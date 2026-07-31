#!/usr/bin/env python3
"""Bounded Chameleon-style portfolio over trained splat populations.

This runner deliberately separates proposal generation from artifact
selection. Training, direct continuation, and Top-K distillation produce raw
splat populations. The portfolio then expands those populations through SVG
recipes, one bounded deployed-artifact recolor pass, and residual native paths.
Only real rasterized SVGs enter the beam. Optionally, a small independent PPTX
shortlist is rendered and captured in Microsoft PowerPoint.
"""

from __future__ import annotations

import argparse
import html
import json
import logging
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

from png2svg_gs.browser_capture import render_svg_in_browser_to_linear_rgb
from png2svg_gs.converter import PNG2SVGConverter
from png2svg_gs.fidelity import (
    FidelityCandidate,
    FidelityConfig,
    FidelityEvaluator,
    FidelityStage,
    RecolorOperator,
    compute_fidelity_metrics,
)
from png2svg_gs.fidelity.metrics import _np_linear_to_srgb, linear_rgb_to_oklab_np
from png2svg_gs.io import (
    compute_quality_metrics,
    generate_svg_content,
    load_png,
    load_splats_json,
    optimize_svg_file,
    save_pptx_with_splats,
    save_splats_json,
    save_svg,
)
from png2svg_gs.mixed_primitives import (
    edge_paths_to_svg_group,
    inject_edge_paths_into_pptx,
    inject_svg_before_close,
    propose_residual_edge_paths,
)
from png2svg_gs.splat import GaussianSplat

RECIPE_NAMES = (
    "standard",
    "browser-compatible",
    "palette-quantized",
    "blur",
    "scripted-matrix",
)
HIGHER_IS_BETTER = {"psnr_srgb", "ssim_srgb", "ms_ssim_luma"}
LOWER_IS_BETTER = {
    "lpips",
    "delta_e_ok_mean",
    "delta_e_ok_p95",
    "edge_chamfer",
    "edge_gradient_l1",
    "worst_roi_error",
}


@dataclass
class Candidate:
    name: str
    svg: Path
    raw: Path
    population: str
    recipe: str
    operations: list[str]
    shape_count: int
    metrics: dict[str, Any]
    parent: str | None = None
    mixed: dict[str, Any] | None = None
    guarded: bool = True
    score: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "svg": str(self.svg),
            "raw": str(self.raw),
            "population": self.population,
            "recipe": self.recipe,
            "operations": self.operations,
            "shape_count": self.shape_count,
            "metrics": self.metrics,
            "parent": self.parent,
            "mixed": self.mixed,
            "guarded": self.guarded,
            "score": self.score,
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--baseline-raw", type=Path, required=True)
    parser.add_argument(
        "--population",
        action="append",
        default=[],
        metavar="NAME=RAW_JSON",
        help="Additional trained population. Repeat for direct/student/postfit.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./tmp/combined-portfolio-mvp"),
    )
    parser.add_argument("--beam-width", type=int, default=3)
    parser.add_argument("--recolor", action="store_true")
    parser.add_argument("--svg-postfit-iters", type=int, default=0)
    parser.add_argument("--blur-postfit-iters", type=int, default=0)
    parser.add_argument(
        "--postfit-population",
        action="append",
        default=[],
        help="Limit compositor post-fit to this population. Repeat as needed.",
    )
    parser.add_argument(
        "--postfit-device",
        choices=["cpu", "mps"],
        default="mps",
    )
    parser.add_argument(
        "--foreground-expert",
        help="Population whose indexed splats should be used inside foreground.",
    )
    parser.add_argument(
        "--background-expert",
        help="Population whose indexed splats should be retained outside foreground.",
    )
    parser.add_argument("--hybrid-name", default="foreground-hybrid")
    parser.add_argument("--capture-powerpoint", action="store_true")
    parser.add_argument("--pptx-raw-limit", type=int, default=2)
    parser.add_argument(
        "--pptx-include-population",
        action="append",
        default=[],
        help="Force this population into the independent PPTX shortlist.",
    )
    return parser


def _parse_population(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"population must be NAME=RAW_JSON, got {value!r}")
    name, raw = value.split("=", 1)
    if not name.strip() or not raw.strip():
        raise ValueError(f"population must be NAME=RAW_JSON, got {value!r}")
    return _slug(name), Path(raw)


def _slug(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in "-_" else "-"
        for character in value.strip().lower()
    ).strip("-")[:80]


def _grid_rois(
    height: int, width: int, tile: int = 64
) -> list[tuple[int, int, int, int]]:
    return [
        (y, x, min(y + tile, height), min(x + tile, width))
        for y in range(0, height, tile)
        for x in range(0, width, tile)
    ]


def _finite_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    result = dict(metrics)
    for key, value in result.items():
        if isinstance(value, float) and not np.isfinite(value):
            result[key] = None
    return result


def evaluate_svg(
    path: Path,
    *,
    target: np.ndarray,
    width: int,
    height: int,
    shape_count: int,
    foreground_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    rendered, renderer = render_svg_in_browser_to_linear_rgb(str(path), width, height)
    metrics = _finite_metrics(
        compute_fidelity_metrics(
            target,
            rendered,
            fixed_rois=_grid_rois(height, width),
            splat_count=shape_count,
            file_size_bytes=path.stat().st_size,
            render_method=renderer,
        ).as_dict()
    )
    if foreground_mask is not None:
        metrics["regions"] = _regional_metrics(
            target, rendered, foreground_mask=foreground_mask
        )
    return metrics


def evaluate_raster(
    path: Path,
    *,
    target: np.ndarray,
    width: int,
    height: int,
    shape_count: int,
    artifact_size: int,
    render_method: str,
    foreground_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    rendered = load_png(str(path), target_size=(width, height))[..., :3]
    metrics = _finite_metrics(
        compute_fidelity_metrics(
            target,
            rendered,
            fixed_rois=_grid_rois(height, width),
            splat_count=shape_count,
            file_size_bytes=artifact_size,
            render_method=render_method,
        ).as_dict()
    )
    if foreground_mask is not None:
        metrics["regions"] = _regional_metrics(
            target, rendered, foreground_mask=foreground_mask
        )
    return metrics


def _regional_metrics(
    target: np.ndarray,
    rendered: np.ndarray,
    *,
    foreground_mask: np.ndarray,
) -> dict[str, Any]:
    target = np.asarray(target, dtype=np.float32)[..., :3]
    rendered = np.asarray(rendered, dtype=np.float32)[..., :3]
    mask = np.asarray(foreground_mask, dtype=bool)
    if mask.shape != target.shape[:2] or not mask.any() or mask.all():
        return {}
    target_srgb = _np_linear_to_srgb(target)
    rendered_srgb = _np_linear_to_srgb(rendered)
    target_ok = linear_rgb_to_oklab_np(target)
    rendered_ok = linear_rgb_to_oklab_np(rendered)
    delta_e = np.sqrt(np.sum((target_ok - rendered_ok) ** 2, axis=-1))

    def masked(name: str, selected: np.ndarray) -> dict[str, Any]:
        difference = target_srgb[selected] - rendered_srgb[selected]
        mse = float(np.mean(difference**2))
        return {
            "name": name,
            "pixels": int(np.count_nonzero(selected)),
            "l1_srgb": float(np.mean(np.abs(difference))),
            "psnr_srgb": float(-10.0 * np.log10(max(mse, 1e-12))),
            "delta_e_ok_mean": float(np.mean(delta_e[selected])),
            "delta_e_ok_p95": float(np.percentile(delta_e[selected], 95)),
        }

    foreground = masked("foreground-mask", mask)
    background = masked("background-mask", ~mask)
    ys, xs = np.nonzero(mask)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    foreground["bbox"] = [y0, x0, y1, x1]
    foreground["bbox_metrics"] = _finite_metrics(
        compute_fidelity_metrics(
            target[y0:y1, x0:x1],
            rendered[y0:y1, x0:x1],
            fixed_rois=_grid_rois(y1 - y0, x1 - x0),
            render_method="foreground-bbox",
        ).as_dict()
    )
    from scipy.ndimage import uniform_filter

    height, width = mask.shape
    focus_size = min(256, max(64, int(round(0.70 * min(height, width)))))
    half = focus_size // 2
    density = uniform_filter(
        mask.astype(np.float32),
        size=focus_size,
        mode="constant",
    )
    density[:half, :] = -1
    density[height - (focus_size - half) :, :] = -1
    density[:, :half] = -1
    density[:, width - (focus_size - half) :] = -1
    focus_y, focus_x = np.unravel_index(np.argmax(density), density.shape)
    focus_y0 = int(np.clip(focus_y - half, 0, height - focus_size))
    focus_x0 = int(np.clip(focus_x - half, 0, width - focus_size))
    focus_y1 = focus_y0 + focus_size
    focus_x1 = focus_x0 + focus_size
    focus_metrics = _finite_metrics(
        compute_fidelity_metrics(
            target[focus_y0:focus_y1, focus_x0:focus_x1],
            rendered[focus_y0:focus_y1, focus_x0:focus_x1],
            fixed_rois=_grid_rois(focus_size, focus_size),
            render_method="foreground-focus-roi",
        ).as_dict()
    )
    focus = {
        "roi": [focus_y0, focus_x0, focus_y1, focus_x1],
        "foreground_density": float(
            np.mean(mask[focus_y0:focus_y1, focus_x0:focus_x1])
        ),
        "metrics": focus_metrics,
    }
    return {
        "foreground": foreground,
        "background": background,
        "focus": focus,
    }


def metric_deltas(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for key in sorted(HIGHER_IS_BETTER | LOWER_IS_BETTER):
        before = baseline.get(key)
        after = candidate.get(key)
        result[key] = (
            None if before is None or after is None else float(after) - float(before)
        )
    result["file_size_bytes"] = float(candidate["file_size_bytes"]) - float(
        baseline["file_size_bytes"]
    )
    return result


def passes_guard(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> tuple[bool, list[str]]:
    """Conservative MVP guard against winning by one blurred metric."""

    failures: list[str] = []
    if candidate["ssim_srgb"] < baseline["ssim_srgb"] - 0.002:
        failures.append("ssim")
    if candidate["ms_ssim_luma"] < baseline["ms_ssim_luma"] - 0.003:
        failures.append("ms-ssim")
    if (
        baseline.get("lpips") is not None
        and candidate.get("lpips") is not None
        and candidate["lpips"] > baseline["lpips"] + 0.005
    ):
        failures.append("lpips")
    if candidate["edge_chamfer"] > baseline["edge_chamfer"] + 0.5:
        failures.append("edge")
    if candidate["worst_roi_error"] > baseline["worst_roi_error"] * 1.03:
        failures.append("worst-roi")
    return not failures, failures


def quality_score(baseline: dict[str, Any], candidate: dict[str, Any]) -> float:
    """Balanced delta score used only inside this bounded experiment."""

    delta = metric_deltas(baseline, candidate)

    def number(key: str) -> float:
        value = delta.get(key)
        return 0.0 if value is None else float(value)

    return float(
        1.50 * number("ssim_srgb")
        + 1.00 * number("ms_ssim_luma")
        - 0.70 * number("lpips")
        - 0.15 * number("delta_e_ok_p95")
        - 0.02 * number("edge_chamfer")
        - 0.30 * number("edge_gradient_l1")
        - 0.25 * number("worst_roi_error")
    )


def _emit_recipe_candidates(
    populations: Iterable[tuple[str, Path, list[str]]],
    *,
    output_dir: Path,
    background: np.ndarray,
    target: np.ndarray,
    foreground_mask: np.ndarray,
    width: int,
    height: int,
) -> list[Candidate]:
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates: list[Candidate] = []
    for population, raw_path, operations in populations:
        splats = load_splats_json(str(raw_path))
        for recipe in RECIPE_NAMES:
            name = f"{population}--{recipe}"
            svg = output_dir / f"{name}.svg"
            save_svg(
                splats,
                width,
                height,
                str(svg),
                background_linear_rgb=background,
                export_recipe=recipe,
            )
            candidates.append(
                Candidate(
                    name=name,
                    svg=svg,
                    raw=raw_path,
                    population=population,
                    recipe=recipe,
                    operations=[*operations, f"svg:{recipe}"],
                    shape_count=len(splats),
                    metrics=evaluate_svg(
                        svg,
                        target=target,
                        width=width,
                        height=height,
                        shape_count=len(splats),
                        foreground_mask=foreground_mask,
                    ),
                )
            )
            if recipe in {"standard", "palette-quantized"}:
                optimized = output_dir / f"{name}--svgo.svg"
                shutil.copy2(svg, optimized)
                optimization = optimize_svg_file(str(optimized), precision=2)
                if optimization.get("applied"):
                    candidates.append(
                        Candidate(
                            name=f"{name}--svgo",
                            svg=optimized,
                            raw=raw_path,
                            population=population,
                            recipe=recipe,
                            operations=[
                                *operations,
                                f"svg:{recipe}",
                                "lossless-svgo",
                            ],
                            shape_count=len(splats),
                            metrics=evaluate_svg(
                                optimized,
                                target=target,
                                width=width,
                                height=height,
                                shape_count=len(splats),
                                foreground_mask=foreground_mask,
                            ),
                        )
                    )
                else:
                    optimized.unlink(missing_ok=True)
    return candidates


def _recolor_population(
    population: str,
    raw_path: Path,
    *,
    output_dir: Path,
    target: np.ndarray,
    background: np.ndarray,
    width: int,
    height: int,
) -> tuple[str, Path, list[str]] | None:
    trace_path = output_dir / f"{population}-recolor.json"
    winner_name = f"{population}-recolor"
    winner_raw = output_dir / f"{winner_name}.raw.json"
    if trace_path.is_file():
        cached = json.loads(trace_path.read_text())
        if cached.get("winner") == population:
            return None
        if winner_raw.is_file():
            return winner_name, winner_raw, ["deployed-artifact-recolor"]
    splats = load_splats_json(str(raw_path))
    config = replace(
        FidelityConfig(mode="max"),
        max_passes=1,
        max_candidates_per_pass=4,
    )
    evaluator = FidelityEvaluator(
        target_linear_rgb=target,
        background_linear_rgb=background,
        compositing_space="srgb",
        emit_svg=lambda items: generate_svg_content(
            items,
            width,
            height,
            background_linear_rgb=background,
            export_recipe="standard",
        ),
        work_dir=str(output_dir / f"{population}-recolor-eval"),
        config=config,
        keep_candidate_artifacts=False,
    )
    result = FidelityStage(config, evaluator, [RecolorOperator()]).run(
        FidelityCandidate(name=population, splats=tuple(splats))
    )
    trace = {
        "population": population,
        "winner": result.winner.name,
        "passes_run": result.passes_run,
        "candidates_evaluated": result.candidates_evaluated,
        "stop_reason": result.stop_reason,
        "baseline_metrics": result.baseline_metrics.as_dict(),
        "final_metrics": result.final_metrics.as_dict(),
        "decisions": result.decisions,
    }
    trace_path.write_text(json.dumps(trace, indent=2, sort_keys=True) + "\n")
    if result.winner.name == population:
        return None
    save_splats_json(list(result.winner.splats), str(winner_raw))
    return winner_name, winner_raw, ["deployed-artifact-recolor"]


def _postfit_populations(
    populations: Iterable[tuple[str, Path, list[str]]],
    *,
    output_dir: Path,
    converter: PNG2SVGConverter,
    target: np.ndarray,
    width: int,
    height: int,
    svg_iters: int,
    blur_iters: int,
) -> list[tuple[str, Path, list[str]]]:
    proposals: list[tuple[str, Path, list[str]]] = []
    for population, raw_path, operations in populations:
        splats = load_splats_json(str(raw_path))
        for kind, iterations in (
            ("svg-postfit", max(0, svg_iters)),
            ("blur-postfit", max(0, blur_iters)),
        ):
            if iterations == 0:
                continue
            name = f"{population}-{kind}"
            fitted_raw = output_dir / f"{name}.raw.json"
            metrics_path = output_dir / f"{name}.json"
            if fitted_raw.is_file() and metrics_path.is_file():
                cached_metrics = json.loads(metrics_path.read_text())
                if int(cached_metrics.get("iterations", -1)) == iterations:
                    proposals.append(
                        (
                            name,
                            fitted_raw,
                            [*operations, f"{kind}:{iterations}"],
                        )
                    )
                    continue
            if kind == "svg-postfit":
                fitted, metrics = converter._postfit_splats_for_svg_proxy(
                    splats,
                    target,
                    width,
                    height,
                    num_iters=iterations,
                    verbose=True,
                )
            else:
                fitted, metrics = converter._postfit_splats_for_blur_proxy(
                    splats,
                    target,
                    width,
                    height,
                    num_iters=iterations,
                    verbose=True,
                )
            save_splats_json(fitted, str(fitted_raw))
            metrics_path.write_text(
                json.dumps(metrics, indent=2, sort_keys=True) + "\n"
            )
            proposals.append(
                (
                    name,
                    fitted_raw,
                    [*operations, f"{kind}:{iterations}"],
                )
            )
    return proposals


def _build_foreground_hybrid(
    *,
    name: str,
    foreground_population: tuple[str, Path, list[str]],
    background_population: tuple[str, Path, list[str]],
    foreground_mask: np.ndarray,
    output_dir: Path,
) -> tuple[str, Path, list[str]]:
    foreground_splats = load_splats_json(str(foreground_population[1]))
    background_splats = load_splats_json(str(background_population[1]))
    if len(foreground_splats) != len(background_splats):
        raise ValueError(
            "foreground/background experts need indexed populations with the "
            f"same splat count, got {len(foreground_splats)} and "
            f"{len(background_splats)}"
        )
    height, width = foreground_mask.shape
    hybrid: list[GaussianSplat] = []
    selected_foreground = 0
    center_distances: list[float] = []
    for foreground_splat, background_splat in zip(
        foreground_splats, background_splats, strict=True
    ):
        foreground_raw = foreground_splat.to_raw_splat()
        background_raw = background_splat.to_raw_splat()
        center_distances.append(
            float(
                np.hypot(
                    foreground_raw.x - background_raw.x,
                    foreground_raw.y - background_raw.y,
                )
            )
        )
        x = int(np.clip(round(foreground_raw.x), 0, width - 1))
        y = int(np.clip(round(foreground_raw.y), 0, height - 1))
        if bool(foreground_mask[y, x]):
            # Preserve the background expert's stable global draw-order slot.
            # Only the region expert's geometry/color/alpha is substituted.
            selected = replace(
                foreground_raw,
                importance=background_raw.importance,
                layer=background_raw.layer,
                source=f"foreground:{foreground_population[0]}",
            )
            selected_foreground += 1
        else:
            selected = replace(
                background_raw,
                source=f"background:{background_population[0]}",
            )
        hybrid.append(GaussianSplat.from_raw_splat(selected))
    raw_path = output_dir / f"{name}.raw.json"
    save_splats_json(hybrid, str(raw_path))
    (output_dir / f"{name}.json").write_text(
        json.dumps(
            {
                "name": name,
                "foreground_population": foreground_population[0],
                "background_population": background_population[0],
                "splat_count": len(hybrid),
                "foreground_splats": selected_foreground,
                "background_splats": len(hybrid) - selected_foreground,
                "foreground_fraction": selected_foreground / max(len(hybrid), 1),
                "median_index_center_distance": float(np.median(center_distances)),
                "p95_index_center_distance": float(np.percentile(center_distances, 95)),
                "order_policy": "background-expert importance and layer",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return (
        name,
        raw_path,
        [
            f"foreground-expert:{foreground_population[0]}",
            f"background-expert:{background_population[0]}",
            "region-conditioned-indexed-hybrid",
        ],
    )


def _mixed_winner(
    parent: Candidate,
    *,
    output_dir: Path,
    target: np.ndarray,
    foreground_mask: np.ndarray,
    width: int,
    height: int,
) -> Candidate | None:
    parent_render, renderer = render_svg_in_browser_to_linear_rgb(
        str(parent.svg), width, height
    )
    parent_quality = compute_quality_metrics(target, parent_render)
    base_content = parent.svg.read_text()
    scratch = output_dir / f"{parent.name}--mixed-scratch.svg"
    best: dict[str, Any] | None = None
    best_content: str | None = None
    for opacity in (0.50, 0.70):
        for length in (12.0, 24.0):
            for stroke_width in (1.0, 2.0):
                proposals = propose_residual_edge_paths(
                    target,
                    parent_render,
                    max_paths=64,
                    path_length=length,
                    width=stroke_width,
                    opacity=opacity,
                )
                for count in (16, 32, 64):
                    paths = proposals[:count]
                    content = inject_svg_before_close(
                        base_content, edge_paths_to_svg_group(paths)
                    )
                    scratch.write_text(content)
                    rendered, actual_renderer = render_svg_in_browser_to_linear_rgb(
                        str(scratch), width, height
                    )
                    metrics = compute_quality_metrics(target, rendered)
                    ssim_gain = float(metrics["ssim_srgb"]) - float(
                        parent_quality["ssim_srgb"]
                    )
                    psnr_regression = float(parent_quality["psnr_srgb"]) - float(
                        metrics["psnr_srgb"]
                    )
                    record = {
                        "count": len(paths),
                        "length": length,
                        "width": stroke_width,
                        "opacity": opacity,
                        "ssim_gain": ssim_gain,
                        "psnr_regression": psnr_regression,
                        "renderer": actual_renderer,
                    }
                    if ssim_gain < 0.0002 or psnr_regression > 0.15:
                        continue
                    key = (
                        ssim_gain,
                        -psnr_regression,
                        -len(paths),
                        -stroke_width,
                    )
                    if best is None or key > best["_key"]:
                        best = {**record, "_key": key}
                        best_content = content
    scratch.unlink(missing_ok=True)
    if best is None or best_content is None:
        return None
    best.pop("_key")
    output = output_dir / f"{parent.name}--mixed-paths.svg"
    output.write_text(best_content)
    metrics = evaluate_svg(
        output,
        target=target,
        width=width,
        height=height,
        shape_count=parent.shape_count + int(best["count"]),
        foreground_mask=foreground_mask,
    )
    return Candidate(
        name=f"{parent.name}--mixed-paths",
        svg=output,
        raw=parent.raw,
        population=parent.population,
        recipe=parent.recipe,
        operations=[*parent.operations, "residual-native-paths"],
        shape_count=parent.shape_count + int(best["count"]),
        metrics=metrics,
        parent=parent.name,
        mixed=best,
    )


def _rank(candidates: list[Candidate], baseline: Candidate) -> list[Candidate]:
    for candidate in candidates:
        candidate.guarded, failures = passes_guard(baseline.metrics, candidate.metrics)
        candidate.score = quality_score(baseline.metrics, candidate.metrics)
        if failures:
            candidate.operations = [
                operation
                for operation in candidate.operations
                if not operation.startswith("guard-failed:")
            ]
            candidate.operations = [
                *candidate.operations,
                f"guard-failed:{','.join(failures)}",
            ]
    return sorted(
        candidates,
        key=lambda candidate: (
            candidate.guarded,
            candidate.score,
            -candidate.metrics["file_size_bytes"],
        ),
        reverse=True,
    )


def _unique_raw_candidates(
    ranked: Iterable[Candidate],
    limit: int,
    include_populations: Iterable[str] = (),
) -> list[Candidate]:
    ranked = list(ranked)
    result: list[Candidate] = []
    seen: set[Path] = set()
    for population in include_populations:
        forced = next(
            (candidate for candidate in ranked if candidate.population == population),
            None,
        )
        if forced is None or forced.raw.resolve() in seen:
            continue
        seen.add(forced.raw.resolve())
        result.append(forced)
        if len(result) >= limit:
            return result
    for candidate in ranked:
        resolved = candidate.raw.resolve()
        if not candidate.guarded or resolved in seen:
            continue
        seen.add(resolved)
        result.append(candidate)
        if len(result) >= limit:
            break
    return result


def _diverse_beam(ranked: Iterable[Candidate], width: int) -> list[Candidate]:
    """Retain lineage diversity so near-duplicate recipes do not fill the beam."""

    eligible = [candidate for candidate in ranked if candidate.guarded]
    selected: list[Candidate] = []
    seen_names: set[str] = set()
    seen_populations: set[str] = set()
    for candidate in eligible:
        if candidate.population in seen_populations:
            continue
        selected.append(candidate)
        seen_names.add(candidate.name)
        seen_populations.add(candidate.population)
        if len(selected) >= width:
            return selected
    for candidate in eligible:
        if candidate.name in seen_names:
            continue
        selected.append(candidate)
        seen_names.add(candidate.name)
        if len(selected) >= width:
            break
    return selected


def _emit_and_capture_pptx(
    ranked: list[Candidate],
    *,
    output_dir: Path,
    target: np.ndarray,
    foreground_mask: np.ndarray,
    background: np.ndarray,
    width: int,
    height: int,
    raw_limit: int,
    include_populations: Iterable[str],
    capture: bool,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_candidates = _unique_raw_candidates(
        ranked,
        raw_limit,
        include_populations=include_populations,
    )
    proposals: list[tuple[Candidate, str]] = []
    for index, candidate in enumerate(raw_candidates):
        proposals.extend((candidate, style) for style in ("gradient", "blur"))
        if index == 0:
            proposals.append((candidate, "soft-edge"))

    records: list[dict[str, Any]] = []
    for parent, style in proposals:
        splats = load_splats_json(str(parent.raw))
        name = f"{parent.population}--pptx-{style}"
        pptx = output_dir / f"{name}.pptx"
        save_pptx_with_splats(
            splats,
            width,
            height,
            str(pptx),
            background_linear_rgb=background,
            splat_style=style,
        )
        record: dict[str, Any] = {
            "name": name,
            "population": parent.population,
            "style": style,
            "operations": [*parent.operations, f"pptx:{style}"],
            "shape_count": len(splats),
            "pptx": str(pptx),
            "bytes": pptx.stat().st_size,
            "capture": None,
            "metrics": None,
        }
        if capture:
            from tools.full_corpus_mvp import _capture_powerpoint_slideshow

            screenshot = output_dir / f"{name}.png"
            returncode, message = _capture_powerpoint_slideshow(
                pptx, screenshot, width, height
            )
            record["capture_log"] = message
            if returncode == 0:
                record["capture"] = str(screenshot)
                record["metrics"] = evaluate_raster(
                    screenshot,
                    target=target,
                    width=width,
                    height=height,
                    shape_count=len(splats),
                    artifact_size=pptx.stat().st_size,
                    render_method="microsoft-powerpoint",
                    foreground_mask=foreground_mask,
                )
            else:
                record["capture_error"] = returncode
        records.append(record)

    mixed_parent = next(
        (
            candidate
            for candidate in ranked
            if candidate.guarded and candidate.mixed is not None
        ),
        None,
    )
    if mixed_parent is not None:
        splats = load_splats_json(str(mixed_parent.raw))
        base_pptx = output_dir / f"{mixed_parent.population}--pptx-gradient.pptx"
        if not base_pptx.exists():
            save_pptx_with_splats(
                splats,
                width,
                height,
                str(base_pptx),
                background_linear_rgb=background,
                splat_style="gradient",
            )
        parent_render, renderer = render_svg_in_browser_to_linear_rgb(
            str(
                next(
                    candidate.svg
                    for candidate in ranked
                    if candidate.name == mixed_parent.parent
                )
            ),
            width,
            height,
        )
        spec = mixed_parent.mixed
        paths = propose_residual_edge_paths(
            target,
            parent_render,
            max_paths=int(spec["count"]),
            path_length=float(spec["length"]),
            width=float(spec["width"]),
            opacity=float(spec["opacity"]),
        )[: int(spec["count"])]
        pptx = output_dir / f"{mixed_parent.population}--pptx-gradient-mixed.pptx"
        segment_count = inject_edge_paths_into_pptx(
            base_pptx,
            pptx,
            paths,
            width=width,
            height=height,
        )
        record = {
            "name": f"{mixed_parent.population}--pptx-gradient-mixed",
            "population": mixed_parent.population,
            "style": "gradient+mixed-paths",
            "operations": [*mixed_parent.operations, "pptx:gradient"],
            "shape_count": len(splats) + segment_count,
            "pptx": str(pptx),
            "bytes": pptx.stat().st_size,
            "native_segment_shapes": segment_count,
            "capture": None,
            "metrics": None,
        }
        if capture:
            from tools.full_corpus_mvp import _capture_powerpoint_slideshow

            screenshot = output_dir / f"{record['name']}.png"
            returncode, message = _capture_powerpoint_slideshow(
                pptx, screenshot, width, height
            )
            record["capture_log"] = message
            if returncode == 0:
                record["capture"] = str(screenshot)
                record["metrics"] = evaluate_raster(
                    screenshot,
                    target=target,
                    width=width,
                    height=height,
                    shape_count=record["shape_count"],
                    artifact_size=pptx.stat().st_size,
                    render_method="microsoft-powerpoint",
                    foreground_mask=foreground_mask,
                )
            else:
                record["capture_error"] = returncode
        records.append(record)
    pptx_baseline = next(
        (record for record in records if record.get("metrics") is not None),
        None,
    )
    if pptx_baseline is not None:
        baseline_metrics = pptx_baseline["metrics"]
        baseline_foreground = (
            baseline_metrics.get("regions", {})
            .get("foreground", {})
            .get("bbox_metrics")
        )
        baseline_focus = (
            baseline_metrics.get("regions", {}).get("focus", {}).get("metrics")
        )
        for record in records:
            metrics = record.get("metrics")
            if metrics is None:
                record["guarded"] = False
                record["score"] = None
                continue
            guarded, failures = passes_guard(baseline_metrics, metrics)
            record["guarded"] = guarded
            record["guard_failures"] = failures
            record["score"] = quality_score(baseline_metrics, metrics)
            foreground = (
                metrics.get("regions", {}).get("foreground", {}).get("bbox_metrics")
            )
            if baseline_foreground is not None and foreground is not None:
                foreground_guarded, foreground_failures = passes_guard(
                    baseline_foreground, foreground
                )
                record["foreground_guarded"] = foreground_guarded
                record["foreground_guard_failures"] = foreground_failures
                record["foreground_score"] = quality_score(
                    baseline_foreground, foreground
                )
            focus = metrics.get("regions", {}).get("focus", {}).get("metrics")
            if baseline_focus is not None and focus is not None:
                focus_guarded, focus_failures = passes_guard(baseline_focus, focus)
                record["focus_guarded"] = focus_guarded
                record["focus_guard_failures"] = focus_failures
                record["focus_score"] = quality_score(baseline_focus, focus)
    return records


def _rel(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def _write_html(
    *,
    output: Path,
    target_copy: Path,
    candidates: list[Candidate],
    pptx_records: list[dict[str, Any]],
    baseline: Candidate,
    source_label: str,
    width: int,
    height: int,
) -> None:
    root = output.parent
    svg_cards = []
    for rank, candidate in enumerate(candidates, start=1):
        delta = metric_deltas(baseline.metrics, candidate.metrics)
        regions = candidate.metrics.get("regions", {})
        foreground = regions.get("foreground", {})
        foreground_bbox = foreground.get("bbox_metrics", {})
        background = regions.get("background", {})
        focus = regions.get("focus", {}).get("metrics", {})
        svg_cards.append(
            f"""
<article>
  <header><b>#{rank} {html.escape(candidate.name)}</b><span class="{'ok' if candidate.guarded else 'bad'}">{'guarded' if candidate.guarded else 'rejected'}</span></header>
  <div class="pair">
    <figure><img src="{_rel(target_copy, root)}"><figcaption>target</figcaption></figure>
    <figure><img src="{_rel(candidate.svg, root)}"><figcaption>actual SVG · {candidate.metrics['render_method']}</figcaption></figure>
  </div>
  <p>{html.escape(' + '.join(candidate.operations))}</p>
  <table><tr><th>metric</th><th>value</th><th>vs baseline</th></tr>
    <tr><td>SSIM</td><td>{_fmt(candidate.metrics['ssim_srgb'], 5)}</td><td>{_fmt(delta['ssim_srgb'], 5)}</td></tr>
    <tr><td>MS-SSIM</td><td>{_fmt(candidate.metrics['ms_ssim_luma'], 5)}</td><td>{_fmt(delta['ms_ssim_luma'], 5)}</td></tr>
    <tr><td>LPIPS ↓</td><td>{_fmt(candidate.metrics['lpips'], 5)}</td><td>{_fmt(delta['lpips'], 5)}</td></tr>
    <tr><td>PSNR</td><td>{_fmt(candidate.metrics['psnr_srgb'], 2)}</td><td>{_fmt(delta['psnr_srgb'], 2)}</td></tr>
    <tr><td>OKLab p95 ↓</td><td>{_fmt(candidate.metrics['delta_e_ok_p95'], 5)}</td><td>{_fmt(delta['delta_e_ok_p95'], 5)}</td></tr>
    <tr><td>edge chamfer ↓</td><td>{_fmt(candidate.metrics['edge_chamfer'], 3)}</td><td>{_fmt(delta['edge_chamfer'], 3)}</td></tr>
    <tr><td>worst ROI ↓</td><td>{_fmt(candidate.metrics['worst_roi_error'], 5)}</td><td>{_fmt(delta['worst_roi_error'], 5)}</td></tr>
    <tr><td>foreground SSIM</td><td>{_fmt(foreground_bbox.get('ssim_srgb'), 5)}</td><td></td></tr>
    <tr><td>foreground LPIPS ↓</td><td>{_fmt(foreground_bbox.get('lpips'), 5)}</td><td></td></tr>
    <tr><td>foreground masked L1 ↓</td><td>{_fmt(foreground.get('l1_srgb'), 5)}</td><td></td></tr>
    <tr><td>focus SSIM</td><td>{_fmt(focus.get('ssim_srgb'), 5)}</td><td></td></tr>
    <tr><td>focus LPIPS ↓</td><td>{_fmt(focus.get('lpips'), 5)}</td><td></td></tr>
    <tr><td>focus edge ↓</td><td>{_fmt(focus.get('edge_chamfer'), 3)}</td><td></td></tr>
    <tr><td>background masked L1 ↓</td><td>{_fmt(background.get('l1_srgb'), 5)}</td><td></td></tr>
    <tr><td>bytes</td><td>{candidate.metrics['file_size_bytes']:,}</td><td>{delta['file_size_bytes']:+,.0f}</td></tr>
    <tr><td>MVP score</td><td>{candidate.score:+.5f}</td><td></td></tr>
  </table>
</article>"""
        )
    pptx_cards = []
    for record in pptx_records:
        metrics = record.get("metrics")
        screenshot = record.get("capture")
        visual = (
            f'<img src="{_rel(Path(screenshot), root)}">'
            if screenshot
            else "<div class='missing'>not captured</div>"
        )
        metric_line = (
            "not scored"
            if not metrics
            else (
                f"SSIM {_fmt(metrics['ssim_srgb'], 5)} · "
                f"MS-SSIM {_fmt(metrics['ms_ssim_luma'], 5)} · "
                f"LPIPS {_fmt(metrics['lpips'], 5)} · "
                f"foreground SSIM {_fmt(metrics.get('regions', {}).get('foreground', {}).get('bbox_metrics', {}).get('ssim_srgb'), 5)} · "
                f"foreground L1 {_fmt(metrics.get('regions', {}).get('foreground', {}).get('l1_srgb'), 5)} · "
                f"focus LPIPS {_fmt(metrics.get('regions', {}).get('focus', {}).get('metrics', {}).get('lpips'), 5)} · "
                f"focus edge {_fmt(metrics.get('regions', {}).get('focus', {}).get('metrics', {}).get('edge_chamfer'), 3)} · "
                f"background L1 {_fmt(metrics.get('regions', {}).get('background', {}).get('l1_srgb'), 5)} · "
                f"{record['bytes']:,} bytes"
            )
        )
        pptx_cards.append(
            f"""
<article>
  <header><b>{html.escape(record['name'])}</b><span>{html.escape(record['style'])}</span></header>
  <div class="pair">
    <figure><img src="{_rel(target_copy, root)}"><figcaption>target</figcaption></figure>
    <figure>{visual}<figcaption>real Microsoft PowerPoint</figcaption></figure>
  </div>
  <p>{metric_line}</p>
</article>"""
        )
    output.write_text(
        f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Combined splat portfolio MVP</title>
<style>
:root {{ color-scheme: dark; font-family: Inter, system-ui, sans-serif; }}
body {{ margin: 0 auto; max-width: 1500px; padding: 28px; background: #11151b; color: #edf2f7; }}
h1,h2 {{ margin: 0.4rem 0; }} .lede {{ color: #a8b3c2; max-width: 90ch; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(430px,1fr)); gap:18px; }}
article {{ background:#1b222c; border:1px solid #303a48; border-radius:12px; padding:14px; }}
header {{ display:flex; justify-content:space-between; gap:12px; }}
.pair {{ display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-top:10px; }}
figure {{ margin:0; }} img,.missing {{ width:100%; aspect-ratio:{width}/{height}; object-fit:contain; background:#0a0d11; }}
figcaption {{ color:#9aa6b5; font-size:.8rem; margin-top:4px; }}
table {{ border-collapse:collapse; width:100%; font-variant-numeric:tabular-nums; }}
td,th {{ border-bottom:1px solid #303a48; padding:4px 6px; text-align:right; }}
td:first-child,th:first-child {{ text-align:left; }} .ok {{ color:#5ee0a0; }} .bad {{ color:#ff7e8a; }}
</style></head><body>
<h1>Combined algorithm portfolio · {html.escape(source_label)} MVP</h1>
<p class="lede">Real SVG artifacts are ranked with a guarded metric vector, not SSIM alone. The beam combines trained splat populations, export recipes, deployed-artifact recolor, and residual native paths. PowerPoint is selected independently from real slideshow captures.</p>
<h2>SVG beam</h2><div class="grid">{''.join(svg_cards)}</div>
<h2>Native PowerPoint shortlist</h2><div class="grid">{''.join(pptx_cards)}</div>
</body></html>"""
    )


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = args.source.resolve()
    baseline_raw = args.baseline_raw.resolve()
    if not source.is_file() or not baseline_raw.is_file():
        raise FileNotFoundError("source or baseline raw checkpoint does not exist")

    with Image.open(source) as image:
        width, height = image.size
    target = load_png(str(source), target_size=(width, height))[..., :3]
    converter = PNG2SVGConverter(
        max_splats=len(load_splats_json(str(baseline_raw))),
        stages=[0],
        target_size=(width, height),
        quality_profile="fast",
        device=args.postfit_device,
        apple_silicon_splat_cap=None,
    )
    guidance = converter._compute_region_guidance(target)
    converter._region_weight_map = guidance["weight_map"]
    converter._region_saliency_map = guidance.get("saliency_map")
    converter._region_detail_priority_map = guidance.get("detail_priority_map")
    converter._region_background_penalty_map = guidance.get("background_penalty_map")
    converter._region_foreground_mask = guidance["foreground_mask"]
    converter._region_background_safe_mask = guidance["background_safe_mask"]
    converter._region_edge_band_mask = guidance["edge_band_mask"]
    background = guidance["background_linear_rgb"]
    converter._background_linear_rgb = background

    populations: list[tuple[str, Path, list[str]]] = [
        ("base4k", baseline_raw, ["target-aware-mlx", "densify-split-prune"])
    ]
    for value in args.population:
        name, raw = _parse_population(value)
        raw = raw.resolve()
        if not raw.is_file():
            raise FileNotFoundError(raw)
        populations.append((name, raw, [name]))

    if bool(args.foreground_expert) != bool(args.background_expert):
        raise ValueError(
            "--foreground-expert and --background-expert must be supplied together"
        )
    if args.foreground_expert and args.background_expert:
        by_name = {population[0]: population for population in populations}
        foreground_name = _slug(args.foreground_expert)
        background_name = _slug(args.background_expert)
        if foreground_name not in by_name or background_name not in by_name:
            raise ValueError(
                "hybrid experts must name supplied populations: "
                f"{foreground_name!r}, {background_name!r}"
            )
        populations.append(
            _build_foreground_hybrid(
                name=_slug(args.hybrid_name),
                foreground_population=by_name[foreground_name],
                background_population=by_name[background_name],
                foreground_mask=guidance["foreground_mask"],
                output_dir=args.output_dir,
            )
        )

    base_populations = tuple(populations)
    requested_postfit = {_slug(population) for population in args.postfit_population}
    postfit_populations = tuple(
        population
        for population in base_populations
        if not requested_postfit or population[0] in requested_postfit
    )
    postfit_records = _postfit_populations(
        postfit_populations,
        output_dir=args.output_dir,
        converter=converter,
        target=target,
        width=width,
        height=height,
        svg_iters=args.svg_postfit_iters,
        blur_iters=args.blur_postfit_iters,
    )
    recolor_records: list[tuple[str, Path, list[str]]] = []
    if args.recolor:
        for population, raw, _ in base_populations:
            result = _recolor_population(
                population,
                raw,
                output_dir=args.output_dir,
                target=target,
                background=background,
                width=width,
                height=height,
            )
            if result is not None:
                recolor_records.append(result)
    populations.extend(postfit_records)
    populations.extend(recolor_records)

    recipe_dir = args.output_dir / "svg"
    candidates = _emit_recipe_candidates(
        populations,
        output_dir=recipe_dir,
        background=background,
        target=target,
        foreground_mask=guidance["foreground_mask"],
        width=width,
        height=height,
    )
    baseline = next(
        candidate
        for candidate in candidates
        if candidate.population == "base4k" and candidate.recipe == "standard"
    )
    ranked = _rank(candidates, baseline)
    beam = _diverse_beam(ranked, max(1, args.beam_width))
    mixed_candidates = [
        candidate
        for candidate in (
            _mixed_winner(
                parent,
                output_dir=recipe_dir,
                target=target,
                foreground_mask=guidance["foreground_mask"],
                width=width,
                height=height,
            )
            for parent in beam
        )
        if candidate is not None
    ]
    ranked = _rank([*candidates, *mixed_candidates], baseline)

    pptx_records = _emit_and_capture_pptx(
        ranked,
        output_dir=args.output_dir / "pptx",
        target=target,
        foreground_mask=guidance["foreground_mask"],
        background=background,
        width=width,
        height=height,
        raw_limit=max(1, args.pptx_raw_limit),
        include_populations=(
            _slug(population) for population in args.pptx_include_population
        ),
        capture=args.capture_powerpoint,
    )
    target_copy = args.output_dir / "target.png"
    shutil.copy2(source, target_copy)
    report = {
        "source": str(source),
        "size": [width, height],
        "background_linear_rgb": [float(value) for value in background],
        "beam_width": args.beam_width,
        "excluded": {
            "top-k-teacher": (
                "Teacher is a non-exportable optimization ceiling; its exportable "
                "student population enters through --population."
            ),
        },
        "baseline": baseline.as_dict(),
        "winner_svg": ranked[0].as_dict(),
        "svg_candidates": [candidate.as_dict() for candidate in ranked],
        "pptx_candidates": pptx_records,
        "winner_pptx": next(
            (
                record
                for record in sorted(
                    (
                        item
                        for item in pptx_records
                        if item.get("metrics") and item.get("guarded")
                    ),
                    key=lambda item: item["score"],
                    reverse=True,
                )
            ),
            None,
        ),
        "winner_pptx_foreground": next(
            (
                record
                for record in sorted(
                    (
                        item
                        for item in pptx_records
                        if item.get("metrics") and item.get("foreground_guarded")
                    ),
                    key=lambda item: item["foreground_score"],
                    reverse=True,
                )
            ),
            None,
        ),
        "winner_pptx_focus": next(
            (
                record
                for record in sorted(
                    (
                        item
                        for item in pptx_records
                        if item.get("metrics") and item.get("focus_guarded")
                    ),
                    key=lambda item: item["focus_score"],
                    reverse=True,
                )
            ),
            None,
        ),
        "best_pptx_focus_lpips": min(
            (
                record
                for record in pptx_records
                if record.get("metrics")
                and record["metrics"].get("regions", {}).get("focus")
            ),
            key=lambda item: item["metrics"]["regions"]["focus"]["metrics"]["lpips"],
            default=None,
        ),
        "best_pptx_focus_edge": min(
            (
                record
                for record in pptx_records
                if record.get("metrics")
                and record["metrics"].get("regions", {}).get("focus")
            ),
            key=lambda item: item["metrics"]["regions"]["focus"]["metrics"][
                "edge_chamfer"
            ],
            default=None,
        ),
        "best_pptx_background_l1": min(
            (
                record
                for record in pptx_records
                if record.get("metrics")
                and record["metrics"].get("regions", {}).get("background")
            ),
            key=lambda item: item["metrics"]["regions"]["background"]["l1_srgb"],
            default=None,
        ),
    }
    comparison = args.output_dir / "comparison.json"
    comparison.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    overview = args.output_dir / "index.html"
    _write_html(
        output=overview,
        target_copy=target_copy,
        candidates=ranked,
        pptx_records=pptx_records,
        baseline=baseline,
        source_label=source.stem.replace("_", " ").title(),
        width=width,
        height=height,
    )
    print(
        json.dumps(
            {
                "winner_svg": report["winner_svg"],
                "winner_pptx": report["winner_pptx"],
                "winner_pptx_foreground": report["winner_pptx_foreground"],
                "winner_pptx_focus": report["winner_pptx_focus"],
                "best_pptx_focus_lpips": report["best_pptx_focus_lpips"],
                "best_pptx_focus_edge": report["best_pptx_focus_edge"],
                "best_pptx_background_l1": report["best_pptx_background_l1"],
                "svg_candidates": len(ranked),
                "pptx_candidates": len(pptx_records),
                "comparison": str(comparison),
                "overview": str(overview),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
