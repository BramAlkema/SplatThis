#!/usr/bin/env python3
"""Evaluate bounded SVG recipes over existing full-corpus splat populations.

No splats are trained or mutated. For every selected image the canonical seed-0
SVG population is exported as standard, palette-quantized, and native-blur SVG.
Every artifact is captured at native dimensions by Playwright Chromium, scored
with fixed baseline ROIs, and accepted or reverted under one predeclared policy.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import statistics
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from png2svg_gs.browser_capture import PlaywrightSvgRenderer, resolve_browser_executable
from png2svg_gs.fidelity.analysis import analyze_residual
from png2svg_gs.fidelity.metrics import compute_fidelity_metrics
from png2svg_gs.io import (
    atomic_write_text,
    load_png,
    load_splats_json,
    save_svg,
    srgb_to_linear,
)
from png2svg_gs.svg_recipe_gate import (
    SvgRecipeGatePolicy,
    metric_deltas,
    select_recipe_candidate,
)

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = REPO / "result" / "corpus"
DEFAULT_OUTPUT = REPO / "tmp" / "svg-recipe-gate"
RECIPES = ("standard", "palette-quantized", "blur")
FloatArray = NDArray[np.float32]


def _write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _finite(value: Any) -> Any:
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(item) for item in value]
    return value


def _median(values: Iterable[float]) -> float:
    materialized = list(values)
    return float(statistics.median(materialized)) if materialized else 0.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    try:
        relative = path.resolve().relative_to(REPO)
        rendered = str(relative)
        return f"./{rendered}" if relative.parts[:1] == ("tmp",) else rendered
    except ValueError:
        return str(path.resolve())


def _relative_url(path: Path, root: Path) -> str:
    return Path(os.path.relpath(path.resolve(), root.resolve())).as_posix()


def _format_metric(value: Any, digits: int = 5) -> str:
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def _measure_recipe(
    *,
    image: str,
    recipe: str,
    splats: list[Any],
    target: FloatArray,
    background: FloatArray,
    fixed_rois: Sequence[tuple[int, int, int, int]],
    output_dir: Path,
    renderer: PlaywrightSvgRenderer,
    render_repeats: int,
    renderer_version: str,
) -> tuple[dict[str, Any], FloatArray]:
    height, width = target.shape[:2]
    image_dir = output_dir / image
    image_dir.mkdir(parents=True, exist_ok=True)
    svg_path = image_dir / f"{recipe}.svg"
    png_path = image_dir / f"{recipe}.png"
    save_svg(
        splats,
        width,
        height,
        str(svg_path),
        background_linear_rgb=background,
        export_recipe=recipe,
    )
    capture = renderer.capture(
        svg_path,
        png_path,
        width=width,
        height=height,
        repeats=render_repeats,
        samples_dir=image_dir / "capture-samples" / recipe,
    )
    if not capture.pixel_stable:
        raise RuntimeError(
            f"Chromium produced differing pixel hashes for {image}/{recipe}"
        )
    with Image.open(png_path) as raster:
        srgb = np.asarray(raster.convert("RGB"), dtype=np.float32) / 255.0
    rendered = srgb_to_linear(srgb)
    timings = [sample / 1000.0 for sample in capture.capture_time_ms_samples]
    metrics = _finite(
        compute_fidelity_metrics(
            target,
            rendered,
            fixed_rois=fixed_rois,
            splat_count=len(splats),
            file_size_bytes=svg_path.stat().st_size,
            render_method=renderer_version,
        ).as_dict()
    )
    measurement = {
        "recipe": recipe,
        "svg": _display_path(svg_path),
        "raster": _display_path(png_path),
        "svg_sha256": _sha256(svg_path),
        "raster_sha256": _sha256(png_path),
        "render_time_sec": _median(timings),
        "render_time_samples_sec": timings,
        "raster_sample_sha256": list(capture.sample_sha256),
        "pixel_stable": capture.pixel_stable,
        "warmup_captures": capture.warmup_captures,
        "capture_method": "Playwright Chromium viewport-clipped PNG screenshot",
        **metrics,
    }
    return measurement, rendered


def _evaluate_image(
    *,
    image: str,
    source_path: Path,
    raw_path: Path,
    manifest_path: Path,
    output_dir: Path,
    renderer: PlaywrightSvgRenderer,
    render_repeats: int,
    renderer_version: str,
    policy: SvgRecipeGatePolicy,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text())
    config = manifest.get("config", {})
    resolved_size = config.get("resolved_target_size")
    target_size = (
        (int(resolved_size[0]), int(resolved_size[1]))
        if isinstance(resolved_size, list) and len(resolved_size) == 2
        else None
    )
    target = load_png(str(source_path), target_size=target_size)[..., :3]
    background = np.asarray(
        config.get("background_linear_rgb", [0.0, 0.0, 0.0]),
        dtype=np.float32,
    )
    splats = load_splats_json(str(raw_path))

    baseline, baseline_render = _measure_recipe(
        image=image,
        recipe="standard",
        splats=splats,
        target=target,
        background=background,
        fixed_rois=(),
        output_dir=output_dir,
        renderer=renderer,
        render_repeats=render_repeats,
        renderer_version=renderer_version,
    )
    fixed_rois = analyze_residual(
        target,
        baseline_render,
        roi_size=min(64, target.shape[0], target.shape[1]),
        roi_count=8,
    ).fixed_rois
    # Recompute the baseline with the same frozen ROI contract as candidates.
    baseline_metrics = _finite(
        compute_fidelity_metrics(
            target,
            baseline_render,
            fixed_rois=fixed_rois,
            splat_count=len(splats),
            file_size_bytes=int(baseline["file_size_bytes"]),
            render_method=renderer_version,
        ).as_dict()
    )
    baseline.update(baseline_metrics)

    candidates = []
    for recipe in RECIPES[1:]:
        measured, _ = _measure_recipe(
            image=image,
            recipe=recipe,
            splats=splats,
            target=target,
            background=background,
            fixed_rois=fixed_rois,
            output_dir=output_dir,
            renderer=renderer,
            render_repeats=render_repeats,
            renderer_version=renderer_version,
        )
        candidates.append(measured)
    selection = select_recipe_candidate(baseline, candidates, policy)
    selected = selection["selected"]
    return {
        "image": image,
        "source": _display_path(source_path),
        "source_sha256": _sha256(source_path),
        "raw_splats": _display_path(raw_path),
        "raw_splats_sha256": _sha256(raw_path),
        "manifest": _display_path(manifest_path),
        "splat_count": len(splats),
        "size": [int(target.shape[1]), int(target.shape[0])],
        "fixed_rois": [list(roi) for roi in fixed_rois],
        "recipes": {item["recipe"]: item for item in [baseline, *candidates]},
        "selection": selection,
        "selected_deltas": metric_deltas(baseline, selected),
    }


def _summarize(
    *,
    expected_images: Sequence[str],
    results: Sequence[Mapping[str, Any]],
    failures: Sequence[Mapping[str, str]],
    policy: SvgRecipeGatePolicy,
    browser_version: str,
    browser_executable: Path,
    render_repeats: int,
) -> dict[str, Any]:
    accepted = [
        result for result in results if result["selection"]["accepted_candidate"]
    ]
    size_growth = []
    render_growth = []
    accepted_lpips_gains = []
    accepted_delta_e_gains = []
    for result in results:
        recipes = result["recipes"]
        baseline = recipes["standard"]
        selected = result["selection"]["selected"]
        size_growth.append(
            float(selected["file_size_bytes"]) / float(baseline["file_size_bytes"])
            - 1.0
        )
        render_growth.append(
            float(selected["render_time_sec"])
            / max(float(baseline["render_time_sec"]), 1e-9)
            - 1.0
        )
        if result["selection"]["accepted_candidate"]:
            delta = result["selected_deltas"]
            if delta["lpips"] is not None:
                accepted_lpips_gains.append(-float(delta["lpips"]))
            if delta["delta_e_ok_p95"] is not None:
                accepted_delta_e_gains.append(-float(delta["delta_e_ok_p95"]))

    median_lpips_gain = _median(accepted_lpips_gains)
    median_delta_e_gain = _median(accepted_delta_e_gains)
    perceptual_gate = bool(
        median_lpips_gain >= policy.min_lpips_gain
        or median_delta_e_gain >= policy.min_delta_e_p95_gain
    )
    median_size_growth = _median(size_growth)
    median_render_growth = _median(render_growth)
    complete = len(results) == len(expected_images) and not failures
    criteria = {
        "complete_full_frame_corpus": complete,
        "minimum_accepted_images": len(accepted) >= policy.minimum_accepted_images,
        "accepted_median_perceptual_gain": perceptual_gate,
        "median_size_growth_within_limit": median_size_growth
        <= policy.maximum_median_size_growth_fraction,
        "median_render_time_growth_within_limit": median_render_growth
        <= policy.maximum_median_render_time_growth_fraction,
    }
    integrate = all(criteria.values())
    return {
        "schema": "splatthis.svg-recipe-gate-run/2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "evidence": {
            "artifact": "emitted SVG captured at native dimensions",
            "renderer": f"Chromium {browser_version}",
            "browser_executable": str(browser_executable),
            "capture_method": "Playwright Chromium viewport-clipped PNG screenshot",
            "device_scale_factor": 1,
            "animations": "disabled during screenshot",
            "warmup_captures_per_artifact": 1,
            "pixel_repeat_stability_required": True,
            "render_repeats": render_repeats,
            "training_performed": False,
            "full_frames": True,
            "baseline_recipe": "standard",
            "candidate_recipes": list(RECIPES[1:]),
        },
        "policy": policy.as_dict(),
        "coverage": {
            "expected_images": len(expected_images),
            "completed_images": len(results),
            "failed_images": len(failures),
            "artifacts_rasterized": len(results) * len(RECIPES),
        },
        "outcome": {
            "accepted_images": len(accepted),
            "accepted_fraction": len(accepted) / len(results) if results else 0.0,
            "selected_recipe_counts": dict(
                Counter(result["selection"]["selected_recipe"] for result in results)
            ),
            "median_accepted_lpips_gain": median_lpips_gain,
            "median_accepted_delta_e_p95_gain": median_delta_e_gain,
            "median_selected_size_growth_fraction": median_size_growth,
            "median_selected_render_time_growth_fraction": median_render_growth,
            "criteria": criteria,
            "go_no_go": (
                "integrate-recipe-selector" if integrate else "do-not-integrate"
            ),
        },
        "failures": list(failures),
        "results": list(results),
    }


def _markdown(summary: Mapping[str, Any]) -> str:
    outcome = summary["outcome"]
    lines = [
        "# SVG recipe gate",
        "",
        "No training was performed. Every candidate is an emitted SVG rendered",
        "by the recorded Chromium version and compared with frozen baseline ROIs.",
        "",
        f"- Verdict: **{outcome['go_no_go']}**",
        f"- Accepted images: {outcome['accepted_images']}/"
        f"{summary['coverage']['completed_images']}",
        f"- Median accepted LPIPS gain: "
        f"{outcome['median_accepted_lpips_gain']:.6f}",
        f"- Median accepted OKLab p95 gain: "
        f"{outcome['median_accepted_delta_e_p95_gain']:.6f}",
        f"- Median selected size growth: "
        f"{outcome['median_selected_size_growth_fraction']:.1%}",
        f"- Median selected capture-time growth: "
        f"{outcome['median_selected_render_time_growth_fraction']:.1%}",
        "",
        "| Image | Selected | Accepted | SSIM Δ | LPIPS Δ | Size Δ | Capture Δ |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for result in summary["results"]:
        baseline = result["recipes"]["standard"]
        selected = result["selection"]["selected"]
        delta = result["selected_deltas"]
        size_delta = (
            float(selected["file_size_bytes"]) / float(baseline["file_size_bytes"])
            - 1.0
        )
        render_delta = (
            float(selected["render_time_sec"])
            / max(float(baseline["render_time_sec"]), 1e-9)
            - 1.0
        )
        lines.append(
            f"| {result['image']} | {result['selection']['selected_recipe']} | "
            f"{'yes' if result['selection']['accepted_candidate'] else 'no'} | "
            f"{float(delta['ssim_srgb'] or 0.0):+.5f} | "
            f"{float(delta['lpips'] or 0.0):+.5f} | {size_delta:+.1%} | "
            f"{render_delta:+.1%} |"
        )
    lines.append("")
    return "\n".join(lines)


def _html(summary: Mapping[str, Any], output_dir: Path) -> str:
    outcome = summary["outcome"]
    rows = []
    cards = []
    for result in summary["results"]:
        baseline = result["recipes"]["standard"]
        selected = result["selection"]["selected"]
        delta = result["selected_deltas"]
        size_delta = (
            float(selected["file_size_bytes"]) / float(baseline["file_size_bytes"])
            - 1.0
        )
        render_delta = (
            float(selected["render_time_sec"])
            / max(float(baseline["render_time_sec"]), 1e-9)
            - 1.0
        )
        rows.append(
            "<tr>"
            f"<td>{html.escape(result['image'])}</td>"
            f"<td>{html.escape(result['selection']['selected_recipe'])}</td>"
            f"<td>{'yes' if result['selection']['accepted_candidate'] else 'no'}</td>"
            f"<td>{float(delta['ssim_srgb'] or 0.0):+.5f}</td>"
            f"<td>{float(delta['lpips'] or 0.0):+.5f}</td>"
            f"<td>{size_delta:+.1%}</td><td>{render_delta:+.1%}</td>"
            "</tr>"
        )
        source = REPO / result["source"]
        panels = [
            "<figure>"
            f'<img src="{html.escape(_relative_url(source, output_dir))}" '
            f'alt="{html.escape(result["image"])} source">'
            "<figcaption><b>source PNG</b><br>full-frame target</figcaption>"
            "</figure>"
        ]
        for recipe in RECIPES:
            measurement = result["recipes"][recipe]
            raster = REPO / measurement["raster"]
            svg = REPO / measurement["svg"]
            selected_class = (
                " selected" if recipe == result["selection"]["selected_recipe"] else ""
            )
            panels.append(
                f'<figure class="{selected_class.strip()}">'
                f'<a href="{html.escape(_relative_url(svg, output_dir))}">'
                f'<img src="{html.escape(_relative_url(raster, output_dir))}" '
                f'alt="{html.escape(result["image"])} {html.escape(recipe)}"></a>'
                f"<figcaption><b>{html.escape(recipe)}</b><br>"
                f"SSIM {_format_metric(measurement['ssim_srgb'])} · "
                f"LPIPS {_format_metric(measurement['lpips'])}<br>"
                f"{measurement['file_size_bytes'] / 1024:.1f} KiB · "
                f"{measurement['render_time_sec'] * 1000:.1f} ms capture</figcaption>"
                "</figure>"
            )
        cards.append(
            "<section class=card>"
            f"<h2>{html.escape(result['image'])} — selected "
            f"{html.escape(result['selection']['selected_recipe'])}</h2>"
            '<div class="panels">' + "".join(panels) + "</div></section>"
        )
    criteria = "".join(
        f"<li class={'pass' if passed else 'fail'}>{html.escape(name)}: "
        f"{'pass' if passed else 'fail'}</li>"
        for name, passed in outcome["criteria"].items()
    )
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>SVG recipe gate</title>
<style>
body{{margin:0;background:#101216;color:#e8ecf1;font:14px system-ui,sans-serif}}
main{{max-width:1500px;margin:auto;padding:24px}} h1,h2{{margin:.2em 0 .6em}}
.summary,.card{{background:#191d24;border:1px solid #303744;border-radius:12px;padding:18px;margin:16px 0}}
table{{border-collapse:collapse;width:100%}} th,td{{padding:7px;border-bottom:1px solid #303744;text-align:right}}
th:first-child,td:first-child{{text-align:left}} .panels{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}}
figure{{margin:0;background:#0b0d10;padding:10px;border:2px solid transparent;border-radius:8px}} figure.selected{{border-color:#72d790}} img{{width:100%;height:auto;display:block}}
figcaption{{padding-top:8px;color:#b8c1cc}} .pass{{color:#72d790}} .fail{{color:#ff8d8d}} a{{color:inherit}}
@media(max-width:800px){{.panels{{grid-template-columns:1fr}}}}
</style></head><body><main>
<h1>Full-corpus SVG recipe gate</h1>
<div class="summary"><p><b>{html.escape(outcome['go_no_go'])}</b> ·
{outcome['accepted_images']}/{summary['coverage']['completed_images']} accepted ·
renderer {html.escape(summary['evidence']['renderer'])}</p><ul>{criteria}</ul></div>
<div class="summary"><table><thead><tr><th>Image</th><th>Selected</th><th>Accepted</th><th>SSIM Δ</th><th>LPIPS Δ</th><th>Size Δ</th><th>Capture Δ</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></div>
{''.join(cards)}
</main></body></html>"""


def _parse_csv(value: Optional[str]) -> Optional[set[str]]:
    if value is None:
        return None
    return {part.strip() for part in value.split(",") if part.strip()}


def main() -> int:  # noqa: C901 - orchestration keeps one auditable CLI path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--only", help="comma-separated corpus image names")
    parser.add_argument("--render-repeats", type=int, default=3)
    parser.add_argument("--minimum-accepted-images", type=int, default=5)
    parser.add_argument(
        "--browser-executable", type=Path, default=resolve_browser_executable()
    )
    parser.add_argument("--timeout-ms", type=int, default=120_000)
    args = parser.parse_args()
    if args.render_repeats < 1:
        parser.error("--render-repeats must be positive")
    if args.minimum_accepted_images < 1:
        parser.error("--minimum-accepted-images must be positive")

    corpus_root = args.corpus_root.resolve()
    output_dir = args.output_dir.resolve()
    if args.browser_executable is None:
        parser.error(
            "installed Chrome not found; set --browser-executable or "
            "SPLATTHIS_BROWSER_EXECUTABLE"
        )
    browser_executable = args.browser_executable.expanduser().resolve()
    if not browser_executable.is_file():
        parser.error(f"browser executable not found: {browser_executable}")
    metadata = json.loads((corpus_root / "corpus.json").read_text())["images"]
    requested = _parse_csv(args.only)
    if requested is not None:
        unknown = requested - set(metadata)
        if unknown:
            parser.error(f"unknown corpus images: {', '.join(sorted(unknown))}")
    selected_images = sorted(requested if requested is not None else metadata)
    policy = replace(
        SvgRecipeGatePolicy(), minimum_accepted_images=args.minimum_accepted_images
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    failures = []
    try:
        with PlaywrightSvgRenderer(
            browser_executable=browser_executable,
            timeout_ms=args.timeout_ms,
        ) as renderer:
            browser_version = renderer.browser_version
            renderer_version = f"Chromium {browser_version}"
            for index, image in enumerate(selected_images, 1):
                print(
                    f"[{index}/{len(selected_images)}] {image} ... ",
                    end="",
                    flush=True,
                )
                source = corpus_root / str(metadata[image]["path"])
                artifacts = corpus_root / "runs" / f"{image}_svg_s0_art"
                raw = artifacts / "final.raw.json"
                manifest = artifacts / "run_manifest.json"
                try:
                    if (
                        not source.is_file()
                        or not raw.is_file()
                        or not manifest.is_file()
                    ):
                        raise FileNotFoundError(
                            "source, raw splats, or manifest missing"
                        )
                    result = _evaluate_image(
                        image=image,
                        source_path=source,
                        raw_path=raw,
                        manifest_path=manifest,
                        output_dir=output_dir,
                        renderer=renderer,
                        render_repeats=args.render_repeats,
                        renderer_version=renderer_version,
                        policy=policy,
                    )
                    results.append(result)
                    print(result["selection"]["selected_recipe"])
                except Exception as exc:
                    failures.append(
                        {"image": image, "error": f"{type(exc).__name__}: {exc}"}
                    )
                    print("FAILED")
    except Exception as exc:
        parser.error(f"could not start Chromium capture: {exc}")
    summary = _finite(
        _summarize(
            expected_images=selected_images,
            results=results,
            failures=failures,
            policy=policy,
            browser_version=browser_version,
            browser_executable=browser_executable,
            render_repeats=args.render_repeats,
        )
    )
    _write_json(output_dir / "summary.json", summary)
    atomic_write_text(output_dir / "report.md", _markdown(summary))
    atomic_write_text(output_dir / "index.html", _html(summary, output_dir))
    print(f"wrote {_display_path(output_dir / 'summary.json')}")
    print(f"wrote {_display_path(output_dir / 'index.html')}")
    print(
        f"{summary['outcome']['accepted_images']}/{len(results)} accepted; "
        f"{summary['outcome']['go_no_go']}"
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
