#!/usr/bin/env python3
"""
Fast fidelity iteration harness for PNG->splat tuning.

Runs multiple converter trials on:
1) a downscaled full image
2) a high-detail crop selected by gradient energy

It outputs a compact leaderboard so algorithm tweaks can be compared in minutes
before committing to expensive full-resolution runs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image

from png2svg_gs.converter import PNG2SVGConverter


def _parse_stages(value: str) -> List[int]:
    items = [v.strip() for v in str(value).split(",") if v.strip()]
    if not items:
        raise ValueError("stages cannot be empty")
    return [int(v) for v in items]


def _resize_longest_edge(image: Image.Image, longest_edge: int) -> Image.Image:
    width, height = image.size
    if max(width, height) <= longest_edge:
        return image.copy()
    if width >= height:
        new_w = int(longest_edge)
        new_h = int(round(height * (new_w / width)))
    else:
        new_h = int(longest_edge)
        new_w = int(round(width * (new_h / height)))
    return image.resize((max(new_w, 1), max(new_h, 1)), Image.Resampling.LANCZOS)


def _gradient_energy(gray: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(gray.astype(np.float32))
    return np.sqrt(gx * gx + gy * gy)


def _best_detail_crop_box(image: Image.Image, crop_size: int) -> Tuple[int, int, int, int]:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    energy = _gradient_energy(gray)

    h, w = energy.shape
    crop_w = min(crop_size, w)
    crop_h = min(crop_size, h)
    if crop_w == w and crop_h == h:
        return (0, 0, w, h)

    # Integral image for fast box sums.
    integral = np.pad(np.cumsum(np.cumsum(energy, axis=0), axis=1), ((1, 0), (1, 0)))

    def box_sum(x0: int, y0: int, x1: int, y1: int) -> float:
        return float(integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0])

    best_score = -1.0
    best_x = 0
    best_y = 0
    step = max(1, min(crop_w, crop_h) // 10)
    max_x = w - crop_w
    max_y = h - crop_h
    for y in range(0, max_y + 1, step):
        for x in range(0, max_x + 1, step):
            score = box_sum(x, y, x + crop_w, y + crop_h)
            if score > best_score:
                best_score = score
                best_x = x
                best_y = y
    return (best_x, best_y, best_x + crop_w, best_y + crop_h)


def _default_trials(args: argparse.Namespace) -> List[Dict[str, Any]]:
    base = {
        "max_splats": int(args.max_splats),
        "stages": _parse_stages(args.stages),
        "quality_profile": args.quality_profile,
        "blend_mode": args.blend_mode,
        "seed": int(args.seed),
        "device": "cpu",
    }
    return [
        {
            "name": "baseline",
            "converter": dict(base),
            "refinement_config": {},
            "learning_rates": {},
        },
        {
            "name": "detail_bias",
            "converter": dict(base),
            "refinement_config": {
                "densify_weight_error": 0.46,
                "densify_weight_uncovered": 0.44,
                "densify_weight_edge": 0.10,
                "sigma_max": 2.6,
            },
            "learning_rates": {},
        },
        {
            "name": "edge_bias",
            "converter": dict(base),
            "refinement_config": {
                "densify_weight_error": 0.38,
                "densify_weight_uncovered": 0.44,
                "densify_weight_edge": 0.18,
                "sigma_max": 2.4,
            },
            "learning_rates": {},
        },
    ]


def _load_trials(args: argparse.Namespace) -> List[Dict[str, Any]]:
    if args.trials_file:
        data = json.loads(Path(args.trials_file).read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("trials file must contain a JSON list")
        return data
    return _default_trials(args)


def _run_case(
    case_name: str,
    input_path: Path,
    output_dir: Path,
    trial: Dict[str, Any],
) -> Dict[str, float]:
    trial_name = str(trial.get("name", "trial"))
    case_dir = output_dir / trial_name / case_name
    case_dir.mkdir(parents=True, exist_ok=True)

    svg_path = case_dir / "output.svg"
    artifacts_dir = case_dir / "artifacts"
    preview_png = case_dir / "preview.png"

    converter_cfg = dict(trial.get("converter", {}))
    refinement_cfg = dict(trial.get("refinement_config", {}))
    learning_rates = dict(trial.get("learning_rates", {}))

    converter = PNG2SVGConverter(
        max_splats=int(converter_cfg.get("max_splats", 500)),
        stages=list(converter_cfg.get("stages", [2, 1, 1])),
        quality_profile=str(converter_cfg.get("quality_profile", "max-fidelity")),
        blend_mode=str(converter_cfg.get("blend_mode", "alpha-over")),
        seed=int(converter_cfg.get("seed", 7)),
        device=str(converter_cfg.get("device", "cpu")),
        refinement_config=refinement_cfg or None,
        learning_rates=learning_rates or None,
    )

    converter.convert(
        input_path=str(input_path),
        output_path=str(svg_path),
        output_format="svg",
        save_json=True,
        verbose=False,
        artifacts_dir=str(artifacts_dir),
        preview_png_path=str(preview_png),
    )

    manifest_path = artifacts_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metrics = dict(manifest.get("final_metrics", {}))
    metrics["runtime_sec"] = float(metrics.get("runtime_sec", 0.0))
    metrics["psnr"] = float(metrics.get("psnr", 0.0))
    metrics["ssim"] = float(metrics.get("ssim", 0.0))
    metrics["coverage"] = float(metrics.get("coverage", 0.0))
    return metrics


def _score(psnr: float, ssim: float, coverage: float, runtime_sec: float) -> float:
    psnr_term = min(psnr / 30.0, 1.0)
    coverage_term = min(coverage, 1.0)
    runtime_penalty = min(runtime_sec / 120.0, 1.0)
    return 0.42 * psnr_term + 0.42 * ssim + 0.22 * coverage_term - 0.06 * runtime_penalty


def main() -> int:
    parser = argparse.ArgumentParser(description="Fast PNG2splat fidelity iteration harness")
    parser.add_argument("input", help="Input image path")
    parser.add_argument("--output-dir", default="fidelity_lab", help="Output directory")
    parser.add_argument("--longest-edge", type=int, default=224, help="Downscaled full-image longest edge")
    parser.add_argument("--crop-size", type=int, default=192, help="High-detail crop size")
    parser.add_argument("--max-splats", type=int, default=500, help="Trial max splats")
    parser.add_argument("--stages", type=str, default="2,1,1", help="Trial stage schedule")
    parser.add_argument("--quality-profile", type=str, default="max-fidelity", help="Converter quality profile")
    parser.add_argument("--blend-mode", type=str, default="alpha-over", help="Converter blend mode")
    parser.add_argument("--seed", type=int, default=7, help="Deterministic seed")
    parser.add_argument("--trials-file", type=str, default="", help="Optional JSON file with trial definitions")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    source = Image.open(input_path).convert("RGB")
    full_small = _resize_longest_edge(source, int(args.longest_edge))
    crop_box = _best_detail_crop_box(full_small, int(args.crop_size))
    detail_crop = full_small.crop(crop_box)

    full_small_path = output_dir / "case_full_small.png"
    detail_crop_path = output_dir / "case_detail_crop.png"
    full_small.save(full_small_path)
    detail_crop.save(detail_crop_path)

    trials = _load_trials(args)
    results: List[Dict[str, Any]] = []
    for trial in trials:
        trial_name = str(trial.get("name", f"trial_{len(results)}"))
        full_metrics = _run_case("full_small", full_small_path, output_dir, trial)
        crop_metrics = _run_case("detail_crop", detail_crop_path, output_dir, trial)

        combined_psnr = 0.55 * full_metrics["psnr"] + 0.45 * crop_metrics["psnr"]
        combined_ssim = 0.55 * full_metrics["ssim"] + 0.45 * crop_metrics["ssim"]
        combined_coverage = 0.55 * full_metrics["coverage"] + 0.45 * crop_metrics["coverage"]
        combined_runtime = full_metrics["runtime_sec"] + crop_metrics["runtime_sec"]
        combined_score = _score(combined_psnr, combined_ssim, combined_coverage, combined_runtime)

        results.append(
            {
                "name": trial_name,
                "full_small": full_metrics,
                "detail_crop": crop_metrics,
                "aggregate": {
                    "psnr": combined_psnr,
                    "ssim": combined_ssim,
                    "coverage": combined_coverage,
                    "runtime_sec": combined_runtime,
                    "score": combined_score,
                },
            }
        )

    results.sort(key=lambda r: r["aggregate"]["score"], reverse=True)

    out_json = output_dir / "leaderboard.json"
    out_json.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# Fidelity Lab Leaderboard",
        "",
        "| Trial | Score | PSNR | SSIM | Coverage | Runtime (s) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        agg = row["aggregate"]
        lines.append(
            "| {name} | {score:.4f} | {psnr:.3f} | {ssim:.3f} | {coverage:.3f} | {runtime:.2f} |".format(
                name=row["name"],
                score=agg["score"],
                psnr=agg["psnr"],
                ssim=agg["ssim"],
                coverage=agg["coverage"],
                runtime=agg["runtime_sec"],
            )
        )
    out_md = output_dir / "leaderboard.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"wrote {out_json}")
    print(f"wrote {out_md}")
    if results:
        best = results[0]
        agg = best["aggregate"]
        print(
            "best={name} score={score:.4f} psnr={psnr:.3f} ssim={ssim:.3f} coverage={cov:.3f} runtime={rt:.2f}s".format(
                name=best["name"],
                score=agg["score"],
                psnr=agg["psnr"],
                ssim=agg["ssim"],
                cov=agg["coverage"],
                rt=agg["runtime_sec"],
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

