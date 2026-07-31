#!/usr/bin/env python3
"""Run the bounded top-K teacher/student MVP on one source image."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from png2svg_gs.browser_capture import render_svg_in_browser_to_linear_rgb
from png2svg_gs.converter import PNG2SVGConverter
from png2svg_gs.distillation import run_distillation_mvp, summarize_mvp_metrics
from png2svg_gs.io import (
    compute_quality_metrics,
    load_png,
    load_splats_json,
    save_pptx_with_splats,
    save_splats_json,
    save_svg,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("./tmp/topk-mvp"))
    parser.add_argument("--max-edge", type=int, default=96)
    parser.add_argument("--splats", type=int, default=128)
    parser.add_argument(
        "--initial-splats-json",
        type=Path,
        default=None,
        help="Start all three arms from an existing converged splat checkpoint "
        "instead of constructing a fresh initialization.",
    )
    parser.add_argument("--teacher-iters", type=int, default=40)
    parser.add_argument("--student-iters", type=int, default=40)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--handoff-mode", choices=["full", "color-only"], default="full"
    )
    parser.add_argument("--teacher-weight", type=float, default=0.25)
    parser.add_argument("--teacher-exportability-weight", type=float, default=0.0)
    parser.add_argument(
        "--svg-postfit-iters",
        type=int,
        default=0,
        help="Apply the existing SVG color/alpha proxy post-fit equally to direct "
        "and student artifacts before emission.",
    )
    parser.add_argument("--min-svg-ssim-gain", type=float, default=0.01)
    parser.add_argument(
        "--constant-teacher-weight",
        action="store_true",
        help="Keep teacher guidance constant instead of decaying it to zero.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "mps"],
        help="Torch execution device. Use mps for Apple Metal.",
    )
    parser.add_argument(
        "--optimizer-backend",
        default="torch",
        choices=["torch", "mlx"],
        help="Optimization implementation for all three arms.",
    )
    parser.add_argument(
        "--renderer-backend",
        default="torch",
        choices=["torch", "torch-batched"],
        help="Differentiable renderer used by all three equal-budget arms.",
    )
    parser.add_argument("--tile-size", type=int, default=32)
    parser.add_argument("--batch-tile-count", type=int, default=16)
    parser.add_argument(
        "--mlx-tile-plan-rebuild-interval",
        type=int,
        default=1,
        help="Rebuild MLX tile membership every N optimizer steps. The "
        "distillation parity default is 1 because geometry is trainable.",
    )
    return parser


def main() -> int:
    run_t0 = time.perf_counter()
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with Image.open(args.input) as source:
        source_w, source_h = source.size
    scale = min(1.0, float(args.max_edge) / max(source_w, source_h))
    target_size = (
        max(1, int(round(source_w * scale))),
        max(1, int(round(source_h * scale))),
    )
    target = load_png(str(args.input), target_size=target_size)[..., :3]
    height, width = target.shape[:2]

    converter = PNG2SVGConverter(
        max_splats=args.splats,
        stages=[0],
        target_size=(width, height),
        quality_profile="fast",
        device=args.device,
        seed=args.seed,
        refinement_config={
            "initial_splat_fraction": 1.0,
            "initial_splat_cap": args.splats,
        },
        apple_silicon_splat_cap=None,
    )
    converter._background_linear_rgb = converter._estimate_background_color(target)
    if args.svg_postfit_iters > 0:
        guidance = converter._compute_region_guidance(target)
        converter._region_weight_map = guidance["weight_map"]
        converter._region_saliency_map = guidance.get("saliency_map")
        converter._region_detail_priority_map = guidance.get("detail_priority_map")
        converter._region_background_penalty_map = guidance.get(
            "background_penalty_map"
        )
        converter._region_foreground_mask = guidance["foreground_mask"]
        converter._region_background_safe_mask = guidance["background_safe_mask"]
        converter._region_edge_band_mask = guidance["edge_band_mask"]
        converter._background_linear_rgb = guidance["background_linear_rgb"]
    initial = (
        load_splats_json(str(args.initial_splats_json))
        if args.initial_splats_json is not None
        else converter._initialize_splats(target, rng=np.random.default_rng(args.seed))
    )

    setup_elapsed = time.perf_counter() - run_t0
    distillation_t0 = time.perf_counter()
    result = run_distillation_mvp(
        initial,
        target,
        teacher_iterations=args.teacher_iters,
        student_iterations=args.student_iters,
        normalized_top_k=args.top_k,
        teacher_weight=args.teacher_weight,
        decay_teacher_weight=not args.constant_teacher_weight,
        teacher_exportability_weight=args.teacher_exportability_weight,
        handoff_mode=args.handoff_mode,
        device=args.device,
        renderer_backend=args.renderer_backend,
        tile_size=args.tile_size,
        batch_tile_count=args.batch_tile_count,
        background_linear_rgb=converter._background_linear_rgb,
        optimization_backend=args.optimizer_backend,
        mlx_tile_plan_rebuild_interval=args.mlx_tile_plan_rebuild_interval,
    )
    distillation_elapsed = time.perf_counter() - distillation_t0
    proxy_t0 = time.perf_counter()
    proxy_metrics = summarize_mvp_metrics(result, target)
    proxy_elapsed = time.perf_counter() - proxy_t0

    stem = args.input.stem
    records = {
        "input": str(args.input),
        "size": [width, height],
        "seed": args.seed,
        "splats": len(initial),
        "initial_splats_json": (
            None if args.initial_splats_json is None else str(args.initial_splats_json)
        ),
        "background_linear_rgb": [
            float(value) for value in converter._background_linear_rgb
        ],
        "top_k": args.top_k,
        "handoff_mode": args.handoff_mode,
        "teacher_weight": args.teacher_weight,
        "teacher_weight_schedule": (
            "constant" if args.constant_teacher_weight else "linear-decay"
        ),
        "teacher_exportability_weight": args.teacher_exportability_weight,
        "svg_postfit_iterations": args.svg_postfit_iters,
        "min_svg_ssim_gain": args.min_svg_ssim_gain,
        "teacher_iterations": args.teacher_iters,
        "student_iterations": args.student_iters,
        "device": args.device,
        "optimizer_backend": args.optimizer_backend,
        "renderer_backend": (
            "mlx-batched" if args.optimizer_backend == "mlx" else args.renderer_backend
        ),
        "tile_size": args.tile_size,
        "batch_tile_count": args.batch_tile_count,
        "mlx_tile_plan_rebuild_interval": args.mlx_tile_plan_rebuild_interval,
        "proxy": proxy_metrics,
        "runner_timings_sec": {
            "setup": float(setup_elapsed),
            "distillation": float(distillation_elapsed),
            "proxy_metrics": float(proxy_elapsed),
            "artifacts": {},
        },
        "timings_sec": {
            "direct": float(result.direct.elapsed_sec),
            "teacher": float(result.teacher.elapsed_sec),
            "student": float(result.student.elapsed_sec),
            "total": float(
                result.direct.elapsed_sec
                + result.teacher.elapsed_sec
                + result.student.elapsed_sec
            ),
        },
        "losses": {
            arm_name: {
                "start": float(getattr(result, arm_name).start_loss),
                "end": float(getattr(result, arm_name).end_loss),
            }
            for arm_name in ("direct", "teacher", "student")
        },
        "artifacts": {},
    }
    for arm_name in ("direct", "student"):
        arm = getattr(result, arm_name)
        export_splats = arm.splats
        arm_timings = {}
        postfit_metrics = None
        raw_path = args.output_dir / f"{stem}_{arm_name}.raw.json"
        save_splats_json(arm.splats, str(raw_path))
        if args.svg_postfit_iters > 0:
            postfit_t0 = time.perf_counter()
            export_splats, postfit_metrics = converter._postfit_splats_for_svg_proxy(
                export_splats,
                target,
                width,
                height,
                num_iters=args.svg_postfit_iters,
                verbose=False,
            )
            arm_timings["svg_postfit"] = float(time.perf_counter() - postfit_t0)
        export_raw_path = args.output_dir / f"{stem}_{arm_name}_export.raw.json"
        save_splats_json(export_splats, str(export_raw_path))
        svg_path = args.output_dir / f"{stem}_{arm_name}.svg"
        pptx_path = args.output_dir / f"{stem}_{arm_name}.pptx"
        svg_t0 = time.perf_counter()
        save_svg(
            export_splats,
            width,
            height,
            str(svg_path),
            background_linear_rgb=converter._background_linear_rgb,
        )
        arm_timings["emit_svg"] = float(time.perf_counter() - svg_t0)
        pptx_t0 = time.perf_counter()
        save_pptx_with_splats(
            arm.splats,
            width,
            height,
            str(pptx_path),
            background_linear_rgb=converter._background_linear_rgb,
            splat_style="gradient",
        )
        arm_timings["emit_pptx"] = float(time.perf_counter() - pptx_t0)
        raster_t0 = time.perf_counter()
        svg_render, renderer = render_svg_in_browser_to_linear_rgb(
            str(svg_path), width, height
        )
        arm_timings["rasterize_svg"] = float(time.perf_counter() - raster_t0)
        records["runner_timings_sec"]["artifacts"][arm_name] = arm_timings
        records["artifacts"][arm_name] = {
            "svg": str(svg_path),
            "svg_bytes": svg_path.stat().st_size,
            "raw_splats": str(raw_path),
            "export_raw_splats": str(export_raw_path),
            "pptx": str(pptx_path),
            "pptx_bytes": pptx_path.stat().st_size,
            "actual_svg_renderer": renderer,
            "svg_postfit": postfit_metrics,
            "actual_svg": compute_quality_metrics(target, svg_render),
            "actual_pptx": None,
        }

    direct_svg = records["artifacts"]["direct"]["actual_svg"]
    student_svg = records["artifacts"]["student"]["actual_svg"]
    teacher_advantage = float(
        proxy_metrics["teacher"]["ssim_srgb"] - proxy_metrics["direct"]["ssim_srgb"]
    )
    actual_gain = float(student_svg["ssim_srgb"] - direct_svg["ssim_srgb"])
    psnr_regression = float(direct_svg["psnr_srgb"] - student_svg["psnr_srgb"])
    accepted = bool(
        teacher_advantage > 0.0
        and actual_gain >= float(args.min_svg_ssim_gain)
        and psnr_regression <= 0.1
    )
    if teacher_advantage <= 0.0:
        decision_reason = "teacher ceiling did not beat direct proxy"
    elif actual_gain < float(args.min_svg_ssim_gain):
        decision_reason = "actual SVG SSIM gain below threshold"
    elif psnr_regression > 0.1:
        decision_reason = "actual SVG PSNR regressed"
    else:
        decision_reason = "student cleared SVG MVP gates"
    records["decision"] = {
        "accepted": accepted,
        "winner": "student" if accepted else "direct",
        "reason": decision_reason,
        "teacher_proxy_ssim_advantage": teacher_advantage,
        "actual_svg_ssim_gain": actual_gain,
        "powerpoint_capture_required": accepted,
    }
    records["runner_timings_sec"]["total_before_write"] = float(
        time.perf_counter() - run_t0
    )

    output_json = args.output_dir / f"{stem}_comparison.json"
    output_json.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")
    print(json.dumps(records, indent=2, sort_keys=True))
    print(
        "PPTX files were emitted but not scored: capture them in real Microsoft "
        "PowerPoint before filling actual_pptx."
    )
    print(f"wrote {output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
