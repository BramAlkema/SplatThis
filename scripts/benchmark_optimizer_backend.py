#!/usr/bin/env python
"""Benchmark experimental optimizer backends on a fixed raw-splat JSON file."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from png2svg_gs.io import (  # noqa: E402
    compute_quality_metrics,
    load_png,
    load_splats_json,
)
from png2svg_gs.mlx_losses import MlxLossConfig  # noqa: E402
from png2svg_gs.mlx_renderer import (  # noqa: E402
    MlxBatchedGaussianRenderer,
    splats_to_numpy_table,
)
from png2svg_gs.mlx_stage import (  # noqa: E402
    MlxRendererConfig,
    MlxStageConfig,
    is_mlx_available,
    optimize_stage_mlx,
)


def _parse_csv(text: str) -> List[str]:
    return [item.strip() for item in str(text).split(",") if item.strip()]


def _parse_background(text: str) -> Tuple[float, float, float]:
    parts = [float(item.strip()) for item in str(text).split(",") if item.strip()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "--background must have exactly three comma values"
        )
    return tuple(float(np.clip(v, 0.0, 1.0)) for v in parts)  # type: ignore[return-value]


def _display_path(path: Path) -> str:
    text = str(path)
    if path.is_absolute() or text.startswith("."):
        return text
    return f"./{text}"


def _infer_dimensions(
    image_path: Path,
    width: Optional[int],
    height: Optional[int],
    max_edge: Optional[int],
) -> Tuple[int, int]:
    if width is not None and height is not None:
        return int(width), int(height)
    with Image.open(image_path) as image:
        src_w, src_h = image.size
    if max_edge is None:
        return int(width or src_w), int(height or src_h)
    scale = float(max_edge) / float(max(src_w, src_h))
    inferred_w = max(1, int(round(src_w * scale)))
    inferred_h = max(1, int(round(src_h * scale)))
    return int(width or inferred_w), int(height or inferred_h)


def _run_mlx(args: argparse.Namespace, width: int, height: int) -> Dict[str, Any]:
    if not is_mlx_available():
        return {
            "backend": "mlx",
            "status": "skipped",
            "reason": "MLX is not installed",
        }

    splats = load_splats_json(str(args.splats_json))
    target = load_png(str(args.input), target_size=(width, height))[:, :, :3]
    spatial_weights = None
    if args.weight_mode == "center":
        y = np.linspace(-1.0, 1.0, height, dtype=np.float32).reshape(-1, 1)
        x = np.linspace(-1.0, 1.0, width, dtype=np.float32).reshape(1, -1)
        spatial_weights = 0.25 + 0.75 * np.exp(-2.5 * (x * x + y * y)).astype(
            np.float32
        )
    elif args.weight_mode == "uniform":
        spatial_weights = np.ones((height, width), dtype=np.float32)
    import mlx.core as mx

    table = splats_to_numpy_table(splats)
    start_renderer = MlxBatchedGaussianRenderer(
        width=width,
        height=height,
        tile_size=args.tile_size,
        batch_tile_count=args.batch_tile_count,
        blend_mode=args.blend_mode,
        background_color=args.background,
        culling_sigma=args.culling_sigma,
        max_active_splats_per_tile=args.max_active_splats_per_tile,
    )
    start_rendered = start_renderer.render(table, plan=start_renderer.build_plan(table))
    mx.eval(start_rendered)
    start_quality = compute_quality_metrics(
        target, np.asarray(start_rendered, dtype=np.float32)
    )
    config = MlxStageConfig(
        renderer=MlxRendererConfig(
            tile_size=args.tile_size,
            batch_tile_count=args.batch_tile_count,
            blend_mode=args.blend_mode,
            background_color=args.background,
            culling_sigma=args.culling_sigma,
            max_active_splats_per_tile=args.max_active_splats_per_tile,
        ),
        loss=MlxLossConfig(args.loss),
        trainable_groups=tuple(_parse_csv(args.trainable_groups)),
        grad_clip_norm=args.grad_clip_norm,
        tile_plan_mode=args.tile_plan,
        tile_plan_rebuild_interval=args.tile_plan_rebuild_interval,
        progress_interval=args.progress_interval,
    )
    started = time.perf_counter()
    result = optimize_stage_mlx(
        splats,
        target,
        width,
        height,
        args.iters,
        config=config,
        spatial_weight_map=spatial_weights,
        verbose=bool(args.verbose),
    )
    wall_sec = time.perf_counter() - started
    quality = compute_quality_metrics(target, result.rendered_linear_rgb)
    row: Dict[str, Any] = {
        "backend": "mlx",
        "status": "ok",
        "width": width,
        "height": height,
        "num_splats": len(splats),
        "wall_sec": float(wall_sec),
        "start_quality": start_quality,
        "quality": quality,
        "quality_delta": {
            "l1": float(quality["l1"] - start_quality["l1"]),
            "ssim_srgb": float(quality["ssim_srgb"] - start_quality["ssim_srgb"]),
            "psnr_srgb": float(quality["psnr_srgb"] - start_quality["psnr_srgb"]),
        },
    }
    row.update(result.metrics)
    return row


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--splats-json", required=True, type=Path)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--max-edge", type=int)
    parser.add_argument("--backend", default="mlx", choices=["mlx"])
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument(
        "--loss",
        default="linear-l1",
        choices=["linear-l1", "oklab-l1", "weighted-oklab-l1"],
    )
    parser.add_argument(
        "--weight-mode", default="none", choices=["none", "uniform", "center"]
    )
    parser.add_argument("--trainable-groups", default="color,alpha")
    parser.add_argument("--tile-plan", default="static", choices=["static", "periodic"])
    parser.add_argument("--tile-plan-rebuild-interval", type=int, default=10)
    parser.add_argument("--tile-size", type=int, default=16)
    parser.add_argument("--batch-tile-count", type=int, default=16)
    parser.add_argument(
        "--blend-mode", default="alpha-over", choices=["alpha-over", "weighted"]
    )
    parser.add_argument("--background", type=_parse_background, default=(0.0, 0.0, 0.0))
    parser.add_argument("--culling-sigma", type=float, default=3.0)
    parser.add_argument("--max-active-splats-per-tile", type=int)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--progress-interval", type=int, default=0)
    parser.add_argument(
        "--out", type=Path, default=Path("./tmp/optimizer_backend_benchmark.json")
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    width, height = _infer_dimensions(
        args.input, args.width, args.height, args.max_edge
    )

    if args.backend == "mlx":
        result = _run_mlx(args, width, height)
    else:  # pragma: no cover - argparse constrains this today.
        result = {
            "backend": args.backend,
            "status": "skipped",
            "reason": "unknown backend",
        }

    payload = {
        "input": _display_path(args.input),
        "splats_json": _display_path(args.splats_json),
        "python": sys.version,
        "platform": platform.platform(),
        "result": result,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

    if result.get("status") == "ok":
        print(
            "mlx optimizer: "
            f"iters={result['iterations']} "
            f"loss {result['start_loss']:.6f}->{result['best_loss']:.6f} "
            f"median_iter={result['median_iter_sec'] * 1000.0:.1f}ms "
            f"ssim_srgb={result['start_quality']['ssim_srgb']:.4f}->{result['quality']['ssim_srgb']:.4f}"
        )
    else:
        print(
            f"{result.get('backend', args.backend)} optimizer: {result.get('status')} {result.get('reason', '')}"
        )
    print(f"wrote {_display_path(args.out)}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
