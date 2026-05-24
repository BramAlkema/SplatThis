"""Command-line interface for the png2svg_gs PNG->SVG Gaussian-splatting pipeline."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image

from .converter import TIME_BUDGET_ALIASES, TIME_BUDGET_PRESETS, PNG2SVGConverter

DEFAULT_MAX_SPLATS = 2000
DEFAULT_APPLE_SILICON_SPLAT_CAP = 2000
DISABLE_APPLE_SILICON_SPLAT_CAP = 0


def _parse_stages(text: str) -> List[int]:
    try:
        stages = [int(p) for p in text.split(",") if p.strip() != ""]
    except ValueError as exc:  # pragma: no cover - argparse surfaces the message
        raise argparse.ArgumentTypeError(f"invalid --stages '{text}': {exc}") from exc
    if not stages or any(s <= 0 for s in stages):
        raise argparse.ArgumentTypeError(
            "--stages must be positive integers, e.g. 200,150,100,50"
        )
    return stages


def _target_size(input_path: str, max_edge: Optional[int]) -> Optional[Tuple[int, int]]:
    """Map a longest-edge cap to a (width, height) target, preserving aspect."""
    if not max_edge or max_edge <= 0:
        return None
    with Image.open(input_path) as img:
        w, h = img.size
    if max(w, h) <= max_edge:
        return None
    scale = max_edge / float(max(w, h))
    return max(1, round(w * scale)), max(1, round(h * scale))


def _normalize_time_budget_label(time_budget: Optional[str]) -> Optional[str]:
    if time_budget is None:
        return None
    key = str(time_budget).strip().lower().replace("_", "-")
    return TIME_BUDGET_ALIASES.get(key, key)


def _preset_exact_splat_count(time_budget: Optional[str]) -> Optional[int]:
    normalized = _normalize_time_budget_label(time_budget)
    if normalized is None:
        return None
    preset = TIME_BUDGET_PRESETS.get(normalized)
    if not preset:
        return None
    preset_cap = preset.get("max_splats")
    if preset_cap is None:
        return None
    min_splats = int(preset.get("min_splats", 0))
    max_splats = int(preset_cap)
    if min_splats == max_splats:
        return max_splats
    return None


def _resolve_cli_resource_limits(
    time_budget: Optional[str],
    splats: Optional[int],
    apple_silicon_splat_cap: Optional[int],
) -> Tuple[int, Optional[int]]:
    exact_splats = _preset_exact_splat_count(time_budget)
    max_splats = (
        int(splats) if splats is not None else int(exact_splats or DEFAULT_MAX_SPLATS)
    )

    if apple_silicon_splat_cap == DISABLE_APPLE_SILICON_SPLAT_CAP:
        resolved_cap = None
    elif apple_silicon_splat_cap is not None:
        resolved_cap = int(apple_silicon_splat_cap)
    elif exact_splats is not None:
        # Exact-count photo presets are opt-in long runs. Do not let the safety
        # cap silently collapse photo-10k/photo-20k back to the interactive 2k
        # ceiling unless the user explicitly supplies a cap.
        resolved_cap = None
    else:
        resolved_cap = DEFAULT_APPLE_SILICON_SPLAT_CAP

    return max_splats, resolved_cap


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="splatlify",
        description="Convert a PNG/JPG image to an SVG (or PPTX) using 2D Gaussian splatting.",
    )
    parser.add_argument("input", help="Input image path (PNG/JPG)")
    parser.add_argument("-o", "--output", help="Output path (default: <input>.svg)")
    parser.add_argument(
        "--splats",
        type=int,
        default=None,
        help="Max number of splats (default: 2000, or exact photo preset count).",
    )
    parser.add_argument(
        "--time-budget",
        default=None,
        choices=[
            "smoke",
            "1m",
            "5m",
            "10m",
            "20m",
            "30m",
            "photo-native-10k",
            "photo-10k",
            "native-10k",
            "photo-native-20k",
            "photo-20k",
            "native-20k",
        ],
        help="Use a content-aware training budget preset. Presets set stage schedule, "
        "splat cap, and residual-detail cost; 'smoke' is an alias for 1m.",
    )
    parser.add_argument(
        "--stages",
        type=_parse_stages,
        default=_parse_stages("200,150,100,50"),
        help="Per-stage iteration schedule, comma-separated (default: 200,150,100,50)",
    )
    parser.add_argument(
        "--profile",
        default="max-fidelity",
        help="Quality profile (default: max-fidelity)",
    )
    parser.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "torch", "torch-batched", "gsplat"],
        help="Renderer backend. 'torch-batched' batches tiles for MPS/GPU experiments.",
    )
    parser.add_argument(
        "--optimizer-backend",
        default="torch",
        choices=["torch", "mlx"],
        help="Optimizer backend. 'mlx' is experimental.",
    )
    parser.add_argument(
        "--mlx-loss",
        default=None,
        choices=["linear-l1", "oklab-l1", "weighted-oklab-l1", "l1-ssim"],
        help="MLX optimizer loss profile when --optimizer-backend=mlx. "
        "Default 'l1-ssim' matches the torch path's combined L1+SSIM loss "
        "and avoids the color-cast regression that pure linear-l1 produces "
        "on PPTX/SVG export.",
    )
    parser.add_argument(
        "--mlx-tile-plan",
        default=None,
        choices=["static", "periodic"],
        help="MLX tile-plan mode. Use 'periodic' for geometry training.",
    )
    parser.add_argument(
        "--mlx-tile-plan-rebuild-interval",
        type=int,
        default=None,
        help="For --mlx-tile-plan periodic, rebuild tile membership every N iterations.",
    )
    parser.add_argument(
        "--mlx-trainable-groups",
        default=None,
        help="Comma-separated MLX trainable groups. Static mode currently supports color,alpha.",
    )
    parser.add_argument(
        "--renderer-tile-size",
        type=int,
        default=None,
        help="Override renderer tile size for backend tuning.",
    )
    parser.add_argument(
        "--renderer-batch-tile-count",
        type=int,
        default=None,
        help="For torch-batched, render this many tiles per tensor batch.",
    )
    parser.add_argument(
        "--renderer-max-active-splats-per-tile",
        type=int,
        default=None,
        help="For torch-batched, cap padded active splats per tile; default is uncapped.",
    )
    parser.add_argument(
        "--initial-splat-cap",
        type=int,
        default=None,
        help="Hard cap on the initial splat population before staged densification "
        "(default 1200). Raise this when --splats is large and you want the "
        "optimizer to actually use the full budget instead of being throttled at "
        "the historical initial cap.",
    )
    parser.add_argument(
        "--initial-splat-fraction",
        type=float,
        default=None,
        help="Fraction of --splats to seed the initial population with before "
        "densification (default 0.50). Clipped to [0.05, 1.0].",
    )
    parser.add_argument(
        "--blend-mode",
        default="alpha-over",
        choices=["alpha-over", "weighted"],
        help="Compositing blend mode",
    )
    parser.add_argument(
        "--max-edge",
        type=int,
        default=None,
        help="Downscale so the longest edge is at most N px",
    )
    parser.add_argument(
        "--format",
        default="svg",
        choices=["svg", "pptx", "canvas"],
        dest="fmt",
        help="Output format. 'canvas' emits a self-contained HTML that renders the "
        "splats via a JS canvas runtime with real linear-space alpha-over "
        "compositing (breaks the SVG primitive's representational cap).",
    )
    parser.add_argument(
        "--pptx-splat-style",
        default="soft-edge",
        choices=["soft-edge", "gradient"],
        help="Native PPTX splat primitive style. 'soft-edge' (default) is the "
        "LibreOffice-tolerant style; 'gradient' uses DrawingML radial gradients "
        "with semi-transparent stops and produces noticeably better fidelity "
        "in PowerPoint, especially when combined with --pptx-proxy-postfit-iters "
        "(e.g. 40 or 120) for a post-fit color/alpha refinement against the "
        "gradient compositor.",
    )
    parser.add_argument(
        "--training-export-target",
        default="auto",
        choices=["auto", "canvas", "svg", "pptx-softedge"],
        help="Renderer target used during optimization. 'auto' (default) picks "
        "based on --format: svg->svg (sRGB compositing, matches browser SVG "
        "blending), pptx->canvas (linear-light training; safer across PPTX "
        "viewers), canvas->canvas. The 'pptx-softedge' target trains against "
        "PowerPoint's actual brighter-than-Gaussian soft-edge rendering; it "
        "produces the closest match in real PowerPoint but can look washed "
        "out in soffice/LibreOffice (which renders the file more literally). "
        "Pass --training-export-target pptx-softedge explicitly if your "
        "deployment target is real PowerPoint.",
    )
    parser.add_argument(
        "--svg-recipe",
        default=None,
        choices=["standard", "browser-compatible", "scripted-matrix"],
        help="SVG export recipe (default comes from quality profile). "
        "'scripted-matrix' stores compact splat rows and expands browser-compatible "
        "gradients at load time.",
    )
    parser.add_argument(
        "--svg-proxy-postfit-iters",
        type=int,
        default=0,
        help="For SVG output, run N post-fit iterations on color/alpha using a "
        "browser-like SVG compositing proxy (default: 0).",
    )
    parser.add_argument(
        "--pptx-proxy-postfit-iters",
        type=int,
        default=0,
        help="For PPTX output, run N post-fit iterations on color/alpha using a "
        "PowerPoint soft-edge proxy with contrast/saturation terms (default: 0).",
    )
    parser.add_argument(
        "--region-weighting",
        dest="region_weighting",
        action="store_true",
        default=None,
        help="Enable segmentation-derived spatial loss/sampling weights.",
    )
    parser.add_argument(
        "--no-region-weighting",
        dest="region_weighting",
        action="store_false",
        help="Disable segmentation-derived spatial loss/sampling weights.",
    )
    parser.add_argument(
        "--layered-saliency",
        dest="layered_saliency",
        action="store_true",
        default=False,
        help="Tag splats into base/mass/detail/edge layers and export nested layer groups.",
    )
    parser.add_argument(
        "--no-layered-saliency",
        dest="layered_saliency",
        action="store_false",
        help="Disable layered saliency tagging/export grouping.",
    )
    parser.add_argument(
        "--apple-silicon-splat-cap",
        dest="apple_silicon_splat_cap",
        type=int,
        default=None,
        help="Safety cap applied on Apple Silicon before budget selection "
        "(default: 2000, disabled by default for exact photo presets).",
    )
    parser.add_argument(
        "--no-apple-silicon-splat-cap",
        dest="apple_silicon_splat_cap",
        action="store_const",
        const=DISABLE_APPLE_SILICON_SPLAT_CAP,
        help="Disable the conservative Apple Silicon splat cap for exploratory runs.",
    )
    parser.add_argument("--device", default="cpu", help="Torch device (cpu or cuda)")
    parser.add_argument("--seed", type=int, default=0, help="Deterministic seed")
    parser.add_argument(
        "--artifacts-dir",
        default=None,
        help="Optional directory for run manifest + iteration dumps",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s"
    )

    input_path = args.input
    if not Path(input_path).is_file():
        print(f"error: input not found: {input_path}", file=sys.stderr)
        return 2

    output = args.output or str(Path(input_path).with_suffix(f".{args.fmt}"))

    refinement_config = {}
    if args.svg_recipe is not None:
        refinement_config["svg_export_recipe"] = args.svg_recipe
    # Resolve "auto" training_export_target. SVG output trains under sRGB
    # compositing (matches browser/rsvg blend). PPTX defaults to canvas
    # (linear-light) because the pptx-softedge proxy is calibrated for
    # PowerPoint's brighter-than-Gaussian rendering and produces washed-out
    # output in soffice/LibreOffice viewers; users targeting real PowerPoint
    # should pass --training-export-target pptx-softedge explicitly.
    training_export_target = args.training_export_target
    if training_export_target == "auto":
        if args.fmt == "svg":
            training_export_target = "svg"
        else:
            training_export_target = "canvas"
    if training_export_target != "canvas":
        refinement_config["training_export_target"] = training_export_target
    if args.svg_proxy_postfit_iters > 0:
        refinement_config["svg_proxy_postfit_iters"] = int(args.svg_proxy_postfit_iters)
    if args.pptx_proxy_postfit_iters > 0:
        refinement_config["pptx_proxy_postfit_iters"] = int(
            args.pptx_proxy_postfit_iters
        )
    if args.region_weighting is not None:
        refinement_config["region_weighting_enabled"] = bool(args.region_weighting)
    if args.renderer_tile_size is not None:
        refinement_config["renderer_tile_size"] = int(args.renderer_tile_size)
    if args.renderer_batch_tile_count is not None:
        refinement_config["renderer_batch_tile_count"] = int(
            args.renderer_batch_tile_count
        )
    if args.renderer_max_active_splats_per_tile is not None:
        refinement_config["renderer_max_active_splats_per_tile"] = int(
            args.renderer_max_active_splats_per_tile
        )
    if args.mlx_loss is not None:
        refinement_config["mlx_loss"] = args.mlx_loss
    if args.mlx_tile_plan is not None:
        refinement_config["mlx_tile_plan"] = args.mlx_tile_plan
    if args.mlx_tile_plan_rebuild_interval is not None:
        refinement_config["mlx_tile_plan_rebuild_interval"] = int(
            args.mlx_tile_plan_rebuild_interval
        )
    if args.mlx_trainable_groups is not None:
        refinement_config["mlx_trainable_groups"] = args.mlx_trainable_groups
    if args.initial_splat_cap is not None:
        refinement_config["initial_splat_cap"] = int(args.initial_splat_cap)
    if args.initial_splat_fraction is not None:
        refinement_config["initial_splat_fraction"] = float(args.initial_splat_fraction)

    max_splats, apple_silicon_splat_cap = _resolve_cli_resource_limits(
        time_budget=args.time_budget,
        splats=args.splats,
        apple_silicon_splat_cap=args.apple_silicon_splat_cap,
    )

    converter = PNG2SVGConverter(
        max_splats=max_splats,
        stages=args.stages,
        target_size=_target_size(input_path, args.max_edge),
        quality_profile=args.profile,
        blend_mode=args.blend_mode,
        device=args.device,
        seed=args.seed,
        refinement_config=refinement_config or None,
        renderer_backend=args.backend,
        optimizer_backend=args.optimizer_backend,
        time_budget=args.time_budget,
        apple_silicon_splat_cap=apple_silicon_splat_cap,
        layered_saliency=args.layered_saliency,
        pptx_splat_style=args.pptx_splat_style,
    )
    converter.convert(
        input_path=input_path,
        output_path=output,
        output_format=args.fmt,
        seed=args.seed,
        artifacts_dir=args.artifacts_dir,
        verbose=args.verbose,
    )
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
