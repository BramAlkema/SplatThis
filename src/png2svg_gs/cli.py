"""Command-line interface for the png2svg_gs PNG->SVG Gaussian-splatting pipeline."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image

from ._version import __version__
from .adaptive_compute import (
    DEFAULT_CHROME_PSNR_SAFETY_MARGIN,
    DEFAULT_CHROME_SSIM_SAFETY_MARGIN,
)
from .budgets import TIME_BUDGET_ALIASES, TIME_BUDGET_PRESETS
from .converter import PNG2SVGConverter

DEFAULT_MAX_SPLATS = 2000
DEFAULT_APPLE_SILICON_SPLAT_CAP = 2000
DISABLE_APPLE_SILICON_SPLAT_CAP = 0


def _positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def _non_negative_int(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("must be zero or a positive integer")
    return value


def _non_negative_float(text: str) -> float:
    value = float(text)
    if value < 0:
        raise argparse.ArgumentTypeError("must be zero or positive")
    return value


def _unit_interval_float(text: str) -> float:
    value = float(text)
    if not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return value


def _initial_splat_fraction(text: str) -> float:
    value = float(text)
    if not 0.05 <= value <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0.05 and 1.0")
    return value


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
    elif splats is not None:
        # Same reasoning for an explicit --splats: the user stated a count, so
        # honor it rather than silently clamping to the interactive ceiling.
        resolved_cap = None
    else:
        resolved_cap = DEFAULT_APPLE_SILICON_SPLAT_CAP

    return max_splats, resolved_cap


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="splatlify",
        description="Convert a PNG/JPG image to an SVG (or PPTX) using 2D Gaussian splatting.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument("input", help="Input image path (PNG/JPG)")
    parser.add_argument("-o", "--output", help="Output path (default: <input>.svg)")
    parser.add_argument(
        "--splats",
        type=_positive_int,
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
        default=None,
        help="Per-stage iteration schedule, comma-separated. Defaults to the "
        "selected profile (max-fidelity: 1000,500,250).",
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
        default="mlx",
        choices=["torch", "mlx"],
        help="Optimizer backend. 'mlx' (default) is Apple-Silicon-native and "
        "~5x faster than torch on M-series hardware; pass 'torch' for "
        "cross-platform CUDA / CPU runs.",
    )
    parser.add_argument(
        "--mlx-loss",
        default=None,
        choices=[
            "linear-l1",
            "oklab-l1",
            "weighted-oklab-l1",
            "l1-ssim",
            "oklab-l1-ssim",
        ],
        help="MLX optimizer loss profile when --optimizer-backend=mlx. "
        "Default 'oklab-l1-ssim' mirrors the torch default objective "
        "(OKLab L1 + SSIM on the L channel + optional gradient term); "
        "'l1-ssim' is the older linear-RGB variant.",
    )
    parser.add_argument(
        "--mlx-tile-plan",
        default=None,
        choices=["static", "periodic"],
        help="MLX tile-plan mode. Use 'periodic' for geometry training.",
    )
    parser.add_argument(
        "--mlx-tile-plan-rebuild-interval",
        type=_positive_int,
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
        type=_positive_int,
        default=None,
        help="Override renderer tile size for backend tuning.",
    )
    parser.add_argument(
        "--renderer-batch-tile-count",
        type=_positive_int,
        default=None,
        help="For torch-batched, render this many tiles per tensor batch.",
    )
    parser.add_argument(
        "--renderer-max-active-splats-per-tile",
        type=_positive_int,
        default=None,
        help="For torch-batched, cap padded active splats per tile; default is uncapped.",
    )
    parser.add_argument(
        "--canvas-parallax-strength",
        type=_non_negative_float,
        default=None,
        help="For --format=canvas: enable mouse-driven parallax over native "
        "Canvas-API splat planes. Requires --layered-saliency for meaningful "
        "depth. Default: 0 = one static Canvas compositor.",
    )
    parser.add_argument(
        "--pixel-runtime-parallax-strength",
        type=_non_negative_float,
        default=None,
        help="For --format=pixel-runtime: enable the historical multi-plane "
        "ImageData parallax runtime. Default: 0 = one generated pixel buffer.",
    )
    parser.add_argument(
        "--css-parallax-strength",
        type=_non_negative_float,
        default=None,
        help="For --format=css: enable scriptless hover parallax with this max "
        "foreground offset in pixels. A CSS-only hover grid moves midground "
        "and foreground depth planes; no JavaScript is emitted. Combine with "
        "--layered-saliency for meaningful depth. Default: 0 = static.",
    )
    parser.add_argument(
        "--css-hover-grid-size",
        type=_positive_int,
        default=None,
        help="CSS parallax hover-grid width and height (1-20, default: 10).",
    )
    parser.add_argument(
        "--adaptive-compute",
        action="store_true",
        help="For pixel-runtime output, stop before later densification/stages once an "
        "observed deployed pixel-runtime checkpoint reaches the quality target. "
        "Default: off. This conservative controller does not use plateau or "
        "future-budget prediction.",
    )
    parser.add_argument(
        "--adaptive-target-ssim-srgb",
        type=_unit_interval_float,
        default=None,
        help="Desired SSIM_sRGB target for --adaptive-compute, scored with the "
        "byte-exact CPU boundary model; the selected final browser backend is "
        "graded separately (default: 0.98).",
    )
    parser.add_argument(
        "--adaptive-target-psnr-srgb",
        type=_non_negative_float,
        default=None,
        help="Optional desired Chrome PSNR_sRGB target for --adaptive-compute. "
        "When supplied, both margin-adjusted SSIM and PSNR targets must be met.",
    )
    parser.add_argument(
        "--adaptive-min-checkpoints",
        type=_positive_int,
        default=None,
        help="Minimum completed pixel-runtime stages before adaptive stopping "
        "(default: 2).",
    )
    parser.add_argument(
        "--adaptive-chrome-ssim-margin",
        type=_non_negative_float,
        default=None,
        help="Advanced cross-version SSIM safety-margin override for the "
        "byte-exact ImageData runtime model "
        f"(calibrated default: {DEFAULT_CHROME_SSIM_SAFETY_MARGIN:g}).",
    )
    parser.add_argument(
        "--adaptive-chrome-psnr-margin",
        type=_non_negative_float,
        default=None,
        help="Advanced cross-version PSNR safety-margin override for the "
        "byte-exact ImageData runtime model "
        f"in dB (calibrated default: {DEFAULT_CHROME_PSNR_SAFETY_MARGIN:g}).",
    )
    parser.add_argument(
        "--initial-splat-cap",
        type=_positive_int,
        default=None,
        help="Hard cap on the initial splat population before staged densification "
        "(default 1200). Raise this when --splats is large and you want the "
        "optimizer to actually use the full budget instead of being throttled at "
        "the historical initial cap.",
    )
    parser.add_argument(
        "--initial-splat-fraction",
        type=_initial_splat_fraction,
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
        type=_positive_int,
        default=None,
        help="Downscale so the longest edge is at most N px",
    )
    parser.add_argument(
        "--format",
        default="svg",
        choices=["svg", "pptx", "canvas", "css", "pixel-runtime"],
        dest="fmt",
        help="Output format. 'canvas' submits one browser-native Canvas 2D "
        "radial-gradient primitive per splat. 'pixel-runtime' tries WebGL2 "
        "float splats, then exact Worker/main-thread CPU fallbacks. 'css' emits "
        "scriptless DOM gradient splats.",
    )
    parser.add_argument(
        "--pptx-splat-style",
        default="gradient",
        choices=["soft-edge", "gradient", "blur"],
        help="Native PPTX splat primitive style. 'gradient' (default) uses "
        "DrawingML radial gradients with per-stop alpha falloff -- each splat's "
        "color stays radially confined and there is no inter-splat color "
        "spreading. 'soft-edge' fills each splat's bounding ellipse with a "
        "uniform color plus an outer feather; in PowerPoint that spreads pink "
        "splats placed near (e.g.) the chameleon's eye across the surrounding "
        "teal body. Use --pptx-proxy-postfit-iters 40-120 for an additional "
        "post-fit color/alpha refinement against the gradient compositor.",
    )
    parser.add_argument(
        "--pptx-painter-order",
        default="legacy",
        choices=["legacy", "back-to-front"],
        help="Native DrawingML shape order. 'legacy' remains the safe default; "
        "'back-to-front' uses corrected painter semantics demonstrated by the "
        "real-PowerPoint corpus MVP. Use the external artifact gate before "
        "promoting it per image.",
    )
    parser.add_argument(
        "--training-export-target",
        default="auto",
        choices=[
            "auto",
            "pixel-runtime",
            "browser-gradient",
            "svg",
            "canvas",
            "pptx-softedge",
        ],
        help="Renderer target used during optimization. 'auto' (default) picks "
        "based on --format: svg/css/canvas->browser-gradient and pixel-runtime "
        "->pixel-runtime. PPTX defaults to pixel-runtime training, which is "
        "safer across viewers. 'canvas' remains a deprecated alias for "
        "'pixel-runtime'. The 'pptx-softedge' target trains against "
        "PowerPoint's actual brighter-than-Gaussian soft-edge rendering; it "
        "produces the closest match in real PowerPoint but can look washed "
        "out in soffice/LibreOffice (which renders the file more literally). "
        "Pass --training-export-target pptx-softedge explicitly if your "
        "deployment target is real PowerPoint.",
    )
    parser.add_argument(
        "--svg-recipe",
        default=None,
        choices=[
            "standard",
            "browser-compatible",
            "scripted-matrix",
            "palette-quantized",
            "blur",
        ],
        help="SVG export recipe (default comes from quality profile). "
        "'scripted-matrix' stores compact splat rows and expands "
        "browser-compatible gradients at load time (JS required). "
        "'palette-quantized' k-means-clusters splat colors into a small "
        "palette (default 128 colors), defines one shared <radialGradient> "
        "per palette color, and references it per-splat with per-element "
        "opacity for the alpha scale. ~3-4x smaller than 'standard' at "
        "high splat counts and renders in any SVG-capable surface "
        "(browsers AND headless rasterizers); slight color banding at very "
        "small palette sizes.",
    )
    parser.add_argument(
        "--svg-gradient-quality",
        default=None,
        choices=["standard", "high"],
        help="SVG Gaussian-stop policy. 'standard' uses compact adaptive "
        "gradients; 'high' uses a stricter adaptive error bound and up to "
        "nine stops. Default: standard; the max-fidelity compositor gate can "
        "select high when the deployed browser artifact improves.",
    )
    parser.add_argument(
        "--svg-painter-order",
        default=None,
        choices=["back-to-front", "legacy"],
        help="SVG element order. Correct back-to-front is the default; legacy "
        "retains historical forward DOM emission for compatibility checks.",
    )
    parser.add_argument(
        "--svg-compositor-gate",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Browser-grade legacy, corrected-standard and corrected-high SVG "
        "candidates, then accept or revert. Enabled by default for max-fidelity.",
    )
    parser.add_argument(
        "--svg-optimize",
        action="store_true",
        help="Post-process the emitted SVG with `svgo` (must be on PATH). "
        "At the default precision this is free: identical LPIPS/SSIM to four "
        "decimals while shrinking the file. Skipped with a warning if svgo "
        "is missing; the original is kept if svgo fails or degrades output.",
    )
    parser.add_argument(
        "--svg-optimize-precision",
        type=_non_negative_int,
        default=2,
        help="Decimal precision for --svg-optimize (default: 2). Below 2 is "
        "measurably lossy for splats: stop-opacity quantization collapses the "
        "alpha vocabulary and truncates Gaussian tails.",
    )
    parser.add_argument(
        "--fidelity-stage",
        default=None,
        choices=["off", "balanced", "max"],
        help="ADR-003 fidelity stage: bounded accept-or-revert polish "
        "evaluated on the emitted, actually-rasterized SVG (default: off). "
        "'balanced' runs the monotonic evaluator shell; 'max' also enables "
        "the bounded operator portfolio. Candidates are kept only on a "
        "measured deployed-artifact gain with hard no-regression gates.",
    )
    parser.add_argument(
        "--svg-proxy-postfit-iters",
        type=_non_negative_int,
        default=0,
        help="For SVG output, run N post-fit iterations on color/alpha using a "
        "browser-like SVG compositing proxy (default: 0).",
    )
    parser.add_argument(
        "--pptx-proxy-postfit-iters",
        type=_non_negative_int,
        default=0,
        help="For PPTX output, run N post-fit iterations on color/alpha using a "
        "PowerPoint soft-edge proxy with contrast/saturation terms (default: 0). "
        "Skipped automatically when --pptx-splat-style=blur (the blur recipe "
        "uses --blur-postfit-iters instead, since the gradient compositor's "
        "alpha-attenuation boost over-saturates true Gaussian splats).",
    )
    parser.add_argument(
        "--blur-postfit-iters",
        type=_non_negative_int,
        default=0,
        help="For blur-recipe output (--svg-recipe blur or --pptx-splat-style "
        "blur), run N post-fit iterations on color/alpha against a Gaussian-"
        "convolution proxy. Closes the train→deploy gap for the blur recipe; "
        "ignored when neither output target is blur. Recommended: 40-120.",
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


def _run_conversion(args: argparse.Namespace) -> int:
    input_path = args.input
    if not Path(input_path).is_file():
        print(f"error: input not found: {input_path}", file=sys.stderr)
        return 2

    default_suffix = (
        ".html" if args.fmt in {"canvas", "css", "pixel-runtime"} else f".{args.fmt}"
    )
    output = args.output or str(Path(input_path).with_suffix(default_suffix))

    refinement_config = {}
    if args.svg_recipe is not None:
        refinement_config["svg_export_recipe"] = args.svg_recipe
    if args.svg_gradient_quality is not None:
        refinement_config["svg_gradient_quality"] = args.svg_gradient_quality
    if args.svg_painter_order is not None:
        refinement_config["svg_painter_order"] = args.svg_painter_order
    if args.svg_compositor_gate is not None:
        refinement_config["svg_compositor_gate"] = args.svg_compositor_gate
    if args.fidelity_stage is not None:
        refinement_config["fidelity_stage"] = args.fidelity_stage
    if args.svg_optimize:
        refinement_config["svg_optimize"] = True
        refinement_config["svg_optimize_precision"] = int(args.svg_optimize_precision)
    # Resolve "auto" training_export_target. Browser-gradient outputs train
    # under sRGB compositing. PPTX defaults to the exact pixel-runtime model
    # (linear-light) because the pptx-softedge proxy is calibrated for
    # PowerPoint's brighter-than-Gaussian rendering and produces washed-out
    # output in soffice/LibreOffice viewers; users targeting real PowerPoint
    # should pass --training-export-target pptx-softedge explicitly.
    training_export_target = args.training_export_target
    if training_export_target == "auto":
        if args.fmt in {"svg", "css", "canvas"}:
            training_export_target = "svg"
        else:
            training_export_target = "pixel-runtime"
    elif training_export_target == "canvas":
        training_export_target = "pixel-runtime"
    elif training_export_target == "browser-gradient":
        training_export_target = "svg"
    if training_export_target != "pixel-runtime":
        refinement_config["training_export_target"] = training_export_target
    if args.svg_proxy_postfit_iters > 0:
        refinement_config["svg_proxy_postfit_iters"] = int(args.svg_proxy_postfit_iters)
    if args.pptx_proxy_postfit_iters > 0:
        refinement_config["pptx_proxy_postfit_iters"] = int(
            args.pptx_proxy_postfit_iters
        )
    if args.blur_postfit_iters > 0:
        refinement_config["blur_proxy_postfit_iters"] = int(args.blur_postfit_iters)
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
    if args.canvas_parallax_strength is not None:
        refinement_config["canvas_parallax_strength"] = float(
            args.canvas_parallax_strength
        )
    if args.pixel_runtime_parallax_strength is not None:
        refinement_config["pixel_runtime_parallax_strength"] = float(
            args.pixel_runtime_parallax_strength
        )
    if args.css_parallax_strength is not None:
        refinement_config["css_parallax_strength"] = float(args.css_parallax_strength)
    if args.css_hover_grid_size is not None:
        refinement_config["css_hover_grid_size"] = int(args.css_hover_grid_size)
    if args.adaptive_compute:
        refinement_config["adaptive_compute_enabled"] = True
    if args.adaptive_target_ssim_srgb is not None:
        refinement_config["adaptive_compute_target_ssim_srgb"] = float(
            args.adaptive_target_ssim_srgb
        )
    if args.adaptive_target_psnr_srgb is not None:
        refinement_config["adaptive_compute_target_psnr_srgb"] = float(
            args.adaptive_target_psnr_srgb
        )
    if args.adaptive_min_checkpoints is not None:
        refinement_config["adaptive_compute_min_checkpoints"] = int(
            args.adaptive_min_checkpoints
        )
    if args.adaptive_chrome_ssim_margin is not None:
        refinement_config["adaptive_compute_chrome_ssim_margin"] = float(
            args.adaptive_chrome_ssim_margin
        )
    if args.adaptive_chrome_psnr_margin is not None:
        refinement_config["adaptive_compute_chrome_psnr_margin"] = float(
            args.adaptive_chrome_psnr_margin
        )
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
        pptx_painter_order=args.pptx_painter_order,
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


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s"
    )
    try:
        return _run_conversion(args)
    except (OSError, RuntimeError, ValueError) as exc:
        if args.verbose:
            logging.getLogger(__name__).exception("Conversion failed")
        else:
            print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
