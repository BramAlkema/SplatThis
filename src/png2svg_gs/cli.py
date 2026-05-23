"""Command-line interface for the png2svg_gs PNG->SVG Gaussian-splatting pipeline."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image

from .converter import PNG2SVGConverter


def _parse_stages(text: str) -> List[int]:
    try:
        stages = [int(p) for p in text.split(",") if p.strip() != ""]
    except ValueError as exc:  # pragma: no cover - argparse surfaces the message
        raise argparse.ArgumentTypeError(f"invalid --stages '{text}': {exc}") from exc
    if not stages or any(s <= 0 for s in stages):
        raise argparse.ArgumentTypeError("--stages must be positive integers, e.g. 200,150,100,50")
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="splatlify",
        description="Convert a PNG/JPG image to an SVG (or PPTX) using 2D Gaussian splatting.",
    )
    parser.add_argument("input", help="Input image path (PNG/JPG)")
    parser.add_argument("-o", "--output", help="Output path (default: <input>.svg)")
    parser.add_argument("--splats", type=int, default=2000, help="Max number of splats (default: 2000)")
    parser.add_argument(
        "--time-budget",
        default=None,
        choices=["smoke", "1m", "5m", "10m", "30m"],
        help="Use a content-aware training budget preset. Presets set stage schedule, "
             "splat cap, and residual-detail cost; 'smoke' is an alias for 1m.",
    )
    parser.add_argument(
        "--stages", type=_parse_stages, default=_parse_stages("200,150,100,50"),
        help="Per-stage iteration schedule, comma-separated (default: 200,150,100,50)",
    )
    parser.add_argument("--profile", default="max-fidelity", help="Quality profile (default: max-fidelity)")
    parser.add_argument("--blend-mode", default="alpha-over", choices=["alpha-over", "weighted"], help="Compositing blend mode")
    parser.add_argument("--max-edge", type=int, default=None, help="Downscale so the longest edge is at most N px")
    parser.add_argument("--format", default="svg", choices=["svg", "pptx", "canvas"], dest="fmt",
                        help="Output format. 'canvas' emits a self-contained HTML that renders the "
                             "splats via a JS canvas runtime with real linear-space alpha-over "
                             "compositing (breaks the SVG primitive's representational cap).")
    parser.add_argument("--pptx-splat-style", default="soft-edge", choices=["soft-edge", "gradient"],
                        help="Native PPTX splat primitive style. 'soft-edge' is the current "
                             "PowerPoint-friendly default; 'gradient' preserves the old radial-gradient path.")
    parser.add_argument("--svg-recipe", default=None, choices=["standard", "browser-compatible"],
                        help="SVG export recipe (default comes from quality profile). "
                             "'browser-compatible' feathers gradients and compensates browser blending.")
    parser.add_argument("--region-weighting", dest="region_weighting", action="store_true", default=None,
                        help="Enable segmentation-derived spatial loss/sampling weights.")
    parser.add_argument("--no-region-weighting", dest="region_weighting", action="store_false",
                        help="Disable segmentation-derived spatial loss/sampling weights.")
    parser.add_argument("--layered-saliency", dest="layered_saliency", action="store_true", default=False,
                        help="Tag splats into base/mass/detail/edge layers and export nested layer groups.")
    parser.add_argument("--no-layered-saliency", dest="layered_saliency", action="store_false",
                        help="Disable layered saliency tagging/export grouping.")
    parser.add_argument(
        "--apple-silicon-splat-cap",
        dest="apple_silicon_splat_cap",
        type=int,
        default=2000,
        help="Safety cap applied on Apple Silicon before budget selection (default: 2000).",
    )
    parser.add_argument(
        "--no-apple-silicon-splat-cap",
        dest="apple_silicon_splat_cap",
        action="store_const",
        const=None,
        help="Disable the conservative Apple Silicon splat cap for exploratory runs.",
    )
    parser.add_argument("--device", default="cpu", help="Torch device (cpu or cuda)")
    parser.add_argument("--seed", type=int, default=0, help="Deterministic seed")
    parser.add_argument("--artifacts-dir", default=None, help="Optional directory for run manifest + iteration dumps")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s")

    input_path = args.input
    if not Path(input_path).is_file():
        print(f"error: input not found: {input_path}", file=sys.stderr)
        return 2

    output = args.output or str(Path(input_path).with_suffix(f".{args.fmt}"))

    refinement_config = {}
    if args.svg_recipe is not None:
        refinement_config["svg_export_recipe"] = args.svg_recipe
    if args.region_weighting is not None:
        refinement_config["region_weighting_enabled"] = bool(args.region_weighting)

    converter = PNG2SVGConverter(
        max_splats=args.splats,
        stages=args.stages,
        target_size=_target_size(input_path, args.max_edge),
        quality_profile=args.profile,
        blend_mode=args.blend_mode,
        device=args.device,
        seed=args.seed,
        refinement_config=refinement_config or None,
        time_budget=args.time_budget,
        apple_silicon_splat_cap=args.apple_silicon_splat_cap,
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
