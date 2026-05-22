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
        "--stages", type=_parse_stages, default=_parse_stages("200,150,100,50"),
        help="Per-stage iteration schedule, comma-separated (default: 200,150,100,50)",
    )
    parser.add_argument("--profile", default="max-fidelity", help="Quality profile (default: max-fidelity)")
    parser.add_argument("--blend-mode", default="alpha-over", choices=["alpha-over", "weighted"], help="Compositing blend mode")
    parser.add_argument("--max-edge", type=int, default=None, help="Downscale so the longest edge is at most N px")
    parser.add_argument("--format", default="svg", choices=["svg", "pptx"], dest="fmt", help="Output format")
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

    converter = PNG2SVGConverter(
        max_splats=args.splats,
        stages=args.stages,
        target_size=_target_size(input_path, args.max_edge),
        quality_profile=args.profile,
        blend_mode=args.blend_mode,
        device=args.device,
        seed=args.seed,
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
