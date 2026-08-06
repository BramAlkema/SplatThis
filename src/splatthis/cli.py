"""Command-line interface for SplatThis's five direct output formats."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image

from ._version import __version__
from .converter import OUTPUT_SUFFIXES, PNG2SVGConverter


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _parse_stages(value: str) -> List[int]:
    try:
        stages = [int(part) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if not stages or any(stage <= 0 for stage in stages):
        raise argparse.ArgumentTypeError(
            "use positive integers, for example 200,150,100"
        )
    return stages


def _target_size(path: Path, max_edge: Optional[int]) -> Optional[Tuple[int, int]]:
    if max_edge is None:
        return None
    with Image.open(path) as image:
        width, height = image.size
    if max(width, height) <= max_edge:
        return None
    scale = max_edge / max(width, height)
    return max(1, round(width * scale)), max(1, round(height * scale))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="splatthis",
        description=(
            "Fit 2D Gaussian splats to an image and export SVG, PPTX, Canvas, "
            "CSS, or CSS-in-EML."
        ),
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    parser.add_argument("input", type=Path, help="PNG or JPEG input")
    parser.add_argument("-o", "--output", type=Path, help="output artifact path")
    parser.add_argument(
        "--format",
        choices=list(OUTPUT_SUFFIXES),
        default="svg",
        help="artifact format (default: svg)",
    )
    parser.add_argument("--splats", type=_positive_int, default=2000)
    parser.add_argument("--stages", type=_parse_stages, default=[200, 150, 100, 50])
    parser.add_argument("--max-edge", type=_positive_int)
    parser.add_argument(
        "--profile",
        choices=["fast", "balanced", "max-fidelity"],
        default="max-fidelity",
    )
    parser.add_argument(
        "--device", default="cpu", help="Torch device: cpu, cuda, or mps"
    )
    parser.add_argument(
        "--optimizer-backend",
        choices=["torch", "mlx"],
        default="torch",
        help="optimizer backend; MLX requires the optional mlx extra and Apple Metal",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--artifacts-dir", type=Path)
    parser.add_argument("--save-json", action="store_true")
    parser.add_argument("--preview", type=Path, help="optional PNG preview path")
    parser.add_argument(
        "--capture",
        type=Path,
        help="capture SVG/CSS/Canvas with Chromium or PPTX with PowerPoint OSA",
    )
    parser.add_argument(
        "--capture-browser",
        type=Path,
        help="Chrome/Chromium executable for --capture",
    )
    parser.add_argument(
        "--embed-population",
        action="store_true",
        help="embed a recoverable population in SVG/PPTX and any preview PNG",
    )
    parser.add_argument(
        "--embed-population-in-pixels",
        action="store_true",
        help="also hide the population in preview PNG low bits; requires steg extra",
    )
    parser.add_argument(
        "--email-subject",
        default="Gaussian splats, drawn by your mail client",
        help="subject for --format eml",
    )
    parser.add_argument(
        "--email-from", default="splatthis@localhost", help="sender for EML output"
    )
    parser.add_argument(
        "--email-to", default="you@example.com", help="recipient for EML output"
    )
    parser.add_argument(
        "--email-splats",
        type=_positive_int,
        default=285,
        help="maximum CSS splats in EML output (default: 285)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(message)s",
    )
    if not args.input.is_file():
        print(f"error: input not found: {args.input}", file=sys.stderr)
        return 2
    output = args.output or args.input.with_suffix(OUTPUT_SUFFIXES[args.format])
    target_size = _target_size(args.input, args.max_edge)
    try:
        converter = PNG2SVGConverter(
            max_splats=args.splats,
            stages=args.stages,
            target_size=target_size,
            quality_profile=args.profile,
            device=args.device,
            optimizer_backend=args.optimizer_backend,
            seed=args.seed,
            blend_mode="alpha-over",
        )
        converter.convert(
            str(args.input),
            str(output),
            save_json=args.save_json,
            seed=args.seed,
            artifacts_dir=(
                None if args.artifacts_dir is None else str(args.artifacts_dir)
            ),
            preview_png_path=None if args.preview is None else str(args.preview),
            output_format=args.format,
            email_subject=args.email_subject,
            email_sender=args.email_from,
            email_recipient=args.email_to,
            email_max_splats=args.email_splats,
            embed_population=args.embed_population,
            embed_population_in_pixels=args.embed_population_in_pixels,
            verbose=args.verbose,
        )
        if args.capture:
            if args.format == "pptx":
                if args.capture_browser is not None:
                    raise ValueError("--capture-browser does not apply to PPTX capture")
                from .powerpoint_osa import capture_pptx_with_powerpoint

                capture_pptx_with_powerpoint(output, args.capture)
            else:
                if args.format not in {"svg", "css", "canvas"}:
                    raise ValueError(
                        "--capture supports --format svg, pptx, css, or canvas"
                    )
                if target_size is None:
                    with Image.open(args.input) as source:
                        width, height = source.size
                else:
                    width, height = target_size
                from .browser_capture import capture_artifact_to_png

                capture_artifact_to_png(
                    output,
                    args.capture,
                    artifact_format=args.format,
                    width=width,
                    height=height,
                    browser_executable=args.capture_browser,
                )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
