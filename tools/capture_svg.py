#!/usr/bin/env python3
"""Capture a real SVG with dimension-locked Playwright Chromium."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from splatthis.browser_capture import (
    PlaywrightSvgRenderer,
    read_svg_pixel_size,
    resolve_browser_executable,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("svg", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument(
        "--browser-executable", type=Path, default=resolve_browser_executable()
    )
    parser.add_argument("--timeout-ms", type=int, default=120_000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--samples-dir", type=Path)
    args = parser.parse_args()
    if (args.width is None) != (args.height is None):
        parser.error("--width and --height must be supplied together")
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    width, height = (
        read_svg_pixel_size(args.svg.resolve())
        if args.width is None
        else (args.width, args.height)
    )
    with PlaywrightSvgRenderer(
        browser_executable=args.browser_executable,
        timeout_ms=args.timeout_ms,
    ) as renderer:
        result = renderer.capture(
            args.svg,
            args.output,
            width=width,
            height=height,
            repeats=args.repeats,
            samples_dir=args.samples_dir,
        )
    print(json.dumps(result.as_dict(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
