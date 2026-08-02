#!/usr/bin/env python3
"""Re-emit the corpus decks with current code and capture them in PowerPoint.

The attended half of the schema-v2 ledger regeneration. For each corpus
image, the stored seed-0 pptx-trained population is re-emitted as a deck by
the shipped emitter under the current defaults (gradient style, corrected
back-to-front order), then rendered by real Microsoft PowerPoint in
slideshow mode and captured at native size. This takes over the desktop for
roughly twenty seconds per deck; run it while nobody needs the screen.
LibreOffice is never used.

The July-era decks and captures are backed up beside the new ones before
being replaced. Afterwards, score the fresh captures with::

    PYTHONPATH=src python tools/corpus_benchmark.py --score-powerpoint

Usage::

    PYTHONPATH=src python tools/run_powerpoint_pass.py [--only image,image]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import List

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

from full_corpus_mvp import _capture_powerpoint_slideshow  # noqa: E402

from splatthis.export_common import _sort_splats_for_export  # noqa: E402
from splatthis.io import load_png  # noqa: E402
from splatthis.pptx_export import (  # noqa: E402
    generate_drawingml_slide_content,
    save_pptx_with_drawingml_content,
)
from splatthis.storage import load_splats_json  # noqa: E402

RUNS = REPO / "result" / "corpus" / "runs"
SOURCES = REPO / "result" / "corpus" / "images"
BACKUP_SUFFIX = "_july2026"


def process_image(image: str) -> str:
    art = RUNS / f"{image}_pptx_s0_art"
    deck_path = RUNS / f"{image}_pptx_s0.pptx"
    capture_path = RUNS / f"{image}_pptx_s0_powerpoint_slide.png"

    splats = load_splats_json(str(art / "final.raw.json"))
    manifest = json.loads((art / "run_manifest.json").read_text(encoding="utf-8"))
    background = np.asarray(
        manifest["config"]["background_linear_rgb"], dtype=np.float32
    )
    source = load_png(str(SOURCES / f"{image}.png"))
    height, width = source.shape[:2]

    for original in (deck_path, capture_path):
        backup = original.with_name(f"{original.stem}{BACKUP_SUFFIX}{original.suffix}")
        if original.exists() and not backup.exists():
            shutil.copy2(original, backup)

    slide_xml = generate_drawingml_slide_content(
        _sort_splats_for_export(splats),
        width,
        height,
        2.5,
        background_linear_rgb=background,
        splat_style="gradient",
    )
    save_pptx_with_drawingml_content(
        slide_xml=slide_xml,
        width=width,
        height=height,
        output_path=str(deck_path),
        splat_count=len(splats),
    )

    returncode, message = _capture_powerpoint_slideshow(
        deck_path, capture_path, width, height
    )
    if returncode or not capture_path.exists():
        return f"FAILED: {message.strip()[-200:]}"
    return f"ok ({len(splats)} splats, {width}x{height})"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", help="comma-separated corpus image names")
    args = parser.parse_args()

    images = sorted(p.stem for p in SOURCES.glob("*.png"))
    if args.only:
        wanted = [part.strip() for part in args.only.split(",") if part.strip()]
        unknown = sorted(set(wanted) - set(images))
        if unknown:
            parser.error(f"unknown corpus images: {', '.join(unknown)}")
        images = wanted

    failures: List[str] = []
    for index, image in enumerate(images, 1):
        print(f"[{index}/{len(images)}] {image} ... ", end="", flush=True)
        try:
            outcome = process_image(image)
        except Exception as exc:  # keep driving the remaining decks
            outcome = f"FAILED: {type(exc).__name__}: {exc}"
        print(outcome, flush=True)
        if outcome.startswith("FAILED"):
            failures.append(image)

    if failures:
        print(f"\n{len(failures)} capture(s) failed: {', '.join(failures)}")
        print("Re-run with --only to retry them.")
        return 1
    print("\nAll decks re-emitted and captured. Next:")
    print("  PYTHONPATH=src python tools/corpus_benchmark.py --score-powerpoint")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
