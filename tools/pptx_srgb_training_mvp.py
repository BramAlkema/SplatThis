#!/usr/bin/env python3
"""MVP: does sRGB-compositing training fix PowerPoint's washed-out decks?

The gradient-style PPTX target trains against the linear-light pixel-runtime
model by default, while PowerPoint composites its gradients in display sRGB
-- the same train/deploy mismatch that washed out SVG until browser targets
moved to sRGB training. This experiment trains a subset of corpus images
with ``--training-export-target svg`` (the sRGB gradient proxy), emits the
decks through the normal CLI under current defaults, captures them in real
Microsoft PowerPoint, and scores capture-vs-source beside the regenerated
baseline pass -- including OKLab delta-E, the metric that quantifies the
vibrance loss a viewer sees.

Judged per the fidelity protocol: real PowerPoint captures only, wins kept
only if they clear the noise floor, and the result decides whether the full
corpus run and any default change happen at all.

Usage::

    PYTHONPATH=src python tools/pptx_srgb_training_mvp.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

from full_corpus_mvp import _capture_powerpoint_slideshow  # noqa: E402

from splatthis.fidelity.metrics import compute_fidelity_metrics  # noqa: E402
from splatthis.io import load_png  # noqa: E402

SOURCES = REPO / "result" / "corpus" / "images"
RUNS = REPO / "result" / "corpus" / "runs"
WORK = REPO / "tmp" / "pptx-srgb-mvp"

IMAGES = ("chameleon", "moon", "gravel", "text")
METRICS = ("ssim_srgb", "lpips", "delta_e_ok_mean", "delta_e_ok_p95")


def _grid_rois(height: int, width: int, tile: int = 64) -> List[tuple]:
    return [
        (y, x, min(y + tile, height), min(x + tile, width))
        for y in range(0, height, tile)
        for x in range(0, width, tile)
    ]


def score(source: np.ndarray, capture_path: Path) -> Dict[str, float]:
    rendered = np.asarray(load_png(str(capture_path))[..., :3], dtype=np.float32)
    height, width = source.shape[:2]
    metrics = compute_fidelity_metrics(
        source,
        rendered,
        fixed_rois=_grid_rois(height, width),
        render_method="Microsoft PowerPoint slideshow",
    ).as_dict()
    return {name: float(metrics[name]) for name in METRICS}


def main() -> int:
    WORK.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Any] = {}
    for index, image in enumerate(IMAGES, 1):
        source_path = SOURCES / f"{image}.png"
        deck = WORK / f"{image}-srgb.pptx"
        capture = WORK / f"{image}-srgb-powerpoint.png"
        source = np.asarray(load_png(str(source_path))[..., :3], dtype=np.float32)

        if not deck.exists():
            print(
                f"[{index}/{len(IMAGES)}] {image}: training (sRGB target) ...",
                flush=True,
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "splatthis.cli",
                    str(source_path),
                    "-o",
                    str(deck),
                    "--format",
                    "pptx",
                    "--training-export-target",
                    "svg",
                    "--seed",
                    "0",
                ],
                cwd=REPO,
                capture_output=True,
                text=True,
            )
            if completed.returncode:
                print(completed.stdout[-500:], completed.stderr[-500:], file=sys.stderr)
                return 2
        if not capture.exists():
            print(
                f"[{index}/{len(IMAGES)}] {image}: capturing in PowerPoint ...",
                flush=True,
            )
            height, width = source.shape[:2]
            returncode, message = _capture_powerpoint_slideshow(
                deck, capture, width, height
            )
            if returncode or not capture.exists():
                print(f"capture failed: {message.strip()[-300:]}", file=sys.stderr)
                return 2

        baseline_capture = RUNS / f"{image}_pptx_s0_powerpoint_slide.png"
        results[image] = {
            "srgb_trained": score(source, capture),
            "baseline_linear": score(source, baseline_capture),
            "deck_bytes": deck.stat().st_size,
        }
        print(f"[{index}/{len(IMAGES)}] {image}: scored", flush=True)

    (WORK / "results.json").write_text(json.dumps(results, indent=1) + "\n")
    print(f"\nwrote {WORK / 'results.json'}\n")
    header = f"{'image':22s}" + "".join(f"{m:>18s}" for m in METRICS)
    print(header)
    for image, entry in results.items():
        for label in ("baseline_linear", "srgb_trained"):
            row = entry[label]
            print(
                f"{image + ' ' + ('base' if label.startswith('b') else 'sRGB'):22s}"
                + "".join(f"{row[m]:>18.4f}" for m in METRICS)
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
