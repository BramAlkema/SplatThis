#!/usr/bin/env python3
"""Measure deployed pixel-runtime fidelity from re-emitted current artifacts.

The fidelity registry deliberately publishes no deployed expectation for the
pixel runtime: its stored corpus captures predate the current code, and
publishing them would repeat the exact provenance mistake the registry exists
to prevent. This script closes the gap the honest way -- it re-emits each
corpus image's pixel-runtime HTML from the stored seed-0 canvas-trained
population with the shipped ``generate_pixel_runtime_html``, captures the
selected backend's actual framebuffer in governing Chrome, scores it against
the original image, and writes the deployed medians and per-image evidence
into ``src/splatthis/data/compositor-fidelity.json``.

Usage::

    PYTHONPATH=src python tools/measure_pixel_runtime_deployed.py
"""

from __future__ import annotations

import json
import statistics
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from splatthis.browser_capture import (  # noqa: E402
    render_pixel_runtime_html_in_browser_to_linear_rgb,
)
from splatthis.fidelity.metrics import compute_fidelity_metrics  # noqa: E402
from splatthis.io import load_png  # noqa: E402
from splatthis.pixel_runtime import generate_pixel_runtime_html  # noqa: E402
from splatthis.storage import load_splats_json  # noqa: E402

RUNS = REPO / "result" / "corpus" / "runs"
SOURCES = REPO / "result" / "corpus" / "images"
REGISTRY = REPO / "src" / "splatthis" / "data" / "compositor-fidelity.json"


def _median(values: List[float]) -> float:
    return statistics.median(values)


def _percentile(values: List[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    fraction = position - low
    return ordered[low] * (1 - fraction) + ordered[high] * fraction


def _grid_rois(height: int, width: int, tile: int = 64) -> List[tuple]:
    return [
        (y, x, min(y + tile, height), min(x + tile, width))
        for y in range(0, height, tile)
        for x in range(0, width, tile)
    ]


def measure_image(image: str, work_dir: Path) -> Dict[str, Any]:
    run = RUNS / f"{image}_canvas_s0_art"
    splats = load_splats_json(str(run / "final.raw.json"))
    manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    background = np.asarray(
        manifest["config"]["background_linear_rgb"], dtype=np.float32
    )
    source = np.asarray(
        load_png(str(SOURCES / f"{image}.png"))[..., :3], dtype=np.float32
    )
    height, width = source.shape[:2]

    html_path = work_dir / f"{image}.html"
    html_path.write_text(
        generate_pixel_runtime_html(
            splats, width, height, background_linear_rgb=background
        ),
        encoding="utf-8",
    )
    rendered, renderer = render_pixel_runtime_html_in_browser_to_linear_rgb(
        html_path, width, height
    )
    metrics = compute_fidelity_metrics(
        source,
        rendered,
        fixed_rois=_grid_rois(height, width),
        splat_count=len(splats),
        render_method=renderer,
    ).as_dict()
    return {
        "image": image,
        "splats": len(splats),
        "renderer": renderer,
        "deployed_ssim_srgb": round(float(metrics["ssim_srgb"]), 4),
        "deployed_lpips": round(float(metrics["lpips"]), 4),
    }


def main() -> int:
    images = sorted(p.stem for p in SOURCES.glob("*.png"))
    if len(images) != 21:
        print(f"error: expected 21 corpus images, found {len(images)}", file=sys.stderr)
        return 2
    rows = []
    with tempfile.TemporaryDirectory(prefix="pixel-deployed-") as tmp:
        for index, image in enumerate(images, 1):
            print(f"[{index}/{len(images)}] {image} ...", flush=True)
            rows.append(measure_image(image, Path(tmp)))

    renderers = {row["renderer"] for row in rows}
    if len(renderers) != 1:
        print(f"error: renderer changed mid-run: {renderers}", file=sys.stderr)
        return 2
    lpips = [row["deployed_lpips"] for row in rows]
    ssim = [row["deployed_ssim_srgb"] for row in rows]

    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    entry = registry["formats"]["pixel-runtime"]
    entry["method"][
        "deployed"
    ] = f"{renderers.pop()} capture of the re-emitted current artifact"
    entry["expectation"]["deployed"] = {
        "lpips_median": round(_median(lpips), 4),
        "lpips_p90": round(_percentile(lpips, 0.9), 4),
        "ssim_srgb_median": round(_median(ssim), 4),
        "ssim_srgb_p10": round(_percentile(ssim, 0.1), 4),
        "against": "the original image",
    }
    entry["expectation"]["summary"] = (
        "Evaluates the splat formula rather than approximating it, so it is a "
        "parity model of the reference renderer rather than a compositor. "
        "Effectively lossless as an emitter. Deployed figures re-measured "
        "August 2026 from artifacts re-emitted by the current code and "
        "captured in governing Chrome; they describe the seed-0 2k "
        "canvas-trained populations and are dominated, like every deployed "
        "figure, by fitting error."
    )
    per_image = {p["image"]: p for p in registry["per_image"]["pixel-runtime"]}
    for row in rows:
        per_image[row["image"]]["deployed_lpips"] = row["deployed_lpips"]
        per_image[row["image"]]["deployed_ssim_srgb"] = row["deployed_ssim_srgb"]
    REGISTRY.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")

    print(f"\nwrote deployed pixel-runtime figures into {REGISTRY.relative_to(REPO)}")
    print(
        f"deployed: lpips median {_median(lpips):.4f} (p90 "
        f"{_percentile(lpips, 0.9):.4f}), ssim median {_median(ssim):.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
