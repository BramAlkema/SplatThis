#!/usr/bin/env python3
"""A/B the PPTX post-fit alpha law on a real PowerPoint capture.

``_postfit_splats_for_pptx_proxy`` refines colour and alpha against a model
of the deck. For the ``gradient`` splat style that model used the soft-edge
law ``-log1p(-((1 - exp(-a)) * scale))`` while the emitter writes
``a * scale`` -- agreeing as alpha goes to zero but diverging to 27% at
alpha 1.0. Because the stage also *selects its best iterate* with the same
model, a wrong law can degrade the deployed deck while reporting a gain.

This isolates that one variable: same stored population, same iteration
count, same emitter, same capture protocol -- only the post-fit law differs
between the two runs. Run it once before the fix and once after; each run
writes its own tagged deck, capture and score.

Usage::

    PYTHONPATH=src python tools/measure_pptx_postfit_law.py --tag before
    # ...apply the fix...
    PYTHONPATH=src python tools/measure_pptx_postfit_law.py --tag after
    PYTHONPATH=src python tools/measure_pptx_postfit_law.py --compare
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

from full_corpus_mvp import _capture_powerpoint_slideshow  # noqa: E402

from splatthis.converter import PNG2SVGConverter  # noqa: E402
from splatthis.fidelity.metrics import compute_fidelity_metrics  # noqa: E402
from splatthis.io import load_png  # noqa: E402
from splatthis.pptx_export import (  # noqa: E402
    generate_drawingml_slide_content,
    save_pptx_with_drawingml_content,
)
from splatthis.storage import load_splats_json  # noqa: E402

RUNS = REPO / "result" / "corpus" / "runs"
SOURCES = REPO / "result" / "corpus" / "images"
WORK = REPO / "tmp" / "pptx-postfit-law"
IMAGE = "chameleon"
ITERS = 60


def _metrics(source: np.ndarray, capture: Path) -> Dict[str, float]:
    rendered = np.asarray(load_png(str(capture))[..., :3], dtype=np.float32)
    height, width = source.shape[:2]
    rois = [
        (y, x, min(y + 64, height), min(x + 64, width))
        for y in range(0, height, 64)
        for x in range(0, width, 64)
    ]
    m = compute_fidelity_metrics(
        source, rendered, fixed_rois=rois, render_method="pp"
    ).as_dict()
    return {
        k: round(float(m[k]), 4)
        for k in ("ssim_srgb", "lpips", "delta_e_ok_mean", "delta_e_ok_p95")
    }


def run(tag: str) -> Dict[str, Any]:
    WORK.mkdir(parents=True, exist_ok=True)
    art = RUNS / f"{IMAGE}_pptx_s0_art"
    splats = load_splats_json(str(art / "final.raw.json"))
    manifest = json.loads((art / "run_manifest.json").read_text(encoding="utf-8"))
    background = np.asarray(
        manifest["config"]["background_linear_rgb"], dtype=np.float32
    )
    image = load_png(str(SOURCES / f"{IMAGE}.png"))
    height, width = image.shape[:2]

    converter = PNG2SVGConverter(
        max_splats=len(splats),
        refinement_config={"pptx_proxy_postfit_iters": ITERS},
    )
    converter._background_linear_rgb = background
    refined, metric = converter._postfit_splats_for_pptx_proxy(
        splats=splats,
        image=image,
        width=width,
        height=height,
        num_iters=ITERS,
        verbose=False,
    )
    print(
        f"  post-fit stage reported: {metric.get('stage_type')} "
        f"iterations={metric.get('iterations')}"
    )

    deck = WORK / f"{IMAGE}-{tag}.pptx"
    save_pptx_with_drawingml_content(
        slide_xml=generate_drawingml_slide_content(
            refined, width=width, height=height, background_linear_rgb=background
        ),
        width=width,
        height=height,
        output_path=str(deck),
        splat_count=len(refined),
    )
    capture = WORK / f"{IMAGE}-{tag}-powerpoint.png"
    print(f"  capturing {tag} deck in PowerPoint ...", flush=True)
    code, message = _capture_powerpoint_slideshow(deck, capture, width, height)
    if code or not capture.exists():
        raise SystemExit(f"capture failed: {message.strip()[-300:]}")

    source_linear = np.asarray(image[..., :3], dtype=np.float32)
    scores = _metrics(source_linear, capture)
    (WORK / f"{tag}.json").write_text(json.dumps(scores, indent=1) + "\n")
    print(f"  {tag}: {scores}")
    return scores


def compare() -> int:
    baseline = RUNS / f"{IMAGE}_pptx_s0_powerpoint_slide.png"
    source = np.asarray(
        load_png(str(SOURCES / f"{IMAGE}.png"))[..., :3], dtype=np.float32
    )
    rows = [("no post-fit (shipped deck)", _metrics(source, baseline))]
    for tag in ("before", "after"):
        path = WORK / f"{tag}.json"
        if path.is_file():
            rows.append((f"post-fit, {tag} fix", json.loads(path.read_text())))
    print(f"\n{'variant':30s}{'ssim':>9s}{'lpips':>9s}{'dE mean':>10s}{'dE p95':>9s}")
    for label, s in rows:
        print(
            f"{label:30s}{s['ssim_srgb']:>9.4f}{s['lpips']:>9.4f}"
            f"{s['delta_e_ok_mean']:>10.4f}{s['delta_e_ok_p95']:>9.4f}"
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", choices=("before", "after"))
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()
    if args.tag:
        run(args.tag)
    if args.compare or not args.tag:
        return compare()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
