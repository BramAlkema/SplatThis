#!/usr/bin/env python3
"""MVP: circumvent PowerPoint's gradient rendering with solid-alpha rings.

The color-space probe proved PowerPoint composites *solid-alpha* shapes in
clean, predictable display sRGB (within 0.006 of the model), while its
alpha-ramp gradients leave the pptx emitter with a median 0.10 LPIPS of
loss against the reference render -- three to five times the SVG or CSS
emitters. This MVP replaces every gradient splat with K concentric
solid-alpha ellipses: a stepwise Gaussian whose ring alphas are solved so
the cumulative sRGB alpha-over composite matches the Gaussian profile at
each ring's midpoint. Same populations, same painter order, same governing
real-PowerPoint capture; only the primitive changes.

Usage::

    PYTHONPATH=src python tools/pptx_ring_stack_mvp.py
"""

from __future__ import annotations

import json
import math
import os
import re
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
from splatthis.pptx_export import (  # noqa: E402
    generate_drawingml_slide_content,
    save_pptx_with_drawingml_content,
)
from splatthis.storage import load_splats_json  # noqa: E402

RUNS = REPO / "result" / "corpus" / "runs"
SOURCES = REPO / "result" / "corpus" / "images"
WORK = REPO / "tmp" / "pptx-ring-stack-mvp"

IMAGES = tuple(
    os.environ.get("RING_IMAGES", "chameleon,text,hubble_deep_field,colorwheel").split(
        ","
    )
)
K_SIGMA = 2.5
_K = int(os.environ.get("RING_COUNT", "4"))
RING_FRACTIONS = tuple((_K - i) / _K for i in range(_K))
#: Feather each ring with the calibrated DrawingML blur (sigma = rad/3.25)
#: to melt the stepwise falloff into a smooth ramp; the factor scales the
#: blur radius relative to the ring gap. 0 disables.
_BLUR = float(os.environ.get("RING_BLUR", "0"))
SUFFIX = f"-k{_K}" + (f"-b{_BLUR:g}" if _BLUR else "")
if SUFFIX == "-k4":
    SUFFIX = ""

SP_PATTERN = re.compile(r"<p:sp>.*?</p:sp>", re.S)
XFRM_PATTERN = re.compile(
    r'<a:xfrm(?:\s+rot="(-?\d+)")?>\s*<a:off x="(-?\d+)" y="(-?\d+)"/>\s*'
    r'<a:ext cx="(\d+)" cy="(\d+)"/>'
)
COLOR_PATTERN = re.compile(r'<a:srgbClr val="([0-9A-Fa-f]{6})">')
PEAK_ALPHA_PATTERN = re.compile(
    r'<a:gs pos="0">\s*<a:srgbClr val="[0-9A-Fa-f]{6}">\s*<a:alpha val="(\d+)"/>'
)


def ring_alphas(peak_alpha: float) -> List[float]:
    """Per-ring alphas so cumulative sRGB-over matches the Gaussian profile.

    Region j (between ring j+1 and ring j, outermost first) is covered by
    rings 1..j; its target is the Gaussian at the region midpoint.
    """
    fractions = list(RING_FRACTIONS) + [0.0]
    midpoints = [
        (fractions[i] + fractions[i + 1]) / 2 for i in range(len(RING_FRACTIONS))
    ]
    targets = [peak_alpha * math.exp(-((K_SIGMA * m) ** 2) / 2.0) for m in midpoints]
    alphas: List[float] = []
    remaining = 1.0
    for target in targets:
        cumulative_before = 1.0 - remaining
        ring = (target - cumulative_before) / remaining if remaining > 1e-6 else 0.0
        ring = min(max(ring, 0.0), 1.0)
        alphas.append(ring)
        remaining *= 1.0 - ring
    return alphas


def transform_shape(match: re.Match, counter: List[int]) -> str:
    shape = match.group(0)
    if "<a:gradFill>" not in shape:
        return shape
    xfrm = XFRM_PATTERN.search(shape)
    color = COLOR_PATTERN.search(shape)
    peak = PEAK_ALPHA_PATTERN.search(shape)
    if not (xfrm and color and peak):
        return shape
    rot = f' rot="{xfrm.group(1)}"' if xfrm.group(1) else ""
    off_x, off_y = int(xfrm.group(2)), int(xfrm.group(3))
    ext_x, ext_y = int(xfrm.group(4)), int(xfrm.group(5))
    peak_alpha = int(peak.group(1)) / 100000.0

    rings = []
    for fraction, alpha in zip(RING_FRACTIONS, ring_alphas(peak_alpha)):
        if alpha <= 0.0005:
            continue
        counter[0] += 1
        cx = max(1, int(round(ext_x * fraction)))
        cy = max(1, int(round(ext_y * fraction)))
        ox = off_x + (ext_x - cx) // 2
        oy = off_y + (ext_y - cy) // 2
        rings.append(
            f'<p:sp><p:nvSpPr><p:cNvPr id="{counter[0]}" '
            f'name="Ring {counter[0]}"/><p:cNvSpPr>'
            f'<a:spLocks noGrp="1"/></p:cNvSpPr><p:nvPr/></p:nvSpPr>'
            f"<p:spPr><a:xfrm{rot}>"
            f'<a:off x="{ox}" y="{oy}"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>'
            f'<a:prstGeom prst="ellipse"><a:avLst/></a:prstGeom>'
            f'<a:solidFill><a:srgbClr val="{color.group(1).upper()}">'
            f'<a:alpha val="{int(round(alpha * 100000))}"/></a:srgbClr>'
            f"</a:solidFill><a:ln><a:noFill/></a:ln>"
            + (
                f'<a:effectLst><a:blur rad="{int(round(_BLUR * (ext_x + ext_y) / (4 * _K)))}"/>'
                f"</a:effectLst>"
                if _BLUR
                else ""
            )
            + "</p:spPr></p:sp>"
        )
    return "".join(rings) if rings else shape


def build_ring_deck(image: str, deck_path: Path) -> int:
    run = RUNS / f"{image}_pptx_s0_art"
    splats = load_splats_json(str(run / "final.raw.json"))
    manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    background = np.asarray(
        manifest["config"]["background_linear_rgb"], dtype=np.float32
    )
    source = load_png(str(SOURCES / f"{image}.png"))
    height, width = source.shape[:2]
    slide_xml = generate_drawingml_slide_content(
        splats,
        width=width,
        height=height,
        background_linear_rgb=background,
    )
    counter = [10000]
    transformed = SP_PATTERN.sub(lambda m: transform_shape(m, counter), slide_xml)
    save_pptx_with_drawingml_content(
        slide_xml=transformed,
        width=width,
        height=height,
        output_path=str(deck_path),
        splat_count=len(splats),
    )
    return counter[0] - 10000


def score(image: str, capture_path: Path) -> Dict[str, float]:
    source = np.asarray(
        load_png(str(SOURCES / f"{image}.png"))[..., :3], dtype=np.float32
    )
    rendered = np.asarray(load_png(str(capture_path))[..., :3], dtype=np.float32)
    height, width = source.shape[:2]
    rois = [
        (y, x, min(y + 64, height), min(x + 64, width))
        for y in range(0, height, 64)
        for x in range(0, width, 64)
    ]
    metrics = compute_fidelity_metrics(
        source, rendered, fixed_rois=rois, render_method="pp"
    ).as_dict()
    return {
        name: round(float(metrics[name]), 4)
        for name in ("ssim_srgb", "lpips", "delta_e_ok_mean", "delta_e_ok_p95")
    }


def main() -> int:
    WORK.mkdir(parents=True, exist_ok=True)
    baseline = {
        row["image"]: row
        for row in (
            json.loads(line)
            for line in (REPO / "result" / "corpus" / "powerpoint_results.jsonl")
            .read_text()
            .splitlines()
            if line
        )
    }
    results: Dict[str, Any] = {}
    for index, image in enumerate(IMAGES, 1):
        deck = WORK / f"{image}-rings{SUFFIX}.pptx"
        capture = WORK / f"{image}-rings{SUFFIX}-powerpoint.png"
        if not deck.exists():
            rings = build_ring_deck(image, deck)
            print(
                f"[{index}/{len(IMAGES)}] {image}: emitted {rings} ring shapes",
                flush=True,
            )
        if not capture.exists():
            source = load_png(str(SOURCES / f"{image}.png"))
            height, width = source.shape[:2]
            print(f"[{index}/{len(IMAGES)}] {image}: capturing ...", flush=True)
            returncode, message = _capture_powerpoint_slideshow(
                deck, capture, width, height
            )
            if returncode or not capture.exists():
                print(f"capture failed: {message.strip()[-300:]}", file=sys.stderr)
                return 2
        results[image] = {
            "rings": score(image, capture),
            "baseline_gradient": {
                k: round(float(baseline[image][j]), 4)
                for k, j in (("ssim_srgb", "ssim_srgb"), ("lpips", "lpips"))
            },
            "deck_bytes": deck.stat().st_size,
        }
        print(f"[{index}/{len(IMAGES)}] {image}: scored", flush=True)

    (WORK / f"results{SUFFIX}.json").write_text(json.dumps(results, indent=1) + "\n")
    print(f"\nwrote {WORK / ('results' + SUFFIX + '.json')}\n")
    print(
        f"{'image':22s}{'variant':>10s}{'ssim':>10s}{'lpips':>10s}"
        f"{'dE-mean':>10s}{'dE-p95':>10s}{'deck KB':>10s}"
    )
    for image, entry in results.items():
        b = entry["baseline_gradient"]
        r = entry["rings"]
        print(
            f"{image:22s}{'gradient':>10s}{b['ssim_srgb']:>10.4f}"
            f"{b['lpips']:>10.4f}{'—':>10s}{'—':>10s}{'—':>10s}"
        )
        print(
            f"{image:22s}{'rings':>10s}{r['ssim_srgb']:>10.4f}"
            f"{r['lpips']:>10.4f}{r['delta_e_ok_mean']:>10.4f}"
            f"{r['delta_e_ok_p95']:>10.4f}"
            f"{entry['deck_bytes'] / 1024:>10.0f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
