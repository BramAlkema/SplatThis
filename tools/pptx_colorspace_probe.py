#!/usr/bin/env python3
"""Measure which color space real PowerPoint composites in.

"PowerPoint renders less vibrant" hides two separate questions: in what
space does it interpolate gradFill stops, and in what space does it
alpha-blend shapes? This probe answers both with one slide and one
20-second real-PowerPoint capture, plus a corpus-wide tally against the
existing captures.

**Probe deck.** A 384x384 slide with designed patches: a 50%-alpha red
rectangle over a green field (alpha-blend probe), a red-to-green gradient
ramp (stop-interpolation probe), a black-to-transparent ramp over white
(alpha-ramp probe), and opaque calibration swatches. Linear-light and
display-sRGB compositing predict very different midpoints (roughly sRGB 188
versus 128 per channel); predictions are computed from the *captured*
calibration swatches, so a display-profile shift in the capture chain
cancels out.

**Corpus tally.** Each stored pptx population is rendered by the internal
reference renderer under both compositing spaces and compared with its real
PowerPoint capture; the winner per image says which model describes the
whole chain better.

Usage::

    PYTHONPATH=src python tools/pptx_colorspace_probe.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

from full_corpus_mvp import _capture_powerpoint_slideshow  # noqa: E402

from splatthis.color import linear_to_srgb, srgb_to_linear  # noqa: E402
from splatthis.io import load_png, px_to_emu  # noqa: E402
from splatthis.pptx_export import (  # noqa: E402
    generate_drawingml_slide_content,
    save_pptx_with_drawingml_content,
)
from splatthis.renderer import render_splats_numpy  # noqa: E402
from splatthis.splat import create_isotropic_splat  # noqa: E402
from splatthis.storage import load_splats_json  # noqa: E402

RUNS = REPO / "result" / "corpus" / "runs"
SOURCES = REPO / "result" / "corpus" / "images"
WORK = REPO / "tmp" / "pptx-colorspace-probe"
SIZE = 384


def _rect(shape_id: int, x: int, y: int, w: int, h: int, fill: str) -> str:
    return (
        f'<p:sp><p:nvSpPr><p:cNvPr id="{shape_id}" name="probe{shape_id}"/>'
        f"<p:cNvSpPr/><p:nvPr/></p:nvSpPr><p:spPr><a:xfrm>"
        f'<a:off x="{px_to_emu(x)}" y="{px_to_emu(y)}"/>'
        f'<a:ext cx="{px_to_emu(w)}" cy="{px_to_emu(h)}"/></a:xfrm>'
        f'<a:prstGeom prst="rect"><a:avLst/></a:prstGeom>{fill}'
        f"<a:ln><a:noFill/></a:ln></p:spPr></p:sp>"
    )


def _solid(color: str, alpha_pct: float = 100.0) -> str:
    alpha = (
        f'<a:alpha val="{int(round(alpha_pct * 1000))}"/>' if alpha_pct < 100 else ""
    )
    return f'<a:solidFill><a:srgbClr val="{color}">{alpha}</a:srgbClr></a:solidFill>'


def _gradient(start: str, end: str, start_alpha: float, end_alpha: float) -> str:
    def stop(pos: int, color: str, alpha_pct: float) -> str:
        alpha = (
            f'<a:alpha val="{int(round(alpha_pct * 1000))}"/>'
            if alpha_pct < 100
            else ""
        )
        return f'<a:gs pos="{pos}"><a:srgbClr val="{color}">{alpha}</a:srgbClr></a:gs>'

    return (
        f"<a:gradFill><a:gsLst>{stop(0, start, start_alpha)}"
        f'{stop(100000, end, end_alpha)}</a:gsLst><a:lin ang="0" scaled="1"/>'
        f"</a:gradFill>"
    )


def build_probe_deck(path: Path) -> None:
    """One slide of probe patches inside the emitter's own slide envelope."""
    dummy = create_isotropic_splat(
        center=[10.0, 10.0], sigma=2.0, color=[0.5, 0.5, 0.5], alpha=0.5
    )
    envelope = generate_drawingml_slide_content([dummy], width=SIZE, height=SIZE)
    shapes = "".join(
        (
            _rect(100, 0, 0, SIZE, SIZE, _solid("00FF00")),
            _rect(101, 0, 0, 192, SIZE, _solid("FF0000", 50.0)),
            _rect(102, 200, 8, 72, 72, _solid("FF0000")),
            _rect(103, 280, 8, 72, 72, _solid("FFFFFF")),
            _rect(104, 200, 88, 72, 72, _solid("000000")),
            _rect(105, 200, 168, 176, 80, _gradient("FF0000", "00FF00", 100, 100)),
            _rect(106, 200, 264, 176, 80, _solid("FFFFFF")),
            _rect(107, 200, 264, 176, 80, _gradient("000000", "000000", 100, 0)),
        )
    )
    start = envelope.index("<p:grpSp>")
    stop = envelope.rindex("</p:grpSp>") + len("</p:grpSp>")
    slide_xml = envelope[:start] + shapes + envelope[stop:]
    save_pptx_with_drawingml_content(
        slide_xml=slide_xml,
        width=SIZE,
        height=SIZE,
        output_path=str(path),
        splat_count=8,
    )


def _patch(srgb: np.ndarray, x: int, y: int, half: int = 5) -> np.ndarray:
    return srgb[y - half : y + half, x - half : x + half].reshape(-1, 3).mean(axis=0)


def classify(name: str, measured, srgb_pred, linear_pred) -> Dict[str, object]:
    d_srgb = float(np.linalg.norm(measured - srgb_pred))
    d_linear = float(np.linalg.norm(measured - linear_pred))
    verdict = "srgb" if d_srgb < d_linear else "linear"
    print(
        f"{name:18s} measured {np.round(measured * 255)} · "
        f"sRGB-pred {np.round(srgb_pred * 255)} (d={d_srgb:.3f}) · "
        f"linear-pred {np.round(linear_pred * 255)} (d={d_linear:.3f}) "
        f"→ {verdict.upper()}"
    )
    return {
        "measured_srgb255": [float(v) for v in measured * 255],
        "distance_srgb_model": d_srgb,
        "distance_linear_model": d_linear,
        "verdict": verdict,
    }


def run_probe() -> Dict[str, object]:
    WORK.mkdir(parents=True, exist_ok=True)
    deck = WORK / "probe.pptx"
    capture = WORK / "probe-powerpoint.png"
    if not deck.exists():
        build_probe_deck(deck)
    if not capture.exists():
        returncode, message = _capture_powerpoint_slideshow(deck, capture, SIZE, SIZE)
        if returncode or not capture.exists():
            raise RuntimeError(f"probe capture failed: {message.strip()[-300:]}")

    linear = np.asarray(load_png(str(capture))[..., :3], dtype=np.float32)
    srgb = np.clip(linear_to_srgb(linear), 0.0, 1.0)

    red = _patch(srgb, 236, 44)
    white = _patch(srgb, 316, 44)
    black = _patch(srgb, 236, 124)
    green = _patch(srgb, 316, 124)
    print(
        f"calibration (sRGB 0-255): red {np.round(red * 255)}, "
        f"green {np.round(green * 255)}, white {np.round(white * 255)}, "
        f"black {np.round(black * 255)}"
    )

    def mix_srgb(a, b, t=0.5):
        return (1 - t) * a + t * b

    def mix_linear(a, b, t=0.5):
        return linear_to_srgb(
            (1 - t) * srgb_to_linear(a.reshape(1, 1, 3))
            + t * srgb_to_linear(b.reshape(1, 1, 3))
        ).reshape(3)

    results: Dict[str, object] = {}
    results["alpha_blend"] = classify(
        "alpha blend",
        _patch(srgb, 96, 192),
        mix_srgb(green, red),
        mix_linear(green, red),
    )
    results["gradient_interp"] = classify(
        "gradient interp",
        _patch(srgb, 288, 208),
        mix_srgb(_patch(srgb, 210, 208), _patch(srgb, 366, 208)),
        mix_linear(_patch(srgb, 210, 208), _patch(srgb, 366, 208)),
    )
    results["alpha_ramp"] = classify(
        "alpha ramp",
        _patch(srgb, 288, 304),
        mix_srgb(white, black),
        mix_linear(white, black),
    )
    return results


def corpus_tally() -> Dict[str, object]:
    from splatthis.fidelity.metrics import compute_fidelity_metrics

    images = sorted(p.stem for p in SOURCES.glob("*.png"))
    tally = {"linear": 0, "srgb": 0}
    rows: List[Dict[str, object]] = []
    for image in images:
        run = RUNS / f"{image}_pptx_s0_art"
        capture = RUNS / f"{image}_pptx_s0_powerpoint_slide.png"
        if not (run / "final.raw.json").exists() or not capture.exists():
            continue
        splats = load_splats_json(str(run / "final.raw.json"))
        manifest = json.loads((run / "run_manifest.json").read_text())
        bg = np.asarray(manifest["config"]["background_linear_rgb"], dtype=np.float32)
        rendered_capture = np.asarray(load_png(str(capture))[..., :3], dtype=np.float32)
        height, width = rendered_capture.shape[:2]
        scores = {}
        for space in ("linear", "srgb"):
            model = render_splats_numpy(
                splats,
                width,
                height,
                background_linear_rgb=bg,
                compositing_space=space,
            )[..., :3]
            metrics = compute_fidelity_metrics(
                model.astype(np.float32),
                rendered_capture,
                fixed_rois=[(0, 0, height, width)],
                render_method="probe",
            ).as_dict()
            scores[space] = float(metrics["lpips"])
        winner = min(scores, key=scores.get)
        tally[winner] += 1
        rows.append({"image": image, **scores, "winner": winner})
        print(
            f"{image:22s} model-vs-capture LPIPS: linear {scores['linear']:.4f} "
            f"srgb {scores['srgb']:.4f} → {winner.upper()}"
        )
    print(f"\ncorpus tally: linear {tally['linear']}, srgb {tally['srgb']}")
    return {"tally": tally, "rows": rows}


def main() -> int:
    print("== synthetic probe (one real-PowerPoint capture) ==")
    probe = run_probe()
    print("\n== corpus tally: internal model vs real captures ==")
    tally = corpus_tally()
    out = WORK / "results.json"
    out.write_text(json.dumps({"probe": probe, "corpus": tally}, indent=1) + "\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
