#!/usr/bin/env python3
"""Build a side-by-side page for the PPTX training-target experiments.

The fidelity protocol says a win is not a win until it survives a
side-by-side look at the deployed artifact. This assembles one page from the
real PowerPoint captures of every variant -- shipping gradient, hybrid-aware
gradient fit, and the closed ring-stack line -- at native size and at 4x on
a smooth region where banding and colour shifts show, with each cell's
measured scores underneath.

Missing variants are skipped, so it can be run while a fit is still going.

Usage::

    PYTHONPATH=src python tools/pptx_side_by_side.py [--open]
"""

from __future__ import annotations

import argparse
import base64
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from splatthis.fidelity.metrics import compute_fidelity_metrics  # noqa: E402
from splatthis.io import load_png  # noqa: E402

SOURCES = REPO / "result" / "corpus" / "images"
OUT = REPO / "tmp" / "pptx-side-by-side.html"

#: label -> capture path template. Order is display order.
VARIANTS: Tuple[Tuple[str, str, str], ...] = (
    (
        "Gradient (shipping)",
        "result/corpus/runs/{image}_pptx_s0_powerpoint_slide.png",
        "1,675 shapes · 161 KB · opens 4.1 s",
    ),
    (
        "Hybrid-aware fit",
        "tmp/pptx-gradient-training-mvp/{image}-gradtrained-powerpoint.png",
        "same shapes · same size · same open time",
    ),
    (
        "Ring stack + fit (closed)",
        "tmp/pptx-ring-training-mvp/{image}-ringtrained-powerpoint.png",
        "13,393 shapes · 522 KB · opens 34.0 s",
    ),
)

#: Per image, a crop that shows smooth gradient falloff (x, y, w, h).
ZOOMS: Dict[str, Tuple[int, int, int, int]] = {
    "chameleon": (0, 0, 140, 120),
    "colorwheel": (95, 95, 140, 120),
    "text": (0, 0, 140, 100),
}


def _b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _zoom_b64(path: Path, box: Tuple[int, int, int, int], factor: int = 4) -> str:
    from io import BytesIO

    from PIL import Image

    x, y, w, h = box
    with Image.open(path) as img:
        crop = img.convert("RGB").crop((x, y, x + w, y + h))
        crop = crop.resize((w * factor, h * factor), Image.NEAREST)
        buffer = BytesIO()
        crop.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def score(image: str, capture: Path) -> Optional[Dict[str, float]]:
    source = np.asarray(
        load_png(str(SOURCES / f"{image}.png"))[..., :3], dtype=np.float32
    )
    rendered = np.asarray(load_png(str(capture))[..., :3], dtype=np.float32)
    if rendered.shape != source.shape:
        return None
    height, width = source.shape[:2]
    rois = [
        (y, x, min(y + 64, height), min(x + 64, width))
        for y in range(0, height, 64)
        for x in range(0, width, 64)
    ]
    m = compute_fidelity_metrics(
        source, rendered, fixed_rois=rois, render_method="pp"
    ).as_dict()
    return {k: float(m[k]) for k in ("ssim_srgb", "lpips", "delta_e_ok_mean")}


HEAD = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>PPTX training targets — real PowerPoint captures</title>
<style>
 *{box-sizing:border-box}
 body{margin:0;padding:40px 24px 72px;background:#101012;color:#eaeaea;
      font-family:-apple-system,BlinkMacSystemFont,system-ui,sans-serif}
 main{max-width:1500px;margin:0 auto}
 h1{font-size:26px;margin:0 0 6px}
 p.note{color:#9c9ca5;font-size:13px;max-width:80ch;line-height:1.55}
 h2{font-size:16px;font-family:ui-monospace,monospace;margin:38px 0 4px}
 .row{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));
      gap:12px;margin:12px 0 4px}
 figure{margin:0;background:#1a1a1d;border:1px solid #2a2a2e;border-radius:8px;
        padding:9px}
 img{width:100%;height:auto;display:block;border-radius:4px;
     image-rendering:pixelated}
 figcaption{font-family:ui-monospace,monospace;font-size:11px;color:#9c9ca5;
            margin-top:7px;line-height:1.55}
 figcaption strong{color:#eaeaea}
 .win{color:#7fd17f}.loss{color:#e58a8a}
</style></head><body><main>
<h1>PPTX training targets, judged on real PowerPoint</h1>
<p class="note">Every image below is a real Microsoft PowerPoint slideshow
capture of the deployed deck, ICC-converted to sRGB. The second row of each
pair is a 4x nearest-neighbour zoom on a smooth region, where banding and
colour shifts are visible and metrics are not. Deltas are against the
shipping gradient deck; green means better.</p>
"""


def build(images: List[str]) -> str:
    parts = [HEAD]
    for image in images:
        source_path = SOURCES / f"{image}.png"
        if not source_path.is_file():
            continue
        available = [
            (label, REPO / template.format(image=image), note)
            for label, template, note in VARIANTS
            if (REPO / template.format(image=image)).is_file()
        ]
        if not available:
            continue
        base = score(image, available[0][1]) if available else None
        parts.append(f"<h2>{image}</h2>")

        cells = [
            '<figure><img src="data:image/png;base64,'
            + _b64(source_path)
            + '"><figcaption><strong>Source</strong><br>original bitmap'
            "</figcaption></figure>"
        ]
        for label, path, note in available:
            s = score(image, path)
            if s is None:
                metrics = "size mismatch"
            else:

                def delta(key: str, lower_better: bool) -> str:
                    if base is None or label == available[0][0]:
                        return ""
                    d = s[key] - base[key]
                    good = (d < 0) if lower_better else (d > 0)
                    cls = "win" if good else "loss"
                    return f' <span class="{cls}">({d:+.4f})</span>'

                metrics = (
                    f"LPIPS {s['lpips']:.4f}{delta('lpips', True)}<br>"
                    f"SSIM {s['ssim_srgb']:.4f}{delta('ssim_srgb', False)}<br>"
                    f"ΔE {s['delta_e_ok_mean']:.4f}"
                    f"{delta('delta_e_ok_mean', True)}"
                )
            cells.append(
                '<figure><img src="data:image/png;base64,'
                + _b64(path)
                + f'"><figcaption><strong>{label}</strong><br>{metrics}'
                f"<br>{note}</figcaption></figure>"
            )
        parts.append('<div class="row">' + "".join(cells) + "</div>")

        box = ZOOMS.get(image, (0, 0, 140, 120))
        zoom_cells = [
            '<figure><img src="data:image/png;base64,'
            + _zoom_b64(source_path, box)
            + '"><figcaption>source, 4x</figcaption></figure>'
        ]
        for label, path, _ in available:
            zoom_cells.append(
                '<figure><img src="data:image/png;base64,'
                + _zoom_b64(path, box)
                + f'"><figcaption>{label}, 4x</figcaption></figure>'
            )
        parts.append('<div class="row">' + "".join(zoom_cells) + "</div>")

    parts.append("</main></body></html>")
    return "\n".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", default="chameleon,colorwheel,text")
    parser.add_argument("--open", action="store_true", help="open in the browser")
    args = parser.parse_args()

    images = [i.strip() for i in args.images.split(",") if i.strip()]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(build(images), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")
    if args.open:
        import subprocess

        subprocess.run(["open", str(OUT)], check=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
