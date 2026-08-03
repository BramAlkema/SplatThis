#!/usr/bin/env python3
"""Can a CSS splat build survive an email client's size budget?

Gmail clips a message at roughly 102 KB of HTML source, so the question is
not "does CSS splatting work in email" but "what does it look like at the
splat count that fits". The standard recipe also leans on things mail
clients do not have -- a shared ``<style>`` block, ``mask-image``, and
``color(srgb-linear ...)`` -- so ``email_safe=True`` emits a variant that
inlines every declaration, folds the colour into the gradient's own stops,
and writes legacy ``rgb()``.

Folding the colour in is not free: the standard recipe masks a solid fill
precisely to stop the browser interpolating colour and opacity together.
This measures that cost instead of assuming it away.

Three cells, so recipe cost is not confused with splat-count cost:

    standard @ full     the published baseline
    standard @ 300      same recipe, email-sized population
    email-safe @ 300    same population, email-safe recipe

Everything is captured through the governing renderer (Playwright Chromium)
and scored with LPIPS alongside SSIM, since SSIM over-rewards the smoothness
that dropping from 9 stops to 6 produces.

Usage::

    PYTHONPATH=src python tools/email_css_mvp.py
    PYTHONPATH=src python tools/email_css_mvp.py --splats 300 --open
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from splatthis.browser_capture import (  # noqa: E402
    render_css_html_in_browser_to_linear_rgb,
)
from splatthis.browser_export import generate_css_splat_html  # noqa: E402
from splatthis.fidelity.metrics import compute_fidelity_metrics  # noqa: E402
from splatthis.io import load_png  # noqa: E402
from splatthis.storage import load_splats_json  # noqa: E402

#: Gmail clips beyond roughly this much HTML source. Measured on the source,
#: not on the transfer encoding: base64 buys about 5 KB, quoted-printable
#: none. The real budget is smaller still, since a message also carries copy,
#: a header, a footer and an unsubscribe link.
GMAIL_CLIP_BYTES = 102 * 1024

WORK = REPO / "tmp" / "email-css"
SOURCE = REPO / "result" / "corpus" / "images" / "chameleon.png"


def _score(source_linear: np.ndarray, html: Path) -> Dict[str, float]:
    height, width = source_linear.shape[:2]
    captured, _renderer = render_css_html_in_browser_to_linear_rgb(
        html_path=str(html), width=width, height=height
    )
    rendered = np.asarray(captured, dtype=np.float32)
    if rendered.ndim == 3 and rendered.shape[2] > 3:
        rendered = rendered[..., :3]
    rois = [
        (y, x, min(y + 64, height), min(x + 64, width))
        for y in range(0, height, 64)
        for x in range(0, width, 64)
    ]
    metrics = compute_fidelity_metrics(
        source_linear, rendered, fixed_rois=rois, render_method="chromium"
    ).as_dict()
    return {k: float(metrics[k]) for k in ("ssim_srgb", "lpips", "delta_e_ok_mean")}


def _background_for(population: Path) -> Optional[np.ndarray]:
    """The backdrop the population was fitted against, from its manifest."""
    manifest = population.parent / "run_manifest.json"
    if not manifest.is_file():
        return None
    config = json.loads(manifest.read_text(encoding="utf-8")).get("config", {})
    value = config.get("background_linear_rgb")
    return None if value is None else np.asarray(value, dtype=np.float32)


def _colour_space_check(
    splats: List[Any],
    width: int,
    height: int,
    background: Optional[np.ndarray],
) -> None:
    """Is rgb() the same colour as color(srgb-linear ...) once rendered?

    If the conversion is not exact, part of any measured delta is a colour
    shift rather than the recipe change, so this is checked before the
    numbers are attributed to anything.
    """
    one = splats[:1]
    out = {}
    for tag, email in (("linear", False), ("srgb", True)):
        path = WORK / f"colourcheck-{tag}.html"
        path.write_text(
            generate_css_splat_html(
                one,
                width=width,
                height=height,
                background_linear_rgb=background,
                email_safe=email,
            ),
            encoding="utf-8",
        )
        captured, _renderer = render_css_html_in_browser_to_linear_rgb(
            html_path=str(path), width=width, height=height
        )
        out[tag] = np.asarray(captured, dtype=np.float32)[..., :3]
    delta = np.abs(out["linear"] - out["srgb"])
    print(
        f"  colour-space check (one splat): max channel delta "
        f"{delta.max():.5f}, mean {delta.mean():.6f} in linear RGB"
    )
    if delta.max() > 0.01:
        print(
            "  NOTE: rgb() and color(srgb-linear ...) do not match; part of "
            "any recipe delta below is a colour shift, not stop count."
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splats", type=int, default=300)
    parser.add_argument(
        "--population",
        default=str(WORK / "art300" / "final.raw.json"),
        help="fit at the email budget (not a subsample of a larger fit)",
    )
    parser.add_argument(
        "--full-population",
        default=None,
        help="optional full-size population for the baseline row",
    )
    parser.add_argument("--open", action="store_true")
    args = parser.parse_args()

    WORK.mkdir(parents=True, exist_ok=True)
    source_linear = np.asarray(load_png(str(SOURCE))[..., :3], dtype=np.float32)
    height, width = source_linear.shape[:2]

    # The fitted backdrop, not the emitter default. Without it a sparse
    # population composites over the wrong colour and the whole build reads
    # far too dark -- which is a harness bug, not a splat-count result.
    background = _background_for(Path(args.population))
    small = load_splats_json(args.population)
    print(f"chameleon {width}x{height}, budget population {len(small)} splats")
    _colour_space_check(small, width, height, background)

    cells = [
        ("standard", small, False),
        ("email-safe", small, True),
    ]
    if args.full_population and Path(args.full_population).is_file():
        full = load_splats_json(args.full_population)
        cells.insert(0, ("standard (full)", full, False))

    rows = []
    for label, population, email in cells:
        name = label.replace(" ", "-").replace("(", "").replace(")", "")
        path = WORK / f"{name}-{len(population)}.html"
        html = generate_css_splat_html(
            population,
            width=width,
            height=height,
            background_linear_rgb=background,
            email_safe=email,
        )
        path.write_text(html, encoding="utf-8")
        scores = _score(source_linear, path)
        rows.append((label, len(population), len(html), scores, path))

    print(
        f"\n{'recipe':18s}{'splats':>7s}{'HTML':>10s}{'fits Gmail':>12s}"
        f"{'SSIM':>9s}{'LPIPS':>9s}{'dE':>8s}"
    )
    for label, count, size, scores, _ in rows:
        fits = "yes" if size <= GMAIL_CLIP_BYTES else "NO"
        print(
            f"{label:18s}{count:>7d}{size / 1024:>9.1f}K{fits:>12s}"
            f"{scores['ssim_srgb']:>9.4f}{scores['lpips']:>9.4f}"
            f"{scores['delta_e_ok_mean']:>8.4f}"
        )

    same = [r for r in rows if r[1] == len(small)]
    if len(same) == 2:
        std, mail = same[0][3], same[1][3]
        print(
            f"\nrecipe cost at {len(small)} splats (email-safe minus standard): "
            f"LPIPS {mail['lpips'] - std['lpips']:+.4f}, "
            f"SSIM {mail['ssim_srgb'] - std['ssim_srgb']:+.4f}, "
            f"bytes {100 * (same[1][2] / same[0][2] - 1):+.0f}%"
        )

    if args.open:
        import subprocess

        subprocess.run(["open", str(rows[-1][4])], check=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
