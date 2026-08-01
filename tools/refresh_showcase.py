#!/usr/bin/env python3
"""Pick the best surviving demo artifacts and install them into ``docs/demo/``.

The repository accumulates experiment output under ``tmp/``. Those runs hold the
good artifacts, but a recorded metric is not a usable index into them: names get
reused across runs, byte sizes drift as candidates are regenerated, and the
artifact behind a given number may sit several directories away from the JSON
that reports it.

This is not hypothetical. The CSS experiment recorded a winner at 0.8748
SSIM_sRGB. Nothing in the directory named in that report scored above 0.8563,
and no file there matched the recorded byte size — the actual winner was in a
``stage2/`` subdirectory, under a different name. Matching by name or size
would have shipped a weaker artifact while quoting the better score.

So this tool re-measures. It reads nothing but the artifacts themselves, scores
each in its governing renderer, and ranks on the numbers it just computed.
Recorded metrics are ignored entirely — they are a record of what a past run
saw, not of what is on disk now.

Usage::

    python tools/refresh_showcase.py --dry-run     # rank candidates, change nothing
    python tools/refresh_showcase.py               # install the winners

PPTX is excluded by default: scoring it means driving a real PowerPoint window,
which is interactive and cannot run headless. Pass ``--with-pptx`` and see
``docs/PROVENANCE_AND_BENCHMARKS.md`` for that path.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from splatthis import evaluate_svg_export_quality  # noqa: E402
from splatthis.color import srgb_to_linear  # noqa: E402
from splatthis.quality import compute_quality_metrics  # noqa: E402

SOURCE = REPO / "docs" / "demo" / "source.png"
DEMO = REPO / "docs" / "demo"
VIEWBOX = re.compile(r'viewBox="0 0 (\d+) (\d+)"')


@dataclass
class Candidate:
    path: Path
    score: Optional[float]
    width: int
    height: int
    note: str = ""

    @property
    def megabytes(self) -> float:
        return self.path.stat().st_size / 1048576


def _target(width: int, height: int) -> np.ndarray:
    src = Image.open(SOURCE).convert("RGB").resize((width, height), Image.LANCZOS)
    return srgb_to_linear(np.asarray(src, dtype=np.float32) / 255.0)


def score_svg(path: Path) -> Candidate:
    """Score an SVG in Chromium, the governing renderer for browser targets."""
    match = VIEWBOX.search(path.read_text(errors="ignore")[:2000])
    if not match:
        return Candidate(path, None, 0, 0, "no viewBox")
    width, height = int(match.group(1)), int(match.group(2))
    result = evaluate_svg_export_quality(_target(width, height), str(path))
    if not result.get("available"):
        return Candidate(path, None, width, height, "capture unavailable")
    metrics = result.get("metrics", result)
    return Candidate(path, metrics.get("ssim_srgb"), width, height)


def score_capture(path: Path) -> Candidate:
    """Score an already-captured PNG (a CSS or pixel-runtime render)."""
    img = Image.open(path).convert("RGB")
    candidate = srgb_to_linear(np.asarray(img, dtype=np.float32) / 255.0)
    metrics = compute_quality_metrics(_target(*img.size), candidate)
    return Candidate(path, metrics.get("ssim_srgb"), img.width, img.height)


# Scores within this of the best are treated as a tie and settled on file size.
# These assets are embedded in the README and served by GitHub Pages, so a
# fraction of a thousandth of SSIM does not justify a megabyte: the top SVG
# scored 0.8668 at 2.08 MB against 0.8665 at 1.51 MB, a difference no viewer
# can see attached to 38% more weight.
TIE_EPSILON = 0.001


def rank(candidates: list[Candidate]) -> list[Candidate]:
    """Rank by score, then prefer the smallest artifact among near-ties."""
    scored = [c for c in candidates if c.score is not None]
    if not scored:
        return []
    scored.sort(key=lambda c: -c.score)  # type: ignore[operator,arg-type]
    best = scored[0].score or 0.0
    contenders = [c for c in scored if (best - (c.score or 0.0)) <= TIE_EPSILON]
    # Bucket by size to two decimals before comparing, so a few hundred bytes
    # cannot outrank a real quality difference. Within a bucket the best score
    # wins; across buckets the lighter artifact does.
    contenders.sort(key=lambda c: (round(c.megabytes, 2), -(c.score or 0.0)))
    remainder = [c for c in scored if c not in contenders]
    return contenders + remainder


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true", help="rank candidates without installing"
    )
    parser.add_argument(
        "--with-pptx",
        action="store_true",
        help="also rank PPTX (opens a real PowerPoint window; not headless)",
    )
    parser.add_argument("--tmp", type=Path, default=REPO / "tmp")
    args = parser.parse_args()

    if not SOURCE.is_file():
        print(f"missing source image: {SOURCE}", file=sys.stderr)
        return 1

    svgs = [
        p
        for p in args.tmp.rglob("*.svg")
        if p.stat().st_size > 256_000  # skip fixtures and per-splat fragments
    ]
    print(f"scoring {len(svgs)} SVG candidates in Chromium ...")
    svg_ranked = rank([score_svg(p) for p in svgs])

    # A CSS candidate is a capture with a sibling .html that emitted it. Matching
    # on directory name alone is not enough: a directory called
    # "css-compositor-*" also holds canvas and pixel-runtime renders, and one of
    # those outscores every real CSS build — selecting it would put a canvas
    # render in the README under a CSS label.
    css_pngs = sorted(
        p
        for p in args.tmp.rglob("*.png")
        if p.with_suffix(".html").is_file()
        and not any(token in p.name.lower() for token in ("canvas", "pixel", "runtime"))
    )
    print(f"scoring {len(css_pngs)} CSS captures (each has a sibling .html) ...")
    css_ranked = rank([score_capture(p) for p in css_pngs])

    def report(title: str, ranked: list[Candidate], limit: int = 5) -> None:
        print(f"\n=== {title} ===")
        if not ranked:
            print("  (no scorable candidates)")
            return
        for i, c in enumerate(ranked[:limit]):
            flag = "  <== winner" if i == 0 else ""
            rel = c.path.relative_to(REPO)
            print(
                f"  {c.score:.4f}  {c.width}x{c.height}  "
                f"{c.megabytes:5.2f} MB  {rel}{flag}"
            )

    report("SVG (Chromium)", svg_ranked)
    report("CSS (Chromium capture)", css_ranked)

    if args.with_pptx:
        print(
            "\n=== PPTX ===\n"
            "  Not automated: scoring requires a real PowerPoint slideshow capture.\n"
            "  Use ~/projects/svg2ooxml/tools/ppt_research/powerpoint_capture_cli.py\n"
            "  and never soffice, which renders DrawingML incorrectly."
        )

    if args.dry_run:
        print("\ndry run: nothing installed")
        return 0

    if svg_ranked:
        shutil.copy2(svg_ranked[0].path, DEMO / "chameleon.svg")
        print(f"\ninstalled {svg_ranked[0].path.name} -> docs/demo/chameleon.svg")
    if css_ranked:
        html = css_ranked[0].path.with_suffix(".html")
        if html.is_file():
            shutil.copy2(html, DEMO / "chameleon-css.html")
            print(f"installed {html.name} -> docs/demo/chameleon-css.html")
        else:
            print(f"no sibling .html for {css_ranked[0].path.name}; CSS left unchanged")

    print("\nUpdate the scores quoted in README.md to match the numbers above.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
