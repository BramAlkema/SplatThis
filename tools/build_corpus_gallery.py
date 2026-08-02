#!/usr/bin/env python3
"""Build the live corpus gallery for GitHub Pages.

The gallery shows the whole 21-image governing corpus side by side, one row
per image, five columns: the original, the scripted pixel runtime rendering
live, the scriptless CSS build as live DOM, the corrected-exporter SVG as a
live vector, and the PowerPoint deck as a real slideshow capture with the
editable deck for download. Everything except PowerPoint renders live --
these are the deployed artifacts themselves, not screenshots -- because
GitHub Pages, unlike the README, applies no sanitizer.

Two modes:

- ``--emit`` (local only) materializes the artifact assets under
  ``docs/corpus/`` from repository state that CI does not have: source
  images and PowerPoint captures/decks from ``result/corpus/``, the
  current-emitter SVGs from ``result/svg-quality/``, and CSS plus
  pixel-runtime builds emitted by the shipped emitters from the stored
  seed-0 populations (the same populations the fidelity registry measured,
  so the quoted numbers describe these exact artifacts). Splat counts are
  recorded in ``docs/corpus/stats.json`` at emit time.
- default / ``--check`` renders ``docs/corpus/index.html`` from the
  committed assets plus the cross-checked ledgers, failing closed on any
  missing artifact, missing stats, or registry mismatch.

``tests/unit/test_corpus_gallery.py`` runs ``--check`` in CI.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO = Path(__file__).resolve().parents[1]
GALLERY = REPO / "docs" / "corpus"
INDEX = GALLERY / "index.html"
STATS = GALLERY / "stats.json"
SOURCES = REPO / "result" / "corpus" / "images"
SVG_LAB = REPO / "result" / "svg-quality"
RUNS = REPO / "result" / "corpus" / "runs"
PPT_RESULTS = REPO / "result" / "corpus" / "powerpoint_results.jsonl"

#: Pixel-runtime pages are flex-centered with 16px padding and a status line.
RUNTIME_PAD_W = 32
RUNTIME_PAD_H = 52


def _load_readme_tool() -> Any:
    spec = importlib.util.spec_from_file_location(
        "update_readme", Path(__file__).with_name("update_readme.py")
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TOOL = _load_readme_tool()


def corpus_images(registry: Dict[str, Any]) -> List[str]:
    return sorted(p["image"] for p in registry["per_image"]["svg"])


def content_classes() -> Dict[str, str]:
    classes: Dict[str, str] = {}
    for line in TOOL.RESULTS.read_text(encoding="utf-8").splitlines():
        if line:
            row = json.loads(line)
            if "content_class" in row:
                classes[row["image"]] = row["content_class"]
    return classes


def powerpoint_rows() -> Dict[str, Dict[str, Any]]:
    rows = {
        row["image"]: row
        for row in (
            json.loads(line)
            for line in PPT_RESULTS.read_text(encoding="utf-8").splitlines()
            if line
        )
    }
    if len(rows) != TOOL.CORPUS_IMAGES:
        raise TOOL.LedgerError("PowerPoint results do not cover the corpus")
    return rows


def _png_size(path: Path) -> Tuple[int, int]:
    from PIL import Image

    with Image.open(path) as img:
        return img.width, img.height


def _lf_kb(path: Path) -> float:
    """Size in KB with newlines normalized to LF (checkout-independent)."""
    return len(path.read_bytes().replace(b"\r\n", b"\n")) / 1024


def emit_assets(registry: Dict[str, Any]) -> None:
    """Materialize gallery assets from local repository state (not in CI)."""
    sys.path.insert(0, str(REPO / "src"))
    import numpy as np

    from splatthis.browser_export import generate_css_splat_html
    from splatthis.pixel_runtime import generate_pixel_runtime_html
    from splatthis.storage import load_splats_json

    for sub in ("src", "svg", "css", "runtime", "pptx"):
        (GALLERY / sub).mkdir(parents=True, exist_ok=True)

    def population(run_dir: Path):
        manifest = json.loads(
            (run_dir / "run_manifest.json").read_text(encoding="utf-8")
        )
        background = np.asarray(
            manifest["config"]["background_linear_rgb"], dtype=np.float32
        )
        return load_splats_json(str(run_dir / "final.raw.json")), background

    stats: Dict[str, Dict[str, int]] = {}
    for image in corpus_images(registry):
        source = SOURCES / f"{image}.png"
        svg = SVG_LAB / f"{image}-standard.svg"
        svg_run = RUNS / f"{image}_svg_s0_art"
        runtime_run = RUNS / f"{image}_canvas_s0_art"
        pptx_run = RUNS / f"{image}_pptx_s0_art"
        deck = RUNS / f"{image}_pptx_s0.pptx"
        capture = RUNS / f"{image}_pptx_s0_powerpoint_slide.png"
        for needed in (source, svg, deck, capture) + tuple(
            run / name
            for run in (svg_run, runtime_run, pptx_run)
            for name in ("final.raw.json", "run_manifest.json")
        ):
            if not needed.exists():
                raise TOOL.LedgerError(
                    f"cannot emit gallery assets: missing {needed.relative_to(REPO)}"
                )
        width, height = _png_size(source)

        shutil.copy2(source, GALLERY / "src" / f"{image}.png")
        shutil.copy2(svg, GALLERY / "svg" / f"{image}.svg")
        shutil.copy2(deck, GALLERY / "pptx" / f"{image}.pptx")
        shutil.copy2(capture, GALLERY / "pptx" / f"{image}.png")

        svg_splats, svg_bg = population(svg_run)
        (GALLERY / "css" / f"{image}.html").write_text(
            generate_css_splat_html(
                svg_splats, width, height, background_linear_rgb=svg_bg
            ),
            encoding="utf-8",
        )
        runtime_splats, runtime_bg = population(runtime_run)
        (GALLERY / "runtime" / f"{image}.html").write_text(
            generate_pixel_runtime_html(
                runtime_splats, width, height, background_linear_rgb=runtime_bg
            ),
            encoding="utf-8",
        )
        pptx_splats, _ = population(pptx_run)
        stats[image] = {
            "svg_splats": len(svg_splats),
            "runtime_splats": len(runtime_splats),
            "pptx_splats": len(pptx_splats),
        }
        print(f"emitted {image}: src, svg, css, runtime, pptx")
    STATS.write_text(json.dumps(stats, indent=1, sort_keys=True) + "\n")
    print(f"wrote {STATS.relative_to(REPO)}")


PAGE_HEAD = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>SplatThis — corpus gallery, rendered live</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
  * { box-sizing: border-box; }
  html, body { margin: 0; padding: 0; background: #0e0e10; color: #eaeaea; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif; }
  main { max-width: 1680px; margin: 0 auto; padding: 48px 24px 80px; }
  h1 { font-size: 30px; margin: 0 0 8px; letter-spacing: -0.02em; }
  p { line-height: 1.55; color: #d5d5dd; max-width: 76ch; }
  a { color: #6fa8ff; }
  .note { color: #9c9ca5; font-size: 13px; }
  .row { display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px; margin: 24px 0; }
  .row figure { background: #1a1a1d; border-radius: 8px; padding: 9px; margin: 0; border: 1px solid #2a2a2e; }
  .row img { width: 100%; height: auto; display: block; border-radius: 4px; }
  .row figcaption { color: #9c9ca5; font-size: 11px; line-height: 1.5; margin-top: 7px; font-family: ui-monospace, monospace; }
  .row figcaption strong { color: #eaeaea; }
  h2.img { font-size: 17px; margin: 40px 0 0; font-family: ui-monospace, monospace; }
  h2.img span { color: #9c9ca5; font-weight: 400; font-size: 13px; }
  .fitframe { position: relative; overflow: hidden; border-radius: 4px; }
  .fitframe iframe { position: absolute; top: 0; left: 0; border: 0; transform-origin: 0 0; }
  @media (max-width: 1100px) { .row { grid-template-columns: repeat(2, 1fr); } }
</style>
</head>
<body>
<main>
<h1>The whole corpus, rendered live</h1>
<p>All 21 governing-corpus images side by side: the original, the scripted
pixel runtime evaluating the splat formula in your browser, the scriptless
CSS build as real DOM composited from a stylesheet, and the
corrected-exporter SVG as a true vector — every one of them the deployed
artifact itself, drawn live by your browser, because GitHub Pages applies no
sanitizer. PowerPoint is the one target that cannot render live: its column
is a real Microsoft PowerPoint slideshow capture, with the editable deck one
click away.</p>
<p class="note">Rows are ordered best-to-worst by deployed SVG LPIPS. Every
score is a seed-0 measurement of these exact artifact families against the
source in the governing renderer — Chromium for the browser targets, real
PowerPoint for the decks — quoted from the versioned ledgers. The page is
generated by <code>tools/build_corpus_gallery.py</code> and goes stale
loudly in CI.</p>
"""

PAGE_FOOT = """
<p class="note"><a href="../">← SplatThis project page</a> ·
<a href="../paper/">technical report</a></p>
</main>
<script>
  const fitFrames = () => {
    document.querySelectorAll(".fitframe").forEach((box) => {
      const frame = box.querySelector("iframe");
      if (!frame) return;
      const s = box.clientWidth / box.dataset.w;
      const ox = box.dataset.ox || 0, oy = box.dataset.oy || 0;
      frame.style.transform = `scale(${s}) translate(${-ox}px, ${-oy}px)`;
    });
  };
  addEventListener("resize", fitFrames);
  addEventListener("load", fitFrames);
  fitFrames();
</script>
</body>
</html>
"""


def build_index(registry: Dict[str, Any]) -> str:
    classes = content_classes()
    stats = json.loads(STATS.read_text(encoding="utf-8")) if STATS.is_file() else None
    if not stats:
        raise TOOL.LedgerError(
            "missing docs/corpus/stats.json (regenerate locally with --emit)"
        )
    ppt = powerpoint_rows()
    per_image = {
        fmt: {p["image"]: p for p in registry["per_image"][fmt]}
        for fmt in ("svg", "css", "pixel-runtime")
    }
    parts: List[str] = [PAGE_HEAD]
    ordered = sorted(
        corpus_images(registry),
        key=lambda image: per_image["svg"][image]["deployed_lpips"],
    )
    for image in ordered:
        assets = {
            "source": GALLERY / "src" / f"{image}.png",
            "svg": GALLERY / "svg" / f"{image}.svg",
            "css": GALLERY / "css" / f"{image}.html",
            "runtime": GALLERY / "runtime" / f"{image}.html",
            "pptx_png": GALLERY / "pptx" / f"{image}.png",
            "pptx_deck": GALLERY / "pptx" / f"{image}.pptx",
        }
        for needed in assets.values():
            if not needed.is_file():
                raise TOOL.LedgerError(
                    f"gallery asset missing: {needed.relative_to(REPO)} "
                    f"(regenerate locally with --emit)"
                )
        if image not in stats:
            raise TOOL.LedgerError(f"stats.json is missing {image}")
        width, height = _png_size(assets["source"])
        frame_w, frame_h = width + RUNTIME_PAD_W, height + RUNTIME_PAD_H
        svg_row = per_image["svg"][image]
        css_row = per_image["css"][image]
        runtime_row = per_image["pixel-runtime"][image]
        ppt_row = ppt[image]
        counts = stats[image]
        parts.append(
            f'<h2 class="img">{image} '
            f"<span>· {classes.get(image, 'unclassified')} · "
            f"{width}×{height}</span></h2>\n"
            f'<div class="row">\n'
            f"  <figure>\n"
            f'    <img src="src/{image}.png" loading="lazy" '
            f'alt="{image} source image">\n'
            f"    <figcaption><strong>Source</strong> · original bitmap · "
            f"{_lf_kb(assets['source']):,.0f} KB</figcaption>\n"
            f"  </figure>\n"
            f"  <figure>\n"
            f'    <div class="fitframe" data-w="{width}" data-h="{height}" '
            f'data-ox="{RUNTIME_PAD_W // 2}" data-oy="{RUNTIME_PAD_W // 2}" '
            f'style="aspect-ratio:{width}/{height};background:#111">\n'
            f'      <iframe src="runtime/{image}.html" width="{frame_w}" '
            f'height="{frame_h}" loading="lazy" '
            f'title="{image} scripted pixel runtime, live"></iframe>\n'
            f"    </div>\n"
            f"    <figcaption><strong>Scripted runtime</strong> · LPIPS "
            f"{runtime_row['deployed_lpips']:.4f} · SSIM "
            f"{runtime_row['deployed_ssim_srgb']:.4f} · "
            f"{counts['runtime_splats']:,} splats · "
            f"{_lf_kb(assets['runtime']):,.0f} KB · live JS/WebGL"
            f"</figcaption>\n"
            f"  </figure>\n"
            f"  <figure>\n"
            f'    <div class="fitframe" data-w="{width}" data-h="{height}" '
            f'style="aspect-ratio:{width}/{height}">\n'
            f'      <iframe src="css/{image}.html" width="{width}" '
            f'height="{height}" loading="lazy" '
            f'title="{image} as scriptless CSS splats"></iframe>\n'
            f"    </div>\n"
            f"    <figcaption><strong>Scriptless CSS</strong> · LPIPS "
            f"{css_row['deployed_lpips']:.4f} · SSIM "
            f"{css_row['deployed_ssim_srgb']:.4f} · "
            f"{counts['svg_splats']:,} splats · "
            f"{_lf_kb(assets['css']):,.0f} KB · live DOM</figcaption>\n"
            f"  </figure>\n"
            f"  <figure>\n"
            f'    <img src="svg/{image}.svg" loading="lazy" '
            f'alt="{image} as corrected-exporter SVG">\n'
            f"    <figcaption><strong>SVG</strong> · LPIPS "
            f"{svg_row['deployed_lpips']:.4f} · SSIM "
            f"{svg_row['deployed_ssim_srgb']:.4f} · "
            f"{counts['svg_splats']:,} splats · "
            f"{_lf_kb(assets['svg']):,.0f} KB · live vector</figcaption>\n"
            f"  </figure>\n"
            f"  <figure>\n"
            f'    <img src="pptx/{image}.png" loading="lazy" '
            f'alt="{image} deck rendered by real PowerPoint">\n'
            f"    <figcaption><strong>PowerPoint</strong> · LPIPS "
            f"{ppt_row['lpips']:.4f} · SSIM {ppt_row['ssim_srgb']:.4f} · "
            f"{counts['pptx_splats']:,} shapes · "
            f"{ppt_row['pptx_bytes'] / 1024:,.0f} KB · real capture · "
            f'<a href="pptx/{image}.pptx" download>deck</a></figcaption>\n'
            f"  </figure>\n"
            f"</div>"
        )
    parts.append(PAGE_FOOT)
    return "\n".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--emit",
        action="store_true",
        help="materialize gallery assets from local corpus state first",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify docs/corpus/index.html matches assets and ledgers",
    )
    args = parser.parse_args()

    try:
        registry = TOOL.declarative_expectations()
        if args.emit:
            emit_assets(registry)
        rendered = build_index(registry)
    except TOOL.LedgerError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    current = INDEX.read_text(encoding="utf-8") if INDEX.is_file() else ""
    if args.check:
        if current != rendered:
            print(
                "docs/corpus/index.html is stale: it no longer matches the "
                "assets and ledgers. Run `python tools/build_corpus_gallery.py`.",
                file=sys.stderr,
            )
            return 1
        print("docs/corpus/index.html is current.")
        return 0

    if current == rendered:
        print("docs/corpus/index.html already current.")
        return 0
    INDEX.parent.mkdir(parents=True, exist_ok=True)
    INDEX.write_text(rendered, encoding="utf-8")
    print(f"docs/corpus/index.html rendered ({len(rendered):,} bytes).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
