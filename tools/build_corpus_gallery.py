#!/usr/bin/env python3
"""Build the live corpus gallery for GitHub Pages.

The gallery shows the whole 21-image governing corpus side by side, one row
per image: the source, the corrected-exporter SVG, and the scriptless CSS
build. Everything except PowerPoint renders live -- these are the deployed
artifacts themselves, not captures -- because GitHub Pages, unlike the
README, applies no sanitizer.

Two modes:

- ``--emit`` (local only) materializes the artifact assets under
  ``docs/corpus/`` from repository state that CI does not have: source
  images from ``result/corpus/images/``, the current-emitter SVGs from
  ``result/svg-quality/``, and CSS builds emitted by the shipped
  ``generate_css_splat_html`` from the stored seed-0 populations in
  ``result/corpus/runs/`` (the same populations the fidelity registry
  measured, so the quoted numbers describe these exact artifacts).
- default / ``--check`` renders ``docs/corpus/index.html`` from the
  committed assets plus the cross-checked ledgers, failing closed on any
  missing artifact or registry mismatch.

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
SOURCES = REPO / "result" / "corpus" / "images"
SVG_LAB = REPO / "result" / "svg-quality"
RUNS = REPO / "result" / "corpus" / "runs"


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


def _png_size(path: Path) -> Tuple[int, int]:
    from PIL import Image

    with Image.open(path) as img:
        return img.width, img.height


def emit_assets(registry: Dict[str, Any]) -> None:
    """Materialize gallery assets from local repository state (not in CI)."""
    sys.path.insert(0, str(REPO / "src"))
    import numpy as np

    from splatthis.browser_export import generate_css_splat_html
    from splatthis.storage import load_splats_json

    for sub in ("src", "svg", "css"):
        (GALLERY / sub).mkdir(parents=True, exist_ok=True)

    for image in corpus_images(registry):
        source = SOURCES / f"{image}.png"
        svg = SVG_LAB / f"{image}-standard.svg"
        run = RUNS / f"{image}_svg_s0_art"
        for needed in (source, svg, run / "final.raw.json", run / "run_manifest.json"):
            if not needed.exists():
                raise TOOL.LedgerError(
                    f"cannot emit gallery assets: missing {needed.relative_to(REPO)}"
                )
        shutil.copy2(source, GALLERY / "src" / f"{image}.png")
        shutil.copy2(svg, GALLERY / "svg" / f"{image}.svg")

        manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
        background = np.asarray(
            manifest["config"]["background_linear_rgb"], dtype=np.float32
        )
        splats = load_splats_json(str(run / "final.raw.json"))
        width, height = _png_size(source)
        html = generate_css_splat_html(
            splats, width, height, background_linear_rgb=background
        )
        (GALLERY / "css" / f"{image}.html").write_text(html, encoding="utf-8")
        print(f"emitted {image}: src, svg, css ({len(splats)} splats)")


PAGE_HEAD = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>SplatThis — corpus gallery, rendered live</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
  * { box-sizing: border-box; }
  html, body { margin: 0; padding: 0; background: #0e0e10; color: #eaeaea; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif; }
  main { max-width: 1180px; margin: 0 auto; padding: 48px 24px 80px; }
  h1 { font-size: 30px; margin: 0 0 8px; letter-spacing: -0.02em; }
  p { line-height: 1.55; color: #d5d5dd; max-width: 72ch; }
  a { color: #6fa8ff; }
  .note { color: #9c9ca5; font-size: 13px; }
  .row { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin: 26px 0; }
  .row figure { background: #1a1a1d; border-radius: 8px; padding: 10px; margin: 0; border: 1px solid #2a2a2e; }
  .row img { width: 100%; height: auto; display: block; border-radius: 4px; }
  .row figcaption { color: #9c9ca5; font-size: 12px; margin-top: 8px; font-family: ui-monospace, monospace; }
  .row figcaption strong { color: #eaeaea; }
  h2.img { font-size: 17px; margin: 40px 0 0; font-family: ui-monospace, monospace; }
  h2.img span { color: #9c9ca5; font-weight: 400; font-size: 13px; }
  .fitframe { position: relative; overflow: hidden; border-radius: 4px; }
  .fitframe iframe { position: absolute; top: 0; left: 0; border: 0; transform-origin: 0 0; }
</style>
</head>
<body>
<main>
<h1>The whole corpus, rendered live</h1>
<p>All 21 governing-corpus images side by side: the source, the
corrected-exporter SVG as a real vector your browser is drawing, and the
scriptless CSS build as real DOM elements composited from a stylesheet.
Nothing here is a screenshot — GitHub Pages applies no sanitizer, so the
deployed artifacts themselves render. PowerPoint is the one target that
cannot appear live; its example lives on the
<a href="../">project page</a> as a real slideshow capture.</p>
<p class="note">Rows are ordered best-to-worst by deployed SVG LPIPS. Scores
are seed-0 measurements of these exact artifacts against the source in
governing Chromium, quoted from the versioned ledgers; the page is generated
by <code>tools/build_corpus_gallery.py</code> and goes stale loudly in CI.</p>
"""

PAGE_FOOT = """
<p class="note"><a href="../">← SplatThis project page</a> ·
<a href="../paper/">technical report</a></p>
</main>
<script>
  const fitFrames = () => {
    document.querySelectorAll(".fitframe").forEach((box) => {
      const frame = box.querySelector("iframe");
      if (frame) frame.style.transform = `scale(${box.clientWidth / box.dataset.w})`;
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
    per_image = {
        fmt: {p["image"]: p for p in registry["per_image"][fmt]}
        for fmt in ("svg", "css")
    }
    parts: List[str] = [PAGE_HEAD]
    ordered = sorted(
        corpus_images(registry),
        key=lambda image: per_image["svg"][image]["deployed_lpips"],
    )
    for image in ordered:
        source = GALLERY / "src" / f"{image}.png"
        svg = GALLERY / "svg" / f"{image}.svg"
        css = GALLERY / "css" / f"{image}.html"
        for needed in (source, svg, css):
            if not needed.is_file():
                raise TOOL.LedgerError(
                    f"gallery asset missing: {needed.relative_to(REPO)} "
                    f"(regenerate locally with --emit)"
                )
        width, height = _png_size(source)
        svg_row = per_image["svg"][image]
        css_row = per_image["css"][image]
        svg_kb = svg.stat().st_size / 1024
        css_kb = css.stat().st_size / 1024
        parts.append(
            f'<h2 class="img">{image} '
            f"<span>· {classes.get(image, 'unclassified')} · "
            f"{width}×{height}</span></h2>\n"
            f'<div class="row">\n'
            f"  <figure>\n"
            f'    <img src="src/{image}.png" loading="lazy" '
            f'alt="{image} source image">\n'
            f"    <figcaption><strong>Source</strong></figcaption>\n"
            f"  </figure>\n"
            f"  <figure>\n"
            f'    <img src="svg/{image}.svg" loading="lazy" '
            f'alt="{image} as corrected-exporter SVG">\n'
            f"    <figcaption><strong>SVG</strong> · LPIPS "
            f"{svg_row['deployed_lpips']:.4f} · SSIM "
            f"{svg_row['deployed_ssim_srgb']:.4f} · {svg_kb:,.0f} KB · "
            f"live vector</figcaption>\n"
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
            f"{css_row['deployed_ssim_srgb']:.4f} · {css_kb:,.0f} KB · "
            f"live DOM</figcaption>\n"
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
