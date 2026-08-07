#!/usr/bin/env python3
"""Build the live corpus study and artifact appendix for GitHub Pages.

The page presents the aggregate evidence as a small artifact study, then shows
the whole 21-image governing corpus side by side as its qualitative appendix.
Each row has five columns: the original, the scripted pixel runtime rendering
live, the scriptless CSS build as live DOM, the corrected-exporter SVG as a
live vector, and the PowerPoint deck as a real slideshow capture with the
editable deck for download. Everything except PowerPoint renders live -- these
are the deployed artifacts themselves, not screenshots -- because GitHub
Pages, unlike the README, applies no sanitizer.

Two modes:

- ``--emit`` (local only) materializes the artifact assets under
  ``docs/corpus/`` from repository state that CI does not have: source
  images and PowerPoint captures/decks from ``result/corpus/``, the latest
  successful deployed SVGs from the governing results ledger, and CSS emitted
  by the current package from the stored seed-0 populations. The historical
  pixel-runtime pages remain committed deployment evidence; that removed
  runtime is not restored to the package. Splat counts are recorded in
  ``docs/corpus/stats.json`` at emit time.
- default / ``--check`` renders ``docs/corpus/index.html`` from the
  committed assets plus the cross-checked ledgers, failing closed on any
  missing artifact, missing stats, or registry mismatch.

``tests/test_corpus_gallery.py`` runs ``--check`` in CI.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO = Path(__file__).resolve().parents[1]
GALLERY = REPO / "docs" / "corpus"
INDEX = GALLERY / "index.html"
STATS = GALLERY / "stats.json"
SOURCES = REPO / "result" / "corpus" / "images"
RESULTS = REPO / "result" / "corpus" / "results.jsonl"
PPT_RESULTS = REPO / "result" / "corpus" / "powerpoint_results.jsonl"
FIDELITY = REPO / "src" / "splatthis" / "data" / "compositor-fidelity.json"
SVG_MEASUREMENTS = REPO / "result" / "svg-quality" / "measurements.json"
CORPUS_IMAGES = 21

#: Pixel-runtime pages are flex-centered with 16px padding and a status line.
RUNTIME_PAD_W = 32
RUNTIME_PAD_H = 52


class GalleryError(RuntimeError):
    """A committed gallery asset or its provenance ledger is inconsistent."""


def _load_json(path: Path) -> Any:
    if not path.is_file():
        raise GalleryError(f"missing gallery ledger: {path.relative_to(REPO)}")
    return json.loads(path.read_text(encoding="utf-8"))


def _percentile(values: List[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    fraction = position - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def _require_corpus(images: set[str], label: str) -> None:
    if len(images) != CORPUS_IMAGES:
        raise GalleryError(
            f"{label}: expected {CORPUS_IMAGES} images, found {len(images)}"
        )


def fidelity_registry() -> Dict[str, Any]:
    """Load and internally cross-check the gallery's published measurements."""

    registry = _load_json(FIDELITY)

    def pin(label: str, published: float, measured: float) -> None:
        if abs(float(published) - float(measured)) > 5e-5:
            raise GalleryError(
                f"{label}: registry has {published}, evidence gives {measured:.4f}"
            )

    for output_format in ("svg", "svg-high", "css", "pixel-runtime"):
        rows = registry["per_image"][output_format]
        _require_corpus(
            {str(row["image"]) for row in rows},
            f"fidelity registry {output_format}",
        )
        deployed = registry["formats"][output_format]["expectation"]["deployed"]
        lpips = [float(row["deployed_lpips"]) for row in rows]
        ssim = [float(row["deployed_ssim_srgb"]) for row in rows]
        pin(
            f"{output_format} deployed LPIPS median",
            deployed["lpips_median"],
            statistics.median(lpips),
        )
        pin(
            f"{output_format} deployed LPIPS p90",
            deployed["lpips_p90"],
            _percentile(lpips, 0.9),
        )
        pin(
            f"{output_format} deployed SSIM median",
            deployed["ssim_srgb_median"],
            statistics.median(ssim),
        )
        pin(
            f"{output_format} deployed SSIM p10",
            deployed["ssim_srgb_p10"],
            _percentile(ssim, 0.1),
        )

    measurements = _load_json(SVG_MEASUREMENTS)
    _require_corpus(set(measurements), "SVG measurements")
    for output_format, quality in (("svg", "standard"), ("svg-high", "high")):
        published = registry["formats"][output_format]["expectation"]["compositor"][
            "ssim_srgb_median"
        ]
        measured = statistics.median(
            float(measurements[image][quality][0]) for image in measurements
        )
        pin(f"{output_format} compositor SSIM median", published, measured)
    return registry


def corpus_images(registry: Dict[str, Any]) -> List[str]:
    return sorted(p["image"] for p in registry["per_image"]["svg"])


def content_classes() -> Dict[str, str]:
    classes: Dict[str, str] = {}
    for line in RESULTS.read_text(encoding="utf-8").splitlines():
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
    if len(rows) != CORPUS_IMAGES:
        raise GalleryError("PowerPoint results do not cover the corpus")
    return rows


def svg_rows() -> Dict[str, Dict[str, Any]]:
    """Return the newest successful deployed SVG row per corpus image."""
    rows = [
        json.loads(line)
        for line in RESULTS.read_text(encoding="utf-8").splitlines()
        if line
    ]
    selected = {
        row["image"]: row
        for row in rows
        if row.get("format") == "svg"
        and row.get("is_deployed_artifact") is True
        and row.get("returncode") == 0
    }
    if len(selected) != CORPUS_IMAGES:
        raise GalleryError(
            "deployed SVG results do not cover the corpus "
            f"({len(selected)} of {CORPUS_IMAGES})"
        )
    return selected


def _png_size(path: Path) -> Tuple[int, int]:
    from PIL import Image

    with Image.open(path) as img:
        return img.width, img.height


def _lf_kb(path: Path) -> float:
    """Size in KB with newlines normalized to LF (checkout-independent)."""
    return len(path.read_bytes().replace(b"\r\n", b"\n")) / 1024


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _gallery_asset_paths(
    source: Path,
    svg_path: Path,
    deck: Path,
    capture: Path,
    svg_run: Path,
    pptx_run: Path,
    image: str,
    generate_css_assets: bool,
) -> List[Path]:
    """Return every local input needed to emit one gallery row."""
    required_runs = [svg_run, pptx_run]
    paths = [source, svg_path, deck, capture]
    paths.extend(
        run / name
        for run in required_runs
        for name in ("final.raw.json", "run_manifest.json")
    )
    if not generate_css_assets:
        paths.append(GALLERY / "css" / f"{image}.html")
    paths.append(GALLERY / "runtime" / f"{image}.html")
    return paths


def _require_gallery_assets(paths: List[Path]) -> None:
    """Fail closed when any input needed by the gallery is absent."""
    for path in paths:
        if not path.exists():
            raise GalleryError(
                f"cannot emit gallery assets: missing {path.relative_to(REPO)}"
            )


def emit_assets(registry: Dict[str, Any]) -> None:
    """Materialize gallery assets from local repository state (not in CI).

    CSS regeneration needs the full training environment. When that environment
    is unavailable, preserve the committed CSS. The historical pixel-runtime
    evidence is always preserved because that emitter is no longer packaged.
    """
    sys.path.insert(0, str(REPO / "src"))
    try:
        import numpy as np

        from splatthis.browser_export import generate_css_splat_html
        from splatthis.io import load_splats_json
    except ModuleNotFoundError as error:
        if error.name != "torch":
            raise
        generate_css_assets = False
        print("training dependencies unavailable; preserving committed CSS assets")
    else:
        generate_css_assets = True

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

    previous_stats = (
        json.loads(STATS.read_text(encoding="utf-8")) if STATS.is_file() else {}
    )

    stats: Dict[str, Dict[str, int]] = {}
    svg = svg_rows()
    ppt = powerpoint_rows()
    for image in corpus_images(registry):
        source = SOURCES / f"{image}.png"
        svg_row = svg[image]
        svg_path = REPO / "result" / "corpus" / svg_row["output_path"]
        svg_run = REPO / "result" / "corpus" / svg_row["artifacts_path"]
        ppt_row = ppt[image]
        capture = REPO / "result" / "corpus" / ppt_row["capture"]
        deck = capture.with_name(capture.name.replace("_powerpoint_slide.png", ".pptx"))
        pptx_run = deck.with_name(f"{deck.stem}_art")
        needed_paths = _gallery_asset_paths(
            source,
            svg_path,
            deck,
            capture,
            svg_run,
            pptx_run,
            image,
            generate_css_assets,
        )
        _require_gallery_assets(needed_paths)
        width, height = _png_size(source)

        shutil.copy2(source, GALLERY / "src" / f"{image}.png")
        shutil.copy2(svg_path, GALLERY / "svg" / f"{image}.svg")
        shutil.copy2(deck, GALLERY / "pptx" / f"{image}.pptx")
        shutil.copy2(capture, GALLERY / "pptx" / f"{image}.png")

        old = previous_stats.get(image, {})
        if generate_css_assets:
            svg_splats, svg_bg = population(svg_run)
            (GALLERY / "css" / f"{image}.html").write_text(
                generate_css_splat_html(
                    svg_splats, width, height, background_linear_rgb=svg_bg
                ),
                encoding="utf-8",
            )
            css_splat_count = len(svg_splats)
        else:
            css_splat_count = int(old.get("css_splats", old["svg_splats"]))
        runtime_splat_count = int(old["runtime_splats"])
        pptx_splat_count = int(
            json.loads((pptx_run / "final.raw.json").read_text(encoding="utf-8"))[
                "num_splats"
            ]
        )
        stats[image] = {
            "svg_splats": int(svg_row["splats_final"]),
            "css_splats": css_splat_count,
            "runtime_splats": runtime_splat_count,
            "pptx_splats": pptx_splat_count,
        }
        print(
            f"emitted {image}: src, svg, pptx"
            + (", css" if generate_css_assets else "")
        )
    STATS.write_text(json.dumps(stats, indent=1, sort_keys=True) + "\n")
    print(f"wrote {STATS.relative_to(REPO)}")


PAGE_HEAD = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Portable 2D Gaussian Splats Across Document Renderers — corpus study</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
  * { box-sizing: border-box; }
  html, body { margin: 0; padding: 0; background: #0e0e10; color: #eaeaea; font-family: Charter, "Bitstream Charter", Georgia, serif; }
  main { max-width: 1680px; margin: 0 auto; padding: 48px 24px 80px; }
  .paper { max-width: 960px; margin: 0 auto 64px; }
  .eyebrow { color: #9c9ca5; font: 12px/1.4 ui-monospace, monospace; letter-spacing: 0.08em; text-transform: uppercase; }
  h1 { font-size: clamp(34px, 5vw, 58px); line-height: 1.04; margin: 12px 0 12px; letter-spacing: -0.035em; }
  h2 { font-size: 24px; line-height: 1.2; margin: 42px 0 12px; letter-spacing: -0.015em; }
  .byline { color: #b9b9c2; margin: 0 0 36px; }
  p, li { line-height: 1.62; color: #d5d5dd; max-width: 78ch; }
  a { color: #6fa8ff; }
  .note { color: #9c9ca5; font-size: 13px; }
  .abstract { border: 1px solid #323238; border-width: 1px 0; padding: 5px 0 18px; }
  .abstract h2 { font-size: 18px; margin-top: 18px; }
  .study-nav { display: flex; flex-wrap: wrap; gap: 8px 18px; margin: 20px 0 0; font: 12px/1.5 ui-monospace, monospace; }
  .study-nav a { color: #aaaab4; text-decoration: none; }
  .table-wrap { overflow-x: auto; margin: 18px 0 8px; }
  table { width: 100%; border-collapse: collapse; font: 13px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif; }
  th, td { padding: 9px 10px; border-bottom: 1px solid #303036; text-align: right; vertical-align: top; white-space: nowrap; }
  th { color: #b9b9c2; font-weight: 600; }
  th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) { text-align: left; }
  .findings { padding-left: 22px; }
  .findings li { margin: 8px 0; }
  .appendix-intro { max-width: 960px; margin: 0 auto 28px; }
  .row { display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px; margin: 24px 0; }
  .row figure { background: #1a1a1d; border-radius: 8px; padding: 9px; margin: 0; border: 1px solid #2a2a2e; }
  .row img { width: 100%; height: auto; display: block; border-radius: 4px; }
  .row figcaption { color: #9c9ca5; font-size: 11px; line-height: 1.5; margin-top: 7px; font-family: ui-monospace, monospace; }
  .row figcaption strong { color: #eaeaea; }
  h3.img { font-size: 17px; margin: 40px 0 0; font-family: ui-monospace, monospace; }
  h3.img span { color: #9c9ca5; font-weight: 400; font-size: 13px; }
  .fitframe { position: relative; overflow: hidden; border-radius: 4px; }
  .fitframe iframe { position: absolute; top: 0; left: 0; border: 0; transform-origin: 0 0; }
  @media (max-width: 1100px) { .row { grid-template-columns: repeat(2, 1fr); } }
</style>
</head>
<body>
<main>
"""

PAGE_FOOT = """
</section>
<p class="note"><a href="https://github.com/BramAlkema/SplatThis">← SplatThis repository</a></p>
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


def _study_intro(
    registry: Dict[str, Any],
    ppt: Dict[str, Dict[str, Any]],
    svg: Dict[str, Dict[str, Any]],
    stats: Dict[str, Dict[str, int]],
    classes: Dict[str, str],
) -> str:
    """Build the paper-like study framing from the evidence shown below it."""

    def summary(
        rows: List[Dict[str, Any]], lpips_key: str, ssim_key: str
    ) -> Dict[str, float]:
        lpips = [float(row[lpips_key]) for row in rows]
        ssim = [float(row[ssim_key]) for row in rows]
        return {
            "lpips_median": statistics.median(lpips),
            "lpips_p90": _percentile(lpips, 0.9),
            "ssim_median": statistics.median(ssim),
        }

    runtime = summary(
        registry["per_image"]["pixel-runtime"],
        "deployed_lpips",
        "deployed_ssim_srgb",
    )
    css = summary(
        registry["per_image"]["css"],
        "deployed_lpips",
        "deployed_ssim_srgb",
    )
    svg_summary = summary(list(svg.values()), "lpips", "ssim_srgb")
    ppt_summary = summary(list(ppt.values()), "lpips", "ssim_srgb")
    summaries = (
        (
            "Historical pixel runtime",
            "Chromium / WebGL",
            runtime,
            statistics.median(row["runtime_splats"] for row in stats.values()),
            "splats",
        ),
        (
            "Scriptless CSS",
            "Chromium / CSS",
            css,
            statistics.median(row["css_splats"] for row in stats.values()),
            "splats",
        ),
        (
            "SVG",
            "Chromium / SVG",
            svg_summary,
            statistics.median(row["svg_splats"] for row in stats.values()),
            "splats",
        ),
        (
            "PowerPoint",
            "Microsoft PowerPoint",
            ppt_summary,
            statistics.median(row["pptx_splats"] for row in stats.values()),
            "shapes",
        ),
    )
    result_rows = "\n".join(
        "<tr>"
        f"<td>{name}</td><td>{renderer}</td>"
        f"<td>{values['lpips_median']:.4f}</td>"
        f"<td>{values['lpips_p90']:.4f}</td>"
        f"<td>{values['ssim_median']:.4f}</td>"
        f"<td>{complexity:,.0f} {unit}</td>"
        "</tr>"
        for name, renderer, values, complexity, unit in summaries
    )
    browser_medians = [
        runtime["lpips_median"],
        css["lpips_median"],
        svg_summary["lpips_median"],
    ]
    easiest = min(svg.values(), key=lambda row: float(row["lpips"]))
    hardest = max(svg.values(), key=lambda row: float(row["lpips"]))
    class_count = len({classes[image] for image in svg})

    return f"""
<article class="paper">
<div class="eyebrow">Technical report · August 2026 · artifact study</div>
<h1>Portable 2D Gaussian Splats Across Document Renderers</h1>
<p class="byline">Bram Alkema · SplatThis · 21-image governing corpus</p>

<section class="abstract" id="abstract">
<h2>Abstract</h2>
<p>We study how fitted anisotropic 2D Gaussian splats survive deployment into
portable document artifacts rather than judging only the optimizer's internal
render. The corpus contains 21 images across {class_count} content classes at
a maximum edge of 384 pixels. Browser artifacts are evaluated in Chromium;
editable decks are captured from Microsoft PowerPoint. LPIPS is the primary
metric. In the evidence snapshot presented here, median deployed LPIPS is
{svg_summary["lpips_median"]:.4f} for the newest SVG artifacts,
{css["lpips_median"]:.4f} for the versioned scriptless-CSS comparator,
{runtime["lpips_median"]:.4f} for the historical pixel runtime, and
{ppt_summary["lpips_median"]:.4f} for PowerPoint. The narrow browser-format
median range suggests that fitting error, not browser emitter choice, dominates
typical deployed fidelity. Per-image tails remain wide, so the aggregate is
not evidence that splats suit every kind of image.</p>
</section>

<nav class="study-nav" aria-label="Study sections">
  <a href="#question">1. Question</a>
  <a href="#method">2. Method</a>
  <a href="#results">3. Results</a>
  <a href="#limitations">4. Limitations</a>
  <a href="#artifact-appendix">5. Artifact appendix</a>
</nav>

<section id="question">
<h2>1. Research question</h2>
<p>How much image fidelity remains when fitted Gaussian populations are
delivered through document renderers that the optimizer does not control? The
study focuses on deployment fidelity and inspectable native artifacts. It is
not a compression benchmark: editability, script policy, and document-format
compatibility are the reasons to choose these outputs.</p>
</section>

<section id="method">
<h2>2. Experimental design</h2>
<p><strong>Corpus.</strong> The governing set comprises 21 source images in
{class_count} labelled content classes, normalized to at most 384 pixels on
the longest edge. Each row below is one image observed across the available
artifact families.</p>
<p><strong>Protocol.</strong> Scores are seed-0 measurements against the source
at native source resolution. Browser targets are rasterized in Chromium and
PowerPoint decks in the native Microsoft PowerPoint slideshow renderer. The
SVG column selects the newest successful deployed ledger row for each image;
PowerPoint selects the versioned real-renderer capture. CSS and the removed
pixel runtime remain versioned comparator snapshots.</p>
<p><strong>Metrics.</strong> LPIPS is primary and lower is better. SSIM is shown
for continuity and higher is better, but it is not used to rank conclusions:
on this corpus its preference for smooth output can reward blur. Median reports
the typical image; p90 LPIPS exposes the difficult tail.</p>
<p><strong>Artifact audit.</strong> The generator checks corpus coverage,
cross-checks registry aggregates against all per-image measurements, verifies
the selected SVG and PPTX hashes, and fails when committed evidence or the
generated page goes stale.</p>
</section>

<section id="results">
<h2>3. Results</h2>
<div class="table-wrap">
<table>
  <thead><tr><th>Artifact family</th><th>Governing renderer</th><th>Median LPIPS ↓</th><th>p90 LPIPS ↓</th><th>Median SSIM ↑</th><th>Median complexity</th></tr></thead>
  <tbody>
{result_rows}
  </tbody>
</table>
</div>
<ol class="findings">
  <li>The three browser-delivered families have median LPIPS between
  {min(browser_medians):.4f} and {max(browser_medians):.4f}. At the level of
  this single-seed snapshot, typical fidelity is substantially more stable
  across browser emitters than across images.</li>
  <li>The difficult-image tail is material: p90 LPIPS ranges from
  {svg_summary["lpips_p90"]:.4f} for the newest SVG snapshot to
  {ppt_summary["lpips_p90"]:.4f} for PowerPoint. Median-only reporting would
  conceal that applicability boundary.</li>
  <li>In the current SVG evidence, <code>{easiest["image"]}</code> is best
  (LPIPS {float(easiest["lpips"]):.4f}) and
  <code>{hardest["image"]}</code> is worst
  (LPIPS {float(hardest["lpips"]):.4f}). The qualitative appendix makes the
  corresponding content differences inspectable.</li>
</ol>
</section>

<section id="limitations">
<h2>4. Limitations and validity</h2>
<p>This is a small, deliberately heterogeneous artifact study, not a claim of
population-level performance. It uses one seed per published artifact, has no
confidence intervals or hypothesis tests, and is limited to images no larger
than 384 pixels. The four columns are not one synchronized experimental run:
SVG and PowerPoint are the newest verified deployed rows, while CSS and the
historical runtime are retained comparator measurements. Consequently the
table supports an evidence-snapshot comparison, not a causal format ranking.
The PowerPoint ledger identifies the real capture backend but does not record
the application and operating-system versions for these captures. Finally,
LPIPS and SSIM are proxies for visual similarity; neither measures editability,
accessibility, browser cost, or authoring usefulness.</p>
<p class="note"><strong>Reproduce the audit:</strong>
<code>./venv/bin/python tools/build_corpus_gallery.py --check</code>. Local
evidence holders can use <code>--emit</code> first to rematerialize the gallery.
Rows below are ordered best-to-worst by the displayed SVG LPIPS.</p>
</section>
</article>

<section id="artifact-appendix">
<div class="appendix-intro">
<h2>5. Qualitative artifact appendix</h2>
<p>Each row shows the source and four deployed output families. Browser targets
render live on this page. PowerPoint is represented by its native slideshow
capture; the editable deck is linked. The historical pixel runtime is retained
as evidence but is no longer part of the installable package.</p>
</div>
"""


def build_index(registry: Dict[str, Any]) -> str:
    classes = content_classes()
    stats = json.loads(STATS.read_text(encoding="utf-8")) if STATS.is_file() else None
    if not stats:
        raise GalleryError(
            "missing docs/corpus/stats.json (regenerate locally with --emit)"
        )
    ppt = powerpoint_rows()
    svg_rows_by_image = svg_rows()
    per_image = {
        fmt: {p["image"]: p for p in registry["per_image"][fmt]}
        for fmt in ("svg", "css", "pixel-runtime")
    }
    ordered = sorted(
        corpus_images(registry),
        key=lambda image: svg_rows_by_image[image]["lpips"],
    )
    parts: List[str] = [
        PAGE_HEAD,
        _study_intro(registry, ppt, svg_rows_by_image, stats, classes),
    ]
    for image in ordered:
        svg_row = svg_rows_by_image[image]
        ppt_row = ppt[image]
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
                raise GalleryError(
                    f"gallery asset missing: {needed.relative_to(REPO)} "
                    f"(regenerate locally with --emit)"
                )
        if _sha256(assets["svg"]) != svg_row["artifact_sha256"]:
            raise GalleryError(
                f"gallery SVG is not the ledger artifact for {image} "
                "(regenerate locally with --emit)"
            )
        if _sha256(assets["pptx_deck"]) != ppt_row["pptx_sha256"]:
            raise GalleryError(
                f"gallery PPTX is not the ledger artifact for {image} "
                "(regenerate locally with --emit)"
            )
        if _sha256(assets["pptx_png"]) != ppt_row["capture_sha256"]:
            raise GalleryError(
                f"gallery PowerPoint capture is not the ledger artifact for {image} "
                "(regenerate locally with --emit)"
            )
        if image not in stats:
            raise GalleryError(f"stats.json is missing {image}")
        width, height = _png_size(assets["source"])
        frame_w, frame_h = width + RUNTIME_PAD_W, height + RUNTIME_PAD_H
        css_row = per_image["css"][image]
        runtime_row = per_image["pixel-runtime"][image]
        counts = stats[image]
        parts.append(
            f'<h3 class="img">{image} '
            f"<span>· {classes.get(image, 'unclassified')} · "
            f"{width}×{height}</span></h3>\n"
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
            f"    <figcaption><strong>Historical scripted runtime</strong> · LPIPS "
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
            f"{counts['css_splats']:,} splats · "
            f"{_lf_kb(assets['css']):,.0f} KB · live DOM</figcaption>\n"
            f"  </figure>\n"
            f"  <figure>\n"
            f'    <img src="svg/{image}.svg" loading="lazy" '
            f'alt="{image} as corrected-exporter SVG">\n'
            f"    <figcaption><strong>SVG</strong> · LPIPS "
            f"{svg_row['lpips']:.4f} · SSIM "
            f"{svg_row['ssim_srgb']:.4f} · "
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
        registry = fidelity_registry()
        if args.emit:
            emit_assets(registry)
        rendered = build_index(registry)
    except GalleryError as error:
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
