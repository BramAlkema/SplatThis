#!/usr/bin/env python3
"""Reference-corpus benchmark (ADR-003 Phase 0).

Materializes a fixed corpus of standard, redistributable test images, runs the
conversion pipeline over it per export format, and scores every run on the
**deployed artifact** — the emitted SVG captured in Chromium, never the
internal renderer.

Everything is content-addressed and resumable: each run writes one JSONL
record keyed by (image, format, seed, config), and re-invocations skip records
that already exist. Source images come from ``skimage.data`` so the corpus is
reproducible from a fresh checkout with no network and no licensing questions.

Usage
-----
    python tools/corpus_benchmark.py --materialize          # write corpus/
    python tools/corpus_benchmark.py --run --formats svg    # score SVG
    python tools/corpus_benchmark.py --run --formats svg --optimizer-backend torch --jobs 2
    python tools/corpus_benchmark.py --run --formats svg,pptx --seeds 0,1,2
    python tools/corpus_benchmark.py --summarize            # tables
    python tools/corpus_benchmark.py --html                 # standalone gallery
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
import mimetypes
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from splatthis.adaptive_compute import (  # noqa: E402
    DEFAULT_CHROME_PSNR_SAFETY_MARGIN,
    DEFAULT_CHROME_SSIM_SAFETY_MARGIN,
)
from splatthis.export_common import SRGB_TRAINING_TARGETS  # noqa: E402

DEFAULT_ROOT = REPO / "result" / "corpus"
MAX_EDGE = 384  # keeps a 20-image x 2-format sweep tractable; recorded in meta
PIXEL_RUNTIME_CAPTURE_KINDS = {"canvas-pixel-buffer", "pixel-runtime-buffer"}
DEPLOYED_CANVAS_CAPTURE_KINDS = {
    *PIXEL_RUNTIME_CAPTURE_KINDS,
    "canvas-api-pixel-buffer",
}


@dataclass(frozen=True)
class CorpusImage:
    name: str
    loader: str
    content_class: str
    note: str


# Content classes follow ADR-003 Phase 0: portrait, fur, landscape, graphic,
# transparency, smooth gradients, tiny hard edges, text-like detail.
CORPUS: List[CorpusImage] = [
    CorpusImage(
        "astronaut", "astronaut", "portrait", "skin tones, fabric, flat backdrop"
    ),
    CorpusImage("chelsea", "chelsea", "fur", "dense high-frequency animal fur"),
    CorpusImage("moon", "moon", "smooth-gradient", "low-contrast smooth grayscale"),
    CorpusImage("coffee", "coffee", "natural", "specular highlights, curved edges"),
    CorpusImage(
        "rocket", "rocket", "landscape", "sky gradient plus hard machine edges"
    ),
    CorpusImage(
        "hubble_deep_field",
        "hubble_deep_field",
        "dark-sparse",
        "near-black field, tiny point sources",
    ),
    CorpusImage(
        "immunohistochemistry",
        "immunohistochemistry",
        "texture",
        "stained biological texture",
    ),
    CorpusImage(
        "retina", "retina", "smooth-gradient", "smooth vignette with fine vessels"
    ),
    CorpusImage("colorwheel", "colorwheel", "graphic", "saturated flat colour regions"),
    CorpusImage("logo", "logo", "transparency", "RGBA flat graphic with alpha"),
    CorpusImage(
        "checkerboard", "checkerboard", "hard-edges", "worst case for soft splats"
    ),
    CorpusImage("page", "page", "text-like", "printed text, thin strokes"),
    CorpusImage("text", "text", "text-like", "handwriting-scale strokes"),
    CorpusImage("brick", "brick", "texture", "regular structured texture"),
    CorpusImage("gravel", "gravel", "texture", "stochastic texture"),
    CorpusImage("grass", "grass", "texture", "fine directional texture"),
    CorpusImage("camera", "camera", "grayscale", "classic grayscale portrait"),
    CorpusImage("coins", "coins", "grayscale", "objects on flat ground"),
    CorpusImage("cell", "cell", "grayscale", "low-contrast microscopy"),
    CorpusImage(
        "stereo_motorcycle", "stereo_motorcycle", "natural", "cluttered natural scene"
    ),
    CorpusImage(
        "chameleon", "__local__", "reference", "the project's standing test image"
    ),
]


def _to_rgb_uint8(arr: np.ndarray) -> np.ndarray:
    from skimage.color import gray2rgb, rgba2rgb
    from skimage.util import img_as_ubyte

    if isinstance(arr, tuple):
        arr = arr[0]
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = gray2rgb(arr)
    elif arr.ndim == 3 and arr.shape[-1] == 4:
        arr = img_as_ubyte(rgba2rgb(arr))  # composite alpha over white
    return img_as_ubyte(arr)


def materialize(root: Path) -> Dict[str, dict]:
    """Write the corpus to disk at a normalized max edge, with hashes."""
    from PIL import Image
    from skimage import data

    images_dir = root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    meta: Dict[str, dict] = {}

    for item in CORPUS:
        if item.loader == "__local__":
            src = Image.open(REPO / "docs" / "demo" / "source.png").convert("RGB")
            arr = np.asarray(src)
        else:
            arr = _to_rgb_uint8(getattr(data, item.loader)())

        im = Image.fromarray(arr)
        scale = MAX_EDGE / max(im.size)
        if scale < 1.0:
            im = im.resize(
                (max(1, round(im.width * scale)), max(1, round(im.height * scale))),
                Image.Resampling.LANCZOS,
            )
        path = images_dir / f"{item.name}.png"
        im.save(path, optimize=True)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        meta[item.name] = {
            **asdict(item),
            "path": str(path.relative_to(root)),
            "size": list(im.size),
            "sha256_16": digest,
            "bytes": path.stat().st_size,
        }
        print(f"  {item.name:<22} {str(im.size):<12} {item.content_class:<16} {digest}")

    # A corpus with duplicate entries silently halves its own statistical
    # power; skimage aliases some samples (cat == chelsea), so assert.
    by_hash: Dict[str, List[str]] = {}
    for name, entry in meta.items():
        by_hash.setdefault(entry["sha256_16"], []).append(name)
    dupes = {h: names for h, names in by_hash.items() if len(names) > 1}
    if dupes:
        raise SystemExit(f"duplicate images in corpus: {dupes}")

    (root / "corpus.json").write_text(
        json.dumps({"max_edge": MAX_EDGE, "images": meta}, indent=2)
    )
    return meta


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

_LPIPS = None


def _lpips_score(a_srgb: np.ndarray, b_srgb: np.ndarray) -> float:
    global _LPIPS
    import lpips
    import torch

    if _LPIPS is None:
        _LPIPS = lpips.LPIPS(net="alex", verbose=False)

    def prep(x):
        t = torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1)[None]
        return t.float() * 2.0 - 1.0

    with torch.no_grad():
        return float(_LPIPS(prep(a_srgb), prep(b_srgb)).item())


# Bumped whenever a scorer starts emitting a field that downstream analysis
# relies on. The runner is content-addressed and resumable, so without this a
# row scored under an older schema is never re-scored and silently persists
# beside newer ones.
#
# Version 2 is the first to guarantee `render_kind` and `is_deployed_artifact`
# on every row. Before it, `results.jsonl` mixed four renderers -- rsvg-convert,
# proxy-srgb, canvas-linear and Chrome -- with a free-text `renderer` string as
# the only discriminator, so an aggregate over rows silently mixed deployed
# evidence with proxies. That is not academic: a content-vs-fidelity
# correlation computed across those rows came out at r=+0.456, against +0.863
# on governing Chromium data alone.
RESULT_SCHEMA_VERSION = 2


def score_svg(source_png: Path, svg_path: Path) -> Optional[dict]:
    """Metrics on the emitted SVG captured in governing Chromium."""
    from splatthis.browser_capture import render_svg_in_browser_to_linear_rgb
    from splatthis.io import compute_quality_metrics, linear_to_srgb, load_png

    target_lin = load_png(str(source_png))[..., :3]
    h, w = target_lin.shape[:2]
    try:
        rendered_lin, method = render_svg_in_browser_to_linear_rgb(str(svg_path), w, h)
    except RuntimeError:
        return None
    m = compute_quality_metrics(target_lin, rendered_lin)
    lp = _lpips_score(linear_to_srgb(target_lin), linear_to_srgb(rendered_lin))
    return {
        "renderer": method,
        "render_kind": "svg-rasterization",
        "is_deployed_artifact": True,
        "lpips": round(lp, 4),
        "ssim_srgb": round(float(m["ssim_srgb"]), 4),
        "psnr_srgb": round(float(m["psnr_srgb"]), 3),
    }


def score_pptx_proxy(source_png: Path, splats_json: Path) -> Optional[dict]:
    """Score an internal splat proxy; never label it as a PPTX render."""
    from splatthis.io import (
        compute_quality_metrics,
        linear_to_srgb,
        load_png,
        load_splats_json,
    )
    from splatthis.renderer import render_splats_numpy

    if not splats_json.exists():
        return None
    target_lin = load_png(str(source_png))[..., :3]
    h, w = target_lin.shape[:2]
    splats = load_splats_json(str(splats_json))
    rendered = render_splats_numpy(splats, width=w, height=h, compositing_space="srgb")
    m = compute_quality_metrics(target_lin, rendered)
    lp = _lpips_score(linear_to_srgb(target_lin), linear_to_srgb(rendered))
    return {
        "renderer": "internal-splat-renderer",
        "render_kind": "pptx-proxy",
        "is_deployed_artifact": False,
        "lpips": round(lp, 4),
        "ssim_srgb": round(float(m["ssim_srgb"]), 4),
        "psnr_srgb": round(float(m["psnr_srgb"]), 3),
    }


def score_canvas(
    source_png: Path, splats_json: Path, manifest_path: Path
) -> Optional[dict]:
    """Score the byte-exact ImageData pixel-runtime model."""
    from splatthis.io import (
        compute_quality_metrics,
        linear_to_srgb,
        load_png,
        load_splats_json,
    )
    from splatthis.renderer import render_pixel_runtime_numpy

    if not splats_json.exists() or not manifest_path.exists():
        return None
    target_lin = load_png(str(source_png))[..., :3]
    h, w = target_lin.shape[:2]
    splats = load_splats_json(str(splats_json))
    manifest = json.loads(manifest_path.read_text())
    config = manifest.get("config", {})
    background = config.get("background_linear_rgb", [0.0, 0.0, 0.0])
    training_target = str(config.get("training_export_target", "canvas"))
    compositing_space = str(config.get("compositing_space", "linear"))
    if training_target in {"svg", "pptx-softedge"}:
        compositing_space = "srgb"
    rendered = render_pixel_runtime_numpy(
        splats,
        width=w,
        height=h,
        background_linear_rgb=np.asarray(background, dtype=np.float32),
        compositing_space=compositing_space,
    )
    metrics = compute_quality_metrics(target_lin, rendered)
    lpips_score = _lpips_score(linear_to_srgb(target_lin), linear_to_srgb(rendered))
    return {
        "renderer": "canvas-image-data-byte-v1",
        "render_kind": "pixel-runtime-model",
        "is_deployed_artifact": False,
        "lpips": round(lpips_score, 4),
        "ssim_srgb": round(float(metrics["ssim_srgb"]), 4),
        "psnr_srgb": round(float(metrics["psnr_srgb"]), 3),
    }


def _grid_rois(
    height: int, width: int, tile: int = 64
) -> list[tuple[int, int, int, int]]:
    return [
        (y, x, min(y + tile, height), min(x + tile, width))
        for y in range(0, height, tile)
        for x in range(0, width, tile)
    ]


def score_canvas_capture(
    source_png: Path,
    capture_png: Path,
    *,
    splat_count: int,
    artifact_bytes: int,
    compositor: str = "pixel-runtime",
) -> Optional[dict]:
    """Score the browser's exact canvas pixel buffer, not a Python proxy."""

    from splatthis.fidelity.metrics import compute_fidelity_metrics
    from splatthis.io import load_png

    if not capture_png.exists():
        return None
    target = load_png(str(source_png))[..., :3]
    rendered = load_png(str(capture_png))[..., :3]
    if target.shape != rendered.shape:
        raise ValueError(
            f"canvas capture shape mismatch: target={target.shape} "
            f"capture={rendered.shape}"
        )
    height, width = target.shape[:2]
    metrics = compute_fidelity_metrics(
        target,
        rendered,
        fixed_rois=_grid_rois(height, width),
        splat_count=splat_count,
        file_size_bytes=artifact_bytes,
        render_method="Google Chrome canvas.toDataURL",
    ).as_dict()
    for key, value in list(metrics.items()):
        if isinstance(value, float) and not np.isfinite(value):
            metrics[key] = None
    render_kind = (
        "canvas-api-pixel-buffer"
        if compositor == "canvas-api-splats"
        else "pixel-runtime-buffer"
    )
    return {
        **metrics,
        "renderer": "Google Chrome canvas.toDataURL",
        "render_kind": render_kind,
        "compositor": compositor,
        "is_deployed_artifact": True,
    }


def capture_canvas_artifact(
    html_path: Path,
    output_path: Path,
    *,
    capture_python: Path,
    browser_executable: Path,
) -> tuple[Optional[dict[str, Any]], str]:
    """Capture one canvas through the Playwright environment supplied by caller."""

    command = [
        str(capture_python),
        str(REPO / "tools" / "capture_canvas_html.py"),
        str(html_path),
        str(output_path),
        "--browser-executable",
        str(browser_executable),
    ]
    result = subprocess.run(
        command,
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=150,
    )
    if result.returncode or not output_path.exists():
        return None, result.stdout.strip()
    try:
        return json.loads(result.stdout.strip().splitlines()[-1]), result.stdout.strip()
    except (IndexError, json.JSONDecodeError):
        return None, result.stdout.strip()


def score_powerpoint_captures(root: Path) -> None:
    """Score slide crops captured from real Microsoft PowerPoint."""
    from splatthis.io import compute_quality_metrics, linear_to_srgb, load_png

    meta = json.loads((root / "corpus.json").read_text())["images"]
    selected_runs = _latest_seed_zero_runs(root)
    records = []
    for name, entry in meta.items():
        source_path = root / entry["path"]
        run_record = selected_runs.get((name, "pptx"), {})
        recorded_output = run_record.get("output_path")
        pptx_path = (
            root / recorded_output
            if recorded_output
            else root / "runs" / f"{name}_pptx_s0.pptx"
        )
        capture_path = pptx_path.with_name(f"{pptx_path.stem}_powerpoint_slide.png")
        if not capture_path.exists():
            print(f"warning: no PowerPoint slide capture for {name}")
            continue
        target = load_png(str(source_path))[..., :3]
        rendered = load_png(str(capture_path))[..., :3]
        if rendered.shape != target.shape:
            print(
                f"warning: capture size mismatch for {name}: "
                f"{rendered.shape[:2]} != {target.shape[:2]}"
            )
            continue
        metrics = compute_quality_metrics(target, rendered)
        lpips_score = _lpips_score(linear_to_srgb(target), linear_to_srgb(rendered))
        metadata_path = capture_path.with_suffix(".json")
        capture_metadata = {}
        if metadata_path.exists():
            try:
                capture_metadata = json.loads(metadata_path.read_text())
            except json.JSONDecodeError:
                print(f"warning: invalid capture metadata: {metadata_path}")
        record = {
            "key": f"{name}|pptx|powerpoint",
            "image": name,
            "content_class": entry["content_class"],
            "renderer": "Microsoft PowerPoint slideshow",
            "render_kind": "powerpoint-capture",
            "is_deployed_artifact": True,
            "capture": str(capture_path.relative_to(root)),
            "pptx_bytes": pptx_path.stat().st_size if pptx_path.exists() else None,
            "pptx_sha256": (
                hashlib.sha256(pptx_path.read_bytes()).hexdigest()
                if pptx_path.exists()
                else None
            ),
            "capture_sha256": hashlib.sha256(capture_path.read_bytes()).hexdigest(),
            "capture_metadata": {
                "source_size": [int(target.shape[1]), int(target.shape[0])],
                "capture_size": [int(rendered.shape[1]), int(rendered.shape[0])],
                "crop_rectangle": capture_metadata.get("crop_rectangle"),
                "powerpoint_version": capture_metadata.get("powerpoint_version"),
                "operating_system": capture_metadata.get("operating_system"),
                "capture_backend": capture_metadata.get(
                    "capture_backend", "Microsoft PowerPoint slideshow"
                ),
            },
            "lpips": round(lpips_score, 4),
            "ssim_srgb": round(float(metrics["ssim_srgb"]), 4),
            "psnr_srgb": round(float(metrics["psnr_srgb"]), 3),
        }
        records.append(record)
        print(
            f"  {name:<22} LPIPS {record['lpips']:.4f}  "
            f"SSIM {record['ssim_srgb']:.4f}"
        )
    output = root / "powerpoint_results.jsonl"
    output.write_text("".join(json.dumps(record) + "\n" for record in records))
    print(f"wrote {output} ({len(records)} real-PowerPoint scores)")


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------


def _latest_seed_zero_runs(root: Path) -> Dict[tuple, dict]:
    """Return the newest successful seed-0 record for each image/format."""
    results_path = root / "results.jsonl"
    selected: Dict[tuple, dict] = {}
    if not results_path.exists():
        return selected
    for line in results_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("seed") != 0 or record.get("returncode", 0) != 0:
            continue
        selected[(record.get("image"), record.get("format"))] = record
    return selected


def _data_uri(path: Path) -> str:
    """Return a file as an embeddable data URI."""
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{payload}"


def _generate_legacy_proxy_html(root: Path, output: Path) -> None:
    """Deprecated proxy-only report retained for old result directories."""
    corpus_path = root / "corpus.json"
    results_path = root / "results.jsonl"
    if not corpus_path.exists():
        raise SystemExit(f"missing corpus manifest: {corpus_path}")
    if not results_path.exists():
        raise SystemExit(f"missing benchmark results: {results_path}")

    meta = json.loads(corpus_path.read_text())["images"]
    results = [
        json.loads(line)
        for line in results_path.read_text().splitlines()
        if line.strip()
    ]
    scored = [r for r in results if r.get("lpips") is not None]
    primary = {(r["image"], r["format"]): r for r in scored if r["seed"] == 0}

    baselines = {}
    baselines_path = root / "baselines.jsonl"
    if baselines_path.exists():
        for line in baselines_path.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            # The seed-0 SVG record is the visual report's primary comparison.
            if "|seed0|" in record["key"]:
                baselines[record["image"]] = record

    classes = sorted({entry["content_class"] for entry in meta.values()})
    cards = []
    for name, entry in meta.items():
        source_path = root / entry["path"]
        source_uri = _data_uri(source_path)
        panels = [
            f"""
            <figure>
              <div class="image-wrap">
                <img loading="lazy" src="{source_uri}" alt="{html.escape(name)} source">
              </div>
              <figcaption><strong>Source</strong><span>{entry["bytes"] / 1024:.0f} KB</span></figcaption>
            </figure>"""
        ]

        for fmt, label in (("svg", "SVG"), ("pptx", "PPTX proxy")):
            record = primary.get((name, fmt))
            preview = root / "runs" / f"{name}_{fmt}_s0_splat_proxy.png"
            if record and preview.exists():
                preview_uri = _data_uri(preview)
                panels.append(
                    f"""
                    <figure>
                      <div class="image-wrap">
                        <img loading="lazy" src="{preview_uri}" alt="{html.escape(name)} {label} render">
                      </div>
                      <figcaption>
                        <strong>{label}</strong>
                        <span>LPIPS {record["lpips"]:.4f} · SSIM {record["ssim_srgb"]:.4f}</span>
                      </figcaption>
                    </figure>"""
                )
            else:
                panels.append(
                    f"""
                    <figure class="missing">
                      <div class="image-wrap"><span>No {label} run</span></div>
                      <figcaption><strong>{label}</strong><span>—</span></figcaption>
                    </figure>"""
                )

        svg = primary.get((name, "svg"))
        baseline = baselines.get(name)
        svg_lpips = svg["lpips"] if svg else 999
        metrics = ""
        if svg:
            metrics = (
                f'<span>SVG {svg["artifact_bytes"] / 1024:.0f} KB</span>'
                f'<span>{svg["splats_final"]:,} splats</span>'
                f'<span>{svg["runtime_sec"]:.0f} s</span>'
            )
        if baseline:
            metrics += (
                f'<span>matched JPEG LPIPS {baseline["jpeg"]["lpips"]:.4f}</span>'
            )

        cards.append(
            f"""
            <article class="card" data-name="{html.escape(name.lower())}"
                     data-class="{html.escape(entry["content_class"])}"
                     data-lpips="{svg_lpips}">
              <header>
                <div>
                  <h2>{html.escape(name.replace("_", " "))}</h2>
                  <p>{html.escape(entry["note"])}</p>
                </div>
                <span class="tag">{html.escape(entry["content_class"])}</span>
              </header>
              <div class="panels">{"".join(panels)}</div>
              <footer>{metrics}</footer>
            </article>"""
        )

    svg_runs = [r for r in scored if r["format"] == "svg" and r["seed"] == 0]
    pptx_runs = [r for r in scored if r["format"] == "pptx" and r["seed"] == 0]
    svg_median = statistics.median(r["lpips"] for r in svg_runs)
    pptx_median = (
        f"{statistics.median(r['lpips'] for r in pptx_runs):.4f}" if pptx_runs else "—"
    )
    options = "".join(
        f'<option value="{html.escape(cls)}">{html.escape(cls)}</option>'
        for cls in classes
    )

    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SplatThis corpus</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #0c0e12; --panel: #151820; --line: #2a2f3a;
      --muted: #9199a8; --text: #f2f4f8; --accent: #c8ff64;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0; color: var(--text); background: var(--bg);
      font: 15px/1.45 ui-sans-serif, system-ui, -apple-system, sans-serif;
    }}
    .page {{ width: min(1800px, 100%); margin: auto; padding: 32px; }}
    .masthead {{
      display: flex; gap: 32px; justify-content: space-between;
      align-items: end; padding: 28px 0 32px; border-bottom: 1px solid var(--line);
    }}
    h1 {{ margin: 0; font-size: clamp(34px, 5vw, 72px); letter-spacing: -.05em; }}
    .lede {{ max-width: 650px; margin: 8px 0 0; color: var(--muted); font-size: 17px; }}
    .summary {{ display: flex; flex-wrap: wrap; gap: 24px; }}
    .stat strong {{ display: block; color: var(--accent); font-size: 24px; }}
    .stat span {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .08em; }}
    .toolbar {{
      position: sticky; top: 0; z-index: 5; display: flex; gap: 12px;
      padding: 16px 0; background: color-mix(in srgb, var(--bg) 92%, transparent);
      backdrop-filter: blur(14px);
    }}
    input, select {{
      min-height: 42px; border: 1px solid var(--line); border-radius: 8px;
      color: var(--text); background: var(--panel); padding: 0 12px; font: inherit;
    }}
    input {{ flex: 1; }}
    .count {{ align-self: center; min-width: 92px; text-align: right; color: var(--muted); }}
    .grid {{ display: grid; gap: 18px; }}
    .card {{ padding: 18px; border: 1px solid var(--line); border-radius: 14px; background: var(--panel); }}
    .card > header {{ display: flex; justify-content: space-between; gap: 24px; align-items: start; margin-bottom: 14px; }}
    h2 {{ margin: 0; font-size: 20px; text-transform: capitalize; }}
    .card p {{ margin: 2px 0 0; color: var(--muted); }}
    .tag {{ border: 1px solid #435029; border-radius: 99px; padding: 4px 9px; color: var(--accent); font-size: 12px; }}
    .panels {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; }}
    figure {{ min-width: 0; margin: 0; border: 1px solid var(--line); border-radius: 9px; overflow: hidden; background: #090a0d; }}
    .image-wrap {{ display: grid; place-items: center; height: clamp(180px, 23vw, 390px); padding: 8px; }}
    img {{ display: block; width: 100%; height: 100%; object-fit: contain; cursor: zoom-in; }}
    figcaption {{ display: flex; justify-content: space-between; gap: 8px; padding: 9px 11px; border-top: 1px solid var(--line); font-size: 12px; }}
    figcaption span {{ color: var(--muted); text-align: right; }}
    .missing {{ color: #59606d; }}
    .card footer {{ display: flex; flex-wrap: wrap; gap: 16px; min-height: 18px; margin-top: 12px; color: var(--muted); font-size: 12px; }}
    dialog {{ width: 96vw; height: 96vh; padding: 20px; border: 1px solid var(--line); background: #050608; }}
    dialog img {{ cursor: zoom-out; }}
    dialog::backdrop {{ background: rgb(0 0 0 / .85); }}
    @media (max-width: 800px) {{
      .page {{ padding: 16px; }} .masthead {{ display: block; }}
      .summary {{ margin-top: 20px; }} .panels {{ grid-template-columns: 1fr; }}
      .image-wrap {{ height: 70vw; }} .toolbar {{ flex-wrap: wrap; }}
      input {{ flex-basis: 100%; }}
    }}
  </style>
</head>
<body>
  <main class="page">
    <section class="masthead">
      <div>
        <h1>SplatThis corpus</h1>
        <p class="lede">The complete reference set, with deployed-artifact previews and seed-0 measurements. Everything in this page is embedded; no server or companion files are required.</p>
      </div>
      <div class="summary">
        <div class="stat"><strong>{len(meta)}</strong><span>images</span></div>
        <div class="stat"><strong>{svg_median:.4f}</strong><span>SVG median LPIPS</span></div>
        <div class="stat"><strong>{pptx_median}</strong><span>PPTX median LPIPS</span></div>
      </div>
    </section>
    <nav class="toolbar">
      <input id="search" type="search" placeholder="Filter by name or note…" aria-label="Filter corpus">
      <select id="class-filter" aria-label="Filter by content class">
        <option value="">All content classes</option>{options}
      </select>
      <select id="sort" aria-label="Sort corpus">
        <option value="manifest">Corpus order</option>
        <option value="name">Name</option>
        <option value="best">Best SVG LPIPS</option>
        <option value="worst">Worst SVG LPIPS</option>
      </select>
      <span class="count" id="count">{len(meta)} shown</span>
    </nav>
    <section class="grid" id="grid">{"".join(cards)}</section>
  </main>
  <dialog id="lightbox"><img alt="Enlarged corpus render"></dialog>
  <script>
    const grid = document.querySelector("#grid");
    const cards = [...grid.children];
    const search = document.querySelector("#search");
    const classFilter = document.querySelector("#class-filter");
    const sort = document.querySelector("#sort");
    const count = document.querySelector("#count");
    const lightbox = document.querySelector("#lightbox");
    function update() {{
      const query = search.value.trim().toLowerCase();
      const cls = classFilter.value;
      let shown = 0;
      for (const card of cards) {{
        const matches = (!query || card.textContent.toLowerCase().includes(query))
          && (!cls || card.dataset.class === cls);
        card.hidden = !matches;
        shown += matches ? 1 : 0;
      }}
      const direction = sort.value === "worst" ? -1 : 1;
      const ordered = [...cards].sort((a, b) => {{
        if (sort.value === "name") return a.dataset.name.localeCompare(b.dataset.name);
        if (sort.value === "best" || sort.value === "worst")
          return direction * (Number(a.dataset.lpips) - Number(b.dataset.lpips));
        return cards.indexOf(a) - cards.indexOf(b);
      }});
      ordered.forEach(card => grid.append(card));
      count.textContent = `${{shown}} shown`;
    }}
    search.addEventListener("input", update);
    classFilter.addEventListener("change", update);
    sort.addEventListener("change", update);
    document.querySelectorAll("figure img").forEach(image => {{
      image.addEventListener("click", () => {{
        lightbox.querySelector("img").src = image.src;
        lightbox.showModal();
      }});
    }});
    lightbox.addEventListener("click", () => lightbox.close());
  </script>
</body>
</html>
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document)
    print(f"wrote {output} ({output.stat().st_size / 1024 / 1024:.1f} MB)")


def generate_canvas_corpus_html(root: Path, output: Path) -> None:
    """Build one HTML containing a live canvas-splat render for every image."""
    from splatthis.color import linear_to_srgb
    from splatthis.io import atomic_write_text, load_splats_json
    from splatthis.splat import render_importance_for_raw

    corpus_path = root / "corpus.json"
    if not corpus_path.exists():
        raise SystemExit(f"missing corpus manifest: {corpus_path}")
    meta = json.loads(corpus_path.read_text())["images"]
    selected_runs = _latest_seed_zero_runs(root)
    canvas_history: dict[str, dict[int, dict[str, Any]]] = {}
    results_path = root / "results.jsonl"
    if results_path.exists():
        for line in results_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (
                record.get("format") != "canvas"
                or record.get("seed") != 0
                or record.get("returncode", 0) != 0
                or not isinstance(record.get("splats_requested"), (int, float))
                or not isinstance(record.get("lpips"), (int, float))
            ):
                continue
            canvas_history.setdefault(str(record["image"]), {})[
                int(record["splats_requested"])
            ] = record
    powerpoint_scores = {}
    powerpoint_results = root / "powerpoint_results.jsonl"
    if powerpoint_results.exists():
        for line in powerpoint_results.read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                powerpoint_scores[record["image"]] = record

    models = []
    for name, entry in meta.items():
        selected = None
        for fmt in ("canvas", "svg", "pptx"):
            run_record = selected_runs.get((name, fmt), {})
            recorded_artifacts = run_record.get("artifacts_path")
            artifact_dir = (
                root / recorded_artifacts
                if recorded_artifacts
                else root / "runs" / f"{name}_{fmt}_s0_art"
            )
            raw_path = artifact_dir / "final.raw.json"
            manifest_path = artifact_dir / "run_manifest.json"
            if raw_path.exists() and manifest_path.exists():
                selected = (fmt, raw_path, manifest_path)
                break
        if selected is None:
            print(f"warning: no seed-0 splats for {name}; skipping")
            continue

        fmt, raw_path, manifest_path = selected
        manifest = json.loads(manifest_path.read_text())
        config = manifest.get("config", {})
        training_seconds = (
            manifest.get("acceptance", {}).get("measured", {}).get("runtime_sec")
        )
        width, height = config.get("resolved_target_size", entry["size"])
        training_target = str(
            config.get(
                "training_export_target",
                config.get("refinement_config", {}).get(
                    "training_export_target", "canvas"
                ),
            )
        )
        srgb_mode = training_target in SRGB_TRAINING_TARGETS

        background = np.asarray(
            config.get("background_linear_rgb", [0.0, 0.0, 0.0]),
            dtype=np.float32,
        )
        if srgb_mode:
            background = linear_to_srgb(background)

        rows = []
        for splat in load_splats_json(str(raw_path)):
            raw = splat.to_raw_splat()
            rgb = np.clip(np.asarray([raw.r, raw.g, raw.b], dtype=np.float32), 0.0, 1.0)
            if srgb_mode:
                rgb = linear_to_srgb(rgb)
            rows.append(
                [
                    round(float(raw.x), 5),
                    round(float(raw.y), 5),
                    round(float(raw.sx), 5),
                    round(float(raw.sy), 5),
                    round(float(raw.theta), 6),
                    round(float(rgb[0]), 5),
                    round(float(rgb[1]), 5),
                    round(float(rgb[2]), 5),
                    round(float(raw.a), 5),
                    round(float(render_importance_for_raw(raw)), 6),
                ]
            )
        rows.sort(key=lambda row: row[9])

        models.append(
            {
                "name": name,
                "label": name.replace("_", " "),
                "content_class": entry["content_class"],
                "note": entry["note"],
                "width": int(width),
                "height": int(height),
                "background": [round(float(value), 6) for value in background[:3]],
                "srgb": srgb_mode,
                "source_format": fmt,
                "training_seconds": training_seconds,
                "splats": rows,
            }
        )

    if not models:
        raise SystemExit(f"no seed-0 splat artifacts found below {root / 'runs'}")

    def metric_text(value: object, decimals: int) -> str:
        if not isinstance(value, (int, float)):
            return "—"
        return f"{float(value):.{decimals}f}"

    def duration_text(value: object) -> str:
        if not isinstance(value, (int, float)):
            return "—"
        seconds = float(value)
        return f"{seconds / 60:.1f} min" if seconds >= 60 else f"{seconds:.1f} s"

    def bytes_text(value: object) -> str:
        if not isinstance(value, (int, float)):
            return "—"
        size = float(value)
        return (
            f"{size / 1024 / 1024:.1f} MB"
            if size >= 1024 * 1024
            else f"{size / 1024:.0f} KB"
        )

    cards_parts = []
    for index, model in enumerate(models):
        name = model["name"]
        entry = meta[name]
        source_path = root / entry["path"]
        source_uri = _data_uri(source_path)
        canvas_record = selected_runs.get((name, model["source_format"]), {})
        svg_record = selected_runs.get((model["name"], "svg"), {})
        svg_path = (
            root / svg_record["output_path"]
            if svg_record.get("output_path")
            else root / "runs" / f"{model['name']}_svg_s0.svg"
        )
        if svg_path.exists():
            svg_uri = _data_uri(svg_path)
            svg_panel = (
                f'<a class="svg-artifact" href="{svg_uri}" '
                f'download="{html.escape(model["name"])}.svg">'
                f'<img src="{svg_uri}" alt="{html.escape(model["label"])} SVG">'
                "</a>"
            )
            svg_caption = (
                "Emitted SVG"
                f" · LPIPS {metric_text(svg_record.get('lpips'), 4)}"
                f" · SSIM {metric_text(svg_record.get('ssim_srgb'), 4)}"
                f" · {bytes_text(svg_path.stat().st_size)} · download"
            )
        else:
            svg_panel = '<span class="missing">No SVG artifact</span>'
            svg_caption = "No SVG artifact"
        pptx_record = selected_runs.get((model["name"], "pptx"), {})
        pptx_path = (
            root / pptx_record["output_path"]
            if pptx_record.get("output_path")
            else root / "runs" / f"{model['name']}_pptx_s0.pptx"
        )
        if pptx_path.exists():
            pptx_uri = _data_uri(pptx_path)
            capture_path = pptx_path.with_name(f"{pptx_path.stem}_powerpoint_slide.png")
            if capture_path.exists():
                pptx_panel = (
                    f'<a class="pptx-capture" href="{pptx_uri}" '
                    f'download="{html.escape(model["name"])}.pptx">'
                    f'<img src="{_data_uri(capture_path)}" '
                    f'alt="{html.escape(model["label"])} rendered by PowerPoint">'
                    "</a>"
                )
            else:
                pptx_panel = (
                    f'<a class="document-artifact" href="{pptx_uri}" '
                    f'download="{html.escape(model["name"])}.pptx">'
                    "<strong>PPTX</strong><span>Native DrawingML</span>"
                    "</a>"
                )
            score = powerpoint_scores.get(model["name"])
            score_text = (
                f" · LPIPS {score['lpips']:.4f} · SSIM {score['ssim_srgb']:.4f}"
                if score
                else ""
            )
            pptx_caption = (
                f"PowerPoint capture{score_text} · "
                f"{pptx_path.stat().st_size / 1024:.0f} KB · download"
            )
        else:
            pptx_panel = '<span class="missing">No PPTX artifact</span>'
            pptx_caption = "No PPTX artifact"

        score = powerpoint_scores.get(name, {})
        comparison_rows = [
            {
                "output": (
                    "Pixel runtime"
                    if canvas_record.get("render_kind") in PIXEL_RUNTIME_CAPTURE_KINDS
                    else "Native Canvas"
                ),
                "evaluation": (
                    "actual browser pixel buffer"
                    if canvas_record.get("render_kind") in DEPLOYED_CANVAS_CAPTURE_KINDS
                    else "runtime-model proxy"
                ),
                "record": canvas_record,
                "splats": len(model["splats"]),
                "runtime": model["training_seconds"],
                "bytes": canvas_record.get("artifact_bytes"),
            },
            {
                "output": "SVG",
                "evaluation": str(svg_record.get("renderer") or "artifact raster"),
                "record": svg_record,
                "splats": svg_record.get("splats_final"),
                "runtime": svg_record.get("runtime_sec"),
                "bytes": svg_path.stat().st_size if svg_path.exists() else None,
            },
            {
                "output": "PowerPoint",
                "evaluation": (
                    "actual slideshow capture" if score else "capture unavailable"
                ),
                "record": score,
                "splats": pptx_record.get("splats_final"),
                "runtime": pptx_record.get("runtime_sec"),
                "bytes": pptx_path.stat().st_size if pptx_path.exists() else None,
            },
        ]
        available = [
            row
            for row in comparison_rows
            if isinstance(row["record"].get("lpips"), (int, float))
        ]
        best_lpips = min(
            (float(row["record"]["lpips"]) for row in available), default=None
        )
        best_ssim = max(
            (float(row["record"]["ssim_srgb"]) for row in available), default=None
        )
        best_psnr = max(
            (float(row["record"]["psnr_srgb"]) for row in available), default=None
        )
        stats_rows = []
        for row in comparison_rows:
            record = row["record"]

            def quality_cell(field: str, decimals: int, best: object) -> str:
                value = record.get(field)
                class_name = (
                    ' class="best"'
                    if isinstance(value, (int, float))
                    and best is not None
                    and abs(float(value) - float(best)) < 1e-9
                    else ""
                )
                return f"<td{class_name}>{metric_text(value, decimals)}</td>"

            splats_value = row["splats"]
            splats_text = (
                f"{int(splats_value):,}"
                if isinstance(splats_value, (int, float))
                else "—"
            )
            stats_rows.append(
                "<tr>"
                f"<th scope=\"row\">{html.escape(str(row['output']))}"
                f"<small>{html.escape(str(row['evaluation']))}</small></th>"
                f"{quality_cell('lpips', 4, best_lpips)}"
                f"{quality_cell('ssim_srgb', 4, best_ssim)}"
                f"{quality_cell('psnr_srgb', 2, best_psnr)}"
                f"<td>{splats_text}</td>"
                f"<td>{duration_text(row['runtime'])}</td>"
                f"<td>{bytes_text(row['bytes'])}</td>"
                "</tr>"
            )
        stats_table = (
            '<div class="comparison"><table>'
            "<caption>Artifact comparison</caption>"
            "<thead><tr><th>Output / evaluator</th><th>LPIPS ↓</th>"
            "<th>SSIM ↑</th><th>PSNR ↑</th><th>Splats</th>"
            "<th>Training</th><th>Size</th></tr></thead>"
            f"<tbody>{''.join(stats_rows)}</tbody></table>"
            '<p class="stats-note">Best perceptual score in green. SVG and '
            "PowerPoint are measured from the emitted artifacts; canvas quality "
            "uses the browser pixel buffer when an actual capture is available; "
            "otherwise the equivalent runtime model is marked as a proxy.</p></div>"
        )
        history_records = [
            canvas_history[name][budget]
            for budget in sorted(canvas_history.get(name, {}))
        ]
        history_rows = []
        for history_record in history_records:
            evaluator = str(
                history_record.get("renderer")
                or history_record.get("render_kind")
                or "unknown"
            )
            actual = history_record.get("render_kind") in PIXEL_RUNTIME_CAPTURE_KINDS
            browser_ms = history_record.get("browser_render_ms")
            browser_samples = [
                float(value)
                for value in history_record.get("browser_render_ms_samples", [])
                if isinstance(value, (int, float))
            ]
            if isinstance(browser_ms, (int, float)):
                browser_text = f"{float(browser_ms):.0f} ms"
                if len(browser_samples) > 1:
                    browser_text += (
                        f"<small>{min(browser_samples):.0f}–"
                        f"{max(browser_samples):.0f} ms</small>"
                    )
            else:
                browser_text = "—"
            history_rows.append(
                "<tr>"
                f"<td>{int(history_record['splats_requested']):,}</td>"
                f"<td>{int(history_record.get('splats_final') or 0):,}</td>"
                f"<td>{metric_text(history_record.get('lpips'), 4)}</td>"
                f"<td>{metric_text(history_record.get('ssim_srgb'), 4)}</td>"
                f"<td>{metric_text(history_record.get('ms_ssim_luma'), 4)}</td>"
                f"<td>{metric_text(history_record.get('psnr_srgb'), 2)}</td>"
                f"<td>{metric_text(history_record.get('worst_roi_error'), 4)}</td>"
                f"<td>{metric_text(history_record.get('edge_chamfer'), 2)}</td>"
                f"<td>{duration_text(history_record.get('runtime_sec'))}</td>"
                f"<td>{browser_text}</td>"
                f"<td>{bytes_text(history_record.get('artifact_bytes'))}</td>"
                f"<td>{html.escape(evaluator)}"
                f"{' · actual' if actual else ' · proxy'}</td>"
                "</tr>"
            )
        history_table = (
            '<div class="comparison canvas-history"><table>'
            "<caption>Canvas budget history · full frame · seed 0</caption>"
            "<thead><tr><th>Requested</th><th>Final</th><th>LPIPS ↓</th>"
            "<th>SSIM ↑</th><th>MS-SSIM ↑</th><th>PSNR ↑</th>"
            "<th>Worst ROI ↓</th><th>Edge ↓</th><th>Training</th>"
            "<th>Browser render (median)</th><th>HTML size</th><th>Evaluator</th>"
            "</tr></thead>"
            f"<tbody>{''.join(history_rows)}</tbody></table></div>"
            if history_rows
            else ""
        )
        cards_parts.append(
            f"""
        <article class="card" data-name="{html.escape(model["name"])}"
                 data-model-index="{index}"
                 data-class="{html.escape(model["content_class"])}">
          <header>
            <div>
              <h2>{html.escape(model["label"])}</h2>
              <p>{html.escape(model["note"])}</p>
            </div>
            <span>{html.escape(model["content_class"])}</span>
          </header>
          <div class="panels">
            <figure>
              <div class="stage">
                <img src="{source_uri}" alt="{html.escape(model["label"])} source">
              </div>
              <figcaption>Source PNG · {entry["size"][0]}×{entry["size"][1]} · {bytes_text(entry["bytes"])}</figcaption>
            </figure>
            <figure class="canvas-panel">
              <div class="stage">
                <canvas id="canvas-{index}" width="{model["width"]}" height="{model["height"]}"></canvas>
              </div>
              <figcaption>
                <strong>HTML canvas splat</strong>
                <span id="status-{index}">queued · {len(model["splats"]):,} {html.escape(model["source_format"])}-trained splats</span>
              </figcaption>
            </figure>
            <figure>
              <div class="stage">{svg_panel}</div>
              <figcaption>{svg_caption}</figcaption>
            </figure>
            <figure>
              <div class="stage">{pptx_panel}</div>
              <figcaption>{pptx_caption}</figcaption>
            </figure>
          </div>
          {stats_table}
          {history_table}
        </article>"""
        )
    cards = "".join(cards_parts)
    classes = "".join(
        f'<option value="{html.escape(value)}">{html.escape(value)}</option>'
        for value in sorted({model["content_class"] for model in models})
    )
    paired_canvas = [
        (history[2000], history[4000])
        for history in canvas_history.values()
        if 2000 in history
        and 4000 in history
        and history[2000].get("render_kind") in PIXEL_RUNTIME_CAPTURE_KINDS
        and history[4000].get("render_kind") in PIXEL_RUNTIME_CAPTURE_KINDS
    ]
    if paired_canvas:
        median_2k_ssim = statistics.median(
            before["ssim_srgb"] for before, _ in paired_canvas
        )
        median_4k_ssim = statistics.median(
            after["ssim_srgb"] for _, after in paired_canvas
        )
        median_delta_ssim = statistics.median(
            after["ssim_srgb"] - before["ssim_srgb"] for before, after in paired_canvas
        )
        median_2k_lpips = statistics.median(
            before["lpips"] for before, _ in paired_canvas
        )
        median_4k_lpips = statistics.median(
            after["lpips"] for _, after in paired_canvas
        )
        median_delta_lpips = statistics.median(
            after["lpips"] - before["lpips"] for before, after in paired_canvas
        )
        improved_both = sum(
            after["ssim_srgb"] > before["ssim_srgb"]
            and after["lpips"] < before["lpips"]
            for before, after in paired_canvas
        )
        paired_summary = (
            f"Paired actual-Chrome 2k→4k · n={len(paired_canvas)} · "
            f"median SSIM {median_2k_ssim:.4f}→{median_4k_ssim:.4f} "
            f"(Δ {median_delta_ssim:+.4f}) · "
            f"LPIPS {median_2k_lpips:.4f}→{median_4k_lpips:.4f} "
            f"(Δ {median_delta_lpips:+.4f}) · "
            f"{improved_both}/{len(paired_canvas)} improve both"
        )
    else:
        paired_summary = "Paired 2k→4k canvas statistics are not complete yet."
    models_json = json.dumps(models, separators=(",", ":"))

    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SplatThis canvas corpus</title>
  <style>
    :root {{
      color-scheme: dark; --bg:#090b0f; --panel:#151820; --line:#2b303b;
      --text:#f1f4f8; --muted:#929bad; --accent:#c8ff64;
    }}
    * {{ box-sizing:border-box }}
    body {{
      margin:0; background:var(--bg); color:var(--text);
      font:15px/1.4 ui-sans-serif,system-ui,-apple-system,sans-serif;
    }}
    main {{ width:min(1800px,100%); margin:auto; padding:32px }}
    .masthead {{
      display:flex; justify-content:space-between; align-items:end; gap:32px;
      padding:28px 0 32px; border-bottom:1px solid var(--line)
    }}
    h1 {{ margin:0; font-size:clamp(36px,6vw,76px); letter-spacing:-.055em }}
    .lede {{ max-width:700px; margin:8px 0 0; color:var(--muted); font-size:17px }}
    .total {{ color:var(--accent); font-size:26px; white-space:nowrap }}
    .toolbar {{
      position:sticky; top:0; z-index:4; display:flex; gap:12px; padding:16px 0;
      background:rgb(9 11 15/.92); backdrop-filter:blur(12px)
    }}
    input,select {{
      min-height:42px; padding:0 12px; border:1px solid var(--line);
      border-radius:8px; background:var(--panel); color:var(--text); font:inherit
    }}
    input {{ flex:1 }}
    .count {{ align-self:center; min-width:90px; text-align:right; color:var(--muted) }}
    .grid {{ display:grid; gap:18px }}
    .card {{
      min-width:0; overflow:hidden; border:1px solid var(--line);
      border-radius:14px; background:var(--panel)
    }}
    .card header {{ display:flex; justify-content:space-between; gap:18px; padding:16px }}
    h2 {{ margin:0; font-size:19px; text-transform:capitalize }}
    .card p {{ margin:3px 0 0; color:var(--muted); font-size:13px }}
    .card header span {{
      height:max-content; padding:4px 8px; border:1px solid #435029;
      border-radius:99px; color:var(--accent); font-size:11px
    }}
    .panels {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)) }}
    figure {{ min-width:0; margin:0; border-top:1px solid var(--line) }}
    figure + figure {{ border-left:1px solid var(--line) }}
    .stage {{
      display:grid; place-items:center; height:clamp(240px,25vw,430px);
      padding:10px; background:#050608
    }}
    .stage img,.stage canvas {{
      display:block; width:auto; height:auto; max-width:100%; max-height:100%;
      object-fit:contain; image-rendering:auto
    }}
    .svg-artifact {{
      display:grid; place-items:center; width:100%; height:100%; cursor:zoom-in
    }}
    .pptx-capture {{
      display:grid; place-items:center; width:100%; height:100%; cursor:pointer
    }}
    .document-artifact {{
      display:grid; place-items:center; align-content:center; gap:8px;
      width:100%; height:100%; color:var(--text); text-decoration:none
    }}
    .document-artifact strong {{
      padding:13px 18px; border:2px solid var(--accent); border-radius:8px;
      color:var(--accent); font-size:28px; letter-spacing:.08em
    }}
    .document-artifact span {{ color:var(--muted); font-size:12px }}
    figcaption {{
      padding:9px 12px; border-top:1px solid var(--line); color:var(--muted);
      font-size:12px; text-transform:uppercase; letter-spacing:.06em
    }}
    figcaption strong {{ display:block; color:var(--text); font-size:12px }}
    figcaption span {{
      display:block; margin-top:4px; text-transform:none; letter-spacing:0;
      font:11px/1.35 ui-monospace,SFMono-Regular,Menlo,monospace
    }}
    .comparison {{
      overflow-x:auto; padding:14px 16px 16px; border-top:1px solid var(--line)
    }}
    table {{ width:100%; border-collapse:collapse; font-variant-numeric:tabular-nums }}
    caption {{
      padding:0 0 9px; color:var(--text); text-align:left; font-size:13px;
      font-weight:700; text-transform:uppercase; letter-spacing:.06em
    }}
    th,td {{
      padding:8px 10px; border-top:1px solid var(--line); text-align:right;
      white-space:nowrap; font-size:12px
    }}
    thead th {{
      border-top:0; color:var(--muted); font-size:10px; text-transform:uppercase;
      letter-spacing:.06em
    }}
    th:first-child {{ text-align:left }}
    tbody th {{ color:var(--text); font-weight:650 }}
    tbody th small {{
      display:block; color:var(--muted); font-weight:400; text-transform:none
    }}
    tbody td small {{ display:block; color:var(--muted); font-weight:400 }}
    td.best {{ color:var(--accent); font-weight:750 }}
    .stats-note {{ margin:9px 0 0!important; font-size:11px!important }}
    .missing {{ color:var(--muted) }}
    @media (max-width:700px) {{
      main {{ padding:16px }} .masthead {{ display:block }} .total {{ display:block; margin-top:18px }}
      .toolbar {{ flex-wrap:wrap }} input {{ flex-basis:100% }}
      .panels {{ grid-template-columns:1fr }} figure + figure {{ border-left:0 }}
      .stage {{ height:80vw }}
      .comparison {{ padding-inline:10px }}
    }}
  </style>
</head>
<body>
  <main>
    <section class="masthead">
      <div>
        <h1>Canvas corpus</h1>
        <p class="lede">The complete corpus: source image, live Gaussian-splat canvas, actual SVG, and native PPTX captured from Microsoft PowerPoint. Internal proxy previews are never presented as PowerPoint renders.</p>
        <p class="lede">{html.escape(paired_summary)}</p>
      </div>
      <strong class="total">{len(models)} canvases · {len(powerpoint_scores)} PowerPoint captures</strong>
    </section>
    <nav class="toolbar">
      <input id="search" type="search" placeholder="Filter the corpus…" aria-label="Filter corpus">
      <select id="class-filter" aria-label="Filter by content class">
        <option value="">All content classes</option>{classes}
      </select>
      <span class="count" id="count">{len(models)} shown</span>
    </nav>
    <section class="grid"> {cards} </section>
  </main>
  <script>
    const MODELS={models_json};
    const FOOTPRINT=3.0;
    function render(index) {{
      const model=MODELS[index], canvas=document.querySelector(`#canvas-${{index}}`);
      if(model.splats===null||!canvas)return;
      const status=document.querySelector(`#status-${{index}}`), t0=performance.now();
      const W=model.width,H=model.height,ctx=canvas.getContext("2d");
      const rgb=new Float32Array(W*H*3),transmittance=new Float32Array(W*H).fill(1);
      for(const s of model.splats) {{
        const [x,y,sx0,sy0,theta,r,g,b,alpha]=s;
        const sx=Math.max(sx0,1e-4),sy=Math.max(sy0,1e-4);
        const ct=Math.cos(theta),st=Math.sin(theta);
        const rx=Math.max(1,Math.ceil(FOOTPRINT*Math.sqrt((sx*ct)*(sx*ct)+(sy*st)*(sy*st))));
        const ry=Math.max(1,Math.ceil(FOOTPRINT*Math.sqrt((sx*st)*(sx*st)+(sy*ct)*(sy*ct))));
        const x0=Math.max(0,Math.floor(x-rx)),x1=Math.min(W,Math.ceil(x+rx+1));
        const y0=Math.max(0,Math.floor(y-ry)),y1=Math.min(H,Math.ceil(y+ry+1));
        const isx=1/(sx*sx),isy=1/(sy*sy);
        for(let py=y0;py<y1;py++) for(let px=x0;px<x1;px++) {{
          const dx=px-x,dy=py-y,u=ct*dx+st*dy,v=-st*dx+ct*dy;
          const layerAlpha=1-Math.exp(-Math.min(1,Math.max(0,alpha))*Math.exp(-.5*(u*u*isx+v*v*isy)));
          const i=py*W+px,t=transmittance[i],amount=t*layerAlpha,j=i*3;
          rgb[j]+=amount*r;rgb[j+1]+=amount*g;rgb[j+2]+=amount*b;
          transmittance[i]=t*(1-layerAlpha);
        }}
      }}
      const image=ctx.createImageData(W,H),out=image.data,threshold=.0031308;
      const encode=value=>value<=threshold?12.92*value:1.055*Math.pow(value,1/2.4)-.055;
      for(let i=0;i<W*H;i++) {{
        const j=i*3,k=i*4,t=transmittance[i];
        let r=Math.min(1,Math.max(0,rgb[j]+t*model.background[0]));
        let g=Math.min(1,Math.max(0,rgb[j+1]+t*model.background[1]));
        let b=Math.min(1,Math.max(0,rgb[j+2]+t*model.background[2]));
        if(!model.srgb) {{ r=encode(r);g=encode(g);b=encode(b) }}
        out[k]=(r*255+.5)|0;out[k+1]=(g*255+.5)|0;out[k+2]=(b*255+.5)|0;out[k+3]=255;
      }}
      ctx.putImageData(image,0,0);
      const training=model.training_seconds==null?"training time unavailable":`training ${{(model.training_seconds/60).toFixed(1)}} min`;
      status.textContent=`${{model.splats.length.toLocaleString()}} ${{model.source_format}}-trained splats · ${{model.srgb?"sRGB":"linear"}} alpha-over · browser render ${{(performance.now()-t0).toFixed(0)}} ms · ${{training}}`;
      model.splats=null;
    }}
    const cards=[...document.querySelectorAll(".card")],pending=new Set();
    function scheduleRender(index) {{
      if(MODELS[index].splats===null||pending.has(index))return;
      pending.add(index);
      const work=()=>{{pending.delete(index);render(index)}};
      if("requestIdleCallback" in window)requestIdleCallback(work,{{timeout:250}});
      else setTimeout(work,0);
    }}
    if("IntersectionObserver" in window) {{
      const observer=new IntersectionObserver(entries=>{{
        for(const entry of entries)if(entry.isIntersecting){{
          scheduleRender(Number(entry.target.dataset.modelIndex));
          observer.unobserve(entry.target);
        }}
      }},{{rootMargin:"800px 0px"}});
      cards.forEach(card=>observer.observe(card));
    }} else {{
      cards.forEach(card=>scheduleRender(Number(card.dataset.modelIndex)));
    }}

    const search=document.querySelector("#search");
    const classFilter=document.querySelector("#class-filter"),count=document.querySelector("#count");
    function filter() {{
      const query=search.value.trim().toLowerCase(),cls=classFilter.value;
      let shown=0;
      for(const card of cards) {{
        const matches=(!query||card.textContent.toLowerCase().includes(query))&&(!cls||card.dataset.class===cls);
        card.hidden=!matches;shown+=matches?1:0;
      }}
      count.textContent=`${{shown}} shown`;
    }}
    search.addEventListener("input",filter);classFilter.addEventListener("change",filter);
  </script>
</body>
</html>
"""
    atomic_write_text(output, document)
    total_splats = sum(len(model["splats"]) for model in models)
    print(
        f"wrote {output} ({output.stat().st_size / 1024 / 1024:.1f} MB, "
        f"{len(models)} canvases, {total_splats:,} splats)"
    )


def jpeg_at_matched_bytes(source_png: Path, target_bytes: int) -> Optional[dict]:
    """Best JPEG that fits in `target_bytes` — the obvious alternative.

    A reviewer's first question about any image-to-vector pipeline is whether
    it beats simply shipping a raster of the same size. Binary-searches JPEG
    quality for the largest file that still fits the budget.
    """
    import io as _io

    from PIL import Image

    from splatthis.io import (
        compute_quality_metrics,
        linear_to_srgb,
        load_png,
        srgb_to_linear,
    )

    src = Image.open(source_png).convert("RGB")
    lo, hi, best = 1, 95, None
    while lo <= hi:
        q = (lo + hi) // 2
        buf = _io.BytesIO()
        src.save(buf, "JPEG", quality=q, optimize=True)
        size = buf.tell()
        if size <= target_bytes:
            best = (q, size, buf.getvalue())
            lo = q + 1
        else:
            hi = q - 1
    if best is None:
        return None

    q, size, payload = best
    cand = (
        np.asarray(Image.open(_io.BytesIO(payload)).convert("RGB"), dtype=np.float32)
        / 255.0
    )
    target_lin = load_png(str(source_png))[..., :3]
    m = compute_quality_metrics(target_lin, srgb_to_linear(cand))
    lp = _lpips_score(linear_to_srgb(target_lin), cand)
    return {
        "quality": q,
        "bytes": size,
        "lpips": round(lp, 4),
        "ssim_srgb": round(float(m["ssim_srgb"]), 4),
        "psnr_srgb": round(float(m["psnr_srgb"]), 3),
    }


def run_baselines(root: Path) -> None:
    """For every scored SVG run, the JPEG that fits the same byte budget."""
    results_path = root / "results.jsonl"
    out_path = root / "baselines.jsonl"
    recs = [json.loads(x) for x in results_path.read_text().splitlines() if x.strip()]
    svg = [r for r in recs if r["format"] == "svg" and r.get("artifact_bytes")]
    done = load_done(out_path)
    meta = json.loads((root / "corpus.json").read_text())["images"]

    print(f"{len(svg)} svg runs; {len(done)} baselines already recorded")
    for r in svg:
        key = f"jpeg|{r['key']}"
        if key in done:
            continue
        src = root / meta[r["image"]]["path"]
        j = jpeg_at_matched_bytes(src, r["artifact_bytes"])
        if j is None:
            continue
        rec = {
            "key": key,
            "image": r["image"],
            "content_class": r["content_class"],
            "svg_bytes": r["artifact_bytes"],
            "svg_lpips": r["lpips"],
            "jpeg": j,
            "lpips_gap": round(r["lpips"] - j["lpips"], 4),
        }
        with out_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(
            f"  {r['image']:<22} svg {r['lpips']:.4f}  jpeg-q{j['quality']:<3} "
            f"{j['lpips']:.4f}  gap {rec['lpips_gap']:+.4f}"
        )


def _code_fingerprint() -> str:
    """Fingerprint executable benchmark/converter code, including dirty edits."""
    digest = hashlib.sha256()
    paths = sorted((REPO / "src" / "splatthis").glob("*.py"))
    paths.append(Path(__file__))
    for path in paths:
        digest.update(str(path.relative_to(REPO)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def _run_config(
    source: Path,
    fmt: str,
    seed: int,
    splats: int,
    stages: Optional[str],
    profile: str,
    optimizer_backend: str,
    full_geometry: bool,
    initial_splat_cap: Optional[int] = None,
    initial_splat_fraction: Optional[float] = None,
    adaptive_compute: bool = False,
    adaptive_target_ssim_srgb: float = 0.98,
    adaptive_min_checkpoints: int = 2,
    adaptive_chrome_ssim_margin: float = DEFAULT_CHROME_SSIM_SAFETY_MARGIN,
    adaptive_chrome_psnr_margin: float = DEFAULT_CHROME_PSNR_SAFETY_MARGIN,
    training_export_target: str = "auto",
) -> dict:
    """Return the canonical, content-addressed identity of a corpus run."""
    geometry = full_geometry or profile in {"balanced", "max-fidelity"}
    return {
        "schema": 3,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "format": fmt,
        "seed": seed,
        "splats": splats,
        "stages": stages or "cli-default",
        "profile": profile,
        "optimizer_backend": optimizer_backend,
        "training_export_target": training_export_target,
        "mlx_tile_plan": "periodic" if geometry else "profile-default",
        "mlx_tile_plan_rebuild_interval": 10 if full_geometry else "profile-default",
        "mlx_trainable_groups": (
            "position,scale,theta,color,alpha" if geometry else "profile-default"
        ),
        "initial_splat_cap": (
            int(initial_splat_cap)
            if initial_splat_cap is not None
            else "profile-default"
        ),
        "initial_splat_fraction": (
            float(initial_splat_fraction)
            if initial_splat_fraction is not None
            else "profile-default"
        ),
        "adaptive_compute": bool(adaptive_compute),
        "adaptive_target_ssim_srgb": float(adaptive_target_ssim_srgb),
        "adaptive_min_checkpoints": int(adaptive_min_checkpoints),
        "adaptive_chrome_ssim_margin": float(adaptive_chrome_ssim_margin),
        "adaptive_chrome_psnr_margin": float(adaptive_chrome_psnr_margin),
        "code_fingerprint": _code_fingerprint(),
    }


def _config_hash(config: dict) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def run_key(image: str, fmt: str, seed: int, config_hash: str) -> str:
    return f"{image}|{fmt}|seed{seed}|cfg-{config_hash}"


def load_done(results_path: Path) -> set:
    """Return successful content-addressed runs whose primary artifact exists."""

    if not results_path.exists():
        return set()
    done = set()
    for line in results_path.read_text().splitlines():
        if line.strip():
            try:
                record = json.loads(line)
            except Exception:
                continue
            if int(record.get("returncode", 0)) != 0 or record.get("error"):
                continue
            output_path = record.get("output_path")
            if output_path:
                artifact = results_path.parent / output_path
                if not artifact.is_file():
                    continue
                expected_size = record.get("artifact_bytes")
                if expected_size is not None and artifact.stat().st_size != int(
                    expected_size
                ):
                    continue
                expected_sha = record.get("artifact_sha256")
                if expected_sha and hashlib.sha256(
                    artifact.read_bytes()
                ).hexdigest() != str(expected_sha):
                    continue
            # A row written under an older schema is missing fields that later
            # analysis depends on, so it must not count as done. Re-scoring it
            # is cheap relative to trusting it: the alternative is an aggregate
            # that silently mixes deployed evidence with proxies.
            if int(record.get("schema_version", 1)) < RESULT_SCHEMA_VERSION:
                continue
            done.add(record["key"])
    return done


@dataclass(frozen=True)
class CorpusRunJob:
    """One isolated converter subprocess and its content-addressed outputs."""

    index: int
    total: int
    key: str
    name: str
    fmt: str
    seed: int
    config: dict
    config_hash: str
    source: Path
    output: Path
    artifacts: Path
    command: tuple[str, ...]


def _execute_corpus_job(
    job: CorpusRunJob,
) -> tuple[CorpusRunJob, subprocess.CompletedProcess[str], float]:
    """Run one converter in an isolated process; safe for a thread scheduler."""

    started = time.perf_counter()
    process = subprocess.run(job.command, capture_output=True, text=True)
    return job, process, time.perf_counter() - started


def _execute_corpus_jobs(
    jobs: List[CorpusRunJob], max_workers: int
) -> Iterator[tuple[CorpusRunJob, subprocess.CompletedProcess[str], float]]:
    """Yield completed runs serially or concurrently without concurrent writes."""

    if max_workers <= 1:
        for job in jobs:
            yield _execute_corpus_job(job)
        return
    with ThreadPoolExecutor(
        max_workers=max_workers, thread_name_prefix="splatthis-corpus"
    ) as executor:
        futures = [executor.submit(_execute_corpus_job, job) for job in jobs]
        for future in as_completed(futures):
            yield future.result()


def _resolve_corpus_worker_count(requested: int, optimizer_backend: str) -> int:
    """Protect seeded MLX runs from shared-Metal cross-process drift."""

    workers = int(requested)
    if workers < 1:
        raise ValueError("corpus jobs must be positive")
    if workers > 1 and str(optimizer_backend).strip().lower() == "mlx":
        raise ValueError(
            "parallel MLX corpus jobs are disabled: concurrent processes share "
            "one Metal device and produced seed-identical parameter drift; use "
            "--jobs 1 for MLX or select the Torch backend"
        )
    return workers


def _build_corpus_job(
    *,
    index: int,
    total: int,
    runs_dir: Path,
    source: Path,
    name: str,
    fmt: str,
    seed: int,
    config: dict,
    config_hash: str,
) -> CorpusRunJob:
    """Materialize the command and unique output paths for one run."""

    stem = runs_dir / f"{name}_{fmt}_s{seed}_{config_hash}"
    suffix = {
        "svg": ".svg",
        "pptx": ".pptx",
        "canvas": ".html",
        "pixel-runtime": ".html",
    }.get(fmt)
    if suffix is None:
        raise ValueError(f"unsupported corpus format: {fmt}")
    output = stem.with_suffix(suffix)
    artifacts = Path(str(stem) + "_art")
    command = [
        sys.executable,
        "-m",
        "splatthis.cli",
        str(source),
        "-o",
        str(output),
        "--seed",
        str(seed),
        "--splats",
        str(config["splats"]),
        "--format",
        fmt,
        "--profile",
        str(config["profile"]),
        "--optimizer-backend",
        str(config["optimizer_backend"]),
        "--artifacts-dir",
        str(artifacts),
    ]
    if config.get("training_export_target", "auto") != "auto":
        command += [
            "--training-export-target",
            str(config["training_export_target"]),
        ]
    if config["stages"] != "cli-default":
        command += ["--stages", str(config["stages"])]
    if config["initial_splat_cap"] != "profile-default":
        command += ["--initial-splat-cap", str(config["initial_splat_cap"])]
    if config["initial_splat_fraction"] != "profile-default":
        command += [
            "--initial-splat-fraction",
            str(config["initial_splat_fraction"]),
        ]
    if config["mlx_tile_plan_rebuild_interval"] != "profile-default":
        command += [
            "--mlx-tile-plan",
            str(config["mlx_tile_plan"]),
            "--mlx-tile-plan-rebuild-interval",
            str(config["mlx_tile_plan_rebuild_interval"]),
            "--mlx-trainable-groups",
            str(config["mlx_trainable_groups"]),
        ]
    if config["adaptive_compute"]:
        command += [
            "--adaptive-compute",
            "--adaptive-target-ssim-srgb",
            str(config["adaptive_target_ssim_srgb"]),
            "--adaptive-min-checkpoints",
            str(config["adaptive_min_checkpoints"]),
            "--adaptive-chrome-ssim-margin",
            str(config["adaptive_chrome_ssim_margin"]),
            "--adaptive-chrome-psnr-margin",
            str(config["adaptive_chrome_psnr_margin"]),
        ]
    return CorpusRunJob(
        index=index,
        total=total,
        key=run_key(name, fmt, seed, config_hash),
        name=name,
        fmt=fmt,
        seed=seed,
        config=config,
        config_hash=config_hash,
        source=source,
        output=output,
        artifacts=artifacts,
        command=tuple(command),
    )


def _finalize_corpus_job(
    job: CorpusRunJob,
    process: subprocess.CompletedProcess[str],
    elapsed: float,
    *,
    root: Path,
    image_meta: dict,
    run_tag: Optional[str],
    canvas_capture_python: Optional[Path],
    browser_executable: Path,
) -> tuple[dict, str]:
    """Score a completed job and build its single-writer JSONL record."""

    record = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "key": job.key,
        "image": job.name,
        "format": job.fmt,
        "seed": job.seed,
        "splats_requested": job.config["splats"],
        "content_class": image_meta["content_class"],
        "source_size": image_meta["size"],
        "runtime_sec": round(elapsed, 1),
        "returncode": process.returncode,
        "config_hash": job.config_hash,
        "run_config": job.config,
        "output_path": str(job.output.relative_to(root)),
        "artifacts_path": str(job.artifacts.relative_to(root)),
    }
    if run_tag:
        record["run_tag"] = run_tag
    if process.returncode != 0:
        record["error"] = (process.stderr or "")[-400:]
        return record, f"FAILED ({elapsed:.0f}s)"

    record["artifact_bytes"] = (
        job.output.stat().st_size if job.output.exists() else None
    )
    if job.output.is_file():
        record["artifact_sha256"] = hashlib.sha256(job.output.read_bytes()).hexdigest()
    final = job.artifacts / "final.raw.json"
    if final.exists():
        record["splats_final"] = len(json.loads(final.read_text()).get("splats", []))
    if job.fmt == "svg":
        scored = score_svg(job.source, job.output)
    elif job.fmt in {"canvas", "pixel-runtime"}:
        scored = None
        if canvas_capture_python is not None:
            capture_path = job.output.with_name(f"{job.output.stem}_chrome_canvas.png")
            capture_metadata, capture_log = capture_canvas_artifact(
                job.output,
                capture_path,
                capture_python=canvas_capture_python,
                browser_executable=browser_executable,
            )
            if capture_metadata is not None:
                record["canvas_capture_path"] = str(capture_path.relative_to(root))
                record["canvas_capture_bytes"] = capture_path.stat().st_size
                record["browser_render_ms"] = capture_metadata["render_ms"]
                record["browser_render_ms_samples"] = capture_metadata.get(
                    "render_ms_samples", [capture_metadata["render_ms"]]
                )
                record["browser_version"] = capture_metadata["browser"]
                scored = score_canvas_capture(
                    job.source,
                    capture_path,
                    splat_count=int(record.get("splats_final") or 0),
                    artifact_bytes=int(record.get("artifact_bytes") or 0),
                    compositor=str(capture_metadata.get("compositor") or job.fmt),
                )
            else:
                record["canvas_capture_error"] = capture_log[-1000:]
        if scored is None:
            scored = score_canvas(
                job.source, final, job.artifacts / "run_manifest.json"
            )
    else:
        scored = score_pptx_proxy(job.source, final)
    if not scored:
        record["error"] = "scoring-failed"
        return record, f"unscored ({elapsed:.0f}s)"
    record.update(scored)
    return (
        record,
        f"LPIPS {scored['lpips']:.4f}  SSIM {scored['ssim_srgb']:.4f}  "
        f"({elapsed:.0f}s)",
    )


def run(
    root: Path,
    formats: List[str],
    seeds: List[int],
    splats: int,
    stages: Optional[str],
    only: Optional[List[str]],
    run_tag: Optional[str],
    full_geometry: bool,
    profile: str,
    optimizer_backend: str,
    canvas_capture_python: Optional[Path],
    browser_executable: Path,
    initial_splat_cap: Optional[int],
    initial_splat_fraction: Optional[float],
    adaptive_compute: bool,
    adaptive_target_ssim_srgb: float,
    adaptive_min_checkpoints: int,
    adaptive_chrome_ssim_margin: float,
    adaptive_chrome_psnr_margin: float,
    training_export_target: str = "auto",
    jobs: int = 1,
) -> None:
    """Run content-addressed corpus jobs with single-writer result handling."""

    meta = json.loads((root / "corpus.json").read_text())["images"]
    results_path = root / "results.jsonl"
    done = load_done(results_path)
    runs_dir = root / "runs"
    runs_dir.mkdir(exist_ok=True)

    todo = []
    for name in meta:
        if only is not None and name not in only:
            continue
        source = root / meta[name]["path"]
        for fmt in formats:
            for seed in seeds:
                config = _run_config(
                    source,
                    fmt,
                    seed,
                    splats,
                    stages,
                    profile,
                    optimizer_backend,
                    full_geometry,
                    initial_splat_cap,
                    initial_splat_fraction,
                    adaptive_compute,
                    adaptive_target_ssim_srgb,
                    adaptive_min_checkpoints,
                    adaptive_chrome_ssim_margin,
                    adaptive_chrome_psnr_margin,
                    training_export_target,
                )
                config_hash = _config_hash(config)
                if run_key(name, fmt, seed, config_hash) not in done:
                    todo.append((name, fmt, seed, source, config, config_hash))

    worker_count = _resolve_corpus_worker_count(jobs, optimizer_backend)
    print(
        f"{len(todo)} runs to do ({len(done)} valid cached; " f"jobs={worker_count})\n"
    )
    prepared = [
        _build_corpus_job(
            index=index,
            total=len(todo),
            runs_dir=runs_dir,
            source=source,
            name=name,
            fmt=fmt,
            seed=seed,
            config=config,
            config_hash=config_hash,
        )
        for index, (name, fmt, seed, source, config, config_hash) in enumerate(todo, 1)
    ]
    for job in prepared:
        print(
            f"[queue {job.index}/{job.total}] {job.name} {job.fmt} " f"seed={job.seed}"
        )

    for completed, (job, process, elapsed) in enumerate(
        _execute_corpus_jobs(prepared, worker_count), 1
    ):
        record, status = _finalize_corpus_job(
            job,
            process,
            elapsed,
            root=root,
            image_meta=meta[job.name],
            run_tag=run_tag,
            canvas_capture_python=canvas_capture_python,
            browser_executable=browser_executable,
        )
        # Only the coordinator writes JSONL, so records cannot interleave even
        # when converter subprocesses complete simultaneously.
        with results_path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        print(
            f"[done {completed}/{len(prepared)}; job {job.index}] "
            f"{job.name} {job.fmt}: {status}"
        )


def capture_existing_canvas_runs(
    root: Path,
    *,
    capture_python: Path,
    browser_executable: Path,
    only: Optional[List[str]],
    refresh_html: bool,
) -> None:
    """Append browser-pixel-buffer scores for cached canvas HTML runs."""

    results_path = root / "results.jsonl"
    if not results_path.exists():
        raise SystemExit(f"missing benchmark results: {results_path}")
    meta = json.loads((root / "corpus.json").read_text())["images"]
    latest_by_key: dict[str, dict[str, Any]] = {}
    for line in results_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (
            record.get("format") in {"canvas", "pixel-runtime"}
            and record.get("returncode", 0) == 0
            and (only is None or record.get("image") in only)
        ):
            latest_by_key[str(record["key"])] = record

    pending = [
        record
        for record in latest_by_key.values()
        if refresh_html
        or record.get("render_kind") not in DEPLOYED_CANVAS_CAPTURE_KINDS
        or not record.get("canvas_capture_path")
        or not (root / record["canvas_capture_path"]).exists()
    ]
    print(f"{len(pending)} cached canvas runs need browser capture")
    for index, record in enumerate(pending, 1):
        name = str(record["image"])
        recorded_output = record.get("output_path")
        html_path = (
            root / recorded_output
            if recorded_output
            else root / "runs" / f"{name}_{record.get('format', 'canvas')}_s0.html"
        )
        if not html_path.exists():
            print(
                f"[canvas capture {index}/{len(pending)}] {name}: "
                f"missing {html_path}"
            )
            continue
        if refresh_html:
            from splatthis.io import generate_native_canvas_html, load_splats_json
            from splatthis.pixel_runtime import generate_pixel_runtime_html

            recorded_artifacts = record.get("artifacts_path")
            artifact_dir = (
                root / recorded_artifacts
                if recorded_artifacts
                else root / "runs" / f"{name}_canvas_s0_art"
            )
            raw_path = artifact_dir / "final.raw.json"
            manifest_path = artifact_dir / "run_manifest.json"
            if not raw_path.exists() or not manifest_path.exists():
                print(
                    f"[canvas capture {index}/{len(pending)}] {name}: "
                    "missing splats or manifest"
                )
                continue
            manifest = json.loads(manifest_path.read_text())
            config = manifest.get("config", {})
            width, height = config.get("resolved_target_size", meta[name]["size"])
            training_target = str(config.get("training_export_target", "canvas"))
            splats = load_splats_json(str(raw_path))
            background = np.asarray(
                config.get("background_linear_rgb", [0.0, 0.0, 0.0]),
                dtype=np.float32,
            )
            render_kind = str(
                manifest.get("artifact_evaluation", {}).get("render_kind", "")
            )
            if render_kind.startswith("canvas-api-"):
                refreshed_html = generate_native_canvas_html(
                    splats,
                    int(width),
                    int(height),
                    background_linear_rgb=background,
                    title=f"{name} · SplatThis native canvas",
                )
            else:
                refreshed_html = generate_pixel_runtime_html(
                    splats,
                    int(width),
                    int(height),
                    background_linear_rgb=background,
                    title=f"{name} · SplatThis pixel runtime",
                    compositing_space=(
                        "srgb"
                        if training_target in {"svg", "pptx-softedge"}
                        else "linear"
                    ),
                )
            html_path.write_text(refreshed_html)
            record["artifact_bytes"] = html_path.stat().st_size
        capture_path = html_path.with_name(f"{html_path.stem}_chrome_canvas.png")
        print(
            f"[canvas capture {index}/{len(pending)}] {name} "
            f"n={record.get('splats_requested')} ... ",
            end="",
            flush=True,
        )
        metadata, capture_log = capture_canvas_artifact(
            html_path,
            capture_path,
            capture_python=capture_python,
            browser_executable=browser_executable,
        )
        if metadata is None:
            print("FAILED")
            continue
        scored = score_canvas_capture(
            root / meta[name]["path"],
            capture_path,
            splat_count=int(record.get("splats_final") or 0),
            artifact_bytes=int(record.get("artifact_bytes") or 0),
            compositor=str(metadata.get("compositor") or "pixel-runtime"),
        )
        if scored is None:
            print("UNSCORED")
            continue
        enriched = {
            **record,
            **scored,
            "canvas_capture_path": str(capture_path.relative_to(root)),
            "canvas_capture_bytes": capture_path.stat().st_size,
            "browser_render_ms": metadata["render_ms"],
            "browser_render_ms_samples": metadata.get(
                "render_ms_samples", [metadata["render_ms"]]
            ),
            "browser_version": metadata["browser"],
            "canvas_capture_log": capture_log[-1000:],
        }
        with results_path.open("a") as output:
            output.write(json.dumps(enriched) + "\n")
        print(
            f"LPIPS {enriched['lpips']:.4f}  "
            f"SSIM {enriched['ssim_srgb']:.4f}  "
            f"browser {enriched['browser_render_ms']:.0f}ms"
        )


def summarize(root: Path) -> None:
    results_path = root / "results.jsonl"
    if not results_path.exists():
        print("no results yet")
        return
    recs = [json.loads(x) for x in results_path.read_text().splitlines() if x.strip()]
    ok = [r for r in recs if r.get("lpips") is not None]
    print(f"{len(ok)}/{len(recs)} runs scored\n")

    # Per-format summary.
    print(
        f"{'format':<8}{'n':>4}{'LPIPS med':>11}{'SSIM med':>10}{'KB med':>9}{'sec med':>9}"
    )
    for fmt in sorted({r["format"] for r in ok}):
        g = [r for r in ok if r["format"] == fmt]
        print(
            f"{fmt:<8}{len(g):>4}{statistics.median(r['lpips'] for r in g):>11.4f}"
            f"{statistics.median(r['ssim_srgb'] for r in g):>10.4f}"
            f"{statistics.median((r.get('artifact_bytes') or 0)/1024 for r in g):>9.0f}"
            f"{statistics.median(r['runtime_sec'] for r in g):>9.0f}"
        )

    # Latest actual-browser canvas budget curve, with effective-4k identity.
    canvas_budget: Dict[tuple[str, int], dict] = {}
    for record in ok:
        budget = record.get("splats_requested")
        if (
            record.get("format") not in {"canvas", "pixel-runtime"}
            or record.get("seed") != 0
            or record.get("render_kind") not in PIXEL_RUNTIME_CAPTURE_KINDS
            or budget not in {2000, 4000}
        ):
            continue
        if (
            budget == 4000
            and record.get("run_config", {}).get("initial_splat_cap") != 4000
        ):
            continue
        canvas_budget[(str(record["image"]), int(budget))] = record
    paired_names = sorted(
        name
        for name, budget in canvas_budget
        if budget == 2000 and (name, 4000) in canvas_budget
    )
    if paired_names:
        before = [canvas_budget[(name, 2000)] for name in paired_names]
        after = [canvas_budget[(name, 4000)] for name in paired_names]
        delta_ssim = [
            new["ssim_srgb"] - old["ssim_srgb"] for old, new in zip(before, after)
        ]
        delta_lpips = [new["lpips"] - old["lpips"] for old, new in zip(before, after)]
        print(f"\nactual Chrome canvas 2k -> effective 4k (n={len(paired_names)}):")
        print(
            "  median SSIM  "
            f"{statistics.median(r['ssim_srgb'] for r in before):.4f} -> "
            f"{statistics.median(r['ssim_srgb'] for r in after):.4f}  "
            f"(paired delta {statistics.median(delta_ssim):+.4f})"
        )
        print(
            "  median LPIPS "
            f"{statistics.median(r['lpips'] for r in before):.4f} -> "
            f"{statistics.median(r['lpips'] for r in after):.4f}  "
            f"(paired delta {statistics.median(delta_lpips):+.4f})"
        )
        print(
            "  improved     "
            f"SSIM {sum(value > 0 for value in delta_ssim)}/{len(paired_names)}, "
            f"LPIPS {sum(value < 0 for value in delta_lpips)}/{len(paired_names)}"
        )
        print(
            "  median cost   "
            f"{statistics.median(r['splats_final'] for r in after):.0f} splats, "
            f"{statistics.median(r['artifact_bytes'] / 1024 for r in after):.0f} KB, "
            f"{statistics.median(r['browser_render_ms'] for r in after):.0f} ms browser, "
            f"{statistics.median(r['runtime_sec'] / 60 for r in after):.1f} min training"
        )
        print(
            "  thresholds    "
            f"SSIM >= .90 {sum(r['ssim_srgb'] >= .90 for r in after)}/{len(after)}, "
            f">= .95 {sum(r['ssim_srgb'] >= .95 for r in after)}/{len(after)}, "
            f">= .99 {sum(r['ssim_srgb'] >= .99 for r in after)}/{len(after)}"
        )

    # Per-content-class, SVG only.
    print(f"\n{'content class':<18}{'n':>4}{'LPIPS med':>11}{'best':>9}{'worst':>9}")
    svg = [r for r in ok if r["format"] == "svg"]
    for cls in sorted({r["content_class"] for r in svg}):
        g = [r for r in svg if r["content_class"] == cls]
        vals = [r["lpips"] for r in g]
        print(
            f"{cls:<18}{len(g):>4}{statistics.median(vals):>11.4f}"
            f"{min(vals):>9.4f}{max(vals):>9.4f}"
        )

    # JPEG-at-matched-bytes baseline, if computed.
    bpath = root / "baselines.jsonl"
    if bpath.exists():
        b = [json.loads(x) for x in bpath.read_text().splitlines() if x.strip()]
        if b:
            gaps = [r["lpips_gap"] for r in b]
            headroom = [
                100.0 * (1 - r["jpeg"]["bytes"] / max(r["svg_bytes"], 1)) for r in b
            ]
            print(f"\nJPEG at matched bytes (n={len(b)}):")
            print(
                f"  svg  LPIPS median {statistics.median(r['svg_lpips'] for r in b):.4f}"
            )
            print(
                f"  jpeg LPIPS median {statistics.median(r['jpeg']['lpips'] for r in b):.4f}"
            )
            print(
                f"  gap  median {statistics.median(gaps):+.4f}  "
                f"(svg better in {sum(1 for g in gaps if g < 0)}/{len(gaps)} images)"
            )
            print(
                f"  jpeg used {statistics.median(headroom):.0f}% less than the byte "
                f"budget at median"
            )

    # Per-image detail, for the appendix.
    print(f"\n{'image':<22}{'class':<17}{'LPIPS':>8}{'SSIM':>8}{'KB':>7}{'sec':>6}")
    for r in sorted(svg, key=lambda x: x["lpips"]):
        print(
            f"{r['image']:<22}{r['content_class']:<17}{r['lpips']:>8.4f}"
            f"{r['ssim_srgb']:>8.4f}{(r.get('artifact_bytes') or 0)/1024:>7.0f}"
            f"{r['runtime_sec']:>6.0f}"
        )

    # Seed noise floor.
    print("\nseed variance (same image+format, >1 seed):")
    groups: Dict[str, List[float]] = {}
    for r in ok:
        groups.setdefault(f"{r['image']}|{r['format']}", []).append(r["lpips"])
    multi = {k: v for k, v in groups.items() if len(v) > 1}
    if not multi:
        print("  (single seed only — rerun with --seeds 0,1,2 for a noise floor)")
    else:
        spreads = [max(v) - min(v) for v in multi.values()]
        print(
            f"  {len(multi)} groups; LPIPS spread median {statistics.median(spreads):.4f}"
            f"  max {max(spreads):.4f}"
        )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--materialize", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument(
        "--capture-canvas-runs",
        action="store_true",
        help="capture and rescore cached canvas HTML through a real browser",
    )
    ap.add_argument(
        "--refresh-canvas-html",
        action="store_true",
        help="regenerate cached canvas HTML from final splats before capture",
    )
    ap.add_argument(
        "--html",
        nargs="?",
        const="",
        default=None,
        metavar="PATH",
        help="write the self-contained live-canvas/SVG corpus (default: ROOT/index.html)",
    )
    ap.add_argument(
        "--baselines",
        action="store_true",
        help="JPEG at matched bytes for each scored SVG run",
    )
    ap.add_argument(
        "--score-powerpoint",
        action="store_true",
        help="score *_pptx_s0_powerpoint_slide.png captures from real PowerPoint",
    )
    ap.add_argument("--formats", default="svg")
    ap.add_argument(
        "--training-export-target",
        default="auto",
        help="Training compositor passed through to splatthis; part of the "
        "run identity, so targets resume as separate jobs",
    )
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--splats", type=int, default=2000)
    ap.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Torch/CPU converter subprocesses to run concurrently (default: 1; "
        "MLX requires 1 for seeded reproducibility)",
    )
    ap.add_argument("--stages", default=None, help="e.g. 60,40,25 to shorten runs")
    ap.add_argument("--only", default=None, help="comma-separated image names")
    ap.add_argument(
        "--run-tag",
        default=None,
        help="configuration label included in resumability keys",
    )
    ap.add_argument(
        "--full-geometry",
        action="store_true",
        help="legacy explicit override for periodic full-geometry MLX optimization",
    )
    ap.add_argument(
        "--profile",
        default="max-fidelity",
        help="converter quality profile (default: max-fidelity)",
    )
    ap.add_argument(
        "--optimizer-backend",
        default="mlx",
        choices=["mlx", "torch"],
        help="converter optimizer backend (default: mlx)",
    )
    ap.add_argument(
        "--initial-splat-cap",
        type=int,
        default=None,
        help="forward an explicit initial population cap to the converter; "
        "use the requested budget for a genuine high-budget scaling run",
    )
    ap.add_argument(
        "--initial-splat-fraction",
        type=float,
        default=None,
        help="forward the initial population fraction to the converter",
    )
    ap.add_argument(
        "--adaptive-compute",
        action="store_true",
        help="enable the online Canvas quality-target controller",
    )
    ap.add_argument(
        "--adaptive-target-ssim-srgb",
        type=float,
        default=0.98,
        help="deployed Canvas SSIM target for adaptive runs (default: 0.98)",
    )
    ap.add_argument(
        "--adaptive-min-checkpoints",
        type=int,
        default=2,
        help="minimum completed stages before adaptive stopping (default: 2)",
    )
    ap.add_argument(
        "--adaptive-chrome-ssim-margin",
        type=float,
        default=DEFAULT_CHROME_SSIM_SAFETY_MARGIN,
        help="cross-version SSIM safety-margin override for adaptive runs "
        f"(default: {DEFAULT_CHROME_SSIM_SAFETY_MARGIN:g})",
    )
    ap.add_argument(
        "--adaptive-chrome-psnr-margin",
        type=float,
        default=DEFAULT_CHROME_PSNR_SAFETY_MARGIN,
        help="cross-version PSNR safety-margin override in dB for adaptive runs "
        f"(default: {DEFAULT_CHROME_PSNR_SAFETY_MARGIN:g})",
    )
    ap.add_argument(
        "--canvas-capture-python",
        type=Path,
        default=None,
        help="Python interpreter containing Playwright; when set, new canvas "
        "runs are scored from Chrome's actual canvas pixel buffer",
    )
    ap.add_argument(
        "--browser-executable",
        type=Path,
        default=Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
        help="browser executable used for canvas artifact capture",
    )
    args = ap.parse_args()
    if args.jobs < 1:
        ap.error("--jobs must be positive")
    if args.run and args.optimizer_backend == "mlx" and args.jobs > 1:
        ap.error("--jobs greater than 1 is unsafe with MLX; use Torch or --jobs 1")
    if not 0.0 <= args.adaptive_target_ssim_srgb <= 1.0:
        ap.error("--adaptive-target-ssim-srgb must be between 0 and 1")
    if args.adaptive_min_checkpoints < 1:
        ap.error("--adaptive-min-checkpoints must be positive")
    if args.adaptive_chrome_ssim_margin < 0.0:
        ap.error("--adaptive-chrome-ssim-margin must be non-negative")
    if args.adaptive_chrome_psnr_margin < 0.0:
        ap.error("--adaptive-chrome-psnr-margin must be non-negative")
    if (
        args.adaptive_compute
        and args.adaptive_target_ssim_srgb + args.adaptive_chrome_ssim_margin > 1.0
    ):
        ap.error("adaptive SSIM target plus Chrome safety margin must not exceed 1")
    requested_formats = [
        value.strip() for value in args.formats.split(",") if value.strip()
    ]
    if (
        args.run
        and args.adaptive_compute
        and any(value != "pixel-runtime" for value in requested_formats)
    ):
        ap.error("--adaptive-compute currently supports only --formats pixel-runtime")

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    if args.materialize:
        print("materializing corpus:")
        materialize(root)
    if args.run:
        run(
            root,
            formats=requested_formats,
            seeds=[int(s) for s in args.seeds.split(",") if s.strip()],
            splats=args.splats,
            stages=args.stages,
            only=[o.strip() for o in args.only.split(",")] if args.only else None,
            run_tag=args.run_tag,
            full_geometry=args.full_geometry,
            profile=args.profile,
            optimizer_backend=args.optimizer_backend,
            canvas_capture_python=args.canvas_capture_python,
            browser_executable=args.browser_executable,
            initial_splat_cap=args.initial_splat_cap,
            initial_splat_fraction=args.initial_splat_fraction,
            adaptive_compute=args.adaptive_compute,
            adaptive_target_ssim_srgb=args.adaptive_target_ssim_srgb,
            adaptive_min_checkpoints=args.adaptive_min_checkpoints,
            adaptive_chrome_ssim_margin=args.adaptive_chrome_ssim_margin,
            adaptive_chrome_psnr_margin=args.adaptive_chrome_psnr_margin,
            training_export_target=args.training_export_target,
            jobs=args.jobs,
        )
    if args.capture_canvas_runs:
        if args.canvas_capture_python is None:
            raise SystemExit("--capture-canvas-runs requires --canvas-capture-python")
        capture_existing_canvas_runs(
            root,
            capture_python=args.canvas_capture_python,
            browser_executable=args.browser_executable,
            only=[o.strip() for o in args.only.split(",")] if args.only else None,
            refresh_html=args.refresh_canvas_html,
        )
    if args.baselines:
        run_baselines(root)
    if args.score_powerpoint:
        score_powerpoint_captures(root)
    if args.summarize:
        summarize(root)
    if args.html is not None:
        output = Path(args.html) if args.html else root / "index.html"
        generate_canvas_corpus_html(root, output)
    if not (
        args.materialize
        or args.run
        or args.summarize
        or args.capture_canvas_runs
        or args.baselines
        or args.score_powerpoint
        or args.html is not None
    ):
        ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
