#!/usr/bin/env python3
"""Reference-corpus benchmark (ADR-003 Phase 0).

Materializes a fixed corpus of standard, redistributable test images, runs the
conversion pipeline over it per export format, and scores every run on the
**deployed artifact** — the emitted SVG put through a real rasterizer, never
the internal renderer.

Everything is content-addressed and resumable: each run writes one JSONL
record keyed by (image, format, seed, config), and re-invocations skip records
that already exist. Source images come from ``skimage.data`` so the corpus is
reproducible from a fresh checkout with no network and no licensing questions.

Usage
-----
    python tools/corpus_benchmark.py --materialize          # write corpus/
    python tools/corpus_benchmark.py --run --formats svg    # score SVG
    python tools/corpus_benchmark.py --run --formats svg,pptx --seeds 0,1,2
    python tools/corpus_benchmark.py --summarize            # tables
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

DEFAULT_ROOT = REPO / "result" / "corpus"
MAX_EDGE = 384  # keeps a 20-image x 2-format sweep tractable; recorded in meta


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


def score_svg(source_png: Path, svg_path: Path) -> Optional[dict]:
    """Metrics on the actually-rasterized SVG."""
    from png2svg_gs.io import (
        _try_rasterize_svg_to_linear_rgb,
        compute_quality_metrics,
        linear_to_srgb,
        load_png,
    )

    target_lin = load_png(str(source_png))[..., :3]
    h, w = target_lin.shape[:2]
    rendered_lin, method = _try_rasterize_svg_to_linear_rgb(str(svg_path), w, h)
    if rendered_lin is None:
        return None
    m = compute_quality_metrics(target_lin, rendered_lin)
    lp = _lpips_score(linear_to_srgb(target_lin), linear_to_srgb(rendered_lin))
    return {
        "renderer": method,
        "lpips": round(lp, 4),
        "ssim_srgb": round(float(m["ssim_srgb"]), 4),
        "psnr_srgb": round(float(m["psnr_srgb"]), 3),
    }


def score_pptx_proxy(source_png: Path, splats_json: Path) -> Optional[dict]:
    """PPTX scored through the calibrated soft-edge proxy.

    Real PowerPoint is not scriptable at corpus scale, so the proxy is used
    here and validated against genuine PowerPoint captures on a subset (see
    result/README.md). Reported separately from SVG for that reason.
    """
    from png2svg_gs.io import (
        compute_quality_metrics,
        linear_to_srgb,
        load_png,
        load_splats_json,
    )
    from png2svg_gs.renderer import render_splats_numpy

    if not splats_json.exists():
        return None
    target_lin = load_png(str(source_png))[..., :3]
    h, w = target_lin.shape[:2]
    splats = load_splats_json(str(splats_json))
    rendered = render_splats_numpy(splats, width=w, height=h, compositing_space="srgb")
    m = compute_quality_metrics(target_lin, rendered)
    lp = _lpips_score(linear_to_srgb(target_lin), linear_to_srgb(rendered))
    return {
        "renderer": "proxy-srgb",
        "lpips": round(lp, 4),
        "ssim_srgb": round(float(m["ssim_srgb"]), 4),
        "psnr_srgb": round(float(m["psnr_srgb"]), 3),
    }


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------


def run_key(image: str, fmt: str, seed: int, splats: int) -> str:
    return f"{image}|{fmt}|seed{seed}|n{splats}"


def load_done(results_path: Path) -> set:
    if not results_path.exists():
        return set()
    done = set()
    for line in results_path.read_text().splitlines():
        if line.strip():
            try:
                done.add(json.loads(line)["key"])
            except Exception:
                pass
    return done


def run(
    root: Path,
    formats: List[str],
    seeds: List[int],
    splats: int,
    stages: Optional[str],
    only: Optional[List[str]],
) -> None:
    meta = json.loads((root / "corpus.json").read_text())["images"]
    results_path = root / "results.jsonl"
    done = load_done(results_path)
    runs_dir = root / "runs"
    runs_dir.mkdir(exist_ok=True)

    todo = [
        (n, f, s)
        for n in meta
        for f in formats
        for s in seeds
        if (only is None or n in only) and run_key(n, f, s, splats) not in done
    ]
    print(f"{len(todo)} runs to do ({len(done)} already recorded)\n")

    for idx, (name, fmt, seed) in enumerate(todo, 1):
        key = run_key(name, fmt, seed, splats)
        src = root / meta[name]["path"]
        stem = runs_dir / f"{name}_{fmt}_s{seed}"
        out = stem.with_suffix(".svg" if fmt == "svg" else ".pptx")
        artifacts = Path(str(stem) + "_art")

        cmd = [
            "splatlify",
            str(src),
            "-o",
            str(out),
            "--seed",
            str(seed),
            "--splats",
            str(splats),
            "--format",
            fmt,
            "--artifacts-dir",
            str(artifacts),
        ]
        if stages:
            cmd += ["--stages", stages]

        print(f"[{idx}/{len(todo)}] {name} {fmt} seed={seed} ... ", end="", flush=True)
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, capture_output=True, text=True)
        elapsed = time.perf_counter() - t0

        rec = {
            "key": key,
            "image": name,
            "format": fmt,
            "seed": seed,
            "splats_requested": splats,
            "content_class": meta[name]["content_class"],
            "source_size": meta[name]["size"],
            "runtime_sec": round(elapsed, 1),
            "returncode": proc.returncode,
        }
        if proc.returncode != 0:
            rec["error"] = (proc.stderr or "")[-400:]
            print(f"FAILED ({elapsed:.0f}s)")
        else:
            rec["artifact_bytes"] = out.stat().st_size if out.exists() else None
            final = artifacts / "final.raw.json"
            if final.exists():
                rec["splats_final"] = len(
                    json.loads(final.read_text()).get("splats", [])
                )
            scored = (
                score_svg(src, out) if fmt == "svg" else score_pptx_proxy(src, final)
            )
            if scored:
                rec.update(scored)
                print(
                    f"LPIPS {scored['lpips']:.4f}  SSIM {scored['ssim_srgb']:.4f}  ({elapsed:.0f}s)"
                )
            else:
                rec["error"] = "scoring-failed"
                print(f"unscored ({elapsed:.0f}s)")

        with results_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")


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
    ap.add_argument("--formats", default="svg")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--splats", type=int, default=2000)
    ap.add_argument("--stages", default=None, help="e.g. 60,40,25 to shorten runs")
    ap.add_argument("--only", default=None, help="comma-separated image names")
    args = ap.parse_args()

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    if args.materialize:
        print("materializing corpus:")
        materialize(root)
    if args.run:
        run(
            root,
            formats=[f.strip() for f in args.formats.split(",") if f.strip()],
            seeds=[int(s) for s in args.seeds.split(",") if s.strip()],
            splats=args.splats,
            stages=args.stages,
            only=[o.strip() for o in args.only.split(",")] if args.only else None,
        )
    if args.summarize:
        summarize(root)
    if not (args.materialize or args.run or args.summarize):
        ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
