#!/usr/bin/env python3
"""Compare production SVG painter-order and gradient policies in Chromium."""

from __future__ import annotations

import argparse
import gzip
import html
import json
import os
from pathlib import Path
from typing import Sequence

import numpy as np

from splatthis.browser_capture import PlaywrightSvgRenderer, resolve_browser_executable
from splatthis.fidelity.metrics import compute_fidelity_metrics
from splatthis.io import (
    _sort_splats_for_export,
    atomic_write_text,
    generate_svg_content,
    load_png,
    load_splats_json,
)


def _write_overview(
    output_dir: Path,
    *,
    source: Path,
    results: Sequence[dict[str, object]],
    width: int,
    height: int,
) -> None:
    baseline = next(result for result in results if result["name"] == "legacy-order")
    cards = [
        f'<figure><img src="{html.escape(os.path.relpath(source, output_dir))}">'
        "<figcaption>Source</figcaption></figure>"
    ]
    for result in results:
        name = str(result["name"])
        cards.append(
            f'<figure><img src="{html.escape(name)}.png">'
            f"<figcaption>{html.escape(name)}"
            f"<small>SSIM {float(result['ssim_srgb']):.6f} · "
            f"Δ {float(result['ssim_srgb']) - float(baseline['ssim_srgb']):+.6f} · "
            f"LPIPS {float(result['lpips']):.6f} · "
            f"Δ {float(result['lpips']) - float(baseline['lpips']):+.6f}<br>"
            f"raw {int(result['file_size_bytes']) / 1000.0:.0f} KB · "
            f"gzip {int(result['gzip_size_bytes']) / 1000.0:.0f} KB · "
            f"capture {float(result['capture_time_ms']):.1f} ms</small>"
            f'<a href="{html.escape(name)}.svg">Open live SVG</a>'
            "</figcaption></figure>"
        )
    document = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>SVG compositor MVP</title>
<style>
:root{{color-scheme:dark;font:15px system-ui,sans-serif}}
body{{margin:0;padding:24px;background:#111;color:#eee}}
h1,.grid{{max-width:1600px;margin:0 auto 18px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:18px}}
figure{{margin:0;padding:12px;background:#1c1c1c;border:1px solid #333;border-radius:10px}}
img{{display:block;width:100%;max-width:{width}px;aspect-ratio:{width}/{height};margin:auto}}
figcaption{{margin-top:10px;font-weight:650}}small{{display:block;color:#aaa;font-weight:400}}
a{{color:#8cc8ff}}
</style></head><body><h1>SVG compositor MVP</h1>
<div class="grid">{''.join(cards)}</div></body></html>
"""
    atomic_write_text(output_dir / "index.html", document)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--splats-json", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--browser-executable", type=Path, default=None)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout-ms", type=int, default=120_000)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    return args


def main() -> int:
    args = _parse_args()
    source = args.source.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.resolve().read_text())
    background = np.asarray(
        manifest["config"]["background_linear_rgb"], dtype=np.float32
    )
    splats = _sort_splats_for_export(load_splats_json(str(args.splats_json.resolve())))
    target = load_png(str(source))[:, :, :3]
    height, width = target.shape[:2]
    k_sigma = float(manifest["config"].get("k_sigma", 2.5))
    corrected_standard = generate_svg_content(
        splats,
        width=width,
        height=height,
        k_sigma=k_sigma,
        background_linear_rgb=background,
    )
    legacy_order = generate_svg_content(
        splats,
        width=width,
        height=height,
        k_sigma=k_sigma,
        background_linear_rgb=background,
        painter_order="legacy",
    )
    corrected_high = generate_svg_content(
        splats,
        width=width,
        height=height,
        k_sigma=k_sigma,
        background_linear_rgb=background,
        gradient_quality="high",
    )
    variants = {
        "legacy-order": legacy_order,
        "corrected-standard": corrected_standard,
        "corrected-high": corrected_high,
    }
    for name, content in variants.items():
        atomic_write_text(output_dir / f"{name}.svg", content)

    results: list[dict[str, object]] = []
    executable = resolve_browser_executable(args.browser_executable)
    with PlaywrightSvgRenderer(
        browser_executable=executable, timeout_ms=args.timeout_ms
    ) as renderer:
        for name in variants:
            artifact = output_dir / f"{name}.svg"
            output = output_dir / f"{name}.png"
            capture = renderer.capture(
                artifact,
                output,
                width=width,
                height=height,
                repeats=args.repeats,
            )
            rendered = renderer.render_linear_rgb(artifact, width=width, height=height)
            metrics = compute_fidelity_metrics(
                target,
                rendered,
                splat_count=len(splats),
                file_size_bytes=artifact.stat().st_size,
                render_method=name,
            ).as_dict()
            results.append(
                {
                    "name": name,
                    "capture_time_ms": capture.capture_time_ms,
                    "pixel_stable": capture.pixel_stable,
                    "gzip_size_bytes": len(
                        gzip.compress(artifact.read_bytes(), compresslevel=9)
                    ),
                    **metrics,
                }
            )
        browser = renderer.browser_version

    report = {
        "schema": "splatthis.svg-order-compositor-mvp/2",
        "source": str(source),
        "splats_json": str(args.splats_json.resolve()),
        "manifest": str(args.manifest.resolve()),
        "browser": browser,
        "width": width,
        "height": height,
        "splat_count": len(splats),
        "results": results,
    }
    atomic_write_text(output_dir / "results.json", json.dumps(report, indent=2))
    _write_overview(
        output_dir,
        source=source,
        results=results,
        width=width,
        height=height,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
