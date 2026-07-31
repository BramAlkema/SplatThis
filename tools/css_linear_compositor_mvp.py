#!/usr/bin/env python3
"""Reproduce the scriptless CSS linear-compositor MVP in Chromium."""

from __future__ import annotations

import argparse
import html
import io
import json
import math
import os
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

from png2svg_gs.browser_capture import resolve_browser_executable
from png2svg_gs.fidelity.metrics import compute_fidelity_metrics
from png2svg_gs.io import (
    ELLIPSE_OVERLAP_BOOST,
    MIN_ELLIPSE_RADIUS_PX,
    _adaptive_gradient_stops,
    _density_aware_stop_error,
    _gaussian_opacity_curve,
    _sort_splats_for_export,
    atomic_output_path,
    atomic_write_text,
    linear_to_srgb,
    load_png,
    load_splats_json,
    srgb_to_linear,
)
from png2svg_gs.splat import GaussianSplat


@dataclass(frozen=True)
class CssVariant:
    name: str
    reverse_order: bool
    linear_color: bool
    linear_interpolation: bool
    alpha_mask: bool
    exact_stops: int = 0
    retain_data_indices: bool = True


VARIANTS = (
    CssVariant("baseline", False, False, False, False),
    CssVariant("reverse-linear", True, True, True, False),
    CssVariant("reverse-linear-mask", True, True, False, True),
    CssVariant(
        "reverse-linear-mask-exact9",
        True,
        True,
        False,
        True,
        exact_stops=9,
        retain_data_indices=False,
    ),
)


def _display_color(rgb: np.ndarray, alpha: float | None = None) -> str:
    srgb = linear_to_srgb(np.clip(np.asarray(rgb), 0.0, 1.0))
    channels = [int(np.clip(np.round(value * 255.0), 0, 255)) for value in srgb]
    if alpha is None:
        return f"rgb({channels[0]},{channels[1]},{channels[2]})"
    return f"rgba({channels[0]},{channels[1]},{channels[2]},{alpha:.4f})"


def _linear_color(rgb: np.ndarray, alpha: float | None = None) -> str:
    values = np.clip(np.asarray(rgb, dtype=np.float32), 0.0, 1.0)
    alpha_part = "" if alpha is None else f" / {alpha:.4f}"
    return (
        f"color(srgb-linear {float(values[0]):.6f} "
        f"{float(values[1]):.6f} {float(values[2]):.6f}{alpha_part})"
    )


def _variant_stops(
    splat: GaussianSplat,
    variant: CssVariant,
    *,
    footprint: float,
    splat_count: int,
) -> list[tuple[float, float]]:
    alpha = float(np.clip(splat.alpha, 0.0, 1.0))
    if variant.exact_stops:
        offsets = np.linspace(0.0, 1.0, variant.exact_stops)
        opacities = _gaussian_opacity_curve(offsets, alpha, footprint)
        return [
            (float(offset), float(opacity))
            for offset, opacity in zip(offsets, opacities)
        ]
    return _adaptive_gradient_stops(
        alpha,
        footprint,
        1.0,
        max_error=_density_aware_stop_error(splat_count),
    )


def _splat_element(
    splat: GaussianSplat,
    index: int,
    variant: CssVariant,
    *,
    footprint: float,
    splat_count: int,
) -> str:
    eigenvalues, eigenvectors = splat.eigendecomposition()
    radius_x = max(
        MIN_ELLIPSE_RADIUS_PX,
        footprint * math.sqrt(max(float(eigenvalues[0]), 1e-8)),
    )
    radius_y = max(
        MIN_ELLIPSE_RADIUS_PX,
        footprint * math.sqrt(max(float(eigenvalues[1]), 1e-8)),
    )
    rotation = math.degrees(
        math.atan2(float(eigenvectors[1, 0]), float(eigenvectors[0, 0]))
    )
    color = np.asarray(splat.color[:3], dtype=np.float32)
    stops = _variant_stops(splat, variant, footprint=footprint, splat_count=splat_count)
    geometry = (
        f"left:{float(splat.mu[0]):.2f}px;top:{float(splat.mu[1]):.2f}px;"
        f"width:{2.0 * radius_x:.2f}px;height:{2.0 * radius_y:.2f}px;"
        f"transform:translate(-50%,-50%) rotate({rotation:.2f}deg);"
    )
    interpolation = " in srgb-linear" if variant.linear_interpolation else ""
    color_function = _linear_color if variant.linear_color else _display_color
    if variant.alpha_mask:
        mask_stops = ",".join(
            f"rgba(0,0,0,{opacity:.4f}) {offset * 100.0:.2f}%"
            for offset, opacity in stops
        )
        paint = (
            f"background:{color_function(color)};"
            "mask-image:radial-gradient(ellipse 50% 50% at center"
            f"{interpolation},{mask_stops});"
            "mask-mode:alpha;mask-repeat:no-repeat"
        )
    else:
        gradient_stops = ",".join(
            f"{color_function(color, opacity)} {offset * 100.0:.2f}%"
            for offset, opacity in stops
        )
        paint = (
            "background:radial-gradient(ellipse 50% 50% at center"
            f"{interpolation},{gradient_stops})"
        )
    data_index = f' data-splat="{index}"' if variant.retain_data_indices else ""
    return f'<i class="splat"{data_index} style="{geometry}{paint}"></i>'


def generate_css_variant(
    splats: Sequence[GaussianSplat],
    *,
    width: int,
    height: int,
    background_linear_rgb: np.ndarray,
    variant: CssVariant,
    k_sigma: float = 2.5,
) -> str:
    """Generate one scriptless CSS compositor candidate."""

    ordered = _sort_splats_for_export(list(splats))
    if variant.reverse_order:
        ordered.reverse()
    footprint = ELLIPSE_OVERLAP_BOOST * float(k_sigma)
    elements = [
        _splat_element(
            splat,
            index,
            variant,
            footprint=footprint,
            splat_count=len(ordered),
        )
        for index, splat in enumerate(ordered)
    ]
    background = _display_color(background_linear_rgb)
    css = (
        "*{box-sizing:border-box}"
        f"html,body{{margin:0;width:{width}px;height:{height}px;overflow:hidden}}"
        f"#scene{{position:relative;width:{width}px;height:{height}px;"
        f"overflow:hidden;background:{background};isolation:isolate}}"
        ".splat{position:absolute;display:block;border-radius:50%;"
        "pointer-events:none;transform-origin:center;mix-blend-mode:normal}"
    )
    return (
        '<!doctype html><html><head><meta charset="utf-8">'
        f"<title>{html.escape(variant.name)}</title><style>{css}</style></head>"
        '<body><main id="scene" data-compositor="css-linear-mvp">'
        + "".join(elements)
        + "</main></body></html>\n"
    )


def _decode_png_linear(payload: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(payload)) as image:
        srgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return srgb_to_linear(srgb)


def _capture_variant(
    page: Any,
    path: Path,
    *,
    repeats: int,
    timeout_ms: int,
) -> tuple[bytes, list[float]]:
    uri = path.resolve().as_uri()
    settle_samples: list[float] = []
    payload = b""
    for repeat in range(repeats + 1):
        page.goto(uri, wait_until="load", timeout=timeout_ms)
        settle_ms = page.evaluate(
            """async () => {
                if (document.fonts) await document.fonts.ready;
                await new Promise(resolve => requestAnimationFrame(
                    () => requestAnimationFrame(resolve)));
                return performance.now();
            }"""
        )
        payload = page.locator("#scene").screenshot(
            type="png", animations="disabled", scale="css"
        )
        if repeat:
            settle_samples.append(float(settle_ms))
    return payload, settle_samples


def _write_overview(
    output_dir: Path,
    *,
    source_path: Path,
    results: Sequence[dict[str, Any]],
    width: int,
    height: int,
) -> None:
    source_relative = os.path.relpath(source_path, output_dir)
    cards = [
        f'<figure><img src="{html.escape(source_relative)}">'
        "<figcaption>Source</figcaption></figure>"
    ]
    for result in results:
        name = str(result["name"])
        cards.append(
            f'<figure><img src="{html.escape(name)}.png">'
            f"<figcaption>{html.escape(name)}"
            f"<small>SSIM {float(result['ssim_srgb']):.6f} · "
            f"LPIPS {float(result['lpips']):.6f} · "
            f"{int(result['file_size_bytes']) / 1000.0:.0f} KB</small>"
            f'<a href="{html.escape(name)}.html">Open live CSS</a>'
            "</figcaption></figure>"
        )
    document = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>CSS linear compositor MVP</title>
<style>
:root{{color-scheme:dark;font:15px system-ui,sans-serif}}
body{{margin:0;padding:24px;background:#111;color:#eee}}
h1,.grid{{max-width:1600px;margin:0 auto 18px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:18px}}
figure{{margin:0;padding:12px;background:#1c1c1c;border:1px solid #333;border-radius:10px}}
img{{display:block;width:100%;max-width:{width}px;aspect-ratio:{width}/{height};margin:auto}}
figcaption{{margin-top:10px;font-weight:650}}small{{display:block;color:#aaa;font-weight:400}}
a{{color:#8cc8ff}}
</style></head><body><h1>CSS linear compositor MVP</h1>
<div class="grid">{''.join(cards)}</div></body></html>
"""
    atomic_write_text(str(output_dir / "index.html"), document)


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
    source_path = args.source.resolve()
    splats_path = args.splats_json.resolve()
    manifest_path = args.manifest.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(manifest_path.read_text())
    background = np.asarray(
        manifest["config"]["background_linear_rgb"], dtype=np.float32
    )
    splats = load_splats_json(str(splats_path))
    target = load_png(str(source_path))[:, :, :3]
    height, width = target.shape[:2]
    for variant in VARIANTS:
        atomic_write_text(
            str(output_dir / f"{variant.name}.html"),
            generate_css_variant(
                splats,
                width=width,
                height=height,
                background_linear_rgb=background,
                variant=variant,
            ),
        )

    executable = resolve_browser_executable(args.browser_executable)
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError('install the "capture" extra to run this MVP') from exc

    results: list[dict[str, Any]] = []
    with sync_playwright() as playwright:
        options: dict[str, Any] = {"headless": True}
        if executable is not None:
            options["executable_path"] = str(executable)
        browser = playwright.chromium.launch(**options)
        browser_version = browser.version
        try:
            page = browser.new_page(
                viewport={"width": width, "height": height}, device_scale_factor=1
            )
            for variant in VARIANTS:
                artifact = output_dir / f"{variant.name}.html"
                payload, settle_samples = _capture_variant(
                    page,
                    artifact,
                    repeats=args.repeats,
                    timeout_ms=args.timeout_ms,
                )
                with atomic_output_path(
                    output_dir / f"{variant.name}.png"
                ) as temporary:
                    temporary.write_bytes(payload)
                metrics = compute_fidelity_metrics(
                    target,
                    _decode_png_linear(payload),
                    splat_count=len(splats),
                    file_size_bytes=artifact.stat().st_size,
                    render_method=variant.name,
                ).as_dict()
                results.append(
                    {
                        "name": variant.name,
                        "settle_ms_median": float(statistics.median(settle_samples)),
                        **metrics,
                    }
                )
        finally:
            browser.close()

    results.sort(key=lambda result: float(result["ssim_srgb"]), reverse=True)
    report = {
        "schema": "splatthis.css-linear-compositor-mvp/1",
        "source": str(source_path),
        "splats_json": str(splats_path),
        "manifest": str(manifest_path),
        "browser": browser_version,
        "width": width,
        "height": height,
        "splat_count": len(splats),
        "results": results,
    }
    atomic_write_text(str(output_dir / "results.json"), json.dumps(report, indent=2))
    _write_overview(
        output_dir,
        source_path=source_path,
        results=results,
        width=width,
        height=height,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
