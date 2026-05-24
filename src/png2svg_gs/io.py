"""
Input/Output utilities for PNG→SVG and PNG→DrawingML/PPTX pipelines.

Handles PNG loading, vector export generation, side-by-side comparison output,
and optional artifact grading helpers.
"""

from __future__ import annotations

import io
import logging
import math
import shutil
import subprocess
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from .splat import (
    LAYER_BASE,
    RAW_SPLAT_SCHEMA_VERSION,
    SPLAT_LAYER_NAMES,
    GaussianSplat,
    RawSplat,
    render_importance_for_raw,
    render_order_key,
)

logger = logging.getLogger(__name__)
EMU_PER_PX = 9525
# Favor fidelity over blanket coverage in export geometry.
ELLIPSE_OVERLAP_BOOST = 1.15
MIN_ELLIPSE_RADIUS_PX = 0.35
# Upper bound on radial-gradient stops used to approximate each splat's
# Gaussian opacity falloff in exported SVG. Per-splat the count is chosen
# adaptively (see _adaptive_gradient_stops) so that the piecewise-linear
# stop interpolation stays within SVG_GRADIENT_STOP_MAX_ERROR of the true
# Gaussian curve. Low-alpha splats often need just 2; mid-alpha 3-4; only
# high-alpha sharp curves want the full 8.
SVG_GRADIENT_STOPS = 8
SVG_GRADIENT_STOPS_MIN = 2
# Max absolute opacity error (0..1) tolerated between the true Gaussian
# curve and the linear interpolation between adjacent gradient stops.
# This is the per-export *baseline* used when callers don't compute a
# density-aware value via `_density_aware_stop_error(splat_count)`.
#
# Empirically (May 2026): sparse runs (~1800 splats / 1Mpx, large splats,
# few overlaps per pixel) tolerate 0.05 -- the piecewise-linear stop
# interpolation under sRGB compositing happens to land slightly closer to
# source than the "more accurate" many-stop Gaussian does. Dense runs
# (~3900 splats / 1Mpx, small splats, deep overlap stacks) need ~0.02
# because per-splat 2-stop linear ramps stack visibly as "unsmoothed"
# artifacts. The density-aware helper interpolates between them so users
# don't have to think about it. See tmp/stops_sweep_visual.html and
# tmp/forced_4000_thresholds_visual.html for the data.
SVG_GRADIENT_STOP_MAX_ERROR = 0.05


def _density_aware_stop_error(
    splat_count: int,
    *,
    baseline: float = SVG_GRADIENT_STOP_MAX_ERROR,
    floor: float = 0.01,
    ceiling: float = 0.05,
) -> float:
    """Pick the adaptive-stop error threshold for an export.

    The two empirical fit points are 1862 splats -> 0.05 and 3905 splats ->
    0.02, both at ~1Mpx canvas. `threshold ~ 100 / N` interpolates them
    cleanly: at N=1862 it gives 0.054 (clamped to ceiling 0.05), at N=3905
    it gives 0.026. Floor at 0.01 prevents runaway stop counts on
    pathological dense scenes.
    """

    if splat_count <= 0:
        return float(baseline)
    raw = 100.0 / float(splat_count)
    return float(np.clip(raw, floor, ceiling))


DEFAULT_EXPORT_ORDER = "importance"
DEFAULT_PPTX_SPLAT_STYLE = "gradient"
PPTX_SOFT_EDGE_ALPHA_SCALE = 0.25
PPTX_SOFT_EDGE_RADIUS_FACTOR = 0.20
PPTX_SOFT_EDGE_K_SIGMA_SCALE = 0.92
PPTX_GRADIENT_ALPHA_SCALE = 0.40
SVG_BROWSER_COMPAT_RECIPE = "browser-compatible"
SVG_SCRIPTED_MATRIX_RECIPE = "scripted-matrix"
SVG_PALETTE_QUANTIZED_RECIPE = "palette-quantized"
# Default palette size when the palette-quantized recipe is selected without an
# explicit override. 128 colors visually covers photographic input without
# bloating <defs>; raise via refinement_config['svg_palette_size'] if needed.
SVG_PALETTE_QUANTIZED_DEFAULT_SIZE = 128
SVG_BACKGROUND_ALPHA_CAP = 0.20
SVG_FEATHER_EXTENT = 2.0
SVG_PRECOMP_ALPHA_THRESHOLD = 0.90
SVG_PRECOMP_MAX_SRGB = 160.0


def _normalize_svg_export_recipe(export_recipe: str) -> str:
    normalized = str(export_recipe).strip().lower().replace("_", "-")
    if normalized in {"browser", "browser-compatible"}:
        return SVG_BROWSER_COMPAT_RECIPE
    if normalized in {"scripted", "scripted-standard", "scripted-matrix", "matrix"}:
        return SVG_SCRIPTED_MATRIX_RECIPE
    if normalized in {
        "palette",
        "palette-quantized",
        "quantized",
        "shared",
        "shared-currentcolor",
    }:
        return SVG_PALETTE_QUANTIZED_RECIPE
    if normalized == "standard":
        return "standard"
    raise ValueError(f"Unsupported SVG export recipe: {export_recipe}")


def _layer_name(layer: Optional[int]) -> str:
    if layer is None:
        return "unassigned"
    return SPLAT_LAYER_NAMES.get(int(layer), f"layer-{int(layer)}")


def _layer_title(layer: Optional[int]) -> str:
    return _layer_name(layer).replace("-", " ").title()


def _splat_layer(splat: GaussianSplat) -> Optional[int]:
    return splat.to_raw_splat().layer


def _gaussian_opacity_curve(
    t: np.ndarray, alpha: float, gradient_footprint: float
) -> np.ndarray:
    """The opacity curve sampled by SVG gradient stops.

    Matches the renderer's per-splat alpha-over opacity at a normalized radius
    t in [0, 1] where 1.0 is `gradient_footprint` sigmas from the center.
    """
    return 1.0 - np.exp(
        -float(alpha) * np.exp(-0.5 * (t * float(gradient_footprint)) ** 2)
    )


def _adaptive_gradient_stops(
    alpha: float,
    gradient_footprint: float,
    inner_end: float,
    *,
    min_stops: int = SVG_GRADIENT_STOPS_MIN,
    max_stops: int = SVG_GRADIENT_STOPS,
    max_error: float = SVG_GRADIENT_STOP_MAX_ERROR,
) -> List[Tuple[float, float]]:
    """Return (offset, opacity) tuples approximating the Gaussian opacity curve.

    Picks the smallest stop count N in [min_stops, max_stops] whose linear
    interpolation between adjacent stops has max absolute error <= max_error
    against the true curve. Offsets span [0, inner_end].
    """

    min_stops = max(2, int(min_stops))
    max_stops = max(min_stops, int(max_stops))

    if float(alpha) <= 1e-6 or float(gradient_footprint) <= 0.0:
        # Effectively transparent or degenerate: a flat zero ramp suffices.
        return [(0.0, 0.0), (float(inner_end), 0.0)]

    sample_t = np.linspace(0.0, 1.0, 65)
    true_op = _gaussian_opacity_curve(sample_t, alpha, gradient_footprint)

    for n_stops in range(min_stops, max_stops + 1):
        stop_t = np.linspace(0.0, 1.0, n_stops)
        stop_op = _gaussian_opacity_curve(stop_t, alpha, gradient_footprint)
        interp_op = np.interp(sample_t, stop_t, stop_op)
        if float(np.max(np.abs(interp_op - true_op))) <= float(max_error):
            return [(float(t * inner_end), float(op)) for t, op in zip(stop_t, stop_op)]

    stop_t = np.linspace(0.0, 1.0, max_stops)
    stop_op = _gaussian_opacity_curve(stop_t, alpha, gradient_footprint)
    return [(float(t * inner_end), float(op)) for t, op in zip(stop_t, stop_op)]


def _normalize_pptx_splat_style(splat_style: str) -> str:
    normalized = str(splat_style).strip().lower().replace("_", "-")
    if normalized in {"softedge", "soft-edge", "soft"}:
        return "soft-edge"
    if normalized in {"gradient", "grad"}:
        return "gradient"
    raise ValueError(f"Unsupported PPTX splat style: {splat_style}")


def load_png(
    path: str,
    target_size: Optional[Tuple[int, int]] = None,
    linearize_srgb: bool = True,
) -> np.ndarray:
    """
    Load PNG image with proper preprocessing.

    Args:
        path: Path to PNG file
        target_size: Optional (width, height) for resizing
        linearize_srgb: Whether to convert sRGB to linear RGB

    Returns:
        Float32 RGB(A) array normalized to [0,1]
    """
    try:
        # Load image
        img = Image.open(path)
        logger.info(f"Loaded {img.size[0]}×{img.size[1]} image: {path}")

        # Convert to RGB(A)
        if img.mode == "P":  # Palette
            img = img.convert("RGBA")
        elif img.mode in ["L", "LA"]:  # Grayscale
            img = img.convert("RGB")
        elif img.mode == "RGBA":
            pass  # Keep alpha
        else:
            img = img.convert("RGB")

        # Resize if requested
        if target_size is not None:
            img = img.resize(target_size, Image.Resampling.LANCZOS)
            logger.info(f"Resized to {target_size[0]}×{target_size[1]}")

        # Convert to numpy array
        img_array = np.array(img, dtype=np.float32) / 255.0

        # sRGB to linear RGB conversion
        if linearize_srgb and img_array.shape[-1] >= 3:
            img_array[..., :3] = srgb_to_linear(img_array[..., :3])
            logger.info("Applied sRGB → linear RGB conversion")

        logger.info(
            f"Final image shape: {img_array.shape}, range: [{img_array.min():.3f}, {img_array.max():.3f}]"
        )
        return img_array

    except Exception as e:
        logger.error(f"Failed to load PNG {path}: {e}")
        raise


def srgb_to_linear(srgb: np.ndarray) -> np.ndarray:
    """
    Convert sRGB to linear RGB.

    Args:
        srgb: sRGB values in [0,1]

    Returns:
        Linear RGB values in [0,1]
    """
    # Standard sRGB → linear conversion
    linear = np.where(
        srgb <= 0.04045, srgb / 12.92, np.power((srgb + 0.055) / 1.055, 2.4)
    )
    return linear


def linear_to_srgb(linear: np.ndarray) -> np.ndarray:
    """
    Convert linear RGB to sRGB.

    Args:
        linear: Linear RGB values in [0,1]

    Returns:
        sRGB values in [0,1]
    """
    srgb = np.where(
        linear <= 0.0031308, 12.92 * linear, 1.055 * np.power(linear, 1.0 / 2.4) - 0.055
    )
    return np.clip(srgb, 0.0, 1.0)


def _sort_splats_for_export(
    splats: List[GaussianSplat],
    sort_mode: str = DEFAULT_EXPORT_ORDER,
    sort_by_area: bool = False,
) -> List[GaussianSplat]:
    """
    Return an export-ordered splat list.

    `importance` order is ascending so higher-importance splats render last/front-most,
    matching renderer behavior.
    """
    if sort_by_area:
        return sorted(splats, key=lambda s: s.area(), reverse=True)

    normalized = str(sort_mode).strip().lower()
    if normalized == "input":
        return list(splats)
    if normalized == "area":
        return sorted(splats, key=lambda s: s.area(), reverse=True)
    if normalized == "importance":
        return sorted(splats, key=render_order_key)
    raise ValueError(f"Unsupported export sort mode: {sort_mode}")


def save_svg(
    splats: List[GaussianSplat],
    width: int,
    height: int,
    output_path: str,
    k_sigma: float = 2.5,
    sort_by_area: bool = False,
    sort_mode: str = DEFAULT_EXPORT_ORDER,
    background_linear_rgb: Optional[np.ndarray] = None,
    export_recipe: str = "standard",
    foreground_mask: Optional[np.ndarray] = None,
    background_safe_mask: Optional[np.ndarray] = None,
    edge_band_mask: Optional[np.ndarray] = None,
) -> None:
    """
    Save splats as SVG file.

    Args:
        splats: List of Gaussian splats
        width: Image width in pixels
        height: Image height in pixels
        output_path: Output SVG file path
        k_sigma: Sigma multiplier for ellipse size (2-3 for 95-99.7% coverage)
        sort_by_area: Legacy flag to sort by area descending
        sort_mode: Export order: importance|area|input
        background_linear_rgb: Optional background color in linear RGB [0,1]
    """
    ordered_splats = _sort_splats_for_export(
        splats=splats,
        sort_mode=sort_mode,
        sort_by_area=sort_by_area,
    )

    # Generate SVG content
    svg_content = generate_svg_content(
        ordered_splats,
        width,
        height,
        k_sigma,
        background_linear_rgb=background_linear_rgb,
        export_recipe=export_recipe,
        foreground_mask=foreground_mask,
        background_safe_mask=background_safe_mask,
        edge_band_mask=edge_band_mask,
    )

    # Write to file
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(svg_content)
        logger.info(f"Saved SVG with {len(ordered_splats)} splats to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save SVG {output_path}: {e}")
        raise


def px_to_emu(value: float) -> int:
    """Convert pixels to EMU units used by DrawingML."""
    return int(round(max(0.0, value) * EMU_PER_PX))


def save_drawingml(
    splats: List[GaussianSplat],
    width: int,
    height: int,
    output_path: str,
    k_sigma: float = 2.5,
    sort_by_area: bool = False,
    sort_mode: str = DEFAULT_EXPORT_ORDER,
    background_linear_rgb: Optional[np.ndarray] = None,
    splat_style: str = DEFAULT_PPTX_SPLAT_STYLE,
) -> None:
    """
    Save splats as PresentationML slide XML with DrawingML ellipse shapes.

    The resulting XML can be inserted into `ppt/slides/slideN.xml` in a PPTX package.
    """
    ordered_splats = _sort_splats_for_export(
        splats=splats,
        sort_mode=sort_mode,
        sort_by_area=sort_by_area,
    )

    drawingml_content = generate_drawingml_slide_content(
        ordered_splats,
        width,
        height,
        k_sigma,
        background_linear_rgb=background_linear_rgb,
        splat_style=splat_style,
    )

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(drawingml_content)
        logger.info(
            f"Saved DrawingML with {len(ordered_splats)} splats to {output_path}"
        )
    except Exception as e:
        logger.error(f"Failed to save DrawingML {output_path}: {e}")
        raise


def generate_parallax_canvas_html(
    splats: List[GaussianSplat],
    width: int,
    height: int,
    background_linear_rgb: Optional[np.ndarray] = None,
    title: str = "SplatThis Parallax",
    parallax_strength: float = 28.0,
) -> str:
    """Parallax canvas runtime: per-layer canvases driven by mouse position.

    Splats with ``raw.layer`` set (via ``--layered-saliency``) get bucketed
    into base/mass/detail/edge canvases. Each canvas runs the same
    linear-light alpha-over render as ``generate_canvas_html`` but only on
    its own splats. The canvases stack absolutely; on mousemove a
    ``translate3d`` is applied per canvas scaled by its depth (base
    stationary, edge moves the most). Background-rect plate is painted
    behind layer 0 so areas revealed by foreground translation show the
    scene background color, not black/empty.

    Quality caveat: each layer composites linear-light internally, but the
    DOM composites the layers in sRGB display space. Tiny color drift vs
    the single-canvas runtime at static rest. The parallax effect itself
    is the goal here, not pixel-perfect render parity.

    Splats without a layer tag fall back to layer 1 ("mass") so they get
    a modest parallax offset.
    """

    import json

    bg_lin = (
        [0.0, 0.0, 0.0]
        if background_linear_rgb is None
        else [
            float(np.clip(c, 0.0, 1.0))
            for c in np.asarray(background_linear_rgb).reshape(-1)[:3]
        ]
    )
    bg_srgb = linear_to_srgb(np.array(bg_lin, dtype=np.float32))
    bg_rgb = tuple(int(np.clip(np.round(c * 255), 0, 255)) for c in bg_srgb)
    bg_css = f"rgb({bg_rgb[0]},{bg_rgb[1]},{bg_rgb[2]})"

    # Bucket splats into three parallax planes (not four). The saliency
    # layers (base=0, mass=1, detail=2, edge=3) get collapsed:
    #   base (0)    -> background plane (stationary)
    #   mass (1)    -> midground plane
    #   detail (2)  -> foreground plane     (merged with edge)
    #   edge (3)    -> foreground plane     (merged with detail)
    # Merging detail+edge avoids visible tearing between near-foreground
    # planes (the man's eye+glasses-rim splats would otherwise track at
    # different speeds than the face-skin splats) and eliminates one
    # cross-layer sRGB compositing seam at the most detailed region.
    # Untagged splats fall back to midground.
    PLANE_BACKGROUND = "background"
    PLANE_MIDGROUND = "midground"
    PLANE_FOREGROUND = "foreground"
    PLANE_DEPTHS = {
        PLANE_BACKGROUND: 0.0,
        PLANE_MIDGROUND: 0.4,
        PLANE_FOREGROUND: 1.0,
    }

    def _layer_to_plane(layer_id: Optional[int]) -> str:
        if layer_id is None:
            return PLANE_MIDGROUND
        if layer_id <= 0:
            return PLANE_BACKGROUND
        if layer_id == 1:
            return PLANE_MIDGROUND
        return PLANE_FOREGROUND

    buckets: Dict[str, List[List[float]]] = {
        PLANE_BACKGROUND: [],
        PLANE_MIDGROUND: [],
        PLANE_FOREGROUND: [],
    }
    for splat in splats:
        raw = splat.to_raw_splat()
        plane = _layer_to_plane(raw.layer)
        buckets[plane].append(
            [
                float(raw.x),
                float(raw.y),
                float(raw.sx),
                float(raw.sy),
                float(raw.theta),
                float(raw.r),
                float(raw.g),
                float(raw.b),
                float(raw.a),
                render_importance_for_raw(raw),
            ]
        )

    layer_data_json = json.dumps(
        [
            {
                "layer": plane,
                "depth": PLANE_DEPTHS[plane],
                "splats": buckets[plane],
            }
            for plane in (PLANE_BACKGROUND, PLANE_MIDGROUND, PLANE_FOREGROUND)
            if buckets[plane]
        ],
        separators=(",", ":"),
    )
    counts = {k: len(v) for k, v in buckets.items()}

    js = (
        r"""
(function(){
  const t0 = performance.now();
  const W = __W__, H = __H__;
  const BG = __BG__;
  const STRENGTH = __STRENGTH__;
  const LAYERS = __LAYERS__;
  const status = document.getElementById('status');
  const stack = document.getElementById('stack');

  function renderLayer(canvas, splats) {
    const ctx = canvas.getContext('2d', { willReadFrequently: false });
    splats.sort((a, b) => a[9] - b[9]);
    const lin = new Float32Array(W * H * 3);
    const T = new Float32Array(W * H).fill(1);
    const FOOTPRINT = 3.0;
    for (let si = 0; si < splats.length; si++) {
      const s = splats[si];
      const x = s[0], y = s[1];
      const sx = Math.max(s[2], 1e-4), sy = Math.max(s[3], 1e-4);
      const theta = s[4];
      const r = s[5], g = s[6], b = s[7];
      const a = Math.min(1, Math.max(0, s[8]));
      const rx = Math.max(1, Math.ceil(FOOTPRINT * sx));
      const ry = Math.max(1, Math.ceil(FOOTPRINT * sy));
      const x0 = Math.max(0, Math.floor(x - rx));
      const x1 = Math.min(W, Math.ceil(x + rx + 1));
      const y0 = Math.max(0, Math.floor(y - ry));
      const y1 = Math.min(H, Math.ceil(y + ry + 1));
      if (x0 >= x1 || y0 >= y1) continue;
      const ct = Math.cos(theta), st = Math.sin(theta);
      const invSx2 = 1 / (sx * sx), invSy2 = 1 / (sy * sy);
      for (let py = y0; py < y1; py++) {
        const baseRow = py * W;
        for (let px = x0; px < x1; px++) {
          const dx = px - x, dy = py - y;
          const u = ct * dx + st * dy;
          const v = -st * dx + ct * dy;
          const q = u * u * invSx2 + v * v * invSy2;
          const w = Math.exp(-0.5 * q);
          const la = 1 - Math.exp(-a * w);
          const idx = baseRow + px;
          const tt = T[idx];
          const contrib = tt * la;
          const j = idx * 3;
          lin[j]     += contrib * r;
          lin[j + 1] += contrib * g;
          lin[j + 2] += contrib * b;
          T[idx] = tt * (1 - la);
        }
      }
    }
    const img = ctx.createImageData(W, H);
    const out = img.data;
    const THR = 0.0031308;
    // Per-layer canvases are stacked over the bg plate; transparent pixels
    // (T near 1) reveal the layer below, so write alpha = 1 - T.
    for (let i = 0; i < W * H; i++) {
      const j = i * 3, k = i * 4;
      const tt = T[i];
      let rL = lin[j], gL = lin[j + 1], bL = lin[j + 2];
      const denom = (1 - tt) > 1e-6 ? (1 - tt) : 1;
      rL = rL / denom;
      gL = gL / denom;
      bL = bL / denom;
      if (rL < 0) rL = 0; else if (rL > 1) rL = 1;
      if (gL < 0) gL = 0; else if (gL > 1) gL = 1;
      if (bL < 0) bL = 0; else if (bL > 1) bL = 1;
      const rS = rL <= THR ? 12.92 * rL : 1.055 * Math.pow(rL, 1/2.4) - 0.055;
      const gS = gL <= THR ? 12.92 * gL : 1.055 * Math.pow(gL, 1/2.4) - 0.055;
      const bS = bL <= THR ? 12.92 * bL : 1.055 * Math.pow(bL, 1/2.4) - 0.055;
      out[k]     = (rS * 255 + 0.5) | 0;
      out[k + 1] = (gS * 255 + 0.5) | 0;
      out[k + 2] = (bS * 255 + 0.5) | 0;
      out[k + 3] = ((1 - tt) * 255 + 0.5) | 0;
    }
    ctx.putImageData(img, 0, 0);
  }

  const layerEls = [];
  for (const ld of LAYERS) {
    const cnv = document.createElement('canvas');
    cnv.width = W; cnv.height = H;
    cnv.className = 'layer';
    cnv.dataset.depth = ld.depth;
    stack.appendChild(cnv);
    renderLayer(cnv, ld.splats);
    layerEls.push(cnv);
  }

  let total = 0;
  for (const ld of LAYERS) total += ld.splats.length;
  status.textContent = 'parallax ready: ' + LAYERS.length + ' layers, ' + total + ' splats, rendered in ' + (performance.now() - t0).toFixed(0) + 'ms';

  function onMove(e) {
    const rect = stack.getBoundingClientRect();
    const mx = ((e.clientX - rect.left) / rect.width  - 0.5) * 2;
    const my = ((e.clientY - rect.top)  / rect.height - 0.5) * 2;
    for (const el of layerEls) {
      const d = parseFloat(el.dataset.depth);
      // Foreground tracks the mouse, background stays still. We translate
      // OPPOSITE the mouse so the parallax feels like looking through the scene.
      const tx = -mx * d * STRENGTH;
      const ty = -my * d * STRENGTH;
      el.style.transform = 'translate3d(' + tx.toFixed(2) + 'px,' + ty.toFixed(2) + 'px,0)';
    }
  }
  stack.addEventListener('mousemove', onMove);
  stack.addEventListener('mouseleave', () => {
    for (const el of layerEls) el.style.transform = 'translate3d(0,0,0)';
  });
})();
""".replace(
            "__W__", str(int(width))
        )
        .replace("__H__", str(int(height)))
        .replace("__BG__", f"[{bg_lin[0]:.6f},{bg_lin[1]:.6f},{bg_lin[2]:.6f}]")
        .replace("__STRENGTH__", f"{float(parallax_strength):.3f}")
        .replace("__LAYERS__", layer_data_json)
    )

    safe_title = title.replace("<", "&lt;").replace(">", "&gt;")
    return (
        "<!doctype html>\n"
        f'<html><head><meta charset="utf-8"><title>{safe_title}</title>\n'
        "<style>\n"
        "  body { margin: 0; background: #111; color: #eee;"
        "    font: 14px -apple-system, sans-serif;"
        "    display: flex; flex-direction: column; align-items: center; padding: 16px; }\n"
        f"  #stack {{ position: relative; width: {int(width)}px; height: {int(height)}px;"
        f"    background: {bg_css}; overflow: hidden; border: 1px solid #333; border-radius: 6px; }}\n"
        "  #stack .layer { position: absolute; top: 0; left: 0;"
        "    transition: transform 0.06s cubic-bezier(0.2,0.7,0.3,1.0);"
        "    pointer-events: none; image-rendering: pixelated; }\n"
        "  #status { color: #7fd17f; font-family: ui-monospace, monospace;"
        "    font-size: 12px; margin: 8px 0; }\n"
        "</style></head>\n"
        "<body>\n"
        '<div id="status">rendering...</div>\n'
        f'<div id="stack" data-layers="{len(layer_data_json)}"></div>\n'
        "<script>\n" + js + "\n</script>\n"
        "</body></html>\n"
    )


def generate_canvas_html(
    splats: List[GaussianSplat],
    width: int,
    height: int,
    background_linear_rgb: Optional[np.ndarray] = None,
    title: str = "SplatThis Canvas",
) -> str:
    """Self-contained HTML that renders the splats via a JS canvas runtime
    doing real linear-space alpha-over compositing.

    Mirrors `render_splats_numpy` math exactly (3σ footprint, sorted by
    importance ascending = back-to-front, per-splat alpha-over with
    `layer_alpha = 1 - exp(-a * exp(-0.5 * q))`, then linear -> sRGB on
    output). The browser's gamma-space SVG compositing and 8-stop gradient
    discretization are the things you can't reproduce with `radialGradient`;
    a JS canvas can. So the displayed result == the optimizer's own forward,
    breaking the SVG primitive's structural cap.
    """
    import json

    bg_lin = (
        [0.0, 0.0, 0.0]
        if background_linear_rgb is None
        else [
            float(np.clip(c, 0.0, 1.0))
            for c in np.asarray(background_linear_rgb).reshape(-1)[:3]
        ]
    )

    # Compact per-splat record: x, y, sx, sy, theta, r, g, b, a, render_order, layer.
    rows: List[List[float]] = []
    for splat in splats:
        raw = splat.to_raw_splat()
        rows.append(
            [
                float(raw.x),
                float(raw.y),
                float(raw.sx),
                float(raw.sy),
                float(raw.theta),
                float(raw.r),
                float(raw.g),
                float(raw.b),
                float(raw.a),
                render_importance_for_raw(raw),
                -1.0 if raw.layer is None else float(raw.layer),
            ]
        )
    splats_json = json.dumps(rows, separators=(",", ":"))

    js = (
        r"""
(function(){
  const t0 = performance.now();
  const W = __W__, H = __H__;
  const BG = __BG__;
  const SPLATS = __SPLATS__;
  const FOOTPRINT = 3.0;
  const status = document.getElementById('status');
  const canvas = document.getElementById('c');
  const ctx = canvas.getContext('2d', { willReadFrequently: false });

  // Back-to-front: lowest render_order first, highest last (painted on top).
  SPLATS.sort((a, b) => a[9] - b[9]);

  const lin = new Float32Array(W * H * 3);
  const T = new Float32Array(W * H).fill(1);

  for (let si = 0; si < SPLATS.length; si++) {
    const s = SPLATS[si];
    const x = s[0], y = s[1];
    const sx = Math.max(s[2], 1e-4), sy = Math.max(s[3], 1e-4);
    const theta = s[4];
    const r = s[5], g = s[6], b = s[7];
    const a = Math.min(1, Math.max(0, s[8]));
    const rx = Math.max(1, Math.ceil(FOOTPRINT * sx));
    const ry = Math.max(1, Math.ceil(FOOTPRINT * sy));
    const x0 = Math.max(0, Math.floor(x - rx));
    const x1 = Math.min(W, Math.ceil(x + rx + 1));
    const y0 = Math.max(0, Math.floor(y - ry));
    const y1 = Math.min(H, Math.ceil(y + ry + 1));
    if (x0 >= x1 || y0 >= y1) continue;
    const ct = Math.cos(theta), st = Math.sin(theta);
    const invSx2 = 1 / (sx * sx), invSy2 = 1 / (sy * sy);
    for (let py = y0; py < y1; py++) {
      const baseRow = py * W;
      for (let px = x0; px < x1; px++) {
        const dx = px - x, dy = py - y;
        const u = ct * dx + st * dy;
        const v = -st * dx + ct * dy;
        const q = u * u * invSx2 + v * v * invSy2;
        const w = Math.exp(-0.5 * q);
        const la = 1 - Math.exp(-a * w);
        const idx = baseRow + px;
        const tt = T[idx];
        const contrib = tt * la;
        const j = idx * 3;
        lin[j]     += contrib * r;
        lin[j + 1] += contrib * g;
        lin[j + 2] += contrib * b;
        T[idx] = tt * (1 - la);
      }
    }
  }

  // Linear -> sRGB and pack into ImageData.
  const img = ctx.createImageData(W, H);
  const out = img.data;
  const THR = 0.0031308;
  for (let i = 0; i < W * H; i++) {
    const j = i * 3, k = i * 4;
    const tt = T[i];
    let rL = lin[j]     + tt * BG[0];
    let gL = lin[j + 1] + tt * BG[1];
    let bL = lin[j + 2] + tt * BG[2];
    if (rL < 0) rL = 0; else if (rL > 1) rL = 1;
    if (gL < 0) gL = 0; else if (gL > 1) gL = 1;
    if (bL < 0) bL = 0; else if (bL > 1) bL = 1;
    const rS = rL <= THR ? 12.92 * rL : 1.055 * Math.pow(rL, 1/2.4) - 0.055;
    const gS = gL <= THR ? 12.92 * gL : 1.055 * Math.pow(gL, 1/2.4) - 0.055;
    const bS = bL <= THR ? 12.92 * bL : 1.055 * Math.pow(bL, 1/2.4) - 0.055;
    out[k]     = (rS * 255 + 0.5) | 0;
    out[k + 1] = (gS * 255 + 0.5) | 0;
    out[k + 2] = (bS * 255 + 0.5) | 0;
    out[k + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  status.textContent = 'rendered ' + SPLATS.length + ' splats at ' + W + '×' + H + ' in ' + (performance.now() - t0).toFixed(0) + 'ms (linear-space alpha-over)';
})();
""".replace(
            "__W__", str(int(width))
        )
        .replace("__H__", str(int(height)))
        .replace("__BG__", f"[{bg_lin[0]:.6f},{bg_lin[1]:.6f},{bg_lin[2]:.6f}]")
        .replace("__SPLATS__", splats_json)
    )

    safe_title = title.replace("<", "&lt;").replace(">", "&gt;")
    return (
        "<!doctype html>\n"
        '<html><head><meta charset="utf-8"><title>' + safe_title + "</title>\n"
        "<style>\n"
        "  body{margin:0;background:#111;color:#eee;font:14px -apple-system,sans-serif;"
        "display:flex;flex-direction:column;align-items:center;padding:16px}\n"
        "  #c{image-rendering:pixelated;border:1px solid #333;border-radius:6px;max-width:100%}\n"
        "  #status{color:#7fd17f;font-family:ui-monospace,monospace;font-size:12px;margin:8px 0}\n"
        "</style></head>\n"
        "<body>\n"
        '<div id="status">rendering...</div>\n'
        f'<canvas id="c" width="{int(width)}" height="{int(height)}"></canvas>\n'
        "<script>\n" + js + "\n</script>\n"
        "</body></html>\n"
    )


def generate_svg_content(
    splats: List[GaussianSplat],
    width: int,
    height: int,
    k_sigma: float = 2.5,
    background_linear_rgb: Optional[np.ndarray] = None,
    export_recipe: str = "standard",
    foreground_mask: Optional[np.ndarray] = None,
    background_safe_mask: Optional[np.ndarray] = None,
    edge_band_mask: Optional[np.ndarray] = None,
) -> str:
    """
    Generate SVG content from splats.

    Args:
        splats: List of Gaussian splats
        width: Image width
        height: Image height
        k_sigma: Sigma multiplier for ellipse size
        background_linear_rgb: Optional background color in linear RGB [0,1]
        export_recipe: "standard" or "browser-compatible". The browser recipe
            feathers gradients, pre-compensates dark opaque splats against the
            background, and caps alpha in safe background regions.

    Returns:
        Complete SVG document as string
    """
    normalized_recipe = _normalize_svg_export_recipe(export_recipe)
    if normalized_recipe == SVG_SCRIPTED_MATRIX_RECIPE:
        return generate_scripted_svg_content(
            splats=splats,
            width=width,
            height=height,
            k_sigma=k_sigma,
            background_linear_rgb=background_linear_rgb,
            foreground_mask=foreground_mask,
            background_safe_mask=background_safe_mask,
            edge_band_mask=edge_band_mask,
        )
    if normalized_recipe == SVG_PALETTE_QUANTIZED_RECIPE:
        return generate_palette_quantized_svg_content(
            splats=splats,
            width=width,
            height=height,
            k_sigma=k_sigma,
            background_linear_rgb=background_linear_rgb,
            foreground_mask=foreground_mask,
            background_safe_mask=background_safe_mask,
            edge_band_mask=edge_band_mask,
        )
    use_browser_recipe = normalized_recipe == SVG_BROWSER_COMPAT_RECIPE

    def _valid_mask(mask: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if mask is None:
            return None
        arr = np.asarray(mask)
        if arr.shape != (int(height), int(width)):
            raise ValueError("SVG region masks must match output height/width")
        return arr.astype(bool, copy=False)

    foreground = _valid_mask(foreground_mask)
    background_safe = _valid_mask(background_safe_mask)
    edge_band = _valid_mask(edge_band_mask)

    bg_linear: Optional[np.ndarray] = None
    bg_srgb: Optional[np.ndarray] = None
    background_rect_line: Optional[str] = None
    if background_linear_rgb is not None:
        bg = np.asarray(background_linear_rgb, dtype=np.float32).reshape(-1)
        if bg.size != 3:
            raise ValueError("background_linear_rgb must have exactly 3 components")
        bg = np.clip(bg, 0.0, 1.0)
        bg_linear = bg
        bg_srgb = linear_to_srgb(bg)
        bg_r = int(np.clip(np.round(bg_srgb[0] * 255), 0, 255))
        bg_g = int(np.clip(np.round(bg_srgb[1] * 255), 0, 255))
        bg_b = int(np.clip(np.round(bg_srgb[2] * 255), 0, 255))
        background_rect_line = (
            f'  <rect x="0" y="0" width="{width}" height="{height}" '
            f'fill="rgb({bg_r},{bg_g},{bg_b})" class="background"/>'
        )

    svg_lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" '
            'xmlns="http://www.w3.org/2000/svg">'
        ),
        "  <desc>Generated by PNG2SVG Gaussian Splatting Pipeline</desc>",
        "  <defs>",
        "    <style>",
        "      .splat { mix-blend-mode: normal; }",
        "    </style>",
    ]

    def _splat_center(splat: GaussianSplat) -> Tuple[int, int]:
        x = int(np.clip(round(float(splat.mu[0])), 0, max(int(width) - 1, 0)))
        y = int(np.clip(round(float(splat.mu[1])), 0, max(int(height) - 1, 0)))
        return x, y

    def _in_safe_background(splat: GaussianSplat) -> bool:
        if background_safe is None:
            return False
        x, y = _splat_center(splat)
        if not bool(background_safe[y, x]):
            return False
        if foreground is not None and bool(foreground[y, x]):
            return False
        if edge_band is not None and bool(edge_band[y, x]):
            return False
        return True

    def _browser_compensated_color(splat: GaussianSplat, alpha: float) -> np.ndarray:
        color_linear = np.clip(np.array(splat.color[:3], dtype=np.float32), 0.0, 1.0)
        color_srgb = linear_to_srgb(color_linear)
        if (
            not use_browser_recipe
            or bg_linear is None
            or bg_srgb is None
            or float(splat.alpha) < SVG_PRECOMP_ALPHA_THRESHOLD
            or float(np.max(color_srgb) * 255.0) > SVG_PRECOMP_MAX_SRGB
        ):
            return color_srgb

        # Browsers blend SVG stops in display space. Solve the stop color that
        # gives the same center-over-background result as linear alpha-over.
        paint_alpha = 1.0 - math.exp(-float(np.clip(alpha, 0.0, 1.0)))
        target_srgb = linear_to_srgb(
            paint_alpha * color_linear + (1.0 - paint_alpha) * bg_linear
        )
        if paint_alpha <= 1e-6:
            return color_srgb
        return np.clip(
            (target_srgb - (1.0 - paint_alpha) * bg_srgb) / paint_alpha, 0.0, 1.0
        )

    gradient_footprint = ELLIPSE_OVERLAP_BOOST * k_sigma
    feather_extent = SVG_FEATHER_EXTENT if use_browser_recipe else 1.0
    inner_end = 1.0 / feather_extent
    # Density-aware stop-error threshold: sparse scenes tolerate fewer stops
    # per splat; dense scenes need more because per-splat 2-stop ramps stack
    # into visible "unsmoothed" artifacts.
    stop_error = _density_aware_stop_error(len(splats))

    # Per-splat radial gradients approximate gaussian opacity in exported SVG.
    for i, splat in enumerate(splats):
        gradient_id = f"splat_grad_{i}"
        alpha = float(np.clip(splat.alpha, 0.0, 1.0))
        if use_browser_recipe and _in_safe_background(splat):
            alpha = min(alpha, SVG_BACKGROUND_ALPHA_CAP)

        rgb_srgb = _browser_compensated_color(splat, alpha)
        r = int(np.clip(np.round(rgb_srgb[0] * 255), 0, 255))
        g = int(np.clip(np.round(rgb_srgb[1] * 255), 0, 255))
        b = int(np.clip(np.round(rgb_srgb[2] * 255), 0, 255))
        color = f"rgb({r},{g},{b})"
        # True-Gaussian gradient stops with adaptive count: reproduce the
        # renderer's per-splat alpha-over opacity 1 - exp(-a * exp(-0.5 * r^2))
        # using only as many stops as needed to keep the piecewise-linear
        # interpolation within `stop_error` of the true curve. The threshold
        # is density-aware (see _density_aware_stop_error): looser for sparse
        # scenes, tighter for dense ones so 2-stop linear ramps don't pile up
        # into visible artifacts.
        adaptive_stops = _adaptive_gradient_stops(
            alpha, gradient_footprint, inner_end, max_error=stop_error
        )
        stop_lines = [
            f'      <stop offset="{offset * 100:.1f}%" stop-color="{color}" stop-opacity="{opacity:.5f}"/>'
            for offset, opacity in adaptive_stops
        ]
        if use_browser_recipe:
            mid_fade = (inner_end + 1.0) / 2.0
            stop_lines.append(
                f'      <stop offset="{mid_fade * 100:.1f}%" stop-color="{color}" stop-opacity="0"/>'
            )
            stop_lines.append(
                f'      <stop offset="100.0%" stop-color="{color}" stop-opacity="0"/>'
            )
        svg_lines.append(
            f'    <radialGradient id="{gradient_id}" cx="50%" cy="50%" r="50%" '
            'gradientUnits="objectBoundingBox">'
        )
        svg_lines.extend(stop_lines)
        svg_lines.append("    </radialGradient>")
    svg_lines.extend(["  </defs>", ""])
    if background_rect_line is not None:
        svg_lines.append(background_rect_line)
        svg_lines.append("")

    for i, splat in enumerate(splats):
        ellipse_element = splat_to_svg_ellipse(
            splat=splat,
            k_sigma=k_sigma,
            element_id=f"splat_{i}",
            gradient_id=f"splat_grad_{i}",
            radius_scale=feather_extent,
        )
        svg_lines.append(f"  {ellipse_element}")

    svg_lines.extend(["", "</svg>"])
    return "\n".join(svg_lines)


def generate_scripted_svg_content(
    splats: List[GaussianSplat],
    width: int,
    height: int,
    k_sigma: float = 2.5,
    background_linear_rgb: Optional[np.ndarray] = None,
    foreground_mask: Optional[np.ndarray] = None,
    background_safe_mask: Optional[np.ndarray] = None,
    edge_band_mask: Optional[np.ndarray] = None,
) -> str:
    """
    Generate a compact browser SVG that stores splats as a numeric matrix.

    The SVG source contains one data row per splat plus a small script. On load,
    the script expands the rows into normal SVG radial gradients and matrix-
    transformed unit ellipses. This keeps the source small and gzip-friendly
    while matching the browser-compatible static SVG rendering in browsers that
    execute inline SVG scripts.
    """

    def _valid_mask(mask: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if mask is None:
            return None
        arr = np.asarray(mask)
        if arr.shape != (int(height), int(width)):
            raise ValueError("SVG region masks must match output height/width")
        return arr.astype(bool, copy=False)

    foreground = _valid_mask(foreground_mask)
    background_safe = _valid_mask(background_safe_mask)
    edge_band = _valid_mask(edge_band_mask)

    bg_linear = np.ones(3, dtype=np.float32)
    if background_linear_rgb is not None:
        bg = np.asarray(background_linear_rgb, dtype=np.float32).reshape(-1)
        if bg.size != 3:
            raise ValueError("background_linear_rgb must have exactly 3 components")
        bg_linear = np.clip(bg, 0.0, 1.0)
    bg_srgb = linear_to_srgb(bg_linear)
    bg_rgb = tuple(int(np.clip(np.round(c * 255), 0, 255)) for c in bg_srgb)

    def _splat_center(splat: GaussianSplat) -> Tuple[int, int]:
        x = int(np.clip(round(float(splat.mu[0])), 0, max(int(width) - 1, 0)))
        y = int(np.clip(round(float(splat.mu[1])), 0, max(int(height) - 1, 0)))
        return x, y

    def _in_safe_background(splat: GaussianSplat) -> bool:
        if background_safe is None:
            return False
        x, y = _splat_center(splat)
        if not bool(background_safe[y, x]):
            return False
        if foreground is not None and bool(foreground[y, x]):
            return False
        if edge_band is not None and bool(edge_band[y, x]):
            return False
        return True

    def _scripted_color_and_alpha(
        splat: GaussianSplat,
    ) -> Tuple[Tuple[int, int, int], float]:
        alpha = float(np.clip(splat.alpha, 0.0, 1.0))
        if _in_safe_background(splat):
            alpha = min(alpha, SVG_BACKGROUND_ALPHA_CAP)

        color_linear = np.clip(np.array(splat.color[:3], dtype=np.float32), 0.0, 1.0)
        color_srgb = linear_to_srgb(color_linear)
        if (
            float(splat.alpha) >= SVG_PRECOMP_ALPHA_THRESHOLD
            and float(np.max(color_srgb) * 255.0) <= SVG_PRECOMP_MAX_SRGB
        ):
            paint_alpha = 1.0 - math.exp(-alpha)
            if paint_alpha > 1e-6:
                target_srgb = linear_to_srgb(
                    paint_alpha * color_linear + (1.0 - paint_alpha) * bg_linear
                )
                color_srgb = np.clip(
                    (target_srgb - (1.0 - paint_alpha) * bg_srgb) / paint_alpha,
                    0.0,
                    1.0,
                )

        rgb = tuple(int(np.clip(np.round(c * 255), 0, 255)) for c in color_srgb)
        return rgb, alpha

    def _matrix_row(splat: GaussianSplat) -> str:
        eigenvals, eigenvecs = splat.eigendecomposition()
        rx = max(
            MIN_ELLIPSE_RADIUS_PX,
            SVG_FEATHER_EXTENT
            * ELLIPSE_OVERLAP_BOOST
            * k_sigma
            * np.sqrt(max(float(eigenvals[0]), 1e-8)),
        )
        ry = max(
            MIN_ELLIPSE_RADIUS_PX,
            SVG_FEATHER_EXTENT
            * ELLIPSE_OVERLAP_BOOST
            * k_sigma
            * np.sqrt(max(float(eigenvals[1]), 1e-8)),
        )
        theta = float(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
        cos_t = float(np.cos(theta))
        sin_t = float(np.sin(theta))
        # SVG matrix(a b c d e f) maps the unit circle to the rotated ellipse.
        a = rx * cos_t
        b = rx * sin_t
        c = -ry * sin_t
        d = ry * cos_t
        e = float(splat.mu[0])
        f = float(splat.mu[1])
        rgb, alpha = _scripted_color_and_alpha(splat)
        values = [
            f"{a:.2f}",
            f"{b:.2f}",
            f"{c:.2f}",
            f"{d:.2f}",
            f"{e:.2f}",
            f"{f:.2f}",
            str(rgb[0]),
            str(rgb[1]),
            str(rgb[2]),
            f"{alpha:.4f}",
        ]
        return ",".join(values)

    rows = ";".join(_matrix_row(splat) for splat in splats)
    gradient_footprint = ELLIPSE_OVERLAP_BOOST * k_sigma
    inner_end = 1.0 / SVG_FEATHER_EXTENT
    script = f"""
(function(){{
  const NS = 'http://www.w3.org/2000/svg';
  const data = document.getElementById('splat-data').textContent.trim();
  const rows = data ? data.split(';') : [];
  const defs = document.getElementById('defs');
  const layer = document.getElementById('splats');
  const gradFrag = document.createDocumentFragment();
  const splatFrag = document.createDocumentFragment();
  const stops = {SVG_GRADIENT_STOPS};
  const footprint = {gradient_footprint:.8f};
  const innerEnd = {inner_end:.8f};
  function addStop(grad, offset, color, opacity) {{
    const stop = document.createElementNS(NS, 'stop');
    stop.setAttribute('offset', (offset * 100).toFixed(1) + '%');
    stop.setAttribute('stop-color', color);
    stop.setAttribute('stop-opacity', opacity.toFixed(5));
    grad.appendChild(stop);
  }}
  for (let i = 0; i < rows.length; i++) {{
    const v = rows[i].split(',');
    const color = 'rgb(' + v[6] + ',' + v[7] + ',' + v[8] + ')';
    const alpha = +v[9];
    const grad = document.createElementNS(NS, 'radialGradient');
    grad.id = 'g' + i;
    grad.setAttribute('cx', '50%');
    grad.setAttribute('cy', '50%');
    grad.setAttribute('r', '50%');
    grad.setAttribute('gradientUnits', 'objectBoundingBox');
    for (let j = 0; j < stops; j++) {{
      const t = j / (stops - 1);
      const opacity = 1 - Math.exp(-alpha * Math.exp(-0.5 * Math.pow(t * footprint, 2)));
      addStop(grad, t * innerEnd, color, opacity);
    }}
    addStop(grad, (innerEnd + 1) / 2, color, 0);
    addStop(grad, 1, color, 0);
    gradFrag.appendChild(grad);

    const ellipse = document.createElementNS(NS, 'ellipse');
    ellipse.setAttribute('cx', '0');
    ellipse.setAttribute('cy', '0');
    ellipse.setAttribute('rx', '1');
    ellipse.setAttribute('ry', '1');
    ellipse.setAttribute('transform', 'matrix(' + v[0] + ' ' + v[1] + ' ' + v[2] + ' ' + v[3] + ' ' + v[4] + ' ' + v[5] + ')');
    ellipse.setAttribute('fill', 'url(#g' + i + ')');
    ellipse.setAttribute('class', 'splat');
    splatFrag.appendChild(ellipse);
  }}
  defs.appendChild(gradFrag);
  layer.appendChild(splatFrag);
  document.documentElement.setAttribute('data-rendered', String(rows.length));
}})();
""".strip()

    return "\n".join(
        [
            '<?xml version="1.0" encoding="UTF-8"?>',
            (
                f'<svg width="{width}" height="{height}" '
                f'viewBox="0 0 {width} {height}" '
                'xmlns="http://www.w3.org/2000/svg">'
            ),
            (
                "  <desc>Compact scripted matrix-driven Gaussian splat SVG. "
                "The source stores one data row per splat and expands browser-compatible "
                "SVG gradients at load time.</desc>"
            ),
            '  <defs id="defs"><style>.splat { mix-blend-mode: normal; }</style></defs>',
            f'  <rect width="{width}" height="{height}" fill="rgb({bg_rgb[0]},{bg_rgb[1]},{bg_rgb[2]})" class="background"/>',
            '  <g id="splats"></g>',
            f'  <script id="splat-data" type="application/octet-stream">{rows}</script>',
            f"  <script><![CDATA[{script}]]></script>",
            "</svg>",
        ]
    )


def generate_palette_quantized_svg_content(
    splats: List[GaussianSplat],
    width: int,
    height: int,
    k_sigma: float = 2.5,
    background_linear_rgb: Optional[np.ndarray] = None,
    foreground_mask: Optional[np.ndarray] = None,
    background_safe_mask: Optional[np.ndarray] = None,
    edge_band_mask: Optional[np.ndarray] = None,
    palette_size: int = SVG_PALETTE_QUANTIZED_DEFAULT_SIZE,
) -> str:
    """Compact SVG that quantizes splat colors into a shared palette.

    Generates one <radialGradient> per palette color in <defs> (with the
    palette color baked into every stop) and references it per-splat via
    ``fill="url(#p{label})"``. Per-element ``opacity="..."`` scales the
    Gaussian profile to the splat's trained alpha. Works in every renderer
    that supports radial gradients (browsers, rsvg-convert, cairosvg).

    The naive "one gradient per splat" `standard` recipe writes ~400 bytes
    per splat in gradient defs alone. This recipe writes one gradient
    block per palette color (~300 bytes * N) plus a thin ~100-byte ellipse
    per splat. At 40k splats / 128 palette colors that's ~4 MB vs ~16 MB
    for the standard recipe.

    The earlier "shared-currentcolor" attempt failed because per SVG spec
    ``currentColor`` inside a paint server resolves at the gradient's
    DEFINITION context, not the reference context. Color quantization
    sidesteps the spec issue by baking real colors into shared gradients.

    Trade-offs:
    - Color quantization introduces banding when the palette is too small;
      defaults to 128 colors which is visually clean for photographic input.
      Tune via refinement_config['svg_palette_size'].
    - The shared Gaussian stop profile uses an alpha-independent shape and
      relies on per-element opacity to scale to the splat's alpha. Exact at
      stop t=0 (when element_opacity = 1-exp(-alpha)); slight underestimate
      in the falloff at high alpha. Visually negligible for typical content.
    """
    from scipy.cluster.vq import kmeans2 as _kmeans2

    def _valid_mask(mask: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if mask is None:
            return None
        arr = np.asarray(mask)
        if arr.shape != (int(height), int(width)):
            raise ValueError("SVG region masks must match output height/width")
        return arr.astype(bool, copy=False)

    foreground = _valid_mask(foreground_mask)
    background_safe = _valid_mask(background_safe_mask)
    edge_band = _valid_mask(edge_band_mask)

    bg_srgb_str: Optional[str] = None
    if background_linear_rgb is not None:
        bg = np.asarray(background_linear_rgb, dtype=np.float32).reshape(-1)
        if bg.size != 3:
            raise ValueError("background_linear_rgb must have exactly 3 components")
        bg_srgb = linear_to_srgb(np.clip(bg, 0.0, 1.0))
        bg_r, bg_g, bg_b = (int(np.clip(np.round(c * 255), 0, 255)) for c in bg_srgb)
        bg_srgb_str = f"rgb({bg_r},{bg_g},{bg_b})"

    def _splat_center(splat: GaussianSplat) -> Tuple[int, int]:
        x = int(np.clip(round(float(splat.mu[0])), 0, max(int(width) - 1, 0)))
        y = int(np.clip(round(float(splat.mu[1])), 0, max(int(height) - 1, 0)))
        return x, y

    def _in_safe_background(splat: GaussianSplat) -> bool:
        if background_safe is None:
            return False
        x, y = _splat_center(splat)
        if not bool(background_safe[y, x]):
            return False
        if foreground is not None and bool(foreground[y, x]):
            return False
        if edge_band is not None and bool(edge_band[y, x]):
            return False
        return True

    # Palette-quantize the splats' sRGB colors via k-means. Use a fixed RNG
    # seed so the same input produces the same SVG byte-for-byte.
    splat_colors_srgb = np.empty((len(splats), 3), dtype=np.float64)
    for i, splat in enumerate(splats):
        c_lin = np.clip(np.array(splat.color[:3], dtype=np.float32), 0.0, 1.0)
        splat_colors_srgb[i] = linear_to_srgb(c_lin)
    actual_palette_size = int(max(1, min(int(palette_size), len(splats))))
    if actual_palette_size >= len(splats):
        # No clustering needed; each splat is its own palette entry.
        centroids = splat_colors_srgb
        labels = np.arange(len(splats), dtype=np.int64)
    else:
        rng = np.random.default_rng(42)
        try:
            centroids, labels = _kmeans2(
                splat_colors_srgb,
                actual_palette_size,
                minit="++",
                seed=rng,
            )
        except TypeError:
            # Older scipy: kmeans2 took an int seed, not a Generator.
            centroids, labels = _kmeans2(
                splat_colors_srgb,
                actual_palette_size,
                minit="++",
                seed=42,
            )
        labels = np.asarray(labels, dtype=np.int64)
        # k-means can converge to empty clusters; replace those centroids with
        # the mean of any orphans they would have hosted. With "++" init this is
        # rare but worth guarding against.
        centroids = np.clip(centroids, 0.0, 1.0)

    # Palette-shared gradient stops use a Gaussian falloff in opacity space.
    # The palette color is baked into stop-color; the per-element opacity
    # then scales the whole splat to its trained alpha.
    footprint = ELLIPSE_OVERLAP_BOOST * k_sigma
    n_stops = 5
    stop_t = np.linspace(0.0, 1.0, n_stops)
    stop_op = np.exp(-0.5 * (stop_t * footprint) ** 2)

    svg_lines: List[str] = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" '
            'xmlns="http://www.w3.org/2000/svg">'
        ),
        (
            "  <desc>Palette-quantized Gaussian splat SVG"
            f" ({actual_palette_size} palette colors).</desc>"
        ),
        "  <defs>",
    ]
    palette_hex: List[str] = []
    for i, centroid in enumerate(centroids):
        r = int(np.clip(np.round(centroid[0] * 255), 0, 255))
        g = int(np.clip(np.round(centroid[1] * 255), 0, 255))
        b = int(np.clip(np.round(centroid[2] * 255), 0, 255))
        color_str = f"rgb({r},{g},{b})"
        palette_hex.append(color_str)
        svg_lines.append(
            f'    <radialGradient id="p{i}" cx="50%" cy="50%" r="50%" '
            'gradientUnits="objectBoundingBox">'
        )
        for t, op in zip(stop_t, stop_op):
            svg_lines.append(
                f'      <stop offset="{t * 100:.1f}%" '
                f'stop-color="{color_str}" stop-opacity="{float(op):.4f}"/>'
            )
        svg_lines.append("    </radialGradient>")
    svg_lines.extend(["  </defs>", ""])

    if bg_srgb_str is not None:
        svg_lines.append(
            f'  <rect width="{width}" height="{height}" fill="{bg_srgb_str}"/>'
        )
        svg_lines.append("")

    for splat, label in zip(splats, labels):
        eigenvals, eigenvecs = splat.eigendecomposition()
        rx = max(
            MIN_ELLIPSE_RADIUS_PX,
            ELLIPSE_OVERLAP_BOOST
            * k_sigma
            * float(np.sqrt(max(float(eigenvals[0]), 1e-8))),
        )
        ry = max(
            MIN_ELLIPSE_RADIUS_PX,
            ELLIPSE_OVERLAP_BOOST
            * k_sigma
            * float(np.sqrt(max(float(eigenvals[1]), 1e-8))),
        )
        rotation_rad = float(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
        rotation_deg = float(np.degrees(rotation_rad))
        cx = float(splat.mu[0])
        cy = float(splat.mu[1])

        alpha = float(np.clip(splat.alpha, 0.0, 1.0))
        if _in_safe_background(splat):
            alpha = min(alpha, SVG_BACKGROUND_ALPHA_CAP)
        # Per-element opacity scales the shared Gaussian profile so the
        # center pixel reaches the true alpha-over center opacity.
        element_opacity = 1.0 - math.exp(-alpha)
        if element_opacity <= 0.0:
            continue

        transform_attr = ""
        if abs(rotation_deg) > 0.1:
            transform_attr = (
                f' transform="rotate({rotation_deg:.1f} {cx:.1f} {cy:.1f})"'
            )

        svg_lines.append(
            f'  <ellipse cx="{cx:.1f}" cy="{cy:.1f}" rx="{rx:.2f}" ry="{ry:.2f}"'
            f' opacity="{element_opacity:.4f}"{transform_attr}'
            f' fill="url(#p{int(label)})"/>'
        )

    svg_lines.extend(["", "</svg>"])
    return "\n".join(svg_lines)


def generate_drawingml_slide_content(
    splats: List[GaussianSplat],
    width: int,
    height: int,
    k_sigma: float = 2.5,
    background_linear_rgb: Optional[np.ndarray] = None,
    splat_style: str = DEFAULT_PPTX_SPLAT_STYLE,
) -> str:
    """Generate PresentationML slide XML containing DrawingML ellipse shapes."""
    normalized_splat_style = _normalize_pptx_splat_style(splat_style)
    slide_width_emu = max(px_to_emu(width), 1)
    slide_height_emu = max(px_to_emu(height), 1)

    lines = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
        'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">',
        "  <p:cSld>",
        "    <p:spTree>",
        "      <p:nvGrpSpPr>",
        '        <p:cNvPr id="1" name="Slide Background"/>',
        "        <p:cNvGrpSpPr/>",
        "        <p:nvPr/>",
        "      </p:nvGrpSpPr>",
        "      <p:grpSpPr>",
        "        <a:xfrm>",
        '          <a:off x="0" y="0"/>',
        f'          <a:ext cx="{slide_width_emu}" cy="{slide_height_emu}"/>',
        '          <a:chOff x="0" y="0"/>',
        f'          <a:chExt cx="{slide_width_emu}" cy="{slide_height_emu}"/>',
        "        </a:xfrm>",
        "      </p:grpSpPr>",
    ]

    group_shape_id = 2
    lines.extend(
        _drawingml_group_start_lines(
            width_emu=slide_width_emu,
            height_emu=slide_height_emu,
            shape_id=group_shape_id,
            name="Splat Group",
        )
    )

    shape_id = group_shape_id + 1
    has_layers = any(_splat_layer(splat) is not None for splat in splats)
    if has_layers:
        by_layer: Dict[Optional[int], List[GaussianSplat]] = {}
        for splat in splats:
            by_layer.setdefault(_splat_layer(splat), []).append(splat)
        layer_ids = sorted(
            by_layer,
            key=lambda layer: (layer is None, 0 if layer is None else int(layer)),
        )
        if background_linear_rgb is not None and LAYER_BASE not in by_layer:
            layer_ids.insert(0, LAYER_BASE)

        for layer_id in layer_ids:
            layer_splats = by_layer.get(layer_id, [])
            if not layer_splats and not (
                layer_id == LAYER_BASE and background_linear_rgb is not None
            ):
                continue
            lines.extend(
                _drawingml_group_start_lines(
                    width_emu=slide_width_emu,
                    height_emu=slide_height_emu,
                    shape_id=shape_id,
                    name=f"{_layer_title(layer_id)} Layer",
                )
            )
            shape_id += 1
            if layer_id == LAYER_BASE and background_linear_rgb is not None:
                lines.extend(
                    _background_to_drawingml_shape_lines(
                        width_emu=slide_width_emu,
                        height_emu=slide_height_emu,
                        shape_id=shape_id,
                        background_linear_rgb=background_linear_rgb,
                    )
                )
                shape_id += 1
            for splat in layer_splats:
                lines.extend(
                    _splat_to_drawingml_shape_lines(
                        splat,
                        shape_id,
                        k_sigma,
                        splat_style=normalized_splat_style,
                    )
                )
                shape_id += 1
            lines.append("      </p:grpSp>")
    else:
        if background_linear_rgb is not None:
            lines.extend(
                _background_to_drawingml_shape_lines(
                    width_emu=slide_width_emu,
                    height_emu=slide_height_emu,
                    shape_id=shape_id,
                    background_linear_rgb=background_linear_rgb,
                )
            )
            shape_id += 1
        for splat in splats:
            lines.extend(
                _splat_to_drawingml_shape_lines(
                    splat,
                    shape_id,
                    k_sigma,
                    splat_style=normalized_splat_style,
                )
            )
            shape_id += 1
    lines.append("      </p:grpSp>")

    lines.extend(
        [
            "    </p:spTree>",
            "  </p:cSld>",
            "  <p:clrMapOvr>",
            "    <a:masterClrMapping/>",
            "  </p:clrMapOvr>",
            "</p:sld>",
        ]
    )

    return "\n".join(lines)


def _drawingml_group_start_lines(
    width_emu: int,
    height_emu: int,
    shape_id: int,
    name: str = "Splat Group",
) -> List[str]:
    """Create the opening XML for a native DrawingML group containing all splats."""
    return [
        "      <p:grpSp>",
        "        <p:nvGrpSpPr>",
        f'          <p:cNvPr id="{shape_id}" name="{name}"/>',
        "          <p:cNvGrpSpPr/>",
        "          <p:nvPr/>",
        "        </p:nvGrpSpPr>",
        "        <p:grpSpPr>",
        "          <a:xfrm>",
        '            <a:off x="0" y="0"/>',
        f'            <a:ext cx="{width_emu}" cy="{height_emu}"/>',
        '            <a:chOff x="0" y="0"/>',
        f'            <a:chExt cx="{width_emu}" cy="{height_emu}"/>',
        "          </a:xfrm>",
        "        </p:grpSpPr>",
    ]


def _background_to_drawingml_shape_lines(
    width_emu: int,
    height_emu: int,
    shape_id: int,
    background_linear_rgb: np.ndarray,
) -> List[str]:
    """Create a native DrawingML rectangle for the estimated canvas background."""
    bg = np.asarray(background_linear_rgb, dtype=np.float32).reshape(-1)
    if bg.size != 3:
        raise ValueError("background_linear_rgb must have exactly 3 components")
    bg_srgb = linear_to_srgb(np.clip(bg, 0.0, 1.0))
    r = int(np.clip(np.round(bg_srgb[0] * 255), 0, 255))
    g = int(np.clip(np.round(bg_srgb[1] * 255), 0, 255))
    b = int(np.clip(np.round(bg_srgb[2] * 255), 0, 255))
    color_hex = f"{r:02X}{g:02X}{b:02X}"
    return [
        "      <p:sp>",
        "        <p:nvSpPr>",
        f'          <p:cNvPr id="{shape_id}" name="Splat Background"/>',
        "          <p:cNvSpPr>",
        '            <a:spLocks noGrp="1"/>',
        "          </p:cNvSpPr>",
        "          <p:nvPr/>",
        "        </p:nvSpPr>",
        "        <p:spPr>",
        "          <a:xfrm>",
        '            <a:off x="0" y="0"/>',
        f'            <a:ext cx="{width_emu}" cy="{height_emu}"/>',
        "          </a:xfrm>",
        '          <a:prstGeom prst="rect">',
        "            <a:avLst/>",
        "          </a:prstGeom>",
        "          <a:solidFill>",
        f'            <a:srgbClr val="{color_hex}"/>',
        "          </a:solidFill>",
        "          <a:ln>",
        "            <a:noFill/>",
        "          </a:ln>",
        "        </p:spPr>",
        "        <p:txBody>",
        "          <a:bodyPr/>",
        "          <a:lstStyle/>",
        "          <a:p>",
        "            <a:endParaRPr/>",
        "          </a:p>",
        "        </p:txBody>",
        "      </p:sp>",
    ]


def _splat_geometry_for_drawingml(
    splat: GaussianSplat,
    k_sigma: float,
) -> Tuple[int, int, int, int, str, str]:
    """Return common DrawingML geometry and color fields for one splat."""
    eigenvals, eigenvecs = splat.eigendecomposition()
    rx = float(
        max(
            MIN_ELLIPSE_RADIUS_PX,
            ELLIPSE_OVERLAP_BOOST * k_sigma * np.sqrt(max(float(eigenvals[0]), 1e-8)),
        )
    )
    ry = float(
        max(
            MIN_ELLIPSE_RADIUS_PX,
            ELLIPSE_OVERLAP_BOOST * k_sigma * np.sqrt(max(float(eigenvals[1]), 1e-8)),
        )
    )
    cx, cy = float(splat.mu[0]), float(splat.mu[1])

    x = cx - rx
    y = cy - ry
    w = max(2.0 * rx, 1e-3)
    h = max(2.0 * ry, 1e-3)

    x_emu = px_to_emu(x)
    y_emu = px_to_emu(y)
    w_emu = max(px_to_emu(w), 1)
    h_emu = max(px_to_emu(h), 1)

    rotation_rad = float(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
    rotation_deg = float(np.degrees(rotation_rad))
    rotation_units = int(round(rotation_deg * 60000.0))
    rot_attr = f' rot="{rotation_units}"' if abs(rotation_units) > 0 else ""

    rgb_srgb = linear_to_srgb(np.array(splat.color[:3], dtype=np.float32))
    r = int(np.clip(np.round(rgb_srgb[0] * 255), 0, 255))
    g = int(np.clip(np.round(rgb_srgb[1] * 255), 0, 255))
    b = int(np.clip(np.round(rgb_srgb[2] * 255), 0, 255))
    color_hex = f"{r:02X}{g:02X}{b:02X}"
    return x_emu, y_emu, w_emu, h_emu, rot_attr, color_hex


def _splat_to_drawingml_shape_lines(
    splat: GaussianSplat,
    shape_id: int,
    k_sigma: float,
    splat_style: str = DEFAULT_PPTX_SPLAT_STYLE,
) -> List[str]:
    """Convert one Gaussian splat to a DrawingML ellipse shape."""
    normalized_splat_style = _normalize_pptx_splat_style(splat_style)
    if normalized_splat_style == "soft-edge":
        return _splat_to_drawingml_soft_edge_shape_lines(splat, shape_id, k_sigma)

    x_emu, y_emu, w_emu, h_emu, rot_attr, color_hex = _splat_geometry_for_drawingml(
        splat, k_sigma
    )

    # Radial gradient stops mirroring the SVG path: the renderer's per-splat
    # alpha-over opacity 1-exp(-a*exp(-0.5*(t*footprint)^2)). PowerPoint's
    # path="circle" profile is too broad for a canvas Gaussian; path="shape"
    # and a lower alpha scale matched the measured PowerPoint roundtrip best.
    alpha_clamped = float(np.clip(splat.alpha * PPTX_GRADIENT_ALPHA_SCALE, 0.0, 1.0))
    footprint = ELLIPSE_OVERLAP_BOOST * k_sigma
    gradient_stop_lines: List[str] = []
    for j in range(SVG_GRADIENT_STOPS):
        t = j / (SVG_GRADIENT_STOPS - 1)
        opacity = 1.0 - math.exp(-alpha_clamped * math.exp(-0.5 * (t * footprint) ** 2))
        pos = int(round(t * 100000.0))
        a_units = int(np.clip(round(opacity * 100000.0), 0, 100000))
        gradient_stop_lines.extend(
            [
                f'              <a:gs pos="{pos}">',
                f'                <a:srgbClr val="{color_hex}"><a:alpha val="{a_units}"/></a:srgbClr>',
                "              </a:gs>",
            ]
        )

    return [
        "      <p:sp>",
        "        <p:nvSpPr>",
        f'          <p:cNvPr id="{shape_id}" name="Splat {shape_id}"/>',
        "          <p:cNvSpPr>",
        '            <a:spLocks noGrp="1"/>',
        "          </p:cNvSpPr>",
        "          <p:nvPr/>",
        "        </p:nvSpPr>",
        "        <p:spPr>",
        f"          <a:xfrm{rot_attr}>",
        f'            <a:off x="{x_emu}" y="{y_emu}"/>',
        f'            <a:ext cx="{w_emu}" cy="{h_emu}"/>',
        "          </a:xfrm>",
        '          <a:prstGeom prst="ellipse">',
        "            <a:avLst/>",
        "          </a:prstGeom>",
        "          <a:gradFill>",
        "            <a:gsLst>",
        *gradient_stop_lines,
        "            </a:gsLst>",
        '            <a:path path="shape">',
        "            </a:path>",
        "          </a:gradFill>",
        "          <a:ln>",
        "            <a:noFill/>",
        "          </a:ln>",
        "        </p:spPr>",
        "        <p:txBody>",
        "          <a:bodyPr/>",
        "          <a:lstStyle/>",
        "          <a:p>",
        "            <a:endParaRPr/>",
        "          </a:p>",
        "        </p:txBody>",
        "      </p:sp>",
    ]


def _splat_to_drawingml_soft_edge_shape_lines(
    splat: GaussianSplat,
    shape_id: int,
    k_sigma: float,
) -> List[str]:
    """Convert one splat to a PowerPoint-friendly soft-edge native ellipse."""
    effective_k_sigma = float(k_sigma) * PPTX_SOFT_EDGE_K_SIGMA_SCALE
    x_emu, y_emu, w_emu, h_emu, rot_attr, color_hex = _splat_geometry_for_drawingml(
        splat,
        effective_k_sigma,
    )
    center_opacity = 1.0 - math.exp(-float(np.clip(splat.alpha, 0.0, 1.0)))
    alpha_units = int(
        np.clip(
            round(center_opacity * PPTX_SOFT_EDGE_ALPHA_SCALE * 100000.0), 0, 100000
        )
    )
    soft_radius = int(max(0, round(min(w_emu, h_emu) * PPTX_SOFT_EDGE_RADIUS_FACTOR)))
    return [
        "      <p:sp>",
        "        <p:nvSpPr>",
        f'          <p:cNvPr id="{shape_id}" name="Splat {shape_id}"/>',
        "          <p:cNvSpPr>",
        '            <a:spLocks noGrp="1"/>',
        "          </p:cNvSpPr>",
        "          <p:nvPr/>",
        "        </p:nvSpPr>",
        "        <p:spPr>",
        f"          <a:xfrm{rot_attr}>",
        f'            <a:off x="{x_emu}" y="{y_emu}"/>',
        f'            <a:ext cx="{w_emu}" cy="{h_emu}"/>',
        "          </a:xfrm>",
        '          <a:prstGeom prst="ellipse">',
        "            <a:avLst/>",
        "          </a:prstGeom>",
        "          <a:solidFill>",
        f'            <a:srgbClr val="{color_hex}"><a:alpha val="{alpha_units}"/></a:srgbClr>',
        "          </a:solidFill>",
        "          <a:ln>",
        "            <a:noFill/>",
        "          </a:ln>",
        "          <a:effectLst>",
        f'            <a:softEdge rad="{soft_radius}"/>',
        "          </a:effectLst>",
        "        </p:spPr>",
        "        <p:txBody>",
        "          <a:bodyPr/>",
        "          <a:lstStyle/>",
        "          <a:p>",
        "            <a:endParaRPr/>",
        "          </a:p>",
        "        </p:txBody>",
        "      </p:sp>",
    ]


def splat_to_svg_ellipse(
    splat: GaussianSplat,
    k_sigma: float = 2.5,
    element_id: Optional[str] = None,
    gradient_id: Optional[str] = None,
    radius_scale: float = 1.0,
) -> str:
    """
    Convert Gaussian splat to SVG ellipse element.

    Args:
        splat: Gaussian splat
        k_sigma: Sigma multiplier for ellipse size
        element_id: Optional element ID

    Returns:
        SVG ellipse element string
    """
    # Get eigendecomposition
    eigenvals, eigenvecs = splat.eigendecomposition()

    # Compute ellipse parameters
    # Semi-axes lengths (k * σ where σ = sqrt(eigenvalue))
    rx = max(
        MIN_ELLIPSE_RADIUS_PX,
        float(max(radius_scale, 0.0))
        * ELLIPSE_OVERLAP_BOOST
        * k_sigma
        * np.sqrt(eigenvals[0]),
    )
    ry = max(
        MIN_ELLIPSE_RADIUS_PX,
        float(max(radius_scale, 0.0))
        * ELLIPSE_OVERLAP_BOOST
        * k_sigma
        * np.sqrt(eigenvals[1]),
    )

    # Rotation angle (from first eigenvector)
    # Note: SVG rotation is in degrees, positive clockwise
    rotation_rad = np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0])
    rotation_deg = np.degrees(rotation_rad)

    # Center position
    cx, cy = splat.mu

    # Color fallback for environments that do not resolve referenced gradients.
    rgb_srgb = linear_to_srgb(np.array(splat.color[:3], dtype=np.float32))
    color_int = tuple(int(np.clip(np.round(c * 255), 0, 255)) for c in rgb_srgb)
    fallback_color = f"rgb({color_int[0]},{color_int[1]},{color_int[2]})"

    # Build ellipse element
    id_attr = f' id="{element_id}"' if element_id else ""
    transform_attr = (
        f' transform="rotate({rotation_deg:.2f} {cx:.2f} {cy:.2f})"'
        if abs(rotation_deg) > 0.1
        else ""
    )

    fill_attr = f"url(#{gradient_id})" if gradient_id else fallback_color
    alpha_attr = (
        ""
        if gradient_id
        else f' fill-opacity="{float(np.clip(splat.alpha, 0.0, 1.0)):.3f}"'
    )
    ellipse = (
        f"<ellipse{id_attr} "
        f'cx="{cx:.2f}" cy="{cy:.2f}" '
        f'rx="{rx:.2f}" ry="{ry:.2f}" '
        f'fill="{fill_attr}" '
        f'data-fallback-fill="{fallback_color}"'
        f"{alpha_attr}"
        f"{transform_attr} "
        f'class="splat"/>'
    )

    return ellipse


def estimate_svg_size(splats: List[GaussianSplat]) -> int:
    """
    Estimate SVG file size in bytes.

    Args:
        splats: List of splats

    Returns:
        Estimated file size in bytes
    """
    # Rough estimate: ~120 bytes per ellipse + overhead
    bytes_per_ellipse = 120
    overhead_bytes = 500  # Headers, etc.

    return len(splats) * bytes_per_ellipse + overhead_bytes


def save_splats_json(splats: List[GaussianSplat], output_path: str) -> None:
    """
    Save splats to canonical raw JSON format.

    Args:
        splats: List of splats
        output_path: Output JSON file path
    """
    import json

    raw_splats = [splat.to_raw_splat().to_dict() for splat in splats]

    payload = {
        "schema": RAW_SPLAT_SCHEMA_VERSION,
        "num_splats": len(raw_splats),
        "splats": raw_splats,
    }
    layer_counts: Dict[int, int] = {}
    for item in raw_splats:
        layer = item.get("layer")
        if layer is None:
            continue
        layer_counts[int(layer)] = layer_counts.get(int(layer), 0) + 1
    if layer_counts:
        payload["layers"] = [
            {
                "id": layer,
                "name": _layer_name(layer),
                "count": count,
            }
            for layer, count in sorted(layer_counts.items())
        ]

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        logger.info(f"Saved {len(splats)} splats to JSON: {output_path}")
    except Exception as e:
        logger.error(f"Failed to save JSON {output_path}: {e}")
        raise


def load_splats_json(input_path: str) -> List[GaussianSplat]:
    """
    Load splats from JSON.

    Supports canonical raw schema and a legacy schema for backward compatibility.
    """
    import json

    try:
        with open(input_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as e:
        logger.error(f"Failed to read JSON {input_path}: {e}")
        raise

    splat_items = payload.get("splats", [])
    schema = payload.get("schema")

    # Canonical raw schema
    if schema == RAW_SPLAT_SCHEMA_VERSION:
        splats: List[GaussianSplat] = []
        for item in splat_items:
            raw = RawSplat.from_dict(item)
            splats.append(GaussianSplat.from_raw_splat(raw))
        return splats

    # Legacy fallback schema
    splats_legacy: List[GaussianSplat] = []
    for item in splat_items:
        if {"mu", "sigma", "color", "alpha"}.issubset(item):
            splats_legacy.append(
                GaussianSplat(
                    mu=np.array(item["mu"], dtype=np.float32),
                    sigma=np.array(item["sigma"], dtype=np.float32),
                    color=np.array(item["color"], dtype=np.float32),
                    alpha=float(item["alpha"]),
                    importance=float(item.get("importance", 0.0)),
                )
            )
            continue

        # If schema is missing but raw fields are present, accept as canonical-like.
        raw = RawSplat.from_dict(item)
        splats_legacy.append(GaussianSplat.from_raw_splat(raw))

    return splats_legacy


def _image_ssim(x: np.ndarray, y: np.ndarray) -> float:
    """Standard windowed SSIM (the metric people mean by "SSIM").

    Uses skimage's local-window SSIM. Falls back to a global single-window SSIM
    only if skimage is unavailable -- note the global form over-reports by ~0.1-0.2
    versus windowed SSIM, so the fallback is a last resort, not an equivalent.
    """
    x = np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0)
    y = np.clip(np.asarray(y, dtype=np.float64), 0.0, 1.0)
    try:
        from skimage.metrics import structural_similarity

        channel_axis = 2 if (x.ndim == 3 and x.shape[2] > 1) else None
        return float(
            structural_similarity(x, y, channel_axis=channel_axis, data_range=1.0)
        )
    except Exception:
        logger.warning("skimage unavailable; falling back to inflated global SSIM")
        return _global_ssim_np(x, y)


def _global_ssim_np(x: np.ndarray, y: np.ndarray) -> float:
    """Global single-window SSIM (legacy fallback; over-reports vs windowed SSIM)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    c1 = 0.01**2
    c2 = 0.03**2

    mu_x = np.mean(x, axis=(0, 1))
    mu_y = np.mean(y, axis=(0, 1))
    x_centered = x - mu_x.reshape(1, 1, -1)
    y_centered = y - mu_y.reshape(1, 1, -1)
    sigma_x = np.mean(x_centered * x_centered, axis=(0, 1))
    sigma_y = np.mean(y_centered * y_centered, axis=(0, 1))
    sigma_xy = np.mean(x_centered * y_centered, axis=(0, 1))

    numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
    ssim_per_channel = numerator / np.maximum(denominator, 1e-8)
    return float(np.clip(np.mean(ssim_per_channel), -1.0, 1.0))


def compute_quality_metrics(
    target_linear_rgb: np.ndarray,
    candidate_linear_rgb: np.ndarray,
) -> Dict[str, float]:
    """Compute core quality metrics for two linear-RGB images in [0,1]."""
    target = np.clip(np.asarray(target_linear_rgb, dtype=np.float32), 0.0, 1.0)
    candidate = np.clip(np.asarray(candidate_linear_rgb, dtype=np.float32), 0.0, 1.0)
    if target.shape != candidate.shape:
        raise ValueError(
            f"Quality metric shape mismatch: target={target.shape}, candidate={candidate.shape}"
        )

    l1 = float(np.mean(np.abs(candidate - target)))
    mse = float(np.mean((candidate - target) ** 2))
    psnr = float(-10.0 * np.log10(max(mse, 1e-12)))
    ssim = _image_ssim(candidate, target)

    # Perceptual (sRGB-display) metrics: what the eye actually sees once the
    # linear-RGB values are gamma-encoded for a display. Linear-space SSIM
    # systematically over-reports quality because it de-weights mid/high tones,
    # so report both and gate on the perceptual one.
    target_srgb = linear_to_srgb(target)
    candidate_srgb = linear_to_srgb(candidate)
    mse_srgb = float(np.mean((candidate_srgb - target_srgb) ** 2))
    return {
        "l1": l1,
        "mse": mse,
        "psnr": psnr,
        "ssim": ssim,
        "psnr_srgb": float(-10.0 * np.log10(max(mse_srgb, 1e-12))),
        "ssim_srgb": _image_ssim(candidate_srgb, target_srgb),
    }


def render_splats_preview_png(
    splats: List[GaussianSplat],
    width: int,
    height: int,
    output_path: str,
    scale: float = 1.0,
    background_linear_rgb: Optional[np.ndarray] = None,
) -> str:
    """
    Render splats to a PNG preview image (linear->sRGB conversion included).
    """
    from .renderer import render_splats_numpy

    render_width = max(1, int(round(float(width) * float(scale))))
    render_height = max(1, int(round(float(height) * float(scale))))

    rendered = render_splats_numpy(
        splats,
        width,
        height,
        background_linear_rgb=background_linear_rgb,
    )
    rendered_srgb = linear_to_srgb(np.clip(rendered, 0.0, 1.0))
    image = Image.fromarray((rendered_srgb * 255.0).astype(np.uint8), mode="RGB")
    if render_width != width or render_height != height:
        image = image.resize((render_width, render_height), Image.Resampling.LANCZOS)
    image.save(output_path, format="PNG")
    return output_path


def _try_rasterize_svg_to_linear_rgb(
    svg_path: str,
    width: int,
    height: int,
) -> Tuple[Optional[np.ndarray], str]:
    """
    Try rasterizing SVG to linear-RGB float image.

    Returns `(image_or_none, method_label)`.
    """
    # Preferred backend: cairosvg (in-process).
    try:
        import cairosvg  # type: ignore

        png_bytes = cairosvg.svg2png(
            url=svg_path, output_width=int(width), output_height=int(height)
        )
        image = Image.open(io.BytesIO(png_bytes)).convert("RGB")
        srgb = np.asarray(image, dtype=np.float32) / 255.0
        return srgb_to_linear(srgb), "cairosvg"
    except Exception:
        pass

    # Fallback backend: rsvg-convert (librsvg CLI), common on macOS/Linux.
    rsvg = shutil.which("rsvg-convert")
    if rsvg is not None:
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp:
                subprocess.run(
                    [
                        rsvg,
                        "-w",
                        str(int(width)),
                        "-h",
                        str(int(height)),
                        svg_path,
                        "-o",
                        tmp.name,
                    ],
                    check=True,
                    capture_output=True,
                )
                image = Image.open(tmp.name).convert("RGB")
                srgb = np.asarray(image, dtype=np.float32) / 255.0
            return srgb_to_linear(srgb), "rsvg-convert"
        except Exception as exc:
            return None, f"error:rsvg:{type(exc).__name__}"

    return None, "unavailable:cairosvg,rsvg-convert"


def evaluate_svg_export_quality(
    target_linear_rgb: np.ndarray,
    svg_path: str,
    fallback_linear_rgb: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Grade SVG export quality against a target image.

    Tries actual SVG rasterization first; if unavailable, uses provided fallback render.
    """
    target = np.asarray(target_linear_rgb, dtype=np.float32)
    h, w = target.shape[:2]
    rendered, method = _try_rasterize_svg_to_linear_rgb(
        svg_path=svg_path, width=w, height=h
    )
    used_fallback = False

    if rendered is None and fallback_linear_rgb is not None:
        logger.warning(
            "SVG could not be rasterized (%s); export-quality metrics fall back to the "
            "numpy proxy render, which does NOT reflect real SVG fidelity. Install cairosvg "
            "or rsvg-convert to measure the actual SVG.",
            method,
        )
        rendered = np.asarray(fallback_linear_rgb, dtype=np.float32)
        method = "proxy-fallback"
        used_fallback = True

    if rendered is None:
        return {
            "available": False,
            "method": method,
            "used_fallback": False,
            "metrics": None,
        }

    metrics = compute_quality_metrics(target, rendered)
    return {
        "available": True,
        "method": method,
        "used_fallback": bool(used_fallback),
        "metrics": metrics,
    }


def _pptx_content_types_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Default Extension="png" ContentType="image/png"/>
  <Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>
  <Override PartName="/ppt/slides/slide1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>
  <Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/>
  <Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/>
  <Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>
  <Override PartName="/ppt/presProps.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presProps+xml"/>
  <Override PartName="/ppt/viewProps.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.viewProps+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>
"""


def _pptx_root_rels_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>
"""


def _pptx_core_props_xml(now_iso: str) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
                   xmlns:dc="http://purl.org/dc/elements/1.1/"
                   xmlns:dcterms="http://purl.org/dc/terms/"
                   xmlns:dcmitype="http://purl.org/dc/dcmitype/"
                   xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>SplatThis Export</dc:title>
  <dc:creator>SplatThis</dc:creator>
  <cp:lastModifiedBy>SplatThis</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">{now_iso}</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">{now_iso}</dcterms:modified>
</cp:coreProperties>
"""


def _pptx_app_props_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"
            xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>SplatThis</Application>
  <Slides>1</Slides>
  <PresentationFormat>Custom</PresentationFormat>
</Properties>
"""


def _pptx_presentation_xml(slide_cx: int, slide_cy: int) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
                xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
                xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:sldMasterIdLst>
    <p:sldMasterId id="2147483648" r:id="rId1"/>
  </p:sldMasterIdLst>
  <p:sldIdLst>
    <p:sldId id="256" r:id="rId2"/>
  </p:sldIdLst>
  <p:sldSz cx="{slide_cx}" cy="{slide_cy}" type="custom"/>
  <p:notesSz cx="6858000" cy="9144000"/>
  <p:defaultTextStyle/>
</p:presentation>
"""


def _pptx_presentation_rels_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="slideMasters/slideMaster1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide1.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/presProps" Target="presProps.xml"/>
  <Relationship Id="rId4" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/viewProps" Target="viewProps.xml"/>
  <Relationship Id="rId5" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="theme/theme1.xml"/>
</Relationships>
"""


def _pptx_pres_props_xml() -> str:
    """Minimal presentationPr part. PowerPoint emits its 'unreadable content'
    repair dialog when this relationship is absent, even when everything else
    is valid (per openxml-audit semantic check)."""
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:presentationPr xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
                  xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
                  xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"/>
"""


def _pptx_view_props_xml() -> str:
    """Minimal viewPr part. Same repair-dialog trigger as presProps."""
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:viewPr xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
          xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
          xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"/>
"""


def _pptx_slide_xml(slide_cx: int, slide_cy: int) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
       xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
       xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld>
    <p:spTree>
      <p:nvGrpSpPr>
        <p:cNvPr id="1" name=""/>
        <p:cNvGrpSpPr/>
        <p:nvPr/>
      </p:nvGrpSpPr>
      <p:grpSpPr>
        <a:xfrm>
          <a:off x="0" y="0"/>
          <a:ext cx="0" cy="0"/>
          <a:chOff x="0" y="0"/>
          <a:chExt cx="0" cy="0"/>
        </a:xfrm>
      </p:grpSpPr>
      <p:pic>
        <p:nvPicPr>
          <p:cNvPr id="2" name="Splat PNG"/>
          <p:cNvPicPr>
            <a:picLocks noChangeAspect="1"/>
          </p:cNvPicPr>
          <p:nvPr/>
        </p:nvPicPr>
        <p:blipFill>
          <a:blip r:embed="rId1"/>
          <a:stretch>
            <a:fillRect/>
          </a:stretch>
        </p:blipFill>
        <p:spPr>
          <a:xfrm>
            <a:off x="0" y="0"/>
            <a:ext cx="{slide_cx}" cy="{slide_cy}"/>
          </a:xfrm>
          <a:prstGeom prst="rect">
            <a:avLst/>
          </a:prstGeom>
        </p:spPr>
      </p:pic>
    </p:spTree>
  </p:cSld>
  <p:clrMapOvr>
    <a:masterClrMapping/>
  </p:clrMapOvr>
</p:sld>
"""


def _pptx_slide_rels_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="../media/image1.png"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>
</Relationships>
"""


def _pptx_vector_slide_rels_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>
</Relationships>
"""


def _pptx_slide_layout_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldLayout xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
             xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
             xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"
             type="blank" preserve="1">
  <p:cSld name="Blank">
    <p:spTree>
      <p:nvGrpSpPr>
        <p:cNvPr id="1" name=""/>
        <p:cNvGrpSpPr/>
        <p:nvPr/>
      </p:nvGrpSpPr>
      <p:grpSpPr>
        <a:xfrm>
          <a:off x="0" y="0"/>
          <a:ext cx="0" cy="0"/>
          <a:chOff x="0" y="0"/>
          <a:chExt cx="0" cy="0"/>
        </a:xfrm>
      </p:grpSpPr>
    </p:spTree>
  </p:cSld>
  <p:clrMapOvr>
    <a:masterClrMapping/>
  </p:clrMapOvr>
</p:sldLayout>
"""


def _pptx_slide_layout_rels_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="../slideMasters/slideMaster1.xml"/>
</Relationships>
"""


def _pptx_slide_master_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldMaster xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
             xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
             xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld>
    <p:bg>
      <p:bgRef idx="1001">
        <a:schemeClr val="bg1"/>
      </p:bgRef>
    </p:bg>
    <p:spTree>
      <p:nvGrpSpPr>
        <p:cNvPr id="1" name=""/>
        <p:cNvGrpSpPr/>
        <p:nvPr/>
      </p:nvGrpSpPr>
      <p:grpSpPr>
        <a:xfrm>
          <a:off x="0" y="0"/>
          <a:ext cx="0" cy="0"/>
          <a:chOff x="0" y="0"/>
          <a:chExt cx="0" cy="0"/>
        </a:xfrm>
      </p:grpSpPr>
    </p:spTree>
  </p:cSld>
  <p:clrMap bg1="lt1" tx1="dk1" bg2="lt2" tx2="dk2"
            accent1="accent1" accent2="accent2" accent3="accent3"
            accent4="accent4" accent5="accent5" accent6="accent6"
            hlink="hlink" folHlink="folHlink"/>
  <p:sldLayoutIdLst>
    <p:sldLayoutId id="2147483649" r:id="rId1"/>
  </p:sldLayoutIdLst>
  <p:txStyles>
    <p:titleStyle/>
    <p:bodyStyle/>
    <p:otherStyle/>
  </p:txStyles>
</p:sldMaster>
"""


def _pptx_slide_master_rels_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="../theme/theme1.xml"/>
</Relationships>
"""


def _pptx_theme_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="SplatThis Theme">
  <a:themeElements>
    <a:clrScheme name="SplatThis">
      <a:dk1><a:sysClr val="windowText" lastClr="000000"/></a:dk1>
      <a:lt1><a:sysClr val="window" lastClr="FFFFFF"/></a:lt1>
      <a:dk2><a:srgbClr val="1F1F1F"/></a:dk2>
      <a:lt2><a:srgbClr val="F2F2F2"/></a:lt2>
      <a:accent1><a:srgbClr val="4F81BD"/></a:accent1>
      <a:accent2><a:srgbClr val="C0504D"/></a:accent2>
      <a:accent3><a:srgbClr val="9BBB59"/></a:accent3>
      <a:accent4><a:srgbClr val="8064A2"/></a:accent4>
      <a:accent5><a:srgbClr val="4BACC6"/></a:accent5>
      <a:accent6><a:srgbClr val="F79646"/></a:accent6>
      <a:hlink><a:srgbClr val="0000FF"/></a:hlink>
      <a:folHlink><a:srgbClr val="800080"/></a:folHlink>
    </a:clrScheme>
    <a:fontScheme name="SplatThis">
      <a:majorFont>
        <a:latin typeface="Calibri"/>
        <a:ea typeface=""/>
        <a:cs typeface=""/>
      </a:majorFont>
      <a:minorFont>
        <a:latin typeface="Calibri"/>
        <a:ea typeface=""/>
        <a:cs typeface=""/>
      </a:minorFont>
    </a:fontScheme>
    <a:fmtScheme name="SplatThis">
      <a:fillStyleLst>
        <a:solidFill><a:schemeClr val="phClr"/></a:solidFill>
        <a:solidFill><a:schemeClr val="phClr"/></a:solidFill>
        <a:solidFill><a:schemeClr val="phClr"/></a:solidFill>
      </a:fillStyleLst>
      <a:lnStyleLst>
        <a:ln w="9525" cap="flat" cmpd="sng" algn="ctr"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln>
        <a:ln w="19050" cap="flat" cmpd="sng" algn="ctr"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln>
        <a:ln w="38100" cap="flat" cmpd="sng" algn="ctr"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln>
      </a:lnStyleLst>
      <a:effectStyleLst>
        <a:effectStyle><a:effectLst/></a:effectStyle>
        <a:effectStyle><a:effectLst/></a:effectStyle>
        <a:effectStyle><a:effectLst/></a:effectStyle>
      </a:effectStyleLst>
      <a:bgFillStyleLst>
        <a:solidFill><a:schemeClr val="phClr"/></a:solidFill>
        <a:solidFill><a:schemeClr val="phClr"/></a:solidFill>
        <a:solidFill><a:schemeClr val="phClr"/></a:solidFill>
      </a:bgFillStyleLst>
    </a:fmtScheme>
  </a:themeElements>
  <a:objectDefaults/>
  <a:extraClrSchemeLst/>
</a:theme>
"""


def save_pptx_with_splat_png(
    splats: List[GaussianSplat],
    width: int,
    height: int,
    output_path: str,
    sort_mode: str = DEFAULT_EXPORT_ORDER,
    sort_by_area: bool = False,
    render_scale: float = 1.0,
    background_linear_rgb: Optional[np.ndarray] = None,
) -> None:
    """
    Save a self-contained PPTX containing one slide with a rendered splat PNG.
    """
    ordered_splats = _sort_splats_for_export(
        splats=splats,
        sort_mode=sort_mode,
        sort_by_area=sort_by_area,
    )

    from .renderer import render_splats_numpy

    render_width = max(1, int(round(float(width) * float(render_scale))))
    render_height = max(1, int(round(float(height) * float(render_scale))))
    rendered = render_splats_numpy(
        ordered_splats,
        width=width,
        height=height,
        background_linear_rgb=background_linear_rgb,
    )
    rendered_srgb = linear_to_srgb(np.clip(rendered, 0.0, 1.0))
    image = Image.fromarray((rendered_srgb * 255.0).astype(np.uint8), mode="RGB")
    if (render_width, render_height) != (width, height):
        image = image.resize((render_width, render_height), Image.Resampling.LANCZOS)

    png_buffer = io.BytesIO()
    image.save(png_buffer, format="PNG")
    png_bytes = png_buffer.getvalue()

    slide_cx = max(px_to_emu(render_width), 1)
    slide_cy = max(px_to_emu(render_height), 1)
    now_iso = (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )

    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", _pptx_content_types_xml())
        zf.writestr("_rels/.rels", _pptx_root_rels_xml())
        zf.writestr("docProps/core.xml", _pptx_core_props_xml(now_iso))
        zf.writestr("docProps/app.xml", _pptx_app_props_xml())
        zf.writestr(
            "ppt/presentation.xml",
            _pptx_presentation_xml(slide_cx=slide_cx, slide_cy=slide_cy),
        )
        zf.writestr("ppt/_rels/presentation.xml.rels", _pptx_presentation_rels_xml())
        zf.writestr(
            "ppt/slides/slide1.xml",
            _pptx_slide_xml(slide_cx=slide_cx, slide_cy=slide_cy),
        )
        zf.writestr("ppt/slides/_rels/slide1.xml.rels", _pptx_slide_rels_xml())
        zf.writestr("ppt/slideLayouts/slideLayout1.xml", _pptx_slide_layout_xml())
        zf.writestr(
            "ppt/slideLayouts/_rels/slideLayout1.xml.rels",
            _pptx_slide_layout_rels_xml(),
        )
        zf.writestr("ppt/slideMasters/slideMaster1.xml", _pptx_slide_master_xml())
        zf.writestr(
            "ppt/slideMasters/_rels/slideMaster1.xml.rels",
            _pptx_slide_master_rels_xml(),
        )
        zf.writestr("ppt/theme/theme1.xml", _pptx_theme_xml())
        zf.writestr("ppt/presProps.xml", _pptx_pres_props_xml())
        zf.writestr("ppt/viewProps.xml", _pptx_view_props_xml())
        zf.writestr("ppt/media/image1.png", png_bytes)

    logger.info("Saved PPTX with rasterized splat image: %s", output_path)


def save_pptx_with_splats(
    splats: List[GaussianSplat],
    width: int,
    height: int,
    output_path: str,
    k_sigma: float = 2.5,
    sort_mode: str = DEFAULT_EXPORT_ORDER,
    sort_by_area: bool = False,
    background_linear_rgb: Optional[np.ndarray] = None,
    splat_style: str = DEFAULT_PPTX_SPLAT_STYLE,
) -> None:
    """
    Save a self-contained PPTX containing native DrawingML splat shapes.

    This is the real vector-PPTX path: it writes one ellipse shape per splat
    into `ppt/slides/slide1.xml` and does not embed a raster preview image.
    """
    ordered_splats = _sort_splats_for_export(
        splats=splats,
        sort_mode=sort_mode,
        sort_by_area=sort_by_area,
    )
    slide_cx = max(px_to_emu(width), 1)
    slide_cy = max(px_to_emu(height), 1)
    slide_xml = generate_drawingml_slide_content(
        ordered_splats,
        width=width,
        height=height,
        k_sigma=k_sigma,
        background_linear_rgb=background_linear_rgb,
        splat_style=splat_style,
    )
    now_iso = (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )

    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", _pptx_content_types_xml())
        zf.writestr("_rels/.rels", _pptx_root_rels_xml())
        zf.writestr("docProps/core.xml", _pptx_core_props_xml(now_iso))
        zf.writestr("docProps/app.xml", _pptx_app_props_xml())
        zf.writestr(
            "ppt/presentation.xml",
            _pptx_presentation_xml(slide_cx=slide_cx, slide_cy=slide_cy),
        )
        zf.writestr("ppt/_rels/presentation.xml.rels", _pptx_presentation_rels_xml())
        zf.writestr("ppt/slides/slide1.xml", slide_xml)
        zf.writestr("ppt/slides/_rels/slide1.xml.rels", _pptx_vector_slide_rels_xml())
        zf.writestr("ppt/slideLayouts/slideLayout1.xml", _pptx_slide_layout_xml())
        zf.writestr(
            "ppt/slideLayouts/_rels/slideLayout1.xml.rels",
            _pptx_slide_layout_rels_xml(),
        )
        zf.writestr("ppt/slideMasters/slideMaster1.xml", _pptx_slide_master_xml())
        zf.writestr(
            "ppt/slideMasters/_rels/slideMaster1.xml.rels",
            _pptx_slide_master_rels_xml(),
        )
        zf.writestr("ppt/theme/theme1.xml", _pptx_theme_xml())
        zf.writestr("ppt/presProps.xml", _pptx_pres_props_xml())
        zf.writestr("ppt/viewProps.xml", _pptx_view_props_xml())

    logger.info(
        "Saved PPTX with %s native splat shapes: %s", len(ordered_splats), output_path
    )


def save_side_by_side_html(
    output_path: str,
    source_png_path: str,
    svg_path: str,
    preview_png_path: Optional[str] = None,
    title: str = "PNG2Splat Side-by-Side",
    metrics: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save a side-by-side HTML comparison page (source, SVG, preview PNG).
    """
    output_dir = Path(output_path).resolve().parent

    def _rel(path_value: Optional[str]) -> str:
        if not path_value:
            return ""
        p = Path(path_value).resolve()
        try:
            return p.relative_to(output_dir).as_posix()
        except Exception:
            return p.as_posix()

    source_ref = _rel(source_png_path)
    svg_ref = _rel(svg_path)
    preview_ref = _rel(preview_png_path) if preview_png_path else ""
    svg_view = (
        f'<img src="{svg_ref}" alt="SVG Splats" />'
        if svg_ref
        else '<div style="padding:16px;color:#9ba5bc;">No SVG artifact for this run.</div>'
    )
    preview_view = (
        f'<img src="{preview_ref}" alt="Splat Preview PNG" />'
        if preview_ref
        else '<div style="padding:16px;color:#9ba5bc;">No preview PNG generated.</div>'
    )

    metrics_rows: List[str] = []
    if metrics:
        for key, value in metrics.items():
            if isinstance(value, dict):
                metrics_rows.append(
                    f"<tr><td colspan='2'><strong>{key}</strong></td></tr>"
                )
                for sub_key, sub_val in value.items():
                    metrics_rows.append(
                        f"<tr><td>{key}.{sub_key}</td><td>{sub_val}</td></tr>"
                    )
            else:
                metrics_rows.append(f"<tr><td>{key}</td><td>{value}</td></tr>")
    metrics_table = (
        "<table>" + "".join(metrics_rows) + "</table>"
        if metrics_rows
        else "<p>No metrics recorded.</p>"
    )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <style>
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #0f1117; color: #e8ebf2; }}
    .wrap {{ max-width: 1800px; margin: 0 auto; padding: 16px; }}
    h1 {{ margin: 0 0 8px 0; font-size: 24px; }}
    .grid {{ display: grid; grid-template-columns: repeat(3, minmax(280px, 1fr)); gap: 12px; }}
    .card {{ background: #171b24; border: 1px solid #2b3344; border-radius: 10px; overflow: hidden; }}
    .card h2 {{ margin: 0; padding: 10px 12px; font-size: 15px; border-bottom: 1px solid #2b3344; }}
    .view {{ background: #0b0e14; min-height: 420px; display: flex; align-items: center; justify-content: center; }}
    .view img, .view object {{ width: 100%; height: 420px; object-fit: contain; background: #fff; }}
    .metrics {{ margin-top: 12px; background: #171b24; border: 1px solid #2b3344; border-radius: 10px; padding: 12px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    td {{ border-bottom: 1px solid #2b3344; padding: 6px 4px; vertical-align: top; }}
    td:first-child {{ color: #9ba5bc; width: 48%; }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>{title}</h1>
    <div class="grid">
      <div class="card">
        <h2>Source PNG</h2>
        <div class="view"><img src="{source_ref}" alt="Source PNG" /></div>
      </div>
      <div class="card">
        <h2>SVG Splats</h2>
        <div class="view">{svg_view}</div>
      </div>
      <div class="card">
        <h2>Splat Preview PNG</h2>
        <div class="view">{preview_view}</div>
      </div>
    </div>
    <div class="metrics">
      <h2>Metrics</h2>
      {metrics_table}
    </div>
  </div>
</body>
</html>
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)


def validate_export_roundtrip(
    splats: List[GaussianSplat],
    width: int,
    height: int,
    k_sigma: float = 2.5,
    atol: float = 1e-4,
) -> Dict[str, Any]:
    """
    Validate exporter round-trip integrity.

    Steps:
    1. Gaussian splats -> canonical raw dict.
    2. Raw dict -> Gaussian splats.
    3. Generate SVG and DrawingML from reconstructed splats.
    4. Compare key parameters and basic export structure counts.
    """
    raw_dicts = [s.to_raw_splat().to_dict() for s in splats]
    reconstructed = [
        GaussianSplat.from_raw_splat(RawSplat.from_dict(d)) for d in raw_dicts
    ]

    if len(splats) == 0:
        svg_content = generate_svg_content(reconstructed, width, height, k_sigma)
        dml_content = generate_drawingml_slide_content(
            reconstructed, width, height, k_sigma
        )
        return {
            "pass": True,
            "num_splats": 0,
            "max_mu_delta": 0.0,
            "max_color_delta": 0.0,
            "max_alpha_delta": 0.0,
            "svg_ellipse_count": svg_content.count("<ellipse"),
            "svg_gradient_count": svg_content.count("<radialGradient"),
            "drawingml_shape_count": dml_content.count("<p:sp>"),
        }

    mu_delta = 0.0
    color_delta = 0.0
    alpha_delta = 0.0
    for original, restored in zip(splats, reconstructed):
        mu_delta = max(
            mu_delta, float(np.max(np.abs(original.mu[:2] - restored.mu[:2])))
        )
        color_delta = max(
            color_delta, float(np.max(np.abs(original.color[:3] - restored.color[:3])))
        )
        alpha_delta = max(
            alpha_delta, float(abs(float(original.alpha) - float(restored.alpha)))
        )

    svg_content = generate_svg_content(reconstructed, width, height, k_sigma)
    dml_content = generate_drawingml_slide_content(
        reconstructed, width, height, k_sigma
    )

    svg_count = svg_content.count("<ellipse")
    svg_gradient_count = svg_content.count("<radialGradient")
    dml_count = dml_content.count("<p:sp>")
    passed = (
        mu_delta <= atol
        and color_delta <= atol
        and alpha_delta <= atol
        and svg_count == len(splats)
        and svg_gradient_count == len(splats)
        and dml_count == len(splats)
    )

    return {
        "pass": bool(passed),
        "num_splats": len(splats),
        "max_mu_delta": float(mu_delta),
        "max_color_delta": float(color_delta),
        "max_alpha_delta": float(alpha_delta),
        "svg_ellipse_count": int(svg_count),
        "svg_gradient_count": int(svg_gradient_count),
        "drawingml_shape_count": int(dml_count),
        "atol": float(atol),
    }
