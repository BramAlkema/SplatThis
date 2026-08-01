"""Browser-native CSS and Canvas splat scene emitters."""

from __future__ import annotations

import math
from html import escape as escape_html
from typing import Any, Dict, List, Optional

import numpy as np
import numpy.typing as npt

from .color import linear_to_srgb
from .export_common import (
    ELLIPSE_OVERLAP_BOOST,
    MIN_ELLIPSE_RADIUS_PX,
    _adaptive_gradient_stops,
    _density_aware_stop_error,
    _sort_splats_for_export,
    _splat_layer,
)
from .splat import LAYER_BASE, LAYER_MASS, GaussianSplat


def generate_css_splat_html(
    splats: List[GaussianSplat],
    width: int,
    height: int,
    background_linear_rgb: Optional[npt.NDArray[Any]] = None,
    title: str = "SplatThis CSS",
    parallax_strength: float = 0.0,
    hover_grid_size: int = 10,
    k_sigma: float = 2.5,
) -> str:
    """Emit a scriptless HTML compositor made from CSS gradient splats.

    Every Gaussian is an absolutely positioned, rotated DOM ellipse whose
    radial-gradient stops approximate the same per-splat opacity curve as the
    standard SVG exporter.  When ``parallax_strength`` is positive, splats are
    grouped into three depth planes and a transparent hover grid drives the
    transforms with sibling CSS selectors.  No JavaScript, canvas, SVG, or
    embedded bitmap is required at runtime.

    CSS compositing, like browser SVG compositing, happens in display sRGB.  It
    is therefore a browser-native alternative to SVG, not a replacement for
    the Canvas target's explicit linear-light pixel compositor.
    """

    if int(width) <= 0 or int(height) <= 0:
        raise ValueError("CSS compositor width and height must be positive integers")
    if float(parallax_strength) < 0.0:
        raise ValueError("CSS parallax strength must be non-negative")
    if not 1 <= int(hover_grid_size) <= 20:
        raise ValueError("CSS hover grid size must be between 1 and 20")
    if float(k_sigma) <= 0.0:
        raise ValueError("CSS k_sigma must be positive")

    width = int(width)
    height = int(height)
    hover_grid_size = int(hover_grid_size)
    parallax_strength = float(parallax_strength)

    bg_linear = (
        np.zeros(3, dtype=np.float32)
        if background_linear_rgb is None
        else np.asarray(background_linear_rgb, dtype=np.float32).reshape(-1)
    )
    if bg_linear.size != 3:
        raise ValueError("background_linear_rgb must have exactly 3 components")
    bg_srgb = linear_to_srgb(np.clip(bg_linear, 0.0, 1.0))
    bg_rgb = tuple(int(np.clip(np.round(channel * 255), 0, 255)) for channel in bg_srgb)
    background_css = f"rgb({bg_rgb[0]},{bg_rgb[1]},{bg_rgb[2]})"

    ordered_splats = _sort_splats_for_export(splats)
    gradient_footprint = ELLIPSE_OVERLAP_BOOST * float(k_sigma)
    stop_error = _density_aware_stop_error(len(ordered_splats))

    def _plane_for(splat: GaussianSplat) -> str:
        layer = _splat_layer(splat)
        if layer is not None and layer <= LAYER_BASE:
            return "background"
        if layer is None or layer == LAYER_MASS:
            return "midground"
        return "foreground"

    def _splat_element(splat: GaussianSplat, index: int) -> str:
        eigenvals, eigenvecs = splat.eigendecomposition()
        rx = max(
            MIN_ELLIPSE_RADIUS_PX,
            gradient_footprint * math.sqrt(max(float(eigenvals[0]), 1e-8)),
        )
        ry = max(
            MIN_ELLIPSE_RADIUS_PX,
            gradient_footprint * math.sqrt(max(float(eigenvals[1]), 1e-8)),
        )
        rotation = math.degrees(
            math.atan2(float(eigenvecs[1, 0]), float(eigenvecs[0, 0]))
        )
        color_srgb = linear_to_srgb(
            np.clip(np.asarray(splat.color[:3], dtype=np.float32), 0.0, 1.0)
        )
        color = tuple(
            int(np.clip(np.round(channel * 255), 0, 255)) for channel in color_srgb
        )
        stops = _adaptive_gradient_stops(
            float(np.clip(splat.alpha, 0.0, 1.0)),
            gradient_footprint,
            1.0,
            max_error=stop_error,
        )
        gradient = ",".join(
            f"rgba({color[0]},{color[1]},{color[2]},{opacity:.2f}) {offset * 100:.1f}%"
            for offset, opacity in stops
        )
        cx, cy = (float(splat.mu[0]), float(splat.mu[1]))
        style = (
            f"left:{cx:.2f}px;top:{cy:.2f}px;"
            f"width:{2.0 * rx:.2f}px;height:{2.0 * ry:.2f}px;"
            f"transform:translate(-50%,-50%) rotate({rotation:.2f}deg);"
            f"background:radial-gradient(ellipse 50% 50% at center,{gradient})"
        )
        return f'<i class="splat" data-splat="{index}" style="{style}"></i>'

    plane_names = ("background", "midground", "foreground")
    planes: Dict[str, List[str]] = {name: [] for name in plane_names}
    for index, splat in enumerate(ordered_splats):
        target_plane = _plane_for(splat) if parallax_strength > 0.0 else "midground"
        planes[target_plane].append(_splat_element(splat, index))

    css_lines = [
        "*{box-sizing:border-box}",
        (
            f"html,body{{margin:0;width:{width}px;height:{height}px;"
            "overflow:hidden;background:transparent}"
        ),
        (
            f"#scene{{position:relative;width:{width}px;height:{height}px;"
            f"overflow:hidden;background:{background_css};isolation:isolate}}"
        ),
        (
            ".plane{position:absolute;inset:0;pointer-events:none;"
            "transform-origin:center;transition:transform 240ms "
            "cubic-bezier(.2,.7,.3,1)}"
        ),
        ".plane-background{z-index:1}.plane-midground{z-index:2}.plane-foreground{z-index:3}",
        (
            ".splat{position:absolute;display:block;border-radius:50%;"
            "pointer-events:none;transform-origin:center;mix-blend-mode:normal}"
        ),
    ]

    hit_cells: List[str] = []
    if parallax_strength > 0.0:
        cell_size = 100.0 / float(hover_grid_size)
        css_lines.append(
            ".depth-hit{position:absolute;z-index:10;display:block;background:transparent}"
        )
        for row in range(hover_grid_size):
            for column in range(hover_grid_size):
                index = row * hover_grid_size + column
                x_normalized = ((column + 0.5) / hover_grid_size - 0.5) * 2.0
                y_normalized = ((row + 0.5) / hover_grid_size - 0.5) * 2.0
                # Move the scene opposite to the pointer, as if the viewer were
                # looking around foreground objects rather than dragging them.
                mid_x = -x_normalized * parallax_strength * 0.4
                mid_y = -y_normalized * parallax_strength * 0.4
                fore_x = -x_normalized * parallax_strength
                fore_y = -y_normalized * parallax_strength
                css_lines.extend(
                    [
                        (
                            f".h{index}:hover~.plane-midground{{"
                            f"transform:translate3d({mid_x:.2f}px,{mid_y:.2f}px,0)}}"
                        ),
                        (
                            f".h{index}:hover~.plane-foreground{{"
                            f"transform:translate3d({fore_x:.2f}px,{fore_y:.2f}px,0)}}"
                        ),
                    ]
                )
                hit_cells.append(
                    f'<b class="depth-hit h{index}" aria-hidden="true" '
                    f'style="left:{column * cell_size:.3f}%;top:{row * cell_size:.3f}%;'
                    f'width:{cell_size:.3f}%;height:{cell_size:.3f}%"></b>'
                )

    plane_html = []
    for plane_name in plane_names:
        if parallax_strength <= 0.0 and plane_name != "midground":
            continue
        plane_html.append(
            f'<div class="plane plane-{plane_name}" data-depth="{plane_name}">'
            + "".join(planes[plane_name])
            + "</div>"
        )

    safe_title = escape_html(title)
    grid_value = hover_grid_size if parallax_strength > 0.0 else 0
    return (
        "<!doctype html>\n"
        f'<html><head><meta charset="utf-8"><title>{safe_title}</title>'
        f"<style>{''.join(css_lines)}</style></head>\n"
        f'<body><main id="scene" data-compositor="css-splats" '
        f'data-splat-count="{len(ordered_splats)}" data-grid="{grid_value}">'
        + "".join(hit_cells)
        + "".join(plane_html)
        + "</main></body></html>\n"
    )


def generate_native_canvas_html(
    splats: List[GaussianSplat],
    width: int,
    height: int,
    background_linear_rgb: Optional[npt.NDArray[Any]] = None,
    title: str = "SplatThis Canvas",
    parallax_strength: float = 0.0,
    k_sigma: float = 2.5,
) -> str:
    """Emit browser-native Canvas radial-gradient splats.

    Unlike :func:`generate_pixel_runtime_html`, this function never computes or
    uploads an ``ImageData`` framebuffer. JavaScript submits one transformed
    radial-gradient circle per Gaussian to the Canvas 2D API, so Chromium owns
    gradient interpolation, antialiasing, and source-over compositing. Optional
    parallax uses three pre-rendered Canvas planes.
    """

    import json

    if int(width) <= 0 or int(height) <= 0:
        raise ValueError("Canvas width and height must be positive integers")
    if float(parallax_strength) < 0.0:
        raise ValueError("Canvas parallax strength must be non-negative")
    if float(k_sigma) <= 0.0:
        raise ValueError("Canvas k_sigma must be positive")

    width = int(width)
    height = int(height)
    parallax_strength = float(parallax_strength)
    ordered_splats = _sort_splats_for_export(splats)
    gradient_footprint = ELLIPSE_OVERLAP_BOOST * float(k_sigma)
    stop_error = _density_aware_stop_error(len(ordered_splats))

    bg_linear = (
        np.zeros(3, dtype=np.float32)
        if background_linear_rgb is None
        else np.asarray(background_linear_rgb, dtype=np.float32).reshape(-1)
    )
    if bg_linear.size != 3:
        raise ValueError("background_linear_rgb must have exactly 3 components")
    bg_srgb = linear_to_srgb(np.clip(bg_linear, 0.0, 1.0))
    bg_rgb = [int(np.clip(np.round(channel * 255), 0, 255)) for channel in bg_srgb]

    def _plane_for(splat: GaussianSplat) -> str:
        layer = _splat_layer(splat)
        if layer is not None and layer <= LAYER_BASE:
            return "background"
        if layer is None or layer == LAYER_MASS:
            return "midground"
        return "foreground"

    def _record(splat: GaussianSplat) -> List[Any]:
        eigenvals, eigenvecs = splat.eigendecomposition()
        rx = max(
            MIN_ELLIPSE_RADIUS_PX,
            gradient_footprint * math.sqrt(max(float(eigenvals[0]), 1e-8)),
        )
        ry = max(
            MIN_ELLIPSE_RADIUS_PX,
            gradient_footprint * math.sqrt(max(float(eigenvals[1]), 1e-8)),
        )
        rotation = math.atan2(float(eigenvecs[1, 0]), float(eigenvecs[0, 0]))
        color_srgb = linear_to_srgb(
            np.clip(np.asarray(splat.color[:3], dtype=np.float32), 0.0, 1.0)
        )
        color = [
            int(np.clip(np.round(channel * 255), 0, 255)) for channel in color_srgb
        ]
        stops = _adaptive_gradient_stops(
            float(np.clip(splat.alpha, 0.0, 1.0)),
            gradient_footprint,
            1.0,
            max_error=stop_error,
        )
        return [
            round(float(splat.mu[0]), 4),
            round(float(splat.mu[1]), 4),
            round(float(rx), 4),
            round(float(ry), 4),
            round(float(rotation), 6),
            *color,
            [[round(offset, 6), round(opacity, 4)] for offset, opacity in stops],
        ]

    if parallax_strength > 0.0:
        plane_records: Dict[str, List[List[Any]]] = {
            "background": [],
            "midground": [],
            "foreground": [],
        }
        for splat in ordered_splats:
            plane_records[_plane_for(splat)].append(_record(splat))
        planes = [
            {
                "name": "background",
                "depth": 0.0,
                "splats": plane_records["background"],
            },
            {
                "name": "midground",
                "depth": 0.4,
                "splats": plane_records["midground"],
            },
            {
                "name": "foreground",
                "depth": 1.0,
                "splats": plane_records["foreground"],
            },
        ]
    else:
        planes = [
            {
                "name": "scene",
                "depth": 0.0,
                "splats": [_record(splat) for splat in ordered_splats],
            }
        ]

    plane_json = json.dumps(planes, separators=(",", ":"))
    js = (
        r"""
(() => {
  const t0 = performance.now();
  const W = __W__, H = __H__, BG = __BG__;
  const STRENGTH = __STRENGTH__, PLANES = __PLANES__;
  const scene = document.getElementById('scene');
  const canvases = [];

  function drawSplat(ctx, splat) {
    const [x, y, rx, ry, theta, r, g, b, stops] = splat;
    ctx.save();
    ctx.translate(x, y);
    ctx.rotate(theta);
    ctx.scale(rx, ry);
    const gradient = ctx.createRadialGradient(0, 0, 0, 0, 0, 1);
    for (const [offset, opacity] of stops) {
      gradient.addColorStop(offset, `rgba(${r},${g},${b},${opacity})`);
    }
    ctx.fillStyle = gradient;
    ctx.beginPath();
    ctx.arc(0, 0, 1, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  }

  for (const plane of PLANES) {
    const canvas = document.createElement('canvas');
    canvas.width = W;
    canvas.height = H;
    if (plane.name === 'scene') canvas.id = 'c';
    canvas.className = 'plane';
    canvas.dataset.compositor = 'canvas-api-splats';
    canvas.dataset.depth = plane.depth;
    canvas.dataset.plane = plane.name;
    const ctx = canvas.getContext('2d', {alpha: true});
    if (plane.name === 'scene' || plane.name === 'background') {
      ctx.fillStyle = `rgb(${BG[0]},${BG[1]},${BG[2]})`;
      ctx.fillRect(0, 0, W, H);
    }
    ctx.globalCompositeOperation = 'source-over';
    for (const splat of plane.splats) drawSplat(ctx, splat);
    scene.appendChild(canvas);
    canvases.push(canvas);
  }

  if (STRENGTH > 0) {
    let queued = false, pointerX = 0, pointerY = 0;
    const update = () => {
      queued = false;
      for (const canvas of canvases) {
        const depth = Number(canvas.dataset.depth);
        const tx = -pointerX * depth * STRENGTH;
        const ty = -pointerY * depth * STRENGTH;
        canvas.style.transform = `translate3d(${tx.toFixed(2)}px,${ty.toFixed(2)}px,0)`;
      }
    };
    scene.addEventListener('mousemove', event => {
      const rect = scene.getBoundingClientRect();
      pointerX = ((event.clientX - rect.left) / rect.width - 0.5) * 2;
      pointerY = ((event.clientY - rect.top) / rect.height - 0.5) * 2;
      if (!queued) { queued = true; requestAnimationFrame(update); }
    });
    scene.addEventListener('mouseleave', () => {
      pointerX = 0; pointerY = 0;
      if (!queued) { queued = true; requestAnimationFrame(update); }
    });
  }

  const renderMs = performance.now() - t0;
  window.__SPLATTHIS_RENDER_MS = renderMs;
  document.documentElement.dataset.splatthisRenderDone = 'true';
  scene.dataset.renderMs = renderMs.toFixed(3);
})();
""".replace(
            "__W__", str(width)
        )
        .replace("__H__", str(height))
        .replace("__BG__", json.dumps(bg_rgb, separators=(",", ":")))
        .replace("__STRENGTH__", f"{parallax_strength:.4f}")
        .replace("__PLANES__", plane_json)
    )

    safe_title = escape_html(title)
    return (
        "<!doctype html>\n"
        f'<html><head><meta charset="utf-8"><title>{safe_title}</title>'
        "<style>*{box-sizing:border-box}"
        f"html,body{{margin:0;width:{width}px;height:{height}px;overflow:hidden}}"
        f"#scene{{position:relative;width:{width}px;height:{height}px;overflow:hidden}}"
        ".plane{position:absolute;inset:0;display:block;pointer-events:none;"
        "transition:transform 60ms cubic-bezier(.2,.7,.3,1)}"
        "</style></head>\n"
        f'<body><main id="scene" data-compositor="canvas-api-splats" '
        f'data-splat-count="{len(ordered_splats)}" '
        f'data-parallax="{str(parallax_strength > 0.0).lower()}"></main>'
        "<script>" + js + "</script></body></html>\n"
    )
