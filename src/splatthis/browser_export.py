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
    _gaussian_opacity_curve,
    _sort_splats_for_export,
    _splat_layer,
)
from .splat import LAYER_BASE, LAYER_MASS, GaussianSplat

#: Evenly spaced samples of the exact Gaussian opacity curve per CSS splat.
#: Nine was the value the compositor MVP selected; fewer loses the tail that
#: alpha-over accumulation depends on, more costs bytes for no measured gain.
CSS_EXACT_GRADIENT_STOPS = 9

#: Stop count for the email-safe variant. Fewer stops is the smaller half of
#: the size win; the larger half is dropping the mask entirely.
CSS_EMAIL_GRADIENT_STOPS = 6


def generate_css_splat_html(
    splats: List[GaussianSplat],
    width: int,
    height: int,
    background_linear_rgb: Optional[npt.NDArray[Any]] = None,
    title: str = "SplatThis CSS",
    parallax_strength: float = 0.0,
    hover_grid_size: int = 10,
    k_sigma: float = 2.5,
    email_safe: bool = False,
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

    # CSS composites in DOM order, permanently, so emitting back-to-front is
    # the difference between a splat stack that accumulates correctly and one
    # that does not. This mirrors the corrected painter order the SVG exporter
    # already defaults to.
    ordered_splats = _sort_splats_for_export(splats)
    ordered_splats.reverse()
    gradient_footprint = ELLIPSE_OVERLAP_BOOST * float(k_sigma)

    def _plane_for(splat: GaussianSplat) -> str:
        layer = _splat_layer(splat)
        if layer is not None and layer <= LAYER_BASE:
            return "background"
        if layer is None or layer == LAYER_MASS:
            return "midground"
        return "foreground"

    # Running bottom edge of the last email-safe block, so the next one's
    # margin-top can be expressed as a delta. Gmail strips position/left/top,
    # but it keeps margins, and sibling margins collapse against a zero
    # margin-bottom to exactly the value asked for -- negative included.
    flow_bottom = [0.0]

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
        # Colour is emitted in linear sRGB and the Gaussian falloff is applied
        # as an alpha mask over a solid fill, rather than baked into the
        # gradient's own colour stops. Painting the colour through the gradient
        # makes the browser interpolate colour and opacity together, which
        # darkens the skirt of every splat; masking separates them.
        colour = np.clip(np.asarray(splat.color[:3], dtype=np.float32), 0.0, 1.0)
        if email_safe:
            # color(srgb-linear ...) is CSS Color 4 and mail clients do not
            # have it, so the same colour is converted to display sRGB and
            # written as legacy rgb().
            srgb = linear_to_srgb(colour)
            channels = tuple(int(round(float(c) * 255.0)) for c in srgb)
            fill = f"rgb({channels[0]},{channels[1]},{channels[2]})"
        else:
            fill = (
                f"color(srgb-linear {float(colour[0]):.6f} "
                f"{float(colour[1]):.6f} {float(colour[2]):.6f})"
            )

        # Nine evenly spaced samples of the exact Gaussian opacity curve. The
        # adaptive placement used previously spends its stop budget where the
        # curve bends most, which is correct for minimising fitted error but
        # loses the tail that alpha-over compositing accumulates over hundreds
        # of overlapping splats.
        stop_count = (
            CSS_EMAIL_GRADIENT_STOPS if email_safe else CSS_EXACT_GRADIENT_STOPS
        )
        offsets = np.linspace(0.0, 1.0, stop_count)
        opacities = _gaussian_opacity_curve(
            offsets, float(np.clip(splat.alpha, 0.0, 1.0)), gradient_footprint
        )

        cx, cy = (float(splat.mu[0]), float(splat.mu[1]))
        if email_safe:
            # No mask: the colour is painted through the gradient's own stops.
            # That is the thing the standard recipe deliberately avoids, so it
            # is a measured quality cost, not a free substitution. Every stop
            # carries the same rgb and varies only alpha, which is what keeps
            # the interpolation from running toward black.
            stops = ",".join(
                f"rgba({channels[0]},{channels[1]},{channels[2]},"
                f"{float(opacity):.3f}) {float(offset) * 100.0:.0f}%"
                for offset, opacity in zip(offsets, opacities)
            )
            # Laid out by margins in normal flow, not by absolute position.
            # Gmail strips position, left, top and transform from inline
            # styles; with position gone an inline <i> also loses width and
            # height, which is why 285 splats rendered as an empty backdrop.
            # A block element sized and offset by margins uses only
            # properties Gmail keeps.
            width_px = max(1.0, round(2.0 * rx))
            height_px = max(1.0, round(2.0 * ry))
            left = cx - width_px / 2.0
            top = cy - height_px / 2.0
            # Track the *rounded* position, not the ideal one. Each margin is
            # relative to the previous element's bottom, so rounding error
            # accumulates down the chain rather than staying under half a
            # pixel per splat the way it does with absolute coordinates.
            margin_top = round(top - flow_bottom[0])
            flow_bottom[0] += margin_top + height_px
            # rotate() is kept, deliberately, and without the translate the
            # absolute version needed: clients that support transforms get
            # the fitted orientation, and the ones that strip it still get
            # the splat in the right place, merely axis-aligned.
            rotate = (
                f"transform:rotate({rotation:.0f}deg);" if abs(rotation) >= 0.5 else ""
            )
            # <div>, not <i>: block display is the default, so nothing depends
            # on a display declaration surviving the sanitiser.
            style = (
                "border-radius:50%;"
                f"width:{width_px:.0f}px;height:{height_px:.0f}px;"
                f"margin:{margin_top:.0f}px 0 0 {left:.0f}px;"
                f"{rotate}"
                # Size stated explicitly, as in the standard recipe. Omitting
                # it defaults to farthest-corner, which is sqrt(2) larger and
                # would show up as a recipe win that is really a size change.
                f"background:radial-gradient(ellipse 50% 50% at center,{stops})"
            )
            return f'<div style="{style}"></div>'

        mask_stops = ",".join(
            f"rgba(0,0,0,{float(opacity):.4f}) {float(offset) * 100.0:.2f}%"
            for offset, opacity in zip(offsets, opacities)
        )
        style = (
            f"left:{cx:.2f}px;top:{cy:.2f}px;"
            f"width:{2.0 * rx:.2f}px;height:{2.0 * ry:.2f}px;"
            f"transform:translate(-50%,-50%) rotate({rotation:.2f}deg);"
            f"background:{fill};"
            f"mask-image:radial-gradient(ellipse 50% 50% at center,{mask_stops});"
            "mask-mode:alpha;mask-repeat:no-repeat"
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
    if email_safe:
        # The scene rules move onto the element itself for the same reason the
        # splat rules did: without a positioned ancestor every absolutely
        # positioned splat would lay out against the viewport instead, and the
        # backdrop colour would vanish. The <style> block is dropped rather
        # than kept as a fallback, since nothing outside it is now needed.
        # overflow:hidden does double duty: it clips the splats that hang
        # over the edge, and it stops the first splat's margin collapsing
        # out through the container.
        scene_style = (
            f"width:{width}px;height:{height}px;"
            f"overflow:hidden;background:{background_css}"
        )
        return (
            "<!doctype html>\n"
            f'<html><head><meta charset="utf-8"><title>{safe_title}</title>'
            "</head>\n"
            f'<body style="margin:0"><div id="scene" '
            f'data-compositor="css-splats-email" '
            f'data-splat-count="{len(ordered_splats)}" style="{scene_style}">'
            + "".join(planes["midground"])
            + "</div></body></html>\n"
        )
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
    # Canvas source-over paints later calls on top. The numerical renderer's
    # front-most splat is first, so Canvas must submit the population reversed.
    ordered_splats.reverse()
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
""".replace("__W__", str(width))
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
