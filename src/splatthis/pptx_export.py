"""DrawingML emitters and deterministic one-slide PPTX packaging."""

from __future__ import annotations

import io
import logging
import math
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from .color import linear_to_srgb
from .export_common import (
    DEFAULT_EXPORT_ORDER,
    DEFAULT_PPTX_SPLAT_STYLE,
    ELLIPSE_OVERLAP_BOOST,
    EMU_PER_PX,
    MIN_ELLIPSE_RADIUS_PX,
    PPTX_BLUR_CORE_K_SIGMA,
    PPTX_BLUR_RAD_PER_SIGMA,
    PPTX_GRADIENT_ALPHA_SCALE,
    PPTX_PAINTER_ORDER_LEGACY,
    PPTX_SOFT_EDGE_ALPHA_SCALE,
    PPTX_SOFT_EDGE_K_SIGMA_SCALE,
    PPTX_SOFT_EDGE_RADIUS_FACTOR,
    SVG_GRADIENT_STOPS,
    _layer_title,
    _normalize_pptx_painter_order,
    _normalize_pptx_splat_style,
    _pptx_painter_splats,
    _sort_splats_for_export,
    _splat_layer,
    pptx_emu_scale,
    px_to_emu,
)
from .splat import LAYER_BASE, GaussianSplat
from .storage import atomic_output_path, atomic_write_text
from .template_assets import load_template, render_template, render_template_lines

logger = logging.getLogger(__name__)


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
    painter_order: str = PPTX_PAINTER_ORDER_LEGACY,
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
        painter_order=painter_order,
    )

    try:
        atomic_write_text(output_path, drawingml_content)
        logger.info(
            f"Saved DrawingML with {len(ordered_splats)} splats to {output_path}"
        )
    except Exception as e:
        logger.error(f"Failed to save DrawingML {output_path}: {e}")
        raise


def generate_drawingml_slide_content(
    splats: List[GaussianSplat],
    width: int,
    height: int,
    k_sigma: float = 2.5,
    background_linear_rgb: Optional[np.ndarray] = None,
    splat_style: str = DEFAULT_PPTX_SPLAT_STYLE,
    painter_order: str = PPTX_PAINTER_ORDER_LEGACY,
) -> str:
    """Generate PresentationML slide XML containing DrawingML ellipse shapes."""
    normalized_splat_style = _normalize_pptx_splat_style(splat_style)
    normalized_painter_order = _normalize_pptx_painter_order(painter_order)
    # Small canvases would emit a sub-inch slide, which is schema-invalid;
    # scale the whole composition uniformly instead (see pptx_emu_scale).
    emu_scale = pptx_emu_scale(width, height)
    slide_width_emu = max(px_to_emu(width, emu_scale), 1)
    slide_height_emu = max(px_to_emu(height, emu_scale), 1)

    lines: List[str] = []

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
        # None-layer splats have render keys in [0, 1) — the LAYER_BASE band
        # (see render_importance_for_raw) — so they must draw with that band,
        # not front-most above the edge layer. Base group first, then the
        # None bucket, then the numbered layers.
        layer_ids = sorted(
            by_layer,
            key=lambda layer: (0 if layer is None else int(layer), layer is None),
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
            for splat in _pptx_painter_splats(layer_splats, normalized_painter_order):
                lines.extend(
                    _splat_to_drawingml_shape_lines(
                        splat,
                        shape_id,
                        k_sigma,
                        emu_scale=emu_scale,
                        splat_style=normalized_splat_style,
                    )
                )
                shape_id += 1
            lines.extend(render_template_lines("drawingml/close_group.xml"))
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
        for splat in _pptx_painter_splats(splats, normalized_painter_order):
            lines.extend(
                _splat_to_drawingml_shape_lines(
                    splat,
                    shape_id,
                    k_sigma,
                    emu_scale=emu_scale,
                    splat_style=normalized_splat_style,
                )
            )
            shape_id += 1
    lines.extend(render_template_lines("drawingml/close_group.xml"))

    return render_template(
        "drawingml/slide.xml",
        slide_width_emu=slide_width_emu,
        slide_height_emu=slide_height_emu,
        body="\n".join(lines),
    ).rstrip("\n")


def _drawingml_group_start_lines(
    width_emu: int,
    height_emu: int,
    shape_id: int,
    name: str = "Splat Group",
) -> List[str]:
    """Create the opening XML for a native DrawingML group containing all splats."""
    return render_template_lines(
        "drawingml/group.xml",
        width_emu=width_emu,
        height_emu=height_emu,
        shape_id=shape_id,
        name=name,
    )


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
    return render_template_lines(
        "drawingml/background_shape.xml",
        width_emu=width_emu,
        height_emu=height_emu,
        shape_id=shape_id,
        color_hex=color_hex,
    )


def _splat_geometry_for_drawingml(
    splat: GaussianSplat,
    k_sigma: float,
    emu_scale: float = 1.0,
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

    x_emu = px_to_emu(x, emu_scale)
    y_emu = px_to_emu(y, emu_scale)
    w_emu = max(px_to_emu(w, emu_scale), 1)
    h_emu = max(px_to_emu(h, emu_scale), 1)

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


def _drawingml_shape_lines(
    shape_id: int,
    x_emu: int,
    y_emu: int,
    w_emu: int,
    h_emu: int,
    rot_attr: str,
    fill_lines: List[str],
    effect_lines: Optional[List[str]] = None,
) -> List[str]:
    """Wrap a per-splat fill + effect body in the standard DrawingML
    ellipse-shape scaffold. All three splat styles (gradient / soft-edge /
    blur) share this 30-line boilerplate; only the inner fill / effect
    differs."""
    template_name = "drawingml/shape_no_effect.xml"
    effect_block = ""
    if effect_lines:
        template_name = "drawingml/shape.xml"
        effect_block = render_template(
            "drawingml/effect_list.xml",
            effect_lines="\n".join(effect_lines),
        ).rstrip("\n")
    return render_template_lines(
        template_name,
        shape_id=shape_id,
        x_emu=x_emu,
        y_emu=y_emu,
        w_emu=w_emu,
        h_emu=h_emu,
        rot_attr=rot_attr,
        fill_lines="\n".join(fill_lines),
        effect_block=effect_block,
    )


def _solid_fill_lines(color_hex: str, alpha_units: int) -> List[str]:
    return render_template_lines(
        "drawingml/solid_fill.xml",
        color_hex=color_hex,
        alpha_units=alpha_units,
    )


def _splat_to_drawingml_shape_lines(
    splat: GaussianSplat,
    shape_id: int,
    k_sigma: float,
    emu_scale: float = 1.0,
    splat_style: str = DEFAULT_PPTX_SPLAT_STYLE,
) -> List[str]:
    """Convert one Gaussian splat to a DrawingML ellipse shape.

    Three styles dispatched by splat_style: 'gradient' (radial gradient
    stops mirroring the renderer's alpha-over Gaussian), 'soft-edge'
    (solid fill plus a soft edge), and 'blur' (solid fill plus a blur with
    radius calibrated to sigma times 3.25 EMU)."""
    normalized = _normalize_pptx_splat_style(splat_style)
    if normalized == "soft-edge":
        return _splat_to_drawingml_soft_edge_shape_lines(
            splat, shape_id, k_sigma, emu_scale
        )
    if normalized == "blur":
        return _splat_to_drawingml_blur_shape_lines(splat, shape_id, k_sigma, emu_scale)

    x_emu, y_emu, w_emu, h_emu, rot_attr, color_hex = _splat_geometry_for_drawingml(
        splat, k_sigma, emu_scale
    )

    # Radial gradient stops mirroring the renderer's per-splat alpha-over
    # opacity 1 - exp(-α · exp(-r²/2σ²)). PPTX gradient-fill renders at
    # ~0.4× the trainer's alpha; PPTX_GRADIENT_ALPHA_SCALE compensates.
    # path="shape" matched the PowerPoint roundtrip better than "circle".
    alpha_clamped = float(np.clip(splat.alpha * PPTX_GRADIENT_ALPHA_SCALE, 0.0, 1.0))
    footprint = ELLIPSE_OVERLAP_BOOST * k_sigma
    stop_lines: List[str] = []
    for j in range(SVG_GRADIENT_STOPS):
        t = j / (SVG_GRADIENT_STOPS - 1)
        opacity = 1.0 - math.exp(-alpha_clamped * math.exp(-0.5 * (t * footprint) ** 2))
        pos = int(round(t * 100000.0))
        a_units = int(np.clip(round(opacity * 100000.0), 0, 100000))
        stop_lines.extend(
            render_template_lines(
                "drawingml/gradient_stop.xml",
                position=pos,
                color_hex=color_hex,
                alpha_units=a_units,
            )
        )
    fill_lines = render_template_lines(
        "drawingml/gradient_fill.xml",
        stop_lines="\n".join(stop_lines),
    )
    return _drawingml_shape_lines(
        shape_id, x_emu, y_emu, w_emu, h_emu, rot_attr, fill_lines
    )


def _splat_to_drawingml_soft_edge_shape_lines(
    splat: GaussianSplat,
    shape_id: int,
    k_sigma: float,
    emu_scale: float = 1.0,
) -> List[str]:
    """Solid-fill ellipse with a soft edge that feathers the outer
    `rad` ring inward — NOT a Gaussian, but cheap to render and visually
    smoother than a hard shape."""
    effective_k_sigma = float(k_sigma) * PPTX_SOFT_EDGE_K_SIGMA_SCALE
    x_emu, y_emu, w_emu, h_emu, rot_attr, color_hex = _splat_geometry_for_drawingml(
        splat,
        effective_k_sigma,
        emu_scale,
    )
    center_opacity = 1.0 - math.exp(-float(np.clip(splat.alpha, 0.0, 1.0)))
    alpha_units = int(
        np.clip(
            round(center_opacity * PPTX_SOFT_EDGE_ALPHA_SCALE * 100000.0),
            0,
            100000,
        )
    )
    soft_radius = int(max(0, round(min(w_emu, h_emu) * PPTX_SOFT_EDGE_RADIUS_FACTOR)))
    return _drawingml_shape_lines(
        shape_id,
        x_emu,
        y_emu,
        w_emu,
        h_emu,
        rot_attr,
        fill_lines=_solid_fill_lines(color_hex, alpha_units),
        effect_lines=render_template_lines(
            "drawingml/soft_edge.xml", radius=soft_radius
        ),
    )


def _splat_to_drawingml_blur_shape_lines(
    splat: GaussianSplat,
    shape_id: int,
    k_sigma: float,
    emu_scale: float = 1.0,
) -> List[str]:
    """Small solid-fill ellipse plus an isotropic blur. The blur produces
    the Gaussian shape; the ellipse is the small coloured core. Mass-
    fraction alpha compensation keeps the convolved peak ≈ α even though
    The blur spreads the core's mass. See
    svg2ooxml/docs/reference/research/blur-fidelity-results.md for the
    σ_blur = rad / 3.25 calibration."""
    # Use a smaller core (k=1.0) than the gradient/soft-edge styles. The
    # convolution math expects a near-point source for the output to
    # approximate a true Gaussian.
    x_emu, y_emu, w_emu, h_emu, rot_attr, color_hex = _splat_geometry_for_drawingml(
        splat,
        float(PPTX_BLUR_CORE_K_SIGMA) / ELLIPSE_OVERLAP_BOOST,
        emu_scale,
    )
    # Geometric-mean sigma drives the isotropic blur radius; ellipse
    # aspect-ratio (baked into w_emu/h_emu) absorbs the anisotropy.
    eigenvals, _ = splat.eigendecomposition()
    sigma_major = float(np.sqrt(max(float(eigenvals[0]), 1e-8)))
    sigma_minor = float(np.sqrt(max(float(eigenvals[1]), 1e-8)))
    sigma_geo_px = float(np.sqrt(sigma_major * sigma_minor))

    mass_fraction = 1.0 - math.exp(-0.5 * float(PPTX_BLUR_CORE_K_SIGMA) ** 2)
    # Target the renderer's peak per-splat opacity 1 − e^(−α), not raw α
    # (same correction as the SVG blur recipe).
    peak_opacity = 1.0 - math.exp(-min(max(float(splat.alpha), 0.0), 1.0))
    alpha_compensated = min(1.0, peak_opacity / max(mass_fraction, 1e-6))
    alpha_units = int(np.clip(round(alpha_compensated * 100000.0), 0, 100000))
    rad_emu = max(1, int(round(sigma_geo_px * EMU_PER_PX * PPTX_BLUR_RAD_PER_SIGMA)))

    return _drawingml_shape_lines(
        shape_id,
        x_emu,
        y_emu,
        w_emu,
        h_emu,
        rot_attr,
        fill_lines=_solid_fill_lines(color_hex, alpha_units),
        effect_lines=render_template_lines("drawingml/blur.xml", radius=rad_emu),
    )


def _pptx_content_types_xml() -> str:
    return load_template("pptx/content_types.xml").rstrip("\n") + "\n"


def _pptx_root_rels_xml() -> str:
    return load_template("pptx/root_rels.xml").rstrip("\n") + "\n"


def _pptx_core_props_xml(now_iso: str) -> str:
    return render_template("pptx/core_props.xml", modified=now_iso).rstrip("\n") + "\n"


def _pptx_app_props_xml() -> str:
    return load_template("pptx/app_props.xml").rstrip("\n") + "\n"


def _pptx_presentation_xml(slide_cx: int, slide_cy: int) -> str:
    return (
        render_template(
            "pptx/presentation.xml",
            slide_cx=slide_cx,
            slide_cy=slide_cy,
        ).rstrip("\n")
        + "\n"
    )


def _pptx_presentation_rels_xml() -> str:
    return load_template("pptx/presentation_rels.xml").rstrip("\n") + "\n"


def _pptx_pres_props_xml() -> str:
    return load_template("pptx/pres_props.xml").rstrip("\n") + "\n"


def _pptx_view_props_xml() -> str:
    return load_template("pptx/view_props.xml").rstrip("\n") + "\n"


def _pptx_slide_xml(slide_cx: int, slide_cy: int) -> str:
    return (
        render_template(
            "pptx/raster_slide.xml",
            slide_cx=slide_cx,
            slide_cy=slide_cy,
        ).rstrip("\n")
        + "\n"
    )


def _pptx_slide_rels_xml() -> str:
    return load_template("pptx/raster_slide_rels.xml").rstrip("\n") + "\n"


def _pptx_vector_slide_rels_xml() -> str:
    return load_template("pptx/vector_slide_rels.xml").rstrip("\n") + "\n"


def _pptx_slide_layout_xml() -> str:
    return load_template("pptx/slide_layout.xml").rstrip("\n") + "\n"


def _pptx_slide_layout_rels_xml() -> str:
    return load_template("pptx/slide_layout_rels.xml").rstrip("\n") + "\n"


def _pptx_slide_master_xml() -> str:
    return load_template("pptx/slide_master.xml").rstrip("\n") + "\n"


def _pptx_slide_master_rels_xml() -> str:
    return load_template("pptx/slide_master_rels.xml").rstrip("\n") + "\n"


def _pptx_theme_xml() -> str:
    return load_template("pptx/theme.xml").rstrip("\n") + "\n"


def _write_pptx_package(
    output_path: str | Path,
    members: List[Tuple[str, str | bytes]],
) -> None:
    """Write a complete OOXML package and publish it atomically."""

    with atomic_output_path(output_path) as temporary:
        with zipfile.ZipFile(
            temporary, "w", compression=zipfile.ZIP_DEFLATED
        ) as archive:
            for member_path, payload in members:
                archive.writestr(member_path, payload)


def save_pptx_with_splat_png(
    splats: List[GaussianSplat],
    width: int,
    height: int,
    output_path: str,
    sort_mode: str = DEFAULT_EXPORT_ORDER,
    sort_by_area: bool = False,
    render_scale: float = 1.0,
    background_linear_rgb: Optional[np.ndarray] = None,
    compositing_space: str = "linear",
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
        compositing_space=compositing_space,
    )
    rendered_srgb = linear_to_srgb(np.clip(rendered, 0.0, 1.0))
    image = Image.fromarray((rendered_srgb * 255.0).astype(np.uint8), mode="RGB")
    if (render_width, render_height) != (width, height):
        image = image.resize((render_width, render_height), Image.Resampling.LANCZOS)

    png_buffer = io.BytesIO()
    image.save(png_buffer, format="PNG")
    png_bytes = png_buffer.getvalue()

    _emu_scale = pptx_emu_scale(render_width, render_height)
    slide_cx = max(px_to_emu(render_width, _emu_scale), 1)
    slide_cy = max(px_to_emu(render_height, _emu_scale), 1)
    now_iso = (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )

    _write_pptx_package(
        output_path,
        [
            ("[Content_Types].xml", _pptx_content_types_xml()),
            ("_rels/.rels", _pptx_root_rels_xml()),
            ("docProps/core.xml", _pptx_core_props_xml(now_iso)),
            ("docProps/app.xml", _pptx_app_props_xml()),
            (
                "ppt/presentation.xml",
                _pptx_presentation_xml(slide_cx=slide_cx, slide_cy=slide_cy),
            ),
            ("ppt/_rels/presentation.xml.rels", _pptx_presentation_rels_xml()),
            (
                "ppt/slides/slide1.xml",
                _pptx_slide_xml(slide_cx=slide_cx, slide_cy=slide_cy),
            ),
            ("ppt/slides/_rels/slide1.xml.rels", _pptx_slide_rels_xml()),
            ("ppt/slideLayouts/slideLayout1.xml", _pptx_slide_layout_xml()),
            (
                "ppt/slideLayouts/_rels/slideLayout1.xml.rels",
                _pptx_slide_layout_rels_xml(),
            ),
            ("ppt/slideMasters/slideMaster1.xml", _pptx_slide_master_xml()),
            (
                "ppt/slideMasters/_rels/slideMaster1.xml.rels",
                _pptx_slide_master_rels_xml(),
            ),
            ("ppt/theme/theme1.xml", _pptx_theme_xml()),
            ("ppt/presProps.xml", _pptx_pres_props_xml()),
            ("ppt/viewProps.xml", _pptx_view_props_xml()),
            ("ppt/media/image1.png", png_bytes),
        ],
    )

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
    painter_order: str = PPTX_PAINTER_ORDER_LEGACY,
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
    slide_xml = generate_drawingml_slide_content(
        ordered_splats,
        width=width,
        height=height,
        k_sigma=k_sigma,
        background_linear_rgb=background_linear_rgb,
        splat_style=splat_style,
        painter_order=painter_order,
    )
    save_pptx_with_drawingml_content(
        slide_xml=slide_xml,
        width=width,
        height=height,
        output_path=output_path,
        splat_count=len(ordered_splats),
    )


def save_pptx_with_drawingml_content(
    *,
    slide_xml: str,
    width: int,
    height: int,
    output_path: str,
    splat_count: int,
) -> None:
    """Package already-emitted DrawingML without generating it a second time."""

    _emu_scale = pptx_emu_scale(width, height)
    slide_cx = max(px_to_emu(width, _emu_scale), 1)
    slide_cy = max(px_to_emu(height, _emu_scale), 1)
    now_iso = (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )

    _write_pptx_package(
        output_path,
        [
            ("[Content_Types].xml", _pptx_content_types_xml()),
            ("_rels/.rels", _pptx_root_rels_xml()),
            ("docProps/core.xml", _pptx_core_props_xml(now_iso)),
            ("docProps/app.xml", _pptx_app_props_xml()),
            (
                "ppt/presentation.xml",
                _pptx_presentation_xml(slide_cx=slide_cx, slide_cy=slide_cy),
            ),
            ("ppt/_rels/presentation.xml.rels", _pptx_presentation_rels_xml()),
            ("ppt/slides/slide1.xml", slide_xml),
            (
                "ppt/slides/_rels/slide1.xml.rels",
                _pptx_vector_slide_rels_xml(),
            ),
            ("ppt/slideLayouts/slideLayout1.xml", _pptx_slide_layout_xml()),
            (
                "ppt/slideLayouts/_rels/slideLayout1.xml.rels",
                _pptx_slide_layout_rels_xml(),
            ),
            ("ppt/slideMasters/slideMaster1.xml", _pptx_slide_master_xml()),
            (
                "ppt/slideMasters/_rels/slideMaster1.xml.rels",
                _pptx_slide_master_rels_xml(),
            ),
            ("ppt/theme/theme1.xml", _pptx_theme_xml()),
            ("ppt/presProps.xml", _pptx_pres_props_xml()),
            ("ppt/viewProps.xml", _pptx_view_props_xml()),
        ],
    )

    logger.info("Saved PPTX with %s native splat shapes: %s", splat_count, output_path)
