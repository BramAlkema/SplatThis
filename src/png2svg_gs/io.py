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

from .splat import GaussianSplat, RAW_SPLAT_SCHEMA_VERSION, RawSplat

logger = logging.getLogger(__name__)
EMU_PER_PX = 9525
# Favor fidelity over blanket coverage in export geometry.
ELLIPSE_OVERLAP_BOOST = 1.15
MIN_ELLIPSE_RADIUS_PX = 0.35
# Number of radial-gradient stops used to approximate each splat's Gaussian
# opacity falloff in exported SVG. ~6-8 captures the profile; more adds bytes
# without measurable fidelity gain.
SVG_GRADIENT_STOPS = 8
DEFAULT_EXPORT_ORDER = "importance"


def load_png(path: str, target_size: Optional[Tuple[int, int]] = None,
             linearize_srgb: bool = True) -> np.ndarray:
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
        if img.mode == 'P':  # Palette
            img = img.convert('RGBA')
        elif img.mode in ['L', 'LA']:  # Grayscale
            img = img.convert('RGB')
        elif img.mode == 'RGBA':
            pass  # Keep alpha
        else:
            img = img.convert('RGB')

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

        logger.info(f"Final image shape: {img_array.shape}, range: [{img_array.min():.3f}, {img_array.max():.3f}]")
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
        srgb <= 0.04045,
        srgb / 12.92,
        np.power((srgb + 0.055) / 1.055, 2.4)
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
        linear <= 0.0031308,
        12.92 * linear,
        1.055 * np.power(linear, 1.0/2.4) - 0.055
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
        return sorted(splats, key=lambda s: float(s.importance))
    raise ValueError(f"Unsupported export sort mode: {sort_mode}")


def save_svg(splats: List[GaussianSplat], width: int, height: int,
             output_path: str, k_sigma: float = 2.5,
             sort_by_area: bool = False,
             sort_mode: str = DEFAULT_EXPORT_ORDER,
             background_linear_rgb: Optional[np.ndarray] = None) -> None:
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
    )

    # Write to file
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
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

    drawingml_content = generate_drawingml_slide_content(ordered_splats, width, height, k_sigma)

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(drawingml_content)
        logger.info(f"Saved DrawingML with {len(ordered_splats)} splats to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save DrawingML {output_path}: {e}")
        raise


def generate_svg_content(
    splats: List[GaussianSplat],
    width: int,
    height: int,
    k_sigma: float = 2.5,
    background_linear_rgb: Optional[np.ndarray] = None,
) -> str:
    """
    Generate SVG content from splats.

    Args:
        splats: List of Gaussian splats
        width: Image width
        height: Image height
        k_sigma: Sigma multiplier for ellipse size
        background_linear_rgb: Optional background color in linear RGB [0,1]

    Returns:
        Complete SVG document as string
    """
    background_rect_line: Optional[str] = None
    if background_linear_rgb is not None:
        bg = np.asarray(background_linear_rgb, dtype=np.float32).reshape(-1)
        if bg.size != 3:
            raise ValueError("background_linear_rgb must have exactly 3 components")
        bg = np.clip(bg, 0.0, 1.0)
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

    # Per-splat radial gradients approximate gaussian opacity in exported SVG.
    for i, splat in enumerate(splats):
        gradient_id = f"splat_grad_{i}"
        rgb_srgb = linear_to_srgb(np.array(splat.color[:3], dtype=np.float32))
        r = int(np.clip(np.round(rgb_srgb[0] * 255), 0, 255))
        g = int(np.clip(np.round(rgb_srgb[1] * 255), 0, 255))
        b = int(np.clip(np.round(rgb_srgb[2] * 255), 0, 255))
        color = f"rgb({r},{g},{b})"
        alpha = float(np.clip(splat.alpha, 0.0, 1.0))
        # True-Gaussian gradient stops: reproduce the renderer's per-splat
        # alpha-over opacity, 1 - exp(-a * exp(-0.5 * r^2)), sampled across the
        # ellipse. The gradient edge (offset 100%) is the ellipse boundary at
        # `footprint` sigmas, so the normalized radius t maps to t*footprint
        # sigmas. This matches render_splats_numpy far better than the old
        # 3-stop ramp (+~0.02 perceptual SSIM at no extra cost).
        footprint = ELLIPSE_OVERLAP_BOOST * k_sigma
        stop_lines = []
        for j in range(SVG_GRADIENT_STOPS):
            t = j / (SVG_GRADIENT_STOPS - 1)
            opacity = 1.0 - math.exp(-alpha * math.exp(-0.5 * (t * footprint) ** 2))
            stop_lines.append(
                f'      <stop offset="{t * 100:.1f}%" stop-color="{color}" stop-opacity="{opacity:.5f}"/>'
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
        )
        svg_lines.append(f"  {ellipse_element}")

    svg_lines.extend(["", "</svg>"])
    return "\n".join(svg_lines)


def generate_drawingml_slide_content(
    splats: List[GaussianSplat], width: int, height: int, k_sigma: float = 2.5
) -> str:
    """Generate PresentationML slide XML containing DrawingML ellipse shapes."""
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

    shape_id = 2
    for splat in splats:
        lines.extend(_splat_to_drawingml_shape_lines(splat, shape_id, k_sigma))
        shape_id += 1

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


def _splat_to_drawingml_shape_lines(
    splat: GaussianSplat, shape_id: int, k_sigma: float
) -> List[str]:
    """Convert one Gaussian splat to a DrawingML ellipse shape."""
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

    # Radial gradient stops mirroring the SVG path: the renderer's per-splat
    # alpha-over opacity 1-exp(-a*exp(-0.5*(t*footprint)^2)). DrawingML pos/alpha
    # are in thousandths of a percent (0..100000). This replaces the old flat
    # solidFill, which gave each splat a uniform-opacity disc with no falloff.
    alpha_clamped = float(np.clip(splat.alpha, 0.0, 1.0))
    footprint = ELLIPSE_OVERLAP_BOOST * k_sigma
    gradient_stop_lines: List[str] = []
    for j in range(SVG_GRADIENT_STOPS):
        t = j / (SVG_GRADIENT_STOPS - 1)
        opacity = 1.0 - math.exp(-alpha_clamped * math.exp(-0.5 * (t * footprint) ** 2))
        pos = int(round(t * 100000.0))
        a_units = int(np.clip(round(opacity * 100000.0), 0, 100000))
        gradient_stop_lines.extend([
            f'              <a:gs pos="{pos}">',
            f'                <a:srgbClr val="{color_hex}"><a:alpha val="{a_units}"/></a:srgbClr>',
            "              </a:gs>",
        ])

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
        '            <a:path path="circle">',
        '              <a:fillToRect l="50000" t="50000" r="50000" b="50000"/>',
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


def splat_to_svg_ellipse(
    splat: GaussianSplat,
    k_sigma: float = 2.5,
    element_id: Optional[str] = None,
    gradient_id: Optional[str] = None,
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
        ELLIPSE_OVERLAP_BOOST * k_sigma * np.sqrt(eigenvals[0]),
    )
    ry = max(
        MIN_ELLIPSE_RADIUS_PX,
        ELLIPSE_OVERLAP_BOOST * k_sigma * np.sqrt(eigenvals[1]),
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
    id_attr = f' id="{element_id}"' if element_id else ''
    transform_attr = f' transform="rotate({rotation_deg:.2f} {cx:.2f} {cy:.2f})"' if abs(rotation_deg) > 0.1 else ''

    fill_attr = f'url(#{gradient_id})' if gradient_id else fallback_color
    alpha_attr = "" if gradient_id else f' fill-opacity="{float(np.clip(splat.alpha, 0.0, 1.0)):.3f}"'
    ellipse = (
        f'<ellipse{id_attr} '
        f'cx="{cx:.2f}" cy="{cy:.2f}" '
        f'rx="{rx:.2f}" ry="{ry:.2f}" '
        f'fill="{fill_attr}" '
        f'data-fallback-fill="{fallback_color}"'
        f'{alpha_attr}'
        f'{transform_attr} '
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
        return float(structural_similarity(x, y, channel_axis=channel_axis, data_range=1.0))
    except Exception:
        logger.warning("skimage unavailable; falling back to inflated global SSIM")
        return _global_ssim_np(x, y)


def _global_ssim_np(x: np.ndarray, y: np.ndarray) -> float:
    """Global single-window SSIM (legacy fallback; over-reports vs windowed SSIM)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

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

        png_bytes = cairosvg.svg2png(url=svg_path, output_width=int(width), output_height=int(height))
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
                    [rsvg, "-w", str(int(width)), "-h", str(int(height)), svg_path, "-o", tmp.name],
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
    rendered, method = _try_rasterize_svg_to_linear_rgb(svg_path=svg_path, width=w, height=h)
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
</Relationships>
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
      <a:majorFont><a:latin typeface="Calibri"/></a:majorFont>
      <a:minorFont><a:latin typeface="Calibri"/></a:minorFont>
    </a:fontScheme>
    <a:fmtScheme name="SplatThis">
      <a:fillStyleLst>
        <a:solidFill><a:schemeClr val="phClr"/></a:solidFill>
      </a:fillStyleLst>
      <a:lnStyleLst>
        <a:ln w="9525" cap="flat" cmpd="sng" algn="ctr">
          <a:solidFill><a:schemeClr val="phClr"/></a:solidFill>
        </a:ln>
      </a:lnStyleLst>
      <a:effectStyleLst>
        <a:effectStyle><a:effectLst/></a:effectStyle>
      </a:effectStyleLst>
      <a:bgFillStyleLst>
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
    now_iso = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", _pptx_content_types_xml())
        zf.writestr("_rels/.rels", _pptx_root_rels_xml())
        zf.writestr("docProps/core.xml", _pptx_core_props_xml(now_iso))
        zf.writestr("docProps/app.xml", _pptx_app_props_xml())
        zf.writestr("ppt/presentation.xml", _pptx_presentation_xml(slide_cx=slide_cx, slide_cy=slide_cy))
        zf.writestr("ppt/_rels/presentation.xml.rels", _pptx_presentation_rels_xml())
        zf.writestr("ppt/slides/slide1.xml", _pptx_slide_xml(slide_cx=slide_cx, slide_cy=slide_cy))
        zf.writestr("ppt/slides/_rels/slide1.xml.rels", _pptx_slide_rels_xml())
        zf.writestr("ppt/slideLayouts/slideLayout1.xml", _pptx_slide_layout_xml())
        zf.writestr("ppt/slideLayouts/_rels/slideLayout1.xml.rels", _pptx_slide_layout_rels_xml())
        zf.writestr("ppt/slideMasters/slideMaster1.xml", _pptx_slide_master_xml())
        zf.writestr("ppt/slideMasters/_rels/slideMaster1.xml.rels", _pptx_slide_master_rels_xml())
        zf.writestr("ppt/theme/theme1.xml", _pptx_theme_xml())
        zf.writestr("ppt/media/image1.png", png_bytes)

    logger.info("Saved PPTX with rasterized splat image: %s", output_path)


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
                metrics_rows.append(f"<tr><td colspan='2'><strong>{key}</strong></td></tr>")
                for sub_key, sub_val in value.items():
                    metrics_rows.append(f"<tr><td>{key}.{sub_key}</td><td>{sub_val}</td></tr>")
            else:
                metrics_rows.append(f"<tr><td>{key}</td><td>{value}</td></tr>")
    metrics_table = (
        "<table>" + "".join(metrics_rows) + "</table>" if metrics_rows else "<p>No metrics recorded.</p>"
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
    reconstructed = [GaussianSplat.from_raw_splat(RawSplat.from_dict(d)) for d in raw_dicts]

    if len(splats) == 0:
        svg_content = generate_svg_content(reconstructed, width, height, k_sigma)
        dml_content = generate_drawingml_slide_content(reconstructed, width, height, k_sigma)
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
        mu_delta = max(mu_delta, float(np.max(np.abs(original.mu[:2] - restored.mu[:2]))))
        color_delta = max(color_delta, float(np.max(np.abs(original.color[:3] - restored.color[:3]))))
        alpha_delta = max(alpha_delta, float(abs(float(original.alpha) - float(restored.alpha))))

    svg_content = generate_svg_content(reconstructed, width, height, k_sigma)
    dml_content = generate_drawingml_slide_content(reconstructed, width, height, k_sigma)

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
