"""Residual-guided native vector primitives for bounded fidelity experiments."""

from __future__ import annotations

import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy import ndimage
from skimage import measure

from .color import linear_to_srgb
from .export_common import pptx_emu_scale, px_to_emu
from .template_assets import load_template, render_template


@dataclass(frozen=True)
class EdgeStroke:
    """Short source-over stroke aligned to a target-image edge."""

    x1: float
    y1: float
    x2: float
    y2: float
    width: float
    color_srgb: tuple[float, float, float]
    opacity: float
    score: float


@dataclass(frozen=True)
class EdgePath:
    """Connected contour fragment represented as an editable SVG path."""

    points: tuple[tuple[float, float], ...]
    width: float
    color_srgb: tuple[float, float, float]
    opacity: float
    score: float


def _display_maps(
    target_linear_rgb: np.ndarray,
    rendered_linear_rgb: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    target = np.clip(np.asarray(target_linear_rgb, dtype=np.float32)[..., :3], 0, 1)
    rendered = np.clip(np.asarray(rendered_linear_rgb, dtype=np.float32)[..., :3], 0, 1)
    if target.shape != rendered.shape:
        raise ValueError("target and rendered images must have identical shapes")
    if target.ndim != 3 or target.shape[-1] != 3:
        raise ValueError("target and rendered images must be HxWx3")
    target_srgb = linear_to_srgb(target)
    rendered_srgb = linear_to_srgb(rendered)
    luma = (
        0.2126 * target_srgb[..., 0]
        + 0.7152 * target_srgb[..., 1]
        + 0.0722 * target_srgb[..., 2]
    )
    grad_x = ndimage.sobel(luma, axis=1, mode="reflect") / 8.0
    grad_y = ndimage.sobel(luma, axis=0, mode="reflect") / 8.0
    gradient = np.hypot(grad_x, grad_y).astype(np.float32)
    color_error = np.sqrt(np.sum((target_srgb - rendered_srgb) ** 2, axis=-1)).astype(
        np.float32
    )
    gradient_norm = gradient / max(float(np.percentile(gradient, 95)), 1e-6)
    error_norm = color_error / max(float(np.percentile(color_error, 95)), 1e-6)
    return target_srgb, rendered_srgb, luma, gradient_norm * error_norm


def propose_residual_edge_strokes(
    target_linear_rgb: np.ndarray,
    rendered_linear_rgb: np.ndarray,
    *,
    max_strokes: int = 32,
    length: float = 5.0,
    width: float = 1.0,
    opacity: float = 0.65,
    min_spacing: int = 3,
) -> list[EdgeStroke]:
    """Place deterministic short strokes at high-error target edges.

    Stroke color is solved in display sRGB for source-over compositing:
    ``source = (target - (1-opacity) * baseline) / opacity``. This makes the
    proposal corrective rather than merely copying the target color on top.
    """

    count = max(0, int(max_strokes))
    if count == 0:
        return []
    alpha = float(np.clip(opacity, 1e-3, 1.0))
    stroke_length = float(max(0.25, length))
    stroke_width = float(max(0.1, width))

    target_srgb, rendered_srgb, luma, priority = _display_maps(
        target_linear_rgb, rendered_linear_rgb
    )
    grad_x = ndimage.sobel(luma, axis=1, mode="reflect") / 8.0
    grad_y = ndimage.sobel(luma, axis=0, mode="reflect") / 8.0

    suppressed = priority.copy()
    height, width_px = priority.shape
    proposals: list[EdgeStroke] = []
    radius = max(1, int(min_spacing))
    half = 0.5 * stroke_length

    for _ in range(count):
        flat_index = int(np.argmax(suppressed))
        score = float(suppressed.flat[flat_index])
        if not np.isfinite(score) or score <= 1e-5:
            break
        y, x = np.unravel_index(flat_index, suppressed.shape)

        # Image gradient points across an edge; the corrective stroke follows
        # its tangent.
        tangent = float(np.arctan2(grad_y[y, x], grad_x[y, x]) + np.pi / 2.0)
        dx = half * float(np.cos(tangent))
        dy = half * float(np.sin(tangent))
        x1 = float(np.clip(float(x) - dx, 0.0, max(width_px - 1.0, 0.0)))
        y1 = float(np.clip(float(y) - dy, 0.0, max(height - 1.0, 0.0)))
        x2 = float(np.clip(float(x) + dx, 0.0, max(width_px - 1.0, 0.0)))
        y2 = float(np.clip(float(y) + dy, 0.0, max(height - 1.0, 0.0)))

        source_color = np.clip(
            (target_srgb[y, x] - (1.0 - alpha) * rendered_srgb[y, x]) / alpha,
            0.0,
            1.0,
        )
        proposals.append(
            EdgeStroke(
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                width=stroke_width,
                color_srgb=tuple(float(v) for v in source_color),
                opacity=alpha,
                score=score,
            )
        )

        y0, y3 = max(0, y - radius), min(height, y + radius + 1)
        x0, x3 = max(0, x - radius), min(width_px, x + radius + 1)
        suppressed[y0:y3, x0:x3] = 0.0

    return proposals


def propose_residual_edge_paths(
    target_linear_rgb: np.ndarray,
    rendered_linear_rgb: np.ndarray,
    *,
    max_paths: int = 8,
    path_length: float = 12.0,
    width: float = 0.8,
    opacity: float = 0.65,
    min_spacing: float = 4.0,
    simplify_tolerance: float = 0.55,
) -> list[EdgePath]:
    """Trace short connected iso-luma contours through high residual edges."""

    target_srgb, rendered_srgb, luma, priority = _display_maps(
        target_linear_rgb, rendered_linear_rgb
    )
    alpha = float(np.clip(opacity, 1e-3, 1.0))
    wanted = max(0, int(max_paths))
    if wanted == 0:
        return []

    candidates = []
    levels = np.unique(np.quantile(luma, [0.15, 0.3, 0.45, 0.6, 0.75, 0.9]))
    height, width_px = luma.shape
    half_length = max(1.0, float(path_length)) * 0.5

    for level in levels:
        for contour_yx in measure.find_contours(luma, float(level)):
            if len(contour_yx) < 4:
                continue
            deltas = np.diff(contour_yx, axis=0)
            cumulative = np.concatenate(
                ([0.0], np.cumsum(np.hypot(deltas[:, 0], deltas[:, 1])))
            )
            if float(cumulative[-1]) < 2.0:
                continue
            ys = np.clip(np.rint(contour_yx[:, 0]).astype(int), 0, height - 1)
            xs = np.clip(np.rint(contour_yx[:, 1]).astype(int), 0, width_px - 1)
            point_priority = priority[ys, xs]
            center_index = int(np.argmax(point_priority))
            center_distance = float(cumulative[center_index])
            keep = (cumulative >= center_distance - half_length) & (
                cumulative <= center_distance + half_length
            )
            fragment_yx = contour_yx[keep]
            fragment_priority = point_priority[keep]
            if len(fragment_yx) < 2 or float(fragment_priority.max()) <= 1e-5:
                continue
            simplified_yx = measure.approximate_polygon(
                fragment_yx, tolerance=float(max(0.0, simplify_tolerance))
            )
            if len(simplified_yx) < 2:
                continue
            frag_ys = np.clip(np.rint(fragment_yx[:, 0]).astype(int), 0, height - 1)
            frag_xs = np.clip(np.rint(fragment_yx[:, 1]).astype(int), 0, width_px - 1)
            solved = np.clip(
                (
                    target_srgb[frag_ys, frag_xs]
                    - (1.0 - alpha) * rendered_srgb[frag_ys, frag_xs]
                )
                / alpha,
                0.0,
                1.0,
            )
            weights = np.maximum(fragment_priority, 1e-4)
            color = tuple(float(v) for v in np.average(solved, axis=0, weights=weights))
            arc_length = float(
                np.sum(np.hypot(np.diff(fragment_yx[:, 0]), np.diff(fragment_yx[:, 1])))
            )
            score = float(fragment_priority.mean() * np.sqrt(max(arc_length, 1.0)))
            midpoint = simplified_yx[len(simplified_yx) // 2]
            candidates.append(
                (
                    score,
                    (float(midpoint[1]), float(midpoint[0])),
                    simplified_yx,
                    color,
                )
            )

    selected: list[EdgePath] = []
    centers: list[tuple[float, float]] = []
    for score, center, points_yx, color in sorted(
        candidates, key=lambda item: item[0], reverse=True
    ):
        if any(
            np.hypot(center[0] - old[0], center[1] - old[1]) < min_spacing
            for old in centers
        ):
            continue
        selected.append(
            EdgePath(
                points=tuple((float(point[1]), float(point[0])) for point in points_yx),
                width=float(max(0.1, width)),
                color_srgb=color,
                opacity=alpha,
                score=score,
            )
        )
        centers.append(center)
        if len(selected) >= wanted:
            break
    return selected


def edge_strokes_to_svg_group(
    strokes: Sequence[EdgeStroke],
    *,
    group_id: str = "residual-edge-strokes",
) -> str:
    """Emit editable native SVG line elements; no raster fallback."""

    children: list[str] = []
    for index, stroke in enumerate(strokes):
        rgb = tuple(
            int(np.clip(np.round(channel * 255.0), 0, 255))
            for channel in stroke.color_srgb
        )
        children.append(
            render_template(
                "svg/edge_line.svg",
                index=index,
                x1=f"{stroke.x1:.3f}",
                y1=f"{stroke.y1:.3f}",
                x2=f"{stroke.x2:.3f}",
                y2=f"{stroke.y2:.3f}",
                stroke=f"rgb({rgb[0]},{rgb[1]},{rgb[2]})",
                opacity=f"{stroke.opacity:.4f}",
                width=f"{stroke.width:.3f}",
            ).rstrip("\n")
        )
    child_block = "\n".join(children)
    if child_block:
        child_block += "\n"
    return render_template(
        "svg/edge_group.svg",
        group_id=group_id,
        group_class="edge-strokes",
        children=child_block,
    ).rstrip("\n")


def edge_paths_to_svg_group(
    paths: Sequence[EdgePath],
    *,
    group_id: str = "residual-edge-paths",
) -> str:
    children: list[str] = []
    for index, path in enumerate(paths):
        if len(path.points) < 2:
            continue
        rgb = tuple(
            int(np.clip(np.round(channel * 255.0), 0, 255))
            for channel in path.color_srgb
        )
        commands = [f"M {path.points[0][0]:.3f} {path.points[0][1]:.3f}"]
        commands.extend(f"L {x:.3f} {y:.3f}" for x, y in path.points[1:])
        children.append(
            render_template(
                "svg/edge_path.svg",
                index=index,
                commands=" ".join(commands),
                stroke=f"rgb({rgb[0]},{rgb[1]},{rgb[2]})",
                opacity=f"{path.opacity:.4f}",
                width=f"{path.width:.3f}",
            ).rstrip("\n")
        )
    child_block = "\n".join(children)
    if child_block:
        child_block += "\n"
    return render_template(
        "svg/edge_group.svg",
        group_id=group_id,
        group_class="edge-paths",
        children=child_block,
    ).rstrip("\n")


def _edge_segment_to_drawingml(
    path: EdgePath,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    shape_id: int,
    emu_scale: float,
) -> str:
    x1, y1 = start
    x2, y2 = end
    dx, dy = x2 - x1, y2 - y1
    length = float(np.hypot(dx, dy))
    if length <= 1e-4:
        return ""
    cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    stroke_width = float(max(0.1, path.width))
    x_emu = px_to_emu(cx - 0.5 * length, emu_scale)
    y_emu = px_to_emu(cy - 0.5 * stroke_width, emu_scale)
    w_emu = max(px_to_emu(length, emu_scale), 1)
    h_emu = max(px_to_emu(stroke_width, emu_scale), 1)
    rotation_units = int(round(float(np.degrees(np.arctan2(dy, dx))) * 60000.0))
    rgb = tuple(
        int(np.clip(np.round(channel * 255.0), 0, 255)) for channel in path.color_srgb
    )
    color_hex = f"{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}"
    alpha_units = int(np.clip(round(path.opacity * 100000.0), 0, 100000))
    return render_template(
        "drawingml/edge_segment.xml",
        shape_id=shape_id,
        rotation_units=rotation_units,
        x_emu=x_emu,
        y_emu=y_emu,
        w_emu=w_emu,
        h_emu=h_emu,
        color_hex=color_hex,
        alpha_units=alpha_units,
    ).rstrip("\n")


def inject_edge_paths_into_pptx(
    baseline_pptx: str | Path,
    output_pptx: str | Path,
    paths: Sequence[EdgePath],
    *,
    width: int,
    height: int,
) -> int:
    """Copy a PPTX and append editable rounded path segments to slide 1."""

    source = Path(baseline_pptx)
    output = Path(output_pptx)
    output.parent.mkdir(parents=True, exist_ok=True)
    emu_scale = pptx_emu_scale(width, height)
    segment_count = sum(max(0, len(path.points) - 1) for path in paths)

    with zipfile.ZipFile(source, "r") as input_zip:
        slide_xml = input_zip.read("ppt/slides/slide1.xml").decode("utf-8")
        root = ET.fromstring(slide_xml)
        ids = [
            int(value)
            for element in root.iter()
            if element.tag.rsplit("}", 1)[-1] == "cNvPr"
            if (value := element.attrib.get("id")) is not None
        ]
        shape_id = max(ids, default=1) + 1
        shape_fragments = []
        for path in paths:
            for start, end in zip(path.points, path.points[1:]):
                fragment = _edge_segment_to_drawingml(
                    path,
                    start,
                    end,
                    shape_id=shape_id,
                    emu_scale=emu_scale,
                )
                if fragment:
                    shape_fragments.append(fragment)
                    shape_id += 1
        insertion = "\n".join(shape_fragments)
        marker = load_template("drawingml/close_group.xml").strip("\n")
        marker_index = slide_xml.rfind(marker)
        if marker_index < 0:
            raise ValueError("slide1.xml has no root DrawingML group close")
        slide_xml = (
            slide_xml[:marker_index]
            + insertion
            + ("\n" if insertion else "")
            + slide_xml[marker_index:]
        )

        with zipfile.ZipFile(output, "w") as output_zip:
            for info in input_zip.infolist():
                data = (
                    slide_xml.encode("utf-8")
                    if info.filename == "ppt/slides/slide1.xml"
                    else input_zip.read(info.filename)
                )
                output_zip.writestr(info, data)
    return segment_count


def inject_svg_before_close(svg_content: str, fragment: str) -> str:
    marker = load_template("svg/close.svg").strip("\n")
    index = svg_content.rfind(marker)
    if index < 0:
        raise ValueError("SVG content has no closing root tag")
    return svg_content[:index] + fragment.rstrip() + "\n" + svg_content[index:]


__all__ = [
    "EdgePath",
    "EdgeStroke",
    "edge_paths_to_svg_group",
    "edge_strokes_to_svg_group",
    "inject_edge_paths_into_pptx",
    "inject_svg_before_close",
    "propose_residual_edge_paths",
    "propose_residual_edge_strokes",
]
