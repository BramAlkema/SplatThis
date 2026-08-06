"""Small shared geometry layer for CSS, Canvas, and DrawingML exporters."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt

from .splat import SPLAT_LAYER_NAMES, GaussianSplat, render_order_key

EMU_PER_PX = 9525
ELLIPSE_OVERLAP_BOOST = 1.15
MIN_ELLIPSE_RADIUS_PX = 0.35
SVG_GRADIENT_STOPS = 8
SVG_GRADIENT_STOPS_MIN = 2
SVG_GRADIENT_STOP_MAX_ERROR = 0.05

DEFAULT_EXPORT_ORDER = "importance"
DEFAULT_PPTX_SPLAT_STYLE = "gradient"
PPTX_PAINTER_ORDER_LEGACY = "legacy"
PPTX_PAINTER_ORDER_BACK_TO_FRONT = "back-to-front"
PPTX_SOFT_EDGE_ALPHA_SCALE = 0.25
PPTX_SOFT_EDGE_RADIUS_FACTOR = 0.20
PPTX_SOFT_EDGE_K_SIGMA_SCALE = 0.92
PPTX_GRADIENT_ALPHA_SCALE = 0.40
PPTX_BLUR_RAD_PER_SIGMA = 3.25
PPTX_BLUR_CORE_K_SIGMA = 1.0
PPTX_PROXY_MODES = ("none", "softedge", "gradient")
MIN_SLIDE_EMU = 914400


def _density_aware_stop_error(
    splat_count: int,
    *,
    baseline: float = SVG_GRADIENT_STOP_MAX_ERROR,
    floor: float = 0.01,
    ceiling: float = 0.05,
) -> float:
    if splat_count <= 0:
        return float(baseline)
    return float(np.clip(100.0 / float(splat_count), floor, ceiling))


def _splat_layer(splat: GaussianSplat) -> Optional[int]:
    return splat.to_raw_splat().layer


def _layer_title(layer: Optional[int]) -> str:
    name = (
        "unassigned"
        if layer is None
        else SPLAT_LAYER_NAMES.get(layer, f"layer-{layer}")
    )
    return name.replace("-", " ").title()


def _gaussian_opacity_curve(
    t: npt.NDArray[np.floating], alpha: float, gradient_footprint: float
) -> npt.NDArray[np.floating]:
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
    opacity_precision: int = 2,
) -> List[Tuple[float, float]]:
    min_stops = max(2, int(min_stops))
    max_stops = max(min_stops, int(max_stops))
    if alpha <= 1e-6 or gradient_footprint <= 0.0:
        return [(0.0, 0.0), (float(inner_end), 0.0)]

    sample_t = np.linspace(0.0, 1.0, 65)
    true_opacity = _gaussian_opacity_curve(sample_t, alpha, gradient_footprint)
    for count in range(min_stops, max_stops + 1):
        stop_t = np.linspace(0.0, 1.0, count)
        stop_opacity = np.round(
            _gaussian_opacity_curve(stop_t, alpha, gradient_footprint),
            opacity_precision,
        )
        if (
            float(
                np.max(np.abs(np.interp(sample_t, stop_t, stop_opacity) - true_opacity))
            )
            <= max_error
        ):
            return [
                (float(t * inner_end), float(opacity))
                for t, opacity in zip(stop_t, stop_opacity)
            ]

    stop_t = np.linspace(0.0, 1.0, max_stops)
    stop_opacity = np.round(
        _gaussian_opacity_curve(stop_t, alpha, gradient_footprint),
        opacity_precision,
    )
    return [
        (float(t * inner_end), float(opacity))
        for t, opacity in zip(stop_t, stop_opacity)
    ]


def _normalize_pptx_splat_style(value: str) -> str:
    normalized = str(value).strip().lower().replace("_", "-")
    aliases = {
        "softedge": "soft-edge",
        "soft-edge": "soft-edge",
        "soft": "soft-edge",
        "gradient": "gradient",
        "grad": "gradient",
        "blur": "blur",
        "gaussian-blur": "blur",
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported PPTX splat style: {value}")
    return aliases[normalized]


def _normalize_pptx_painter_order(value: str) -> str:
    normalized = str(value).strip().lower().replace("_", "-")
    if normalized in {"", "legacy", "forward", "front-to-back"}:
        return PPTX_PAINTER_ORDER_LEGACY
    if normalized in {"back-to-front", "correct", "corrected", "painter"}:
        return PPTX_PAINTER_ORDER_BACK_TO_FRONT
    raise ValueError(f"Unsupported PPTX painter order: {value}")


def _pptx_painter_splats(
    splats: List[GaussianSplat],
    painter_order: str = PPTX_PAINTER_ORDER_LEGACY,
) -> List[GaussianSplat]:
    if _normalize_pptx_painter_order(painter_order) == PPTX_PAINTER_ORDER_LEGACY:
        return list(splats)
    return list(reversed(splats))


def _sort_splats_for_export(
    splats: List[GaussianSplat],
    sort_mode: str = DEFAULT_EXPORT_ORDER,
    sort_by_area: bool = False,
) -> List[GaussianSplat]:
    if sort_by_area:
        return sorted(splats, key=lambda splat: splat.area(), reverse=True)
    normalized = str(sort_mode).strip().lower()
    if normalized == "input":
        return list(splats)
    if normalized == "area":
        return sorted(splats, key=lambda splat: splat.area(), reverse=True)
    if normalized == "importance":
        return sorted(splats, key=render_order_key)
    raise ValueError(f"Unsupported export sort mode: {sort_mode}")


def pptx_emu_scale(width: int, height: int) -> float:
    width_emu = max(int(round(width * EMU_PER_PX)), 1)
    height_emu = max(int(round(height * EMU_PER_PX)), 1)
    return max(1.0, MIN_SLIDE_EMU / width_emu, MIN_SLIDE_EMU / height_emu)


def px_to_emu(value: float, scale: float = 1.0) -> int:
    return int(round(value * EMU_PER_PX * scale))
