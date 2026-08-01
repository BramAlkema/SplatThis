"""Governing browser evaluation for emitted deployment artifacts."""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import numpy.typing as npt

from .quality import compute_quality_metrics

logger = logging.getLogger(__name__)


def _evaluate_browser_export_quality(
    target_linear_rgb: npt.NDArray[Any],
    fallback_linear_rgb: Optional[npt.NDArray[Any]] = None,
    *,
    artifact_name: str,
    capture: Callable[[int, int], Tuple[npt.NDArray[Any], str]],
) -> Dict[str, Any]:
    """Grade a browser artifact while keeping proxy evidence diagnostic-only."""

    target = np.asarray(target_linear_rgb, dtype=np.float32)
    height, width = target.shape[:2]
    try:
        rendered, method = capture(width, height)
    except RuntimeError as exc:
        method = f"unavailable:{exc}"
        fallback = (
            None
            if fallback_linear_rgb is None
            else np.asarray(fallback_linear_rgb, dtype=np.float32)
        )
        if fallback is not None:
            logger.warning(
                "%s could not be captured in Chromium (%s); export-quality metrics "
                "fall back to the numpy proxy render, which does NOT reflect deployed "
                "browser fidelity. Install the capture extra and configure Chrome to "
                "measure it.",
                artifact_name,
                method,
            )
        return {
            "available": False,
            "method": "proxy-fallback" if fallback is not None else method,
            "governing_method": method,
            "used_fallback": fallback is not None,
            "metrics": (
                None if fallback is None else compute_quality_metrics(target, fallback)
            ),
        }
    return {
        "available": True,
        "method": method,
        "used_fallback": False,
        "metrics": compute_quality_metrics(target, rendered),
    }


def evaluate_svg_export_quality(
    target_linear_rgb: npt.NDArray[Any],
    svg_path: str,
    fallback_linear_rgb: Optional[npt.NDArray[Any]] = None,
) -> Dict[str, Any]:
    from .browser_capture import render_svg_in_browser_to_linear_rgb

    return _evaluate_browser_export_quality(
        target_linear_rgb,
        fallback_linear_rgb,
        artifact_name="SVG",
        capture=lambda width, height: render_svg_in_browser_to_linear_rgb(
            svg_path=svg_path, width=width, height=height
        ),
    )


def evaluate_css_export_quality(
    target_linear_rgb: npt.NDArray[Any],
    html_path: str,
    fallback_linear_rgb: Optional[npt.NDArray[Any]] = None,
) -> Dict[str, Any]:
    from .browser_capture import render_css_html_in_browser_to_linear_rgb

    return _evaluate_browser_export_quality(
        target_linear_rgb,
        fallback_linear_rgb,
        artifact_name="CSS compositor",
        capture=lambda width, height: render_css_html_in_browser_to_linear_rgb(
            html_path=html_path, width=width, height=height
        ),
    )


def evaluate_native_canvas_export_quality(
    target_linear_rgb: npt.NDArray[Any],
    html_path: str,
    fallback_linear_rgb: Optional[npt.NDArray[Any]] = None,
) -> Dict[str, Any]:
    from .browser_capture import render_canvas_html_in_browser_to_linear_rgb

    return _evaluate_browser_export_quality(
        target_linear_rgb,
        fallback_linear_rgb,
        artifact_name="Canvas compositor",
        capture=lambda width, height: render_canvas_html_in_browser_to_linear_rgb(
            html_path=html_path, width=width, height=height
        ),
    )


def evaluate_pixel_runtime_export_quality(
    target_linear_rgb: npt.NDArray[Any],
    html_path: str,
    fallback_linear_rgb: Optional[npt.NDArray[Any]] = None,
) -> Dict[str, Any]:
    from .browser_capture import render_pixel_runtime_html_in_browser_to_linear_rgb

    return _evaluate_browser_export_quality(
        target_linear_rgb,
        fallback_linear_rgb,
        artifact_name="Pixel runtime",
        capture=lambda width, height: render_pixel_runtime_html_in_browser_to_linear_rgb(
            html_path=html_path, width=width, height=height
        ),
    )
