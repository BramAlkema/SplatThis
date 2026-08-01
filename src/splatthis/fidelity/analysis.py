"""Residual analysis for the fidelity stage (ADR-003).

Phase-1 scope: OKLab residual, priority map, and deterministic fixed ROIs.
The full residual-topology classification (edge displacement, coverage,
opacity-order error) is Phase 3 of the ADR delivery plan.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt

from .metrics import Roi, linear_rgb_to_oklab_np


def centered_crop(*, x: int, y: int, size: int, shape: Tuple[int, int]) -> Roi:
    """Fixed-size crop centered on (x, y), clamped inside the image."""
    h, w = shape
    size = int(min(size, h, w))
    half = size // 2
    y0 = int(np.clip(y - half, 0, h - size))
    x0 = int(np.clip(x - half, 0, w - size))
    return (y0, x0, y0 + size, x0 + size)


def suppress_neighborhood(
    priority: npt.NDArray[Any], *, x: int, y: int, radius: int
) -> None:
    """Zero a square neighborhood in-place so ROIs spread out."""
    h, w = priority.shape
    y0, y1 = max(0, y - radius), min(h, y + radius + 1)
    x0, x1 = max(0, x - radius), min(w, x + radius + 1)
    priority[y0:y1, x0:x1] = 0.0


def select_fixed_rois(
    error_map: npt.NDArray[Any],
    saliency: Optional[npt.NDArray[Any]] = None,
    size: int = 64,
    count: int = 8,
) -> Tuple[Roi, ...]:
    """Deterministically pick the worst fixed-size windows from the baseline.

    The ROIs stay FIXED while candidates are compared — otherwise each
    candidate would be judged on a different set of easy or hard crops.
    """
    error_map = np.asarray(error_map, dtype=np.float32)
    if saliency is not None:
        priority = error_map * (1.0 + np.asarray(saliency, dtype=np.float32))
    else:
        priority = error_map.copy()
    suppressed = priority.copy()
    rois = []
    for _ in range(int(count)):
        if not np.isfinite(suppressed).any() or suppressed.max() <= 0.0:
            break
        y, x = np.unravel_index(int(np.argmax(suppressed)), suppressed.shape)
        rois.append(centered_crop(x=int(x), y=int(y), size=size, shape=priority.shape))
        suppress_neighborhood(suppressed, x=int(x), y=int(y), radius=size // 2)
    return tuple(rois)


@dataclass(frozen=True)
class ResidualAnalysis:
    """Shared analysis bundle handed to candidate operators."""

    residual_oklab: npt.NDArray[
        Any
    ]  # [H, W, 3] signed OKLab residual (target - rendered)
    residual_linear: npt.NDArray[Any]  # [H, W, 3] signed linear-RGB residual
    absolute_color_error: npt.NDArray[Any]  # [H, W] OKLab distance
    priority: npt.NDArray[Any]  # [H, W] error x (1 + saliency)
    fixed_rois: Tuple[Roi, ...]


def analyze_residual(
    target_linear_rgb: npt.NDArray[Any],
    rendered_linear_rgb: npt.NDArray[Any],
    *,
    saliency: Optional[npt.NDArray[Any]] = None,
    fixed_rois: Optional[Sequence[Roi]] = None,
    roi_size: int = 64,
    roi_count: int = 8,
) -> ResidualAnalysis:
    target = np.clip(np.asarray(target_linear_rgb, dtype=np.float32)[..., :3], 0, 1)
    rendered = np.clip(np.asarray(rendered_linear_rgb, dtype=np.float32)[..., :3], 0, 1)
    lab_t = linear_rgb_to_oklab_np(target)
    lab_r = linear_rgb_to_oklab_np(rendered)
    residual_oklab = lab_t - lab_r
    error = np.sqrt(np.sum(residual_oklab**2, axis=-1)).astype(np.float32)
    if saliency is not None:
        priority = error * (1.0 + np.asarray(saliency, dtype=np.float32))
    else:
        priority = error.copy()
    if fixed_rois is None:
        fixed_rois = select_fixed_rois(error, saliency, size=roi_size, count=roi_count)
    return ResidualAnalysis(
        residual_oklab=residual_oklab,
        residual_linear=(target - rendered).astype(np.float32),
        absolute_color_error=error,
        priority=priority,
        fixed_rois=tuple(fixed_rois),
    )
