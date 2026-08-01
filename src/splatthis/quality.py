"""Pure image-quality metrics shared by all artifact evaluators."""

from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np
import numpy.typing as npt

from .color import linear_to_srgb

logger = logging.getLogger(__name__)


def _image_ssim(x: npt.NDArray[Any], y: npt.NDArray[Any]) -> float:
    """Compute standard windowed SSIM, with a documented legacy fallback."""

    x = np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0)
    y = np.clip(np.asarray(y, dtype=np.float64), 0.0, 1.0)
    # Two named reasons may fall back, and nothing else. The fallback is a
    # single-window global SSIM that reads about 0.10 higher than the windowed
    # metric on real artifacts -- 0.9776 against 0.8797 on the shipped demo
    # pair -- which is far above the 0.50 acceptance floor. Catching every
    # exception here meant any transient failure inside skimage silently
    # promoted a failing run to a passing one and recorded the inflated number
    # as though it were real. A computation that fails must raise.
    try:
        from skimage.metrics import structural_similarity
    except ImportError:
        logger.warning("skimage unavailable; falling back to global SSIM")
        return _global_ssim_np(x, y)

    # Windowed SSIM needs an odd window no larger than the shorter side. Tiny
    # inputs (test fixtures, degenerate crops) cannot carry one, and that is a
    # property of the input rather than a failure, so they fall back too.
    shortest_side = min(x.shape[0], x.shape[1])
    window = min(7, shortest_side if shortest_side % 2 else shortest_side - 1)
    if window < 3:
        logger.warning(
            "image too small for windowed SSIM (%dx%d); using global SSIM",
            x.shape[0],
            x.shape[1],
        )
        return _global_ssim_np(x, y)

    channel_axis = 2 if (x.ndim == 3 and x.shape[2] > 1) else None
    return float(
        structural_similarity(
            x, y, channel_axis=channel_axis, data_range=1.0, win_size=window
        )
    )


def _global_ssim_np(x: npt.NDArray[Any], y: npt.NDArray[Any]) -> float:
    """Global single-window SSIM retained only as a compatibility fallback."""

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
    return float(np.clip(np.mean(numerator / np.maximum(denominator, 1e-8)), -1.0, 1.0))


def compute_quality_metrics(
    target_linear_rgb: npt.NDArray[Any],
    candidate_linear_rgb: npt.NDArray[Any],
) -> Dict[str, float]:
    """Compute linear and display-sRGB L1, MSE, PSNR, and SSIM metrics."""

    target = np.clip(np.asarray(target_linear_rgb, dtype=np.float32), 0.0, 1.0)
    candidate = np.clip(np.asarray(candidate_linear_rgb, dtype=np.float32), 0.0, 1.0)
    if target.shape != candidate.shape:
        raise ValueError(
            f"Quality metric shape mismatch: target={target.shape}, "
            f"candidate={candidate.shape}"
        )
    l1 = float(np.mean(np.abs(candidate - target)))
    mse = float(np.mean((candidate - target) ** 2))
    target_srgb = linear_to_srgb(target)
    candidate_srgb = linear_to_srgb(candidate)
    mse_srgb = float(np.mean((candidate_srgb - target_srgb) ** 2))
    return {
        "l1": l1,
        "mse": mse,
        "psnr": float(-10.0 * np.log10(max(mse, 1e-12))),
        "ssim": _image_ssim(candidate, target),
        "psnr_srgb": float(-10.0 * np.log10(max(mse_srgb, 1e-12))),
        "ssim_srgb": _image_ssim(candidate_srgb, target_srgb),
    }
