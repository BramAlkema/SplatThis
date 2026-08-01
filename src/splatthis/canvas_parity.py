"""Calibration helpers for the in-process Canvas model versus Chrome pixels."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class CanvasParityObservation:
    """One unchanged checkpoint scored by the model and by real Chrome."""

    image: str
    checkpoint: str
    model_ssim_srgb: float
    browser_ssim_srgb: float
    model_psnr_srgb: float
    browser_psnr_srgb: float
    pixel_parity_ssim_srgb: float
    pixel_mae_linear: float
    pixel_max_abs_linear: float

    @property
    def ssim_overstatement(self) -> float:
        """Positive when the model score is higher than Chrome's score."""

        return float(self.model_ssim_srgb - self.browser_ssim_srgb)

    @property
    def psnr_overstatement(self) -> float:
        """Positive when the model score is higher than Chrome's score."""

        return float(self.model_psnr_srgb - self.browser_psnr_srgb)

    def as_dict(self) -> Dict[str, Any]:
        return {
            **self.__dict__,
            "ssim_overstatement": self.ssim_overstatement,
            "psnr_overstatement": self.psnr_overstatement,
        }


def ceil_margin(value: float, quantum: float) -> float:
    """Round a non-negative observed bias up to a readable safety quantum."""

    if quantum <= 0.0:
        raise ValueError("quantum must be positive")
    positive = max(0.0, float(value))
    if positive == 0.0:
        return 0.0
    units = math.ceil((positive / quantum) - 1e-12)
    return float(units * quantum)


def _distribution(values: Iterable[float]) -> Dict[str, float]:
    materialized = [float(value) for value in values]
    if not materialized:
        raise ValueError("at least one value is required")
    return {
        "min": min(materialized),
        "median": float(statistics.median(materialized)),
        "p95": float(np.percentile(materialized, 95)),
        "max": max(materialized),
    }


def summarize_canvas_parity(
    observations: Sequence[CanvasParityObservation],
    *,
    ssim_quantum: float = 0.0001,
    psnr_quantum: float = 0.01,
) -> Dict[str, Any]:
    """Summarize model-to-browser bias and recommend conservative margins."""

    if not observations:
        raise ValueError("at least one parity observation is required")
    ssim_overstatement = [item.ssim_overstatement for item in observations]
    psnr_overstatement = [item.psnr_overstatement for item in observations]
    max_ssim_overstatement = max(0.0, max(ssim_overstatement))
    max_psnr_overstatement = max(0.0, max(psnr_overstatement))
    return {
        "observation_count": len(observations),
        "image_count": len({item.image for item in observations}),
        "ssim_overstatement": _distribution(ssim_overstatement),
        "psnr_overstatement": _distribution(psnr_overstatement),
        "pixel_parity_ssim_srgb": _distribution(
            item.pixel_parity_ssim_srgb for item in observations
        ),
        "pixel_mae_linear": _distribution(
            item.pixel_mae_linear for item in observations
        ),
        "pixel_max_abs_linear": _distribution(
            item.pixel_max_abs_linear for item in observations
        ),
        "recommended_ssim_safety_margin": ceil_margin(
            max_ssim_overstatement, ssim_quantum
        ),
        "recommended_psnr_safety_margin": ceil_margin(
            max_psnr_overstatement, psnr_quantum
        ),
        "ssim_margin_quantum": float(ssim_quantum),
        "psnr_margin_quantum": float(psnr_quantum),
    }
