"""Color-space transforms shared by renderers and artifact exporters."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt


def srgb_to_linear(srgb: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """Convert normalized sRGB values to linear RGB."""

    values = np.asarray(srgb)
    return np.where(
        values <= 0.04045,
        values / 12.92,
        np.power((values + 0.055) / 1.055, 2.4),
    )


def linear_to_srgb(linear: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """Convert normalized linear RGB values to clipped sRGB."""

    values = np.asarray(linear)
    srgb = np.where(
        values <= 0.0031308,
        12.92 * values,
        1.055 * np.power(values, 1.0 / 2.4) - 0.055,
    )
    return np.clip(srgb, 0.0, 1.0)


def linear_to_srgb_float32(linear: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """Convert to sRGB with the clipped float32 semantics used by scorers."""

    values = np.clip(np.asarray(linear), 0.0, 1.0)
    return np.where(
        values <= 0.0031308,
        12.92 * values,
        1.055 * np.power(np.maximum(values, 1e-12), 1.0 / 2.4) - 0.055,
    ).astype(np.float32)


def srgb_to_linear_float32(srgb: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """Convert to linear RGB with the clipped float32 scorer semantics."""

    values = np.clip(np.asarray(srgb), 0.0, 1.0)
    return np.where(
        values <= 0.04045,
        values / 12.92,
        np.power((values + 0.055) / 1.055, 2.4),
    ).astype(np.float32)
