#!/usr/bin/env python3
"""
Reconstruction error computation utilities for progressive allocation.

This module provides functions for computing per-pixel reconstruction error
between target and rendered images, supporting both L1 and L2 metrics.
"""

import numpy as np
import logging
from typing import Optional, Union

logger = logging.getLogger(__name__)


def compute_l1_error(
    target: np.ndarray,
    rendered: np.ndarray,
    normalize: bool = True
) -> np.ndarray:
    """Compute per-pixel L1 (Manhattan) reconstruction error.

    Args:
        target: Target image (H, W) or (H, W, C), any numeric dtype
        rendered: Rendered image (H, W) or (H, W, C), any numeric dtype
        normalize: Whether to normalize images to [0, 1] range

    Returns:
        Per-pixel L1 error map (H, W) as float32

    Raises:
        ValueError: If images have incompatible shapes
    """
    # Validate input shapes
    if target.shape != rendered.shape:
        raise ValueError(
            f"Target and rendered images must have same shape. "
            f"Got target: {target.shape}, rendered: {rendered.shape}"
        )

    # Convert to float32 for computation
    target_f = _normalize_image(target) if normalize else target.astype(np.float32)
    rendered_f = _normalize_image(rendered) if normalize else rendered.astype(np.float32)

    # Compute L1 error per pixel
    if len(target_f.shape) == 3:
        # Multi-channel image: average error across channels
        error_map = np.mean(np.abs(target_f - rendered_f), axis=2)
    else:
        # Single-channel image
        error_map = np.abs(target_f - rendered_f)

    return error_map.astype(np.float32)


def compute_l2_error(
    target: np.ndarray,
    rendered: np.ndarray,
    normalize: bool = True
) -> np.ndarray:
    """Compute per-pixel L2 (Euclidean) reconstruction error.

    Args:
        target: Target image (H, W) or (H, W, C), any numeric dtype
        rendered: Rendered image (H, W) or (H, W, C), any numeric dtype
        normalize: Whether to normalize images to [0, 1] range

    Returns:
        Per-pixel L2 error map (H, W) as float32

    Raises:
        ValueError: If images have incompatible shapes
    """
    if target.shape != rendered.shape:
        raise ValueError(
            f"Target and rendered images must have same shape. "
            f"Got target: {target.shape}, rendered: {rendered.shape}"
        )

    target_f = _normalize_image(target) if normalize else target.astype(np.float32)
    rendered_f = _normalize_image(rendered) if normalize else rendered.astype(np.float32)

    if len(target_f.shape) == 3:
        # Multi-channel: Euclidean distance across channels
        error_map = np.sqrt(np.sum((target_f - rendered_f) ** 2, axis=2))
    else:
        # Single-channel: same as L1 for single values
        error_map = np.abs(target_f - rendered_f)

    return error_map.astype(np.float32)


def compute_mse_error(
    target: np.ndarray,
    rendered: np.ndarray,
    normalize: bool = True
) -> np.ndarray:
    """Compute per-pixel Mean Squared Error (MSE).

    Args:
        target: Target image (H, W) or (H, W, C), any numeric dtype
        rendered: Rendered image (H, W) or (H, W, C), any numeric dtype
        normalize: Whether to normalize images to [0, 1] range

    Returns:
        Per-pixel MSE error map (H, W) as float32

    Raises:
        ValueError: If images have incompatible shapes
    """
    if target.shape != rendered.shape:
        raise ValueError(
            f"Target and rendered images must have same shape. "
            f"Got target: {target.shape}, rendered: {rendered.shape}"
        )

    target_f = _normalize_image(target) if normalize else target.astype(np.float32)
    rendered_f = _normalize_image(rendered) if normalize else rendered.astype(np.float32)

    if len(target_f.shape) == 3:
        # Multi-channel: average MSE across channels
        error_map = np.mean((target_f - rendered_f) ** 2, axis=2)
    else:
        # Single-channel
        error_map = (target_f - rendered_f) ** 2

    return error_map.astype(np.float32)


def compute_weighted_error(
    target: np.ndarray,
    rendered: np.ndarray,
    weight_map: Optional[np.ndarray] = None,
    metric: str = "l1",
    normalize: bool = True
) -> np.ndarray:
    """Compute weighted reconstruction error with optional importance map.

    Args:
        target: Target image (H, W) or (H, W, C), any numeric dtype
        rendered: Rendered image (H, W) or (H, W, C), any numeric dtype
        weight_map: Optional weight map (H, W) with importance weights
        metric: Error metric to use ("l1", "l2", "mse")
        normalize: Whether to normalize images to [0, 1] range

    Returns:
        Per-pixel weighted error map (H, W) as float32

    Raises:
        ValueError: If images have incompatible shapes or invalid metric
    """
    # Compute base error
    if metric == "l1":
        error_map = compute_l1_error(target, rendered, normalize)
    elif metric == "l2":
        error_map = compute_l2_error(target, rendered, normalize)
    elif metric == "mse":
        error_map = compute_mse_error(target, rendered, normalize)
    else:
        raise ValueError(f"Unknown metric: {metric}. Use 'l1', 'l2', or 'mse'")

    # Apply weights if provided
    if weight_map is not None:
        if weight_map.shape != error_map.shape:
            raise ValueError(
                f"Weight map shape {weight_map.shape} doesn't match "
                f"error map shape {error_map.shape}"
            )
        error_map = error_map * weight_map.astype(np.float32)

    return error_map


def compute_error_statistics(error_map: np.ndarray) -> dict:
    """Compute comprehensive statistics about reconstruction error.

    Args:
        error_map: Per-pixel error map (H, W)

    Returns:
        Dictionary with error statistics
    """
    if error_map.size == 0:
        return {'empty': True}

    flat_errors = error_map.flatten()
    non_zero_errors = flat_errors[flat_errors > 0]

    stats = {
        'mean_error': float(np.mean(flat_errors)),
        'max_error': float(np.max(flat_errors)),
        'min_error': float(np.min(flat_errors)),
        'std_error': float(np.std(flat_errors)),
        'total_error': float(np.sum(flat_errors)),
        'rms_error': float(np.sqrt(np.mean(flat_errors ** 2))),
        'error_pixels': int(np.sum(flat_errors > 0)),
        'zero_error_pixels': int(np.sum(flat_errors == 0)),
        'shape': error_map.shape,
        'percentiles': {
            '50': float(np.percentile(flat_errors, 50)),
            '75': float(np.percentile(flat_errors, 75)),
            '90': float(np.percentile(flat_errors, 90)),
            '95': float(np.percentile(flat_errors, 95)),
            '99': float(np.percentile(flat_errors, 99))
        }
    }

    # Add statistics for non-zero errors if they exist
    if len(non_zero_errors) > 0:
        stats['non_zero_stats'] = {
            'mean': float(np.mean(non_zero_errors)),
            'std': float(np.std(non_zero_errors)),
            'min': float(np.min(non_zero_errors)),
            'max': float(np.max(non_zero_errors))
        }

    return stats


def compute_psnr(
    target: np.ndarray,
    rendered: np.ndarray,
    max_value: Optional[float] = None,
    normalize: bool = True
) -> float:
    """Compute Peak Signal-to-Noise Ratio (PSNR) between images.

    Args:
        target: Target image (H, W) or (H, W, C), any numeric dtype
        rendered: Rendered image (H, W) or (H, W, C), any numeric dtype
        max_value: Maximum possible pixel value. If None, inferred from dtype
        normalize: Whether to normalize images to [0, 1] range

    Returns:
        PSNR value in dB

    Raises:
        ValueError: If images have incompatible shapes
    """
    if target.shape != rendered.shape:
        raise ValueError(
            f"Target and rendered images must have same shape. "
            f"Got target: {target.shape}, rendered: {rendered.shape}"
        )

    target_f = _normalize_image(target) if normalize else target.astype(np.float32)
    rendered_f = _normalize_image(rendered) if normalize else rendered.astype(np.float32)

    # Determine maximum value
    if max_value is None:
        if normalize:
            max_value = 1.0
        else:
            if target.dtype == np.uint8:
                max_value = 255.0
            elif target.dtype == np.uint16:
                max_value = 65535.0
            else:
                max_value = float(np.max(target))

    # Compute MSE
    mse = np.mean((target_f - rendered_f) ** 2)

    # Handle perfect match
    if mse == 0:
        return float('inf')

    # Compute PSNR
    psnr = 20 * np.log10(max_value / np.sqrt(mse))
    return float(psnr)


def _normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize image to [0, 1] range based on data type.

    Args:
        image: Input image

    Returns:
        Normalized image as float32
    """
    if image.dtype == np.uint8:
        return image.astype(np.float32) / 255.0
    elif image.dtype == np.uint16:
        return image.astype(np.float32) / 65535.0
    elif image.dtype == np.uint32:
        return image.astype(np.float32) / 4294967295.0
    else:
        # Assume already normalized or float
        return image.astype(np.float32)


def validate_images(target: np.ndarray, rendered: np.ndarray) -> None:
    """Validate that images are compatible for error computation.

    Args:
        target: Target image
        rendered: Rendered image

    Raises:
        ValueError: If images are incompatible
    """
    if target.shape != rendered.shape:
        raise ValueError(
            f"Images must have same shape. "
            f"Got target: {target.shape}, rendered: {rendered.shape}"
        )

    if target.size == 0:
        raise ValueError("Images cannot be empty")

    if len(target.shape) not in [2, 3]:
        raise ValueError(
            f"Images must be 2D or 3D arrays, got {len(target.shape)}D"
        )

    if len(target.shape) == 3 and target.shape[2] not in [1, 3, 4]:
        logger.warning(
            f"Unusual number of channels: {target.shape[2]}. "
            f"Expected 1, 3, or 4 channels"
        )


# Convenience function that matches the interface used in ErrorGuidedPlacement
def compute_reconstruction_error(
    target: np.ndarray,
    rendered: np.ndarray,
    metric: str = "l1"
) -> np.ndarray:
    """Compute reconstruction error using specified metric.

    This is a convenience function that provides the same interface as
    ErrorGuidedPlacement.compute_reconstruction_error().

    Args:
        target: Target image (H, W) or (H, W, C)
        rendered: Rendered image (H, W) or (H, W, C)
        metric: Error metric ("l1", "l2", "mse")

    Returns:
        Per-pixel error map (H, W) as float32
    """
    validate_images(target, rendered)

    if metric == "l1":
        return compute_l1_error(target, rendered)
    elif metric == "l2":
        return compute_l2_error(target, rendered)
    elif metric == "mse":
        return compute_mse_error(target, rendered)
    else:
        raise ValueError(f"Unknown metric: {metric}. Use 'l1', 'l2', or 'mse'")