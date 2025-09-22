#!/usr/bin/env python3
"""
Error-guided splat placement for progressive allocation.

This module implements error computation and sampling strategies for placing
new Gaussian splats based on reconstruction error.
"""

import logging
from typing import List, Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)


class ErrorGuidedPlacement:
    """Manages error-guided placement of new Gaussian splats.

    This class computes reconstruction error maps and samples new splat positions
    based on error distribution with temperature-controlled sampling.
    """

    def __init__(self, temperature: float = 2.0):
        """Initialize error-guided placement.

        Args:
            temperature: Sampling temperature for error distribution.
                         Lower values = more focused sampling on high-error regions.
                         Higher values = more uniform sampling.
        """
        if temperature <= 0:
            raise ValueError(f"Temperature must be positive, got {temperature}")

        self.temperature = temperature

    def compute_reconstruction_error(self, target: np.ndarray, rendered: np.ndarray) -> np.ndarray:
        """Compute per-pixel reconstruction error between target and rendered images.

        Args:
            target: Original target image (H, W, C) or (H, W)
            rendered: Rendered splat image (H, W, C) or (H, W)

        Returns:
            Error map (H, W) with per-pixel L1 distance

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
        target_f = self._normalize_image(target)
        rendered_f = self._normalize_image(rendered)

        # Compute L1 error per pixel
        if len(target_f.shape) == 3:
            # Multi-channel image: average error across channels
            error_map = np.mean(np.abs(target_f - rendered_f), axis=2)
        else:
            # Single-channel image
            error_map = np.abs(target_f - rendered_f)

        return error_map

    def compute_l2_error(self, target: np.ndarray, rendered: np.ndarray) -> np.ndarray:
        """Compute per-pixel L2 (Euclidean) reconstruction error.

        Args:
            target: Original target image (H, W, C) or (H, W)
            rendered: Rendered splat image (H, W, C) or (H, W)

        Returns:
            Error map (H, W) with per-pixel L2 distance
        """
        if target.shape != rendered.shape:
            raise ValueError(
                f"Target and rendered images must have same shape. "
                f"Got target: {target.shape}, rendered: {rendered.shape}"
            )

        target_f = self._normalize_image(target)
        rendered_f = self._normalize_image(rendered)

        if len(target_f.shape) == 3:
            # Multi-channel: Euclidean distance across channels
            error_map = np.sqrt(np.sum((target_f - rendered_f) ** 2, axis=2))
        else:
            # Single-channel
            error_map = np.abs(target_f - rendered_f)

        return error_map

    def create_placement_probability(self, error_map: np.ndarray) -> np.ndarray:
        """Convert error map to placement probability distribution.

        Args:
            error_map: Per-pixel error (H, W)

        Returns:
            Probability map (H, W) normalized to sum to 1

        Raises:
            ValueError: If error_map is empty or contains invalid values
        """
        if error_map.size == 0:
            raise ValueError("Error map cannot be empty")

        if np.any(error_map < 0):
            raise ValueError("Error map cannot contain negative values")

        # Handle edge case: all zeros
        if np.all(error_map == 0):
            logger.warning("Error map is all zeros, using uniform distribution")
            uniform_prob = np.ones_like(error_map) / error_map.size
            return uniform_prob

        # Add small epsilon to avoid division by zero
        error_safe = error_map + 1e-8

        # Apply temperature for sampling control
        # Lower temperature = more focused on high-error regions
        # Higher temperature = more uniform sampling
        prob_map = error_safe ** (1.0 / self.temperature)

        # Normalize to probability distribution
        prob_sum = np.sum(prob_map)
        if prob_sum == 0:
            # Fallback to uniform if something went wrong
            logger.warning("Probability sum is zero, using uniform distribution")
            return np.ones_like(error_map) / error_map.size

        prob_map = prob_map / prob_sum

        return prob_map

    def sample_positions(self, prob_map: np.ndarray, count: int,
                        min_distance: Optional[float] = None) -> List[Tuple[int, int]]:
        """Sample new splat positions from probability distribution.

        Args:
            prob_map: Probability distribution (H, W)
            count: Number of positions to sample
            min_distance: Minimum distance between samples (optional)

        Returns:
            List of (y, x) positions

        Raises:
            ValueError: If count is invalid or prob_map is malformed
        """
        if count <= 0:
            return []

        if prob_map.size == 0:
            raise ValueError("Probability map cannot be empty")

        if not np.isclose(np.sum(prob_map), 1.0, atol=1e-6):
            logger.warning(f"Probability map sum is {np.sum(prob_map):.6f}, should be 1.0")

        height, width = prob_map.shape
        total_pixels = height * width

        # Limit count to available pixels
        actual_count = min(count, total_pixels)

        # Flatten probability map for sampling
        flat_probs = prob_map.flatten()

        # Sample indices without replacement
        try:
            sampled_indices = np.random.choice(
                total_pixels,
                size=actual_count,
                replace=False,
                p=flat_probs
            )
        except ValueError as e:
            logger.error(f"Sampling failed: {e}")
            # Fallback to uniform sampling
            sampled_indices = np.random.choice(
                total_pixels,
                size=actual_count,
                replace=False
            )

        # Convert flat indices back to 2D coordinates
        positions = []
        for idx in sampled_indices:
            y, x = divmod(idx, width)
            positions.append((y, x))

        # Apply minimum distance constraint if specified
        if min_distance is not None and min_distance > 0:
            positions = self._enforce_min_distance(positions, min_distance)

        return positions

    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image to [0, 1] range.

        Args:
            image: Input image

        Returns:
            Normalized image as float32
        """
        if image.dtype == np.uint8:
            return image.astype(np.float32) / 255.0
        elif image.dtype in [np.uint16]:
            return image.astype(np.float32) / 65535.0
        else:
            # Assume already normalized or float
            return image.astype(np.float32)

    def _enforce_min_distance(self, positions: List[Tuple[int, int]],
                             min_distance: float) -> List[Tuple[int, int]]:
        """Remove positions that are too close to each other.

        Args:
            positions: List of (y, x) positions
            min_distance: Minimum distance between positions

        Returns:
            Filtered list of positions
        """
        if len(positions) <= 1:
            return positions

        filtered_positions = [positions[0]]  # Always keep first position

        for candidate in positions[1:]:
            candidate_y, candidate_x = candidate

            # Check distance to all accepted positions
            too_close = False
            for accepted_y, accepted_x in filtered_positions:
                distance = np.sqrt((candidate_y - accepted_y)**2 + (candidate_x - accepted_x)**2)
                if distance < min_distance:
                    too_close = True
                    break

            if not too_close:
                filtered_positions.append(candidate)

        if len(filtered_positions) < len(positions):
            logger.debug(
                f"Filtered {len(positions) - len(filtered_positions)} positions "
                f"due to min_distance constraint ({min_distance})"
            )

        return filtered_positions

    def get_error_statistics(self, error_map: np.ndarray) -> dict:
        """Compute statistics about the error map for analysis.

        Args:
            error_map: Per-pixel error (H, W)

        Returns:
            Dictionary with error statistics
        """
        if error_map.size == 0:
            return {'empty': True}

        return {
            'mean_error': float(np.mean(error_map)),
            'max_error': float(np.max(error_map)),
            'min_error': float(np.min(error_map)),
            'std_error': float(np.std(error_map)),
            'total_error': float(np.sum(error_map)),
            'error_pixels': int(np.sum(error_map > 0)),
            'zero_error_pixels': int(np.sum(error_map == 0)),
            'shape': error_map.shape,
            'percentiles': {
                '50': float(np.percentile(error_map, 50)),
                '75': float(np.percentile(error_map, 75)),
                '90': float(np.percentile(error_map, 90)),
                '95': float(np.percentile(error_map, 95)),
                '99': float(np.percentile(error_map, 99))
            }
        }

    def visualize_probability_map(self, prob_map: np.ndarray,
                                 sampled_positions: Optional[List[Tuple[int, int]]] = None) -> np.ndarray:
        """Create a visualization of the probability map with optional sample points.

        Args:
            prob_map: Probability distribution (H, W)
            sampled_positions: Optional list of sampled positions to highlight

        Returns:
            Visualization image (H, W, 3) as uint8
        """
        # Normalize probability map to [0, 255] for visualization
        prob_vis = (prob_map * 255 / np.max(prob_map)).astype(np.uint8)

        # Create RGB image (red channel for probability)
        vis_image = np.zeros((*prob_map.shape, 3), dtype=np.uint8)
        vis_image[:, :, 0] = prob_vis  # Red channel shows probability

        # Mark sampled positions if provided
        if sampled_positions:
            for y, x in sampled_positions:
                if 0 <= y < vis_image.shape[0] and 0 <= x < vis_image.shape[1]:
                    # Mark with green cross
                    vis_image[max(0, y-2):min(vis_image.shape[0], y+3), x, 1] = 255
                    vis_image[y, max(0, x-2):min(vis_image.shape[1], x+3), 1] = 255

        return vis_image