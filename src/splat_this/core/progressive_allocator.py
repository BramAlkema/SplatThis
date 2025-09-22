#!/usr/bin/env python3
"""
Progressive Gaussian allocation with error-guided placement.

This module implements the progressive allocation strategy inspired by Image-GS,
where splats are added incrementally based on reconstruction error.
"""

import logging
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ProgressiveConfig:
    """Configuration for progressive Gaussian allocation.

    This configuration controls how splats are allocated progressively,
    starting with a sparse set and adding more where reconstruction error is highest.
    """

    # Initial allocation parameters
    initial_ratio: float = 0.3              # Start with 30% of target splats
    max_splats: int = 2000                  # Maximum total splats allowed

    # Progressive addition parameters
    add_interval: int = 50                  # Add splats every N iterations
    max_add_per_step: int = 20              # Limit splats added per step

    # Quality control parameters
    error_threshold: float = 0.01           # Minimum error for new placement
    convergence_patience: int = 5           # Steps without improvement to stop

    # Sampling parameters
    temperature: float = 2.0                # Sampling temperature for error distribution

    def __post_init__(self):
        """Validate configuration parameters after initialization."""
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        """Validate all configuration parameters.

        Raises:
            ValueError: If any parameter is invalid
        """
        # Initial allocation validation
        if not 0.1 <= self.initial_ratio <= 0.8:
            raise ValueError(f"initial_ratio must be between 0.1 and 0.8, got {self.initial_ratio}")

        if self.max_splats <= 0:
            raise ValueError(f"max_splats must be positive, got {self.max_splats}")

        # Progressive addition validation
        if self.add_interval <= 0:
            raise ValueError(f"add_interval must be positive, got {self.add_interval}")

        if self.max_add_per_step <= 0:
            raise ValueError(f"max_add_per_step must be positive, got {self.max_add_per_step}")

        # Quality control validation
        if self.error_threshold < 0:
            raise ValueError(f"error_threshold must be non-negative, got {self.error_threshold}")

        if self.convergence_patience <= 0:
            raise ValueError(f"convergence_patience must be positive, got {self.convergence_patience}")

        # Sampling validation
        if self.temperature <= 0:
            raise ValueError(f"temperature must be positive, got {self.temperature}")

    def get_initial_count(self) -> int:
        """Calculate the initial number of splats to allocate.

        Returns:
            Number of splats for initial allocation
        """
        return int(self.max_splats * self.initial_ratio)

    def validate_compatibility(self, image_size: Tuple[int, int]) -> None:
        """Validate configuration compatibility with image size.

        Args:
            image_size: (height, width) of target image

        Raises:
            ValueError: If configuration is incompatible with image size
        """
        height, width = image_size
        total_pixels = height * width

        # Check if max_splats is reasonable for image size
        max_reasonable_splats = total_pixels // 4  # At most 1 splat per 4 pixels
        if self.max_splats > max_reasonable_splats:
            logger.warning(
                f"max_splats ({self.max_splats}) is very high for image size {image_size}. "
                f"Consider reducing to ≤{max_reasonable_splats}"
            )

        # Check if initial allocation is reasonable
        initial_count = self.get_initial_count()
        if initial_count < 10:
            raise ValueError(
                f"Initial allocation ({initial_count}) is too small. "
                f"Increase max_splats or initial_ratio."
            )


class ProgressiveAllocator:
    """Manages progressive Gaussian allocation with error-guided placement.

    This class tracks allocation state, monitors reconstruction error, and decides
    when and how many splats to add during progressive optimization.
    """

    def __init__(self, config: ProgressiveConfig):
        """Initialize the progressive allocator.

        Args:
            config: Configuration for progressive allocation
        """
        self.config = config
        self.iteration_count = 0
        self.error_history: List[float] = []
        self.last_addition_iteration = -1
        self._converged = False

    def should_add_splats(self, current_error: float) -> bool:
        """Determine if new splats should be added.

        Args:
            current_error: Current mean reconstruction error

        Returns:
            True if splats should be added, False otherwise
        """
        # Check if already converged
        if self._converged:
            return False

        # Check if enough iterations have passed since last addition
        # For first addition, use iteration_count directly
        if self.last_addition_iteration == -1:
            iterations_since_addition = self.iteration_count
        else:
            iterations_since_addition = self.iteration_count - self.last_addition_iteration

        if iterations_since_addition < self.config.add_interval:
            return False

        # Check if error is above threshold
        if current_error < self.config.error_threshold:
            logger.debug(f"Error {current_error:.4f} below threshold {self.config.error_threshold}")
            return False

        # Check for convergence (error not improving)
        if self._check_convergence():
            self._converged = True
            logger.info("Progressive allocation converged - error stabilized")
            return False

        return True

    def get_addition_count(self, current_splat_count: int) -> int:
        """Calculate how many splats to add this step.

        Args:
            current_splat_count: Number of splats currently allocated

        Returns:
            Number of splats to add (0 if budget exhausted)
        """
        remaining_budget = self.config.max_splats - current_splat_count
        if remaining_budget <= 0:
            return 0

        return min(self.config.max_add_per_step, remaining_budget)

    def record_iteration(self, error: float, added_splats: int = 0) -> None:
        """Record iteration results for tracking and convergence detection.

        Args:
            error: Mean reconstruction error for this iteration
            added_splats: Number of splats added this iteration
        """
        self.iteration_count += 1
        self.error_history.append(error)

        if added_splats > 0:
            self.last_addition_iteration = self.iteration_count
            logger.debug(f"Iteration {self.iteration_count}: Added {added_splats} splats, error: {error:.4f}")

    def _check_convergence(self) -> bool:
        """Check if the error has converged (stopped improving).

        Returns:
            True if converged, False otherwise
        """
        if len(self.error_history) < self.config.convergence_patience:
            return False

        # Check recent error history for stability
        recent_errors = self.error_history[-self.config.convergence_patience:]
        error_range = max(recent_errors) - min(recent_errors)

        # Use a more reasonable convergence threshold
        # Should be much smaller than the error_threshold
        convergence_threshold = max(self.config.error_threshold * 0.01, 0.001)

        return error_range < convergence_threshold

    def get_stats(self) -> dict:
        """Get allocation statistics for monitoring and debugging.

        Returns:
            Dictionary with allocation statistics
        """
        return {
            'iteration_count': self.iteration_count,
            'error_history_length': len(self.error_history),
            'current_error': self.error_history[-1] if self.error_history else None,
            'last_addition_iteration': self.last_addition_iteration,
            'converged': self._converged,
            'config': self.config
        }

    def reset(self) -> None:
        """Reset allocator state for new allocation session."""
        self.iteration_count = 0
        self.error_history.clear()
        self.last_addition_iteration = -1
        self._converged = False