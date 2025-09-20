"""Mathematical utilities for splat processing."""

import numpy as np
from typing import Tuple


def safe_eigendecomposition(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Safely compute eigenvalues/vectors with fallback for edge cases."""
    try:
        eigenvals, eigenvecs = np.linalg.eigh(matrix)
        return eigenvals, eigenvecs
    except np.linalg.LinAlgError:
        # Fallback to identity for degenerate cases
        return np.array([1.0, 1.0]), np.eye(2)


def clamp_value(value: float, min_val: float, max_val: float) -> float:
    """Clamp value to range [min_val, max_val]."""
    return max(min_val, min(max_val, value))


def normalize_angle(angle: float) -> float:
    """Normalize angle to [-π, π] range."""
    return ((angle + np.pi) % (2 * np.pi)) - np.pi
