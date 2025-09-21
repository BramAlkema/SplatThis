"""Adaptive Gaussian 2D with full covariance matrix support."""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any
import logging

# Import current Gaussian for compatibility
from .extract import Gaussian

logger = logging.getLogger(__name__)


@dataclass
class AdaptiveGaussian2D:
    """
    Anisotropic 2D Gaussian with full covariance matrix support.

    Based on Image-GS methodology for content-adaptive Gaussian splatting.
    Uses inverse scales for numerical stability during optimization.
    """

    # Core parameters (optimizable)
    mu: np.ndarray = field(default_factory=lambda: np.array([0.5, 0.5]))  # Position [x, y] in [0,1]²
    inv_s: np.ndarray = field(default_factory=lambda: np.array([0.2, 0.2]))  # Inverse scales [1/sx, 1/sy]
    theta: float = 0.0           # Rotation angle in [0, π)
    color: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0, 1.0]))  # RGB color
    alpha: float = 0.8           # Opacity [0, 1]

    # Optional metadata
    content_complexity: float = 0.0    # Local content measure
    saliency_score: float = 0.0        # Importance weighting
    refinement_count: int = 0          # Number of optimization updates

    def __post_init__(self):
        """Validate and normalize parameters after initialization."""
        self.mu = np.array(self.mu, dtype=np.float32)
        self.inv_s = np.array(self.inv_s, dtype=np.float32)
        self.color = np.array(self.color, dtype=np.float32)
        self.validate_parameters()

    def validate_parameters(self) -> None:
        """Validate all parameters are in valid ranges."""
        if self.mu.shape != (2,):
            raise ValueError(f"Position mu must be 2D, got shape {self.mu.shape}")
        if self.inv_s.shape != (2,):
            raise ValueError(f"Inverse scales inv_s must be 2D, got shape {self.inv_s.shape}")
        if len(self.color) < 3:
            raise ValueError(f"Color must have at least 3 components, got {len(self.color)}")

        # Parameter range validation
        if not (0 <= self.alpha <= 1):
            raise ValueError(f"Alpha must be in [0,1], got {self.alpha}")
        if np.any(self.inv_s <= 0):
            raise ValueError(f"Inverse scales must be positive, got {self.inv_s}")

    def clip_parameters(self) -> None:
        """Clip parameters to valid ranges for numerical stability."""
        # Position in normalized coordinates [0,1]²
        self.mu = np.clip(self.mu, 0.0, 1.0)

        # Inverse scales: prevent degenerate (too small) or extreme (too large) values
        self.inv_s = np.clip(self.inv_s, 1e-3, 1e3)

        # Rotation angle in [0, π)
        self.theta = self.theta % np.pi

        # Color and alpha in [0,1]
        self.color = np.clip(self.color, 0.0, 1.0)
        self.alpha = np.clip(self.alpha, 0.0, 1.0)

        # Metadata bounds
        self.content_complexity = np.clip(self.content_complexity, 0.0, 1.0)
        self.saliency_score = np.clip(self.saliency_score, 0.0, 1.0)
        self.refinement_count = max(0, self.refinement_count)

    @property
    def covariance_matrix(self) -> np.ndarray:
        """
        Compute 2x2 covariance matrix from inverse scales and rotation.

        Uses the formula: Σ = (R S^-1)^-1 (R S^-1)^-T
        where R is rotation matrix and S^-1 is diagonal inverse scale matrix.

        Returns:
            2x2 covariance matrix
        """
        # Rotation matrix
        cos_t, sin_t = np.cos(self.theta), np.sin(self.theta)
        R = np.array([[cos_t, -sin_t],
                      [sin_t, cos_t]], dtype=np.float32)

        # Inverse scale matrix (diagonal)
        S_inv = np.diag(self.inv_s)

        # Build covariance: Σ = (R S^-1)^-1 (R S^-1)^-T
        RS_inv = R @ S_inv

        try:
            # More numerically stable computation
            RS_inv_inv = np.linalg.inv(RS_inv)
            cov = RS_inv_inv @ RS_inv_inv.T
        except np.linalg.LinAlgError:
            logger.warning(f"Singular matrix in covariance computation, using fallback")
            # Fallback to identity with small regularization
            cov = np.eye(2, dtype=np.float32) * 1e-3

        return cov

    @property
    def covariance_inverse(self) -> np.ndarray:
        """
        Compute inverse covariance matrix efficiently.

        More efficient than inverting covariance_matrix for Gaussian evaluation.

        Returns:
            2x2 inverse covariance matrix
        """
        # Direct computation: Σ^-1 = (R S^-1) (R S^-1)^T
        cos_t, sin_t = np.cos(self.theta), np.sin(self.theta)
        R = np.array([[cos_t, -sin_t],
                      [sin_t, cos_t]], dtype=np.float32)

        S_inv = np.diag(self.inv_s)
        RS_inv = R @ S_inv

        return RS_inv @ RS_inv.T

    @property
    def aspect_ratio(self) -> float:
        """
        Compute anisotropy ratio from inverse scales.

        Returns:
            Aspect ratio (always >= 1.0)
        """
        return max(self.inv_s) / min(self.inv_s)

    @property
    def orientation(self) -> float:
        """
        Return primary orientation angle.

        Returns:
            Rotation angle in radians [0, π)
        """
        return self.theta

    @property
    def eigenvalues(self) -> np.ndarray:
        """
        Compute eigenvalues of covariance matrix.

        Returns:
            Array of eigenvalues [lambda1, lambda2] sorted descending
        """
        cov = self.covariance_matrix
        eigenvals = np.linalg.eigvals(cov)
        return np.sort(eigenvals)[::-1]  # Sort descending

    @property
    def principal_axis_length(self) -> float:
        """
        Compute length of principal (major) axis.

        Returns:
            Length of major axis (3σ extent)
        """
        return 3.0 * np.sqrt(max(self.eigenvalues))

    @property
    def minor_axis_length(self) -> float:
        """
        Compute length of minor axis.

        Returns:
            Length of minor axis (3σ extent)
        """
        return 3.0 * np.sqrt(min(self.eigenvalues))

    def evaluate_at(self, point: np.ndarray, normalize_coords: bool = True) -> float:
        """
        Evaluate Gaussian at given point.

        Args:
            point: [x, y] coordinates
            normalize_coords: If True, point is in [0,1]², else in pixels

        Returns:
            Gaussian value at point
        """
        if normalize_coords:
            eval_point = point
            center = self.mu
        else:
            # Convert pixel coordinates to normalized
            eval_point = point  # Assume already converted by caller
            center = self.mu

        delta = eval_point - center
        cov_inv = self.covariance_inverse

        quadratic_form = delta.T @ cov_inv @ delta
        return np.exp(-0.5 * quadratic_form)

    def compute_3sigma_radius_px(self, image_size: Tuple[int, int]) -> float:
        """
        Compute 3σ radius in pixels for spatial binning.

        Args:
            image_size: (H, W) image dimensions

        Returns:
            Radius in pixels
        """
        H, W = image_size
        max_eigenval = max(self.eigenvalues)
        return 3.0 * np.sqrt(max_eigenval) * min(H, W)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize to dictionary for storage/transmission.

        Returns:
            Dictionary representation
        """
        return {
            'mu': self.mu.tolist(),
            'inv_s': self.inv_s.tolist(),
            'theta': float(self.theta),
            'color': self.color.tolist(),
            'alpha': float(self.alpha),
            'content_complexity': float(self.content_complexity),
            'saliency_score': float(self.saliency_score),
            'refinement_count': int(self.refinement_count),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AdaptiveGaussian2D':
        """
        Deserialize from dictionary.

        Args:
            data: Dictionary representation

        Returns:
            AdaptiveGaussian2D instance
        """
        return cls(
            mu=np.array(data['mu'], dtype=np.float32),
            inv_s=np.array(data['inv_s'], dtype=np.float32),
            theta=data['theta'],
            color=np.array(data['color'], dtype=np.float32),
            alpha=data['alpha'],
            content_complexity=data.get('content_complexity', 0.0),
            saliency_score=data.get('saliency_score', 0.0),
            refinement_count=data.get('refinement_count', 0),
        )

    @classmethod
    def from_gaussian(cls, gaussian: Gaussian, image_size: Optional[Tuple[int, int]] = None) -> 'AdaptiveGaussian2D':
        """
        Convert from current Gaussian class for backward compatibility.

        Args:
            gaussian: Current Gaussian instance
            image_size: Optional (H, W) for coordinate normalization

        Returns:
            AdaptiveGaussian2D equivalent
        """
        if image_size is not None:
            H, W = image_size
            # Normalize position coordinates
            mu = np.array([gaussian.x / W, gaussian.y / H], dtype=np.float32)

            # Convert radii to inverse scales (normalized)
            inv_sx = W / (gaussian.rx * W) if gaussian.rx > 0 else 0.2
            inv_sy = H / (gaussian.ry * H) if gaussian.ry > 0 else 0.2
            inv_s = np.array([inv_sx, inv_sy], dtype=np.float32)
        else:
            # Use raw coordinates (assume already normalized)
            mu = np.array([gaussian.x, gaussian.y], dtype=np.float32)
            inv_s = np.array([1.0/gaussian.rx, 1.0/gaussian.ry], dtype=np.float32)

        return cls(
            mu=mu,
            inv_s=inv_s,
            theta=gaussian.theta,
            color=np.array([gaussian.r/255.0, gaussian.g/255.0, gaussian.b/255.0], dtype=np.float32),
            alpha=gaussian.a,
            content_complexity=getattr(gaussian, 'score', 0.0),
        )

    def to_gaussian(self, image_size: Tuple[int, int]) -> Gaussian:
        """
        Convert to current Gaussian class for backward compatibility.

        Args:
            image_size: (H, W) for coordinate denormalization

        Returns:
            Gaussian instance
        """
        H, W = image_size

        # Denormalize coordinates
        x = self.mu[0] * W
        y = self.mu[1] * H

        # Convert inverse scales to radii
        rx = 1.0 / (self.inv_s[0] * W) if self.inv_s[0] > 0 else 1.0
        ry = 1.0 / (self.inv_s[1] * H) if self.inv_s[1] > 0 else 1.0

        # Convert color to 0-255 range
        r = int(np.clip(self.color[0] * 255, 0, 255))
        g = int(np.clip(self.color[1] * 255, 0, 255))
        b = int(np.clip(self.color[2] * 255, 0, 255))

        return Gaussian(
            x=x, y=y, rx=rx, ry=ry, theta=self.theta,
            r=r, g=g, b=b, a=self.alpha,
            score=self.content_complexity
        )

    def copy(self) -> 'AdaptiveGaussian2D':
        """Create a deep copy of this Gaussian."""
        return AdaptiveGaussian2D(
            mu=self.mu.copy(),
            inv_s=self.inv_s.copy(),
            theta=self.theta,
            color=self.color.copy(),
            alpha=self.alpha,
            content_complexity=self.content_complexity,
            saliency_score=self.saliency_score,
            refinement_count=self.refinement_count,
        )

    def __str__(self) -> str:
        """String representation for debugging."""
        return (f"AdaptiveGaussian2D(mu={self.mu}, inv_s={self.inv_s}, "
                f"theta={self.theta:.3f}, aspect={self.aspect_ratio:.2f}, "
                f"alpha={self.alpha:.3f})")

    def __repr__(self) -> str:
        """Detailed representation for debugging."""
        return self.__str__()


def create_isotropic_gaussian(center: np.ndarray, scale: float, color: np.ndarray, alpha: float = 0.8) -> AdaptiveGaussian2D:
    """
    Helper function to create isotropic Gaussian splat.

    Args:
        center: [x, y] position in [0,1]²
        scale: Scale factor (will be converted to inverse scale)
        color: RGB color in [0,1]
        alpha: Opacity

    Returns:
        Isotropic AdaptiveGaussian2D
    """
    inv_scale = 1.0 / scale if scale > 0 else 0.2
    return AdaptiveGaussian2D(
        mu=np.array(center, dtype=np.float32),
        inv_s=np.array([inv_scale, inv_scale], dtype=np.float32),
        theta=0.0,
        color=np.array(color, dtype=np.float32),
        alpha=alpha
    )


def create_anisotropic_gaussian(center: np.ndarray, scales: Tuple[float, float],
                              orientation: float, color: np.ndarray, alpha: float = 0.8) -> AdaptiveGaussian2D:
    """
    Helper function to create anisotropic Gaussian splat.

    Args:
        center: [x, y] position in [0,1]²
        scales: (scale_x, scale_y) scale factors
        orientation: Rotation angle in radians
        color: RGB color in [0,1]
        alpha: Opacity

    Returns:
        Anisotropic AdaptiveGaussian2D
    """
    scale_x, scale_y = scales
    inv_scale_x = 1.0 / scale_x if scale_x > 0 else 0.2
    inv_scale_y = 1.0 / scale_y if scale_y > 0 else 0.2

    return AdaptiveGaussian2D(
        mu=np.array(center, dtype=np.float32),
        inv_s=np.array([inv_scale_x, inv_scale_y], dtype=np.float32),
        theta=orientation % np.pi,
        color=np.array(color, dtype=np.float32),
        alpha=alpha
    )