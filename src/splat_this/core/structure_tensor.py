#!/usr/bin/env python3
"""Structure tensor analysis for local orientation and anisotropy detection."""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
import logging
from scipy import ndimage
from scipy.linalg import eigh

logger = logging.getLogger(__name__)


@dataclass
class StructureTensorResult:
    """Results from structure tensor analysis."""
    orientation: np.ndarray  # Dominant orientation angle at each point (radians)
    anisotropy: np.ndarray   # Anisotropy ratio (0=isotropic, 1=highly anisotropic)
    coherence: np.ndarray    # Edge coherence measure (0=no edge, 1=strong edge)
    eigenvalues: np.ndarray  # Raw eigenvalues (height, width, 2)
    eigenvectors: np.ndarray # Raw eigenvectors (height, width, 2, 2)
    tensor_trace: np.ndarray # Trace of structure tensor (overall gradient magnitude)


@dataclass
class StructureTensorConfig:
    """Configuration for structure tensor computation."""
    gradient_sigma: float = 1.0      # Sigma for gradient computation
    integration_sigma: float = 2.0   # Sigma for tensor integration
    anisotropy_threshold: float = 0.1 # Minimum anisotropy for orientation reliability
    coherence_threshold: float = 0.2  # Minimum coherence for edge detection
    edge_enhancement: bool = True     # Whether to enhance edge responses
    normalization: str = 'trace'     # 'trace', 'determinant', or 'none'

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.gradient_sigma <= 0:
            raise ValueError("gradient_sigma must be positive")
        if self.integration_sigma <= 0:
            raise ValueError("integration_sigma must be positive")
        if not 0 <= self.anisotropy_threshold <= 1:
            raise ValueError("anisotropy_threshold must be in [0, 1]")
        if not 0 <= self.coherence_threshold <= 1:
            raise ValueError("coherence_threshold must be in [0, 1]")


class StructureTensorAnalyzer:
    """Compute and analyze structure tensors for local image analysis."""

    def __init__(self, config: Optional[StructureTensorConfig] = None):
        """Initialize structure tensor analyzer.

        Args:
            config: Configuration parameters, defaults to StructureTensorConfig()
        """
        self.config = config or StructureTensorConfig()

    def compute_structure_tensor(self, image: np.ndarray) -> StructureTensorResult:
        """Compute structure tensor for the entire image.

        Args:
            image: Input image (H, W, C) or (H, W) in range [0, 1]

        Returns:
            StructureTensorResult with all tensor analysis results
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image.copy()

        # Ensure float64 for numerical stability
        gray = gray.astype(np.float64)

        # Compute gradients with Gaussian derivatives
        gx = ndimage.gaussian_filter(gray, sigma=self.config.gradient_sigma, order=[0, 1])
        gy = ndimage.gaussian_filter(gray, sigma=self.config.gradient_sigma, order=[1, 0])

        # Compute structure tensor components
        gxx = gx * gx
        gyy = gy * gy
        gxy = gx * gy

        # Apply Gaussian integration to smooth tensor components
        sigma = self.config.integration_sigma
        Jxx = ndimage.gaussian_filter(gxx, sigma=sigma)
        Jyy = ndimage.gaussian_filter(gyy, sigma=sigma)
        Jxy = ndimage.gaussian_filter(gxy, sigma=sigma)

        # Edge enhancement if enabled
        if self.config.edge_enhancement:
            # Enhance diagonal terms relative to off-diagonal
            enhancement_factor = 1.1
            Jxx *= enhancement_factor
            Jyy *= enhancement_factor

        # Compute eigenvalues and eigenvectors at each pixel
        h, w = gray.shape
        eigenvalues = np.zeros((h, w, 2))
        eigenvectors = np.zeros((h, w, 2, 2))

        for i in range(h):
            for j in range(w):
                # Structure tensor matrix at this pixel
                J = np.array([[Jxx[i, j], Jxy[i, j]],
                             [Jxy[i, j], Jyy[i, j]]])

                # Compute eigenvalues and eigenvectors
                # eigh returns eigenvalues in ascending order
                eigvals, eigvecs = eigh(J)

                eigenvalues[i, j] = eigvals
                eigenvectors[i, j] = eigvecs

        # Extract larger and smaller eigenvalues
        lambda1 = eigenvalues[:, :, 1]  # Larger eigenvalue
        lambda2 = eigenvalues[:, :, 0]  # Smaller eigenvalue

        # Compute derived measures
        orientation = self._compute_orientation(eigenvectors)
        anisotropy = self._compute_anisotropy(lambda1, lambda2)
        coherence = self._compute_coherence(lambda1, lambda2)
        tensor_trace = Jxx + Jyy

        # Apply normalization if requested
        if self.config.normalization == 'trace':
            norm_factor = tensor_trace + 1e-10
            anisotropy = anisotropy * np.sqrt(norm_factor / np.max(norm_factor))
        elif self.config.normalization == 'determinant':
            determinant = lambda1 * lambda2
            norm_factor = np.sqrt(determinant + 1e-10)
            anisotropy = anisotropy * (norm_factor / np.max(norm_factor))

        return StructureTensorResult(
            orientation=orientation,
            anisotropy=anisotropy,
            coherence=coherence,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            tensor_trace=tensor_trace
        )

    def _compute_orientation(self, eigenvectors: np.ndarray) -> np.ndarray:
        """Compute dominant orientation from eigenvectors.

        Args:
            eigenvectors: Eigenvectors array (H, W, 2, 2)

        Returns:
            Orientation angles in radians (H, W)
        """
        # Dominant eigenvector (corresponding to larger eigenvalue)
        dominant_vec = eigenvectors[:, :, :, 1]  # Second column (larger eigenvalue)

        # Compute angle of dominant eigenvector
        orientation = np.arctan2(dominant_vec[:, :, 1], dominant_vec[:, :, 0])

        # Ensure orientation is in [0, π) for consistency
        orientation = np.mod(orientation, np.pi)

        return orientation

    def _compute_anisotropy(self, lambda1: np.ndarray, lambda2: np.ndarray) -> np.ndarray:
        """Compute anisotropy measure from eigenvalues.

        Args:
            lambda1: Larger eigenvalues (H, W)
            lambda2: Smaller eigenvalues (H, W)

        Returns:
            Anisotropy ratios in [0, 1] (H, W)
        """
        # Avoid division by zero
        lambda_sum = lambda1 + lambda2
        epsilon = 1e-10

        # Compute anisotropy as (λ1 - λ2) / (λ1 + λ2)
        anisotropy = np.where(
            lambda_sum > epsilon,
            (lambda1 - lambda2) / (lambda_sum + epsilon),
            0.0
        )

        return np.clip(anisotropy, 0.0, 1.0)

    def _compute_coherence(self, lambda1: np.ndarray, lambda2: np.ndarray) -> np.ndarray:
        """Compute edge coherence measure.

        Args:
            lambda1: Larger eigenvalues (H, W)
            lambda2: Smaller eigenvalues (H, W)

        Returns:
            Coherence values in [0, 1] (H, W)
        """
        # Coherence as square of anisotropy weighted by total energy
        anisotropy = self._compute_anisotropy(lambda1, lambda2)
        total_energy = lambda1 + lambda2

        # Normalize energy to [0, 1]
        max_energy = np.max(total_energy)
        if max_energy > 0:
            normalized_energy = total_energy / max_energy
        else:
            normalized_energy = np.zeros_like(total_energy)

        # Coherence combines anisotropy and energy
        coherence = anisotropy * anisotropy * normalized_energy

        return coherence

    def analyze_local_structure(self, image: np.ndarray,
                               points: np.ndarray) -> Dict[str, np.ndarray]:
        """Analyze local structure at specific points.

        Args:
            image: Input image (H, W, C) or (H, W)
            points: Points to analyze, shape (N, 2) as (y, x) coordinates

        Returns:
            Dictionary with orientation, anisotropy, coherence at each point
        """
        # Compute full structure tensor
        result = self.compute_structure_tensor(image)

        # Extract values at specified points
        points = np.round(points).astype(int)
        h, w = image.shape[:2]

        # Clip points to image bounds
        points[:, 0] = np.clip(points[:, 0], 0, h - 1)
        points[:, 1] = np.clip(points[:, 1], 0, w - 1)

        orientations = result.orientation[points[:, 0], points[:, 1]]
        anisotropies = result.anisotropy[points[:, 0], points[:, 1]]
        coherences = result.coherence[points[:, 0], points[:, 1]]

        return {
            'orientations': orientations,
            'anisotropies': anisotropies,
            'coherences': coherences,
            'points': points
        }

    def detect_edge_following_locations(self, image: np.ndarray,
                                       min_coherence: Optional[float] = None,
                                       min_anisotropy: Optional[float] = None) -> np.ndarray:
        """Detect locations suitable for edge-following splats.

        Args:
            image: Input image (H, W, C) or (H, W)
            min_coherence: Minimum coherence threshold
            min_anisotropy: Minimum anisotropy threshold

        Returns:
            Array of (y, x) coordinates for edge-following splat locations
        """
        if min_coherence is None:
            min_coherence = self.config.coherence_threshold
        if min_anisotropy is None:
            min_anisotropy = self.config.anisotropy_threshold

        # Compute structure tensor
        result = self.compute_structure_tensor(image)

        # Find locations with sufficient edge strength
        edge_mask = (result.coherence >= min_coherence) & (result.anisotropy >= min_anisotropy)

        # Extract coordinates
        y_coords, x_coords = np.where(edge_mask)
        locations = np.column_stack((y_coords, x_coords))

        logger.info(f"Detected {len(locations)} edge-following locations "
                   f"(coherence >= {min_coherence}, anisotropy >= {min_anisotropy})")

        return locations

    def create_orientation_field_visualization(self, result: StructureTensorResult,
                                             stride: int = 8) -> Dict[str, Any]:
        """Create visualization data for orientation field.

        Args:
            result: Structure tensor analysis result
            stride: Sampling stride for visualization vectors

        Returns:
            Dictionary with visualization data
        """
        h, w = result.orientation.shape

        # Sample points for visualization
        y_coords = np.arange(stride//2, h, stride)
        x_coords = np.arange(stride//2, w, stride)
        yy, xx = np.meshgrid(y_coords, x_coords, indexing='ij')

        # Extract orientations and strengths at sample points
        orientations = result.orientation[yy, xx]
        anisotropies = result.anisotropy[yy, xx]
        coherences = result.coherence[yy, xx]

        # Compute vector components
        cos_theta = np.cos(orientations)
        sin_theta = np.sin(orientations)

        # Scale vectors by anisotropy (stronger edges = longer vectors)
        scale = anisotropies * stride * 0.4
        dx = cos_theta * scale
        dy = sin_theta * scale

        return {
            'x_positions': xx,
            'y_positions': yy,
            'dx': dx,
            'dy': dy,
            'orientations': orientations,
            'anisotropies': anisotropies,
            'coherences': coherences,
            'stride': stride
        }

    def validate_orientation_accuracy(self, image: np.ndarray,
                                    test_points: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Validate orientation accuracy against known edge directions.

        Args:
            image: Input image (H, W, C) or (H, W)
            test_points: Optional specific points to test, shape (N, 2)

        Returns:
            Dictionary with validation metrics
        """
        result = self.compute_structure_tensor(image)

        if test_points is None:
            # Sample points uniformly across the image
            h, w = image.shape[:2]
            num_samples = min(1000, h * w // 16)
            y_samples = np.random.randint(0, h, num_samples)
            x_samples = np.random.randint(0, w, num_samples)
            test_points = np.column_stack((y_samples, x_samples))

        # Analyze structure at test points
        analysis = self.analyze_local_structure(image, test_points)

        # Compute validation metrics
        high_coherence_mask = analysis['coherences'] >= self.config.coherence_threshold
        high_anisotropy_mask = analysis['anisotropies'] >= self.config.anisotropy_threshold

        reliable_points = high_coherence_mask & high_anisotropy_mask

        return {
            'total_test_points': len(test_points),
            'high_coherence_fraction': np.mean(high_coherence_mask),
            'high_anisotropy_fraction': np.mean(high_anisotropy_mask),
            'reliable_orientations_fraction': np.mean(reliable_points),
            'mean_coherence': np.mean(analysis['coherences']),
            'mean_anisotropy': np.mean(analysis['anisotropies']),
            'coherence_std': np.std(analysis['coherences']),
            'anisotropy_std': np.std(analysis['anisotropies'])
        }


def compute_structure_tensor(image: np.ndarray,
                           gradient_sigma: float = 1.0,
                           integration_sigma: float = 2.0) -> StructureTensorResult:
    """Convenience function to compute structure tensor with default settings.

    Args:
        image: Input image (H, W, C) or (H, W)
        gradient_sigma: Sigma for gradient computation
        integration_sigma: Sigma for tensor integration

    Returns:
        StructureTensorResult with tensor analysis
    """
    config = StructureTensorConfig(
        gradient_sigma=gradient_sigma,
        integration_sigma=integration_sigma
    )
    analyzer = StructureTensorAnalyzer(config)
    return analyzer.compute_structure_tensor(image)


def analyze_local_orientations(image: np.ndarray, points: np.ndarray) -> Dict[str, np.ndarray]:
    """Convenience function to analyze orientations at specific points.

    Args:
        image: Input image (H, W, C) or (H, W)
        points: Points to analyze, shape (N, 2) as (y, x) coordinates

    Returns:
        Dictionary with orientation analysis at each point
    """
    analyzer = StructureTensorAnalyzer()
    return analyzer.analyze_local_structure(image, points)