#!/usr/bin/env python3
"""
Advanced Feature Detection and Splat Placement

Implements sophisticated algorithms for content-aware splat initialization,
including edge detection, structure tensor analysis, and adaptive sizing.
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass

from .splat import GaussianSplat, create_isotropic_splat


@dataclass
class EdgeDetectionConfig:
    """Configuration for edge detection parameters."""
    sobel_threshold: float = 0.1
    canny_low: float = 0.05
    canny_high: float = 0.15
    harris_k: float = 0.04
    harris_threshold: float = 0.01
    non_max_suppression: bool = True
    gaussian_sigma: float = 1.0


@dataclass
class StructureTensorConfig:
    """Configuration for structure tensor analysis."""
    window_size: int = 5
    sigma_inner: float = 1.0
    sigma_outer: float = 2.0
    coherence_threshold: float = 0.5
    anisotropy_threshold: float = 0.3


class EdgeDetector:
    """
    Advanced edge detection using multiple methods for robust boundary detection.
    Combines Sobel, Canny, and Harris corner detection for comprehensive analysis.
    """

    def __init__(self, config: EdgeDetectionConfig = None):
        self.config = config or EdgeDetectionConfig()

    def sobel_edges(self, image: np.ndarray) -> np.ndarray:
        """Compute Sobel edge magnitude."""
        if len(image.shape) == 3:
            gray = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
        else:
            gray = image

        # Sobel operators
        sobel_x = ndimage.sobel(gray, axis=1)
        sobel_y = ndimage.sobel(gray, axis=0)

        magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

        # Normalize
        if magnitude.max() > 0:
            magnitude = magnitude / magnitude.max()

        return magnitude

    def canny_edges(self, image: np.ndarray) -> np.ndarray:
        """Implement Canny edge detection."""
        if len(image.shape) == 3:
            gray = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
        else:
            gray = image

        # Gaussian smoothing
        smoothed = gaussian_filter(gray, self.config.gaussian_sigma)

        # Gradients
        grad_x = ndimage.sobel(smoothed, axis=1)
        grad_y = ndimage.sobel(smoothed, axis=0)

        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        direction = np.arctan2(grad_y, grad_x)

        # Non-maximum suppression
        if self.config.non_max_suppression:
            magnitude = self._non_max_suppression(magnitude, direction)

        # Double thresholding
        high_threshold = self.config.canny_high * magnitude.max()
        low_threshold = self.config.canny_low * magnitude.max()

        strong_edges = magnitude > high_threshold
        weak_edges = (magnitude >= low_threshold) & (magnitude <= high_threshold)

        # Edge tracking by hysteresis
        edges = self._hysteresis_tracking(strong_edges, weak_edges)

        return edges.astype(np.float32)

    def _non_max_suppression(self, magnitude: np.ndarray, direction: np.ndarray) -> np.ndarray:
        """Apply non-maximum suppression to thin edges."""
        height, width = magnitude.shape
        suppressed = np.zeros_like(magnitude)

        # Convert angles to 0-180 degrees
        angle = direction * 180.0 / np.pi
        angle[angle < 0] += 180

        for i in range(1, height - 1):
            for j in range(1, width - 1):
                # Get neighboring pixels based on gradient direction
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    neighbors = [magnitude[i, j-1], magnitude[i, j+1]]
                elif 22.5 <= angle[i, j] < 67.5:
                    neighbors = [magnitude[i-1, j-1], magnitude[i+1, j+1]]
                elif 67.5 <= angle[i, j] < 112.5:
                    neighbors = [magnitude[i-1, j], magnitude[i+1, j]]
                else:  # 112.5 <= angle[i, j] < 157.5
                    neighbors = [magnitude[i-1, j+1], magnitude[i+1, j-1]]

                # Suppress if not local maximum
                if magnitude[i, j] >= max(neighbors):
                    suppressed[i, j] = magnitude[i, j]

        return suppressed

    def _hysteresis_tracking(self, strong: np.ndarray, weak: np.ndarray) -> np.ndarray:
        """Track edges using hysteresis."""
        edges = strong.copy()

        # Find weak edges connected to strong edges
        changed = True
        while changed:
            changed = False
            for i in range(1, weak.shape[0] - 1):
                for j in range(1, weak.shape[1] - 1):
                    if weak[i, j] and not edges[i, j]:
                        # Check if connected to strong edge
                        if np.any(edges[i-1:i+2, j-1:j+2]):
                            edges[i, j] = True
                            changed = True

        return edges

    def harris_corners(self, image: np.ndarray) -> np.ndarray:
        """Detect Harris corners."""
        if len(image.shape) == 3:
            gray = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
        else:
            gray = image

        # Compute gradients
        grad_x = ndimage.sobel(gray, axis=1)
        grad_y = ndimage.sobel(gray, axis=0)

        # Compute products of gradients
        grad_xx = grad_x * grad_x
        grad_xy = grad_x * grad_y
        grad_yy = grad_y * grad_y

        # Apply Gaussian filtering
        sigma = self.config.gaussian_sigma
        grad_xx = gaussian_filter(grad_xx, sigma)
        grad_xy = gaussian_filter(grad_xy, sigma)
        grad_yy = gaussian_filter(grad_yy, sigma)

        # Harris response
        k = self.config.harris_k
        det = grad_xx * grad_yy - grad_xy**2
        trace = grad_xx + grad_yy

        harris_response = det - k * trace**2

        # Threshold
        threshold = self.config.harris_threshold * harris_response.max()
        corners = harris_response > threshold

        return corners.astype(np.float32)

    def detect_edges_comprehensive(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Run comprehensive edge detection using multiple methods."""
        return {
            'sobel': self.sobel_edges(image),
            'canny': self.canny_edges(image),
            'harris': self.harris_corners(image)
        }


class StructureTensorAnalyzer:
    """
    Structure tensor analysis for understanding local image structure.
    Detects orientation, coherence, and anisotropy for informed splat placement.
    """

    def __init__(self, config: StructureTensorConfig = None):
        self.config = config or StructureTensorConfig()

    def compute_structure_tensor(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute structure tensor and derived properties."""
        if len(image.shape) == 3:
            gray = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
        else:
            gray = image

        # Compute gradients with inner Gaussian
        grad_x = ndimage.sobel(
            gaussian_filter(gray, self.config.sigma_inner), axis=1
        )
        grad_y = ndimage.sobel(
            gaussian_filter(gray, self.config.sigma_inner), axis=0
        )

        # Structure tensor components
        j11 = grad_x * grad_x
        j12 = grad_x * grad_y
        j22 = grad_y * grad_y

        # Apply outer Gaussian
        j11 = gaussian_filter(j11, self.config.sigma_outer)
        j12 = gaussian_filter(j12, self.config.sigma_outer)
        j22 = gaussian_filter(j22, self.config.sigma_outer)

        # Compute eigenvalues and eigenvectors
        height, width = gray.shape
        coherence = np.zeros((height, width))
        orientation = np.zeros((height, width))
        anisotropy = np.zeros((height, width))

        for i in range(height):
            for j in range(width):
                # Structure tensor at this pixel
                J = np.array([[j11[i, j], j12[i, j]],
                             [j12[i, j], j22[i, j]]])

                # Eigenvalues
                eigenvals = np.linalg.eigvals(J)
                lambda1, lambda2 = sorted(eigenvals, reverse=True)

                # Coherence (normalized difference)
                if lambda1 + lambda2 > 1e-8:
                    coherence[i, j] = (lambda1 - lambda2) / (lambda1 + lambda2)

                # Orientation (direction of principal eigenvector)
                if lambda1 > 1e-8:
                    eigenvecs = np.linalg.eig(J)[1]
                    principal_vec = eigenvecs[:, 0]
                    orientation[i, j] = np.arctan2(principal_vec[1], principal_vec[0])

                # Anisotropy (ratio of eigenvalues)
                if lambda2 > 1e-8:
                    anisotropy[i, j] = lambda1 / lambda2
                else:
                    anisotropy[i, j] = 1.0

        return {
            'coherence': coherence,
            'orientation': orientation,
            'anisotropy': anisotropy,
            'j11': j11,
            'j12': j12,
            'j22': j22
        }


class ContentAdaptiveSplatPlacer:
    """
    Advanced splat placement using edge detection and structure tensor analysis.
    Places splats intelligently based on image content and local structure.
    """

    def __init__(self,
                 edge_config: EdgeDetectionConfig = None,
                 structure_config: StructureTensorConfig = None):

        self.edge_detector = EdgeDetector(edge_config)
        self.structure_analyzer = StructureTensorAnalyzer(structure_config)

    def analyze_image_structure(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Comprehensive image structure analysis."""
        # Edge detection
        edges = self.edge_detector.detect_edges_comprehensive(image)

        # Structure tensor analysis
        structure = self.structure_analyzer.compute_structure_tensor(image)

        # Combine edge information
        edge_strength = np.maximum.reduce([
            edges['sobel'],
            edges['canny'].astype(np.float32),
            edges['harris']
        ])

        return {
            'edge_strength': edge_strength,
            'coherence': structure['coherence'],
            'orientation': structure['orientation'],
            'anisotropy': structure['anisotropy']
        }

    def generate_splat_candidates(self,
                                 image: np.ndarray,
                                 target_count: int,
                                 strategy: str = 'adaptive') -> List[Dict[str, Any]]:
        """Generate splat placement candidates based on content analysis."""

        analysis = self.analyze_image_structure(image)
        height, width = image.shape[:2]

        candidates = []

        if strategy == 'adaptive':
            # Edge-based placement (40% of splats)
            edge_count = int(target_count * 0.4)
            edge_candidates = self._place_on_edges(
                analysis['edge_strength'], edge_count, image
            )
            candidates.extend(edge_candidates)

            # Structure-guided placement (30% of splats)
            structure_count = int(target_count * 0.3)
            structure_candidates = self._place_on_structure(
                analysis, structure_count, image
            )
            candidates.extend(structure_candidates)

            # Grid-based filling (30% of splats)
            grid_count = target_count - len(candidates)
            grid_candidates = self._place_on_grid(grid_count, image)
            candidates.extend(grid_candidates)

        elif strategy == 'edge_focused':
            # Focus heavily on edges
            candidates = self._place_on_edges(
                analysis['edge_strength'], target_count, image
            )

        elif strategy == 'structure_guided':
            # Use structure tensor for all placement
            candidates = self._place_on_structure(
                analysis, target_count, image
            )

        else:  # 'uniform'
            candidates = self._place_on_grid(target_count, image)

        return candidates[:target_count]

    def _place_on_edges(self,
                       edge_strength: np.ndarray,
                       count: int,
                       image: np.ndarray) -> List[Dict[str, Any]]:
        """Place splats on detected edges."""
        candidates = []

        # Find strong edge pixels
        threshold = np.percentile(edge_strength, 85)
        edge_pixels = np.where(edge_strength > threshold)

        if len(edge_pixels[0]) > 0:
            # Sample from edge pixels
            indices = np.random.choice(
                len(edge_pixels[0]), min(count, len(edge_pixels[0])), replace=False
            )

            for idx in indices:
                y, x = edge_pixels[0][idx], edge_pixels[1][idx]

                # Local color
                color = self._sample_local_color(image, x, y)

                # Smaller splats on edges for precision
                base_sigma = min(image.shape[:2]) / 64
                sigma = base_sigma * (0.5 + 0.5 * edge_strength[y, x])

                candidates.append({
                    'position': np.array([float(x), float(y)]),
                    'sigma': sigma,
                    'color': color,
                    'alpha': 0.7 + 0.3 * edge_strength[y, x],
                    'type': 'edge',
                    'importance': edge_strength[y, x]
                })

        return candidates

    def _place_on_structure(self,
                           analysis: Dict[str, np.ndarray],
                           count: int,
                           image: np.ndarray) -> List[Dict[str, Any]]:
        """Place splats based on structure tensor analysis."""
        candidates = []

        coherence = analysis['coherence']
        orientation = analysis['orientation']
        anisotropy = analysis['anisotropy']

        # Find structured regions
        structure_mask = (coherence > 0.3) & (anisotropy > 1.5)
        structure_pixels = np.where(structure_mask)

        if len(structure_pixels[0]) > 0:
            # Sample structured pixels
            indices = np.random.choice(
                len(structure_pixels[0]), min(count, len(structure_pixels[0])), replace=False
            )

            for idx in indices:
                y, x = structure_pixels[0][idx], structure_pixels[1][idx]

                # Local properties
                color = self._sample_local_color(image, x, y)
                local_coherence = coherence[y, x]
                local_orientation = orientation[y, x]
                local_anisotropy = anisotropy[y, x]

                # Adaptive sizing based on structure
                base_sigma = min(image.shape[:2]) / 48
                sigma = base_sigma * (1.0 + local_coherence)

                candidates.append({
                    'position': np.array([float(x), float(y)]),
                    'sigma': sigma,
                    'color': color,
                    'alpha': 0.6 + 0.4 * local_coherence,
                    'orientation': local_orientation,
                    'anisotropy': min(local_anisotropy, 5.0),  # Cap anisotropy
                    'type': 'structure',
                    'importance': local_coherence
                })

        return candidates

    def _place_on_grid(self, count: int, image: np.ndarray) -> List[Dict[str, Any]]:
        """Place splats on regular grid with jitter."""
        candidates = []
        height, width = image.shape[:2]

        grid_size = int(np.sqrt(count))

        for i in range(grid_size):
            for j in range(grid_size):
                if len(candidates) >= count:
                    break

                # Grid position with jitter
                x = (j + 0.5 + np.random.uniform(-0.3, 0.3)) * width / grid_size
                y = (i + 0.5 + np.random.uniform(-0.3, 0.3)) * height / grid_size

                x = np.clip(x, 0, width - 1)
                y = np.clip(y, 0, height - 1)

                color = self._sample_local_color(image, int(x), int(y))
                sigma = min(width, height) / (grid_size * 2)

                candidates.append({
                    'position': np.array([float(x), float(y)]),
                    'sigma': sigma,
                    'color': color,
                    'alpha': 0.8,
                    'type': 'grid',
                    'importance': 0.5
                })

        return candidates

    def _sample_local_color(self, image: np.ndarray, x: int, y: int, window_size: int = 3) -> np.ndarray:
        """Sample local color with window averaging."""
        height, width = image.shape[:2]

        x1 = max(0, x - window_size//2)
        x2 = min(width, x + window_size//2 + 1)
        y1 = max(0, y - window_size//2)
        y2 = min(height, y + window_size//2 + 1)

        patch = image[y1:y2, x1:x2, :3]
        color = np.mean(patch.reshape(-1, 3), axis=0)

        return color.astype(np.float32)

    def create_splats_from_candidates(self, candidates: List[Dict[str, Any]]) -> List[GaussianSplat]:
        """Convert placement candidates to GaussianSplat objects."""
        splats = []

        for candidate in candidates:
            if candidate.get('anisotropy', 1.0) > 1.2 and 'orientation' in candidate:
                # Create anisotropic splat
                sigma = candidate['sigma']
                anisotropy = candidate['anisotropy']
                orientation = candidate['orientation']

                # Create covariance matrix
                # Major axis length
                major_sigma = sigma * np.sqrt(anisotropy)
                minor_sigma = sigma / np.sqrt(anisotropy)

                # Rotation matrix
                cos_theta = np.cos(orientation)
                sin_theta = np.sin(orientation)

                # Diagonal covariance in local coordinates
                local_cov = np.array([[major_sigma**2, 0],
                                    [0, minor_sigma**2]])

                # Rotation to global coordinates
                rotation = np.array([[cos_theta, -sin_theta],
                                   [sin_theta, cos_theta]])

                global_cov = rotation @ local_cov @ rotation.T

                splat = GaussianSplat(
                    mu=candidate['position'],
                    sigma=global_cov,
                    color=candidate['color'],
                    alpha=candidate['alpha'],
                    importance=candidate.get('importance', 1.0)
                )
            else:
                # Create isotropic splat
                splat = create_isotropic_splat(
                    center=candidate['position'],
                    sigma=candidate['sigma'],
                    color=candidate['color'],
                    alpha=candidate['alpha']
                )
                splat.importance = candidate.get('importance', 1.0)

            splats.append(splat)

        return splats


if __name__ == "__main__":
    # Test the advanced features
    print("Testing Advanced Feature Detection...")

    # Create test image
    test_image = np.random.rand(64, 64, 3)

    # Test edge detection
    edge_detector = EdgeDetector()
    edges = edge_detector.detect_edges_comprehensive(test_image)
    print(f"Detected edges - Sobel max: {edges['sobel'].max():.3f}")

    # Test structure analysis
    structure_analyzer = StructureTensorAnalyzer()
    structure = structure_analyzer.compute_structure_tensor(test_image)
    print(f"Structure coherence max: {structure['coherence'].max():.3f}")

    # Test splat placement
    placer = ContentAdaptiveSplatPlacer()
    candidates = placer.generate_splat_candidates(test_image, 50, 'adaptive')
    print(f"Generated {len(candidates)} splat candidates")

    # Create splats
    splats = placer.create_splats_from_candidates(candidates)
    print(f"Created {len(splats)} splats")