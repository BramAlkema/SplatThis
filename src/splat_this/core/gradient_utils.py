"""Gradient computation and structure tensor analysis utilities for adaptive Gaussian splatting."""

import numpy as np
import logging
from typing import Tuple, Optional, Dict, Any, List
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from skimage import filters, feature
from skimage.color import rgb2gray
from skimage.filters import rank

logger = logging.getLogger(__name__)


class GradientAnalyzer:
    """Gradient computation and analysis for content-adaptive splat placement."""

    def __init__(self, sigma: float = 1.0, method: str = 'sobel'):
        """
        Initialize gradient analyzer.

        Args:
            sigma: Gaussian smoothing parameter for structure tensor
            method: Gradient computation method ('sobel', 'scharr', 'roberts', 'prewitt')
        """
        self.sigma = sigma
        self.method = method
        self.valid_methods = ['sobel', 'scharr', 'roberts', 'prewitt', 'gaussian']

        if method not in self.valid_methods:
            raise ValueError(f"Method must be one of {self.valid_methods}, got {method}")

    def compute_gradients(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute image gradients using specified method.

        Args:
            image: Input image (H, W) grayscale or (H, W, C) color

        Returns:
            Tuple of (grad_x, grad_y) gradient arrays
        """
        # Convert to grayscale if needed
        if image.ndim == 3:
            gray = rgb2gray(image)
        else:
            gray = image.copy()

        # Ensure float type for gradient computation
        gray = gray.astype(np.float64)

        if self.method == 'sobel':
            grad_x = filters.sobel_h(gray)
            grad_y = filters.sobel_v(gray)
        elif self.method == 'scharr':
            grad_x = filters.scharr_h(gray)
            grad_y = filters.scharr_v(gray)
        elif self.method == 'roberts':
            grad_x = filters.roberts_pos_diag(gray)
            grad_y = filters.roberts_neg_diag(gray)
        elif self.method == 'prewitt':
            grad_x = filters.prewitt_h(gray)
            grad_y = filters.prewitt_v(gray)
        elif self.method == 'gaussian':
            # Use Gaussian derivatives
            grad_x = gaussian_filter(gray, self.sigma, order=[0, 1])
            grad_y = gaussian_filter(gray, self.sigma, order=[1, 0])
        else:
            raise ValueError(f"Unknown gradient method: {self.method}")

        return grad_x, grad_y

    def compute_gradient_magnitude(self, image: np.ndarray) -> np.ndarray:
        """
        Compute gradient magnitude field.

        Args:
            image: Input image

        Returns:
            Gradient magnitude array (H, W)
        """
        grad_x, grad_y = self.compute_gradients(image)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        return magnitude

    def compute_gradient_orientation(self, image: np.ndarray) -> np.ndarray:
        """
        Compute gradient orientation field.

        Args:
            image: Input image

        Returns:
            Gradient orientation array (H, W) in radians [0, π)
        """
        grad_x, grad_y = self.compute_gradients(image)
        orientation = np.arctan2(grad_y, grad_x) % np.pi
        return orientation

    def compute_structure_tensor(self, image: np.ndarray,
                               smoothing_sigma: Optional[float] = None) -> np.ndarray:
        """
        Compute structure tensor for local orientation analysis.

        Args:
            image: Input image
            smoothing_sigma: Gaussian smoothing parameter (defaults to self.sigma)

        Returns:
            Structure tensor field (H, W, 2, 2)
        """
        if smoothing_sigma is None:
            smoothing_sigma = self.sigma

        # Compute gradients
        grad_x, grad_y = self.compute_gradients(image)

        # Structure tensor components
        J11 = gaussian_filter(grad_x * grad_x, smoothing_sigma)
        J12 = gaussian_filter(grad_x * grad_y, smoothing_sigma)
        J22 = gaussian_filter(grad_y * grad_y, smoothing_sigma)

        # Build tensor field
        H, W = grad_x.shape
        tensor_field = np.zeros((H, W, 2, 2), dtype=np.float64)
        tensor_field[:, :, 0, 0] = J11
        tensor_field[:, :, 0, 1] = J12
        tensor_field[:, :, 1, 0] = J12
        tensor_field[:, :, 1, 1] = J22

        return tensor_field

    def analyze_local_structure(self, tensor_field: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract edge strength, orientation, and coherence from structure tensor.

        Args:
            tensor_field: Structure tensor field (H, W, 2, 2)

        Returns:
            Tuple of (edge_strength, orientation, coherence) arrays
        """
        H, W = tensor_field.shape[:2]
        edge_strength = np.zeros((H, W))
        orientation = np.zeros((H, W))
        coherence = np.zeros((H, W))

        for i in range(H):
            for j in range(W):
                T = tensor_field[i, j]

                try:
                    eigenvals, eigenvecs = np.linalg.eigh(T)

                    # Sort eigenvalues (largest first)
                    idx = np.argsort(eigenvals)[::-1]
                    lambda1, lambda2 = eigenvals[idx]

                    # Edge strength (largest eigenvalue)
                    edge_strength[i, j] = np.sqrt(max(lambda1, 0))

                    # Orientation (principal eigenvector angle)
                    if lambda1 > 1e-10:
                        principal_vec = eigenvecs[:, idx[0]]
                        orientation[i, j] = np.arctan2(principal_vec[1], principal_vec[0]) % np.pi

                    # Coherence measure
                    if lambda1 + lambda2 > 1e-10:
                        coherence[i, j] = (lambda1 - lambda2) / (lambda1 + lambda2)
                        coherence[i, j] = max(0.0, coherence[i, j])  # Ensure non-negative

                except np.linalg.LinAlgError:
                    # Handle singular matrices
                    edge_strength[i, j] = 0.0
                    orientation[i, j] = 0.0
                    coherence[i, j] = 0.0

        return edge_strength, orientation, coherence


class ProbabilityMapGenerator:
    """Generate probability maps for content-adaptive splat placement."""

    def __init__(self, gradient_weight: float = 0.7, uniform_weight: float = 0.3):
        """
        Initialize probability map generator.

        Args:
            gradient_weight: Weight for gradient-based placement [0,1]
            uniform_weight: Weight for uniform coverage [0,1]
        """
        if abs(gradient_weight + uniform_weight - 1.0) > 1e-6:
            logger.warning(f"Weights don't sum to 1.0: {gradient_weight + uniform_weight}")

        self.gradient_weight = gradient_weight
        self.uniform_weight = uniform_weight

    def create_gradient_probability_map(self, gradient_magnitude: np.ndarray,
                                      power: float = 1.0) -> np.ndarray:
        """
        Create probability map from gradient magnitude.

        Args:
            gradient_magnitude: Gradient magnitude field (H, W)
            power: Power to raise gradient values (>1 increases contrast)

        Returns:
            Normalized probability map (H, W)
        """
        # Avoid division by zero
        magnitude = gradient_magnitude.copy()
        max_val = np.max(magnitude)

        if max_val > 1e-10:
            magnitude = magnitude / max_val

        # Apply power for contrast adjustment
        if power != 1.0:
            magnitude = np.power(magnitude, power)

        # Normalize to probability distribution
        total = np.sum(magnitude)
        if total > 1e-10:
            prob_map = magnitude / total
        else:
            # Fallback to uniform if no gradients
            prob_map = np.ones_like(magnitude) / magnitude.size

        return prob_map

    def create_mixed_probability_map(self, image: np.ndarray,
                                   gradient_analyzer: GradientAnalyzer,
                                   gradient_power: float = 1.0) -> np.ndarray:
        """
        Create mixed probability map combining gradient and uniform components.

        Args:
            image: Input image
            gradient_analyzer: Gradient analyzer instance
            gradient_power: Power for gradient contrast adjustment

        Returns:
            Mixed probability map (H, W)
        """
        H, W = image.shape[:2]

        # Compute gradient component
        grad_magnitude = gradient_analyzer.compute_gradient_magnitude(image)
        grad_prob = self.create_gradient_probability_map(grad_magnitude, gradient_power)

        # Uniform component
        uniform_prob = np.ones((H, W)) / (H * W)

        # Mix components
        mixed_prob = (self.gradient_weight * grad_prob +
                     self.uniform_weight * uniform_prob)

        # Ensure normalization
        total = np.sum(mixed_prob)
        if total > 1e-10:
            mixed_prob = mixed_prob / total

        return mixed_prob

    def create_saliency_probability_map(self, image: np.ndarray,
                                      edge_threshold: float = 0.1) -> np.ndarray:
        """
        Create probability map based on saliency and edge detection.

        Args:
            image: Input image
            edge_threshold: Threshold for edge detection

        Returns:
            Saliency-based probability map (H, W)
        """
        # Convert to grayscale if needed
        if image.ndim == 3:
            gray = rgb2gray(image)
        else:
            gray = image.copy()

        # Compute multiple saliency cues
        # 1. Edge strength
        edges = feature.canny(gray, sigma=1.0, low_threshold=edge_threshold/2,
                             high_threshold=edge_threshold)
        edge_saliency = gaussian_filter(edges.astype(float), sigma=2.0)

        # 2. Local variance (texture)
        variance_saliency = rank.variance(
            (gray * 255).astype(np.uint8),
            np.ones((5, 5))
        ).astype(float)
        variance_saliency = gaussian_filter(variance_saliency, sigma=1.0)

        # 3. Gradient magnitude
        grad_magnitude = filters.sobel(gray)
        grad_saliency = gaussian_filter(grad_magnitude, sigma=1.0)

        # Combine saliency measures
        saliency = (0.4 * edge_saliency +
                   0.3 * variance_saliency +
                   0.3 * grad_saliency)

        # Normalize to probability distribution
        saliency = saliency / (np.max(saliency) + 1e-10)
        prob_map = saliency / (np.sum(saliency) + 1e-10)

        return prob_map


class SpatialSampler:
    """Sample spatial positions from probability distributions."""

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize spatial sampler.

        Args:
            seed: Random seed for reproducible sampling
        """
        if seed is not None:
            np.random.seed(seed)

    def sample_from_probability_map(self, prob_map: np.ndarray,
                                  n_samples: int) -> List[Tuple[int, int]]:
        """
        Sample spatial positions from probability map.

        Args:
            prob_map: Probability distribution (H, W)
            n_samples: Number of samples to generate

        Returns:
            List of (y, x) positions
        """
        H, W = prob_map.shape

        # Flatten probability map
        flat_prob = prob_map.flatten()

        # Ensure probabilities sum to 1
        total_prob = np.sum(flat_prob)
        if total_prob > 1e-10:
            flat_prob = flat_prob / total_prob
        else:
            # Fallback to uniform
            flat_prob = np.ones_like(flat_prob) / flat_prob.size

        # Sample indices
        flat_indices = np.random.choice(
            flat_prob.size,
            size=n_samples,
            p=flat_prob,
            replace=True
        )

        # Convert back to 2D coordinates
        positions = []
        for idx in flat_indices:
            y, x = divmod(idx, W)
            positions.append((y, x))

        return positions

    def sample_with_minimum_distance(self, prob_map: np.ndarray,
                                    n_samples: int,
                                    min_distance: float = 5.0,
                                    max_attempts: int = 1000) -> List[Tuple[int, int]]:
        """
        Sample positions with minimum distance constraint.

        Args:
            prob_map: Probability distribution (H, W)
            n_samples: Number of samples to generate
            min_distance: Minimum distance between samples (pixels)
            max_attempts: Maximum attempts to place each sample

        Returns:
            List of (y, x) positions
        """
        positions = []

        for _ in range(n_samples):
            placed = False

            for attempt in range(max_attempts):
                # Sample candidate position
                candidates = self.sample_from_probability_map(prob_map, 1)
                candidate = candidates[0]

                # Check distance to existing positions
                valid = True
                for existing in positions:
                    dist = np.sqrt((candidate[0] - existing[0])**2 +
                                 (candidate[1] - existing[1])**2)
                    if dist < min_distance:
                        valid = False
                        break

                if valid:
                    positions.append(candidate)
                    placed = True
                    break

            if not placed:
                # Fallback: place anyway after max attempts
                candidates = self.sample_from_probability_map(prob_map, 1)
                positions.append(candidates[0])

        return positions

    def sample_stratified(self, prob_map: np.ndarray,
                         n_samples: int,
                         grid_divisions: int = 4) -> List[Tuple[int, int]]:
        """
        Stratified sampling to ensure coverage across image regions.

        Args:
            prob_map: Probability distribution (H, W)
            n_samples: Number of samples to generate
            grid_divisions: Number of grid divisions per dimension

        Returns:
            List of (y, x) positions
        """
        H, W = prob_map.shape
        positions = []

        # Calculate samples per cell
        total_cells = grid_divisions * grid_divisions
        samples_per_cell = n_samples // total_cells
        remaining_samples = n_samples % total_cells

        cell_h = H // grid_divisions
        cell_w = W // grid_divisions

        for grid_y in range(grid_divisions):
            for grid_x in range(grid_divisions):
                # Cell boundaries
                y_start = grid_y * cell_h
                y_end = min((grid_y + 1) * cell_h, H)
                x_start = grid_x * cell_w
                x_end = min((grid_x + 1) * cell_w, W)

                # Extract cell probability
                cell_prob = prob_map[y_start:y_end, x_start:x_end].copy()

                # Normalize cell probability
                total_cell_prob = np.sum(cell_prob)
                if total_cell_prob > 1e-10:
                    cell_prob = cell_prob / total_cell_prob
                else:
                    cell_prob = np.ones_like(cell_prob) / cell_prob.size

                # Number of samples for this cell
                n_cell_samples = samples_per_cell
                if remaining_samples > 0:
                    n_cell_samples += 1
                    remaining_samples -= 1

                # Sample within cell
                if n_cell_samples > 0:
                    cell_positions = self.sample_from_probability_map(cell_prob, n_cell_samples)

                    # Convert to global coordinates
                    for cy, cx in cell_positions:
                        global_y = y_start + cy
                        global_x = x_start + cx
                        positions.append((global_y, global_x))

        return positions


class EdgeDetector:
    """Edge detection and smoothing utilities."""

    def __init__(self):
        """Initialize edge detector."""
        pass

    def detect_edges_canny(self, image: np.ndarray,
                          sigma: float = 1.0,
                          low_threshold: float = 0.1,
                          high_threshold: float = 0.2) -> np.ndarray:
        """
        Detect edges using Canny edge detector.

        Args:
            image: Input image
            sigma: Gaussian smoothing parameter
            low_threshold: Low threshold for edge linking
            high_threshold: High threshold for edge detection

        Returns:
            Binary edge map (H, W)
        """
        # Convert to grayscale if needed
        if image.ndim == 3:
            gray = rgb2gray(image)
        else:
            gray = image.copy()

        edges = feature.canny(gray, sigma=sigma,
                             low_threshold=low_threshold,
                             high_threshold=high_threshold)

        return edges

    def detect_edges_gradient(self, image: np.ndarray,
                            threshold: float = 0.1) -> np.ndarray:
        """
        Detect edges using gradient magnitude thresholding.

        Args:
            image: Input image
            threshold: Gradient magnitude threshold

        Returns:
            Binary edge map (H, W)
        """
        analyzer = GradientAnalyzer(method='sobel')
        grad_magnitude = analyzer.compute_gradient_magnitude(image)

        # Normalize gradient magnitude
        max_grad = np.max(grad_magnitude)
        if max_grad > 1e-10:
            grad_magnitude = grad_magnitude / max_grad

        edges = grad_magnitude > threshold
        return edges

    def apply_gaussian_smoothing(self, image: np.ndarray,
                                sigma: float) -> np.ndarray:
        """
        Apply Gaussian smoothing to image.

        Args:
            image: Input image
            sigma: Gaussian standard deviation

        Returns:
            Smoothed image
        """
        if image.ndim == 3:
            # Apply to each channel separately
            smoothed = np.zeros_like(image)
            for c in range(image.shape[2]):
                smoothed[:, :, c] = gaussian_filter(image[:, :, c], sigma)
            return smoothed
        else:
            return gaussian_filter(image, sigma)

    def compute_edge_orientation_map(self, image: np.ndarray,
                                   sigma: float = 1.0) -> np.ndarray:
        """
        Compute edge orientation map.

        Args:
            image: Input image
            sigma: Smoothing parameter

        Returns:
            Edge orientation map (H, W) in radians [0, π)
        """
        analyzer = GradientAnalyzer(sigma=sigma, method='gaussian')
        orientation = analyzer.compute_gradient_orientation(image)
        return orientation


def visualize_gradient_analysis(image: np.ndarray,
                               output_path: Optional[str] = None) -> Dict[str, np.ndarray]:
    """
    Comprehensive gradient analysis visualization.

    Args:
        image: Input image
        output_path: Optional path to save visualization

    Returns:
        Dictionary of analysis results
    """
    # Initialize analyzer
    analyzer = GradientAnalyzer(sigma=1.0, method='sobel')
    prob_generator = ProbabilityMapGenerator()
    edge_detector = EdgeDetector()

    # Compute all analysis components
    grad_x, grad_y = analyzer.compute_gradients(image)
    grad_magnitude = analyzer.compute_gradient_magnitude(image)
    grad_orientation = analyzer.compute_gradient_orientation(image)

    structure_tensor = analyzer.compute_structure_tensor(image)
    edge_strength, principal_orientation, coherence = analyzer.analyze_local_structure(structure_tensor)

    prob_map = prob_generator.create_mixed_probability_map(image, analyzer)
    edges = edge_detector.detect_edges_canny(image)

    results = {
        'original_image': image,
        'gradient_x': grad_x,
        'gradient_y': grad_y,
        'gradient_magnitude': grad_magnitude,
        'gradient_orientation': grad_orientation,
        'edge_strength': edge_strength,
        'principal_orientation': principal_orientation,
        'coherence': coherence,
        'probability_map': prob_map,
        'edges': edges
    }

    logger.info(f"Gradient analysis complete. Results shape: {image.shape}")
    logger.info(f"Max gradient magnitude: {np.max(grad_magnitude):.4f}")
    logger.info(f"Mean edge strength: {np.mean(edge_strength):.4f}")
    logger.info(f"Mean coherence: {np.mean(coherence):.4f}")

    return results


# Convenience functions for common operations
def compute_image_gradients(image: np.ndarray, method: str = 'sobel') -> Tuple[np.ndarray, np.ndarray]:
    """Convenience function to compute image gradients."""
    analyzer = GradientAnalyzer(method=method)
    return analyzer.compute_gradients(image)


def create_content_probability_map(image: np.ndarray,
                                 gradient_weight: float = 0.7) -> np.ndarray:
    """Convenience function to create content-adaptive probability map."""
    analyzer = GradientAnalyzer()
    generator = ProbabilityMapGenerator(gradient_weight=gradient_weight)
    return generator.create_mixed_probability_map(image, analyzer)


def sample_adaptive_positions(image: np.ndarray,
                            n_samples: int,
                            method: str = 'mixed') -> List[Tuple[int, int]]:
    """Convenience function to sample adaptive positions."""
    sampler = SpatialSampler()

    if method == 'mixed':
        prob_map = create_content_probability_map(image)
    elif method == 'saliency':
        generator = ProbabilityMapGenerator()
        prob_map = generator.create_saliency_probability_map(image)
    elif method == 'uniform':
        H, W = image.shape[:2]
        prob_map = np.ones((H, W)) / (H * W)
    else:
        raise ValueError(f"Unknown sampling method: {method}")

    return sampler.sample_from_probability_map(prob_map, n_samples)