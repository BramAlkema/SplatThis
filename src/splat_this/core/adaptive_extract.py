"""Adaptive Gaussian splat extraction inspired by NYU Image-GS.

This module implements content-adaptive splat sizing and placement based on:
- Error-guided optimization
- Saliency-aware initialization
- Progressive refinement
- Anisotropic Gaussian optimization
"""

import numpy as np
import logging
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass
from skimage.segmentation import slic, felzenszwalb
# peak_local_maxima not available in all skimage versions, implement our own
from skimage.filters import gaussian, sobel
from skimage.measure import regionprops
from scipy.optimize import minimize
from scipy.ndimage import binary_dilation, maximum_filter
import math


def peak_local_maxima(image: np.ndarray, min_distance: int = 1,
                     threshold_abs: float = None) -> Tuple[np.ndarray, np.ndarray]:
    """Simple implementation of peak local maxima detection."""
    if threshold_abs is None:
        threshold_abs = np.mean(image)

    # Use maximum filter to find local maxima
    neighborhood = np.ones((min_distance*2+1, min_distance*2+1))
    local_maxima = maximum_filter(image, footprint=neighborhood) == image

    # Apply threshold
    above_threshold = image > threshold_abs
    peaks = local_maxima & above_threshold

    # Return coordinates as tuple of arrays (row, col)
    return np.where(peaks)

from .extract import Gaussian
from ..utils.math import safe_eigendecomposition, clamp_value
from .progressive_allocator import ProgressiveConfig, ProgressiveAllocator
from .error_guided_placement import ErrorGuidedPlacement
from ..utils.reconstruction_error import compute_reconstruction_error

logger = logging.getLogger(__name__)


@dataclass
class AdaptiveSplatConfig:
    """Configuration for adaptive splat extraction."""

    # Progressive allocation mode
    enable_progressive: bool = True  # Use progressive allocation by default

    # Initialization strategies
    init_strategy: str = "saliency"  # "random", "gradient", "saliency"
    random_ratio: float = 0.3  # Ratio of randomly initialized splats

    # Size adaptation
    min_scale: float = 2.0  # Minimum splat scale (increased for visibility)
    max_scale: float = 20.0  # Maximum splat scale
    scale_variance_threshold: float = 0.1  # Threshold for scale adaptation

    # Legacy parameters (kept for compatibility with existing code)
    refinement_iterations: int = 5
    error_threshold: float = 0.01  # Reconstruction error threshold

    # Saliency parameters
    edge_weight: float = 0.4
    variance_weight: float = 0.3
    gradient_weight: float = 0.3


class SaliencyAnalyzer:
    """Analyzes image content to guide splat placement."""

    def __init__(self, config: AdaptiveSplatConfig):
        self.config = config

    def compute_saliency_map(self, image: np.ndarray) -> np.ndarray:
        """Compute content-aware saliency map for splat placement.

        Args:
            image: Input image (H, W, 3)

        Returns:
            Saliency map (H, W) with values 0-1
        """
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image

        # Edge detection component
        edges = sobel(gray)
        edge_saliency = (edges - edges.min()) / (edges.max() - edges.min() + 1e-8)

        # Local variance component
        variance_map = self._compute_local_variance(gray)
        variance_saliency = (variance_map - variance_map.min()) / (variance_map.max() - variance_map.min() + 1e-8)

        # Gradient magnitude component
        gy, gx = np.gradient(gray)
        gradient_mag = np.sqrt(gx**2 + gy**2)
        gradient_saliency = (gradient_mag - gradient_mag.min()) / (gradient_mag.max() - gradient_mag.min() + 1e-8)

        # Combine components
        saliency = (
            self.config.edge_weight * edge_saliency +
            self.config.variance_weight * variance_saliency +
            self.config.gradient_weight * gradient_saliency
        )

        # Smooth the saliency map
        saliency = gaussian(saliency, sigma=1.0)

        return saliency

    def _compute_local_variance(self, image: np.ndarray, window_size: int = 5) -> np.ndarray:
        """Compute local variance at each pixel using efficient convolution."""
        from scipy.ndimage import uniform_filter

        # Use convolution-based approach for efficiency
        # Var(X) = E[X^2] - E[X]^2

        # Compute local mean using uniform filter (convolution)
        local_mean = uniform_filter(image.astype(np.float64), size=window_size, mode='constant')

        # Compute local mean of squares
        local_mean_sq = uniform_filter(image.astype(np.float64)**2, size=window_size, mode='constant')

        # Compute variance using the formula above
        variance_map = local_mean_sq - local_mean**2

        # Ensure non-negative values (numerical precision issues)
        variance_map = np.maximum(variance_map, 0)

        return variance_map

    def _find_local_maxima(
        self,
        image: np.ndarray,
        min_distance: int = 8,
        threshold_abs: float = 0.3
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Find local maxima in an image using scipy.

        Args:
            image: Input image
            min_distance: Minimum distance between peaks
            threshold_abs: Absolute threshold for peak detection

        Returns:
            Tuple of (row_indices, col_indices) for peaks
        """
        from scipy.ndimage import maximum_filter, binary_erosion

        # Apply maximum filter to find local maxima
        neighborhood = np.ones((min_distance, min_distance))
        local_maxima = maximum_filter(image, footprint=neighborhood) == image

        # Apply threshold
        above_threshold = image > threshold_abs

        # Combine conditions
        peaks = local_maxima & above_threshold

        # Get coordinates
        peak_coords = np.where(peaks)
        return peak_coords

    def detect_saliency_peaks(
        self,
        saliency_map: np.ndarray,
        min_distance: int = 8,
        threshold_abs: float = 0.3
    ) -> List[Tuple[int, int]]:
        """Detect peaks in saliency map for high-importance regions.

        Args:
            saliency_map: Computed saliency map (H, W)
            min_distance: Minimum distance between peaks
            threshold_abs: Absolute threshold for peak detection

        Returns:
            List of (y, x) peak coordinates
        """
        peaks = self._find_local_maxima(
            saliency_map,
            min_distance=min_distance,
            threshold_abs=threshold_abs
        )

        # Convert to list of tuples and sort by saliency value (descending)
        peak_coords = list(zip(peaks[0], peaks[1]))
        peak_values = [saliency_map[y, x] for y, x in peak_coords]

        # Sort by saliency value (highest first)
        sorted_peaks = sorted(zip(peak_coords, peak_values), key=lambda x: x[1], reverse=True)

        return [coord for coord, _ in sorted_peaks]

    def compute_multi_scale_saliency(self, image: np.ndarray, scales: List[float] = None) -> np.ndarray:
        """Compute multi-scale saliency for better feature detection.

        Args:
            image: Input image (H, W, 3)
            scales: List of Gaussian sigma values for multi-scale analysis

        Returns:
            Multi-scale saliency map (H, W)
        """
        if scales is None:
            scales = [0.5, 1.0, 2.0, 4.0]

        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image

        multi_scale_saliency = np.zeros_like(gray)

        for i, sigma in enumerate(scales):
            # Apply Gaussian smoothing at this scale
            smoothed = gaussian(gray, sigma=sigma)

            # Compute gradients at this scale
            gy, gx = np.gradient(smoothed)
            gradient_mag = np.sqrt(gx**2 + gy**2)

            # Normalize and weight by scale
            if gradient_mag.max() > 0:
                scale_saliency = gradient_mag / gradient_mag.max()
                weight = 1.0 / (i + 1)  # Give more weight to finer scales
                multi_scale_saliency += weight * scale_saliency

        # Normalize final result
        if multi_scale_saliency.max() > 0:
            multi_scale_saliency = multi_scale_saliency / multi_scale_saliency.max()

        return multi_scale_saliency

    def analyze_content_complexity(self, image: np.ndarray) -> Dict[str, float]:
        """Analyze overall image content complexity metrics.

        Args:
            image: Input image (H, W, 3)

        Returns:
            Dictionary of complexity metrics
        """
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image

        # Compute various complexity measures

        # 1. Global variance
        global_variance = np.var(gray)

        # 2. Edge density
        edges = sobel(gray)
        edge_density = np.mean(edges > np.percentile(edges, 75))

        # 3. Local variance statistics
        local_var = self._compute_local_variance(gray, window_size=7)
        variance_mean = np.mean(local_var)
        variance_std = np.std(local_var)

        # 4. Gradient statistics
        gy, gx = np.gradient(gray)
        gradient_mag = np.sqrt(gx**2 + gy**2)
        gradient_mean = np.mean(gradient_mag)
        gradient_std = np.std(gradient_mag)

        # 5. Texture measure (using local standard deviation)
        from scipy.ndimage import generic_filter
        texture_measure = np.mean(generic_filter(gray, np.std, size=5))

        return {
            'global_variance': float(global_variance),
            'edge_density': float(edge_density),
            'variance_mean': float(variance_mean),
            'variance_std': float(variance_std),
            'gradient_mean': float(gradient_mean),
            'gradient_std': float(gradient_std),
            'texture_measure': float(texture_measure),
            'complexity_score': float(
                0.3 * min(global_variance / 1000, 1.0) +
                0.3 * edge_density +
                0.2 * min(variance_std / 100, 1.0) +
                0.2 * min(gradient_std / 50, 1.0)
            )
        }


class AdaptiveSplatExtractor:
    """Extract Gaussian splats with adaptive sizing and placement."""

    def __init__(self, config: Optional[AdaptiveSplatConfig] = None,
                 progressive_config: Optional[ProgressiveConfig] = None):
        self.config = config or AdaptiveSplatConfig()
        self.saliency_analyzer = SaliencyAnalyzer(self.config)

        # Initialize progressive components if enabled
        if self.config.enable_progressive:
            self.progressive_config = progressive_config or ProgressiveConfig()
            self.allocator = ProgressiveAllocator(self.progressive_config)
            self.placer = ErrorGuidedPlacement(self.progressive_config.temperature)

    def extract_adaptive_splats(
        self,
        image: np.ndarray,
        n_splats: int = 1500,
        verbose: bool = False
    ) -> List[Gaussian]:
        """Extract splats with adaptive sizing based on content.

        Args:
            image: Input image (H, W, 3)
            n_splats: Target number of splats
            verbose: Enable detailed logging

        Returns:
            List of adaptive Gaussian splats
        """
        if verbose:
            logger.info(f"Starting adaptive splat extraction with {n_splats} target splats")

        # Route to progressive or static allocation based on configuration
        if self.config.enable_progressive:
            return self._extract_progressive_splats(image, n_splats, verbose)
        else:
            return self._extract_static_splats(image, n_splats, verbose)

    def _extract_progressive_splats(
        self,
        image: np.ndarray,
        n_splats: int,
        verbose: bool
    ) -> List[Gaussian]:
        """Extract splats using progressive allocation strategy."""
        if verbose:
            logger.info("Using progressive allocation strategy")

        # Update progressive config max_splats from parameter
        self.progressive_config.max_splats = n_splats
        self.progressive_config.validate_compatibility(image.shape[:2])

        # Reset allocator for new session
        self.allocator.reset()

        # Step 1: Compute saliency map for guidance
        saliency_map = self.saliency_analyzer.compute_saliency_map(image)

        # Step 2: Initialize with sparse allocation
        initial_count = self.progressive_config.get_initial_count()
        if verbose:
            logger.info(f"Starting with {initial_count} initial splats ({initial_count/n_splats:.1%} of target)")

        current_splats = self._initialize_splats(image, saliency_map, initial_count, verbose)

        # Step 3: Progressive allocation loop with real error computation
        iteration = 0
        last_error_map = None

        while len(current_splats) < n_splats and iteration < 100:  # Safety limit
            iteration += 1

            # Render current splats to compute reconstruction error
            rendered_image = self._render_splats_to_image(current_splats, image.shape[:2])

            # Compute reconstruction error
            error_map = compute_reconstruction_error(image, rendered_image)
            mean_error = float(np.mean(error_map))

            if verbose and iteration % 10 == 0:
                logger.info(f"Iteration {iteration}: Mean error = {mean_error:.4f}")

            # Check if we should add more splats
            if self.allocator.should_add_splats(mean_error):
                add_count = self.allocator.get_addition_count(len(current_splats))

                if add_count > 0:
                    # Use error-guided placement to determine where to add new splats
                    prob_map = self.placer.create_placement_probability(error_map)
                    new_positions = self.placer.sample_positions(prob_map, add_count, min_distance=2.0)

                    # Create new splats at selected positions
                    new_splats = self._create_splats_at_positions(image, new_positions, verbose)
                    current_splats.extend(new_splats)

                    self.allocator.record_iteration(mean_error, add_count)

                    if verbose:
                        logger.info(f"Iteration {iteration}: Added {add_count} splats, total: {len(current_splats)}")
                        logger.info(f"  Error reduction: {mean_error:.4f}")
                else:
                    self.allocator.record_iteration(mean_error)
            else:
                self.allocator.record_iteration(mean_error)

            # Store error map for potential use in next iteration
            last_error_map = error_map

        if verbose:
            stats = self.allocator.get_stats()
            logger.info(f"Progressive allocation completed: {len(current_splats)} splats in {iteration} iterations")
            logger.info(f"Final error: {stats['current_error']:.4f}, Converged: {stats['converged']}")

        return current_splats

    def _extract_static_splats(
        self,
        image: np.ndarray,
        n_splats: int,
        verbose: bool
    ) -> List[Gaussian]:
        """Extract splats using static (legacy) allocation strategy."""
        if verbose:
            logger.info("Using static allocation strategy")

        # Step 1: Compute saliency map
        saliency_map = self.saliency_analyzer.compute_saliency_map(image)

        # Step 2: Initialize splats based on strategy
        initial_splats = self._initialize_splats(image, saliency_map, n_splats, verbose)

        # Step 3: Progressive refinement (temporarily disabled to preserve size variation)
        # refined_splats = self._progressive_refinement(image, initial_splats, saliency_map, verbose)
        refined_splats = initial_splats

        # Step 4: Adaptive scale optimization (temporarily disabled to preserve size variation)
        # final_splats = self._optimize_adaptive_scales(image, refined_splats, verbose)
        final_splats = refined_splats

        if verbose:
            self._log_scale_statistics(final_splats)

        return final_splats

    def _initialize_splats(
        self,
        image: np.ndarray,
        saliency_map: np.ndarray,
        n_splats: int,
        verbose: bool
    ) -> List[Gaussian]:
        """Initialize splats using the configured strategy."""

        if self.config.init_strategy == "saliency":
            return self._saliency_based_initialization(image, saliency_map, n_splats, verbose)
        elif self.config.init_strategy == "gradient":
            return self._gradient_based_initialization(image, n_splats, verbose)
        else:  # random
            return self._random_initialization(image, n_splats, verbose)

    def _saliency_based_initialization(
        self,
        image: np.ndarray,
        saliency_map: np.ndarray,
        n_splats: int,
        verbose: bool
    ) -> List[Gaussian]:
        """Initialize splats using saliency-guided placement."""

        h, w = image.shape[:2]
        splats = []

        # Find high-saliency regions
        peaks = peak_local_maxima(saliency_map, min_distance=8, threshold_abs=0.3)
        peak_coords = list(zip(peaks[0], peaks[1]))

        if verbose:
            logger.info(f"Found {len(peak_coords)} saliency peaks")

        # Segment image with fewer, larger regions for high-saliency areas
        segments = slic(image, n_segments=n_splats//2, compactness=15.0, sigma=1.5)

        # Process high-saliency regions first
        saliency_splats = []
        used_segments = set()

        for y, x in peak_coords:
            if len(saliency_splats) >= n_splats * 0.7:  # Use 70% for high-saliency
                break

            segment_id = segments[y, x]
            if segment_id in used_segments:
                continue

            mask = segments == segment_id
            if np.sum(mask) < 4:  # Skip tiny segments
                continue

            used_segments.add(segment_id)

            # Create larger splat for high-saliency region
            splat = self._create_adaptive_splat_from_mask(image, mask, saliency_map, scale_boost=1.5)
            if splat:
                saliency_splats.append(splat)

        # Fill remaining with smaller splats for detail
        detail_segments = felzenszwalb(image, scale=50, sigma=0.8, min_size=3)
        detail_splats = []

        for segment_id in np.unique(detail_segments):
            if len(saliency_splats) + len(detail_splats) >= n_splats:
                break

            if segment_id in used_segments:
                continue

            mask = detail_segments == segment_id
            if np.sum(mask) < 3:
                continue

            # Create smaller splats for detail areas
            splat = self._create_adaptive_splat_from_mask(image, mask, saliency_map, scale_boost=0.7)
            if splat:
                detail_splats.append(splat)

        splats = saliency_splats + detail_splats

        if verbose:
            logger.info(f"Created {len(saliency_splats)} saliency splats + {len(detail_splats)} detail splats")

        return splats[:n_splats]

    def _create_adaptive_splat_from_mask(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        saliency_map: np.ndarray,
        scale_boost: float = 1.0
    ) -> Optional[Gaussian]:
        """Create a splat from a segmented region with adaptive sizing."""

        coords = np.column_stack(np.where(mask))
        if len(coords) < 3:
            return None

        # Get region properties
        mean_color = image[mask].mean(axis=0)

        # Compute size based on region extent and saliency
        region_saliency = saliency_map[mask].mean()

        # Standard ellipse fitting
        try:
            cov_matrix = np.cov(coords.T)
            eigenvals, eigenvecs = safe_eigendecomposition(cov_matrix)

            if eigenvals is None:
                return None

            # Adaptive scaling based on saliency and region properties
            # Normalize eigenvalues by image dimensions for more reasonable scales
            height, width = image.shape[:2]
            img_scale = max(height, width)
            normalized_eigenvals = eigenvals / (img_scale * 0.05)  # Less aggressive normalization for visibility

            base_scale = np.sqrt(normalized_eigenvals) * scale_boost
            saliency_scale = 1.0 + region_saliency * 2.0  # Scale 1.0-3.0 based on saliency

            rx = clamp_value(base_scale[0] * saliency_scale, self.config.min_scale, self.config.max_scale)
            ry = clamp_value(base_scale[1] * saliency_scale, self.config.min_scale, self.config.max_scale)

            # Rotation angle
            if eigenvals[0] > eigenvals[1]:
                theta = np.arctan2(eigenvecs[0, 1], eigenvecs[0, 0])
            else:
                theta = np.arctan2(eigenvecs[1, 1], eigenvecs[1, 0])

            # Center position
            center = coords.mean(axis=0)
            x, y = center[1], center[0]  # Note: coords are (row, col)

            # Alpha based on saliency
            alpha = clamp_value(0.4 + region_saliency * 0.4, 0.1, 0.9)

            return Gaussian(
                x=float(x), y=float(y),
                rx=float(rx), ry=float(ry),
                theta=float(theta),
                r=int(mean_color[0] * 255), g=int(mean_color[1] * 255), b=int(mean_color[2] * 255),
                a=float(alpha),
                score=float(region_saliency)
            )

        except Exception as e:
            logger.debug(f"Failed to create adaptive splat: {e}")
            return None

    def _gradient_based_initialization(self, image: np.ndarray, n_splats: int, verbose: bool) -> List[Gaussian]:
        """Initialize splats based on gradient information."""
        # Simplified gradient-based approach - can be enhanced
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image

        # Find gradient peaks
        gy, gx = np.gradient(gray)
        gradient_mag = np.sqrt(gx**2 + gy**2)

        peaks = peak_local_maxima(gradient_mag, min_distance=5, threshold_abs=np.mean(gradient_mag))
        peak_coords = list(zip(peaks[0], peaks[1]))

        # Use SLIC segmentation but bias towards gradient regions
        segments = slic(image, n_segments=n_splats, compactness=10.0, sigma=1.0)

        splats = []
        for segment_id in np.unique(segments):
            mask = segments == segment_id
            splat = self._create_adaptive_splat_from_mask(image, mask, gradient_mag, scale_boost=1.0)
            if splat:
                splats.append(splat)

        return splats[:n_splats]

    def _random_initialization(self, image: np.ndarray, n_splats: int, verbose: bool) -> List[Gaussian]:
        """Random initialization strategy."""
        # Simple random segmentation
        segments = felzenszwalb(image, scale=100, sigma=1.0, min_size=4)

        splats = []
        for segment_id in np.unique(segments):
            if len(splats) >= n_splats:
                break

            mask = segments == segment_id
            # Use uniform saliency for random init
            uniform_saliency = np.ones_like(image[:,:,0]) * 0.5
            splat = self._create_adaptive_splat_from_mask(image, mask, uniform_saliency)
            if splat:
                splats.append(splat)

        return splats[:n_splats]

    def _progressive_refinement(
        self,
        image: np.ndarray,
        splats: List[Gaussian],
        saliency_map: np.ndarray,
        verbose: bool
    ) -> List[Gaussian]:
        """Progressive refinement of splat parameters."""

        if verbose:
            logger.info(f"Starting progressive refinement for {len(splats)} splats")

        refined_splats = splats.copy()

        for iteration in range(self.config.refinement_iterations):
            # Compute reconstruction error (simplified)
            error_map = self._compute_reconstruction_error(image, refined_splats)

            # Refine splats in high-error regions
            high_error_regions = error_map > np.percentile(error_map, 80)

            for i, splat in enumerate(refined_splats):
                # Check if splat overlaps with high-error region
                x, y = int(splat.x), int(splat.y)
                if 0 <= y < high_error_regions.shape[0] and 0 <= x < high_error_regions.shape[1]:
                    if high_error_regions[y, x]:
                        # Refine this splat
                        refined_splats[i] = self._refine_single_splat(splat, saliency_map, error_map)

            if verbose:
                avg_error = np.mean(error_map)
                logger.info(f"Refinement iteration {iteration + 1}: avg error = {avg_error:.4f}")

        return refined_splats

    def _compute_reconstruction_error(self, image: np.ndarray, splats: List[Gaussian]) -> np.ndarray:
        """Compute reconstruction error map (simplified version)."""
        # This is a simplified error computation
        # In practice, you'd render the splats and compare to original

        h, w = image.shape[:2]
        reconstructed = np.zeros_like(image)

        # Simple reconstruction by drawing ellipses (very simplified)
        for splat in splats:
            y, x = int(splat.y), int(splat.x)
            if 0 <= y < h and 0 <= x < w:
                # Very simple: just put the color at the center
                reconstructed[y, x] = [splat.r, splat.g, splat.b]

        # Compute error
        if len(image.shape) == 3:
            error = np.mean((image - reconstructed) ** 2, axis=2)
        else:
            error = (image - np.mean(reconstructed, axis=2)) ** 2

        return error

    def _refine_single_splat(
        self,
        splat: Gaussian,
        saliency_map: np.ndarray,
        error_map: np.ndarray
    ) -> Gaussian:
        """Refine a single splat based on local error and saliency."""

        # Get local saliency and error
        x, y = int(splat.x), int(splat.y)
        h, w = saliency_map.shape

        if 0 <= y < h and 0 <= x < w:
            local_saliency = saliency_map[y, x]
            local_error = error_map[y, x]

            # Adjust scale based on error (more conservative scaling)
            error_scale = 1.0 + local_error * 0.1  # Reduced from 0.5 to 0.1
            new_rx = clamp_value(splat.rx * error_scale, self.config.min_scale, self.config.max_scale)
            new_ry = clamp_value(splat.ry * error_scale, self.config.min_scale, self.config.max_scale)

            # Adjust alpha based on saliency
            new_alpha = clamp_value(splat.a + local_saliency * 0.1 - 0.05, 0.1, 0.9)

            return Gaussian(
                x=splat.x, y=splat.y,
                rx=new_rx, ry=new_ry, theta=splat.theta,
                r=splat.r, g=splat.g, b=splat.b,
                a=new_alpha, score=splat.score
            )

        return splat

    def _optimize_adaptive_scales(
        self,
        image: np.ndarray,
        splats: List[Gaussian],
        verbose: bool
    ) -> List[Gaussian]:
        """Final optimization of adaptive scales."""

        if verbose:
            logger.info("Optimizing adaptive scales")

        optimized_splats = []

        for splat in splats:
            # Analyze local region around splat
            x, y = int(splat.x), int(splat.y)
            h, w = image.shape[:2]

            # Define local region
            radius = max(int(max(splat.rx, splat.ry)), 5)
            y1, y2 = max(0, y - radius), min(h, y + radius)
            x1, x2 = max(0, x - radius), min(w, x + radius)

            if y2 > y1 and x2 > x1:
                local_region = image[y1:y2, x1:x2]

                # Compute local statistics
                local_variance = np.var(local_region)
                local_gradient = np.mean(np.gradient(np.mean(local_region, axis=2) if len(local_region.shape) == 3 else local_region))

                # Adaptive scale based on local content (more conservative)
                content_scale = 1.0 + local_variance * 0.0005 + abs(local_gradient) * 0.02

                new_rx = clamp_value(splat.rx * content_scale, self.config.min_scale, self.config.max_scale)
                new_ry = clamp_value(splat.ry * content_scale, self.config.min_scale, self.config.max_scale)

                optimized_splat = Gaussian(
                    x=splat.x, y=splat.y,
                    rx=new_rx, ry=new_ry, theta=splat.theta,
                    r=splat.r, g=splat.g, b=splat.b,
                    a=splat.a, score=splat.score
                )
                optimized_splats.append(optimized_splat)
            else:
                optimized_splats.append(splat)

        return optimized_splats

    def _log_scale_statistics(self, splats: List[Gaussian]) -> None:
        """Log statistics about splat scale distribution."""

        scales = [(s.rx + s.ry) / 2 for s in splats]

        logger.info(f"Splat scale statistics:")
        logger.info(f"  Min scale: {min(scales):.2f}")
        logger.info(f"  Max scale: {max(scales):.2f}")
        logger.info(f"  Mean scale: {np.mean(scales):.2f}")
        logger.info(f"  Std scale: {np.std(scales):.2f}")

        # Scale distribution
        small_splats = sum(1 for s in scales if s < 2.0)
        medium_splats = sum(1 for s in scales if 2.0 <= s < 5.0)
        large_splats = sum(1 for s in scales if s >= 5.0)

        logger.info(f"  Small splats (<2.0): {small_splats}")
        logger.info(f"  Medium splats (2.0-5.0): {medium_splats}")
        logger.info(f"  Large splats (>=5.0): {large_splats}")

    def _create_splats_at_positions(
        self,
        image: np.ndarray,
        positions: List[Tuple[int, int]],
        verbose: bool = False
    ) -> List[Gaussian]:
        """Create new splats at specified positions with adaptive sizing.

        Args:
            image: Input image (H, W, 3)
            positions: List of (y, x) pixel positions
            verbose: Enable detailed logging

        Returns:
            List of new Gaussian splats
        """
        splats = []
        height, width = image.shape[:2]

        for y, x in positions:
            # Ensure positions are within image bounds
            if not (0 <= y < height and 0 <= x < width):
                if verbose:
                    logger.warning(f"Position ({y}, {x}) is outside image bounds, skipping")
                continue

            # Extract local image patch for color and scale estimation
            patch_size = 3
            y_start = max(0, y - patch_size // 2)
            y_end = min(height, y + patch_size // 2 + 1)
            x_start = max(0, x - patch_size // 2)
            x_end = min(width, x + patch_size // 2 + 1)

            if len(image.shape) == 3:
                local_patch = image[y_start:y_end, x_start:x_end, :]
                # Use mean color of local patch
                color = np.mean(local_patch.reshape(-1, 3), axis=0)
            else:
                local_patch = image[y_start:y_end, x_start:x_end]
                # Convert grayscale to RGB
                mean_intensity = np.mean(local_patch)
                color = np.array([mean_intensity, mean_intensity, mean_intensity])

            # Adaptive scale based on local variance
            local_variance = np.var(local_patch)
            # Scale between min_scale and max_scale based on variance
            variance_factor = np.clip(local_variance / 0.1, 0.0, 1.0)  # Normalize variance
            scale = self.config.min_scale + variance_factor * (self.config.max_scale - self.config.min_scale)

            # Create Gaussian splat
            splat = Gaussian(
                x=float(x),
                y=float(y),
                rx=scale,
                ry=scale,
                theta=0.0,
                r=int(np.clip(color[0] * 255, 0, 255)),
                g=int(np.clip(color[1] * 255, 0, 255)),
                b=int(np.clip(color[2] * 255, 0, 255)),
                a=1.0
            )

            splats.append(splat)

        if verbose and splats:
            scales = [(s.rx + s.ry) / 2 for s in splats]
            logger.info(f"Created {len(splats)} splats with scale range: {min(scales):.1f}-{max(scales):.1f}")

        return splats

    def _render_splats_to_image(
        self,
        splats: List[Gaussian],
        image_shape: Tuple[int, int]
    ) -> np.ndarray:
        """Render list of splats back to an image for error computation.

        Args:
            splats: List of Gaussian splats
            image_shape: (height, width) of target image

        Returns:
            Rendered image (H, W, 3) as float32
        """
        height, width = image_shape
        rendered = np.zeros((height, width, 3), dtype=np.float32)

        # Create coordinate grids
        y_coords, x_coords = np.mgrid[0:height, 0:width]
        coords = np.stack([x_coords, y_coords], axis=-1)  # Shape: (H, W, 2)

        for splat in splats:
            # Splat center position
            center = np.array([splat.x, splat.y])

            # Compute distance from center
            diff = coords - center  # Shape: (H, W, 2)

            # Apply rotation if needed (for now, assume no rotation)
            if hasattr(splat, 'theta') and splat.theta != 0:
                cos_theta = np.cos(splat.theta)
                sin_theta = np.sin(splat.theta)
                rotation_matrix = np.array([[cos_theta, -sin_theta],
                                          [sin_theta, cos_theta]])
                diff_rotated = np.dot(diff, rotation_matrix.T)
            else:
                diff_rotated = diff

            # Compute elliptical distance
            x_normalized = diff_rotated[..., 0] / max(splat.rx, 0.1)
            y_normalized = diff_rotated[..., 1] / max(splat.ry, 0.1)
            distance_sq = x_normalized**2 + y_normalized**2

            # Gaussian weight (using 2D Gaussian)
            weight = np.exp(-0.5 * distance_sq)

            # Apply opacity
            if hasattr(splat, 'a'):
                weight *= splat.a

            # Convert RGB from [0, 255] to [0, 1] range
            color = np.array([splat.r, splat.g, splat.b], dtype=np.float32) / 255.0

            # Add contribution to rendered image
            weight_expanded = weight[..., np.newaxis]  # Shape: (H, W, 1)
            contribution = weight_expanded * color  # Shape: (H, W, 3)

            # Alpha blending (additive for now, could be more sophisticated)
            rendered += contribution

        # Clip to valid range
        rendered = np.clip(rendered, 0.0, 1.0)

        return rendered