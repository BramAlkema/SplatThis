"""Gradient-guided placement algorithm for content-adaptive Gaussian splatting.

This module implements intelligent splat placement that replaces uniform SLIC segmentation
with gradient-guided, content-aware initialization following Image-GS methodology.
"""

import math
import numpy as np
import logging
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass, field
from scipy import ndimage
from scipy.ndimage import maximum_filter, label
from skimage.feature import peak_local_max

from .gradient_utils import GradientAnalyzer, ProbabilityMapGenerator, SpatialSampler
from .adaptive_gaussian import AdaptiveGaussian2D

logger = logging.getLogger(__name__)


@dataclass
class PlacementConfig:
    """Configuration for gradient-guided placement algorithm."""

    # Density control
    base_splat_density: float = 0.01        # Base splats per pixel (1% coverage baseline)
    density_multiplier_range: Tuple[float, float] = (0.5, 4.0)  # Density scaling range
    max_splats: int = 2000                  # Maximum number of splats
    min_splats: int = 50                    # Minimum number of splats

    # Probability mixing
    gradient_weight: float = 0.7            # Weight for gradient-based placement [0,1]
    uniform_weight: float = 0.3             # Weight for uniform coverage [0,1]

    # Local maxima detection
    maxima_threshold: float = 0.1           # Threshold for gradient maxima detection
    maxima_min_distance: int = 5            # Minimum distance between maxima (pixels)
    maxima_weight: float = 0.2              # Weight for maxima-based placement

    # Adaptive complexity
    complexity_smoothing: float = 2.0       # Gaussian smoothing for complexity estimation
    complexity_power: float = 1.5           # Power for complexity-based density scaling

    # Spatial constraints
    min_splat_distance: float = 2.0         # Minimum distance between splats (pixels)
    border_margin: int = 2                  # Margin from image borders (pixels)

    # Quality validation
    coverage_target: float = 0.95           # Target coverage ratio for validation
    distribution_uniformity_threshold: float = 0.8  # Uniformity requirement [0,1]

    def __post_init__(self):
        """Validate configuration parameters."""
        if abs(self.gradient_weight + self.uniform_weight - 1.0) > 1e-6:
            logger.warning(f"Gradient and uniform weights don't sum to 1.0: "
                         f"{self.gradient_weight + self.uniform_weight}")

        if self.max_splats <= self.min_splats:
            raise ValueError(f"max_splats ({self.max_splats}) must be > min_splats ({self.min_splats})")

        if not (0 < self.gradient_weight < 1) or not (0 < self.uniform_weight < 1):
            raise ValueError("Gradient and uniform weights must be in (0,1)")

        if self.base_splat_density <= 0:
            raise ValueError(f"Base splat density must be positive, got {self.base_splat_density}")


@dataclass
class PlacementResult:
    """Result of gradient-guided placement algorithm."""

    positions: List[Tuple[int, int]]        # Splat positions (y, x) in pixels
    normalized_positions: List[Tuple[float, float]]  # Positions in [0,1]²
    complexity_map: np.ndarray              # Local complexity measure (H, W)
    density_map: np.ndarray                 # Adaptive density map (H, W)
    probability_map: np.ndarray             # Final placement probability map (H, W)

    # Quality metrics
    coverage_achieved: float = 0.0          # Actual coverage ratio
    distribution_uniformity: float = 0.0    # Spatial distribution uniformity
    gradient_alignment: float = 0.0         # Alignment with gradient features

    # Placement statistics
    total_splats: int = 0                   # Total number of placed splats
    gradient_guided_splats: int = 0         # Splats placed by gradient guidance
    uniform_coverage_splats: int = 0        # Splats placed for uniform coverage
    maxima_based_splats: int = 0            # Splats placed at local maxima

    def __post_init__(self):
        """Update derived statistics."""
        self.total_splats = len(self.positions)


class GradientGuidedPlacer:
    """Gradient-guided splat placement algorithm."""

    def __init__(self, image_size: Tuple[int, int], config: Optional[PlacementConfig] = None):
        """
        Initialize gradient-guided placer.

        Args:
            image_size: (height, width) of target image
            config: Placement configuration
        """
        self.image_height, self.image_width = image_size
        self.config = config or PlacementConfig()

        # Initialize components
        self.gradient_analyzer = GradientAnalyzer(sigma=1.0, method='sobel')
        self.prob_generator = ProbabilityMapGenerator(
            gradient_weight=self.config.gradient_weight,
            uniform_weight=self.config.uniform_weight
        )
        self.spatial_sampler = SpatialSampler()

        logger.info(f"Initialized GradientGuidedPlacer for {self.image_width}x{self.image_height} image")

    def compute_image_complexity(self, image: np.ndarray) -> np.ndarray:
        """
        Compute local image complexity for adaptive density control.

        Args:
            image: Input image (H, W, C) or (H, W)

        Returns:
            Complexity map (H, W) with values in [0, 1]
        """
        # Compute gradient magnitude
        grad_magnitude = self.gradient_analyzer.compute_gradient_magnitude(image)

        # Compute local variance (texture complexity)
        if image.ndim == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image.copy()

        # Local variance using uniform filter
        kernel_size = 5
        gray_squared = gray ** 2
        local_mean = ndimage.uniform_filter(gray, size=kernel_size)
        local_mean_squared = ndimage.uniform_filter(gray_squared, size=kernel_size)
        local_variance = local_mean_squared - local_mean ** 2

        # Combine gradient and texture measures
        complexity = 0.6 * grad_magnitude + 0.4 * local_variance

        # Smooth complexity map
        complexity = ndimage.gaussian_filter(complexity, sigma=self.config.complexity_smoothing)

        # Normalize to [0, 1]
        if np.max(complexity) > 1e-10:
            complexity = complexity / np.max(complexity)

        return complexity

    def compute_adaptive_density(self, complexity_map: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Compute adaptive splat density from complexity map.

        Args:
            complexity_map: Local complexity measure (H, W)

        Returns:
            Tuple of (density_map, total_splat_count)
        """
        H, W = complexity_map.shape
        total_pixels = H * W

        # Apply power scaling to complexity
        # Ensure no NaN values by clipping very small values
        safe_complexity = np.clip(complexity_map, 1e-10, 1.0)
        powered_complexity = np.power(safe_complexity, self.config.complexity_power)

        # Compute base density
        base_count = int(total_pixels * self.config.base_splat_density)

        # Scale density by complexity
        min_mult, max_mult = self.config.density_multiplier_range
        density_multipliers = min_mult + (max_mult - min_mult) * powered_complexity

        # Compute per-pixel splat density
        density_map = self.config.base_splat_density * density_multipliers

        # Compute total splat count from density integral
        density_sum = np.sum(density_map)
        if np.isnan(density_sum) or np.isinf(density_sum):
            logger.warning(f"Invalid density sum {density_sum}, using base count")
            total_splat_count = base_count
        else:
            total_splat_count = int(density_sum)

        # Enforce splat count limits
        total_splat_count = np.clip(total_splat_count, self.config.min_splats, self.config.max_splats)

        # Renormalize density map to match target count
        current_sum = np.sum(density_map)
        if current_sum > 1e-10:
            density_map = density_map * (total_splat_count / current_sum)

        return density_map, total_splat_count

    def detect_gradient_maxima(self, image: np.ndarray) -> List[Tuple[int, int]]:
        """
        Detect local maxima in gradient magnitude for strategic placement.

        Args:
            image: Input image

        Returns:
            List of maxima positions (y, x)
        """
        # Compute gradient magnitude
        grad_magnitude = self.gradient_analyzer.compute_gradient_magnitude(image)

        # Normalize gradient magnitude
        if np.max(grad_magnitude) > 1e-10:
            grad_magnitude = grad_magnitude / np.max(grad_magnitude)

        # Apply threshold
        thresholded = grad_magnitude > self.config.maxima_threshold

        # Find local maxima
        try:
            # Use skimage for robust peak detection
            peaks = peak_local_max(
                grad_magnitude,
                min_distance=self.config.maxima_min_distance,
                threshold_abs=self.config.maxima_threshold
            )
            maxima_positions = [(peak[0], peak[1]) for peak in peaks]
        except Exception:
            # Fallback to scipy maximum filter
            local_maxima = maximum_filter(grad_magnitude, size=self.config.maxima_min_distance) == grad_magnitude
            local_maxima = local_maxima & thresholded

            # Extract positions
            maxima_positions = list(zip(*np.where(local_maxima)))

        # Filter out border positions
        margin = self.config.border_margin
        filtered_maxima = []
        for y, x in maxima_positions:
            if (margin <= y < self.image_height - margin and
                margin <= x < self.image_width - margin):
                filtered_maxima.append((y, x))

        logger.debug(f"Detected {len(filtered_maxima)} gradient maxima")
        return filtered_maxima

    def create_placement_probability_map(self, image: np.ndarray,
                                       complexity_map: np.ndarray,
                                       density_map: np.ndarray) -> np.ndarray:
        """
        Create comprehensive placement probability map.

        Args:
            image: Input image
            complexity_map: Local complexity measure
            density_map: Adaptive density map

        Returns:
            Placement probability map (H, W)
        """
        # Base gradient-guided probability
        base_prob = self.prob_generator.create_mixed_probability_map(image, self.gradient_analyzer)

        # Weight by density map
        density_weighted_prob = base_prob * density_map

        # Add local maxima emphasis
        maxima_positions = self.detect_gradient_maxima(image)
        maxima_map = np.zeros_like(base_prob)

        # Create Gaussian blobs around maxima
        sigma = self.config.maxima_min_distance / 2.0
        for y, x in maxima_positions:
            # Create small Gaussian around each maximum
            y_indices, x_indices = np.ogrid[:self.image_height, :self.image_width]
            gaussian_blob = np.exp(-((y_indices - y)**2 + (x_indices - x)**2) / (2 * sigma**2))
            maxima_map += gaussian_blob

        # Normalize maxima map
        if np.max(maxima_map) > 1e-10:
            maxima_map = maxima_map / np.max(maxima_map)

        # Combine all probability sources
        combined_prob = (
            (1.0 - self.config.maxima_weight) * density_weighted_prob +
            self.config.maxima_weight * maxima_map
        )

        # Normalize to probability distribution
        total_prob = np.sum(combined_prob)
        if total_prob > 1e-10:
            combined_prob = combined_prob / total_prob
        else:
            # Fallback to uniform distribution
            combined_prob = np.ones_like(combined_prob) / combined_prob.size

        return combined_prob

    def place_splats(self, image: np.ndarray, target_count: Optional[int] = None) -> PlacementResult:
        """
        Perform gradient-guided splat placement.

        Args:
            image: Input image for guidance
            target_count: Optional override for splat count

        Returns:
            PlacementResult with positions and analysis
        """
        logger.info(f"Starting gradient-guided placement for {image.shape}")

        # Step 1: Compute image complexity
        complexity_map = self.compute_image_complexity(image)

        # Step 2: Compute adaptive density
        density_map, auto_splat_count = self.compute_adaptive_density(complexity_map)
        final_splat_count = target_count if target_count is not None else auto_splat_count

        logger.info(f"Target splat count: {final_splat_count} "
                   f"(auto: {auto_splat_count}, override: {target_count is not None})")

        # Step 3: Create placement probability map
        probability_map = self.create_placement_probability_map(image, complexity_map, density_map)

        # Step 4: Sample splat positions
        positions = self.spatial_sampler.sample_with_minimum_distance(
            probability_map,
            n_samples=final_splat_count,
            min_distance=self.config.min_splat_distance,
            max_attempts=1000
        )

        # Step 5: Convert to normalized coordinates
        normalized_positions = []
        for y, x in positions:
            norm_x = x / self.image_width
            norm_y = y / self.image_height
            normalized_positions.append((norm_y, norm_x))  # Note: (y, x) order for consistency

        # Step 6: Compute quality metrics
        coverage_achieved = self._compute_coverage(positions)
        distribution_uniformity = self._compute_distribution_uniformity(positions)
        gradient_alignment = self._compute_gradient_alignment(positions, image)

        # Step 7: Analyze placement statistics
        gradient_positions = self.detect_gradient_maxima(image)
        maxima_based_count = self._count_maxima_aligned_splats(positions, gradient_positions)

        # Create result
        result = PlacementResult(
            positions=positions,
            normalized_positions=normalized_positions,
            complexity_map=complexity_map,
            density_map=density_map,
            probability_map=probability_map,
            coverage_achieved=coverage_achieved,
            distribution_uniformity=distribution_uniformity,
            gradient_alignment=gradient_alignment,
            total_splats=len(positions),
            gradient_guided_splats=int(len(positions) * self.config.gradient_weight),
            uniform_coverage_splats=int(len(positions) * self.config.uniform_weight),
            maxima_based_splats=maxima_based_count
        )

        logger.info(f"Placement complete: {result.total_splats} splats, "
                   f"coverage: {result.coverage_achieved:.3f}, "
                   f"uniformity: {result.distribution_uniformity:.3f}")

        return result

    def validate_placement_quality(self, result: PlacementResult) -> Dict[str, Any]:
        """
        Validate placement quality against configuration requirements.

        Args:
            result: Placement result to validate

        Returns:
            Validation report with quality assessment
        """
        validation = {
            'passed': True,
            'issues': [],
            'metrics': {},
            'recommendations': []
        }

        # Check coverage requirement
        if result.coverage_achieved < self.config.coverage_target:
            validation['passed'] = False
            validation['issues'].append(
                f"Coverage {result.coverage_achieved:.3f} below target {self.config.coverage_target:.3f}"
            )
            validation['recommendations'].append("Increase base_splat_density or max_splats")

        # Check distribution uniformity
        if result.distribution_uniformity < self.config.distribution_uniformity_threshold:
            validation['passed'] = False
            validation['issues'].append(
                f"Distribution uniformity {result.distribution_uniformity:.3f} below threshold "
                f"{self.config.distribution_uniformity_threshold:.3f}"
            )
            validation['recommendations'].append("Increase uniform_weight or reduce min_splat_distance")

        # Check splat count bounds
        if result.total_splats < self.config.min_splats:
            validation['passed'] = False
            validation['issues'].append(f"Splat count {result.total_splats} below minimum {self.config.min_splats}")

        if result.total_splats > self.config.max_splats:
            validation['passed'] = False
            validation['issues'].append(f"Splat count {result.total_splats} above maximum {self.config.max_splats}")

        # Collect metrics
        validation['metrics'] = {
            'coverage_achieved': result.coverage_achieved,
            'distribution_uniformity': result.distribution_uniformity,
            'gradient_alignment': result.gradient_alignment,
            'total_splats': result.total_splats,
            'complexity_mean': float(np.mean(result.complexity_map)),
            'density_variance': float(np.var(result.density_map))
        }

        return validation

    def _compute_coverage(self, positions: List[Tuple[int, int]]) -> float:
        """Compute spatial coverage ratio."""
        if not positions:
            return 0.0

        # Create coverage map with splat influence
        coverage_map = np.zeros((self.image_height, self.image_width))
        influence_radius = max(2, int(self.config.min_splat_distance))

        for y, x in positions:
            # Add Gaussian influence around each splat
            y_indices, x_indices = np.ogrid[:self.image_height, :self.image_width]
            influence = np.exp(-((y_indices - y)**2 + (x_indices - x)**2) / (2 * influence_radius**2))
            coverage_map = np.maximum(coverage_map, influence)

        # Compute coverage as fraction of pixels with significant influence
        coverage_ratio = np.mean(coverage_map > 0.1)  # 10% threshold
        return coverage_ratio

    def _compute_distribution_uniformity(self, positions: List[Tuple[int, int]]) -> float:
        """Compute spatial distribution uniformity using grid-based analysis."""
        if not positions:
            return 0.0

        # Divide image into grid cells
        grid_size = 8  # 8x8 grid
        cell_height = self.image_height // grid_size
        cell_width = self.image_width // grid_size

        # Count splats per cell
        cell_counts = np.zeros((grid_size, grid_size))
        for y, x in positions:
            cell_y = min(y // cell_height, grid_size - 1)
            cell_x = min(x // cell_width, grid_size - 1)
            cell_counts[cell_y, cell_x] += 1

        # Compute uniformity as inverse of coefficient of variation
        mean_count = np.mean(cell_counts)
        if mean_count > 1e-10:
            std_count = np.std(cell_counts)
            coefficient_of_variation = std_count / mean_count
            uniformity = 1.0 / (1.0 + coefficient_of_variation)
        else:
            uniformity = 0.0

        return uniformity

    def _compute_gradient_alignment(self, positions: List[Tuple[int, int]], image: np.ndarray) -> float:
        """Compute alignment of splats with gradient features."""
        if not positions:
            return 0.0

        grad_magnitude = self.gradient_analyzer.compute_gradient_magnitude(image)

        # Evaluate gradient magnitude at splat positions
        gradient_values = []
        for y, x in positions:
            if 0 <= y < self.image_height and 0 <= x < self.image_width:
                gradient_values.append(grad_magnitude[y, x])

        if not gradient_values:
            return 0.0

        # Normalize by maximum gradient
        max_gradient = np.max(grad_magnitude)
        if max_gradient > 1e-10:
            normalized_values = np.array(gradient_values) / max_gradient
            alignment = np.mean(normalized_values)
        else:
            alignment = 0.0

        return alignment

    def _count_maxima_aligned_splats(self, positions: List[Tuple[int, int]],
                                   maxima_positions: List[Tuple[int, int]]) -> int:
        """Count splats that are close to gradient maxima."""
        if not positions or not maxima_positions:
            return 0

        aligned_count = 0
        threshold_distance = self.config.maxima_min_distance

        for splat_y, splat_x in positions:
            for max_y, max_x in maxima_positions:
                distance = np.sqrt((splat_y - max_y)**2 + (splat_x - max_x)**2)
                if distance <= threshold_distance:
                    aligned_count += 1
                    break  # Count each splat only once

        return aligned_count

    def get_placement_statistics(self, result: PlacementResult) -> Dict[str, Any]:
        """Get comprehensive placement statistics."""
        return {
            'total_splats': result.total_splats,
            'coverage_achieved': result.coverage_achieved,
            'distribution_uniformity': result.distribution_uniformity,
            'gradient_alignment': result.gradient_alignment,
            'complexity_statistics': {
                'mean': float(np.mean(result.complexity_map)),
                'std': float(np.std(result.complexity_map)),
                'max': float(np.max(result.complexity_map)),
                'high_complexity_fraction': float(np.mean(result.complexity_map > 0.7))
            },
            'density_statistics': {
                'mean': float(np.mean(result.density_map)),
                'std': float(np.std(result.density_map)),
                'max': float(np.max(result.density_map)),
                'density_range': float(np.max(result.density_map) - np.min(result.density_map))
            },
            'spatial_distribution': {
                'gradient_guided_fraction': result.gradient_guided_splats / result.total_splats,
                'uniform_coverage_fraction': result.uniform_coverage_splats / result.total_splats,
                'maxima_aligned_fraction': result.maxima_based_splats / result.total_splats
            }
        }


# Convenience functions
def create_gradient_guided_placer(image_size: Tuple[int, int],
                                config: Optional[Dict[str, Any]] = None) -> GradientGuidedPlacer:
    """Convenience function to create gradient-guided placer."""
    if config:
        placement_config = PlacementConfig(**config)
    else:
        placement_config = PlacementConfig()

    return GradientGuidedPlacer(image_size, placement_config)


def place_adaptive_splats(image: np.ndarray,
                         target_count: Optional[int] = None,
                         config: Optional[Dict[str, Any]] = None) -> PlacementResult:
    """Convenience function for adaptive splat placement."""
    placer = create_gradient_guided_placer(image.shape[:2], config)
    return placer.place_splats(image, target_count)