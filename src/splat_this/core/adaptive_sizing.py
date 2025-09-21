#!/usr/bin/env python3
"""Adaptive sizing strategy for content-aware Gaussian splat initialization."""

import numpy as np
from typing import Tuple, Optional, Dict, Any, List
from dataclasses import dataclass
import logging
from scipy import ndimage

logger = logging.getLogger(__name__)


@dataclass
class SizingConfig:
    """Configuration for adaptive sizing strategy."""
    base_size: float = 0.02                    # Base splat size (fraction of image diagonal)
    size_range: Tuple[float, float] = (0.005, 0.08)  # Min/max size limits
    complexity_sensitivity: float = 0.8        # How much complexity affects size (0-1)
    variance_weight: float = 0.3              # Weight for local variance in sizing
    edge_density_weight: float = 0.4          # Weight for edge density in sizing
    anisotropy_influence: float = 0.3         # How much anisotropy affects sizing
    smoothing_sigma: float = 1.5              # Gaussian smoothing for size maps
    size_quantization: int = 20               # Number of discrete size levels
    adaptive_range: bool = True               # Whether to adapt size range to content
    normalization: str = 'percentile'        # 'percentile', 'minmax', or 'adaptive'

    def __post_init__(self):
        """Validate configuration parameters."""
        if not 0 < self.base_size < 1:
            raise ValueError("base_size must be in (0, 1)")
        if not (0 < self.size_range[0] < self.size_range[1] < 1):
            raise ValueError("size_range must be (min, max) with 0 < min < max < 1")
        if not 0 <= self.complexity_sensitivity <= 1:
            raise ValueError("complexity_sensitivity must be in [0, 1]")
        if not 0 <= self.variance_weight <= 1:
            raise ValueError("variance_weight must be in [0, 1]")
        if not 0 <= self.edge_density_weight <= 1:
            raise ValueError("edge_density_weight must be in [0, 1]")
        if not 0 <= self.anisotropy_influence <= 1:
            raise ValueError("anisotropy_influence must be in [0, 1]")
        if self.smoothing_sigma < 0:
            raise ValueError("smoothing_sigma must be non-negative")
        if self.size_quantization < 2:
            raise ValueError("size_quantization must be at least 2")


@dataclass
class SizingResult:
    """Results from adaptive sizing analysis."""
    size_map: np.ndarray           # Adaptive size at each pixel
    complexity_map: np.ndarray     # Local complexity measure
    variance_map: np.ndarray       # Local variance measure
    edge_density_map: np.ndarray   # Local edge density
    size_distribution: np.ndarray  # Histogram of size values
    statistics: Dict[str, float]   # Summary statistics


@dataclass
class SplatSizeAllocation:
    """Size allocation for individual splats."""
    positions: np.ndarray      # Splat positions (N, 2)
    sizes: np.ndarray         # Allocated sizes (N,)
    complexity_scores: np.ndarray  # Complexity at each position (N,)
    size_rationale: List[str] # Rationale for each size decision


class AdaptiveSizer:
    """Compute adaptive sizes for Gaussian splats based on image content."""

    def __init__(self, image_shape: Tuple[int, int], config: Optional[SizingConfig] = None):
        """Initialize adaptive sizer.

        Args:
            image_shape: Image dimensions (height, width)
            config: Sizing configuration, defaults to SizingConfig()
        """
        self.image_shape = image_shape
        self.config = config or SizingConfig()
        self.image_diagonal = np.sqrt(image_shape[0]**2 + image_shape[1]**2)

    def compute_adaptive_sizes(self, image: np.ndarray,
                              complexity_map: Optional[np.ndarray] = None,
                              anisotropy_map: Optional[np.ndarray] = None) -> SizingResult:
        """Compute adaptive size map for the entire image.

        Args:
            image: Input image (H, W, C) or (H, W) in range [0, 1]
            complexity_map: Precomputed complexity map (optional)
            anisotropy_map: Precomputed anisotropy map (optional)

        Returns:
            SizingResult with adaptive size maps and statistics
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image.copy()

        # Compute complexity if not provided
        if complexity_map is None:
            complexity_map = self._compute_local_complexity(gray)

        # Compute local variance
        variance_map = self._compute_local_variance(gray)

        # Compute edge density
        edge_density_map = self._compute_edge_density(gray)

        # Combine factors into size influence map
        size_influence = self._combine_sizing_factors(
            complexity_map, variance_map, edge_density_map, anisotropy_map
        )

        # Convert influence to actual sizes
        size_map = self._compute_size_map(size_influence)

        # Apply smoothing if requested
        if self.config.smoothing_sigma > 0:
            size_map = ndimage.gaussian_filter(size_map, sigma=self.config.smoothing_sigma)

        # Apply size constraints
        size_map = np.clip(size_map, self.config.size_range[0], self.config.size_range[1])

        # Quantize sizes if requested
        if self.config.size_quantization > 2:
            size_map = self._quantize_sizes(size_map)

        # Compute size distribution and statistics
        size_distribution = self._compute_size_distribution(size_map)
        statistics = self._compute_sizing_statistics(size_map, complexity_map, variance_map)

        return SizingResult(
            size_map=size_map,
            complexity_map=complexity_map,
            variance_map=variance_map,
            edge_density_map=edge_density_map,
            size_distribution=size_distribution,
            statistics=statistics
        )

    def _compute_local_complexity(self, image: np.ndarray) -> np.ndarray:
        """Compute local complexity measure."""
        # Gradient magnitude
        grad_x = ndimage.sobel(image, axis=1)
        grad_y = ndimage.sobel(image, axis=0)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

        # Local standard deviation (texture complexity)
        kernel_size = 5
        local_mean = ndimage.uniform_filter(image, size=kernel_size)
        local_variance = ndimage.uniform_filter(image**2, size=kernel_size) - local_mean**2
        local_std = np.sqrt(np.maximum(local_variance, 0))

        # Combine gradient and texture measures
        complexity = 0.6 * gradient_magnitude + 0.4 * local_std

        # Normalize to [0, 1]
        if np.max(complexity) > 0:
            complexity = complexity / np.max(complexity)

        return complexity

    def _compute_local_variance(self, image: np.ndarray) -> np.ndarray:
        """Compute local variance using sliding window."""
        window_size = 7

        # Compute local mean and variance
        local_mean = ndimage.uniform_filter(image, size=window_size)
        local_mean_sq = ndimage.uniform_filter(image**2, size=window_size)
        local_variance = local_mean_sq - local_mean**2

        # Ensure non-negative variance
        local_variance = np.maximum(local_variance, 0)

        # Normalize to [0, 1]
        if np.max(local_variance) > 0:
            local_variance = local_variance / np.max(local_variance)

        return local_variance

    def _compute_edge_density(self, image: np.ndarray) -> np.ndarray:
        """Compute local edge density."""
        # Canny-like edge detection
        grad_x = ndimage.sobel(image, axis=1)
        grad_y = ndimage.sobel(image, axis=0)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

        # Threshold for edge detection
        edge_threshold = np.percentile(gradient_magnitude, 75)
        edges = gradient_magnitude > edge_threshold

        # Compute local edge density using convolution
        kernel_size = 9
        kernel = np.ones((kernel_size, kernel_size)) / (kernel_size**2)
        edge_density = ndimage.convolve(edges.astype(float), kernel)

        return edge_density

    def _combine_sizing_factors(self, complexity_map: np.ndarray,
                               variance_map: np.ndarray,
                               edge_density_map: np.ndarray,
                               anisotropy_map: Optional[np.ndarray] = None) -> np.ndarray:
        """Combine different factors into a unified size influence map."""
        # Start with complexity as the primary factor
        influence = complexity_map * self.config.complexity_sensitivity

        # Add variance contribution
        influence += variance_map * self.config.variance_weight

        # Add edge density contribution
        influence += edge_density_map * self.config.edge_density_weight

        # Add anisotropy influence if available
        if anisotropy_map is not None:
            influence += anisotropy_map * self.config.anisotropy_influence

        # Normalize weights
        total_weight = (self.config.complexity_sensitivity +
                       self.config.variance_weight +
                       self.config.edge_density_weight)
        if anisotropy_map is not None:
            total_weight += self.config.anisotropy_influence

        if total_weight > 0:
            influence = influence / total_weight

        # Ensure values are in [0, 1]
        influence = np.clip(influence, 0, 1)

        return influence

    def _compute_size_map(self, influence_map: np.ndarray) -> np.ndarray:
        """Convert influence map to actual size values."""
        # Invert influence: high complexity/detail -> smaller sizes
        size_factor = 1.0 - influence_map

        # Apply adaptive range if enabled
        if self.config.adaptive_range:
            # Adjust size range based on content characteristics
            content_complexity = np.mean(influence_map)
            if content_complexity > 0.7:  # High detail image
                size_range = (self.config.size_range[0], self.config.size_range[1] * 0.8)
            elif content_complexity < 0.3:  # Low detail image
                size_range = (self.config.size_range[0] * 1.2, self.config.size_range[1])
            else:
                size_range = self.config.size_range
        else:
            size_range = self.config.size_range

        # Map size factor to actual sizes
        size_min, size_max = size_range
        size_map = size_min + size_factor * (size_max - size_min)

        # Apply normalization
        if self.config.normalization == 'percentile':
            # Use percentiles to handle outliers
            p5, p95 = np.percentile(size_map, [5, 95])
            size_map = np.clip(size_map, p5, p95)
        elif self.config.normalization == 'minmax':
            # Standard min-max normalization
            size_map = (size_map - np.min(size_map)) / (np.max(size_map) - np.min(size_map))
            size_map = size_min + size_map * (size_max - size_min)
        # 'adaptive' normalization keeps the computed values

        return size_map

    def _quantize_sizes(self, size_map: np.ndarray) -> np.ndarray:
        """Quantize sizes to discrete levels."""
        size_min, size_max = self.config.size_range
        size_levels = np.linspace(size_min, size_max, self.config.size_quantization)

        # Find closest quantization level for each pixel
        quantized = np.zeros_like(size_map)
        for i in range(size_map.shape[0]):
            for j in range(size_map.shape[1]):
                distances = np.abs(size_levels - size_map[i, j])
                closest_idx = np.argmin(distances)
                quantized[i, j] = size_levels[closest_idx]

        return quantized

    def _compute_size_distribution(self, size_map: np.ndarray) -> np.ndarray:
        """Compute histogram of size values."""
        size_min, size_max = self.config.size_range
        bins = np.linspace(size_min, size_max, 50)
        histogram, _ = np.histogram(size_map.flatten(), bins=bins)
        return histogram

    def _compute_sizing_statistics(self, size_map: np.ndarray,
                                  complexity_map: np.ndarray,
                                  variance_map: np.ndarray) -> Dict[str, float]:
        """Compute summary statistics for sizing analysis."""
        flat_sizes = size_map.flatten()
        flat_complexity = complexity_map.flatten()

        stats = {
            'mean_size': np.mean(flat_sizes),
            'std_size': np.std(flat_sizes),
            'min_size': np.min(flat_sizes),
            'max_size': np.max(flat_sizes),
            'size_range': np.max(flat_sizes) - np.min(flat_sizes),
            'size_diversity': np.std(flat_sizes) / np.mean(flat_sizes),  # Coefficient of variation
            'complexity_correlation': np.corrcoef(flat_complexity, 1.0 - flat_sizes)[0, 1],
            'small_size_fraction': np.mean(flat_sizes < np.percentile(flat_sizes, 25)),
            'large_size_fraction': np.mean(flat_sizes > np.percentile(flat_sizes, 75)),
            'size_entropy': self._compute_entropy(flat_sizes)
        }

        # Handle NaN values
        for key, value in stats.items():
            if np.isnan(value):
                stats[key] = 0.0

        return stats

    def _compute_entropy(self, values: np.ndarray) -> float:
        """Compute entropy of size distribution."""
        # Discretize values into bins
        bins = 20
        histogram, _ = np.histogram(values, bins=bins)

        # Normalize to probabilities
        probabilities = histogram / np.sum(histogram)

        # Remove zero probabilities to avoid log(0)
        probabilities = probabilities[probabilities > 0]

        # Compute Shannon entropy
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy

    def allocate_splat_sizes(self, positions: np.ndarray,
                            image: np.ndarray,
                            complexity_map: Optional[np.ndarray] = None) -> SplatSizeAllocation:
        """Allocate sizes for specific splat positions.

        Args:
            positions: Splat positions as (N, 2) array of (y, x) coordinates
            image: Input image (H, W, C) or (H, W)
            complexity_map: Precomputed complexity map (optional)

        Returns:
            SplatSizeAllocation with sizes and rationale for each splat
        """
        # Compute sizing maps
        sizing_result = self.compute_adaptive_sizes(image, complexity_map)

        # Extract positions and clip to image bounds
        positions = np.round(positions).astype(int)
        h, w = image.shape[:2]
        positions[:, 0] = np.clip(positions[:, 0], 0, h - 1)
        positions[:, 1] = np.clip(positions[:, 1], 0, w - 1)

        # Extract sizes at splat positions
        sizes = sizing_result.size_map[positions[:, 0], positions[:, 1]]
        complexity_scores = sizing_result.complexity_map[positions[:, 0], positions[:, 1]]

        # Generate rationale for each size decision
        size_rationale = []
        for i, (pos, size, complexity) in enumerate(zip(positions, sizes, complexity_scores)):
            if complexity > 0.7:
                rationale = "Small size: High complexity region"
            elif complexity > 0.4:
                rationale = "Medium size: Moderate complexity"
            elif complexity > 0.1:
                rationale = "Large size: Low complexity region"
            else:
                rationale = "Large size: Uniform region"

            size_rationale.append(rationale)

        return SplatSizeAllocation(
            positions=positions,
            sizes=sizes,
            complexity_scores=complexity_scores,
            size_rationale=size_rationale
        )

    def validate_size_distribution(self, sizing_result: SizingResult) -> Dict[str, Any]:
        """Validate that size distribution meets quality criteria.

        Args:
            sizing_result: Result from compute_adaptive_sizes

        Returns:
            Dictionary with validation results
        """
        stats = sizing_result.statistics
        issues = []
        recommendations = []

        # Check size diversity
        if stats['size_diversity'] < 0.2:
            issues.append("Low size diversity - sizes too uniform")
            recommendations.append("Increase complexity_sensitivity or variance_weight")

        if stats['size_diversity'] > 0.8:
            issues.append("Excessive size diversity - sizes too variable")
            recommendations.append("Increase smoothing_sigma or reduce sensitivity parameters")

        # Check size range utilization
        effective_range = (stats['max_size'] - stats['min_size']) / self.config.base_size
        if effective_range < 1.5:
            issues.append("Poor size range utilization")
            recommendations.append("Increase size_range or adjust complexity_sensitivity")

        # Check complexity correlation
        if abs(stats['complexity_correlation']) < 0.3:
            issues.append("Weak correlation between complexity and size")
            recommendations.append("Adjust complexity_sensitivity or normalization method")

        # Check for degenerate sizes
        if stats['min_size'] < self.config.size_range[0] * 1.1:
            issues.append("Sizes too close to minimum limit")
            recommendations.append("Increase minimum size or adjust size computation")

        if stats['max_size'] > self.config.size_range[1] * 0.9:
            issues.append("Sizes too close to maximum limit")
            recommendations.append("Decrease maximum size or adjust size computation")

        return {
            'passed': len(issues) == 0,
            'issues': issues,
            'recommendations': recommendations,
            'statistics': stats,
            'size_quality_score': self._compute_quality_score(stats)
        }

    def _compute_quality_score(self, stats: Dict[str, float]) -> float:
        """Compute overall quality score for size distribution."""
        # Diversity score (target around 0.4-0.6)
        diversity_score = 1.0 - abs(stats['size_diversity'] - 0.5) * 2
        diversity_score = max(0, diversity_score)

        # Correlation score (higher absolute correlation is better)
        correlation_score = abs(stats['complexity_correlation'])

        # Range utilization score
        range_utilization = stats['size_range'] / (self.config.size_range[1] - self.config.size_range[0])
        range_score = min(1.0, range_utilization)

        # Entropy score (normalized by max possible entropy)
        max_entropy = np.log2(20)  # Assuming 20 bins
        entropy_score = stats['size_entropy'] / max_entropy

        # Weighted combination
        quality_score = (0.3 * diversity_score +
                        0.3 * correlation_score +
                        0.2 * range_score +
                        0.2 * entropy_score)

        return quality_score


def compute_adaptive_sizes(image: np.ndarray,
                          base_size: float = 0.02,
                          complexity_sensitivity: float = 0.8) -> SizingResult:
    """Convenience function to compute adaptive sizes with default settings.

    Args:
        image: Input image (H, W, C) or (H, W)
        base_size: Base splat size (fraction of image diagonal)
        complexity_sensitivity: How much complexity affects size

    Returns:
        SizingResult with adaptive size analysis
    """
    config = SizingConfig(
        base_size=base_size,
        complexity_sensitivity=complexity_sensitivity
    )
    sizer = AdaptiveSizer(image.shape[:2], config)
    return sizer.compute_adaptive_sizes(image)


def allocate_splat_sizes(positions: np.ndarray, image: np.ndarray) -> SplatSizeAllocation:
    """Convenience function to allocate sizes for splat positions.

    Args:
        positions: Splat positions as (N, 2) array
        image: Input image (H, W, C) or (H, W)

    Returns:
        SplatSizeAllocation with sizes for each position
    """
    sizer = AdaptiveSizer(image.shape[:2])
    return sizer.allocate_splat_sizes(positions, image)