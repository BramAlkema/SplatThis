#!/usr/bin/env python3
"""Color sampling and validation for adaptive Gaussian splat initialization."""

import numpy as np
from typing import Tuple, Optional, Dict, Any, List, Union
from dataclasses import dataclass
import logging
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator

logger = logging.getLogger(__name__)


@dataclass
class ColorSamplingConfig:
    """Configuration for color sampling and validation."""
    interpolation_method: str = 'bilinear'      # 'nearest', 'bilinear', 'bicubic'
    outlier_detection: bool = True              # Enable outlier detection
    outlier_threshold: float = 3.0              # Standard deviations for outlier detection
    color_space: str = 'RGB'                   # 'RGB', 'LAB', 'HSV', 'YUV'
    gamma_correction: bool = True               # Apply gamma correction
    gamma_value: float = 2.2                   # Gamma correction value
    normalization: str = 'minmax'              # 'minmax', 'zscore', 'none'
    smoothing_radius: float = 0.0              # Gaussian smoothing radius (0 = no smoothing)
    boundary_handling: str = 'clamp'           # 'clamp', 'wrap', 'mirror'
    multi_channel_support: bool = True         # Support for >3 channel images
    validation_tolerance: float = 0.1          # Tolerance for color accuracy validation

    def __post_init__(self):
        """Validate configuration parameters."""
        valid_interpolations = ['nearest', 'bilinear', 'bicubic']
        if self.interpolation_method not in valid_interpolations:
            raise ValueError(f"interpolation_method must be one of {valid_interpolations}")

        if not 0 < self.outlier_threshold < 10:
            raise ValueError("outlier_threshold must be in (0, 10)")

        valid_color_spaces = ['RGB', 'LAB', 'HSV', 'YUV']
        if self.color_space not in valid_color_spaces:
            raise ValueError(f"color_space must be one of {valid_color_spaces}")

        if not 0.1 < self.gamma_value < 5.0:
            raise ValueError("gamma_value must be in (0.1, 5.0)")

        valid_normalizations = ['minmax', 'zscore', 'none']
        if self.normalization not in valid_normalizations:
            raise ValueError(f"normalization must be one of {valid_normalizations}")

        if self.smoothing_radius < 0:
            raise ValueError("smoothing_radius must be non-negative")

        valid_boundary = ['clamp', 'wrap', 'mirror']
        if self.boundary_handling not in valid_boundary:
            raise ValueError(f"boundary_handling must be one of {valid_boundary}")

        if not 0 < self.validation_tolerance < 1:
            raise ValueError("validation_tolerance must be in (0, 1)")


@dataclass
class ColorSample:
    """Single color sample with metadata."""
    position: np.ndarray      # Position (y, x) coordinates
    color: np.ndarray        # Sampled color values
    confidence: float        # Sampling confidence [0, 1]
    is_outlier: bool        # Whether sample is detected as outlier
    interpolated: bool      # Whether position required interpolation
    original_color: np.ndarray  # Color before any transformations


@dataclass
class ColorSamplingResult:
    """Results from color sampling operation."""
    samples: List[ColorSample]      # Individual color samples
    positions: np.ndarray           # All sample positions (N, 2)
    colors: np.ndarray             # All sampled colors (N, C)
    outlier_mask: np.ndarray       # Boolean mask for outliers (N,)
    statistics: Dict[str, Any]     # Sampling statistics
    validation_results: Dict[str, Any]  # Color accuracy validation


class ColorSampler:
    """Sample colors from images at specified positions with validation."""

    def __init__(self, config: Optional[ColorSamplingConfig] = None):
        """Initialize color sampler.

        Args:
            config: Sampling configuration, defaults to ColorSamplingConfig()
        """
        self.config = config or ColorSamplingConfig()

    def sample_colors(self, image: np.ndarray,
                     positions: np.ndarray) -> ColorSamplingResult:
        """Sample colors from image at specified positions.

        Args:
            image: Input image (H, W, C) or (H, W) in range [0, 1]
            positions: Sample positions as (N, 2) array of (y, x) coordinates

        Returns:
            ColorSamplingResult with sampled colors and validation
        """
        # Ensure image is in correct format
        if len(image.shape) == 2:
            image = image[:, :, np.newaxis]  # Add channel dimension

        original_image = image.copy()

        # Apply preprocessing
        processed_image = self._preprocess_image(image)

        # Perform color sampling
        sampled_colors, interpolation_mask = self._sample_at_positions(
            processed_image, positions
        )

        # Apply color space conversion if needed
        if self.config.color_space != 'RGB':
            sampled_colors = self._convert_color_space(sampled_colors, 'RGB', self.config.color_space)

        # Detect outliers
        outlier_mask = np.zeros(len(positions), dtype=bool)
        if self.config.outlier_detection:
            outlier_mask = self._detect_outliers(sampled_colors)

        # Create individual samples
        samples = []
        for i in range(len(positions)):
            # Sample original colors for comparison
            orig_colors, _ = self._sample_at_positions(original_image, positions[i:i+1])

            sample = ColorSample(
                position=positions[i],
                color=sampled_colors[i],
                confidence=self._compute_sampling_confidence(
                    processed_image, positions[i], interpolation_mask[i]
                ),
                is_outlier=outlier_mask[i],
                interpolated=interpolation_mask[i],
                original_color=orig_colors[0]
            )
            samples.append(sample)

        # Compute statistics
        statistics = self._compute_color_statistics(sampled_colors, outlier_mask)

        # Validate color accuracy
        validation_results = self._validate_color_accuracy(
            original_image, positions, sampled_colors
        )

        return ColorSamplingResult(
            samples=samples,
            positions=positions,
            colors=sampled_colors,
            outlier_mask=outlier_mask,
            statistics=statistics,
            validation_results=validation_results
        )

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Apply preprocessing to image before sampling."""
        processed = image.copy()

        # Apply gamma correction
        if self.config.gamma_correction:
            processed = np.power(processed, 1.0 / self.config.gamma_value)

        # Apply smoothing if requested
        if self.config.smoothing_radius > 0:
            for c in range(processed.shape[2]):
                processed[:, :, c] = ndimage.gaussian_filter(
                    processed[:, :, c], sigma=self.config.smoothing_radius
                )

        # Apply normalization
        if self.config.normalization == 'minmax':
            for c in range(processed.shape[2]):
                channel = processed[:, :, c]
                min_val, max_val = np.min(channel), np.max(channel)
                if max_val > min_val:
                    processed[:, :, c] = (channel - min_val) / (max_val - min_val)
        elif self.config.normalization == 'zscore':
            for c in range(processed.shape[2]):
                channel = processed[:, :, c]
                mean_val, std_val = np.mean(channel), np.std(channel)
                if std_val > 0:
                    processed[:, :, c] = (channel - mean_val) / std_val

        return processed

    def _sample_at_positions(self, image: np.ndarray,
                           positions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Sample colors at specified positions with interpolation.

        Args:
            image: Input image (H, W, C)
            positions: Positions to sample (N, 2)

        Returns:
            Tuple of (sampled_colors, interpolation_mask)
        """
        h, w, c = image.shape
        n_positions = len(positions)
        sampled_colors = np.zeros((n_positions, c))
        interpolation_mask = np.zeros(n_positions, dtype=bool)

        for i, pos in enumerate(positions):
            y, x = pos

            # Handle boundary conditions
            if self.config.boundary_handling == 'clamp':
                y = np.clip(y, 0, h - 1)
                x = np.clip(x, 0, w - 1)
            elif self.config.boundary_handling == 'wrap':
                y = y % h
                x = x % w
            elif self.config.boundary_handling == 'mirror':
                y = self._mirror_coordinate(y, h)
                x = self._mirror_coordinate(x, w)

            if self.config.interpolation_method == 'nearest':
                # Nearest neighbor sampling
                yi, xi = int(round(y)), int(round(x))
                yi = np.clip(yi, 0, h - 1)
                xi = np.clip(xi, 0, w - 1)
                sampled_colors[i] = image[yi, xi]
                interpolation_mask[i] = False

            elif self.config.interpolation_method == 'bilinear':
                # Bilinear interpolation
                sampled_colors[i], interpolation_mask[i] = self._bilinear_sample(
                    image, y, x
                )

            elif self.config.interpolation_method == 'bicubic':
                # Bicubic interpolation (using scipy)
                sampled_colors[i], interpolation_mask[i] = self._bicubic_sample(
                    image, y, x
                )

        return sampled_colors, interpolation_mask

    def _bilinear_sample(self, image: np.ndarray, y: float, x: float) -> Tuple[np.ndarray, bool]:
        """Perform bilinear interpolation at a position."""
        h, w = image.shape[:2]

        # Check if position is exactly on a pixel
        if abs(y - round(y)) < 1e-6 and abs(x - round(x)) < 1e-6:
            yi, xi = int(round(y)), int(round(x))
            yi = np.clip(yi, 0, h - 1)
            xi = np.clip(xi, 0, w - 1)
            return image[yi, xi], False

        # Get integer coordinates
        y0, x0 = int(np.floor(y)), int(np.floor(x))
        y1, x1 = y0 + 1, x0 + 1

        # Clamp to image bounds
        y0 = np.clip(y0, 0, h - 1)
        y1 = np.clip(y1, 0, h - 1)
        x0 = np.clip(x0, 0, w - 1)
        x1 = np.clip(x1, 0, w - 1)

        # Compute interpolation weights
        wy = y - y0
        wx = x - x0

        # Bilinear interpolation
        color = (image[y0, x0] * (1 - wx) * (1 - wy) +
                image[y0, x1] * wx * (1 - wy) +
                image[y1, x0] * (1 - wx) * wy +
                image[y1, x1] * wx * wy)

        return color, True

    def _bicubic_sample(self, image: np.ndarray, y: float, x: float) -> Tuple[np.ndarray, bool]:
        """Perform bicubic interpolation at a position."""
        h, w, c = image.shape

        # Check if position is exactly on a pixel
        if abs(y - round(y)) < 1e-6 and abs(x - round(x)) < 1e-6:
            yi, xi = int(round(y)), int(round(x))
            yi = np.clip(yi, 0, h - 1)
            xi = np.clip(xi, 0, w - 1)
            return image[yi, xi], False

        # Use scipy interpolation for bicubic
        color = np.zeros(c)
        for ch in range(c):
            # Create interpolator for this channel
            y_coords = np.arange(h)
            x_coords = np.arange(w)
            interpolator = RegularGridInterpolator(
                (y_coords, x_coords), image[:, :, ch],
                method='cubic', bounds_error=False, fill_value=0.0
            )

            # Sample at position
            color[ch] = float(interpolator([y, x]))

        return color, True

    def _mirror_coordinate(self, coord: float, size: int) -> float:
        """Apply mirror boundary condition to coordinate."""
        if coord < 0:
            return -coord
        elif coord >= size:
            return 2 * (size - 1) - coord
        else:
            return coord

    def _detect_outliers(self, colors: np.ndarray) -> np.ndarray:
        """Detect color outliers using statistical methods."""
        if len(colors) < 3:
            return np.zeros(len(colors), dtype=bool)

        outliers = np.zeros(len(colors), dtype=bool)

        # Per-channel outlier detection
        for c in range(colors.shape[1]):
            channel_values = colors[:, c]
            mean_val = np.mean(channel_values)
            std_val = np.std(channel_values)

            if std_val > 0:
                z_scores = np.abs((channel_values - mean_val) / std_val)
                channel_outliers = z_scores > self.config.outlier_threshold
                outliers |= channel_outliers

        return outliers

    def _compute_sampling_confidence(self, image: np.ndarray, position: np.ndarray,
                                   interpolated: bool) -> float:
        """Compute confidence score for a color sample."""
        y, x = position
        h, w = image.shape[:2]

        # Base confidence
        confidence = 1.0

        # Reduce confidence for interpolated samples
        if interpolated:
            confidence *= 0.9

        # Reduce confidence near boundaries
        margin = 2.0
        if (y < margin or y > h - margin - 1 or
            x < margin or x > w - margin - 1):
            boundary_factor = min(
                y / margin, (h - 1 - y) / margin,
                x / margin, (w - 1 - x) / margin, 1.0
            )
            confidence *= max(0.1, boundary_factor)

        # Consider local variance (lower variance = higher confidence)
        if h > 3 and w > 3:
            yi, xi = int(np.clip(y, 1, h - 2)), int(np.clip(x, 1, w - 2))
            local_patch = image[yi-1:yi+2, xi-1:xi+2]
            local_variance = np.var(local_patch)
            variance_factor = np.exp(-local_variance * 5)  # Exponential decay
            confidence *= variance_factor

        return np.clip(confidence, 0.0, 1.0)

    def _convert_color_space(self, colors: np.ndarray,
                           from_space: str, to_space: str) -> np.ndarray:
        """Convert colors between color spaces."""
        if from_space == to_space:
            return colors

        # Currently only supporting RGB as input
        if from_space != 'RGB':
            raise NotImplementedError(f"Conversion from {from_space} not implemented")

        converted = colors.copy()

        if to_space == 'LAB':
            # RGB to LAB conversion (simplified)
            # This is a basic implementation - in practice you'd use colorspacious or similar
            converted = self._rgb_to_lab(converted)
        elif to_space == 'HSV':
            converted = self._rgb_to_hsv(converted)
        elif to_space == 'YUV':
            converted = self._rgb_to_yuv(converted)

        return converted

    def _rgb_to_lab(self, rgb: np.ndarray) -> np.ndarray:
        """Convert RGB to LAB color space (simplified)."""
        # Simplified RGB to LAB conversion
        # In practice, use proper color conversion libraries
        lab = np.zeros_like(rgb)

        # Simple luminance calculation
        lab[:, 0] = 0.299 * rgb[:, 0] + 0.587 * rgb[:, 1] + 0.114 * rgb[:, 2]  # L
        lab[:, 1] = rgb[:, 0] - lab[:, 0]  # A (simplified)
        lab[:, 2] = rgb[:, 2] - lab[:, 0]  # B (simplified)

        return lab

    def _rgb_to_hsv(self, rgb: np.ndarray) -> np.ndarray:
        """Convert RGB to HSV color space."""
        hsv = np.zeros_like(rgb)

        for i in range(len(rgb)):
            r, g, b = rgb[i]
            max_val = max(r, g, b)
            min_val = min(r, g, b)
            diff = max_val - min_val

            # Value
            hsv[i, 2] = max_val

            # Saturation
            if max_val == 0:
                hsv[i, 1] = 0
            else:
                hsv[i, 1] = diff / max_val

            # Hue
            if diff == 0:
                hsv[i, 0] = 0
            elif max_val == r:
                hsv[i, 0] = (60 * ((g - b) / diff) + 360) % 360
            elif max_val == g:
                hsv[i, 0] = (60 * ((b - r) / diff) + 120) % 360
            else:
                hsv[i, 0] = (60 * ((r - g) / diff) + 240) % 360

            hsv[i, 0] /= 360  # Normalize to [0, 1]

        return hsv

    def _rgb_to_yuv(self, rgb: np.ndarray) -> np.ndarray:
        """Convert RGB to YUV color space."""
        # YUV conversion matrix
        yuv = np.zeros_like(rgb)
        yuv[:, 0] = 0.299 * rgb[:, 0] + 0.587 * rgb[:, 1] + 0.114 * rgb[:, 2]  # Y
        yuv[:, 1] = -0.14713 * rgb[:, 0] - 0.28886 * rgb[:, 1] + 0.436 * rgb[:, 2]  # U
        yuv[:, 2] = 0.615 * rgb[:, 0] - 0.51499 * rgb[:, 1] - 0.10001 * rgb[:, 2]  # V

        return yuv

    def _compute_color_statistics(self, colors: np.ndarray,
                                outlier_mask: np.ndarray) -> Dict[str, Any]:
        """Compute color sampling statistics."""
        valid_colors = colors[~outlier_mask]

        stats = {
            'total_samples': len(colors),
            'valid_samples': len(valid_colors),
            'outlier_count': np.sum(outlier_mask),
            'outlier_fraction': np.mean(outlier_mask),
            'color_channels': colors.shape[1]
        }

        if len(valid_colors) > 0:
            stats.update({
                'mean_color': np.mean(valid_colors, axis=0),
                'std_color': np.std(valid_colors, axis=0),
                'min_color': np.min(valid_colors, axis=0),
                'max_color': np.max(valid_colors, axis=0),
                'color_range': np.max(valid_colors, axis=0) - np.min(valid_colors, axis=0),
                'color_diversity': np.mean(np.std(valid_colors, axis=0))
            })
        else:
            # Handle case with no valid colors
            c = colors.shape[1]
            stats.update({
                'mean_color': np.zeros(c),
                'std_color': np.zeros(c),
                'min_color': np.zeros(c),
                'max_color': np.zeros(c),
                'color_range': np.zeros(c),
                'color_diversity': 0.0
            })

        return stats

    def _validate_color_accuracy(self, image: np.ndarray, positions: np.ndarray,
                                sampled_colors: np.ndarray) -> Dict[str, Any]:
        """Validate color sampling accuracy."""
        h, w = image.shape[:2]

        # Sample validation positions (integer positions only)
        validation_positions = []
        expected_colors = []
        actual_colors = []

        for i, pos in enumerate(positions):
            y, x = pos

            # Only validate positions that are exactly on pixels
            if (abs(y - round(y)) < 1e-6 and abs(x - round(x)) < 1e-6 and
                0 <= round(y) < h and 0 <= round(x) < w):

                yi, xi = int(round(y)), int(round(x))
                validation_positions.append([yi, xi])
                expected_colors.append(image[yi, xi])
                actual_colors.append(sampled_colors[i])

        if len(validation_positions) == 0:
            return {
                'validation_count': 0,
                'mean_error': 0.0,
                'max_error': 0.0,
                'accuracy_within_tolerance': 1.0,
                'channel_errors': np.array([])
            }

        expected_colors = np.array(expected_colors)
        actual_colors = np.array(actual_colors)

        # Compute errors
        errors = np.abs(expected_colors - actual_colors)
        mean_error = np.mean(errors)
        max_error = np.max(errors)
        channel_errors = np.mean(errors, axis=0)

        # Check accuracy within tolerance
        within_tolerance = np.all(errors <= self.config.validation_tolerance, axis=1)
        accuracy_fraction = np.mean(within_tolerance)

        return {
            'validation_count': len(validation_positions),
            'mean_error': mean_error,
            'max_error': max_error,
            'accuracy_within_tolerance': accuracy_fraction,
            'channel_errors': channel_errors
        }

    def validate_sampling_quality(self, result: ColorSamplingResult) -> Dict[str, Any]:
        """Validate overall sampling quality."""
        issues = []
        recommendations = []

        stats = result.statistics
        validation = result.validation_results

        # Check outlier rate
        if stats['outlier_fraction'] > 0.2:
            issues.append(f"High outlier rate: {stats['outlier_fraction']:.1%}")
            recommendations.append("Reduce outlier_threshold or improve preprocessing")

        # Check color diversity
        if stats['color_diversity'] < 0.05:
            issues.append("Low color diversity - samples too uniform")
            recommendations.append("Check image content or sampling positions")

        # Check validation accuracy
        if validation['validation_count'] > 0:
            if validation['accuracy_within_tolerance'] < 0.95:
                issues.append(f"Low accuracy: {validation['accuracy_within_tolerance']:.1%}")
                recommendations.append("Adjust interpolation method or tolerance")

            if validation['mean_error'] > self.config.validation_tolerance:
                issues.append(f"High mean error: {validation['mean_error']:.4f}")
                recommendations.append("Check preprocessing or interpolation settings")

        # Compute quality score
        quality_score = self._compute_sampling_quality_score(result)

        return {
            'passed': len(issues) == 0,
            'issues': issues,
            'recommendations': recommendations,
            'quality_score': quality_score,
            'validation_summary': {
                'outlier_rate': stats['outlier_fraction'],
                'color_diversity': stats['color_diversity'],
                'accuracy': validation.get('accuracy_within_tolerance', 0.0),
                'mean_error': validation.get('mean_error', 0.0)
            }
        }

    def _compute_sampling_quality_score(self, result: ColorSamplingResult) -> float:
        """Compute overall quality score for color sampling."""
        stats = result.statistics
        validation = result.validation_results

        # Outlier score (lower outlier rate is better)
        outlier_score = max(0, 1 - stats['outlier_fraction'] * 2)

        # Diversity score (moderate diversity is good)
        diversity = stats['color_diversity']
        diversity_score = 1 - abs(diversity - 0.2) / 0.2  # Target around 0.2
        diversity_score = max(0, min(1, diversity_score))

        # Accuracy score
        if validation['validation_count'] > 0:
            accuracy_score = validation['accuracy_within_tolerance']
            error_score = max(0, 1 - validation['mean_error'] / self.config.validation_tolerance)
        else:
            accuracy_score = 0.5  # Neutral score when no validation possible
            error_score = 0.5

        # Weighted combination
        quality_score = (0.3 * outlier_score +
                        0.2 * diversity_score +
                        0.3 * accuracy_score +
                        0.2 * error_score)

        return quality_score


def sample_colors_at_positions(image: np.ndarray, positions: np.ndarray,
                              interpolation: str = 'bilinear') -> ColorSamplingResult:
    """Convenience function to sample colors with default settings.

    Args:
        image: Input image (H, W, C) or (H, W)
        positions: Sample positions as (N, 2) array
        interpolation: Interpolation method ('nearest', 'bilinear', 'bicubic')

    Returns:
        ColorSamplingResult with sampled colors
    """
    config = ColorSamplingConfig(interpolation_method=interpolation)
    sampler = ColorSampler(config)
    return sampler.sample_colors(image, positions)


def validate_color_sampling(image: np.ndarray, positions: np.ndarray,
                           sampled_colors: np.ndarray) -> Dict[str, Any]:
    """Convenience function to validate color sampling accuracy.

    Args:
        image: Original image (H, W, C) or (H, W)
        positions: Sample positions as (N, 2) array
        sampled_colors: Sampled color values (N, C)

    Returns:
        Dictionary with validation results
    """
    sampler = ColorSampler()
    return sampler._validate_color_accuracy(image, positions, sampled_colors)