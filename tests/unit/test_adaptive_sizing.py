#!/usr/bin/env python3
"""Unit tests for adaptive sizing module."""

import pytest
import numpy as np
from src.splat_this.core.adaptive_sizing import (
    AdaptiveSizer,
    SizingConfig,
    SizingResult,
    SplatSizeAllocation,
    compute_adaptive_sizes,
    allocate_splat_sizes
)


class TestSizingConfig:
    """Test sizing configuration."""

    def test_init_defaults(self):
        """Test default configuration initialization."""
        config = SizingConfig()
        assert config.base_size == 0.02
        assert config.size_range == (0.005, 0.08)
        assert config.complexity_sensitivity == 0.8
        assert config.variance_weight == 0.3
        assert config.edge_density_weight == 0.4
        assert config.anisotropy_influence == 0.3
        assert config.smoothing_sigma == 1.5
        assert config.size_quantization == 20
        assert config.adaptive_range is True
        assert config.normalization == 'percentile'

    def test_init_custom(self):
        """Test custom configuration initialization."""
        config = SizingConfig(
            base_size=0.03,
            size_range=(0.01, 0.06),
            complexity_sensitivity=0.9,
            variance_weight=0.4,
            edge_density_weight=0.5,
            anisotropy_influence=0.2,
            smoothing_sigma=2.0,
            size_quantization=15,
            adaptive_range=False,
            normalization='minmax'
        )
        assert config.base_size == 0.03
        assert config.size_range == (0.01, 0.06)
        assert config.complexity_sensitivity == 0.9
        assert config.variance_weight == 0.4
        assert config.edge_density_weight == 0.5
        assert config.anisotropy_influence == 0.2
        assert config.smoothing_sigma == 2.0
        assert config.size_quantization == 15
        assert config.adaptive_range is False
        assert config.normalization == 'minmax'

    def test_validation_base_size(self):
        """Test validation of base size."""
        with pytest.raises(ValueError, match="base_size must be in"):
            SizingConfig(base_size=0.0)

        with pytest.raises(ValueError, match="base_size must be in"):
            SizingConfig(base_size=1.0)

    def test_validation_size_range(self):
        """Test validation of size range."""
        with pytest.raises(ValueError, match="size_range must be"):
            SizingConfig(size_range=(0.1, 0.05))  # min > max

        with pytest.raises(ValueError, match="size_range must be"):
            SizingConfig(size_range=(0.0, 0.1))  # min = 0

        with pytest.raises(ValueError, match="size_range must be"):
            SizingConfig(size_range=(0.1, 1.0))  # max = 1

    def test_validation_weight_ranges(self):
        """Test validation of weight parameters."""
        with pytest.raises(ValueError, match="complexity_sensitivity must be in"):
            SizingConfig(complexity_sensitivity=-0.1)

        with pytest.raises(ValueError, match="complexity_sensitivity must be in"):
            SizingConfig(complexity_sensitivity=1.1)

        with pytest.raises(ValueError, match="variance_weight must be in"):
            SizingConfig(variance_weight=2.0)

        with pytest.raises(ValueError, match="edge_density_weight must be in"):
            SizingConfig(edge_density_weight=-1.0)

        with pytest.raises(ValueError, match="anisotropy_influence must be in"):
            SizingConfig(anisotropy_influence=1.5)

    def test_validation_other_params(self):
        """Test validation of other parameters."""
        with pytest.raises(ValueError, match="smoothing_sigma must be non-negative"):
            SizingConfig(smoothing_sigma=-1.0)

        with pytest.raises(ValueError, match="size_quantization must be at least 2"):
            SizingConfig(size_quantization=1)


class TestSizingResult:
    """Test sizing result data structure."""

    def test_result_creation(self):
        """Test creation of sizing result."""
        h, w = 32, 32
        result = SizingResult(
            size_map=np.ones((h, w)) * 0.02,
            complexity_map=np.zeros((h, w)),
            variance_map=np.zeros((h, w)),
            edge_density_map=np.zeros((h, w)),
            size_distribution=np.zeros(50),
            statistics={'mean_size': 0.02}
        )

        assert result.size_map.shape == (h, w)
        assert result.complexity_map.shape == (h, w)
        assert result.variance_map.shape == (h, w)
        assert result.edge_density_map.shape == (h, w)
        assert len(result.size_distribution) == 50
        assert isinstance(result.statistics, dict)


class TestSplatSizeAllocation:
    """Test splat size allocation data structure."""

    def test_allocation_creation(self):
        """Test creation of splat size allocation."""
        n_splats = 10
        allocation = SplatSizeAllocation(
            positions=np.random.rand(n_splats, 2) * 32,
            sizes=np.ones(n_splats) * 0.02,
            complexity_scores=np.random.rand(n_splats),
            size_rationale=['test'] * n_splats
        )

        assert allocation.positions.shape == (n_splats, 2)
        assert allocation.sizes.shape == (n_splats,)
        assert allocation.complexity_scores.shape == (n_splats,)
        assert len(allocation.size_rationale) == n_splats


class TestAdaptiveSizer:
    """Test adaptive sizer functionality."""

    def test_init_default(self):
        """Test sizer initialization with defaults."""
        sizer = AdaptiveSizer((64, 64))
        assert isinstance(sizer.config, SizingConfig)
        assert sizer.image_shape == (64, 64)
        assert sizer.image_diagonal == pytest.approx(np.sqrt(64**2 + 64**2))

    def test_init_custom_config(self):
        """Test sizer initialization with custom config."""
        config = SizingConfig(base_size=0.03)
        sizer = AdaptiveSizer((32, 32), config)
        assert sizer.config.base_size == 0.03
        assert sizer.image_shape == (32, 32)

    def test_compute_adaptive_sizes_grayscale(self):
        """Test adaptive sizing on grayscale image."""
        # Create test image with varying complexity
        image = np.zeros((64, 64))
        # Uniform region
        image[:32, :32] = 0.5
        # High complexity region (checkerboard)
        for i in range(32, 64):
            for j in range(32, 64):
                if (i + j) % 2 == 0:
                    image[i, j] = 1.0

        sizer = AdaptiveSizer((64, 64))
        result = sizer.compute_adaptive_sizes(image)

        assert isinstance(result, SizingResult)
        assert result.size_map.shape == (64, 64)
        assert result.complexity_map.shape == (64, 64)
        assert result.variance_map.shape == (64, 64)
        assert result.edge_density_map.shape == (64, 64)

        # High complexity region should have smaller sizes
        uniform_region_size = np.mean(result.size_map[:32, :32])
        complex_region_size = np.mean(result.size_map[32:, 32:])
        assert uniform_region_size > complex_region_size

    def test_compute_adaptive_sizes_color(self):
        """Test adaptive sizing on color image."""
        # Create color test image
        image = np.zeros((32, 32, 3))
        image[:, :, 0] = np.linspace(0, 1, 32).reshape(1, -1)  # Red gradient
        image[:, :, 1] = 0.5  # Constant green
        image[:, :, 2] = 0.3  # Constant blue

        sizer = AdaptiveSizer((32, 32))
        result = sizer.compute_adaptive_sizes(image)

        assert result.size_map.shape == (32, 32)
        assert np.all(np.isfinite(result.size_map))
        assert np.all(result.size_map > 0)

    def test_compute_local_complexity(self):
        """Test local complexity computation."""
        # Create image with clear complexity differences
        image = np.zeros((32, 32))
        # Simple gradient
        image[:16, :] = np.linspace(0, 1, 32)
        # High frequency pattern
        for i in range(16, 32):
            for j in range(32):
                image[i, j] = 0.5 + 0.5 * np.sin(j * np.pi / 2)

        sizer = AdaptiveSizer((32, 32))
        complexity = sizer._compute_local_complexity(image)

        assert complexity.shape == (32, 32)
        assert np.all(0 <= complexity)
        assert np.all(complexity <= 1)

        # High frequency region should have higher complexity
        simple_complexity = np.mean(complexity[:16, :])
        complex_complexity = np.mean(complexity[16:, :])
        assert complex_complexity > simple_complexity

    def test_compute_local_variance(self):
        """Test local variance computation."""
        # Create image with varying local variance
        image = np.zeros((32, 32))
        # Uniform region
        image[:16, :16] = 0.5
        # Variable region
        image[16:, 16:] = np.random.rand(16, 16)

        sizer = AdaptiveSizer((32, 32))
        variance = sizer._compute_local_variance(image)

        assert variance.shape == (32, 32)
        assert np.all(variance >= 0)
        assert np.all(variance <= 1)

        # Variable region should have higher variance
        uniform_variance = np.mean(variance[:16, :16])
        variable_variance = np.mean(variance[16:, 16:])
        assert variable_variance > uniform_variance

    def test_compute_edge_density(self):
        """Test edge density computation."""
        # Create image with edges
        image = np.zeros((32, 32))
        image[:, 15:17] = 1.0  # Vertical edge
        image[15:17, :] = 0.5  # Horizontal edge

        sizer = AdaptiveSizer((32, 32))
        edge_density = sizer._compute_edge_density(image)

        assert edge_density.shape == (32, 32)
        assert np.all(edge_density >= 0)
        assert np.all(edge_density <= 1)

        # Edge regions should have higher density
        edge_region_density = np.mean(edge_density[14:18, 14:18])
        background_density = np.mean(edge_density[:10, :10])
        assert edge_region_density > background_density

    def test_combine_sizing_factors(self):
        """Test combination of sizing factors."""
        complexity = np.ones((32, 32)) * 0.8
        variance = np.ones((32, 32)) * 0.6
        edge_density = np.ones((32, 32)) * 0.4
        anisotropy = np.ones((32, 32)) * 0.5

        sizer = AdaptiveSizer((32, 32))
        influence = sizer._combine_sizing_factors(
            complexity, variance, edge_density, anisotropy
        )

        assert influence.shape == (32, 32)
        assert np.all(0 <= influence)
        assert np.all(influence <= 1)

        # Test without anisotropy
        influence_no_aniso = sizer._combine_sizing_factors(
            complexity, variance, edge_density, None
        )
        assert influence_no_aniso.shape == (32, 32)

    def test_compute_size_map(self):
        """Test size map computation."""
        influence = np.array([[0.1, 0.5, 0.9],
                             [0.0, 0.3, 1.0],
                             [0.2, 0.7, 0.4]])

        sizer = AdaptiveSizer((3, 3))
        size_map = sizer._compute_size_map(influence)

        assert size_map.shape == (3, 3)
        assert np.all(size_map >= sizer.config.size_range[0])
        assert np.all(size_map <= sizer.config.size_range[1])

        # High influence should lead to smaller sizes
        assert size_map[0, 0] > size_map[0, 2]  # Low vs high influence

    def test_quantize_sizes(self):
        """Test size quantization."""
        size_map = np.random.uniform(0.01, 0.05, (32, 32))

        config = SizingConfig(size_quantization=5, size_range=(0.01, 0.05))
        sizer = AdaptiveSizer((32, 32), config)
        quantized = sizer._quantize_sizes(size_map)

        assert quantized.shape == (32, 32)

        # Check that values are from the quantization levels
        expected_levels = np.linspace(0.01, 0.05, 5)
        for value in np.unique(quantized):
            assert np.any(np.isclose(value, expected_levels, atol=1e-6))

    def test_compute_size_distribution(self):
        """Test size distribution computation."""
        size_map = np.random.uniform(0.01, 0.05, (32, 32))

        sizer = AdaptiveSizer((32, 32))
        distribution = sizer._compute_size_distribution(size_map)

        assert len(distribution) == 49  # np.histogram returns n-1 bins for n bin edges
        assert np.sum(distribution) == 32 * 32  # Total pixel count

    def test_compute_sizing_statistics(self):
        """Test sizing statistics computation."""
        size_map = np.random.uniform(0.01, 0.05, (32, 32))
        complexity_map = np.random.rand(32, 32)
        variance_map = np.random.rand(32, 32)

        sizer = AdaptiveSizer((32, 32))
        stats = sizer._compute_sizing_statistics(size_map, complexity_map, variance_map)

        required_keys = [
            'mean_size', 'std_size', 'min_size', 'max_size', 'size_range',
            'size_diversity', 'complexity_correlation', 'small_size_fraction',
            'large_size_fraction', 'size_entropy'
        ]

        for key in required_keys:
            assert key in stats
            assert isinstance(stats[key], (int, float))
            assert np.isfinite(stats[key])

    def test_allocate_splat_sizes(self):
        """Test splat size allocation."""
        # Create test image
        image = np.random.rand(64, 64)

        # Create test positions
        positions = np.array([[16, 16], [32, 32], [48, 48]])

        sizer = AdaptiveSizer((64, 64))
        allocation = sizer.allocate_splat_sizes(positions, image)

        assert isinstance(allocation, SplatSizeAllocation)
        assert allocation.positions.shape == (3, 2)
        assert allocation.sizes.shape == (3,)
        assert allocation.complexity_scores.shape == (3,)
        assert len(allocation.size_rationale) == 3

        # Sizes should be within valid range
        assert np.all(allocation.sizes >= sizer.config.size_range[0])
        assert np.all(allocation.sizes <= sizer.config.size_range[1])

    def test_allocate_splat_sizes_bounds_clipping(self):
        """Test that out-of-bounds positions are clipped."""
        image = np.ones((32, 32))
        positions = np.array([[-5, -5], [40, 40], [16, 16]])

        sizer = AdaptiveSizer((32, 32))
        allocation = sizer.allocate_splat_sizes(positions, image)

        clipped_positions = allocation.positions
        assert np.all(clipped_positions[:, 0] >= 0)
        assert np.all(clipped_positions[:, 0] < 32)
        assert np.all(clipped_positions[:, 1] >= 0)
        assert np.all(clipped_positions[:, 1] < 32)

    def test_validate_size_distribution(self):
        """Test size distribution validation."""
        # Create test with good size distribution
        image = np.random.rand(64, 64)

        sizer = AdaptiveSizer((64, 64))
        result = sizer.compute_adaptive_sizes(image)
        validation = sizer.validate_size_distribution(result)

        assert 'passed' in validation
        assert 'issues' in validation
        assert 'recommendations' in validation
        assert 'statistics' in validation
        assert 'size_quality_score' in validation

        assert isinstance(validation['passed'], bool)
        assert isinstance(validation['issues'], list)
        assert isinstance(validation['recommendations'], list)
        assert isinstance(validation['statistics'], dict)
        assert 0 <= validation['size_quality_score'] <= 1

    def test_compute_quality_score(self):
        """Test quality score computation."""
        stats = {
            'size_diversity': 0.5,
            'complexity_correlation': 0.7,
            'size_range': 0.03,
            'size_entropy': 2.0
        }

        sizer = AdaptiveSizer((64, 64))
        score = sizer._compute_quality_score(stats)

        assert 0 <= score <= 1
        assert isinstance(score, float)

    def test_different_normalization_methods(self):
        """Test different normalization methods."""
        image = np.random.rand(32, 32)

        normalizations = ['percentile', 'minmax', 'adaptive']

        for norm in normalizations:
            config = SizingConfig(normalization=norm)
            sizer = AdaptiveSizer((32, 32), config)
            result = sizer.compute_adaptive_sizes(image)

            assert result.size_map.shape == (32, 32)
            assert np.all(np.isfinite(result.size_map))
            assert np.all(result.size_map > 0)

    def test_adaptive_range_effect(self):
        """Test adaptive range adjustment."""
        # High complexity image
        high_complexity_image = np.random.rand(32, 32)

        # Low complexity image
        low_complexity_image = np.ones((32, 32)) * 0.5

        config_adaptive = SizingConfig(adaptive_range=True)
        config_fixed = SizingConfig(adaptive_range=False)

        sizer_adaptive = AdaptiveSizer((32, 32), config_adaptive)
        sizer_fixed = AdaptiveSizer((32, 32), config_fixed)

        # Test with both images
        result_high_adaptive = sizer_adaptive.compute_adaptive_sizes(high_complexity_image)
        result_high_fixed = sizer_fixed.compute_adaptive_sizes(high_complexity_image)

        # Both should complete without error
        assert result_high_adaptive.size_map.shape == (32, 32)
        assert result_high_fixed.size_map.shape == (32, 32)

    def test_smoothing_effect(self):
        """Test smoothing effect on size maps."""
        image = np.random.rand(32, 32)

        # No smoothing
        config_no_smooth = SizingConfig(smoothing_sigma=0.0)
        sizer_no_smooth = AdaptiveSizer((32, 32), config_no_smooth)
        result_no_smooth = sizer_no_smooth.compute_adaptive_sizes(image)

        # With smoothing
        config_smooth = SizingConfig(smoothing_sigma=2.0)
        sizer_smooth = AdaptiveSizer((32, 32), config_smooth)
        result_smooth = sizer_smooth.compute_adaptive_sizes(image)

        # Smoothed version should have lower variance
        variance_no_smooth = np.var(result_no_smooth.size_map)
        variance_smooth = np.var(result_smooth.size_map)
        assert variance_smooth <= variance_no_smooth


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_compute_adaptive_sizes_default(self):
        """Test convenience function with default parameters."""
        image = np.random.rand(32, 32)
        result = compute_adaptive_sizes(image)

        assert isinstance(result, SizingResult)
        assert result.size_map.shape == (32, 32)

    def test_compute_adaptive_sizes_custom_params(self):
        """Test convenience function with custom parameters."""
        image = np.random.rand(32, 32)
        result = compute_adaptive_sizes(
            image, base_size=0.03, complexity_sensitivity=0.9
        )

        assert isinstance(result, SizingResult)
        assert result.size_map.shape == (32, 32)

    def test_allocate_splat_sizes_convenience(self):
        """Test convenience function for splat size allocation."""
        image = np.random.rand(32, 32)
        positions = np.array([[16, 16], [8, 8]])

        allocation = allocate_splat_sizes(positions, image)

        assert isinstance(allocation, SplatSizeAllocation)
        assert allocation.positions.shape == (2, 2)
        assert allocation.sizes.shape == (2,)


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_very_small_image(self):
        """Test adaptive sizing on very small image."""
        image = np.ones((3, 3))

        sizer = AdaptiveSizer((3, 3))
        result = sizer.compute_adaptive_sizes(image)

        assert result.size_map.shape == (3, 3)
        assert np.all(np.isfinite(result.size_map))

    def test_single_pixel_image(self):
        """Test adaptive sizing on single pixel image."""
        image = np.array([[0.5]])

        sizer = AdaptiveSizer((1, 1))
        result = sizer.compute_adaptive_sizes(image)

        assert result.size_map.shape == (1, 1)
        assert np.all(np.isfinite(result.size_map))

    def test_uniform_image(self):
        """Test adaptive sizing on uniform image."""
        image = np.ones((32, 32)) * 0.7

        sizer = AdaptiveSizer((32, 32))
        result = sizer.compute_adaptive_sizes(image)

        # Uniform image should have relatively uniform sizes
        size_std = np.std(result.size_map)
        assert size_std < 0.01  # Very small variation

    def test_zero_image(self):
        """Test adaptive sizing on zero image."""
        image = np.zeros((32, 32))

        sizer = AdaptiveSizer((32, 32))
        result = sizer.compute_adaptive_sizes(image)

        assert np.all(np.isfinite(result.size_map))
        assert np.all(result.size_map > 0)

    def test_high_contrast_image(self):
        """Test adaptive sizing on high contrast image."""
        image = np.zeros((32, 32))
        image[::2, ::2] = 1.0  # Checkerboard

        sizer = AdaptiveSizer((32, 32))
        result = sizer.compute_adaptive_sizes(image)

        assert np.all(np.isfinite(result.size_map))
        assert np.all(result.size_map > 0)

        # Should have high complexity
        assert np.mean(result.complexity_map) > 0.1

    def test_different_image_sizes(self):
        """Test adaptive sizing on different image sizes."""
        sizes = [(16, 16), (32, 64), (128, 32)]

        for h, w in sizes:
            image = np.random.rand(h, w)

            sizer = AdaptiveSizer((h, w))
            result = sizer.compute_adaptive_sizes(image)

            assert result.size_map.shape == (h, w)
            assert result.complexity_map.shape == (h, w)
            assert result.variance_map.shape == (h, w)

    def test_empty_positions_array(self):
        """Test size allocation with empty positions array."""
        image = np.ones((32, 32))
        positions = np.empty((0, 2))

        sizer = AdaptiveSizer((32, 32))
        allocation = sizer.allocate_splat_sizes(positions, image)

        assert allocation.positions.shape == (0, 2)
        assert allocation.sizes.shape == (0,)
        assert len(allocation.size_rationale) == 0

    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        # Very small values
        image = np.ones((32, 32)) * 1e-10

        sizer = AdaptiveSizer((32, 32))
        result = sizer.compute_adaptive_sizes(image)

        assert np.all(np.isfinite(result.size_map))
        assert np.all(np.isfinite(result.complexity_map))

        # Very large values
        image = np.ones((32, 32)) * 1e10

        result = sizer.compute_adaptive_sizes(image)

        assert np.all(np.isfinite(result.size_map))
        assert np.all(np.isfinite(result.complexity_map))

    def test_extreme_config_values(self):
        """Test with extreme configuration values."""
        # Minimal sensitivity
        config = SizingConfig(
            complexity_sensitivity=0.0,
            variance_weight=0.0,
            edge_density_weight=0.0,
            anisotropy_influence=0.0
        )

        image = np.random.rand(32, 32)
        sizer = AdaptiveSizer((32, 32), config)
        result = sizer.compute_adaptive_sizes(image)

        assert np.all(np.isfinite(result.size_map))

        # Maximum sensitivity
        config = SizingConfig(
            complexity_sensitivity=1.0,
            variance_weight=1.0,
            edge_density_weight=1.0,
            anisotropy_influence=1.0
        )

        sizer = AdaptiveSizer((32, 32), config)
        result = sizer.compute_adaptive_sizes(image)

        assert np.all(np.isfinite(result.size_map))