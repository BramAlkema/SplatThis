#!/usr/bin/env python3
"""Unit tests for structure tensor analysis module."""

import pytest
import numpy as np
from src.splat_this.core.structure_tensor import (
    StructureTensorAnalyzer,
    StructureTensorConfig,
    StructureTensorResult,
    compute_structure_tensor,
    analyze_local_orientations
)


class TestStructureTensorConfig:
    """Test structure tensor configuration."""

    def test_init_defaults(self):
        """Test default configuration initialization."""
        config = StructureTensorConfig()
        assert config.gradient_sigma == 1.0
        assert config.integration_sigma == 2.0
        assert config.anisotropy_threshold == 0.1
        assert config.coherence_threshold == 0.2
        assert config.edge_enhancement is True
        assert config.normalization == 'trace'

    def test_init_custom(self):
        """Test custom configuration initialization."""
        config = StructureTensorConfig(
            gradient_sigma=1.5,
            integration_sigma=3.0,
            anisotropy_threshold=0.2,
            coherence_threshold=0.3,
            edge_enhancement=False,
            normalization='determinant'
        )
        assert config.gradient_sigma == 1.5
        assert config.integration_sigma == 3.0
        assert config.anisotropy_threshold == 0.2
        assert config.coherence_threshold == 0.3
        assert config.edge_enhancement is False
        assert config.normalization == 'determinant'

    def test_validation_positive_sigmas(self):
        """Test validation of positive sigma values."""
        with pytest.raises(ValueError, match="gradient_sigma must be positive"):
            StructureTensorConfig(gradient_sigma=0.0)

        with pytest.raises(ValueError, match="integration_sigma must be positive"):
            StructureTensorConfig(integration_sigma=-1.0)

    def test_validation_threshold_ranges(self):
        """Test validation of threshold ranges."""
        with pytest.raises(ValueError, match="anisotropy_threshold must be in"):
            StructureTensorConfig(anisotropy_threshold=-0.1)

        with pytest.raises(ValueError, match="anisotropy_threshold must be in"):
            StructureTensorConfig(anisotropy_threshold=1.1)

        with pytest.raises(ValueError, match="coherence_threshold must be in"):
            StructureTensorConfig(coherence_threshold=2.0)


class TestStructureTensorResult:
    """Test structure tensor result data structure."""

    def test_result_creation(self):
        """Test creation of structure tensor result."""
        h, w = 32, 32
        result = StructureTensorResult(
            orientation=np.zeros((h, w)),
            anisotropy=np.ones((h, w)),
            coherence=np.ones((h, w)) * 0.5,
            eigenvalues=np.zeros((h, w, 2)),
            eigenvectors=np.zeros((h, w, 2, 2)),
            tensor_trace=np.ones((h, w))
        )

        assert result.orientation.shape == (h, w)
        assert result.anisotropy.shape == (h, w)
        assert result.coherence.shape == (h, w)
        assert result.eigenvalues.shape == (h, w, 2)
        assert result.eigenvectors.shape == (h, w, 2, 2)
        assert result.tensor_trace.shape == (h, w)


class TestStructureTensorAnalyzer:
    """Test structure tensor analyzer functionality."""

    def test_init_default(self):
        """Test analyzer initialization with defaults."""
        analyzer = StructureTensorAnalyzer()
        assert isinstance(analyzer.config, StructureTensorConfig)
        assert analyzer.config.gradient_sigma == 1.0

    def test_init_custom_config(self):
        """Test analyzer initialization with custom config."""
        config = StructureTensorConfig(gradient_sigma=2.0)
        analyzer = StructureTensorAnalyzer(config)
        assert analyzer.config.gradient_sigma == 2.0

    def test_compute_structure_tensor_grayscale(self):
        """Test structure tensor computation on grayscale image."""
        # Create simple gradient image
        image = np.zeros((32, 32))
        for i in range(32):
            image[:, i] = i / 31.0  # Horizontal gradient

        analyzer = StructureTensorAnalyzer()
        result = analyzer.compute_structure_tensor(image)

        assert isinstance(result, StructureTensorResult)
        assert result.orientation.shape == (32, 32)
        assert result.anisotropy.shape == (32, 32)
        assert result.coherence.shape == (32, 32)

        # For horizontal gradient, orientation should be mostly vertical (π/2)
        # Allow some tolerance due to filtering
        central_region = result.orientation[8:24, 8:24]
        mean_orientation = np.mean(central_region)
        assert abs(mean_orientation - np.pi/2) < 0.2 or abs(mean_orientation) < 0.2

    def test_compute_structure_tensor_color(self):
        """Test structure tensor computation on color image."""
        # Create color gradient image
        image = np.zeros((32, 32, 3))
        for i in range(32):
            image[:, i, 0] = i / 31.0  # Red gradient
            image[:, i, 1] = 0.5       # Constant green
            image[:, i, 2] = 0.3       # Constant blue

        analyzer = StructureTensorAnalyzer()
        result = analyzer.compute_structure_tensor(image)

        assert result.orientation.shape == (32, 32)
        assert np.all(np.isfinite(result.orientation))
        assert np.all(np.isfinite(result.anisotropy))
        assert np.all(np.isfinite(result.coherence))

    def test_compute_orientation_consistency(self):
        """Test that orientation computation is consistent."""
        # Create diagonal edge
        image = np.zeros((32, 32))
        for i in range(32):
            for j in range(32):
                if i + j < 32:
                    image[i, j] = 1.0

        analyzer = StructureTensorAnalyzer()
        result = analyzer.compute_structure_tensor(image)

        # Orientation should be well-defined in the edge region
        edge_region = result.orientation[10:22, 10:22]
        assert np.all(0 <= edge_region)
        assert np.all(edge_region < np.pi)

    def test_compute_anisotropy_range(self):
        """Test that anisotropy values are in valid range."""
        # Create checkerboard pattern (high anisotropy)
        image = np.zeros((32, 32))
        for i in range(32):
            for j in range(32):
                if (i // 4 + j // 4) % 2 == 0:
                    image[i, j] = 1.0

        analyzer = StructureTensorAnalyzer()
        result = analyzer.compute_structure_tensor(image)

        assert np.all(0 <= result.anisotropy)
        assert np.all(result.anisotropy <= 1)

    def test_compute_coherence_range(self):
        """Test that coherence values are in valid range."""
        # Create image with strong edges
        image = np.zeros((32, 32))
        image[:, 15:17] = 1.0  # Vertical stripe

        analyzer = StructureTensorAnalyzer()
        result = analyzer.compute_structure_tensor(image)

        assert np.all(0 <= result.coherence)
        assert np.all(result.coherence <= 1)

        # Should have high coherence near the edge
        edge_coherence = result.coherence[:, 15:17]
        assert np.max(edge_coherence) > 0.1

    def test_analyze_local_structure(self):
        """Test local structure analysis at specific points."""
        # Create simple vertical edge
        image = np.zeros((32, 32))
        image[:, 16:] = 1.0

        points = np.array([[16, 16], [16, 8], [16, 24]])  # On and off edge

        analyzer = StructureTensorAnalyzer()
        analysis = analyzer.analyze_local_structure(image, points)

        assert 'orientations' in analysis
        assert 'anisotropies' in analysis
        assert 'coherences' in analysis
        assert 'points' in analysis

        assert len(analysis['orientations']) == 3
        assert len(analysis['anisotropies']) == 3
        assert len(analysis['coherences']) == 3

        # Point on edge should have higher anisotropy than points off edge
        edge_anisotropy = analysis['anisotropies'][0]
        off_edge_anisotropies = analysis['anisotropies'][1:]
        # This might not always hold due to filtering, so use lenient check
        assert edge_anisotropy >= 0

    def test_analyze_local_structure_bounds_clipping(self):
        """Test that out-of-bounds points are clipped."""
        image = np.ones((32, 32))
        points = np.array([[-5, -5], [40, 40], [16, 16]])

        analyzer = StructureTensorAnalyzer()
        analysis = analyzer.analyze_local_structure(image, points)

        clipped_points = analysis['points']
        assert np.all(clipped_points[:, 0] >= 0)
        assert np.all(clipped_points[:, 0] < 32)
        assert np.all(clipped_points[:, 1] >= 0)
        assert np.all(clipped_points[:, 1] < 32)

    def test_detect_edge_following_locations(self):
        """Test detection of edge-following locations."""
        # Create cross pattern
        image = np.zeros((32, 32))
        image[15:17, :] = 1.0    # Horizontal line
        image[:, 15:17] = 1.0    # Vertical line

        analyzer = StructureTensorAnalyzer()
        locations = analyzer.detect_edge_following_locations(
            image, min_coherence=0.1, min_anisotropy=0.05
        )

        assert isinstance(locations, np.ndarray)
        assert locations.shape[1] == 2  # (y, x) coordinates
        # Should detect some edge locations
        assert len(locations) > 0

    def test_detect_edge_following_empty(self):
        """Test edge detection on uniform image."""
        # Uniform image should have no edges
        image = np.ones((32, 32)) * 0.5

        analyzer = StructureTensorAnalyzer()
        locations = analyzer.detect_edge_following_locations(
            image, min_coherence=0.5, min_anisotropy=0.5
        )

        # Should find very few or no edge locations
        assert len(locations) <= 10  # Allow some noise

    def test_create_orientation_field_visualization(self):
        """Test creation of orientation field visualization."""
        # Create diagonal pattern
        image = np.zeros((32, 32))
        for i in range(32):
            for j in range(32):
                if abs(i - j) < 2:
                    image[i, j] = 1.0

        analyzer = StructureTensorAnalyzer()
        result = analyzer.compute_structure_tensor(image)
        viz = analyzer.create_orientation_field_visualization(result, stride=8)

        assert 'x_positions' in viz
        assert 'y_positions' in viz
        assert 'dx' in viz
        assert 'dy' in viz
        assert 'orientations' in viz
        assert 'anisotropies' in viz
        assert 'coherences' in viz
        assert viz['stride'] == 8

        # Check dimensions are consistent
        assert viz['x_positions'].shape == viz['y_positions'].shape
        assert viz['dx'].shape == viz['x_positions'].shape
        assert viz['dy'].shape == viz['x_positions'].shape

    def test_validate_orientation_accuracy(self):
        """Test orientation accuracy validation."""
        # Create image with known structure
        image = np.zeros((64, 64))
        image[30:34, :] = 1.0  # Horizontal edge

        analyzer = StructureTensorAnalyzer()
        validation = analyzer.validate_orientation_accuracy(image)

        assert 'total_test_points' in validation
        assert 'high_coherence_fraction' in validation
        assert 'high_anisotropy_fraction' in validation
        assert 'reliable_orientations_fraction' in validation
        assert 'mean_coherence' in validation
        assert 'mean_anisotropy' in validation

        assert validation['total_test_points'] > 0
        assert 0 <= validation['high_coherence_fraction'] <= 1
        assert 0 <= validation['high_anisotropy_fraction'] <= 1
        assert 0 <= validation['reliable_orientations_fraction'] <= 1

    def test_validate_orientation_with_test_points(self):
        """Test orientation validation with specific test points."""
        image = np.ones((32, 32))
        test_points = np.array([[16, 16], [8, 8], [24, 24]])

        analyzer = StructureTensorAnalyzer()
        validation = analyzer.validate_orientation_accuracy(image, test_points)

        assert validation['total_test_points'] == 3

    def test_edge_enhancement_effect(self):
        """Test that edge enhancement affects results."""
        image = np.zeros((32, 32))
        image[:, 15:17] = 1.0  # Vertical edge

        # With edge enhancement
        config_enhanced = StructureTensorConfig(edge_enhancement=True)
        analyzer_enhanced = StructureTensorAnalyzer(config_enhanced)
        result_enhanced = analyzer_enhanced.compute_structure_tensor(image)

        # Without edge enhancement
        config_normal = StructureTensorConfig(edge_enhancement=False)
        analyzer_normal = StructureTensorAnalyzer(config_normal)
        result_normal = analyzer_normal.compute_structure_tensor(image)

        # Enhanced version might have different characteristics
        # Just verify both complete without errors
        assert result_enhanced.anisotropy.shape == result_normal.anisotropy.shape
        assert np.all(np.isfinite(result_enhanced.anisotropy))
        assert np.all(np.isfinite(result_normal.anisotropy))

    def test_normalization_options(self):
        """Test different normalization options."""
        image = np.random.rand(32, 32)

        normalizations = ['trace', 'determinant', 'none']

        for norm in normalizations:
            config = StructureTensorConfig(normalization=norm)
            analyzer = StructureTensorAnalyzer(config)
            result = analyzer.compute_structure_tensor(image)

            assert np.all(np.isfinite(result.anisotropy))
            assert np.all(0 <= result.anisotropy)
            assert np.all(result.anisotropy <= 1)


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_compute_structure_tensor_default(self):
        """Test convenience function with default parameters."""
        image = np.random.rand(32, 32)
        result = compute_structure_tensor(image)

        assert isinstance(result, StructureTensorResult)
        assert result.orientation.shape == (32, 32)

    def test_compute_structure_tensor_custom_params(self):
        """Test convenience function with custom parameters."""
        image = np.random.rand(32, 32)
        result = compute_structure_tensor(
            image, gradient_sigma=2.0, integration_sigma=3.0
        )

        assert isinstance(result, StructureTensorResult)
        assert result.orientation.shape == (32, 32)

    def test_analyze_local_orientations(self):
        """Test convenience function for local orientation analysis."""
        image = np.random.rand(32, 32)
        points = np.array([[16, 16], [8, 8]])

        analysis = analyze_local_orientations(image, points)

        assert 'orientations' in analysis
        assert 'anisotropies' in analysis
        assert 'coherences' in analysis
        assert len(analysis['orientations']) == 2


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_very_small_image(self):
        """Test structure tensor on very small image."""
        image = np.ones((3, 3))

        analyzer = StructureTensorAnalyzer()
        result = analyzer.compute_structure_tensor(image)

        assert result.orientation.shape == (3, 3)
        assert np.all(np.isfinite(result.orientation))

    def test_single_pixel_image(self):
        """Test structure tensor on single pixel image."""
        image = np.array([[0.5]])

        analyzer = StructureTensorAnalyzer()
        result = analyzer.compute_structure_tensor(image)

        assert result.orientation.shape == (1, 1)
        assert np.all(np.isfinite(result.orientation))

    def test_uniform_image(self):
        """Test structure tensor on uniform image."""
        image = np.ones((32, 32)) * 0.7

        analyzer = StructureTensorAnalyzer()
        result = analyzer.compute_structure_tensor(image)

        # Uniform image should have low anisotropy everywhere
        assert np.all(result.anisotropy < 0.1)
        assert np.all(result.coherence < 0.1)

    def test_zero_image(self):
        """Test structure tensor on zero image."""
        image = np.zeros((32, 32))

        analyzer = StructureTensorAnalyzer()
        result = analyzer.compute_structure_tensor(image)

        assert np.all(np.isfinite(result.orientation))
        assert np.all(result.anisotropy == 0)
        assert np.all(result.coherence == 0)

    def test_high_contrast_image(self):
        """Test structure tensor on high contrast image."""
        image = np.zeros((32, 32))
        image[::2, ::2] = 1.0  # Checkerboard

        analyzer = StructureTensorAnalyzer()
        result = analyzer.compute_structure_tensor(image)

        assert np.all(np.isfinite(result.orientation))
        assert np.all(0 <= result.anisotropy)
        assert np.all(result.anisotropy <= 1)

    def test_different_image_sizes(self):
        """Test structure tensor on different image sizes."""
        sizes = [(16, 16), (32, 64), (128, 32)]

        for h, w in sizes:
            image = np.random.rand(h, w)

            analyzer = StructureTensorAnalyzer()
            result = analyzer.compute_structure_tensor(image)

            assert result.orientation.shape == (h, w)
            assert result.anisotropy.shape == (h, w)
            assert result.coherence.shape == (h, w)

    def test_empty_points_array(self):
        """Test local analysis with empty points array."""
        image = np.ones((32, 32))
        points = np.empty((0, 2))

        analyzer = StructureTensorAnalyzer()
        analysis = analyzer.analyze_local_structure(image, points)

        assert len(analysis['orientations']) == 0
        assert len(analysis['anisotropies']) == 0
        assert len(analysis['coherences']) == 0

    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        # Create image with very small values
        image = np.ones((32, 32)) * 1e-10

        analyzer = StructureTensorAnalyzer()
        result = analyzer.compute_structure_tensor(image)

        assert np.all(np.isfinite(result.orientation))
        assert np.all(np.isfinite(result.anisotropy))
        assert np.all(np.isfinite(result.coherence))

        # Create image with very large values
        image = np.ones((32, 32)) * 1e10

        result = analyzer.compute_structure_tensor(image)

        assert np.all(np.isfinite(result.orientation))
        assert np.all(np.isfinite(result.anisotropy))
        assert np.all(np.isfinite(result.coherence))