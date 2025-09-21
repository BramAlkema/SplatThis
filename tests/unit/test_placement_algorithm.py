"""Unit tests for gradient-guided placement algorithm."""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from src.splat_this.core.placement_algorithm import (
    PlacementConfig,
    PlacementResult,
    GradientGuidedPlacer,
    create_gradient_guided_placer,
    place_adaptive_splats
)


class TestPlacementConfig:
    """Test PlacementConfig dataclass."""

    def test_init_defaults(self):
        """Test default configuration values."""
        config = PlacementConfig()

        assert config.base_splat_density == 0.01
        assert config.gradient_weight == 0.7
        assert config.uniform_weight == 0.3
        assert config.max_splats == 2000
        assert config.min_splats == 50
        assert config.maxima_threshold == 0.1

    def test_init_custom(self):
        """Test custom configuration values."""
        config = PlacementConfig(
            base_splat_density=0.02,
            gradient_weight=0.8,
            uniform_weight=0.2,
            max_splats=1500
        )

        assert config.base_splat_density == 0.02
        assert config.gradient_weight == 0.8
        assert config.uniform_weight == 0.2
        assert config.max_splats == 1500

    def test_validation_weights_sum(self):
        """Test warning for weights that don't sum to 1.0."""
        with patch('src.splat_this.core.placement_algorithm.logger') as mock_logger:
            config = PlacementConfig(gradient_weight=0.8, uniform_weight=0.5)
            mock_logger.warning.assert_called_once()

    def test_validation_splat_count_bounds(self):
        """Test validation of splat count bounds."""
        with pytest.raises(ValueError, match="max_splats .* must be > min_splats"):
            PlacementConfig(max_splats=50, min_splats=100)

    def test_validation_weight_ranges(self):
        """Test validation of weight ranges."""
        with pytest.raises(ValueError, match="Gradient and uniform weights must be in \\(0,1\\)"):
            PlacementConfig(gradient_weight=0.0, uniform_weight=1.0)

        with pytest.raises(ValueError, match="Gradient and uniform weights must be in \\(0,1\\)"):
            PlacementConfig(gradient_weight=1.0, uniform_weight=0.0)

    def test_validation_positive_density(self):
        """Test validation of positive base density."""
        with pytest.raises(ValueError, match="Base splat density must be positive"):
            PlacementConfig(base_splat_density=0.0)

        with pytest.raises(ValueError, match="Base splat density must be positive"):
            PlacementConfig(base_splat_density=-0.01)


class TestPlacementResult:
    """Test PlacementResult dataclass."""

    def test_init_basic(self):
        """Test basic PlacementResult initialization."""
        positions = [(10, 20), (30, 40), (50, 60)]
        normalized_positions = [(0.1, 0.2), (0.3, 0.4), (0.5, 0.6)]
        complexity_map = np.random.rand(64, 64)
        density_map = np.random.rand(64, 64)
        probability_map = np.random.rand(64, 64)

        result = PlacementResult(
            positions=positions,
            normalized_positions=normalized_positions,
            complexity_map=complexity_map,
            density_map=density_map,
            probability_map=probability_map
        )

        assert result.positions == positions
        assert result.normalized_positions == normalized_positions
        assert result.total_splats == 3  # Auto-computed from positions

    def test_post_init_total_splats(self):
        """Test automatic computation of total_splats."""
        positions = [(i, i*2) for i in range(10)]

        result = PlacementResult(
            positions=positions,
            normalized_positions=[],
            complexity_map=np.zeros((10, 10)),
            density_map=np.zeros((10, 10)),
            probability_map=np.zeros((10, 10))
        )

        assert result.total_splats == 10


class TestGradientGuidedPlacer:
    """Test GradientGuidedPlacer class."""

    def test_init_basic(self):
        """Test basic initialization."""
        placer = GradientGuidedPlacer((64, 128))

        assert placer.image_height == 64
        assert placer.image_width == 128
        assert placer.config.base_splat_density == 0.01
        assert placer.gradient_analyzer is not None
        assert placer.prob_generator is not None
        assert placer.spatial_sampler is not None

    def test_init_custom_config(self):
        """Test initialization with custom configuration."""
        config = PlacementConfig(base_splat_density=0.02, max_splats=1000)
        placer = GradientGuidedPlacer((32, 32), config)

        assert placer.config.base_splat_density == 0.02
        assert placer.config.max_splats == 1000

    def test_compute_image_complexity_grayscale(self):
        """Test image complexity computation with grayscale image."""
        placer = GradientGuidedPlacer((20, 20))

        # Create test image with edge
        image = np.zeros((20, 20))
        image[:, 10:] = 1.0  # Vertical edge

        complexity_map = placer.compute_image_complexity(image)

        assert complexity_map.shape == (20, 20)
        assert np.all(complexity_map >= 0)
        assert np.all(complexity_map <= 1)
        assert np.max(complexity_map) > 0  # Should detect edge complexity

    def test_compute_image_complexity_color(self):
        """Test image complexity computation with color image."""
        placer = GradientGuidedPlacer((15, 15))

        # Create color test image
        image = np.random.rand(15, 15, 3)

        complexity_map = placer.compute_image_complexity(image)

        assert complexity_map.shape == (15, 15)
        assert np.all(complexity_map >= 0)
        assert np.all(complexity_map <= 1)

    def test_compute_adaptive_density(self):
        """Test adaptive density computation."""
        placer = GradientGuidedPlacer((10, 10))

        # Create complexity map with varying complexity
        complexity_map = np.zeros((10, 10))
        complexity_map[2:8, 2:8] = 1.0  # High complexity center

        density_map, total_count = placer.compute_adaptive_density(complexity_map)

        assert density_map.shape == (10, 10)
        assert np.all(density_map >= 0)
        assert total_count >= placer.config.min_splats
        assert total_count <= placer.config.max_splats

        # High complexity region should have higher density
        center_density = np.mean(density_map[2:8, 2:8])
        corner_density = density_map[0, 0]
        assert center_density > corner_density

    def test_compute_adaptive_density_clipping(self):
        """Test density computation with count clipping."""
        config = PlacementConfig(min_splats=100, max_splats=200, base_splat_density=0.001)
        placer = GradientGuidedPlacer((20, 20), config)

        # Very low complexity should trigger min_splats
        low_complexity = np.ones((20, 20)) * 0.1
        _, count_low = placer.compute_adaptive_density(low_complexity)
        assert count_low == config.min_splats

        # Very high complexity should trigger max_splats
        config_high = PlacementConfig(min_splats=50, max_splats=100, base_splat_density=0.1)
        placer_high = GradientGuidedPlacer((20, 20), config_high)
        high_complexity = np.ones((20, 20))
        _, count_high = placer_high.compute_adaptive_density(high_complexity)
        assert count_high == config_high.max_splats

    @patch('src.splat_this.core.placement_algorithm.peak_local_max')
    def test_detect_gradient_maxima_success(self, mock_peak_detection):
        """Test gradient maxima detection with successful peak detection."""
        # Mock peak detection to return specific peaks (as array of coordinates)
        mock_peak_detection.return_value = np.array([[5, 8], [15, 12]])

        placer = GradientGuidedPlacer((20, 20))
        image = np.random.rand(20, 20)

        maxima = placer.detect_gradient_maxima(image)

        assert len(maxima) == 2
        assert (5, 8) in maxima
        assert (15, 12) in maxima
        mock_peak_detection.assert_called_once()

    @patch('src.splat_this.core.placement_algorithm.peak_local_max')
    def test_detect_gradient_maxima_fallback(self, mock_peak_detection):
        """Test gradient maxima detection with fallback to scipy."""
        # Mock peak detection to fail
        mock_peak_detection.side_effect = Exception("Peak detection failed")

        placer = GradientGuidedPlacer((10, 10))

        # Create image with known maximum
        image = np.zeros((10, 10))
        image[5, 5] = 1.0  # Central maximum

        maxima = placer.detect_gradient_maxima(image)

        # Should fall back to scipy and detect some maxima
        assert isinstance(maxima, list)
        # Note: exact results depend on fallback implementation

    def test_detect_gradient_maxima_border_filtering(self):
        """Test border filtering in maxima detection."""
        config = PlacementConfig(border_margin=3)
        placer = GradientGuidedPlacer((10, 10), config)

        # Mock the peak detection to return border positions
        with patch('src.splat_this.core.placement_algorithm.peak_local_max') as mock_peak:
            # Return peaks near borders (as array of coordinates)
            mock_peak.return_value = np.array([[1, 1], [5, 5], [9, 9]])

            image = np.random.rand(10, 10)
            maxima = placer.detect_gradient_maxima(image)

            # Only the center peak (5, 5) should remain after border filtering
            assert len(maxima) == 1
            assert (5, 5) in maxima

    def test_create_placement_probability_map(self):
        """Test placement probability map creation."""
        placer = GradientGuidedPlacer((16, 16))

        image = np.random.rand(16, 16, 3)
        complexity_map = np.random.rand(16, 16)
        density_map = np.ones((16, 16)) * 0.01

        prob_map = placer.create_placement_probability_map(image, complexity_map, density_map)

        assert prob_map.shape == (16, 16)
        assert np.all(prob_map >= 0)
        assert np.abs(np.sum(prob_map) - 1.0) < 1e-6  # Should sum to 1

    def test_place_splats_basic(self):
        """Test basic splat placement."""
        placer = GradientGuidedPlacer((20, 20))

        # Simple test image
        image = np.random.rand(20, 20, 3)

        result = placer.place_splats(image, target_count=50)

        assert isinstance(result, PlacementResult)
        assert result.total_splats == 50
        assert len(result.positions) == 50
        assert len(result.normalized_positions) == 50

        # Check normalized coordinates are in [0, 1]
        for norm_y, norm_x in result.normalized_positions:
            assert 0 <= norm_y <= 1
            assert 0 <= norm_x <= 1

        # Check pixel coordinates are in bounds
        for y, x in result.positions:
            assert 0 <= y < 20
            assert 0 <= x < 20

    def test_place_splats_auto_count(self):
        """Test splat placement with automatic count determination."""
        config = PlacementConfig(base_splat_density=0.02, min_splats=10, max_splats=100)
        placer = GradientGuidedPlacer((15, 15), config)

        image = np.random.rand(15, 15, 3)

        result = placer.place_splats(image)  # No target_count specified

        assert 10 <= result.total_splats <= 100
        assert result.complexity_map.shape == (15, 15)
        assert result.density_map.shape == (15, 15)
        assert result.probability_map.shape == (15, 15)

    def test_place_splats_quality_metrics(self):
        """Test quality metrics computation in placement."""
        placer = GradientGuidedPlacer((12, 12))

        image = np.random.rand(12, 12, 3)

        result = placer.place_splats(image, target_count=30)

        # Quality metrics should be computed
        assert 0 <= result.coverage_achieved <= 1
        assert 0 <= result.distribution_uniformity <= 1
        assert 0 <= result.gradient_alignment <= 1

        # Placement statistics should be reasonable
        assert result.gradient_guided_splats >= 0
        assert result.uniform_coverage_splats >= 0
        assert result.maxima_based_splats >= 0

    def test_validate_placement_quality_pass(self):
        """Test placement quality validation with passing result."""
        config = PlacementConfig(coverage_target=0.5, distribution_uniformity_threshold=0.3)
        placer = GradientGuidedPlacer((10, 10), config)

        # Create result that should pass validation
        result = PlacementResult(
            positions=[(i, i) for i in range(60)],  # 60 splats
            normalized_positions=[(i/10, i/10) for i in range(60)],
            complexity_map=np.ones((10, 10)),
            density_map=np.ones((10, 10)),
            probability_map=np.ones((10, 10)) / 100,
            coverage_achieved=0.8,
            distribution_uniformity=0.7
        )

        validation = placer.validate_placement_quality(result)

        assert validation['passed'] == True
        assert len(validation['issues']) == 0
        assert 'metrics' in validation

    def test_validate_placement_quality_fail(self):
        """Test placement quality validation with failing result."""
        config = PlacementConfig(coverage_target=0.9, distribution_uniformity_threshold=0.8)
        placer = GradientGuidedPlacer((10, 10), config)

        # Create result that should fail validation
        result = PlacementResult(
            positions=[(5, 5)],  # Only 1 splat
            normalized_positions=[(0.5, 0.5)],
            complexity_map=np.ones((10, 10)),
            density_map=np.ones((10, 10)),
            probability_map=np.ones((10, 10)) / 100,
            coverage_achieved=0.2,  # Below target
            distribution_uniformity=0.1  # Below threshold
        )

        validation = placer.validate_placement_quality(result)

        assert validation['passed'] == False
        assert len(validation['issues']) >= 2  # Coverage and uniformity issues
        assert 'recommendations' in validation

    def test_compute_coverage(self):
        """Test coverage computation."""
        placer = GradientGuidedPlacer((20, 20))

        # Test with well-distributed positions
        positions = [(5, 5), (5, 15), (15, 5), (15, 15)]  # Corner-like distribution
        coverage = placer._compute_coverage(positions)

        assert 0 <= coverage <= 1
        assert coverage > 0  # Should have some coverage

        # Test with empty positions
        empty_coverage = placer._compute_coverage([])
        assert empty_coverage == 0.0

    def test_compute_distribution_uniformity(self):
        """Test distribution uniformity computation."""
        placer = GradientGuidedPlacer((16, 16))

        # Test with uniform distribution
        uniform_positions = [(i*2, j*2) for i in range(8) for j in range(8)]
        uniform_score = placer._compute_distribution_uniformity(uniform_positions)

        # Test with clustered distribution
        clustered_positions = [(8 + i//8, 8 + i%8) for i in range(64)]
        clustered_score = placer._compute_distribution_uniformity(clustered_positions)

        assert 0 <= uniform_score <= 1
        assert 0 <= clustered_score <= 1
        assert uniform_score > clustered_score  # Uniform should score higher

    def test_compute_gradient_alignment(self):
        """Test gradient alignment computation."""
        placer = GradientGuidedPlacer((10, 10))

        # Create image with known gradient structure
        image = np.zeros((10, 10))
        image[:, 5:] = 1.0  # Vertical edge at x=5

        # Positions on the edge should have higher alignment
        edge_positions = [(i, 5) for i in range(10)]
        edge_alignment = placer._compute_gradient_alignment(edge_positions, image)

        # Positions away from edge should have lower alignment
        non_edge_positions = [(i, 0) for i in range(10)]
        non_edge_alignment = placer._compute_gradient_alignment(non_edge_positions, image)

        assert 0 <= edge_alignment <= 1
        assert 0 <= non_edge_alignment <= 1
        assert edge_alignment > non_edge_alignment

    def test_count_maxima_aligned_splats(self):
        """Test counting of maxima-aligned splats."""
        config = PlacementConfig(maxima_min_distance=3)
        placer = GradientGuidedPlacer((20, 20), config)

        maxima_positions = [(5, 5), (15, 15)]
        splat_positions = [(4, 4), (5, 6), (10, 10), (16, 14)]  # Some close, some far

        aligned_count = placer._count_maxima_aligned_splats(splat_positions, maxima_positions)

        # Should count splats within distance threshold
        assert aligned_count >= 1  # At least (4,4) near (5,5) and (16,14) near (15,15)
        assert aligned_count <= len(splat_positions)

    def test_get_placement_statistics(self):
        """Test placement statistics collection."""
        placer = GradientGuidedPlacer((10, 10))

        result = PlacementResult(
            positions=[(i, i) for i in range(20)],
            normalized_positions=[(i/10, i/10) for i in range(20)],
            complexity_map=np.random.rand(10, 10),
            density_map=np.random.rand(10, 10),
            probability_map=np.ones((10, 10)) / 100,
            gradient_guided_splats=14,
            uniform_coverage_splats=6,
            maxima_based_splats=3
        )

        stats = placer.get_placement_statistics(result)

        expected_keys = [
            'total_splats', 'coverage_achieved', 'distribution_uniformity',
            'gradient_alignment', 'complexity_statistics', 'density_statistics',
            'spatial_distribution'
        ]

        for key in expected_keys:
            assert key in stats

        assert stats['total_splats'] == 20
        assert 'mean' in stats['complexity_statistics']
        assert 'gradient_guided_fraction' in stats['spatial_distribution']


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_create_gradient_guided_placer_default(self):
        """Test convenience function with default config."""
        placer = create_gradient_guided_placer((32, 64))

        assert isinstance(placer, GradientGuidedPlacer)
        assert placer.image_height == 32
        assert placer.image_width == 64
        assert placer.config.base_splat_density == 0.01

    def test_create_gradient_guided_placer_custom(self):
        """Test convenience function with custom config."""
        config = {'base_splat_density': 0.03, 'max_splats': 500}
        placer = create_gradient_guided_placer((16, 16), config)

        assert placer.config.base_splat_density == 0.03
        assert placer.config.max_splats == 500

    def test_place_adaptive_splats(self):
        """Test convenience function for adaptive placement."""
        image = np.random.rand(15, 15, 3)

        result = place_adaptive_splats(image, target_count=25)

        assert isinstance(result, PlacementResult)
        assert result.total_splats == 25

    def test_place_adaptive_splats_with_config(self):
        """Test adaptive placement with custom config."""
        image = np.random.rand(12, 12)
        config = {'gradient_weight': 0.9, 'uniform_weight': 0.1}

        result = place_adaptive_splats(image, config=config)

        assert isinstance(result, PlacementResult)
        assert result.total_splats > 0


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_very_small_image(self):
        """Test with very small image."""
        placer = GradientGuidedPlacer((5, 5))
        image = np.random.rand(5, 5, 3)

        result = placer.place_splats(image, target_count=3)

        assert result.total_splats == 3
        assert result.complexity_map.shape == (5, 5)

    def test_single_pixel_image(self):
        """Test with single pixel image."""
        placer = GradientGuidedPlacer((1, 1))
        image = np.array([[[0.5, 0.5, 0.5]]])

        result = placer.place_splats(image, target_count=1)

        assert result.total_splats == 1
        assert result.positions[0] == (0, 0)

    def test_zero_target_count(self):
        """Test with zero target count."""
        placer = GradientGuidedPlacer((10, 10))
        image = np.random.rand(10, 10, 3)

        result = placer.place_splats(image, target_count=0)

        assert result.total_splats == 0
        assert len(result.positions) == 0

    def test_uniform_image(self):
        """Test with uniform (no gradient) image."""
        placer = GradientGuidedPlacer((10, 10))
        image = np.ones((10, 10, 3)) * 0.5  # Uniform gray

        result = placer.place_splats(image, target_count=20)

        assert result.total_splats == 20
        # Should still place splats despite no gradients
        assert np.max(result.complexity_map) >= 0

    def test_high_contrast_image(self):
        """Test with high contrast image."""
        placer = GradientGuidedPlacer((12, 12))

        # Checkerboard pattern (high contrast)
        image = np.zeros((12, 12, 3))
        for i in range(12):
            for j in range(12):
                if (i + j) % 2 == 0:
                    image[i, j] = [1.0, 1.0, 1.0]

        result = placer.place_splats(image, target_count=30)

        assert result.total_splats == 30
        # High contrast should result in high complexity
        assert np.mean(result.complexity_map) > 0.1


if __name__ == "__main__":
    pytest.main([__file__])