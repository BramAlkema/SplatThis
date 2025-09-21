"""Unit tests for error computation and analysis framework."""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from src.splat_this.core.error_analysis import (
    ErrorMetrics,
    ErrorRegion,
    ErrorAnalyzer,
    compute_reconstruction_error,
    create_error_visualization
)


class TestErrorMetrics:
    """Test ErrorMetrics dataclass."""

    def test_init_basic(self):
        """Test basic ErrorMetrics initialization."""
        metrics = ErrorMetrics(l1_error=0.1, l2_error=0.05)

        assert metrics.l1_error == 0.1
        assert metrics.l2_error == 0.05
        assert metrics.mae == 0.1  # Alias
        assert metrics.mse == 0.05  # Alias

    def test_post_init_derived_metrics(self):
        """Test automatic computation of derived metrics."""
        metrics = ErrorMetrics(l2_error=0.04)  # MSE = 0.04

        assert abs(metrics.rmse - 0.2) < 1e-6  # sqrt(0.04) = 0.2
        assert abs(metrics.psnr - 13.979) < 0.01  # -10*log10(0.04)

    def test_post_init_zero_mse(self):
        """Test PSNR computation with zero MSE."""
        metrics = ErrorMetrics(l2_error=0.0)

        assert metrics.rmse == 0.0
        assert metrics.psnr == float('inf')

    def test_to_dict(self):
        """Test dictionary conversion."""
        metrics = ErrorMetrics(
            l1_error=0.1,
            l2_error=0.05,
            ssim_score=0.8,
            coverage_ratio=0.9
        )

        data = metrics.to_dict()

        expected_keys = [
            'l1_error', 'l2_error', 'rmse', 'ssim_score', 'psnr',
            'edge_error', 'smooth_error', 'gradient_error',
            'coverage_ratio', 'alpha_mean', 'alpha_std'
        ]

        for key in expected_keys:
            assert key in data
            assert isinstance(data[key], (int, float))


class TestErrorRegion:
    """Test ErrorRegion dataclass."""

    def test_init_basic(self):
        """Test basic ErrorRegion initialization."""
        region = ErrorRegion(
            center=(10, 20),
            bbox=(5, 15, 15, 25),
            area=100,
            mean_error=0.3,
            max_error=0.8,
            error_type='edge',
            priority=30.0
        )

        assert region.center == (10, 20)
        assert region.bbox == (5, 15, 15, 25)
        assert region.area == 100
        assert region.error_type == 'edge'

    def test_size_property(self):
        """Test region size computation."""
        region = ErrorRegion(
            center=(0, 0),
            bbox=(10, 20, 30, 50),  # height=20, width=30
            area=0,
            mean_error=0.0,
            max_error=0.0,
            error_type='general',
            priority=0.0
        )

        size = region.size
        assert size == (20, 30)  # (height, width)


class TestErrorAnalyzer:
    """Test ErrorAnalyzer class."""

    def test_init_default(self):
        """Test default initialization."""
        analyzer = ErrorAnalyzer()

        assert analyzer.window_size == 7
        assert analyzer.edge_threshold == 0.1
        assert analyzer.error_history == []

    def test_init_custom(self):
        """Test custom initialization."""
        analyzer = ErrorAnalyzer(window_size=11, edge_threshold=0.2)

        assert analyzer.window_size == 11
        assert analyzer.edge_threshold == 0.2

    def test_prepare_images_grayscale(self):
        """Test image preparation with grayscale images."""
        analyzer = ErrorAnalyzer()

        target = np.random.rand(10, 10)
        rendered = np.random.rand(10, 10)

        target_rgb, rendered_rgb = analyzer._prepare_images(target, rendered)

        assert target_rgb.shape == (10, 10, 1)
        assert rendered_rgb.shape == (10, 10, 1)
        assert target_rgb.dtype == np.float32
        assert rendered_rgb.dtype == np.float32

    def test_prepare_images_color(self):
        """Test image preparation with color images."""
        analyzer = ErrorAnalyzer()

        target = np.random.rand(10, 10, 3)
        rendered = np.random.rand(10, 10, 3)

        target_rgb, rendered_rgb = analyzer._prepare_images(target, rendered)

        assert target_rgb.shape == (10, 10, 3)
        assert rendered_rgb.shape == (10, 10, 3)

    def test_prepare_images_rgba(self):
        """Test image preparation with RGBA rendered image."""
        analyzer = ErrorAnalyzer()

        target = np.random.rand(10, 10, 3)
        rendered = np.random.rand(10, 10, 4)  # RGBA

        target_rgb, rendered_rgb = analyzer._prepare_images(target, rendered)

        assert target_rgb.shape == (10, 10, 3)
        assert rendered_rgb.shape == (10, 10, 3)  # Alpha dropped

    def test_prepare_images_mixed_channels(self):
        """Test image preparation with different channel counts."""
        analyzer = ErrorAnalyzer()

        target = np.random.rand(10, 10, 1)  # Grayscale
        rendered = np.random.rand(10, 10, 3)  # Color

        target_rgb, rendered_rgb = analyzer._prepare_images(target, rendered)

        assert target_rgb.shape == (10, 10, 3)  # Expanded to color
        assert rendered_rgb.shape == (10, 10, 3)

    def test_prepare_images_dimension_mismatch(self):
        """Test error handling for dimension mismatch."""
        analyzer = ErrorAnalyzer()

        target = np.random.rand(10, 10, 3)
        rendered = np.random.rand(8, 12, 3)  # Different spatial dimensions

        with pytest.raises(ValueError, match="Image dimensions must match"):
            analyzer._prepare_images(target, rendered)

    def test_compute_basic_metrics_perfect_match(self):
        """Test basic metrics with perfect reconstruction."""
        analyzer = ErrorAnalyzer()

        target = np.random.rand(20, 20, 3)
        rendered = target.copy()  # Perfect match

        metrics = analyzer.compute_basic_metrics(target, rendered)

        assert abs(metrics.l1_error) < 1e-6
        assert abs(metrics.l2_error) < 1e-6
        assert abs(metrics.rmse) < 1e-6

    def test_compute_basic_metrics_with_difference(self):
        """Test basic metrics with known difference."""
        analyzer = ErrorAnalyzer()

        target = np.zeros((10, 10, 3))
        rendered = np.ones((10, 10, 3)) * 0.1  # Uniform 0.1 error

        metrics = analyzer.compute_basic_metrics(target, rendered)

        assert abs(metrics.l1_error - 0.1) < 1e-6
        assert abs(metrics.l2_error - 0.01) < 1e-6
        assert abs(metrics.rmse - 0.1) < 1e-6

    def test_compute_basic_metrics_with_alpha(self):
        """Test basic metrics with alpha channel."""
        analyzer = ErrorAnalyzer()

        target = np.random.rand(10, 10, 3)
        rendered = np.random.rand(10, 10, 4)
        rendered[:, :, 3] = 0.8  # Set alpha

        metrics = analyzer.compute_basic_metrics(target, rendered)

        assert metrics.coverage_ratio == 1.0  # All pixels above threshold
        assert abs(metrics.alpha_mean - 0.8) < 1e-6  # Allow for floating point precision
        assert metrics.alpha_std < 1e-10  # Should be very close to zero

    def test_compute_basic_metrics_with_mask(self):
        """Test basic metrics with mask."""
        analyzer = ErrorAnalyzer()

        target = np.zeros((10, 10, 3))
        rendered = np.ones((10, 10, 3))
        mask = np.zeros((10, 10))
        mask[2:8, 2:8] = 1.0  # Only center region

        metrics = analyzer.compute_basic_metrics(target, rendered, mask)

        # Should only compute error in masked region
        assert metrics.l1_error > 0  # Non-zero error in center
        assert metrics.l2_error > 0

    @patch('src.splat_this.core.error_analysis.ssim')
    def test_compute_ssim_success(self, mock_ssim):
        """Test SSIM computation success."""
        mock_ssim.return_value = 0.85

        analyzer = ErrorAnalyzer()
        target = np.random.rand(20, 20, 3)
        rendered = np.random.rand(20, 20, 3)

        ssim_score = analyzer.compute_ssim(target, rendered)

        assert ssim_score == 0.85
        mock_ssim.assert_called_once()

    @patch('src.splat_this.core.error_analysis.ssim')
    def test_compute_ssim_failure(self, mock_ssim):
        """Test SSIM computation failure handling."""
        mock_ssim.side_effect = Exception("SSIM failed")

        analyzer = ErrorAnalyzer()
        target = np.random.rand(10, 10, 3)
        rendered = np.random.rand(10, 10, 3)

        ssim_score = analyzer.compute_ssim(target, rendered)

        assert ssim_score == 0.0  # Fallback value

    def test_gradient_edges_fallback(self):
        """Test gradient-based edge detection fallback."""
        analyzer = ErrorAnalyzer()

        # Create image with vertical edge
        image = np.zeros((20, 20))
        image[:, 10:] = 1.0

        edges = analyzer._gradient_edges(image, threshold=0.1)

        assert edges.shape == image.shape
        assert edges.dtype == bool
        assert np.any(edges)  # Should detect some edges

    @patch('skimage.feature.canny')
    def test_compute_perceptual_metrics_with_canny(self, mock_canny):
        """Test perceptual metrics with Canny edge detection."""
        # Mock Canny edge detection
        mock_canny.side_effect = lambda img, **kwargs: np.random.rand(*img.shape) > 0.8

        analyzer = ErrorAnalyzer()
        target = np.random.rand(15, 15, 3)
        rendered = np.random.rand(15, 15, 3)

        metrics = analyzer.compute_perceptual_metrics(target, rendered)

        expected_keys = ['edge_error', 'smooth_error', 'gradient_error']
        for key in expected_keys:
            assert key in metrics
            assert isinstance(metrics[key], (float, np.floating))  # Allow numpy float types
            assert metrics[key] >= 0

    def test_compute_perceptual_metrics_fallback(self):
        """Test perceptual metrics with fallback edge detection."""
        analyzer = ErrorAnalyzer()

        # Create simple test images
        target = np.zeros((15, 15, 3))
        target[7, :, :] = 1.0  # Horizontal edge

        rendered = np.zeros((15, 15, 3))
        rendered[8, :, :] = 1.0  # Shifted edge

        with patch('skimage.feature.canny', side_effect=ImportError):
            metrics = analyzer.compute_perceptual_metrics(target, rendered)

        assert 'edge_error' in metrics
        assert 'smooth_error' in metrics
        assert 'gradient_error' in metrics

    def test_create_error_map_l1(self):
        """Test L1 error map creation."""
        analyzer = ErrorAnalyzer()

        target = np.zeros((10, 10, 3))
        rendered = np.ones((10, 10, 3)) * 0.2

        error_map = analyzer.create_error_map(target, rendered, 'l1')

        assert error_map.shape == (10, 10)
        assert np.allclose(error_map, 0.2)  # Uniform 0.2 error

    def test_create_error_map_l2(self):
        """Test L2 error map creation."""
        analyzer = ErrorAnalyzer()

        target = np.zeros((10, 10, 3))
        rendered = np.ones((10, 10, 3)) * 0.1

        error_map = analyzer.create_error_map(target, rendered, 'l2')

        assert error_map.shape == (10, 10)
        assert np.allclose(error_map, 0.01)  # (0.1)^2 = 0.01

    def test_create_error_map_ssim_local(self):
        """Test local SSIM error map creation."""
        analyzer = ErrorAnalyzer()

        target = np.random.rand(20, 20, 3)
        rendered = target + 0.1  # Small difference

        error_map = analyzer.create_error_map(target, rendered, 'ssim_local')

        assert error_map.shape == (20, 20)
        assert np.all(error_map >= 0)  # SSIM error should be non-negative

    def test_create_error_map_invalid_type(self):
        """Test error map creation with invalid type."""
        analyzer = ErrorAnalyzer()

        target = np.random.rand(10, 10, 3)
        rendered = np.random.rand(10, 10, 3)

        with pytest.raises(ValueError, match="Unknown error type"):
            analyzer.create_error_map(target, rendered, 'invalid')

    def test_detect_high_error_regions_simple(self):
        """Test high-error region detection with simple pattern."""
        analyzer = ErrorAnalyzer()

        # Create error map with high-error region in center
        error_map = np.zeros((20, 20))
        error_map[8:12, 8:12] = 1.0  # 4x4 high-error region

        regions = analyzer.detect_high_error_regions(error_map, threshold=0.5, min_area=10)

        assert len(regions) == 1
        region = regions[0]
        assert region.area == 16  # 4x4 = 16 pixels
        assert region.center[0] in [9, 10]  # Center row
        assert region.center[1] in [9, 10]  # Center column

    def test_detect_high_error_regions_auto_threshold(self):
        """Test high-error region detection with auto threshold."""
        analyzer = ErrorAnalyzer()

        # Create error map with outliers
        error_map = np.random.normal(0.1, 0.02, (30, 30))  # Low baseline error
        error_map[10:15, 10:15] = 0.5  # High-error region

        regions = analyzer.detect_high_error_regions(error_map, threshold=None, min_area=5)

        assert len(regions) >= 1  # Should detect at least the high-error region

    def test_detect_high_error_regions_min_area_filter(self):
        """Test region filtering by minimum area."""
        analyzer = ErrorAnalyzer()

        # Create small scattered high-error pixels
        error_map = np.zeros((20, 20))
        error_map[5, 5] = 1.0  # Single pixel
        error_map[10:12, 10:12] = 1.0  # 2x2 region (4 pixels)

        regions = analyzer.detect_high_error_regions(error_map, threshold=0.5, min_area=5)

        assert len(regions) == 0  # Both regions too small (< 5 pixels)

    def test_track_error_history(self):
        """Test error history tracking."""
        analyzer = ErrorAnalyzer()

        metrics1 = ErrorMetrics(l1_error=0.1, l2_error=0.01)
        metrics2 = ErrorMetrics(l1_error=0.08, l2_error=0.008)

        analyzer.track_error_history(metrics1)
        analyzer.track_error_history(metrics2)

        assert len(analyzer.error_history) == 2
        assert analyzer.error_history[0].l1_error == 0.1
        assert analyzer.error_history[1].l1_error == 0.08

    def test_track_error_history_max_length(self):
        """Test error history maximum length enforcement."""
        analyzer = ErrorAnalyzer()

        # Add many metrics (more than max)
        for i in range(1100):  # More than max_history=1000
            metrics = ErrorMetrics(l1_error=i * 0.001)
            analyzer.track_error_history(metrics)

        assert len(analyzer.error_history) == 1000  # Should be capped

    def test_analyze_convergence_insufficient_data(self):
        """Test convergence analysis with insufficient data."""
        analyzer = ErrorAnalyzer()

        # Add only a few metrics
        for i in range(5):
            metrics = ErrorMetrics(l1_error=0.1 - i * 0.01)
            analyzer.track_error_history(metrics)

        convergence = analyzer.analyze_convergence(window_size=10)

        assert convergence['converged'] == False
        assert convergence['trend'] == 'insufficient_data'

    def test_analyze_convergence_decreasing_trend(self):
        """Test convergence analysis with decreasing error trend."""
        analyzer = ErrorAnalyzer()

        # Add decreasing error metrics with slope > 1e-6 threshold
        for i in range(15):
            # Start high, decrease significantly (slope will be negative > 1e-6 in magnitude)
            metrics = ErrorMetrics(l1_error=0.1, l2_error=0.1 - i * 0.002)  # Use l2_error for trend
            analyzer.track_error_history(metrics)

        convergence = analyzer.analyze_convergence(window_size=10)

        assert convergence['trend'] == 'decreasing'
        assert convergence['slope'] < -1e-6  # Verify slope is below threshold

    def test_analyze_convergence_plateau(self):
        """Test convergence analysis with plateau detection."""
        analyzer = ErrorAnalyzer()

        # Add completely constant error metrics (perfect plateau)
        base_error = 0.05
        for i in range(15):
            # Use exactly constant values for perfect plateau
            metrics = ErrorMetrics(l1_error=0.1, l2_error=base_error)  # No variation
            analyzer.track_error_history(metrics)

        convergence = analyzer.analyze_convergence(window_size=10)

        # Check plateau detection criteria
        assert convergence['plateau_detected'] == True
        assert convergence['trend'] == 'stable'
        assert abs(convergence['slope']) <= 1e-6  # Should be very close to zero

    def test_create_quality_report_basic(self):
        """Test basic quality report creation."""
        analyzer = ErrorAnalyzer()

        target = np.random.rand(20, 20, 3)
        rendered = target + 0.1  # Add some error

        report = analyzer.create_quality_report(target, rendered, include_regions=False)

        expected_keys = ['metrics', 'error_statistics']
        for key in expected_keys:
            assert key in report

        assert 'l1_error' in report['metrics']
        assert 'l1_map_mean' in report['error_statistics']

    def test_create_quality_report_with_regions(self):
        """Test quality report with region analysis."""
        analyzer = ErrorAnalyzer()

        # Create target with clear structure
        target = np.zeros((30, 30, 3))
        target[10:20, 10:20, :] = 1.0  # White square

        # Create rendered with error
        rendered = target.copy()
        rendered[12:18, 12:18, :] = 0.5  # Gray error region

        report = analyzer.create_quality_report(target, rendered, include_regions=True)

        assert 'high_error_regions' in report
        assert 'count' in report['high_error_regions']
        assert 'regions' in report['high_error_regions']

    def test_create_quality_report_with_convergence(self):
        """Test quality report with convergence analysis."""
        analyzer = ErrorAnalyzer()

        # Add some history first
        for i in range(5):
            metrics = ErrorMetrics(l1_error=0.1 - i * 0.01)
            analyzer.track_error_history(metrics)

        target = np.random.rand(15, 15, 3)
        rendered = target + 0.05

        report = analyzer.create_quality_report(target, rendered)

        assert 'convergence' in report
        assert 'trend' in report['convergence']


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_compute_reconstruction_error(self):
        """Test convenience function for reconstruction error."""
        target = np.random.rand(10, 10, 3)
        rendered = target + 0.1

        metrics = compute_reconstruction_error(target, rendered)

        assert isinstance(metrics, ErrorMetrics)
        assert metrics.l1_error > 0
        assert metrics.ssim_score >= 0

    def test_create_error_visualization(self):
        """Test convenience function for error visualization."""
        target = np.random.rand(10, 10, 3)
        rendered = target + 0.1

        visualization = create_error_visualization(target, rendered, 'l1')

        expected_keys = ['error_map', 'target', 'rendered', 'difference']
        for key in expected_keys:
            assert key in visualization

        assert visualization['error_map'].shape == (10, 10)


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_single_pixel_image(self):
        """Test with single pixel image."""
        analyzer = ErrorAnalyzer()

        target = np.array([[[0.5, 0.5, 0.5]]])  # 1x1x3
        rendered = np.array([[[0.7, 0.6, 0.8]]])  # 1x1x3

        metrics = analyzer.compute_basic_metrics(target, rendered)

        assert metrics.l1_error > 0
        assert metrics.l2_error > 0

    def test_zero_images(self):
        """Test with all-zero images."""
        analyzer = ErrorAnalyzer()

        target = np.zeros((10, 10, 3))
        rendered = np.zeros((10, 10, 3))

        metrics = analyzer.compute_basic_metrics(target, rendered)

        assert metrics.l1_error == 0.0
        assert metrics.l2_error == 0.0
        assert metrics.rmse == 0.0

    def test_large_values(self):
        """Test with large pixel values (should be clipped)."""
        analyzer = ErrorAnalyzer()

        target = np.ones((5, 5, 3)) * 2.0  # Values > 1.0
        rendered = np.ones((5, 5, 3)) * -0.5  # Values < 0.0

        target_prep, rendered_prep = analyzer._prepare_images(target, rendered)

        assert np.all(target_prep <= 1.0)
        assert np.all(target_prep >= 0.0)
        assert np.all(rendered_prep <= 1.0)
        assert np.all(rendered_prep >= 0.0)

    def test_empty_error_map(self):
        """Test high-error region detection with empty error map."""
        analyzer = ErrorAnalyzer()

        error_map = np.zeros((10, 10))  # No errors

        regions = analyzer.detect_high_error_regions(error_map, threshold=0.1)

        assert len(regions) == 0

    def test_uniform_error_map(self):
        """Test with uniform error map (no regions)."""
        analyzer = ErrorAnalyzer()

        error_map = np.ones((10, 10)) * 0.1  # Uniform low error

        regions = analyzer.detect_high_error_regions(error_map, threshold=0.5)

        assert len(regions) == 0


if __name__ == "__main__":
    pytest.main([__file__])