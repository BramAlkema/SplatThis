"""Unit tests for SaliencyAnalyzer in adaptive splat extraction."""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from splat_this.core.adaptive_extract import SaliencyAnalyzer, AdaptiveSplatConfig


class TestSaliencyAnalyzer:
    """Test the SaliencyAnalyzer class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = AdaptiveSplatConfig(
            edge_weight=0.4,
            variance_weight=0.3,
            gradient_weight=0.3
        )
        self.analyzer = SaliencyAnalyzer(self.config)

        # Create test images
        self.test_image_rgb = self._create_test_image_rgb()
        self.test_image_gray = self._create_test_image_gray()
        self.complex_image = self._create_complex_test_image()

    def _create_test_image_rgb(self) -> np.ndarray:
        """Create a simple RGB test image."""
        image = np.zeros((50, 50, 3), dtype=np.uint8)

        # Add some features
        image[10:20, 10:20] = [255, 0, 0]  # Red square
        image[30:40, 30:40] = [0, 255, 0]  # Green square
        image[15:25, 35:45] = [0, 0, 255]  # Blue square

        return image

    def _create_test_image_gray(self) -> np.ndarray:
        """Create a simple grayscale test image."""
        image = np.zeros((50, 50), dtype=np.uint8)

        # Add some features
        image[10:20, 10:20] = 200  # Bright square
        image[30:40, 30:40] = 150  # Medium square

        return image

    def _create_complex_test_image(self) -> np.ndarray:
        """Create a complex test image with various features."""
        image = np.random.randint(0, 50, (100, 100, 3), dtype=np.uint8)

        # Add high-contrast edges
        image[20:30, :] = 255  # Horizontal stripe
        image[:, 40:50] = 255  # Vertical stripe

        # Add gradient
        for i in range(60, 80):
            image[i, 60:80] = int((i - 60) * 255 / 20)

        # Add textured region
        texture_region = np.random.randint(100, 200, (15, 15, 3))
        image[70:85, 10:25] = texture_region

        return image

    def test_initialization(self):
        """Test SaliencyAnalyzer initialization."""
        assert self.analyzer.config == self.config
        assert self.analyzer.config.edge_weight == 0.4
        assert self.analyzer.config.variance_weight == 0.3
        assert self.analyzer.config.gradient_weight == 0.3

    def test_compute_saliency_map_rgb(self):
        """Test saliency map computation with RGB image."""
        saliency_map = self.analyzer.compute_saliency_map(self.test_image_rgb)

        # Check output shape and type
        assert saliency_map.shape == (50, 50)
        assert saliency_map.dtype == np.float64

        # Check value range
        assert 0 <= saliency_map.min() <= saliency_map.max() <= 1

        # Check that edges have higher saliency
        # The edges of our colored squares should have higher saliency
        center_saliency = saliency_map[25, 25]  # Empty area
        edge_saliency = saliency_map[10, 10]   # Edge of red square

        # Edge regions should generally have higher saliency than empty regions
        assert np.mean(saliency_map[8:12, 8:12]) > center_saliency

    def test_compute_saliency_map_grayscale(self):
        """Test saliency map computation with grayscale image."""
        saliency_map = self.analyzer.compute_saliency_map(self.test_image_gray)

        # Check output shape and type
        assert saliency_map.shape == (50, 50)
        assert saliency_map.dtype == np.float64

        # Check value range
        assert 0 <= saliency_map.min() <= saliency_map.max() <= 1

    def test_compute_local_variance(self):
        """Test local variance computation."""
        # Use grayscale image for simplicity
        variance_map = self.analyzer._compute_local_variance(
            self.test_image_gray.astype(np.float64),
            window_size=5
        )

        # Check output shape
        assert variance_map.shape == self.test_image_gray.shape

        # Check that variance is non-negative
        assert np.all(variance_map >= 0)

        # Check that uniform regions have low variance
        # The center of our bright square should have low variance
        uniform_region_variance = variance_map[15, 15]
        edge_region_variance = variance_map[10, 10]  # Edge of square

        # Edges should generally have higher variance than uniform regions
        assert edge_region_variance >= uniform_region_variance

    def test_local_variance_performance(self):
        """Test that local variance computation is efficient."""
        import time

        # Create larger image for performance test
        large_image = np.random.rand(200, 200)

        start_time = time.time()
        variance_map = self.analyzer._compute_local_variance(large_image, window_size=7)
        end_time = time.time()

        # Should complete in reasonable time (less than 1 second)
        assert end_time - start_time < 1.0
        assert variance_map.shape == large_image.shape

    def test_detect_saliency_peaks(self):
        """Test saliency peak detection."""
        # Create a saliency map with known peaks
        saliency_map = np.zeros((50, 50))
        saliency_map[10, 10] = 0.8  # High peak
        saliency_map[30, 30] = 0.6  # Medium peak
        saliency_map[40, 40] = 0.9  # Highest peak

        # Apply some smoothing to make realistic peaks
        from skimage.filters import gaussian
        saliency_map = gaussian(saliency_map, sigma=1.0)

        peaks = self.analyzer.detect_saliency_peaks(
            saliency_map,
            min_distance=5,
            threshold_abs=0.1  # Lower threshold to detect smoothed peaks
        )

        # Should detect peaks (exact coordinates may vary due to smoothing)
        assert len(peaks) >= 2  # Should find at least the major peaks
        assert all(isinstance(coord, tuple) and len(coord) == 2 for coord in peaks)

        # Peaks should be sorted by importance (highest first)
        if len(peaks) >= 2:
            peak1_value = saliency_map[peaks[0][0], peaks[0][1]]
            peak2_value = saliency_map[peaks[1][0], peaks[1][1]]
            assert peak1_value >= peak2_value

    def test_compute_multi_scale_saliency(self):
        """Test multi-scale saliency computation."""
        multi_scale_saliency = self.analyzer.compute_multi_scale_saliency(
            self.complex_image,
            scales=[0.5, 1.0, 2.0]
        )

        # Check output shape
        assert multi_scale_saliency.shape == self.complex_image.shape[:2]

        # Check value range
        assert 0 <= multi_scale_saliency.min() <= multi_scale_saliency.max() <= 1

        # Multi-scale should detect features at different scales
        assert np.std(multi_scale_saliency) > 0  # Should have variation

    def test_analyze_content_complexity(self):
        """Test content complexity analysis."""
        complexity_metrics = self.analyzer.analyze_content_complexity(self.complex_image)

        # Check that all expected metrics are present
        expected_keys = [
            'global_variance', 'edge_density', 'variance_mean', 'variance_std',
            'gradient_mean', 'gradient_std', 'texture_measure', 'complexity_score'
        ]
        for key in expected_keys:
            assert key in complexity_metrics
            assert isinstance(complexity_metrics[key], float)
            assert complexity_metrics[key] >= 0

        # Complexity score should be between 0 and 1
        assert 0 <= complexity_metrics['complexity_score'] <= 1

        # Test with simple image for comparison
        simple_complexity = self.analyzer.analyze_content_complexity(self.test_image_rgb)

        # Complex image should have higher complexity score
        assert complexity_metrics['complexity_score'] > simple_complexity['complexity_score']

    def test_saliency_weight_configuration(self):
        """Test that saliency weights are properly applied."""
        # Test with edge-heavy weighting
        edge_config = AdaptiveSplatConfig(
            edge_weight=0.8,
            variance_weight=0.1,
            gradient_weight=0.1
        )
        edge_analyzer = SaliencyAnalyzer(edge_config)
        edge_saliency = edge_analyzer.compute_saliency_map(self.complex_image)

        # Test with variance-heavy weighting
        variance_config = AdaptiveSplatConfig(
            edge_weight=0.1,
            variance_weight=0.8,
            gradient_weight=0.1
        )
        variance_analyzer = SaliencyAnalyzer(variance_config)
        variance_saliency = variance_analyzer.compute_saliency_map(self.complex_image)

        # The results should be different
        assert not np.allclose(edge_saliency, variance_saliency, rtol=0.1)

    def test_saliency_with_uniform_image(self):
        """Test saliency computation with uniform image."""
        uniform_image = np.full((50, 50, 3), 128, dtype=np.uint8)
        saliency_map = self.analyzer.compute_saliency_map(uniform_image)

        # Uniform image should have low saliency everywhere
        assert np.all(saliency_map >= 0)
        assert np.std(saliency_map) < 0.1  # Very low variation

    def test_saliency_with_edge_cases(self):
        """Test saliency computation with edge cases."""
        # Test with very small image
        tiny_image = np.random.randint(0, 255, (5, 5, 3), dtype=np.uint8)
        tiny_saliency = self.analyzer.compute_saliency_map(tiny_image)
        assert tiny_saliency.shape == (5, 5)

        # Test with single channel image
        single_channel = np.random.randint(0, 255, (20, 20), dtype=np.uint8)
        single_saliency = self.analyzer.compute_saliency_map(single_channel)
        assert single_saliency.shape == (20, 20)

    def test_peak_detection_edge_cases(self):
        """Test peak detection with edge cases."""
        # Test with no peaks
        flat_saliency = np.full((30, 30), 0.2)
        peaks = self.analyzer.detect_saliency_peaks(flat_saliency, threshold_abs=0.5)
        assert len(peaks) == 0

        # Test with single peak
        single_peak_saliency = np.zeros((30, 30))
        single_peak_saliency[15, 15] = 1.0
        peaks = self.analyzer.detect_saliency_peaks(single_peak_saliency, threshold_abs=0.5)
        assert len(peaks) >= 1

    def test_multi_scale_with_default_scales(self):
        """Test multi-scale saliency with default scale parameters."""
        # Should work with default scales
        multi_scale = self.analyzer.compute_multi_scale_saliency(self.test_image_rgb)
        assert multi_scale.shape == (50, 50)
        assert 0 <= multi_scale.min() <= multi_scale.max() <= 1

    @patch('scipy.ndimage.uniform_filter')
    def test_local_variance_calls_uniform_filter(self, mock_uniform_filter):
        """Test that local variance uses uniform_filter for efficiency."""
        mock_uniform_filter.return_value = np.zeros((10, 10))

        test_image = np.random.rand(10, 10)
        self.analyzer._compute_local_variance(test_image, window_size=3)

        # Should call uniform_filter twice (for mean and mean of squares)
        assert mock_uniform_filter.call_count == 2

    def test_memory_efficiency(self):
        """Test memory efficiency with large images."""
        # Create a reasonably large image
        large_image = np.random.randint(0, 255, (500, 500, 3), dtype=np.uint8)

        # These operations should complete without memory errors
        saliency_map = self.analyzer.compute_saliency_map(large_image)
        assert saliency_map.shape == (500, 500)

        complexity = self.analyzer.analyze_content_complexity(large_image)
        assert 'complexity_score' in complexity

    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        # Test with high contrast image
        high_contrast = np.zeros((30, 30, 3), dtype=np.uint8)
        high_contrast[:15, :] = 255

        saliency = self.analyzer.compute_saliency_map(high_contrast)
        assert np.all(np.isfinite(saliency))
        assert np.all(saliency >= 0)

        # Test with low contrast image
        low_contrast = np.full((30, 30, 3), 128, dtype=np.uint8)
        noise = np.random.randint(-2, 3, (30, 30, 3)).astype(np.int16)
        low_contrast = np.clip(low_contrast.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        saliency_low = self.analyzer.compute_saliency_map(low_contrast)
        assert np.all(np.isfinite(saliency_low))
        assert np.all(saliency_low >= 0)