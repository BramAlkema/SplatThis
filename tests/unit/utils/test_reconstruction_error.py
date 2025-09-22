#!/usr/bin/env python3
"""Unit tests for reconstruction error utilities."""

import pytest
import numpy as np
from src.splat_this.utils.reconstruction_error import (
    compute_l1_error,
    compute_l2_error,
    compute_mse_error,
    compute_weighted_error,
    compute_error_statistics,
    compute_psnr,
    compute_reconstruction_error,
    validate_images,
    _normalize_image
)


class TestL1Error:
    """Test cases for L1 error computation."""

    def test_perfect_match(self):
        """Test L1 error for identical images."""
        image = np.array([[0.1, 0.5], [0.8, 0.2]], dtype=np.float32)
        error = compute_l1_error(image, image)
        assert error.shape == (2, 2)
        assert np.allclose(error, 0.0)

    def test_known_values_grayscale(self):
        """Test L1 error with known values for grayscale."""
        target = np.array([[1.0, 0.0], [0.5, 0.7]], dtype=np.float32)
        rendered = np.array([[0.0, 1.0], [0.5, 0.2]], dtype=np.float32)
        error = compute_l1_error(target, rendered, normalize=False)

        expected = np.array([[1.0, 1.0], [0.0, 0.5]], dtype=np.float32)
        assert np.allclose(error, expected)

    def test_known_values_rgb(self):
        """Test L1 error with known values for RGB."""
        target = np.zeros((2, 2, 3), dtype=np.float32)
        target[0, 0] = [1.0, 0.0, 0.0]  # Red
        target[1, 1] = [0.0, 1.0, 0.0]  # Green

        rendered = np.zeros((2, 2, 3), dtype=np.float32)
        rendered[0, 0] = [0.0, 1.0, 0.0]  # Green instead of red
        rendered[1, 1] = [0.0, 1.0, 0.0]  # Correct green

        error = compute_l1_error(target, rendered, normalize=False)

        # Error at (0,0) should be mean of [1.0, 1.0, 0.0] = 2/3
        # Error at (1,1) should be 0
        assert error.shape == (2, 2)
        assert np.isclose(error[0, 0], 2.0/3.0)
        assert np.isclose(error[1, 1], 0.0)

    def test_uint8_normalization(self):
        """Test L1 error with uint8 images."""
        target = np.array([[255, 0], [127, 255]], dtype=np.uint8)
        rendered = np.array([[0, 255], [127, 0]], dtype=np.uint8)

        error = compute_l1_error(target, rendered)

        # Expected normalized errors: [1.0, 1.0], [0.0, 1.0]
        expected = np.array([[1.0, 1.0], [0.0, 1.0]], dtype=np.float32)
        assert np.allclose(error, expected)

    def test_shape_mismatch(self):
        """Test error handling for mismatched shapes."""
        target = np.zeros((2, 2))
        rendered = np.zeros((3, 3))

        with pytest.raises(ValueError, match="Target and rendered images must have same shape"):
            compute_l1_error(target, rendered)


class TestL2Error:
    """Test cases for L2 error computation."""

    def test_perfect_match(self):
        """Test L2 error for identical images."""
        image = np.array([[0.1, 0.5], [0.8, 0.2]], dtype=np.float32)
        error = compute_l2_error(image, image)
        assert error.shape == (2, 2)
        assert np.allclose(error, 0.0)

    def test_known_values_grayscale(self):
        """Test L2 error with known values for grayscale."""
        target = np.array([[1.0, 0.0]], dtype=np.float32)
        rendered = np.array([[0.0, 1.0]], dtype=np.float32)

        error = compute_l2_error(target, rendered, normalize=False)
        expected = np.array([[1.0, 1.0]], dtype=np.float32)
        assert np.allclose(error, expected)

    def test_known_values_rgb(self):
        """Test L2 error with known values for RGB."""
        target = np.array([[[1.0, 0.0, 0.0]]], dtype=np.float32)  # Red
        rendered = np.array([[[0.0, 1.0, 0.0]]], dtype=np.float32)  # Green

        error = compute_l2_error(target, rendered, normalize=False)
        # L2 distance = sqrt(1^2 + 1^2 + 0^2) = sqrt(2)
        expected = np.array([[np.sqrt(2.0)]], dtype=np.float32)
        assert np.allclose(error, expected)

    def test_shape_mismatch(self):
        """Test error handling for mismatched shapes."""
        target = np.zeros((2, 2))
        rendered = np.zeros((3, 3))

        with pytest.raises(ValueError, match="Target and rendered images must have same shape"):
            compute_l2_error(target, rendered)


class TestMSEError:
    """Test cases for MSE error computation."""

    def test_perfect_match(self):
        """Test MSE error for identical images."""
        image = np.array([[0.1, 0.5], [0.8, 0.2]], dtype=np.float32)
        error = compute_mse_error(image, image)
        assert error.shape == (2, 2)
        assert np.allclose(error, 0.0)

    def test_known_values(self):
        """Test MSE error with known values."""
        target = np.array([[1.0, 0.0]], dtype=np.float32)
        rendered = np.array([[0.0, 0.5]], dtype=np.float32)

        error = compute_mse_error(target, rendered, normalize=False)
        expected = np.array([[1.0, 0.25]], dtype=np.float32)  # (1-0)^2=1, (0-0.5)^2=0.25
        assert np.allclose(error, expected)

    def test_rgb_channels(self):
        """Test MSE error for RGB channels."""
        target = np.array([[[1.0, 0.0, 0.5]]], dtype=np.float32)
        rendered = np.array([[[0.0, 1.0, 0.0]]], dtype=np.float32)

        error = compute_mse_error(target, rendered, normalize=False)
        # MSE = mean of [(1-0)^2, (0-1)^2, (0.5-0)^2] = mean of [1, 1, 0.25] = 0.75
        expected = np.array([[0.75]], dtype=np.float32)
        assert np.allclose(error, expected)


class TestWeightedError:
    """Test cases for weighted error computation."""

    def test_uniform_weights(self):
        """Test weighted error with uniform weights."""
        target = np.array([[1.0, 0.0]], dtype=np.float32)
        rendered = np.array([[0.0, 1.0]], dtype=np.float32)
        weights = np.ones((1, 2), dtype=np.float32)

        error = compute_weighted_error(target, rendered, weights, "l1", normalize=False)
        expected = np.array([[1.0, 1.0]], dtype=np.float32)
        assert np.allclose(error, expected)

    def test_variable_weights(self):
        """Test weighted error with variable weights."""
        target = np.array([[1.0, 1.0]], dtype=np.float32)
        rendered = np.array([[0.0, 0.0]], dtype=np.float32)
        weights = np.array([[2.0, 0.5]], dtype=np.float32)

        error = compute_weighted_error(target, rendered, weights, "l1", normalize=False)
        expected = np.array([[2.0, 0.5]], dtype=np.float32)  # 1.0*2.0, 1.0*0.5
        assert np.allclose(error, expected)

    def test_different_metrics(self):
        """Test weighted error with different metrics."""
        target = np.array([[1.0]], dtype=np.float32)
        rendered = np.array([[0.0]], dtype=np.float32)
        weights = np.array([[2.0]], dtype=np.float32)

        l1_error = compute_weighted_error(target, rendered, weights, "l1", normalize=False)
        l2_error = compute_weighted_error(target, rendered, weights, "l2", normalize=False)
        mse_error = compute_weighted_error(target, rendered, weights, "mse", normalize=False)

        assert l1_error[0, 0] == 2.0  # |1-0| * 2 = 2
        assert l2_error[0, 0] == 2.0  # |1-0| * 2 = 2 (for single channel)
        assert mse_error[0, 0] == 2.0  # (1-0)^2 * 2 = 2

    def test_invalid_metric(self):
        """Test error handling for invalid metric."""
        target = np.array([[1.0]])
        rendered = np.array([[0.0]])

        with pytest.raises(ValueError, match="Unknown metric: invalid"):
            compute_weighted_error(target, rendered, None, "invalid")

    def test_weight_shape_mismatch(self):
        """Test error handling for mismatched weight shape."""
        target = np.array([[1.0, 0.0]])
        rendered = np.array([[0.0, 1.0]])
        weights = np.array([[1.0], [1.0]])  # Wrong shape

        with pytest.raises(ValueError, match="Weight map shape .* doesn't match"):
            compute_weighted_error(target, rendered, weights, "l1")


class TestErrorStatistics:
    """Test cases for error statistics computation."""

    def test_empty_error_map(self):
        """Test statistics for empty error map."""
        stats = compute_error_statistics(np.array([]))
        assert stats['empty'] == True

    def test_basic_statistics(self):
        """Test basic statistics computation."""
        error_map = np.array([[0.0, 0.1], [0.5, 1.0]], dtype=np.float32)
        stats = compute_error_statistics(error_map)

        assert np.isclose(stats['mean_error'], 0.4)
        assert np.isclose(stats['max_error'], 1.0)
        assert np.isclose(stats['min_error'], 0.0)
        assert np.isclose(stats['total_error'], 1.6)
        assert stats['error_pixels'] == 3  # Non-zero pixels
        assert stats['zero_error_pixels'] == 1
        assert stats['shape'] == (2, 2)
        assert 'percentiles' in stats

    def test_all_zero_errors(self):
        """Test statistics for all-zero error map."""
        error_map = np.zeros((3, 3), dtype=np.float32)
        stats = compute_error_statistics(error_map)

        assert stats['mean_error'] == 0.0
        assert stats['max_error'] == 0.0
        assert stats['error_pixels'] == 0
        assert stats['zero_error_pixels'] == 9

    def test_non_zero_statistics(self):
        """Test statistics for non-zero errors."""
        error_map = np.array([[0.0, 0.2], [0.4, 0.6]], dtype=np.float32)
        stats = compute_error_statistics(error_map)

        assert 'non_zero_stats' in stats
        assert np.isclose(stats['non_zero_stats']['mean'], 0.4)  # (0.2 + 0.4 + 0.6) / 3
        assert np.isclose(stats['non_zero_stats']['min'], 0.2)
        assert np.isclose(stats['non_zero_stats']['max'], 0.6)


class TestPSNR:
    """Test cases for PSNR computation."""

    def test_perfect_match(self):
        """Test PSNR for identical images."""
        image = np.array([[100, 150], [200, 50]], dtype=np.uint8)
        psnr = compute_psnr(image, image)
        assert psnr == float('inf')

    def test_known_values(self):
        """Test PSNR with known values."""
        target = np.array([[255, 0]], dtype=np.uint8)
        rendered = np.array([[0, 255]], dtype=np.uint8)

        psnr = compute_psnr(target, rendered, normalize=False, max_value=255.0)

        # MSE = mean of [(255-0)^2, (0-255)^2] = mean of [65025, 65025] = 65025
        # PSNR = 20 * log10(255 / sqrt(65025)) = 20 * log10(255 / 255) = 20 * log10(1) = 0
        assert np.isclose(psnr, 0.0, atol=1e-10)

    def test_normalized_images(self):
        """Test PSNR with normalized images."""
        target = np.array([[1.0, 0.0]], dtype=np.float32)
        rendered = np.array([[0.5, 0.5]], dtype=np.float32)

        psnr = compute_psnr(target, rendered, normalize=False, max_value=1.0)

        # MSE = mean of [(1-0.5)^2, (0-0.5)^2] = mean of [0.25, 0.25] = 0.25
        # PSNR = 20 * log10(1.0 / sqrt(0.25)) = 20 * log10(1.0 / 0.5) = 20 * log10(2) ≈ 6.02
        expected_psnr = 20 * np.log10(2)
        assert np.isclose(psnr, expected_psnr)

    def test_shape_mismatch(self):
        """Test PSNR error handling for mismatched shapes."""
        target = np.zeros((2, 2))
        rendered = np.zeros((3, 3))

        with pytest.raises(ValueError, match="Target and rendered images must have same shape"):
            compute_psnr(target, rendered)


class TestNormalizeImage:
    """Test cases for image normalization."""

    def test_uint8_normalization(self):
        """Test uint8 image normalization."""
        image = np.array([[0, 127, 255]], dtype=np.uint8)
        normalized = _normalize_image(image)
        expected = np.array([[0.0, 127.0/255.0, 1.0]], dtype=np.float32)
        assert np.allclose(normalized, expected)

    def test_uint16_normalization(self):
        """Test uint16 image normalization."""
        image = np.array([[0, 32767, 65535]], dtype=np.uint16)
        normalized = _normalize_image(image)
        expected = np.array([[0.0, 32767.0/65535.0, 1.0]], dtype=np.float32)
        assert np.allclose(normalized, expected)

    def test_uint32_normalization(self):
        """Test uint32 image normalization."""
        image = np.array([[0, 2147483647, 4294967295]], dtype=np.uint32)
        normalized = _normalize_image(image)
        expected = np.array([[0.0, 2147483647.0/4294967295.0, 1.0]], dtype=np.float32)
        assert np.allclose(normalized, expected)

    def test_float_passthrough(self):
        """Test float image passthrough."""
        image = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)
        normalized = _normalize_image(image)
        assert np.allclose(normalized, image)

    def test_float64_conversion(self):
        """Test float64 to float32 conversion."""
        image = np.array([[0.0, 0.5, 1.0]], dtype=np.float64)
        normalized = _normalize_image(image)
        assert normalized.dtype == np.float32
        assert np.allclose(normalized, image)


class TestValidateImages:
    """Test cases for image validation."""

    def test_valid_images(self):
        """Test validation of valid images."""
        target = np.random.rand(10, 10, 3)
        rendered = np.random.rand(10, 10, 3)
        validate_images(target, rendered)  # Should not raise

    def test_shape_mismatch(self):
        """Test validation error for shape mismatch."""
        target = np.zeros((5, 5))
        rendered = np.zeros((10, 10))

        with pytest.raises(ValueError, match="Images must have same shape"):
            validate_images(target, rendered)

    def test_empty_images(self):
        """Test validation error for empty images."""
        target = np.array([])
        rendered = np.array([])

        with pytest.raises(ValueError, match="Images cannot be empty"):
            validate_images(target, rendered)

    def test_invalid_dimensions(self):
        """Test validation error for invalid dimensions."""
        target = np.zeros((5, 5, 5, 5))  # 4D array
        rendered = np.zeros((5, 5, 5, 5))

        with pytest.raises(ValueError, match="Images must be 2D or 3D arrays"):
            validate_images(target, rendered)

    def test_unusual_channels_warning(self, caplog):
        """Test warning for unusual number of channels."""
        target = np.zeros((5, 5, 7))  # 7 channels
        rendered = np.zeros((5, 5, 7))

        validate_images(target, rendered)
        assert "Unusual number of channels" in caplog.text


class TestReconstructionError:
    """Test cases for convenience reconstruction error function."""

    def test_l1_metric(self):
        """Test reconstruction error with L1 metric."""
        target = np.array([[1.0, 0.0]], dtype=np.float32)
        rendered = np.array([[0.0, 1.0]], dtype=np.float32)

        error = compute_reconstruction_error(target, rendered, "l1")
        expected = np.array([[1.0, 1.0]], dtype=np.float32)
        assert np.allclose(error, expected)

    def test_l2_metric(self):
        """Test reconstruction error with L2 metric."""
        target = np.array([[[1.0, 0.0, 0.0]]], dtype=np.float32)
        rendered = np.array([[[0.0, 1.0, 0.0]]], dtype=np.float32)

        error = compute_reconstruction_error(target, rendered, "l2")
        expected = np.array([[np.sqrt(2.0)]], dtype=np.float32)
        assert np.allclose(error, expected)

    def test_mse_metric(self):
        """Test reconstruction error with MSE metric."""
        target = np.array([[1.0, 0.0]], dtype=np.float32)
        rendered = np.array([[0.0, 0.5]], dtype=np.float32)

        error = compute_reconstruction_error(target, rendered, "mse")
        expected = np.array([[1.0, 0.25]], dtype=np.float32)
        assert np.allclose(error, expected)

    def test_invalid_metric(self):
        """Test error handling for invalid metric."""
        target = np.array([[1.0]])
        rendered = np.array([[0.0]])

        with pytest.raises(ValueError, match="Unknown metric: invalid"):
            compute_reconstruction_error(target, rendered, "invalid")

    def test_validation_called(self):
        """Test that image validation is called."""
        target = np.zeros((2, 2))
        rendered = np.zeros((3, 3))

        with pytest.raises(ValueError, match="Images must have same shape"):
            compute_reconstruction_error(target, rendered, "l1")