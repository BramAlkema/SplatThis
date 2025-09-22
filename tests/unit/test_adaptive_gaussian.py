"""Unit tests for AdaptiveGaussian2D class."""

import pytest
import numpy as np
import math
from unittest.mock import Mock

from splat_this.core.adaptive_gaussian import (
    AdaptiveGaussian2D,
    create_isotropic_gaussian,
    create_anisotropic_gaussian
)
from splat_this.core.extract import Gaussian


class TestAdaptiveGaussian2D:
    """Test the AdaptiveGaussian2D class."""

    def test_basic_initialization(self):
        """Test basic initialization with default parameters."""
        gaussian = AdaptiveGaussian2D()

        assert gaussian.mu.shape == (2,)
        assert gaussian.inv_s.shape == (2,)
        assert len(gaussian.color) >= 3
        assert 0 <= gaussian.alpha <= 1
        assert 0 <= gaussian.theta < np.pi

    def test_initialization_with_parameters(self):
        """Test initialization with specific parameters."""
        mu = np.array([0.3, 0.7])
        inv_s = np.array([0.5, 0.8])
        theta = np.pi / 4
        color = np.array([0.8, 0.6, 0.4])
        alpha = 0.9

        gaussian = AdaptiveGaussian2D(
            mu=mu, inv_s=inv_s, theta=theta, color=color, alpha=alpha
        )

        np.testing.assert_array_almost_equal(gaussian.mu, mu)
        np.testing.assert_array_almost_equal(gaussian.inv_s, inv_s)
        assert gaussian.theta == theta
        np.testing.assert_array_almost_equal(gaussian.color, color)
        assert gaussian.alpha == alpha

    def test_parameter_validation(self):
        """Test parameter validation during initialization."""
        # Invalid mu shape
        with pytest.raises(ValueError, match="Position mu must be 2D"):
            AdaptiveGaussian2D(mu=np.array([0.5]))

        # Invalid inv_s shape
        with pytest.raises(ValueError, match="Inverse scales inv_s must be 2D"):
            AdaptiveGaussian2D(inv_s=np.array([0.5]))

        # Invalid color shape
        with pytest.raises(ValueError, match="Color must have at least 3 components"):
            AdaptiveGaussian2D(color=np.array([0.5, 0.6]))

        # Invalid alpha range
        with pytest.raises(ValueError, match="Alpha must be in"):
            AdaptiveGaussian2D(alpha=1.5)

        # Invalid inverse scales (negative)
        with pytest.raises(ValueError, match="Inverse scales must be positive"):
            AdaptiveGaussian2D(inv_s=np.array([-0.1, 0.5]))

    def test_parameter_clipping(self):
        """Test parameter clipping functionality."""
        # Create Gaussian with valid initial values, then modify them
        gaussian = AdaptiveGaussian2D()

        # Set out-of-bounds values directly (bypassing validation)
        gaussian.mu = np.array([-0.5, 1.5])
        gaussian.inv_s = np.array([1e-5, 1e5])
        gaussian.theta = 2 * np.pi + 0.5
        gaussian.color = np.array([-0.2, 1.5, 0.5])
        gaussian.alpha = 1.2

        gaussian.clip_parameters()

        # Check clipping
        assert 0.0 <= gaussian.mu[0] <= 1.0
        assert 0.0 <= gaussian.mu[1] <= 1.0
        assert 1e-3 <= gaussian.inv_s[0] <= 1e3
        assert 1e-3 <= gaussian.inv_s[1] <= 1e3
        assert 0.0 <= gaussian.theta < np.pi
        assert 0.0 <= gaussian.color[0] <= 1.0
        assert 0.0 <= gaussian.color[1] <= 1.0
        assert 0.0 <= gaussian.alpha <= 1.0

    def test_covariance_matrix_computation(self):
        """Test covariance matrix computation."""
        gaussian = AdaptiveGaussian2D(
            mu=np.array([0.5, 0.5]),
            inv_s=np.array([1.0, 1.0]),  # Isotropic
            theta=0.0  # No rotation
        )

        cov = gaussian.covariance_matrix

        # Should be identity matrix for this configuration
        expected = np.eye(2)
        np.testing.assert_array_almost_equal(cov, expected, decimal=4)

        # Test with rotation
        gaussian.theta = np.pi / 2
        cov_rotated = gaussian.covariance_matrix

        # Should still be identity (isotropic + rotation = identity)
        np.testing.assert_array_almost_equal(cov_rotated, expected, decimal=4)

    def test_covariance_matrix_anisotropic(self):
        """Test covariance matrix for anisotropic Gaussian."""
        gaussian = AdaptiveGaussian2D(
            mu=np.array([0.5, 0.5]),
            inv_s=np.array([2.0, 1.0]),  # 2:1 anisotropy
            theta=0.0  # No rotation
        )

        cov = gaussian.covariance_matrix

        # Should be diagonal with different eigenvalues
        assert cov.shape == (2, 2)

        # Check eigenvalues reflect anisotropy
        eigenvals = np.linalg.eigvals(cov)
        eigenvals = np.sort(eigenvals)

        # Larger eigenvalue should correspond to smaller inverse scale
        assert eigenvals[1] > eigenvals[0]

    def test_covariance_inverse_computation(self):
        """Test inverse covariance matrix computation."""
        gaussian = AdaptiveGaussian2D(
            mu=np.array([0.5, 0.5]),
            inv_s=np.array([1.0, 1.0]),
            theta=np.pi / 6
        )

        cov = gaussian.covariance_matrix
        cov_inv_direct = gaussian.covariance_inverse
        cov_inv_computed = np.linalg.inv(cov)

        # Direct computation should match inverse of covariance
        np.testing.assert_array_almost_equal(cov_inv_direct, cov_inv_computed, decimal=4)

    def test_aspect_ratio_computation(self):
        """Test aspect ratio computation."""
        # Isotropic case
        gaussian = AdaptiveGaussian2D(inv_s=np.array([1.0, 1.0]))
        assert gaussian.aspect_ratio == pytest.approx(1.0)

        # Anisotropic case
        gaussian = AdaptiveGaussian2D(inv_s=np.array([2.0, 1.0]))
        assert gaussian.aspect_ratio == pytest.approx(2.0)

        # Reverse anisotropy
        gaussian = AdaptiveGaussian2D(inv_s=np.array([1.0, 3.0]))
        assert gaussian.aspect_ratio == pytest.approx(3.0)

    def test_orientation_property(self):
        """Test orientation property."""
        gaussian = AdaptiveGaussian2D(theta=np.pi / 4)
        assert gaussian.orientation == pytest.approx(np.pi / 4)

    def test_eigenvalues_computation(self):
        """Test eigenvalues computation."""
        gaussian = AdaptiveGaussian2D(
            inv_s=np.array([2.0, 1.0]),
            theta=0.0
        )

        eigenvals = gaussian.eigenvalues

        # Should be sorted in descending order
        assert eigenvals[0] >= eigenvals[1]
        assert len(eigenvals) == 2
        assert all(ev > 0 for ev in eigenvals)

    def test_axis_lengths(self):
        """Test principal and minor axis length computation."""
        gaussian = AdaptiveGaussian2D(
            inv_s=np.array([1.0, 1.0]),
            theta=0.0
        )

        principal_length = gaussian.principal_axis_length
        minor_length = gaussian.minor_axis_length

        # For isotropic case, should be equal
        assert principal_length == pytest.approx(minor_length)

        # Should be 3σ extent
        eigenvals = gaussian.eigenvalues
        expected_principal = 3.0 * np.sqrt(max(eigenvals))
        expected_minor = 3.0 * np.sqrt(min(eigenvals))

        assert principal_length == pytest.approx(expected_principal)
        assert minor_length == pytest.approx(expected_minor)

    def test_gaussian_evaluation(self):
        """Test Gaussian evaluation at points."""
        gaussian = AdaptiveGaussian2D(
            mu=np.array([0.5, 0.5]),
            inv_s=np.array([1.0, 1.0]),
            theta=0.0
        )

        # Evaluate at center
        center_value = gaussian.evaluate_at(np.array([0.5, 0.5]))
        assert center_value == pytest.approx(1.0)

        # Evaluate at distant point
        distant_value = gaussian.evaluate_at(np.array([0.0, 0.0]))
        assert 0.0 < distant_value < 1.0

        # Center should have higher value than distant point
        assert center_value > distant_value

    def test_3sigma_radius_computation(self):
        """Test 3σ radius computation in pixels."""
        gaussian = AdaptiveGaussian2D(
            inv_s=np.array([0.5, 0.5]),  # Large Gaussian
            theta=0.0
        )

        image_size = (100, 100)
        radius = gaussian.compute_3sigma_radius_px(image_size)

        assert radius > 0
        assert isinstance(radius, (float, np.floating))

        # Larger Gaussian should have larger radius
        larger_gaussian = AdaptiveGaussian2D(inv_s=np.array([0.2, 0.2]))
        larger_radius = larger_gaussian.compute_3sigma_radius_px(image_size)
        assert larger_radius > radius

    def test_serialization_roundtrip(self):
        """Test serialization and deserialization."""
        original = AdaptiveGaussian2D(
            mu=np.array([0.3, 0.7]),
            inv_s=np.array([0.8, 1.2]),
            theta=np.pi / 3,
            color=np.array([0.8, 0.6, 0.4]),
            alpha=0.9,
            content_complexity=0.5,
            saliency_score=0.7,
            refinement_count=5
        )

        # Serialize and deserialize
        data = original.to_dict()
        restored = AdaptiveGaussian2D.from_dict(data)

        # Check all fields match
        np.testing.assert_array_almost_equal(original.mu, restored.mu)
        np.testing.assert_array_almost_equal(original.inv_s, restored.inv_s)
        assert original.theta == pytest.approx(restored.theta)
        np.testing.assert_array_almost_equal(original.color, restored.color)
        assert original.alpha == pytest.approx(restored.alpha)
        assert original.content_complexity == pytest.approx(restored.content_complexity)
        assert original.saliency_score == pytest.approx(restored.saliency_score)
        assert original.refinement_count == restored.refinement_count

    def test_conversion_from_gaussian(self):
        """Test conversion from current Gaussian class."""
        original_gaussian = Gaussian(
            x=50, y=100, rx=10, ry=15, theta=np.pi/4,
            r=200, g=150, b=100, a=0.8, score=0.6
        )

        image_size = (200, 400)  # H, W
        adaptive = AdaptiveGaussian2D.from_gaussian(original_gaussian, image_size)

        # Check coordinate normalization
        expected_mu = np.array([50/400, 100/200])  # x/W, y/H
        np.testing.assert_array_almost_equal(adaptive.mu, expected_mu)

        # Check color normalization
        expected_color = np.array([200/255, 150/255, 100/255])
        np.testing.assert_array_almost_equal(adaptive.color, expected_color)

        # Check other properties
        assert adaptive.alpha == pytest.approx(0.8)
        assert adaptive.theta == pytest.approx(np.pi/4)
        assert adaptive.content_complexity == pytest.approx(0.6)

    def test_conversion_to_gaussian(self):
        """Test conversion to current Gaussian class."""
        adaptive = AdaptiveGaussian2D(
            mu=np.array([0.25, 0.5]),  # Normalized coordinates
            inv_s=np.array([0.1, 0.2]),  # Will become radii
            theta=np.pi/6,
            color=np.array([0.8, 0.6, 0.4]),
            alpha=0.9,
            content_complexity=0.7
        )

        image_size = (200, 400)  # H, W
        gaussian = adaptive.to_gaussian(image_size)

        # Check coordinate denormalization
        assert gaussian.x == pytest.approx(0.25 * 400)  # mu_x * W
        assert gaussian.y == pytest.approx(0.5 * 200)   # mu_y * H

        # Check color denormalization
        assert gaussian.r == int(0.8 * 255)
        assert gaussian.g == int(0.6 * 255)
        assert gaussian.b == int(0.4 * 255)

        # Check other properties
        assert gaussian.a == pytest.approx(0.9)
        assert gaussian.theta == pytest.approx(np.pi/6)
        assert gaussian.score == pytest.approx(0.7)

    def test_pixel_space_conversion_roundtrip(self):
        """Test round-trip conversion from Gaussian to AdaptiveGaussian2D and back."""
        # Create original Gaussian with specific pixel values
        original_gaussian = Gaussian(
            x=120.5, y=75.3, rx=15.7, ry=8.2, theta=np.pi/3,
            r=180, g=90, b=45, a=0.75, score=0.85
        )

        image_size = (150, 300)  # H, W

        # Convert to AdaptiveGaussian2D
        adaptive = AdaptiveGaussian2D.from_gaussian(original_gaussian, image_size)

        # Convert back to Gaussian
        recovered_gaussian = adaptive.to_gaussian(image_size)

        # Check round-trip accuracy for all parameters
        assert recovered_gaussian.x == pytest.approx(original_gaussian.x, rel=1e-6)
        assert recovered_gaussian.y == pytest.approx(original_gaussian.y, rel=1e-6)
        assert recovered_gaussian.rx == pytest.approx(original_gaussian.rx, rel=1e-6)
        assert recovered_gaussian.ry == pytest.approx(original_gaussian.ry, rel=1e-6)
        assert recovered_gaussian.theta == pytest.approx(original_gaussian.theta, rel=1e-6)
        assert recovered_gaussian.r == original_gaussian.r
        assert recovered_gaussian.g == original_gaussian.g
        assert recovered_gaussian.b == original_gaussian.b
        assert recovered_gaussian.a == pytest.approx(original_gaussian.a, rel=1e-6)
        assert recovered_gaussian.score == pytest.approx(original_gaussian.score, rel=1e-6)

    def test_normalized_sigma_conversion_with_zeros_guard(self):
        """Test pixel-space to normalized σ conversion with zero protection."""
        # Test with very small radii (close to zero, simulating the zero case)
        very_small_gaussian = Gaussian(
            x=50, y=100, rx=1e-10, ry=1e-10, theta=0,
            r=255, g=128, b=64, a=1.0, score=0.5
        )

        image_size = (200, 400)  # H, W
        adaptive = AdaptiveGaussian2D.from_gaussian(very_small_gaussian, image_size)

        # Should have valid non-zero inverse scales protected by minimum values
        assert adaptive.inv_s[0] > 0
        assert adaptive.inv_s[1] > 0
        assert np.isfinite(adaptive.inv_s[0])
        assert np.isfinite(adaptive.inv_s[1])

        # The minimum protection should kick in
        # sigma_x_norm = max(1e-10 / 400, 1e-6) = 1e-6
        # sigma_y_norm = max(1e-10 / 200, 1e-6) = 1e-6
        expected_inv_sx = 1.0 / 1e-6  # 1e6
        expected_inv_sy = 1.0 / 1e-6  # 1e6

        assert adaptive.inv_s[0] == pytest.approx(expected_inv_sx, rel=1e-3)
        assert adaptive.inv_s[1] == pytest.approx(expected_inv_sy, rel=1e-3)

        # Should convert back to valid radii
        recovered = adaptive.to_gaussian(image_size)
        assert recovered.rx > 0
        assert recovered.ry > 0
        assert np.isfinite(recovered.rx)
        assert np.isfinite(recovered.ry)

    def test_normalized_sigma_conversion_small_values(self):
        """Test conversion with very small pixel radii."""
        # Test with tiny radii
        small_radius_gaussian = Gaussian(
            x=50, y=100, rx=0.001, ry=0.002, theta=np.pi/4,
            r=100, g=200, b=150, a=0.8, score=0.3
        )

        image_size = (1000, 2000)  # Large image
        adaptive = AdaptiveGaussian2D.from_gaussian(small_radius_gaussian, image_size)

        # Should handle small normalized σ values correctly
        # sigma_x_norm = 0.001 / 2000 = 0.0000005
        # sigma_y_norm = 0.002 / 1000 = 0.000002
        # But should be clamped to minimum value (1e-6)

        expected_sigma_x_norm = max(0.001 / 2000, 1e-6)
        expected_sigma_y_norm = max(0.002 / 1000, 1e-6)
        expected_inv_sx = 1.0 / expected_sigma_x_norm
        expected_inv_sy = 1.0 / expected_sigma_y_norm

        assert adaptive.inv_s[0] == pytest.approx(expected_inv_sx, rel=1e-3)
        assert adaptive.inv_s[1] == pytest.approx(expected_inv_sy, rel=1e-3)

        # Round-trip should recover at least the minimum protected values
        # Due to clamping, we might not get exactly the original tiny values back
        recovered = adaptive.to_gaussian(image_size)

        # The recovered values should be at least as large as the minimum protected values
        min_rx_expected = 2000 / 1e6  # W / (1/1e-6) = W * 1e-6 = 2000 * 1e-6 = 0.002
        min_ry_expected = 1000 / 1e6  # H / (1/1e-6) = H * 1e-6 = 1000 * 1e-6 = 0.001

        assert recovered.rx >= min_rx_expected * 0.9  # Allow small numerical tolerance
        assert recovered.ry >= min_ry_expected * 0.9

    def test_copy_method(self):
        """Test deep copying."""
        original = AdaptiveGaussian2D(
            mu=np.array([0.3, 0.7]),
            inv_s=np.array([0.8, 1.2]),
            color=np.array([0.8, 0.6, 0.4])
        )

        copy = original.copy()

        # Should be equal but different objects
        np.testing.assert_array_equal(original.mu, copy.mu)
        np.testing.assert_array_equal(original.inv_s, copy.inv_s)
        np.testing.assert_array_equal(original.color, copy.color)

        # Should be different objects
        assert copy.mu is not original.mu
        assert copy.inv_s is not original.inv_s
        assert copy.color is not original.color

        # Modifying copy shouldn't affect original
        copy.mu[0] = 0.9
        assert original.mu[0] != copy.mu[0]


class TestHelperFunctions:
    """Test helper functions for creating Gaussians."""

    def test_create_isotropic_gaussian(self):
        """Test isotropic Gaussian creation."""
        center = [0.3, 0.7]
        scale = 0.1
        color = [0.8, 0.6, 0.4]
        alpha = 0.9

        gaussian = create_isotropic_gaussian(center, scale, color, alpha)

        np.testing.assert_array_almost_equal(gaussian.mu, center)
        assert gaussian.inv_s[0] == pytest.approx(gaussian.inv_s[1])  # Isotropic
        assert gaussian.inv_s[0] == pytest.approx(1.0 / scale)
        np.testing.assert_array_almost_equal(gaussian.color, color)
        assert gaussian.alpha == pytest.approx(alpha)
        assert gaussian.theta == pytest.approx(0.0)

    def test_create_anisotropic_gaussian(self):
        """Test anisotropic Gaussian creation."""
        center = [0.4, 0.6]
        scales = (0.1, 0.2)
        orientation = np.pi / 4
        color = [0.7, 0.5, 0.3]
        alpha = 0.8

        gaussian = create_anisotropic_gaussian(center, scales, orientation, color, alpha)

        np.testing.assert_array_almost_equal(gaussian.mu, center)
        assert gaussian.inv_s[0] == pytest.approx(1.0 / scales[0])
        assert gaussian.inv_s[1] == pytest.approx(1.0 / scales[1])
        assert gaussian.theta == pytest.approx(orientation % np.pi)
        np.testing.assert_array_almost_equal(gaussian.color, color)
        assert gaussian.alpha == pytest.approx(alpha)

    def test_create_isotropic_gaussian_zero_scale(self):
        """Test isotropic Gaussian with zero scale."""
        gaussian = create_isotropic_gaussian([0.5, 0.5], 0.0, [1.0, 1.0, 1.0])

        # Should fallback to default inverse scale
        assert gaussian.inv_s[0] == pytest.approx(0.2)
        assert gaussian.inv_s[1] == pytest.approx(0.2)


class TestNumericalStability:
    """Test numerical stability and edge cases."""

    def test_singular_matrix_handling(self):
        """Test handling of singular matrices in covariance computation."""
        # Create near-singular case
        gaussian = AdaptiveGaussian2D(
            inv_s=np.array([1e-6, 1e6]),  # Extreme anisotropy
            theta=0.0
        )

        # Should not crash and should return valid matrix
        cov = gaussian.covariance_matrix
        assert cov.shape == (2, 2)
        assert np.all(np.isfinite(cov))

        # Eigenvalues should be positive
        eigenvals = np.linalg.eigvals(cov)
        assert np.all(eigenvals > 0)

    def test_extreme_parameter_values(self):
        """Test handling of extreme parameter values."""
        gaussian = AdaptiveGaussian2D(
            mu=np.array([1e-10, 1 - 1e-10]),
            inv_s=np.array([1e-3, 1e3]),
            theta=np.pi - 1e-10,
            color=np.array([1e-10, 1 - 1e-10, 0.5]),
            alpha=1e-10
        )

        # Should not crash
        cov = gaussian.covariance_matrix
        assert np.all(np.isfinite(cov))

        value = gaussian.evaluate_at(gaussian.mu)
        assert np.isfinite(value)

    def test_parameter_consistency(self):
        """Test internal parameter consistency."""
        gaussian = AdaptiveGaussian2D(
            inv_s=np.array([2.0, 1.0]),
            theta=np.pi / 4
        )

        # Aspect ratio should be consistent with inverse scales
        expected_aspect = 2.0 / 1.0
        assert gaussian.aspect_ratio == pytest.approx(expected_aspect)

        # Eigenvalues should reflect the inverse scale relationship
        eigenvals = gaussian.eigenvalues
        eigenval_ratio = max(eigenvals) / min(eigenvals)

        # The relationship is more complex due to rotation, but should be related
        assert eigenval_ratio > 1.0  # Should still show anisotropy


if __name__ == "__main__":
    pytest.main([__file__])