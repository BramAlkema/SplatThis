#!/usr/bin/env python3
"""Unit tests for manual gradient computation module."""

import pytest
import numpy as np
from src.splat_this.core.manual_gradients import (
    ManualGradientComputer,
    GradientConfig,
    SplatGradients,
    GradientValidation,
    compute_splat_gradients,
    validate_gradient_computation
)
from src.splat_this.core.adaptive_gaussian import AdaptiveGaussian2D, create_isotropic_gaussian


class TestGradientConfig:
    """Test gradient configuration."""

    def test_init_defaults(self):
        """Test default configuration initialization."""
        config = GradientConfig()
        assert config.position_step == 0.1
        assert config.scale_step == 0.001
        assert config.rotation_step == 0.01
        assert config.color_step == 0.001
        assert config.gradient_clipping is True
        assert config.clip_threshold == 10.0
        assert config.numerical_validation is True
        assert config.finite_diff_method == 'central'
        assert config.stability_epsilon == 1e-8

    def test_init_custom(self):
        """Test custom configuration initialization."""
        config = GradientConfig(
            position_step=0.05,
            scale_step=0.0005,
            rotation_step=0.005,
            color_step=0.0005,
            gradient_clipping=False,
            clip_threshold=5.0,
            numerical_validation=False,
            finite_diff_method='forward',
            stability_epsilon=1e-10
        )
        assert config.position_step == 0.05
        assert config.scale_step == 0.0005
        assert config.rotation_step == 0.005
        assert config.color_step == 0.0005
        assert config.gradient_clipping is False
        assert config.clip_threshold == 5.0
        assert config.numerical_validation is False
        assert config.finite_diff_method == 'forward'
        assert config.stability_epsilon == 1e-10

    def test_validation_positive_steps(self):
        """Test validation of positive step sizes."""
        with pytest.raises(ValueError, match="position_step must be positive"):
            GradientConfig(position_step=0.0)

        with pytest.raises(ValueError, match="scale_step must be positive"):
            GradientConfig(scale_step=-0.001)

        with pytest.raises(ValueError, match="rotation_step must be positive"):
            GradientConfig(rotation_step=0.0)

        with pytest.raises(ValueError, match="color_step must be positive"):
            GradientConfig(color_step=-0.001)

    def test_validation_clip_threshold(self):
        """Test validation of clipping threshold."""
        with pytest.raises(ValueError, match="clip_threshold must be positive"):
            GradientConfig(clip_threshold=0.0)

        with pytest.raises(ValueError, match="clip_threshold must be positive"):
            GradientConfig(clip_threshold=-1.0)

    def test_validation_stability_epsilon(self):
        """Test validation of stability epsilon."""
        with pytest.raises(ValueError, match="stability_epsilon must be positive"):
            GradientConfig(stability_epsilon=0.0)

    def test_validation_finite_diff_method(self):
        """Test validation of finite difference method."""
        with pytest.raises(ValueError, match="finite_diff_method must be one of"):
            GradientConfig(finite_diff_method='invalid')

        # Valid methods should work
        for method in ['forward', 'backward', 'central']:
            config = GradientConfig(finite_diff_method=method)
            assert config.finite_diff_method == method


class TestSplatGradients:
    """Test splat gradients data structure."""

    def test_gradients_creation(self):
        """Test creation of splat gradients."""
        gradients = SplatGradients(
            position_grad=np.array([0.1, -0.2]),
            scale_grad=np.array([0.05, 0.03]),
            rotation_grad=0.02,
            color_grad=np.array([0.01, -0.01, 0.03]),
            alpha_grad=-0.005
        )

        assert gradients.position_grad.shape == (2,)
        assert gradients.scale_grad.shape == (2,)
        assert gradients.color_grad.shape == (3,)
        assert isinstance(gradients.rotation_grad, (float, int, np.number))
        assert isinstance(gradients.alpha_grad, (float, int, np.number))

    def test_gradients_validation_shapes(self):
        """Test validation of gradient shapes."""
        with pytest.raises(ValueError, match="position_grad must have shape"):
            SplatGradients(
                position_grad=np.array([0.1]),  # Wrong shape
                scale_grad=np.array([0.05, 0.03]),
                rotation_grad=0.02,
                color_grad=np.array([0.01, -0.01, 0.03]),
                alpha_grad=-0.005
            )

        with pytest.raises(ValueError, match="scale_grad must have shape"):
            SplatGradients(
                position_grad=np.array([0.1, -0.2]),
                scale_grad=np.array([0.05]),  # Wrong shape
                rotation_grad=0.02,
                color_grad=np.array([0.01, -0.01, 0.03]),
                alpha_grad=-0.005
            )

        with pytest.raises(ValueError, match="color_grad must have shape"):
            SplatGradients(
                position_grad=np.array([0.1, -0.2]),
                scale_grad=np.array([0.05, 0.03]),
                rotation_grad=0.02,
                color_grad=np.array([0.01, -0.01]),  # Wrong shape
                alpha_grad=-0.005
            )


class TestGradientValidation:
    """Test gradient validation data structure."""

    def test_validation_creation(self):
        """Test creation of gradient validation."""
        validation = GradientValidation(
            position_error=0.01,
            scale_error=0.005,
            rotation_error=0.002,
            color_error=0.008,
            alpha_error=0.001,
            max_error=0.01,
            passed=True
        )

        assert validation.position_error == 0.01
        assert validation.scale_error == 0.005
        assert validation.rotation_error == 0.002
        assert validation.color_error == 0.008
        assert validation.alpha_error == 0.001
        assert validation.max_error == 0.01
        assert validation.passed is True


class TestManualGradientComputer:
    """Test manual gradient computer functionality."""

    def test_init_default(self):
        """Test computer initialization with defaults."""
        computer = ManualGradientComputer()
        assert isinstance(computer.config, GradientConfig)

    def test_init_custom_config(self):
        """Test computer initialization with custom config."""
        config = GradientConfig(position_step=0.05)
        computer = ManualGradientComputer(config)
        assert computer.config.position_step == 0.05

    def test_compute_position_gradient(self):
        """Test position gradient computation."""
        # Create test splat
        splat = create_isotropic_gaussian([0.5, 0.5], 0.1, [0.8, 0.2, 0.1], 0.8)

        # Create test images
        target_image = np.ones((64, 64, 3)) * 0.5
        rendered_image = np.ones((64, 64, 3)) * 0.4
        error_map = np.abs(target_image[:, :, 0] - rendered_image[:, :, 0])

        computer = ManualGradientComputer()
        grad = computer.compute_position_gradient(splat, target_image, rendered_image, error_map)

        assert grad.shape == (2,)
        assert np.all(np.isfinite(grad))

    def test_compute_scale_gradient(self):
        """Test scale gradient computation."""
        splat = create_isotropic_gaussian([0.5, 0.5], 0.1, [0.8, 0.2, 0.1], 0.8)

        target_image = np.ones((64, 64, 3)) * 0.5
        rendered_image = np.ones((64, 64, 3)) * 0.4
        error_map = np.abs(target_image[:, :, 0] - rendered_image[:, :, 0])

        computer = ManualGradientComputer()
        grad = computer.compute_scale_gradient(splat, target_image, rendered_image, error_map)

        assert grad.shape == (2,)
        assert np.all(np.isfinite(grad))

    def test_compute_rotation_gradient(self):
        """Test rotation gradient computation."""
        splat = create_isotropic_gaussian([0.5, 0.5], 0.1, [0.8, 0.2, 0.1], 0.8)

        target_image = np.ones((64, 64, 3)) * 0.5
        rendered_image = np.ones((64, 64, 3)) * 0.4
        error_map = np.abs(target_image[:, :, 0] - rendered_image[:, :, 0])

        computer = ManualGradientComputer()
        grad = computer.compute_rotation_gradient(splat, target_image, rendered_image, error_map)

        assert isinstance(grad, (float, int, np.number))
        assert np.isfinite(grad)

    def test_compute_color_gradient(self):
        """Test color gradient computation."""
        splat = create_isotropic_gaussian([0.5, 0.5], 0.1, [0.8, 0.2, 0.1], 0.8)

        target_image = np.ones((64, 64, 3)) * 0.5
        rendered_image = np.ones((64, 64, 3)) * 0.4
        error_map = np.abs(target_image[:, :, 0] - rendered_image[:, :, 0])

        computer = ManualGradientComputer()
        grad = computer.compute_color_gradient(splat, target_image, rendered_image, error_map)

        assert grad.shape == (3,)
        assert np.all(np.isfinite(grad))

    def test_compute_alpha_gradient(self):
        """Test alpha gradient computation."""
        splat = create_isotropic_gaussian([0.5, 0.5], 0.1, [0.8, 0.2, 0.1], 0.8)

        target_image = np.ones((64, 64, 3)) * 0.5
        rendered_image = np.ones((64, 64, 3)) * 0.4
        error_map = np.abs(target_image[:, :, 0] - rendered_image[:, :, 0])

        computer = ManualGradientComputer()
        grad = computer.compute_alpha_gradient(splat, target_image, rendered_image, error_map)

        assert isinstance(grad, (float, int, np.number))
        assert np.isfinite(grad)

    def test_compute_all_gradients(self):
        """Test computation of all gradients."""
        splat = create_isotropic_gaussian([0.5, 0.5], 0.1, [0.8, 0.2, 0.1], 0.8)

        target_image = np.ones((64, 64, 3)) * 0.5
        rendered_image = np.ones((64, 64, 3)) * 0.4
        error_map = np.abs(target_image[:, :, 0] - rendered_image[:, :, 0])

        computer = ManualGradientComputer()
        gradients = computer.compute_all_gradients(splat, target_image, rendered_image, error_map)

        assert isinstance(gradients, SplatGradients)
        assert gradients.position_grad.shape == (2,)
        assert gradients.scale_grad.shape == (2,)
        assert gradients.color_grad.shape == (3,)
        assert np.all(np.isfinite(gradients.position_grad))
        assert np.all(np.isfinite(gradients.scale_grad))
        assert np.all(np.isfinite(gradients.color_grad))
        assert np.isfinite(gradients.rotation_grad)
        assert np.isfinite(gradients.alpha_grad)

    def test_finite_difference_methods(self):
        """Test different finite difference methods."""
        splat = create_isotropic_gaussian([0.5, 0.5], 0.1, [0.8, 0.2, 0.1], 0.8)

        target_image = np.ones((64, 64, 3)) * 0.5
        rendered_image = np.ones((64, 64, 3)) * 0.4
        error_map = np.abs(target_image[:, :, 0] - rendered_image[:, :, 0])

        methods = ['forward', 'backward', 'central']

        for method in methods:
            config = GradientConfig(finite_diff_method=method)
            computer = ManualGradientComputer(config)
            gradients = computer.compute_all_gradients(splat, target_image, rendered_image, error_map)

            assert isinstance(gradients, SplatGradients)
            assert np.all(np.isfinite(gradients.position_grad))
            assert np.all(np.isfinite(gradients.scale_grad))
            assert np.all(np.isfinite(gradients.color_grad))

    def test_gradient_clipping(self):
        """Test gradient clipping functionality."""
        splat = create_isotropic_gaussian([0.5, 0.5], 0.1, [0.8, 0.2, 0.1], 0.8)

        target_image = np.ones((64, 64, 3)) * 0.5
        rendered_image = np.ones((64, 64, 3)) * 10.0  # Large difference to create large gradients
        error_map = np.abs(target_image[:, :, 0] - rendered_image[:, :, 0])

        # With clipping
        config_clipped = GradientConfig(gradient_clipping=True, clip_threshold=1.0)
        computer_clipped = ManualGradientComputer(config_clipped)
        gradients_clipped = computer_clipped.compute_all_gradients(
            splat, target_image, rendered_image, error_map
        )

        # Without clipping
        config_unclipped = GradientConfig(gradient_clipping=False)
        computer_unclipped = ManualGradientComputer(config_unclipped)
        gradients_unclipped = computer_unclipped.compute_all_gradients(
            splat, target_image, rendered_image, error_map
        )

        # Clipped gradients should have bounded magnitude
        assert np.linalg.norm(gradients_clipped.position_grad) <= config_clipped.clip_threshold
        assert np.linalg.norm(gradients_clipped.scale_grad) <= config_clipped.clip_threshold
        assert np.linalg.norm(gradients_clipped.color_grad) <= config_clipped.clip_threshold
        assert abs(gradients_clipped.rotation_grad) <= config_clipped.clip_threshold
        assert abs(gradients_clipped.alpha_grad) <= config_clipped.clip_threshold

    def test_create_perturbed_splat(self):
        """Test creation of perturbed splats."""
        original_splat = create_isotropic_gaussian([0.5, 0.5], 0.1, [0.8, 0.2, 0.1], 0.8)

        computer = ManualGradientComputer()

        # Test position perturbation
        new_position = [0.4, 0.4]
        perturbed = computer._create_perturbed_splat(original_splat, position=new_position)
        assert np.allclose(perturbed.mu, new_position)
        assert np.allclose(1.0 / perturbed.inv_s, 1.0 / original_splat.inv_s)
        assert perturbed.theta == original_splat.theta

        # Test scale perturbation
        new_scale = [1.5, 1.5]
        perturbed = computer._create_perturbed_splat(original_splat, scale=new_scale)
        assert np.allclose(1.0 / perturbed.inv_s, new_scale)
        assert np.allclose(perturbed.mu, original_splat.mu)

        # Test rotation perturbation
        new_rotation = 0.5
        perturbed = computer._create_perturbed_splat(original_splat, rotation=new_rotation)
        assert perturbed.theta == new_rotation
        assert np.allclose(perturbed.mu, original_splat.mu)

        # Test color perturbation
        new_color = [0.5, 0.5, 0.5]
        perturbed = computer._create_perturbed_splat(original_splat, color=new_color)
        assert np.allclose(perturbed.color, new_color)
        assert np.allclose(perturbed.mu, original_splat.mu)

        # Test alpha perturbation
        new_alpha = 0.5
        perturbed = computer._create_perturbed_splat(original_splat, alpha=new_alpha)
        assert perturbed.alpha == new_alpha
        assert np.allclose(perturbed.mu, original_splat.mu)

    def test_clip_gradient(self):
        """Test gradient clipping helper function."""
        computer = ManualGradientComputer()

        # Test vector gradient clipping
        large_gradient = np.array([10.0, 15.0])  # Magnitude ~18
        clipped = computer._clip_gradient(large_gradient, 5.0)
        assert np.linalg.norm(clipped) <= 5.0 + 1e-10
        # Should maintain direction
        assert np.allclose(clipped / np.linalg.norm(clipped),
                          large_gradient / np.linalg.norm(large_gradient))

        # Test small gradient (no clipping)
        small_gradient = np.array([1.0, 2.0])
        clipped = computer._clip_gradient(small_gradient, 5.0)
        assert np.allclose(clipped, small_gradient)

    def test_clip_scalar_gradient(self):
        """Test scalar gradient clipping helper function."""
        computer = ManualGradientComputer()

        # Test large positive gradient
        large_grad = 15.0
        clipped = computer._clip_scalar_gradient(large_grad, 5.0)
        assert clipped == 5.0

        # Test large negative gradient
        large_neg_grad = -15.0
        clipped = computer._clip_scalar_gradient(large_neg_grad, 5.0)
        assert clipped == -5.0

        # Test small gradient (no clipping)
        small_grad = 2.0
        clipped = computer._clip_scalar_gradient(small_grad, 5.0)
        assert clipped == 2.0

    def test_validate_gradients(self):
        """Test gradient validation framework."""
        splat = create_isotropic_gaussian([0.5, 0.5], 0.1, [0.8, 0.2, 0.1], 0.8)

        target_image = np.ones((64, 64, 3)) * 0.5
        rendered_image = np.ones((64, 64, 3)) * 0.4
        error_map = np.abs(target_image[:, :, 0] - rendered_image[:, :, 0])

        computer = ManualGradientComputer()
        gradients = computer.compute_all_gradients(splat, target_image, rendered_image, error_map)
        validation = computer.validate_gradients(
            splat, target_image, rendered_image, error_map, gradients
        )

        assert isinstance(validation, GradientValidation)
        assert validation.position_error >= 0
        assert validation.scale_error >= 0
        assert validation.rotation_error >= 0
        assert validation.color_error >= 0
        assert validation.alpha_error >= 0
        assert validation.max_error >= 0
        assert isinstance(validation.passed, bool)


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_compute_splat_gradients_default(self):
        """Test convenience function with default config."""
        splat = create_isotropic_gaussian([0.5, 0.5], 0.1, [0.8, 0.2, 0.1], 0.8)

        target_image = np.ones((64, 64, 3)) * 0.5
        rendered_image = np.ones((64, 64, 3)) * 0.4
        error_map = np.abs(target_image[:, :, 0] - rendered_image[:, :, 0])

        gradients = compute_splat_gradients(splat, target_image, rendered_image, error_map)

        assert isinstance(gradients, SplatGradients)
        assert gradients.position_grad.shape == (2,)
        assert gradients.scale_grad.shape == (2,)
        assert gradients.color_grad.shape == (3,)

    def test_compute_splat_gradients_custom_config(self):
        """Test convenience function with custom config."""
        splat = create_isotropic_gaussian([0.5, 0.5], 0.1, [0.8, 0.2, 0.1], 0.8)

        target_image = np.ones((64, 64, 3)) * 0.5
        rendered_image = np.ones((64, 64, 3)) * 0.4
        error_map = np.abs(target_image[:, :, 0] - rendered_image[:, :, 0])

        config = GradientConfig(finite_diff_method='forward')
        gradients = compute_splat_gradients(
            splat, target_image, rendered_image, error_map, config
        )

        assert isinstance(gradients, SplatGradients)

    def test_validate_gradient_computation(self):
        """Test convenience function for gradient validation."""
        splat = create_isotropic_gaussian([0.5, 0.5], 0.1, [0.8, 0.2, 0.1], 0.8)

        target_image = np.ones((64, 64, 3)) * 0.5
        rendered_image = np.ones((64, 64, 3)) * 0.4
        error_map = np.abs(target_image[:, :, 0] - rendered_image[:, :, 0])

        validation = validate_gradient_computation(
            splat, target_image, rendered_image, error_map
        )

        assert isinstance(validation, GradientValidation)
        assert isinstance(validation.passed, bool)


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_zero_error_map(self):
        """Test gradient computation with zero error map."""
        splat = create_isotropic_gaussian([0.5, 0.5], 0.1, [0.8, 0.2, 0.1], 0.8)

        target_image = np.ones((64, 64, 3)) * 0.5
        rendered_image = target_image.copy()  # Perfect match
        error_map = np.zeros((64, 64))

        computer = ManualGradientComputer()
        gradients = computer.compute_all_gradients(splat, target_image, rendered_image, error_map)

        # Gradients might be zero or very small
        assert np.all(np.isfinite(gradients.position_grad))
        assert np.all(np.isfinite(gradients.scale_grad))
        assert np.all(np.isfinite(gradients.color_grad))
        assert np.isfinite(gradients.rotation_grad)
        assert np.isfinite(gradients.alpha_grad)

    def test_uniform_error_map(self):
        """Test gradient computation with uniform error map."""
        splat = create_isotropic_gaussian([0.5, 0.5], 0.1, [0.8, 0.2, 0.1], 0.8)

        target_image = np.ones((64, 64, 3)) * 0.5
        rendered_image = np.ones((64, 64, 3)) * 0.3
        error_map = np.ones((64, 64)) * 0.2  # Uniform error

        computer = ManualGradientComputer()
        gradients = computer.compute_all_gradients(splat, target_image, rendered_image, error_map)

        assert np.all(np.isfinite(gradients.position_grad))
        assert np.all(np.isfinite(gradients.scale_grad))
        assert np.all(np.isfinite(gradients.color_grad))

    def test_extreme_splat_parameters(self):
        """Test gradient computation with extreme splat parameters."""
        # Very small splat
        small_splat = create_isotropic_gaussian([0.5, 0.5], 0.001, [0.8, 0.2, 0.1], 0.01)

        # Very large splat
        large_splat = create_isotropic_gaussian([0.5, 0.5], 10.0, [0.8, 0.2, 0.1], 1.0)

        target_image = np.ones((64, 64, 3)) * 0.5
        rendered_image = np.ones((64, 64, 3)) * 0.4
        error_map = np.abs(target_image[:, :, 0] - rendered_image[:, :, 0])

        computer = ManualGradientComputer()

        # Test small splat
        gradients_small = computer.compute_all_gradients(
            small_splat, target_image, rendered_image, error_map
        )
        assert np.all(np.isfinite(gradients_small.position_grad))

        # Test large splat
        gradients_large = computer.compute_all_gradients(
            large_splat, target_image, rendered_image, error_map
        )
        assert np.all(np.isfinite(gradients_large.position_grad))

    def test_small_step_sizes(self):
        """Test gradient computation with very small step sizes."""
        splat = create_isotropic_gaussian([0.5, 0.5], 0.1, [0.8, 0.2, 0.1], 0.8)

        target_image = np.ones((64, 64, 3)) * 0.5
        rendered_image = np.ones((64, 64, 3)) * 0.4
        error_map = np.abs(target_image[:, :, 0] - rendered_image[:, :, 0])

        config = GradientConfig(
            position_step=1e-6,
            scale_step=1e-8,
            rotation_step=1e-6,
            color_step=1e-8
        )
        computer = ManualGradientComputer(config)
        gradients = computer.compute_all_gradients(splat, target_image, rendered_image, error_map)

        # Should still produce finite gradients
        assert np.all(np.isfinite(gradients.position_grad))
        assert np.all(np.isfinite(gradients.scale_grad))
        assert np.all(np.isfinite(gradients.color_grad))

    def test_large_step_sizes(self):
        """Test gradient computation with large step sizes."""
        splat = create_isotropic_gaussian([0.5, 0.5], 0.1, [0.8, 0.2, 0.1], 0.8)

        target_image = np.ones((64, 64, 3)) * 0.5
        rendered_image = np.ones((64, 64, 3)) * 0.4
        error_map = np.abs(target_image[:, :, 0] - rendered_image[:, :, 0])

        config = GradientConfig(
            position_step=10.0,
            scale_step=1.0,
            rotation_step=1.0,
            color_step=0.5
        )
        computer = ManualGradientComputer(config)
        gradients = computer.compute_all_gradients(splat, target_image, rendered_image, error_map)

        # Should still produce finite gradients
        assert np.all(np.isfinite(gradients.position_grad))
        assert np.all(np.isfinite(gradients.scale_grad))
        assert np.all(np.isfinite(gradients.color_grad))

    def test_boundary_splat(self):
        """Test gradient computation for splat near image boundary."""
        # Splat near edge
        boundary_splat = create_isotropic_gaussian([0.05, 0.05], 0.1, [0.8, 0.2, 0.1], 0.8)

        target_image = np.ones((64, 64, 3)) * 0.5
        rendered_image = np.ones((64, 64, 3)) * 0.4
        error_map = np.abs(target_image[:, :, 0] - rendered_image[:, :, 0])

        computer = ManualGradientComputer()
        gradients = computer.compute_all_gradients(
            boundary_splat, target_image, rendered_image, error_map
        )

        assert np.all(np.isfinite(gradients.position_grad))
        assert np.all(np.isfinite(gradients.scale_grad))

    def test_outside_image_splat(self):
        """Test gradient computation for splat outside image bounds."""
        # Splat outside image
        outside_splat = create_isotropic_gaussian([-0.1, -0.1], 0.1, [0.8, 0.2, 0.1], 0.8)

        target_image = np.ones((64, 64, 3)) * 0.5
        rendered_image = np.ones((64, 64, 3)) * 0.4
        error_map = np.abs(target_image[:, :, 0] - rendered_image[:, :, 0])

        computer = ManualGradientComputer()
        gradients = computer.compute_all_gradients(
            outside_splat, target_image, rendered_image, error_map
        )

        # Should still produce finite results
        assert np.all(np.isfinite(gradients.position_grad))
        assert np.all(np.isfinite(gradients.scale_grad))

    def test_numerical_stability(self):
        """Test numerical stability with various parameter combinations."""
        splat = create_isotropic_gaussian([0.5, 0.5], 0.1, [0.8, 0.2, 0.1], 0.8)

        # Test with very small values
        target_image = np.ones((64, 64, 3)) * 1e-10
        rendered_image = np.ones((64, 64, 3)) * 1e-11
        error_map = np.abs(target_image[:, :, 0] - rendered_image[:, :, 0])

        computer = ManualGradientComputer()
        gradients = computer.compute_all_gradients(splat, target_image, rendered_image, error_map)

        assert np.all(np.isfinite(gradients.position_grad))
        assert np.all(np.isfinite(gradients.scale_grad))
        assert np.all(np.isfinite(gradients.color_grad))

        # Test with very large values
        target_image = np.ones((64, 64, 3)) * 1e10
        rendered_image = np.ones((64, 64, 3)) * 1e9
        error_map = np.abs(target_image[:, :, 0] - rendered_image[:, :, 0])

        gradients = computer.compute_all_gradients(splat, target_image, rendered_image, error_map)

        assert np.all(np.isfinite(gradients.position_grad))
        assert np.all(np.isfinite(gradients.scale_grad))
        assert np.all(np.isfinite(gradients.color_grad))