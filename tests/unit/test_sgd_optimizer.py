"""
Unit tests for SGD optimization system.

Tests for T3.2: SGD Optimization Loop implementation.
Comprehensive testing of SGD optimizer functionality including:
- Configuration validation
- Optimization methods (SGD, momentum, Adam)
- Learning rate scheduling
- Convergence criteria
- Gradient clipping and stability
- Batch processing
- Integration with manual gradients
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
import copy

from src.splat_this.core.sgd_optimizer import (
    SGDConfig,
    SGDOptimizer,
    OptimizationState,
    OptimizationResult,
    OptimizationMethod,
    LearningRateSchedule,
    optimize_splats_sgd,
    create_sgd_config_preset
)
from src.splat_this.core.manual_gradients import GradientConfig, SplatGradients
from src.splat_this.core.adaptive_gaussian import AdaptiveGaussian2D, create_isotropic_gaussian


class TestSGDConfig:
    """Test SGD configuration class."""

    def test_init_defaults(self):
        """Test default configuration initialization."""
        config = SGDConfig()

        assert config.position_lr == 0.001
        assert config.scale_lr == 0.0005
        assert config.rotation_lr == 0.0001
        assert config.color_lr == 0.001
        assert config.alpha_lr == 0.0005
        assert config.method == OptimizationMethod.SGD_MOMENTUM
        assert config.momentum == 0.9
        assert config.lr_schedule == LearningRateSchedule.EXPONENTIAL_DECAY
        assert config.max_iterations == 1000
        assert config.convergence_threshold == 1e-6

    def test_init_custom(self):
        """Test custom configuration initialization."""
        config = SGDConfig(
            position_lr=0.01,
            scale_lr=0.005,
            method=OptimizationMethod.ADAM,
            max_iterations=500,
            momentum=0.8
        )

        assert config.position_lr == 0.01
        assert config.scale_lr == 0.005
        assert config.method == OptimizationMethod.ADAM
        assert config.max_iterations == 500
        assert config.momentum == 0.8

    def test_validation_positive_learning_rates(self):
        """Test validation of positive learning rates."""
        with pytest.raises(ValueError, match="position_lr must be positive"):
            SGDConfig(position_lr=-0.001)

        with pytest.raises(ValueError, match="scale_lr must be positive"):
            SGDConfig(scale_lr=0.0)

        with pytest.raises(ValueError, match="rotation_lr must be positive"):
            SGDConfig(rotation_lr=-0.1)

    def test_validation_momentum_range(self):
        """Test validation of momentum parameter range."""
        with pytest.raises(ValueError, match="momentum must be in \\[0,1\\]"):
            SGDConfig(momentum=-0.1)

        with pytest.raises(ValueError, match="momentum must be in \\[0,1\\]"):
            SGDConfig(momentum=1.5)

    def test_validation_beta_ranges(self):
        """Test validation of Adam beta parameters."""
        with pytest.raises(ValueError, match="beta1 must be in \\[0,1\\]"):
            SGDConfig(beta1=1.1)

        with pytest.raises(ValueError, match="beta2 must be in \\[0,1\\]"):
            SGDConfig(beta2=-0.1)

    def test_validation_positive_iterations(self):
        """Test validation of positive max_iterations."""
        with pytest.raises(ValueError, match="max_iterations must be positive"):
            SGDConfig(max_iterations=0)

        with pytest.raises(ValueError, match="max_iterations must be positive"):
            SGDConfig(max_iterations=-100)

    def test_validation_positive_threshold(self):
        """Test validation of positive convergence threshold."""
        with pytest.raises(ValueError, match="convergence_threshold must be positive"):
            SGDConfig(convergence_threshold=0.0)

        with pytest.raises(ValueError, match="convergence_threshold must be positive"):
            SGDConfig(convergence_threshold=-1e-6)


class TestOptimizationState:
    """Test optimization state tracking."""

    def test_state_initialization(self):
        """Test optimization state initialization."""
        state = OptimizationState()

        assert state.iteration == 0
        assert state.current_loss == float('inf')
        assert state.best_loss == float('inf')
        assert len(state.loss_history) == 0
        assert len(state.gradient_norms) == 0
        assert not state.converged
        assert not state.early_stopped

    def test_state_reset(self):
        """Test optimization state reset functionality."""
        state = OptimizationState()

        # Modify state
        state.iteration = 100
        state.current_loss = 0.5
        state.best_loss = 0.3
        state.loss_history = [1.0, 0.8, 0.5]
        state.gradient_norms = [0.1, 0.05, 0.02]
        state.converged = True
        state.patience_counter = 10

        # Reset state
        state.reset()

        assert state.iteration == 0
        assert state.current_loss == float('inf')
        assert state.best_loss == float('inf')
        assert len(state.loss_history) == 0
        assert len(state.gradient_norms) == 0
        assert not state.converged
        assert not state.early_stopped
        assert state.patience_counter == 0


class TestSGDOptimizer:
    """Test SGD optimizer functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = SGDConfig(max_iterations=10, log_every=5, validate_every=5)
        self.optimizer = SGDOptimizer(self.config)

        # Create test splats
        self.splats = [
            create_isotropic_gaussian([0.3, 0.3], 0.1, [0.8, 0.2, 0.1], 0.8),
            create_isotropic_gaussian([0.7, 0.7], 0.08, [0.1, 0.8, 0.2], 0.7),
            create_isotropic_gaussian([0.5, 0.5], 0.12, [0.2, 0.1, 0.8], 0.9)
        ]

        # Create test images
        self.target_image = np.ones((32, 32, 3)) * 0.5
        self.rendered_image = np.ones((32, 32, 3)) * 0.4
        self.error_map = np.abs(self.target_image[:, :, 0] - self.rendered_image[:, :, 0])

    def test_optimizer_initialization(self):
        """Test optimizer initialization."""
        assert self.optimizer.config == self.config
        assert self.optimizer.gradient_computer is not None
        assert isinstance(self.optimizer.state, OptimizationState)

    def test_momentum_state_initialization(self):
        """Test momentum state initialization for different methods."""
        # SGD momentum
        config_momentum = SGDConfig(method=OptimizationMethod.SGD_MOMENTUM)
        optimizer_momentum = SGDOptimizer(config_momentum)
        assert 'position_momentum' in optimizer_momentum.state.momentum_states

        # Adam
        config_adam = SGDConfig(method=OptimizationMethod.ADAM)
        optimizer_adam = SGDOptimizer(config_adam)
        assert 'position_momentum' in optimizer_adam.state.momentum_states
        assert 'position_moment2' in optimizer_adam.state.momentum_states

    def test_prepare_batches_full_batch(self):
        """Test batch preparation for full batch mode."""
        batches = self.optimizer._prepare_batches(5)
        assert len(batches) == 1
        assert len(batches[0]) == 5

    def test_prepare_batches_mini_batch(self):
        """Test batch preparation for mini-batch mode."""
        config = SGDConfig(batch_size=2)
        optimizer = SGDOptimizer(config)

        batches = optimizer._prepare_batches(5)
        assert len(batches) == 3  # 2, 2, 1
        assert len(batches[0]) == 2
        assert len(batches[1]) == 2
        assert len(batches[2]) == 1

    def test_gradient_clipping(self):
        """Test gradient clipping functionality."""
        # Create large gradients
        gradients = SplatGradients(
            position_grad=np.array([10.0, 15.0]),
            scale_grad=np.array([8.0, 12.0]),
            rotation_grad=20.0,
            color_grad=np.array([5.0, 10.0, 15.0]),
            alpha_grad=25.0
        )

        config = SGDConfig(gradient_clipping=True, clip_threshold=1.0)
        optimizer = SGDOptimizer(config)

        clipped = optimizer._clip_gradients(gradients)

        # Check that large gradients are clipped
        assert np.linalg.norm(clipped.position_grad) <= 1.0 + 1e-10
        assert np.linalg.norm(clipped.scale_grad) <= 1.0 + 1e-10
        assert abs(clipped.rotation_grad) <= 1.0 + 1e-10
        assert np.linalg.norm(clipped.color_grad) <= 1.0 + 1e-10
        assert abs(clipped.alpha_grad) <= 1.0 + 1e-10

    def test_learning_rate_schedules(self):
        """Test different learning rate scheduling strategies."""
        schedules = [
            LearningRateSchedule.CONSTANT,
            LearningRateSchedule.LINEAR_DECAY,
            LearningRateSchedule.EXPONENTIAL_DECAY,
            LearningRateSchedule.COSINE_ANNEALING,
            LearningRateSchedule.STEP_DECAY
        ]

        for schedule in schedules:
            config = SGDConfig(lr_schedule=schedule, max_iterations=100)
            optimizer = SGDOptimizer(config)

            # Test learning rate at different iterations
            optimizer.state.iteration = 0
            lrs_start = optimizer._get_current_learning_rates()

            optimizer.state.iteration = 50
            lrs_mid = optimizer._get_current_learning_rates()

            optimizer.state.iteration = 99
            lrs_end = optimizer._get_current_learning_rates()

            # All learning rates should be positive
            for lr_dict in [lrs_start, lrs_mid, lrs_end]:
                for lr in lr_dict.values():
                    assert lr > 0

            # For decay schedules, learning rate should decrease
            if schedule != LearningRateSchedule.CONSTANT:
                assert lrs_end['position'] <= lrs_start['position']

    def test_sgd_parameter_update(self):
        """Test standard SGD parameter updates."""
        splat = create_isotropic_gaussian([0.5, 0.5], 0.1, [0.8, 0.2, 0.1], 0.8)
        original_mu = splat.mu.copy()
        original_inv_s = splat.inv_s.copy()
        original_theta = splat.theta
        original_color = splat.color.copy()
        original_alpha = splat.alpha

        gradients = SplatGradients(
            position_grad=np.array([0.1, -0.1]),
            scale_grad=np.array([0.05, 0.05]),
            rotation_grad=0.02,
            color_grad=np.array([0.01, -0.01, 0.01]),
            alpha_grad=-0.05
        )

        lrs = {'position': 0.01, 'scale': 0.005, 'rotation': 0.001, 'color': 0.001, 'alpha': 0.001}

        self.optimizer._sgd_update(splat, gradients, lrs)

        # Check that parameters were updated
        assert not np.allclose(splat.mu, original_mu)
        assert not np.allclose(splat.inv_s, original_inv_s)
        assert splat.theta != original_theta
        assert not np.allclose(splat.color, original_color)
        assert splat.alpha != original_alpha

    def test_sgd_momentum_update(self):
        """Test SGD with momentum updates."""
        config = SGDConfig(method=OptimizationMethod.SGD_MOMENTUM, momentum=0.9)
        optimizer = SGDOptimizer(config)

        splat = create_isotropic_gaussian([0.5, 0.5], 0.1, [0.8, 0.2, 0.1], 0.8)
        original_mu = splat.mu.copy()

        gradients = SplatGradients(
            position_grad=np.array([0.1, -0.1]),
            scale_grad=np.array([0.05, 0.05]),
            rotation_grad=0.02,
            color_grad=np.array([0.01, -0.01, 0.01]),
            alpha_grad=-0.05
        )

        lrs = {'position': 0.01, 'scale': 0.005, 'rotation': 0.001, 'color': 0.001, 'alpha': 0.001}

        # First update (initializes momentum)
        optimizer._sgd_momentum_update(splat, gradients, lrs, 0)
        mu_after_first = splat.mu.copy()

        # Second update (uses accumulated momentum)
        optimizer._sgd_momentum_update(splat, gradients, lrs, 0)
        mu_after_second = splat.mu.copy()

        # Check that momentum was applied
        assert not np.allclose(mu_after_first, original_mu)
        assert not np.allclose(mu_after_second, mu_after_first)

    def test_convergence_checking(self):
        """Test convergence criteria checking."""
        # Test gradient norm convergence
        self.optimizer.state.gradient_norms = [0.1, 0.05, 1e-7]
        self.optimizer.config.convergence_threshold = 1e-6
        assert self.optimizer._check_convergence()

        # Test relative improvement convergence
        self.optimizer.state.loss_history = [1.0] * 10 + [0.999] * 10
        self.optimizer.config.relative_tolerance = 1e-2
        assert self.optimizer._check_convergence()

    def test_early_stopping(self):
        """Test early stopping criteria."""
        self.optimizer.config.early_stopping_patience = 5
        self.optimizer.state.patience_counter = 3
        assert not self.optimizer._check_early_stopping()

        self.optimizer.state.patience_counter = 6
        assert self.optimizer._check_early_stopping()

    @patch('src.splat_this.core.sgd_optimizer.logger')
    def test_logging(self, mock_logger):
        """Test optimization progress logging."""
        self.optimizer.state.iteration = 10
        self.optimizer.state.current_loss = 0.5
        self.optimizer.state.gradient_norms = [0.1]

        self.optimizer._log_progress()

        # Check that logger was called
        mock_logger.info.assert_called()

    def test_optimization_integration(self):
        """Test full optimization integration."""
        # Mock the gradient computer to return predictable gradients
        mock_gradients = SplatGradients(
            position_grad=np.array([0.01, -0.01]),
            scale_grad=np.array([0.005, 0.005]),
            rotation_grad=0.001,
            color_grad=np.array([0.001, -0.001, 0.001]),
            alpha_grad=-0.001
        )

        with patch.object(self.optimizer.gradient_computer, 'compute_all_gradients',
                         return_value=mock_gradients):
            result = self.optimizer.optimize_splats(
                self.splats, self.target_image, self.rendered_image, self.error_map
            )

        # Check result structure
        assert isinstance(result, OptimizationResult)
        assert len(result.optimized_splats) == len(self.splats)
        assert result.iterations > 0
        assert result.final_loss >= 0
        assert isinstance(result.optimization_history, OptimizationState)

    def test_different_optimization_methods(self):
        """Test different optimization methods work."""
        methods = [
            OptimizationMethod.SGD,
            OptimizationMethod.SGD_MOMENTUM,
            OptimizationMethod.ADAM,
            OptimizationMethod.RMSPROP,
            OptimizationMethod.ADAGRAD
        ]

        for method in methods:
            config = SGDConfig(method=method, max_iterations=5)
            optimizer = SGDOptimizer(config)

            # Mock gradients
            mock_gradients = SplatGradients(
                position_grad=np.array([0.01, -0.01]),
                scale_grad=np.array([0.005, 0.005]),
                rotation_grad=0.001,
                color_grad=np.array([0.001, -0.001, 0.001]),
                alpha_grad=-0.001
            )

            with patch.object(optimizer.gradient_computer, 'compute_all_gradients',
                             return_value=mock_gradients):
                result = optimizer.optimize_splats(
                    self.splats[:1], self.target_image, self.rendered_image, self.error_map
                )

            assert isinstance(result, OptimizationResult)
            assert len(result.optimized_splats) == 1

    def test_batch_processing(self):
        """Test batch and mini-batch processing."""
        # Test mini-batch mode
        config = SGDConfig(batch_size=2, max_iterations=3)
        optimizer = SGDOptimizer(config)

        mock_gradients = SplatGradients(
            position_grad=np.array([0.01, -0.01]),
            scale_grad=np.array([0.005, 0.005]),
            rotation_grad=0.001,
            color_grad=np.array([0.001, -0.001, 0.001]),
            alpha_grad=-0.001
        )

        with patch.object(optimizer.gradient_computer, 'compute_all_gradients',
                         return_value=mock_gradients):
            result = optimizer.optimize_splats(
                self.splats, self.target_image, self.rendered_image, self.error_map
            )

        assert isinstance(result, OptimizationResult)
        assert len(result.optimized_splats) == len(self.splats)


class TestConvenienceFunctions:
    """Test convenience functions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.splats = [create_isotropic_gaussian([0.5, 0.5], 0.1, [0.8, 0.2, 0.1], 0.8)]
        self.target_image = np.ones((16, 16, 3)) * 0.5
        self.rendered_image = np.ones((16, 16, 3)) * 0.4
        self.error_map = np.abs(self.target_image[:, :, 0] - self.rendered_image[:, :, 0])

    def test_optimize_splats_sgd_default(self):
        """Test convenience function with default configuration."""
        # Mock gradients to avoid dependency on rendering
        mock_gradients = SplatGradients(
            position_grad=np.array([0.01, -0.01]),
            scale_grad=np.array([0.005, 0.005]),
            rotation_grad=0.001,
            color_grad=np.array([0.001, -0.001, 0.001]),
            alpha_grad=-0.001
        )

        with patch('src.splat_this.core.sgd_optimizer.ManualGradientComputer') as mock_computer_class:
            mock_computer = mock_computer_class.return_value
            mock_computer.compute_all_gradients.return_value = mock_gradients
            mock_computer.validate_gradients.return_value = Mock(max_error=0.001, passed=True)

            result = optimize_splats_sgd(
                self.splats, self.target_image, self.rendered_image, self.error_map
            )

        assert isinstance(result, OptimizationResult)
        assert len(result.optimized_splats) == len(self.splats)

    def test_optimize_splats_sgd_custom_config(self):
        """Test convenience function with custom configuration."""
        config = SGDConfig(max_iterations=5, method=OptimizationMethod.ADAM)

        mock_gradients = SplatGradients(
            position_grad=np.array([0.01, -0.01]),
            scale_grad=np.array([0.005, 0.005]),
            rotation_grad=0.001,
            color_grad=np.array([0.001, -0.001, 0.001]),
            alpha_grad=-0.001
        )

        with patch('src.splat_this.core.sgd_optimizer.ManualGradientComputer') as mock_computer_class:
            mock_computer = mock_computer_class.return_value
            mock_computer.compute_all_gradients.return_value = mock_gradients
            mock_computer.validate_gradients.return_value = Mock(max_error=0.001, passed=True)

            result = optimize_splats_sgd(
                self.splats, self.target_image, self.rendered_image, self.error_map, config
            )

        assert isinstance(result, OptimizationResult)

    def test_create_sgd_config_presets(self):
        """Test SGD configuration presets."""
        presets = ["fast", "balanced", "high_quality"]

        for preset in presets:
            config = create_sgd_config_preset(preset)
            assert isinstance(config, SGDConfig)

            # Check that preset-specific parameters are set
            if preset == "fast":
                assert config.max_iterations == 200
                assert config.method == OptimizationMethod.SGD_MOMENTUM
            elif preset == "high_quality":
                assert config.max_iterations == 2000
                assert config.method == OptimizationMethod.ADAM
            else:  # balanced
                assert config.max_iterations == 1000
                assert config.method == OptimizationMethod.SGD_MOMENTUM


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_splat_list(self):
        """Test optimization with empty splat list."""
        config = SGDConfig(max_iterations=5)
        optimizer = SGDOptimizer(config)

        target_image = np.ones((16, 16, 3)) * 0.5
        rendered_image = np.ones((16, 16, 3)) * 0.4
        error_map = np.abs(target_image[:, :, 0] - rendered_image[:, :, 0])

        result = optimizer.optimize_splats([], target_image, rendered_image, error_map)

        assert isinstance(result, OptimizationResult)
        assert len(result.optimized_splats) == 0

    def test_single_splat_optimization(self):
        """Test optimization with single splat."""
        config = SGDConfig(max_iterations=3)
        optimizer = SGDOptimizer(config)

        splat = [create_isotropic_gaussian([0.5, 0.5], 0.1, [0.8, 0.2, 0.1], 0.8)]
        target_image = np.ones((16, 16, 3)) * 0.5
        rendered_image = np.ones((16, 16, 3)) * 0.4
        error_map = np.abs(target_image[:, :, 0] - rendered_image[:, :, 0])

        mock_gradients = SplatGradients(
            position_grad=np.array([0.01, -0.01]),
            scale_grad=np.array([0.005, 0.005]),
            rotation_grad=0.001,
            color_grad=np.array([0.001, -0.001, 0.001]),
            alpha_grad=-0.001
        )

        with patch.object(optimizer.gradient_computer, 'compute_all_gradients',
                         return_value=mock_gradients):
            result = optimizer.optimize_splats(splat, target_image, rendered_image, error_map)

        assert len(result.optimized_splats) == 1

    def test_zero_learning_rates(self):
        """Test behavior with very small learning rates."""
        config = SGDConfig(
            position_lr=1e-10,
            scale_lr=1e-10,
            rotation_lr=1e-10,
            color_lr=1e-10,
            alpha_lr=1e-10,
            max_iterations=5
        )
        optimizer = SGDOptimizer(config)

        splats = [create_isotropic_gaussian([0.5, 0.5], 0.1, [0.8, 0.2, 0.1], 0.8)]
        target_image = np.ones((16, 16, 3)) * 0.5
        rendered_image = np.ones((16, 16, 3)) * 0.4
        error_map = np.abs(target_image[:, :, 0] - rendered_image[:, :, 0])

        original_splat = splats[0].copy()

        mock_gradients = SplatGradients(
            position_grad=np.array([1.0, -1.0]),
            scale_grad=np.array([0.5, 0.5]),
            rotation_grad=0.1,
            color_grad=np.array([0.1, -0.1, 0.1]),
            alpha_grad=-0.1
        )

        with patch.object(optimizer.gradient_computer, 'compute_all_gradients',
                         return_value=mock_gradients):
            result = optimizer.optimize_splats(splats, target_image, rendered_image, error_map)

        # With very small learning rates, parameters should barely change
        optimized_splat = result.optimized_splats[0]
        assert np.allclose(optimized_splat.mu, original_splat.mu, atol=1e-4)

    def test_extreme_gradients(self):
        """Test optimization with extreme gradient values."""
        config = SGDConfig(gradient_clipping=True, clip_threshold=1.0, max_iterations=3)
        optimizer = SGDOptimizer(config)

        splats = [create_isotropic_gaussian([0.5, 0.5], 0.1, [0.8, 0.2, 0.1], 0.8)]
        target_image = np.ones((16, 16, 3)) * 0.5
        rendered_image = np.ones((16, 16, 3)) * 0.4
        error_map = np.abs(target_image[:, :, 0] - rendered_image[:, :, 0])

        # Extreme gradients that should be clipped
        extreme_gradients = SplatGradients(
            position_grad=np.array([1000.0, -1000.0]),
            scale_grad=np.array([500.0, 500.0]),
            rotation_grad=100.0,
            color_grad=np.array([200.0, -200.0, 200.0]),
            alpha_grad=-300.0
        )

        with patch.object(optimizer.gradient_computer, 'compute_all_gradients',
                         return_value=extreme_gradients):
            result = optimizer.optimize_splats(splats, target_image, rendered_image, error_map)

        # Should complete without errors due to gradient clipping
        assert isinstance(result, OptimizationResult)
        assert len(result.optimized_splats) == 1

    def test_convergence_edge_cases(self):
        """Test convergence detection edge cases."""
        optimizer = SGDOptimizer(SGDConfig())

        # Test convergence with empty history
        assert not optimizer._check_convergence()

        # Test convergence with insufficient history for relative improvement
        optimizer.state.gradient_norms = [1e-7]
        optimizer.config.convergence_threshold = 1e-6
        assert optimizer._check_convergence()

    def test_numerical_stability(self):
        """Test numerical stability with edge case parameters."""
        config = SGDConfig(max_iterations=3)
        optimizer = SGDOptimizer(config)

        # Create splat with extreme parameters
        splat = AdaptiveGaussian2D(
            mu=np.array([0.999, 0.001]),  # Near boundary
            inv_s=np.array([1e-3, 1e3]),  # Extreme scales
            theta=np.pi - 1e-6,           # Near boundary
            color=np.array([1.0, 0.0, 1.0]),  # Extreme colors
            alpha=1e-6                    # Very small alpha
        )

        target_image = np.ones((16, 16, 3)) * 0.5
        rendered_image = np.ones((16, 16, 3)) * 0.4
        error_map = np.abs(target_image[:, :, 0] - rendered_image[:, :, 0])

        mock_gradients = SplatGradients(
            position_grad=np.array([0.01, -0.01]),
            scale_grad=np.array([0.005, 0.005]),
            rotation_grad=0.001,
            color_grad=np.array([0.001, -0.001, 0.001]),
            alpha_grad=-0.001
        )

        with patch.object(optimizer.gradient_computer, 'compute_all_gradients',
                         return_value=mock_gradients):
            result = optimizer.optimize_splats([splat], target_image, rendered_image, error_map)

        # Should complete without numerical errors
        assert isinstance(result, OptimizationResult)
        optimized_splat = result.optimized_splats[0]

        # Parameters should remain valid after optimization
        assert np.all(np.isfinite(optimized_splat.mu))
        assert np.all(optimized_splat.inv_s > 0)
        assert np.isfinite(optimized_splat.theta)
        assert np.all(0 <= optimized_splat.color) and np.all(optimized_splat.color <= 1)
        assert 0 <= optimized_splat.alpha <= 1