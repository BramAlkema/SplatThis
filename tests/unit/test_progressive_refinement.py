"""
Unit tests for Progressive Refinement System.

Tests for T3.3: Progressive Refinement System implementation.
Comprehensive testing of progressive refinement functionality including:
- Configuration validation
- Error map computation and analysis
- High-error region identification
- Splat selection and refinement operations
- Integration with SGD optimization
- Convergence criteria and validation
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import copy
import cv2

from src.splat_this.core.progressive_refinement import (
    RefinementConfig,
    ProgressiveRefiner,
    RefinementState,
    RefinementResult,
    ErrorRegion,
    RefinementStrategy,
    RefinementOperation,
    refine_splats_progressively,
    create_refinement_config_preset
)
from src.splat_this.core.adaptive_gaussian import AdaptiveGaussian2D, create_isotropic_gaussian
from src.splat_this.core.sgd_optimizer import OptimizationResult, OptimizationState


class TestRefinementConfig:
    """Test refinement configuration class."""

    def test_init_defaults(self):
        """Test default configuration initialization."""
        config = RefinementConfig()

        assert config.strategy == RefinementStrategy.ERROR_DRIVEN
        assert RefinementOperation.SCALE_ADJUSTMENT in config.enabled_operations
        assert config.error_percentile_threshold == 80.0
        assert config.max_refinement_iterations == 10
        assert config.convergence_threshold == 0.001
        assert config.content_analysis_enabled is True

    def test_init_custom(self):
        """Test custom configuration initialization."""
        config = RefinementConfig(
            strategy=RefinementStrategy.SALIENCY_GUIDED,
            error_percentile_threshold=85.0,
            max_refinement_iterations=15,
            enabled_operations=[RefinementOperation.SCALE_ADJUSTMENT]
        )

        assert config.strategy == RefinementStrategy.SALIENCY_GUIDED
        assert config.error_percentile_threshold == 85.0
        assert config.max_refinement_iterations == 15
        assert len(config.enabled_operations) == 1

    def test_validation_error_percentile(self):
        """Test validation of error percentile threshold."""
        with pytest.raises(ValueError, match="error_percentile_threshold must be in \\[0,100\\]"):
            RefinementConfig(error_percentile_threshold=-10)

        with pytest.raises(ValueError, match="error_percentile_threshold must be in \\[0,100\\]"):
            RefinementConfig(error_percentile_threshold=110)

    def test_validation_min_error_threshold(self):
        """Test validation of minimum error threshold."""
        with pytest.raises(ValueError, match="min_error_threshold must be non-negative"):
            RefinementConfig(min_error_threshold=-0.1)

    def test_validation_max_iterations(self):
        """Test validation of maximum iterations."""
        with pytest.raises(ValueError, match="max_refinement_iterations must be positive"):
            RefinementConfig(max_refinement_iterations=0)

        with pytest.raises(ValueError, match="max_refinement_iterations must be positive"):
            RefinementConfig(max_refinement_iterations=-5)

    def test_validation_convergence_threshold(self):
        """Test validation of convergence threshold."""
        with pytest.raises(ValueError, match="convergence_threshold must be positive"):
            RefinementConfig(convergence_threshold=0.0)

        with pytest.raises(ValueError, match="convergence_threshold must be positive"):
            RefinementConfig(convergence_threshold=-0.001)

    def test_validation_overlap_threshold(self):
        """Test validation of splat overlap threshold."""
        with pytest.raises(ValueError, match="splat_overlap_threshold must be in \\[0,1\\]"):
            RefinementConfig(splat_overlap_threshold=-0.1)

        with pytest.raises(ValueError, match="splat_overlap_threshold must be in \\[0,1\\]"):
            RefinementConfig(splat_overlap_threshold=1.5)


class TestErrorRegion:
    """Test error region data structure."""

    def test_error_region_creation(self):
        """Test error region creation."""
        region = ErrorRegion(
            center=(10, 15),
            bbox=(5, 10, 15, 20),
            error_magnitude=0.5,
            area=50,
            overlapping_splats=[0, 1, 2]
        )

        assert region.center == (10, 15)
        assert region.bbox == (5, 10, 15, 20)
        assert region.error_magnitude == 0.5
        assert region.area == 50
        assert region.overlapping_splats == [0, 1, 2]
        assert region.content_complexity == 0.0


class TestRefinementState:
    """Test refinement state tracking."""

    def test_state_initialization(self):
        """Test refinement state initialization."""
        state = RefinementState()

        assert state.iteration == 0
        assert state.total_error == float('inf')
        assert state.best_error == float('inf')
        assert len(state.error_history) == 0
        assert len(state.refined_splats) == 0
        assert not state.converged

    def test_state_reset(self):
        """Test refinement state reset functionality."""
        state = RefinementState()

        # Modify state
        state.iteration = 5
        state.total_error = 0.5
        state.best_error = 0.3
        state.error_history = [1.0, 0.8, 0.5]
        state.refined_splats = {0, 1, 2}
        state.converged = True

        # Reset state
        state.reset()

        assert state.iteration == 0
        assert state.total_error == float('inf')
        assert state.best_error == float('inf')
        assert len(state.error_history) == 0
        assert len(state.refined_splats) == 0
        assert not state.converged


class TestProgressiveRefiner:
    """Test progressive refiner functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = RefinementConfig(max_refinement_iterations=3, log_progress=False)
        self.refiner = ProgressiveRefiner(self.config)

        # Create test splats
        self.splats = [
            create_isotropic_gaussian([0.3, 0.3], 0.1, [0.8, 0.2, 0.1], 0.8),
            create_isotropic_gaussian([0.7, 0.7], 0.08, [0.1, 0.8, 0.2], 0.7),
            create_isotropic_gaussian([0.5, 0.5], 0.12, [0.2, 0.1, 0.8], 0.9)
        ]

        # Create test images
        self.target_image = np.ones((32, 32, 3)) * 0.6
        self.rendered_image = np.ones((32, 32, 3)) * 0.4

        # Create error map with some high-error regions
        error_map = np.ones((32, 32)) * 0.2
        error_map[10:15, 10:15] = 0.8  # High error region
        error_map[20:25, 20:25] = 0.7  # Another high error region
        self.error_map = error_map

    def test_refiner_initialization(self):
        """Test refiner initialization."""
        assert self.refiner.config == self.config
        assert self.refiner.sgd_optimizer is not None
        assert self.refiner.gradient_computer is not None
        assert isinstance(self.refiner.state, RefinementState)

    def test_compute_error_map(self):
        """Test error map computation."""
        error_map = self.refiner._compute_error_map(self.target_image, self.rendered_image)

        assert error_map.shape == (32, 32)
        assert np.all(error_map >= 0)
        expected_error = np.linalg.norm(self.target_image - self.rendered_image, axis=-1)
        assert np.allclose(error_map, expected_error, atol=1e-6)

    def test_compute_error_map_with_smoothing(self):
        """Test error map computation with smoothing kernel."""
        config = RefinementConfig(error_analysis_kernel_size=3)
        refiner = ProgressiveRefiner(config)

        error_map = refiner._compute_error_map(self.target_image, self.rendered_image)

        assert error_map.shape == (32, 32)
        assert np.all(error_map >= 0)

    def test_identify_high_error_regions(self):
        """Test high-error region identification."""
        # Create error map with clear high-error regions
        error_map = np.zeros((32, 32))
        error_map[5:10, 5:10] = 1.0   # High error region
        error_map[20:25, 15:20] = 0.9  # Another high error region

        regions = self.refiner._identify_high_error_regions(
            error_map, self.target_image, self.rendered_image
        )

        assert len(regions) >= 1  # Should find at least one region
        assert all(isinstance(r, ErrorRegion) for r in regions)

        # Check region properties
        for region in regions:
            assert region.error_magnitude > 0
            assert region.area > 0
            assert len(region.bbox) == 4
            assert len(region.center) == 2

    def test_select_splats_for_refinement(self):
        """Test splat selection for refinement."""
        # Create mock error regions
        regions = [
            ErrorRegion(
                center=(10, 10),
                bbox=(8, 8, 12, 12),
                error_magnitude=0.8,
                area=16,
                overlapping_splats=[]
            ),
            ErrorRegion(
                center=(20, 20),
                bbox=(18, 18, 22, 22),
                error_magnitude=0.7,
                area=16,
                overlapping_splats=[]
            )
        ]

        selected = self.refiner._select_splats_for_refinement(
            self.splats, regions, self.error_map
        )

        assert isinstance(selected, list)
        assert len(selected) >= self.config.min_splats_per_iteration
        assert len(selected) <= self.config.max_splats_per_iteration
        assert all(0 <= idx < len(self.splats) for idx in selected)

    def test_perform_refinement_operations(self):
        """Test refinement operations performance."""
        # Create mock error region
        region = ErrorRegion(
            center=(16, 16),
            bbox=(14, 14, 18, 18),
            error_magnitude=0.8,
            area=16,
            overlapping_splats=[0, 1]
        )

        # Store original splat parameters
        original_splats = [splat.copy() for splat in self.splats]

        operations_count = self.refiner._perform_refinement_operations(
            self.splats, [0, 1], [region], self.target_image, self.rendered_image, self.error_map
        )

        assert operations_count >= 0

        # Check that some parameters were modified (at least for enabled operations)
        if operations_count > 0:
            changes_detected = False
            for i in [0, 1]:
                if (not np.allclose(self.splats[i].mu, original_splats[i].mu) or
                    not np.allclose(self.splats[i].inv_s, original_splats[i].inv_s) or
                    self.splats[i].alpha != original_splats[i].alpha):
                    changes_detected = True
                    break
            assert changes_detected

    def test_refine_scale(self):
        """Test scale refinement operation."""
        splat = self.splats[0].copy()
        original_inv_s = splat.inv_s.copy()

        region = ErrorRegion(
            center=(16, 16),
            bbox=(14, 14, 18, 18),
            error_magnitude=0.5,
            area=16,
            overlapping_splats=[0],
            content_complexity=0.3
        )

        changed = self.refiner._refine_scale(splat, region, self.error_map)

        if changed:
            assert not np.allclose(splat.inv_s, original_inv_s)
            assert np.all(splat.inv_s > 0)  # Ensure valid scales

    def test_refine_position(self):
        """Test position refinement operation."""
        splat = self.splats[0].copy()
        original_mu = splat.mu.copy()

        region = ErrorRegion(
            center=(20, 20),  # Different from splat position
            bbox=(18, 18, 22, 22),
            error_magnitude=0.5,
            area=16,
            overlapping_splats=[0]
        )

        changed = self.refiner._refine_position(splat, region, self.error_map)

        if changed:
            assert not np.allclose(splat.mu, original_mu)
            assert np.all(0 <= splat.mu) and np.all(splat.mu <= 1)  # Ensure valid positions

    def test_refine_alpha(self):
        """Test alpha refinement operation."""
        splat = self.splats[0].copy()
        original_alpha = splat.alpha

        region = ErrorRegion(
            center=(16, 16),
            bbox=(14, 14, 18, 18),
            error_magnitude=0.8,
            area=16,
            overlapping_splats=[0]
        )

        changed = self.refiner._refine_alpha(splat, region, self.error_map)

        if changed:
            assert splat.alpha != original_alpha
            assert 0 <= splat.alpha <= 1  # Ensure valid alpha

    def test_refine_anisotropy(self):
        """Test anisotropy refinement operation."""
        splat = self.splats[0].copy()
        original_inv_s = splat.inv_s.copy()
        original_theta = splat.theta

        # Create error map with gradient structure
        error_map = np.zeros((32, 32))
        error_map[10:20, 15] = 1.0  # Vertical line of high error

        region = ErrorRegion(
            center=(15, 15),
            bbox=(10, 14, 20, 16),
            error_magnitude=0.8,
            area=60,
            overlapping_splats=[0]
        )

        changed = self.refiner._refine_anisotropy(splat, region, error_map)

        # Anisotropy refinement may or may not change parameters depending on gradient
        if changed:
            assert (not np.allclose(splat.inv_s, original_inv_s) or
                   splat.theta != original_theta)

    def test_analyze_content_complexity(self):
        """Test content complexity analysis."""
        # Create image with varying complexity
        image = np.random.random((32, 32, 3))

        # Smooth region
        image[5:10, 5:10] = 0.5

        # Complex region
        image[20:25, 20:25] = np.random.random((5, 5, 3))

        smooth_complexity = self.refiner._analyze_content_complexity(image, (5, 5, 10, 10))
        complex_complexity = self.refiner._analyze_content_complexity(image, (20, 20, 25, 25))

        assert 0 <= smooth_complexity <= 1
        assert 0 <= complex_complexity <= 1
        # Complex region should have higher complexity
        assert complex_complexity >= smooth_complexity

    def test_check_convergence(self):
        """Test convergence checking."""
        # Test no convergence with insufficient history
        assert not self.refiner._check_convergence()

        # Test convergence with small relative improvement
        self.refiner.state.error_history = [1.0, 0.999, 0.998]
        self.refiner.config.convergence_threshold = 0.01
        assert self.refiner._check_convergence()

        # Test convergence with patience exceeded
        self.refiner.state.error_history = [1.0, 0.9, 0.8]  # Good improvement
        self.refiner.state.patience_counter = 5
        self.refiner.config.refinement_patience = 3
        assert self.refiner._check_convergence()

    def test_validate_refinement(self):
        """Test refinement validation."""
        # Should not raise any exceptions with valid splats
        self.refiner._validate_refinement(self.splats, self.target_image, self.rendered_image)

        # Test with invalid splat (should log warning)
        invalid_splat = self.splats[0].copy()
        invalid_splat.alpha = 2.0  # Invalid alpha

        with patch('src.splat_this.core.progressive_refinement.logger') as mock_logger:
            self.refiner._validate_refinement([invalid_splat], self.target_image, self.rendered_image)
            mock_logger.warning.assert_called()

    @patch('src.splat_this.core.progressive_refinement.ProgressiveRefiner._apply_sgd_optimization')
    def test_refine_splats_integration(self, mock_sgd):
        """Test full refinement integration."""
        # Mock SGD optimization result
        mock_result = OptimizationResult(
            optimized_splats=[splat.copy() for splat in self.splats[:2]],
            final_loss=0.5,
            iterations=10,
            converged=True,
            early_stopped=False,
            optimization_history=OptimizationState()
        )
        mock_sgd.return_value = mock_result

        # Create target with high error in specific region
        target = np.ones((32, 32, 3)) * 0.8
        rendered = np.ones((32, 32, 3)) * 0.2
        rendered[10:15, 10:15] = 0.9  # Create error pattern

        result = self.refiner.refine_splats(self.splats, target, rendered)

        assert isinstance(result, RefinementResult)
        assert len(result.refined_splats) == len(self.splats)
        assert result.iterations > 0
        assert result.final_error >= 0
        assert isinstance(result.refinement_history, RefinementState)

    def test_update_rendered_image(self):
        """Test rendered image update (placeholder)."""
        updated = self.refiner._update_rendered_image(self.splats, self.target_image)

        assert updated.shape == self.target_image.shape
        assert np.all(np.isfinite(updated))

    def test_default_loss_function(self):
        """Test default loss function."""
        loss = self.refiner._default_loss_function(
            self.splats, self.target_image, self.rendered_image, self.error_map
        )

        assert isinstance(loss, (int, float))
        assert loss >= 0


class TestConvenienceFunctions:
    """Test convenience functions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.splats = [create_isotropic_gaussian([0.5, 0.5], 0.1, [0.8, 0.2, 0.1], 0.8)]
        self.target_image = np.ones((16, 16, 3)) * 0.6
        self.rendered_image = np.ones((16, 16, 3)) * 0.4

    @patch('src.splat_this.core.progressive_refinement.ProgressiveRefiner.refine_splats')
    def test_refine_splats_progressively(self, mock_refine):
        """Test convenience function for progressive refinement."""
        # Mock refinement result
        mock_result = RefinementResult(
            refined_splats=self.splats,
            final_error=0.5,
            iterations=5,
            converged=True,
            refinement_history=RefinementState(),
            total_operations=10
        )
        mock_refine.return_value = mock_result

        result = refine_splats_progressively(
            self.splats, self.target_image, self.rendered_image
        )

        assert isinstance(result, RefinementResult)
        mock_refine.assert_called_once()

    def test_create_refinement_config_presets(self):
        """Test refinement configuration presets."""
        presets = ["fast", "balanced", "high_quality"]

        for preset in presets:
            config = create_refinement_config_preset(preset)
            assert isinstance(config, RefinementConfig)

            # Check that preset-specific parameters are set
            if preset == "fast":
                assert config.max_refinement_iterations == 5
                assert config.error_percentile_threshold == 85.0
            elif preset == "high_quality":
                assert config.max_refinement_iterations == 20
                assert config.error_percentile_threshold == 75.0
                assert config.content_analysis_enabled is True
            else:  # balanced
                assert config.max_refinement_iterations == 10
                assert config.error_percentile_threshold == 80.0


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = RefinementConfig(max_refinement_iterations=2, log_progress=False)
        self.refiner = ProgressiveRefiner(self.config)

    def test_empty_splat_list(self):
        """Test refinement with empty splat list."""
        target = np.ones((16, 16, 3)) * 0.5
        rendered = np.ones((16, 16, 3)) * 0.4

        result = self.refiner.refine_splats([], target, rendered)

        assert isinstance(result, RefinementResult)
        assert len(result.refined_splats) == 0
        assert result.total_operations == 0

    def test_single_splat_refinement(self):
        """Test refinement with single splat."""
        splat = [create_isotropic_gaussian([0.5, 0.5], 0.1, [0.8, 0.2, 0.1], 0.8)]
        target = np.ones((16, 16, 3)) * 0.8
        rendered = np.ones((16, 16, 3)) * 0.2

        with patch.object(self.refiner, '_apply_sgd_optimization') as mock_sgd:
            mock_sgd.return_value = OptimizationResult(
                optimized_splats=splat,
                final_loss=0.5,
                iterations=5,
                converged=True,
                early_stopped=False,
                optimization_history=OptimizationState()
            )

            result = self.refiner.refine_splats(splat, target, rendered)

        assert len(result.refined_splats) == 1

    def test_no_high_error_regions(self):
        """Test refinement when no high-error regions are found."""
        splats = [create_isotropic_gaussian([0.5, 0.5], 0.1, [0.8, 0.2, 0.1], 0.8)]
        target = np.ones((16, 16, 3)) * 0.5
        rendered = np.ones((16, 16, 3)) * 0.5  # Perfect match

        result = self.refiner.refine_splats(splats, target, rendered)

        assert isinstance(result, RefinementResult)
        # Should stop early due to no high-error regions

    def test_uniform_error_map(self):
        """Test refinement with uniform error map."""
        splats = [create_isotropic_gaussian([0.5, 0.5], 0.1, [0.8, 0.2, 0.1], 0.8)]
        target = np.ones((16, 16, 3)) * 0.6
        rendered = np.ones((16, 16, 3)) * 0.4  # Uniform error

        with patch.object(self.refiner, '_apply_sgd_optimization') as mock_sgd:
            mock_sgd.return_value = OptimizationResult(
                optimized_splats=splats,
                final_loss=0.5,
                iterations=5,
                converged=True,
                early_stopped=False,
                optimization_history=OptimizationState()
            )

            result = self.refiner.refine_splats(splats, target, rendered)

        assert isinstance(result, RefinementResult)

    def test_extreme_error_values(self):
        """Test refinement with extreme error values."""
        splats = [create_isotropic_gaussian([0.5, 0.5], 0.1, [0.8, 0.2, 0.1], 0.8)]
        target = np.ones((16, 16, 3)) * 1.0
        rendered = np.zeros((16, 16, 3))  # Maximum error

        with patch.object(self.refiner, '_apply_sgd_optimization') as mock_sgd:
            mock_sgd.return_value = OptimizationResult(
                optimized_splats=splats,
                final_loss=0.5,
                iterations=5,
                converged=True,
                early_stopped=False,
                optimization_history=OptimizationState()
            )

            result = self.refiner.refine_splats(splats, target, rendered)

        assert isinstance(result, RefinementResult)
        assert result.final_error >= 0

    def test_small_image_refinement(self):
        """Test refinement with very small images."""
        splats = [create_isotropic_gaussian([0.5, 0.5], 0.1, [0.8, 0.2, 0.1], 0.8)]
        target = np.ones((4, 4, 3)) * 0.6
        rendered = np.ones((4, 4, 3)) * 0.4

        with patch.object(self.refiner, '_apply_sgd_optimization') as mock_sgd:
            mock_sgd.return_value = OptimizationResult(
                optimized_splats=splats,
                final_loss=0.5,
                iterations=5,
                converged=True,
                early_stopped=False,
                optimization_history=OptimizationState()
            )

            result = self.refiner.refine_splats(splats, target, rendered)

        assert isinstance(result, RefinementResult)

    def test_convergence_edge_cases(self):
        """Test convergence detection edge cases."""
        # Test convergence with empty history
        assert not self.refiner._check_convergence()

        # Test convergence with single error value
        self.refiner.state.error_history = [1.0]
        assert not self.refiner._check_convergence()

        # Test convergence with identical errors
        self.refiner.state.error_history = [1.0, 1.0, 1.0]
        self.refiner.config.convergence_threshold = 0.01
        assert self.refiner._check_convergence()

    def test_numerical_stability(self):
        """Test numerical stability with edge case parameters."""
        # Create splat with extreme parameters
        splat = AdaptiveGaussian2D(
            mu=np.array([0.001, 0.999]),  # Near boundaries
            inv_s=np.array([1e-3, 1e3]),  # Extreme scales
            theta=np.pi - 1e-6,           # Near boundary
            color=np.array([1.0, 0.0, 1.0]),  # Extreme colors
            alpha=1e-6                    # Very small alpha
        )

        target = np.ones((16, 16, 3)) * 0.5
        rendered = np.ones((16, 16, 3)) * 0.4

        with patch.object(self.refiner, '_apply_sgd_optimization') as mock_sgd:
            mock_sgd.return_value = OptimizationResult(
                optimized_splats=[splat],
                final_loss=0.5,
                iterations=5,
                converged=True,
                early_stopped=False,
                optimization_history=OptimizationState()
            )

            result = self.refiner.refine_splats([splat], target, rendered)

        # Should complete without numerical errors
        assert isinstance(result, RefinementResult)
        refined_splat = result.refined_splats[0]

        # Parameters should remain valid after refinement
        assert np.all(np.isfinite(refined_splat.mu))
        assert np.all(refined_splat.inv_s > 0)
        assert np.isfinite(refined_splat.theta)
        assert np.all(0 <= refined_splat.color) and np.all(refined_splat.color <= 1)
        assert 0 <= refined_splat.alpha <= 1

    def test_invalid_image_dimensions(self):
        """Test refinement with mismatched image dimensions."""
        splats = [create_isotropic_gaussian([0.5, 0.5], 0.1, [0.8, 0.2, 0.1], 0.8)]
        target = np.ones((16, 16, 3)) * 0.6
        rendered = np.ones((8, 8, 3)) * 0.4  # Different size

        # Should handle gracefully or raise appropriate error
        try:
            result = self.refiner.refine_splats(splats, target, rendered)
            # If it succeeds, ensure result is reasonable
            assert isinstance(result, RefinementResult)
        except (ValueError, AttributeError):
            # Expected behavior for mismatched dimensions
            pass

    def test_disabled_operations(self):
        """Test refinement with all operations disabled."""
        config = RefinementConfig(
            enabled_operations=[],  # No operations enabled
            max_refinement_iterations=2,
            log_progress=False
        )
        refiner = ProgressiveRefiner(config)

        splats = [create_isotropic_gaussian([0.5, 0.5], 0.1, [0.8, 0.2, 0.1], 0.8)]
        target = np.ones((16, 16, 3)) * 0.8
        rendered = np.ones((16, 16, 3)) * 0.2

        result = refiner.refine_splats(splats, target, rendered)

        assert isinstance(result, RefinementResult)
        # Should have minimal operations performed
        assert result.total_operations >= 0