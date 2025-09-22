"""
Unit tests for Anisotropic Refinement System.

Tests for T4.1: Anisotropic Refinement System implementation.
Comprehensive testing of edge-aware anisotropic refinement functionality including:
- Configuration validation
- Structure tensor analysis and edge detection
- Aspect ratio enhancement and orientation alignment
- Edge sharpening and quality metrics
- Anisotropic constraints and validation
- Integration testing and convergence
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import copy

from src.splat_this.core.anisotropic_refinement import (
    AnisotropicConfig,
    AnisotropicRefiner,
    AnisotropicAnalysis,
    AnisotropicRefinementResult,
    AnisotropyStrategy,
    AnisotropyOperation,
    create_anisotropic_config_preset,
    refine_splats_anisotropically
)
from src.splat_this.core.adaptive_gaussian import AdaptiveGaussian2D, create_isotropic_gaussian


class TestAnisotropicConfig:
    """Test anisotropic configuration validation and creation."""

    def test_init_defaults(self):
        """Test default configuration initialization."""
        config = AnisotropicConfig()

        assert config.strategy == AnisotropyStrategy.HYBRID
        assert AnisotropyOperation.ASPECT_RATIO_ENHANCEMENT in config.enabled_operations
        assert AnisotropyOperation.ORIENTATION_ALIGNMENT in config.enabled_operations
        assert AnisotropyOperation.EDGE_SHARPENING in config.enabled_operations

        assert config.max_aspect_ratio == 8.0
        assert config.min_aspect_ratio == 1.0
        assert config.gradient_method == "sobel"
        assert config.edge_strength_threshold == 0.1

    def test_init_custom(self):
        """Test custom configuration initialization."""
        config = AnisotropicConfig(
            strategy=AnisotropyStrategy.EDGE_FOLLOWING,
            max_aspect_ratio=12.0,
            gradient_method="scharr",
            edge_strength_threshold=0.05
        )

        assert config.strategy == AnisotropyStrategy.EDGE_FOLLOWING
        assert config.max_aspect_ratio == 12.0
        assert config.gradient_method == "scharr"
        assert config.edge_strength_threshold == 0.05

    def test_validation_aspect_ratio(self):
        """Test aspect ratio validation."""
        with pytest.raises(ValueError, match="max_aspect_ratio must be greater than min_aspect_ratio"):
            AnisotropicConfig(max_aspect_ratio=2.0, min_aspect_ratio=3.0)

    def test_validation_edge_threshold(self):
        """Test edge strength threshold validation."""
        with pytest.raises(ValueError, match="edge_strength_threshold must be non-negative"):
            AnisotropicConfig(edge_strength_threshold=-0.1)

    def test_validation_coherence_threshold(self):
        """Test coherence threshold validation."""
        with pytest.raises(ValueError, match="coherence_threshold must be in"):
            AnisotropicConfig(coherence_threshold=1.5)

        with pytest.raises(ValueError, match="coherence_threshold must be in"):
            AnisotropicConfig(coherence_threshold=-0.1)


class TestAnisotropicAnalysis:
    """Test anisotropic analysis data structure."""

    def test_analysis_creation(self):
        """Test creation of anisotropic analysis."""
        edge_strength = np.random.rand(64, 64)
        orientation = np.random.rand(64, 64) * np.pi
        coherence = np.random.rand(64, 64)
        gradient_magnitude = np.random.rand(64, 64)
        structure_tensor = np.random.rand(64, 64, 2, 2)
        quality_map = np.random.rand(64, 64)

        analysis = AnisotropicAnalysis(
            edge_strength=edge_strength,
            orientation=orientation,
            coherence=coherence,
            gradient_magnitude=gradient_magnitude,
            structure_tensor=structure_tensor,
            quality_map=quality_map
        )

        assert analysis.edge_strength.shape == (64, 64)
        assert analysis.orientation.shape == (64, 64)
        assert analysis.coherence.shape == (64, 64)
        assert analysis.gradient_magnitude.shape == (64, 64)
        assert analysis.structure_tensor.shape == (64, 64, 2, 2)
        assert analysis.quality_map.shape == (64, 64)


class TestAnisotropicRefiner:
    """Test anisotropic refiner functionality."""

    def test_refiner_initialization(self):
        """Test anisotropic refiner initialization."""
        config = AnisotropicConfig()
        refiner = AnisotropicRefiner(config)

        assert refiner.config == config
        assert refiner.gradient_analyzer is not None
        assert refiner.refinement_history == []

    def test_analyze_anisotropic_structure(self):
        """Test anisotropic structure analysis."""
        config = AnisotropicConfig()
        refiner = AnisotropicRefiner(config)

        # Create test image with edge structure
        image = np.zeros((64, 64, 3))
        image[20:44, :, :] = 1.0  # Horizontal edge
        image[:, 20:44, :] = 0.5  # Vertical edge

        analysis = refiner.analyze_anisotropic_structure(image)

        assert isinstance(analysis, AnisotropicAnalysis)
        assert analysis.edge_strength.shape == (64, 64)
        assert analysis.orientation.shape == (64, 64)
        assert analysis.coherence.shape == (64, 64)
        assert analysis.gradient_magnitude.shape == (64, 64)
        assert analysis.structure_tensor.shape == (64, 64, 2, 2)
        assert analysis.quality_map.shape == (64, 64)

        # Check that edges are detected
        assert np.any(analysis.edge_strength > 0.1)
        assert np.any(analysis.gradient_magnitude > 0.05)

    def test_compute_quality_map(self):
        """Test quality map computation."""
        config = AnisotropicConfig()
        refiner = AnisotropicRefiner(config)

        edge_strength = np.random.rand(32, 32) * 0.5
        coherence = np.random.rand(32, 32) * 0.8
        gradient_magnitude = np.random.rand(32, 32) * 0.3

        quality_map = refiner._compute_quality_map(edge_strength, coherence, gradient_magnitude)

        assert quality_map.shape == (32, 32)
        assert np.all(quality_map >= 0)
        assert np.all(quality_map <= 1)

    def test_refine_splat_anisotropy(self):
        """Test single splat anisotropic refinement."""
        config = AnisotropicConfig()
        refiner = AnisotropicRefiner(config)

        # Create test splat
        splat = create_isotropic_gaussian(
            center=np.array([0.5, 0.5]),
            scale=0.1,
            color=np.array([1.0, 0.5, 0.3]),
            alpha=0.8
        )

        # Create mock analysis with strong edge
        analysis = Mock(spec=AnisotropicAnalysis)
        analysis.edge_strength = np.ones((64, 64)) * 0.5
        analysis.orientation = np.zeros((64, 64))
        analysis.coherence = np.ones((64, 64)) * 0.8
        analysis.quality_map = np.ones((64, 64)) * 0.9

        refined_splat = refiner.refine_splat_anisotropy(splat, analysis, (64, 64))

        assert isinstance(refined_splat, AdaptiveGaussian2D)
        assert refined_splat.refinement_count == splat.refinement_count + 1

    def test_enhance_aspect_ratio(self):
        """Test aspect ratio enhancement."""
        config = AnisotropicConfig()
        refiner = AnisotropicRefiner(config)

        splat = create_isotropic_gaussian(
            center=np.array([0.5, 0.5]),
            scale=0.1,
            color=np.array([1.0, 0.5, 0.3]),
            alpha=0.8
        )

        original_scales = 1.0 / splat.inv_s
        original_aspect_ratio = max(original_scales) / min(original_scales)

        # Test with strong edge
        enhanced_splat = refiner._enhance_aspect_ratio(splat, edge_strength=0.5, coherence=0.8)

        enhanced_scales = 1.0 / enhanced_splat.inv_s
        enhanced_aspect_ratio = max(enhanced_scales) / min(enhanced_scales)

        # Should increase aspect ratio for strong edges
        assert enhanced_aspect_ratio >= original_aspect_ratio

        # Test with weak edge (should not change much)
        weak_enhanced = refiner._enhance_aspect_ratio(splat, edge_strength=0.05, coherence=0.2)
        weak_scales = 1.0 / weak_enhanced.inv_s
        weak_aspect_ratio = max(weak_scales) / min(weak_scales)

        assert abs(weak_aspect_ratio - original_aspect_ratio) < 0.5

    def test_align_orientation(self):
        """Test orientation alignment."""
        config = AnisotropicConfig()
        refiner = AnisotropicRefiner(config)

        splat = create_isotropic_gaussian(
            center=np.array([0.5, 0.5]),
            scale=0.1,
            color=np.array([1.0, 0.5, 0.3]),
            alpha=0.8
        )
        splat.theta = 0.0  # Start with zero orientation
        original_theta = splat.theta

        target_orientation = np.pi / 4  # 45 degrees

        # Test with strong coherence
        aligned_splat = refiner._align_orientation(splat, target_orientation, coherence=0.8)

        # Orientation should move toward target
        if abs(target_orientation - original_theta) > config.orientation_tolerance:
            assert abs(aligned_splat.theta - target_orientation) < abs(original_theta - target_orientation)

        # Test with weak coherence (should not change much)
        splat.theta = 0.0  # Reset
        weak_aligned = refiner._align_orientation(splat, target_orientation, coherence=0.1)
        assert abs(weak_aligned.theta - 0.0) < 0.1

    def test_sharpen_edges(self):
        """Test edge sharpening."""
        config = AnisotropicConfig()
        refiner = AnisotropicRefiner(config)

        splat = create_isotropic_gaussian(
            center=np.array([0.5, 0.5]),
            scale=0.1,
            color=np.array([1.0, 0.5, 0.3]),
            alpha=0.8
        )

        original_scales = 1.0 / splat.inv_s

        # Test with strong edge
        sharpened_splat = refiner._sharpen_edges(splat, edge_strength=0.5, quality=0.8)

        sharpened_scales = 1.0 / sharpened_splat.inv_s

        # Should increase anisotropy (one scale should decrease)
        aspect_ratio_original = max(original_scales) / min(original_scales)
        aspect_ratio_sharpened = max(sharpened_scales) / min(sharpened_scales)

        assert aspect_ratio_sharpened >= aspect_ratio_original

    def test_apply_anisotropic_constraints(self):
        """Test anisotropic constraint application."""
        config = AnisotropicConfig(max_aspect_ratio=5.0)
        refiner = AnisotropicRefiner(config)

        # Create splat with extreme aspect ratio
        splat = create_isotropic_gaussian(
            center=np.array([0.5, 0.5]),
            scale=0.1,
            color=np.array([1.0, 0.5, 0.3]),
            alpha=0.8
        )

        # Make extremely anisotropic
        splat.inv_s = np.array([0.01, 1.0])  # Very large aspect ratio

        constrained_splat = refiner._apply_anisotropic_constraints(splat)

        # Check aspect ratio constraint
        constrained_scales = 1.0 / constrained_splat.inv_s
        aspect_ratio = max(constrained_scales) / min(constrained_scales)
        assert aspect_ratio <= config.max_aspect_ratio

        # Check parameter constraints
        assert 0.0 <= constrained_splat.alpha <= 1.0
        assert np.all(constrained_splat.color >= 0.0)
        assert np.all(constrained_splat.color <= 1.0)

    def test_compute_aspect_ratio_quality(self):
        """Test aspect ratio quality computation."""
        config = AnisotropicConfig()
        refiner = AnisotropicRefiner(config)

        splat = create_isotropic_gaussian(
            center=np.array([0.5, 0.5]),
            scale=0.1,
            color=np.array([1.0, 0.5, 0.3]),
            alpha=0.8
        )

        # Create mock analysis
        analysis = Mock(spec=AnisotropicAnalysis)
        analysis.edge_strength = np.ones((64, 64)) * 0.5
        analysis.coherence = np.ones((64, 64)) * 0.8

        quality = refiner._compute_aspect_ratio_quality(splat, analysis, (32, 32))

        assert 0.0 <= quality <= 1.0

    def test_compute_orientation_quality(self):
        """Test orientation quality computation."""
        config = AnisotropicConfig()
        refiner = AnisotropicRefiner(config)

        splat = create_isotropic_gaussian(
            center=np.array([0.5, 0.5]),
            scale=0.1,
            color=np.array([1.0, 0.5, 0.3]),
            alpha=0.8
        )
        splat.theta = np.pi / 4

        # Create mock analysis
        analysis = Mock(spec=AnisotropicAnalysis)
        analysis.orientation = np.ones((64, 64)) * (np.pi / 4)  # Same orientation
        analysis.coherence = np.ones((64, 64)) * 0.8

        quality = refiner._compute_orientation_quality(splat, analysis, (32, 32))

        assert 0.0 <= quality <= 1.0
        assert quality > 0.8  # Should be high for aligned orientations

    def test_validate_refinement(self):
        """Test refinement validation."""
        config = AnisotropicConfig()
        refiner = AnisotropicRefiner(config)

        original_splat = create_isotropic_gaussian(
            center=np.array([0.5, 0.5]),
            scale=0.1,
            color=np.array([1.0, 0.5, 0.3]),
            alpha=0.8
        )

        # Create better aligned splat
        refined_splat = create_isotropic_gaussian(
            center=np.array([0.5, 0.5]),
            scale=0.1,
            color=np.array([1.0, 0.5, 0.3]),
            alpha=0.8
        )
        refined_splat.theta = np.pi / 4  # Better aligned

        # Create mock analysis that prefers the refined orientation
        analysis = Mock(spec=AnisotropicAnalysis)
        analysis.edge_strength = np.ones((64, 64)) * 0.5
        analysis.orientation = np.ones((64, 64)) * (np.pi / 4)
        analysis.coherence = np.ones((64, 64)) * 0.8

        is_better = refiner._validate_refinement(original_splat, refined_splat, analysis)

        assert isinstance(is_better, (bool, np.bool_))

    def test_refine_splats_anisotropically_integration(self):
        """Test full anisotropic refinement integration."""
        config = AnisotropicConfig(max_refinement_iterations=2)
        refiner = AnisotropicRefiner(config)

        # Create test splats
        splats = [
            create_isotropic_gaussian(
                center=np.array([0.3, 0.3]),
                scale=0.1,
                color=np.array([1.0, 0.0, 0.0]),
                alpha=0.8
            ),
            create_isotropic_gaussian(
                center=np.array([0.7, 0.7]),
                scale=0.1,
                color=np.array([0.0, 1.0, 0.0]),
                alpha=0.8
            )
        ]

        # Create test image with edges
        target_image = np.zeros((64, 64, 3))
        target_image[20:44, :, :] = 1.0  # Horizontal edge

        result = refiner.refine_splats_anisotropically(splats, target_image)

        assert isinstance(result, AnisotropicRefinementResult)
        assert len(result.refined_splats) == len(splats)
        assert result.iterations >= 1
        assert isinstance(result.converged, bool)
        assert isinstance(result.final_quality, float)
        assert isinstance(result.quality_improvement, float)
        assert len(result.refinement_history) == result.iterations
        assert isinstance(result.anisotropic_analysis, AnisotropicAnalysis)

    def test_compute_overall_quality(self):
        """Test overall quality computation."""
        config = AnisotropicConfig()
        refiner = AnisotropicRefiner(config)

        splats = [
            create_isotropic_gaussian(
                center=np.array([0.25, 0.25]),
                scale=0.1,
                color=np.array([1.0, 0.0, 0.0]),
                alpha=0.8
            ),
            create_isotropic_gaussian(
                center=np.array([0.75, 0.75]),
                scale=0.1,
                color=np.array([0.0, 1.0, 0.0]),
                alpha=0.8
            )
        ]

        # Create mock analysis
        analysis = Mock(spec=AnisotropicAnalysis)
        analysis.edge_strength = np.ones((64, 64)) * 0.3
        analysis.orientation = np.zeros((64, 64))
        analysis.coherence = np.ones((64, 64)) * 0.5
        analysis.quality_map = np.ones((64, 64)) * 0.7

        quality = refiner._compute_overall_quality(splats, analysis)

        assert isinstance(quality, float)
        assert 0.0 <= quality <= 1.0

        # Test with empty splats
        empty_quality = refiner._compute_overall_quality([], analysis)
        assert empty_quality == 0.0


class TestConvenienceFunctions:
    """Test convenience functions and presets."""

    def test_create_anisotropic_config_presets(self):
        """Test anisotropic configuration preset creation."""
        presets = ["conservative", "balanced", "aggressive", "experimental"]

        for preset in presets:
            config = create_anisotropic_config_preset(preset)
            assert isinstance(config, AnisotropicConfig)

            # Check that each preset has different characteristics
            if preset == "conservative":
                assert config.max_aspect_ratio <= 3.0
                assert config.learning_rate <= 0.05
            elif preset == "aggressive":
                assert config.max_aspect_ratio >= 8.0
                assert config.learning_rate >= 0.15
            elif preset == "experimental":
                assert config.max_aspect_ratio >= 10.0
                assert AnisotropyOperation.COHERENCE_OPTIMIZATION in config.enabled_operations

    def test_create_anisotropic_config_presets_invalid(self):
        """Test invalid preset handling."""
        with pytest.raises(ValueError, match="Unknown preset"):
            create_anisotropic_config_preset("invalid_preset")

    def test_refine_splats_anisotropically_convenience(self):
        """Test convenience function for anisotropic refinement."""
        splats = [
            create_isotropic_gaussian(
                center=np.array([0.5, 0.5]),
                scale=0.1,
                color=np.array([1.0, 0.5, 0.3]),
                alpha=0.8
            )
        ]

        # Create test image
        target_image = np.random.rand(32, 32, 3)

        # Test with default config
        result = refine_splats_anisotropically(splats, target_image)

        assert isinstance(result, AnisotropicRefinementResult)
        assert len(result.refined_splats) == 1

        # Test with custom config
        custom_config = AnisotropicConfig(max_refinement_iterations=1)
        result_custom = refine_splats_anisotropically(splats, target_image, custom_config)

        assert isinstance(result_custom, AnisotropicRefinementResult)
        assert result_custom.iterations == 1


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_splat_list(self):
        """Test refinement with empty splat list."""
        config = AnisotropicConfig()
        refiner = AnisotropicRefiner(config)

        target_image = np.random.rand(32, 32, 3)

        result = refiner.refine_splats_anisotropically([], target_image)

        assert isinstance(result, AnisotropicRefinementResult)
        assert len(result.refined_splats) == 0
        assert result.final_quality >= 0.0

    def test_single_splat_refinement(self):
        """Test refinement with single splat."""
        config = AnisotropicConfig()
        refiner = AnisotropicRefiner(config)

        splat = create_isotropic_gaussian(
            center=np.array([0.5, 0.5]),
            scale=0.1,
            color=np.array([1.0, 0.5, 0.3]),
            alpha=0.8
        )

        target_image = np.random.rand(32, 32, 3)

        result = refiner.refine_splats_anisotropically([splat], target_image)

        assert isinstance(result, AnisotropicRefinementResult)
        assert len(result.refined_splats) == 1

    def test_uniform_image_refinement(self):
        """Test refinement on uniform image (no edges)."""
        config = AnisotropicConfig()
        refiner = AnisotropicRefiner(config)

        splat = create_isotropic_gaussian(
            center=np.array([0.5, 0.5]),
            scale=0.1,
            color=np.array([1.0, 0.5, 0.3]),
            alpha=0.8
        )

        # Uniform image (no edges)
        target_image = np.ones((32, 32, 3)) * 0.5

        result = refiner.refine_splats_anisotropically([splat], target_image)

        assert isinstance(result, AnisotropicRefinementResult)
        # Should not significantly change splats for uniform images
        refined_scales = 1.0 / result.refined_splats[0].inv_s
        aspect_ratio = max(refined_scales) / min(refined_scales)
        assert aspect_ratio < 2.0  # Should remain relatively isotropic

    def test_extreme_anisotropy_constraints(self):
        """Test constraints with extreme anisotropy."""
        config = AnisotropicConfig(max_aspect_ratio=2.0)  # Very restrictive
        refiner = AnisotropicRefiner(config)

        # Create extremely anisotropic splat
        splat = create_isotropic_gaussian(
            center=np.array([0.5, 0.5]),
            scale=0.1,
            color=np.array([1.0, 0.5, 0.3]),
            alpha=0.8
        )
        splat.inv_s = np.array([0.001, 10.0])  # Extreme aspect ratio

        constrained = refiner._apply_anisotropic_constraints(splat)

        scales = 1.0 / constrained.inv_s
        aspect_ratio = max(scales) / min(scales)
        assert aspect_ratio <= config.max_aspect_ratio

    def test_invalid_image_dimensions(self):
        """Test handling of invalid image dimensions."""
        config = AnisotropicConfig()
        refiner = AnisotropicRefiner(config)

        splat = create_isotropic_gaussian(
            center=np.array([0.5, 0.5]),
            scale=0.1,
            color=np.array([1.0, 0.5, 0.3]),
            alpha=0.8
        )

        # Very small image
        small_image = np.random.rand(2, 2, 3)

        # Should handle gracefully
        result = refiner.refine_splats_anisotropically([splat], small_image)
        assert isinstance(result, AnisotropicRefinementResult)

    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        config = AnisotropicConfig()
        refiner = AnisotropicRefiner(config)

        # Create splat at edge of image
        splat = create_isotropic_gaussian(
            center=np.array([0.99, 0.99]),  # Near edge
            scale=0.001,  # Very small
            color=np.array([1.0, 1.0, 1.0]),
            alpha=1.0
        )

        target_image = np.random.rand(10, 10, 3)

        # Should handle without errors
        result = refiner.refine_splats_anisotropically([splat], target_image)
        assert isinstance(result, AnisotropicRefinementResult)

        # Check that refined splat is valid
        refined = result.refined_splats[0]
        assert np.all(np.isfinite(refined.inv_s))
        assert np.all(np.isfinite(refined.mu))
        assert np.isfinite(refined.theta)
        assert np.all(np.isfinite(refined.color))
        assert np.isfinite(refined.alpha)

    def test_convergence_edge_cases(self):
        """Test convergence detection edge cases."""
        config = AnisotropicConfig(
            max_refinement_iterations=10,
            convergence_threshold=1e-10  # Very strict
        )
        refiner = AnisotropicRefiner(config)

        splats = [
            create_isotropic_gaussian(
                center=np.array([0.5, 0.5]),
                scale=0.1,
                color=np.array([1.0, 0.5, 0.3]),
                alpha=0.8
            )
        ]

        target_image = np.random.rand(16, 16, 3)

        result = refiner.refine_splats_anisotropically(splats, target_image)

        # Should either converge or reach max iterations
        assert result.iterations <= config.max_refinement_iterations
        if result.converged:
            assert result.iterations < config.max_refinement_iterations


if __name__ == "__main__":
    pytest.main([__file__])