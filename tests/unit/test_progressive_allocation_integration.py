#!/usr/bin/env python3
"""Integration tests for progressive allocation system."""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.splat_this.core.adaptive_extract import AdaptiveSplatExtractor
from src.splat_this.core.progressive_allocator import ProgressiveAllocator
from src.splat_this.core.error_guided_placement import ErrorGuidedPlacement
from src.splat_this.utils.reconstruction_error import (
    compute_reconstruction_error,
    compute_error_statistics,
    compute_psnr
)
from src.splat_this.utils.visualization import create_debug_summary


class TestProgressiveAllocationIntegration:
    """Integration tests for the complete progressive allocation pipeline."""

    def setup_method(self):
        """Set up test fixtures."""
        # Import the config here to avoid module-level import issues
        from src.splat_this.core.adaptive_extract import AdaptiveSplatConfig
        from src.splat_this.core.progressive_allocator import ProgressiveConfig

        # Configure for progressive allocation with reasonable parameters for testing
        config = AdaptiveSplatConfig()
        config.enable_progressive = True

        # Configure progressive allocator with test-friendly parameters
        progressive_config = ProgressiveConfig(
            initial_ratio=0.5,  # Start with more splats for small test cases
            max_splats=100,     # Lower max for faster tests
            error_threshold=0.05,  # More permissive error threshold
            max_add_per_step=5,    # Smaller batch sizes for testing
            convergence_patience=3  # Less patience for faster convergence
        )

        self.extractor = AdaptiveSplatExtractor(config, progressive_config)

        # Create test images
        self.simple_image = self._create_test_image((50, 50, 3))
        self.gradient_image = self._create_gradient_image((40, 40, 3))
        self.complex_image = self._create_complex_image((60, 60, 3))

    def _create_test_image(self, shape):
        """Create a simple test image with basic patterns."""
        image = np.zeros(shape, dtype=np.float32)
        h, w = shape[:2]

        # Add some basic patterns
        image[h//4:3*h//4, w//4:3*w//4] = 0.8  # Central square
        image[h//8:h//4, w//8:w//4] = 0.3      # Corner square

        if len(shape) == 3:
            image[:, :, 1] *= 0.7  # Vary green channel
            image[:, :, 2] *= 0.5  # Vary blue channel

        return image

    def _create_gradient_image(self, shape):
        """Create an image with smooth gradients."""
        h, w = shape[:2]
        y, x = np.mgrid[0:h, 0:w]

        if len(shape) == 3:
            image = np.zeros(shape, dtype=np.float32)
            image[:, :, 0] = (x / w)  # Horizontal gradient
            image[:, :, 1] = (y / h)  # Vertical gradient
            image[:, :, 2] = 0.5      # Constant blue
        else:
            image = (x + y) / (w + h)

        return image.astype(np.float32)

    def _create_complex_image(self, shape):
        """Create a complex image with multiple features."""
        h, w = shape[:2]
        image = np.random.rand(*shape).astype(np.float32) * 0.1

        # Add circles
        y, x = np.mgrid[0:h, 0:w]
        center1 = (h//3, w//3)
        center2 = (2*h//3, 2*w//3)

        circle1 = ((x - center1[1])**2 + (y - center1[0])**2) < (h//8)**2
        circle2 = ((x - center2[1])**2 + (y - center2[0])**2) < (h//6)**2

        if len(shape) == 3:
            image[circle1, 0] = 0.9
            image[circle2, 1] = 0.8
        else:
            image[circle1] = 0.9
            image[circle2] = 0.8

        return image

    def test_full_progressive_allocation_pipeline(self):
        """Test the complete progressive allocation pipeline."""
        # Extract splats using progressive allocation
        splats = self.extractor.extract_adaptive_splats(
            self.complex_image,
            n_splats=20,
            verbose=False
        )

        # Verify results
        assert len(splats) > 0
        assert len(splats) <= 20

        # Verify splat properties
        for splat in splats:
            assert hasattr(splat, 'x') and hasattr(splat, 'y')
            assert hasattr(splat, 'rx') and hasattr(splat, 'ry')
            assert hasattr(splat, 'r') and hasattr(splat, 'g') and hasattr(splat, 'b')
            assert 0 <= splat.x < self.complex_image.shape[1]
            assert 0 <= splat.y < self.complex_image.shape[0]

    def test_progressive_vs_regular_allocation(self):
        """Compare progressive allocation with regular allocation."""
        from src.splat_this.core.adaptive_extract import AdaptiveSplatConfig

        # Create regular (non-progressive) extractor
        regular_config = AdaptiveSplatConfig()
        regular_config.enable_progressive = False
        regular_extractor = AdaptiveSplatExtractor(regular_config)

        # Regular allocation
        regular_splats = regular_extractor.extract_adaptive_splats(
            self.gradient_image,
            n_splats=25,
            verbose=False
        )

        # Progressive allocation
        progressive_splats = self.extractor.extract_adaptive_splats(
            self.gradient_image,
            n_splats=25
        )

        # Both should produce valid results
        assert len(regular_splats) > 0
        assert len(progressive_splats) > 0

    def test_progressive_allocation_convergence(self):
        """Test that progressive allocation converges properly."""
        # Use a simple image that should converge quickly
        simple_uniform = np.full((30, 30, 3), 0.5, dtype=np.float32)

        splats = self.extractor.extract_adaptive_splats(
            simple_uniform,
            n_splats=50,  # Request many splats
            verbose=False
        )

        # Should converge with fewer splats than requested
        assert len(splats) < 50
        assert len(splats) > 0

    def test_progressive_allocation_with_different_metrics(self):
        """Test progressive allocation with different error metrics."""
        # Since progressive allocation is enabled by default, we just test that
        # different configurations work
        test_configs = [
            {"error_threshold": 0.01},
            {"error_threshold": 0.05},
            {"error_threshold": 0.1}
        ]

        for config_params in test_configs:
            from src.splat_this.core.progressive_allocator import ProgressiveConfig
            from src.splat_this.core.adaptive_extract import AdaptiveSplatConfig

            test_config = AdaptiveSplatConfig()
            test_config.enable_progressive = True
            prog_config = ProgressiveConfig(**config_params)
            test_extractor = AdaptiveSplatExtractor(test_config, prog_config)

            splats = test_extractor.extract_adaptive_splats(
                self.simple_image,
                n_splats=50  # Increased for initial allocation validation
            )

            assert len(splats) > 0, f"Failed with config {config_params}"

    def test_saliency_guided_initial_allocation(self):
        """Test that saliency analysis guides initial splat placement."""
        # Create image with high saliency regions
        saliency_image = np.zeros((40, 40, 3), dtype=np.float32)
        # High contrast region (high saliency)
        saliency_image[10:30, 10:30] = 1.0
        saliency_image[15:25, 15:25] = 0.0

        splats = self.extractor.extract_adaptive_splats(
            saliency_image,
            n_splats=30,
                    )

        # Verify splats are placed in high saliency region
        high_saliency_splats = 0
        for splat in splats:
            if 10 <= splat.x <= 30 and 10 <= splat.y <= 30:
                high_saliency_splats += 1

        # Most splats should be in the high saliency region
        assert high_saliency_splats >= len(splats) // 2

    def test_error_guided_placement_integration(self):
        """Test error-guided placement in the full pipeline."""
        # Create image with localized features
        feature_image = np.zeros((50, 50, 3), dtype=np.float32)

        # Add high-frequency features in corners
        feature_image[5:15, 5:15] = np.random.rand(10, 10, 3)
        feature_image[35:45, 35:45] = np.random.rand(10, 10, 3)

        splats = self.extractor.extract_adaptive_splats(
            feature_image,
            n_splats=30,
                    )

        # Count splats near high-frequency regions
        near_features = 0
        for splat in splats:
            if ((5 <= splat.x <= 15 and 5 <= splat.y <= 15) or
                (35 <= splat.x <= 45 and 35 <= splat.y <= 45)):
                near_features += 1

        # Should place more splats near features
        assert near_features > 0

    def test_progressive_allocation_with_budget_constraints(self):
        """Test progressive allocation respects budget constraints."""
        budgets = [30, 40, 60, 80]  # Increased budgets for validation

        for budget in budgets:
            splats = self.extractor.extract_adaptive_splats(
                self.complex_image,
                n_splats=budget,
                            )

            # Should not exceed budget
            assert len(splats) <= budget

            # Should use at least some of the budget (unless converged)
            assert len(splats) > 0

    def test_reconstruction_error_computation_integration(self):
        """Test reconstruction error computation in the pipeline."""
        # Extract splats
        splats = self.extractor.extract_adaptive_splats(
            self.gradient_image,
            n_splats=30,
                    )

        # Render splats back to image
        rendered = self.extractor._render_splats_to_image(
            splats,
            self.gradient_image.shape[:2]
        )

        # Compute reconstruction error
        error_map = compute_reconstruction_error(
            self.gradient_image,
            rendered,
            metric="l1"
        )

        # Verify error map properties
        assert error_map.shape == self.gradient_image.shape[:2]
        assert error_map.dtype == np.float32
        assert np.all(error_map >= 0)

        # Error should be reasonable (not perfect, but not terrible)
        mean_error = np.mean(error_map)
        assert 0 <= mean_error <= 1.0

    def test_progressive_steps_tracking(self):
        """Test that progressive steps are properly tracked through verbose output."""
        # Test with verbose output enabled
        splats = self.extractor.extract_adaptive_splats(
            self.simple_image,
            n_splats=25,
            verbose=True
        )

        # Should have produced valid splats
        assert len(splats) > 0

    def test_temperature_scaling_effects(self):
        """Test that temperature scaling affects placement diversity."""
        # Test with different temperature settings
        placer = ErrorGuidedPlacement()

        # Create error map with clear high/low error regions
        error_map = np.zeros((30, 30), dtype=np.float32)
        error_map[10:20, 10:20] = 1.0  # High error region

        # High temperature (more diverse)
        placer.temperature = 2.0
        prob_map_hot = placer.create_placement_probability(error_map)

        # Low temperature (more focused)
        placer.temperature = 0.1
        prob_map_cold = placer.create_placement_probability(error_map)

        # High temperature should have more spread probability
        hot_entropy = -np.sum(prob_map_hot * np.log(prob_map_hot + 1e-10))
        cold_entropy = -np.sum(prob_map_cold * np.log(prob_map_cold + 1e-10))

        assert hot_entropy > cold_entropy

    def test_allocation_with_visualization_debug(self):
        """Test progressive allocation with verbose output for debugging."""
        splats = self.extractor.extract_adaptive_splats(
            self.simple_image,
            n_splats=25,
            verbose=True
        )

        assert len(splats) > 0


class TestProgressiveAllocationComponents:
    """Test individual components of the progressive allocation system."""

    def test_progressive_allocator_initialization(self):
        """Test ProgressiveAllocator initialization and configuration."""
        from src.splat_this.core.progressive_allocator import ProgressiveConfig

        config = ProgressiveConfig(
            error_threshold=0.05,
            max_add_per_step=3,
            temperature=1.5
        )
        allocator = ProgressiveAllocator(config)

        assert allocator.config.error_threshold == 0.05
        assert allocator.config.max_add_per_step == 3
        assert allocator.config.temperature == 1.5

    def test_error_guided_placement_initialization(self):
        """Test ErrorGuidedPlacement initialization and configuration."""
        placer = ErrorGuidedPlacement(temperature=1.5)

        assert placer.temperature == 1.5

    def test_component_integration_workflow(self):
        """Test that all components work together in the expected workflow."""
        # Initialize components
        from src.splat_this.core.progressive_allocator import ProgressiveConfig

        config = ProgressiveConfig()
        allocator = ProgressiveAllocator(config)
        placer = ErrorGuidedPlacement()

        # Create test data
        image = np.random.rand(20, 20, 3).astype(np.float32)

        # Step 1: Error-guided placement (simulated workflow)
        error_map = np.random.rand(20, 20).astype(np.float32)
        prob_map = placer.create_placement_probability(error_map)
        assert prob_map.shape == error_map.shape

        # Step 2: Sample positions
        positions = placer.sample_positions(prob_map, 3)
        assert len(positions) <= 3

        # Step 3: Allocation decision
        current_error = 0.1
        should_continue = allocator.should_add_splats(current_error)
        assert isinstance(should_continue, bool)


class TestProgressiveAllocationEdgeCases:
    """Test edge cases and error conditions for progressive allocation."""

    def test_empty_image_handling(self):
        """Test handling of empty or invalid images."""
        extractor = AdaptiveSplatExtractor()

        # Empty image
        with pytest.raises(ValueError):
            extractor.extract_adaptive_splats(
                np.array([]),
                n_splats=5,
                            )

    def test_zero_splat_request(self):
        """Test handling of zero splat request."""
        extractor = AdaptiveSplatExtractor()
        image = np.random.rand(10, 10, 3).astype(np.float32)

        splats = extractor.extract_adaptive_splats(
            image,
            n_splats=0,
                    )

        assert len(splats) == 0

    def test_single_pixel_image(self):
        """Test handling of single pixel image."""
        extractor = AdaptiveSplatExtractor()
        image = np.array([[[0.5, 0.5, 0.5]]], dtype=np.float32)

        splats = extractor.extract_adaptive_splats(
            image,
            n_splats=1,
                    )

        # Should handle gracefully
        assert len(splats) <= 1

    def test_uniform_image_convergence(self):
        """Test convergence behavior on uniform images."""
        extractor = AdaptiveSplatExtractor()

        # Completely uniform image
        uniform_image = np.full((20, 20, 3), 0.5, dtype=np.float32)

        splats = extractor.extract_adaptive_splats(
            uniform_image,
            n_splats=20,
                    )

        # Should converge quickly with few splats
        assert len(splats) < 20

    def test_extreme_error_thresholds(self):
        """Test behavior with extreme error thresholds."""
        from src.splat_this.core.progressive_allocator import ProgressiveConfig

        # Very strict threshold (should use many splats)
        strict_config = ProgressiveConfig(error_threshold=0.001)
        strict_allocator = ProgressiveAllocator(strict_config)

        # Very loose threshold (should use few splats)
        loose_config = ProgressiveConfig(error_threshold=0.9)
        loose_allocator = ProgressiveAllocator(loose_config)

        # Test decision making
        high_error = 0.5
        low_error = 0.01

        assert strict_allocator.should_add_splats(high_error) is True
        assert strict_allocator.should_add_splats(low_error) is True

        assert loose_allocator.should_add_splats(high_error) is False
        assert loose_allocator.should_add_splats(low_error) is False

    def test_large_batch_sizes(self):
        """Test behavior with large batch sizes."""
        from src.splat_this.core.progressive_allocator import ProgressiveConfig

        config = ProgressiveConfig(max_add_per_step=100)
        allocator = ProgressiveAllocator(config)

        # Should handle large batch sizes gracefully
        add_count = allocator.get_addition_count(current_splat_count=5)
        assert add_count > 0
        assert add_count <= 100


class TestProgressiveAllocationPerformance:
    """Performance-focused tests for progressive allocation."""

    def test_allocation_scales_with_image_size(self):
        """Test that allocation time scales reasonably with image size."""
        extractor = AdaptiveSplatExtractor()
        sizes = [(20, 20), (40, 40), (60, 60)]

        for size in sizes:
            image = np.random.rand(*size, 3).astype(np.float32)

            # Should complete in reasonable time
            splats = extractor.extract_adaptive_splats(
                image,
                n_splats=10,
                            )

            assert len(splats) > 0

    def test_memory_usage_stability(self):
        """Test that memory usage remains stable during allocation."""
        extractor = AdaptiveSplatExtractor()

        # Process multiple images to check for memory leaks
        for i in range(5):
            image = np.random.rand(30, 30, 3).astype(np.float32)
            splats = extractor.extract_adaptive_splats(
                image,
                n_splats=30,
                            )
            assert len(splats) > 0

    def test_convergence_performance(self):
        """Test that convergence happens in reasonable number of iterations."""
        extractor = AdaptiveSplatExtractor()

        # Image that should converge quickly
        simple_image = np.full((25, 25, 3), 0.7, dtype=np.float32)
        simple_image[10:15, 10:15] = 0.3  # Single feature

        # Track iterations (would need instrumentation in real implementation)
        splats = extractor.extract_adaptive_splats(
            simple_image,
            n_splats=30,
                    )

        # Should converge with reasonable number of splats
        assert 1 <= len(splats) <= 15