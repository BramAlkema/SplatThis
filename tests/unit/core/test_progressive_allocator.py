#!/usr/bin/env python3
"""Unit tests for progressive allocation components."""

import pytest
import numpy as np
from src.splat_this.core.progressive_allocator import ProgressiveConfig, ProgressiveAllocator


class TestProgressiveConfig:
    """Test cases for ProgressiveConfig dataclass."""

    def test_default_config_is_valid(self):
        """Test that default configuration passes validation."""
        config = ProgressiveConfig()
        # Should not raise any exceptions
        assert config.initial_ratio == 0.3
        assert config.max_splats == 2000
        assert config.add_interval == 50

    def test_custom_config_validation(self):
        """Test custom configuration validation."""
        # Valid custom config
        config = ProgressiveConfig(
            initial_ratio=0.5,
            max_splats=1000,
            add_interval=25,
            max_add_per_step=10,
            error_threshold=0.005,
            convergence_patience=3,
            temperature=1.5
        )
        assert config.initial_ratio == 0.5
        assert config.max_splats == 1000

    def test_initial_ratio_validation(self):
        """Test initial_ratio parameter validation."""
        # Too low
        with pytest.raises(ValueError, match="initial_ratio must be between 0.1 and 0.8"):
            ProgressiveConfig(initial_ratio=0.05)

        # Too high
        with pytest.raises(ValueError, match="initial_ratio must be between 0.1 and 0.8"):
            ProgressiveConfig(initial_ratio=0.9)

        # Valid boundary values
        ProgressiveConfig(initial_ratio=0.1)  # Should not raise
        ProgressiveConfig(initial_ratio=0.8)  # Should not raise

    def test_max_splats_validation(self):
        """Test max_splats parameter validation."""
        # Zero splats
        with pytest.raises(ValueError, match="max_splats must be positive"):
            ProgressiveConfig(max_splats=0)

        # Negative splats
        with pytest.raises(ValueError, match="max_splats must be positive"):
            ProgressiveConfig(max_splats=-100)

        # Valid positive value
        ProgressiveConfig(max_splats=1)  # Should not raise

    def test_add_interval_validation(self):
        """Test add_interval parameter validation."""
        # Zero interval
        with pytest.raises(ValueError, match="add_interval must be positive"):
            ProgressiveConfig(add_interval=0)

        # Negative interval
        with pytest.raises(ValueError, match="add_interval must be positive"):
            ProgressiveConfig(add_interval=-10)

        # Valid positive value
        ProgressiveConfig(add_interval=1)  # Should not raise

    def test_max_add_per_step_validation(self):
        """Test max_add_per_step parameter validation."""
        with pytest.raises(ValueError, match="max_add_per_step must be positive"):
            ProgressiveConfig(max_add_per_step=0)

        with pytest.raises(ValueError, match="max_add_per_step must be positive"):
            ProgressiveConfig(max_add_per_step=-5)

    def test_error_threshold_validation(self):
        """Test error_threshold parameter validation."""
        # Negative threshold
        with pytest.raises(ValueError, match="error_threshold must be non-negative"):
            ProgressiveConfig(error_threshold=-0.01)

        # Zero threshold (should be valid)
        ProgressiveConfig(error_threshold=0.0)  # Should not raise

        # Positive threshold
        ProgressiveConfig(error_threshold=0.1)  # Should not raise

    def test_convergence_patience_validation(self):
        """Test convergence_patience parameter validation."""
        with pytest.raises(ValueError, match="convergence_patience must be positive"):
            ProgressiveConfig(convergence_patience=0)

        with pytest.raises(ValueError, match="convergence_patience must be positive"):
            ProgressiveConfig(convergence_patience=-2)

    def test_temperature_validation(self):
        """Test temperature parameter validation."""
        with pytest.raises(ValueError, match="temperature must be positive"):
            ProgressiveConfig(temperature=0.0)

        with pytest.raises(ValueError, match="temperature must be positive"):
            ProgressiveConfig(temperature=-1.0)

        # Valid positive temperature
        ProgressiveConfig(temperature=0.1)  # Should not raise

    def test_get_initial_count(self):
        """Test initial splat count calculation."""
        config = ProgressiveConfig(max_splats=1000, initial_ratio=0.3)
        assert config.get_initial_count() == 300

        config = ProgressiveConfig(max_splats=2000, initial_ratio=0.25)
        assert config.get_initial_count() == 500

        config = ProgressiveConfig(max_splats=100, initial_ratio=0.1)
        assert config.get_initial_count() == 10

    def test_validate_compatibility(self):
        """Test image size compatibility validation."""
        config = ProgressiveConfig(max_splats=1000)

        # Normal image size should work
        config.validate_compatibility((512, 512))  # Should not raise

        # Very small image with too many splats should warn but not error
        config = ProgressiveConfig(max_splats=100000)
        config.validate_compatibility((100, 100))  # Should warn but not raise

        # Config that produces too few initial splats should error
        config = ProgressiveConfig(max_splats=10, initial_ratio=0.1)
        with pytest.raises(ValueError, match="Initial allocation .* is too small"):
            config.validate_compatibility((512, 512))


class TestProgressiveAllocator:
    """Test cases for ProgressiveAllocator class."""

    def test_allocator_initialization(self):
        """Test allocator initialization."""
        config = ProgressiveConfig()
        allocator = ProgressiveAllocator(config)

        assert allocator.config == config
        assert allocator.iteration_count == 0
        assert len(allocator.error_history) == 0
        assert allocator.last_addition_iteration == -1

    def test_should_add_splats_initial_state(self):
        """Test should_add_splats in initial state."""
        config = ProgressiveConfig(add_interval=10, error_threshold=0.01)
        allocator = ProgressiveAllocator(config)

        # Should not add immediately (no iterations yet)
        assert not allocator.should_add_splats(0.1)

        # After recording some iterations with varying error (to avoid convergence)
        for i in range(15):
            error = 0.1 + 0.01 * (i % 3)  # Vary error to prevent convergence
            allocator.record_iteration(error)

        # Now should add (enough iterations passed, error above threshold)
        assert allocator.should_add_splats(0.1)

    def test_should_add_splats_error_threshold(self):
        """Test should_add_splats respects error threshold."""
        config = ProgressiveConfig(add_interval=5, error_threshold=0.01)
        allocator = ProgressiveAllocator(config)

        # Add some iterations with varying error
        for i in range(10):
            error = 0.02 + 0.005 * (i % 2)  # Vary error to prevent convergence
            allocator.record_iteration(error)

        # High error should trigger addition
        assert allocator.should_add_splats(0.02)

        # Low error should not trigger addition
        assert not allocator.should_add_splats(0.005)

    def test_should_add_splats_convergence_detection(self):
        """Test convergence detection prevents further additions."""
        config = ProgressiveConfig(
            add_interval=5,
            error_threshold=0.01,
            convergence_patience=3
        )
        allocator = ProgressiveAllocator(config)

        # Add iterations with stable error (converged)
        stable_error = 0.02
        for i in range(10):
            allocator.record_iteration(stable_error)  # Same error indicates convergence

        # Should detect convergence and not add splats
        assert not allocator.should_add_splats(stable_error)

    def test_get_addition_count(self):
        """Test addition count calculation."""
        config = ProgressiveConfig(max_splats=100, max_add_per_step=20)
        allocator = ProgressiveAllocator(config)

        # Normal case - should return max_add_per_step
        assert allocator.get_addition_count(50) == 20

        # Near budget limit - should return remaining budget
        assert allocator.get_addition_count(90) == 10

        # At budget limit - should return 0
        assert allocator.get_addition_count(100) == 0

        # Over budget limit - should return 0
        assert allocator.get_addition_count(150) == 0

    def test_record_iteration(self):
        """Test iteration recording."""
        config = ProgressiveConfig()
        allocator = ProgressiveAllocator(config)

        # Record first iteration
        allocator.record_iteration(0.1)
        assert allocator.iteration_count == 1
        assert len(allocator.error_history) == 1
        assert allocator.error_history[0] == 0.1
        assert allocator.last_addition_iteration == -1

        # Record iteration with splat addition
        allocator.record_iteration(0.08, added_splats=5)
        assert allocator.iteration_count == 2
        assert len(allocator.error_history) == 2
        assert allocator.error_history[1] == 0.08
        assert allocator.last_addition_iteration == 2

    def test_convergence_detection(self):
        """Test internal convergence detection logic."""
        config = ProgressiveConfig(
            convergence_patience=3,
            error_threshold=0.01
        )
        allocator = ProgressiveAllocator(config)

        # Not enough history - should not converge
        allocator.record_iteration(0.1)
        allocator.record_iteration(0.09)
        assert not allocator._check_convergence()

        # Add very stable error history - should converge
        # Error range needs to be < 0.0001 (0.01 * 0.01) to trigger convergence
        allocator.record_iteration(0.020000)
        allocator.record_iteration(0.020001)
        allocator.record_iteration(0.020000)
        assert allocator._check_convergence()

        # Add varying error - should not converge
        allocator.error_history.clear()
        allocator.record_iteration(0.1)
        allocator.record_iteration(0.05)
        allocator.record_iteration(0.2)
        assert not allocator._check_convergence()

    def test_get_stats(self):
        """Test statistics retrieval."""
        config = ProgressiveConfig()
        allocator = ProgressiveAllocator(config)

        # Initial stats
        stats = allocator.get_stats()
        assert stats['iteration_count'] == 0
        assert stats['error_history_length'] == 0
        assert stats['current_error'] is None
        assert stats['last_addition_iteration'] == -1
        assert not stats['converged']
        assert stats['config'] == config

        # After some iterations
        allocator.record_iteration(0.1)
        allocator.record_iteration(0.08, added_splats=5)

        stats = allocator.get_stats()
        assert stats['iteration_count'] == 2
        assert stats['error_history_length'] == 2
        assert stats['current_error'] == 0.08
        assert stats['last_addition_iteration'] == 2

    def test_reset(self):
        """Test allocator state reset."""
        config = ProgressiveConfig()
        allocator = ProgressiveAllocator(config)

        # Add some state
        allocator.record_iteration(0.1)
        allocator.record_iteration(0.08, added_splats=5)
        allocator._converged = True

        # Reset
        allocator.reset()

        # Check all state is cleared
        assert allocator.iteration_count == 0
        assert len(allocator.error_history) == 0
        assert allocator.last_addition_iteration == -1
        assert not allocator._converged

    def test_full_allocation_workflow(self):
        """Test complete allocation workflow simulation."""
        config = ProgressiveConfig(
            max_splats=100,
            add_interval=10,
            max_add_per_step=15,
            error_threshold=0.01,
            convergence_patience=3
        )
        allocator = ProgressiveAllocator(config)
        current_splat_count = config.get_initial_count()  # Start with initial allocation

        iteration = 0
        while current_splat_count < config.max_splats:
            iteration += 1
            # Simulate decreasing error over time
            error = 0.1 * np.exp(-iteration * 0.05)

            # Check if we should add splats
            if allocator.should_add_splats(error):
                add_count = allocator.get_addition_count(current_splat_count)
                current_splat_count += add_count
                allocator.record_iteration(error, add_count)
            else:
                allocator.record_iteration(error)

            # Safety break to prevent infinite loop
            if iteration > 1000:
                break

        # Should have allocated some splats
        assert current_splat_count > config.get_initial_count()

        # Should not exceed budget
        assert current_splat_count <= config.max_splats

        # Should have recorded iterations
        assert allocator.iteration_count > 0
        assert len(allocator.error_history) > 0