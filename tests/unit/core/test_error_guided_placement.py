#!/usr/bin/env python3
"""Unit tests for error-guided placement components."""

import pytest
import numpy as np
from src.splat_this.core.error_guided_placement import ErrorGuidedPlacement


class TestErrorGuidedPlacement:
    """Test cases for ErrorGuidedPlacement class."""

    def test_initialization(self):
        """Test ErrorGuidedPlacement initialization."""
        # Default initialization
        placer = ErrorGuidedPlacement()
        assert placer.temperature == 2.0

        # Custom temperature
        placer = ErrorGuidedPlacement(temperature=1.5)
        assert placer.temperature == 1.5

        # Invalid temperature should raise error
        with pytest.raises(ValueError, match="Temperature must be positive"):
            ErrorGuidedPlacement(temperature=0.0)

        with pytest.raises(ValueError, match="Temperature must be positive"):
            ErrorGuidedPlacement(temperature=-1.0)

    def test_compute_reconstruction_error_basic(self):
        """Test basic reconstruction error computation."""
        placer = ErrorGuidedPlacement()

        # Perfect match - should have zero error
        target = np.array([[1.0, 0.5], [0.2, 0.8]], dtype=np.float32)
        rendered = target.copy()
        error = placer.compute_reconstruction_error(target, rendered)

        assert error.shape == (2, 2)
        assert np.allclose(error, 0.0)

        # Known error case
        target = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        rendered = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
        error = placer.compute_reconstruction_error(target, rendered)

        expected_error = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
        assert np.allclose(error, expected_error)

    def test_compute_reconstruction_error_multichannel(self):
        """Test reconstruction error computation for multi-channel images."""
        placer = ErrorGuidedPlacement()

        # RGB images
        target = np.zeros((2, 2, 3), dtype=np.float32)
        target[0, 0] = [1.0, 0.0, 0.0]  # Red pixel
        target[1, 1] = [0.0, 1.0, 0.0]  # Green pixel

        rendered = np.zeros((2, 2, 3), dtype=np.float32)
        rendered[0, 0] = [0.0, 1.0, 0.0]  # Green instead of red
        rendered[1, 1] = [0.0, 1.0, 0.0]  # Correct green

        error = placer.compute_reconstruction_error(target, rendered)

        # Error at (0,0) should be mean of [1.0, 1.0, 0.0] = 2/3
        # Error at (1,1) should be 0 (perfect match)
        assert error.shape == (2, 2)
        assert np.isclose(error[0, 0], 2.0/3.0)
        assert np.isclose(error[1, 1], 0.0)

    def test_compute_reconstruction_error_uint8(self):
        """Test reconstruction error with uint8 images."""
        placer = ErrorGuidedPlacement()

        # uint8 images should be normalized to [0,1] range
        target = np.array([[255, 0], [127, 255]], dtype=np.uint8)
        rendered = np.array([[0, 255], [127, 0]], dtype=np.uint8)

        error = placer.compute_reconstruction_error(target, rendered)

        # Expected normalized errors: [1.0, 1.0], [0.0, 1.0]
        expected_error = np.array([[1.0, 1.0], [0.0, 1.0]], dtype=np.float32)
        assert np.allclose(error, expected_error)

    def test_compute_reconstruction_error_shape_mismatch(self):
        """Test error handling for mismatched image shapes."""
        placer = ErrorGuidedPlacement()

        target = np.zeros((2, 2))
        rendered = np.zeros((3, 3))

        with pytest.raises(ValueError, match="Target and rendered images must have same shape"):
            placer.compute_reconstruction_error(target, rendered)

    def test_compute_l2_error(self):
        """Test L2 error computation."""
        placer = ErrorGuidedPlacement()

        # Test with known values
        target = np.array([[1.0, 0.0]], dtype=np.float32)
        rendered = np.array([[0.0, 1.0]], dtype=np.float32)

        error = placer.compute_l2_error(target, rendered)
        expected_error = np.array([[1.0, 1.0]], dtype=np.float32)
        assert np.allclose(error, expected_error)

        # Multi-channel test
        target = np.array([[[1.0, 0.0, 0.0]]], dtype=np.float32)  # Red
        rendered = np.array([[[0.0, 1.0, 0.0]]], dtype=np.float32)  # Green

        error = placer.compute_l2_error(target, rendered)
        # L2 distance = sqrt(1^2 + 1^2 + 0^2) = sqrt(2)
        expected_error = np.array([[np.sqrt(2.0)]], dtype=np.float32)
        assert np.allclose(error, expected_error)

    def test_create_placement_probability_basic(self):
        """Test basic probability distribution creation."""
        placer = ErrorGuidedPlacement(temperature=1.0)

        # Simple error map
        error_map = np.array([[1.0, 2.0], [0.0, 3.0]], dtype=np.float32)
        prob_map = placer.create_placement_probability(error_map)

        # Should be normalized (sum to 1)
        assert np.isclose(np.sum(prob_map), 1.0)

        # Should preserve relative ordering (higher error = higher probability)
        assert prob_map[1, 1] > prob_map[0, 1]  # 3.0 > 2.0
        assert prob_map[0, 1] > prob_map[0, 0]  # 2.0 > 1.0
        assert prob_map[0, 0] > prob_map[1, 0]  # 1.0 > 0.0 (with epsilon)

    def test_create_placement_probability_temperature_effects(self):
        """Test temperature effects on probability distribution."""
        error_map = np.array([[1.0, 4.0]], dtype=np.float32)

        # Low temperature (sharp distribution)
        placer_sharp = ErrorGuidedPlacement(temperature=0.5)
        prob_sharp = placer_sharp.create_placement_probability(error_map)

        # High temperature (more uniform distribution)
        placer_smooth = ErrorGuidedPlacement(temperature=4.0)
        prob_smooth = placer_smooth.create_placement_probability(error_map)

        # Sharp distribution should have bigger difference between high/low error
        ratio_sharp = prob_sharp[0, 1] / prob_sharp[0, 0]
        ratio_smooth = prob_smooth[0, 1] / prob_smooth[0, 0]
        assert ratio_sharp > ratio_smooth

    def test_create_placement_probability_edge_cases(self):
        """Test edge cases for probability creation."""
        placer = ErrorGuidedPlacement()

        # Empty error map
        with pytest.raises(ValueError, match="Error map cannot be empty"):
            placer.create_placement_probability(np.array([]))

        # Negative values
        error_map = np.array([[-1.0, 1.0]])
        with pytest.raises(ValueError, match="Error map cannot contain negative values"):
            placer.create_placement_probability(error_map)

        # All zeros (should use uniform distribution)
        error_map = np.zeros((2, 2))
        prob_map = placer.create_placement_probability(error_map)
        assert np.allclose(prob_map, 0.25)  # Uniform probability
        assert np.isclose(np.sum(prob_map), 1.0)

    def test_sample_positions_basic(self):
        """Test basic position sampling."""
        placer = ErrorGuidedPlacement()

        # Create simple probability map
        prob_map = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)

        # Sample positions
        positions = placer.sample_positions(prob_map, count=2)

        assert len(positions) == 2
        assert all(isinstance(pos, tuple) and len(pos) == 2 for pos in positions)
        assert all(0 <= y < 2 and 0 <= x < 2 for y, x in positions)

        # All positions should be unique (no replacement)
        assert len(set(positions)) == len(positions)

    def test_sample_positions_edge_cases(self):
        """Test edge cases for position sampling."""
        placer = ErrorGuidedPlacement()

        prob_map = np.array([[0.5, 0.5]], dtype=np.float32)

        # Zero count
        positions = placer.sample_positions(prob_map, count=0)
        assert len(positions) == 0

        # Negative count
        positions = placer.sample_positions(prob_map, count=-5)
        assert len(positions) == 0

        # Count larger than available pixels
        positions = placer.sample_positions(prob_map, count=10)
        assert len(positions) == 2  # Should be limited to available pixels

        # Empty probability map
        with pytest.raises(ValueError, match="Probability map cannot be empty"):
            placer.sample_positions(np.array([]), count=1)

    def test_sample_positions_min_distance(self):
        """Test minimum distance constraint."""
        placer = ErrorGuidedPlacement()

        # Create a larger probability map
        prob_map = np.ones((10, 10), dtype=np.float32) / 100  # Uniform

        # Sample with minimum distance constraint
        positions = placer.sample_positions(prob_map, count=20, min_distance=3.0)

        # Check that all positions respect minimum distance
        for i, (y1, x1) in enumerate(positions):
            for j, (y2, x2) in enumerate(positions):
                if i != j:
                    distance = np.sqrt((y1 - y2)**2 + (x1 - x2)**2)
                    assert distance >= 3.0

    def test_normalize_image(self):
        """Test image normalization."""
        placer = ErrorGuidedPlacement()

        # uint8 image
        img_uint8 = np.array([[0, 127, 255]], dtype=np.uint8)
        normalized = placer._normalize_image(img_uint8)
        expected = np.array([[0.0, 127.0/255.0, 1.0]], dtype=np.float32)
        assert np.allclose(normalized, expected)

        # uint16 image
        img_uint16 = np.array([[0, 32767, 65535]], dtype=np.uint16)
        normalized = placer._normalize_image(img_uint16)
        expected = np.array([[0.0, 32767.0/65535.0, 1.0]], dtype=np.float32)
        assert np.allclose(normalized, expected)

        # Already float image
        img_float = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)
        normalized = placer._normalize_image(img_float)
        assert np.allclose(normalized, img_float)

    def test_enforce_min_distance(self):
        """Test minimum distance enforcement."""
        placer = ErrorGuidedPlacement()

        # Positions that are too close
        positions = [(0, 0), (0, 1), (0, 5), (5, 5)]
        filtered = placer._enforce_min_distance(positions, min_distance=2.0)

        # Should keep first position and filter out close ones
        assert (0, 0) in filtered  # Always keep first
        assert (0, 1) not in filtered  # Too close to (0, 0)
        assert (0, 5) in filtered  # Far enough from (0, 0)
        assert (5, 5) in filtered  # Far enough from others

        # Empty list
        assert placer._enforce_min_distance([], 1.0) == []

        # Single position
        assert placer._enforce_min_distance([(0, 0)], 1.0) == [(0, 0)]

    def test_get_error_statistics(self):
        """Test error statistics computation."""
        placer = ErrorGuidedPlacement()

        error_map = np.array([[0.0, 0.1], [0.5, 1.0]], dtype=np.float32)
        stats = placer.get_error_statistics(error_map)

        assert np.isclose(stats['mean_error'], 0.4)
        assert np.isclose(stats['max_error'], 1.0)
        assert np.isclose(stats['min_error'], 0.0)
        assert np.isclose(stats['total_error'], 1.6)
        assert stats['error_pixels'] == 3  # Non-zero pixels
        assert stats['zero_error_pixels'] == 1
        assert stats['shape'] == (2, 2)
        assert 'percentiles' in stats

        # Empty error map
        empty_stats = placer.get_error_statistics(np.array([]))
        assert empty_stats['empty'] == True

    def test_visualize_probability_map(self):
        """Test probability map visualization."""
        placer = ErrorGuidedPlacement()

        prob_map = np.array([[0.1, 0.4], [0.2, 0.3]], dtype=np.float32)
        vis = placer.visualize_probability_map(prob_map)

        # Should be RGB image
        assert vis.shape == (2, 2, 3)
        assert vis.dtype == np.uint8

        # Red channel should reflect probability values
        assert vis[0, 1, 0] > vis[0, 0, 0]  # Higher probability = brighter red

        # With sample positions
        positions = [(0, 0), (1, 1)]
        vis_with_samples = placer.visualize_probability_map(prob_map, positions)

        # Should have green markers at sample positions
        assert vis_with_samples[0, 0, 1] > 0  # Green at (0, 0)
        assert vis_with_samples[1, 1, 1] > 0  # Green at (1, 1)

    def test_full_workflow_integration(self):
        """Test complete error-guided placement workflow."""
        placer = ErrorGuidedPlacement(temperature=2.0)

        # Create synthetic target and rendered images
        target = np.random.rand(10, 10).astype(np.float32)
        rendered = target + np.random.normal(0, 0.1, target.shape).astype(np.float32)
        rendered = np.clip(rendered, 0, 1)  # Keep in valid range

        # Compute error
        error_map = placer.compute_reconstruction_error(target, rendered)
        assert error_map.shape == (10, 10)
        assert np.all(error_map >= 0)

        # Create probability distribution
        prob_map = placer.create_placement_probability(error_map)
        assert np.isclose(np.sum(prob_map), 1.0)

        # Sample positions
        positions = placer.sample_positions(prob_map, count=5)
        assert len(positions) <= 5
        assert all(0 <= y < 10 and 0 <= x < 10 for y, x in positions)

        # Get statistics
        stats = placer.get_error_statistics(error_map)
        assert 'mean_error' in stats
        assert stats['mean_error'] >= 0

        # Create visualization
        vis = placer.visualize_probability_map(prob_map, positions)
        assert vis.shape == (10, 10, 3)

    def test_sampling_distribution_quality(self):
        """Test that sampling follows the probability distribution."""
        placer = ErrorGuidedPlacement(temperature=1.0)

        # Create probability map with clear bias
        prob_map = np.zeros((4, 4), dtype=np.float32)
        prob_map[0, 0] = 0.8  # High probability corner
        prob_map[3, 3] = 0.2  # Low probability corner
        prob_map = prob_map / np.sum(prob_map)

        # Sample many times and check distribution
        position_counts = {}
        num_trials = 1000

        for _ in range(num_trials):
            positions = placer.sample_positions(prob_map, count=1)
            if positions:
                pos = positions[0]
                position_counts[pos] = position_counts.get(pos, 0) + 1

        # High probability position should be sampled more often
        high_prob_count = position_counts.get((0, 0), 0)
        low_prob_count = position_counts.get((3, 3), 0)

        # Should sample high-probability position more often (allowing some variance)
        assert high_prob_count > low_prob_count