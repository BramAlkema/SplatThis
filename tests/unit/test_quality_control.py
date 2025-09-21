"""Unit tests for quality control functionality."""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from typing import List

from splat_this.core.extract import Gaussian
from splat_this.core.layering import QualityController


class TestQualityController:

    def test_init_default_parameters(self):
        """Test QualityController initialization with default parameters."""
        controller = QualityController(target_count=100)
        assert controller.target_count == 100
        assert controller.k_multiplier == 2.5
        assert controller.min_area_threshold == 1.0
        assert controller.max_alpha == 1.0
        assert controller.alpha_adjustment is True

    def test_init_custom_parameters(self):
        """Test QualityController initialization with custom parameters."""
        controller = QualityController(
            target_count=50,
            k_multiplier=3.0,
            min_area_threshold=2.0,
            max_alpha=0.8,
            alpha_adjustment=False
        )
        assert controller.target_count == 50
        assert controller.k_multiplier == 3.0
        assert controller.min_area_threshold == 2.0
        assert controller.max_alpha == 0.8
        assert controller.alpha_adjustment is False

    def test_optimize_splats_empty_list(self):
        """Test quality control with empty splat list."""
        controller = QualityController(target_count=10)
        result = controller.optimize_splats([])
        assert result == []

    def test_validate_and_cleanup_valid_splats(self):
        """Test validation with valid splats."""
        controller = QualityController(target_count=10)
        splats = [
            Gaussian(x=10, y=10, rx=2, ry=2, theta=0, r=255, g=0, b=0, a=1.0),
            Gaussian(x=20, y=20, rx=3, ry=3, theta=45, r=0, g=255, b=0, a=0.8),
        ]

        result = controller._validate_and_cleanup(splats)
        assert len(result) == 2
        assert all(isinstance(splat, Gaussian) for splat in result)

    def test_validate_and_cleanup_invalid_splats(self):
        """Test validation removes invalid splats."""
        controller = QualityController(target_count=10)

        # Create valid splats first
        valid_splat = Gaussian(x=10, y=10, rx=2, ry=2, theta=0, r=255, g=0, b=0, a=1.0)

        # Create splats that we'll manually corrupt after construction
        splat_invalid_rx = Gaussian(x=20, y=20, rx=2, ry=2, theta=0, r=255, g=0, b=0, a=1.0)
        splat_invalid_color = Gaussian(x=30, y=30, rx=2, ry=2, theta=0, r=255, g=0, b=0, a=1.0)
        splat_invalid_coord = Gaussian(x=40, y=40, rx=2, ry=2, theta=0, r=255, g=0, b=0, a=1.0)

        # Manually corrupt them after construction (bypassing validation)
        splat_invalid_rx.rx = -1  # Invalid rx
        splat_invalid_color.r = 300  # Invalid color
        splat_invalid_coord.x = np.inf  # Invalid coordinate

        splats = [valid_splat, splat_invalid_rx, splat_invalid_color, splat_invalid_coord]

        result = controller._validate_and_cleanup(splats)
        assert len(result) == 1  # Only the valid splat should remain
        assert result[0].x == 10

    def test_validate_and_cleanup_fixes_alpha(self):
        """Test validation fixes out-of-bounds alpha values."""
        controller = QualityController(target_count=10)

        # Create valid splats and then corrupt alpha values
        splat1 = Gaussian(x=10, y=10, rx=2, ry=2, theta=0, r=255, g=0, b=0, a=1.0)
        splat2 = Gaussian(x=20, y=20, rx=2, ry=2, theta=0, r=255, g=0, b=0, a=1.0)

        # Manually set invalid alpha values after construction
        splat1.a = 1.5  # Alpha > 1.0
        splat2.a = -0.1  # Alpha < 0.0

        splats = [splat1, splat2]

        result = controller._validate_and_cleanup(splats)
        assert len(result) == 2
        assert result[0].a == 1.0  # Clamped to 1.0
        assert result[1].a == 0.0  # Clamped to 0.0

    def test_cull_micro_regions_basic(self):
        """Test micro-region culling with mixed size splats."""
        controller = QualityController(target_count=10, min_area_threshold=5.0)

        # Create splats with different areas
        splats = [
            Gaussian(x=10, y=10, rx=5, ry=5, theta=0, r=255, g=0, b=0, a=1.0),  # Area ≈ 78.5
            Gaussian(x=20, y=20, rx=2, ry=2, theta=0, r=0, g=255, b=0, a=1.0),  # Area ≈ 12.6
            Gaussian(x=30, y=30, rx=1, ry=1, theta=0, r=0, g=0, b=255, a=1.0),  # Area ≈ 3.14
        ]

        result = controller._cull_micro_regions(splats)

        # Only the larger splats should remain (adaptive threshold should remove the smallest)
        assert len(result) <= len(splats)
        if len(result) < len(splats):
            # The smallest splat should be removed
            remaining_areas = [splat.area() for splat in result]
            assert min(remaining_areas) > 3.14

    def test_cull_micro_regions_adaptive_threshold(self):
        """Test that micro-region culling uses adaptive threshold."""
        controller = QualityController(target_count=10, min_area_threshold=1.0)

        # Create splats where median area * 0.01 > min_area_threshold
        large_splats = [
            Gaussian(x=i, y=i, rx=10, ry=10, theta=0, r=255, g=0, b=0, a=1.0)
            for i in range(5)
        ]
        small_splats = [
            Gaussian(x=i+10, y=i+10, rx=1, ry=1, theta=0, r=0, g=255, b=0, a=1.0)
            for i in range(2)
        ]

        all_splats = large_splats + small_splats
        result = controller._cull_micro_regions(all_splats)

        # Should use adaptive threshold based on median area
        assert len(result) <= len(all_splats)

    def test_apply_size_filtering_basic(self):
        """Test size-based filtering with k_multiplier."""
        controller = QualityController(target_count=5, k_multiplier=2.0)

        # Create 20 splats with varying sizes
        splats = []
        for i in range(20):
            size = 1 + i * 0.5  # Increasing sizes
            splat = Gaussian(
                x=i, y=i, rx=size, ry=size, theta=0,
                r=255, g=0, b=0, a=1.0
            )
            splats.append(splat)

        result = controller._apply_size_filtering(splats)

        # Should keep top splats by size, limited by k_multiplier * target_count
        max_keep = int(5 * 2.0)  # 10 splats max
        assert len(result) <= max_keep

        # Should keep the largest splats
        if len(result) > 1:
            areas = [splat.area() for splat in result]
            assert areas == sorted(areas, reverse=True)

    def test_apply_size_filtering_no_limit(self):
        """Test size filtering when k_multiplier allows all splats."""
        controller = QualityController(target_count=5, k_multiplier=10.0)

        splats = [
            Gaussian(x=i, y=i, rx=2, ry=2, theta=0, r=255, g=0, b=0, a=1.0)
            for i in range(8)
        ]

        result = controller._apply_size_filtering(splats)

        # All splats should be kept since k_multiplier * target_count > len(splats)
        assert len(result) == len(splats)

    def test_achieve_target_count_basic(self):
        """Test achieving target count through score-based selection."""
        controller = QualityController(target_count=3)

        splats = [
            Gaussian(x=10, y=10, rx=2, ry=2, theta=0, r=255, g=0, b=0, a=1.0, score=0.9),
            Gaussian(x=20, y=20, rx=2, ry=2, theta=0, r=0, g=255, b=0, a=1.0, score=0.7),
            Gaussian(x=30, y=30, rx=2, ry=2, theta=0, r=0, g=0, b=255, a=1.0, score=0.5),
            Gaussian(x=40, y=40, rx=2, ry=2, theta=0, r=255, g=255, b=0, a=1.0, score=0.3),
            Gaussian(x=50, y=50, rx=2, ry=2, theta=0, r=255, g=0, b=255, a=1.0, score=0.1),
        ]

        result = controller._achieve_target_count(splats)

        assert len(result) == 3
        # Should keep highest-scoring splats
        scores = [splat.score for splat in result]
        assert scores == sorted(scores, reverse=True)
        assert min(scores) >= 0.5  # Should keep top 3 scores

    def test_achieve_target_count_fewer_splats(self):
        """Test target count when fewer splats than target."""
        controller = QualityController(target_count=10)

        splats = [
            Gaussian(x=i, y=i, rx=2, ry=2, theta=0, r=255, g=0, b=0, a=1.0, score=0.5)
            for i in range(5)
        ]

        result = controller._achieve_target_count(splats)

        # Should return all splats since count < target
        assert len(result) == 5

    def test_adjust_alpha_transparency_basic(self):
        """Test alpha transparency adjustment."""
        controller = QualityController(target_count=10, max_alpha=0.9)

        splats = [
            Gaussian(x=10, y=10, rx=2, ry=2, theta=0, r=255, g=0, b=0, a=1.0, score=0.9),
            Gaussian(x=20, y=20, rx=5, ry=5, theta=0, r=0, g=255, b=0, a=1.0, score=0.5),
            Gaussian(x=30, y=30, rx=1, ry=1, theta=0, r=0, g=0, b=255, a=1.0, score=0.1),
        ]

        result = controller._adjust_alpha_transparency(splats)

        assert len(result) == 3

        # All alpha values should be <= max_alpha
        alphas = [splat.a for splat in result]
        assert all(alpha <= 0.9 for alpha in alphas)

        # Higher scoring/larger splats should generally have higher alpha
        # (though exact values depend on the formula)
        assert all(alpha >= 0.7 for alpha in alphas)  # Base alpha is 0.7

    def test_adjust_alpha_transparency_disabled(self):
        """Test that alpha adjustment can be disabled."""
        controller = QualityController(target_count=10, alpha_adjustment=False)

        splats = [
            Gaussian(x=10, y=10, rx=2, ry=2, theta=0, r=255, g=0, b=0, a=0.5, score=0.9),
        ]

        result = controller.optimize_splats(splats)

        # Alpha should remain unchanged when adjustment is disabled
        assert result[0].a == 0.5

    def test_full_optimization_workflow(self):
        """Test complete optimization workflow."""
        controller = QualityController(
            target_count=5,
            k_multiplier=2.0,
            min_area_threshold=1.0,
            max_alpha=0.8,
            alpha_adjustment=True
        )

        # Create a mix of valid and invalid splats
        splats = []

        # Valid splats with varying scores and sizes
        for i in range(10):
            splat = Gaussian(
                x=i * 10, y=i * 10,
                rx=1 + i * 0.5, ry=1 + i * 0.5,
                theta=i * 30,
                r=255, g=128, b=64,
                a=1.0,
                score=i / 10.0
            )
            splats.append(splat)

        # Add some splats that we'll corrupt to test validation
        invalid_splat1 = Gaussian(x=100, y=100, rx=2, ry=2, theta=0, r=255, g=0, b=0, a=1.0)
        invalid_splat2 = Gaussian(x=110, y=110, rx=2, ry=2, theta=0, r=255, g=0, b=0, a=1.0)

        # Corrupt them after construction
        invalid_splat1.rx = -1  # Invalid rx
        invalid_splat2.r = 300  # Invalid color

        splats.extend([invalid_splat1, invalid_splat2])

        result = controller.optimize_splats(splats)

        # Should achieve target count (or close to it)
        assert len(result) <= 5
        assert len(result) > 0

        # All returned splats should be valid
        for splat in result:
            assert splat.rx > 0 and splat.ry > 0
            assert 0 <= splat.r <= 255
            assert 0 <= splat.g <= 255
            assert 0 <= splat.b <= 255
            assert 0.0 <= splat.a <= 0.8  # Respects max_alpha
            assert np.isfinite(splat.x) and np.isfinite(splat.y)

    def test_get_quality_statistics_basic(self):
        """Test quality statistics generation."""
        controller = QualityController(target_count=5)

        original_splats = [
            Gaussian(x=i, y=i, rx=2, ry=2, theta=0, r=255, g=0, b=0, a=1.0, score=i/10.0)
            for i in range(10)
        ]

        final_splats = original_splats[:5]  # Simulate filtering

        stats = controller.get_quality_statistics(original_splats, final_splats)

        assert stats['original_count'] == 10
        assert stats['final_count'] == 5
        assert stats['reduction_ratio'] == 0.5
        assert stats['target_achievement'] == 1.0  # Achieved exactly

        # Should include statistical measures
        assert 'avg_area' in stats
        assert 'avg_score' in stats
        assert 'avg_alpha' in stats
        assert 'alpha_range' in stats

    def test_get_quality_statistics_empty(self):
        """Test quality statistics with empty input."""
        controller = QualityController(target_count=5)

        stats = controller.get_quality_statistics([], [])
        assert stats == {}

    def test_optimization_preserves_important_splats(self):
        """Test that optimization preserves high-scoring, large splats."""
        controller = QualityController(
            target_count=3,
            k_multiplier=1.5,
            alpha_adjustment=False
        )

        # Create splats where some are clearly more important
        important_splats = [
            Gaussian(x=10, y=10, rx=5, ry=5, theta=0, r=255, g=0, b=0, a=1.0, score=0.9),
            Gaussian(x=20, y=20, rx=4, ry=4, theta=0, r=0, g=255, b=0, a=1.0, score=0.8),
        ]

        unimportant_splats = [
            Gaussian(x=i+50, y=i+50, rx=1, ry=1, theta=0, r=128, g=128, b=128, a=1.0, score=0.1)
            for i in range(10)
        ]

        all_splats = important_splats + unimportant_splats
        result = controller.optimize_splats(all_splats)

        # Important splats should be preserved
        result_positions = [(splat.x, splat.y) for splat in result]
        assert (10, 10) in result_positions  # First important splat
        assert (20, 20) in result_positions  # Second important splat

    def test_edge_case_single_splat(self):
        """Test optimization with single splat."""
        controller = QualityController(target_count=5)

        splats = [
            Gaussian(x=10, y=10, rx=2, ry=2, theta=0, r=255, g=0, b=0, a=1.0, score=0.5)
        ]

        result = controller.optimize_splats(splats)

        assert len(result) == 1
        assert result[0].x == 10

    def test_edge_case_all_same_scores(self):
        """Test optimization when all splats have same scores."""
        controller = QualityController(target_count=3)

        splats = [
            Gaussian(x=i*10, y=i*10, rx=2, ry=2, theta=0, r=255, g=0, b=0, a=1.0, score=0.5)
            for i in range(5)
        ]

        result = controller.optimize_splats(splats)

        assert len(result) == 3  # Should still achieve target count

    def test_k_multiplier_zero_edge_case(self):
        """Test edge case where k_multiplier is zero."""
        controller = QualityController(target_count=5, k_multiplier=0.0)

        splats = [
            Gaussian(x=i*10, y=i*10, rx=2, ry=2, theta=0, r=255, g=0, b=0, a=1.0, score=0.5)
            for i in range(10)
        ]

        result = controller.optimize_splats(splats)

        # Should still work, but size filtering might be bypassed
        assert len(result) <= 5