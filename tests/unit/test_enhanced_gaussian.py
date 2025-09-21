"""Tests for enhanced Gaussian features and SplatCollection."""

import pytest
import math

from splat_this.core.extract import Gaussian, SplatCollection


class TestGaussianComparison:
    """Test Gaussian comparison operators."""

    def test_depth_comparison(self):
        """Test comparison operators based on depth."""
        splat1 = Gaussian(
            x=10, y=10, rx=5, ry=5, theta=0, r=255, g=0, b=0, a=0.8, depth=0.3
        )
        splat2 = Gaussian(
            x=20, y=20, rx=5, ry=5, theta=0, r=0, g=255, b=0, a=0.8, depth=0.7
        )

        # Test less than
        assert splat1 < splat2
        assert not (splat2 < splat1)

        # Test less than or equal
        assert splat1 <= splat2
        assert not (splat2 <= splat1)

        # Test greater than
        assert splat2 > splat1
        assert not (splat1 > splat2)

        # Test greater than or equal
        assert splat2 >= splat1
        assert not (splat1 >= splat2)

    def test_equality_comparison(self):
        """Test equality comparison."""
        splat1 = Gaussian(
            x=10,
            y=10,
            rx=5,
            ry=5,
            theta=0.5,
            r=255,
            g=128,
            b=64,
            a=0.8,
            score=0.9,
            depth=0.5,
        )
        splat2 = Gaussian(
            x=10,
            y=10,
            rx=5,
            ry=5,
            theta=0.5,
            r=255,
            g=128,
            b=64,
            a=0.8,
            score=0.9,
            depth=0.5,
        )
        splat3 = Gaussian(
            x=11,
            y=10,
            rx=5,
            ry=5,
            theta=0.5,
            r=255,
            g=128,
            b=64,
            a=0.8,
            score=0.9,
            depth=0.5,
        )

        assert splat1 == splat2
        assert not (splat1 == splat3)
        assert splat1 != splat3

    def test_comparison_with_non_gaussian(self):
        """Test comparison with non-Gaussian objects."""
        splat = Gaussian(x=10, y=10, rx=5, ry=5, theta=0, r=255, g=0, b=0, a=0.8)

        # Ordering comparisons should raise TypeError for incompatible types
        with pytest.raises(TypeError):
            splat < "not a gaussian"

        with pytest.raises(TypeError):
            splat <= "not a gaussian"

        with pytest.raises(TypeError):
            splat > "not a gaussian"

        with pytest.raises(TypeError):
            splat >= "not a gaussian"

        # Equality comparisons should work (return False/True)
        assert not (splat == "not a gaussian")
        assert splat != "not a gaussian"


class TestGaussianTransformations:
    """Test Gaussian transformation methods."""

    def test_translate(self):
        """Test translation method."""
        original = Gaussian(
            x=10, y=20, rx=5, ry=3, theta=0.5, r=255, g=128, b=64, a=0.8
        )
        translated = original.translate(5, -10)

        assert translated.x == 15
        assert translated.y == 10
        # Other properties should remain unchanged
        assert translated.rx == original.rx
        assert translated.ry == original.ry
        assert translated.theta == original.theta
        assert translated.r == original.r
        assert translated.g == original.g
        assert translated.b == original.b
        assert translated.a == original.a

    def test_scale_uniform(self):
        """Test uniform scaling."""
        original = Gaussian(
            x=10, y=20, rx=5, ry=3, theta=0.5, r=255, g=128, b=64, a=0.8
        )
        scaled = original.scale(2.0)

        assert scaled.x == 20
        assert scaled.y == 40
        assert scaled.rx == 10
        assert scaled.ry == 6
        # Other properties should remain unchanged
        assert scaled.theta == original.theta
        assert scaled.r == original.r

    def test_scale_non_uniform(self):
        """Test non-uniform scaling."""
        original = Gaussian(
            x=10, y=20, rx=5, ry=3, theta=0.5, r=255, g=128, b=64, a=0.8
        )
        scaled = original.scale(2.0, 3.0)

        assert scaled.x == 20
        assert scaled.y == 60
        assert scaled.rx == 10
        assert scaled.ry == 9

    def test_rotate(self):
        """Test rotation method."""
        original = Gaussian(x=10, y=0, rx=5, ry=3, theta=0, r=255, g=128, b=64, a=0.8)
        rotated = original.rotate(math.pi / 2)  # 90 degrees

        # After 90-degree rotation around origin, (10, 0) becomes (0, 10)
        assert abs(rotated.x - 0) < 1e-6
        assert abs(rotated.y - 10) < 1e-6
        assert abs(rotated.theta - math.pi / 2) < 1e-6

    def test_rotate_around_center(self):
        """Test rotation around a specific center."""
        original = Gaussian(x=20, y=10, rx=5, ry=3, theta=0, r=255, g=128, b=64, a=0.8)
        rotated = original.rotate(
            math.pi / 2, center_x=10, center_y=10
        )  # 90 degrees around (10, 10)

        # (20, 10) rotated 90 degrees around (10, 10) becomes (10, 20)
        assert abs(rotated.x - 10) < 1e-6
        assert abs(rotated.y - 20) < 1e-6


class TestGaussianBlending:
    """Test Gaussian blending functionality."""

    def test_blend_with_equal_weight(self):
        """Test blending with equal weight (0.5)."""
        splat1 = Gaussian(
            x=10,
            y=10,
            rx=5,
            ry=5,
            theta=0,
            r=255,
            g=0,
            b=0,
            a=0.8,
            score=0.9,
            depth=0.3,
        )
        splat2 = Gaussian(
            x=20,
            y=30,
            rx=10,
            ry=8,
            theta=1,
            r=0,
            g=255,
            b=0,
            a=0.6,
            score=0.7,
            depth=0.7,
        )

        blended = splat1.blend_with(splat2, 0.5)

        assert blended.x == 15  # (10 + 20) / 2
        assert blended.y == 20  # (10 + 30) / 2
        assert blended.rx == 7.5  # (5 + 10) / 2
        assert blended.ry == 6.5  # (5 + 8) / 2
        assert blended.theta == 0.5  # (0 + 1) / 2
        assert blended.r == 127  # (255 + 0) / 2
        assert blended.g == 127  # (0 + 255) / 2
        assert blended.b == 0  # (0 + 0) / 2
        assert abs(blended.a - 0.7) < 1e-6  # (0.8 + 0.6) / 2
        assert abs(blended.score - 0.8) < 1e-6  # (0.9 + 0.7) / 2
        assert abs(blended.depth - 0.5) < 1e-6  # (0.3 + 0.7) / 2

    def test_blend_with_extreme_weights(self):
        """Test blending with extreme weights."""
        splat1 = Gaussian(x=10, y=10, rx=5, ry=5, theta=0, r=255, g=0, b=0, a=0.8)
        splat2 = Gaussian(x=20, y=30, rx=10, ry=8, theta=1, r=0, g=255, b=0, a=0.6)

        # Weight 0 should return splat1
        blended0 = splat1.blend_with(splat2, 0.0)
        assert abs(blended0.x - splat1.x) < 1e-6
        assert abs(blended0.y - splat1.y) < 1e-6

        # Weight 1 should return splat2
        blended1 = splat1.blend_with(splat2, 1.0)
        assert abs(blended1.x - splat2.x) < 1e-6
        assert abs(blended1.y - splat2.y) < 1e-6

    def test_blend_with_invalid_type(self):
        """Test blending with invalid type."""
        splat = Gaussian(x=10, y=10, rx=5, ry=5, theta=0, r=255, g=0, b=0, a=0.8)

        with pytest.raises(TypeError):
            splat.blend_with("not a gaussian")


class TestGaussianUtilities:
    """Test Gaussian utility methods."""

    def test_distance_to(self):
        """Test distance calculation."""
        splat1 = Gaussian(x=0, y=0, rx=5, ry=5, theta=0, r=255, g=0, b=0, a=0.8)
        splat2 = Gaussian(x=3, y=4, rx=5, ry=5, theta=0, r=0, g=255, b=0, a=0.8)

        distance = splat1.distance_to(splat2)
        assert abs(distance - 5.0) < 1e-6  # 3-4-5 triangle

    def test_distance_to_invalid_type(self):
        """Test distance calculation with invalid type."""
        splat = Gaussian(x=10, y=10, rx=5, ry=5, theta=0, r=255, g=0, b=0, a=0.8)

        with pytest.raises(TypeError):
            splat.distance_to("not a gaussian")

    def test_overlaps_with_overlapping(self):
        """Test overlap detection with overlapping Gaussians."""
        splat1 = Gaussian(x=10, y=10, rx=5, ry=5, theta=0, r=255, g=0, b=0, a=0.8)
        splat2 = Gaussian(x=12, y=12, rx=5, ry=5, theta=0, r=0, g=255, b=0, a=0.8)

        assert splat1.overlaps_with(splat2)
        assert splat2.overlaps_with(splat1)

    def test_overlaps_with_non_overlapping(self):
        """Test overlap detection with non-overlapping Gaussians."""
        splat1 = Gaussian(x=0, y=0, rx=5, ry=5, theta=0, r=255, g=0, b=0, a=0.8)
        splat2 = Gaussian(x=20, y=20, rx=5, ry=5, theta=0, r=0, g=255, b=0, a=0.8)

        assert not splat1.overlaps_with(splat2)
        assert not splat2.overlaps_with(splat1)

    def test_overlaps_with_invalid_type(self):
        """Test overlap check with invalid type."""
        splat = Gaussian(x=10, y=10, rx=5, ry=5, theta=0, r=255, g=0, b=0, a=0.8)

        with pytest.raises(TypeError):
            splat.overlaps_with("not a gaussian")


class TestSplatCollection:
    """Test SplatCollection functionality."""

    def create_test_splats(self):
        """Create a list of test splats."""
        return [
            Gaussian(
                x=10,
                y=10,
                rx=5,
                ry=5,
                theta=0,
                r=255,
                g=0,
                b=0,
                a=0.8,
                score=0.9,
                depth=0.1,
            ),
            Gaussian(
                x=20,
                y=20,
                rx=3,
                ry=3,
                theta=0,
                r=0,
                g=255,
                b=0,
                a=0.6,
                score=0.5,
                depth=0.5,
            ),
            Gaussian(
                x=30,
                y=30,
                rx=8,
                ry=8,
                theta=0,
                r=0,
                g=0,
                b=255,
                a=0.7,
                score=0.8,
                depth=0.9,
            ),
        ]

    def test_collection_creation(self):
        """Test creating a SplatCollection."""
        splats = self.create_test_splats()
        collection = SplatCollection(splats)

        assert len(collection) == 3
        assert collection[0] == splats[0]
        assert collection[1] == splats[1]
        assert collection[2] == splats[2]

    def test_collection_iteration(self):
        """Test iterating over a SplatCollection."""
        splats = self.create_test_splats()
        collection = SplatCollection(splats)

        collected = list(collection)
        assert len(collected) == 3
        assert collected == splats

    def test_filter_by_score(self):
        """Test filtering by score."""
        splats = self.create_test_splats()
        collection = SplatCollection(splats)

        filtered = collection.filter_by_score(0.7)
        assert len(filtered) == 2  # Score 0.9 and 0.8 pass threshold 0.7
        assert filtered[0].score >= 0.7
        assert filtered[1].score >= 0.7

    def test_filter_by_area(self):
        """Test filtering by area."""
        splats = self.create_test_splats()
        collection = SplatCollection(splats)

        # Filter by minimum area
        min_filtered = collection.filter_by_area(min_area=50)
        assert len(min_filtered) == 2  # Only splats with area >= 50

        # Filter by maximum area
        max_filtered = collection.filter_by_area(max_area=100)
        assert len(max_filtered) == 2  # Only splats with area <= 100

        # Filter by range
        range_filtered = collection.filter_by_area(min_area=30, max_area=100)
        assert len(range_filtered) == 1  # Only medium-sized splats

    def test_sort_by_depth(self):
        """Test sorting by depth."""
        splats = self.create_test_splats()
        collection = SplatCollection(splats)

        # Sort front to back (default)
        sorted_front = collection.sort_by_depth()
        depths = [s.depth for s in sorted_front]
        assert depths == [0.1, 0.5, 0.9]

        # Sort back to front
        sorted_back = collection.sort_by_depth(reverse=True)
        depths_back = [s.depth for s in sorted_back]
        assert depths_back == [0.9, 0.5, 0.1]

    def test_sort_by_score(self):
        """Test sorting by score."""
        splats = self.create_test_splats()
        collection = SplatCollection(splats)

        # Sort highest first (default)
        sorted_high = collection.sort_by_score()
        scores = [s.score for s in sorted_high]
        assert scores == [0.9, 0.8, 0.5]

        # Sort lowest first
        sorted_low = collection.sort_by_score(reverse=False)
        scores_low = [s.score for s in sorted_low]
        assert scores_low == [0.5, 0.8, 0.9]

    def test_sort_by_area(self):
        """Test sorting by area."""
        splats = self.create_test_splats()
        collection = SplatCollection(splats)

        # Sort largest first (default)
        sorted_large = collection.sort_by_area()
        areas = [s.area() for s in sorted_large]
        assert areas[0] > areas[1] > areas[2]

    def test_get_statistics(self):
        """Test getting collection statistics."""
        splats = self.create_test_splats()
        collection = SplatCollection(splats)

        stats = collection.get_statistics()
        assert stats["count"] == 3
        assert stats["score_range"] == (0.5, 0.9)
        assert stats["depth_range"] == (0.1, 0.9)
        assert stats["avg_score"] == (0.9 + 0.5 + 0.8) / 3

    def test_get_statistics_empty(self):
        """Test getting statistics for empty collection."""
        empty_collection = SplatCollection([])
        stats = empty_collection.get_statistics()

        assert stats["count"] == 0
        assert stats["avg_score"] == 0

    def test_remove_overlapping(self):
        """Test removing overlapping splats."""
        # Create overlapping splats (close together with significant overlap)
        splats = [
            Gaussian(
                x=10, y=10, rx=5, ry=5, theta=0, r=255, g=0, b=0, a=0.8, score=0.9
            ),
            Gaussian(
                x=11, y=11, rx=5, ry=5, theta=0, r=0, g=255, b=0, a=0.6, score=0.7
            ),  # Close overlap with first
            Gaussian(
                x=30, y=30, rx=5, ry=5, theta=0, r=0, g=0, b=255, a=0.7, score=0.8
            ),  # Separate
        ]
        collection = SplatCollection(splats)

        # Test with a lower threshold to ensure overlap is detected
        filtered = collection.remove_overlapping(threshold=0.3)
        # Should keep the highest scoring splat from overlapping pair plus the separate one
        assert len(filtered) == 2

        # Check that the highest scoring splats are kept
        scores = [s.score for s in filtered]
        assert 0.9 in scores  # Highest score should be kept
        assert 0.8 in scores  # Separate splat should be kept
        assert 0.7 not in scores  # Lower scoring overlapping splat should be removed

    def test_translate_all(self):
        """Test translating all splats."""
        splats = self.create_test_splats()
        collection = SplatCollection(splats)

        translated = collection.translate_all(5, -10)
        assert len(translated) == 3
        assert translated[0].x == 15  # 10 + 5
        assert translated[0].y == 0  # 10 - 10
        assert translated[1].x == 25  # 20 + 5
        assert translated[1].y == 10  # 20 - 10

    def test_scale_all(self):
        """Test scaling all splats."""
        splats = self.create_test_splats()
        collection = SplatCollection(splats)

        scaled = collection.scale_all(2.0)
        assert len(scaled) == 3
        assert scaled[0].x == 20  # 10 * 2
        assert scaled[0].rx == 10  # 5 * 2

    def test_to_list(self):
        """Test converting collection to list."""
        splats = self.create_test_splats()
        collection = SplatCollection(splats)

        result_list = collection.to_list()
        assert isinstance(result_list, list)
        assert len(result_list) == 3
        assert result_list == splats

    def test_to_dicts(self):
        """Test converting collection to dictionaries."""
        splats = self.create_test_splats()
        collection = SplatCollection(splats)

        dicts = collection.to_dicts()
        assert isinstance(dicts, list)
        assert len(dicts) == 3
        assert all(isinstance(d, dict) for d in dicts)

    def test_from_dicts(self):
        """Test creating collection from dictionaries."""
        splats = self.create_test_splats()
        dicts = [splat.to_dict() for splat in splats]

        collection = SplatCollection.from_dicts(dicts)
        assert len(collection) == 3
        # Test that the recreated splats are equivalent
        for original, recreated in zip(splats, collection):
            assert original == recreated

    def test_merge_with(self):
        """Test merging collections."""
        splats1 = self.create_test_splats()[:2]
        splats2 = self.create_test_splats()[2:]

        collection1 = SplatCollection(splats1)
        collection2 = SplatCollection(splats2)

        merged = collection1.merge_with(collection2)
        assert len(merged) == 3
        assert merged[0] == splats1[0]
        assert merged[1] == splats1[1]
        assert merged[2] == splats2[0]

    def test_merge_with_invalid_type(self):
        """Test merging with invalid type."""
        splats = self.create_test_splats()
        collection = SplatCollection(splats)

        with pytest.raises(TypeError):
            collection.merge_with("not a collection")


class TestSplatCollectionEdgeCases:
    """Test edge cases for SplatCollection."""

    def test_empty_collection(self):
        """Test operations on empty collection."""
        empty = SplatCollection([])

        assert len(empty) == 0
        assert list(empty) == []

        # Test filtering operations
        filtered = empty.filter_by_score(0.5)
        assert len(filtered) == 0

        # Test sorting operations
        sorted_empty = empty.sort_by_depth()
        assert len(sorted_empty) == 0

        # Test transformation operations
        translated = empty.translate_all(10, 10)
        assert len(translated) == 0

    def test_single_splat_collection(self):
        """Test operations on single-splat collection."""
        splat = Gaussian(
            x=10, y=10, rx=5, ry=5, theta=0, r=255, g=0, b=0, a=0.8, score=0.9
        )
        single = SplatCollection([splat])

        assert len(single) == 1
        assert single[0] == splat

        # Test operations return valid results
        stats = single.get_statistics()
        assert stats["count"] == 1
        assert stats["avg_score"] == 0.9

        filtered = single.filter_by_score(0.5)
        assert len(filtered) == 1

        sorted_single = single.sort_by_depth()
        assert len(sorted_single) == 1
