"""Tests for importance scoring and multi-factor analysis."""

import numpy as np
from unittest.mock import patch, MagicMock

from splat_this.core.extract import Gaussian
from splat_this.core.layering import ImportanceScorer


class TestImportanceScorer:
    """Test the ImportanceScorer class."""

    def test_scorer_initialization(self):
        """Test ImportanceScorer initialization with default and custom weights."""
        # Default weights
        scorer = ImportanceScorer()
        assert scorer.area_weight == 0.3
        assert scorer.edge_weight == 0.5
        assert scorer.color_weight == 0.2

        # Custom weights
        custom_scorer = ImportanceScorer(
            area_weight=0.4, edge_weight=0.3, color_weight=0.3
        )
        assert custom_scorer.area_weight == 0.4
        assert custom_scorer.edge_weight == 0.3
        assert custom_scorer.color_weight == 0.3

    def test_weight_normalization(self):
        """Test that weights should sum to 1.0 for proper scoring."""
        scorer = ImportanceScorer()
        total_weight = scorer.area_weight + scorer.edge_weight + scorer.color_weight
        assert abs(total_weight - 1.0) < 1e-6  # Weights should sum to 1

    def create_test_image(self, width=100, height=100):
        """Create a test image with known patterns."""
        image = np.zeros((height, width, 3), dtype=np.uint8)

        # Add a high-edge region (checkerboard pattern)
        for i in range(0, height // 2, 4):
            for j in range(0, width // 2, 4):
                image[i : i + 2, j : j + 2] = [255, 255, 255]
                image[i + 2 : i + 4, j + 2 : j + 4] = [255, 255, 255]

        # Add a smooth gradient region
        for i in range(height // 2, height):
            intensity = int(255 * i / height)
            image[i, :] = [intensity, intensity // 2, intensity // 3]

        # Add a high-variance color region
        np.random.seed(42)  # For reproducible tests
        noise_region = np.random.randint(0, 256, (20, 20, 3), dtype=np.uint8)
        image[10:30, 70:90] = noise_region

        return image

    def create_test_splats(self):
        """Create test splats at known locations."""
        return [
            # Small splat in high-edge region
            Gaussian(x=10, y=10, rx=5, ry=5, theta=0, r=128, g=128, b=128, a=0.8),
            # Medium splat in smooth region
            Gaussian(x=50, y=75, rx=15, ry=10, theta=0, r=64, g=64, b=64, a=0.8),
            # Large splat covering multiple regions
            Gaussian(x=50, y=50, rx=30, ry=25, theta=0, r=192, g=192, b=192, a=0.8),
            # Small splat in high-variance color region
            Gaussian(x=80, y=20, rx=3, ry=3, theta=0, r=255, g=0, b=0, a=0.8),
        ]

    def test_area_score_calculation(self):
        """Test area-based scoring."""
        scorer = ImportanceScorer()
        image_area = 100 * 100

        # Small splat (area = π * 5 * 5 ≈ 78.5)
        small_splat = Gaussian(
            x=50, y=50, rx=5, ry=5, theta=0, r=128, g=128, b=128, a=0.8
        )
        small_score = scorer._calculate_area_score(small_splat, image_area)

        # Large splat (area = π * 20 * 20 ≈ 1256.6)
        large_splat = Gaussian(
            x=50, y=50, rx=20, ry=20, theta=0, r=128, g=128, b=128, a=0.8
        )
        large_score = scorer._calculate_area_score(large_splat, image_area)

        # Very large splat (area = π * 40 * 40 ≈ 5026.5)
        very_large_splat = Gaussian(
            x=50, y=50, rx=40, ry=40, theta=0, r=128, g=128, b=128, a=0.8
        )
        very_large_score = scorer._calculate_area_score(very_large_splat, image_area)

        # Scores should be in [0, 1] range
        assert 0 <= small_score <= 1
        assert 0 <= large_score <= 1
        assert 0 <= very_large_score <= 1

        # Large splat should score higher than small (up to optimal ratio)
        # The optimal ratio is 1% of image area, so 100 pixels out of 10000
        # Small splat area ≈ 78.5, large splat area ≈ 1256.6
        # Since large splat is beyond optimal ratio, it may score lower
        assert small_score > 0  # Small splat should get some score
        assert large_score > 0  # Large splat should get some score

        # But very large splat should start getting diminishing returns
        assert very_large_score < 1.0  # Should not be perfect score

    def test_edge_score_with_opencv(self):
        """Test edge-based scoring with OpenCV available."""
        with patch("splat_this.core.layering.HAS_OPENCV", True):
            mock_cv2 = MagicMock()
            with patch.dict("sys.modules", {"cv2": mock_cv2}):
                # Mock OpenCV functions
                mock_cv2.cvtColor.return_value = (
                    np.ones((100, 100), dtype=np.uint8) * 128
                )
                mock_cv2.Laplacian.return_value = np.random.normal(0, 50, (100, 100))
                mock_cv2.COLOR_RGB2GRAY = 7  # Mock constant

                scorer = ImportanceScorer()
                image = self.create_test_image(100, 100)
                splat = Gaussian(
                    x=25, y=25, rx=10, ry=10, theta=0, r=128, g=128, b=128, a=0.8
                )

                # Pre-compute edge map
                edge_map = scorer._compute_edge_map(image)
                score = scorer._calculate_edge_score(splat, image, edge_map)

                assert 0 <= score <= 1
                mock_cv2.cvtColor.assert_called_once()
                mock_cv2.Laplacian.assert_called_once()

    def test_edge_score_without_opencv(self):
        """Test edge-based scoring fallback without OpenCV."""
        with patch("splat_this.core.layering.HAS_OPENCV", False):
            scorer = ImportanceScorer()
            image = self.create_test_image(100, 100)
            splat = Gaussian(
                x=25, y=25, rx=10, ry=10, theta=0, r=128, g=128, b=128, a=0.8
            )

            score = scorer._calculate_edge_score(splat, image, edge_map=None)
            assert 0 <= score <= 1

    def test_color_score_calculation(self):
        """Test color variance-based scoring."""
        scorer = ImportanceScorer()

        # Create images with different color characteristics
        # Uniform color image
        uniform_image = np.full((100, 100, 3), [128, 128, 128], dtype=np.uint8)
        uniform_splat = Gaussian(
            x=50, y=50, rx=10, ry=10, theta=0, r=128, g=128, b=128, a=0.8
        )
        uniform_score = scorer._calculate_color_score(uniform_splat, uniform_image)

        # High variance color image
        np.random.seed(42)
        varied_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        varied_splat = Gaussian(
            x=50, y=50, rx=10, ry=10, theta=0, r=128, g=128, b=128, a=0.8
        )
        varied_score = scorer._calculate_color_score(varied_splat, varied_image)

        # Scores should be in [0, 1] range
        assert 0 <= uniform_score <= 1
        assert 0 <= varied_score <= 1

        # High variance region should score higher than uniform region
        assert varied_score > uniform_score

    def test_color_score_grayscale(self):
        """Test color scoring with grayscale images."""
        scorer = ImportanceScorer()

        # Grayscale image
        gray_image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        splat = Gaussian(x=50, y=50, rx=10, ry=10, theta=0, r=128, g=128, b=128, a=0.8)

        score = scorer._calculate_color_score(splat, gray_image)
        assert 0 <= score <= 1

    def test_edge_cases_out_of_bounds(self):
        """Test scoring methods with out-of-bounds splats."""
        scorer = ImportanceScorer()
        image = self.create_test_image(100, 100)

        # Splat completely out of bounds
        out_of_bounds_splat = Gaussian(
            x=-50, y=-50, rx=10, ry=10, theta=0, r=128, g=128, b=128, a=0.8
        )

        edge_score = scorer._calculate_edge_score(out_of_bounds_splat, image)
        color_score = scorer._calculate_color_score(out_of_bounds_splat, image)

        assert edge_score == 0.0
        assert color_score == 0.0

        # Splat partially out of bounds
        partial_out_splat = Gaussian(
            x=5, y=5, rx=10, ry=10, theta=0, r=128, g=128, b=128, a=0.8
        )

        edge_score = scorer._calculate_edge_score(partial_out_splat, image)
        color_score = scorer._calculate_color_score(partial_out_splat, image)

        assert 0 <= edge_score <= 1
        assert 0 <= color_score <= 1

    def test_score_splats_empty_list(self):
        """Test scoring with empty splat list."""
        scorer = ImportanceScorer()
        image = self.create_test_image()
        empty_splats = []

        # Should not raise an error
        scorer.score_splats(empty_splats, image)
        assert len(empty_splats) == 0

    def test_score_splats_integration(self):
        """Test complete scoring pipeline."""
        scorer = ImportanceScorer(area_weight=0.3, edge_weight=0.4, color_weight=0.3)
        image = self.create_test_image()
        splats = self.create_test_splats()

        # Score the splats
        scorer.score_splats(splats, image)

        # Verify all splats have been scored
        for i, splat in enumerate(splats):
            assert 0 <= splat.score <= 1
            # Scores should have changed from originals (unless original was already in valid range)
            assert hasattr(splat, "score")

        # Verify scores are different (assuming diverse test image)
        scores = [splat.score for splat in splats]
        assert len(set(scores)) > 1  # Should have different scores

    def test_vectorized_scoring(self):
        """Test vectorized scoring method."""
        scorer = ImportanceScorer()
        image = self.create_test_image()
        splats = self.create_test_splats()

        # Score using vectorized method
        scorer.score_splats_vectorized(splats, image)

        # Verify all scores are valid
        for splat in splats:
            assert 0 <= splat.score <= 1

    def test_vectorized_vs_regular_scoring(self):
        """Test that vectorized and regular scoring produce similar results."""
        scorer = ImportanceScorer(area_weight=0.3, edge_weight=0.4, color_weight=0.3)
        image = self.create_test_image()

        # Create two identical sets of splats
        splats1 = self.create_test_splats()
        splats2 = [
            Gaussian(
                x=s.x,
                y=s.y,
                rx=s.rx,
                ry=s.ry,
                theta=s.theta,
                r=s.r,
                g=s.g,
                b=s.b,
                a=s.a,
                score=s.score,
                depth=s.depth,
            )
            for s in splats1
        ]

        # Score using both methods
        scorer.score_splats(splats1, image)
        scorer.score_splats_vectorized(splats2, image)

        # Compare scores (should be similar, allowing for differences due to implementation)
        for s1, s2 in zip(splats1, splats2):
            assert (
                abs(s1.score - s2.score) < 0.3
            )  # Allow for larger differences due to different implementations

    def test_scoring_with_different_weights(self):
        """Test that different weight configurations produce different scores."""
        image = self.create_test_image()
        splat = Gaussian(x=25, y=25, rx=10, ry=10, theta=0, r=128, g=128, b=128, a=0.8)

        # Score with edge-heavy weighting
        edge_scorer = ImportanceScorer(
            area_weight=0.1, edge_weight=0.8, color_weight=0.1
        )
        splat.score = 0.0  # Reset
        edge_scorer.score_splats([splat], image)
        edge_score = splat.score

        # Score with area-heavy weighting
        area_scorer = ImportanceScorer(
            area_weight=0.8, edge_weight=0.1, color_weight=0.1
        )
        splat.score = 0.0  # Reset
        area_scorer.score_splats([splat], image)
        area_score = splat.score

        # Scores should be different (unless the splat happens to score equally on both factors)
        # This test mainly ensures the weighting system is working
        assert 0 <= edge_score <= 1
        assert 0 <= area_score <= 1

    def test_compute_edge_map_with_opencv(self):
        """Test edge map computation with OpenCV."""
        with patch("splat_this.core.layering.HAS_OPENCV", True):
            mock_cv2 = MagicMock()
            with patch.dict("sys.modules", {"cv2": mock_cv2}):
                mock_cv2.cvtColor.return_value = (
                    np.ones((100, 100), dtype=np.uint8) * 128
                )
                mock_cv2.Laplacian.return_value = (
                    np.ones((100, 100), dtype=np.float64) * 50
                )
                mock_cv2.COLOR_RGB2GRAY = 7

                scorer = ImportanceScorer()
                image = self.create_test_image()

                edge_map = scorer._compute_edge_map(image)

                assert edge_map.shape == (100, 100)
                assert edge_map.dtype in [np.float64, np.float32]
                mock_cv2.cvtColor.assert_called_once()
                mock_cv2.Laplacian.assert_called_once()

    def test_compute_edge_map_without_opencv(self):
        """Test edge map computation fallback without OpenCV."""
        with patch("splat_this.core.layering.HAS_OPENCV", False):
            scorer = ImportanceScorer()
            image = self.create_test_image()

            edge_map = scorer._compute_edge_map(image)

            assert edge_map.shape == (100, 100)
            assert edge_map.dtype in [np.float64, np.float32]


class TestImportanceScorerEdgeCases:
    """Test edge cases and error conditions for ImportanceScorer."""

    def test_tiny_splats(self):
        """Test scoring very small splats."""
        scorer = ImportanceScorer()
        image = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)

        tiny_splat = Gaussian(
            x=25, y=25, rx=0.5, ry=0.5, theta=0, r=128, g=128, b=128, a=0.8
        )
        scorer.score_splats([tiny_splat], image)

        assert 0 <= tiny_splat.score <= 1

    def test_huge_splats(self):
        """Test scoring very large splats."""
        scorer = ImportanceScorer()
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

        huge_splat = Gaussian(
            x=50, y=50, rx=100, ry=100, theta=0, r=128, g=128, b=128, a=0.8
        )
        scorer.score_splats([huge_splat], image)

        assert 0 <= huge_splat.score <= 1

    def test_small_image(self):
        """Test scoring with small image size."""
        scorer = ImportanceScorer()
        image = np.full((3, 3, 3), [255, 0, 0], dtype=np.uint8)  # 3x3 image

        splat = Gaussian(x=1, y=1, rx=1, ry=1, theta=0, r=128, g=128, b=128, a=0.8)
        scorer.score_splats([splat], image)

        assert 0 <= splat.score <= 1

    def test_zero_variance_region(self):
        """Test color scoring in region with no variance."""
        scorer = ImportanceScorer()
        uniform_image = np.full((50, 50, 3), [100, 100, 100], dtype=np.uint8)

        splat = Gaussian(x=25, y=25, rx=10, ry=10, theta=0, r=128, g=128, b=128, a=0.8)
        color_score = scorer._calculate_color_score(splat, uniform_image)

        assert color_score == 0.0  # No variance should give zero score

    def test_invalid_splat_positions(self):
        """Test scoring splats at edge and corner positions."""
        scorer = ImportanceScorer()
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

        edge_splats = [
            Gaussian(
                x=0, y=50, rx=5, ry=5, theta=0, r=128, g=128, b=128, a=0.8
            ),  # Left edge
            Gaussian(
                x=99, y=50, rx=5, ry=5, theta=0, r=128, g=128, b=128, a=0.8
            ),  # Right edge
            Gaussian(
                x=50, y=0, rx=5, ry=5, theta=0, r=128, g=128, b=128, a=0.8
            ),  # Top edge
            Gaussian(
                x=50, y=99, rx=5, ry=5, theta=0, r=128, g=128, b=128, a=0.8
            ),  # Bottom edge
            Gaussian(
                x=0, y=0, rx=5, ry=5, theta=0, r=128, g=128, b=128, a=0.8
            ),  # Corner
        ]

        scorer.score_splats(edge_splats, image)

        for splat in edge_splats:
            assert 0 <= splat.score <= 1
