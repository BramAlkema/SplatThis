"""Tests for Gaussian splat extraction functionality."""

import pytest
import numpy as np
from unittest.mock import patch

from splat_this.core.extract import Gaussian, SplatExtractor


class TestGaussian:
    """Test the Gaussian dataclass."""

    def test_valid_gaussian_creation(self):
        """Test creating a valid Gaussian splat."""
        splat = Gaussian(
            x=100.0,
            y=50.0,
            rx=10.0,
            ry=8.0,
            theta=0.5,
            r=255,
            g=128,
            b=64,
            a=0.8,
        )

        assert splat.x == 100.0
        assert splat.y == 50.0
        assert splat.rx == 10.0
        assert splat.ry == 8.0
        assert splat.theta == 0.5
        assert splat.r == 255
        assert splat.g == 128
        assert splat.b == 64
        assert splat.a == 0.8
        assert splat.score == 0.0  # Default value
        assert splat.depth == 0.5  # Default value

    def test_gaussian_validation_invalid_radii(self):
        """Test validation with invalid radii."""
        with pytest.raises(ValueError, match="Invalid radii"):
            Gaussian(
                x=100.0, y=50.0, rx=0.0, ry=8.0, theta=0.5, r=255, g=128, b=64, a=0.8
            )

        with pytest.raises(ValueError, match="Invalid radii"):
            Gaussian(
                x=100.0, y=50.0, rx=10.0, ry=-5.0, theta=0.5, r=255, g=128, b=64, a=0.8
            )

    def test_gaussian_validation_invalid_rgb(self):
        """Test validation with invalid RGB values."""
        with pytest.raises(ValueError, match="Invalid RGB"):
            Gaussian(
                x=100.0, y=50.0, rx=10.0, ry=8.0, theta=0.5, r=256, g=128, b=64, a=0.8
            )

        with pytest.raises(ValueError, match="Invalid RGB"):
            Gaussian(
                x=100.0, y=50.0, rx=10.0, ry=8.0, theta=0.5, r=255, g=-1, b=64, a=0.8
            )

    def test_gaussian_validation_invalid_alpha(self):
        """Test validation with invalid alpha values."""
        with pytest.raises(ValueError, match="Invalid alpha"):
            Gaussian(
                x=100.0, y=50.0, rx=10.0, ry=8.0, theta=0.5, r=255, g=128, b=64, a=1.5
            )

        with pytest.raises(ValueError, match="Invalid alpha"):
            Gaussian(
                x=100.0, y=50.0, rx=10.0, ry=8.0, theta=0.5, r=255, g=128, b=64, a=-0.1
            )

    def test_gaussian_area_calculation(self):
        """Test ellipse area calculation."""
        splat = Gaussian(
            x=100.0, y=50.0, rx=10.0, ry=8.0, theta=0.5, r=255, g=128, b=64, a=0.8
        )

        expected_area = np.pi * 10.0 * 8.0
        assert abs(splat.area() - expected_area) < 1e-6

    def test_gaussian_serialization(self):
        """Test to_dict and from_dict methods."""
        original = Gaussian(
            x=100.0,
            y=50.0,
            rx=10.0,
            ry=8.0,
            theta=0.5,
            r=255,
            g=128,
            b=64,
            a=0.8,
            score=0.9,
            depth=0.3,
        )

        # Test serialization
        data = original.to_dict()
        expected_data = {
            "position": (100.0, 50.0),
            "size": (10.0, 8.0),
            "rotation": 0.5,
            "color": (255, 128, 64, 0.8),
            "score": 0.9,
            "depth": 0.3,
        }
        assert data == expected_data

        # Test deserialization
        reconstructed = Gaussian.from_dict(data)
        assert reconstructed.x == original.x
        assert reconstructed.y == original.y
        assert reconstructed.rx == original.rx
        assert reconstructed.ry == original.ry
        assert reconstructed.theta == original.theta
        assert reconstructed.r == original.r
        assert reconstructed.g == original.g
        assert reconstructed.b == original.b
        assert reconstructed.a == original.a
        assert reconstructed.score == original.score
        assert reconstructed.depth == original.depth

    def test_gaussian_from_dict_defaults(self):
        """Test from_dict with missing optional fields."""
        data = {
            "position": (100.0, 50.0),
            "size": (10.0, 8.0),
            "rotation": 0.5,
            "color": (255, 128, 64, 0.8),
        }

        gaussian = Gaussian.from_dict(data)
        assert gaussian.score == 0.0  # Default
        assert gaussian.depth == 0.5  # Default


class TestSplatExtractor:
    """Test the SplatExtractor class."""

    def test_extractor_initialization(self):
        """Test SplatExtractor initialization."""
        extractor = SplatExtractor()
        assert extractor.k == 2.5  # Default value
        assert extractor.base_alpha == 0.65  # Default value

        # Custom values
        custom_extractor = SplatExtractor(k=3.0, base_alpha=0.8)
        assert custom_extractor.k == 3.0
        assert custom_extractor.base_alpha == 0.8

    def create_test_image(self, width=200, height=150):
        """Create a simple test image for testing."""
        # Create a gradient image with some structure
        image = np.zeros((height, width, 3), dtype=np.uint8)

        # Add some color variation
        for y in range(height):
            for x in range(width):
                image[y, x] = [
                    int(255 * x / width),  # Red gradient
                    int(255 * y / height),  # Green gradient
                    int(128 + 127 * np.sin(x * 0.1)),  # Blue variation
                ]

        return image

    @patch("splat_this.core.extract.slic")
    @patch("splat_this.core.extract.rgb2lab")
    def test_extract_splats_basic(self, mock_rgb2lab, mock_slic):
        """Test basic splat extraction functionality."""
        # Setup mocks
        test_image = self.create_test_image(100, 100)
        mock_rgb2lab.return_value = test_image  # Simplified for testing

        # Create a simple segmentation with 3 segments
        segments = np.zeros((100, 100), dtype=int)
        segments[20:40, 20:40] = 1  # Square segment 1
        segments[60:80, 60:80] = 2  # Square segment 2
        mock_slic.return_value = segments

        extractor = SplatExtractor()
        splats = extractor.extract_splats(test_image, n_splats=3)

        # Verify SLIC was called with correct parameters
        mock_slic.assert_called_once()
        call_args = mock_slic.call_args
        assert call_args[1]["n_segments"] == 3
        assert call_args[1]["compactness"] == 10.0
        assert call_args[1]["sigma"] == 1.0

        # Should get 2 splats (segments 1 and 2, segment 0 is background)
        assert len(splats) == 2
        assert all(isinstance(splat, Gaussian) for splat in splats)

    @patch("splat_this.core.extract.safe_eigendecomposition")
    def test_segment_to_gaussian_valid(self, mock_eigen):
        """Test converting a valid segment to Gaussian."""
        # Setup mock eigendecomposition
        eigenvalues = np.array([25.0, 16.0])  # Will give rx=12.5, ry=10.0 with k=2.5
        eigenvectors = np.array([[1.0, 0.0], [0.0, 1.0]])  # No rotation
        mock_eigen.return_value = (eigenvalues, eigenvectors)

        # Create test image and mask
        test_image = self.create_test_image(100, 100)
        mask = np.zeros((100, 100), dtype=bool)
        mask[40:60, 30:70] = True  # Rectangular region

        extractor = SplatExtractor()
        splat = extractor._segment_to_gaussian(test_image, mask, segment_id=1)

        assert splat is not None
        assert isinstance(splat, Gaussian)

        # Check that centroid is roughly in the middle of the mask
        assert 45 < splat.x < 55  # Around x=50 (middle of 30:70 range)
        assert 45 < splat.y < 55  # Around y=50 (middle of 40:60 range)

        # Check that radii are reasonable
        assert splat.rx > 0
        assert splat.ry > 0

        # Check that color values are in valid range
        assert 0 <= splat.r <= 255
        assert 0 <= splat.g <= 255
        assert 0 <= splat.b <= 255
        assert 0.0 <= splat.a <= 1.0

    def test_segment_to_gaussian_empty_mask(self):
        """Test segment_to_gaussian with empty mask."""
        test_image = self.create_test_image(100, 100)
        empty_mask = np.zeros((100, 100), dtype=bool)

        extractor = SplatExtractor()
        splat = extractor._segment_to_gaussian(test_image, empty_mask, segment_id=1)

        assert splat is None

    def test_segment_to_gaussian_single_pixel(self):
        """Test segment_to_gaussian with single pixel mask."""
        test_image = self.create_test_image(100, 100)
        single_pixel_mask = np.zeros((100, 100), dtype=bool)
        single_pixel_mask[50, 50] = True

        extractor = SplatExtractor()
        splat = extractor._segment_to_gaussian(
            test_image, single_pixel_mask, segment_id=1
        )

        assert splat is None  # Should return None for insufficient coordinates

    @patch("splat_this.core.extract.safe_eigendecomposition")
    def test_segment_to_gaussian_failed_eigen(self, mock_eigen):
        """Test segment_to_gaussian when eigendecomposition fails."""
        mock_eigen.return_value = (None, None)  # Simulate failure

        test_image = self.create_test_image(100, 100)
        mask = np.zeros((100, 100), dtype=bool)
        mask[40:60, 30:70] = True

        extractor = SplatExtractor()
        splat = extractor._segment_to_gaussian(test_image, mask, segment_id=1)

        assert splat is None

    @patch("splat_this.core.extract.slic")
    @patch("splat_this.core.extract.rgb2lab")
    def test_extract_splats_filters_small_segments(self, mock_rgb2lab, mock_slic):
        """Test that small segments are filtered out."""
        test_image = self.create_test_image(100, 100)
        mock_rgb2lab.return_value = test_image

        # Create segmentation with one large and one small segment
        segments = np.zeros((100, 100), dtype=int)
        segments[20:80, 20:80] = 1  # Large segment (3600 pixels)
        segments[5:7, 5:7] = 2  # Small segment (4 pixels, should be filtered)
        mock_slic.return_value = segments

        extractor = SplatExtractor()
        splats = extractor.extract_splats(test_image, n_splats=10)

        # Should only get the large segment
        assert len(splats) == 1

    def test_extract_splats_real_image(self):
        """Test extraction with a real (synthetic) image structure."""
        # Create a more realistic test image with distinct regions
        test_image = np.zeros((200, 200, 3), dtype=np.uint8)

        # Add distinct colored regions
        test_image[50:100, 50:100] = [255, 0, 0]  # Red square
        test_image[120:170, 120:170] = [0, 255, 0]  # Green square
        test_image[50:100, 120:170] = [0, 0, 255]  # Blue square

        # Add some noise to make it more realistic
        noise = np.random.randint(-20, 21, test_image.shape, dtype=np.int16)
        test_image = np.clip(test_image.astype(np.int16) + noise, 0, 255).astype(
            np.uint8
        )

        extractor = SplatExtractor()
        splats = extractor.extract_splats(test_image, n_splats=20)

        # Should extract some splats
        assert len(splats) > 0
        assert len(splats) <= 20

        # All splats should be valid
        for splat in splats:
            assert isinstance(splat, Gaussian)
            assert splat.rx > 0
            assert splat.ry > 0
            assert 0 <= splat.r <= 255
            assert 0 <= splat.g <= 255
            assert 0 <= splat.b <= 255
            assert 0.0 <= splat.a <= 1.0


class TestSplatExtractionIntegration:
    """Integration tests for splat extraction."""

    def test_end_to_end_extraction(self):
        """Test complete extraction pipeline."""
        # Create a simple but realistic test image
        image = np.zeros((150, 200, 3), dtype=np.uint8)

        # Create some geometric shapes with different colors
        # Circle-like region
        y, x = np.ogrid[:150, :200]
        circle_mask = (x - 50) ** 2 + (y - 50) ** 2 <= 30**2
        image[circle_mask] = [255, 100, 100]

        # Rectangle region
        image[100:130, 150:180] = [100, 255, 100]

        # Gradient background
        for i in range(150):
            for j in range(200):
                if not circle_mask[i, j] and not (100 <= i < 130 and 150 <= j < 180):
                    image[i, j] = [i, j, 128]

        extractor = SplatExtractor(k=2.0, base_alpha=0.7)
        splats = extractor.extract_splats(image, n_splats=15)

        # Verify we got reasonable results
        assert len(splats) > 0
        assert len(splats) <= 15

        # Check that splats have reasonable properties
        for splat in splats:
            # Position should be within image bounds
            assert 0 <= splat.x < 200
            assert 0 <= splat.y < 150

            # Size should be reasonable
            assert 1.0 <= splat.rx <= 100.0
            assert 1.0 <= splat.ry <= 100.0

            # Color should be valid
            assert 0 <= splat.r <= 255
            assert 0 <= splat.g <= 255
            assert 0 <= splat.b <= 255

            # Alpha and score should be in range
            assert 0.0 <= splat.a <= 1.0
            assert 0.0 <= splat.score <= 1.0

        # Check that we can serialize all splats
        for splat in splats:
            data = splat.to_dict()
            reconstructed = Gaussian.from_dict(data)
            assert abs(reconstructed.x - splat.x) < 1e-6
            assert abs(reconstructed.y - splat.y) < 1e-6
