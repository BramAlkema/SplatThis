"""Basic tests to verify package structure."""

import pytest
import numpy as np

from splat_this.core.extract import Gaussian, SplatExtractor
from splat_this.core.layering import LayerAssigner
from splat_this.core.svgout import SVGGenerator
from splat_this.utils.image import validate_image_dimensions
from splat_this.utils.math import clamp_value, normalize_angle


class TestGaussianDataclass:
    """Test the Gaussian splat dataclass."""

    def test_gaussian_creation(self):
        """Test creating a valid Gaussian splat."""
        splat = Gaussian(
            x=10.0, y=20.0, rx=5.0, ry=3.0, theta=0.5, r=128, g=64, b=32, a=0.8
        )
        assert splat.x == 10.0
        assert splat.y == 20.0
        assert splat.area() > 0

    def test_gaussian_validation(self):
        """Test Gaussian parameter validation."""
        # Test invalid radii
        with pytest.raises(ValueError, match="Invalid radii"):
            Gaussian(x=0, y=0, rx=0, ry=1, theta=0, r=128, g=128, b=128, a=0.5)

        # Test invalid RGB
        with pytest.raises(ValueError, match="Invalid RGB"):
            Gaussian(x=0, y=0, rx=1, ry=1, theta=0, r=300, g=128, b=128, a=0.5)

        # Test invalid alpha
        with pytest.raises(ValueError, match="Invalid alpha"):
            Gaussian(x=0, y=0, rx=1, ry=1, theta=0, r=128, g=128, b=128, a=1.5)


class TestSplatExtractor:
    """Test the splat extraction system."""

    def test_extractor_creation(self):
        """Test creating a SplatExtractor."""
        extractor = SplatExtractor(k=2.5, base_alpha=0.65)
        assert extractor.k == 2.5
        assert extractor.base_alpha == 0.65

    def test_basic_extraction(self):
        """Test basic splat extraction with placeholder implementation."""
        extractor = SplatExtractor()
        # Create a simple test image
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        splats = extractor.extract_splats(test_image, n_splats=50)

        # Should return at least one splat (placeholder implementation)
        assert len(splats) >= 1
        assert all(isinstance(s, Gaussian) for s in splats)


class TestLayerAssignment:
    """Test the layer assignment system."""

    def test_layer_assigner_creation(self):
        """Test creating a LayerAssigner."""
        assigner = LayerAssigner(n_layers=4)
        assert assigner.n_layers == 4

    def test_basic_layer_assignment(self):
        """Test basic layer assignment with placeholder implementation."""
        # Create test splats
        splats = [
            Gaussian(
                x=i, y=i, rx=1, ry=1, theta=0, r=128, g=128, b=128, a=0.5, score=i / 10
            )
            for i in range(10)
        ]

        assigner = LayerAssigner(n_layers=4)
        layers = assigner.assign_layers(splats)

        # Should return a dictionary with layer data
        assert isinstance(layers, dict)
        assert len(layers) >= 1  # At least one layer should have splats


class TestSVGGenerator:
    """Test the SVG generation system."""

    def test_svg_generator_creation(self):
        """Test creating an SVGGenerator."""
        generator = SVGGenerator(width=1920, height=1080)
        assert generator.width == 1920
        assert generator.height == 1080

    def test_basic_svg_generation(self):
        """Test basic SVG generation with placeholder implementation."""
        generator = SVGGenerator(width=100, height=100)

        # Create test layers
        test_splat = Gaussian(
            x=50, y=50, rx=10, ry=5, theta=0, r=255, g=0, b=0, a=0.8, depth=0.5
        )
        layers = {0: [test_splat]}

        svg_content = generator.generate_svg(layers)

        # Should return valid SVG string
        assert isinstance(svg_content, str)
        assert "svg" in svg_content
        assert "viewBox" in svg_content


class TestUtils:
    """Test utility functions."""

    def test_image_dimension_validation(self):
        """Test image dimension validation."""
        # Valid image
        valid_image = np.zeros((500, 500, 3), dtype=np.uint8)
        validate_image_dimensions(valid_image)  # Should not raise

        # Too small
        small_image = np.zeros((50, 50, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="too small"):
            validate_image_dimensions(small_image)

    def test_math_utilities(self):
        """Test mathematical utility functions."""
        # Test clamp_value
        assert clamp_value(5.0, 0.0, 10.0) == 5.0
        assert clamp_value(-1.0, 0.0, 10.0) == 0.0
        assert clamp_value(15.0, 0.0, 10.0) == 10.0

        # Test normalize_angle
        angle = normalize_angle(4 * np.pi)
        assert -np.pi <= angle <= np.pi
