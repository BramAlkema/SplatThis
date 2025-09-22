"""Unit tests for SVG generation functionality."""

import pytest
import math
import tempfile
from pathlib import Path
from typing import Dict, List
from unittest.mock import Mock, patch

from splat_this.core.extract import Gaussian
from splat_this.core.svgout import SVGGenerator


class TestSVGGenerator:

    def test_init_default_parameters(self):
        """Test SVGGenerator initialization with default parameters."""
        generator = SVGGenerator(width=800, height=600)
        assert generator.width == 800
        assert generator.height == 600
        assert generator.precision == 3
        assert generator.parallax_strength == 40
        assert generator.interactive_top == 0

    def test_init_custom_parameters(self):
        """Test SVGGenerator initialization with custom parameters."""
        generator = SVGGenerator(
            width=1920,
            height=1080,
            precision=2,
            parallax_strength=60,
            interactive_top=5
        )
        assert generator.width == 1920
        assert generator.height == 1080
        assert generator.precision == 2
        assert generator.parallax_strength == 60
        assert generator.interactive_top == 5

    def test_format_number_precision(self):
        """Test number formatting with specified precision."""
        generator = SVGGenerator(width=800, height=600, precision=2)
        assert generator._format_number(3.14159) == "3.14"
        assert generator._format_number(10.0) == "10.00"
        assert generator._format_number(0.123456) == "0.12"

        generator_high_precision = SVGGenerator(width=800, height=600, precision=4)
        assert generator_high_precision._format_number(3.14159) == "3.1416"

    def test_generate_empty_svg(self):
        """Test generation of empty SVG when no layers provided."""
        generator = SVGGenerator(width=800, height=600)
        result = generator.generate_svg({})

        assert '<?xml version="1.0"' in result
        assert 'viewBox="0 0 800 600"' in result
        assert 'xmlns="http://www.w3.org/2000/svg"' in result
        assert 'No splats to display' in result
        assert '</svg>' in result

    def test_generate_header_basic(self):
        """Test SVG header generation."""
        generator = SVGGenerator(width=1024, height=768, parallax_strength=50, interactive_top=3)
        header = generator._generate_header()

        assert '<?xml version="1.0" encoding="UTF-8"?>' in header
        assert 'viewBox="0 0 1024 768"' in header
        assert 'xmlns="http://www.w3.org/2000/svg"' in header
        assert 'data-parallax-strength="50"' in header
        assert 'data-interactive-top="3"' in header
        assert 'class="splat-svg"' in header

    def test_generate_header_with_title(self):
        """Test SVG header generation with title."""
        generator = SVGGenerator(width=800, height=600)
        header = generator._generate_header(title="Test SVG")

        assert '<title>Test SVG</title>' in header

    def test_generate_header_escapes_title(self):
        """Test SVG header escapes special characters in title."""
        generator = SVGGenerator(width=800, height=600)
        header = generator._generate_header(title="<Test & Title>")

        assert '&lt;Test &amp; Title&gt;' in header

    def test_generate_defs_solid_mode(self):
        """Test definitions generation for solid mode."""
        generator = SVGGenerator(width=800, height=600)
        defs = generator._generate_defs(gaussian_mode=False)

        assert '<defs></defs>' in defs

    def test_generate_defs_gaussian_mode(self):
        """Test definitions generation for gaussian mode."""
        generator = SVGGenerator(width=800, height=600)

        # First register a gradient by creating a splat
        test_splat = Gaussian(x=100, y=100, rx=10, ry=10, theta=0, r=255, g=100, b=50, a=0.8)
        gradient_id = generator._register_gradient(test_splat)

        # Now generate defs with the registered gradient
        defs = generator._generate_defs(gaussian_mode=True)

        assert '<defs>' in defs
        assert f'radialGradient id="{gradient_id}"' in defs
        assert 'stop offset="0%"' in defs
        assert 'stop offset="70%"' in defs
        assert 'stop offset="100%"' in defs

    def test_generate_splat_element_solid_mode(self):
        """Test splat element generation in solid mode."""
        generator = SVGGenerator(width=800, height=600, precision=2)
        splat = Gaussian(
            x=100.123, y=200.456,
            rx=10.789, ry=15.321,
            theta=math.radians(45),
            r=255, g=128, b=64,
            a=0.8
        )

        element = generator._generate_splat_element(splat, gaussian_mode=False)

        assert 'cx="100.12"' in element
        assert 'cy="200.46"' in element
        assert 'rx="10.79"' in element
        assert 'ry="15.32"' in element
        assert 'fill: rgba(255, 128, 64, 0.80)' in element
        assert 'rotate(45.00 100.12 200.46)' in element

    def test_generate_splat_element_gaussian_mode(self):
        """Test splat element generation in gaussian mode."""
        generator = SVGGenerator(width=800, height=600)
        splat = Gaussian(
            x=50, y=75,
            rx=20, ry=30,
            theta=0,
            r=200, g=100, b=50,
            a=0.9
        )

        element = generator._generate_splat_element(splat, gaussian_mode=True)

        assert 'fill: url(#gaussian-gradient-' in element
        assert 'fill-opacity: 0.900' in element
        # Check basic structure
        assert 'cx="50.000"' in element
        assert 'cy="75.000"' in element

    def test_generate_splat_element_no_rotation(self):
        """Test splat element generation without rotation."""
        generator = SVGGenerator(width=800, height=600)
        splat = Gaussian(
            x=100, y=200,
            rx=10, ry=15,
            theta=0,  # No rotation
            r=255, g=0, b=0,
            a=1.0
        )

        element = generator._generate_splat_element(splat, gaussian_mode=False)

        assert 'transform=' not in element  # No rotation transform

    def test_generate_layer_groups_basic(self):
        """Test layer groups generation."""
        generator = SVGGenerator(width=800, height=600)

        splats_layer0 = [
            Gaussian(x=100, y=100, rx=10, ry=10, theta=0, r=255, g=0, b=0, a=1.0, depth=0.2),
            Gaussian(x=200, y=200, rx=15, ry=15, theta=0, r=0, g=255, b=0, a=0.8, depth=0.2),
        ]

        splats_layer1 = [
            Gaussian(x=150, y=150, rx=12, ry=12, theta=0, r=0, g=0, b=255, a=0.9, depth=0.8),
        ]

        layers = {0: splats_layer0, 1: splats_layer1}

        layer_groups = generator._generate_layer_groups(layers, gaussian_mode=False)

        # Check layer structure
        assert 'data-depth="0.200"' in layer_groups
        assert 'data-depth="0.800"' in layer_groups
        assert 'data-layer="0"' in layer_groups
        assert 'data-layer="1"' in layer_groups

        # Check splat elements are included
        assert 'cx="100.000"' in layer_groups  # First splat
        assert 'cx="200.000"' in layer_groups  # Second splat
        assert 'cx="150.000"' in layer_groups  # Third splat

    def test_generate_layer_groups_empty_layer(self):
        """Test layer groups generation with empty layers."""
        generator = SVGGenerator(width=800, height=600)

        layers = {0: [], 1: [Gaussian(x=100, y=100, rx=10, ry=10, theta=0, r=255, g=0, b=0, a=1.0)]}

        layer_groups = generator._generate_layer_groups(layers, gaussian_mode=False)

        # Empty layer should not appear in output
        assert 'data-layer="0"' not in layer_groups
        assert 'data-layer="1"' in layer_groups

    def test_generate_styles(self):
        """Test CSS styles generation."""
        generator = SVGGenerator(width=800, height=600)
        styles = generator._generate_styles()

        assert '.splat-svg' in styles
        assert '.layer' in styles
        assert '.interactive-splat' in styles
        assert 'prefers-reduced-motion: reduce' in styles
        assert 'max-width: 768px' in styles
        assert 'transform-style: preserve-3d' in styles

    def test_generate_scripts(self):
        """Test JavaScript generation."""
        generator = SVGGenerator(width=800, height=600, parallax_strength=50, interactive_top=3)
        scripts = generator._generate_scripts()

        assert 'parallaxStrength = 50' in scripts
        assert 'interactiveTop = 3' in scripts
        assert 'prefers-reduced-motion' in scripts
        assert 'deviceorientation' in scripts
        assert 'mousemove' in scripts
        assert 'updateLayerTransforms' in scripts

    def test_validate_svg_structure_valid(self):
        """Test SVG structure validation with valid SVG."""
        generator = SVGGenerator(width=800, height=600)

        valid_svg = '''<?xml version="1.0" encoding="UTF-8"?>
<svg viewBox="0 0 800 600" xmlns="http://www.w3.org/2000/svg">
    <g class="layer"></g>
</svg>'''

        assert generator._validate_svg_structure(valid_svg)

    def test_validate_svg_structure_invalid(self):
        """Test SVG structure validation with invalid SVG."""
        generator = SVGGenerator(width=800, height=600)

        # Missing closing tag
        invalid_svg = '''<?xml version="1.0" encoding="UTF-8"?>
<svg viewBox="0 0 800 600" xmlns="http://www.w3.org/2000/svg">
    <g class="layer"></g>'''

        assert not generator._validate_svg_structure(invalid_svg)

    def test_validate_svg_structure_missing_elements(self):
        """Test SVG structure validation with missing required elements."""
        generator = SVGGenerator(width=800, height=600)

        # Missing viewBox
        invalid_svg = '''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg">
</svg>'''

        assert not generator._validate_svg_structure(invalid_svg)

    def test_generate_svg_complete_workflow(self):
        """Test complete SVG generation workflow."""
        generator = SVGGenerator(width=800, height=600, precision=2, parallax_strength=30)

        splats = [
            Gaussian(x=100, y=100, rx=20, ry=15, theta=math.radians(30),
                    r=255, g=100, b=50, a=0.8, depth=0.3),
            Gaussian(x=200, y=150, rx=25, ry=20, theta=0,
                    r=50, g=200, b=100, a=0.9, depth=0.7),
        ]

        layers = {0: [splats[0]], 1: [splats[1]]}

        svg_content = generator.generate_svg(layers, gaussian_mode=False, title="Test SVG")

        # Check overall structure
        assert '<?xml version="1.0"' in svg_content
        assert '<title>Test SVG</title>' in svg_content
        assert 'viewBox="0 0 800 600"' in svg_content
        assert 'data-parallax-strength="30"' in svg_content

        # Check layers
        assert 'data-depth="0.30"' in svg_content
        assert 'data-depth="0.70"' in svg_content

        # Check splats
        assert 'cx="100.00"' in svg_content
        assert 'cy="100.00"' in svg_content
        assert 'rx="20.00"' in svg_content
        assert 'rotate(30.00' in svg_content

        # Check styles and scripts
        assert '<style>' in svg_content
        assert '<script>' in svg_content
        assert 'parallaxStrength = 30' in svg_content

        # Validate structure
        assert generator._validate_svg_structure(svg_content)

    def test_generate_svg_gaussian_mode(self):
        """Test SVG generation in gaussian mode."""
        generator = SVGGenerator(width=800, height=600)

        splats = [
            Gaussian(x=100, y=100, rx=20, ry=20, theta=0,
                    r=255, g=0, b=0, a=1.0, depth=0.5),
        ]

        layers = {0: splats}

        svg_content = generator.generate_svg(layers, gaussian_mode=True)

        # Check gaussian-specific elements
        assert 'radialGradient id="gaussian-gradient-' in svg_content
        assert 'url(#gaussian-gradient-' in svg_content
        # Note: data-color is not generated by the old SVGGenerator, check other attributes
        assert 'cx="100.000"' in svg_content

    def test_get_svg_info(self):
        """Test SVG info generation."""
        generator = SVGGenerator(width=1920, height=1080, precision=2,
                                parallax_strength=60, interactive_top=5)

        splats1 = [Gaussian(x=100, y=100, rx=10, ry=10, theta=0, r=255, g=0, b=0, a=1.0)] * 3
        splats2 = [Gaussian(x=200, y=200, rx=10, ry=10, theta=0, r=0, g=255, b=0, a=1.0)] * 2

        layers = {0: splats1, 1: splats2}

        info = generator.get_svg_info(layers)

        assert info['width'] == 1920
        assert info['height'] == 1080
        assert info['precision'] == 2
        assert info['layer_count'] == 2
        assert info['total_splats'] == 5
        assert info['parallax_strength'] == 60
        assert info['interactive_top'] == 5

    def test_save_svg_basic(self):
        """Test saving SVG to file."""
        generator = SVGGenerator(width=800, height=600)

        svg_content = '''<?xml version="1.0" encoding="UTF-8"?>
<svg viewBox="0 0 800 600" xmlns="http://www.w3.org/2000/svg">
</svg>'''

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test.svg"
            generator.save_svg(svg_content, output_path)

            assert output_path.exists()
            assert output_path.read_text(encoding="utf-8") == svg_content

    def test_save_svg_creates_directories(self):
        """Test that save_svg creates necessary directories."""
        generator = SVGGenerator(width=800, height=600)

        svg_content = '<svg></svg>'

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "nested" / "directory" / "test.svg"
            generator.save_svg(svg_content, output_path)

            assert output_path.exists()
            assert output_path.read_text(encoding="utf-8") == svg_content

    def test_save_svg_file_error(self):
        """Test save_svg error handling."""
        generator = SVGGenerator(width=800, height=600)

        # Try to save to an invalid path
        invalid_path = Path("/invalid/path/that/should/not/exist/test.svg")

        with pytest.raises(Exception):
            generator.save_svg("<svg></svg>", invalid_path)

    def test_edge_case_very_small_dimensions(self):
        """Test SVG generation with very small dimensions."""
        generator = SVGGenerator(width=1, height=1)

        splats = [
            Gaussian(x=0.5, y=0.5, rx=0.1, ry=0.1, theta=0,
                    r=255, g=255, b=255, a=1.0, depth=0.5),
        ]

        layers = {0: splats}

        svg_content = generator.generate_svg(layers)

        assert 'viewBox="0 0 1 1"' in svg_content
        assert generator._validate_svg_structure(svg_content)

    def test_edge_case_large_dimensions(self):
        """Test SVG generation with large dimensions."""
        generator = SVGGenerator(width=10000, height=8000)

        splats = [
            Gaussian(x=5000, y=4000, rx=100, ry=100, theta=0,
                    r=128, g=128, b=128, a=0.5, depth=0.5),
        ]

        layers = {0: splats}

        svg_content = generator.generate_svg(layers)

        assert 'viewBox="0 0 10000 8000"' in svg_content
        assert 'cx="5000.000"' in svg_content
        assert generator._validate_svg_structure(svg_content)

    def test_edge_case_many_layers(self):
        """Test SVG generation with many layers."""
        generator = SVGGenerator(width=800, height=600)

        layers = {}
        for i in range(20):  # 20 layers
            splat = Gaussian(x=i*10, y=i*10, rx=5, ry=5, theta=0,
                           r=255, g=128, b=64, a=0.5, depth=i/20.0)
            layers[i] = [splat]

        svg_content = generator.generate_svg(layers)

        # Check all layers are present
        for i in range(20):
            assert f'data-layer="{i}"' in svg_content

        assert generator._validate_svg_structure(svg_content)

    def test_edge_case_zero_precision(self):
        """Test SVG generation with zero precision."""
        generator = SVGGenerator(width=800, height=600, precision=0)

        splat = Gaussian(x=123.456, y=789.123, rx=10.789, ry=15.321, theta=0,
                        r=255, g=128, b=64, a=0.567)

        layers = {0: [splat]}

        svg_content = generator.generate_svg(layers)

        # Numbers should be rounded to integers
        assert 'cx="123"' in svg_content
        assert 'cy="789"' in svg_content
        assert 'rx="11"' in svg_content

    def test_edge_case_high_precision(self):
        """Test SVG generation with high precision."""
        generator = SVGGenerator(width=800, height=600, precision=6)

        splat = Gaussian(x=123.123456789, y=456.987654321, rx=10, ry=10, theta=0,
                        r=255, g=0, b=0, a=1.0)

        layers = {0: [splat]}

        svg_content = generator.generate_svg(layers)

        # Should format with 6 decimal places
        assert 'cx="123.123457"' in svg_content  # Rounded to 6 decimals
        assert 'cy="456.987654"' in svg_content

    def test_rotation_angle_conversion(self):
        """Test conversion of rotation angles from radians to degrees."""
        generator = SVGGenerator(width=800, height=600)

        # Test various angles
        test_cases = [
            (0, "0.000"),
            (math.pi/4, "45.000"),  # 45 degrees
            (math.pi/2, "90.000"),  # 90 degrees
            (math.pi, "180.000"),   # 180 degrees
            (2*math.pi, "360.000"), # 360 degrees
        ]

        for radians, expected_deg in test_cases:
            splat = Gaussian(x=100, y=100, rx=10, ry=10, theta=radians,
                           r=255, g=0, b=0, a=1.0)

            element = generator._generate_splat_element(splat, gaussian_mode=False)

            if abs(radians) > 1e-6:  # Only expect transform for non-zero angles
                assert f'rotate({expected_deg}' in element
            else:
                assert 'transform=' not in element

    def test_color_and_alpha_formatting(self):
        """Test proper formatting of colors and alpha values."""
        generator = SVGGenerator(width=800, height=600, precision=2)

        splat = Gaussian(x=100, y=100, rx=10, ry=10, theta=0,
                        r=255, g=128, b=64, a=0.123456)

        element = generator._generate_splat_element(splat, gaussian_mode=False)

        assert 'rgba(255, 128, 64, 0.12)' in element  # Alpha rounded to precision

    def test_layer_sorting_by_depth(self):
        """Test that layers are sorted correctly by depth."""
        generator = SVGGenerator(width=800, height=600)

        # Create layers with specific depths (not in order)
        splat1 = Gaussian(x=100, y=100, rx=10, ry=10, theta=0,
                         r=255, g=0, b=0, a=1.0, depth=0.8)  # Front
        splat2 = Gaussian(x=200, y=200, rx=10, ry=10, theta=0,
                         r=0, g=255, b=0, a=1.0, depth=0.2)  # Back
        splat3 = Gaussian(x=300, y=300, rx=10, ry=10, theta=0,
                         r=0, g=0, b=255, a=1.0, depth=0.5)  # Middle

        layers = {2: [splat1], 0: [splat2], 1: [splat3]}  # Unordered keys

        layer_groups = generator._generate_layer_groups(layers, gaussian_mode=False)

        # Find positions of layers in the output
        layer0_pos = layer_groups.find('data-layer="0"')
        layer1_pos = layer_groups.find('data-layer="1"')
        layer2_pos = layer_groups.find('data-layer="2"')

        # Should be ordered by layer index (which corresponds to depth)
        assert layer0_pos < layer1_pos < layer2_pos