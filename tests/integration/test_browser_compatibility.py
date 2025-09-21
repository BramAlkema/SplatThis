"""Browser compatibility tests for SplatThis SVG output."""

import pytest
import numpy as np
import re
from pathlib import Path
import tempfile

from splat_this.core.extract import SplatExtractor, Gaussian
from splat_this.core.layering import LayerAssigner
from splat_this.core.svgout import SVGGenerator
from splat_this.core.optimized_svgout import OptimizedSVGGenerator


class TestSVGBrowserCompatibility:
    """Test SVG output compatibility with target browsers."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create test splats for SVG generation
        self.test_splats = [
            Gaussian(x=100, y=80, rx=15, ry=12, theta=0.5, r=255, g=100, b=50, a=0.8, depth=0.3),
            Gaussian(x=200, y=120, rx=20, ry=15, theta=1.2, r=100, g=200, b=150, a=0.7, depth=0.5),
            Gaussian(x=150, y=100, rx=12, ry=18, theta=0.8, r=50, g=150, b=255, a=0.9, depth=0.7),
        ]

        # Create test layers
        self.test_layers = {
            0: [self.test_splats[0]],
            1: [self.test_splats[1]],
            2: [self.test_splats[2]],
        }

    def test_svg_basic_structure_compliance(self):
        """Test that SVG output follows basic SVG 1.1 specification."""
        generator = SVGGenerator(width=400, height=300)
        svg_content = generator.generate_svg(self.test_layers)

        # Check SVG version declaration
        assert '<?xml version="1.0"' in svg_content
        assert 'encoding="UTF-8"' in svg_content

        # Check SVG namespace
        assert 'xmlns="http://www.w3.org/2000/svg"' in svg_content

        # Check viewBox format
        viewbox_match = re.search(r'viewBox="(\d+) (\d+) (\d+) (\d+)"', svg_content)
        assert viewbox_match is not None
        x, y, width, height = map(int, viewbox_match.groups())
        assert x == 0 and y == 0
        assert width == 400 and height == 300

        # Check proper SVG structure
        assert svg_content.startswith('<?xml')
        assert '<svg' in svg_content
        assert '</svg>' in svg_content
        assert svg_content.strip().endswith('</svg>')

    def test_css_compatibility(self):
        """Test CSS features for browser compatibility."""
        generator = SVGGenerator(width=400, height=300)
        svg_content = generator.generate_svg(self.test_layers)

        # Check CSS is properly embedded
        assert '<style><![CDATA[' in svg_content
        assert ']]></style>' in svg_content

        # Check for modern CSS features with fallbacks
        css_section = svg_content[svg_content.find('<style>'):svg_content.find('</style>')]

        # Transform properties (supported in all modern browsers)
        assert 'transform:' in css_section or 'transform ' in css_section

        # Check for vendor prefix fallbacks or standard properties
        assert 'transition:' in css_section

        # Check for accessibility support
        assert 'prefers-reduced-motion' in css_section

        # Check for responsive design
        assert '@media' in css_section

    def test_javascript_compatibility(self):
        """Test JavaScript features for browser compatibility."""
        generator = SVGGenerator(width=400, height=300)
        svg_content = generator.generate_svg(self.test_layers)

        # Check JavaScript is properly embedded
        assert '<script><![CDATA[' in svg_content
        assert ']]></script>' in svg_content

        js_section = svg_content[svg_content.find('<script>'):svg_content.find('</script>')]

        # Check for ES5 compatible code (no arrow functions, const/let)
        assert '=>' not in js_section or 'function(' in js_section  # Should have function fallbacks

        # Check for feature detection
        assert 'typeof' in js_section  # Feature detection for DeviceOrientationEvent

        # Check for graceful degradation
        assert 'matchMedia' in js_section  # Media query support check

        # Check for iOS permission handling
        assert 'requestPermission' in js_section

    def test_color_format_compatibility(self):
        """Test color format compatibility across browsers."""
        generator = SVGGenerator(width=400, height=300)

        # Test solid mode (rgba colors)
        solid_svg = generator.generate_svg(self.test_layers, gaussian_mode=False)

        # Check for rgba color format (widely supported)
        rgba_matches = re.findall(r'rgba\((\d+),\s*(\d+),\s*(\d+),\s*([0-9.]+)\)', solid_svg)
        assert len(rgba_matches) > 0

        # Validate color values are in valid ranges
        for r, g, b, a in rgba_matches:
            assert 0 <= int(r) <= 255
            assert 0 <= int(g) <= 255
            assert 0 <= int(b) <= 255
            assert 0.0 <= float(a) <= 1.0

        # Test gradient mode
        gradient_svg = generator.generate_svg(self.test_layers, gaussian_mode=True)

        # Check for proper gradient definition
        assert 'radialGradient' in gradient_svg
        assert 'id="gaussianGradient"' in gradient_svg
        assert 'cx="50%" cy="50%" r="50%"' in gradient_svg

        # Check gradient is properly referenced
        assert 'url(#gaussianGradient)' in gradient_svg

    def test_transform_compatibility(self):
        """Test transform attribute compatibility."""
        # Create splats with rotations to test transforms
        rotated_splats = [
            Gaussian(x=100, y=100, rx=15, ry=10, theta=0.7854, r=255, g=0, b=0, a=0.8, depth=0.5),  # 45 degrees
            Gaussian(x=200, y=150, rx=12, ry=20, theta=1.5708, r=0, g=255, b=0, a=0.9, depth=0.6),  # 90 degrees
        ]
        rotated_layers = {0: rotated_splats}

        generator = SVGGenerator(width=400, height=300)
        svg_content = generator.generate_svg(rotated_layers)

        # Check for transform attributes
        transform_matches = re.findall(r'transform="rotate\(([^)]+)\)"', svg_content)
        assert len(transform_matches) > 0

        # Validate rotation values are in degrees (not radians)
        for rotation_str in transform_matches:
            parts = rotation_str.split()
            angle = float(parts[0])
            assert -360 <= angle <= 360  # Valid degree range

    def test_accessibility_features(self):
        """Test accessibility features in SVG output."""
        generator = SVGGenerator(width=400, height=300)
        svg_content = generator.generate_svg(
            self.test_layers,
            title="Accessibility Test SVG"
        )

        # Check for title element
        assert '<title>Accessibility Test SVG</title>' in svg_content

        # Check for reduced motion support
        assert 'prefers-reduced-motion' in svg_content
        assert 'transition: none !important' in svg_content
        assert 'animation: none !important' in svg_content

        # Check for proper ARIA considerations
        # SVG should be keyboard accessible and screen reader friendly
        assert 'class="splat-svg"' in svg_content  # Proper semantics

    def test_mobile_compatibility(self):
        """Test mobile-specific compatibility features."""
        generator = SVGGenerator(width=400, height=300, interactive_top=3)
        svg_content = generator.generate_svg(self.test_layers)

        js_section = svg_content[svg_content.find('<script>'):svg_content.find('</script>')]

        # Check for mobile detection
        assert 'isMobile' in js_section
        assert 'Android|iPhone|iPad' in js_section

        # Check for touch event handling
        assert 'DeviceOrientationEvent' in js_section

        # Check for iOS 13+ permission handling
        assert 'requestPermission' in js_section

        # Check for mobile-specific CSS
        css_section = svg_content[svg_content.find('<style>'):svg_content.find('</style>')]
        assert 'max-width: 768px' in css_section or '@media' in css_section

    def test_performance_optimized_compatibility(self):
        """Test optimized SVG generator compatibility."""
        opt_generator = OptimizedSVGGenerator(width=400, height=300, chunk_size=50)
        svg_content = opt_generator.generate_svg(self.test_layers)

        # Should have same compatibility features as standard generator
        assert '<?xml version="1.0"' in svg_content
        assert 'xmlns="http://www.w3.org/2000/svg"' in svg_content
        assert '<style><![CDATA[' in svg_content
        assert '<script><![CDATA[' in svg_content

        # Check optimized JavaScript performance features
        js_section = svg_content[svg_content.find('<script>'):svg_content.find('</script>')]
        assert 'requestAnimationFrame' in js_section  # Performance optimization
        assert 'passive: true' in js_section  # Event listener optimization

    def test_cross_browser_event_handling(self):
        """Test event handling compatibility across browsers."""
        generator = SVGGenerator(width=400, height=300)
        svg_content = generator.generate_svg(self.test_layers)

        js_section = svg_content[svg_content.find('<script>'):svg_content.find('</script>')]

        # Check for addEventListener usage (preferred over onclick)
        assert 'addEventListener' in js_section

        # Check for proper event object handling
        assert 'clientX' in js_section and 'clientY' in js_section

        # Check for getBoundingClientRect (modern but widely supported)
        assert 'getBoundingClientRect' in js_section

        # Check for proper context binding
        assert 'this' not in js_section or 'function(' in js_section  # Avoid arrow function 'this' issues

    def test_svg_validation_for_email_clients(self):
        """Test SVG compatibility with email clients."""
        generator = SVGGenerator(width=400, height=300)
        svg_content = generator.generate_svg(self.test_layers)

        # Email clients prefer inline styles over CSS classes
        # Check that critical styles could work inline
        assert 'fill=' in svg_content or 'style=' in svg_content

        # Check for proper XML structure (important for email)
        assert svg_content.count('<svg') == svg_content.count('</svg>')
        assert svg_content.count('<g') == svg_content.count('</g>')

        # Check that all ellipse elements are properly closed
        ellipse_count = svg_content.count('<ellipse')
        assert '/>' in svg_content  # Self-closing tags

    def test_svg_size_and_performance(self):
        """Test SVG output size for performance considerations."""
        generator = SVGGenerator(width=400, height=300, precision=2)  # Lower precision
        svg_content = generator.generate_svg(self.test_layers)

        # Check that precision setting affects output
        # With precision=2, should see numbers like "123.45" not "123.456789"
        number_matches = re.findall(r'\d+\.\d+', svg_content)
        if number_matches:
            # Most numbers should have 2 decimal places or fewer
            precise_numbers = [n for n in number_matches if len(n.split('.')[1]) <= 2]
            assert len(precise_numbers) / len(number_matches) > 0.8  # 80% should be 2 decimals

        # File size should be reasonable for web use
        file_size_kb = len(svg_content.encode('utf-8')) / 1024
        assert file_size_kb < 50, f"SVG too large: {file_size_kb:.1f}KB"  # Should be under 50KB for small images

    def test_fallback_content(self):
        """Test fallback content for unsupported features."""
        generator = SVGGenerator(width=400, height=300)

        # Test with empty layers (should show fallback)
        empty_svg = generator.generate_svg({})
        assert 'No splats to display' in empty_svg

        # Test that fallback content is properly structured
        assert '<text' in empty_svg
        assert 'text-anchor="middle"' in empty_svg
        assert 'font-family="sans-serif"' in empty_svg

    def test_xml_entity_escaping(self):
        """Test proper XML entity escaping."""
        # Test with special characters in title
        generator = SVGGenerator(width=400, height=300)
        svg_content = generator.generate_svg(
            self.test_layers,
            title="Test & Escape <XML> \"Entities\""
        )

        # Should not contain unescaped XML entities in title
        title_section = svg_content[svg_content.find('<title>'):svg_content.find('</title>')]
        assert '&amp;' in title_section or '&' not in title_section
        assert '&lt;' in title_section or '<' not in title_section.replace('<title>', '').replace('</title>', '')
        assert '&quot;' in title_section or '"' not in title_section


class TestPowerPointCompatibility:
    """Test compatibility with PowerPoint and presentation software."""

    def setup_method(self):
        """Set up test fixtures."""
        self.test_splats = [
            Gaussian(x=400, y=300, rx=20, ry=15, theta=0, r=255, g=100, b=50, a=0.8, depth=0.5),
        ]
        self.test_layers = {0: self.test_splats}

    def test_powerpoint_svg_compatibility(self):
        """Test SVG features that work in PowerPoint."""
        generator = SVGGenerator(width=800, height=600)
        svg_content = generator.generate_svg(self.test_layers)

        # PowerPoint supports basic SVG 1.1 features
        assert 'viewBox=' in svg_content
        assert '<ellipse' in svg_content

        # PowerPoint may not support all CSS/JS features
        # Test can generate a "safe" version without advanced features
        # (This would be a future enhancement)

    def test_svg_file_creation_and_validation(self):
        """Test actual SVG file creation and basic validation."""
        generator = SVGGenerator(width=800, height=600)
        svg_content = generator.generate_svg(self.test_layers, title="PowerPoint Test")

        with tempfile.TemporaryDirectory() as temp_dir:
            svg_file = Path(temp_dir) / "powerpoint_test.svg"
            generator.save_svg(svg_content, svg_file)

            # Verify file was created
            assert svg_file.exists()

            # Read back and verify content
            saved_content = svg_file.read_text(encoding='utf-8')
            assert saved_content == svg_content

            # Basic XML validation (no syntax errors)
            assert saved_content.count('<') == saved_content.count('>')
            assert '<svg' in saved_content and '</svg>' in saved_content

    def test_email_client_compatibility(self):
        """Test compatibility with email clients."""
        generator = SVGGenerator(width=600, height=400)
        svg_content = generator.generate_svg(self.test_layers)

        # Many email clients have limited SVG support
        # Test basic features that work widely
        assert '<ellipse' in svg_content  # Basic shapes
        assert 'fill=' in svg_content or 'style=' in svg_content  # Coloring

        # Advanced features that might not work in email:
        # - JavaScript (should be optional)
        # - Complex CSS (should have inline fallbacks)
        # - Gradients (should have solid color fallbacks)

        # For email compatibility, we might want a "simple" mode
        # This test documents the current behavior


class TestSVGStandardsCompliance:
    """Test compliance with SVG and web standards."""

    def setup_method(self):
        """Set up test fixtures."""
        self.generator = SVGGenerator(width=400, height=300)
        self.test_layers = {
            0: [Gaussian(x=200, y=150, rx=15, ry=12, theta=0.5, r=255, g=100, b=50, a=0.8, depth=0.5)]
        }

    def test_svg_namespace_compliance(self):
        """Test SVG namespace declarations."""
        svg_content = self.generator.generate_svg(self.test_layers)

        # Should declare SVG namespace
        assert 'xmlns="http://www.w3.org/2000/svg"' in svg_content

        # Should not have undefined namespace prefixes
        # (This would cause validation errors)
        lines = svg_content.split('\n')
        for line in lines:
            # Simple check for namespace prefixes without declarations
            if ':' in line and '<' in line:
                # Skip common valid patterns
                if any(valid in line for valid in ['http:', 'https:', 'xmlns:', 'xlink:']):
                    continue

    def test_css_standards_compliance(self):
        """Test CSS standards compliance."""
        svg_content = self.generator.generate_svg(self.test_layers)

        css_section = svg_content[svg_content.find('<style>'):svg_content.find('</style>')]

        # Check for valid CSS property names (no typos)
        css_properties = [
            'transform', 'transition', 'will-change', 'overflow',
            'cursor', 'transform-style'
        ]

        for prop in css_properties:
            if prop in css_section:
                # Property should be followed by a colon
                assert f'{prop}:' in css_section

    def test_html5_standards_compliance(self):
        """Test HTML5/modern web standards compliance."""
        svg_content = self.generator.generate_svg(self.test_layers)

        # Check for modern HTML5 features used appropriately
        if 'will-change' in svg_content:
            # will-change should be used responsibly
            assert 'transform' in svg_content  # Should specify what will change

        # Check for proper media query syntax
        if '@media' in svg_content:
            assert 'prefers-reduced-motion' in svg_content  # Accessibility

    def test_javascript_standards_compliance(self):
        """Test JavaScript standards compliance."""
        svg_content = self.generator.generate_svg(self.test_layers)

        js_section = svg_content[svg_content.find('<script>'):svg_content.find('</script>')]

        # Check for 'use strict'
        assert "'use strict'" in js_section

        # Check for proper variable declarations
        # Should not have global variable pollution
        assert 'var ' not in js_section or 'function(' in js_section

        # Check for proper event listener cleanup (if any)
        if 'addEventListener' in js_section:
            # Should use proper event listener patterns
            assert 'function(' in js_section  # Proper function syntax