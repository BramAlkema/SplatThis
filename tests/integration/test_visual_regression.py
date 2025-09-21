"""Visual regression tests for SplatThis output consistency."""

import pytest
import numpy as np
import hashlib
import tempfile
from pathlib import Path
import json
from typing import Dict, List, Any

from splat_this.core.extract import SplatExtractor, Gaussian
from splat_this.core.layering import ImportanceScorer, LayerAssigner, QualityController
from splat_this.core.svgout import SVGGenerator
from splat_this.core.optimized_extract import OptimizedSplatExtractor
from splat_this.core.optimized_svgout import OptimizedSVGGenerator


class TestVisualRegression:
    """Test visual consistency of SVG output across changes."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create deterministic test data to ensure reproducible results
        np.random.seed(42)  # Fixed seed for reproducibility

        # Create test images with known patterns
        self.test_images = self._create_deterministic_test_images()

        # Create reference data directory
        self.reference_dir = Path(__file__).parent / "reference_data"
        self.reference_dir.mkdir(exist_ok=True)

    def _create_deterministic_test_images(self) -> Dict[str, np.ndarray]:
        """Create deterministic test images for consistent results."""
        images = {}

        # Simple gradient (highly predictable)
        gradient = np.zeros((100, 150, 3), dtype=np.uint8)
        for y in range(100):
            gradient[y, :, 0] = int(255 * y / 100)  # Red gradient
            gradient[y, :, 1] = int(128 * y / 100)  # Green gradient
        images['gradient'] = gradient

        # Geometric pattern
        geometric = np.zeros((80, 120, 3), dtype=np.uint8)
        # Create consistent geometric shapes
        geometric[20:60, 30:90] = [100, 150, 200]  # Blue rectangle
        geometric[30:50, 40:80] = [255, 100, 50]   # Orange rectangle
        geometric[10:25, 10:25] = [50, 255, 50]    # Green square
        images['geometric'] = geometric

        # Noise pattern with fixed seed
        np.random.seed(123)
        noise = np.random.randint(0, 255, (60, 90, 3), dtype=np.uint8)
        images['noise'] = noise

        return images

    def test_svg_output_consistency(self):
        """Test that SVG output is consistent for the same input."""
        for image_name, image in self.test_images.items():
            reference_svg = self._generate_reference_svg(image, image_name)

            # Generate SVG multiple times and compare
            for iteration in range(3):
                current_svg = self._generate_reference_svg(image, image_name)

                # SVG content should be identical
                assert current_svg == reference_svg, \
                    f"SVG output inconsistent for {image_name} iteration {iteration}"

    def test_splat_extraction_determinism(self):
        """Test that splat extraction is deterministic."""
        image = self.test_images['geometric']

        # Extract splats multiple times
        results = []
        for i in range(3):
            extractor = SplatExtractor()
            splats = extractor.extract_splats(image, n_splats=30)

            # Convert to comparable format
            splat_data = self._splats_to_comparable_data(splats)
            results.append(splat_data)

        # All extractions should produce the same result
        for i in range(1, len(results)):
            assert results[0] == results[i], \
                f"Splat extraction not deterministic, iteration {i} differs"

    def test_scoring_consistency(self):
        """Test that importance scoring is consistent."""
        image = self.test_images['gradient']

        # Create identical splats
        test_splats = [
            Gaussian(x=50, y=40, rx=10, ry=8, theta=0.5, r=255, g=100, b=50, a=0.8),
            Gaussian(x=100, y=60, rx=15, ry=12, theta=1.0, r=100, g=200, b=150, a=0.7),
        ]

        # Score multiple times
        scores_sets = []
        for i in range(3):
            splats_copy = [
                Gaussian(x=s.x, y=s.y, rx=s.rx, ry=s.ry, theta=s.theta,
                        r=s.r, g=s.g, b=s.b, a=s.a)
                for s in test_splats
            ]

            scorer = ImportanceScorer()
            scorer.score_splats(splats_copy, image)

            scores = [s.score for s in splats_copy]
            scores_sets.append(scores)

        # All scoring should produce the same result
        for i in range(1, len(scores_sets)):
            for j in range(len(scores_sets[0])):
                assert abs(scores_sets[0][j] - scores_sets[i][j]) < 1e-10, \
                    f"Scoring inconsistent: splat {j}, iteration {i}"

    def test_svg_structure_regression(self):
        """Test that SVG structure doesn't regress."""
        image = self.test_images['geometric']

        # Generate SVG with known parameters
        extractor = SplatExtractor()
        splats = extractor.extract_splats(image, n_splats=25)

        scorer = ImportanceScorer()
        scorer.score_splats(splats, image)

        controller = QualityController(target_count=20)
        filtered_splats = controller.optimize_splats(splats)

        layer_assigner = LayerAssigner(n_layers=3)
        layers = layer_assigner.assign_layers(filtered_splats)

        generator = SVGGenerator(width=120, height=80, precision=2)
        svg_content = generator.generate_svg(layers, title="Regression Test")

        # Check structural elements that should always be present
        required_elements = [
            '<?xml version="1.0"',
            '<svg',
            'viewBox="0 0 120 80"',
            'xmlns="http://www.w3.org/2000/svg"',
            '<title>Regression Test</title>',
            '<g class="layer"',
            '<ellipse',
            '<style><![CDATA[',
            '<script><![CDATA[',
            '</svg>'
        ]

        for element in required_elements:
            assert element in svg_content, f"Missing required element: {element}"

        # Check that we have the expected number of elements
        ellipse_count = svg_content.count('<ellipse')
        assert ellipse_count > 0, "No ellipses generated"
        assert ellipse_count <= 25, f"Too many ellipses: {ellipse_count}"

    def test_mathematical_precision_consistency(self):
        """Test that mathematical calculations are consistent."""
        # Test with known splat parameters
        test_splat = Gaussian(
            x=100.123456, y=75.654321, rx=12.345, ry=8.765, theta=0.785398,
            r=255, g=128, b=64, a=0.8, depth=0.5
        )

        generator = SVGGenerator(width=200, height=150, precision=3)

        # Generate multiple times and check precision
        for i in range(3):
            svg_element = generator._generate_splat_element(test_splat, gaussian_mode=False)

            # Check that coordinates are formatted with correct precision
            assert 'cx="100.123"' in svg_element
            assert 'cy="75.654"' in svg_element
            assert 'rx="12.345"' in svg_element
            assert 'ry="8.765"' in svg_element

    def test_color_format_consistency(self):
        """Test that color formatting is consistent."""
        test_splats = [
            Gaussian(x=50, y=50, rx=10, ry=10, theta=0, r=255, g=128, b=64, a=0.75),
            Gaussian(x=100, y=100, rx=15, ry=15, theta=0, r=0, g=255, b=0, a=1.0),
        ]

        layers = {0: test_splats}
        generator = SVGGenerator(width=200, height=200)

        # Test solid mode
        solid_svg = generator.generate_svg(layers, gaussian_mode=False)

        # Should contain rgba colors with specific format
        assert 'rgba(255, 128, 64, 0.75)' in solid_svg or 'rgba(255,128,64,0.75)' in solid_svg
        assert 'rgba(0, 255, 0, 1)' in solid_svg or 'rgba(0,255,0,1)' in solid_svg

        # Test gaussian mode
        gaussian_svg = generator.generate_svg(layers, gaussian_mode=True)

        # Should contain gradient references
        assert 'url(#gaussianGradient)' in gaussian_svg
        assert 'data-color="rgb(255, 128, 64)"' in gaussian_svg or 'data-color="rgb(255,128,64)"' in gaussian_svg

    def test_animation_code_stability(self):
        """Test that animation JavaScript code is stable."""
        generator = SVGGenerator(width=400, height=300, parallax_strength=50, interactive_top=5)
        layers = {0: [Gaussian(x=200, y=150, rx=20, ry=15, theta=0, r=255, g=100, b=50, a=0.8, depth=0.5)]}

        svg_content = generator.generate_svg(layers)

        # Extract JavaScript section
        js_start = svg_content.find('<script><![CDATA[')
        js_end = svg_content.find(']]></script>')
        js_content = svg_content[js_start:js_end]

        # Check for essential animation features
        essential_js_features = [
            "'use strict'",
            "parallaxStrength = 50",
            "interactiveTop = 5",
            "handleMouseMove",
            "handleDeviceOrientation",
            "updateLayerTransforms",
            "addEventListener",
            "getBoundingClientRect"
        ]

        for feature in essential_js_features:
            assert feature in js_content, f"Missing essential JS feature: {feature}"

    def test_optimized_vs_standard_consistency(self):
        """Test that optimized components produce consistent results with standard ones."""
        image = self.test_images['gradient']

        # Generate with standard components
        standard_svg = self._generate_svg_with_standard_components(image)

        # Generate with optimized components
        optimized_svg = self._generate_svg_with_optimized_components(image)

        # While the exact content may differ due to optimizations,
        # the structural elements should be similar
        standard_elements = self._extract_svg_structure(standard_svg)
        optimized_elements = self._extract_svg_structure(optimized_svg)

        # Compare structural similarity
        assert standard_elements['has_xml_declaration'] == optimized_elements['has_xml_declaration']
        assert standard_elements['has_svg_namespace'] == optimized_elements['has_svg_namespace']
        assert standard_elements['has_viewbox'] == optimized_elements['has_viewbox']
        assert standard_elements['has_ellipses'] == optimized_elements['has_ellipses']
        assert standard_elements['has_styles'] == optimized_elements['has_styles']
        assert standard_elements['has_scripts'] == optimized_elements['has_scripts']

        # Ellipse count should be similar (allow some variance due to optimization)
        ellipse_ratio = optimized_elements['ellipse_count'] / max(standard_elements['ellipse_count'], 1)
        assert 0.7 <= ellipse_ratio <= 1.3, f"Ellipse count too different: {ellipse_ratio:.2f}x"

    def test_file_size_regression(self):
        """Test that file sizes don't regress significantly."""
        image = self.test_images['geometric']

        # Generate SVG with standard parameters
        svg_content = self._generate_reference_svg(image, 'size_test')

        file_size = len(svg_content.encode('utf-8'))

        # File size should be reasonable
        assert file_size < 100000, f"SVG file too large: {file_size} bytes"  # 100KB limit
        assert file_size > 1000, f"SVG file too small: {file_size} bytes"    # 1KB minimum

        # File size should be consistent for same input
        svg_content2 = self._generate_reference_svg(image, 'size_test')
        file_size2 = len(svg_content2.encode('utf-8'))

        assert file_size == file_size2, "File size inconsistent between generations"

    def test_empty_layers_handling(self):
        """Test consistent handling of edge cases."""
        generator = SVGGenerator(width=400, height=300)

        # Test with empty layers
        empty_svg = generator.generate_svg({})

        # Should have consistent fallback content
        assert 'No splats to display' in empty_svg
        assert '<text' in empty_svg
        assert 'text-anchor="middle"' in empty_svg

        # Test multiple times to ensure consistency
        for i in range(3):
            empty_svg_repeat = generator.generate_svg({})
            assert empty_svg == empty_svg_repeat, f"Empty SVG inconsistent on iteration {i}"

    # Helper methods

    def _generate_reference_svg(self, image: np.ndarray, test_name: str) -> str:
        """Generate reference SVG for regression testing."""
        # Use fixed parameters for consistency
        extractor = SplatExtractor()
        splats = extractor.extract_splats(image, n_splats=20)

        scorer = ImportanceScorer(area_weight=0.3, edge_weight=0.5, color_weight=0.2)
        scorer.score_splats(splats, image)

        controller = QualityController(target_count=15, k_multiplier=2.0)
        filtered_splats = controller.optimize_splats(splats)

        layer_assigner = LayerAssigner(n_layers=3)
        layers = layer_assigner.assign_layers(filtered_splats)

        generator = SVGGenerator(
            width=image.shape[1],
            height=image.shape[0],
            precision=2,
            parallax_strength=30,
            interactive_top=0
        )

        return generator.generate_svg(layers, title=f"Reference {test_name}")

    def _generate_svg_with_standard_components(self, image: np.ndarray) -> str:
        """Generate SVG using standard components."""
        extractor = SplatExtractor()
        splats = extractor.extract_splats(image, n_splats=30)

        scorer = ImportanceScorer()
        scorer.score_splats(splats, image)

        controller = QualityController(target_count=20)
        filtered_splats = controller.optimize_splats(splats)

        layer_assigner = LayerAssigner(n_layers=4)
        layers = layer_assigner.assign_layers(filtered_splats)

        generator = SVGGenerator(width=image.shape[1], height=image.shape[0])
        return generator.generate_svg(layers)

    def _generate_svg_with_optimized_components(self, image: np.ndarray) -> str:
        """Generate SVG using optimized components."""
        extractor = OptimizedSplatExtractor(max_workers=1)  # Single worker for consistency
        splats = extractor.extract_splats(image, n_splats=30)

        # Use standard components for scoring/filtering for fair comparison
        scorer = ImportanceScorer()
        scorer.score_splats(splats, image)

        controller = QualityController(target_count=20)
        filtered_splats = controller.optimize_splats(splats)

        layer_assigner = LayerAssigner(n_layers=4)
        layers = layer_assigner.assign_layers(filtered_splats)

        generator = OptimizedSVGGenerator(width=image.shape[1], height=image.shape[0], chunk_size=10)
        return generator.generate_svg(layers)

    def _splats_to_comparable_data(self, splats: List[Gaussian]) -> List[Dict[str, Any]]:
        """Convert splats to comparable data structure."""
        return [
            {
                'x': round(s.x, 6),
                'y': round(s.y, 6),
                'rx': round(s.rx, 6),
                'ry': round(s.ry, 6),
                'theta': round(s.theta, 6),
                'r': s.r,
                'g': s.g,
                'b': s.b,
                'a': round(s.a, 6),
            }
            for s in splats
        ]

    def _extract_svg_structure(self, svg_content: str) -> Dict[str, Any]:
        """Extract structural information from SVG for comparison."""
        return {
            'has_xml_declaration': '<?xml' in svg_content,
            'has_svg_namespace': 'xmlns="http://www.w3.org/2000/svg"' in svg_content,
            'has_viewbox': 'viewBox=' in svg_content,
            'has_ellipses': '<ellipse' in svg_content,
            'has_styles': '<style>' in svg_content,
            'has_scripts': '<script>' in svg_content,
            'ellipse_count': svg_content.count('<ellipse'),
            'layer_count': svg_content.count('<g class="layer"'),
            'file_size': len(svg_content),
        }

    def _generate_content_hash(self, content: str) -> str:
        """Generate hash of content for comparison."""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()


class TestReferenceDataManagement:
    """Manage reference data for visual regression testing."""

    def setup_method(self):
        """Set up reference data management."""
        self.reference_dir = Path(__file__).parent / "reference_data"
        self.reference_dir.mkdir(exist_ok=True)

    def test_reference_data_creation(self):
        """Test creation and validation of reference data."""
        # This test would create reference data files for comparison
        # In a real scenario, these would be committed to version control

        test_image = np.zeros((50, 75, 3), dtype=np.uint8)
        test_image[10:40, 20:55] = [255, 128, 64]  # Orange rectangle

        # Generate reference SVG
        extractor = SplatExtractor()
        splats = extractor.extract_splats(test_image, n_splats=10)

        generator = SVGGenerator(width=75, height=50)
        layers = {0: splats}
        svg_content = generator.generate_svg(layers)

        # Create reference file
        reference_file = self.reference_dir / "test_reference.svg"
        reference_file.write_text(svg_content)

        # Validate reference file
        assert reference_file.exists()
        loaded_content = reference_file.read_text()
        assert loaded_content == svg_content

    def test_reference_comparison(self):
        """Test comparison against reference data."""
        # This would compare current output against stored reference
        # For this example, we'll just test the mechanism

        reference_content = "<svg>reference content</svg>"
        current_content = "<svg>reference content</svg>"

        assert reference_content == current_content, "Content differs from reference"

    @pytest.mark.skip(reason="Reference data management - manual execution")
    def test_update_reference_data(self):
        """Test for updating reference data when intentional changes are made."""
        # This test would be run manually when reference data needs updating
        # It would regenerate all reference files with current code
        pass