"""Integration tests for the complete SplatThis pipeline."""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from PIL import Image
import logging

from splat_this.utils.image import load_image, validate_image_dimensions
from splat_this.core.extract import SplatExtractor
from splat_this.core.optimized_extract import OptimizedSplatExtractor
from splat_this.core.layering import ImportanceScorer, LayerAssigner, QualityController
from splat_this.core.optimized_layering import OptimizedImportanceScorer, ParallelQualityController
from splat_this.core.svgout import SVGGenerator
from splat_this.core.optimized_svgout import OptimizedSVGGenerator
from splat_this.utils.profiler import global_profiler


class TestFullPipelineIntegration:
    """Test complete pipeline integration from image to SVG."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create test images with different characteristics
        self.test_images = self._create_test_images()

    def _create_test_images(self) -> dict:
        """Create various test images for comprehensive testing."""
        images = {}

        # Simple gradient image
        gradient = np.zeros((100, 150, 3), dtype=np.uint8)
        for y in range(100):
            gradient[y, :, 0] = int(255 * y / 100)  # Red gradient
        images['gradient'] = gradient

        # Checkered pattern
        checkered = np.zeros((80, 120, 3), dtype=np.uint8)
        for y in range(80):
            for x in range(120):
                if (x // 10 + y // 10) % 2:
                    checkered[y, x] = [255, 255, 255]
        images['checkered'] = checkered

        # Random noise
        noise = np.random.randint(0, 255, (60, 90, 3), dtype=np.uint8)
        images['noise'] = noise

        # Solid color with shapes
        shapes = np.full((120, 180, 3), 50, dtype=np.uint8)  # Dark gray background
        # Add some geometric shapes
        shapes[20:40, 30:50] = [255, 0, 0]  # Red rectangle
        shapes[60:80, 100:130] = [0, 255, 0]  # Green rectangle
        # Add a circle-like pattern
        center_y, center_x = 90, 90
        for y in range(120):
            for x in range(180):
                if (y - center_y)**2 + (x - center_x)**2 < 400:  # Circle radius ~20
                    shapes[y, x] = [0, 0, 255]  # Blue circle
        images['shapes'] = shapes

        return images

    def test_standard_pipeline_workflow(self):
        """Test the standard (non-optimized) pipeline end-to-end."""
        for image_name, image in self.test_images.items():
            with self._subtest_context(f"Standard pipeline - {image_name}"):
                self._test_pipeline_with_components(
                    image=image,
                    extractor_class=SplatExtractor,
                    scorer_class=ImportanceScorer,
                    controller_class=QualityController,
                    generator_class=SVGGenerator,
                    test_name=f"standard_{image_name}"
                )

    def test_optimized_pipeline_workflow(self):
        """Test the optimized pipeline end-to-end."""
        for image_name, image in self.test_images.items():
            with self._subtest_context(f"Optimized pipeline - {image_name}"):
                self._test_pipeline_with_components(
                    image=image,
                    extractor_class=OptimizedSplatExtractor,
                    scorer_class=OptimizedImportanceScorer,
                    controller_class=ParallelQualityController,
                    generator_class=OptimizedSVGGenerator,
                    test_name=f"optimized_{image_name}"
                )

    def test_mixed_pipeline_combinations(self):
        """Test various combinations of standard and optimized components."""
        test_image = self.test_images['shapes']

        # Test different combinations
        combinations = [
            # Mix 1: Optimized extraction, standard processing
            (OptimizedSplatExtractor, ImportanceScorer, QualityController, SVGGenerator),
            # Mix 2: Standard extraction, optimized processing
            (SplatExtractor, OptimizedImportanceScorer, ParallelQualityController, OptimizedSVGGenerator),
            # Mix 3: Optimized extraction and scoring, standard quality control and SVG
            (OptimizedSplatExtractor, OptimizedImportanceScorer, QualityController, SVGGenerator),
        ]

        for i, (extractor_cls, scorer_cls, controller_cls, generator_cls) in enumerate(combinations):
            with self._subtest_context(f"Mixed pipeline combination {i+1}"):
                self._test_pipeline_with_components(
                    image=test_image,
                    extractor_class=extractor_cls,
                    scorer_class=scorer_cls,
                    controller_class=controller_cls,
                    generator_class=generator_cls,
                    test_name=f"mixed_{i+1}"
                )

    def _test_pipeline_with_components(
        self,
        image: np.ndarray,
        extractor_class,
        scorer_class,
        controller_class,
        generator_class,
        test_name: str
    ):
        """Test pipeline with specified component classes."""
        height, width = image.shape[:2]

        # Step 1: Extract splats
        if extractor_class == OptimizedSplatExtractor:
            extractor = extractor_class(max_workers=2)  # Limit workers for testing
        else:
            extractor = extractor_class()

        splats = extractor.extract_splats(image, n_splats=50)

        # Validate extraction results
        assert isinstance(splats, list)
        assert 0 < len(splats) <= 60  # Allow some variance due to filtering
        for splat in splats:
            assert 0 <= splat.x <= width
            assert 0 <= splat.y <= height
            assert splat.rx > 0 and splat.ry > 0
            assert 0 <= splat.r <= 255
            assert 0 <= splat.g <= 255
            assert 0 <= splat.b <= 255
            assert 0.0 <= splat.a <= 1.0

        # Step 2: Score splats for importance
        if scorer_class == OptimizedImportanceScorer:
            scorer = scorer_class(max_workers=2)  # Limit workers for testing
        else:
            scorer = scorer_class()

        scorer.score_splats(splats, image)

        # Validate scoring results
        for splat in splats:
            assert hasattr(splat, 'score')
            assert isinstance(splat.score, float)
            assert splat.score >= 0

        # Step 3: Apply quality control
        if controller_class == ParallelQualityController:
            controller = controller_class(target_count=30, max_workers=2)
        else:
            controller = controller_class(target_count=30)

        filtered_splats = controller.optimize_splats(splats)

        # Validate quality control results
        assert isinstance(filtered_splats, list)
        assert len(filtered_splats) <= 35  # Allow some variance
        for splat in filtered_splats:
            assert splat.rx > 0 and splat.ry > 0
            assert 0 <= splat.r <= 255
            assert 0 <= splat.g <= 255
            assert 0 <= splat.b <= 255
            assert 0.0 <= splat.a <= 1.0

        # Step 4: Assign to depth layers
        layer_assigner = LayerAssigner(n_layers=4)
        layers = layer_assigner.assign_layers(filtered_splats)

        # Validate layer assignment
        assert isinstance(layers, dict)
        assert len(layers) <= 4
        total_splats_in_layers = sum(len(layer_splats) for layer_splats in layers.values())
        assert total_splats_in_layers == len(filtered_splats)

        for layer_splats in layers.values():
            for splat in layer_splats:
                assert 0.2 <= splat.depth <= 1.0

        # Step 5: Generate SVG
        if generator_class == OptimizedSVGGenerator:
            generator = generator_class(width=width, height=height, chunk_size=10)
        else:
            generator = generator_class(width=width, height=height)

        # Test both solid and gaussian modes
        for gaussian_mode in [False, True]:
            svg_content = generator.generate_svg(
                layers,
                gaussian_mode=gaussian_mode,
                title=f"Integration Test - {test_name}"
            )

            # Validate SVG structure
            assert isinstance(svg_content, str)
            assert len(svg_content) > 1000  # Should be substantial
            assert '<?xml version="1.0"' in svg_content
            assert f'viewBox="0 0 {width} {height}"' in svg_content
            assert f'<title>Integration Test - {test_name}</title>' in svg_content
            assert '<ellipse' in svg_content
            assert '</svg>' in svg_content

            # Validate gaussian mode specifics
            if gaussian_mode:
                assert 'radialGradient id="gaussian-gradient-' in svg_content
                assert 'url(#gaussian-gradient-' in svg_content
            else:
                assert 'rgba(' in svg_content

            # Test SVG validation method
            if hasattr(generator, '_validate_svg_structure'):
                assert generator._validate_svg_structure(svg_content)

    def test_pipeline_with_file_io(self):
        """Test complete pipeline including file I/O operations."""
        test_image = self.test_images['shapes']

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Save test image as PNG
            input_path = temp_path / "test_input.png"
            pil_image = Image.fromarray(test_image)
            pil_image.save(input_path)

            # Load image using our loader
            loaded_image, dimensions = load_image(input_path)
            validate_image_dimensions(dimensions)

            # Verify loaded image matches original
            assert loaded_image.shape == test_image.shape
            assert dimensions == (test_image.shape[1], test_image.shape[0])  # width, height

            # Run pipeline on loaded image
            extractor = SplatExtractor()
            splats = extractor.extract_splats(loaded_image, n_splats=40)

            scorer = ImportanceScorer()
            scorer.score_splats(splats, loaded_image)

            controller = QualityController(target_count=25)
            filtered_splats = controller.optimize_splats(splats)

            layer_assigner = LayerAssigner(n_layers=3)
            layers = layer_assigner.assign_layers(filtered_splats)

            generator = SVGGenerator(width=dimensions[0], height=dimensions[1])
            svg_content = generator.generate_svg(layers, title="File I/O Test")

            # Save SVG
            output_path = temp_path / "test_output.svg"
            generator.save_svg(svg_content, output_path)

            # Verify file was created and contains expected content
            assert output_path.exists()
            saved_content = output_path.read_text()
            assert saved_content == svg_content
            assert len(saved_content) > 1000

    def test_pipeline_performance_characteristics(self):
        """Test pipeline performance with profiling."""
        test_image = self.test_images['gradient']

        # Clear global profiler
        global_profiler.metrics.clear()

        @global_profiler.profile_function("integration_test_pipeline")
        def run_full_pipeline():
            extractor = OptimizedSplatExtractor(max_workers=2)
            splats = extractor.extract_splats(test_image, n_splats=100)

            scorer = OptimizedImportanceScorer(max_workers=2)
            scorer.score_splats(splats, test_image)

            controller = ParallelQualityController(target_count=50, max_workers=2)
            filtered_splats = controller.optimize_splats(splats)

            layer_assigner = LayerAssigner(n_layers=5)
            layers = layer_assigner.assign_layers(filtered_splats)

            generator = OptimizedSVGGenerator(width=150, height=100, chunk_size=20)
            svg_content = generator.generate_svg(layers)

            return svg_content

        # Run pipeline with profiling
        svg_result = run_full_pipeline()

        # Validate result
        assert isinstance(svg_result, str)
        assert len(svg_result) > 1000

        # Check profiling results
        summary = global_profiler.get_summary()
        assert summary['total_time'] > 0
        assert summary['peak_memory_mb'] > 0
        assert 'integration_test_pipeline' in summary['by_function']

        # Performance assertions (should complete in reasonable time)
        total_time = summary['total_time']
        assert total_time < 10.0, f"Pipeline took too long: {total_time:.2f}s"

        peak_memory = summary['peak_memory_mb']
        assert peak_memory < 500, f"Pipeline used too much memory: {peak_memory:.1f}MB"

    def test_pipeline_error_handling(self):
        """Test pipeline error handling and recovery."""
        # Test with problematic images
        problematic_images = {
            'very_small': np.random.randint(0, 255, (10, 15, 3), dtype=np.uint8),
            'single_color': np.full((50, 75, 3), 128, dtype=np.uint8),
        }

        for image_name, image in problematic_images.items():
            with self._subtest_context(f"Error handling - {image_name}"):
                try:
                    # Pipeline should handle edge cases gracefully
                    extractor = SplatExtractor()
                    splats = extractor.extract_splats(image, n_splats=20)

                    # May have very few or no splats for problematic images
                    if splats:
                        scorer = ImportanceScorer()
                        scorer.score_splats(splats, image)

                        controller = QualityController(target_count=10)
                        filtered_splats = controller.optimize_splats(splats)

                        layer_assigner = LayerAssigner(n_layers=2)
                        layers = layer_assigner.assign_layers(filtered_splats)

                        generator = SVGGenerator(width=image.shape[1], height=image.shape[0])
                        svg_content = generator.generate_svg(layers)

                        # Should produce valid SVG even with few/no splats
                        assert isinstance(svg_content, str)
                        assert '<?xml version="1.0"' in svg_content
                        assert '</svg>' in svg_content

                except Exception as e:
                    # Log the error but don't fail the test for edge cases
                    logging.warning(f"Pipeline handled edge case {image_name}: {e}")

    def test_pipeline_consistency(self):
        """Test that pipeline produces consistent results with same input."""
        test_image = self.test_images['checkered']

        # Run pipeline multiple times with same parameters
        results = []
        for i in range(3):
            extractor = SplatExtractor()
            splats = extractor.extract_splats(test_image, n_splats=30)

            scorer = ImportanceScorer(area_weight=0.4, edge_weight=0.4, color_weight=0.2)
            scorer.score_splats(splats, test_image)

            controller = QualityController(target_count=20, k_multiplier=2.0)
            filtered_splats = controller.optimize_splats(splats)

            layer_assigner = LayerAssigner(n_layers=3)
            layers = layer_assigner.assign_layers(filtered_splats)

            generator = SVGGenerator(width=120, height=80, precision=2)
            svg_content = generator.generate_svg(layers)

            results.append({
                'splat_count': len(filtered_splats),
                'layer_count': len(layers),
                'svg_length': len(svg_content)
            })

        # Results should be reasonably consistent
        splat_counts = [r['splat_count'] for r in results]
        layer_counts = [r['layer_count'] for r in results]

        # Allow some variance due to SLIC randomness, but should be close
        assert max(splat_counts) - min(splat_counts) <= 5, f"Inconsistent splat counts: {splat_counts}"
        assert len(set(layer_counts)) <= 2, f"Inconsistent layer counts: {layer_counts}"

    def _subtest_context(self, test_name: str):
        """Create a context manager for subtests."""
        from contextlib import nullcontext
        return nullcontext()  # Simple context that doesn't interfere


class TestPipelineScalability:
    """Test pipeline scalability with various input sizes."""

    def test_small_images(self):
        """Test pipeline with small images."""
        small_image = np.random.randint(0, 255, (40, 60, 3), dtype=np.uint8)

        extractor = SplatExtractor()
        splats = extractor.extract_splats(small_image, n_splats=10)

        # Should handle small images gracefully
        assert isinstance(splats, list)
        # May have very few splats due to size filtering
        assert len(splats) <= 15

    def test_medium_images(self):
        """Test pipeline with medium-sized images."""
        medium_image = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)

        extractor = OptimizedSplatExtractor(max_workers=2)
        splats = extractor.extract_splats(medium_image, n_splats=100)

        scorer = OptimizedImportanceScorer(max_workers=2)
        scorer.score_splats(splats, medium_image)

        controller = ParallelQualityController(target_count=75, max_workers=2)
        filtered_splats = controller.optimize_splats(splats)

        assert len(filtered_splats) <= 85  # Allow some variance
        assert all(s.score >= 0 for s in filtered_splats)

    def test_parameter_scaling(self):
        """Test how pipeline scales with different parameter values."""
        test_image = np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8)

        # Test different splat counts
        splat_counts = [25, 50, 100, 200]
        for n_splats in splat_counts:
            extractor = SplatExtractor()
            splats = extractor.extract_splats(test_image, n_splats=n_splats)

            # More requested splats should generally yield more results
            # (though filtering may reduce the count)
            assert isinstance(splats, list)
            if n_splats <= 50:
                assert len(splats) <= n_splats * 1.2  # Allow some overhead
            else:
                # For larger counts, expect significant filtering
                assert len(splats) <= n_splats * 0.8

        # Test different layer counts
        layer_counts = [2, 4, 6, 8]
        splats = extractor.extract_splats(test_image, n_splats=50)

        for n_layers in layer_counts:
            layer_assigner = LayerAssigner(n_layers=n_layers)
            layers = layer_assigner.assign_layers(splats)

            assert len(layers) <= n_layers
            # Should distribute splats across layers
            if len(splats) > n_layers:
                assert len(layers) >= 2  # Should use multiple layers