"""Integration tests for the complete SplatThis pipeline."""

import pytest
import numpy as np
import tempfile
from pathlib import Path

from splat_this.core.extract import Gaussian
from splat_this.core.layering import ImportanceScorer, LayerAssigner, QualityController
from splat_this.core.svgout import SVGGenerator


class TestPipelineIntegration:
    """Test complete pipeline integration from splats to SVG."""

    def test_complete_pipeline_workflow(self):
        """Test the complete pipeline from raw splats to final SVG."""
        # Step 1: Create initial splats (simulating SLIC extraction)
        raw_splats = self._create_test_splats(count=50)

        # Step 2: Score splats for importance
        scorer = ImportanceScorer(area_weight=0.3, edge_weight=0.5, color_weight=0.2)

        # Create mock image for scoring
        test_image = np.random.randint(0, 256, (200, 300, 3), dtype=np.uint8)
        scorer.score_splats(raw_splats, test_image)

        # Verify scoring worked
        assert all(0.0 <= splat.score <= 1.0 for splat in raw_splats)

        # Step 3: Apply quality control filtering
        controller = QualityController(
            target_count=20,
            k_multiplier=2.0,
            alpha_adjustment=True
        )

        quality_splats = controller.optimize_splats(raw_splats)

        # Verify quality control
        assert len(quality_splats) <= 20
        assert all(splat.rx > 0 and splat.ry > 0 for splat in quality_splats)
        assert all(0.0 <= splat.a <= 1.0 for splat in quality_splats)

        # Step 4: Assign to depth layers
        layer_assigner = LayerAssigner(n_layers=4)
        layers = layer_assigner.assign_layers(quality_splats)

        # Verify layer assignment
        assert len(layers) <= 4
        total_layer_splats = sum(len(layer_splats) for layer_splats in layers.values())
        assert total_layer_splats == len(quality_splats)

        # Verify depth assignment
        for layer_splats in layers.values():
            for splat in layer_splats:
                assert 0.2 <= splat.depth <= 1.0

        # Step 5: Generate SVG
        svg_generator = SVGGenerator(
            width=300, height=200,
            precision=2,
            parallax_strength=40
        )

        svg_content = svg_generator.generate_svg(
            layers=layers,
            gaussian_mode=False,
            title="Integration Test SVG"
        )

        # Verify SVG generation
        assert '<?xml version="1.0"' in svg_content
        assert 'viewBox="0 0 300 200"' in svg_content
        assert '<title>Integration Test SVG</title>' in svg_content
        assert svg_generator._validate_svg_structure(svg_content)

        # Step 6: Save SVG to file
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "integration_test.svg"
            svg_generator.save_svg(svg_content, output_path)

            assert output_path.exists()
            saved_content = output_path.read_text(encoding="utf-8")
            assert saved_content == svg_content

    def test_gaussian_mode_pipeline(self):
        """Test pipeline with gaussian mode enabled."""
        # Create test splats
        splats = self._create_test_splats(count=10)

        # Skip scoring step, use default scores
        for splat in splats:
            splat.score = np.random.uniform(0.1, 0.9)

        # Assign to layers
        layer_assigner = LayerAssigner(n_layers=3)
        layers = layer_assigner.assign_layers(splats)

        # Generate SVG in gaussian mode
        svg_generator = SVGGenerator(width=400, height=300)
        svg_content = svg_generator.generate_svg(
            layers=layers,
            gaussian_mode=True
        )

        # Verify gaussian-specific features
        assert 'radialGradient id="gaussianGradient"' in svg_content
        assert 'url(#gaussianGradient)' in svg_content
        assert 'data-color=' in svg_content

    def test_empty_pipeline_handling(self):
        """Test pipeline handling with empty inputs."""
        # Empty splats
        empty_splats = []

        # Quality control should handle empty input
        controller = QualityController(target_count=10)
        result_splats = controller.optimize_splats(empty_splats)
        assert result_splats == []

        # Layer assignment should handle empty input
        layer_assigner = LayerAssigner()
        layers = layer_assigner.assign_layers(result_splats)
        assert layers == {}

        # SVG generation should handle empty layers
        svg_generator = SVGGenerator(width=800, height=600)
        svg_content = svg_generator.generate_svg(layers)

        assert 'No splats to display' in svg_content
        assert svg_generator._validate_svg_structure(svg_content)

    def test_pipeline_with_statistics(self):
        """Test pipeline with comprehensive statistics tracking."""
        # Create test splats
        original_splats = self._create_test_splats(count=100)

        # Add scoring
        for splat in original_splats:
            splat.score = np.random.beta(2, 5)  # Skewed distribution

        # Quality control with statistics
        controller = QualityController(target_count=30, k_multiplier=2.0)
        filtered_splats = controller.optimize_splats(original_splats)

        quality_stats = controller.get_quality_statistics(original_splats, filtered_splats)

        # Verify statistics
        assert quality_stats['original_count'] == 100
        assert quality_stats['final_count'] == len(filtered_splats)
        assert 0.0 <= quality_stats['reduction_ratio'] <= 1.0
        assert quality_stats['avg_area'] > 0
        assert quality_stats['avg_score'] >= 0

        # Layer assignment with statistics
        layer_assigner = LayerAssigner(n_layers=5)
        layers = layer_assigner.assign_layers(filtered_splats)
        layer_stats = layer_assigner.get_layer_statistics(layers)

        # Verify layer statistics
        assert len(layer_stats) <= 5
        for layer_idx, stats in layer_stats.items():
            assert stats['count'] >= 0
            assert 0.2 <= stats['depth'] <= 1.0
            if stats['count'] > 0:
                assert stats['avg_score'] >= 0
                assert stats['avg_area'] > 0

        # SVG info
        svg_generator = SVGGenerator(width=1000, height=800, precision=3)
        svg_info = svg_generator.get_svg_info(layers)

        assert svg_info['width'] == 1000
        assert svg_info['height'] == 800
        assert svg_info['precision'] == 3
        assert svg_info['layer_count'] == len(layers)
        assert svg_info['total_splats'] == len(filtered_splats)

    def test_pipeline_performance_characteristics(self):
        """Test pipeline performance with larger dataset."""
        # Create larger dataset
        large_splats = self._create_test_splats(count=1000)

        # Add realistic score distribution
        for i, splat in enumerate(large_splats):
            # Score based on position (center is more important)
            center_x, center_y = 500, 400
            distance = np.sqrt((splat.x - center_x)**2 + (splat.y - center_y)**2)
            max_distance = np.sqrt(center_x**2 + center_y**2)
            splat.score = 1.0 - (distance / max_distance)

        # Apply aggressive filtering
        controller = QualityController(
            target_count=50,
            k_multiplier=1.5,
            alpha_adjustment=True
        )

        optimized_splats = controller.optimize_splats(large_splats)

        # Verify significant reduction
        assert len(optimized_splats) <= 50
        assert len(optimized_splats) < len(large_splats) * 0.1  # At least 90% reduction

        # Verify highest-scoring splats were preserved
        original_scores = sorted([s.score for s in large_splats], reverse=True)[:50]
        optimized_scores = sorted([s.score for s in optimized_splats], reverse=True)

        # Top scores should be similar (allowing for filtering effects)
        if len(optimized_scores) > 10:
            top_original = np.mean(original_scores[:10])
            top_optimized = np.mean(optimized_scores[:10])
            assert top_optimized >= top_original * 0.8  # Should preserve high-quality splats

    def test_pipeline_layer_balance(self):
        """Test that pipeline maintains good layer balance."""
        # Create splats with varied scores
        splats = []
        for i in range(80):
            score = (i / 80.0) ** 2  # Quadratic distribution
            splat = Gaussian(
                x=np.random.uniform(0, 800),
                y=np.random.uniform(0, 600),
                rx=np.random.uniform(5, 20),
                ry=np.random.uniform(5, 20),
                theta=np.random.uniform(0, 2*np.pi),
                r=np.random.randint(0, 256),
                g=np.random.randint(0, 256),
                b=np.random.randint(0, 256),
                a=0.8,
                score=score
            )
            splats.append(splat)

        # Layer assignment
        layer_assigner = LayerAssigner(n_layers=6)
        layers = layer_assigner.assign_layers(splats)

        # Balance layers
        balanced_layers = layer_assigner.balance_layers(layers, min_per_layer=8)

        # Verify balance
        layer_counts = [len(layer_splats) for layer_splats in balanced_layers.values()]

        if len(layer_counts) > 1:
            # Should be relatively balanced
            min_count = min(layer_counts)
            max_count = max(layer_counts)
            balance_ratio = min_count / max_count if max_count > 0 else 1.0

            # Allow some imbalance but not extreme
            assert balance_ratio >= 0.3, f"Poor layer balance: {layer_counts}"

    def test_error_handling_in_pipeline(self):
        """Test error handling throughout the pipeline."""
        # Test with some corrupted splats
        splats = self._create_test_splats(count=20)

        # Corrupt some splats after creation
        splats[5].rx = -1  # Invalid radius
        splats[10].r = 300  # Invalid color
        splats[15].a = 2.0  # Invalid alpha

        # Quality control should handle and fix/remove invalid splats
        controller = QualityController(target_count=15)
        cleaned_splats = controller.optimize_splats(splats)

        # Verify cleaning worked
        for splat in cleaned_splats:
            assert splat.rx > 0 and splat.ry > 0
            assert 0 <= splat.r <= 255
            assert 0 <= splat.g <= 255
            assert 0 <= splat.b <= 255
            assert 0.0 <= splat.a <= 1.0

        # Rest of pipeline should work normally
        layer_assigner = LayerAssigner(n_layers=3)
        layers = layer_assigner.assign_layers(cleaned_splats)

        svg_generator = SVGGenerator(width=400, height=300)
        svg_content = svg_generator.generate_svg(layers)

        assert svg_generator._validate_svg_structure(svg_content)

    def _create_test_splats(self, count: int) -> list:
        """Create a list of test splats with varied properties."""
        splats = []

        for i in range(count):
            splat = Gaussian(
                x=np.random.uniform(50, 750),
                y=np.random.uniform(50, 550),
                rx=np.random.uniform(3, 25),
                ry=np.random.uniform(3, 25),
                theta=np.random.uniform(0, 2*np.pi),
                r=np.random.randint(0, 256),
                g=np.random.randint(0, 256),
                b=np.random.randint(0, 256),
                a=np.random.uniform(0.3, 1.0),
                score=0.0,  # Will be set by scoring
                depth=0.5   # Will be set by layer assignment
            )
            splats.append(splat)

        return splats