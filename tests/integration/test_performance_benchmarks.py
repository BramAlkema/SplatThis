"""Performance benchmark tests for SplatThis."""

import pytest
import time
import numpy as np
from pathlib import Path
import psutil
import tempfile

from splat_this.core.extract import SplatExtractor
from splat_this.core.optimized_extract import OptimizedSplatExtractor
from splat_this.core.layering import ImportanceScorer, QualityController
from splat_this.core.optimized_layering import OptimizedImportanceScorer, ParallelQualityController
from splat_this.core.svgout import SVGGenerator
from splat_this.core.optimized_svgout import OptimizedSVGGenerator
from splat_this.utils.profiler import PerformanceProfiler, benchmark_function


class TestPerformanceBenchmarks:
    """Performance benchmark tests to ensure requirements are met."""

    def setup_method(self):
        """Set up benchmark test fixtures."""
        # Create test images of various sizes for benchmarking
        self.benchmark_images = {
            'small': np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8),     # ~0.3MP
            'medium': np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),    # ~1.2MP
            'large': np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8),    # ~3.7MP
            'hd': np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8),      # ~8.3MP (1080p)
        }

        # Performance targets (adjusted for testing environment)
        self.performance_targets = {
            'small': {'max_time': 5.0, 'max_memory_mb': 200},
            'medium': {'max_time': 15.0, 'max_memory_mb': 400},
            'large': {'max_time': 30.0, 'max_memory_mb': 800},
            'hd': {'max_time': 60.0, 'max_memory_mb': 1200},
        }

    def test_splat_extraction_performance(self):
        """Benchmark splat extraction performance."""
        for image_name, image in self.benchmark_images.items():
            target_splats = 1000 if image_name == 'hd' else 500

            # Test standard extractor
            standard_metrics = benchmark_function(
                self._extract_splats_standard,
                image, target_splats,
                iterations=3
            )

            # Test optimized extractor
            optimized_metrics = benchmark_function(
                self._extract_splats_optimized,
                image, target_splats,
                iterations=3
            )

            # Validate performance
            max_time = self.performance_targets[image_name]['max_time']
            max_memory = self.performance_targets[image_name]['max_memory_mb']

            assert standard_metrics['avg_time'] < max_time, \
                f"Standard extraction too slow for {image_name}: {standard_metrics['avg_time']:.2f}s > {max_time}s"

            assert optimized_metrics['avg_time'] < max_time, \
                f"Optimized extraction too slow for {image_name}: {optimized_metrics['avg_time']:.2f}s > {max_time}s"

            # Optimized version should be faster or comparable
            improvement_ratio = standard_metrics['avg_time'] / optimized_metrics['avg_time']
            assert improvement_ratio >= 0.7, \
                f"Optimized extraction not faster for {image_name}: {improvement_ratio:.2f}x"

            print(f"Extraction {image_name}: Standard {standard_metrics['avg_time']:.2f}s, "
                  f"Optimized {optimized_metrics['avg_time']:.2f}s ({improvement_ratio:.2f}x)")

    def test_importance_scoring_performance(self):
        """Benchmark importance scoring performance."""
        # Create test splats for scoring
        test_splats = self._create_test_splats(500)

        for image_name, image in self.benchmark_images.items():
            # Test standard scorer
            standard_metrics = benchmark_function(
                self._score_splats_standard,
                test_splats.copy(), image,
                iterations=3
            )

            # Test optimized scorer
            optimized_metrics = benchmark_function(
                self._score_splats_optimized,
                test_splats.copy(), image,
                iterations=3
            )

            # Validate performance (scoring should be fast)
            max_scoring_time = 5.0  # Scoring should be under 5 seconds

            assert standard_metrics['avg_time'] < max_scoring_time, \
                f"Standard scoring too slow for {image_name}: {standard_metrics['avg_time']:.2f}s"

            assert optimized_metrics['avg_time'] < max_scoring_time, \
                f"Optimized scoring too slow for {image_name}: {optimized_metrics['avg_time']:.2f}s"

            # For larger images with many splats, optimized should be faster
            if len(test_splats) > 100:
                improvement_ratio = standard_metrics['avg_time'] / optimized_metrics['avg_time']
                assert improvement_ratio >= 0.8, \
                    f"Optimized scoring not significantly faster for {image_name}: {improvement_ratio:.2f}x"

            print(f"Scoring {image_name}: Standard {standard_metrics['avg_time']:.3f}s, "
                  f"Optimized {optimized_metrics['avg_time']:.3f}s")

    def test_svg_generation_performance(self):
        """Benchmark SVG generation performance."""
        # Create test layers with different splat counts
        test_cases = [
            ('small_count', self._create_test_layers(100)),
            ('medium_count', self._create_test_layers(500)),
            ('large_count', self._create_test_layers(1500)),
            ('very_large_count', self._create_test_layers(3000)),
        ]

        for case_name, layers in test_cases:
            total_splats = sum(len(layer_splats) for layer_splats in layers.values())

            # Test standard SVG generator
            standard_metrics = benchmark_function(
                self._generate_svg_standard,
                layers, 800, 600,
                iterations=3
            )

            # Test optimized SVG generator
            optimized_metrics = benchmark_function(
                self._generate_svg_optimized,
                layers, 800, 600,
                iterations=3
            )

            # Performance expectations based on splat count
            if total_splats < 500:
                max_time = 2.0
            elif total_splats < 1500:
                max_time = 5.0
            else:
                max_time = 15.0

            assert standard_metrics['avg_time'] < max_time, \
                f"Standard SVG generation too slow for {case_name}: {standard_metrics['avg_time']:.2f}s"

            assert optimized_metrics['avg_time'] < max_time, \
                f"Optimized SVG generation too slow for {case_name}: {optimized_metrics['avg_time']:.2f}s"

            # For large splat counts, optimized should be significantly faster
            if total_splats > 1000:
                improvement_ratio = standard_metrics['avg_time'] / optimized_metrics['avg_time']
                assert improvement_ratio >= 1.2, \
                    f"Optimized SVG not faster for {case_name}: {improvement_ratio:.2f}x"

            print(f"SVG {case_name} ({total_splats} splats): Standard {standard_metrics['avg_time']:.3f}s, "
                  f"Optimized {optimized_metrics['avg_time']:.3f}s")

    def test_memory_usage_benchmarks(self):
        """Test memory usage during processing."""
        process = psutil.Process()

        for image_name, image in self.benchmark_images.items():
            # Skip HD image for memory test to avoid system issues
            if image_name == 'hd':
                continue

            profiler = PerformanceProfiler()

            @profiler.profile_function("memory_test")
            def run_memory_test():
                extractor = OptimizedSplatExtractor(max_workers=2)
                splats = extractor.extract_splats(image, n_splats=500)

                scorer = OptimizedImportanceScorer(max_workers=2)
                scorer.score_splats(splats, image)

                controller = ParallelQualityController(target_count=300, max_workers=2)
                filtered_splats = controller.optimize_splats(splats)

                from splat_this.core.layering import LayerAssigner
                layer_assigner = LayerAssigner(n_layers=5)
                layers = layer_assigner.assign_layers(filtered_splats)

                generator = OptimizedSVGGenerator(width=image.shape[1], height=image.shape[0])
                svg_content = generator.generate_svg(layers)

                return len(svg_content)

            # Run memory test
            initial_memory = process.memory_info().rss / (1024 * 1024)  # MB
            result_size = run_memory_test()
            final_memory = process.memory_info().rss / (1024 * 1024)  # MB

            # Get performance summary
            summary = profiler.get_summary()
            peak_memory = summary['peak_memory_mb']

            # Validate memory usage
            max_memory = self.performance_targets[image_name]['max_memory_mb']
            memory_increase = final_memory - initial_memory

            assert peak_memory < max_memory, \
                f"Peak memory too high for {image_name}: {peak_memory:.1f}MB > {max_memory}MB"

            assert memory_increase < max_memory * 0.5, \
                f"Memory increase too high for {image_name}: {memory_increase:.1f}MB"

            print(f"Memory {image_name}: Peak {peak_memory:.1f}MB, "
                  f"Increase {memory_increase:.1f}MB, Result {result_size} bytes")

    def test_end_to_end_performance(self):
        """Test end-to-end pipeline performance."""
        for image_name, image in self.benchmark_images.items():
            # Skip very large images for CI stability
            if image_name == 'hd':
                continue

            target_time = self.performance_targets[image_name]['max_time']

            # Test complete optimized pipeline
            pipeline_metrics = benchmark_function(
                self._run_complete_pipeline,
                image,
                iterations=2  # Fewer iterations for full pipeline
            )

            assert pipeline_metrics['avg_time'] < target_time, \
                f"Complete pipeline too slow for {image_name}: {pipeline_metrics['avg_time']:.2f}s > {target_time}s"

            print(f"End-to-end {image_name}: {pipeline_metrics['avg_time']:.2f}s "
                  f"(target: {target_time}s)")

    def test_file_io_performance(self):
        """Test file I/O performance."""
        # Create test SVG content of various sizes
        svg_sizes = {
            'small': "<svg>" + "x" * 10000 + "</svg>",           # ~10KB
            'medium': "<svg>" + "x" * 100000 + "</svg>",         # ~100KB
            'large': "<svg>" + "x" * 1000000 + "</svg>",         # ~1MB
        }

        for size_name, svg_content in svg_sizes.items():
            with tempfile.TemporaryDirectory() as temp_dir:
                output_path = Path(temp_dir) / f"test_{size_name}.svg"

                # Benchmark file writing
                write_metrics = benchmark_function(
                    self._write_svg_file,
                    svg_content, output_path,
                    iterations=5
                )

                # Benchmark file reading
                read_metrics = benchmark_function(
                    self._read_svg_file,
                    output_path,
                    iterations=5
                )

                # Performance expectations
                max_write_time = 0.1 if len(svg_content) < 500000 else 0.5
                max_read_time = 0.05 if len(svg_content) < 500000 else 0.2

                assert write_metrics['avg_time'] < max_write_time, \
                    f"File writing too slow for {size_name}: {write_metrics['avg_time']:.3f}s"

                assert read_metrics['avg_time'] < max_read_time, \
                    f"File reading too slow for {size_name}: {read_metrics['avg_time']:.3f}s"

                print(f"File I/O {size_name} ({len(svg_content)} bytes): "
                      f"Write {write_metrics['avg_time']:.3f}s, Read {read_metrics['avg_time']:.3f}s")

    # Helper methods for benchmarking

    def _extract_splats_standard(self, image: np.ndarray, n_splats: int):
        """Extract splats using standard extractor."""
        extractor = SplatExtractor()
        return extractor.extract_splats(image, n_splats)

    def _extract_splats_optimized(self, image: np.ndarray, n_splats: int):
        """Extract splats using optimized extractor."""
        extractor = OptimizedSplatExtractor(max_workers=2)
        return extractor.extract_splats(image, n_splats)

    def _score_splats_standard(self, splats, image: np.ndarray):
        """Score splats using standard scorer."""
        scorer = ImportanceScorer()
        scorer.score_splats(splats, image)
        return splats

    def _score_splats_optimized(self, splats, image: np.ndarray):
        """Score splats using optimized scorer."""
        scorer = OptimizedImportanceScorer(max_workers=2)
        scorer.score_splats(splats, image)
        return splats

    def _generate_svg_standard(self, layers, width: int, height: int):
        """Generate SVG using standard generator."""
        generator = SVGGenerator(width=width, height=height)
        return generator.generate_svg(layers)

    def _generate_svg_optimized(self, layers, width: int, height: int):
        """Generate SVG using optimized generator."""
        generator = OptimizedSVGGenerator(width=width, height=height, chunk_size=100)
        return generator.generate_svg(layers)

    def _run_complete_pipeline(self, image: np.ndarray):
        """Run complete optimized pipeline."""
        extractor = OptimizedSplatExtractor(max_workers=2)
        splats = extractor.extract_splats(image, n_splats=300)

        scorer = OptimizedImportanceScorer(max_workers=2)
        scorer.score_splats(splats, image)

        controller = ParallelQualityController(target_count=200, max_workers=2)
        filtered_splats = controller.optimize_splats(splats)

        from splat_this.core.layering import LayerAssigner
        layer_assigner = LayerAssigner(n_layers=4)
        layers = layer_assigner.assign_layers(filtered_splats)

        generator = OptimizedSVGGenerator(
            width=image.shape[1],
            height=image.shape[0],
            chunk_size=50
        )
        svg_content = generator.generate_svg(layers)

        return len(svg_content)

    def _write_svg_file(self, content: str, path: Path):
        """Write SVG content to file."""
        from splat_this.utils.optimized_io import write_svg_optimized
        write_svg_optimized(content, path)
        return path.stat().st_size

    def _read_svg_file(self, path: Path):
        """Read SVG content from file."""
        from splat_this.utils.optimized_io import OptimizedFileReader
        reader = OptimizedFileReader()
        return len(reader.read_file(path))

    def _create_test_splats(self, count: int):
        """Create test splats for benchmarking."""
        from splat_this.core.extract import Gaussian

        splats = []
        for i in range(count):
            splat = Gaussian(
                x=np.random.uniform(10, 790),
                y=np.random.uniform(10, 590),
                rx=np.random.uniform(3, 15),
                ry=np.random.uniform(3, 15),
                theta=np.random.uniform(0, 2*np.pi),
                r=np.random.randint(0, 256),
                g=np.random.randint(0, 256),
                b=np.random.randint(0, 256),
                a=np.random.uniform(0.3, 1.0),
                score=np.random.uniform(0, 1)
            )
            splats.append(splat)

        return splats

    def _create_test_layers(self, total_splats: int):
        """Create test layers for benchmarking."""
        splats = self._create_test_splats(total_splats)

        # Distribute across layers
        layer_count = min(5, max(2, total_splats // 100))
        layers = {}

        splats_per_layer = total_splats // layer_count
        for i in range(layer_count):
            start_idx = i * splats_per_layer
            end_idx = start_idx + splats_per_layer if i < layer_count - 1 else total_splats
            layer_splats = splats[start_idx:end_idx]

            # Set depth based on layer
            depth = 0.2 + (0.8 * i / (layer_count - 1))
            for splat in layer_splats:
                splat.depth = depth

            layers[i] = layer_splats

        return layers


class TestPerformanceRegression:
    """Test for performance regressions between versions."""

    def test_no_performance_regression(self):
        """Ensure optimized components are not slower than standard ones."""
        test_image = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)

        # Test extraction performance
        standard_extractor_time = benchmark_function(
            lambda: SplatExtractor().extract_splats(test_image, 100),
            iterations=3
        )['avg_time']

        optimized_extractor_time = benchmark_function(
            lambda: OptimizedSplatExtractor(max_workers=2).extract_splats(test_image, 100),
            iterations=3
        )['avg_time']

        # Optimized should not be significantly slower
        regression_ratio = optimized_extractor_time / standard_extractor_time
        assert regression_ratio < 1.5, \
            f"Optimized extractor regression: {regression_ratio:.2f}x slower"

        print(f"Extraction regression test: {regression_ratio:.2f}x "
              f"({'slower' if regression_ratio > 1 else 'faster'})")

    def test_memory_efficiency_no_regression(self):
        """Ensure memory usage doesn't regress."""
        test_image = np.random.randint(0, 255, (150, 200, 3), dtype=np.uint8)

        process = psutil.Process()

        # Measure standard pipeline memory
        initial_memory = process.memory_info().rss
        extractor = SplatExtractor()
        splats = extractor.extract_splats(test_image, 100)
        standard_memory = process.memory_info().rss - initial_memory

        # Measure optimized pipeline memory
        initial_memory = process.memory_info().rss
        opt_extractor = OptimizedSplatExtractor(max_workers=2)
        opt_splats = opt_extractor.extract_splats(test_image, 100)
        optimized_memory = process.memory_info().rss - initial_memory

        # Optimized should not use significantly more memory
        memory_ratio = optimized_memory / max(standard_memory, 1)  # Avoid division by zero
        assert memory_ratio < 2.0, \
            f"Optimized extractor memory regression: {memory_ratio:.2f}x more memory"

        print(f"Memory regression test: {memory_ratio:.2f}x memory usage")

    @pytest.mark.slow
    def test_scalability_performance(self):
        """Test that performance scales appropriately with input size."""
        # Test with different image sizes
        sizes = [(100, 150), (200, 300), (400, 600)]
        times = []

        for height, width in sizes:
            test_image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)

            metrics = benchmark_function(
                lambda: OptimizedSplatExtractor(max_workers=2).extract_splats(test_image, 200),
                iterations=2
            )

            times.append(metrics['avg_time'])
            pixel_count = height * width

            print(f"Scalability test {width}×{height} ({pixel_count:,} pixels): {metrics['avg_time']:.2f}s")

        # Performance should scale sub-linearly with image size
        # (due to optimizations and fixed splat count)
        if len(times) >= 2:
            # Check that 4x pixels doesn't take more than 3x time
            size_ratio = (sizes[-1][0] * sizes[-1][1]) / (sizes[0][0] * sizes[0][1])
            time_ratio = times[-1] / times[0]

            efficiency_ratio = time_ratio / size_ratio
            assert efficiency_ratio < 0.8, \
                f"Poor scalability: {size_ratio:.1f}x pixels, {time_ratio:.1f}x time (efficiency: {efficiency_ratio:.2f})"

            print(f"Scalability efficiency: {efficiency_ratio:.2f} (lower is better)")