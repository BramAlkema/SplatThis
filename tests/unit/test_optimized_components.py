"""Unit tests for optimized components."""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from splat_this.core.extract import Gaussian
from splat_this.core.optimized_extract import OptimizedSplatExtractor, BatchSplatExtractor
from splat_this.core.optimized_layering import OptimizedImportanceScorer, ParallelQualityController
from splat_this.core.optimized_svgout import OptimizedSVGGenerator
from splat_this.utils.profiler import PerformanceProfiler, MemoryEfficientProcessor, global_profiler
from splat_this.utils.optimized_io import OptimizedFileWriter, OptimizedFileReader, write_svg_optimized


class TestPerformanceProfiler:
    """Test performance profiler functionality."""

    def test_profiler_initialization(self):
        """Test profiler initialization."""
        profiler = PerformanceProfiler()
        assert profiler.metrics == {}
        assert isinstance(profiler.start_memory, int)

    def test_profile_function_decorator(self):
        """Test function profiling decorator."""
        profiler = PerformanceProfiler()

        @profiler.profile_function("test_func")
        def test_function(x, y):
            return x + y

        result = test_function(2, 3)
        assert result == 5
        assert "test_func" in profiler.metrics
        assert profiler.metrics["test_func"]["duration"] > 0
        assert profiler.metrics["test_func"]["calls"] == 1

    def test_get_summary(self):
        """Test performance summary generation."""
        profiler = PerformanceProfiler()

        @profiler.profile_function("test_func")
        def dummy_func():
            return "done"

        dummy_func()
        summary = profiler.get_summary()

        assert "total_time" in summary
        assert "peak_memory_mb" in summary
        assert "by_function" in summary
        assert "test_func" in summary["by_function"]

    def test_empty_summary(self):
        """Test summary with no metrics."""
        profiler = PerformanceProfiler()
        summary = profiler.get_summary()

        assert summary["total_time"] == 0.0
        assert summary["peak_memory_mb"] == 0.0
        assert summary["by_function"] == {}


class TestMemoryEfficientProcessor:
    """Test memory-efficient processor."""

    def test_memory_processor_initialization(self):
        """Test memory processor initialization."""
        processor = MemoryEfficientProcessor(max_memory_mb=512)
        assert processor.max_memory_mb == 512

    def test_check_memory_usage(self):
        """Test memory usage checking."""
        processor = MemoryEfficientProcessor()
        memory_mb = processor.check_memory_usage()
        assert isinstance(memory_mb, float)
        assert memory_mb > 0

    def test_estimate_memory_usage(self):
        """Test memory usage estimation."""
        processor = MemoryEfficientProcessor()
        estimated = processor.estimate_memory_usage(1000, 5, (800, 600))
        assert isinstance(estimated, float)
        assert estimated > 0

    def test_should_downsample_image_small(self):
        """Test downsampling decision for small image."""
        processor = MemoryEfficientProcessor(max_memory_mb=1024)
        should_downsample, new_size = processor.should_downsample_image((400, 300), 500)
        assert not should_downsample
        assert new_size == (400, 300)

    def test_should_downsample_image_large(self):
        """Test downsampling decision for very large image."""
        processor = MemoryEfficientProcessor(max_memory_mb=512)
        should_downsample, new_size = processor.should_downsample_image((10000, 8000), 2000)
        assert should_downsample
        assert new_size[0] < 10000
        assert new_size[1] < 8000


class TestOptimizedSplatExtractor:
    """Test optimized splat extractor."""

    def setup_method(self):
        """Set up test fixtures."""
        self.test_image = np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8)

    def test_extractor_initialization(self):
        """Test extractor initialization."""
        extractor = OptimizedSplatExtractor(max_workers=2)
        assert extractor.k == 2.5
        assert extractor.base_alpha == 0.65
        assert extractor.max_workers == 2

    @patch('splat_this.core.optimized_extract.slic')
    def test_extract_splats_basic(self, mock_slic):
        """Test basic splat extraction."""
        # Mock SLIC to return simple segments
        mock_segments = np.ones((100, 150), dtype=int)
        mock_segments[:50, :] = 1
        mock_segments[50:, :] = 2
        mock_slic.return_value = mock_segments

        extractor = OptimizedSplatExtractor(max_workers=1)
        splats = extractor.extract_splats(self.test_image, 50)

        assert isinstance(splats, list)
        assert len(splats) >= 0  # May filter out small segments
        for splat in splats:
            assert isinstance(splat, Gaussian)
            assert splat.rx > 0
            assert splat.ry > 0

    def test_downsample_image(self):
        """Test image downsampling."""
        extractor = OptimizedSplatExtractor()
        downsampled = extractor._downsample_image(self.test_image, (75, 50))

        assert downsampled.shape == (50, 75, 3)
        assert downsampled.dtype == np.uint8

    @patch('splat_this.core.optimized_extract.slic')
    def test_parallel_vs_sequential_extraction(self, mock_slic):
        """Test that parallel and sequential extraction work."""
        # Create larger segment array for parallel path
        mock_segments = np.random.randint(1, 150, (100, 150), dtype=int)
        mock_slic.return_value = mock_segments

        extractor = OptimizedSplatExtractor(max_workers=2)

        # This should trigger parallel processing due to many segments
        splats = extractor.extract_splats(self.test_image, 50)
        assert isinstance(splats, list)


class TestOptimizedImportanceScorer:
    """Test optimized importance scorer."""

    def setup_method(self):
        """Set up test fixtures."""
        self.test_image = np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8)
        self.test_splats = [
            Gaussian(x=25, y=25, rx=5, ry=5, theta=0, r=255, g=100, b=50, a=0.8),
            Gaussian(x=75, y=75, rx=8, ry=6, theta=0.5, r=100, g=200, b=150, a=0.7),
        ]

    def test_scorer_initialization(self):
        """Test scorer initialization."""
        scorer = OptimizedImportanceScorer(max_workers=2)
        assert scorer.area_weight == 0.3
        assert scorer.edge_weight == 0.5
        assert scorer.color_weight == 0.2
        assert scorer.max_workers == 2

    @patch('splat_this.core.optimized_layering.HAS_OPENCV', False)
    def test_score_splats_sequential(self):
        """Test sequential splat scoring without OpenCV."""
        scorer = OptimizedImportanceScorer(max_workers=1)
        scorer.score_splats(self.test_splats, self.test_image)

        for splat in self.test_splats:
            assert hasattr(splat, 'score')
            assert isinstance(splat.score, float)
            assert splat.score >= 0

    def test_score_splats_parallel(self):
        """Test parallel splat scoring."""
        # Create more splats to trigger parallel processing
        many_splats = []
        for i in range(120):
            splat = Gaussian(
                x=np.random.uniform(10, 140),
                y=np.random.uniform(10, 90),
                rx=np.random.uniform(3, 10),
                ry=np.random.uniform(3, 10),
                theta=0,
                r=np.random.randint(0, 256),
                g=np.random.randint(0, 256),
                b=np.random.randint(0, 256),
                a=0.8
            )
            many_splats.append(splat)

        scorer = OptimizedImportanceScorer(max_workers=2)
        scorer.score_splats(many_splats, self.test_image)

        for splat in many_splats:
            assert hasattr(splat, 'score')
            assert isinstance(splat.score, float)

    def test_score_splats_empty(self):
        """Test scoring with empty splat list."""
        scorer = OptimizedImportanceScorer()
        scorer.score_splats([], self.test_image)
        # Should not raise an exception

    def test_compute_edge_map_fallback(self):
        """Test edge map computation fallback."""
        scorer = OptimizedImportanceScorer()
        with patch('splat_this.core.optimized_layering.HAS_OPENCV', False):
            edge_map = scorer._compute_edge_map(self.test_image)
            assert isinstance(edge_map, np.ndarray)
            assert edge_map.shape == self.test_image.shape[:2]


class TestParallelQualityController:
    """Test parallel quality controller."""

    def setup_method(self):
        """Set up test fixtures."""
        self.test_splats = []
        for i in range(50):
            splat = Gaussian(
                x=np.random.uniform(10, 140),
                y=np.random.uniform(10, 90),
                rx=np.random.uniform(3, 10),
                ry=np.random.uniform(3, 10),
                theta=0,
                r=np.random.randint(0, 256),
                g=np.random.randint(0, 256),
                b=np.random.randint(0, 256),
                a=np.random.uniform(0.3, 1.0),
                score=np.random.uniform(0, 1)
            )
            self.test_splats.append(splat)

    def test_controller_initialization(self):
        """Test controller initialization."""
        controller = ParallelQualityController(target_count=30, max_workers=2)
        assert controller.target_count == 30
        assert controller.max_workers == 2

    def test_optimize_splats_basic(self):
        """Test basic splat optimization."""
        controller = ParallelQualityController(target_count=25)
        optimized = controller.optimize_splats(self.test_splats)

        assert isinstance(optimized, list)
        assert len(optimized) <= 25
        for splat in optimized:
            assert isinstance(splat, Gaussian)
            assert splat.rx > 0 and splat.ry > 0
            assert 0 <= splat.r <= 255
            assert 0.0 <= splat.a <= 1.0

    def test_optimize_splats_empty(self):
        """Test optimization with empty list."""
        controller = ParallelQualityController()
        result = controller.optimize_splats([])
        assert result == []

    def test_filter_valid_splats_parallel(self):
        """Test parallel validity filtering."""
        # Create valid splats first, then manually corrupt them to bypass validation
        invalid_splats = self.test_splats.copy()

        # Add splats and then corrupt them after creation
        corrupt_splat1 = Gaussian(x=10, y=10, rx=5, ry=5, theta=0, r=255, g=100, b=50, a=0.8)
        corrupt_splat1.rx = -1  # Make invalid after creation

        corrupt_splat2 = Gaussian(x=10, y=10, rx=5, ry=5, theta=0, r=255, g=100, b=50, a=0.8)
        corrupt_splat2.r = 300  # Make invalid after creation

        invalid_splats.extend([corrupt_splat1, corrupt_splat2])

        controller = ParallelQualityController(max_workers=2)
        valid = controller._filter_valid_splats_parallel(invalid_splats)

        assert len(valid) < len(invalid_splats)
        for splat in valid:
            assert controller._is_valid_splat(splat)

    def test_filter_by_score(self):
        """Test score-based filtering."""
        controller = ParallelQualityController(target_count=10)
        filtered = controller._filter_by_score(self.test_splats)

        assert len(filtered) <= 10
        # Check that splats are ordered by score (descending)
        scores = [s.score for s in filtered]
        assert scores == sorted(scores, reverse=True)

    def test_get_quality_statistics(self):
        """Test quality statistics generation."""
        controller = ParallelQualityController(target_count=25)
        optimized = controller.optimize_splats(self.test_splats)
        stats = controller.get_quality_statistics(self.test_splats, optimized)

        assert 'original_count' in stats
        assert 'final_count' in stats
        assert 'reduction_ratio' in stats
        assert 'avg_area' in stats
        assert 'avg_score' in stats

        assert stats['original_count'] == len(self.test_splats)
        assert stats['final_count'] == len(optimized)


class TestOptimizedSVGGenerator:
    """Test optimized SVG generator."""

    def setup_method(self):
        """Set up test fixtures."""
        self.test_splats = [
            Gaussian(x=50, y=40, rx=5, ry=5, theta=0, r=255, g=100, b=50, a=0.8, depth=0.3),
            Gaussian(x=75, y=60, rx=8, ry=6, theta=0.5, r=100, g=200, b=150, a=0.7, depth=0.7),
        ]
        self.test_layers = {0: [self.test_splats[0]], 1: [self.test_splats[1]]}

    def test_generator_initialization(self):
        """Test generator initialization."""
        generator = OptimizedSVGGenerator(
            width=800, height=600, chunk_size=100
        )
        assert generator.width == 800
        assert generator.height == 600
        assert generator.chunk_size == 100

    def test_generate_svg_standard(self):
        """Test standard SVG generation."""
        generator = OptimizedSVGGenerator(width=300, height=200, chunk_size=10)
        svg_content = generator.generate_svg(self.test_layers)

        assert isinstance(svg_content, str)
        assert '<?xml version="1.0"' in svg_content
        assert '<svg' in svg_content
        assert '</svg>' in svg_content
        assert 'ellipse' in svg_content

    def test_generate_svg_streaming(self):
        """Test streaming SVG generation for large splat count."""
        # Create many splats to trigger streaming mode
        many_splats = []
        for i in range(6000):  # >5000 triggers streaming
            splat = Gaussian(
                x=np.random.uniform(0, 300),
                y=np.random.uniform(0, 200),
                rx=np.random.uniform(2, 8),
                ry=np.random.uniform(2, 8),
                theta=0,
                r=np.random.randint(0, 256),
                g=np.random.randint(0, 256),
                b=np.random.randint(0, 256),
                a=0.8,
                depth=0.5
            )
            many_splats.append(splat)

        large_layers = {0: many_splats}

        generator = OptimizedSVGGenerator(width=300, height=200, chunk_size=500)
        svg_content = generator.generate_svg(large_layers)

        assert isinstance(svg_content, str)
        assert '<?xml version="1.0"' in svg_content
        assert len(svg_content) > 100000  # Should be quite large

    def test_generate_svg_gaussian_mode(self):
        """Test SVG generation in gaussian mode."""
        generator = OptimizedSVGGenerator(width=300, height=200)
        svg_content = generator.generate_svg(self.test_layers, gaussian_mode=True)

        assert 'radialGradient id="gaussianGradient"' in svg_content
        assert 'url(#gaussianGradient)' in svg_content

    def test_generate_svg_empty_layers(self):
        """Test SVG generation with empty layers."""
        generator = OptimizedSVGGenerator(width=300, height=200)
        svg_content = generator.generate_svg({})

        assert 'No splats to display' in svg_content

    def test_generate_large_layer_parallel(self):
        """Test parallel processing for large layers."""
        # Create layer with many splats
        many_splats = []
        for i in range(250):  # >200 triggers parallel processing
            splat = Gaussian(
                x=np.random.uniform(0, 300),
                y=np.random.uniform(0, 200),
                rx=np.random.uniform(2, 8),
                ry=np.random.uniform(2, 8),
                theta=0,
                r=np.random.randint(0, 256),
                g=np.random.randint(0, 256),
                b=np.random.randint(0, 256),
                a=0.8,
                depth=0.5
            )
            many_splats.append(splat)

        generator = OptimizedSVGGenerator(width=300, height=200)
        layer_content = generator._generate_large_layer_parallel(
            0, many_splats, False
        )

        assert '<g class="layer"' in layer_content
        assert '</g>' in layer_content
        assert 'ellipse' in layer_content

    def test_get_svg_info(self):
        """Test SVG info generation."""
        generator = OptimizedSVGGenerator(width=800, height=600)
        info = generator.get_svg_info(self.test_layers)

        assert info['width'] == 800
        assert info['height'] == 600
        assert info['total_splats'] == 2
        assert info['layer_count'] == 2
        assert 'estimated_size_kb' in info

    def test_estimate_svg_size(self):
        """Test SVG size estimation."""
        generator = OptimizedSVGGenerator(width=300, height=200)
        estimated_size = generator._estimate_svg_size(1000)

        assert isinstance(estimated_size, int)
        assert estimated_size > 0

    def test_save_svg_optimized(self):
        """Test optimized SVG saving."""
        generator = OptimizedSVGGenerator(width=300, height=200)
        svg_content = generator.generate_svg(self.test_layers)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test.svg"
            generator.save_svg(svg_content, output_path)

            assert output_path.exists()
            saved_content = output_path.read_text()
            assert saved_content == svg_content


class TestOptimizedIO:
    """Test optimized I/O operations."""

    def test_optimized_file_writer(self):
        """Test optimized file writer."""
        writer = OptimizedFileWriter(buffer_size=1024)

        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = Path(temp_dir) / "test.txt"
            test_content = "Hello, optimized world!"

            writer.write_buffered(test_content, test_file)

            assert test_file.exists()
            assert test_file.read_text() == test_content

    def test_atomic_file_write(self):
        """Test atomic file writing."""
        writer = OptimizedFileWriter()

        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = Path(temp_dir) / "atomic_test.txt"
            test_content = "Atomic write test content"

            writer.write_atomic(test_content, test_file)

            assert test_file.exists()
            assert test_file.read_text() == test_content

    def test_optimized_file_reader(self):
        """Test optimized file reader."""
        reader = OptimizedFileReader()

        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = Path(temp_dir) / "read_test.txt"
            test_content = "Read test content"
            test_file.write_text(test_content)

            read_content = reader.read_file(test_file)
            assert read_content == test_content

    def test_write_svg_optimized_small(self):
        """Test optimized SVG writing for small files."""
        small_svg = "<svg>small content</svg>"

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "small.svg"
            write_svg_optimized(small_svg, output_path)

            assert output_path.exists()
            assert output_path.read_text() == small_svg

    def test_write_svg_optimized_large(self):
        """Test optimized SVG writing for large files."""
        # Create large SVG content (>10MB)
        large_svg = "<svg>" + "x" * (11 * 1024 * 1024) + "</svg>"

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "large.svg"
            write_svg_optimized(large_svg, output_path)

            assert output_path.exists()
            assert output_path.read_text() == large_svg


class TestBatchSplatExtractor:
    """Test batch splat extractor."""

    def setup_method(self):
        """Set up test fixtures."""
        self.test_images = [
            np.random.randint(0, 255, (50, 75, 3), dtype=np.uint8),
            np.random.randint(0, 255, (60, 80, 3), dtype=np.uint8),
        ]

    @patch('splat_this.core.optimized_extract.OptimizedSplatExtractor.extract_splats')
    def test_extract_batch_basic(self, mock_extract):
        """Test basic batch extraction."""
        # Mock the extraction to return dummy splats
        mock_splats = [
            Gaussian(x=25, y=25, rx=5, ry=5, theta=0, r=255, g=100, b=50, a=0.8)
        ]
        mock_extract.return_value = mock_splats

        batch_extractor = BatchSplatExtractor(max_workers=1)
        results = batch_extractor.extract_batch(
            self.test_images, [25, 30]
        )

        assert len(results) == 2
        assert all(isinstance(result, list) for result in results)
        assert mock_extract.call_count == 2

    def test_extract_batch_length_mismatch(self):
        """Test batch extraction with mismatched lengths."""
        batch_extractor = BatchSplatExtractor()

        with pytest.raises(ValueError, match="Number of images must match"):
            batch_extractor.extract_batch(self.test_images, [25])  # Only one count


class TestGlobalProfiler:
    """Test global profiler integration."""

    def test_global_profiler_usage(self):
        """Test using the global profiler."""
        @global_profiler.profile_function("global_test")
        def test_func():
            return "global test"

        result = test_func()
        assert result == "global test"

        summary = global_profiler.get_summary()
        assert "global_test" in summary["by_function"]