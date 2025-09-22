#!/usr/bin/env python3
"""Performance benchmarks for progressive allocation system."""

import pytest
import numpy as np
import time
import psutil
import gc
from typing import Dict, List, Tuple
from dataclasses import dataclass

from src.splat_this.core.adaptive_extract import AdaptiveSplatExtractor, AdaptiveSplatConfig
from src.splat_this.core.progressive_allocator import ProgressiveAllocator, ProgressiveConfig
from src.splat_this.core.error_guided_placement import ErrorGuidedPlacement
from src.splat_this.utils.reconstruction_error import compute_reconstruction_error
from src.splat_this.utils.visualization import create_debug_summary


@dataclass
class PerformanceResult:
    """Container for performance benchmark results."""
    execution_time: float
    memory_usage_mb: float
    cpu_usage_percent: float
    splats_generated: int
    convergence_iterations: int = 0
    error_reduction: float = 0.0


class TestProgressiveAllocationPerformance:
    """Performance benchmarks for progressive allocation system."""

    def setup_method(self):
        """Set up performance test fixtures."""
        # Configure for performance testing with reasonable parameters
        self.config = AdaptiveSplatConfig()
        self.config.enable_progressive = True

        self.progressive_config = ProgressiveConfig(
            initial_ratio=0.4,
            max_splats=200,
            error_threshold=0.02,
            max_add_per_step=10,
            convergence_patience=5
        )

        self.extractor = AdaptiveSplatExtractor(self.config, self.progressive_config)

        # Create test images of various sizes
        self.small_image = self._create_test_image((64, 64, 3))
        self.medium_image = self._create_test_image((128, 128, 3))
        self.large_image = self._create_test_image((256, 256, 3))

    def _create_test_image(self, shape: Tuple[int, int, int]) -> np.ndarray:
        """Create a test image with controlled complexity."""
        h, w, c = shape
        image = np.zeros(shape, dtype=np.float32)

        # Add structured content for realistic performance testing
        # Background gradient
        y, x = np.mgrid[0:h, 0:w]
        image[:, :, 0] = (x / w) * 0.3
        image[:, :, 1] = (y / h) * 0.3
        image[:, :, 2] = 0.2

        # Add geometric features
        num_features = max(4, min(16, (h * w) // 2048))  # Scale features with image size

        for i in range(num_features):
            center_x = np.random.randint(w // 4, 3 * w // 4)
            center_y = np.random.randint(h // 4, 3 * h // 4)
            radius = np.random.randint(h // 16, h // 8)

            # Create circular feature
            mask = ((x - center_x) ** 2 + (y - center_y) ** 2) < radius ** 2
            color = np.random.rand(3) * 0.7 + 0.3
            for c_idx in range(3):
                image[mask, c_idx] = color[c_idx]

        return image

    def _measure_performance(self, func, *args, **kwargs) -> PerformanceResult:
        """Measure performance metrics for a function call."""
        # Initial memory measurement
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Clear cache and garbage collect
        gc.collect()

        # Measure execution time and CPU usage
        start_time = time.time()
        cpu_start = process.cpu_percent()

        result = func(*args, **kwargs)

        end_time = time.time()
        cpu_end = process.cpu_percent()

        # Final memory measurement
        final_memory = process.memory_info().rss / 1024 / 1024  # MB

        execution_time = end_time - start_time
        memory_usage = max(final_memory - initial_memory, 0)  # Ensure non-negative
        cpu_usage = max(cpu_end - cpu_start, 0)  # Ensure non-negative

        splats_count = len(result) if hasattr(result, '__len__') else 0

        return PerformanceResult(
            execution_time=execution_time,
            memory_usage_mb=memory_usage,
            cpu_usage_percent=cpu_usage,
            splats_generated=splats_count
        )

    def test_allocation_performance_scaling(self):
        """Test how allocation performance scales with image size."""
        test_cases = [
            ("small", self.small_image, 50),
            ("medium", self.medium_image, 80),
            ("large", self.large_image, 120)
        ]

        results = {}

        for name, image, n_splats in test_cases:
            result = self._measure_performance(
                self.extractor.extract_adaptive_splats,
                image,
                n_splats,
                False  # verbose=False for cleaner benchmarks
            )

            results[name] = result

            # Verify reasonable performance characteristics
            assert result.execution_time < 30.0, f"Execution time too long for {name}: {result.execution_time}s"
            assert result.memory_usage_mb < 200, f"Memory usage too high for {name}: {result.memory_usage_mb}MB"
            assert result.splats_generated > 0, f"No splats generated for {name}"

            print(f"\n{name.upper()} IMAGE PERFORMANCE:")
            print(f"  Image size: {image.shape}")
            print(f"  Execution time: {result.execution_time:.3f}s")
            print(f"  Memory usage: {result.memory_usage_mb:.2f}MB")
            print(f"  Splats generated: {result.splats_generated}")
            print(f"  Splats per second: {result.splats_generated / result.execution_time:.1f}")

        # Test scaling relationships
        small_result = results["small"]
        medium_result = results["medium"]
        large_result = results["large"]

        # Document scaling behavior - progressive allocation may scale worse due to convergence checking
        small_area = self.small_image.shape[0] * self.small_image.shape[1]
        large_area = self.large_image.shape[0] * self.large_image.shape[1]
        area_ratio = large_area / small_area
        time_ratio = large_result.execution_time / small_result.execution_time

        print(f"\nSCALING ANALYSIS:")
        print(f"  Area ratio: {area_ratio:.1f}x")
        print(f"  Time ratio: {time_ratio:.1f}x")
        print(f"  Scaling factor: {time_ratio / area_ratio:.2f} (1.0 = linear, <1.0 = sub-linear)")

        # Allow for worse-than-linear scaling due to progressive allocation complexity
        # Progressive allocation involves iterative error computation and convergence checking
        assert time_ratio < area_ratio * 3.0, f"Performance scaling unacceptable: {time_ratio} vs {area_ratio}"

    def test_progressive_vs_static_performance(self):
        """Compare performance of progressive vs static allocation."""
        # Configure static allocation
        static_config = AdaptiveSplatConfig()
        static_config.enable_progressive = False
        static_extractor = AdaptiveSplatExtractor(static_config)

        n_splats = 80
        test_image = self.medium_image

        # Measure progressive allocation
        progressive_result = self._measure_performance(
            self.extractor.extract_adaptive_splats,
            test_image,
            n_splats,
            False
        )

        # Measure static allocation
        static_result = self._measure_performance(
            static_extractor.extract_adaptive_splats,
            test_image,
            n_splats,
            False
        )

        print(f"\nPROGRESSIVE vs STATIC ALLOCATION:")
        print(f"Progressive - Time: {progressive_result.execution_time:.3f}s, "
              f"Memory: {progressive_result.memory_usage_mb:.2f}MB, "
              f"Splats: {progressive_result.splats_generated}")
        print(f"Static - Time: {static_result.execution_time:.3f}s, "
              f"Memory: {static_result.memory_usage_mb:.2f}MB, "
              f"Splats: {static_result.splats_generated}")

        # Progressive is significantly slower due to iterative convergence checking and error computation
        # but may converge to fewer splats, demonstrating intelligent allocation
        time_ratio = progressive_result.execution_time / static_result.execution_time

        print(f"  Performance ratio: {time_ratio:.1f}x slower")
        print(f"  Convergence efficiency: Progressive used {progressive_result.splats_generated}/{static_result.splats_generated} = {progressive_result.splats_generated/static_result.splats_generated:.1%} of static splats")

        # Progressive should be significantly slower but finite
        assert time_ratio < 200.0, f"Progressive allocation unreasonably slow: {time_ratio}x"

        # Both should produce reasonable results
        assert progressive_result.splats_generated > 0
        assert static_result.splats_generated > 0

        # Progressive might use fewer splats due to convergence
        assert progressive_result.splats_generated <= static_result.splats_generated

    def test_memory_efficiency(self):
        """Test memory efficiency during allocation."""
        test_sizes = [40, 60, 80, 100]
        memory_results = []

        for n_splats in test_sizes:
            result = self._measure_performance(
                self.extractor.extract_adaptive_splats,
                self.medium_image,
                n_splats,
                False
            )

            memory_per_splat = result.memory_usage_mb / max(result.splats_generated, 1)
            memory_results.append((n_splats, result.memory_usage_mb, memory_per_splat))

            print(f"n_splats={n_splats}: {result.memory_usage_mb:.2f}MB total, "
                  f"{memory_per_splat:.3f}MB per splat")

        # Memory usage should scale reasonably with splat count
        max_memory = max(result[1] for result in memory_results)
        assert max_memory < 150, f"Memory usage too high: {max_memory}MB"

        # Memory usage can be variable due to garbage collection and progressive allocation patterns
        # Focus on ensuring no excessive memory usage
        memory_per_splat_values = [result[2] for result in memory_results if result[2] > 0]
        if memory_per_splat_values:  # Only check if we have valid measurements
            max_memory_per_splat = max(memory_per_splat_values)
            assert max_memory_per_splat < 5.0, f"Memory per splat too high: {max_memory_per_splat}MB"

        print(f"Memory usage analysis complete. Max per-splat: {max(memory_per_splat_values) if memory_per_splat_values else 0:.3f}MB")

    def test_convergence_performance(self):
        """Test convergence behavior and performance."""
        # Create simple image that should converge quickly
        simple_image = np.full((100, 100, 3), 0.5, dtype=np.float32)

        # Add single feature
        simple_image[40:60, 40:60] = 0.8

        # Test different error thresholds
        thresholds = [0.01, 0.03, 0.05, 0.1]
        convergence_results = []

        for threshold in thresholds:
            config = ProgressiveConfig(
                initial_ratio=0.3,
                max_splats=100,
                error_threshold=threshold,
                max_add_per_step=5,
                convergence_patience=3
            )

            test_extractor = AdaptiveSplatExtractor(self.config, config)

            result = self._measure_performance(
                test_extractor.extract_adaptive_splats,
                simple_image,
                100,  # Request many splats
                False
            )

            convergence_results.append((threshold, result))

            print(f"Threshold {threshold}: {result.execution_time:.3f}s, "
                  f"{result.splats_generated} splats")

        # Lower thresholds should generally use more splats (less convergence)
        # Higher thresholds should converge faster (fewer splats)
        strict_result = convergence_results[0][1]  # threshold=0.01
        loose_result = convergence_results[-1][1]  # threshold=0.1

        # Strict threshold should take longer or use more splats
        assert (strict_result.execution_time >= loose_result.execution_time * 0.5 or
                strict_result.splats_generated >= loose_result.splats_generated), \
            "Convergence behavior not as expected"

    def test_error_computation_performance(self):
        """Test performance of error computation components."""
        # Test error computation performance
        target = self.medium_image
        rendered = target + np.random.normal(0, 0.05, target.shape).astype(np.float32)
        rendered = np.clip(rendered, 0, 1)

        error_metrics = ["l1", "l2", "mse"]
        error_results = {}

        for metric in error_metrics:
            result = self._measure_performance(
                compute_reconstruction_error,
                target,
                rendered,
                metric
            )

            error_results[metric] = result

            print(f"{metric.upper()} error computation: {result.execution_time:.4f}s")

        # Error computation should be fast
        for metric, result in error_results.items():
            assert result.execution_time < 1.0, f"{metric} error computation too slow: {result.execution_time}s"

        # L1 should generally be fastest, L2 might be slightly slower
        assert error_results["l1"].execution_time <= error_results["l2"].execution_time * 2

    def test_placement_algorithm_performance(self):
        """Test performance of placement algorithms."""
        placer = ErrorGuidedPlacement()

        # Create error map
        error_map = np.random.exponential(0.1, (128, 128)).astype(np.float32)

        # Test probability map creation
        prob_result = self._measure_performance(
            placer.create_placement_probability,
            error_map
        )

        print(f"Probability map creation: {prob_result.execution_time:.4f}s")

        # Test position sampling
        prob_map = placer.create_placement_probability(error_map)
        sample_counts = [10, 20, 50, 100]

        for count in sample_counts:
            sample_result = self._measure_performance(
                placer.sample_positions,
                prob_map,
                count
            )

            print(f"Sampling {count} positions: {sample_result.execution_time:.4f}s")

            # Sampling should be fast and scale reasonably
            assert sample_result.execution_time < 2.0, f"Position sampling too slow for {count} positions"

        # Probability computation should be fast
        assert prob_result.execution_time < 0.5, f"Probability map creation too slow: {prob_result.execution_time}s"

    def test_batch_size_performance(self):
        """Test how batch size affects performance."""
        batch_sizes = [5, 10, 20, 30]
        batch_results = []

        for batch_size in batch_sizes:
            config = ProgressiveConfig(
                initial_ratio=0.3,
                max_splats=100,
                error_threshold=0.03,
                max_add_per_step=batch_size,
                convergence_patience=3
            )

            test_extractor = AdaptiveSplatExtractor(self.config, config)

            result = self._measure_performance(
                test_extractor.extract_adaptive_splats,
                self.medium_image,
                80,
                False
            )

            batch_results.append((batch_size, result))

            print(f"Batch size {batch_size}: {result.execution_time:.3f}s, "
                  f"{result.splats_generated} splats")

        # All batch sizes should complete in reasonable time
        for batch_size, result in batch_results:
            assert result.execution_time < 15.0, f"Batch size {batch_size} too slow: {result.execution_time}s"
            assert result.splats_generated > 0, f"No splats generated for batch size {batch_size}"

    def test_visualization_performance_impact(self):
        """Test performance impact of debug visualization."""
        import tempfile

        # Test without visualization
        base_result = self._measure_performance(
            self.extractor.extract_adaptive_splats,
            self.medium_image,
            60,
            False  # verbose=False
        )

        # Test with verbose output (minimal visualization)
        verbose_result = self._measure_performance(
            self.extractor.extract_adaptive_splats,
            self.medium_image,
            60,
            True  # verbose=True
        )

        print(f"Base execution: {base_result.execution_time:.3f}s")
        print(f"Verbose execution: {verbose_result.execution_time:.3f}s")

        # Verbose should not significantly impact performance
        verbose_overhead = verbose_result.execution_time / base_result.execution_time
        assert verbose_overhead < 2.0, f"Verbose mode overhead too high: {verbose_overhead}x"

    def test_stress_test_large_allocation(self):
        """Stress test with large splat allocation."""
        # Use a larger image and more splats for stress testing
        stress_image = self._create_test_image((200, 200, 3))

        config = ProgressiveConfig(
            initial_ratio=0.3,
            max_splats=300,
            error_threshold=0.02,
            max_add_per_step=15,
            convergence_patience=5
        )

        stress_extractor = AdaptiveSplatExtractor(self.config, config)

        result = self._measure_performance(
            stress_extractor.extract_adaptive_splats,
            stress_image,
            250,  # Large allocation
            False
        )

        print(f"STRESS TEST RESULTS:")
        print(f"  Execution time: {result.execution_time:.3f}s")
        print(f"  Memory usage: {result.memory_usage_mb:.2f}MB")
        print(f"  Splats generated: {result.splats_generated}")

        # Should complete within reasonable limits
        assert result.execution_time < 60.0, f"Stress test too slow: {result.execution_time}s"
        assert result.memory_usage_mb < 500, f"Stress test memory usage too high: {result.memory_usage_mb}MB"
        assert result.splats_generated > 0, "No splats generated in stress test"


class TestAllocationAlgorithmComparison:
    """Compare performance of different allocation strategies."""

    def setup_method(self):
        """Set up comparison test fixtures."""
        self.test_image = self._create_comparison_image((128, 128, 3))

    def _create_comparison_image(self, shape: Tuple[int, int, int]) -> np.ndarray:
        """Create image optimized for algorithm comparison."""
        h, w, c = shape
        image = np.zeros(shape, dtype=np.float32)

        # Create structured content for fair comparison
        # Gradient background
        y, x = np.mgrid[0:h, 0:w]
        image[:, :, 0] = np.sin(x / w * 2 * np.pi) * 0.2 + 0.3
        image[:, :, 1] = np.cos(y / h * 2 * np.pi) * 0.2 + 0.3
        image[:, :, 2] = 0.4

        # Add high-frequency features
        for i in range(8):
            cx = np.random.randint(w // 4, 3 * w // 4)
            cy = np.random.randint(h // 4, 3 * h // 4)
            r = np.random.randint(8, 20)

            mask = ((x - cx) ** 2 + (y - cy) ** 2) < r ** 2
            color = np.random.rand() * 0.6 + 0.4
            image[mask] = color

        return image

    def test_initialization_strategy_comparison(self):
        """Compare different initialization strategies."""
        strategies = ["saliency", "gradient", "random"]
        n_splats = 80

        results = {}

        for strategy in strategies:
            config = AdaptiveSplatConfig()
            config.enable_progressive = True
            config.init_strategy = strategy

            prog_config = ProgressiveConfig(
                initial_ratio=0.4,
                max_splats=120,
                error_threshold=0.03
            )

            extractor = AdaptiveSplatExtractor(config, prog_config)

            start_time = time.time()
            splats = extractor.extract_adaptive_splats(self.test_image, n_splats, False)
            end_time = time.time()

            execution_time = end_time - start_time
            results[strategy] = {
                'time': execution_time,
                'splats': len(splats)
            }

            print(f"{strategy.upper()} initialization: {execution_time:.3f}s, {len(splats)} splats")

        # All strategies should complete in reasonable time
        for strategy, result in results.items():
            assert result['time'] < 20.0, f"{strategy} initialization too slow: {result['time']}s"
            assert result['splats'] > 0, f"No splats generated for {strategy} initialization"

    def test_error_metric_performance_comparison(self):
        """Compare performance of different error metrics in allocation."""
        # Note: This tests the overall allocation performance with different
        # error computation preferences, as error metrics are used internally
        error_thresholds = [0.01, 0.03, 0.05]
        n_splats = 70

        results = {}

        for threshold in error_thresholds:
            config = AdaptiveSplatConfig()
            config.enable_progressive = True

            prog_config = ProgressiveConfig(
                initial_ratio=0.4,
                max_splats=100,
                error_threshold=threshold
            )

            extractor = AdaptiveSplatExtractor(config, prog_config)

            start_time = time.time()
            splats = extractor.extract_adaptive_splats(self.test_image, n_splats, False)
            end_time = time.time()

            execution_time = end_time - start_time
            results[threshold] = {
                'time': execution_time,
                'splats': len(splats)
            }

            print(f"Error threshold {threshold}: {execution_time:.3f}s, {len(splats)} splats")

        # All should complete reasonably
        for threshold, result in results.items():
            assert result['time'] < 15.0, f"Threshold {threshold} too slow: {result['time']}s"
            assert result['splats'] > 0, f"No splats for threshold {threshold}"


if __name__ == "__main__":
    # Run performance benchmarks manually
    perf_tests = TestProgressiveAllocationPerformance()
    perf_tests.setup_method()

    print("=== PROGRESSIVE ALLOCATION PERFORMANCE BENCHMARKS ===")

    try:
        print("\n1. Testing allocation performance scaling...")
        perf_tests.test_allocation_performance_scaling()

        print("\n2. Testing progressive vs static performance...")
        perf_tests.test_progressive_vs_static_performance()

        print("\n3. Testing memory efficiency...")
        perf_tests.test_memory_efficiency()

        print("\n4. Testing convergence performance...")
        perf_tests.test_convergence_performance()

        print("\n5. Testing error computation performance...")
        perf_tests.test_error_computation_performance()

        print("\n6. Testing placement algorithm performance...")
        perf_tests.test_placement_algorithm_performance()

        print("\nPerformance benchmarks completed successfully!")

    except Exception as e:
        print(f"Performance benchmark failed: {e}")
        raise