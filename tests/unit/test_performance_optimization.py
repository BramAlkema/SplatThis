"""
Unit tests for Performance Optimization System.

Tests for T4.3: Performance Optimization System implementation.
Comprehensive testing of performance optimization functionality including:
- Spatial acceleration structures (quadtree, spatial hash, grid)
- Efficient covariance matrix computation with caching
- Memory optimization and chunked processing
- Parallel processing optimization
- Performance benchmarking and monitoring
- Adaptive algorithm selection
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import time
import gc

from src.splat_this.core.performance_optimization import (
    PerformanceConfig,
    PerformanceOptimizer,
    SpatialAccelerator,
    EfficientComputation,
    MemoryOptimizer,
    ParallelProcessor,
    PerformanceMetrics,
    AccelerationStructure,
    ComputationMode,
    create_performance_config_preset,
    optimize_splat_operation
)
from src.splat_this.core.adaptive_gaussian import AdaptiveGaussian2D, create_isotropic_gaussian


class TestPerformanceConfig:
    """Test performance configuration validation and creation."""

    def test_init_defaults(self):
        """Test default configuration initialization."""
        config = PerformanceConfig()

        assert config.acceleration_structure == AccelerationStructure.ADAPTIVE
        assert config.computation_mode == ComputationMode.ADAPTIVE
        assert config.enable_covariance_caching is True
        assert config.enable_parallel_processing is True
        assert config.max_memory_mb == 2048
        assert config.spatial_grid_size == 32

    def test_init_custom(self):
        """Test custom configuration initialization."""
        config = PerformanceConfig(
            acceleration_structure=AccelerationStructure.QUADTREE,
            computation_mode=ComputationMode.VECTORIZED,
            max_memory_mb=4096,
            enable_parallel_processing=False
        )

        assert config.acceleration_structure == AccelerationStructure.QUADTREE
        assert config.computation_mode == ComputationMode.VECTORIZED
        assert config.max_memory_mb == 4096
        assert config.enable_parallel_processing is False

    def test_validation_spatial_grid_size(self):
        """Test spatial grid size validation."""
        with pytest.raises(ValueError, match="spatial_grid_size must be at least 4"):
            PerformanceConfig(spatial_grid_size=2)

    def test_validation_cache_size(self):
        """Test cache size validation."""
        with pytest.raises(ValueError, match="cache_size must be at least 16"):
            PerformanceConfig(cache_size=8)

    def test_max_workers_auto_setting(self):
        """Test automatic max workers setting."""
        config = PerformanceConfig()
        assert config.max_workers is not None
        assert config.max_workers > 0


class TestSpatialAccelerator:
    """Test spatial acceleration structures."""

    def test_accelerator_initialization(self):
        """Test spatial accelerator initialization."""
        config = PerformanceConfig()
        accelerator = SpatialAccelerator(config)

        assert accelerator.config == config
        assert accelerator._spatial_index is None
        assert accelerator._bounds is None

    def test_choose_optimal_structure(self):
        """Test optimal structure selection."""
        config = PerformanceConfig(acceleration_structure=AccelerationStructure.ADAPTIVE)
        accelerator = SpatialAccelerator(config)

        # Test structure selection based on splat count
        assert accelerator._choose_optimal_structure(10) == AccelerationStructure.NONE
        assert accelerator._choose_optimal_structure(100) == AccelerationStructure.GRID
        assert accelerator._choose_optimal_structure(1000) == AccelerationStructure.SPATIAL_HASH
        assert accelerator._choose_optimal_structure(5000) == AccelerationStructure.QUADTREE

    def test_build_index_with_splats(self):
        """Test building spatial index with splats."""
        config = PerformanceConfig(acceleration_structure=AccelerationStructure.GRID)
        accelerator = SpatialAccelerator(config)

        # Create test splats
        splats = [
            create_isotropic_gaussian(center=np.array([0.2, 0.3]), scale=0.1, color=np.array([1, 0, 0])),
            create_isotropic_gaussian(center=np.array([0.7, 0.8]), scale=0.1, color=np.array([0, 1, 0])),
            create_isotropic_gaussian(center=np.array([0.5, 0.5]), scale=0.1, color=np.array([0, 0, 1]))
        ]

        accelerator.build_index(splats)

        assert accelerator._spatial_index is not None
        assert accelerator._bounds is not None

    def test_build_quadtree_index(self):
        """Test quadtree index building."""
        config = PerformanceConfig(
            acceleration_structure=AccelerationStructure.QUADTREE,
            quadtree_max_depth=4,
            quadtree_max_items=2
        )
        accelerator = SpatialAccelerator(config)

        # Create test splats in different regions
        splats = []
        for i in range(10):
            x = (i % 3) * 0.3 + 0.1
            y = (i // 3) * 0.3 + 0.1
            splats.append(create_isotropic_gaussian(
                center=np.array([x, y]), scale=0.05, color=np.array([1, 0, 0])
            ))

        accelerator.build_index(splats)

        assert accelerator._spatial_index is not None
        assert hasattr(accelerator._spatial_index, 'query_region')

    def test_build_spatial_hash_index(self):
        """Test spatial hash index building."""
        config = PerformanceConfig(
            acceleration_structure=AccelerationStructure.SPATIAL_HASH,
            spatial_hash_cell_size=0.2
        )
        accelerator = SpatialAccelerator(config)

        splats = [
            create_isotropic_gaussian(center=np.array([0.1, 0.1]), scale=0.05, color=np.array([1, 0, 0])),
            create_isotropic_gaussian(center=np.array([0.3, 0.3]), scale=0.05, color=np.array([0, 1, 0])),
            create_isotropic_gaussian(center=np.array([0.7, 0.7]), scale=0.05, color=np.array([0, 0, 1]))
        ]

        accelerator.build_index(splats)

        assert accelerator._spatial_index is not None
        assert isinstance(accelerator._spatial_index, dict)

    def test_query_region(self):
        """Test spatial region querying."""
        config = PerformanceConfig(acceleration_structure=AccelerationStructure.GRID)
        accelerator = SpatialAccelerator(config)

        # Create test splats
        splats = [
            create_isotropic_gaussian(center=np.array([0.2, 0.2]), scale=0.05, color=np.array([1, 0, 0])),
            create_isotropic_gaussian(center=np.array([0.8, 0.8]), scale=0.05, color=np.array([0, 1, 0])),
            create_isotropic_gaussian(center=np.array([0.5, 0.5]), scale=0.05, color=np.array([0, 0, 1]))
        ]

        accelerator.build_index(splats)

        # Query region that should contain first splat
        results = accelerator.query_region((0.1, 0.1, 0.3, 0.3))

        assert len(results) >= 1
        # Check that results contain splat-index pairs
        for splat, idx in results:
            assert isinstance(splat, AdaptiveGaussian2D)
            assert isinstance(idx, int)

    def test_empty_splats_handling(self):
        """Test handling of empty splat list."""
        config = PerformanceConfig()
        accelerator = SpatialAccelerator(config)

        # Should not crash with empty list
        accelerator.build_index([])

        results = accelerator.query_region((0, 0, 1, 1))
        assert results == []


class TestEfficientComputation:
    """Test efficient computation utilities."""

    def test_computation_initialization(self):
        """Test efficient computation initialization."""
        config = PerformanceConfig()
        computation = EfficientComputation(config)

        assert computation.config == config
        assert computation._covariance_cache == {}
        assert 'cache_hits' in computation._computation_stats

    def test_cached_covariance_computation(self):
        """Test cached covariance matrix computation."""
        config = PerformanceConfig()
        computation = EfficientComputation(config)

        inv_s = (5.0, 10.0)
        theta = np.pi / 4

        # First call should compute
        cov1, inv_cov1 = computation._cached_covariance_computation(inv_s, theta)

        # Second call should use cache
        cov2, inv_cov2 = computation._cached_covariance_computation(inv_s, theta)

        # Results should be identical
        np.testing.assert_array_almost_equal(cov1, cov2)
        np.testing.assert_array_almost_equal(inv_cov1, inv_cov2)

        # Check shapes
        assert cov1.shape == (2, 2)
        assert inv_cov1.shape == (2, 2)

    def test_compute_covariance_matrices_small_batch(self):
        """Test covariance computation for small batch."""
        config = PerformanceConfig(
            computation_mode=ComputationMode.CACHED,
            vectorization_threshold=100
        )
        computation = EfficientComputation(config)

        splats = [
            create_isotropic_gaussian(center=np.array([0.5, 0.5]), scale=0.1, color=np.array([1, 0, 0])),
            create_isotropic_gaussian(center=np.array([0.3, 0.7]), scale=0.05, color=np.array([0, 1, 0]))
        ]

        covariances, inv_covariances = computation.compute_covariance_matrices(splats)

        assert covariances.shape == (2, 2, 2)
        assert inv_covariances.shape == (2, 2, 2)
        assert np.all(np.isfinite(covariances))
        assert np.all(np.isfinite(inv_covariances))

    def test_compute_covariance_matrices_vectorized(self):
        """Test vectorized covariance computation."""
        config = PerformanceConfig(
            computation_mode=ComputationMode.VECTORIZED,
            vectorization_threshold=5
        )
        computation = EfficientComputation(config)

        # Create enough splats to trigger vectorization
        splats = []
        for i in range(10):
            splats.append(create_isotropic_gaussian(
                center=np.array([0.1 * i, 0.1 * i]), scale=0.05, color=np.array([1, 0, 0])
            ))

        covariances, inv_covariances = computation.compute_covariance_matrices(splats)

        assert covariances.shape == (10, 2, 2)
        assert inv_covariances.shape == (10, 2, 2)
        assert computation._computation_stats['vectorized_operations'] > 0

    def test_computation_stats(self):
        """Test computation statistics tracking."""
        config = PerformanceConfig()
        computation = EfficientComputation(config)

        splats = [
            create_isotropic_gaussian(center=np.array([0.5, 0.5]), scale=0.1, color=np.array([1, 0, 0]))
        ]

        # Compute twice to test cache statistics
        computation.compute_covariance_matrices(splats)
        computation.compute_covariance_matrices(splats)

        stats = computation.get_computation_stats()

        assert 'cache_hit_rate' in stats
        assert 'total_operations' in stats
        assert 'vectorized_operations' in stats
        assert stats['total_operations'] >= 2


class TestMemoryOptimizer:
    """Test memory optimization utilities."""

    def test_memory_optimizer_initialization(self):
        """Test memory optimizer initialization."""
        config = PerformanceConfig()
        optimizer = MemoryOptimizer(config)

        assert optimizer.config == config
        assert 'peak_usage_mb' in optimizer.memory_stats

    def test_process_in_chunks_small_batch(self):
        """Test chunked processing with small batch."""
        config = PerformanceConfig(chunk_size=5)
        optimizer = MemoryOptimizer(config)

        splats = [
            create_isotropic_gaussian(center=np.array([0.1 * i, 0.1 * i]), scale=0.05, color=np.array([1, 0, 0]))
            for i in range(3)
        ]

        def dummy_operation(splat_chunk):
            return len(splat_chunk)

        results = optimizer.process_in_chunks(splats, dummy_operation)

        assert len(results) == 1
        assert results[0] == 3

    def test_process_in_chunks_large_batch(self):
        """Test chunked processing with large batch."""
        config = PerformanceConfig(chunk_size=5)
        optimizer = MemoryOptimizer(config)

        splats = [
            create_isotropic_gaussian(center=np.array([0.1 * i, 0.1 * i]), scale=0.05, color=np.array([1, 0, 0]))
            for i in range(12)
        ]

        def dummy_operation(splat_chunk):
            return len(splat_chunk)

        results = optimizer.process_in_chunks(splats, dummy_operation)

        # Should have multiple chunks
        assert len(results) > 1
        assert sum(results) == 12  # Total should equal original count

    @patch('psutil.Process')
    def test_memory_monitoring(self, mock_process):
        """Test memory usage monitoring."""
        mock_memory_info = Mock()
        mock_memory_info.rss = 1024 * 1024 * 1024  # 1GB
        mock_process.return_value.memory_info.return_value = mock_memory_info

        config = PerformanceConfig(max_memory_mb=512, enable_memory_monitoring=True)
        optimizer = MemoryOptimizer(config)

        # This should trigger memory warning
        optimizer._check_memory_usage("test_operation")

        assert optimizer.memory_stats['peak_usage_mb'] > 0

    def test_memory_stats(self):
        """Test memory statistics collection."""
        config = PerformanceConfig()
        optimizer = MemoryOptimizer(config)

        stats = optimizer.get_memory_stats()

        assert 'peak_usage_mb' in stats
        assert 'gc_collections' in stats
        assert 'chunks_processed' in stats


class TestParallelProcessor:
    """Test parallel processing utilities."""

    def test_parallel_processor_initialization(self):
        """Test parallel processor initialization."""
        config = PerformanceConfig()
        processor = ParallelProcessor(config)

        assert processor.config == config
        assert 'parallel_operations' in processor.processing_stats

    def test_process_parallel_small_batch(self):
        """Test parallel processing with small batch (should use sequential)."""
        config = PerformanceConfig(parallel_threshold=10)
        processor = ParallelProcessor(config)

        splats = [
            create_isotropic_gaussian(center=np.array([0.1 * i, 0.1 * i]), scale=0.05, color=np.array([1, 0, 0]))
            for i in range(5)
        ]

        def dummy_operation(splat_chunk):
            return len(splat_chunk)

        results = processor.process_parallel(splats, dummy_operation)

        assert len(results) == 1
        assert results[0] == 5
        assert processor.processing_stats['sequential_operations'] > 0

    def test_process_parallel_large_batch(self):
        """Test parallel processing with large batch."""
        config = PerformanceConfig(parallel_threshold=5, max_workers=2)
        processor = ParallelProcessor(config)

        splats = [
            create_isotropic_gaussian(center=np.array([0.1 * i, 0.1 * i]), scale=0.05, color=np.array([1, 0, 0]))
            for i in range(20)
        ]

        def dummy_operation(splat_chunk):
            return len(splat_chunk)

        results = processor.process_parallel(splats, dummy_operation)

        # Should use parallel processing
        assert len(results) > 1
        assert sum(results) == 20

    def test_processing_stats(self):
        """Test processing statistics collection."""
        config = PerformanceConfig()
        processor = ParallelProcessor(config)

        stats = processor.get_processing_stats()

        assert 'parallel_ratio' in stats
        assert 'total_operations' in stats
        assert 'avg_threads_per_operation' in stats


class TestPerformanceOptimizer:
    """Test main performance optimizer."""

    def test_optimizer_initialization(self):
        """Test performance optimizer initialization."""
        config = PerformanceConfig()
        optimizer = PerformanceOptimizer(config)

        assert optimizer.config == config
        assert optimizer.spatial_accelerator is not None
        assert optimizer.efficient_computation is not None
        assert optimizer.memory_optimizer is not None
        assert optimizer.parallel_processor is not None

    def test_optimize_splat_operations_small_batch(self):
        """Test optimization for small batch."""
        config = PerformanceConfig(small_batch_threshold=10)
        optimizer = PerformanceOptimizer(config)

        splats = [
            create_isotropic_gaussian(center=np.array([0.5, 0.5]), scale=0.1, color=np.array([1, 0, 0])),
            create_isotropic_gaussian(center=np.array([0.3, 0.7]), scale=0.05, color=np.array([0, 1, 0]))
        ]

        def dummy_operation(splat_list):
            return len(splat_list)

        result = optimizer.optimize_splat_operations(splats, dummy_operation)

        assert result == 2
        assert len(optimizer.optimization_history) > 0

    def test_optimize_splat_operations_medium_batch(self):
        """Test optimization for medium batch."""
        config = PerformanceConfig(
            small_batch_threshold=5,
            medium_batch_threshold=20
        )
        optimizer = PerformanceOptimizer(config)

        splats = [
            create_isotropic_gaussian(center=np.array([0.1 * i, 0.1 * i]), scale=0.05, color=np.array([1, 0, 0]))
            for i in range(10)
        ]

        def dummy_operation(splat_list):
            return len(splat_list)

        result = optimizer.optimize_splat_operations(splats, dummy_operation)

        assert result == 10

    def test_performance_metrics_recording(self):
        """Test performance metrics recording."""
        config = PerformanceConfig()
        optimizer = PerformanceOptimizer(config)

        splats = [create_isotropic_gaussian(center=np.array([0.5, 0.5]), scale=0.1, color=np.array([1, 0, 0]))]

        def dummy_operation(splat_list):
            time.sleep(0.01)  # Small delay for timing
            return len(splat_list)

        optimizer.optimize_splat_operations(splats, dummy_operation, operation_name="test_op")

        assert len(optimizer.optimization_history) == 1
        metrics = optimizer.optimization_history[0]
        assert metrics.operation_name == "test_op"
        assert metrics.duration > 0
        assert metrics.splat_count == 1

    def test_comprehensive_stats(self):
        """Test comprehensive statistics collection."""
        config = PerformanceConfig()
        optimizer = PerformanceOptimizer(config)

        stats = optimizer.get_comprehensive_stats()

        assert 'optimization_history' in stats
        assert 'computation_stats' in stats
        assert 'memory_stats' in stats
        assert 'processing_stats' in stats
        assert 'total_operations' in stats

    def test_benchmark_operations(self):
        """Test operation benchmarking."""
        config = PerformanceConfig(benchmark_iterations=2)
        optimizer = PerformanceOptimizer(config)

        splats = [
            create_isotropic_gaussian(center=np.array([0.5, 0.5]), scale=0.1, color=np.array([1, 0, 0]))
        ]

        operations = {
            'operation1': lambda x: len(x),
            'operation2': lambda x: sum(1 for _ in x)
        }

        results = optimizer.benchmark_operations(splats, operations)

        assert 'operation1' in results
        assert 'operation2' in results

        for op_name, result in results.items():
            assert 'avg_time' in result
            assert 'min_time' in result
            assert 'max_time' in result
            assert 'iterations' in result


class TestConvenienceFunctions:
    """Test convenience functions and presets."""

    def test_create_performance_config_presets(self):
        """Test performance configuration preset creation."""
        presets = ["minimal", "balanced", "performance", "memory_efficient"]

        for preset in presets:
            config = create_performance_config_preset(preset)
            assert isinstance(config, PerformanceConfig)

            # Check that each preset has different characteristics
            if preset == "minimal":
                assert config.acceleration_structure == AccelerationStructure.NONE
                assert config.enable_parallel_processing is False
            elif preset == "performance":
                assert config.acceleration_structure == AccelerationStructure.QUADTREE
                assert config.computation_mode == ComputationMode.VECTORIZED
            elif preset == "memory_efficient":
                assert config.chunk_size == 128
                assert config.enable_memory_monitoring is True

    def test_create_performance_config_presets_invalid(self):
        """Test invalid preset handling."""
        with pytest.raises(ValueError, match="Unknown preset"):
            create_performance_config_preset("invalid_preset")

    def test_optimize_splat_operation_convenience(self):
        """Test convenience function for operation optimization."""
        splats = [
            create_isotropic_gaussian(center=np.array([0.5, 0.5]), scale=0.1, color=np.array([1, 0, 0]))
        ]

        def dummy_operation(splat_list):
            return len(splat_list)

        # Test with default config
        result = optimize_splat_operation(splats, dummy_operation)
        assert result == 1

        # Test with custom config
        custom_config = PerformanceConfig(small_batch_threshold=5)
        result_custom = optimize_splat_operation(splats, dummy_operation, custom_config)
        assert result_custom == 1


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_splat_list(self):
        """Test optimization with empty splat list."""
        config = PerformanceConfig()
        optimizer = PerformanceOptimizer(config)

        def dummy_operation(splat_list):
            return len(splat_list)

        result = optimizer.optimize_splat_operations([], dummy_operation)
        assert result == 0

    def test_single_splat(self):
        """Test optimization with single splat."""
        config = PerformanceConfig()
        optimizer = PerformanceOptimizer(config)

        splat = create_isotropic_gaussian(center=np.array([0.5, 0.5]), scale=0.1, color=np.array([1, 0, 0]))

        def dummy_operation(splat_list):
            return len(splat_list)

        result = optimizer.optimize_splat_operations([splat], dummy_operation)
        assert result == 1

    def test_operation_failure_fallback(self):
        """Test fallback when optimization fails."""
        config = PerformanceConfig()
        optimizer = PerformanceOptimizer(config)

        splats = [create_isotropic_gaussian(center=np.array([0.5, 0.5]), scale=0.1, color=np.array([1, 0, 0]))]

        def working_operation(splat_list):
            return len(splat_list)

        def failing_optimization(*args, **kwargs):
            raise ValueError("Optimization failed")

        # Mock a failure in optimization but working fallback
        with patch.object(optimizer, '_optimize_medium_batch', side_effect=failing_optimization):
            result = optimizer.optimize_splat_operations(splats, working_operation)
            # Should fallback to the original operation
            assert result == 1

    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        config = PerformanceConfig()
        computation = EfficientComputation(config)

        # Create splat with extreme values
        splat = create_isotropic_gaussian(center=np.array([0.5, 0.5]), scale=0.001, color=np.array([1, 0, 0]))
        splat.inv_s = np.array([1000.0, 0.001])  # Extreme anisotropy

        covariances, inv_covariances = computation.compute_covariance_matrices([splat])

        # Should handle gracefully without NaN/Inf
        assert np.all(np.isfinite(covariances))
        assert np.all(np.isfinite(inv_covariances))

    def test_memory_pressure_handling(self):
        """Test handling of memory pressure situations."""
        config = PerformanceConfig(max_memory_mb=1)  # Very low limit
        optimizer = MemoryOptimizer(config)

        # Should not crash with low memory limit
        stats = optimizer.get_memory_stats()
        assert isinstance(stats, dict)

    def test_parallel_processing_edge_cases(self):
        """Test parallel processing edge cases."""
        config = PerformanceConfig(max_workers=1, parallel_threshold=1)
        processor = ParallelProcessor(config)

        # Very small chunks
        splats = [create_isotropic_gaussian(center=np.array([0.5, 0.5]), scale=0.1, color=np.array([1, 0, 0]))]

        def dummy_operation(splat_chunk):
            return len(splat_chunk)

        # Should fall back to sequential processing for too-small chunks
        results = processor.process_parallel(splats, dummy_operation)
        assert len(results) == 1

    def test_acceleration_structure_edge_cases(self):
        """Test spatial acceleration structure edge cases."""
        config = PerformanceConfig(acceleration_structure=AccelerationStructure.GRID)
        accelerator = SpatialAccelerator(config)

        # Test with splats at exact same position
        splats = [
            create_isotropic_gaussian(center=np.array([0.5, 0.5]), scale=0.1, color=np.array([1, 0, 0])),
            create_isotropic_gaussian(center=np.array([0.5, 0.5]), scale=0.1, color=np.array([0, 1, 0]))
        ]

        accelerator.build_index(splats)
        results = accelerator.query_region((0.4, 0.4, 0.6, 0.6))

        # Should find both splats
        assert len(results) == 2


if __name__ == "__main__":
    pytest.main([__file__])