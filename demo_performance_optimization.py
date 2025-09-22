#!/usr/bin/env python3
"""
Performance Optimization System Demo

Demonstrates the capabilities of the Performance Optimization System (T4.3)
for scalable adaptive Gaussian splatting operations.

This demo showcases:
- Spatial acceleration structures for efficient queries
- Covariance matrix computation optimization with caching
- Memory optimization for large splat counts
- Parallel processing for batch operations
- Performance benchmarking and monitoring
- Adaptive algorithm selection based on data characteristics

The performance optimization system improves scalability by:
1. Using spatial acceleration structures (quadtree, spatial hash, grid)
2. Caching expensive covariance matrix computations
3. Processing operations in memory-efficient chunks
4. Leveraging parallel processing for large batches
5. Monitoring and profiling performance characteristics
6. Adaptively selecting optimal algorithms

Usage:
    python demo_performance_optimization.py [--preset PRESET] [--verbose] [--benchmark]

    PRESET options: minimal, balanced, performance, memory_efficient
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import time
from typing import List, Tuple, Dict, Any
import logging

from src.splat_this.core.performance_optimization import (
    PerformanceOptimizer,
    PerformanceConfig,
    SpatialAccelerator,
    EfficientComputation,
    AccelerationStructure,
    ComputationMode,
    create_performance_config_preset,
    optimize_splat_operation
)
from src.splat_this.core.adaptive_gaussian import AdaptiveGaussian2D, create_isotropic_gaussian


def setup_logging(verbose: bool = False):
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def create_test_splats(count: int, distribution: str = "random") -> List[AdaptiveGaussian2D]:
    """Create test splats with different spatial distributions."""
    print(f"🎯 Creating {count} test splats with {distribution} distribution...")

    splats = []

    if distribution == "random":
        # Random distribution across the space
        for i in range(count):
            x = np.random.uniform(0.1, 0.9)
            y = np.random.uniform(0.1, 0.9)
            scale = np.random.uniform(0.02, 0.08)
            color = np.random.uniform(0.2, 1.0, 3)
            alpha = np.random.uniform(0.6, 0.9)

            splat = create_isotropic_gaussian(
                center=np.array([x, y]),
                scale=scale,
                color=color,
                alpha=alpha
            )
            splats.append(splat)

    elif distribution == "clustered":
        # Create clusters of splats
        num_clusters = max(1, count // 20)
        cluster_centers = [(np.random.uniform(0.2, 0.8), np.random.uniform(0.2, 0.8))
                          for _ in range(num_clusters)]

        for i in range(count):
            # Choose a cluster
            cluster_x, cluster_y = cluster_centers[i % num_clusters]

            # Add some spread around cluster center
            x = cluster_x + np.random.normal(0, 0.1)
            y = cluster_y + np.random.normal(0, 0.1)
            x = np.clip(x, 0.05, 0.95)
            y = np.clip(y, 0.05, 0.95)

            scale = np.random.uniform(0.02, 0.06)
            color = np.random.uniform(0.3, 1.0, 3)
            alpha = np.random.uniform(0.7, 0.9)

            splat = create_isotropic_gaussian(
                center=np.array([x, y]),
                scale=scale,
                color=color,
                alpha=alpha
            )
            splats.append(splat)

    elif distribution == "grid":
        # Grid-like distribution
        grid_size = int(np.sqrt(count)) + 1
        for i in range(count):
            grid_x = (i % grid_size) / (grid_size - 1) if grid_size > 1 else 0.5
            grid_y = (i // grid_size) / (grid_size - 1) if grid_size > 1 else 0.5

            # Add small random offset
            x = grid_x + np.random.uniform(-0.05, 0.05)
            y = grid_y + np.random.uniform(-0.05, 0.05)
            x = np.clip(x, 0.05, 0.95)
            y = np.clip(y, 0.05, 0.95)

            scale = np.random.uniform(0.03, 0.07)
            color = np.random.uniform(0.2, 1.0, 3)
            alpha = np.random.uniform(0.6, 0.8)

            splat = create_isotropic_gaussian(
                center=np.array([x, y]),
                scale=scale,
                color=color,
                alpha=alpha
            )
            splats.append(splat)

    print(f"✅ Created {len(splats)} test splats")
    return splats


def benchmark_spatial_acceleration(splat_counts: List[int], query_counts: List[int]) -> Dict[str, Any]:
    """Benchmark spatial acceleration structures."""
    print("\n🔍 SPATIAL ACCELERATION BENCHMARK")
    print("=" * 40)

    results = {
        'splat_counts': splat_counts,
        'query_counts': query_counts,
        'structures': {},
        'scaling_analysis': {}
    }

    structures = [
        AccelerationStructure.NONE,
        AccelerationStructure.GRID,
        AccelerationStructure.SPATIAL_HASH,
        AccelerationStructure.QUADTREE
    ]

    for structure in structures:
        print(f"\n📊 Testing {structure.value.upper()} acceleration...")
        structure_results = {
            'build_times': [],
            'query_times': [],
            'memory_usage': []
        }

        for splat_count in splat_counts:
            # Create test splats
            splats = create_test_splats(splat_count, "clustered")

            # Create accelerator
            config = PerformanceConfig(acceleration_structure=structure)
            accelerator = SpatialAccelerator(config)

            # Benchmark index building
            start_time = time.time()
            accelerator.build_index(splats)
            build_time = time.time() - start_time
            structure_results['build_times'].append(build_time)

            # Benchmark queries
            query_regions = [
                (np.random.uniform(0, 0.5), np.random.uniform(0, 0.5),
                 np.random.uniform(0.5, 1), np.random.uniform(0.5, 1))
                for _ in range(query_counts[0])  # Use first query count for simplicity
            ]

            start_time = time.time()
            total_results = 0
            for region in query_regions:
                results_in_region = accelerator.query_region(region)
                total_results += len(results_in_region)
            query_time = time.time() - start_time

            structure_results['query_times'].append(query_time)

            print(f"   {splat_count:4d} splats: build={build_time*1000:6.2f}ms, "
                  f"query={query_time*1000:6.2f}ms, found={total_results}")

        results['structures'][structure.value] = structure_results

    return results


def benchmark_covariance_computation(splat_counts: List[int]) -> Dict[str, Any]:
    """Benchmark covariance matrix computation methods."""
    print("\n🧮 COVARIANCE COMPUTATION BENCHMARK")
    print("=" * 40)

    results = {
        'splat_counts': splat_counts,
        'methods': {}
    }

    methods = [
        ("standard", ComputationMode.STANDARD),
        ("cached", ComputationMode.CACHED),
        ("vectorized", ComputationMode.VECTORIZED),
    ]

    for method_name, computation_mode in methods:
        print(f"\n📊 Testing {method_name.upper()} computation...")
        method_results = {
            'computation_times': [],
            'cache_hit_rates': []
        }

        for splat_count in splat_counts:
            splats = create_test_splats(splat_count, "random")

            # Create computation engine
            config = PerformanceConfig(
                computation_mode=computation_mode,
                enable_covariance_caching=(computation_mode == ComputationMode.CACHED),
                vectorization_threshold=50
            )
            computation = EfficientComputation(config)

            # Benchmark computation
            start_time = time.time()
            covariances, inv_covariances = computation.compute_covariance_matrices(splats)
            computation_time = time.time() - start_time

            # Compute again to test caching
            start_time_cached = time.time()
            covariances2, inv_covariances2 = computation.compute_covariance_matrices(splats)
            computation_time_cached = time.time() - start_time_cached

            method_results['computation_times'].append(computation_time)

            # Get cache statistics
            stats = computation.get_computation_stats()
            cache_hit_rate = stats.get('cache_hit_rate', 0.0)
            method_results['cache_hit_rates'].append(cache_hit_rate)

            print(f"   {splat_count:4d} splats: time={computation_time*1000:6.2f}ms, "
                  f"cached={computation_time_cached*1000:6.2f}ms, "
                  f"cache_hit_rate={cache_hit_rate:5.1f}%")

        results['methods'][method_name] = method_results

    return results


def benchmark_memory_optimization(splat_counts: List[int]) -> Dict[str, Any]:
    """Benchmark memory optimization techniques."""
    print("\n💾 MEMORY OPTIMIZATION BENCHMARK")
    print("=" * 40)

    results = {
        'splat_counts': splat_counts,
        'peak_memory': [],
        'chunk_processing_times': [],
        'gc_collections': []
    }

    for splat_count in splat_counts:
        splats = create_test_splats(splat_count, "random")

        # Create memory optimizer
        config = PerformanceConfig(
            chunk_size=min(256, splat_count // 4 + 1),
            max_memory_mb=1024,
            enable_memory_monitoring=True
        )

        from src.splat_this.core.performance_optimization import MemoryOptimizer
        memory_optimizer = MemoryOptimizer(config)

        # Define a dummy operation
        def dummy_operation(splat_chunk):
            # Simulate some memory-intensive operation
            data = np.random.rand(len(splat_chunk), 100, 100)
            return np.sum(data)

        # Benchmark chunked processing
        start_time = time.time()
        chunk_results = memory_optimizer.process_in_chunks(splats, dummy_operation)
        processing_time = time.time() - start_time

        # Get memory statistics
        memory_stats = memory_optimizer.get_memory_stats()

        results['chunk_processing_times'].append(processing_time)
        results['peak_memory'].append(memory_stats['peak_usage_mb'])
        results['gc_collections'].append(memory_stats['gc_collections'])

        print(f"   {splat_count:4d} splats: time={processing_time:6.2f}s, "
              f"peak_memory={memory_stats['peak_usage_mb']:6.1f}MB, "
              f"chunks={memory_stats['chunks_processed']}, "
              f"gc={memory_stats['gc_collections']}")

    return results


def benchmark_parallel_processing(splat_counts: List[int]) -> Dict[str, Any]:
    """Benchmark parallel processing performance."""
    print("\n⚡ PARALLEL PROCESSING BENCHMARK")
    print("=" * 40)

    results = {
        'splat_counts': splat_counts,
        'sequential_times': [],
        'parallel_times': [],
        'speedup_ratios': []
    }

    def dummy_computation(splat_list):
        """Simulate computational work."""
        total = 0
        for splat in splat_list:
            # Simulate some computation
            covariance = np.outer(splat.mu, splat.mu)
            eigenvals = np.linalg.eigvals(covariance)
            total += np.sum(eigenvals)
        return total

    for splat_count in splat_counts:
        splats = create_test_splats(splat_count, "random")

        # Sequential processing
        config_sequential = PerformanceConfig(enable_parallel_processing=False)
        from src.splat_this.core.performance_optimization import ParallelProcessor
        processor_sequential = ParallelProcessor(config_sequential)

        start_time = time.time()
        result_sequential = processor_sequential.process_parallel(splats, dummy_computation)
        sequential_time = time.time() - start_time

        # Parallel processing
        config_parallel = PerformanceConfig(
            enable_parallel_processing=True,
            max_workers=4,
            parallel_threshold=max(10, splat_count // 10)
        )
        processor_parallel = ParallelProcessor(config_parallel)

        start_time = time.time()
        result_parallel = processor_parallel.process_parallel(splats, dummy_computation)
        parallel_time = time.time() - start_time

        speedup = sequential_time / max(parallel_time, 0.001)

        results['sequential_times'].append(sequential_time)
        results['parallel_times'].append(parallel_time)
        results['speedup_ratios'].append(speedup)

        print(f"   {splat_count:4d} splats: sequential={sequential_time*1000:6.2f}ms, "
              f"parallel={parallel_time*1000:6.2f}ms, speedup={speedup:4.1f}x")

    return results


def demonstrate_adaptive_optimization(preset_name: str) -> Dict[str, Any]:
    """Demonstrate adaptive optimization for different batch sizes."""
    print(f"\n🔧 ADAPTIVE OPTIMIZATION DEMO - {preset_name.upper()}")
    print("=" * 50)

    config = create_performance_config_preset(preset_name)
    optimizer = PerformanceOptimizer(config)

    print(f"📋 Configuration:")
    print(f"   Acceleration: {config.acceleration_structure.value}")
    print(f"   Computation: {config.computation_mode.value}")
    print(f"   Parallel processing: {config.enable_parallel_processing}")
    print(f"   Memory limit: {config.max_memory_mb}MB")
    print(f"   Cache size: {config.cache_size}")

    def test_operation(splat_list):
        """Test operation that computes some metrics."""
        total_area = 0
        for splat in splat_list:
            # Compute approximate area
            scales = 1.0 / splat.inv_s
            area = np.pi * scales[0] * scales[1]
            total_area += area
        return total_area

    batch_sizes = [10, 50, 200, 1000, 5000]
    results = {
        'batch_sizes': batch_sizes,
        'execution_times': [],
        'optimizations_used': [],
        'memory_efficiency': []
    }

    for batch_size in batch_sizes:
        print(f"\n📊 Testing batch size: {batch_size}")

        splats = create_test_splats(batch_size, "clustered")

        # Run optimized operation
        start_time = time.time()
        result = optimizer.optimize_splat_operations(
            splats, test_operation, operation_name=f"test_batch_{batch_size}"
        )
        execution_time = time.time() - start_time

        results['execution_times'].append(execution_time)

        # Get optimization metrics
        if optimizer.optimization_history:
            latest_metrics = optimizer.optimization_history[-1]
            ops_per_second = latest_metrics.operations_per_second
            memory_efficiency = latest_metrics.memory_efficiency

            # Handle result formatting (could be single value or list)
            if isinstance(result, (list, tuple)):
                result_str = f"[{', '.join(f'{r:.2f}' for r in result[:3])}{'...' if len(result) > 3 else ''}]"
            else:
                result_str = f"{result:.2f}"

            print(f"   Result: {result_str}")
            print(f"   Time: {execution_time*1000:.2f}ms")
            print(f"   Ops/sec: {ops_per_second:.1f}")
            print(f"   Memory efficiency: {memory_efficiency:.2f}MB/1k splats")
            print(f"   Acceleration: {latest_metrics.acceleration_used}")
            print(f"   Computation: {latest_metrics.computation_mode}")

            results['optimizations_used'].append({
                'acceleration': latest_metrics.acceleration_used,
                'computation': latest_metrics.computation_mode,
                'ops_per_second': ops_per_second
            })
            results['memory_efficiency'].append(memory_efficiency)
        else:
            results['optimizations_used'].append({})
            results['memory_efficiency'].append(0.0)

    return results


def create_performance_visualization(benchmark_results: Dict[str, Any]) -> plt.Figure:
    """Create visualization of performance benchmark results."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Spatial acceleration benchmark
    if 'spatial_acceleration' in benchmark_results:
        spatial_data = benchmark_results['spatial_acceleration']
        splat_counts = spatial_data['splat_counts']

        ax = axes[0, 0]
        for structure_name, structure_data in spatial_data['structures'].items():
            if structure_name != 'none':  # Skip none for clarity
                ax.plot(splat_counts, structure_data['build_times'],
                       marker='o', label=f'{structure_name} build')

        ax.set_xlabel('Number of Splats')
        ax.set_ylabel('Build Time (seconds)')
        ax.set_title('Spatial Index Build Performance')
        ax.legend()
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

    # Covariance computation benchmark
    if 'covariance_computation' in benchmark_results:
        cov_data = benchmark_results['covariance_computation']
        splat_counts = cov_data['splat_counts']

        ax = axes[0, 1]
        for method_name, method_data in cov_data['methods'].items():
            ax.plot(splat_counts, method_data['computation_times'],
                   marker='s', label=method_name)

        ax.set_xlabel('Number of Splats')
        ax.set_ylabel('Computation Time (seconds)')
        ax.set_title('Covariance Computation Performance')
        ax.legend()
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

    # Memory optimization benchmark
    if 'memory_optimization' in benchmark_results:
        mem_data = benchmark_results['memory_optimization']
        splat_counts = mem_data['splat_counts']

        ax = axes[0, 2]
        ax.plot(splat_counts, mem_data['peak_memory'],
               marker='^', color='red', label='Peak Memory')
        ax.set_xlabel('Number of Splats')
        ax.set_ylabel('Peak Memory (MB)')
        ax.set_title('Memory Usage')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Parallel processing benchmark
    if 'parallel_processing' in benchmark_results:
        par_data = benchmark_results['parallel_processing']
        splat_counts = par_data['splat_counts']

        ax = axes[1, 0]
        ax.plot(splat_counts, par_data['sequential_times'],
               marker='o', label='Sequential', color='blue')
        ax.plot(splat_counts, par_data['parallel_times'],
               marker='s', label='Parallel', color='green')
        ax.set_xlabel('Number of Splats')
        ax.set_ylabel('Execution Time (seconds)')
        ax.set_title('Sequential vs Parallel Performance')
        ax.legend()
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

    # Speedup ratios
    if 'parallel_processing' in benchmark_results:
        par_data = benchmark_results['parallel_processing']
        splat_counts = par_data['splat_counts']

        ax = axes[1, 1]
        ax.plot(splat_counts, par_data['speedup_ratios'],
               marker='D', color='purple', label='Speedup')
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, label='No speedup')
        ax.set_xlabel('Number of Splats')
        ax.set_ylabel('Speedup Ratio')
        ax.set_title('Parallel Processing Speedup')
        ax.legend()
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)

    # Adaptive optimization comparison
    if 'adaptive_optimization' in benchmark_results:
        adapt_data = benchmark_results['adaptive_optimization']
        batch_sizes = adapt_data['batch_sizes']

        ax = axes[1, 2]
        ax.plot(batch_sizes, adapt_data['execution_times'],
               marker='h', color='orange', label='Execution Time')
        ax.set_xlabel('Batch Size')
        ax.set_ylabel('Execution Time (seconds)')
        ax.set_title('Adaptive Optimization Performance')
        ax.legend()
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def main():
    """Main demonstration function."""
    parser = argparse.ArgumentParser(description='Performance Optimization System Demo')
    parser.add_argument('--preset', choices=['minimal', 'balanced', 'performance', 'memory_efficient'],
                       default='balanced', help='Performance optimization preset to use')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--benchmark', action='store_true', help='Run comprehensive benchmarks')
    parser.add_argument('--quick', action='store_true', help='Run quick demo with smaller datasets')

    args = parser.parse_args()

    setup_logging(args.verbose)

    print("⚡ PERFORMANCE OPTIMIZATION SYSTEM DEMO")
    print("🎯 Adaptive Gaussian Splatting - T4.3 Implementation")
    print("=" * 60)

    try:
        benchmark_results = {}

        if args.quick:
            splat_counts = [10, 50, 200]
            query_counts = [10]
        else:
            splat_counts = [10, 50, 200, 1000, 5000] if args.benchmark else [50, 200, 1000]
            query_counts = [50, 200] if args.benchmark else [50]

        if args.benchmark:
            # Run comprehensive benchmarks
            print("🏃‍♂️ Running comprehensive performance benchmarks...")

            # Spatial acceleration benchmark
            benchmark_results['spatial_acceleration'] = benchmark_spatial_acceleration(
                splat_counts, query_counts
            )

            # Covariance computation benchmark
            benchmark_results['covariance_computation'] = benchmark_covariance_computation(
                splat_counts
            )

            # Memory optimization benchmark
            if not args.quick:
                benchmark_results['memory_optimization'] = benchmark_memory_optimization(
                    splat_counts
                )

            # Parallel processing benchmark
            benchmark_results['parallel_processing'] = benchmark_parallel_processing(
                splat_counts
            )

        # Adaptive optimization demonstration
        adaptive_results = demonstrate_adaptive_optimization(args.preset)
        benchmark_results['adaptive_optimization'] = adaptive_results

        # Create and save visualization if we have benchmark data
        if benchmark_results and len(benchmark_results) > 1:
            print("\n🎨 Creating performance visualization...")
            fig = create_performance_visualization(benchmark_results)

            output_path = f'performance_optimization_{args.preset}.png'
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"💾 Performance visualization saved to: {output_path}")

        print("\n✅ Performance Optimization Demo completed successfully!")
        print("🎯 The system demonstrates scalable optimization with:")
        print("   • Spatial acceleration structures for efficient queries")
        print("   • Cached covariance computation for repeated operations")
        print("   • Memory-efficient chunked processing")
        print("   • Parallel processing for large batch operations")
        print("   • Adaptive algorithm selection based on data size")
        print("   • Comprehensive performance monitoring and profiling")

        # Print final performance summary
        if 'adaptive_optimization' in benchmark_results:
            adapt_data = benchmark_results['adaptive_optimization']
            print(f"\n📊 Performance Summary for {args.preset.upper()} preset:")
            for i, batch_size in enumerate(adapt_data['batch_sizes']):
                exec_time = adapt_data['execution_times'][i]
                throughput = batch_size / max(exec_time, 0.001)
                print(f"   {batch_size:4d} splats: {exec_time*1000:6.2f}ms ({throughput:6.0f} splats/sec)")

    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    exit(main())