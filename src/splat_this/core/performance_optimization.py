"""
Performance Optimization System for Adaptive Gaussian Splatting.

T4.3: Advanced performance optimization that provides spatial acceleration,
efficient computation, memory management, and parallel processing for
large-scale Gaussian splat operations.

This module provides:
- Spatial acceleration structures (Quad-tree, spatial hashing)
- Efficient covariance matrix computation with caching
- Memory optimization for large splat counts
- Parallel processing for batch operations
- Performance profiling and benchmarking
- Adaptive algorithm selection based on data characteristics
"""

import numpy as np
import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any, Union, Callable
from enum import Enum
import time
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from functools import lru_cache, wraps
import gc

from .adaptive_gaussian import AdaptiveGaussian2D
from ..utils.profiler import PerformanceProfiler, benchmark_function
from ..utils.math import safe_eigendecomposition, clamp_value

logger = logging.getLogger(__name__)


class AccelerationStructure(Enum):
    """Types of spatial acceleration structures."""
    NONE = "none"
    QUADTREE = "quadtree"
    SPATIAL_HASH = "spatial_hash"
    GRID = "grid"
    ADAPTIVE = "adaptive"


class ComputationMode(Enum):
    """Computation optimization modes."""
    STANDARD = "standard"
    CACHED = "cached"
    VECTORIZED = "vectorized"
    PARALLEL = "parallel"
    ADAPTIVE = "adaptive"


@dataclass
class PerformanceConfig:
    """Configuration for performance optimization."""

    # Spatial acceleration
    acceleration_structure: AccelerationStructure = AccelerationStructure.ADAPTIVE
    spatial_grid_size: int = 32
    quadtree_max_depth: int = 8
    quadtree_max_items: int = 16
    spatial_hash_cell_size: float = 0.1

    # Computation optimization
    computation_mode: ComputationMode = ComputationMode.ADAPTIVE
    enable_covariance_caching: bool = True
    cache_size: int = 1024
    vectorization_threshold: int = 100

    # Memory optimization
    max_memory_mb: int = 2048
    enable_memory_monitoring: bool = True
    garbage_collection_threshold: int = 1000
    chunk_size: int = 256

    # Parallel processing
    enable_parallel_processing: bool = True
    max_workers: Optional[int] = None
    parallel_threshold: int = 50
    use_process_pool: bool = False

    # Performance monitoring
    enable_profiling: bool = True
    benchmark_iterations: int = 5
    log_performance_warnings: bool = True

    # Adaptive thresholds
    small_batch_threshold: int = 10
    medium_batch_threshold: int = 100
    large_batch_threshold: int = 1000

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.max_workers is None:
            self.max_workers = min(mp.cpu_count(), 8)

        if self.spatial_grid_size < 4:
            raise ValueError("spatial_grid_size must be at least 4")
        if self.cache_size < 16:
            raise ValueError("cache_size must be at least 16")


@dataclass
class PerformanceMetrics:
    """Performance measurement results."""
    operation_name: str
    duration: float
    memory_used_mb: float
    splat_count: int
    operations_per_second: float
    memory_efficiency: float  # MB per 1000 splats
    acceleration_used: str
    computation_mode: str


class SpatialAccelerator:
    """Spatial acceleration structures for efficient splat queries."""

    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.structure_type = config.acceleration_structure
        self._spatial_index = None
        self._bounds = None

    def build_index(self, splats: List[AdaptiveGaussian2D], image_bounds: Tuple[float, float, float, float] = None):
        """Build spatial acceleration structure."""
        if not splats:
            return

        if image_bounds is None:
            # Compute bounds from splats
            positions = np.array([splat.mu for splat in splats])
            min_x, min_y = np.min(positions, axis=0)
            max_x, max_y = np.max(positions, axis=0)
            padding = 0.1
            self._bounds = (min_x - padding, min_y - padding, max_x + padding, max_y + padding)
        else:
            self._bounds = image_bounds

        structure_type = self._choose_optimal_structure(len(splats))

        if structure_type == AccelerationStructure.QUADTREE:
            self._spatial_index = self._build_quadtree(splats)
        elif structure_type == AccelerationStructure.SPATIAL_HASH:
            self._spatial_index = self._build_spatial_hash(splats)
        elif structure_type == AccelerationStructure.GRID:
            self._spatial_index = self._build_grid(splats)
        else:
            self._spatial_index = None

        logger.debug(f"Built {structure_type.value} for {len(splats)} splats")

    def _choose_optimal_structure(self, splat_count: int) -> AccelerationStructure:
        """Choose optimal acceleration structure based on data characteristics."""
        if self.structure_type != AccelerationStructure.ADAPTIVE:
            return self.structure_type

        # Adaptive selection based on splat count
        if splat_count < 50:
            return AccelerationStructure.NONE  # Linear search is fine
        elif splat_count < 500:
            return AccelerationStructure.GRID  # Simple grid is efficient
        elif splat_count < 2000:
            return AccelerationStructure.SPATIAL_HASH  # Hash for medium datasets
        else:
            return AccelerationStructure.QUADTREE  # Tree for large datasets

    def _build_quadtree(self, splats: List[AdaptiveGaussian2D]):
        """Build quadtree spatial index."""
        # Simple quadtree implementation
        class QuadTreeNode:
            def __init__(self, bounds, depth=0, config=None):
                self.bounds = bounds  # (min_x, min_y, max_x, max_y)
                self.splats = []
                self.children = []
                self.depth = depth
                self.config = config or self.config

            def insert(self, splat, splat_idx):
                # Check if splat is within bounds
                x, y = splat.mu[0], splat.mu[1]
                min_x, min_y, max_x, max_y = self.bounds

                if not (min_x <= x <= max_x and min_y <= y <= max_y):
                    return False

                # If we can fit more items or we're at max depth, add here
                if (len(self.splats) < self.config.quadtree_max_items or
                    self.depth >= self.config.quadtree_max_depth):
                    self.splats.append((splat, splat_idx))
                    return True

                # Split if needed
                if not self.children:
                    self._split()

                # Try to insert in children
                for child in self.children:
                    if child.insert(splat, splat_idx):
                        return True

                # If children couldn't take it, add here
                self.splats.append((splat, splat_idx))
                return True

            def _split(self):
                min_x, min_y, max_x, max_y = self.bounds
                mid_x = (min_x + max_x) / 2
                mid_y = (min_y + max_y) / 2

                self.children = [
                    QuadTreeNode((min_x, min_y, mid_x, mid_y), self.depth + 1, self.config),
                    QuadTreeNode((mid_x, min_y, max_x, mid_y), self.depth + 1, self.config),
                    QuadTreeNode((min_x, mid_y, mid_x, max_y), self.depth + 1, self.config),
                    QuadTreeNode((mid_x, mid_y, max_x, max_y), self.depth + 1, self.config)
                ]

            def query_region(self, query_bounds):
                """Query splats in a region."""
                results = []
                min_qx, min_qy, max_qx, max_qy = query_bounds
                min_x, min_y, max_x, max_y = self.bounds

                # Check if regions overlap
                if (max_qx < min_x or min_qx > max_x or
                    max_qy < min_y or min_qy > max_y):
                    return results

                # Add splats from this node
                for splat, idx in self.splats:
                    x, y = splat.mu[0], splat.mu[1]
                    if min_qx <= x <= max_qx and min_qy <= y <= max_qy:
                        results.append((splat, idx))

                # Query children
                for child in self.children:
                    results.extend(child.query_region(query_bounds))

                return results

        # Build the quadtree
        root = QuadTreeNode(self._bounds, config=self.config)
        for idx, splat in enumerate(splats):
            root.insert(splat, idx)

        return root

    def _build_spatial_hash(self, splats: List[AdaptiveGaussian2D]):
        """Build spatial hash index."""
        spatial_hash = {}
        cell_size = self.config.spatial_hash_cell_size

        for idx, splat in enumerate(splats):
            x, y = splat.mu[0], splat.mu[1]
            # Compute hash cell
            cell_x = int(x / cell_size)
            cell_y = int(y / cell_size)
            cell_key = (cell_x, cell_y)

            if cell_key not in spatial_hash:
                spatial_hash[cell_key] = []
            spatial_hash[cell_key].append((splat, idx))

        return spatial_hash

    def _build_grid(self, splats: List[AdaptiveGaussian2D]):
        """Build simple grid index."""
        grid_size = self.config.spatial_grid_size
        min_x, min_y, max_x, max_y = self._bounds

        # Create grid
        grid = [[[] for _ in range(grid_size)] for _ in range(grid_size)]

        for idx, splat in enumerate(splats):
            x, y = splat.mu[0], splat.mu[1]

            # Map to grid coordinates
            grid_x = int((x - min_x) / (max_x - min_x) * (grid_size - 1))
            grid_y = int((y - min_y) / (max_y - min_y) * (grid_size - 1))

            grid_x = np.clip(grid_x, 0, grid_size - 1)
            grid_y = np.clip(grid_y, 0, grid_size - 1)

            grid[grid_y][grid_x].append((splat, idx))

        return grid

    def query_region(self, region_bounds: Tuple[float, float, float, float]) -> List[Tuple[AdaptiveGaussian2D, int]]:
        """Query splats in a spatial region."""
        if self._spatial_index is None:
            return []

        if hasattr(self._spatial_index, 'query_region'):
            # Quadtree
            return self._spatial_index.query_region(region_bounds)
        elif isinstance(self._spatial_index, dict):
            # Spatial hash
            return self._query_hash_region(region_bounds)
        elif isinstance(self._spatial_index, list):
            # Grid
            return self._query_grid_region(region_bounds)
        else:
            return []

    def _query_hash_region(self, region_bounds):
        """Query spatial hash for region."""
        min_x, min_y, max_x, max_y = region_bounds
        cell_size = self.config.spatial_hash_cell_size
        results = []

        # Find all cells that overlap with region
        min_cell_x = int(min_x / cell_size)
        max_cell_x = int(max_x / cell_size)
        min_cell_y = int(min_y / cell_size)
        max_cell_y = int(max_y / cell_size)

        for cell_x in range(min_cell_x, max_cell_x + 1):
            for cell_y in range(min_cell_y, max_cell_y + 1):
                cell_key = (cell_x, cell_y)
                if cell_key in self._spatial_index:
                    for splat, idx in self._spatial_index[cell_key]:
                        x, y = splat.mu[0], splat.mu[1]
                        if min_x <= x <= max_x and min_y <= y <= max_y:
                            results.append((splat, idx))

        return results

    def _query_grid_region(self, region_bounds):
        """Query grid for region."""
        min_x, min_y, max_x, max_y = region_bounds
        bounds_min_x, bounds_min_y, bounds_max_x, bounds_max_y = self._bounds
        grid_size = self.config.spatial_grid_size
        results = []

        # Map region to grid coordinates
        min_grid_x = int((min_x - bounds_min_x) / (bounds_max_x - bounds_min_x) * (grid_size - 1))
        max_grid_x = int((max_x - bounds_min_x) / (bounds_max_x - bounds_min_x) * (grid_size - 1))
        min_grid_y = int((min_y - bounds_min_y) / (bounds_max_y - bounds_min_y) * (grid_size - 1))
        max_grid_y = int((max_y - bounds_min_y) / (bounds_max_y - bounds_min_y) * (grid_size - 1))

        min_grid_x = np.clip(min_grid_x, 0, grid_size - 1)
        max_grid_x = np.clip(max_grid_x, 0, grid_size - 1)
        min_grid_y = np.clip(min_grid_y, 0, grid_size - 1)
        max_grid_y = np.clip(max_grid_y, 0, grid_size - 1)

        for grid_y in range(min_grid_y, max_grid_y + 1):
            for grid_x in range(min_grid_x, max_grid_x + 1):
                for splat, idx in self._spatial_index[grid_y][grid_x]:
                    x, y = splat.mu[0], splat.mu[1]
                    if min_x <= x <= max_x and min_y <= y <= max_y:
                        results.append((splat, idx))

        return results


class EfficientComputation:
    """Efficient computation utilities with caching and vectorization."""

    def __init__(self, config: PerformanceConfig):
        self.config = config
        self._covariance_cache = {}
        self._computation_stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'vectorized_operations': 0
        }

    @lru_cache(maxsize=1024)
    def _cached_covariance_computation(self, inv_s_tuple: Tuple[float, float], theta: float) -> Tuple[np.ndarray, np.ndarray]:
        """Cached covariance matrix computation."""
        inv_s = np.array(inv_s_tuple)

        # Compute rotation matrix
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        rotation_matrix = np.array([
            [cos_theta, -sin_theta],
            [sin_theta, cos_theta]
        ])

        # Scale matrix (inverse scales)
        scale_matrix = np.diag(inv_s)

        # Combined transformation: R * S
        transform_matrix = rotation_matrix @ scale_matrix

        # Covariance matrix: (R * S)^-1 * (R * S)^-T
        try:
            inv_transform = np.linalg.inv(transform_matrix)
            covariance = inv_transform @ inv_transform.T
        except np.linalg.LinAlgError:
            # Fallback for singular matrices
            covariance = np.eye(2) * 0.01

        # Inverse covariance for efficient computation
        try:
            inv_covariance = np.linalg.inv(covariance)
        except np.linalg.LinAlgError:
            inv_covariance = np.eye(2) * 100.0

        return covariance, inv_covariance

    def compute_covariance_matrices(self, splats: List[AdaptiveGaussian2D]) -> Tuple[np.ndarray, np.ndarray]:
        """Efficiently compute covariance matrices for multiple splats."""
        n_splats = len(splats)
        covariances = np.zeros((n_splats, 2, 2))
        inv_covariances = np.zeros((n_splats, 2, 2))

        if self.config.computation_mode == ComputationMode.VECTORIZED and n_splats >= self.config.vectorization_threshold:
            # Vectorized computation
            self._vectorized_covariance_computation(splats, covariances, inv_covariances)
            self._computation_stats['vectorized_operations'] += 1
        else:
            # Individual computation with caching
            for i, splat in enumerate(splats):
                if self.config.enable_covariance_caching:
                    # Use cached computation
                    cache_key = (tuple(splat.inv_s), splat.theta)
                    try:
                        cov, inv_cov = self._cached_covariance_computation(cache_key[0], cache_key[1])
                        self._computation_stats['cache_hits'] += 1
                    except:
                        cov, inv_cov = self._compute_single_covariance(splat)
                        self._computation_stats['cache_misses'] += 1
                else:
                    cov, inv_cov = self._compute_single_covariance(splat)
                    self._computation_stats['cache_misses'] += 1

                covariances[i] = cov
                inv_covariances[i] = inv_cov

        return covariances, inv_covariances

    def _vectorized_covariance_computation(self, splats: List[AdaptiveGaussian2D],
                                         covariances: np.ndarray, inv_covariances: np.ndarray):
        """Vectorized covariance computation for multiple splats."""
        n_splats = len(splats)

        # Extract parameters
        inv_scales = np.array([splat.inv_s for splat in splats])  # (n, 2)
        thetas = np.array([splat.theta for splat in splats])      # (n,)

        # Compute rotation matrices for all splats
        cos_thetas = np.cos(thetas)
        sin_thetas = np.sin(thetas)

        rotation_matrices = np.zeros((n_splats, 2, 2))
        rotation_matrices[:, 0, 0] = cos_thetas
        rotation_matrices[:, 0, 1] = -sin_thetas
        rotation_matrices[:, 1, 0] = sin_thetas
        rotation_matrices[:, 1, 1] = cos_thetas

        # Compute scale matrices
        scale_matrices = np.zeros((n_splats, 2, 2))
        scale_matrices[:, 0, 0] = inv_scales[:, 0]
        scale_matrices[:, 1, 1] = inv_scales[:, 1]

        # Combined transformation matrices
        transform_matrices = rotation_matrices @ scale_matrices

        # Compute covariances and inverse covariances
        for i in range(n_splats):
            try:
                inv_transform = np.linalg.inv(transform_matrices[i])
                covariances[i] = inv_transform @ inv_transform.T
                inv_covariances[i] = np.linalg.inv(covariances[i])
            except np.linalg.LinAlgError:
                # Fallback for singular matrices
                covariances[i] = np.eye(2) * 0.01
                inv_covariances[i] = np.eye(2) * 100.0

    def _compute_single_covariance(self, splat: AdaptiveGaussian2D) -> Tuple[np.ndarray, np.ndarray]:
        """Compute covariance for a single splat."""
        return self._cached_covariance_computation(tuple(splat.inv_s), splat.theta)

    def get_computation_stats(self) -> Dict[str, Any]:
        """Get computation performance statistics."""
        total_operations = self._computation_stats['cache_hits'] + self._computation_stats['cache_misses']
        cache_hit_rate = (self._computation_stats['cache_hits'] / max(total_operations, 1)) * 100

        return {
            'cache_hit_rate': cache_hit_rate,
            'total_operations': total_operations,
            'vectorized_operations': self._computation_stats['vectorized_operations'],
            'cache_size': len(self._covariance_cache)
        }


class MemoryOptimizer:
    """Memory optimization utilities for large splat operations."""

    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.memory_stats = {
            'peak_usage_mb': 0.0,
            'gc_collections': 0,
            'chunks_processed': 0
        }

    def process_in_chunks(self, splats: List[AdaptiveGaussian2D],
                         operation: Callable, *args, **kwargs) -> List[Any]:
        """Process splats in memory-efficient chunks."""
        if len(splats) <= self.config.chunk_size:
            return [operation(splats, *args, **kwargs)]

        results = []
        num_chunks = (len(splats) + self.config.chunk_size - 1) // self.config.chunk_size

        logger.info(f"Processing {len(splats)} splats in {num_chunks} chunks of {self.config.chunk_size}")

        for i in range(0, len(splats), self.config.chunk_size):
            chunk = splats[i:i + self.config.chunk_size]

            # Memory monitoring
            if self.config.enable_memory_monitoring:
                self._check_memory_usage(f"chunk_{i // self.config.chunk_size}")

            # Process chunk
            chunk_result = operation(chunk, *args, **kwargs)
            results.append(chunk_result)

            self.memory_stats['chunks_processed'] += 1

            # Garbage collection if needed
            if (i // self.config.chunk_size) % 10 == 0:  # Every 10 chunks
                self._maybe_collect_garbage()

        return results

    def _check_memory_usage(self, operation_name: str):
        """Check and log memory usage."""
        import psutil
        process = psutil.Process()
        memory_mb = process.memory_info().rss / (1024 * 1024)

        self.memory_stats['peak_usage_mb'] = max(self.memory_stats['peak_usage_mb'], memory_mb)

        if memory_mb > self.config.max_memory_mb * 0.9:  # 90% of limit
            logger.warning(f"High memory usage during {operation_name}: {memory_mb:.1f}MB")
            self._force_garbage_collection()

    def _maybe_collect_garbage(self):
        """Collect garbage if threshold is reached."""
        if self.memory_stats['chunks_processed'] >= self.config.garbage_collection_threshold:
            self._force_garbage_collection()
            self.memory_stats['chunks_processed'] = 0

    def _force_garbage_collection(self):
        """Force garbage collection."""
        collected = gc.collect()
        self.memory_stats['gc_collections'] += 1
        logger.debug(f"Garbage collection freed {collected} objects")

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        return self.memory_stats.copy()


class ParallelProcessor:
    """Parallel processing utilities for batch operations."""

    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.processing_stats = {
            'parallel_operations': 0,
            'sequential_operations': 0,
            'total_threads_used': 0
        }

    def process_parallel(self, splats: List[AdaptiveGaussian2D],
                        operation: Callable, *args, **kwargs) -> List[Any]:
        """Process splats in parallel if beneficial."""
        if not self.config.enable_parallel_processing or len(splats) < self.config.parallel_threshold:
            self.processing_stats['sequential_operations'] += 1
            return [operation(splats, *args, **kwargs)]

        # Determine optimal chunk size for parallel processing
        num_workers = min(self.config.max_workers, len(splats) // 10 + 1)
        chunk_size = len(splats) // num_workers

        if chunk_size < 5:  # Too small chunks aren't worth parallel overhead
            self.processing_stats['sequential_operations'] += 1
            return [operation(splats, *args, **kwargs)]

        chunks = [splats[i:i + chunk_size] for i in range(0, len(splats), chunk_size)]

        self.processing_stats['parallel_operations'] += 1
        self.processing_stats['total_threads_used'] += len(chunks)

        executor_class = ProcessPoolExecutor if self.config.use_process_pool else ThreadPoolExecutor

        try:
            with executor_class(max_workers=num_workers) as executor:
                futures = [executor.submit(operation, chunk, *args, **kwargs) for chunk in chunks]
                results = [future.result() for future in futures]

            logger.debug(f"Processed {len(splats)} splats using {len(chunks)} parallel workers")
            return results

        except Exception as e:
            logger.warning(f"Parallel processing failed, falling back to sequential: {e}")
            self.processing_stats['sequential_operations'] += 1
            return [operation(splats, *args, **kwargs)]

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get parallel processing statistics."""
        total_ops = self.processing_stats['parallel_operations'] + self.processing_stats['sequential_operations']
        parallel_ratio = (self.processing_stats['parallel_operations'] / max(total_ops, 1)) * 100

        return {
            'parallel_ratio': parallel_ratio,
            'total_operations': total_ops,
            'avg_threads_per_operation': self.processing_stats['total_threads_used'] / max(self.processing_stats['parallel_operations'], 1)
        }


class PerformanceOptimizer:
    """Main performance optimization coordinator."""

    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.spatial_accelerator = SpatialAccelerator(config)
        self.efficient_computation = EfficientComputation(config)
        self.memory_optimizer = MemoryOptimizer(config)
        self.parallel_processor = ParallelProcessor(config)
        self.profiler = PerformanceProfiler() if config.enable_profiling else None
        self.optimization_history = []

    def optimize_splat_operations(self, splats: List[AdaptiveGaussian2D],
                                 operation: Callable, *args, **kwargs) -> Any:
        """Optimize a splat operation using all available techniques."""
        operation_name = kwargs.pop('operation_name', operation.__name__)
        start_time = time.time()

        if self.profiler:
            self.profiler.metrics[operation_name] = self.profiler.metrics.get(operation_name, {})

        try:
            # Choose optimization strategy based on splat count
            splat_count = len(splats)

            if splat_count <= self.config.small_batch_threshold:
                # Small batch: use standard processing
                result = operation(splats, *args, **kwargs)
            elif splat_count <= self.config.medium_batch_threshold:
                # Medium batch: use efficient computation and caching
                result = self._optimize_medium_batch(splats, operation, *args, **kwargs)
            elif splat_count <= self.config.large_batch_threshold:
                # Large batch: add parallel processing
                result = self._optimize_large_batch(splats, operation, *args, **kwargs)
            else:
                # Very large batch: use all optimizations
                result = self._optimize_very_large_batch(splats, operation, *args, **kwargs)

            # Record performance metrics
            duration = time.time() - start_time
            self._record_performance_metrics(operation_name, duration, splat_count)

            return result

        except Exception as e:
            logger.error(f"Performance optimization failed for {operation_name}: {e}")
            # Fallback to standard operation
            return operation(splats, *args, **kwargs)

    def _optimize_medium_batch(self, splats: List[AdaptiveGaussian2D],
                              operation: Callable, *args, **kwargs) -> Any:
        """Optimize medium-sized batch operations."""
        # Use efficient computation
        return operation(splats, *args, **kwargs)

    def _optimize_large_batch(self, splats: List[AdaptiveGaussian2D],
                             operation: Callable, *args, **kwargs) -> Any:
        """Optimize large batch operations."""
        # Use parallel processing
        results = self.parallel_processor.process_parallel(splats, operation, *args, **kwargs)

        # Combine results if needed
        if len(results) == 1:
            return results[0]
        else:
            # Need to implement result combination logic based on operation type
            return results

    def _optimize_very_large_batch(self, splats: List[AdaptiveGaussian2D],
                                  operation: Callable, *args, **kwargs) -> Any:
        """Optimize very large batch operations."""
        # Use chunked processing with memory management
        chunk_operation = lambda chunk, *args, **kwargs: self.parallel_processor.process_parallel(
            chunk, operation, *args, **kwargs
        )

        chunk_results = self.memory_optimizer.process_in_chunks(splats, chunk_operation, *args, **kwargs)

        # Flatten results
        results = []
        for chunk_result in chunk_results:
            if isinstance(chunk_result, list):
                results.extend(chunk_result)
            else:
                results.append(chunk_result)

        return results

    def _record_performance_metrics(self, operation_name: str, duration: float, splat_count: int):
        """Record performance metrics for analysis."""
        import psutil
        process = psutil.Process()
        memory_mb = process.memory_info().rss / (1024 * 1024)

        metrics = PerformanceMetrics(
            operation_name=operation_name,
            duration=duration,
            memory_used_mb=memory_mb,
            splat_count=splat_count,
            operations_per_second=splat_count / max(duration, 0.001),
            memory_efficiency=memory_mb / max(splat_count / 1000, 0.001),
            acceleration_used=self.spatial_accelerator.structure_type.value,
            computation_mode=self.config.computation_mode.value
        )

        self.optimization_history.append(metrics)

        if self.config.log_performance_warnings:
            self._check_performance_warnings(metrics)

    def _check_performance_warnings(self, metrics: PerformanceMetrics):
        """Check for performance issues and log warnings."""
        # Slow operations
        if metrics.operations_per_second < 100 and metrics.splat_count > 100:
            logger.warning(f"Slow operation {metrics.operation_name}: "
                          f"{metrics.operations_per_second:.1f} ops/sec")

        # High memory usage
        if metrics.memory_efficiency > 50:  # > 50MB per 1000 splats
            logger.warning(f"High memory usage in {metrics.operation_name}: "
                          f"{metrics.memory_efficiency:.1f}MB per 1000 splats")

    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        return {
            'optimization_history': [
                {
                    'operation': m.operation_name,
                    'duration': m.duration,
                    'splat_count': m.splat_count,
                    'ops_per_second': m.operations_per_second,
                    'memory_efficiency': m.memory_efficiency
                }
                for m in self.optimization_history[-10:]  # Last 10 operations
            ],
            'computation_stats': self.efficient_computation.get_computation_stats(),
            'memory_stats': self.memory_optimizer.get_memory_stats(),
            'processing_stats': self.parallel_processor.get_processing_stats(),
            'total_operations': len(self.optimization_history)
        }

    def benchmark_operations(self, splats: List[AdaptiveGaussian2D],
                           operations: Dict[str, Callable]) -> Dict[str, Dict[str, float]]:
        """Benchmark multiple operations for performance comparison."""
        results = {}

        for op_name, operation in operations.items():
            logger.info(f"Benchmarking {op_name}...")

            benchmark_result = benchmark_function(
                operation, splats,
                iterations=self.config.benchmark_iterations
            )

            results[op_name] = benchmark_result

        return results


# Convenience functions for easy usage

def create_performance_config_preset(preset: str = "balanced") -> PerformanceConfig:
    """Create predefined performance optimization configurations."""
    if preset == "minimal":
        return PerformanceConfig(
            acceleration_structure=AccelerationStructure.NONE,
            computation_mode=ComputationMode.STANDARD,
            enable_parallel_processing=False,
            enable_covariance_caching=False,
            max_memory_mb=512
        )
    elif preset == "balanced":
        return PerformanceConfig(
            acceleration_structure=AccelerationStructure.ADAPTIVE,
            computation_mode=ComputationMode.ADAPTIVE,
            enable_parallel_processing=True,
            enable_covariance_caching=True,
            max_memory_mb=2048
        )
    elif preset == "performance":
        return PerformanceConfig(
            acceleration_structure=AccelerationStructure.QUADTREE,
            computation_mode=ComputationMode.VECTORIZED,
            enable_parallel_processing=True,
            enable_covariance_caching=True,
            max_memory_mb=4096,
            vectorization_threshold=50,
            parallel_threshold=25
        )
    elif preset == "memory_efficient":
        return PerformanceConfig(
            acceleration_structure=AccelerationStructure.GRID,
            computation_mode=ComputationMode.CACHED,
            enable_parallel_processing=False,
            enable_covariance_caching=True,
            max_memory_mb=1024,
            chunk_size=128,
            enable_memory_monitoring=True
        )
    else:
        raise ValueError(f"Unknown preset: {preset}. Available: minimal, balanced, performance, memory_efficient")


def optimize_splat_operation(splats: List[AdaptiveGaussian2D],
                            operation: Callable,
                            config: Optional[PerformanceConfig] = None,
                            *args, **kwargs) -> Any:
    """
    Convenience function for optimizing splat operations.

    Args:
        splats: List of Gaussian splats to process
        operation: Operation function to optimize
        config: Optional performance configuration (defaults to balanced)
        *args, **kwargs: Arguments to pass to the operation

    Returns:
        Result of the optimized operation
    """
    if config is None:
        config = create_performance_config_preset("balanced")

    optimizer = PerformanceOptimizer(config)
    return optimizer.optimize_splat_operations(splats, operation, *args, **kwargs)