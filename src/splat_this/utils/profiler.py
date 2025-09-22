"""Performance profiling utilities for SplatThis."""

import time
import psutil
import functools
import gc
from typing import Dict, Any, Callable, Optional, Tuple
from pathlib import Path


class PerformanceProfiler:
    """Performance profiler for monitoring execution time and memory usage."""

    def __init__(self):
        self.metrics: Dict[str, Dict[str, Any]] = {}
        self.start_memory = psutil.virtual_memory().used
        self.process = psutil.Process()

    def profile_function(self, name: str):
        """Decorator to profile function execution."""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                start_memory = self.process.memory_info().rss

                # Force garbage collection for accurate memory measurement
                gc.collect()

                result = func(*args, **kwargs)

                end_time = time.time()
                end_memory = self.process.memory_info().rss

                # Get existing metrics or create new entry
                existing = self.metrics.get(name, {
                    'total_duration': 0.0,
                    'total_memory_delta': 0,
                    'peak_memory': start_memory,
                    'calls': 0
                })

                # Accumulate timing and memory data
                self.metrics[name] = {
                    'duration': end_time - start_time,  # Last call duration
                    'total_duration': existing['total_duration'] + (end_time - start_time),
                    'memory_delta': end_memory - start_memory,  # Last call memory delta
                    'total_memory_delta': existing['total_memory_delta'] + (end_memory - start_memory),
                    'peak_memory': max(existing['peak_memory'], end_memory),
                    'start_memory': start_memory,  # Last call start memory
                    'calls': existing['calls'] + 1
                }
                return result
            return wrapper
        return decorator

    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        if not self.metrics:
            return {
                'total_time': 0.0,
                'peak_memory_mb': 0.0,
                'by_function': {}
            }

        total_time = sum(m['duration'] for m in self.metrics.values())
        peak_memory = max(m['peak_memory'] for m in self.metrics.values())

        return {
            'total_time': total_time,
            'peak_memory_mb': peak_memory / (1024 * 1024),
            'total_memory_allocated_mb': sum(
                max(0, m['memory_delta']) for m in self.metrics.values()
            ) / (1024 * 1024),
            'by_function': self.metrics
        }

    def print_summary(self, title: str = "Performance Summary"):
        """Print formatted performance summary."""
        summary = self.get_summary()

        print(f"\n📊 {title}")
        print("=" * len(title) + "===")
        print(f"Total Time: {summary['total_time']:.2f}s")
        print(f"Peak Memory: {summary['peak_memory_mb']:.1f}MB")
        print(f"Memory Allocated: {summary['total_memory_allocated_mb']:.1f}MB")
        print()

        if summary['by_function']:
            print("By Function:")
            for name, metrics in summary['by_function'].items():
                print(f"  {name}:")
                print(f"    Time: {metrics['duration']:.3f}s")
                print(f"    Memory: {metrics['memory_delta'] / (1024 * 1024):+.1f}MB")
                print(f"    Calls: {metrics.get('calls', 1)}")


class MemoryEfficientProcessor:
    """Processor with memory management for large images."""

    def __init__(self, max_memory_mb: int = 1024):
        self.max_memory_mb = max_memory_mb
        self.process = psutil.Process()

    def check_memory_usage(self) -> float:
        """Check current memory usage in MB."""
        memory_mb = self.process.memory_info().rss / (1024 * 1024)
        return memory_mb

    def ensure_memory_limit(self, operation_name: str = "operation"):
        """Ensure memory usage doesn't exceed limit."""
        memory_mb = self.check_memory_usage()
        if memory_mb > self.max_memory_mb:
            # Force garbage collection
            gc.collect()
            memory_mb = self.check_memory_usage()

            if memory_mb > self.max_memory_mb:
                raise MemoryError(
                    f"Memory usage ({memory_mb:.0f}MB) exceeds limit "
                    f"({self.max_memory_mb}MB) during {operation_name}"
                )

    def estimate_memory_usage(self, splats: int, layers: int, image_size: tuple) -> float:
        """Estimate memory usage in MB for given parameters."""
        width, height = image_size

        # Base memory overhead
        base_memory = 100  # Base Python overhead

        # Image memory (multiple copies during processing)
        image_memory = (width * height * 3 * 4) / (1024 * 1024)  # RGBA float32
        image_copies = 3  # Original, processed, SLIC result

        # Splat memory
        splat_memory = splats * 0.0001  # ~0.1KB per splat

        # Layer processing memory
        layer_memory = layers * 5  # Layer processing overhead

        # SVG generation memory
        svg_memory = splats * 0.0005  # ~0.5KB per splat in SVG

        total = base_memory + (image_memory * image_copies) + splat_memory + layer_memory + svg_memory
        return total

    def should_downsample_image(self, image_size: tuple, target_splats: int) -> Tuple[bool, tuple]:
        """Determine if image should be downsampled for memory efficiency."""
        width, height = image_size
        total_pixels = width * height

        # Memory-based downsampling
        estimated_memory = self.estimate_memory_usage(target_splats, 5, image_size)

        if estimated_memory > self.max_memory_mb * 0.8:  # 80% of limit
            # Calculate downsample factor to stay under memory limit
            target_memory = self.max_memory_mb * 0.7  # 70% of limit for safety
            scale_factor = (target_memory / estimated_memory) ** 0.5

            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)

            # Ensure minimum reasonable size, but never upscale any dimension
            new_width = max(min(new_width, width), 400) if width >= 400 else new_width
            new_height = max(min(new_height, height), 300) if height >= 300 else new_height

            return True, (new_width, new_height)

        # Pixel count based downsampling (for very large images)
        max_pixels = 8_000_000  # 8MP max for performance
        if total_pixels > max_pixels:
            scale_factor = (max_pixels / total_pixels) ** 0.5
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            return True, (new_width, new_height)

        return False, image_size


def estimate_memory_usage(splats: int, layers: int) -> float:
    """Simple memory estimation for CLI warnings."""
    # Rough estimation based on splat count and layers
    base_memory = 100  # Base overhead
    splat_memory = splats * 0.1  # ~0.1MB per 1000 splats
    layer_memory = layers * 10   # Layer processing overhead
    return base_memory + splat_memory + layer_memory


def benchmark_function(func: Callable, *args, iterations: int = 5, **kwargs) -> Dict[str, float]:
    """Benchmark a function over multiple iterations."""
    times = []
    memories = []
    process = psutil.Process()

    for _ in range(iterations):
        gc.collect()  # Clean start

        start_memory = process.memory_info().rss
        start_time = time.time()

        result = func(*args, **kwargs)

        end_time = time.time()
        end_memory = process.memory_info().rss

        times.append(end_time - start_time)
        memories.append(end_memory - start_memory)

    return {
        'avg_time': sum(times) / len(times),
        'min_time': min(times),
        'max_time': max(times),
        'avg_memory_delta': sum(memories) / len(memories) / (1024 * 1024),  # MB
        'iterations': iterations
    }


# Global profiler instance for easy access
global_profiler = PerformanceProfiler()