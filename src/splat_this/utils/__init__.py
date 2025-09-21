"""Utility modules for SplatThis."""

from .image import load_image, ImageLoader, validate_image_dimensions
from .math import safe_eigendecomposition, clamp_value, normalize_angle
from .profiler import (
    PerformanceProfiler,
    MemoryEfficientProcessor,
    global_profiler,
    estimate_memory_usage,
    benchmark_function
)

__all__ = [
    "load_image",
    "ImageLoader",
    "validate_image_dimensions",
    "safe_eigendecomposition",
    "clamp_value",
    "normalize_angle",
    "PerformanceProfiler",
    "MemoryEfficientProcessor",
    "global_profiler",
    "estimate_memory_usage",
    "benchmark_function",
]
