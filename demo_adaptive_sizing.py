#!/usr/bin/env python3
"""Demonstration of adaptive sizing strategy for content-aware Gaussian splat initialization."""

import numpy as np
import time
from src.splat_this.core.adaptive_sizing import (
    AdaptiveSizer,
    SizingConfig,
    compute_adaptive_sizes,
    allocate_splat_sizes
)


def create_test_images():
    """Create various test images for adaptive sizing demonstration."""
    images = {}

    # 1. Uniform image (should get uniform, large sizes)
    uniform = np.ones((64, 64)) * 0.5
    images['uniform'] = uniform

    # 2. Simple gradient (should get varying sizes)
    gradient = np.zeros((64, 64))
    for i in range(64):
        gradient[:, i] = i / 64.0
    images['gradient'] = gradient

    # 3. High detail checkerboard (should get small sizes)
    checkerboard = np.zeros((64, 64))
    for i in range(64):
        for j in range(64):
            if (i // 4 + j // 4) % 2 == 0:
                checkerboard[i, j] = 1.0
    images['checkerboard'] = checkerboard

    # 4. Mixed complexity image
    mixed = np.ones((64, 64)) * 0.5  # Base uniform
    # Add high detail region
    for i in range(16, 48):
        for j in range(16, 48):
            if (i + j) % 2 == 0:
                mixed[i, j] = 1.0
            else:
                mixed[i, j] = 0.0
    # Add smooth gradient region
    for i in range(64):
        mixed[i, :16] = i / 64.0
    images['mixed_complexity'] = mixed

    # 5. Edge-rich image
    edge_rich = np.zeros((64, 64))
    # Horizontal edges
    edge_rich[20:22, :] = 1.0
    edge_rich[40:42, :] = 1.0
    # Vertical edges
    edge_rich[:, 20:22] = 0.5
    edge_rich[:, 40:42] = 0.5
    # Diagonal edge
    for i in range(64):
        for j in range(64):
            if abs(i - j) < 2:
                edge_rich[i, j] = 0.8
    images['edge_rich'] = edge_rich

    # 6. Natural-like texture
    natural = np.random.rand(64, 64) * 0.3 + 0.3  # Base texture
    # Add structure
    natural[:20, :] = 0.8  # Sky region (smooth)
    natural[20:25, :] = np.linspace(0.8, 0.2, 64)  # Horizon gradient
    natural[25:, :] *= 0.5  # Ground region (keep texture but darker)
    images['natural_texture'] = natural

    # 7. Circular pattern
    circular = np.zeros((64, 64))
    center = 32
    for i in range(64):
        for j in range(64):
            radius = np.sqrt((i - center)**2 + (j - center)**2)
            if 10 < radius < 15:
                circular[i, j] = 1.0
            elif 20 < radius < 25:
                circular[i, j] = 0.5
    images['circular_pattern'] = circular

    return images


def demo_basic_adaptive_sizing():
    """Demonstrate basic adaptive sizing computation."""
    print("=== Basic Adaptive Sizing Demo ===")

    images = create_test_images()
    image = images['mixed_complexity']

    sizer = AdaptiveSizer((64, 64))

    print(f"Input image shape: {image.shape}")

    start_time = time.time()
    result = sizer.compute_adaptive_sizes(image)
    computation_time = time.time() - start_time

    print(f"Adaptive sizing computed in {computation_time:.3f} seconds")
    print(f"Size map shape: {result.size_map.shape}")
    print(f"Complexity map shape: {result.complexity_map.shape}")

    # Analyze size statistics
    print(f"\nSize Statistics:")
    print(f"  Mean size: {result.statistics['mean_size']:.6f}")
    print(f"  Size range: {result.statistics['size_range']:.6f}")
    print(f"  Size diversity: {result.statistics['size_diversity']:.3f}")
    print(f"  Min size: {result.statistics['min_size']:.6f}")
    print(f"  Max size: {result.statistics['max_size']:.6f}")

    # Analyze complexity correlation
    print(f"\nComplexity Correlation:")
    print(f"  Complexity-size correlation: {result.statistics['complexity_correlation']:.3f}")

    # Size distribution
    small_fraction = result.statistics['small_size_fraction']
    large_fraction = result.statistics['large_size_fraction']
    print(f"\nSize Distribution:")
    print(f"  Small sizes fraction: {small_fraction:.3f}")
    print(f"  Large sizes fraction: {large_fraction:.3f}")
    print(f"  Size entropy: {result.statistics['size_entropy']:.3f}")


def demo_content_adaptation():
    """Demonstrate how sizing adapts to different content types."""
    print("\n=== Content Adaptation Demo ===")

    images = create_test_images()
    sizer = AdaptiveSizer((64, 64))

    content_types = ['uniform', 'gradient', 'checkerboard', 'edge_rich', 'natural_texture']

    for content_type in content_types:
        image = images[content_type]
        result = sizer.compute_adaptive_sizes(image)

        print(f"\n{content_type.replace('_', ' ').title()}:")
        print(f"  Mean complexity: {np.mean(result.complexity_map):.3f}")
        print(f"  Mean size: {result.statistics['mean_size']:.6f}")
        print(f"  Size diversity: {result.statistics['size_diversity']:.3f}")
        print(f"  Complexity correlation: {result.statistics['complexity_correlation']:.3f}")

        # Analyze size adaptation
        min_size = result.statistics['min_size']
        max_size = result.statistics['max_size']
        size_adaptation = (max_size - min_size) / sizer.config.base_size
        print(f"  Size adaptation factor: {size_adaptation:.2f}x")


def demo_configuration_effects():
    """Demonstrate effects of different configuration parameters."""
    print("\n=== Configuration Effects Demo ===")

    image = create_test_images()['mixed_complexity']

    configs = [
        ("Default", SizingConfig()),
        ("High Complexity Sensitivity", SizingConfig(complexity_sensitivity=1.0)),
        ("Low Complexity Sensitivity", SizingConfig(complexity_sensitivity=0.3)),
        ("High Variance Weight", SizingConfig(variance_weight=0.7)),
        ("High Edge Weight", SizingConfig(edge_density_weight=0.8)),
        ("Large Base Size", SizingConfig(base_size=0.04)),
        ("Small Base Size", SizingConfig(base_size=0.01)),
        ("Wide Size Range", SizingConfig(size_range=(0.002, 0.12))),
        ("Narrow Size Range", SizingConfig(size_range=(0.01, 0.03))),
        ("Heavy Smoothing", SizingConfig(smoothing_sigma=3.0)),
        ("No Smoothing", SizingConfig(smoothing_sigma=0.0)),
        ("Quantized Sizes", SizingConfig(size_quantization=5))
    ]

    for name, config in configs:
        sizer = AdaptiveSizer((64, 64), config)
        result = sizer.compute_adaptive_sizes(image)

        print(f"\n{name}:")
        print(f"  Mean size: {result.statistics['mean_size']:.6f}")
        print(f"  Size diversity: {result.statistics['size_diversity']:.3f}")
        print(f"  Size range: {result.statistics['size_range']:.6f}")
        print(f"  Complexity correlation: {result.statistics['complexity_correlation']:.3f}")


def demo_splat_size_allocation():
    """Demonstrate size allocation for specific splat positions."""
    print("\n=== Splat Size Allocation Demo ===")

    image = create_test_images()['mixed_complexity']

    # Define test positions in different regions
    test_positions = np.array([
        [32, 8],   # Gradient region
        [32, 32],  # High detail center
        [8, 8],    # Uniform region
        [50, 50],  # Uniform region
        [32, 48],  # High detail region
    ])

    position_names = [
        "Gradient region", "High detail center", "Uniform region 1",
        "Uniform region 2", "High detail region"
    ]

    sizer = AdaptiveSizer((64, 64))
    allocation = sizer.allocate_splat_sizes(test_positions, image)

    print("Size Allocation at Test Positions:")
    for i, (name, pos) in enumerate(zip(position_names, test_positions)):
        size = allocation.sizes[i]
        complexity = allocation.complexity_scores[i]
        rationale = allocation.size_rationale[i]

        print(f"\n{name} at ({pos[0]}, {pos[1]}):")
        print(f"  Allocated size: {size:.6f}")
        print(f"  Complexity score: {complexity:.3f}")
        print(f"  Rationale: {rationale}")

    # Compare with base size
    base_size = sizer.config.base_size
    print(f"\nSize Comparison (base size = {base_size:.6f}):")
    for i, (name, size) in enumerate(zip(position_names, allocation.sizes)):
        ratio = size / base_size
        print(f"  {name}: {ratio:.2f}x base size")


def demo_normalization_methods():
    """Demonstrate different normalization methods."""
    print("\n=== Normalization Methods Demo ===")

    image = create_test_images()['edge_rich']

    normalizations = ['percentile', 'minmax', 'adaptive']

    for normalization in normalizations:
        config = SizingConfig(normalization=normalization)
        sizer = AdaptiveSizer((64, 64), config)
        result = sizer.compute_adaptive_sizes(image)

        print(f"\n{normalization.title()} Normalization:")
        print(f"  Mean size: {result.statistics['mean_size']:.6f}")
        print(f"  Size range: {result.statistics['size_range']:.6f}")
        print(f"  Size diversity: {result.statistics['size_diversity']:.3f}")
        print(f"  Min size: {result.statistics['min_size']:.6f}")
        print(f"  Max size: {result.statistics['max_size']:.6f}")


def demo_size_validation():
    """Demonstrate size distribution validation."""
    print("\n=== Size Distribution Validation Demo ===")

    images = create_test_images()
    test_cases = ['uniform', 'mixed_complexity', 'checkerboard']

    for test_case in test_cases:
        image = images[test_case]

        # Test with default config
        sizer = AdaptiveSizer((64, 64))
        result = sizer.compute_adaptive_sizes(image)
        validation = sizer.validate_size_distribution(result)

        print(f"\n{test_case.replace('_', ' ').title()}:")
        print(f"  Validation passed: {validation['passed']}")
        print(f"  Quality score: {validation['size_quality_score']:.3f}")
        print(f"  Issues: {len(validation['issues'])}")

        if validation['issues']:
            for issue in validation['issues']:
                print(f"    - {issue}")

        if validation['recommendations']:
            print(f"  Recommendations: {len(validation['recommendations'])}")
            for rec in validation['recommendations'][:2]:  # Show first 2
                print(f"    - {rec}")


def demo_adaptive_range():
    """Demonstrate adaptive range adjustment."""
    print("\n=== Adaptive Range Demo ===")

    images = create_test_images()

    # Test with and without adaptive range
    configs = [
        ("Fixed Range", SizingConfig(adaptive_range=False)),
        ("Adaptive Range", SizingConfig(adaptive_range=True))
    ]

    test_images = ['uniform', 'checkerboard', 'natural_texture']

    for config_name, config in configs:
        print(f"\n{config_name}:")

        for image_name in test_images:
            image = images[image_name]
            sizer = AdaptiveSizer((64, 64), config)
            result = sizer.compute_adaptive_sizes(image)

            complexity_level = np.mean(result.complexity_map)
            if complexity_level > 0.7:
                complexity_desc = "High"
            elif complexity_level > 0.3:
                complexity_desc = "Medium"
            else:
                complexity_desc = "Low"

            print(f"  {image_name}: {complexity_desc} complexity ({complexity_level:.3f})")
            print(f"    Size range: {result.statistics['size_range']:.6f}")
            print(f"    Mean size: {result.statistics['mean_size']:.6f}")


def demo_performance_and_scaling():
    """Demonstrate performance characteristics and scaling."""
    print("\n=== Performance and Scaling Demo ===")

    sizes = [(32, 32), (64, 64), (128, 128)]
    sizer_configs = [
        ("Basic", SizingConfig()),
        ("No Smoothing", SizingConfig(smoothing_sigma=0.0)),
        ("Heavy Processing", SizingConfig(smoothing_sigma=2.0, size_quantization=50))
    ]

    for config_name, config in sizer_configs:
        print(f"\n{config_name} Configuration:")

        for h, w in sizes:
            # Create test image
            image = np.random.rand(h, w)

            sizer = AdaptiveSizer((h, w), config)

            start_time = time.time()
            result = sizer.compute_adaptive_sizes(image)
            computation_time = time.time() - start_time

            pixels = h * w
            time_per_pixel = computation_time / pixels * 1e6  # microseconds per pixel

            print(f"  {w}x{h} ({pixels:5d} px): {computation_time:.3f}s, {time_per_pixel:.2f}μs/px")


def demo_quality_metrics():
    """Demonstrate comprehensive quality metrics."""
    print("\n=== Quality Metrics Demo ===")

    images = create_test_images()

    for image_name, image in images.items():
        sizer = AdaptiveSizer((64, 64))
        result = sizer.compute_adaptive_sizes(image)

        print(f"\n{image_name.replace('_', ' ').title()}:")

        # Size statistics
        stats = result.statistics
        print(f"  Size diversity: {stats['size_diversity']:.3f}")
        print(f"  Complexity correlation: {stats['complexity_correlation']:.3f}")
        print(f"  Size entropy: {stats['size_entropy']:.3f}")

        # Quality validation
        validation = sizer.validate_size_distribution(result)
        print(f"  Quality score: {validation['size_quality_score']:.3f}")
        print(f"  Validation passed: {validation['passed']}")


def demo_convenience_functions():
    """Demonstrate convenience functions."""
    print("\n=== Convenience Functions Demo ===")

    image = create_test_images()['circular_pattern']

    # Using convenience function for full analysis
    print("Quick adaptive sizing:")
    result = compute_adaptive_sizes(image, base_size=0.03, complexity_sensitivity=0.9)
    print(f"  Mean size: {result.statistics['mean_size']:.6f}")
    print(f"  Size diversity: {result.statistics['size_diversity']:.3f}")

    # Using convenience function for splat allocation
    print("\nQuick splat size allocation:")
    test_positions = np.array([[32, 32], [16, 16], [48, 48]])
    allocation = allocate_splat_sizes(test_positions, image)

    for i, pos in enumerate(test_positions):
        print(f"  Position ({pos[0]}, {pos[1]}): size={allocation.sizes[i]:.6f}")


if __name__ == "__main__":
    print("🎯 SplatThis Adaptive Sizing Strategy Demonstration")
    print("=" * 75)

    demo_basic_adaptive_sizing()
    demo_content_adaptation()
    demo_configuration_effects()
    demo_splat_size_allocation()
    demo_normalization_methods()
    demo_size_validation()
    demo_adaptive_range()
    demo_performance_and_scaling()
    demo_quality_metrics()
    demo_convenience_functions()

    print("\n" + "=" * 75)
    print("✅ Adaptive Sizing Strategy Demo Complete!")
    print("📏 Content-adaptive: Sizes automatically adjust to image complexity")
    print("🎯 Detail-aware: Small sizes in high-detail regions, large in smooth areas")
    print("⚖️  Quality-driven: Validation ensures appropriate size diversity")
    print("🔧 Configurable: Multiple parameters for fine-tuning behavior")
    print("📊 Comprehensive: Rich statistics and quality metrics")
    print("⚡ Efficient: Scalable performance across image sizes")
    print("🚀 Ready for Phase 3: Progressive Optimization Integration")