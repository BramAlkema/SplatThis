#!/usr/bin/env python3
"""Demonstration of gradient-guided placement algorithm for adaptive Gaussian splatting."""

import numpy as np
import time
from src.splat_this.core.placement_algorithm import (
    GradientGuidedPlacer,
    PlacementConfig,
    create_gradient_guided_placer,
    place_adaptive_splats
)


def create_test_images():
    """Create various test images for placement algorithm demonstration."""
    images = {}

    # 1. Simple gradient image
    simple = np.zeros((64, 64, 3))
    for y in range(64):
        for x in range(64):
            simple[y, x, 0] = x / 64.0  # Red gradient
            simple[y, x, 1] = y / 64.0  # Green gradient
            simple[y, x, 2] = 0.3
    images['simple_gradient'] = simple

    # 2. Edge-rich image
    edge_rich = np.zeros((64, 64, 3))
    # Horizontal stripes
    for i in range(0, 64, 8):
        edge_rich[i:i+4, :, :] = [0.8, 0.2, 0.1]
    # Vertical stripes
    for i in range(0, 64, 12):
        edge_rich[:, i:i+6, :] = [0.1, 0.8, 0.3]
    # Central circle
    center = 32
    for y in range(64):
        for x in range(64):
            if (x - center)**2 + (y - center)**2 < 15**2:
                edge_rich[y, x, :] = [0.9, 0.9, 0.1]
    images['edge_rich'] = edge_rich

    # 3. Smooth regions with detailed areas
    mixed = np.ones((64, 64, 3)) * 0.5  # Gray background
    # Smooth gradient in top-left
    for y in range(32):
        for x in range(32):
            mixed[y, x, :] = [x/32.0, y/32.0, 0.8]
    # Detailed texture in bottom-right
    np.random.seed(42)
    texture = np.random.rand(32, 32, 3) * 0.5 + 0.25
    mixed[32:, 32:, :] = texture
    images['mixed_complexity'] = mixed

    # 4. High-contrast checkerboard
    checkerboard = np.zeros((64, 64, 3))
    for y in range(64):
        for x in range(64):
            if (x // 8 + y // 8) % 2 == 0:
                checkerboard[y, x, :] = [1.0, 1.0, 1.0]
            else:
                checkerboard[y, x, :] = [0.0, 0.0, 0.0]
    images['checkerboard'] = checkerboard

    # 5. Natural-like image (simulated)
    natural = np.zeros((64, 64, 3))
    # Sky region (smooth)
    natural[:20, :, :] = [0.3, 0.5, 0.8]
    # Mountain silhouette (edges)
    mountain_y = 20 + np.sin(np.linspace(0, 4*np.pi, 64)) * 5
    for x in range(64):
        y_level = int(mountain_y[x])
        natural[:y_level, x, :] = [0.3, 0.5, 0.8]  # Sky
        natural[y_level:40, x, :] = [0.2, 0.4, 0.1]  # Mountain
    # Ground with texture
    for y in range(40, 64):
        for x in range(64):
            noise = np.random.normal(0, 0.1)
            natural[y, x, :] = [0.1 + noise, 0.6 + noise, 0.2 + noise]
    images['natural_like'] = np.clip(natural, 0, 1)

    return images


def demo_basic_placement():
    """Demonstrate basic gradient-guided placement."""
    print("=== Basic Gradient-Guided Placement Demo ===")

    image = create_test_images()['simple_gradient']
    placer = GradientGuidedPlacer((64, 64))

    print(f"Input image shape: {image.shape}")

    # Place splats with default settings
    start_time = time.time()
    result = placer.place_splats(image, target_count=100)
    placement_time = time.time() - start_time

    print(f"Placement completed in {placement_time:.3f} seconds")
    print(f"Placed {result.total_splats} splats")
    print(f"Coverage achieved: {result.coverage_achieved:.3f}")
    print(f"Distribution uniformity: {result.distribution_uniformity:.3f}")
    print(f"Gradient alignment: {result.gradient_alignment:.3f}")

    # Analyze placement composition
    print(f"\nPlacement Composition:")
    print(f"  Gradient-guided: {result.gradient_guided_splats}")
    print(f"  Uniform coverage: {result.uniform_coverage_splats}")
    print(f"  Maxima-based: {result.maxima_based_splats}")


def demo_configuration_effects():
    """Demonstrate effects of different configuration parameters."""
    print("\n=== Configuration Effects Demo ===")

    image = create_test_images()['edge_rich']

    configs = [
        ("High Gradient Weight", PlacementConfig(gradient_weight=0.9, uniform_weight=0.1)),
        ("Balanced Weights", PlacementConfig(gradient_weight=0.5, uniform_weight=0.5)),
        ("High Uniform Weight", PlacementConfig(gradient_weight=0.2, uniform_weight=0.8)),
        ("High Density", PlacementConfig(base_splat_density=0.03)),
        ("Low Density", PlacementConfig(base_splat_density=0.005)),
        ("Strong Maxima", PlacementConfig(maxima_weight=0.5, maxima_threshold=0.05)),
        ("Weak Maxima", PlacementConfig(maxima_weight=0.1, maxima_threshold=0.2))
    ]

    for name, config in configs:
        placer = GradientGuidedPlacer((64, 64), config)
        result = placer.place_splats(image, target_count=80)

        print(f"\n{name}:")
        print(f"  Splats placed: {result.total_splats}")
        print(f"  Coverage: {result.coverage_achieved:.3f}")
        print(f"  Uniformity: {result.distribution_uniformity:.3f}")
        print(f"  Gradient alignment: {result.gradient_alignment:.3f}")
        print(f"  Complexity mean: {np.mean(result.complexity_map):.3f}")


def demo_adaptive_density():
    """Demonstrate adaptive density based on image complexity."""
    print("\n=== Adaptive Density Demo ===")

    images = create_test_images()
    placer = GradientGuidedPlacer((64, 64))

    for name, image in images.items():
        print(f"\n{name.replace('_', ' ').title()}:")

        # Compute complexity and density
        complexity_map = placer.compute_image_complexity(image)
        density_map, total_count = placer.compute_adaptive_density(complexity_map)

        print(f"  Complexity - min: {np.min(complexity_map):.3f}, "
              f"max: {np.max(complexity_map):.3f}, "
              f"mean: {np.mean(complexity_map):.3f}")
        print(f"  Density - min: {np.min(density_map):.6f}, "
              f"max: {np.max(density_map):.6f}, "
              f"mean: {np.mean(density_map):.6f}")
        print(f"  Auto splat count: {total_count}")

        # Place splats with automatic count
        result = placer.place_splats(image)
        print(f"  Actual splats placed: {result.total_splats}")
        print(f"  Coverage achieved: {result.coverage_achieved:.3f}")


def demo_quality_validation():
    """Demonstrate placement quality validation."""
    print("\n=== Quality Validation Demo ===")

    image = create_test_images()['mixed_complexity']

    # Test with strict quality requirements
    strict_config = PlacementConfig(
        coverage_target=0.9,
        distribution_uniformity_threshold=0.8,
        base_splat_density=0.02
    )

    # Test with lenient quality requirements
    lenient_config = PlacementConfig(
        coverage_target=0.5,
        distribution_uniformity_threshold=0.3,
        base_splat_density=0.01
    )

    configs = [("Strict", strict_config), ("Lenient", lenient_config)]

    for name, config in configs:
        placer = GradientGuidedPlacer((64, 64), config)
        result = placer.place_splats(image, target_count=120)
        validation = placer.validate_placement_quality(result)

        print(f"\n{name} Validation:")
        print(f"  Passed: {validation['passed']}")
        print(f"  Issues: {len(validation['issues'])}")
        if validation['issues']:
            for issue in validation['issues']:
                print(f"    - {issue}")
        if validation['recommendations']:
            print(f"  Recommendations: {len(validation['recommendations'])}")
            for rec in validation['recommendations'][:2]:  # Show first 2
                print(f"    - {rec}")


def demo_local_maxima_detection():
    """Demonstrate local maxima detection and influence."""
    print("\n=== Local Maxima Detection Demo ===")

    image = create_test_images()['checkerboard']

    # Test with different maxima configurations
    maxima_configs = [
        PlacementConfig(maxima_threshold=0.1, maxima_min_distance=3, maxima_weight=0.0),
        PlacementConfig(maxima_threshold=0.1, maxima_min_distance=3, maxima_weight=0.3),
        PlacementConfig(maxima_threshold=0.05, maxima_min_distance=5, maxima_weight=0.5)
    ]

    config_names = ["No Maxima", "Moderate Maxima", "Strong Maxima"]

    for name, config in zip(config_names, maxima_configs):
        placer = GradientGuidedPlacer((64, 64), config)

        # Detect maxima
        maxima = placer.detect_gradient_maxima(image)

        # Place splats
        result = placer.place_splats(image, target_count=100)

        print(f"\n{name}:")
        print(f"  Detected maxima: {len(maxima)}")
        print(f"  Maxima-aligned splats: {result.maxima_based_splats}")
        print(f"  Gradient alignment: {result.gradient_alignment:.3f}")
        print(f"  Distribution uniformity: {result.distribution_uniformity:.3f}")


def demo_placement_statistics():
    """Demonstrate comprehensive placement statistics."""
    print("\n=== Placement Statistics Demo ===")

    image = create_test_images()['natural_like']
    placer = GradientGuidedPlacer((64, 64))

    result = placer.place_splats(image, target_count=150)
    stats = placer.get_placement_statistics(result)

    print(f"Comprehensive Statistics:")
    print(f"  Total splats: {stats['total_splats']}")
    print(f"  Coverage achieved: {stats['coverage_achieved']:.3f}")
    print(f"  Distribution uniformity: {stats['distribution_uniformity']:.3f}")
    print(f"  Gradient alignment: {stats['gradient_alignment']:.3f}")

    print(f"\nComplexity Statistics:")
    comp_stats = stats['complexity_statistics']
    print(f"  Mean: {comp_stats['mean']:.3f}")
    print(f"  Std: {comp_stats['std']:.3f}")
    print(f"  Max: {comp_stats['max']:.3f}")
    print(f"  High complexity fraction: {comp_stats['high_complexity_fraction']:.3f}")

    print(f"\nDensity Statistics:")
    density_stats = stats['density_statistics']
    print(f"  Mean: {density_stats['mean']:.6f}")
    print(f"  Std: {density_stats['std']:.6f}")
    print(f"  Max: {density_stats['max']:.6f}")
    print(f"  Range: {density_stats['density_range']:.6f}")

    print(f"\nSpatial Distribution:")
    spatial = stats['spatial_distribution']
    print(f"  Gradient-guided fraction: {spatial['gradient_guided_fraction']:.3f}")
    print(f"  Uniform coverage fraction: {spatial['uniform_coverage_fraction']:.3f}")
    print(f"  Maxima-aligned fraction: {spatial['maxima_aligned_fraction']:.3f}")


def demo_performance_scaling():
    """Demonstrate performance scaling with image size and splat count."""
    print("\n=== Performance Scaling Demo ===")

    # Test different image sizes
    image_sizes = [(32, 32), (64, 64), (128, 128), (256, 256)]
    splat_counts = [50, 200, 800, 3200]

    for (h, w), splat_count in zip(image_sizes, splat_counts):
        # Create test image
        image = np.random.rand(h, w, 3)

        placer = GradientGuidedPlacer((h, w))

        start_time = time.time()
        result = placer.place_splats(image, target_count=splat_count)
        placement_time = time.time() - start_time

        pixels = h * w
        splats_per_pixel = result.total_splats / pixels

        print(f"  {w}x{h} ({pixels:6d} px): "
              f"{result.total_splats:4d} splats ({splats_per_pixel:.4f}/px), "
              f"{placement_time:.3f}s, "
              f"{placement_time/result.total_splats*1000:.2f}ms/splat")


def demo_convenience_functions():
    """Demonstrate convenience functions."""
    print("\n=== Convenience Functions Demo ===")

    image = create_test_images()['edge_rich']

    # Using convenience function with default config
    print("Default placement:")
    result1 = place_adaptive_splats(image, target_count=80)
    print(f"  Placed {result1.total_splats} splats")
    print(f"  Coverage: {result1.coverage_achieved:.3f}")

    # Using convenience function with custom config
    print("\nCustom configuration:")
    custom_config = {
        'gradient_weight': 0.8,
        'uniform_weight': 0.2,
        'base_splat_density': 0.02
    }
    result2 = place_adaptive_splats(image, config=custom_config)
    print(f"  Placed {result2.total_splats} splats")
    print(f"  Coverage: {result2.coverage_achieved:.3f}")

    # Using factory function
    print("\nFactory function:")
    placer = create_gradient_guided_placer((64, 64), custom_config)
    result3 = placer.place_splats(image, target_count=100)
    print(f"  Placed {result3.total_splats} splats")
    print(f"  Coverage: {result3.coverage_achieved:.3f}")


def demo_comparison_with_uniform():
    """Compare gradient-guided placement with uniform placement."""
    print("\n=== Comparison with Uniform Placement ===")

    image = create_test_images()['natural_like']

    # Gradient-guided placement
    placer = GradientGuidedPlacer((64, 64))
    gradient_result = placer.place_splats(image, target_count=100)

    # Simulate uniform placement (minimal gradient influence)
    uniform_config = PlacementConfig(gradient_weight=0.01, uniform_weight=0.99)
    uniform_placer = GradientGuidedPlacer((64, 64), uniform_config)
    uniform_result = uniform_placer.place_splats(image, target_count=100)

    print("Gradient-Guided vs Uniform Placement:")
    print(f"  Coverage:")
    print(f"    Gradient-guided: {gradient_result.coverage_achieved:.3f}")
    print(f"    Uniform: {uniform_result.coverage_achieved:.3f}")

    print(f"  Distribution uniformity:")
    print(f"    Gradient-guided: {gradient_result.distribution_uniformity:.3f}")
    print(f"    Uniform: {uniform_result.distribution_uniformity:.3f}")

    print(f"  Gradient alignment:")
    print(f"    Gradient-guided: {gradient_result.gradient_alignment:.3f}")
    print(f"    Uniform: {uniform_result.gradient_alignment:.3f}")

    print(f"  Content responsiveness:")
    gradient_complexity_corr = np.corrcoef(
        gradient_result.complexity_map.flatten(),
        gradient_result.density_map.flatten()
    )[0, 1]
    uniform_complexity_corr = np.corrcoef(
        uniform_result.complexity_map.flatten(),
        uniform_result.density_map.flatten()
    )[0, 1]

    print(f"    Gradient-guided complexity correlation: {gradient_complexity_corr:.3f}")
    print(f"    Uniform complexity correlation: {uniform_complexity_corr:.3f}")


if __name__ == "__main__":
    print("🎯 SplatThis Gradient-Guided Placement Demonstration")
    print("=" * 70)

    demo_basic_placement()
    demo_configuration_effects()
    demo_adaptive_density()
    demo_quality_validation()
    demo_local_maxima_detection()
    demo_placement_statistics()
    demo_performance_scaling()
    demo_convenience_functions()
    demo_comparison_with_uniform()

    print("\n" + "=" * 70)
    print("✅ Gradient-Guided Placement Demo Complete!")
    print("🎯 Content-adaptive: Splats placed based on image complexity")
    print("📊 Quality-driven: Validation ensures coverage and uniformity")
    print("🔍 Feature-aware: Local maxima detection for strategic placement")
    print("⚖️  Balanced approach: Gradient guidance + uniform coverage guarantee")
    print("📈 Scalable performance: Efficient for various image sizes")
    print("🔧 Ready for next phase: Structure Tensor Integration")