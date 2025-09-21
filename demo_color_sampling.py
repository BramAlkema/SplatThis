#!/usr/bin/env python3
"""Demonstration of color sampling and validation for adaptive Gaussian splat initialization."""

import numpy as np
import time
from src.splat_this.core.color_sampling import (
    ColorSampler,
    ColorSamplingConfig,
    sample_colors_at_positions,
    validate_color_sampling
)


def create_test_images():
    """Create various test images for color sampling demonstration."""
    images = {}

    # 1. RGB gradient image
    rgb_gradient = np.zeros((64, 64, 3))
    for i in range(64):
        rgb_gradient[:, i, 0] = i / 64.0  # Red gradient
        rgb_gradient[i, :, 1] = i / 64.0  # Green gradient
        rgb_gradient[:, :, 2] = 0.5       # Constant blue
    images['rgb_gradient'] = rgb_gradient

    # 2. High contrast checkerboard
    checkerboard = np.zeros((64, 64, 3))
    for i in range(64):
        for j in range(64):
            if (i // 8 + j // 8) % 2 == 0:
                checkerboard[i, j] = [1.0, 1.0, 1.0]  # White
            else:
                checkerboard[i, j] = [0.0, 0.0, 0.0]  # Black
    images['checkerboard'] = checkerboard

    # 3. Smooth color regions
    smooth_regions = np.ones((64, 64, 3)) * 0.5  # Gray background
    # Red region
    smooth_regions[:32, :32, :] = [0.8, 0.2, 0.2]
    # Green region
    smooth_regions[:32, 32:, :] = [0.2, 0.8, 0.2]
    # Blue region
    smooth_regions[32:, :32, :] = [0.2, 0.2, 0.8]
    # Yellow region
    smooth_regions[32:, 32:, :] = [0.8, 0.8, 0.2]
    images['smooth_regions'] = smooth_regions

    # 4. Noisy texture
    np.random.seed(42)
    noisy_texture = np.random.rand(64, 64, 3)
    # Add some structure
    noisy_texture[20:44, 20:44, :] = np.random.rand(24, 24, 3) * 0.3 + 0.7  # Bright region
    images['noisy_texture'] = noisy_texture

    # 5. Color outliers image
    outliers_image = np.ones((64, 64, 3)) * 0.5  # Gray background
    # Add some outlier pixels
    outliers_image[10, 10] = [10.0, 0.0, 0.0]  # Bright red outlier
    outliers_image[30, 30] = [0.0, 10.0, 0.0]  # Bright green outlier
    outliers_image[50, 50] = [0.0, 0.0, 10.0]  # Bright blue outlier
    images['color_outliers'] = outliers_image

    # 6. Multi-channel image (5 channels)
    multi_channel = np.random.rand(64, 64, 5)
    # Make channels somewhat correlated
    multi_channel[:, :, 1] = multi_channel[:, :, 0] * 0.8 + 0.1
    multi_channel[:, :, 2] = 1.0 - multi_channel[:, :, 0]
    images['multi_channel'] = multi_channel

    # 7. Grayscale image
    grayscale = np.zeros((64, 64))
    for i in range(64):
        for j in range(64):
            grayscale[i, j] = (np.sin(i * 0.1) + np.cos(j * 0.1)) * 0.25 + 0.5
    images['grayscale'] = grayscale

    return images


def demo_basic_color_sampling():
    """Demonstrate basic color sampling functionality."""
    print("=== Basic Color Sampling Demo ===")

    images = create_test_images()
    image = images['rgb_gradient']

    # Define test positions
    positions = np.array([
        [16, 16],  # Low red, low green
        [16, 48],  # Low red, high green
        [48, 16],  # High red, low green
        [48, 48],  # High red, high green
        [32, 32],  # Middle values
    ])

    sampler = ColorSampler()

    print(f"Input image shape: {image.shape}")
    print(f"Sampling at {len(positions)} positions")

    start_time = time.time()
    result = sampler.sample_colors(image, positions)
    sampling_time = time.time() - start_time

    print(f"Color sampling completed in {sampling_time:.4f} seconds")
    print(f"Sampled colors shape: {result.colors.shape}")

    # Display sampled colors
    print(f"\nSampled Colors:")
    for i, (pos, sample) in enumerate(zip(positions, result.samples)):
        color = sample.color
        confidence = sample.confidence
        print(f"  Position ({pos[0]:2d}, {pos[1]:2d}): "
              f"RGB({color[0]:.3f}, {color[1]:.3f}, {color[2]:.3f}) "
              f"confidence={confidence:.3f}")

    # Display statistics
    stats = result.statistics
    print(f"\nSampling Statistics:")
    print(f"  Total samples: {stats['total_samples']}")
    print(f"  Valid samples: {stats['valid_samples']}")
    print(f"  Outliers detected: {stats['outlier_count']}")
    print(f"  Color diversity: {stats['color_diversity']:.3f}")


def demo_interpolation_methods():
    """Demonstrate different interpolation methods."""
    print("\n=== Interpolation Methods Demo ===")

    image = create_test_images()['checkerboard']

    # Test position between pixels for clear interpolation effect
    positions = np.array([[31.5, 31.5]])  # Between black and white squares

    methods = ['nearest', 'bilinear', 'bicubic']

    for method in methods:
        config = ColorSamplingConfig(interpolation_method=method)
        sampler = ColorSampler(config)
        result = sampler.sample_colors(image, positions)

        color = result.colors[0]
        interpolated = result.samples[0].interpolated

        print(f"\n{method.capitalize()} Interpolation:")
        print(f"  Sampled color: RGB({color[0]:.3f}, {color[1]:.3f}, {color[2]:.3f})")
        print(f"  Interpolated: {interpolated}")

        # Nearest should be exactly 0 or 1, others should be in between
        if method == 'nearest':
            print(f"  Expected: close to 0.0 or 1.0 (exact pixel value)")
        else:
            print(f"  Expected: between 0.0 and 1.0 (interpolated)")


def demo_boundary_handling():
    """Demonstrate different boundary handling methods."""
    print("\n=== Boundary Handling Demo ===")

    image = create_test_images()['smooth_regions']

    # Positions outside image bounds
    positions = np.array([
        [-5, -5],     # Far outside
        [70, 70],     # Far outside
        [2, 2],       # Near boundary, inside
        [62, 62],     # Near boundary, inside
    ])

    boundary_methods = ['clamp', 'wrap', 'mirror']

    for method in boundary_methods:
        config = ColorSamplingConfig(boundary_handling=method)
        sampler = ColorSampler(config)
        result = sampler.sample_colors(image, positions)

        print(f"\n{method.capitalize()} Boundary Handling:")
        for i, (pos, color) in enumerate(zip(positions, result.colors)):
            print(f"  Position ({pos[0]:3d}, {pos[1]:3d}): "
                  f"RGB({color[0]:.3f}, {color[1]:.3f}, {color[2]:.3f})")


def demo_outlier_detection():
    """Demonstrate outlier detection and handling."""
    print("\n=== Outlier Detection Demo ===")

    image = create_test_images()['color_outliers']

    # Sample at outlier and normal positions
    positions = np.array([
        [10, 10],  # Red outlier
        [30, 30],  # Green outlier
        [50, 50],  # Blue outlier
        [5, 5],    # Normal
        [25, 25],  # Normal
        [45, 45],  # Normal
    ])

    # Test with and without outlier detection
    configs = [
        ("No Outlier Detection", ColorSamplingConfig(
            outlier_detection=False,
            gamma_correction=False,
            normalization='none'
        )),
        ("Strict Outlier Detection", ColorSamplingConfig(
            outlier_detection=True,
            outlier_threshold=1.0,
            gamma_correction=False,
            normalization='none'
        )),
        ("Lenient Outlier Detection", ColorSamplingConfig(
            outlier_detection=True,
            outlier_threshold=3.0,
            gamma_correction=False,
            normalization='none'
        ))
    ]

    for name, config in configs:
        sampler = ColorSampler(config)
        result = sampler.sample_colors(image, positions)

        print(f"\n{name}:")
        outlier_count = np.sum(result.outlier_mask)
        print(f"  Outliers detected: {outlier_count}/{len(positions)}")

        for i, (pos, sample) in enumerate(zip(positions, result.samples)):
            outlier_flag = "OUTLIER" if sample.is_outlier else "normal"
            print(f"    Position ({pos[0]:2d}, {pos[1]:2d}): {outlier_flag}")


def demo_preprocessing_effects():
    """Demonstrate effects of preprocessing options."""
    print("\n=== Preprocessing Effects Demo ===")

    image = create_test_images()['noisy_texture']
    positions = np.array([[32, 32]])

    preprocessing_configs = [
        ("No Preprocessing", ColorSamplingConfig(
            gamma_correction=False,
            normalization='none',
            smoothing_radius=0.0
        )),
        ("Gamma Correction", ColorSamplingConfig(
            gamma_correction=True,
            gamma_value=2.2,
            normalization='none',
            smoothing_radius=0.0
        )),
        ("Min-Max Normalization", ColorSamplingConfig(
            gamma_correction=False,
            normalization='minmax',
            smoothing_radius=0.0
        )),
        ("Z-Score Normalization", ColorSamplingConfig(
            gamma_correction=False,
            normalization='zscore',
            smoothing_radius=0.0
        )),
        ("Smoothing", ColorSamplingConfig(
            gamma_correction=False,
            normalization='none',
            smoothing_radius=2.0
        )),
        ("Full Preprocessing", ColorSamplingConfig(
            gamma_correction=True,
            normalization='minmax',
            smoothing_radius=1.0
        ))
    ]

    for name, config in preprocessing_configs:
        sampler = ColorSampler(config)
        result = sampler.sample_colors(image, positions)

        color = result.colors[0]
        original_color = result.samples[0].original_color

        print(f"\n{name}:")
        print(f"  Original: RGB({original_color[0]:.3f}, {original_color[1]:.3f}, {original_color[2]:.3f})")
        print(f"  Processed: RGB({color[0]:.3f}, {color[1]:.3f}, {color[2]:.3f})")


def demo_color_space_conversion():
    """Demonstrate color space conversions."""
    print("\n=== Color Space Conversion Demo ===")

    image = create_test_images()['rgb_gradient']
    positions = np.array([[16, 48], [48, 16]])  # Two different color positions

    color_spaces = ['RGB', 'LAB', 'HSV', 'YUV']

    for color_space in color_spaces:
        config = ColorSamplingConfig(color_space=color_space)
        sampler = ColorSampler(config)
        result = sampler.sample_colors(image, positions)

        print(f"\n{color_space} Color Space:")
        for i, (pos, color) in enumerate(zip(positions, result.colors)):
            if len(color) == 3:
                print(f"  Position ({pos[0]:2d}, {pos[1]:2d}): "
                      f"{color_space}({color[0]:.3f}, {color[1]:.3f}, {color[2]:.3f})")
            else:
                print(f"  Position ({pos[0]:2d}, {pos[1]:2d}): {color}")


def demo_multi_channel_support():
    """Demonstrate multi-channel image support."""
    print("\n=== Multi-Channel Support Demo ===")

    image = create_test_images()['multi_channel']  # 5 channels
    positions = np.array([[32, 32], [16, 48]])

    config = ColorSamplingConfig(multi_channel_support=True)
    sampler = ColorSampler(config)
    result = sampler.sample_colors(image, positions)

    print(f"Multi-channel image shape: {image.shape}")
    print(f"Sampled colors shape: {result.colors.shape}")

    print(f"\nSampled Colors:")
    for i, (pos, color) in enumerate(zip(positions, result.colors)):
        color_str = ", ".join([f"{c:.3f}" for c in color])
        print(f"  Position ({pos[0]:2d}, {pos[1]:2d}): [{color_str}]")


def demo_confidence_scoring():
    """Demonstrate confidence scoring for different sampling scenarios."""
    print("\n=== Confidence Scoring Demo ===")

    image = create_test_images()['smooth_regions']

    # Various positions with different expected confidence levels
    test_scenarios = [
        ("Center pixel", np.array([[32.0, 32.0]])),           # Exact pixel, center
        ("Sub-pixel center", np.array([[32.5, 32.5]])),      # Interpolated, center
        ("Near boundary", np.array([[2.0, 2.0]])),           # Exact pixel, near edge
        ("Sub-pixel boundary", np.array([[1.5, 1.5]])),      # Interpolated, near edge
        ("Very near edge", np.array([[0.5, 0.5]])),          # Very close to edge
    ]

    sampler = ColorSampler()

    for scenario_name, positions in test_scenarios:
        result = sampler.sample_colors(image, positions)
        sample = result.samples[0]

        print(f"\n{scenario_name}:")
        print(f"  Position: ({positions[0][0]:.1f}, {positions[0][1]:.1f})")
        print(f"  Confidence: {sample.confidence:.3f}")
        print(f"  Interpolated: {sample.interpolated}")


def demo_validation_accuracy():
    """Demonstrate color accuracy validation."""
    print("\n=== Validation Accuracy Demo ===")

    image = create_test_images()['rgb_gradient']

    # Test exact pixel sampling vs interpolated sampling
    test_cases = [
        ("Exact Pixels", np.array([[16.0, 16.0], [32.0, 32.0], [48.0, 48.0]])),
        ("Sub-pixel", np.array([[16.5, 16.5], [32.5, 32.5], [48.5, 48.5]])),
        ("Mixed", np.array([[16.0, 16.0], [32.5, 32.5], [48.0, 48.0]])),
    ]

    for case_name, positions in test_cases:
        sampler = ColorSampler()
        result = sampler.sample_colors(image, positions)

        validation = result.validation_results
        print(f"\n{case_name}:")
        print(f"  Validation count: {validation['validation_count']}")
        print(f"  Mean error: {validation['mean_error']:.6f}")
        print(f"  Max error: {validation['max_error']:.6f}")
        print(f"  Accuracy within tolerance: {validation['accuracy_within_tolerance']:.3f}")


def demo_sampling_quality():
    """Demonstrate overall sampling quality assessment."""
    print("\n=== Sampling Quality Demo ===")

    test_images = ['rgb_gradient', 'checkerboard', 'smooth_regions', 'noisy_texture']
    images = create_test_images()

    for image_name in test_images:
        image = images[image_name]

        # Sample multiple positions
        np.random.seed(42)
        positions = np.random.rand(20, 2) * 60 + 2  # Avoid extreme edges

        sampler = ColorSampler()
        result = sampler.sample_colors(image, positions)
        quality = sampler.validate_sampling_quality(result)

        print(f"\n{image_name.replace('_', ' ').title()}:")
        print(f"  Quality score: {quality['quality_score']:.3f}")
        print(f"  Validation passed: {quality['passed']}")
        print(f"  Issues: {len(quality['issues'])}")

        if quality['issues']:
            print(f"  Top issues:")
            for issue in quality['issues'][:2]:
                print(f"    - {issue}")


def demo_performance_scaling():
    """Demonstrate performance scaling with different configurations."""
    print("\n=== Performance Scaling Demo ===")

    # Test different numbers of samples
    sample_counts = [10, 50, 100, 500, 1000]
    image = create_test_images()['rgb_gradient']

    for count in sample_counts:
        # Generate random positions
        np.random.seed(42)
        positions = np.random.rand(count, 2) * 60 + 2

        sampler = ColorSampler()

        start_time = time.time()
        result = sampler.sample_colors(image, positions)
        sampling_time = time.time() - start_time

        time_per_sample = sampling_time / count * 1000  # milliseconds

        print(f"  {count:4d} samples: {sampling_time:.4f}s total, {time_per_sample:.3f}ms/sample")


def demo_convenience_functions():
    """Demonstrate convenience functions."""
    print("\n=== Convenience Functions Demo ===")

    image = create_test_images()['smooth_regions']
    positions = np.array([[16, 16], [48, 48]])

    # Using convenience function for sampling
    print("Quick color sampling:")
    result = sample_colors_at_positions(image, positions, interpolation='bilinear')

    print(f"  Sampled {len(result.colors)} colors")
    for i, (pos, color) in enumerate(zip(positions, result.colors)):
        print(f"    Position ({pos[0]}, {pos[1]}): RGB({color[0]:.3f}, {color[1]:.3f}, {color[2]:.3f})")

    # Using convenience function for validation
    print("\nQuick validation:")
    validation = validate_color_sampling(image, positions, result.colors)

    print(f"  Validation count: {validation['validation_count']}")
    print(f"  Mean error: {validation['mean_error']:.6f}")
    print(f"  Accuracy: {validation['accuracy_within_tolerance']:.3f}")


if __name__ == "__main__":
    print("🎨 SplatThis Color Sampling and Validation Demonstration")
    print("=" * 80)

    demo_basic_color_sampling()
    demo_interpolation_methods()
    demo_boundary_handling()
    demo_outlier_detection()
    demo_preprocessing_effects()
    demo_color_space_conversion()
    demo_multi_channel_support()
    demo_confidence_scoring()
    demo_validation_accuracy()
    demo_sampling_quality()
    demo_performance_scaling()
    demo_convenience_functions()

    print("\n" + "=" * 80)
    print("✅ Color Sampling and Validation Demo Complete!")
    print("🎨 Color accuracy: Precise sampling with sub-pixel interpolation")
    print("🔍 Outlier detection: Automatic identification of problematic samples")
    print("🌈 Multi-channel: Support for RGB, LAB, HSV, YUV, and custom formats")
    print("⚙️  Preprocessing: Gamma correction, normalization, and smoothing")
    print("📊 Quality validation: Comprehensive accuracy and confidence metrics")
    print("⚡ Performance: Efficient sampling for large position sets")
    print("🚀 Phase 2 Complete: Ready for Phase 3 Progressive Optimization!")