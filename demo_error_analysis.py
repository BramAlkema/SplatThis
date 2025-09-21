#!/usr/bin/env python3
"""Demonstration of error computation and analysis framework for adaptive Gaussian splatting."""

import numpy as np
import time
from src.splat_this.core.error_analysis import (
    ErrorAnalyzer,
    ErrorMetrics,
    compute_reconstruction_error,
    create_error_visualization
)
from src.splat_this.core.adaptive_gaussian import AdaptiveGaussian2D, create_isotropic_gaussian
from src.splat_this.core.tile_renderer import TileRenderer


def create_test_target() -> np.ndarray:
    """Create a synthetic test target image with known features."""
    target = np.zeros((64, 64, 3), dtype=np.float32)

    # Background gradient
    for y in range(64):
        for x in range(64):
            target[y, x, 0] = x / 64.0  # Red gradient
            target[y, x, 1] = y / 64.0  # Green gradient
            target[y, x, 2] = 0.3       # Constant blue

    # Add geometric shapes for clear error analysis
    # Central circle
    center_y, center_x = 32, 32
    for y in range(64):
        for x in range(64):
            dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            if dist < 12:
                target[y, x] = [0.8, 0.2, 0.9]  # Purple circle

    # Horizontal edge
    target[20:24, :, :] = [0.9, 0.9, 0.1]  # Yellow bar

    # Vertical edge
    target[:, 50:54, :] = [0.1, 0.9, 0.4]  # Green bar

    return target


def create_rendered_with_errors(target: np.ndarray, error_type: str = 'uniform') -> np.ndarray:
    """Create rendered image with specific types of errors."""
    rendered = target.copy()

    if error_type == 'uniform':
        # Add uniform noise
        noise = np.random.normal(0, 0.1, target.shape)
        rendered = np.clip(rendered + noise, 0, 1)

    elif error_type == 'localized':
        # Add localized high-error regions
        rendered[40:50, 10:20, :] = 0.5  # Gray patch (high error)
        rendered[10:15, 40:50, :] = [1.0, 0.0, 0.0]  # Red patch

    elif error_type == 'edge':
        # Add errors at edges
        # Blur the yellow bar (edge degradation)
        for i in range(3):
            rendered[20:24, :, i] = np.convolve(rendered[20:24, :, i].flatten(),
                                               [0.25, 0.5, 0.25], mode='same').reshape(-1, 64)

        # Shift the green bar (edge misalignment)
        rendered[:, 50:54, :] = 0.3  # Remove original
        rendered[:, 52:56, :] = [0.1, 0.9, 0.4]  # Shift by 2 pixels

    elif error_type == 'gaussian_reconstruction':
        # Simulate Gaussian reconstruction errors
        gaussians = [
            create_isotropic_gaussian([0.5, 0.5], 0.15, [0.8, 0.2, 0.9], 0.7),  # Central purple
            create_isotropic_gaussian([0.8, 0.3], 0.08, [0.9, 0.9, 0.1], 0.8),  # Yellow approx
            create_isotropic_gaussian([0.3, 0.8], 0.08, [0.1, 0.9, 0.4], 0.8),  # Green approx
        ]

        renderer = TileRenderer((64, 64))
        rendered_rgba = renderer.render_full_image(gaussians)
        rendered = rendered_rgba[:, :, :3]  # Drop alpha

    return np.clip(rendered, 0, 1)


def demo_basic_error_metrics():
    """Demonstrate basic error metric computation."""
    print("=== Basic Error Metrics Demo ===")

    target = create_test_target()
    rendered = create_rendered_with_errors(target, 'uniform')

    analyzer = ErrorAnalyzer()
    metrics = analyzer.compute_basic_metrics(target, rendered)

    print(f"Target shape: {target.shape}")
    print(f"Rendered shape: {rendered.shape}")
    print(f"\nBasic Error Metrics:")
    print(f"  L1 Error (MAE): {metrics.l1_error:.6f}")
    print(f"  L2 Error (MSE): {metrics.l2_error:.6f}")
    print(f"  RMSE: {metrics.rmse:.6f}")
    print(f"  PSNR: {metrics.psnr:.2f} dB")

    # Coverage metrics
    print(f"\nCoverage Metrics:")
    print(f"  Coverage ratio: {metrics.coverage_ratio:.3f}")
    print(f"  Alpha mean: {metrics.alpha_mean:.3f}")
    print(f"  Alpha std: {metrics.alpha_std:.6f}")


def demo_perceptual_metrics():
    """Demonstrate perceptual error metrics."""
    print("\n=== Perceptual Error Metrics Demo ===")

    target = create_test_target()
    analyzer = ErrorAnalyzer()

    error_types = ['uniform', 'localized', 'edge']

    for error_type in error_types:
        rendered = create_rendered_with_errors(target, error_type)

        # Comprehensive metrics
        metrics = analyzer.compute_comprehensive_metrics(target, rendered)

        print(f"\n{error_type.capitalize()} Error Type:")
        print(f"  SSIM Score: {metrics.ssim_score:.4f}")
        print(f"  Edge Error: {metrics.edge_error:.6f}")
        print(f"  Smooth Error: {metrics.smooth_error:.6f}")
        print(f"  Gradient Error: {metrics.gradient_error:.6f}")
        print(f"  Overall L1: {metrics.l1_error:.6f}")


def demo_error_maps():
    """Demonstrate error map creation and visualization."""
    print("\n=== Error Map Creation Demo ===")

    target = create_test_target()
    rendered = create_rendered_with_errors(target, 'localized')

    analyzer = ErrorAnalyzer()

    # Create different types of error maps
    error_types = ['l1', 'l2', 'ssim_local']

    for error_type in error_types:
        error_map = analyzer.create_error_map(target, rendered, error_type)

        print(f"\n{error_type.upper()} Error Map:")
        print(f"  Shape: {error_map.shape}")
        print(f"  Min error: {np.min(error_map):.6f}")
        print(f"  Max error: {np.max(error_map):.6f}")
        print(f"  Mean error: {np.mean(error_map):.6f}")
        print(f"  Std error: {np.std(error_map):.6f}")

        # Find high-error pixels
        threshold = np.mean(error_map) + 2 * np.std(error_map)
        high_error_pixels = np.sum(error_map > threshold)
        print(f"  High-error pixels (>{threshold:.4f}): {high_error_pixels}")


def demo_high_error_regions():
    """Demonstrate high-error region detection."""
    print("\n=== High-Error Region Detection Demo ===")

    target = create_test_target()
    rendered = create_rendered_with_errors(target, 'localized')

    analyzer = ErrorAnalyzer()

    # Create error map
    error_map = analyzer.create_error_map(target, rendered, 'l1')

    # Detect high-error regions
    regions = analyzer.detect_high_error_regions(error_map, threshold=None, min_area=5)

    print(f"Detected {len(regions)} high-error regions:")

    for i, region in enumerate(regions[:5]):  # Show top 5
        print(f"\nRegion {i+1}:")
        print(f"  Center: {region.center}")
        print(f"  Area: {region.area} pixels")
        print(f"  Size: {region.size}")
        print(f"  Mean error: {region.mean_error:.6f}")
        print(f"  Max error: {region.max_error:.6f}")
        print(f"  Error type: {region.error_type}")
        print(f"  Priority: {region.priority:.2f}")


def demo_convergence_analysis():
    """Demonstrate convergence analysis and error history tracking."""
    print("\n=== Convergence Analysis Demo ===")

    target = create_test_target()
    analyzer = ErrorAnalyzer()

    # Simulate optimization iterations with decreasing error
    print("Simulating optimization iterations...")

    base_errors = [0.2, 0.15, 0.12, 0.1, 0.08, 0.07, 0.065, 0.063, 0.062, 0.0615,
                  0.061, 0.0608, 0.0607, 0.0606, 0.0605]

    for i, base_error in enumerate(base_errors):
        # Add small random variation
        l1_error = base_error + np.random.normal(0, 0.001)
        l2_error = base_error**2 + np.random.normal(0, 0.0001)

        metrics = ErrorMetrics(l1_error=l1_error, l2_error=l2_error)
        analyzer.track_error_history(metrics)

        if (i + 1) % 5 == 0:
            convergence = analyzer.analyze_convergence(window_size=5)
            print(f"  Iteration {i+1}: L1={l1_error:.6f}, "
                  f"Trend={convergence['trend']}, "
                  f"Plateau={convergence['plateau_detected']}")

    # Final convergence analysis
    final_convergence = analyzer.analyze_convergence()
    print(f"\nFinal Convergence Analysis:")
    print(f"  Converged: {final_convergence['converged']}")
    print(f"  Trend: {final_convergence['trend']}")
    print(f"  Improvement rate: {final_convergence['improvement_rate']:.6f}")
    print(f"  Plateau detected: {final_convergence['plateau_detected']}")
    print(f"  Slope: {final_convergence['slope']:.8f}")


def demo_quality_report():
    """Demonstrate comprehensive quality report generation."""
    print("\n=== Quality Report Demo ===")

    target = create_test_target()
    rendered = create_rendered_with_errors(target, 'gaussian_reconstruction')

    analyzer = ErrorAnalyzer()

    # Generate comprehensive quality report
    start_time = time.time()
    report = analyzer.create_quality_report(target, rendered, include_regions=True)
    report_time = time.time() - start_time

    print(f"Quality report generated in {report_time:.3f} seconds")

    # Display report summary
    print(f"\nQuality Report Summary:")
    metrics = report['metrics']
    print(f"  Overall L1 Error: {metrics['l1_error']:.6f}")
    print(f"  Overall L2 Error: {metrics['l2_error']:.6f}")
    print(f"  SSIM Score: {metrics['ssim_score']:.4f}")
    print(f"  PSNR: {metrics['psnr']:.2f} dB")

    # Error statistics
    error_stats = report['error_statistics']
    print(f"\nError Map Statistics:")
    print(f"  L1 map max: {error_stats['l1_map_max']:.6f}")
    print(f"  L1 map mean: {error_stats['l1_map_mean']:.6f}")
    print(f"  L1 map std: {error_stats['l1_map_std']:.6f}")

    # High-error regions
    if 'high_error_regions' in report:
        region_info = report['high_error_regions']
        print(f"\nHigh-Error Regions:")
        print(f"  Count: {region_info['count']}")
        print(f"  Total area: {region_info['total_area']} pixels")

        if region_info['regions']:
            top_region = region_info['regions'][0]
            print(f"  Top region: center={top_region['center']}, "
                  f"area={top_region['area']}, "
                  f"priority={top_region['priority']:.2f}")


def demo_masking_and_roi():
    """Demonstrate error analysis with masks and regions of interest."""
    print("\n=== Masking and ROI Demo ===")

    target = create_test_target()
    rendered = create_rendered_with_errors(target, 'uniform')

    # Create region of interest mask (center region)
    mask = np.zeros((64, 64))
    mask[20:44, 20:44] = 1.0  # Center 24x24 region

    analyzer = ErrorAnalyzer()

    # Compare full image vs masked analysis
    metrics_full = analyzer.compute_basic_metrics(target, rendered)
    metrics_masked = analyzer.compute_basic_metrics(target, rendered, mask)

    print(f"Error Analysis Comparison:")
    print(f"  Full image L1: {metrics_full.l1_error:.6f}")
    print(f"  Masked region L1: {metrics_masked.l1_error:.6f}")
    print(f"  Full image RMSE: {metrics_full.rmse:.6f}")
    print(f"  Masked region RMSE: {metrics_masked.rmse:.6f}")

    # SSIM comparison
    ssim_full = analyzer.compute_ssim(target, rendered)
    ssim_masked = analyzer.compute_ssim(target, rendered, mask)
    print(f"  Full image SSIM: {ssim_full:.4f}")
    print(f"  Masked region SSIM: {ssim_masked:.4f}")


def demo_convenience_functions():
    """Demonstrate convenience functions."""
    print("\n=== Convenience Functions Demo ===")

    target = create_test_target()
    rendered = create_rendered_with_errors(target, 'edge')

    # Quick reconstruction error
    metrics = compute_reconstruction_error(target, rendered)
    print(f"Quick reconstruction error analysis:")
    print(f"  L1: {metrics.l1_error:.6f}")
    print(f"  SSIM: {metrics.ssim_score:.4f}")

    # Error visualization
    visualization = create_error_visualization(target, rendered, 'l1')
    print(f"\nError visualization components:")
    for key, value in visualization.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}: shape {value.shape}")
        else:
            print(f"  {key}: {type(value)}")


if __name__ == "__main__":
    print("🎯 SplatThis Error Analysis Demonstration")
    print("=" * 60)

    demo_basic_error_metrics()
    demo_perceptual_metrics()
    demo_error_maps()
    demo_high_error_regions()
    demo_convergence_analysis()
    demo_quality_report()
    demo_masking_and_roi()
    demo_convenience_functions()

    print("\n" + "=" * 60)
    print("✅ Error Analysis Demo Complete!")
    print("📊 L1/L2 metrics: Basic reconstruction quality assessment")
    print("🔍 SSIM analysis: Perceptual quality evaluation")
    print("🗺️  Error maps: Spatial error distribution visualization")
    print("🎯 Region detection: High-error area identification")
    print("📈 Convergence tracking: Optimization progress monitoring")
    print("📋 Quality reports: Comprehensive assessment framework")
    print("🎭 Ready for Phase 2: Content-Adaptive Initialization")