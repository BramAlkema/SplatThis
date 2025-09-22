#!/usr/bin/env python3
"""Demonstration of advanced error metrics for adaptive Gaussian splatting.

This script showcases the T4.2: Advanced Error Metrics implementation, demonstrating
LPIPS perceptual metrics, frequency-domain analysis, content-aware error weighting,
and comparative quality assessment.
"""

import sys
import time
import logging
import argparse
import numpy as np
from typing import Dict, List, Any
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.splat_this.core.advanced_error_metrics import (
    AdvancedErrorAnalyzer,
    LPIPSCalculator,
    FrequencyAnalyzer,
    ContentAwareAnalyzer,
    ComparativeQualityAssessment,
    PerceptualMetric,
    compare_reconstruction_methods
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_images(size: int = 128) -> Dict[str, np.ndarray]:
    """Create a set of test images for demonstration."""
    np.random.seed(42)  # For reproducible results

    # Target image: complex scene with different content types
    target = np.zeros((size, size, 3))

    # Add smooth background gradient
    y, x = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size), indexing='ij')
    target[:, :, 0] = 0.3 + 0.4 * x  # Red gradient
    target[:, :, 1] = 0.2 + 0.3 * y  # Green gradient
    target[:, :, 2] = 0.5 + 0.2 * np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)  # Blue pattern

    # Add some textured regions
    center_y, center_x = size // 2, size // 2
    radius = size // 6

    # Textured circle
    for i in range(size):
        for j in range(size):
            dist = np.sqrt((i - center_y)**2 + (j - center_x)**2)
            if dist < radius:
                # Add high-frequency texture
                texture = 0.3 * np.sin(10 * np.pi * i / size) * np.cos(10 * np.pi * j / size)
                target[i, j] += texture

    # Add bright square
    square_size = size // 8
    bright_y = center_y - radius - square_size
    bright_x = center_x
    target[bright_y:bright_y+square_size, bright_x:bright_x+square_size] = [0.9, 0.9, 0.8]

    # Add dark square
    dark_y = center_y + radius
    dark_x = center_x
    target[dark_y:dark_y+square_size, dark_x:dark_x+square_size] = [0.1, 0.1, 0.2]

    # Add some edges
    edge_width = 2
    target[size//4:size//4+edge_width, :] = [1.0, 0.0, 0.0]  # Horizontal red line
    target[:, size//4:size//4+edge_width] = [0.0, 1.0, 0.0]  # Vertical green line

    # Clip to valid range
    target = np.clip(target, 0.0, 1.0)

    # Create different reconstruction methods with varying quality
    reconstructions = {}

    # Method 1: High quality (small errors)
    noise_scale = 0.05
    reconstructions["HighQuality"] = np.clip(target + noise_scale * np.random.randn(*target.shape), 0.0, 1.0)

    # Method 2: Medium quality (moderate blurring)
    from scipy.ndimage import gaussian_filter
    reconstructions["MediumQuality"] = gaussian_filter(target, sigma=1.5)
    reconstructions["MediumQuality"] += 0.1 * np.random.randn(*target.shape)
    reconstructions["MediumQuality"] = np.clip(reconstructions["MediumQuality"], 0.0, 1.0)

    # Method 3: Low quality (significant distortion)
    # Downsample and upsample to simulate poor reconstruction
    from skimage.transform import resize
    low_res = resize(target, (size//4, size//4, 3), anti_aliasing=True)
    reconstructions["LowQuality"] = resize(low_res, (size, size, 3), anti_aliasing=True)
    reconstructions["LowQuality"] += 0.2 * np.random.randn(*target.shape)
    reconstructions["LowQuality"] = np.clip(reconstructions["LowQuality"], 0.0, 1.0)

    # Method 4: Edge-preserving but smooth
    reconstructions["EdgePreserving"] = target.copy()
    # Apply bilateral-like filtering (smooth non-edge regions)
    for c in range(3):
        reconstructions["EdgePreserving"][:, :, c] = gaussian_filter(
            reconstructions["EdgePreserving"][:, :, c], sigma=0.8)
    reconstructions["EdgePreserving"] = np.clip(reconstructions["EdgePreserving"], 0.0, 1.0)

    # Method 5: High-frequency artifacts
    reconstructions["HighFreqArtifacts"] = target.copy()
    # Add high-frequency noise
    high_freq_noise = 0.15 * np.random.randn(*target.shape)
    reconstructions["HighFreqArtifacts"] += high_freq_noise
    reconstructions["HighFreqArtifacts"] = np.clip(reconstructions["HighFreqArtifacts"], 0.0, 1.0)

    return {"target": target, **reconstructions}


def demonstrate_lpips_metrics():
    """Demonstrate LPIPS perceptual metrics."""
    print("\n" + "="*60)
    print("🎯 LPIPS PERCEPTUAL METRICS DEMONSTRATION")
    print("="*60)

    # Create test images
    images = create_test_images(128)
    target = images["target"]

    # Initialize LPIPS calculator
    lpips_calc = LPIPSCalculator(PerceptualMetric.LPIPS_VGG)

    print("\n📊 LPIPS Scores (lower = more perceptually similar):")
    print("-" * 50)

    results = {}
    for method_name, reconstruction in images.items():
        if method_name == "target":
            continue

        start_time = time.time()
        lpips_score = lpips_calc.compute_lpips(target, reconstruction)
        duration = time.time() - start_time

        results[method_name] = lpips_score
        print(f"   {method_name:18s}: {lpips_score:.4f} ({duration*1000:.1f}ms)")

    # Rank methods by LPIPS
    sorted_methods = sorted(results.items(), key=lambda x: x[1])
    print(f"\n🏆 Best to Worst (by LPIPS):")
    for rank, (method, score) in enumerate(sorted_methods, 1):
        print(f"   {rank}. {method} (LPIPS: {score:.4f})")

    return results


def demonstrate_frequency_analysis():
    """Demonstrate frequency-domain error analysis."""
    print("\n" + "="*60)
    print("🌊 FREQUENCY-DOMAIN ERROR ANALYSIS")
    print("="*60)

    # Create test images
    images = create_test_images(128)
    target = images["target"]

    # Initialize frequency analyzer
    freq_analyzer = FrequencyAnalyzer()

    print("\n📊 Frequency Band Analysis:")
    print("-" * 50)

    results = {}
    for method_name, reconstruction in images.items():
        if method_name == "target":
            continue

        start_time = time.time()
        freq_metrics = freq_analyzer.compute_frequency_metrics(target, reconstruction)
        duration = time.time() - start_time

        results[method_name] = freq_metrics

        print(f"\n📈 {method_name} ({duration*1000:.1f}ms):")
        print(f"   Spectral Distortion:     {freq_metrics['spectral_distortion']:.6f}")
        print(f"   High-Freq Preservation:  {freq_metrics['high_freq_preservation']:.4f}")
        print(f"   Low Frequencies:         {freq_metrics['freq_low']:.6f}")
        print(f"   Mid-Low Frequencies:     {freq_metrics['freq_mid_low']:.6f}")
        print(f"   Mid-High Frequencies:    {freq_metrics['freq_mid_high']:.6f}")
        print(f"   High Frequencies:        {freq_metrics['freq_high']:.6f}")

    # Compare high-frequency preservation
    print(f"\n🔊 High-Frequency Preservation Ranking:")
    hf_ranking = sorted(results.items(), key=lambda x: x[1]['high_freq_preservation'], reverse=True)
    for rank, (method, metrics) in enumerate(hf_ranking, 1):
        print(f"   {rank}. {method}: {metrics['high_freq_preservation']:.4f}")

    return results


def demonstrate_content_aware_analysis():
    """Demonstrate content-aware error analysis."""
    print("\n" + "="*60)
    print("🎨 CONTENT-AWARE ERROR ANALYSIS")
    print("="*60)

    # Create test images
    images = create_test_images(128)
    target = images["target"]

    # Initialize content-aware analyzer
    content_analyzer = ContentAwareAnalyzer(num_segments=25)

    # Analyze content regions in target
    print("\n🔍 Analyzing content regions in target image...")
    start_time = time.time()
    content_regions = content_analyzer.analyze_content_regions(target)
    duration = time.time() - start_time

    print(f"   Found {len(content_regions)} content regions ({duration*1000:.1f}ms)")

    # Display region statistics
    content_types = {}
    total_importance = 0.0
    for region in content_regions:
        content_types[region.content_type] = content_types.get(region.content_type, 0) + 1
        total_importance += region.importance_weight * region.area

    print(f"\n📊 Content Region Statistics:")
    print(f"   Total regions: {len(content_regions)}")
    for content_type, count in sorted(content_types.items()):
        print(f"   {content_type:12s}: {count:3d} regions")

    # Compute content-weighted errors
    print(f"\n⚖️  Content-Weighted Error Analysis:")
    print("-" * 50)

    results = {}
    for method_name, reconstruction in images.items():
        if method_name == "target":
            continue

        start_time = time.time()
        content_error = content_analyzer.compute_content_weighted_error(
            target, reconstruction, content_regions)
        duration = time.time() - start_time

        results[method_name] = content_error
        print(f"   {method_name:18s}: {content_error:.6f} ({duration*1000:.1f}ms)")

    # Rank by content-weighted error
    print(f"\n🏆 Best to Worst (by Content-Weighted Error):")
    sorted_methods = sorted(results.items(), key=lambda x: x[1])
    for rank, (method, error) in enumerate(sorted_methods, 1):
        print(f"   {rank}. {method} (Error: {error:.6f})")

    return results


def demonstrate_comparative_assessment():
    """Demonstrate comparative quality assessment."""
    print("\n" + "="*60)
    print("🏁 COMPARATIVE QUALITY ASSESSMENT")
    print("="*60)

    # Create test images
    images = create_test_images(128)
    target = images["target"]

    # Extract reconstructions (exclude target)
    reconstructions = {k: v for k, v in images.items() if k != "target"}

    print(f"\n🔬 Comparing {len(reconstructions)} reconstruction methods...")

    # Initialize comparative assessor
    assessor = ComparativeQualityAssessment()

    # Perform comparison
    start_time = time.time()
    comparison_results = assessor.compare_methods(target, reconstructions)
    duration = time.time() - start_time

    print(f"   Comparison completed in {duration:.2f}s")

    # Display detailed results
    print(f"\n📈 Detailed Quality Assessment:")
    print("=" * 80)

    for method_name, metrics in comparison_results.items():
        print(f"\n🔍 {method_name}:")
        print(f"   Quality Rank:        {metrics['advanced']['quality_rank']}")
        print(f"   Combined Score:      {metrics['combined_score']:.4f}")

        # Basic metrics
        basic = metrics['basic']
        print(f"   Basic Metrics:")
        print(f"     L1 Error:          {basic['l1_error']:.6f}")
        print(f"     SSIM Score:        {basic['ssim_score']:.4f}")
        print(f"     PSNR:              {basic['psnr']:.2f} dB")

        # Advanced metrics
        advanced = metrics['advanced']
        print(f"   Advanced Metrics:")
        print(f"     LPIPS Score:       {advanced['lpips_score']:.4f}")
        print(f"     Gradient Similarity: {advanced['gradient_similarity']:.4f}")
        print(f"     Edge Coherence:    {advanced['edge_coherence']:.4f}")
        print(f"     Spectral Distortion: {advanced['spectral_distortion']:.6f}")
        print(f"     High-Freq Preservation: {advanced['high_freq_preservation']:.4f}")
        print(f"     Content-Weighted Error: {advanced['content_weighted_error']:.6f}")

    # Overall ranking
    print(f"\n🏆 FINAL QUALITY RANKING:")
    print("=" * 40)
    ranked_methods = sorted(comparison_results.items(),
                          key=lambda x: x[1]['advanced']['quality_rank'])

    for rank, (method_name, metrics) in enumerate(ranked_methods, 1):
        score = metrics['combined_score']
        quality_rank = metrics['advanced']['quality_rank']
        print(f"   {rank}. {method_name:18s} (Score: {score:.4f})")

    return comparison_results


def demonstrate_advanced_error_maps():
    """Demonstrate advanced error map creation."""
    print("\n" + "="*60)
    print("🗺️  ADVANCED ERROR MAP DEMONSTRATION")
    print("="*60)

    # Create test images
    images = create_test_images(128)
    target = images["target"]

    # Initialize advanced analyzer
    analyzer = AdvancedErrorAnalyzer()

    # Select a medium-quality reconstruction for demonstration
    reconstruction = images["MediumQuality"]

    print(f"\n🔍 Creating advanced error maps for MediumQuality method...")

    # Create different types of error maps
    error_maps = {}

    # Content-weighted error map
    start_time = time.time()
    content_map = analyzer.create_advanced_error_map(target, reconstruction, 'content_weighted')
    duration = time.time() - start_time
    error_maps['content_weighted'] = content_map
    print(f"   Content-weighted map:    {duration*1000:.1f}ms (range: {np.min(content_map):.4f}-{np.max(content_map):.4f})")

    # Frequency-weighted error map
    start_time = time.time()
    freq_map = analyzer.create_advanced_error_map(target, reconstruction, 'frequency_weighted')
    duration = time.time() - start_time
    error_maps['frequency_weighted'] = freq_map
    print(f"   Frequency-weighted map:  {duration*1000:.1f}ms (range: {np.min(freq_map):.4f}-{np.max(freq_map):.4f})")

    # Standard L1 error map for comparison
    start_time = time.time()
    l1_map = analyzer.create_error_map(target, reconstruction, 'l1')
    duration = time.time() - start_time
    error_maps['l1'] = l1_map
    print(f"   Standard L1 map:         {duration*1000:.1f}ms (range: {np.min(l1_map):.4f}-{np.max(l1_map):.4f})")

    # Compare error map statistics
    print(f"\n📊 Error Map Statistics Comparison:")
    print("-" * 50)
    for map_name, error_map in error_maps.items():
        mean_error = np.mean(error_map)
        std_error = np.std(error_map)
        max_error = np.max(error_map)
        print(f"   {map_name:20s}: mean={mean_error:.6f}, std={std_error:.6f}, max={max_error:.6f}")

    return error_maps


def run_performance_benchmark():
    """Run performance benchmark of advanced metrics."""
    print("\n" + "="*60)
    print("⚡ PERFORMANCE BENCHMARK")
    print("="*60)

    sizes = [64, 128, 256]
    methods = ["HighQuality", "MediumQuality", "LowQuality"]

    print(f"\n⏱️  Benchmarking advanced metrics across different image sizes...")

    analyzer = AdvancedErrorAnalyzer()

    results = {}
    for size in sizes:
        print(f"\n📐 Testing {size}x{size} images:")
        print("-" * 30)

        # Create test images for this size
        images = create_test_images(size)
        target = images["target"]

        size_results = {}
        total_time = 0

        for method in methods:
            reconstruction = images[method]

            # Time the advanced metrics computation
            start_time = time.time()
            advanced_metrics = analyzer.compute_advanced_metrics(target, reconstruction)
            duration = time.time() - start_time

            size_results[method] = duration
            total_time += duration

            print(f"   {method:15s}: {duration*1000:6.1f}ms")

        results[size] = size_results
        print(f"   {'Total':15s}: {total_time*1000:6.1f}ms")

    # Performance scaling analysis
    print(f"\n📈 Performance Scaling Analysis:")
    print("-" * 40)
    base_size = sizes[0]
    base_time = sum(results[base_size].values()) / len(methods)

    for size in sizes:
        avg_time = sum(results[size].values()) / len(methods)
        scaling_factor = (size / base_size) ** 2  # Expected O(n²) scaling
        actual_factor = avg_time / base_time
        efficiency = scaling_factor / actual_factor if actual_factor > 0 else 0

        print(f"   {size}x{size}: {avg_time*1000:6.1f}ms (scaling: {actual_factor:.2f}x, efficiency: {efficiency:.2f})")

    return results


def main():
    """Main demonstration function."""
    parser = argparse.ArgumentParser(
        description="Advanced Error Metrics Demonstration - T4.2 Implementation"
    )
    parser.add_argument(
        "--demo",
        choices=["all", "lpips", "frequency", "content", "comparative", "maps", "performance"],
        default="all",
        help="Which demonstration to run"
    )

    args = parser.parse_args()

    print("⚡ ADVANCED ERROR METRICS SYSTEM DEMO")
    print("🎯 Adaptive Gaussian Splatting - T4.2 Implementation")
    print("=" * 60)

    start_time = time.time()

    try:
        if args.demo in ["all", "lpips"]:
            demonstrate_lpips_metrics()

        if args.demo in ["all", "frequency"]:
            demonstrate_frequency_analysis()

        if args.demo in ["all", "content"]:
            demonstrate_content_aware_analysis()

        if args.demo in ["all", "comparative"]:
            demonstrate_comparative_assessment()

        if args.demo in ["all", "maps"]:
            demonstrate_advanced_error_maps()

        if args.demo in ["all", "performance"]:
            run_performance_benchmark()

    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise

    total_duration = time.time() - start_time

    print("\n" + "="*60)
    print("✅ ADVANCED ERROR METRICS DEMO COMPLETED!")
    print("="*60)
    print(f"🎯 T4.2 implementation successfully demonstrates:")
    print(f"   • LPIPS perceptual similarity metrics")
    print(f"   • Frequency-domain error analysis with FFT")
    print(f"   • Content-aware error weighting")
    print(f"   • Region-based error aggregation")
    print(f"   • Comparative quality assessment framework")
    print(f"   • Advanced error map generation")
    print(f"   • Multi-scale and gradient-based similarity metrics")
    print(f"   • Edge coherence and texture similarity analysis")
    print(f"\n⏱️  Total demo time: {total_duration:.2f}s")
    print(f"\n📊 The advanced error metrics provide:")
    print(f"   • Perceptually-aware quality assessment")
    print(f"   • Frequency-specific reconstruction analysis")
    print(f"   • Content-importance weighted error computation")
    print(f"   • Comprehensive method comparison and ranking")
    print(f"   • State-of-the-art error visualization")


if __name__ == "__main__":
    main()