#!/usr/bin/env python3
"""
Progressive Refinement System Demo

Demonstrates the capabilities of the Progressive Refinement System (T3.3)
for adaptive Gaussian splatting optimization.

This demo showcases:
- Error map computation and analysis
- High-error region identification
- Progressive splat refinement operations
- Integration with SGD optimization
- Convergence monitoring and validation

The progressive refinement system iteratively improves Gaussian splat
representations by:
1. Computing reconstruction error maps
2. Identifying high-error regions (>80th percentile)
3. Selecting splats for refinement based on error overlap
4. Applying targeted refinement operations
5. Optimizing with SGD for fine-tuning
6. Validating convergence criteria

Usage:
    python demo_progressive_refinement.py [--preset PRESET] [--verbose]

    PRESET options: conservative, balanced, aggressive, experimental
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import time
from typing import List, Tuple, Dict, Any
import logging

from src.splat_this.core.progressive_refinement import (
    ProgressiveRefiner,
    RefinementConfig,
    RefinementStrategy,
    RefinementOperation,
    create_refinement_config_preset,
    refine_splats_progressively
)
from src.splat_this.core.adaptive_gaussian import AdaptiveGaussian2D, create_isotropic_gaussian
from src.splat_this.core.sgd_optimizer import SGDConfig


def render_gaussians_to_image(splats: List[AdaptiveGaussian2D], width: int, height: int) -> np.ndarray:
    """Simple renderer for Gaussian splats to create demo images."""
    image = np.zeros((height, width, 3))

    # Create coordinate grids
    x = np.linspace(0, 1, width)
    y = np.linspace(0, 1, height)
    X, Y = np.meshgrid(x, y)

    for splat in splats:
        # Get splat parameters
        mu_x, mu_y = splat.mu[0], splat.mu[1]
        # Convert inverse scales to scales (sigma = 1/inv_s)
        sigma_x, sigma_y = 1.0/splat.inv_s[0], 1.0/splat.inv_s[1]
        color = splat.color
        alpha = splat.alpha

        # Create Gaussian distribution
        dx = X - mu_x
        dy = Y - mu_y

        # Simple isotropic Gaussian for demo (ignoring rotation for simplicity)
        gaussian = np.exp(-(dx**2 / (2*sigma_x**2) + dy**2 / (2*sigma_y**2)))

        # Alpha blend the Gaussian into the image
        for c in range(3):
            image[:, :, c] += alpha * color[c] * gaussian

    # Clip to valid range
    return np.clip(image, 0, 1)


def setup_logging(verbose: bool = False):
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def create_synthetic_target_image(width: int = 256, height: int = 256) -> np.ndarray:
    """Create a synthetic target image with complex patterns for testing."""
    print("📸 Creating synthetic target image with complex patterns...")

    # Create coordinate grids
    x = np.linspace(0, 1, width)
    y = np.linspace(0, 1, height)
    X, Y = np.meshgrid(x, y)

    # Initialize the image
    image = np.zeros((height, width, 3))

    # Add multiple overlapping Gaussian-like patterns
    patterns = [
        # Central bright region
        {"center": (0.5, 0.5), "scale": 0.15, "color": [0.9, 0.3, 0.1], "intensity": 0.8},
        # Upper left pattern
        {"center": (0.25, 0.25), "scale": 0.1, "color": [0.1, 0.7, 0.9], "intensity": 0.6},
        # Lower right pattern
        {"center": (0.75, 0.75), "scale": 0.08, "color": [0.3, 0.9, 0.2], "intensity": 0.7},
        # Background texture
        {"center": (0.6, 0.3), "scale": 0.2, "color": [0.8, 0.8, 0.1], "intensity": 0.4},
        # Sharp edge feature
        {"center": (0.8, 0.2), "scale": 0.05, "color": [0.9, 0.1, 0.8], "intensity": 0.9},
    ]

    for pattern in patterns:
        cx, cy = pattern["center"]
        scale = pattern["scale"]
        color = np.array(pattern["color"])
        intensity = pattern["intensity"]

        # Create Gaussian-like pattern
        dist_sq = ((X - cx)**2 + (Y - cy)**2) / (2 * scale**2)
        gaussian = np.exp(-dist_sq)

        # Add to each color channel
        for c in range(3):
            image[:, :, c] += intensity * color[c] * gaussian

    # Add some noise for realism
    noise = np.random.normal(0, 0.02, image.shape)
    image = np.clip(image + noise, 0, 1)

    print(f"✅ Created {width}x{height} target image with 5 complex patterns")
    return image


def create_initial_splat_approximation(target_image: np.ndarray, num_splats: int = 12) -> List[AdaptiveGaussian2D]:
    """Create an initial rough approximation with fewer, larger splats."""
    print(f"🎯 Creating initial approximation with {num_splats} splats...")

    height, width = target_image.shape[:2]
    splats = []

    # Create a grid-based initial placement with some randomization
    grid_size = int(np.sqrt(num_splats)) + 1
    for i in range(num_splats):
        # Grid position with some randomization
        grid_x = (i % grid_size) / (grid_size - 1) if grid_size > 1 else 0.5
        grid_y = (i // grid_size) / (grid_size - 1) if grid_size > 1 else 0.5

        # Add some random offset
        x = np.clip(grid_x + np.random.normal(0, 0.1), 0.1, 0.9)
        y = np.clip(grid_y + np.random.normal(0, 0.1), 0.1, 0.9)

        # Sample color from target image at this location
        img_x = int(x * (width - 1))
        img_y = int(y * (height - 1))
        color = target_image[img_y, img_x, :]

        # Create splat with initial parameters
        splat = create_isotropic_gaussian(
            center=np.array([x, y]),
            scale=0.08,  # Larger initial scale for rough approximation
            color=color + np.random.normal(0, 0.1, 3),  # Add some color variation
            alpha=0.7 + np.random.uniform(-0.2, 0.2)
        )
        splats.append(splat)

    print(f"✅ Created {len(splats)} initial splats with grid-based placement")
    return splats


def demonstrate_refinement_preset(preset_name: str, target_image: np.ndarray,
                                initial_splats: List[AdaptiveGaussian2D]) -> Dict[str, Any]:
    """Demonstrate refinement using a specific preset configuration."""
    print(f"\n🔧 Testing {preset_name.upper()} refinement preset...")

    # Create refinement configuration
    config = create_refinement_config_preset(preset_name)

    print(f"📋 Configuration:")
    print(f"   Error percentile threshold: {config.error_percentile_threshold}")
    print(f"   Max refinement iterations: {config.max_refinement_iterations}")
    if config.sgd_config:
        print(f"   SGD iterations per round: {config.sgd_config.max_iterations}")
    else:
        print(f"   SGD iterations per round: Not configured")
    print(f"   Enabled operations: {[op.name for op in config.enabled_operations]}")

    # Render initial image for refinement
    initial_rendered = render_gaussians_to_image(
        initial_splats,
        target_image.shape[1],
        target_image.shape[0]
    )

    # Perform refinement
    start_time = time.time()
    result = refine_splats_progressively(
        splats=initial_splats.copy(),
        target_image=target_image,
        rendered_image=initial_rendered,
        config=config
    )
    end_time = time.time()

    print(f"⏱️  Refinement completed in {end_time - start_time:.2f} seconds")
    print(f"🔄 Iterations performed: {result.iterations}")
    print(f"📉 Final error: {result.final_error:.6f}")
    print(f"🎯 Total operations: {result.total_operations}")
    print(f"✅ Converged: {result.converged}")

    # Render final result
    final_rendered = render_gaussians_to_image(
        result.refined_splats,
        target_image.shape[1],  # width
        target_image.shape[0]   # height
    )

    return {
        'preset': preset_name,
        'config': config,
        'result': result,
        'rendered_image': final_rendered,
        'execution_time': end_time - start_time
    }


def analyze_refinement_quality(target_image: np.ndarray, rendered_image: np.ndarray) -> Dict[str, float]:
    """Analyze the quality of refinement using multiple metrics."""
    # Mean Squared Error
    mse = np.mean((target_image - rendered_image) ** 2)

    # Peak Signal-to-Noise Ratio
    psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else float('inf')

    # Structural Similarity (simplified version)
    # Use correlation coefficient as a simple similarity measure
    target_flat = target_image.flatten()
    rendered_flat = rendered_image.flatten()
    correlation = np.corrcoef(target_flat, rendered_flat)[0, 1] if len(target_flat) > 1 else 1.0

    # Compute error map statistics
    error_map = np.mean((target_image - rendered_image) ** 2, axis=2)
    error_std = np.std(error_map)
    error_max = np.max(error_map)
    error_percentile_90 = np.percentile(error_map, 90)

    return {
        'mse': mse,
        'psnr': psnr,
        'correlation': correlation,
        'error_std': error_std,
        'error_max': error_max,
        'error_90th_percentile': error_percentile_90
    }


def create_comparison_visualization(target_image: np.ndarray,
                                  refinement_results: List[Dict[str, Any]]) -> plt.Figure:
    """Create a comprehensive visualization comparing refinement results."""
    num_presets = len(refinement_results)
    fig, axes = plt.subplots(3, num_presets + 1, figsize=(4 * (num_presets + 1), 12))

    if num_presets == 0:
        return fig

    # Ensure axes is 2D
    if axes.ndim == 1:
        axes = axes.reshape(3, -1)

    # Show target image in first column
    axes[0, 0].imshow(target_image)
    axes[0, 0].set_title('Target Image', fontweight='bold')
    axes[0, 0].axis('off')

    axes[1, 0].text(0.5, 0.5, 'Target\nImage', ha='center', va='center',
                   transform=axes[1, 0].transAxes, fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')

    axes[2, 0].text(0.5, 0.5, 'Reference', ha='center', va='center',
                   transform=axes[2, 0].transAxes, fontsize=14, fontweight='bold')
    axes[2, 0].axis('off')

    # Show results for each preset
    for i, result_data in enumerate(refinement_results):
        col = i + 1
        preset = result_data['preset']
        rendered = result_data['rendered_image']
        result = result_data['result']
        exec_time = result_data['execution_time']

        # Rendered image
        axes[0, col].imshow(rendered)
        axes[0, col].set_title(f'{preset.title()} Result', fontweight='bold')
        axes[0, col].axis('off')

        # Error map
        error_map = np.mean((target_image - rendered) ** 2, axis=2)
        im = axes[1, col].imshow(error_map, cmap='hot', vmin=0, vmax=np.max(error_map))
        axes[1, col].set_title(f'{preset.title()} Error Map')
        axes[1, col].axis('off')
        plt.colorbar(im, ax=axes[1, col], fraction=0.046, pad=0.04)

        # Quality metrics
        quality = analyze_refinement_quality(target_image, rendered)
        metrics_text = (
            f"Loss: {result.final_loss:.4f}\n"
            f"PSNR: {quality['psnr']:.2f} dB\n"
            f"Correlation: {quality['correlation']:.3f}\n"
            f"Iterations: {result.total_iterations}\n"
            f"Time: {exec_time:.2f}s\n"
            f"Converged: {'✅' if result.converged else '❌'}"
        )

        axes[2, col].text(0.05, 0.95, metrics_text, transform=axes[2, col].transAxes,
                         verticalalignment='top', fontfamily='monospace', fontsize=10)
        axes[2, col].set_title(f'{preset.title()} Metrics')
        axes[2, col].axis('off')

    plt.tight_layout()
    return fig


def compare_refinement_strategies():
    """Compare different refinement strategies and their effectiveness."""
    print("\n🆚 PROGRESSIVE REFINEMENT STRATEGY COMPARISON")
    print("=" * 60)

    # Create test scenario
    target_image = create_synthetic_target_image(128, 128)  # Smaller for faster demo
    initial_splats = create_initial_splat_approximation(target_image, num_splats=8)

    # Render initial approximation
    initial_rendered = render_gaussians_to_image(
        initial_splats,
        target_image.shape[1],
        target_image.shape[0]
    )

    initial_quality = analyze_refinement_quality(target_image, initial_rendered)
    print(f"\n📊 Initial approximation quality:")
    print(f"   MSE: {initial_quality['mse']:.6f}")
    print(f"   PSNR: {initial_quality['psnr']:.2f} dB")
    print(f"   Correlation: {initial_quality['correlation']:.3f}")

    # Test different presets
    presets = ['conservative', 'balanced', 'aggressive']
    results = []

    for preset in presets:
        try:
            result_data = demonstrate_refinement_preset(preset, target_image, initial_splats)
            results.append(result_data)
        except Exception as e:
            print(f"❌ Error testing {preset} preset: {e}")

    if results:
        # Create and display visualization
        print("\n🎨 Creating comparison visualization...")
        fig = create_comparison_visualization(target_image, results)

        # Save the visualization
        output_path = 'progressive_refinement_comparison.png'
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"💾 Visualization saved to: {output_path}")

        # Print summary comparison
        print(f"\n📈 REFINEMENT COMPARISON SUMMARY")
        print("-" * 50)

        for result_data in results:
            preset = result_data['preset']
            result = result_data['result']
            exec_time = result_data['execution_time']
            quality = analyze_refinement_quality(target_image, result_data['rendered_image'])

            improvement = ((initial_quality['mse'] - quality['mse']) / initial_quality['mse']) * 100

            print(f"{preset.upper():>12}: "
                  f"Loss={result.final_loss:.4f}, "
                  f"PSNR={quality['psnr']:6.2f}dB, "
                  f"Improvement={improvement:5.1f}%, "
                  f"Time={exec_time:5.2f}s")

    return results


def demonstrate_error_analysis():
    """Demonstrate error analysis and high-error region identification."""
    print("\n🔍 ERROR ANALYSIS DEMONSTRATION")
    print("=" * 40)

    # Create test scenario
    target_image = create_synthetic_target_image(64, 64)  # Small for clear visualization
    initial_splats = create_initial_splat_approximation(target_image, num_splats=4)

    # Create refiner
    config = create_refinement_config_preset('balanced')
    refiner = ProgressiveRefiner(config)

    # Render initial approximation
    rendered_image = render_gaussians_to_image(
        initial_splats,
        target_image.shape[1],
        target_image.shape[0]
    )

    print("🔬 Computing error analysis...")

    # Compute error map
    error_map = refiner._compute_error_map(target_image, rendered_image)

    # Identify high-error regions
    high_error_regions = refiner._identify_high_error_regions(error_map, target_image, rendered_image)

    print(f"📊 Error Analysis Results:")
    print(f"   Mean error: {np.mean(error_map):.6f}")
    print(f"   Max error: {np.max(error_map):.6f}")
    print(f"   Error std: {np.std(error_map):.6f}")
    print(f"   High-error regions found: {len(high_error_regions)}")

    for i, region in enumerate(high_error_regions):
        center_y, center_x = region.center
        print(f"   Region {i+1}: center=({center_x:.2f}, {center_y:.2f}), "
              f"area={region.area}, avg_error={region.error_magnitude:.6f}")

    # Select splats for refinement
    splats_to_refine = refiner._select_splats_for_refinement(initial_splats, high_error_regions, error_map)

    print(f"🎯 Splats selected for refinement: {len(splats_to_refine)}")

    return {
        'target_image': target_image,
        'rendered_image': rendered_image,
        'error_map': error_map,
        'high_error_regions': high_error_regions,
        'splats_to_refine': splats_to_refine
    }


def main():
    """Main demonstration function."""
    parser = argparse.ArgumentParser(description='Progressive Refinement System Demo')
    parser.add_argument('--preset', choices=['conservative', 'balanced', 'aggressive', 'experimental'],
                       default='balanced', help='Refinement preset to use')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--compare', action='store_true', help='Compare all refinement presets')
    parser.add_argument('--error-analysis', action='store_true', help='Demonstrate error analysis')

    args = parser.parse_args()

    setup_logging(args.verbose)

    print("🎨 PROGRESSIVE REFINEMENT SYSTEM DEMO")
    print("🎯 Adaptive Gaussian Splatting - T3.3 Implementation")
    print("=" * 60)

    try:
        if args.error_analysis:
            demonstrate_error_analysis()

        if args.compare:
            compare_refinement_strategies()
        else:
            # Single preset demonstration
            print(f"\n🔧 Demonstrating {args.preset.upper()} refinement preset")

            target_image = create_synthetic_target_image()
            initial_splats = create_initial_splat_approximation(target_image)

            result_data = demonstrate_refinement_preset(args.preset, target_image, initial_splats)

            # Create simple visualization
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            axes[0].imshow(target_image)
            axes[0].set_title('Target Image')
            axes[0].axis('off')

            axes[1].imshow(result_data['rendered_image'])
            axes[1].set_title(f'Refined Result ({args.preset})')
            axes[1].axis('off')

            error_map = np.mean((target_image - result_data['rendered_image']) ** 2, axis=2)
            im = axes[2].imshow(error_map, cmap='hot')
            axes[2].set_title('Error Map')
            axes[2].axis('off')
            plt.colorbar(im, ax=axes[2])

            plt.tight_layout()
            output_path = f'progressive_refinement_{args.preset}.png'
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"💾 Result saved to: {output_path}")

        print("\n✅ Progressive Refinement Demo completed successfully!")
        print("🎯 The system demonstrates adaptive optimization with:")
        print("   • Error-driven refinement selection")
        print("   • Multiple refinement strategies")
        print("   • Integrated SGD optimization")
        print("   • Convergence monitoring")
        print("   • Quality validation")

    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    exit(main())