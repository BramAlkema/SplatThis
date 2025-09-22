#!/usr/bin/env python3
"""
Anisotropic Refinement System Demo

Demonstrates the capabilities of the Anisotropic Refinement System (T4.1)
for edge-aware adaptive Gaussian splatting optimization.

This demo showcases:
- Edge-aware anisotropy enhancement
- Dynamic aspect ratio optimization
- Orientation fine-tuning for edge alignment
- Structure tensor analysis and quality metrics
- Integration with existing progressive refinement

The anisotropic refinement system improves splat representations by:
1. Analyzing image structure using gradient and structure tensors
2. Enhancing aspect ratios based on edge strength and coherence
3. Aligning splat orientations with local edge directions
4. Sharpening edges through targeted scale adjustments
5. Maintaining quality through validation and constraints

Usage:
    python demo_anisotropic_refinement.py [--preset PRESET] [--verbose]

    PRESET options: conservative, balanced, aggressive, experimental
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import time
from typing import List, Tuple, Dict, Any
import logging

from src.splat_this.core.anisotropic_refinement import (
    AnisotropicRefiner,
    AnisotropicConfig,
    AnisotropyStrategy,
    AnisotropyOperation,
    create_anisotropic_config_preset,
    refine_splats_anisotropically
)
from src.splat_this.core.adaptive_gaussian import AdaptiveGaussian2D, create_isotropic_gaussian
from src.splat_this.core.progressive_refinement import create_refinement_config_preset


def setup_logging(verbose: bool = False):
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


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
        # Convert inverse scales to scales
        sigma_x, sigma_y = 1.0/splat.inv_s[0], 1.0/splat.inv_s[1]
        color = splat.color
        alpha = splat.alpha
        theta = splat.theta

        # Create Gaussian distribution with rotation
        dx = X - mu_x
        dy = Y - mu_y

        # Apply rotation
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        dx_rot = cos_theta * dx + sin_theta * dy
        dy_rot = -sin_theta * dx + cos_theta * dy

        # Anisotropic Gaussian
        gaussian = np.exp(-(dx_rot**2 / (2*sigma_x**2) + dy_rot**2 / (2*sigma_y**2)))

        # Alpha blend the Gaussian into the image
        for c in range(3):
            image[:, :, c] += alpha * color[c] * gaussian

    # Clip to valid range
    return np.clip(image, 0, 1)


def create_edge_rich_target_image(width: int = 256, height: int = 256) -> np.ndarray:
    """Create a target image with strong edge features for anisotropic refinement testing."""
    print("📸 Creating edge-rich target image for anisotropic refinement...")

    # Initialize the image
    image = np.zeros((height, width, 3))

    # Create coordinate grids
    x = np.linspace(0, 1, width)
    y = np.linspace(0, 1, height)
    X, Y = np.meshgrid(x, y)

    # 1. Sharp diagonal line
    diagonal_mask = np.abs(Y - X) < 0.02
    image[diagonal_mask] = [0.9, 0.1, 0.1]

    # 2. Vertical edge
    vertical_mask = np.abs(X - 0.3) < 0.01
    image[vertical_mask] = [0.1, 0.9, 0.1]

    # 3. Horizontal edge
    horizontal_mask = np.abs(Y - 0.7) < 0.01
    image[horizontal_mask] = [0.1, 0.1, 0.9]

    # 4. Curved edge (circle)
    center_x, center_y = 0.7, 0.3
    radius = 0.15
    circle_dist = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    circle_mask = np.abs(circle_dist - radius) < 0.01
    image[circle_mask] = [0.9, 0.9, 0.1]

    # 5. Corner feature
    corner_mask = ((X > 0.8) & (Y > 0.8)) | ((X < 0.2) & (Y < 0.2))
    image[corner_mask] = [0.8, 0.3, 0.8]

    # Add some soft background texture
    noise = np.random.normal(0, 0.05, image.shape)
    image = np.clip(image + noise, 0, 1)

    print(f"✅ Created {width}x{height} edge-rich target image with various edge orientations")
    return image


def create_initial_isotropic_splats(target_image: np.ndarray, num_splats: int = 15) -> List[AdaptiveGaussian2D]:
    """Create initial isotropic splats that will benefit from anisotropic refinement."""
    print(f"🎯 Creating {num_splats} initial isotropic splats...")

    height, width = target_image.shape[:2]
    splats = []

    # Place splats strategically near edges and features
    positions = [
        # Near diagonal line
        (0.4, 0.4), (0.6, 0.6),
        # Near vertical edge
        (0.3, 0.2), (0.3, 0.5), (0.3, 0.8),
        # Near horizontal edge
        (0.2, 0.7), (0.5, 0.7), (0.8, 0.7),
        # Near circle
        (0.7, 0.3), (0.85, 0.3), (0.7, 0.45),
        # Random positions
        (0.1, 0.1), (0.9, 0.9), (0.5, 0.3), (0.2, 0.9)
    ]

    for i, (pos_x, pos_y) in enumerate(positions[:num_splats]):
        # Sample color from target image at this location
        img_x = int(pos_x * (width - 1))
        img_y = int(pos_y * (height - 1))
        color = target_image[img_y, img_x, :]

        # Create isotropic splat
        splat = create_isotropic_gaussian(
            center=np.array([pos_x, pos_y]),
            scale=0.06 + np.random.uniform(-0.02, 0.02),
            color=color + np.random.normal(0, 0.1, 3),
            alpha=0.7 + np.random.uniform(-0.1, 0.2)
        )
        splats.append(splat)

    print(f"✅ Created {len(splats)} initial isotropic splats")
    return splats


def analyze_anisotropic_improvements(original_splats: List[AdaptiveGaussian2D],
                                   refined_splats: List[AdaptiveGaussian2D]) -> Dict[str, Any]:
    """Analyze the improvements from anisotropic refinement."""
    improvements = {
        'aspect_ratio_changes': [],
        'orientation_changes': [],
        'refined_count': 0,
        'avg_aspect_ratio_before': 0.0,
        'avg_aspect_ratio_after': 0.0,
        'max_aspect_ratio_before': 0.0,
        'max_aspect_ratio_after': 0.0
    }

    for orig, refined in zip(original_splats, refined_splats):
        # Compute aspect ratios
        orig_scales = 1.0 / orig.inv_s
        refined_scales = 1.0 / refined.inv_s

        orig_aspect = max(orig_scales) / min(orig_scales)
        refined_aspect = max(refined_scales) / min(refined_scales)

        improvements['aspect_ratio_changes'].append(refined_aspect - orig_aspect)
        improvements['orientation_changes'].append(abs(refined.theta - orig.theta))

        if refined.refinement_count > orig.refinement_count:
            improvements['refined_count'] += 1

    # Compute statistics
    all_orig_aspects = [max(1.0/s.inv_s) / min(1.0/s.inv_s) for s in original_splats]
    all_refined_aspects = [max(1.0/s.inv_s) / min(1.0/s.inv_s) for s in refined_splats]

    improvements['avg_aspect_ratio_before'] = np.mean(all_orig_aspects)
    improvements['avg_aspect_ratio_after'] = np.mean(all_refined_aspects)
    improvements['max_aspect_ratio_before'] = np.max(all_orig_aspects)
    improvements['max_aspect_ratio_after'] = np.max(all_refined_aspects)

    return improvements


def demonstrate_anisotropic_preset(preset_name: str, target_image: np.ndarray,
                                  initial_splats: List[AdaptiveGaussian2D]) -> Dict[str, Any]:
    """Demonstrate anisotropic refinement using a specific preset configuration."""
    print(f"\n🔧 Testing {preset_name.upper()} anisotropic refinement preset...")

    # Create anisotropic configuration
    config = create_anisotropic_config_preset(preset_name)

    print(f"📋 Anisotropic Configuration:")
    print(f"   Strategy: {config.strategy.value}")
    print(f"   Max aspect ratio: {config.max_aspect_ratio}")
    print(f"   Learning rate: {config.learning_rate}")
    print(f"   Max iterations: {config.max_refinement_iterations}")
    print(f"   Edge threshold: {config.edge_strength_threshold}")
    print(f"   Operations: {[op.name for op in config.enabled_operations]}")

    # Perform anisotropic refinement
    start_time = time.time()
    result = refine_splats_anisotropically(
        splats=initial_splats.copy(),
        target_image=target_image,
        config=config
    )
    end_time = time.time()

    print(f"⏱️  Anisotropic refinement completed in {end_time - start_time:.2f} seconds")
    print(f"🔄 Iterations performed: {result.iterations}")
    print(f"📊 Final quality: {result.final_quality:.6f}")
    print(f"📈 Quality improvement: {result.quality_improvement:.6f}")
    print(f"✅ Converged: {result.converged}")

    # Analyze improvements
    improvements = analyze_anisotropic_improvements(initial_splats, result.refined_splats)

    print(f"🎯 Refinement Analysis:")
    print(f"   Splats refined: {improvements['refined_count']}/{len(initial_splats)}")
    print(f"   Avg aspect ratio: {improvements['avg_aspect_ratio_before']:.2f} → {improvements['avg_aspect_ratio_after']:.2f}")
    print(f"   Max aspect ratio: {improvements['max_aspect_ratio_before']:.2f} → {improvements['max_aspect_ratio_after']:.2f}")
    print(f"   Avg orientation change: {np.mean(improvements['orientation_changes']):.3f} rad")

    # Render final result
    final_rendered = render_gaussians_to_image(
        result.refined_splats,
        target_image.shape[1],
        target_image.shape[0]
    )

    return {
        'preset': preset_name,
        'config': config,
        'result': result,
        'rendered_image': final_rendered,
        'improvements': improvements,
        'execution_time': end_time - start_time
    }


def visualize_anisotropic_analysis(analysis, target_image: np.ndarray) -> plt.Figure:
    """Create visualization of anisotropic structure analysis."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Original image
    axes[0, 0].imshow(target_image)
    axes[0, 0].set_title('Target Image')
    axes[0, 0].axis('off')

    # Edge strength
    im1 = axes[0, 1].imshow(analysis.edge_strength, cmap='hot')
    axes[0, 1].set_title('Edge Strength')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    # Coherence
    im2 = axes[0, 2].imshow(analysis.coherence, cmap='viridis')
    axes[0, 2].set_title('Structure Coherence')
    axes[0, 2].axis('off')
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)

    # Orientation
    im3 = axes[1, 0].imshow(analysis.orientation, cmap='hsv')
    axes[1, 0].set_title('Edge Orientation')
    axes[1, 0].axis('off')
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)

    # Gradient magnitude
    im4 = axes[1, 1].imshow(analysis.gradient_magnitude, cmap='plasma')
    axes[1, 1].set_title('Gradient Magnitude')
    axes[1, 1].axis('off')
    plt.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)

    # Quality map
    im5 = axes[1, 2].imshow(analysis.quality_map, cmap='RdYlGn')
    axes[1, 2].set_title('Quality Map')
    axes[1, 2].axis('off')
    plt.colorbar(im5, ax=axes[1, 2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    return fig


def create_anisotropic_comparison_visualization(target_image: np.ndarray,
                                              refinement_results: List[Dict[str, Any]]) -> plt.Figure:
    """Create comprehensive visualization comparing anisotropic refinement results."""
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
        improvements = result_data['improvements']
        exec_time = result_data['execution_time']

        # Rendered image
        axes[0, col].imshow(rendered)
        axes[0, col].set_title(f'{preset.title()} Result', fontweight='bold')
        axes[0, col].axis('off')

        # Structure analysis visualization (quality map)
        quality_map = result.anisotropic_analysis.quality_map
        im = axes[1, col].imshow(quality_map, cmap='RdYlGn', vmin=0, vmax=1)
        axes[1, col].set_title(f'{preset.title()} Quality Map')
        axes[1, col].axis('off')
        plt.colorbar(im, ax=axes[1, col], fraction=0.046, pad=0.04)

        # Anisotropic metrics
        metrics_text = (
            f"Quality: {result.final_quality:.4f}\n"
            f"Improvement: {result.quality_improvement:.4f}\n"
            f"Iterations: {result.iterations}\n"
            f"Refined: {improvements['refined_count']}/{len(result.refined_splats)}\n"
            f"Avg AR: {improvements['avg_aspect_ratio_after']:.2f}\n"
            f"Max AR: {improvements['max_aspect_ratio_after']:.2f}\n"
            f"Time: {exec_time:.2f}s\n"
            f"Converged: {'✅' if result.converged else '❌'}"
        )

        axes[2, col].text(0.05, 0.95, metrics_text, transform=axes[2, col].transAxes,
                         verticalalignment='top', fontfamily='monospace', fontsize=10)
        axes[2, col].set_title(f'{preset.title()} Metrics')
        axes[2, col].axis('off')

    plt.tight_layout()
    return fig


def compare_anisotropic_strategies():
    """Compare different anisotropic refinement strategies."""
    print("\n🆚 ANISOTROPIC REFINEMENT STRATEGY COMPARISON")
    print("=" * 60)

    # Create test scenario with strong edges
    target_image = create_edge_rich_target_image(128, 128)
    initial_splats = create_initial_isotropic_splats(target_image, num_splats=10)

    # Render initial approximation for comparison
    initial_rendered = render_gaussians_to_image(
        initial_splats,
        target_image.shape[1],
        target_image.shape[0]
    )

    # Analyze initial state
    initial_aspects = [max(1.0/s.inv_s) / min(1.0/s.inv_s) for s in initial_splats]
    print(f"\n📊 Initial splat statistics:")
    print(f"   Average aspect ratio: {np.mean(initial_aspects):.2f}")
    print(f"   Max aspect ratio: {np.max(initial_aspects):.2f}")
    print(f"   All splats isotropic: {all(ar < 1.1 for ar in initial_aspects)}")

    # Test different presets
    presets = ['conservative', 'balanced', 'aggressive']
    results = []

    for preset in presets:
        try:
            result_data = demonstrate_anisotropic_preset(preset, target_image, initial_splats)
            results.append(result_data)
        except Exception as e:
            print(f"❌ Error testing {preset} preset: {e}")

    if results:
        # Create and display visualization
        print("\n🎨 Creating anisotropic comparison visualization...")
        fig = create_anisotropic_comparison_visualization(target_image, results)

        # Save the visualization
        output_path = 'anisotropic_refinement_comparison.png'
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"💾 Visualization saved to: {output_path}")

        # Print summary comparison
        print(f"\n📈 ANISOTROPIC REFINEMENT COMPARISON SUMMARY")
        print("-" * 50)

        for result_data in results:
            preset = result_data['preset']
            result = result_data['result']
            improvements = result_data['improvements']
            exec_time = result_data['execution_time']

            print(f"{preset.upper():>12}: "
                  f"Quality={result.final_quality:.4f}, "
                  f"AR_avg={improvements['avg_aspect_ratio_after']:5.2f}, "
                  f"AR_max={improvements['max_aspect_ratio_after']:5.2f}, "
                  f"Refined={improvements['refined_count']:2d}/{len(result.refined_splats)}, "
                  f"Time={exec_time:5.2f}s")

    return results


def demonstrate_structure_analysis():
    """Demonstrate structure tensor analysis for anisotropic refinement."""
    print("\n🔬 STRUCTURE TENSOR ANALYSIS DEMONSTRATION")
    print("=" * 50)

    # Create test image with varied edge structures
    target_image = create_edge_rich_target_image(64, 64)

    # Create anisotropic refiner
    config = create_anisotropic_config_preset('balanced')
    refiner = AnisotropicRefiner(config)

    print("🔍 Analyzing image structure for anisotropic refinement...")

    # Perform structure analysis
    analysis = refiner.analyze_anisotropic_structure(target_image)

    print(f"📊 Structure Analysis Results:")
    print(f"   Edge strength - mean: {np.mean(analysis.edge_strength):.4f}, max: {np.max(analysis.edge_strength):.4f}")
    print(f"   Coherence - mean: {np.mean(analysis.coherence):.4f}, max: {np.max(analysis.coherence):.4f}")
    print(f"   Gradient magnitude - mean: {np.mean(analysis.gradient_magnitude):.4f}, max: {np.max(analysis.gradient_magnitude):.4f}")
    print(f"   Quality map - mean: {np.mean(analysis.quality_map):.4f}, max: {np.max(analysis.quality_map):.4f}")

    # Identify high-quality regions
    high_quality_mask = analysis.quality_map > 0.5
    high_quality_fraction = np.mean(high_quality_mask)
    print(f"   High-quality regions: {high_quality_fraction:.1%} of image")

    # Create visualization
    print("🎨 Creating structure analysis visualization...")
    fig = visualize_anisotropic_analysis(analysis, target_image)

    output_path = 'anisotropic_structure_analysis.png'
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"💾 Structure analysis saved to: {output_path}")

    return analysis


def main():
    """Main demonstration function."""
    parser = argparse.ArgumentParser(description='Anisotropic Refinement System Demo')
    parser.add_argument('--preset', choices=['conservative', 'balanced', 'aggressive', 'experimental'],
                       default='balanced', help='Anisotropic refinement preset to use')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--compare', action='store_true', help='Compare all anisotropic presets')
    parser.add_argument('--structure-analysis', action='store_true', help='Demonstrate structure analysis')

    args = parser.parse_args()

    setup_logging(args.verbose)

    print("🔧 ANISOTROPIC REFINEMENT SYSTEM DEMO")
    print("🎯 Adaptive Gaussian Splatting - T4.1 Implementation")
    print("=" * 60)

    try:
        if args.structure_analysis:
            demonstrate_structure_analysis()

        if args.compare:
            compare_anisotropic_strategies()
        else:
            # Single preset demonstration
            print(f"\n🔧 Demonstrating {args.preset.upper()} anisotropic refinement preset")

            target_image = create_edge_rich_target_image()
            initial_splats = create_initial_isotropic_splats(target_image)

            result_data = demonstrate_anisotropic_preset(args.preset, target_image, initial_splats)

            # Create simple visualization
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))

            axes[0, 0].imshow(target_image)
            axes[0, 0].set_title('Target Image')
            axes[0, 0].axis('off')

            initial_rendered = render_gaussians_to_image(initial_splats, target_image.shape[1], target_image.shape[0])
            axes[0, 1].imshow(initial_rendered)
            axes[0, 1].set_title('Initial (Isotropic)')
            axes[0, 1].axis('off')

            axes[0, 2].imshow(result_data['rendered_image'])
            axes[0, 2].set_title(f'Refined ({args.preset})')
            axes[0, 2].axis('off')

            # Structure analysis
            analysis = result_data['result'].anisotropic_analysis
            im1 = axes[1, 0].imshow(analysis.edge_strength, cmap='hot')
            axes[1, 0].set_title('Edge Strength')
            axes[1, 0].axis('off')
            plt.colorbar(im1, ax=axes[1, 0])

            im2 = axes[1, 1].imshow(analysis.coherence, cmap='viridis')
            axes[1, 1].set_title('Coherence')
            axes[1, 1].axis('off')
            plt.colorbar(im2, ax=axes[1, 1])

            im3 = axes[1, 2].imshow(analysis.quality_map, cmap='RdYlGn')
            axes[1, 2].set_title('Quality Map')
            axes[1, 2].axis('off')
            plt.colorbar(im3, ax=axes[1, 2])

            plt.tight_layout()
            output_path = f'anisotropic_refinement_{args.preset}.png'
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"💾 Result saved to: {output_path}")

        print("\n✅ Anisotropic Refinement Demo completed successfully!")
        print("🎯 The system demonstrates edge-aware optimization with:")
        print("   • Structure tensor analysis for edge detection")
        print("   • Dynamic aspect ratio enhancement")
        print("   • Orientation alignment with local edges")
        print("   • Quality-driven refinement validation")
        print("   • Multiple refinement strategies and presets")

    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    exit(main())