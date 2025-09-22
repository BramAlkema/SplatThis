#!/usr/bin/env python3
"""Real Gaussian Splats vs PNG comparison using the adaptive splatting system.

This script actually generates Gaussian splats from the image and compares them
with the original using our advanced error metrics.
"""

import sys
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.splat_this.core.adaptive_gaussian_2d import AdaptiveGaussian2D
from src.splat_this.core.gradient_computation import GradientComputer
from src.splat_this.core.tile_renderer import TileRenderer
from src.splat_this.core.content_adaptive_initialization import ContentAdaptiveInitializer
from src.splat_this.core.advanced_error_metrics import AdvancedErrorAnalyzer, compare_reconstruction_methods


def load_image(image_path: str, target_size: tuple = (256, 256)) -> np.ndarray:
    """Load and process image."""
    with Image.open(image_path) as img:
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize(target_size, Image.Resampling.LANCZOS)
        return np.array(img).astype(np.float32) / 255.0


def create_splat_reconstructions(target_image: np.ndarray) -> dict:
    """Create different Gaussian splat reconstructions with varying splat counts."""
    H, W, C = target_image.shape

    # Initialize components
    gradient_computer = GradientComputer()
    initializer = ContentAdaptiveInitializer()
    renderer = TileRenderer(tile_size=64)

    reconstructions = {}
    splat_counts = [50, 100, 200, 500, 1000]

    for splat_count in splat_counts:
        print(f"Creating reconstruction with {splat_count} splats...")

        # Initialize splats
        splats = initializer.initialize_splats(
            target_image,
            num_splats=splat_count,
            use_content_adaptive=True
        )

        # Render splats
        rendered = renderer.render(splats, (H, W))

        # Convert to RGB if RGBA
        if rendered.shape[2] == 4:
            # Alpha composite over white background
            alpha = rendered[:, :, 3:4]
            rgb = rendered[:, :, :3]
            rendered_rgb = rgb * alpha + (1 - alpha)
        else:
            rendered_rgb = rendered

        reconstructions[f"{splat_count}_splats"] = np.clip(rendered_rgb, 0, 1)

    return reconstructions


def create_side_by_side_visualization(target: np.ndarray, reconstructions: dict,
                                    metrics_results: dict) -> None:
    """Create side-by-side visualization with metrics."""

    num_methods = len(reconstructions) + 1  # +1 for original
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    # Show original
    axes[0].imshow(target)
    axes[0].set_title("🎯 Original PNG\n512×512", fontsize=12, fontweight='bold')
    axes[0].axis('off')

    # Show reconstructions
    for i, (method_name, reconstruction) in enumerate(reconstructions.items(), 1):
        if i >= len(axes):
            break

        axes[i].imshow(reconstruction)

        # Get metrics for this method
        if method_name in metrics_results:
            metrics = metrics_results[method_name]
            rank = metrics['advanced']['quality_rank']
            score = metrics['combined_score']
            lpips = metrics['advanced']['lpips_score']
            ssim = metrics['basic']['ssim_score']

            title = f"🔸 {method_name.replace('_', ' ').title()}\n"
            title += f"Rank #{rank} | Score: {score:.3f}\n"
            title += f"LPIPS: {lpips:.4f} | SSIM: {ssim:.3f}"
        else:
            title = f"🔸 {method_name.replace('_', ' ').title()}"

        axes[i].set_title(title, fontsize=10)
        axes[i].axis('off')

    # Hide unused subplots
    for i in range(len(reconstructions) + 1, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.suptitle("🎯 Real Gaussian Splats vs Original PNG - Advanced Error Metrics Comparison",
                fontsize=16, fontweight='bold', y=0.98)

    # Save the visualization
    plt.savefig("real_splats_comparison.png", dpi=150, bbox_inches='tight')
    print("💾 Saved comparison to: real_splats_comparison.png")

    plt.show()


def main():
    """Main function to demonstrate real splats vs PNG."""
    print("🎯 REAL GAUSSIAN SPLATS VS PNG COMPARISON")
    print("=" * 60)

    # Load the image
    image_path = "SCR-20250921-omxs.png"
    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        print("Please ensure the image file exists in the current directory.")
        return

    print(f"📂 Loading image: {image_path}")
    target = load_image(image_path, target_size=(256, 256))  # Smaller for faster processing
    print(f"✅ Loaded image: {target.shape}")

    # Create real splat reconstructions
    print("\n🔸 Creating Gaussian splat reconstructions...")
    try:
        reconstructions = create_splat_reconstructions(target)
        print(f"✅ Created {len(reconstructions)} splat reconstructions")
    except Exception as e:
        print(f"❌ Failed to create splat reconstructions: {e}")
        print("This might be because the full adaptive splatting system isn't implemented yet.")
        print("We have the error metrics (T4.2) but need the rendering pipeline!")
        return

    # Analyze with advanced error metrics
    print("\n📊 Analyzing with advanced error metrics...")
    try:
        metrics_results = compare_reconstruction_methods(target, reconstructions)
        print("✅ Advanced error metrics analysis complete!")

        # Print summary
        print("\n📈 Quality Ranking Summary:")
        print("-" * 40)
        sorted_methods = sorted(metrics_results.items(),
                              key=lambda x: x[1]['advanced']['quality_rank'])

        for method, metrics in sorted_methods:
            rank = metrics['advanced']['quality_rank']
            score = metrics['combined_score']
            lpips = metrics['advanced']['lpips_score']
            ssim = metrics['basic']['ssim_score']
            print(f"#{rank}. {method:15s} | Score: {score:.3f} | LPIPS: {lpips:.4f} | SSIM: {ssim:.3f}")

    except Exception as e:
        print(f"❌ Error metrics analysis failed: {e}")
        return

    # Create visualization
    print("\n🎨 Creating side-by-side visualization...")
    try:
        create_side_by_side_visualization(target, reconstructions, metrics_results)
        print("✅ Visualization complete!")
    except Exception as e:
        print(f"❌ Visualization failed: {e}")

    print("\n🎯 DEMONSTRATION COMPLETE!")
    print("✅ Successfully compared real Gaussian splats with original PNG")
    print("✅ Applied advanced error metrics (T4.2) to real splat reconstructions")
    print("✅ Generated side-by-side comparison visualization")


if __name__ == "__main__":
    main()