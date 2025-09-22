#!/usr/bin/env python3
"""
Generate intermediate visual outputs using the working PNG-to-SVG pipeline.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.splat_this.core.adaptive_extract import AdaptiveSplatExtractor, SaliencyAnalyzer, AdaptiveSplatConfig
from src.splat_this.core.optimized_svgout import OptimizedSVGGenerator

def ensure_output_dir():
    """Ensure the intermediate_outputs directory exists."""
    output_dir = "intermediate_outputs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def load_image(image_path: str, target_size: tuple = (512, 512)) -> tuple:
    """Load and process image, return both original and normalized."""
    with Image.open(image_path) as img:
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img_resized = img.resize(target_size, Image.Resampling.LANCZOS)

        # Return both the PIL image and normalized array
        image_array = np.array(img_resized)
        normalized_array = image_array.astype(np.float32) / 255.0

        return img_resized, image_array, normalized_array

def generate_saliency_png(image_path, output_dir):
    """Generate saliency analysis PNG output."""
    print("🎯 Generating saliency analysis PNG...")

    # Load image (use unnormalized for visualization)
    _, image_array, normalized_array = load_image(image_path, target_size=(512, 512))

    # Create saliency analyzer
    config = AdaptiveSplatConfig()
    saliency_analyzer = SaliencyAnalyzer(config)

    # Get saliency map (use normalized image)
    saliency_map = saliency_analyzer.compute_saliency_map(normalized_array)

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Original image
    ax1.imshow(image_array)
    ax1.set_title('Original Image', fontsize=14, fontweight='bold')
    ax1.axis('off')

    # Saliency map overlay
    ax2.imshow(image_array, alpha=0.6)
    im = ax2.imshow(saliency_map, cmap='hot', alpha=0.7)
    ax2.set_title('Saliency Map', fontsize=14, fontweight='bold')
    ax2.axis('off')
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

    plt.tight_layout()
    saliency_path = os.path.join(output_dir, "step2_saliency_analysis.png")
    plt.savefig(saliency_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✅ Saliency PNG saved: {saliency_path}")
    return saliency_path

def generate_splat_svg(image_path, output_dir, num_splats, filename):
    """Generate SVG with specified number of splats using working pipeline."""
    print(f"⭐ Generating {filename} with {num_splats} splats...")

    # Load image (use normalized for processing)
    _, image_array, normalized_array = load_image(image_path, target_size=(512, 512))

    # Extract splats (use normalized image like the working script)
    extractor = AdaptiveSplatExtractor()
    splats = extractor.extract_adaptive_splats(normalized_array, n_splats=num_splats, verbose=False)

    # Generate SVG
    height, width = image_array.shape[:2]
    svg_generator = OptimizedSVGGenerator(width, height)

    # Format splats as layers (like in working script)
    layers = {0: splats}
    svg_content = svg_generator.generate_svg(
        layers,
        gaussian_mode=True,
        title=f"{num_splats} Gaussian Splats"
    )

    # Save SVG
    svg_path = os.path.join(output_dir, filename)
    with open(svg_path, 'w') as f:
        f.write(svg_content)

    print(f"✅ SVG saved: {svg_path} ({len(splats)} splats)")
    return svg_path, splats

def main():
    """Generate all intermediate outputs."""
    print("🚀 Generating intermediate visual outputs...")

    # Setup
    image_path = "SCR-20250921-omxs.png"
    if not os.path.exists(image_path):
        print(f"❌ Error: Image file not found: {image_path}")
        return

    output_dir = ensure_output_dir()

    try:
        # Step 1: Original image (copy)
        original_path = os.path.join(output_dir, "step1_original.png")
        Image.open(image_path).save(original_path)
        print(f"✅ Original image copied: {original_path}")

        # Step 2: Saliency analysis (PNG)
        saliency_path = generate_saliency_png(image_path, output_dir)

        # Step 3: Initial splat generation (SVG - high starting point)
        splat_path_2000, splats_2000 = generate_splat_svg(
            image_path, output_dir, 2000, "step3_initial_splats.svg"
        )

        # Step 4: Refinement (SVG - enhanced splats)
        splat_path_4000, splats_4000 = generate_splat_svg(
            image_path, output_dir, 4000, "step4_refinement.svg"
        )

        # Step 5: Scale optimization (SVG - high density)
        splat_path_6000, splats_6000 = generate_splat_svg(
            image_path, output_dir, 6000, "step5_scale_optimization.svg"
        )

        # Step 6: Final output (SVG - maximum density)
        final_path, final_splats = generate_splat_svg(
            image_path, output_dir, 10000, "step6_final_output.svg"
        )

        print("\n🎉 All intermediate outputs generated successfully!")
        print(f"📁 Output directory: {output_dir}/")
        print(f"   - step1_original.png")
        print(f"   - step2_saliency_analysis.png")
        print(f"   - step3_initial_splats.svg ({len(splats_2000)} splats)")
        print(f"   - step4_refinement.svg ({len(splats_4000)} splats)")
        print(f"   - step5_scale_optimization.svg ({len(splats_6000)} splats)")
        print(f"   - step6_final_output.svg ({len(final_splats)} splats)")

        return {
            'original': original_path,
            'saliency': saliency_path,
            'initial': splat_path_2000,
            'refinement': splat_path_4000,
            'optimization': splat_path_6000,
            'final': final_path
        }

    except Exception as e:
        print(f"❌ Error generating outputs: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()