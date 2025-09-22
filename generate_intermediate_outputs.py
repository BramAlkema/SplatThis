#!/usr/bin/env python3
"""
Generate all intermediate visual outputs for the SplatThis pipeline.
Creates PNG for saliency and SVG outputs for each processing step.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from splat_this.core.adaptive_extract import AdaptiveSplatExtractor
from splat_this.core.optimized_svgout import OptimizedSVGGenerator

def ensure_output_dir():
    """Ensure the intermediate_outputs directory exists."""
    output_dir = "intermediate_outputs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def generate_saliency_visualization(image_path, output_dir):
    """Generate saliency analysis PNG output."""
    print("🎯 Generating saliency analysis visualization...")

    # Load image
    image = Image.open(image_path).convert('RGB')
    image_array = np.array(image)

    # Create adaptive extractor (has saliency functionality)
    extractor = AdaptiveSplatExtractor()

    # Get saliency map
    saliency_map = extractor.compute_saliency_map(image_array)

    # Find peaks manually using simple peak detection
    from scipy.signal import peak_local_maxima
    from scipy.ndimage import maximum_filter

    # Find local maxima in saliency map
    local_maxima = maximum_filter(saliency_map, size=20) == saliency_map
    threshold = np.percentile(saliency_map, 95)  # Top 5% values
    peaks_mask = local_maxima & (saliency_map > threshold)
    peaks = np.column_stack(np.where(peaks_mask))

    # Create visualization
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    # Original image
    ax1.imshow(image_array)
    ax1.set_title('Original Image', fontsize=14, fontweight='bold')
    ax1.axis('off')

    # Saliency map
    im2 = ax2.imshow(saliency_map, cmap='hot', alpha=0.8)
    ax2.set_title('Saliency Map', fontsize=14, fontweight='bold')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    # Peaks overlay
    ax3.imshow(image_array, alpha=0.7)
    ax3.imshow(saliency_map, cmap='hot', alpha=0.5)

    # Mark peaks
    if len(peaks) > 0:
        peak_coords = np.array(peaks)
        ax3.scatter(peak_coords[:, 1], peak_coords[:, 0],
                   c='lime', s=100, marker='x', linewidths=3, label=f'{len(peaks)} Peaks')
        ax3.legend()

    ax3.set_title(f'Detected Saliency Peaks ({len(peaks)})', fontsize=14, fontweight='bold')
    ax3.axis('off')

    plt.tight_layout()
    saliency_path = os.path.join(output_dir, "step2_saliency_analysis.png")
    plt.savefig(saliency_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✅ Saliency visualization saved: {saliency_path}")
    return len(peaks)

def generate_splat_generation_svg(image_path, output_dir, num_splats=200):
    """Generate initial splat generation SVG."""
    print(f"⭐ Generating initial splat generation SVG ({num_splats} splats)...")

    # Load image
    image = Image.open(image_path).convert('RGB')
    image_array = np.array(image)

    # Extract initial splats
    extractor = AdaptiveSplatExtractor()
    splats = extractor.extract_splats(image_array, num_splats=num_splats)

    # Generate SVG
    svg_generator = OptimizedSVGGenerator()
    svg_content = svg_generator.generate_svg(splats, image_array.shape[:2])

    splat_path = os.path.join(output_dir, "step3_initial_splats.svg")
    with open(splat_path, 'w') as f:
        f.write(svg_content)

    print(f"✅ Initial splats SVG saved: {splat_path} ({len(splats)} splats)")
    return splats

def generate_refinement_svgs(image_path, initial_splats, output_dir, iterations=3):
    """Generate refinement iteration SVGs."""
    print(f"🔄 Generating refinement iteration SVGs ({iterations} iterations)...")

    # Load image
    image = Image.open(image_path).convert('RGB')
    image_array = np.array(image)

    # Create refinement engine
    refiner = ProgressiveRefinementEngine()
    svg_generator = OptimizedSVGGenerator()

    # Perform refinement iterations
    current_splats = initial_splats.copy()
    refinement_paths = []

    for i in range(iterations):
        print(f"  Refinement iteration {i+1}/{iterations}...")

        # Refine splats
        current_splats = refiner.refine_splats(current_splats, image_array, iterations=1)

        # Generate SVG
        svg_content = svg_generator.generate_svg(current_splats, image_array.shape[:2])

        refinement_path = os.path.join(output_dir, f"step4_refinement_iter_{i+1}.svg")
        with open(refinement_path, 'w') as f:
            f.write(svg_content)

        refinement_paths.append(refinement_path)
        print(f"    ✅ Refinement iteration {i+1} saved: {refinement_path}")

    return current_splats, refinement_paths

def generate_scale_optimization_svgs(image_path, refined_splats, output_dir):
    """Generate scale optimization SVGs."""
    print("📊 Generating scale optimization SVGs...")

    # Load image
    image = Image.open(image_path).convert('RGB')
    image_array = np.array(image)

    svg_generator = OptimizedSVGGenerator()
    scale_paths = []

    # Different splat counts for scale optimization
    scale_targets = [100, 500, 1000]

    for target_count in scale_targets:
        print(f"  Generating scale optimization for {target_count} splats...")

        # Sample splats to target count
        if len(refined_splats) > target_count:
            # Select most important splats based on size/alpha
            splat_importance = []
            for splat in refined_splats:
                # Importance based on size and alpha
                importance = splat.sigma_x * splat.sigma_y * splat.alpha
                splat_importance.append(importance)

            # Get indices of most important splats
            important_indices = np.argsort(splat_importance)[-target_count:]
            scale_splats = [refined_splats[i] for i in important_indices]
        else:
            scale_splats = refined_splats

        # Generate SVG
        svg_content = svg_generator.generate_svg(scale_splats, image_array.shape[:2])

        scale_path = os.path.join(output_dir, f"step5_scale_optimization_{target_count}.svg")
        with open(scale_path, 'w') as f:
            f.write(svg_content)

        scale_paths.append(scale_path)
        print(f"    ✅ Scale optimization saved: {scale_path} ({len(scale_splats)} splats)")

    return scale_paths

def generate_final_svg(image_path, final_splats, output_dir):
    """Generate final optimized SVG."""
    print("🎨 Generating final optimized SVG...")

    # Load image
    image = Image.open(image_path).convert('RGB')
    image_array = np.array(image)

    # Generate final SVG
    svg_generator = OptimizedSVGGenerator()
    svg_content = svg_generator.generate_svg(final_splats, image_array.shape[:2])

    final_path = os.path.join(output_dir, "step6_final_output.svg")
    with open(final_path, 'w') as f:
        f.write(svg_content)

    print(f"✅ Final SVG saved: {final_path} ({len(final_splats)} splats)")
    return final_path

def main():
    """Generate all intermediate outputs."""
    print("🚀 Generating all intermediate visual outputs...")

    # Setup
    image_path = "SCR-20250921-omxs.png"
    if not os.path.exists(image_path):
        print(f"❌ Error: Image file not found: {image_path}")
        return

    output_dir = ensure_output_dir()

    try:
        # Step 1: Original image (just copy)
        original_path = os.path.join(output_dir, "step1_original.png")
        Image.open(image_path).save(original_path)
        print(f"✅ Original image copied: {original_path}")

        # Step 2: Saliency analysis
        num_peaks = generate_saliency_visualization(image_path, output_dir)

        # Step 3: Initial splat generation
        initial_splats = generate_splat_generation_svg(image_path, output_dir, num_splats=200)

        # Step 4: Refinement iterations
        refined_splats, refinement_paths = generate_refinement_svgs(
            image_path, initial_splats, output_dir, iterations=3
        )

        # Step 5: Scale optimization
        scale_paths = generate_scale_optimization_svgs(image_path, refined_splats, output_dir)

        # Step 6: Final output
        final_path = generate_final_svg(image_path, refined_splats, output_dir)

        print("\n🎉 All intermediate outputs generated successfully!")
        print(f"📁 Output directory: {output_dir}/")
        print(f"   - step1_original.png")
        print(f"   - step2_saliency_analysis.png")
        print(f"   - step3_initial_splats.svg")
        print(f"   - step4_refinement_iter_1.svg")
        print(f"   - step4_refinement_iter_2.svg")
        print(f"   - step4_refinement_iter_3.svg")
        print(f"   - step5_scale_optimization_100.svg")
        print(f"   - step5_scale_optimization_500.svg")
        print(f"   - step5_scale_optimization_1000.svg")
        print(f"   - step6_final_output.svg")

    except Exception as e:
        print(f"❌ Error generating outputs: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()