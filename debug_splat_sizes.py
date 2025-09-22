#!/usr/bin/env python3
"""
Debug script to investigate why all splats have the same size.
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.splat_this.core.adaptive_extract import AdaptiveSplatExtractor, AdaptiveSplatConfig

def debug_splat_sizing():
    """Debug the splat sizing mechanism."""
    print("🔍 Debugging splat sizing mechanism...")

    # Load test image
    image_path = "SCR-20250921-omxs.png"
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return

    # Load and process image
    with Image.open(image_path) as img:
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img_resized = img.resize((512, 512), Image.Resampling.LANCZOS)
        image_array = np.array(img_resized).astype(np.float32) / 255.0

    # Create extractor with debug config
    config = AdaptiveSplatConfig()
    print(f"📋 Config: min_scale={config.min_scale}, max_scale={config.max_scale}")

    extractor = AdaptiveSplatExtractor(config)

    # Let's monkey-patch the _create_adaptive_splat_from_mask method to add debug output
    original_method = extractor._create_adaptive_splat_from_mask

    def debug_create_splat(image, mask, saliency_map, scale_boost=1.0):
        """Debug version of _create_adaptive_splat_from_mask with detailed logging."""
        from src.splat_this.utils.math import safe_eigendecomposition, clamp_value

        coords = np.column_stack(np.where(mask))
        if len(coords) < 3:
            return None

        # Get region properties
        center = np.mean(coords, axis=0)

        # Calculate covariance matrix
        centered_coords = coords - center
        if len(centered_coords) > 1:
            cov_matrix = np.cov(centered_coords.T)
            eigenvals, eigenvecs = safe_eigendecomposition(cov_matrix)

            # Extract colors
            region_pixels = image[mask]
            mean_color = np.mean(region_pixels, axis=0)

            # Extract saliency
            region_saliency = np.mean(saliency_map[mask])

            print(f"🔍 Debug Region Analysis:")
            print(f"   • Eigenvalues: {eigenvals}")
            print(f"   • sqrt(eigenvals): {np.sqrt(eigenvals)}")

            # Apply normalization like in the updated method
            height, width = image.shape[:2]
            img_scale = max(height, width)
            normalized_eigenvals = eigenvals / (img_scale * 0.1)

            print(f"   • Image scale: {img_scale}")
            print(f"   • Normalized eigenvals: {normalized_eigenvals}")
            print(f"   • sqrt(normalized): {np.sqrt(normalized_eigenvals)}")
            print(f"   • scale_boost: {scale_boost}")
            print(f"   • region_saliency: {region_saliency:.4f}")

            # Adaptive scaling based on saliency and region properties
            base_scale = np.sqrt(normalized_eigenvals) * scale_boost
            saliency_scale = 1.0 + region_saliency * 2.0  # Scale 1.0-3.0 based on saliency

            print(f"   • base_scale (normalized): {base_scale}")
            print(f"   • saliency_scale: {saliency_scale:.4f}")
            print(f"   • base_scale * saliency_scale: {base_scale * saliency_scale}")

            rx_unclamped = base_scale[0] * saliency_scale
            ry_unclamped = base_scale[1] * saliency_scale

            rx = clamp_value(rx_unclamped, config.min_scale, config.max_scale)
            ry = clamp_value(ry_unclamped, config.min_scale, config.max_scale)

            print(f"   • rx: {rx_unclamped:.4f} -> clamped: {rx:.4f}")
            print(f"   • ry: {ry_unclamped:.4f} -> clamped: {ry:.4f}")

            if abs(rx - config.max_scale) < 0.01 and abs(ry - config.max_scale) < 0.01:
                print(f"   ⚠️  BOTH VALUES HIT MAX_SCALE ({config.max_scale})!")

            print("")

            # Continue with original method logic
            return original_method(image, mask, saliency_map, scale_boost)

        return None

    # Apply monkey patch
    extractor._create_adaptive_splat_from_mask = debug_create_splat

    # Extract a small number of splats for debugging
    print("🎯 Extracting 10 splats for debugging...")
    splats = extractor.extract_adaptive_splats(image_array, n_splats=10, verbose=True)

    print(f"\n📊 Final Results:")
    print(f"Generated {len(splats)} splats")

    sizes = [(splat.rx, splat.ry) for splat in splats]
    unique_sizes = set(sizes)

    print(f"Unique sizes: {len(unique_sizes)}")
    for size in unique_sizes:
        count = sizes.count(size)
        print(f"   • Size {size}: {count} splats")

    if len(unique_sizes) == 1:
        print("❌ PROBLEM: All splats have the same size!")
        print(f"   Size: {list(unique_sizes)[0]}")
        print(f"   This equals max_scale: {abs(list(unique_sizes)[0][0] - config.max_scale) < 0.01}")
    else:
        print("✅ Good: Variable splat sizes found!")

if __name__ == "__main__":
    import os
    debug_splat_sizing()