#!/usr/bin/env python3
"""
Test the size progression through all pipeline stages.
"""

import numpy as np
import sys
from pathlib import Path
from PIL import Image

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.splat_this.core.adaptive_extract import AdaptiveSplatExtractor, AdaptiveSplatConfig

def test_size_progression():
    """Test splat sizes at each pipeline stage."""
    print("🔍 Testing size progression through pipeline stages...")

    # Load test image
    image_path = "SCR-20250921-omxs.png"
    with Image.open(image_path) as img:
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img_resized = img.resize((512, 512), Image.Resampling.LANCZOS)
        image_array = np.array(img_resized).astype(np.float32) / 255.0

    # Create extractor
    config = AdaptiveSplatConfig()
    print(f"📋 Config: min_scale={config.min_scale}, max_scale={config.max_scale}")

    extractor = AdaptiveSplatExtractor(config)

    # Monkey patch to track each stage
    original_progressive = extractor._progressive_refinement
    original_optimize = extractor._optimize_adaptive_scales
    original_initialize = extractor._initialize_splats

    def track_initialize(image, saliency_map, n_splats, verbose):
        splats = original_initialize(image, saliency_map, n_splats, verbose)
        sizes = [(s.rx, s.ry) for s in splats[:10]]  # First 10
        unique_sizes = set(sizes)
        print(f"🎯 After initialization: {len(splats)} splats, {len(unique_sizes)} unique sizes")
        print(f"   Sample sizes: {sizes[:5]}")
        return splats

    def track_progressive(image, splats, saliency_map, verbose):
        splats = original_progressive(image, splats, saliency_map, verbose)
        sizes = [(s.rx, s.ry) for s in splats[:10]]  # First 10
        unique_sizes = set(sizes)
        print(f"🔄 After progressive refinement: {len(splats)} splats, {len(unique_sizes)} unique sizes")
        print(f"   Sample sizes: {sizes[:5]}")
        return splats

    def track_optimize(image, splats, verbose):
        splats = original_optimize(image, splats, verbose)
        sizes = [(s.rx, s.ry) for s in splats[:10]]  # First 10
        unique_sizes = set(sizes)
        print(f"⚡ After optimization: {len(splats)} splats, {len(unique_sizes)} unique sizes")
        print(f"   Sample sizes: {sizes[:5]}")
        return splats

    # Apply tracking
    extractor._initialize_splats = track_initialize
    extractor._progressive_refinement = track_progressive
    extractor._optimize_adaptive_scales = track_optimize

    # Extract splats
    print("\n🚀 Running extraction with tracking...")
    splats = extractor.extract_adaptive_splats(image_array, n_splats=100, verbose=False)

    # Final analysis
    print(f"\n📊 Final Results:")
    final_sizes = [(s.rx, s.ry) for s in splats]
    unique_final = set(final_sizes)
    print(f"   Total splats: {len(splats)}")
    print(f"   Unique sizes: {len(unique_final)}")

    if len(unique_final) == 1:
        print(f"   ❌ All splats have same size: {list(unique_final)[0]}")
    else:
        print("   ✅ Variable sizes found!")
        for size in list(unique_final)[:5]:
            count = final_sizes.count(size)
            print(f"      • {size}: {count} splats")

if __name__ == "__main__":
    import os
    test_size_progression()