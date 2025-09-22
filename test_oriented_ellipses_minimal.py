#!/usr/bin/env python3
"""Minimal test for oriented ellipse creation."""

import sys
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.splat_this.core.adaptive_extract import AdaptiveSplatExtractor, AdaptiveSplatConfig
from src.splat_this.utils.image import load_image

def test_direct_oriented_ellipse_creation():
    """Test the _create_splats_at_positions method directly."""
    print("🔬 Testing Direct Oriented Ellipse Creation")
    print("=" * 50)

    # Load test image
    image, _ = load_image(Path("simple_original.png"))
    print(f"✅ Loaded test image: {image.shape}")

    # Create extractor
    config = AdaptiveSplatConfig()
    config.min_scale = 2.0
    config.max_scale = 20.0
    extractor = AdaptiveSplatExtractor(config=config)

    # Test direct splat creation at specific positions
    test_positions = [(100, 100), (200, 200), (300, 300), (150, 350)]
    print(f"🎯 Testing splat creation at {len(test_positions)} positions...")

    try:
        splats = extractor._create_splats_at_positions(image, test_positions, verbose=True)
        print(f"\n📊 Generated {len(splats)} splats")

        if len(splats) == 0:
            print("❌ No splats were generated!")
            return False

        # Analyze splat properties
        print("\n🔍 Splat Analysis:")
        for i, splat in enumerate(splats):
            theta_deg = np.degrees(splat.theta)
            print(f"  Splat {i+1}: pos=({splat.x:.1f}, {splat.y:.1f}), "
                  f"rx={splat.rx:.2f}, ry={splat.ry:.2f}, θ={theta_deg:.1f}°")

        # Check for oriented ellipses
        oriented_count = sum(1 for s in splats if abs(s.theta) > 0.1)
        elliptical_count = sum(1 for s in splats if abs(s.rx - s.ry) > 0.5)

        print(f"\n✨ Results:")
        print(f"  Oriented splats (|θ| > 0.1 rad): {oriented_count}/{len(splats)}")
        print(f"  Elliptical splats (|rx-ry| > 0.5): {elliptical_count}/{len(splats)}")

        # Success if we have any diversity
        if oriented_count > 0 or elliptical_count > 0:
            print("🎉 SUCCESS: Oriented ellipses are being generated!")
            return True
        else:
            print("⚠️ All splats are circular and axis-aligned")
            return False

    except Exception as e:
        print(f"❌ Error during splat creation: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_direct_oriented_ellipse_creation()
    if success:
        print("\n🎉 Test PASSED: Oriented ellipse creation is working!")
    else:
        print("\n❌ Test FAILED: Oriented ellipse creation needs work.")
        sys.exit(1)