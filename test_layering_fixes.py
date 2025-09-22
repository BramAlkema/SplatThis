#!/usr/bin/env python3
"""Test LayerAssigner single layer fix."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.splat_this.core.layering import LayerAssigner
from src.splat_this.core.extract import Gaussian

def test_layer_assigner_single_layer():
    """Test LayerAssigner works with n_layers=1."""
    print("Testing LayerAssigner with n_layers=1...")

    # Create test splats
    test_splats = [
        Gaussian(x=10, y=10, rx=5, ry=5, theta=0, r=255, g=0, b=0, a=0.8, score=0.9, depth=0.5),
        Gaussian(x=20, y=20, rx=3, ry=3, theta=0, r=0, g=255, b=0, a=0.7, score=0.8, depth=0.5),
        Gaussian(x=30, y=30, rx=4, ry=4, theta=0, r=0, g=0, b=255, a=0.6, score=0.7, depth=0.5),
    ]

    try:
        # This should not crash
        assigner = LayerAssigner(n_layers=1)
        layer_data = assigner.assign_layers(test_splats)

        print(f"✅ LayerAssigner(n_layers=1) created successfully")
        print(f"✅ assign_layers completed with {len(layer_data)} layers")

        # Check layer data
        if len(layer_data) == 1:
            print(f"✅ Correct number of layers: {len(layer_data)}")
        else:
            print(f"❌ Wrong number of layers: expected 1, got {len(layer_data)}")
            return False

        # Check all splats are in layer 0
        layer_0_splats = layer_data.get(0, [])
        if len(layer_0_splats) == len(test_splats):
            print(f"✅ All {len(test_splats)} splats assigned to layer 0")
        else:
            print(f"❌ Wrong splat count in layer 0: expected {len(test_splats)}, got {len(layer_0_splats)}")
            return False

        # Check depth values
        depths = [splat.depth for splat in layer_0_splats]
        if all(d == 0.6 for d in depths):
            print(f"✅ All splats have correct depth value: 0.6")
        else:
            print(f"❌ Incorrect depth values: {depths}")
            return False

        return True

    except Exception as e:
        print(f"❌ LayerAssigner(n_layers=1) failed: {e}")
        return False

def test_layer_assigner_multiple_layers():
    """Test LayerAssigner still works with multiple layers."""
    print("\nTesting LayerAssigner with n_layers=3...")

    # Create test splats with different scores
    test_splats = [
        Gaussian(x=10, y=10, rx=5, ry=5, theta=0, r=255, g=0, b=0, a=0.8, score=0.9, depth=0.5),
        Gaussian(x=20, y=20, rx=3, ry=3, theta=0, r=0, g=255, b=0, a=0.7, score=0.5, depth=0.5),
        Gaussian(x=30, y=30, rx=4, ry=4, theta=0, r=0, g=0, b=255, a=0.6, score=0.1, depth=0.5),
    ]

    try:
        assigner = LayerAssigner(n_layers=3)
        layer_data = assigner.assign_layers(test_splats)

        print(f"✅ LayerAssigner(n_layers=3) created successfully")
        print(f"✅ assign_layers completed with {len(layer_data)} layers")

        # Check we have 3 layers
        if len(layer_data) == 3:
            print(f"✅ Correct number of layers: {len(layer_data)}")
        else:
            print(f"❌ Wrong number of layers: expected 3, got {len(layer_data)}")
            return False

        # Check depth values are different
        all_depths = []
        for layer_idx, layer_splats in layer_data.items():
            for splat in layer_splats:
                all_depths.append(splat.depth)

        unique_depths = set(all_depths)
        if len(unique_depths) > 1:
            print(f"✅ Multiple unique depth values: {sorted(unique_depths)}")
        else:
            print(f"❌ Only one depth value found: {unique_depths}")
            return False

        return True

    except Exception as e:
        print(f"❌ LayerAssigner(n_layers=3) failed: {e}")
        return False

def test_layer_assigner_invalid_input():
    """Test LayerAssigner rejects invalid input."""
    print("\nTesting LayerAssigner with invalid input...")

    try:
        LayerAssigner(n_layers=0)
        print("❌ LayerAssigner should have rejected n_layers=0")
        return False
    except ValueError as e:
        if "n_layers must be at least 1" in str(e):
            print("✅ LayerAssigner correctly rejected n_layers=0")
        else:
            print(f"❌ Unexpected error message: {e}")
            return False
    except Exception as e:
        print(f"❌ Unexpected exception type: {e}")
        return False

    try:
        LayerAssigner(n_layers=-1)
        print("❌ LayerAssigner should have rejected n_layers=-1")
        return False
    except ValueError as e:
        if "n_layers must be at least 1" in str(e):
            print("✅ LayerAssigner correctly rejected n_layers=-1")
        else:
            print(f"❌ Unexpected error message: {e}")
            return False
    except Exception as e:
        print(f"❌ Unexpected exception type: {e}")
        return False

    return True

if __name__ == "__main__":
    print("🔧 Testing LayerAssigner Fixes")
    print("=" * 40)

    test1 = test_layer_assigner_single_layer()
    test2 = test_layer_assigner_multiple_layers()
    test3 = test_layer_assigner_invalid_input()

    if all([test1, test2, test3]):
        print("\n🎉 ALL LAYERING TESTS PASSED!")
        print("\nSummary of fixes:")
        print("1. ✅ LayerAssigner works with n_layers=1 (no division by zero)")
        print("2. ✅ LayerAssigner still works with multiple layers")
        print("3. ✅ LayerAssigner rejects invalid n_layers values")
    else:
        print("\n❌ SOME LAYERING TESTS FAILED!")
        sys.exit(1)