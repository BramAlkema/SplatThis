#!/usr/bin/env python3
"""Test LayerAssigner n_layers=1 fix."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.splat_this.core.layering import LayerAssigner
from src.splat_this.core.extract import Gaussian
import numpy as np

def test_single_layer():
    """Test that LayerAssigner works with n_layers=1."""
    print("Testing LayerAssigner with n_layers=1...")

    # Create test splat
    splat = Gaussian(
        x=100.0, y=100.0, rx=10.0, ry=10.0, theta=0.0,
        r=255, g=0, b=0, a=0.5
    )
    splat.score = 0.5

    # Test constructor validation
    try:
        assigner = LayerAssigner(n_layers=0)
        print("❌ FAILED: Should reject n_layers=0")
        return False
    except ValueError as e:
        print(f"✅ Constructor validation works: {e}")

    # Test n_layers=1 (the problematic case)
    try:
        assigner = LayerAssigner(n_layers=1)
        layers = assigner.assign_layers([splat])

        # Should have exactly one layer with one splat
        if len(layers) == 1 and 0 in layers and len(layers[0]) == 1:
            depth = layers[0][0].depth
            print(f"✅ n_layers=1 works! Depth assigned: {depth}")

            # Depth should be 0.6 (middle value)
            if abs(depth - 0.6) < 1e-6:
                print(f"✅ Correct depth value for single layer: {depth}")
                return True
            else:
                print(f"❌ Wrong depth value, expected 0.6, got {depth}")
                return False
        else:
            print(f"❌ Wrong layer structure: {layers}")
            return False

    except Exception as e:
        print(f"❌ FAILED with n_layers=1: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_multiple_layers():
    """Test that multi-layer still works."""
    print("\nTesting LayerAssigner with n_layers=3...")

    # Create test splats with different scores
    splats = []
    for i, score in enumerate([0.1, 0.5, 0.9]):
        splat = Gaussian(
            x=100.0 + i * 50, y=100.0, rx=10.0, ry=10.0, theta=0.0,
            r=255, g=0, b=0, a=0.5
        )
        splat.score = score
        splats.append(splat)

    try:
        assigner = LayerAssigner(n_layers=3)
        layers = assigner.assign_layers(splats)

        print(f"✅ n_layers=3 works! Created {len(layers)} layers")

        # Check depth values
        expected_depths = [0.2, 0.6, 1.0]  # For 3 layers
        for layer_idx in layers:
            if layers[layer_idx]:  # If layer has splats
                actual_depth = layers[layer_idx][0].depth
                expected_depth = expected_depths[layer_idx]
                if abs(actual_depth - expected_depth) < 1e-6:
                    print(f"✅ Layer {layer_idx} has correct depth: {actual_depth}")
                else:
                    print(f"❌ Layer {layer_idx} wrong depth: expected {expected_depth}, got {actual_depth}")
                    return False

        return True

    except Exception as e:
        print(f"❌ FAILED with n_layers=3: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔧 Testing LayerAssigner division-by-zero fixes")
    print("=" * 50)

    success1 = test_single_layer()
    success2 = test_multiple_layers()

    if success1 and success2:
        print("\n🎉 ALL TESTS PASSED! LayerAssigner is fixed.")
    else:
        print("\n❌ SOME TESTS FAILED!")
        sys.exit(1)