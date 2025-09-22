#!/usr/bin/env python3
"""Test memory-related fixes."""

import sys
from pathlib import Path
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.splat_this.utils.profiler import MemoryEfficientProcessor
from src.splat_this.core.optimized_extract import OptimizedSplatExtractor

def test_memory_processor_no_upscaling():
    """Test MemoryEfficientProcessor doesn't upscale images."""
    print("Testing MemoryEfficientProcessor upscaling prevention...")

    processor = MemoryEfficientProcessor(max_memory_mb=512)

    # Test case 1: Normal image that shouldn't be downsampled
    normal_size = (800, 600)  # width, height
    should_downsample, new_size = processor.should_downsample_image(normal_size, 1000)

    if not should_downsample:
        print(f"✅ Normal image {normal_size} not downsampled")
    else:
        print(f"❌ Normal image {normal_size} was unnecessarily downsampled to {new_size}")
        return False

    # Test case 2: Skinny image that could be upscaled in width but shouldn't be
    skinny_size = (200, 600)  # narrow but tall
    should_downsample, new_size = processor.should_downsample_image(skinny_size, 1000)

    if should_downsample:
        # Check that width wasn't upscaled
        if new_size[0] <= skinny_size[0]:
            print(f"✅ Skinny image width not upscaled: {skinny_size[0]} → {new_size[0]}")
        else:
            print(f"❌ Skinny image width was upscaled: {skinny_size[0]} → {new_size[0]}")
            return False

        # Check that height wasn't upscaled
        if new_size[1] <= skinny_size[1]:
            print(f"✅ Skinny image height not upscaled: {skinny_size[1]} → {new_size[1]}")
        else:
            print(f"❌ Skinny image height was upscaled: {skinny_size[1]} → {new_size[1]}")
            return False
    else:
        print(f"✅ Skinny image {skinny_size} not downsampled (acceptable)")

    # Test case 3: Very small image that should never be upscaled
    tiny_size = (50, 50)
    should_downsample, new_size = processor.should_downsample_image(tiny_size, 100)

    if should_downsample:
        # Should never upscale
        if new_size[0] <= tiny_size[0] and new_size[1] <= tiny_size[1]:
            print(f"✅ Tiny image not upscaled: {tiny_size} → {new_size}")
        else:
            print(f"❌ Tiny image was upscaled: {tiny_size} → {new_size}")
            return False
    else:
        print(f"✅ Tiny image {tiny_size} not downsampled")

    return True

def test_max_memory_propagation():
    """Test that max_memory setting is properly propagated."""
    print("\nTesting max_memory propagation...")

    # Test OptimizedSplatExtractor accepts max_memory_mb
    try:
        extractor = OptimizedSplatExtractor(max_memory_mb=512)
        print("✅ OptimizedSplatExtractor accepts max_memory_mb parameter")

        # Check that it creates a memory processor with the right limit
        if hasattr(extractor, 'memory_processor'):
            if extractor.memory_processor.max_memory_mb == 512:
                print("✅ OptimizedSplatExtractor memory processor has correct limit")
            else:
                print(f"❌ Memory processor limit wrong: expected 512, got {extractor.memory_processor.max_memory_mb}")
                return False
        else:
            print("❌ OptimizedSplatExtractor doesn't have memory_processor attribute")
            return False

    except Exception as e:
        print(f"❌ OptimizedSplatExtractor failed with max_memory_mb: {e}")
        return False

    # Test that None still works
    try:
        extractor_none = OptimizedSplatExtractor(max_memory_mb=None)
        print("✅ OptimizedSplatExtractor accepts max_memory_mb=None")

        # Should still have a memory processor with default limit
        if hasattr(extractor_none, 'memory_processor'):
            print("✅ OptimizedSplatExtractor creates default memory processor when max_memory_mb=None")
        else:
            print("❌ OptimizedSplatExtractor doesn't create memory processor when max_memory_mb=None")
            return False

    except Exception as e:
        print(f"❌ OptimizedSplatExtractor failed with max_memory_mb=None: {e}")
        return False

    return True

def test_memory_enforcement_order():
    """Test memory enforcement logic."""
    print("\nTesting memory enforcement order...")

    # Create a processor with low memory limit to trigger downsampling
    processor = MemoryEfficientProcessor(max_memory_mb=100)  # Very low limit

    # Large image that should trigger downsampling
    large_size = (4000, 3000)  # 4K x 3K
    should_downsample, new_size = processor.should_downsample_image(large_size, 5000)

    if should_downsample:
        print(f"✅ Large image {large_size} triggered downsampling to {new_size}")

        # Check that the new size is smaller
        if new_size[0] < large_size[0] and new_size[1] < large_size[1]:
            print(f"✅ Downsampled size is smaller in both dimensions")
        else:
            print(f"❌ Downsampled size is not smaller: {large_size} → {new_size}")
            return False
    else:
        print(f"❌ Large image {large_size} should have triggered downsampling")
        return False

    return True

if __name__ == "__main__":
    print("🔧 Testing Memory-Related Fixes")
    print("=" * 40)

    test1 = test_memory_processor_no_upscaling()
    test2 = test_max_memory_propagation()
    test3 = test_memory_enforcement_order()

    if all([test1, test2, test3]):
        print("\n🎉 ALL MEMORY TESTS PASSED!")
        print("\nSummary of fixes:")
        print("1. ✅ MemoryEfficientProcessor prevents upscaling during downsampling")
        print("2. ✅ max_memory parameter is propagated to OptimizedSplatExtractor")
        print("3. ✅ Memory enforcement considers downsampling appropriately")
    else:
        print("\n❌ SOME MEMORY TESTS FAILED!")
        sys.exit(1)