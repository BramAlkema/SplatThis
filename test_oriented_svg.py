#!/usr/bin/env python3
"""Test that oriented ellipses are properly generated in SVG output."""

import sys
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.splat_this.core.adaptive_extract import AdaptiveSplatExtractor, AdaptiveSplatConfig
from src.splat_this.utils.image import load_image
from src.splat_this.core.optimized_svgout import OptimizedSVGGenerator
from src.splat_this.core.layering import LayerAssigner

def test_oriented_ellipse_svg_generation():
    """Test that oriented ellipses are properly rendered in SVG."""
    print("🎨 Testing Oriented Ellipse SVG Generation")
    print("=" * 50)

    # Load test image
    image, _ = load_image(Path("simple_original.png"))
    print(f"✅ Loaded test image: {image.shape}")

    # Create extractor and generate some oriented splats
    config = AdaptiveSplatConfig()
    config.min_scale = 2.0
    config.max_scale = 20.0
    extractor = AdaptiveSplatExtractor(config=config)

    # Create splats at specific positions to ensure we get them
    test_positions = [(100, 100), (200, 200), (300, 300), (150, 350), (400, 100)]
    splats = extractor._create_splats_at_positions(image, test_positions, verbose=False)

    print(f"✅ Generated {len(splats)} test splats")

    # Verify we have oriented ellipses
    oriented_count = sum(1 for s in splats if abs(s.theta) > 0.1)
    elliptical_count = sum(1 for s in splats if abs(s.rx - s.ry) > 0.5)

    print(f"  Oriented splats: {oriented_count}/{len(splats)}")
    print(f"  Elliptical splats: {elliptical_count}/{len(splats)}")

    if oriented_count == 0 and elliptical_count == 0:
        print("❌ No oriented ellipses to test!")
        return False

    # Assign to layers
    layer_assigner = LayerAssigner(n_layers=3)
    layer_data = layer_assigner.assign_layers(splats)
    print(f"✅ Assigned splats to {len(layer_data)} layers")

    # Generate SVG in Gaussian mode (should preserve orientations)
    svg_generator = OptimizedSVGGenerator(width=512, height=512)
    svg_content = svg_generator.generate_svg(layer_data, gaussian_mode=True)

    print(f"✅ Generated SVG ({len(svg_content):,} characters)")

    # Analyze SVG content for orientation features
    has_transform = "transform" in svg_content
    has_ellipse = "ellipse" in svg_content or "rx" in svg_content
    has_rotation = "rotate(" in svg_content
    has_radial_gradient = "radialGradient" in svg_content

    print(f"\n🔍 SVG Content Analysis:")
    print(f"  Contains transform attributes: {has_transform}")
    print(f"  Contains ellipse elements: {has_ellipse}")
    print(f"  Contains rotation transforms: {has_rotation}")
    print(f"  Contains radial gradients: {has_radial_gradient}")

    # Save for inspection
    svg_path = "test_oriented_ellipses_output.svg"
    with open(svg_path, 'w') as f:
        f.write(svg_content)
    print(f"✅ Saved test SVG to: {svg_path}")

    # Check for specific ellipse parameters in SVG
    import re

    # Look for ellipse elements with rx != ry
    ellipse_pattern = r'<ellipse[^>]*rx="([^"]*)"[^>]*ry="([^"]*)"'
    ellipse_matches = re.findall(ellipse_pattern, svg_content)

    # Look for rotation transforms
    rotation_pattern = r'rotate\(([^)]*)\)'
    rotation_matches = re.findall(rotation_pattern, svg_content)

    print(f"\n📐 Detailed Analysis:")
    print(f"  Found {len(ellipse_matches)} ellipse elements")
    print(f"  Found {len(rotation_matches)} rotation transforms")

    # Check for non-circular ellipses
    non_circular = 0
    if ellipse_matches:
        for rx_str, ry_str in ellipse_matches[:5]:  # Show first 5
            try:
                rx, ry = float(rx_str), float(ry_str)
                if abs(rx - ry) > 0.1:
                    non_circular += 1
                print(f"    Ellipse: rx={rx:.2f}, ry={ry:.2f}")
            except ValueError:
                pass

    # Check for non-zero rotations
    non_zero_rotations = 0
    if rotation_matches:
        for rot_str in rotation_matches[:5]:  # Show first 5
            try:
                angle = float(rot_str.split()[0])  # Get first number
                if abs(angle) > 0.1:
                    non_zero_rotations += 1
                print(f"    Rotation: {angle:.1f}°")
            except (ValueError, IndexError):
                pass

    print(f"\n✨ SVG Results:")
    print(f"  Non-circular ellipses: {non_circular}")
    print(f"  Non-zero rotations: {non_zero_rotations}")

    # Success criteria: SVG contains oriented elements
    success = (has_ellipse or has_radial_gradient) and (has_rotation or non_circular > 0)

    if success:
        print(f"\n🎉 SUCCESS: SVG contains oriented ellipse elements!")
        return True
    else:
        print(f"\n❌ FAILURE: SVG doesn't properly represent oriented ellipses")
        return False

if __name__ == "__main__":
    success = test_oriented_ellipse_svg_generation()
    if success:
        print("\n🎉 OVERALL SUCCESS: Oriented ellipses work end-to-end!")
    else:
        print("\n❌ OVERALL FAILURE: SVG generation needs improvement.")
        sys.exit(1)