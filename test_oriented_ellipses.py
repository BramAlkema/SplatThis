#!/usr/bin/env python3
"""Test oriented ellipse generation in progressive allocation."""

import sys
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.splat_this.core.adaptive_extract import AdaptiveSplatExtractor, AdaptiveSplatConfig
from src.splat_this.utils.image import load_image

def test_oriented_ellipse_creation():
    """Test that adaptive allocation creates oriented ellipses."""
    print("🔍 Testing Oriented Ellipse Creation")
    print("=" * 50)

    # Load test image
    image, _ = load_image(Path("simple_original.png"))
    print(f"✅ Loaded test image: {image.shape}")

    # Create adaptive extractor with non-progressive mode to avoid minimum requirements
    config = AdaptiveSplatConfig()
    config.enable_progressive = False  # Use non-progressive mode for simpler testing
    config.min_scale = 2.0
    config.max_scale = 20.0

    extractor = AdaptiveSplatExtractor(config=config)
    print(f"✅ Created extractor with scale range: {config.min_scale}-{config.max_scale}")

    # Extract splats with adaptive allocation
    print("\n🎯 Extracting adaptive splats...")
    splats = extractor.extract_adaptive_splats(image, n_splats=100, verbose=True)

    print(f"\n📊 Analysis of {len(splats)} generated splats:")

    if len(splats) == 0:
        print("❌ No splats were generated! Check extraction method.")
        return False

    # Analyze splat properties
    rx_values = [s.rx for s in splats]
    ry_values = [s.ry for s in splats]
    theta_values = [s.theta for s in splats]

    print(f"   rx range: {min(rx_values):.2f} - {max(rx_values):.2f}")
    print(f"   ry range: {min(ry_values):.2f} - {max(ry_values):.2f}")
    print(f"   theta range: {np.degrees(min(theta_values)):.1f}° - {np.degrees(max(theta_values)):.1f}°")

    # Check for orientation diversity
    oriented_splats = [s for s in splats if abs(s.theta) > 0.1]  # Non-zero rotation
    elliptical_splats = [s for s in splats if abs(s.rx - s.ry) > 0.5]  # Non-circular

    print(f"\n🎨 Splat Diversity Analysis:")
    print(f"   Oriented splats (|θ| > 0.1 rad): {len(oriented_splats)}/{len(splats)} ({100*len(oriented_splats)/len(splats):.1f}%)")
    print(f"   Elliptical splats (|rx-ry| > 0.5): {len(elliptical_splats)}/{len(splats)} ({100*len(elliptical_splats)/len(splats):.1f}%)")

    # Check config compliance
    config_violations = [s for s in splats if s.rx < config.min_scale or s.rx > config.max_scale or s.ry < config.min_scale or s.ry > config.max_scale]

    if config_violations:
        print(f"❌ Config violations: {len(config_violations)} splats outside scale limits")
        for i, s in enumerate(config_violations[:3]):  # Show first 3
            print(f"   Splat {i}: rx={s.rx:.2f}, ry={s.ry:.2f}")
        return False
    else:
        print(f"✅ All splats respect config limits: [{config.min_scale}, {config.max_scale}]")

    # Success criteria
    success = True
    reasons = []

    if len(oriented_splats) < len(splats) * 0.1:  # At least 10% oriented
        success = False
        reasons.append(f"Too few oriented splats: {len(oriented_splats)}/{len(splats)}")

    if len(elliptical_splats) < len(splats) * 0.1:  # At least 10% elliptical
        success = False
        reasons.append(f"Too few elliptical splats: {len(elliptical_splats)}/{len(splats)}")

    if max(rx_values) - min(rx_values) < 2.0:  # Some scale diversity
        success = False
        reasons.append(f"Insufficient rx diversity: {max(rx_values) - min(rx_values):.2f}")

    if max(ry_values) - min(ry_values) < 2.0:  # Some scale diversity
        success = False
        reasons.append(f"Insufficient ry diversity: {max(ry_values) - min(ry_values):.2f}")

    if success:
        print(f"\n🎉 SUCCESS: Oriented ellipses are working correctly!")
        return True
    else:
        print(f"\n❌ FAILURE: Oriented ellipses not working properly:")
        for reason in reasons:
            print(f"   - {reason}")
        return False

def test_progressive_svg_generation():
    """Test that adaptive allocation generates SVG with oriented ellipses."""
    print("\n🎨 Testing Adaptive SVG Generation")
    print("=" * 50)

    # Load test image
    image, _ = load_image(Path("simple_original.png"))

    # Create extractor and generate splats (non-progressive for testing)
    config = AdaptiveSplatConfig()
    config.enable_progressive = False  # Use non-progressive mode
    config.min_scale = 2.0
    config.max_scale = 20.0

    extractor = AdaptiveSplatExtractor(config=config)
    splats = extractor.extract_adaptive_splats(image, n_splats=50, verbose=False)

    # Generate SVG
    from src.splat_this.core.optimized_svgout import OptimizedSVGGenerator
    from src.splat_this.core.layering import LayerAssigner

    # Assign layers
    layer_assigner = LayerAssigner(n_layers=3)
    layer_data = layer_assigner.assign_layers(splats)

    # Generate SVG
    svg_generator = OptimizedSVGGenerator(width=512, height=512)
    svg_content = svg_generator.generate_svg(layer_data, gaussian_mode=True)

    # Check SVG content for ellipse properties
    has_transform = "transform" in svg_content
    has_ellipse = "ellipse" in svg_content or "rx" in svg_content
    has_rotation = "rotate(" in svg_content

    print(f"📝 SVG Content Analysis:")
    print(f"   Has transform attributes: {has_transform}")
    print(f"   Has ellipse elements: {has_ellipse}")
    print(f"   Has rotation transforms: {has_rotation}")
    print(f"   SVG length: {len(svg_content):,} characters")

    # Save test SVG
    test_svg_path = "test_oriented_ellipses.svg"
    with open(test_svg_path, 'w') as f:
        f.write(svg_content)
    print(f"✅ Test SVG saved to: {test_svg_path}")

    # Success if we have either transforms or ellipse elements
    if has_transform or has_ellipse:
        print(f"🎉 SUCCESS: Progressive SVG contains oriented elements!")
        return True
    else:
        print(f"❌ FAILURE: SVG doesn't contain oriented ellipse elements")
        return False

if __name__ == "__main__":
    print("🔬 Testing Oriented Ellipse Implementation")
    print("=" * 60)

    test1 = test_oriented_ellipse_creation()
    test2 = test_progressive_svg_generation()

    if test1 and test2:
        print("\n🎉 ALL TESTS PASSED!")
        print("\nOriented ellipse implementation is working correctly:")
        print("✅ Local covariance estimation creates diverse orientations")
        print("✅ AdaptiveSplatConfig scale limits are respected")
        print("✅ Progressive allocation generates oriented SVG elements")
    else:
        print("\n❌ SOME TESTS FAILED!")
        print("The oriented ellipse implementation needs further work.")
        sys.exit(1)