#!/usr/bin/env python3
"""
Inspect SVG files to check splat size variation.
"""

import re
import os
from collections import Counter

def extract_splat_sizes_from_svg(svg_file):
    """Extract all splat sizes from SVG file."""
    if not os.path.exists(svg_file):
        print(f"❌ File not found: {svg_file}")
        return []

    with open(svg_file, 'r') as f:
        content = f.read()

    # Find all ellipse elements and extract rx/ry values
    ellipse_pattern = r'<ellipse[^>]+rx="([^"]+)"[^>]+ry="([^"]+)"[^>]*>'
    matches = re.findall(ellipse_pattern, content)

    sizes = []
    for rx, ry in matches:
        try:
            sizes.append((float(rx), float(ry)))
        except ValueError:
            continue

    return sizes

def analyze_size_distribution(sizes, filename):
    """Analyze and report size distribution."""
    print(f"\n🔍 Analyzing {filename}:")
    print(f"   Total splats: {len(sizes)}")

    if not sizes:
        print("   ❌ No sizes found!")
        return

    # Count unique sizes
    unique_sizes = set(sizes)
    print(f"   Unique sizes: {len(unique_sizes)}")

    if len(unique_sizes) == 1:
        print(f"   ❌ All splats have same size: {list(unique_sizes)[0]}")
        return

    # Show size distribution
    size_counter = Counter(sizes)
    print("   📊 Size distribution:")

    # Show top 5 most common sizes
    for size, count in size_counter.most_common(5):
        percentage = (count / len(sizes)) * 100
        print(f"      • {size}: {count} splats ({percentage:.1f}%)")

    # Show size ranges
    rx_values = [rx for rx, ry in sizes]
    ry_values = [ry for rx, ry in sizes]

    print(f"   📏 Size ranges:")
    print(f"      • rx: {min(rx_values):.2f} - {max(rx_values):.2f}")
    print(f"      • ry: {min(ry_values):.2f} - {max(ry_values):.2f}")

    if len(unique_sizes) > 1:
        print("   ✅ GOOD: Variable splat sizes found!")

def main():
    """Inspect all generated SVG files."""
    print("🔍 Inspecting SVG files for splat size variation...")

    output_dir = "intermediate_outputs"
    svg_files = [
        "step3_initial_splats.svg",
        "step4_refinement.svg",
        "step5_scale_optimization.svg",
        "step6_final_output.svg"
    ]

    for svg_file in svg_files:
        filepath = os.path.join(output_dir, svg_file)
        sizes = extract_splat_sizes_from_svg(filepath)
        analyze_size_distribution(sizes, svg_file)

if __name__ == "__main__":
    main()