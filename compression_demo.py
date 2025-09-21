#!/usr/bin/env python3
"""Simple compression analysis demo."""

def analyze_splat_compression():
    """Analyze theoretical compression ratios."""

    # Image dimensions
    width, height = 1920, 1080
    num_splats = 1500

    # Original bitmap
    bitmap_size = width * height * 3  # RGB24 = 6,220,800 bytes

    # Current SVG text format (estimated)
    # Each splat in SVG is roughly 100-200 characters
    svg_text_size = num_splats * 150  # 225,000 bytes

    # Ultra-compact binary format
    # Each splat: 10 bytes (position, size, rotation, color optimally packed)
    compact_binary_size = num_splats * 10 + 16  # +16 for header

    # With compression (typical 2-3x on structured data)
    compressed_size = compact_binary_size // 2.5

    print("=== Splat Compression Analysis ===")
    print()
    print(f"Image: {width}×{height} ({bitmap_size:,} bytes)")
    print(f"Splats: {num_splats}")
    print()
    print("Format Comparison:")
    print(f"  Original bitmap:     {bitmap_size:,} bytes")
    print(f"  SVG text format:     {svg_text_size:,} bytes")
    print(f"  Compact binary:      {compact_binary_size:,} bytes")
    print(f"  Compressed binary:   {compressed_size:,} bytes")
    print()
    print("Compression Ratios:")
    print(f"  SVG vs bitmap:       {bitmap_size/svg_text_size:.1f}:1")
    print(f"  Binary vs bitmap:    {bitmap_size/compact_binary_size:.1f}:1")
    print(f"  Compressed vs bitmap: {bitmap_size/compressed_size:.1f}:1")
    print(f"  Compressed vs SVG:   {svg_text_size/compressed_size:.1f}:1")
    print()

    # Size categories
    print("File Size Categories:")
    if compressed_size < 1024:
        print(f"  Compressed: {compressed_size} bytes (tiny!)")
    elif compressed_size < 10240:
        print(f"  Compressed: {compressed_size/1024:.1f} KB (very small)")
    elif compressed_size < 102400:
        print(f"  Compressed: {compressed_size/1024:.1f} KB (small)")
    else:
        print(f"  Compressed: {compressed_size/1024:.1f} KB")

    print(f"  Original:   {bitmap_size/1024/1024:.1f} MB")
    print()

    # Quality vs size analysis
    print("Quality vs Size Trade-offs:")
    for splat_count in [500, 1000, 1500, 2000, 3000]:
        size = (splat_count * 10 + 16) // 2.5
        ratio = bitmap_size / size
        print(f"  {splat_count:4d} splats: {size:5.0f} bytes ({ratio:5.0f}:1 compression)")

if __name__ == "__main__":
    analyze_splat_compression()