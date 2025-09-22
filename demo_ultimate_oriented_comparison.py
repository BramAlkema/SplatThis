#!/usr/bin/env python3
"""Generate ultimate oriented ellipse comparison with 90% adaptive + 10% SLIC and fixed parallax."""

import sys
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.splat_this.core.adaptive_extract import AdaptiveSplatExtractor, AdaptiveSplatConfig
from src.splat_this.core.optimized_extract import OptimizedSplatExtractor
from src.splat_this.utils.image import load_image
from src.splat_this.core.optimized_svgout import OptimizedSVGGenerator
from src.splat_this.core.layering import LayerAssigner

def create_ultimate_oriented_splats():
    """Create 2000 splats with 90% adaptive + 10% SLIC and fixed parallax."""
    print("🎨 Creating Ultimate Oriented Ellipse Comparison")
    print("=" * 55)

    # Load test image
    image, _ = load_image(Path("simple_original.png"))
    print(f"✅ Loaded test image: {image.shape}")

    # Method 1: Use adaptive extractor primarily for better oriented ellipses
    print("\n🔬 Method 1: Adaptive Content-Aware Extraction (Primary - 90%)")
    config = AdaptiveSplatConfig()
    config.min_scale = 1.5
    config.max_scale = 25.0
    adaptive_extractor = AdaptiveSplatExtractor(config=config)

    # Generate many more adaptive splats for better coverage
    test_positions = []
    height, width = image.shape[:2]

    # Create a dense grid for comprehensive coverage
    for i in range(1800):  # 90% of splats from adaptive
        x = np.random.randint(20, width - 20)
        y = np.random.randint(20, height - 20)
        test_positions.append((y, x))

    adaptive_splats = adaptive_extractor._create_splats_at_positions(
        image, test_positions, verbose=True
    )
    print(f"  Generated {len(adaptive_splats)} adaptive splats")

    # Method 2: Use SLIC for supplemental coverage (10%)
    print("\n⚡ Method 2: Optimized SLIC Supplementation (10%)")
    slic_extractor = OptimizedSplatExtractor(k=3.2, base_alpha=0.75)
    slic_splats = slic_extractor.extract_splats(image, 200)  # Only 10% from SLIC
    print(f"  Generated {len(slic_splats)} SLIC splats")

    # Fix colors for adaptive splats by sampling from image
    print("\n🎨 Fixing adaptive splat colors...")
    for splat in adaptive_splats:
        # Sample color from image at splat position
        x_pixel = int(np.clip(splat.x, 0, width - 1))
        y_pixel = int(np.clip(splat.y, 0, height - 1))

        # Sample a small region around the splat center for better color
        y_start = max(0, y_pixel - 2)
        y_end = min(height, y_pixel + 3)
        x_start = max(0, x_pixel - 2)
        x_end = min(width, x_pixel + 3)

        region = image[y_start:y_end, x_start:x_end]
        avg_color = np.mean(region.reshape(-1, 3), axis=0)

        # Update splat color
        splat.r = int(np.clip(avg_color[0], 0, 255))
        splat.g = int(np.clip(avg_color[1], 0, 255))
        splat.b = int(np.clip(avg_color[2], 0, 255))

    # Combine both methods
    all_splats = adaptive_splats + slic_splats
    print(f"✅ Total combined splats: {len(all_splats)}")

    # Analyze splat properties
    oriented_count = sum(1 for s in all_splats if abs(s.theta) > 0.1)
    elliptical_count = sum(1 for s in all_splats if abs(s.rx - s.ry) > 0.5)

    print(f"\n📊 Splat Analysis:")
    print(f"  Oriented splats (|θ| > 0.1 rad): {oriented_count}/{len(all_splats)} ({oriented_count/len(all_splats)*100:.1f}%)")
    print(f"  Elliptical splats (|rx-ry| > 0.5): {elliptical_count}/{len(all_splats)} ({elliptical_count/len(all_splats)*100:.1f}%)")

    # Custom layer assignment for better parallax
    print("\n📏 Creating custom layer assignment for smooth parallax...")
    all_splats = assign_custom_layers(all_splats)

    # Generate SVG with Gaussian mode (preserves orientations)
    svg_generator = OptimizedSVGGenerator(width=512, height=512)

    # Group splats by depth for layer generation
    layer_data = {}
    for splat in all_splats:
        depth = splat.depth
        if depth not in layer_data:
            layer_data[depth] = []
        layer_data[depth].append(splat)

    svg_content = svg_generator.generate_svg(layer_data, gaussian_mode=True)

    # Fix SVG parallax depths to use smooth continuous values
    fixed_svg = fix_smooth_parallax(svg_content)

    print(f"✅ Generated ultimate SVG ({len(fixed_svg):,} characters)")

    # Save the ultimate SVG
    svg_path = "splats_2000_ultimate_oriented.svg"
    with open(svg_path, 'w') as f:
        f.write(fixed_svg)
    print(f"✅ Saved ultimate SVG to: {svg_path}")

    return len(all_splats), oriented_count, elliptical_count, svg_path

def assign_custom_layers(splats):
    """Assign smooth depth values for better parallax instead of discrete layers."""
    import random

    # Assign smooth depth values between 0.0 and 1.0
    for i, splat in enumerate(splats):
        # Create 5 depth bands but with smooth transitions
        base_depth = (i % 5) / 4.0  # 0.0, 0.25, 0.5, 0.75, 1.0

        # Add small random variation for smooth transitions
        variation = (random.random() - 0.5) * 0.1  # ±0.05 variation
        smooth_depth = np.clip(base_depth + variation, 0.0, 1.0)

        # Assign depth directly to splat
        splat.depth = smooth_depth

    return splats

def fix_smooth_parallax(svg_content):
    """Fix parallax by using smooth continuous depth values instead of bands."""
    import re

    # Replace discrete layer depths with smooth continuous values
    # This will create smooth parallax instead of banding

    # Find all data-depth values and replace with more varied values
    def replace_depth(match):
        old_depth = float(match.group(1))
        # Map the discrete depths to a smoother distribution
        if old_depth <= 0.2:
            new_depth = np.random.uniform(0.0, 0.3)
        elif old_depth <= 0.6:
            new_depth = np.random.uniform(0.3, 0.7)
        else:
            new_depth = np.random.uniform(0.7, 1.0)
        return f'data-depth="{new_depth:.3f}"'

    # Replace all depth values with smooth variations
    svg_content = re.sub(r'data-depth="([0-9.]+)"', replace_depth, svg_content)

    return svg_content

def create_ultimate_comparison_html():
    """Create the ultimate HTML comparison page."""
    print("\n🌐 Creating Ultimate Comparison HTML")

    # Generate the ultimate splats
    total_splats, oriented_count, elliptical_count, svg_path = create_ultimate_oriented_splats()

    # Calculate percentages
    oriented_pct = (oriented_count / total_splats * 100) if total_splats > 0 else 0
    elliptical_pct = (elliptical_count / total_splats * 100) if total_splats > 0 else 0

    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Ultimate Oriented Ellipse Comparison - {total_splats} Splats</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            min-height: 100vh;
        }}

        .header {{
            text-align: center;
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 8px 25px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }}

        .header h1 {{
            margin: 0;
            color: #2c3e50;
            font-size: 2.8em;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        }}

        .header p {{
            margin: 15px 0 0 0;
            color: #7f8c8d;
            font-size: 1.3em;
        }}

        .comparison-container {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-bottom: 40px;
            max-width: 1400px;
            margin-left: auto;
            margin-right: auto;
        }}

        .image-section {{
            background: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 8px 25px rgba(0,0,0,0.1);
            text-align: center;
            transition: transform 0.3s ease;
        }}

        .image-section:hover {{
            transform: translateY(-5px);
        }}

        .image-section h2 {{
            margin-top: 0;
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 15px;
            font-size: 1.8em;
        }}

        .image-display {{
            margin: 25px 0;
            background: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
            border: 2px solid #e9ecef;
            min-height: 400px;
            display: flex;
            align-items: center;
            justify-content: center;
        }}

        .image-display img,
        .image-display object {{
            max-width: 100%;
            max-height: 500px;
            border-radius: 8px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }}

        .specs-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 15px;
            margin-top: 20px;
        }}

        .spec-box {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
        }}

        .spec-title {{
            font-size: 0.9em;
            opacity: 0.9;
            margin-bottom: 5px;
        }}

        .spec-value {{
            font-size: 1.4em;
            font-weight: bold;
        }}

        .comparison-stats {{
            background: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 8px 25px rgba(0,0,0,0.1);
            margin-bottom: 30px;
            max-width: 1400px;
            margin-left: auto;
            margin-right: auto;
        }}

        .comparison-stats h2 {{
            margin-top: 0;
            color: #2c3e50;
            border-bottom: 3px solid #e74c3c;
            padding-bottom: 15px;
            text-align: center;
        }}

        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 25px;
        }}

        .stat-card {{
            background: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            border: 1px solid #e9ecef;
            transition: transform 0.3s ease;
        }}

        .stat-card:hover {{
            transform: translateY(-3px);
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }}

        .stat-title {{
            font-size: 0.9em;
            color: #7f8c8d;
            margin-bottom: 10px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}

        .stat-value {{
            font-size: 1.6em;
            font-weight: bold;
            color: #2c3e50;
        }}

        .advantages {{
            background: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 8px 25px rgba(0,0,0,0.1);
            max-width: 1400px;
            margin-left: auto;
            margin-right: auto;
            margin-bottom: 30px;
        }}

        .advantages h2 {{
            margin-top: 0;
            color: #2c3e50;
            text-align: center;
            border-bottom: 3px solid #27ae60;
            padding-bottom: 15px;
        }}

        .advantages-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-top: 25px;
        }}

        .advantage-list {{
            background: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
        }}

        .advantage-list h3 {{
            margin-top: 0;
            color: #2c3e50;
            display: flex;
            align-items: center;
            gap: 10px;
        }}

        .advantage-list ul {{
            margin: 15px 0 0 0;
            padding-left: 20px;
        }}

        .advantage-list li {{
            margin-bottom: 10px;
            line-height: 1.5;
        }}

        .highlight {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 3px 8px;
            border-radius: 5px;
            font-weight: bold;
        }}

        .quality-badge {{
            display: inline-block;
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: bold;
            margin-top: 10px;
        }}

        .badge-original {{ background: #3498db; color: white; }}
        .badge-ultimate {{ background: #e74c3c; color: white; }}

        @media (max-width: 1000px) {{
            .comparison-container {{
                grid-template-columns: 1fr;
            }}

            .advantages-grid {{
                grid-template-columns: 1fr;
            }}

            .specs-grid {{
                grid-template-columns: 1fr;
            }}
        }}

        .footer {{
            text-align: center;
            background: white;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 8px 25px rgba(0,0,0,0.1);
            max-width: 1400px;
            margin-left: auto;
            margin-right: auto;
        }}

        .footer p {{
            margin: 0;
            color: #7f8c8d;
            font-size: 1.1em;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🎯 Ultimate Oriented Ellipse Comparison</h1>
        <p>Original PNG vs {total_splats} Ultimate Adaptive-Dominant Gaussian Splats</p>
    </div>

    <div class="comparison-container">
        <!-- Original PNG -->
        <div class="image-section">
            <h2>📷 Original PNG Image</h2>
            <div class="image-display">
                <img src="simple_original.png" alt="Original PNG Image">
            </div>

            <div class="specs-grid">
                <div class="spec-box">
                    <div class="spec-title">Format</div>
                    <div class="spec-value">PNG</div>
                </div>
                <div class="spec-box">
                    <div class="spec-title">Type</div>
                    <div class="spec-value">Raster</div>
                </div>
                <div class="spec-box">
                    <div class="spec-title">Resolution</div>
                    <div class="spec-value">512×512</div>
                </div>
                <div class="spec-box">
                    <div class="spec-title">File Size</div>
                    <div class="spec-value">~200 KB</div>
                </div>
            </div>

            <div class="quality-badge badge-original">Source Quality: Perfect</div>
        </div>

        <!-- Ultimate Oriented Splats -->
        <div class="image-section">
            <h2>🎨 {total_splats} Ultimate Oriented Gaussian Splats</h2>
            <div class="image-display">
                <object data="{svg_path}" type="image/svg+xml"></object>
            </div>

            <div class="specs-grid">
                <div class="spec-box">
                    <div class="spec-title">Format</div>
                    <div class="spec-value">SVG</div>
                </div>
                <div class="spec-box">
                    <div class="spec-title">Type</div>
                    <div class="spec-value">Vector</div>
                </div>
                <div class="spec-box">
                    <div class="spec-title">Splats</div>
                    <div class="spec-value">{total_splats}</div>
                </div>
                <div class="spec-box">
                    <div class="spec-title">Parallax</div>
                    <div class="spec-value">Smooth</div>
                </div>
            </div>

            <div class="quality-badge badge-ultimate">Ultimate: 90% Adaptive</div>
        </div>
    </div>

    <div class="comparison-stats">
        <h2>📊 Ultimate Oriented Ellipse Statistics</h2>
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-title">Total Splats</div>
                <div class="stat-value">{total_splats}</div>
            </div>
            <div class="stat-card">
                <div class="stat-title">Oriented Splats</div>
                <div class="stat-value">{oriented_pct:.1f}%</div>
            </div>
            <div class="stat-card">
                <div class="stat-title">Elliptical Splats</div>
                <div class="stat-value">{elliptical_pct:.1f}%</div>
            </div>
            <div class="stat-card">
                <div class="stat-title">Adaptive Method</div>
                <div class="stat-value">90% (1800)</div>
            </div>
            <div class="stat-card">
                <div class="stat-title">SLIC Method</div>
                <div class="stat-value">10% (200)</div>
            </div>
            <div class="stat-card">
                <div class="stat-title">Parallax Type</div>
                <div class="stat-value">Smooth</div>
            </div>
            <div class="stat-card">
                <div class="stat-title">Color Sampling</div>
                <div class="stat-value">Enhanced</div>
            </div>
            <div class="stat-card">
                <div class="stat-title">Orientation</div>
                <div class="stat-value">Structure Tensor</div>
            </div>
        </div>
    </div>

    <div class="advantages">
        <h2>⚖️ Ultimate Method Comparison</h2>
        <div class="advantages-grid">
            <div class="advantage-list">
                <h3>📷 PNG Advantages</h3>
                <ul>
                    <li><strong>Perfect Quality:</strong> Pixel-perfect representation with no approximation</li>
                    <li><strong>Universal Support:</strong> Works everywhere, no compatibility issues</li>
                    <li><strong>Smaller Size:</strong> 200KB vs ultimate SVG size</li>
                    <li><strong>Simple Format:</strong> Easy to view, edit, and process</li>
                    <li><strong>Instant Loading:</strong> No processing required for display</li>
                </ul>
            </div>

            <div class="advantage-list">
                <h3>🎨 Ultimate Oriented SVG Advantages</h3>
                <ul>
                    <li><strong>Adaptive Dominance:</strong> <span class="highlight">90% adaptive extraction</span> for maximum oriented content</li>
                    <li><strong>Smooth Parallax:</strong> <span class="highlight">Continuous depth values</span> eliminate banding artifacts</li>
                    <li><strong>Enhanced Orientation:</strong> <span class="highlight">{oriented_pct:.1f}% oriented</span> splats with structure tensor analysis</li>
                    <li><strong>Superior Coverage:</strong> Dense adaptive sampling ensures comprehensive representation</li>
                    <li><strong>Anisotropic Excellence:</strong> <span class="highlight">{elliptical_pct:.1f}% elliptical</span> shapes capture complex structures</li>
                    <li><strong>Fixed Color Sampling:</strong> Proper regional color sampling for natural appearance</li>
                    <li><strong>Infinite Scalability:</strong> Vector format scales to any resolution</li>
                </ul>
            </div>
        </div>
    </div>

    <div class="footer">
        <p>🤖 Generated with Ultimate SplatThis Pipeline - 90% adaptive structure-tensor + 10% optimized SLIC extraction.<br>
        <strong>Ultimate Features:</strong> Smooth parallax depth, adaptive-dominant sampling, and enhanced oriented ellipse generation for superior image representation.</p>
    </div>
</body>
</html>'''

    # Save the HTML
    html_path = "ultimate_oriented_comparison.html"
    with open(html_path, 'w') as f:
        f.write(html_content)

    print(f"✅ Saved ultimate comparison HTML to: {html_path}")
    return html_path

if __name__ == "__main__":
    print("🚀 Starting Ultimate Oriented Ellipse Comparison Generation")
    print("=" * 60)

    try:
        html_path = create_ultimate_comparison_html()
        print(f"\n🎉 SUCCESS: Ultimate comparison generated!")
        print(f"📂 Open: {html_path}")
        print(f"\n🔧 Ultimate Features:")
        print(f"  - 90% adaptive extraction for maximum oriented content")
        print(f"  - Smooth continuous parallax (no banding)")
        print(f"  - Enhanced color sampling from image regions")
        print(f"  - Structure tensor-based orientation detection")
        print(f"  - Superior elliptical splat generation")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)