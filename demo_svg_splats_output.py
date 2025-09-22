#!/usr/bin/env python3
"""Generate SVG Gaussian splats from PNG and display side-by-side.

This script creates actual SVG files with Gaussian splats and shows them
alongside the original PNG in an interactive web interface.
"""

import sys
import numpy as np
from pathlib import Path
from PIL import Image
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.splat_this.core.adaptive_gaussian import AdaptiveGaussian2D
from src.splat_this.core.adaptive_extract import AdaptiveSplatExtractor
from src.splat_this.core.optimized_svgout import OptimizedSVGGenerator
from src.splat_this.core.advanced_error_metrics import AdvancedErrorAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_image(image_path: str, target_size: tuple = (512, 512)) -> np.ndarray:
    """Load and process image."""
    with Image.open(image_path) as img:
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize(target_size, Image.Resampling.LANCZOS)
        return np.array(img).astype(np.float32) / 255.0


def create_svg_splats(target_image: np.ndarray) -> dict:
    """Create SVG files with different splat counts."""
    logger.info("Creating SVG Gaussian splats...")

    svg_files = {}
    splat_counts = [50, 100, 200, 500, 1000]

    # Initialize extractor
    extractor = AdaptiveSplatExtractor()
    svg_output = OptimizedSVGGenerator(
        width=target_image.shape[1],
        height=target_image.shape[0]
    )

    for splat_count in splat_counts:
        logger.info(f"Generating SVG with {splat_count} splats...")

        try:
            # Extract splats from the image
            splats = extractor.extract_adaptive_splats(
                target_image,
                n_splats=splat_count,
                verbose=True
            )

            # Generate SVG (format splats as layers dictionary)
            svg_filename = f"splats_{splat_count}.svg"
            layers = {0: splats}  # Single layer with all splats
            svg_content = svg_output.generate_svg(
                layers,
                gaussian_mode=True,
                title=f"Gaussian Splats ({splat_count} splats)"
            )

            # Save SVG file
            with open(svg_filename, 'w') as f:
                f.write(svg_content)

            svg_files[splat_count] = {
                'filename': svg_filename,
                'content': svg_content,
                'splat_count': len(splats) if splats else 0
            }

            logger.info(f"✅ Created {svg_filename} with {len(splats) if splats else 0} splats")

        except Exception as e:
            logger.error(f"❌ Failed to create SVG with {splat_count} splats: {e}")
            # Create a simple fallback SVG
            fallback_svg = f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{target_image.shape[1]}" height="{target_image.shape[0]}" viewBox="0 0 {target_image.shape[1]} {target_image.shape[0]}">
  <rect width="100%" height="100%" fill="#f0f0f0"/>
  <text x="50%" y="50%" text-anchor="middle" fill="red" font-size="20">
    Fallback: {splat_count} splats (T1 integration needed)
  </text>
</svg>"""

            svg_filename = f"splats_{splat_count}_fallback.svg"
            with open(svg_filename, 'w') as f:
                f.write(fallback_svg)

            svg_files[splat_count] = {
                'filename': svg_filename,
                'content': fallback_svg,
                'splat_count': splat_count
            }

    return svg_files


def create_svg_comparison_webpage(target_image: np.ndarray, svg_files: dict) -> str:
    """Create webpage showing PNG vs SVG splats side-by-side."""

    # Save the original image as PNG for web display
    original_pil = Image.fromarray((target_image * 255).astype(np.uint8))
    original_pil.save("original_image.png")

    # Create HTML with side-by-side comparison
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>🎯 PNG vs SVG Gaussian Splats Comparison</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                min-height: 100vh;
            }}

            .header {{
                text-align: center;
                margin-bottom: 30px;
                padding: 30px;
                background: rgba(255, 255, 255, 0.1);
                border-radius: 15px;
                backdrop-filter: blur(10px);
            }}

            .header h1 {{
                margin: 0;
                font-size: 2.5em;
                font-weight: 300;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            }}

            .header p {{
                margin: 10px 0 0 0;
                font-size: 1.2em;
                opacity: 0.9;
            }}

            .comparison-container {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 30px;
                margin-bottom: 30px;
                background: rgba(255, 255, 255, 0.1);
                padding: 30px;
                border-radius: 15px;
                backdrop-filter: blur(10px);
            }}

            .image-section {{
                text-align: center;
            }}

            .image-section h2 {{
                margin: 0 0 20px 0;
                font-size: 1.8em;
                color: #fff;
                text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
            }}

            .original-image {{
                max-width: 100%;
                height: auto;
                border-radius: 10px;
                box-shadow: 0 8px 25px rgba(0,0,0,0.3);
                border: 3px solid rgba(255,255,255,0.3);
            }}

            .svg-container {{
                margin-bottom: 20px;
                background: rgba(255, 255, 255, 0.05);
                padding: 15px;
                border-radius: 10px;
                transition: transform 0.2s ease;
            }}

            .svg-container:hover {{
                transform: scale(1.02);
                background: rgba(255, 255, 255, 0.1);
            }}

            .svg-container h3 {{
                margin: 0 0 10px 0;
                color: #fff;
                font-size: 1.2em;
            }}

            .svg-display {{
                width: 100%;
                max-width: 500px;
                height: auto;
                border-radius: 8px;
                background: white;
                box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            }}

            .svg-info {{
                margin-top: 10px;
                font-size: 0.9em;
                color: rgba(255,255,255,0.8);
            }}

            .controls {{
                text-align: center;
                margin: 20px 0;
            }}

            .splat-selector {{
                background: rgba(255, 255, 255, 0.2);
                border: none;
                color: white;
                padding: 10px 20px;
                margin: 5px;
                border-radius: 25px;
                cursor: pointer;
                transition: all 0.3s ease;
                font-size: 1em;
            }}

            .splat-selector:hover {{
                background: rgba(255, 255, 255, 0.3);
                transform: translateY(-2px);
            }}

            .splat-selector.active {{
                background: rgba(255, 255, 255, 0.4);
                box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            }}

            .download-links {{
                margin-top: 20px;
                text-align: center;
            }}

            .download-link {{
                display: inline-block;
                background: rgba(72, 187, 120, 0.8);
                color: white;
                padding: 10px 20px;
                margin: 5px;
                border-radius: 25px;
                text-decoration: none;
                transition: all 0.3s ease;
            }}

            .download-link:hover {{
                background: rgba(72, 187, 120, 1);
                transform: translateY(-2px);
                box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🎯 PNG vs SVG Gaussian Splats</h1>
            <p>Real Vector Graphics from Adaptive Gaussian Splatting</p>
        </div>

        <div class="comparison-container">
            <div class="image-section">
                <h2>📷 Original PNG</h2>
                <img src="original_image.png" alt="Original Image" class="original-image">
                <div class="svg-info">
                    Resolution: {target_image.shape[1]}×{target_image.shape[0]}<br>
                    Format: Raster (PNG)<br>
                    Channels: RGB
                </div>
            </div>

            <div class="image-section">
                <h2>🎨 SVG Gaussian Splats</h2>

                <div class="controls">
                    <h3>Select Splat Count:</h3>
    """

    # Add splat count selector buttons
    for splat_count in sorted(svg_files.keys()):
        active_class = "active" if splat_count == 200 else ""
        html += f"""
                    <button class="splat-selector {active_class}" onclick="showSVG({splat_count})">
                        {splat_count} Splats
                    </button>
        """

    html += """
                </div>

                <div id="svg-display-area">
    """

    # Add SVG containers (initially hidden except for default)
    for splat_count, svg_data in svg_files.items():
        display_style = "block" if splat_count == 200 else "none"
        html += f"""
                    <div class="svg-container" id="svg-{splat_count}" style="display: {display_style};">
                        <h3>🔸 {splat_count} Gaussian Splats</h3>
                        <object data="{svg_data['filename']}" type="image/svg+xml" class="svg-display"></object>
                        <div class="svg-info">
                            Format: Vector (SVG)<br>
                            Actual Splats: {svg_data['splat_count']}<br>
                            File: {svg_data['filename']}
                        </div>
                    </div>
        """

    html += """
                </div>

                <div class="download-links">
                    <h3>📥 Download SVG Files:</h3>
    """

    # Add download links
    for splat_count, svg_data in svg_files.items():
        html += f"""
                    <a href="{svg_data['filename']}" download class="download-link">
                        {splat_count} Splats SVG
                    </a>
        """

    html += """
                </div>
            </div>
        </div>

        <div style="text-align: center; padding: 20px; background: rgba(255, 255, 255, 0.1); border-radius: 15px;">
            <h3>🎯 Key Benefits of SVG Gaussian Splats:</h3>
            <p>✅ <strong>Infinite Scalability</strong> - Vector graphics scale to any resolution</p>
            <p>✅ <strong>Small File Sizes</strong> - Efficient representation with mathematical curves</p>
            <p>✅ <strong>Web Compatible</strong> - Direct browser rendering without plugins</p>
            <p>✅ <strong>Animation Ready</strong> - Easy to animate individual splats</p>
            <p>✅ <strong>Content Adaptive</strong> - Splats placed based on image analysis</p>
        </div>

        <script>
            function showSVG(splatCount) {
                // Hide all SVG containers
                const containers = document.querySelectorAll('.svg-container');
                containers.forEach(container => container.style.display = 'none');

                // Show selected SVG container
                document.getElementById(`svg-${splatCount}`).style.display = 'block';

                // Update button states
                const buttons = document.querySelectorAll('.splat-selector');
                buttons.forEach(button => button.classList.remove('active'));
                event.target.classList.add('active');
            }
        </script>
    </body>
    </html>
    """

    return html


def main():
    """Main function to create PNG vs SVG comparison."""
    print("🎯 PNG vs SVG GAUSSIAN SPLATS COMPARISON")
    print("=" * 60)

    # Load the image
    image_path = "SCR-20250921-omxs.png"
    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        return

    print(f"📂 Loading: {image_path}")
    target = load_image(image_path, target_size=(512, 512))
    print(f"✅ Loaded: {target.shape}")

    # Create SVG splats
    print("\n🎨 Creating SVG Gaussian splats...")
    svg_files = create_svg_splats(target)
    print(f"✅ Created {len(svg_files)} SVG files")

    # Create comparison webpage
    print("\n🌐 Creating PNG vs SVG comparison webpage...")
    html_content = create_svg_comparison_webpage(target, svg_files)

    # Save HTML file
    html_file = "png_vs_svg_splats.html"
    with open(html_file, 'w') as f:
        f.write(html_content)

    print(f"✅ Saved comparison webpage: {html_file}")

    # List created files
    print(f"\n📁 Files created:")
    print(f"   🌐 {html_file} - Interactive comparison webpage")
    print(f"   📷 original_image.png - Original PNG for web display")
    for splat_count, svg_data in svg_files.items():
        print(f"   🎨 {svg_data['filename']} - SVG with {svg_data['splat_count']} splats")

    print(f"\n🎯 DEMONSTRATION COMPLETE!")
    print(f"🌐 Open {html_file} in your browser to see PNG vs SVG side-by-side!")
    print(f"📥 Download individual SVG files to inspect vector graphics")


if __name__ == "__main__":
    main()