#!/usr/bin/env python3
"""Simple PNG to SVG conversion - end-to-end test."""

import sys
import numpy as np
from pathlib import Path
from PIL import Image
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.splat_this.core.adaptive_extract import AdaptiveSplatExtractor
from src.splat_this.core.optimized_svgout import OptimizedSVGGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_image(image_path: str, target_size: tuple = (512, 512)) -> np.ndarray:
    """Load and process image."""
    with Image.open(image_path) as img:
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize(target_size, Image.Resampling.LANCZOS)
        return np.array(img).astype(np.float32) / 255.0


def main():
    """Simple PNG to SVG conversion."""
    print("🎯 SIMPLE PNG → SVG CONVERSION")
    print("=" * 50)

    # Load the image
    image_path = "simple_original.png"
    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        return

    print(f"📂 Loading: {image_path}")
    target = load_image(image_path, target_size=(512, 512))
    print(f"✅ Loaded: {target.shape}")

    # Extract 1000 splats
    print(f"\n🔸 Extracting 1000 Gaussian splats...")
    extractor = AdaptiveSplatExtractor()
    splats = extractor.extract_adaptive_splats(target, n_splats=1000, verbose=True)
    print(f"✅ Extracted {len(splats)} splats")

    # Generate SVG
    print(f"\n🎨 Generating SVG...")
    svg_generator = OptimizedSVGGenerator(
        width=target.shape[1],
        height=target.shape[0]
    )

    # Format splats as layers
    layers = {0: splats}
    svg_content = svg_generator.generate_svg(
        layers,
        gaussian_mode=True,
        title="1000 Gaussian Splats"
    )

    # Save files
    svg_file = "simple_1000_splats.svg"
    with open(svg_file, 'w') as f:
        f.write(svg_content)

    # Save original as PNG for comparison
    original_pil = Image.fromarray((target * 255).astype(np.uint8))
    png_file = "simple_original.png"
    original_pil.save(png_file)

    print(f"✅ Saved: {svg_file}")
    print(f"✅ Saved: {png_file}")

    # Create simple HTML comparison
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>PNG vs SVG - Simple Test</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .comparison {{ display: flex; gap: 20px; }}
        .image-box {{ text-align: center; }}
        .image-box img, .image-box object {{ max-width: 500px; height: auto; border: 1px solid #ccc; }}
    </style>
</head>
<body>
    <h1>🎯 PNG vs SVG - Simple Test</h1>

    <div class="comparison">
        <div class="image-box">
            <h2>📷 Original PNG</h2>
            <img src="{png_file}" alt="Original PNG">
            <p>Format: Raster (PNG)<br>Resolution: {target.shape[1]}×{target.shape[0]}</p>
        </div>

        <div class="image-box">
            <h2>🎨 1000 Gaussian Splats</h2>
            <object data="{svg_file}" type="image/svg+xml" width="500"></object>
            <p>Format: Vector (SVG)<br>Splats: 1000</p>
        </div>
    </div>
</body>
</html>"""

    html_file = "simple_comparison.html"
    with open(html_file, 'w') as f:
        f.write(html)

    print(f"✅ Saved: {html_file}")
    print(f"\n🎯 COMPLETE!")
    print(f"📁 Files: {png_file}, {svg_file}, {html_file}")
    print(f"🌐 Open {html_file} to see PNG vs SVG side-by-side")


if __name__ == "__main__":
    main()