#!/usr/bin/env python3
"""Debug splat colors and structure."""

import sys
import numpy as np
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

from src.splat_this.core.adaptive_extract import AdaptiveSplatExtractor

def load_image(image_path: str) -> np.ndarray:
    with Image.open(image_path) as img:
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize((512, 512), Image.Resampling.LANCZOS)
        return np.array(img).astype(np.float32) / 255.0

def main():
    print("🔍 DEBUG SPLAT COLORS")
    print("=" * 40)

    # Load image
    target = load_image("SCR-20250921-omxs.png")
    print(f"Image shape: {target.shape}")
    print(f"Image range: {target.min():.3f} - {target.max():.3f}")
    print(f"Mean color: {target.mean(axis=(0,1))}")

    # Extract just 5 splats for debugging
    extractor = AdaptiveSplatExtractor()
    splats = extractor.extract_adaptive_splats(target, n_splats=5, verbose=False)

    print(f"\nExtracted {len(splats)} splats")

    for i, splat in enumerate(splats[:3]):
        print(f"\nSplat {i}:")
        print(f"  Position: ({splat.x:.1f}, {splat.y:.1f})")
        print(f"  Color RGB: ({splat.r}, {splat.g}, {splat.b})")
        print(f"  Alpha: {splat.a:.3f}")
        print(f"  Size: rx={splat.rx:.3f}, ry={splat.ry:.3f}")
        print(f"  Rotation: {splat.theta:.3f}")

        # Check the actual pixel color at this position
        x_int = int(np.clip(splat.x, 0, target.shape[1] - 1))
        y_int = int(np.clip(splat.y, 0, target.shape[0] - 1))
        pixel_color = target[y_int, x_int]
        print(f"  Actual pixel at ({x_int}, {y_int}): {pixel_color}")

if __name__ == "__main__":
    main()