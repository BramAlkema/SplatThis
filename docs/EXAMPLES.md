# SplatThis Examples & Tutorials

This guide provides comprehensive examples and tutorials for using SplatThis to create stunning parallax-animated SVG graphics.

## Table of Contents

1. [Quick Start Examples](#quick-start-examples)
2. [Command Line Tutorials](#command-line-tutorials)
3. [Python API Tutorials](#python-api-tutorials)
4. [Advanced Techniques](#advanced-techniques)
5. [Performance Optimization](#performance-optimization)
6. [Troubleshooting Common Issues](#troubleshooting-common-issues)

## Quick Start Examples

### Basic Image Conversion

```bash
# Convert a photo to animated SVG
splatlify photo.jpg -o parallax.svg

# Open the result in a browser
open parallax.svg  # macOS
start parallax.svg  # Windows
xdg-open parallax.svg  # Linux
```

### High-Quality Output

```bash
# More splats for better quality
splatlify landscape.png --splats 3000 --layers 6 -o high_quality.svg

# Enable Gaussian gradients for smoother appearance
splatlify portrait.jpg --gaussian --splats 2500 -o smooth.svg
```

## Command Line Tutorials

### Tutorial 1: Basic Parallax Effect

Create a simple parallax effect from a landscape photo:

```bash
# Step 1: Start with default settings
splatlify landscape.jpg -o step1.svg

# Step 2: Increase parallax strength for more motion
splatlify landscape.jpg --parallax-strength 80 -o step2.svg

# Step 3: Add more layers for depth
splatlify landscape.jpg --parallax-strength 80 --layers 6 -o step3.svg

# Step 4: Fine-tune splat count
splatlify landscape.jpg --parallax-strength 80 --layers 6 --splats 2000 -o final.svg
```

**Expected Results:**
- `step1.svg`: Subtle parallax motion
- `step2.svg`: More pronounced motion
- `step3.svg`: Smoother depth transitions
- `final.svg`: Higher quality with more detail

### Tutorial 2: Interactive Elements

Create an interactive composition with elements that respond to mouse movement:

```bash
# Basic interactive setup
splatlify cityscape.jpg --interactive-top 300 -o interactive_basic.svg

# Enhanced interactivity with stronger parallax
splatlify cityscape.jpg \
  --interactive-top 500 \
  --parallax-strength 100 \
  --layers 5 \
  -o interactive_enhanced.svg

# Fine-tuned for optimal user experience
splatlify cityscape.jpg \
  --interactive-top 400 \
  --parallax-strength 60 \
  --layers 6 \
  --splats 2500 \
  --k 2.8 \
  --alpha 0.75 \
  -o interactive_optimized.svg
```

### Tutorial 3: Animation Frame Processing

Extract specific frames from animated GIFs:

```bash
# Process the first frame (default)
splatlify animation.gif -o frame_0.svg

# Process a specific frame
splatlify animation.gif --frame 10 -o frame_10.svg

# Create multiple frames with consistent settings
for i in {0..15}; do
  splatlify animation.gif --frame $i --splats 1500 -o "frame_${i}.svg"
done
```

### Tutorial 4: Custom Styling

Fine-tune visual appearance:

```bash
# Larger, more transparent splats
splatlify texture.png \
  --k 3.5 \
  --alpha 0.4 \
  --splats 1200 \
  -o large_transparent.svg

# Smaller, more opaque splats for detailed look
splatlify pattern.jpg \
  --k 1.8 \
  --alpha 0.9 \
  --splats 3500 \
  -o detailed_opaque.svg

# Balanced approach with Gaussian gradients
splatlify photo.jpg \
  --gaussian \
  --k 2.5 \
  --alpha 0.65 \
  --splats 2000 \
  --layers 5 \
  -o balanced_smooth.svg
```

## Python API Tutorials

### Tutorial 1: Basic Python Pipeline

```python
from splat_this import SplatExtractor, LayerAssigner, SVGGenerator
from splat_this.utils.image import ImageLoader
import numpy as np

# Load image
loader = ImageLoader()
image = loader.load_image("input.jpg")

print(f"Loaded image: {image.shape}")

# Extract splats
extractor = SplatExtractor(n_segments=2000)
splats = extractor.extract_splats(image, n_splats=1500)

print(f"Extracted {len(splats)} splats")

# Assign to layers
assigner = LayerAssigner(n_layers=4)
layers = assigner.assign_layers(splats)

print(f"Created {len(layers)} layers")
for layer_id, layer_splats in layers.items():
    print(f"  Layer {layer_id}: {len(layer_splats)} splats")

# Generate SVG
generator = SVGGenerator(
    width=image.shape[1],
    height=image.shape[0],
    parallax_strength=40
)
svg_content = generator.generate_svg(layers)

# Save result
with open("output.svg", "w") as f:
    f.write(svg_content)

print("SVG saved to output.svg")
```

### Tutorial 2: Custom Splat Processing

```python
from splat_this.core.extract import SplatExtractor, Gaussian
from splat_this.core.layering import ImportanceScorer
import numpy as np

# Load image
from splat_this.utils.image import ImageLoader
loader = ImageLoader()
image = loader.load_image("input.jpg", max_size=1024)

# Extract with custom parameters
extractor = SplatExtractor(
    n_segments=3000,    # More initial segments
    compactness=15.0,   # More compact regions
    sigma=0.8          # Less smoothing
)
splats = extractor.extract_splats(image, n_splats=2000, min_size=6)

# Custom importance scoring
scorer = ImportanceScorer(
    area_weight=0.4,    # Emphasize larger splats
    edge_weight=0.4,    # Emphasize edge details
    color_weight=0.2    # De-emphasize color variation
)
scorer.score_splats(splats, image)

# Filter splats by custom criteria
def filter_splats(splats, min_score=0.3, max_area=1000):
    """Filter splats by score and area."""
    filtered = []
    for splat in splats:
        if splat.score >= min_score and splat.area() <= max_area:
            filtered.append(splat)
    return filtered

filtered_splats = filter_splats(splats, min_score=0.4)
print(f"Filtered to {len(filtered_splats)} high-quality splats")

# Custom layer assignment
def custom_layer_assignment(splats, n_layers=5):
    """Assign splats to layers based on size and score."""
    layers = {i: [] for i in range(n_layers)}

    for splat in splats:
        # Combine area and score for layer decision
        layer_score = (splat.area() * 0.3) + (splat.score * 0.7)
        layer_id = min(int(layer_score * n_layers), n_layers - 1)
        layers[layer_id].append(splat)

    return layers

custom_layers = custom_layer_assignment(filtered_splats)

# Generate with custom settings
from splat_this.core.svgout import SVGGenerator
generator = SVGGenerator(
    width=image.shape[1],
    height=image.shape[0],
    precision=4,
    parallax_strength=60,
    interactive_top=300
)

svg_content = generator.generate_svg(
    custom_layers,
    gaussian_mode=True,
    title="Custom Processed Image"
)

generator.save_svg(custom_layers, "custom_output.svg", gaussian_mode=True)
```

### Tutorial 3: Batch Processing

```python
import os
from pathlib import Path
from splat_this import SplatExtractor, LayerAssigner, SVGGenerator
from splat_this.utils.image import ImageLoader

def process_image_batch(input_dir, output_dir, **kwargs):
    """Process all images in a directory."""

    # Setup components
    loader = ImageLoader()
    extractor = SplatExtractor()
    assigner = LayerAssigner(n_layers=kwargs.get('layers', 4))

    # Process each image
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    image_extensions = {'.jpg', '.jpeg', '.png', '.gif'}

    for image_file in input_path.iterdir():
        if image_file.suffix.lower() in image_extensions:
            print(f"Processing {image_file.name}...")

            try:
                # Load and process
                image = loader.load_image(str(image_file))
                splats = extractor.extract_splats(
                    image,
                    n_splats=kwargs.get('splats', 1500)
                )
                layers = assigner.assign_layers(splats)

                # Generate SVG
                generator = SVGGenerator(
                    width=image.shape[1],
                    height=image.shape[0],
                    parallax_strength=kwargs.get('parallax_strength', 40),
                    interactive_top=kwargs.get('interactive_top', 0)
                )

                # Save with same name but .svg extension
                output_file = output_path / f"{image_file.stem}.svg"
                generator.save_svg(
                    layers,
                    str(output_file),
                    gaussian_mode=kwargs.get('gaussian', False)
                )

                print(f"  → {output_file.name}")

            except Exception as e:
                print(f"  Error processing {image_file.name}: {e}")

# Usage
process_image_batch(
    input_dir="./input_images",
    output_dir="./output_svgs",
    splats=2000,
    layers=5,
    parallax_strength=60,
    gaussian=True
)
```

### Tutorial 4: Performance Optimization

```python
from splat_this.core.optimized_extract import OptimizedSplatExtractor
from splat_this.core.optimized_layering import OptimizedImportanceScorer
from splat_this.core.optimized_svgout import OptimizedSVGGenerator
from splat_this.utils.profiler import PerformanceProfiler
import time

def performance_comparison(image_path):
    """Compare standard vs optimized components."""

    from splat_this.utils.image import ImageLoader
    loader = ImageLoader()
    image = loader.load_image(image_path)

    profiler = PerformanceProfiler()

    # Standard pipeline
    print("Running standard pipeline...")
    profiler.start_timing("standard_total")

    from splat_this import SplatExtractor, LayerAssigner, SVGGenerator

    profiler.start_timing("standard_extract")
    extractor = SplatExtractor()
    splats = extractor.extract_splats(image, n_splats=2000)
    profiler.end_timing("standard_extract")

    profiler.start_timing("standard_layers")
    assigner = LayerAssigner(n_layers=5)
    layers = assigner.assign_layers(splats)
    profiler.end_timing("standard_layers")

    profiler.start_timing("standard_svg")
    generator = SVGGenerator(image.shape[1], image.shape[0])
    svg_content = generator.generate_svg(layers)
    profiler.end_timing("standard_svg")

    standard_time = profiler.end_timing("standard_total")

    # Optimized pipeline
    print("Running optimized pipeline...")
    profiler.start_timing("optimized_total")

    profiler.start_timing("optimized_extract")
    opt_extractor = OptimizedSplatExtractor()
    opt_splats = opt_extractor.extract_splats(image, n_splats=2000)
    profiler.end_timing("optimized_extract")

    profiler.start_timing("optimized_layers")
    opt_assigner = LayerAssigner(n_layers=5)  # Same as standard
    opt_layers = opt_assigner.assign_layers(opt_splats)
    profiler.end_timing("optimized_layers")

    profiler.start_timing("optimized_svg")
    opt_generator = OptimizedSVGGenerator(image.shape[1], image.shape[0])
    opt_svg_content = opt_generator.generate_svg(opt_layers)
    profiler.end_timing("optimized_svg")

    optimized_time = profiler.end_timing("optimized_total")

    # Results
    speedup = standard_time / optimized_time
    print(f"\nPerformance Results:")
    print(f"Standard pipeline: {standard_time:.2f}s")
    print(f"Optimized pipeline: {optimized_time:.2f}s")
    print(f"Speedup: {speedup:.2f}x")

    # Detailed timing
    stats = profiler.get_stats()
    print(f"\nDetailed Timing:")
    for operation, timing in stats.items():
        print(f"  {operation}: {timing:.3f}s")

# Usage
performance_comparison("large_image.jpg")
```

## Advanced Techniques

### Custom Splat Rendering

```python
from splat_this.core.extract import Gaussian
from splat_this.core.svgout import SVGGenerator

class CustomSVGGenerator(SVGGenerator):
    """Custom SVG generator with additional effects."""

    def _generate_splat_element(self, splat, layer_id, gaussian_mode=False):
        """Override to add custom splat rendering."""

        # Add pulse animation to high-score splats
        if splat.score > 0.8:
            return self._generate_pulsing_splat(splat, layer_id, gaussian_mode)

        # Use parent implementation for normal splats
        return super()._generate_splat_element(splat, layer_id, gaussian_mode)

    def _generate_pulsing_splat(self, splat, layer_id, gaussian_mode):
        """Generate splat with pulsing animation."""

        # Base splat element
        base_element = super()._generate_splat_element(splat, layer_id, gaussian_mode)

        # Add pulsing animation
        pulse_animation = '''
        <animateTransform
            attributeName="transform"
            type="scale"
            values="1;1.2;1"
            dur="3s"
            repeatCount="indefinite"/>'''

        # Insert animation before closing tag
        return base_element.replace('</ellipse>', pulse_animation + '</ellipse>')

# Usage
generator = CustomSVGGenerator(800, 600)
# ... process as normal
```

### Dynamic Layer Effects

```python
def create_depth_blur_effect(layers, max_blur=5):
    """Add depth-based blur to background layers."""

    total_layers = len(layers)

    for layer_id, splats in layers.items():
        # Calculate blur amount (background layers more blurred)
        blur_amount = (layer_id / total_layers) * max_blur

        for splat in splats:
            # Store blur information in splat (custom attribute)
            splat.blur_radius = blur_amount

# Apply effect
create_depth_blur_effect(layers, max_blur=8)
```

### Responsive SVG Generation

```python
def generate_responsive_svg(layers, base_width, base_height):
    """Generate SVG that adapts to different screen sizes."""

    # Define breakpoints
    breakpoints = {
        'mobile': 480,
        'tablet': 768,
        'desktop': 1200
    }

    # Generate base SVG
    generator = SVGGenerator(base_width, base_height)
    svg_content = generator.generate_svg(layers)

    # Add responsive CSS
    responsive_css = """
    <style>
    @media (max-width: 480px) {
        .splat-layer { transform-origin: center; transform: scale(0.8); }
    }
    @media (min-width: 481px) and (max-width: 768px) {
        .splat-layer { transform-origin: center; transform: scale(0.9); }
    }
    @media (min-width: 1200px) {
        .splat-layer { transform-origin: center; transform: scale(1.1); }
    }
    </style>
    """

    # Insert CSS into SVG
    svg_with_responsive = svg_content.replace(
        '<defs>',
        f'<defs>{responsive_css}'
    )

    return svg_with_responsive
```

## Performance Optimization

### Memory Management for Large Images

```python
def process_large_image(image_path, target_size=1920):
    """Efficiently process large images."""

    from PIL import Image
    import numpy as np

    # Load with automatic resizing
    pil_image = Image.open(image_path)

    # Calculate resize ratio
    max_dim = max(pil_image.size)
    if max_dim > target_size:
        ratio = target_size / max_dim
        new_size = (int(pil_image.size[0] * ratio),
                   int(pil_image.size[1] * ratio))
        pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
        print(f"Resized to {new_size}")

    # Convert to numpy
    image = np.array(pil_image)

    # Process with optimized components
    from splat_this.core.optimized_extract import OptimizedSplatExtractor
    extractor = OptimizedSplatExtractor(max_size_limit=target_size)

    # Use moderate splat count for large images
    splat_count = min(2000, max(1000, (image.shape[0] * image.shape[1]) // 1000))
    splats = extractor.extract_splats(image, n_splats=splat_count)

    print(f"Extracted {len(splats)} splats for {image.shape} image")
    return splats, image
```

### Parallel Processing

```python
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import numpy as np

def process_image_chunk(chunk_data):
    """Process a chunk of image data."""
    chunk, chunk_params = chunk_data

    # Initialize extractor for this process
    from splat_this.core.optimized_extract import OptimizedSplatExtractor
    extractor = OptimizedSplatExtractor()

    # Process chunk
    splats = extractor.extract_splats(chunk, **chunk_params)
    return splats

def parallel_image_processing(image, n_chunks=4, **extract_params):
    """Process image in parallel chunks."""

    # Split image into chunks
    chunk_height = image.shape[0] // n_chunks
    chunks = []

    for i in range(n_chunks):
        start_row = i * chunk_height
        end_row = start_row + chunk_height if i < n_chunks - 1 else image.shape[0]
        chunk = image[start_row:end_row, :, :]
        chunks.append((chunk, extract_params))

    # Process in parallel
    with ProcessPoolExecutor(max_workers=n_chunks) as executor:
        chunk_results = list(executor.map(process_image_chunk, chunks))

    # Combine results
    all_splats = []
    for chunk_splats in chunk_results:
        all_splats.extend(chunk_splats)

    return all_splats

# Usage (for very large images)
# large_splats = parallel_image_processing(large_image, n_chunks=4, n_splats=500)
```

## Troubleshooting Common Issues

### Issue 1: SVG Not Displaying in Browser

**Problem**: SVG file loads but shows blank content.

**Solutions**:
```python
# Check SVG syntax
def validate_svg_syntax(svg_content):
    """Basic SVG validation."""
    required_elements = ['<svg', '</svg>', 'viewBox']

    for element in required_elements:
        if element not in svg_content:
            print(f"Missing required element: {element}")
            return False

    print("SVG syntax appears valid")
    return True

# Usage
with open("output.svg", "r") as f:
    svg_content = f.read()
    validate_svg_syntax(svg_content)
```

### Issue 2: Poor Quality Output

**Problem**: Output lacks detail or looks pixelated.

**Solutions**:
```bash
# Increase splat count
splatlify input.jpg --splats 3000 -o high_detail.svg

# Add more layers
splatlify input.jpg --splats 2500 --layers 6 -o more_layers.svg

# Enable Gaussian mode
splatlify input.jpg --gaussian --splats 2000 -o smooth.svg

# Combine all improvements
splatlify input.jpg --gaussian --splats 3500 --layers 6 --k 2.2 -o best_quality.svg
```

### Issue 3: Performance Issues

**Problem**: Processing takes too long or uses too much memory.

**Solutions**:
```python
# Use optimized components
from splat_this.core.optimized_extract import OptimizedSplatExtractor

# Reduce image size
loader = ImageLoader()
image = loader.load_image("input.jpg", max_size=1280)

# Use fewer splats
splats = extractor.extract_splats(image, n_splats=1000)

# Monitor memory usage
import psutil
import os

def monitor_memory():
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"Memory usage: {memory_mb:.1f} MB")

monitor_memory()
# ... processing ...
monitor_memory()
```

### Issue 4: Browser Compatibility

**Problem**: SVG works in some browsers but not others.

**Solutions**:
```python
# Generate with maximum compatibility
generator = SVGGenerator(
    width=800,
    height=600,
    precision=2,  # Lower precision for smaller files
    parallax_strength=30  # Moderate effects
)

# Disable advanced features for older browsers
svg_content = generator.generate_svg(
    layers,
    gaussian_mode=False,  # Use simple circles instead
    title="Compatible SVG"
)
```

This comprehensive guide should help you get the most out of SplatThis for any use case, from simple conversions to advanced custom processing pipelines.