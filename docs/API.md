# SplatThis API Documentation

## Overview

SplatThis provides a comprehensive Python API for converting images into parallax-animated SVG graphics using Gaussian splat technology. The API is organized into core components for extraction, layering, and SVG generation, with both standard and optimized implementations.

## Core Components

### `splat_this.core.extract`

#### Class: `Gaussian`

Represents a single Gaussian splat with position, size, rotation, and color.

```python
@dataclass
class Gaussian:
    x: float          # X position
    y: float          # Y position
    rx: float         # X radius (must be > 0)
    ry: float         # Y radius (must be > 0)
    theta: float      # Rotation angle in radians
    r: int           # Red component (0-255)
    g: int           # Green component (0-255)
    b: int           # Blue component (0-255)
    a: float         # Alpha transparency (0.0-1.0)
    score: float = 0.0    # Importance score
    depth: float = 0.5    # Depth layer (0.0-1.0)
```

**Methods:**
- `validate() -> None`: Validates splat parameters
- `area() -> float`: Calculates ellipse area
- `to_dict() -> dict`: Converts to dictionary representation

#### Class: `SplatExtractor`

Extracts Gaussian splats from images using SLIC superpixel segmentation.

```python
class SplatExtractor:
    def __init__(self,
                 n_segments: int = 2000,
                 compactness: float = 10.0,
                 sigma: float = 1.0,
                 max_size_limit: int = 2048):
```

**Parameters:**
- `n_segments`: Target number of superpixel segments
- `compactness`: Balance between color proximity and space proximity
- `sigma`: Gaussian smoothing parameter
- `max_size_limit`: Maximum image dimension for processing

**Methods:**

```python
def extract_splats(self,
                  image: np.ndarray,
                  n_splats: int = 1500,
                  min_size: int = 4) -> List[Gaussian]:
    """Extract Gaussian splats from image.

    Args:
        image: Input image as numpy array (H, W, 3)
        n_splats: Target number of splats to extract
        min_size: Minimum splat size in pixels

    Returns:
        List of Gaussian splats sorted by importance
    """
```

```python
def estimate_gaussian_params(self,
                           region_coords: np.ndarray,
                           mean_color: np.ndarray) -> Optional[Gaussian]:
    """Estimate Gaussian parameters for a region.

    Args:
        region_coords: Array of (y, x) coordinates
        mean_color: RGB color values [0-255]

    Returns:
        Gaussian splat or None if estimation fails
    """
```

### `splat_this.core.layering`

#### Class: `ImportanceScorer`

Scores splats by visual importance for depth assignment.

```python
class ImportanceScorer:
    def __init__(self,
                 area_weight: float = 0.3,
                 edge_weight: float = 0.5,
                 color_weight: float = 0.2):
```

**Parameters:**
- `area_weight`: Weight for area-based scoring
- `edge_weight`: Weight for edge-based scoring
- `color_weight`: Weight for color-based scoring

**Methods:**

```python
def score_splats(self, splats: List[Gaussian], image: np.ndarray) -> None:
    """Update splat scores based on importance factors.

    Args:
        splats: List of splats to score (modified in-place)
        image: Original image for context analysis
    """
```

#### Class: `LayerAssigner`

Assigns splats to depth layers for parallax effects.

```python
class LayerAssigner:
    def __init__(self, n_layers: int = 4):
```

**Methods:**

```python
def assign_layers(self, splats: List[Gaussian]) -> Dict[int, List[Gaussian]]:
    """Distribute splats across depth layers.

    Args:
        splats: List of scored splats

    Returns:
        Dictionary mapping layer indices to splat lists
    """
```

```python
def assign_depth_values(self, splats: List[Gaussian]) -> None:
    """Assign continuous depth values to splats.

    Args:
        splats: List of splats to assign depths (modified in-place)
    """
```

### `splat_this.core.svgout`

#### Class: `SVGGenerator`

Generates animated SVG from layered splats.

```python
class SVGGenerator:
    def __init__(self,
                 width: int,
                 height: int,
                 precision: int = 3,
                 parallax_strength: int = 40,
                 interactive_top: int = 0):
```

**Parameters:**
- `width`, `height`: SVG viewport dimensions
- `precision`: Decimal precision for coordinates
- `parallax_strength`: Mouse parallax sensitivity (0-200)
- `interactive_top`: Number of top-layer interactive splats

**Methods:**

```python
def generate_svg(self,
                layers: Dict[int, List[Gaussian]],
                gaussian_mode: bool = False,
                title: Optional[str] = None) -> str:
    """Generate complete SVG with layers and animation.

    Args:
        layers: Dictionary mapping layer indices to splat lists
        gaussian_mode: Enable smooth gradient rendering
        title: Optional title for accessibility

    Returns:
        Complete SVG markup as string
    """
```

```python
def save_svg(self,
            layers: Dict[int, List[Gaussian]],
            output_path: str,
            **kwargs) -> None:
    """Generate and save SVG to file.

    Args:
        layers: Layered splats dictionary
        output_path: Output file path
        **kwargs: Additional arguments for generate_svg()
    """
```

## Optimized Components

High-performance variants with the same API but enhanced efficiency:

### `splat_this.core.optimized_extract.OptimizedSplatExtractor`
### `splat_this.core.optimized_layering.OptimizedImportanceScorer`
### `splat_this.core.optimized_svgout.OptimizedSVGGenerator`

Performance improvements:
- **1.5x - 1.85x faster** processing
- **Reduced memory usage** through streaming
- **Parallel processing** for large datasets
- **Automatic optimization** based on input size

## Utility Modules

### `splat_this.utils.image`

#### Class: `ImageLoader`

```python
class ImageLoader:
    @staticmethod
    def load_image(path: str, max_size: Optional[int] = None) -> np.ndarray:
        """Load and validate image from file.

        Args:
            path: Image file path (JPG, PNG, GIF supported)
            max_size: Maximum dimension for automatic resizing

        Returns:
            RGB image array with shape (H, W, 3)

        Raises:
            ValueError: Invalid image format or corrupted file
            FileNotFoundError: Image file not found
        """
```

```python
def validate_image_array(image: np.ndarray) -> None:
    """Validate image array format.

    Args:
        image: Image array to validate

    Raises:
        ValueError: Invalid array format or dimensions
    """
```

### `splat_this.utils.math`

Mathematical utilities for splat processing:

```python
def safe_eigendecomposition(matrix: np.ndarray) -> tuple:
    """Safely compute eigenvalues and eigenvectors.

    Args:
        matrix: 2x2 covariance matrix

    Returns:
        Tuple of (eigenvalues, eigenvectors) or (None, None) if failed
    """
```

```python
def clamp_value(value: float, min_val: float, max_val: float) -> float:
    """Clamp value to specified range.

    Args:
        value: Value to clamp
        min_val: Minimum allowed value
        max_val: Maximum allowed value

    Returns:
        Clamped value
    """
```

### `splat_this.utils.profiler`

#### Class: `PerformanceProfiler`

```python
class PerformanceProfiler:
    def __init__(self):
        """Initialize performance profiler."""

    def start_timing(self, operation: str) -> None:
        """Start timing an operation."""

    def end_timing(self, operation: str) -> float:
        """End timing and return duration in seconds."""

    def get_stats(self) -> dict:
        """Get comprehensive performance statistics."""

    def reset(self) -> None:
        """Reset all timing data."""
```

## High-Level API

### Complete Pipeline Example

```python
from splat_this import (
    SplatExtractor, ImportanceScorer, LayerAssigner, SVGGenerator
)
from splat_this.utils.image import ImageLoader

# Load image
loader = ImageLoader()
image = loader.load_image("input.jpg", max_size=1920)

# Extract splats
extractor = SplatExtractor(n_segments=2000)
splats = extractor.extract_splats(image, n_splats=1500)

# Score by importance
scorer = ImportanceScorer()
scorer.score_splats(splats, image)

# Assign to layers
assigner = LayerAssigner(n_layers=5)
layers = assigner.assign_layers(splats)

# Generate SVG
generator = SVGGenerator(
    width=image.shape[1],
    height=image.shape[0],
    parallax_strength=60,
    interactive_top=200
)
svg_content = generator.generate_svg(layers, gaussian_mode=True)

# Save result
with open("output.svg", "w") as f:
    f.write(svg_content)
```

### Optimized Pipeline

```python
from splat_this.core.optimized_extract import OptimizedSplatExtractor
from splat_this.core.optimized_layering import OptimizedImportanceScorer
from splat_this.core.optimized_svgout import OptimizedSVGGenerator

# Use optimized components for 1.85x speedup
extractor = OptimizedSplatExtractor()
scorer = OptimizedImportanceScorer()
generator = OptimizedSVGGenerator(width, height)

# Same API, better performance
splats = extractor.extract_splats(image, n_splats=2000)
scorer.score_splats(splats, image)
layers = assigner.assign_layers(splats)
svg_content = generator.generate_svg(layers, gaussian_mode=True)
```

## Error Handling

### Common Exceptions

```python
# Image validation errors
try:
    image = loader.load_image("invalid.jpg")
except ValueError as e:
    print(f"Invalid image: {e}")
except FileNotFoundError:
    print("Image file not found")

# Splat validation errors
try:
    splat = Gaussian(x=10, y=10, rx=-1, ry=5, ...)  # Invalid radius
except ValueError as e:
    print(f"Invalid splat parameters: {e}")

# SVG generation errors
try:
    svg = generator.generate_svg({})  # Empty layers
except Exception as e:
    print(f"SVG generation failed: {e}")
```

### Best Practices

1. **Always validate inputs** before processing
2. **Use try-catch blocks** for file operations
3. **Check image dimensions** before processing large files
4. **Monitor memory usage** for batch processing
5. **Use optimized components** for production workloads

## Performance Considerations

### Memory Management
```python
# For large images, use automatic downsampling
extractor = SplatExtractor(max_size_limit=1920)

# Or manually resize
from PIL import Image
pil_image = Image.open("large.jpg")
pil_image.thumbnail((1920, 1920), Image.Resampling.LANCZOS)
image = np.array(pil_image)
```

### Optimization Tips
```python
# Use fewer splats for faster processing
splats = extractor.extract_splats(image, n_splats=800)  # vs 1500

# Reduce layers for simpler output
assigner = LayerAssigner(n_layers=3)  # vs 5

# Lower precision for smaller files
generator = SVGGenerator(width, height, precision=2)  # vs 3
```

## Browser Compatibility

### Generated SVG Features
- **SVG 1.1 compliant** markup
- **CSS3 transforms** for animation
- **ES5+ JavaScript** for interaction
- **Responsive design** with viewport scaling
- **Accessibility support** with ARIA labels

### Feature Detection
```javascript
// In generated SVG JavaScript
const hasGyroscope = 'DeviceOrientationEvent' in window;
const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
```

## Migration Guide

### From v0.0.x to v0.1.0
- `extract_splats()` now returns sorted splats by default
- `LayerAssigner` constructor requires `n_layers` parameter
- `SVGGenerator` width/height are now required constructor parameters
- Optimized components available for performance-critical applications

This API documentation covers the complete SplatThis Python interface. For more examples and tutorials, see the [Examples Documentation](EXAMPLES.md).