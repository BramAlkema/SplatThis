# Implementation Tasks - SplatThis CLI Core

**Spec:** @spec.md | **Tasks Overview:** @tasks.md
**Created:** 2025-01-20 | **Status:** Ready for Development

> 📋 **Purpose:** Actionable, step-by-step implementation tasks broken down from the specification
> 🎯 **Goal:** Enable immediate development start with clear deliverables
> ⏱️ **Timeline:** 4 weeks, organized by daily/weekly milestones

## Quick Start Guide

### Prerequisites
```bash
# Development environment setup
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install --upgrade pip setuptools wheel
```

### Task Execution Order
1. **Week 1:** Foundation & Core Pipeline (T1.1 → T1.4)
2. **Week 2:** Depth & Layering System (T2.1 → T2.3)
3. **Week 3:** SVG Generation & Animation (T3.1 → T3.4)
4. **Week 4:** CLI & Integration (T4.1 → T4.4)

---

## Phase 1: Foundation & Core Pipeline

### T1.1: Project Setup & Structure
**🎯 Goal:** Create professional Python package with CI/CD
**⏱️ Estimate:** 4 hours | **🔗 Dependencies:** None

#### T1.1.1: Package Structure Setup
```bash
# Create directory structure
mkdir -p src/splat_this/{core,utils,templates}
mkdir -p tests/{unit,integration,assets}
mkdir -p docs/examples

# Create package files
touch src/splat_this/__init__.py
touch src/splat_this/core/{__init__.py,extract.py,layering.py,svgout.py}
touch src/splat_this/utils/{__init__.py,image.py,math.py}
touch src/splat_this/cli.py
```

#### T1.1.2: Configuration Files
**Create `pyproject.toml`:**
```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "splat-this"
version = "0.1.0"
description = "Convert images to parallax-animated SVG splats"
dependencies = [
    "Pillow>=9.0.0,<11.0.0",
    "numpy>=1.21.0,<2.0.0",
    "scikit-image>=0.19.0,<1.0.0",
    "click>=8.0.0,<9.0.0"
]

[project.scripts]
splatlify = "splat_this.cli:main"

[tool.black]
line-length = 88
target-version = ['py38']

[tool.mypy]
python_version = "3.8"
strict = true
```

**Create `requirements-dev.txt`:**
```txt
pytest>=7.0.0
black>=22.0.0
mypy>=0.991
coverage>=6.0.0
flake8>=5.0.0
```

#### T1.1.3: Basic CLI Entry Point
**Create `src/splat_this/cli.py`:**
```python
import click
from pathlib import Path

@click.command()
@click.argument('input_file', type=click.Path(exists=True, path_type=Path))
@click.option('--frame', default=0, help='GIF frame number (default: 0)')
@click.option('--splats', default=1500, help='Target splat count (default: 1500)')
@click.option('--layers', default=4, help='Depth layers (default: 4)')
@click.option('--k', default=2.5, help='Splat size multiplier (default: 2.5)')
@click.option('--alpha', default=0.65, help='Base alpha (default: 0.65)')
@click.option('--parallax-strength', default=40, help='Parallax strength (default: 40)')
@click.option('--interactive-top', default=0, help='Interactive splats (default: 0)')
@click.option('--gaussian', is_flag=True, help='Enable gradient mode')
@click.option('--verbose', is_flag=True, help='Verbose output')
@click.option('-o', '--output', required=True, type=click.Path(path_type=Path))
def main(input_file, output, **kwargs):
    """Convert image to parallax-animated SVG splats."""
    click.echo(f"Converting {input_file} to {output}")
    click.echo("SplatThis CLI v0.1.0 - Basic structure ready!")

if __name__ == '__main__':
    main()
```

#### T1.1.4: Development Tools Setup
**Create `.github/workflows/ci.yml`:**
```yaml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, "3.10", "3.11"]
    steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    - run: pip install -e .[dev]
    - run: black --check src/
    - run: mypy src/
    - run: pytest
```

**✅ Acceptance Test:**
```bash
pip install -e .
splatlify --help  # Should show usage
```

---

### T1.2: Image Loading & Validation
**🎯 Goal:** Robust image input handling with format support
**⏱️ Estimate:** 6 hours | **🔗 Dependencies:** T1.1

#### T1.2.1: Basic Image Loading
**Create `src/splat_this/utils/image.py`:**
```python
from PIL import Image
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

class ImageLoader:
    SUPPORTED_FORMATS = {'.png', '.jpg', '.jpeg', '.gif'}

    def __init__(self, path: Path, frame: int = 0):
        self.path = path
        self.frame = frame
        self._validate_format()

    def _validate_format(self) -> None:
        if self.path.suffix.lower() not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {self.path.suffix}")

    def load(self) -> np.ndarray:
        """Load image as RGB numpy array."""
        try:
            with Image.open(self.path) as img:
                if img.format == 'GIF' and img.is_animated:
                    return self._extract_gif_frame(img)
                return self._process_static_image(img)
        except Exception as e:
            raise RuntimeError(f"Failed to load {self.path}: {e}")

    def _extract_gif_frame(self, img: Image.Image) -> np.ndarray:
        """Extract specific frame from animated GIF."""
        if self.frame >= img.n_frames:
            raise ValueError(f"Frame {self.frame} not available (max: {img.n_frames-1})")

        img.seek(self.frame)
        return self._process_static_image(img)

    def _process_static_image(self, img: Image.Image) -> np.ndarray:
        """Convert PIL image to RGB numpy array."""
        # Handle orientation from EXIF
        if hasattr(img, '_getexif'):
            img = self._correct_orientation(img)

        # Convert to RGB
        if img.mode != 'RGB':
            img = img.convert('RGB')

        return np.array(img)

    def _correct_orientation(self, img: Image.Image) -> Image.Image:
        """Apply EXIF orientation correction."""
        # Implementation for EXIF orientation handling
        return img

def load_image(path: Path, frame: int = 0) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Convenience function to load image and return array + dimensions."""
    loader = ImageLoader(path, frame)
    array = loader.load()
    return array, array.shape[:2]
```

#### T1.2.2: Validation & Error Handling
```python
def validate_image_dimensions(image: np.ndarray) -> None:
    """Validate image meets size requirements."""
    height, width = image.shape[:2]

    if width < 100 or height < 100:
        raise ValueError(f"Image too small: {width}x{height} (minimum: 100x100)")

    if width > 8192 or height > 8192:
        raise ValueError(f"Image too large: {width}x{height} (maximum: 8192x8192)")

    total_pixels = width * height
    if total_pixels > 33_554_432:  # 8192^2 / 2 for memory safety
        raise ValueError(f"Image has too many pixels: {total_pixels}")
```

#### T1.2.3: Unit Tests
**Create `tests/unit/test_image_loading.py`:**
```python
import pytest
import numpy as np
from pathlib import Path
from splat_this.utils.image import ImageLoader, load_image

def test_load_png_image():
    # Test with sample PNG
    pass

def test_unsupported_format():
    with pytest.raises(ValueError, match="Unsupported format"):
        ImageLoader(Path("test.bmp"))

def test_gif_frame_extraction():
    # Test GIF frame selection
    pass
```

**✅ Acceptance Test:**
```python
# Test image loading works correctly
image, (height, width) = load_image(Path("test.jpg"))
assert isinstance(image, np.ndarray)
assert image.shape == (height, width, 3)
```

---

### T1.3: SLIC Superpixel Implementation
**🎯 Goal:** Extract Gaussian splats using SLIC segmentation
**⏱️ Estimate:** 8 hours | **🔗 Dependencies:** T1.2

#### T1.3.1: SLIC Segmentation
**Create `src/splat_this/core/extract.py`:**
```python
from skimage.segmentation import slic
from skimage.measure import regionprops
import numpy as np
from typing import List, Tuple
from dataclasses import dataclass

@dataclass
class Gaussian:
    x: float
    y: float
    rx: float
    ry: float
    theta: float
    r: int
    g: int
    b: int
    a: float
    score: float = 0.0
    depth: float = 0.5

class SplatExtractor:
    def __init__(self, k: float = 2.5, base_alpha: float = 0.65):
        self.k = k
        self.base_alpha = base_alpha

    def extract_splats(self, image: np.ndarray, n_splats: int) -> List[Gaussian]:
        """Extract Gaussian splats from image using SLIC."""
        # Step 1: SLIC segmentation with oversampling for filtering
        n_segments = int(n_splats * 1.5)  # Oversample for quality filtering
        segments = slic(
            image,
            n_segments=n_segments,
            compactness=10,
            sigma=1.0,
            multichannel=True,
            convert2lab=True
        )

        # Step 2: Extract splats from each region
        splats = []
        for region_id in np.unique(segments):
            if region_id == 0:  # Skip background
                continue

            splat = self._extract_splat_from_region(image, segments, region_id)
            if splat:
                splats.append(splat)

        # Step 3: Filter to target count
        return self._filter_splats(splats, n_splats)

    def _extract_splat_from_region(self, image: np.ndarray, segments: np.ndarray, region_id: int) -> Optional[Gaussian]:
        """Extract single Gaussian splat from segmented region."""
        mask = segments == region_id

        if not np.any(mask):
            return None

        # Get region properties
        region_pixels = image[mask]
        coords = np.column_stack(np.where(mask))

        if len(coords) < 3:  # Need minimum points for covariance
            return None

        # Calculate centroid
        centroid_y, centroid_x = coords.mean(axis=0)

        # Calculate covariance for ellipse parameters
        rx, ry, theta = self._compute_covariance_ellipse(coords)

        # Scale by k parameter
        rx *= self.k
        ry *= self.k

        # Extract color
        mean_color = region_pixels.mean(axis=0).astype(int)
        r, g, b = mean_color

        # Calculate alpha from local contrast
        local_variance = region_pixels.var()
        alpha = min(1.0, self.base_alpha * (1 + local_variance / 255.0))

        return Gaussian(
            x=centroid_x, y=centroid_y,
            rx=rx, ry=ry, theta=theta,
            r=r, g=g, b=b, a=alpha
        )

    def _compute_covariance_ellipse(self, coords: np.ndarray) -> Tuple[float, float, float]:
        """Compute ellipse parameters from coordinate covariance."""
        # Center coordinates
        centered = coords - coords.mean(axis=0)

        # Compute covariance matrix
        cov_matrix = np.cov(centered.T)

        # Eigenvalue decomposition
        eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)

        # Sort by eigenvalue (largest first)
        order = eigenvals.argsort()[::-1]
        eigenvals = eigenvals[order]
        eigenvecs = eigenvecs[:, order]

        # Convert to ellipse parameters
        rx = np.sqrt(max(eigenvals[0], 1e-6))  # Prevent zero radius
        ry = np.sqrt(max(eigenvals[1], 1e-6))

        # Rotation angle
        theta = np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0])

        return rx, ry, theta

    def _filter_splats(self, splats: List[Gaussian], target_count: int) -> List[Gaussian]:
        """Filter splats to target count based on size/importance."""
        if len(splats) <= target_count:
            return splats

        # Score by area (rx * ry)
        for splat in splats:
            splat.score = splat.rx * splat.ry

        # Sort by score and take top splats
        splats.sort(key=lambda s: s.score, reverse=True)
        return splats[:target_count]
```

#### T1.3.2: Mathematical Utilities
**Create `src/splat_this/utils/math.py`:**
```python
import numpy as np
from typing import Tuple

def safe_eigendecomposition(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Safely compute eigenvalues/vectors with fallback for edge cases."""
    try:
        eigenvals, eigenvecs = np.linalg.eigh(matrix)
        return eigenvals, eigenvecs
    except np.linalg.LinAlgError:
        # Fallback to identity for degenerate cases
        return np.array([1.0, 1.0]), np.eye(2)

def clamp_value(value: float, min_val: float, max_val: float) -> float:
    """Clamp value to range [min_val, max_val]."""
    return max(min_val, min(max_val, value))

def normalize_angle(angle: float) -> float:
    """Normalize angle to [-π, π] range."""
    return ((angle + np.pi) % (2 * np.pi)) - np.pi
```

#### T1.3.3: Integration Test
```python
def test_splat_extraction():
    # Create test image
    test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

    extractor = SplatExtractor()
    splats = extractor.extract_splats(test_image, n_splats=50)

    assert len(splats) <= 50
    assert all(isinstance(s, Gaussian) for s in splats)
    assert all(s.rx > 0 and s.ry > 0 for s in splats)
```

**✅ Acceptance Test:**
```python
# Verify SLIC extraction works with real image
image, _ = load_image(Path("test.jpg"))
extractor = SplatExtractor()
splats = extractor.extract_splats(image, n_splats=1500)
print(f"Extracted {len(splats)} splats from image")
```

---

### T1.4: Gaussian Splat Data Structure
**🎯 Goal:** Complete splat representation with utilities
**⏱️ Estimate:** 3 hours | **🔗 Dependencies:** T1.3

#### T1.4.1: Enhanced Dataclass
**Update `Gaussian` dataclass with methods:**
```python
@dataclass
class Gaussian:
    x: float
    y: float
    rx: float
    ry: float
    theta: float
    r: int
    g: int
    b: int
    a: float
    score: float = 0.0
    depth: float = 0.5

    def __post_init__(self):
        """Validate splat parameters."""
        self.validate()

    def validate(self) -> None:
        """Ensure splat parameters are valid."""
        if self.rx <= 0 or self.ry <= 0:
            raise ValueError(f"Invalid radii: rx={self.rx}, ry={self.ry}")

        if not (0 <= self.r <= 255 and 0 <= self.g <= 255 and 0 <= self.b <= 255):
            raise ValueError(f"Invalid RGB: ({self.r}, {self.g}, {self.b})")

        if not (0.0 <= self.a <= 1.0):
            raise ValueError(f"Invalid alpha: {self.a}")

    def area(self) -> float:
        """Calculate ellipse area."""
        return np.pi * self.rx * self.ry

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'position': (self.x, self.y),
            'size': (self.rx, self.ry),
            'rotation': self.theta,
            'color': (self.r, self.g, self.b, self.a),
            'score': self.score,
            'depth': self.depth
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Gaussian':
        """Create from dictionary."""
        return cls(
            x=data['position'][0], y=data['position'][1],
            rx=data['size'][0], ry=data['size'][1],
            theta=data['rotation'],
            r=data['color'][0], g=data['color'][1],
            b=data['color'][2], a=data['color'][3],
            score=data.get('score', 0.0),
            depth=data.get('depth', 0.5)
        )
```

#### T1.4.2: Splat Collection Utilities
```python
class SplatCollection:
    def __init__(self, splats: List[Gaussian]):
        self.splats = splats

    def filter_by_score(self, threshold: float) -> 'SplatCollection':
        """Filter splats by minimum score."""
        filtered = [s for s in self.splats if s.score >= threshold]
        return SplatCollection(filtered)

    def sort_by_depth(self) -> 'SplatCollection':
        """Sort splats by depth (back to front)."""
        sorted_splats = sorted(self.splats, key=lambda s: s.depth)
        return SplatCollection(sorted_splats)

    def get_statistics(self) -> dict:
        """Get collection statistics."""
        if not self.splats:
            return {}

        scores = [s.score for s in self.splats]
        areas = [s.area() for s in self.splats]

        return {
            'count': len(self.splats),
            'score_range': (min(scores), max(scores)),
            'area_range': (min(areas), max(areas)),
            'depth_range': (min(s.depth for s in self.splats),
                           max(s.depth for s in self.splats))
        }
```

**✅ Acceptance Test:**
```python
# Test splat validation and utilities
splat = Gaussian(x=10, y=20, rx=5, ry=3, theta=0.5,
                r=128, g=64, b=32, a=0.8)
assert splat.area() > 0
assert splat.to_dict()['position'] == (10, 20)
```

---

## Phase 2: Depth & Layering System

### T2.1: Importance Scoring Algorithm
**🎯 Goal:** Prioritize splats for depth assignment
**⏱️ Estimate:** 6 hours | **🔗 Dependencies:** T1.4

#### T2.1.1: Multi-factor Scoring
**Create `src/splat_this/core/layering.py`:**
```python
from scipy import ndimage
import cv2

class ImportanceScorer:
    def __init__(self, area_weight: float = 0.3, edge_weight: float = 0.5,
                 color_weight: float = 0.2):
        self.area_weight = area_weight
        self.edge_weight = edge_weight
        self.color_weight = color_weight

    def score_splats(self, splats: List[Gaussian], image: np.ndarray) -> None:
        """Update splat scores based on importance factors."""
        image_area = image.shape[0] * image.shape[1]

        for splat in splats:
            area_score = self._calculate_area_score(splat, image_area)
            edge_score = self._calculate_edge_score(splat, image)
            color_score = self._calculate_color_score(splat, image)

            # Weighted combination
            splat.score = (
                area_score * self.area_weight +
                edge_score * self.edge_weight +
                color_score * self.color_weight
            )

    def _calculate_area_score(self, splat: Gaussian, image_area: float) -> float:
        """Score based on relative area."""
        return (splat.rx * splat.ry) / image_area

    def _calculate_edge_score(self, splat: Gaussian, image: np.ndarray) -> float:
        """Score based on edge strength in splat region."""
        # Extract region around splat
        x, y = int(splat.x), int(splat.y)
        rx, ry = int(splat.rx), int(splat.ry)

        # Define bounding box
        x1, x2 = max(0, x - rx), min(image.shape[1], x + rx)
        y1, y2 = max(0, y - ry), min(image.shape[0], y + ry)

        if x1 >= x2 or y1 >= y2:
            return 0.0

        region = image[y1:y2, x1:x2]
        if region.size == 0:
            return 0.0

        # Convert to grayscale for edge detection
        gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY) if len(region.shape) == 3 else region

        # Calculate Laplacian variance (edge strength)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return laplacian.var() / 255.0  # Normalize

    def _calculate_color_score(self, splat: Gaussian, image: np.ndarray) -> float:
        """Score based on color variance/complexity."""
        # Extract region around splat
        x, y = int(splat.x), int(splat.y)
        rx, ry = int(splat.rx), int(splat.ry)

        x1, x2 = max(0, x - rx), min(image.shape[1], x + rx)
        y1, y2 = max(0, y - ry), min(image.shape[0], y + ry)

        if x1 >= x2 or y1 >= y2:
            return 0.0

        region = image[y1:y2, x1:x2]
        if region.size == 0:
            return 0.0

        # Calculate color variance
        color_variance = region.var()
        return min(1.0, color_variance / (255.0 ** 2))  # Normalize
```

#### T2.1.2: Performance Optimization
```python
def score_splats_vectorized(self, splats: List[Gaussian], image: np.ndarray) -> None:
    """Optimized vectorized scoring for large splat counts."""
    # Pre-compute edge map for entire image
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edge_map = cv2.Laplacian(gray, cv2.CV_64F)
    edge_variance_map = ndimage.generic_filter(edge_map, np.var, size=3)

    image_area = image.shape[0] * image.shape[1]

    # Vectorized scoring
    for splat in splats:
        x, y = int(splat.x), int(splat.y)

        # Sample edge strength at splat location
        if 0 <= y < edge_variance_map.shape[0] and 0 <= x < edge_variance_map.shape[1]:
            edge_score = edge_variance_map[y, x] / 255.0
        else:
            edge_score = 0.0

        area_score = (splat.rx * splat.ry) / image_area

        # Simplified color score using splat's own color variance
        color_score = (splat.r + splat.g + splat.b) / (3 * 255.0)

        splat.score = (
            area_score * self.area_weight +
            edge_score * self.edge_weight +
            color_score * self.color_weight
        )
```

**✅ Acceptance Test:**
```python
# Test scoring produces reasonable results
scorer = ImportanceScorer()
scorer.score_splats(splats, image)
scores = [s.score for s in splats]
assert all(0 <= score <= 1 for score in scores)
assert len(set(scores)) > 1  # Should have variety
```

---

### T2.2: Layer Assignment System
**🎯 Goal:** Distribute splats across depth layers
**⏱️ Estimate:** 5 hours | **🔗 Dependencies:** T2.1

#### T2.2.1: Percentile-based Layering
```python
class LayerAssigner:
    def __init__(self, n_layers: int = 4):
        self.n_layers = n_layers

    def assign_layers(self, splats: List[Gaussian]) -> Dict[int, List[Gaussian]]:
        """Assign splats to depth layers based on scores."""
        if not splats:
            return {}

        # Sort by score (ascending, so highest scores go to front layers)
        sorted_splats = sorted(splats, key=lambda s: s.score)

        # Calculate layer boundaries using percentiles
        layer_size = len(sorted_splats) // self.n_layers
        remainder = len(sorted_splats) % self.n_layers

        layers = {}
        start_idx = 0

        for layer_idx in range(self.n_layers):
            # Distribute remainder across layers
            current_layer_size = layer_size + (1 if layer_idx < remainder else 0)
            end_idx = start_idx + current_layer_size

            # Get splats for this layer
            layer_splats = sorted_splats[start_idx:end_idx]

            # Assign depth values (0.2 for back layer, 1.0 for front layer)
            depth_value = 0.2 + (layer_idx / (self.n_layers - 1)) * 0.8

            for splat in layer_splats:
                splat.depth = depth_value

            layers[layer_idx] = layer_splats
            start_idx = end_idx

        return layers

    def get_layer_statistics(self, layers: Dict[int, List[Gaussian]]) -> Dict[int, dict]:
        """Get statistics for each layer."""
        stats = {}
        for layer_idx, layer_splats in layers.items():
            if layer_splats:
                scores = [s.score for s in layer_splats]
                stats[layer_idx] = {
                    'count': len(layer_splats),
                    'depth': layer_splats[0].depth,
                    'score_range': (min(scores), max(scores)),
                    'avg_score': sum(scores) / len(scores)
                }
        return stats
```

#### T2.2.2: Layer Balance Optimization
```python
def balance_layers(self, layers: Dict[int, List[Gaussian]],
                  min_per_layer: int = 10) -> Dict[int, List[Gaussian]]:
    """Ensure minimum splats per layer for visual balance."""
    total_splats = sum(len(layer) for layer in layers.values())

    if total_splats < min_per_layer * self.n_layers:
        # Not enough splats for minimum distribution
        return layers

    # Redistribute if any layer is too small
    balanced_layers = {}
    all_splats = []

    # Collect all splats
    for layer_splats in layers.values():
        all_splats.extend(layer_splats)

    # Re-sort by score
    all_splats.sort(key=lambda s: s.score)

    # Redistribute with minimum guarantees
    target_per_layer = max(min_per_layer, len(all_splats) // self.n_layers)

    for layer_idx in range(self.n_layers):
        start_idx = layer_idx * target_per_layer
        end_idx = min((layer_idx + 1) * target_per_layer, len(all_splats))

        layer_splats = all_splats[start_idx:end_idx]
        depth_value = 0.2 + (layer_idx / (self.n_layers - 1)) * 0.8

        for splat in layer_splats:
            splat.depth = depth_value

        balanced_layers[layer_idx] = layer_splats

    return balanced_layers
```

**✅ Acceptance Test:**
```python
# Test layer assignment distributes splats evenly
assigner = LayerAssigner(n_layers=4)
layers = assigner.assign_layers(splats)
assert len(layers) == 4
layer_counts = [len(layer) for layer in layers.values()]
assert max(layer_counts) - min(layer_counts) <= 1  # Balanced distribution
```

---

### T2.3: Quality Control & Filtering
**🎯 Goal:** Optimize splat collection for performance and quality
**⏱️ Estimate:** 4 hours | **🔗 Dependencies:** T2.2

#### T2.3.1: Smart Filtering System
```python
class QualityController:
    def __init__(self, target_count: int, k_multiplier: float = 2.5):
        self.target_count = target_count
        self.k_multiplier = k_multiplier

    def optimize_splats(self, splats: List[Gaussian]) -> List[Gaussian]:
        """Apply quality control and filtering to splat collection."""
        if not splats:
            return splats

        # Step 1: Remove micro-splats (too small to be visible)
        filtered = self._remove_micro_splats(splats)

        # Step 2: Adjust target count if needed
        if len(filtered) > self.target_count:
            filtered = self._reduce_to_target(filtered)

        # Step 3: Validate and clean
        filtered = self._validate_splats(filtered)

        # Step 4: Adjust alpha for better blending
        self._adjust_alpha_values(filtered)

        return filtered

    def _remove_micro_splats(self, splats: List[Gaussian],
                           min_area_ratio: float = 0.0001) -> List[Gaussian]:
        """Remove splats that are too small to be visually meaningful."""
        if not splats:
            return splats

        # Calculate area threshold based on largest splat
        areas = [s.area() for s in splats]
        max_area = max(areas)
        min_area_threshold = max_area * min_area_ratio

        # Filter out micro-splats
        filtered = [s for s in splats if s.area() >= min_area_threshold]

        return filtered

    def _reduce_to_target(self, splats: List[Gaussian]) -> List[Gaussian]:
        """Reduce splat count to target using smart selection."""
        if len(splats) <= self.target_count:
            return splats

        # Sort by composite score (area + importance)
        def composite_score(splat):
            return splat.score * 0.7 + (splat.area() / max(s.area() for s in splats)) * 0.3

        splats.sort(key=composite_score, reverse=True)
        return splats[:self.target_count]

    def _validate_splats(self, splats: List[Gaussian]) -> List[Gaussian]:
        """Remove invalid splats and fix edge cases."""
        valid_splats = []

        for splat in splats:
            try:
                # Check for valid parameters
                if (splat.rx > 0 and splat.ry > 0 and
                    0 <= splat.r <= 255 and 0 <= splat.g <= 255 and 0 <= splat.b <= 255 and
                    0 <= splat.a <= 1.0):

                    # Clamp extreme values
                    splat.rx = min(splat.rx, 500)  # Maximum radius
                    splat.ry = min(splat.ry, 500)
                    splat.a = max(0.1, min(1.0, splat.a))  # Ensure visibility

                    valid_splats.append(splat)
            except (ValueError, TypeError):
                continue  # Skip invalid splats

        return valid_splats

    def _adjust_alpha_values(self, splats: List[Gaussian]) -> None:
        """Adjust alpha values for better visual blending."""
        if not splats:
            return

        # Calculate alpha adjustment based on layer depth
        for splat in splats:
            # Background layers get lower alpha
            depth_factor = splat.depth
            base_alpha = splat.a

            # Adjust: background layers fade more, foreground stays opaque
            adjusted_alpha = base_alpha * (0.3 + 0.7 * depth_factor)
            splat.a = max(0.1, min(1.0, adjusted_alpha))
```

#### T2.3.2: Performance Metrics
```python
def get_quality_metrics(self, original_splats: List[Gaussian],
                       final_splats: List[Gaussian]) -> dict:
    """Calculate quality control metrics."""
    return {
        'original_count': len(original_splats),
        'final_count': len(final_splats),
        'reduction_ratio': len(final_splats) / len(original_splats) if original_splats else 0,
        'score_preservation': self._calculate_score_preservation(original_splats, final_splats),
        'size_distribution': self._analyze_size_distribution(final_splats)
    }

def _calculate_score_preservation(self, original: List[Gaussian],
                                 final: List[Gaussian]) -> float:
    """Calculate how well high-scoring splats were preserved."""
    if not original or not final:
        return 0.0

    original_top_scores = sorted([s.score for s in original], reverse=True)[:len(final)]
    final_scores = sorted([s.score for s in final], reverse=True)

    # Calculate correlation or overlap
    preserved_score = sum(min(o, f) for o, f in zip(original_top_scores, final_scores))
    total_score = sum(original_top_scores)

    return preserved_score / total_score if total_score > 0 else 0.0
```

**✅ Acceptance Test:**
```python
# Test quality control maintains splat count within target ±5%
controller = QualityController(target_count=1500)
optimized = controller.optimize_splats(raw_splats)
assert 1425 <= len(optimized) <= 1575  # Within 5% of target
assert all(s.rx > 0 and s.ry > 0 for s in optimized)  # All valid
```

---

## Phase 3: SVG Generation & Animation

### T3.1: SVG Structure Generation
**🎯 Goal:** Generate well-formed SVG with proper layer structure
**⏱️ Estimate:** 8 hours | **🔗 Dependencies:** T2.3

#### T3.1.1: SVG Template System
**Create `src/splat_this/core/svgout.py`:**
```python
from typing import Dict, List
import xml.etree.ElementTree as ET
from xml.dom import minidom

class SVGGenerator:
    def __init__(self, width: int, height: int, precision: int = 3):
        self.width = width
        self.height = height
        self.precision = precision

    def generate_svg(self, layers: Dict[int, List[Gaussian]],
                    parallax_strength: int = 40,
                    gaussian_mode: bool = False) -> str:
        """Generate complete SVG with layers and animation."""

        # Create root SVG element
        svg = ET.Element('svg')
        svg.set('viewBox', f'0 0 {self.width} {self.height}')
        svg.set('color-interpolation', 'linearRGB')
        svg.set('style', 'width: 100%; height: 100vh;')
        svg.set('xmlns', 'http://www.w3.org/2000/svg')

        # Add definitions if using gradient mode
        if gaussian_mode:
            defs = self._create_gradient_definitions()
            svg.append(defs)

        # Add layer groups
        for layer_idx in sorted(layers.keys()):
            layer_splats = layers[layer_idx]
            if layer_splats:
                depth = layer_splats[0].depth
                layer_group = self._create_layer_group(layer_splats, depth, gaussian_mode)
                svg.append(layer_group)

        # Add inline styles
        style_elem = self._create_styles()
        svg.append(style_elem)

        # Add inline JavaScript
        script_elem = self._create_animation_script(parallax_strength)
        svg.append(script_elem)

        # Convert to pretty XML string
        return self._format_svg(svg)

    def _create_gradient_definitions(self) -> ET.Element:
        """Create shared gradient definitions for gaussian mode."""
        defs = ET.Element('defs')

        # Shared radial gradient
        gradient = ET.Element('radialGradient')
        gradient.set('id', 'splat-gradient')
        gradient.set('cx', '50%')
        gradient.set('cy', '50%')
        gradient.set('r', '50%')

        # Gradient stops
        stop1 = ET.Element('stop')
        stop1.set('offset', '0%')
        stop1.set('stop-opacity', '1')

        stop2 = ET.Element('stop')
        stop2.set('offset', '100%')
        stop2.set('stop-opacity', '0')

        gradient.append(stop1)
        gradient.append(stop2)
        defs.append(gradient)

        return defs

    def _create_layer_group(self, splats: List[Gaussian], depth: float,
                           gaussian_mode: bool) -> ET.Element:
        """Create SVG group for a single depth layer."""
        group = ET.Element('g')
        group.set('class', 'layer')
        group.set('data-depth', str(round(depth, 3)))

        for splat in splats:
            ellipse = self._create_splat_element(splat, gaussian_mode)
            group.append(ellipse)

        return group

    def _create_splat_element(self, splat: Gaussian, gaussian_mode: bool) -> ET.Element:
        """Create SVG ellipse element for a single splat."""
        ellipse = ET.Element('ellipse')

        # Position and size
        ellipse.set('cx', str(round(splat.x, self.precision)))
        ellipse.set('cy', str(round(splat.y, self.precision)))
        ellipse.set('rx', str(round(splat.rx, self.precision)))
        ellipse.set('ry', str(round(splat.ry, self.precision)))

        # Rotation
        if abs(splat.theta) > 0.001:  # Only add rotation if significant
            transform = f'rotate({round(np.degrees(splat.theta), 1)} {round(splat.x, 1)} {round(splat.y, 1)})'
            ellipse.set('transform', transform)

        # Color and fill
        if gaussian_mode:
            ellipse.set('fill', 'url(#splat-gradient)')
            ellipse.set('style', f'color: rgba({splat.r}, {splat.g}, {splat.b}, {round(splat.a, 3)})')
        else:
            rgba = f'rgba({splat.r}, {splat.g}, {splat.b}, {round(splat.a, 3)})'
            ellipse.set('fill', rgba)

        return ellipse
```

#### T3.1.2: CSS Styles Generation
```python
def _create_styles(self) -> ET.Element:
    """Create inline CSS styles for animation."""
    style = ET.Element('style')
    style.text = '''
    .layer {
        transform-style: preserve-3d;
        transition: transform 0.1s ease-out;
    }

    @media (prefers-reduced-motion: reduce) {
        .layer {
            transition: none !important;
            transform: none !important;
        }
    }

    /* Gradient mode styling */
    ellipse[fill="url(#splat-gradient)"] {
        fill: currentColor;
    }
    '''
    return style
```

#### T3.1.3: XML Formatting
```python
def _format_svg(self, svg_element: ET.Element) -> str:
    """Format SVG as pretty-printed XML string."""
    # Convert to string
    rough_string = ET.tostring(svg_element, encoding='unicode')

    # Parse and pretty print
    reparsed = minidom.parseString(rough_string)
    pretty = reparsed.documentElement.toprettyxml(indent='  ')

    # Clean up extra whitespace and add XML declaration
    lines = [line for line in pretty.split('\n') if line.strip()]
    return '\n'.join(lines)

def save_svg(self, svg_content: str, output_path: Path) -> None:
    """Save SVG content to file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write(svg_content)
```

**✅ Acceptance Test:**
```python
# Test SVG generation produces valid XML
generator = SVGGenerator(1920, 1080)
svg_content = generator.generate_svg(layers)
assert '<?xml' in svg_content
assert 'viewBox="0 0 1920 1080"' in svg_content
assert all(f'data-depth=' in svg_content for _ in layers)
```

---

### T3.2: Animation System Implementation
**🎯 Goal:** Interactive parallax animation with accessibility
**⏱️ Estimate:** 10 hours | **🔗 Dependencies:** T3.1

#### T3.2.1: JavaScript Animation Engine
```python
def _create_animation_script(self, parallax_strength: int) -> ET.Element:
    """Create inline JavaScript for parallax interaction."""
    script = ET.Element('script')

    js_code = f'''
    (function() {{
        'use strict';

        // Configuration
        const PARALLAX_STRENGTH = {parallax_strength};
        const EASING_FACTOR = 0.12;

        // State
        let mouseX = 0.5;
        let mouseY = 0.5;
        let currentX = 0.5;
        let currentY = 0.5;
        let animationFrame = null;
        let reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

        // DOM references
        const layers = document.querySelectorAll('.layer');

        // Initialize
        function init() {{
            setupEventListeners();
            if (!reducedMotion) {{
                startAnimationLoop();
            }}
        }}

        function setupEventListeners() {{
            // Mouse tracking
            document.addEventListener('mousemove', handleMouseMove);
            document.addEventListener('mouseleave', handleMouseLeave);

            // Device orientation (mobile)
            if (window.DeviceOrientationEvent) {{
                window.addEventListener('deviceorientation', handleDeviceOrientation);
            }}

            // Touch tracking for mobile
            document.addEventListener('touchmove', handleTouchMove, {{ passive: true }});

            // Reduced motion preference changes
            window.matchMedia('(prefers-reduced-motion: reduce)').addEventListener('change', handleMotionPreferenceChange);
        }}

        function handleMouseMove(event) {{
            if (reducedMotion) return;

            mouseX = event.clientX / window.innerWidth;
            mouseY = event.clientY / window.innerHeight;
        }}

        function handleMouseLeave() {{
            if (reducedMotion) return;

            // Return to center when mouse leaves
            mouseX = 0.5;
            mouseY = 0.5;
        }}

        function handleTouchMove(event) {{
            if (reducedMotion || event.touches.length === 0) return;

            const touch = event.touches[0];
            mouseX = touch.clientX / window.innerWidth;
            mouseY = touch.clientY / window.innerHeight;
        }}

        function handleDeviceOrientation(event) {{
            if (reducedMotion) return;

            // Convert device orientation to mouse coordinates
            const gamma = event.gamma || 0; // Left-right tilt (-90 to 90)
            const beta = event.beta || 0;   // Front-back tilt (-180 to 180)

            // Normalize to 0-1 range with center at 0.5
            mouseX = Math.max(0, Math.min(1, 0.5 + gamma / 90));
            mouseY = Math.max(0, Math.min(1, 0.5 + beta / 180));
        }}

        function handleMotionPreferenceChange(e) {{
            reducedMotion = e.matches;
            if (reducedMotion) {{
                // Reset all transforms
                layers.forEach(layer => {{
                    layer.style.transform = '';
                }});
                if (animationFrame) {{
                    cancelAnimationFrame(animationFrame);
                    animationFrame = null;
                }}
            }} else {{
                startAnimationLoop();
            }}
        }}

        function startAnimationLoop() {{
            function animate() {{
                updateParallax();
                animationFrame = requestAnimationFrame(animate);
            }}
            animate();
        }}

        function updateParallax() {{
            if (reducedMotion) return;

            // Smooth interpolation
            currentX += (mouseX - currentX) * EASING_FACTOR;
            currentY += (mouseY - currentY) * EASING_FACTOR;

            // Apply transforms to each layer
            layers.forEach(layer => {{
                const depth = parseFloat(layer.dataset.depth) || 0.5;

                // Calculate offset based on depth and mouse position
                const offsetX = (currentX - 0.5) * PARALLAX_STRENGTH * depth;
                const offsetY = (currentY - 0.5) * PARALLAX_STRENGTH * depth;

                // Apply slight scale effect for pseudo-3D
                const scale = 1 + depth * 0.02;

                // Apply transform
                layer.style.transform = `translate(${{offsetX.toFixed(2)}}px, ${{offsetY.toFixed(2)}}px) scale(${{scale.toFixed(3)}})`;
            }});
        }}

        // Initialize when DOM is ready
        if (document.readyState === 'loading') {{
            document.addEventListener('DOMContentLoaded', init);
        }} else {{
            init();
        }}
    }})();
    '''

    script.text = js_code
    return script
```

#### T3.2.2: Performance Optimization
```python
def _create_optimized_animation_script(self, parallax_strength: int,
                                     max_fps: int = 60) -> ET.Element:
    """Create performance-optimized animation script."""
    script = ET.Element('script')

    js_code = f'''
    (function() {{
        'use strict';

        const PARALLAX_STRENGTH = {parallax_strength};
        const EASING_FACTOR = 0.12;
        const MAX_FPS = {max_fps};
        const FRAME_INTERVAL = 1000 / MAX_FPS;

        let mouseX = 0.5, mouseY = 0.5;
        let currentX = 0.5, currentY = 0.5;
        let lastFrameTime = 0;
        let animationFrame = null;
        let reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

        const layers = Array.from(document.querySelectorAll('.layer'));
        const layerData = layers.map(layer => ({{
            element: layer,
            depth: parseFloat(layer.dataset.depth) || 0.5
        }}));

        function init() {{
            setupEventListeners();
            if (!reducedMotion) {{
                startAnimationLoop();
            }}
        }}

        function updateParallax(timestamp) {{
            if (reducedMotion) return;

            // Frame rate limiting
            if (timestamp - lastFrameTime < FRAME_INTERVAL) {{
                animationFrame = requestAnimationFrame(updateParallax);
                return;
            }}
            lastFrameTime = timestamp;

            // Smooth interpolation
            currentX += (mouseX - currentX) * EASING_FACTOR;
            currentY += (mouseY - currentY) * EASING_FACTOR;

            // Batch DOM updates
            layerData.forEach(layerInfo => {{
                const offsetX = (currentX - 0.5) * PARALLAX_STRENGTH * layerInfo.depth;
                const offsetY = (currentY - 0.5) * PARALLAX_STRENGTH * layerInfo.depth;
                const scale = 1 + layerInfo.depth * 0.02;

                layerInfo.element.style.transform =
                    `translate(${{offsetX.toFixed(2)}}px, ${{offsetY.toFixed(2)}}px) scale(${{scale.toFixed(3)}})`;
            }});

            animationFrame = requestAnimationFrame(updateParallax);
        }}

        // ... rest of event handlers ...

        init();
    }})();
    '''

    script.text = js_code
    return script
```

**✅ Acceptance Test:**
```python
# Test animation script is properly embedded
svg_content = generator.generate_svg(layers, parallax_strength=40)
assert 'PARALLAX_STRENGTH = 40' in svg_content
assert 'prefers-reduced-motion' in svg_content
assert 'deviceorientation' in svg_content
```

---

### T3.3: Gradient Mode Implementation
**🎯 Goal:** Higher visual fidelity with gradient rendering
**⏱️ Estimate:** 6 hours | **🔗 Dependencies:** T3.1

#### T3.3.1: Gradient System
```python
class GradientSVGGenerator(SVGGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gradient_cache = {}

    def _create_gradient_definitions(self) -> ET.Element:
        """Enhanced gradient definitions with multiple variations."""
        defs = ET.Element('defs')

        # Base gradient for most splats
        base_gradient = self._create_radial_gradient('splat-gradient',
                                                   stops=[(0, 1.0), (70, 0.8), (100, 0.0)])
        defs.append(base_gradient)

        # Sharp gradient for high-importance splats
        sharp_gradient = self._create_radial_gradient('splat-gradient-sharp',
                                                    stops=[(0, 1.0), (50, 0.9), (100, 0.0)])
        defs.append(sharp_gradient)

        # Soft gradient for background splats
        soft_gradient = self._create_radial_gradient('splat-gradient-soft',
                                                   stops=[(0, 0.8), (80, 0.4), (100, 0.0)])
        defs.append(soft_gradient)

        return defs

    def _create_radial_gradient(self, gradient_id: str,
                               stops: List[Tuple[int, float]]) -> ET.Element:
        """Create radial gradient with specified stops."""
        gradient = ET.Element('radialGradient')
        gradient.set('id', gradient_id)
        gradient.set('cx', '50%')
        gradient.set('cy', '50%')
        gradient.set('r', '50%')

        for offset, opacity in stops:
            stop = ET.Element('stop')
            stop.set('offset', f'{offset}%')
            stop.set('stop-opacity', str(opacity))
            gradient.append(stop)

        return gradient

    def _create_splat_element(self, splat: Gaussian, gaussian_mode: bool) -> ET.Element:
        """Enhanced splat element with gradient selection."""
        ellipse = super()._create_splat_element(splat, gaussian_mode)

        if gaussian_mode:
            # Select gradient based on splat importance
            gradient_id = self._select_gradient_for_splat(splat)
            ellipse.set('fill', f'url(#{gradient_id})')

            # Set color using CSS custom property for better caching
            color_rgba = f'rgba({splat.r}, {splat.g}, {splat.b}, {round(splat.a, 3)})'
            ellipse.set('style', f'color: {color_rgba}')

        return ellipse

    def _select_gradient_for_splat(self, splat: Gaussian) -> str:
        """Select appropriate gradient based on splat characteristics."""
        if splat.score > 0.8:
            return 'splat-gradient-sharp'
        elif splat.depth < 0.4:  # Background layers
            return 'splat-gradient-soft'
        else:
            return 'splat-gradient'
```

#### T3.3.2: File Size Optimization
```python
def optimize_gradient_output(self, svg_content: str) -> str:
    """Optimize SVG for smaller file size while preserving gradients."""
    # Remove redundant whitespace but preserve readability
    lines = svg_content.split('\n')
    optimized_lines = []

    for line in lines:
        stripped = line.strip()
        if stripped:  # Skip empty lines
            # Compress style attributes
            if 'style=' in stripped:
                stripped = self._compress_style_attribute(stripped)
            optimized_lines.append(stripped)

    return '\n'.join(optimized_lines)

def _compress_style_attribute(self, line: str) -> str:
    """Compress CSS in style attributes."""
    import re

    # Remove extra spaces in style attributes
    style_pattern = r'style="([^"]*)"'

    def compress_style(match):
        style_content = match.group(1)
        # Remove extra spaces around colons and semicolons
        compressed = re.sub(r'\s*:\s*', ':', style_content)
        compressed = re.sub(r'\s*;\s*', ';', compressed)
        return f'style="{compressed}"'

    return re.sub(style_pattern, compress_style, line)
```

#### T3.3.3: Fallback System
```python
def generate_hybrid_svg(self, layers: Dict[int, List[Gaussian]],
                       gradient_threshold: float = 0.6) -> str:
    """Generate SVG with gradients for important splats, solid for others."""

    # Separate splats by importance
    gradient_splats = {}
    solid_splats = {}

    for layer_idx, layer_splats in layers.items():
        gradient_layer = []
        solid_layer = []

        for splat in layer_splats:
            if splat.score >= gradient_threshold:
                gradient_layer.append(splat)
            else:
                solid_layer.append(splat)

        if gradient_layer:
            gradient_splats[layer_idx] = gradient_layer
        if solid_layer:
            solid_splats[layer_idx] = solid_layer

    # Generate combined SVG
    return self._generate_combined_svg(gradient_splats, solid_splats)
```

**✅ Acceptance Test:**
```python
# Test gradient mode produces larger but higher quality output
solid_svg = generator.generate_svg(layers, gaussian_mode=False)
gradient_svg = generator.generate_svg(layers, gaussian_mode=True)
assert len(gradient_svg) > len(solid_svg)  # Larger file
assert 'radialGradient' in gradient_svg
assert 'url(#splat-gradient)' in gradient_svg
```

---

## Phase 4: CLI & Integration

### T4.1: Complete CLI Implementation
**🎯 Goal:** Production-ready command-line interface
**⏱️ Estimate:** 6 hours | **🔗 Dependencies:** T3.3

#### T4.1.1: Enhanced CLI with Progress
**Update `src/splat_this/cli.py`:**
```python
import click
import sys
from pathlib import Path
from typing import Optional
import time

from .utils.image import load_image, validate_image_dimensions
from .core.extract import SplatExtractor
from .core.layering import ImportanceScorer, LayerAssigner, QualityController
from .core.svgout import SVGGenerator, GradientSVGGenerator

class ProgressBar:
    def __init__(self, total_steps: int, description: str = "Processing"):
        self.total_steps = total_steps
        self.current_step = 0
        self.description = description
        self.start_time = time.time()

    def update(self, step_name: str) -> None:
        self.current_step += 1
        percentage = (self.current_step / self.total_steps) * 100
        elapsed = time.time() - self.start_time

        # Simple progress bar
        bar_length = 30
        filled_length = int(bar_length * self.current_step // self.total_steps)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)

        click.echo(f'\r{self.description}: [{bar}] {percentage:.1f}% - {step_name}', nl=False)

        if self.current_step == self.total_steps:
            click.echo(f' ✓ Complete ({elapsed:.1f}s)')

@click.command()
@click.argument('input_file', type=click.Path(exists=True, path_type=Path))
@click.option('--frame', default=0, help='GIF frame number (default: 0)')
@click.option('--splats', default=1500, help='Target splat count (default: 1500)',
              type=click.IntRange(100, 10000))
@click.option('--layers', default=4, help='Depth layers (default: 4)',
              type=click.IntRange(2, 8))
@click.option('--k', default=2.5, help='Splat size multiplier (default: 2.5)',
              type=click.FloatRange(1.0, 5.0))
@click.option('--alpha', default=0.65, help='Base alpha (default: 0.65)',
              type=click.FloatRange(0.1, 1.0))
@click.option('--parallax-strength', default=40, help='Parallax strength (default: 40)',
              type=click.IntRange(0, 200))
@click.option('--interactive-top', default=0, help='Interactive splats (default: 0)',
              type=click.IntRange(0, 5000))
@click.option('--gaussian', is_flag=True, help='Enable gradient mode')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
@click.option('-o', '--output', required=True, type=click.Path(path_type=Path),
              help='Output SVG file path')
def main(input_file: Path, output: Path, frame: int, splats: int, layers: int,
         k: float, alpha: float, parallax_strength: int, interactive_top: int,
         gaussian: bool, verbose: bool):
    """Convert image to parallax-animated SVG splats.

    Examples:
        splatlify photo.jpg -o parallax.svg
        splatlify animation.gif --frame 5 --splats 2000 -o output.svg
        splatlify image.png --gaussian --layers 6 -o high-quality.svg
    """

    # Validate output path
    if output.exists():
        if not click.confirm(f"Output file {output} exists. Overwrite?"):
            click.echo("Aborted.")
            return

    # Ensure output directory exists
    output.parent.mkdir(parents=True, exist_ok=True)

    # Initialize progress tracking
    total_steps = 6  # Load, Extract, Score, Layer, Generate, Save
    progress = ProgressBar(total_steps, "Converting")

    try:
        # Step 1: Load image
        progress.update("Loading image")
        if verbose:
            click.echo(f"\nLoading image: {input_file}")

        image, (height, width) = load_image(input_file, frame)
        validate_image_dimensions(image)

        if verbose:
            click.echo(f"Image dimensions: {width}x{height}")
            click.echo(f"Image size: {image.nbytes / (1024*1024):.1f} MB")

        # Step 2: Extract splats
        progress.update("Extracting splats")
        extractor = SplatExtractor(k=k, base_alpha=alpha)
        raw_splats = extractor.extract_splats(image, splats)

        if verbose:
            click.echo(f"Extracted {len(raw_splats)} raw splats")

        # Step 3: Score splats
        progress.update("Scoring importance")
        scorer = ImportanceScorer()
        scorer.score_splats(raw_splats, image)

        # Step 4: Assign layers and filter
        progress.update("Assigning layers")
        controller = QualityController(target_count=splats, k_multiplier=k)
        final_splats = controller.optimize_splats(raw_splats)

        assigner = LayerAssigner(n_layers=layers)
        layer_data = assigner.assign_layers(final_splats)

        if verbose:
            click.echo(f"Final splat count: {len(final_splats)}")
            stats = assigner.get_layer_statistics(layer_data)
            for layer_idx, layer_stats in stats.items():
                click.echo(f"Layer {layer_idx}: {layer_stats['count']} splats (depth: {layer_stats['depth']:.2f})")

        # Step 5: Generate SVG
        progress.update("Generating SVG")
        if gaussian:
            generator = GradientSVGGenerator(width, height)
        else:
            generator = SVGGenerator(width, height)

        svg_content = generator.generate_svg(
            layer_data,
            parallax_strength=parallax_strength,
            gaussian_mode=gaussian
        )

        # Step 6: Save output
        progress.update("Saving file")
        generator.save_svg(svg_content, output)

        # Final statistics
        file_size = output.stat().st_size
        file_size_mb = file_size / (1024 * 1024)

        click.echo(f"\n✅ Successfully created {output}")
        click.echo(f"📊 Final statistics:")
        click.echo(f"   Splats: {len(final_splats)}")
        click.echo(f"   Layers: {len(layer_data)}")
        click.echo(f"   File size: {file_size_mb:.2f} MB")

        if verbose:
            click.echo(f"   Mode: {'Gradient' if gaussian else 'Solid'}")
            click.echo(f"   Parallax strength: {parallax_strength}px")
            click.echo(f"   Interactive splats: {interactive_top}")

    except Exception as e:
        progress.update("Error")
        click.echo(f"\n❌ Error: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
```

#### T4.1.2: Parameter Validation
```python
def validate_parameters(input_file: Path, **params) -> None:
    """Validate CLI parameters and provide helpful messages."""

    # Check input file format
    supported_formats = {'.png', '.jpg', '.jpeg', '.gif'}
    if input_file.suffix.lower() not in supported_formats:
        raise click.BadParameter(
            f"Unsupported format '{input_file.suffix}'. "
            f"Supported formats: {', '.join(supported_formats)}"
        )

    # Validate parameter combinations
    if params['interactive_top'] > params['splats']:
        raise click.BadParameter(
            f"Interactive splats ({params['interactive_top']}) cannot exceed "
            f"total splats ({params['splats']})"
        )

    # Memory estimation
    estimated_memory = estimate_memory_usage(params['splats'], params['layers'])
    if estimated_memory > 2048:  # 2GB limit
        click.echo(f"⚠️  Warning: Estimated memory usage: {estimated_memory:.0f}MB", err=True)
        if not click.confirm("Continue anyway?"):
            raise click.Abort()

def estimate_memory_usage(splats: int, layers: int) -> float:
    """Estimate memory usage in MB."""
    # Rough estimation based on splat count and layers
    base_memory = 100  # Base overhead
    splat_memory = splats * 0.1  # ~0.1MB per 1000 splats
    layer_memory = layers * 10   # Layer processing overhead
    return base_memory + splat_memory + layer_memory
```

**✅ Acceptance Test:**
```bash
# Test CLI with various parameter combinations
splatlify test.jpg -o output.svg --verbose
splatlify test.gif --frame 2 --splats 2000 --gaussian -o animated.svg
splatlify test.png --layers 6 --parallax-strength 60 -o high-quality.svg
```

---

### T4.2: Performance Optimization
**🎯 Goal:** Meet performance benchmarks
**⏱️ Estimate:** 8 hours | **🔗 Dependencies:** T4.1

#### T4.2.1: Profiling and Optimization
**Create `src/splat_this/utils/profiler.py`:**
```python
import time
import psutil
import functools
from typing import Dict, Any, Callable

class PerformanceProfiler:
    def __init__(self):
        self.metrics = {}
        self.start_memory = psutil.virtual_memory().used

    def profile_function(self, name: str):
        """Decorator to profile function execution."""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                start_memory = psutil.virtual_memory().used

                result = func(*args, **kwargs)

                end_time = time.time()
                end_memory = psutil.virtual_memory().used

                self.metrics[name] = {
                    'duration': end_time - start_time,
                    'memory_delta': end_memory - start_memory,
                    'peak_memory': psutil.virtual_memory().used
                }

                return result
            return wrapper
        return decorator

    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        total_time = sum(m['duration'] for m in self.metrics.values())
        peak_memory = max(m['peak_memory'] for m in self.metrics.values()) if self.metrics else 0

        return {
            'total_time': total_time,
            'peak_memory_mb': peak_memory / (1024 * 1024),
            'by_function': self.metrics
        }

# Optimized SLIC implementation
class OptimizedSplatExtractor(SplatExtractor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.profiler = PerformanceProfiler()

    @PerformanceProfiler().profile_function("slic_segmentation")
    def extract_splats(self, image: np.ndarray, n_splats: int) -> List[Gaussian]:
        """Optimized splat extraction with performance monitoring."""

        # Adaptive oversampling based on image size
        image_area = image.shape[0] * image.shape[1]
        if image_area > 2_000_000:  # 2MP+
            oversample_factor = 1.3
        else:
            oversample_factor = 1.5

        n_segments = int(n_splats * oversample_factor)

        # Optimized SLIC parameters for speed
        segments = slic(
            image,
            n_segments=n_segments,
            compactness=10,
            sigma=0,  # Disable pre-smoothing for speed
            multichannel=True,
            convert2lab=False,  # Work in RGB for speed
            max_iter=10  # Limit iterations
        )

        return self._extract_splats_vectorized(image, segments, n_splats)

    def _extract_splats_vectorized(self, image: np.ndarray, segments: np.ndarray,
                                  target_count: int) -> List[Gaussian]:
        """Vectorized splat extraction for better performance."""
        splats = []
        unique_segments = np.unique(segments)[1:]  # Skip background

        # Pre-allocate arrays for vectorized operations
        centroids = np.zeros((len(unique_segments), 2))
        areas = np.zeros(len(unique_segments))
        colors = np.zeros((len(unique_segments), 3))

        # Vectorized centroid and color calculation
        for i, region_id in enumerate(unique_segments):
            mask = segments == region_id
            coords = np.column_stack(np.where(mask))

            if len(coords) < 3:
                continue

            centroids[i] = coords.mean(axis=0)
            areas[i] = len(coords)
            colors[i] = image[mask].mean(axis=0)

        # Convert to Gaussian splats
        for i, region_id in enumerate(unique_segments):
            if areas[i] == 0:
                continue

            mask = segments == region_id
            coords = np.column_stack(np.where(mask))

            if len(coords) < 3:
                continue

            # Fast covariance computation
            rx, ry, theta = self._fast_covariance_ellipse(coords)

            splat = Gaussian(
                x=centroids[i, 1], y=centroids[i, 0],  # Note: x/y swap for image coords
                rx=rx * self.k, ry=ry * self.k, theta=theta,
                r=int(colors[i, 0]), g=int(colors[i, 1]), b=int(colors[i, 2]),
                a=self.base_alpha,
                score=areas[i]  # Initial score based on area
            )

            splats.append(splat)

        return self._filter_splats(splats, target_count)

    def _fast_covariance_ellipse(self, coords: np.ndarray) -> Tuple[float, float, float]:
        """Optimized covariance calculation."""
        # Simple axis-aligned bounding box for speed
        y_range = coords[:, 0].max() - coords[:, 0].min()
        x_range = coords[:, 1].max() - coords[:, 1].min()

        rx = max(x_range / 4, 1.0)  # Simple approximation
        ry = max(y_range / 4, 1.0)
        theta = 0.0  # Skip rotation calculation for speed

        return rx, ry, theta
```

#### T4.2.2: Memory Management
```python
class MemoryEfficientProcessor:
    def __init__(self, max_memory_mb: int = 1024):
        self.max_memory_mb = max_memory_mb

    def process_large_image(self, image_path: Path, **params) -> List[Gaussian]:
        """Process large images with memory management."""

        # Check if image needs downscaling
        with Image.open(image_path) as img:
            width, height = img.size
            total_pixels = width * height

        # Downscale if too large
        if total_pixels > 8_000_000:  # 8MP threshold
            scale_factor = np.sqrt(8_000_000 / total_pixels)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)

            click.echo(f"Downscaling large image: {width}x{height} → {new_width}x{new_height}")

            with Image.open(image_path) as img:
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                image = np.array(img)
        else:
            image, _ = load_image(image_path)

        return self._process_with_monitoring(image, **params)

    def _process_with_monitoring(self, image: np.ndarray, **params) -> List[Gaussian]:
        """Process with memory monitoring."""
        process = psutil.Process()

        def check_memory():
            memory_mb = process.memory_info().rss / (1024 * 1024)
            if memory_mb > self.max_memory_mb:
                raise MemoryError(f"Memory usage ({memory_mb:.0f}MB) exceeds limit ({self.max_memory_mb}MB)")

        # Extract with memory checks
        check_memory()
        extractor = OptimizedSplatExtractor()
        splats = extractor.extract_splats(image, params['splats'])

        check_memory()
        # Continue with processing...

        return splats
```

**✅ Acceptance Test:**
```python
# Test performance meets benchmarks
start_time = time.time()
splats = process_image("test_1920x1080.jpg", splats=1500)
duration = time.time() - start_time
assert duration < 30  # Under 30 seconds
assert len(splats) >= 1425  # Within 5% of target
```

---

### T4.3: Testing & Quality Assurance
**🎯 Goal:** Comprehensive test coverage and validation
**⏱️ Estimate:** 10 hours | **🔗 Dependencies:** T4.2

#### T4.3.1: Unit Test Suite
**Create comprehensive test files:**

```python
# tests/unit/test_extract.py
import pytest
import numpy as np
from splat_this.core.extract import SplatExtractor, Gaussian

class TestSplatExtractor:
    def test_extract_basic_functionality(self):
        """Test basic splat extraction."""
        extractor = SplatExtractor()
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        splats = extractor.extract_splats(image, n_splats=50)

        assert len(splats) <= 50
        assert all(isinstance(s, Gaussian) for s in splats)
        assert all(s.rx > 0 and s.ry > 0 for s in splats)

    def test_covariance_calculation(self):
        """Test mathematical correctness of covariance."""
        extractor = SplatExtractor()

        # Test with known circle pattern
        coords = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]])
        rx, ry, theta = extractor._compute_covariance_ellipse(coords)

        # Should be roughly circular
        assert abs(rx - ry) < 0.5

    def test_edge_cases(self):
        """Test edge cases and error handling."""
        extractor = SplatExtractor()

        # Single color image
        solid_image = np.full((50, 50, 3), 128, dtype=np.uint8)
        splats = extractor.extract_splats(solid_image, n_splats=10)
        assert len(splats) >= 1  # Should produce at least some splats

        # Very small image
        tiny_image = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
        splats = extractor.extract_splats(tiny_image, n_splats=5)
        assert len(splats) <= 5

# tests/unit/test_layering.py
import pytest
from splat_this.core.layering import ImportanceScorer, LayerAssigner, QualityController

class TestLayerAssignment:
    def test_layer_distribution(self):
        """Test even distribution across layers."""
        # Create test splats with varying scores
        splats = [Gaussian(x=i, y=i, rx=1, ry=1, theta=0, r=128, g=128, b=128, a=0.5, score=i/100)
                 for i in range(100)]

        assigner = LayerAssigner(n_layers=4)
        layers = assigner.assign_layers(splats)

        assert len(layers) == 4
        layer_counts = [len(layer) for layer in layers.values()]
        assert max(layer_counts) - min(layer_counts) <= 1  # Balanced

    def test_depth_values(self):
        """Test correct depth value assignment."""
        splats = [Gaussian(x=0, y=0, rx=1, ry=1, theta=0, r=128, g=128, b=128, a=0.5, score=0.5)]

        assigner = LayerAssigner(n_layers=4)
        layers = assigner.assign_layers(splats)

        # Check depth values are in expected range
        for layer_splats in layers.values():
            for splat in layer_splats:
                assert 0.2 <= splat.depth <= 1.0

# tests/unit/test_svgout.py
import pytest
from xml.etree import ElementTree as ET
from splat_this.core.svgout import SVGGenerator

class TestSVGGeneration:
    def test_svg_structure(self):
        """Test generated SVG has correct structure."""
        generator = SVGGenerator(100, 100)

        # Create test layers
        test_splat = Gaussian(x=50, y=50, rx=10, ry=5, theta=0, r=255, g=0, b=0, a=0.8, depth=0.5)
        layers = {0: [test_splat]}

        svg_content = generator.generate_svg(layers)

        # Parse and validate XML
        root = ET.fromstring(svg_content)
        assert root.tag == 'svg'
        assert 'viewBox' in root.attrib

        # Check for required elements
        assert root.find('.//g[@class="layer"]') is not None
        assert root.find('.//ellipse') is not None
        assert root.find('.//style') is not None
        assert root.find('.//script') is not None

    def test_gradient_mode(self):
        """Test gradient mode produces different output."""
        generator = SVGGenerator(100, 100)
        test_splat = Gaussian(x=50, y=50, rx=10, ry=5, theta=0, r=255, g=0, b=0, a=0.8, depth=0.5)
        layers = {0: [test_splat]}

        solid_svg = generator.generate_svg(layers, gaussian_mode=False)
        gradient_svg = generator.generate_svg(layers, gaussian_mode=True)

        assert 'radialGradient' in gradient_svg
        assert 'radialGradient' not in solid_svg
        assert len(gradient_svg) > len(solid_svg)
```

#### T4.3.2: Integration Tests
```python
# tests/integration/test_end_to_end.py
import pytest
from pathlib import Path
import tempfile
from splat_this.cli import main
from click.testing import CliRunner

class TestEndToEnd:
    def test_complete_pipeline_png(self, sample_png_image):
        """Test complete pipeline with PNG input."""
        runner = CliRunner()

        with tempfile.NamedTemporaryFile(suffix='.svg', delete=False) as tmp:
            result = runner.invoke(main, [
                str(sample_png_image),
                '-o', tmp.name,
                '--splats', '100',  # Small count for speed
                '--layers', '3'
            ])

            assert result.exit_code == 0
            assert Path(tmp.name).exists()

            # Validate output
            svg_content = Path(tmp.name).read_text()
            assert 'viewBox' in svg_content
            assert 'data-depth' in svg_content

    def test_parameter_validation(self):
        """Test CLI parameter validation."""
        runner = CliRunner()

        # Test invalid splat count
        result = runner.invoke(main, [
            'test.jpg', '-o', 'out.svg', '--splats', '50'  # Below minimum
        ])
        assert result.exit_code != 0

    def test_performance_benchmark(self, sample_large_image):
        """Test processing time meets requirements."""
        import time

        runner = CliRunner()
        start_time = time.time()

        with tempfile.NamedTemporaryFile(suffix='.svg', delete=False) as tmp:
            result = runner.invoke(main, [
                str(sample_large_image),
                '-o', tmp.name,
                '--splats', '1500'
            ])

            duration = time.time() - start_time
            assert result.exit_code == 0
            assert duration < 30  # Under 30 seconds for 1920x1080
```

#### T4.3.3: Browser Compatibility Tests
```python
# tests/compatibility/test_browser_compatibility.py
import pytest
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time

class TestBrowserCompatibility:
    @pytest.fixture
    def chrome_driver(self):
        """Set up Chrome driver for testing."""
        options = Options()
        options.add_argument('--headless')
        driver = webdriver.Chrome(options=options)
        yield driver
        driver.quit()

    def test_svg_loads_in_chrome(self, chrome_driver, sample_svg):
        """Test SVG loads and animates in Chrome."""
        # Load SVG file
        chrome_driver.get(f'file://{sample_svg}')

        # Wait for load
        time.sleep(2)

        # Check for SVG elements
        svg_elements = chrome_driver.find_elements('tag name', 'svg')
        assert len(svg_elements) > 0

        # Test animation works (basic check)
        layers = chrome_driver.find_elements('css selector', '.layer')
        assert len(layers) > 0

    def test_performance_in_browser(self, chrome_driver, sample_svg):
        """Test animation performance in browser."""
        chrome_driver.get(f'file://{sample_svg}')

        # Execute JavaScript to measure FPS
        fps_script = """
        return new Promise(resolve => {
            let frames = 0;
            let start = performance.now();

            function countFrame() {
                frames++;
                if (frames < 60) {
                    requestAnimationFrame(countFrame);
                } else {
                    let end = performance.now();
                    let fps = frames / ((end - start) / 1000);
                    resolve(fps);
                }
            }
            requestAnimationFrame(countFrame);
        });
        """

        fps = chrome_driver.execute_async_script(fps_script)
        assert fps > 45  # Minimum acceptable FPS
```

**✅ Acceptance Test:**
```bash
# Run complete test suite
pytest tests/ -v --cov=src/splat_this --cov-report=html
# Coverage should be >80%
# All tests should pass
# Performance benchmarks should meet requirements
```

---

### T4.4: Documentation & Examples
**🎯 Goal:** Complete user documentation and examples
**⏱️ Estimate:** 4 hours | **🔗 Dependencies:** T4.3

#### T4.4.1: Comprehensive README
**Create `README.md`:**
```markdown
# SplatThis 🎨

Convert images into self-contained parallax-animated SVG splats.

## Quick Start

```bash
pip install splat-this
splatlify your-image.jpg -o parallax.svg
```

## Features

✨ **Universal Compatibility** - Works in browsers, PowerPoint, email clients
🚀 **Self-Contained** - No external dependencies or files
📱 **Mobile Ready** - Touch and gyroscope parallax support
⚡ **Fast Processing** - Typically under 30 seconds for 1080p images
🎯 **High Quality** - SSIM fidelity within 3% of original
♿ **Accessible** - Respects `prefers-reduced-motion`

## Installation

```bash
pip install splat-this
```

Or from source:
```bash
git clone https://github.com/username/splat-this
cd splat-this
pip install -e .
```

## Basic Usage

### Simple Conversion
```bash
splatlify photo.jpg -o output.svg
```

### High Quality Mode
```bash
splatlify photo.jpg --gaussian --splats 3000 --layers 6 -o high-quality.svg
```

### GIF Animation Frame
```bash
splatlify animation.gif --frame 5 -o frame-5.svg
```

## CLI Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--splats` | 1500 | Target number of splats (100-10000) |
| `--layers` | 4 | Depth layers for parallax (2-8) |
| `--k` | 2.5 | Splat size multiplier (1.0-5.0) |
| `--alpha` | 0.65 | Base transparency (0.1-1.0) |
| `--parallax-strength` | 40 | Parallax intensity in pixels (0-200) |
| `--gaussian` | off | Enable gradient mode for higher quality |
| `--frame` | 0 | GIF frame to extract (for animated GIFs) |
| `--verbose` | off | Show detailed processing information |

## Examples

### Web Background
Perfect for hero sections and landing pages:
```bash
splatlify hero-image.jpg --parallax-strength 60 --layers 5 -o hero-bg.svg
```

### PowerPoint Slide
Optimized for presentations:
```bash
splatlify slide-bg.png --splats 1000 --layers 3 -o slide.svg
```

### High Fidelity Art
Maximum quality for artistic images:
```bash
splatlify artwork.jpg --gaussian --splats 5000 --k 3.0 -o art.svg
```

## Technical Details

### How It Works
1. **SLIC Segmentation** - Divides image into perceptually meaningful regions
2. **Gaussian Extraction** - Converts regions to elliptical splats with covariance analysis
3. **Importance Scoring** - Ranks splats by visual significance (area + edge strength)
4. **Layer Assignment** - Distributes splats across depth layers for parallax effect
5. **SVG Generation** - Creates self-contained SVG with inline CSS/JS animation

### Performance
- **Processing**: <30s for 1920×1080 images on 2019+ hardware
- **Memory**: <1GB peak usage during processing
- **Output Size**: ~1-2MB for 1500 splats (solid mode), ~2-3MB (gradient mode)
- **Animation**: >60fps on desktop, >45fps on mobile

### Compatibility
- **Browsers**: Chrome 80+, Firefox 75+, Safari 13+, Edge 80+
- **Applications**: PowerPoint 2019+, modern email clients
- **Platforms**: Windows 10+, macOS 10.15+, Linux (Ubuntu 20.04+)

## Troubleshooting

### Large Images
For images >4K, the tool automatically downscales:
```bash
splatlify huge-image.jpg -o output.svg --verbose
# Shows: "Downscaling large image: 7680x4320 → 2844x1601"
```

### Memory Issues
Reduce splat count and layers:
```bash
splatlify image.jpg --splats 800 --layers 3 -o output.svg
```

### Poor Quality
Increase splats and enable gradient mode:
```bash
splatlify image.jpg --gaussian --splats 3000 --k 3.5 -o output.svg
```

### Animation Too Subtle
Increase parallax strength:
```bash
splatlify image.jpg --parallax-strength 80 -o output.svg
```

## Development

### Setup
```bash
git clone https://github.com/username/splat-this
cd splat-this
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -e .[dev]
```

### Testing
```bash
pytest tests/ -v --cov=src/splat_this
```

### Code Quality
```bash
black src/ tests/
mypy src/
flake8 src/
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make changes and add tests
4. Run quality checks: `black`, `mypy`, `pytest`
5. Submit a pull request

## License

MIT License - see [LICENSE](LICENSE) file.

## Acknowledgments

- Uses [SLIC superpixel segmentation](https://scikit-image.org/) from scikit-image
- Inspired by [Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) techniques
- Built with [Click](https://click.palletsprojects.com/) CLI framework
```

#### T4.4.2: API Documentation
**Create `docs/api.md`:**
```markdown
# API Documentation

## Core Modules

### SplatExtractor
Extract Gaussian splats from images using SLIC segmentation.

```python
from splat_this.core.extract import SplatExtractor

extractor = SplatExtractor(k=2.5, base_alpha=0.65)
splats = extractor.extract_splats(image, n_splats=1500)
```

### LayerAssigner
Assign splats to depth layers for parallax effect.

```python
from splat_this.core.layering import LayerAssigner

assigner = LayerAssigner(n_layers=4)
layers = assigner.assign_layers(splats)
```

### SVGGenerator
Generate animated SVG from layered splats.

```python
from splat_this.core.svgout import SVGGenerator

generator = SVGGenerator(width=1920, height=1080)
svg_content = generator.generate_svg(layers, parallax_strength=40)
```

## Data Structures

### Gaussian
Represents a single splat with position, size, rotation, and color.

```python
@dataclass
class Gaussian:
    x: float        # X coordinate
    y: float        # Y coordinate
    rx: float       # X-axis radius
    ry: float       # Y-axis radius
    theta: float    # Rotation angle (radians)
    r: int         # Red (0-255)
    g: int         # Green (0-255)
    b: int         # Blue (0-255)
    a: float       # Alpha (0.0-1.0)
    score: float   # Importance score
    depth: float   # Depth layer (0.2-1.0)
```

## Usage Examples

### Basic Processing
```python
from splat_this.utils.image import load_image
from splat_this.core.extract import SplatExtractor
from splat_this.core.layering import LayerAssigner
from splat_this.core.svgout import SVGGenerator

# Load image
image, (height, width) = load_image("photo.jpg")

# Extract splats
extractor = SplatExtractor()
splats = extractor.extract_splats(image, n_splats=1500)

# Assign layers
assigner = LayerAssigner(n_layers=4)
layers = assigner.assign_layers(splats)

# Generate SVG
generator = SVGGenerator(width, height)
svg_content = generator.generate_svg(layers)

# Save
with open("output.svg", "w") as f:
    f.write(svg_content)
```

### Custom Processing Pipeline
```python
from splat_this.core.layering import ImportanceScorer, QualityController

# Score splats by importance
scorer = ImportanceScorer(area_weight=0.4, edge_weight=0.6)
scorer.score_splats(splats, image)

# Apply quality control
controller = QualityController(target_count=1500)
filtered_splats = controller.optimize_splats(splats)

# Custom layer assignment
assigner = LayerAssigner(n_layers=6)
layers = assigner.assign_layers(filtered_splats)
```
```

#### T4.4.3: Usage Examples and Tutorials
**Create `examples/` directory with sample files:**

```python
# examples/basic_usage.py
"""Basic usage example for SplatThis."""

from pathlib import Path
from splat_this.utils.image import load_image
from splat_this.core.extract import SplatExtractor
from splat_this.core.layering import LayerAssigner
from splat_this.core.svgout import SVGGenerator

def basic_conversion(input_path: str, output_path: str):
    """Convert image to parallax SVG with default settings."""

    # Load image
    print(f"Loading {input_path}...")
    image, (height, width) = load_image(Path(input_path))
    print(f"Image size: {width}×{height}")

    # Extract splats
    print("Extracting splats...")
    extractor = SplatExtractor(k=2.5, base_alpha=0.65)
    splats = extractor.extract_splats(image, n_splats=1500)
    print(f"Extracted {len(splats)} splats")

    # Assign to layers
    print("Assigning depth layers...")
    assigner = LayerAssigner(n_layers=4)
    layers = assigner.assign_layers(splats)

    # Generate SVG
    print("Generating SVG...")
    generator = SVGGenerator(width, height)
    svg_content = generator.generate_svg(layers, parallax_strength=40)

    # Save
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(svg_content)

    print(f"✓ Saved to {output_path}")

if __name__ == "__main__":
    basic_conversion("sample.jpg", "output.svg")
```

```python
# examples/advanced_customization.py
"""Advanced customization example."""

def high_quality_conversion(input_path: str, output_path: str):
    """High-quality conversion with gradient mode."""

    from splat_this.core.svgout import GradientSVGGenerator
    from splat_this.core.layering import ImportanceScorer, QualityController

    # Load and process
    image, (height, width) = load_image(Path(input_path))

    # High splat count extraction
    extractor = SplatExtractor(k=3.0, base_alpha=0.7)
    splats = extractor.extract_splats(image, n_splats=3000)

    # Advanced scoring
    scorer = ImportanceScorer(
        area_weight=0.2,
        edge_weight=0.6,
        color_weight=0.2
    )
    scorer.score_splats(splats, image)

    # Quality optimization
    controller = QualityController(target_count=2500)
    final_splats = controller.optimize_splats(splats)

    # More layers for smoother parallax
    assigner = LayerAssigner(n_layers=6)
    layers = assigner.assign_layers(final_splats)

    # Gradient mode for higher fidelity
    generator = GradientSVGGenerator(width, height)
    svg_content = generator.generate_svg(
        layers,
        parallax_strength=60,
        gaussian_mode=True
    )

    # Save optimized output
    svg_content = generator.optimize_gradient_output(svg_content)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(svg_content)

    print(f"✓ High-quality SVG saved to {output_path}")

if __name__ == "__main__":
    high_quality_conversion("artwork.jpg", "high-quality.svg")
```

**✅ Acceptance Test:**
```bash
# Test documentation examples work
cd examples/
python basic_usage.py
python advanced_customization.py

# Verify outputs are valid
ls -la *.svg
# Files should exist and be >100KB
```

---

## Summary

This comprehensive implementation plan provides:

### ✅ **Completed Structure**
- **16 major tasks** across 4 phases
- **60+ subtasks** with detailed implementation steps
- **Clear dependencies** and effort estimates
- **Acceptance criteria** for each deliverable

### 🎯 **Key Deliverables**
- **Production CLI** with all specified parameters
- **High-performance processing** (<30s for 1080p)
- **Universal SVG output** (browsers, PowerPoint, email)
- **Comprehensive testing** (>80% coverage)
- **Complete documentation** with examples

### 📊 **Timeline Summary**
- **Week 1:** Foundation & Core Pipeline (T1.1-T1.4)
- **Week 2:** Depth & Layering System (T2.1-T2.3)
- **Week 3:** SVG Generation & Animation (T3.1-T3.4)
- **Week 4:** CLI & Integration (T4.1-T4.4)

### 🚀 **Ready for Implementation**
Each task includes:
- Step-by-step implementation code
- Performance optimization strategies
- Error handling and edge cases
- Unit and integration tests
- Documentation examples

The tasks are now ready for immediate development start with clear success criteria and measurable deliverables.

---

## Future Enhancement: Image-GS Integration

### T5.1: NYU-ICL Image-GS Integration (Future Phase)
**🎯 Goal:** Add support for pre-trained 2D Gaussian Splatting models
**⏱️ Estimate:** 12 hours | **🔗 Dependencies:** Core pipeline complete

#### T5.1.1: Image-GS Model Loading
**Repository:** https://github.com/NYU-ICL/image-gs

```python
# Future integration code structure
class ImageGSExtractor(SplatExtractor):
    def __init__(self, checkpoint_path: Optional[Path] = None):
        super().__init__()
        self.model = None
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)

    def load_checkpoint(self, checkpoint_path: Path):
        """Load pre-trained Image-GS model."""
        import torch
        # Load NYU-ICL model checkpoint
        self.model = torch.load(checkpoint_path)

    def extract_splats_from_model(self, image: np.ndarray) -> List[Gaussian]:
        """Extract splats using trained Image-GS model."""
        # Convert to model format and run inference
        # Return higher-quality Gaussian splats
        pass
```

#### T5.1.2: CLI Integration
```bash
# Future CLI enhancement
splatlify image.jpg --checkpoint models/image-gs-model.pth -o output.svg
```

**Implementation Priority:** Post-MVP, based on user demand for higher quality output.

**Benefits:**
- Significantly higher visual fidelity
- Better preservation of image details
- More accurate splat positioning and sizing

**Requirements:**
- PyTorch dependency
- GPU acceleration recommended
- Pre-trained model files (~50-100MB each)

<function_calls>
<invoke name="TodoWrite">
<parameter name="todos">[{"content": "Create actionable tasks file structure", "status": "completed", "activeForm": "Creating actionable tasks file structure"}, {"content": "Break down Phase 1 tasks into detailed steps", "status": "completed", "activeForm": "Breaking down Phase 1 tasks into detailed steps"}, {"content": "Break down Phase 2 tasks into detailed steps", "status": "completed", "activeForm": "Breaking down Phase 2 tasks into detailed steps"}, {"content": "Break down Phase 3 tasks into detailed steps", "status": "completed", "activeForm": "Breaking down Phase 3 tasks into detailed steps"}, {"content": "Break down Phase 4 tasks into detailed steps", "status": "completed", "activeForm": "Breaking down Phase 4 tasks into detailed steps"}, {"content": "Add GitHub Issues integration", "status": "completed", "activeForm": "Adding GitHub Issues integration"}]