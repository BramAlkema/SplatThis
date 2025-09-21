# Adaptive Gaussian Splats Specification

## Overview

This specification defines an adaptive Gaussian splat system inspired by NYU's Image-GS research that addresses the critical limitation of uniform splat sizing in current implementations. Instead of generating splats of similar sizes, this system creates variable-sized splats based on image content, saliency, and reconstruction error.

## Problem Statement

### Current Limitations
- **Uniform Splat Sizes**: All splats are approximately the same size (~25 pixels radius)
- **Poor Content Adaptation**: No consideration of image complexity or detail levels
- **Inefficient Representation**: Large uniform splats for fine details, small splats for smooth areas
- **Suboptimal Compression**: Missed opportunities for better compression ratios

### Target Improvements
- **Content-Adaptive Sizing**: Splats sized based on local image complexity
- **Saliency-Guided Placement**: Important features get priority and appropriate sizing
- **Progressive Refinement**: Iterative optimization of splat parameters
- **Variable Detail Levels**: Small splats for fine details, large splats for smooth areas

## Technical Architecture

### 1. Saliency Analysis Engine

#### Components
```
SaliencyAnalyzer
├── Edge Detection (Sobel)
├── Local Variance Analysis
├── Gradient Magnitude
└── Multi-Scale Feature Detection
```

#### Saliency Map Generation
```python
saliency = w1 * edge_saliency + w2 * variance_saliency + w3 * gradient_saliency
```

**Parameters:**
- `edge_weight`: 0.4 (emphasis on edges)
- `variance_weight`: 0.3 (local texture importance)
- `gradient_weight`: 0.3 (directional features)

#### Output
- **Saliency Map**: 2D array (H×W) with values [0,1]
- **Peak Detection**: Local maxima indicating high-importance regions
- **Multi-Scale Representation**: Different detail levels across image

### 2. Adaptive Initialization Strategies

#### A. Saliency-Based Initialization
```
1. Compute saliency map
2. Detect saliency peaks (local maxima)
3. Create large splats (1.5x scale) for high-saliency regions (70% of budget)
4. Create small splats (0.7x scale) for detail areas (30% of budget)
5. Ensure non-overlapping placement
```

#### B. Gradient-Based Initialization
```
1. Compute gradient magnitude
2. Find gradient peaks (edges, corners)
3. Bias splat placement toward high-gradient areas
4. Size splats based on local gradient strength
```

#### C. Random Initialization (Baseline)
```
1. Random segmentation with felzenszwalb
2. Uniform splat sizing
3. Used for comparison/fallback
```

### 3. Progressive Refinement System

#### Iterative Optimization Loop
```
for iteration in range(refinement_iterations):
    1. Compute reconstruction error map
    2. Identify high-error regions (>80th percentile)
    3. Refine splats overlapping high-error areas
    4. Update splat parameters based on local content
    5. Evaluate convergence criteria
```

#### Refinement Operations
- **Scale Adjustment**: Increase/decrease splat size based on error
- **Position Refinement**: Fine-tune splat centers
- **Alpha Modulation**: Adjust transparency based on importance
- **Anisotropy Optimization**: Adjust ellipse aspect ratio

### 4. Adaptive Scale Optimization

#### Content-Adaptive Scaling
```python
content_scale = 1.0 + local_variance * 0.001 + abs(local_gradient) * 0.1
new_scale = base_scale * content_scale * saliency_boost
```

#### Scale Constraints
- **Minimum Scale**: 0.5 pixels (fine detail preservation)
- **Maximum Scale**: 8.0 pixels (efficient smooth area coverage)
- **Anisotropy Ratio**: 1:4 max (prevents degenerate ellipses)

#### Dynamic Range Distribution
- **Small Splats** (<2.0px): Fine details, textures, edges
- **Medium Splats** (2.0-5.0px): Regular content, moderate complexity
- **Large Splats** (≥5.0px): Smooth areas, backgrounds, gradients

## Implementation Specification

### 1. Core Classes

#### AdaptiveSplatConfig
```python
@dataclass
class AdaptiveSplatConfig:
    # Initialization
    init_strategy: str = "saliency"  # "saliency", "gradient", "random"
    random_ratio: float = 0.3

    # Scale adaptation
    min_scale: float = 0.5
    max_scale: float = 8.0
    scale_variance_threshold: float = 0.1

    # Progressive optimization
    refinement_iterations: int = 5
    error_threshold: float = 0.01

    # Saliency weights
    edge_weight: float = 0.4
    variance_weight: float = 0.3
    gradient_weight: float = 0.3
```

#### AdaptiveSplatExtractor
```python
class AdaptiveSplatExtractor:
    def extract_adaptive_splats(
        self,
        image: np.ndarray,
        n_splats: int = 1500,
        verbose: bool = False
    ) -> List[Gaussian]:
        """Main extraction pipeline with adaptive sizing."""
```

### 2. Algorithm Pipeline

#### Stage 1: Content Analysis
```
Input: RGB Image (H×W×3)
↓
Saliency Analysis:
├── Edge Detection (Sobel filter)
├── Local Variance (5×5 windows)
├── Gradient Magnitude
└── Peak Detection
↓
Output: Saliency Map (H×W), Peak Coordinates
```

#### Stage 2: Adaptive Initialization
```
Input: Image + Saliency Map
↓
Strategy Selection:
├── Saliency-Based (70% high-importance, 30% detail)
├── Gradient-Based (edge-focused placement)
└── Random (uniform distribution)
↓
Splat Generation:
├── Variable sizing (0.5x to 3.0x scale multiplier)
├── Anisotropic ellipses
└── Content-aware alpha values
↓
Output: Initial Splat Set
```

#### Stage 3: Progressive Refinement
```
Input: Initial Splats + Image
↓
For each iteration:
├── Render current splat configuration
├── Compute reconstruction error map
├── Identify high-error regions (>80th percentile)
├── Refine affected splats:
│   ├── Scale adjustment (±50% based on error)
│   ├── Position fine-tuning (±2 pixels)
│   └── Alpha modulation (±0.1)
└── Check convergence (error < threshold)
↓
Output: Refined Splat Set
```

#### Stage 4: Final Optimization
```
Input: Refined Splats + Image
↓
Local Content Analysis:
├── Extract local region around each splat
├── Compute local variance and gradient
├── Apply content-adaptive scaling
└── Enforce scale constraints [0.5, 8.0]
↓
Output: Optimized Adaptive Splats
```

### 3. Data Structures

#### Enhanced Gaussian Splat
```python
@dataclass
class AdaptiveGaussian(Gaussian):
    # Inherited: x, y, rx, ry, theta, r, g, b, a, score, depth

    # Adaptive extensions
    content_complexity: float = 0.0    # Local content measure
    refinement_count: int = 0          # Times refined
    error_contribution: float = 0.0    # Reconstruction error
    saliency_score: float = 0.0       # Local saliency value
    scale_factor: float = 1.0         # Content-adaptive scale multiplier
```

#### Refinement State
```python
@dataclass
class RefinementState:
    iteration: int
    total_error: float
    error_map: np.ndarray
    convergence_rate: float
    refinement_count: int
```

### 4. Performance Specifications

#### Computational Complexity
- **Saliency Analysis**: O(H×W) - single pass operations
- **Initialization**: O(N) where N = number of segments
- **Refinement**: O(I×M) where I = iterations, M = splats in error regions
- **Final Optimization**: O(S) where S = total splats

#### Memory Requirements
- **Saliency Maps**: 3×(H×W) float32 arrays
- **Error Maps**: (H×W) float32 per iteration
- **Splat Storage**: S×AdaptiveGaussian structures
- **Peak Memory**: ~4×(H×W×4) + S×200 bytes

#### Quality Metrics
- **Scale Diversity**: Standard deviation of splat scales ≥ 1.5
- **Content Adaptation**: Correlation between splat size and local complexity ≥ 0.6
- **Reconstruction Quality**: PSNR improvement ≥ 3dB vs uniform splats
- **Compression Efficiency**: 10-30% better compression ratio

### 5. Configuration Profiles

#### High Quality Profile
```python
config = AdaptiveSplatConfig(
    init_strategy="saliency",
    refinement_iterations=8,
    min_scale=0.3,
    max_scale=12.0,
    error_threshold=0.005
)
# Use case: Photography, detailed images
# Performance: ~3x slower, +25% quality
```

#### Balanced Profile
```python
config = AdaptiveSplatConfig(
    init_strategy="saliency",
    refinement_iterations=5,
    min_scale=0.5,
    max_scale=8.0,
    error_threshold=0.01
)
# Use case: General web graphics
# Performance: ~1.5x slower, +15% quality
```

#### Fast Profile
```python
config = AdaptiveSplatConfig(
    init_strategy="gradient",
    refinement_iterations=3,
    min_scale=0.8,
    max_scale=6.0,
    error_threshold=0.02
)
# Use case: Real-time applications
# Performance: ~1.2x slower, +8% quality
```

## Integration Specification

### 1. CLI Integration
```bash
# Enable adaptive splats
splatlify image.jpg --adaptive --quality high -o adaptive.svg

# Configure adaptive parameters
splatlify image.jpg --adaptive \
    --init-strategy saliency \
    --refinement-iterations 6 \
    --min-scale 0.4 --max-scale 10.0 \
    -o custom_adaptive.svg

# Compare with uniform splats
splatlify image.jpg --adaptive --compare-uniform -o comparison.html
```

### 2. Python API Integration
```python
from splat_this.core.adaptive_extract import AdaptiveSplatExtractor, AdaptiveSplatConfig

# Basic usage
extractor = AdaptiveSplatExtractor()
splats = extractor.extract_adaptive_splats(image, n_splats=2000, verbose=True)

# Custom configuration
config = AdaptiveSplatConfig(
    init_strategy="saliency",
    refinement_iterations=8,
    min_scale=0.3,
    max_scale=15.0
)
extractor = AdaptiveSplatExtractor(config)
splats = extractor.extract_adaptive_splats(image, n_splats=1500)

# Drop-in replacement for existing SplatExtractor
from splat_this.core.extract import SplatExtractor as UniformExtractor
uniform_splats = UniformExtractor().extract_splats(image)
adaptive_splats = AdaptiveSplatExtractor().extract_adaptive_splats(image)
```

### 3. Backward Compatibility
- **Existing API**: All current SplatExtractor methods remain unchanged
- **Optional Feature**: Adaptive sizing is opt-in via new classes/parameters
- **Fallback Mode**: Graceful degradation to uniform sizing on errors
- **Configuration**: Existing pipelines work without modification

## Validation & Testing

### 1. Unit Tests
```python
def test_saliency_map_generation():
    """Test saliency map has correct properties."""

def test_adaptive_initialization_strategies():
    """Test all three initialization modes."""

def test_progressive_refinement():
    """Test iterative improvement of splats."""

def test_scale_diversity():
    """Ensure variable splat sizes are generated."""
```

### 2. Integration Tests
```python
def test_end_to_end_adaptive_pipeline():
    """Test complete adaptive extraction pipeline."""

def test_performance_comparison():
    """Compare adaptive vs uniform splat quality."""

def test_cli_adaptive_integration():
    """Test CLI with adaptive parameters."""
```

### 3. Quality Benchmarks
- **Image Types**: Photography, graphics, screenshots, art
- **Metrics**: PSNR, SSIM, compression ratio, visual quality scores
- **Performance**: Processing time, memory usage, scalability
- **Comparison**: Adaptive vs uniform splats across test suite

### 4. Visual Validation
- **A/B Testing**: Side-by-side uniform vs adaptive results
- **Scale Distribution Plots**: Histogram of splat sizes
- **Saliency Overlay**: Visualization of saliency-guided placement
- **Error Map Evolution**: Progressive refinement visualization

## Expected Outcomes

### 1. Quality Improvements
- **25-40% better compression** for complex images
- **15-25% improved visual quality** (PSNR/SSIM)
- **Better detail preservation** in high-frequency areas
- **More efficient smooth area representation**

### 2. Adaptive Behavior Examples

#### Fine Detail Image (Text, Circuits)
- **70% small splats** (0.5-2.0px) for text/lines
- **25% medium splats** (2.0-4.0px) for moderate detail
- **5% large splats** (4.0+px) for background areas

#### Natural Photography
- **40% small splats** for texture/detail areas
- **45% medium splats** for general content
- **15% large splats** for sky/smooth areas

#### Graphics/UI Screenshots
- **50% small splats** for UI elements/text
- **30% medium splats** for icons/graphics
- **20% large splats** for solid color areas

### 3. Performance Characteristics
- **Processing Time**: 1.2-3x slower than uniform (profile dependent)
- **Memory Usage**: +50% during processing (temporary maps)
- **Output Size**: 10-30% smaller SVG files
- **Quality/Speed Trade-off**: Configurable via profiles

This adaptive system transforms SplatThis from a uniform splat generator into a content-aware, intelligently-sized representation that rivals modern image compression while maintaining the benefits of vector scalability and interactivity.