# PNG→SVG Gaussian Splatting Pipeline Specification

**Feature**: PNG to SVG conversion using anisotropic 2D Gaussian splats
**Priority**: High
**Status**: New Feature
**Effort**: 3-4 weeks
**Goal**: Practical tool for PNG→usable SVG with Gaussian splats à la Image-GS

## Overview

Create a lean, production-focused pipeline that converts PNG images into vector SVG representations using anisotropic 2D Gaussian "splats". This approach prioritizes practical usability over research compliance, targeting real-world PNG→SVG conversion needs.

## Problem Statement

Current PNG→SVG conversion tools produce either:
- Traced paths that lose fine details and gradients
- Raster embeddings that aren't truly vectorized
- Complex mesh representations that are hard to edit

Our solution uses content-adaptive Gaussian splats to create:
- True vector representation with smooth gradients
- Zoomable, parallax-friendly output
- Deterministic size control (N splats → N SVG elements)
- Clean level-of-detail by truncating tail splats

## Technical Requirements

### 1. Pipeline Architecture

```
PNG Input → Pre-process → Content-Adaptive Seeding → Progressive Fitting → SVG Export
     ↓            ↓                   ↓                       ↓              ↓
  Load PNG   Linearize RGB     Gradient-based placement   Stage refinement   Ellipse transforms
```

### 2. Core Data Structures

#### Splat Representation
```python
@dataclass
class GaussianSplat:
    mu: np.ndarray      # mean μᵢ = (x, y) in image coords
    sigma: np.ndarray   # covariance Σᵢ ∈ ℝ²ˣ² (positive-definite)
    color: np.ndarray   # color cᵢ = (r, g, b)
    alpha: float        # opacity αᵢ ∈ [0,1]
```

#### Module Structure
```
png2svg_gs/
├── io.py                # load_png(), save_svg()
├── features.py          # gradients, seeding, tiling/BSP
├── splat.py             # Splat dataclass, covariance utils
├── renderer.py          # differentiable canvas (PyTorch)
├── optimise.py          # fit_loop(), refine_on_residuals()
├── export_svg.py        # covariance→ellipse transform
└── main.py              # CLI interface
```

### 3. Implementation Phases

#### Phase 1: Core Infrastructure (Week 1)
- **Input Processing**: PNG loading, linearization, normalization
- **Splat Data Structure**: Basic Gaussian splat with covariance utilities
- **Simple Renderer**: Basic rasterization for validation

#### Phase 2: Content-Adaptive Seeding (Week 2)
- **Gradient Analysis**: Sobel/Scharr gradient magnitude computation
- **Adaptive Placement**: Dense seeding in high-gradient regions
- **Poisson Disk**: Coverage for low-texture flat regions
- **Spatial Organization**: Grid/BSP for fast refinement access

#### Phase 3: Progressive Fitting (Week 3)
- **Differentiable Renderer**: PyTorch-based alpha compositing
- **Loss Function**: `λ₁·MSE + λ₂·TV(α) + λ₃·area_penalty`
- **Stage-Based Optimization**: Multi-stage refinement with Adam
- **Error-Driven Refinement**: Spawn new splats in high-residual areas

#### Phase 4: SVG Export & Polish (Week 4)
- **Covariance Transform**: Eigendecomposition → SVG ellipse transforms
- **Post-Processing**: Prune transparent splats, merge overlaps
- **SVG Generation**: Optimized ellipse elements with proper ordering
- **CLI Interface**: Complete command-line tool

## Detailed Technical Specifications

### 1. Input Processing
```python
def load_png(path: str) -> np.ndarray:
    """Load PNG → float32 RGB(A), linearize sRGB, normalize [0,1]"""
    # - Read PNG with PIL/OpenCV
    # - Convert sRGB → linear RGB (gamma correction)
    # - Normalize to [0,1] range
    # - Optional: resize to working resolution
```

### 2. Content-Adaptive Seeding
```python
def init_seeds(img: np.ndarray) -> List[Tuple[float, float]]:
    """Generate content-adaptive seed positions"""
    # - Compute gradient magnitude (Sobel/Scharr)
    # - Create probability map: higher gradients → more seeds
    # - Poisson disk sampling in low-gradient regions
    # - Return normalized (x,y) positions
```

### 3. Progressive Fitting Algorithm
```python
def fit_stage(img: np.ndarray, splats: List[GaussianSplat],
              iters: int) -> List[GaussianSplat]:
    """Single optimization stage"""
    # Setup:
    # - Convert splats to PyTorch tensors
    # - Initialize Adam optimizer

    # Loss function:
    # L = λ₁·MSE(render, target) + λ₂·TV(α) + λ₃·area_penalty

    # Optimization loop:
    # for i in range(iters):
    #     rendered = render(splats, H, W)
    #     loss = compute_loss(rendered, img, splats)
    #     loss.backward()
    #     optimizer.step()

    return optimized_splats
```

### 4. SVG Export
```python
def cov_to_svg_transform(mu: np.ndarray, sigma: np.ndarray,
                        k: float = 2.5) -> str:
    """Convert covariance matrix to SVG transform"""
    # Eigendecomposition: Σ = R·diag(σ₁², σ₂²)·Rᵀ
    vals, vecs = np.linalg.eigh(sigma)

    # Scale by k-sigma (k≈2-3 for 95-99.7% mass)
    scale = np.diag([k * np.sqrt(vals[0]), k * np.sqrt(vals[1])])
    R = vecs  # rotation matrix

    # Compose transform: T = Translate(μ) · R · Scale
    transform_matrix = compose_affine(translate=mu, linear=R @ scale)
    return matrix_to_svg_transform(transform_matrix)

def export_svg(splats: List[GaussianSplat], width: int, height: int) -> str:
    """Generate SVG with ellipse elements"""
    # Sort by area (largest first) for proper alpha compositing
    # Generate <ellipse> elements with transforms
    # Return complete SVG document
```

## API Design

### Command Line Interface
```bash
# Basic usage
png2svg input.png output.svg

# With parameters
png2svg input.png output.svg \
    --max-splats 5000 \
    --k-sigma 2.5 \
    --stages 4 \
    --working-size 512

# With quality control
png2svg input.png output.svg \
    --target-psnr 35 \
    --max-iterations 1000
```

### Python API
```python
from png2svg_gs import PNG2SVGConverter

converter = PNG2SVGConverter(
    max_splats=5000,
    k_sigma=2.5,
    stages=4
)

# Convert single image
svg_content = converter.convert("input.png")
with open("output.svg", "w") as f:
    f.write(svg_content)

# Batch processing
converter.convert_batch("input_dir/", "output_dir/")
```

## Performance Targets

### Quality Metrics
- **PSNR**: >30 dB for typical natural images
- **SSIM**: >0.9 structural similarity
- **Visual Quality**: Smooth gradients, sharp edges preserved
- **File Size**: 5-20× smaller than equivalent PNG

### Performance Characteristics
- **Processing Time**: 30-120 seconds for 512×512 image
- **Memory Usage**: <2GB RAM for 1024×1024 images
- **Splat Budget**: 1,000-20,000 splats depending on complexity
- **SVG Size**: 50-500KB for typical images

### Tunables
```python
# Key parameters that affect quality vs performance
MAX_SPLATS = 5000        # Size vs fidelity knob
K_SIGMA = 2.5           # Ellipse size (2-3 range)
STAGES = [500, 300, 200, 100]  # Iterations per stage
TILE_SIZE = 32          # Refinement granularity
LOSS_WEIGHTS = {        # Loss function balance
    'mse': 1.0,
    'tv': 0.1,
    'area': 0.01
}
```

## Success Criteria

### Functional Requirements
- [ ] PNG input loading with proper sRGB handling
- [ ] Content-adaptive seed placement
- [ ] Progressive stage-based optimization
- [ ] SVG export with proper ellipse transforms
- [ ] CLI interface with parameter control

### Quality Requirements
- [ ] Visually faithful reproduction of input image
- [ ] Smooth gradients without banding artifacts
- [ ] Sharp edge preservation in high-contrast areas
- [ ] Reasonable file sizes (competitive with PNG)

### Technical Requirements
- [ ] Pure Python/PyTorch implementation (no CUDA required)
- [ ] Memory efficient for large images (tiling support)
- [ ] Deterministic results (same input → same output)
- [ ] Clean, readable SVG output

## Integration with Existing Codebase

### Reusable Components
- **AdaptiveGaussian2D**: Core splat representation ✅
- **Structure Tensor Analysis**: Enhanced gradient computation ✅
- **Tile Renderer**: Spatial organization framework ✅
- **SVG Export**: Ellipse transform generation ✅

### New Components Needed
- **PNG Input Pipeline**: sRGB linearization, normalization
- **Stage-Based Optimizer**: Simplified progressive fitting
- **PyTorch Renderer**: Differentiable alpha compositing
- **CLI Interface**: User-friendly command-line tool

## Risk Assessment

### Technical Risks
- **Convergence Issues**: Covariance matrices becoming ill-conditioned
- **Memory Usage**: Large images requiring tiling strategies
- **Performance**: Optimization time for high-quality results

### Mitigation Strategies
- **Covariance Clamping**: Ensure positive-definite matrices
- **Adaptive Tiling**: Process large images in overlapping tiles
- **Progressive Quality**: Allow early termination with good-enough results

## Future Extensions

### Short-term (1-2 months)
- **Batch Processing**: Directory-level conversion
- **Quality Presets**: Low/Medium/High quality profiles
- **Animation Support**: Temporal coherence for video frames

### Long-term (3-6 months)
- **Interactive Editor**: GUI for manual splat adjustment
- **Style Transfer**: Artistic style preservation during conversion
- **Web Service**: Cloud-based conversion API

## Dependencies

### Core Dependencies
- **PyTorch**: Differentiable rendering and optimization
- **NumPy**: Numerical computations and array operations
- **PIL/OpenCV**: Image I/O and processing
- **scikit-image**: Image analysis utilities

### Optional Dependencies
- **Click**: Command-line interface framework
- **tqdm**: Progress bars for long operations
- **matplotlib**: Visualization and debugging

## Validation Strategy

### Unit Tests
- Splat mathematical operations (covariance, transforms)
- PNG loading and preprocessing correctness
- SVG export format validation

### Integration Tests
- End-to-end pipeline on reference images
- Quality metric validation against known targets
- Performance benchmarking on various image sizes

### User Acceptance Testing
- Artist/designer feedback on SVG usability
- Comparison with existing PNG→SVG tools
- Real-world usage scenarios and edge cases

---

**Next Steps**: Begin with Phase 1 implementation, focusing on core infrastructure and basic PNG→splat→SVG pipeline before adding progressive optimization complexity.