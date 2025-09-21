# Adaptive Gaussian Splatting Implementation Spec

**Created:** 2025-01-21
**Version:** 1.0
**Status:** Ready for Implementation
**Priority:** High
**Related:** @../adaptive-splats/tasks.md

## Executive Summary

Implement research-grade adaptive Gaussian splatting based on Image-GS methodology to replace the current naive uniform splat placement approach. This addresses critical quality issues including rasterization at high density, poor edge handling, and inefficient splat allocation.

## Problem Statement

### Current Limitations
1. **Uniform placement** - SLIC segmentation ignores image content structure
2. **Fixed circular splats** - Cannot follow edges or adapt to local geometry
3. **Rasterization at high density** - 4000+ uniform splats look pixelated
4. **Gap vs blur trade-off** - Cannot achieve both sharp edges AND smooth coverage
5. **No optimization** - Static parameters lead to suboptimal quality

### Root Cause Analysis
The fundamental issue is treating all image regions identically. Edges require different splat characteristics than smooth areas, but our current approach forces uniform circular splats everywhere.

## Solution Overview

Implement **content-adaptive, anisotropic Gaussian splatting** with:

1. **Gradient-guided initialization** - Place splats based on image structure
2. **Progressive optimization** - Start coarse, refine where needed
3. **Anisotropic ellipses** - Elongated splats following edge orientations
4. **Top-K tile rendering** - Prevent muddy overdraw at high density
5. **Error-driven refinement** - Add splats where reconstruction fails

## Technical Architecture

### Core Components

#### 1. AdaptiveGaussian2D Class
```python
@dataclass
class AdaptiveGaussian2D:
    mu: np.ndarray[2]        # Position in [0,1]² normalized coordinates
    inv_s: np.ndarray[2]     # Inverse scales (1/sx, 1/sy) for optimization
    theta: float             # Rotation angle in [0, π)
    color: np.ndarray[C]     # RGB or multi-channel color

    @property
    def covariance_matrix(self) -> np.ndarray:
        """Compute full 2x2 covariance matrix from parameters."""

    @property
    def aspect_ratio(self) -> float:
        """Compute anisotropy ratio from inverse scales."""
```

#### 2. AdaptiveSplatExtractor Class
```python
class AdaptiveSplatExtractor:
    def __init__(self, max_splats: int = 2000, base_scale_px: float = 5.0):
        self.max_splats = max_splats
        self.base_scale_px = base_scale_px

    def extract_adaptive_splats(self, image: np.ndarray) -> List[AdaptiveGaussian2D]:
        """Main extraction pipeline with progressive optimization."""

    def gradient_guided_initialization(self, image: np.ndarray) -> List[AdaptiveGaussian2D]:
        """Content-adaptive initial placement."""

    def progressive_optimization(self, gaussians: List[AdaptiveGaussian2D],
                               target: np.ndarray) -> List[AdaptiveGaussian2D]:
        """Iterative refinement with error-guided splat addition."""
```

#### 3. TileRenderer Class
```python
class TileRenderer:
    def __init__(self, tile_size: int = 16, top_k: int = 8):
        self.tile_size = tile_size
        self.top_k = top_k

    def render(self, gaussians: List[AdaptiveGaussian2D],
               image_size: Tuple[int, int]) -> np.ndarray:
        """Tile-based top-K rendering for clean blending."""
```

### Algorithm Flow

#### Phase 1: Content-Adaptive Initialization
1. **Gradient Analysis**
   ```python
   grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
   prob_map = (1 - p_uniform) * normalize(grad_magnitude) + p_uniform * uniform_dist
   initial_positions = sample_from_probability(prob_map, n_initial)
   ```

2. **Structure Tensor Computation**
   ```python
   structure_tensor = compute_structure_tensor(image, sigma=1.0)
   eigenvals, eigenvecs = np.linalg.eigh(structure_tensor)
   anisotropy = eigenvals[1] / eigenvals[0]
   orientation = np.arctan2(eigenvecs[1, 1], eigenvecs[0, 1])
   ```

3. **Initial Splat Creation**
   ```python
   for position in initial_positions:
       splat = AdaptiveGaussian2D(
           mu=normalize_position(position),
           inv_s=[1/base_scale, 1/base_scale],  # Start isotropic
           theta=local_orientation[position],
           color=sample_color(image, position)
       )
   ```

#### Phase 2: Progressive Optimization
1. **Reconstruction Error Computation**
   ```python
   rendered = tile_render(gaussians, image_size)
   error_map = np.abs(rendered - target).mean(axis=2)
   high_error_regions = error_map > error_threshold
   ```

2. **Gradient-Based Parameter Updates**
   ```python
   for gaussian in gaussians:
       grad_mu = compute_position_gradient(gaussian, error_map)
       grad_inv_s = compute_scale_gradient(gaussian, error_map)
       grad_theta = compute_rotation_gradient(gaussian, error_map)
       grad_color = compute_color_gradient(gaussian, error_map)

       # SGD updates with clipping
       gaussian.mu += lr_mu * grad_mu
       gaussian.inv_s += lr_scale * grad_inv_s
       gaussian.theta += lr_theta * grad_theta
       gaussian.color += lr_color * grad_color

       clip_parameters(gaussian)
   ```

3. **Error-Driven Splat Addition**
   ```python
   if len(gaussians) < max_splats:
       error_prob = normalize(error_map ** temperature)
       new_positions = sample_from_probability(error_prob, n_add)
       gaussians.extend(create_splats_at(new_positions))
   ```

#### Phase 3: Anisotropic Refinement
1. **Edge Detection and Orientation**
   ```python
   edges = canny_edge_detection(image)
   edge_orientations = compute_edge_orientations(image)

   for gaussian in edge_gaussians:
       if is_near_edge(gaussian.position, edges):
           local_orientation = edge_orientations[gaussian.position]
           gaussian.theta = local_orientation
           gaussian.inv_s = adapt_to_edge_strength(local_edge_strength)
   ```

2. **Covariance Matrix Optimization**
   ```python
   def compute_covariance_matrix(inv_s, theta):
       # Build rotation matrix
       cos_t, sin_t = np.cos(theta), np.sin(theta)
       R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])

       # Scale matrix from inverse scales
       S_inv = np.diag(inv_s)

       # Covariance: (R * S_inv)^-1 * (R * S_inv)^-T
       RS_inv = R @ S_inv
       return np.linalg.inv(RS_inv @ RS_inv.T)
   ```

### Rendering Pipeline

#### Tile-Based Spatial Binning
```python
def precompute_tile_bins(gaussians, H, W, tile_size):
    tile_to_gauss = defaultdict(list)

    for g_id, gaussian in enumerate(gaussians):
        # Compute 3σ radius in pixels
        cov = gaussian.covariance_matrix
        eigenvals = np.linalg.eigvals(cov)
        radius_px = 3 * np.sqrt(max(eigenvals)) * min(H, W)

        # Find intersecting tiles
        center_px = gaussian.mu * np.array([W, H])
        tiles = get_intersecting_tiles(center_px, radius_px, tile_size)

        for tile_id in tiles:
            tile_to_gauss[tile_id].append(g_id)

    return tile_to_gauss
```

#### Per-Pixel Top-K Evaluation
```python
def render_tile(gaussians, tile_bounds, top_k):
    H_tile, W_tile = tile_bounds.shape[:2]
    output = np.zeros((H_tile, W_tile, channels))

    for y in range(H_tile):
        for x in range(W_tile):
            # Evaluate all candidate Gaussians at this pixel
            values, colors = [], []
            for g_id in tile_gaussians:
                gaussian = gaussians[g_id]
                value = evaluate_gaussian_2d(gaussian, x, y)
                values.append(value)
                colors.append(gaussian.color)

            # Keep top-K contributions
            if values:
                top_indices = np.argsort(values)[-top_k:]
                weights = normalize([values[i] for i in top_indices])
                final_color = sum(w * colors[i] for w, i in zip(weights, top_indices))
                output[y, x] = final_color

    return output
```

## Implementation Plan

### Phase 1: Core Infrastructure (Week 1)
- **T1.1:** Implement `AdaptiveGaussian2D` class with covariance matrix computation
- **T1.2:** Create gradient computation utilities for initialization
- **T1.3:** Build structure tensor analysis for anisotropy detection
- **T1.4:** Implement basic tile-based rendering framework

### Phase 2: Adaptive Initialization (Week 2)
- **T2.1:** Gradient-guided splat placement algorithm
- **T2.2:** Content-adaptive sizing based on local image complexity
- **T2.3:** Initial orientation estimation from structure tensor
- **T2.4:** Color sampling and validation pipeline

### Phase 3: Progressive Optimization (Week 3)
- **T3.1:** Error map computation and analysis
- **T3.2:** Manual gradient computation for parameter updates
- **T3.3:** SGD optimization loop with parameter clipping
- **T3.4:** Error-driven splat addition mechanism

### Phase 4: Anisotropic Refinement (Week 4)
- **T4.1:** Edge detection and orientation analysis
- **T4.2:** Anisotropic splat shaping for edge regions
- **T4.3:** Advanced covariance matrix optimization
- **T4.4:** Top-K rendering with proper weight normalization

### Phase 5: Integration & Testing (Week 5)
- **T5.1:** CLI integration with `--adaptive` flag
- **T5.2:** SVG output support for anisotropic ellipses
- **T5.3:** Performance optimization and profiling
- **T5.4:** Quality validation against Image-GS benchmarks

## Success Criteria

### Quality Metrics
1. **Edge Sharpness:** Chameleon facial features clearly preserved without rasterization
2. **Smooth Coverage:** No visible gaps in background regions
3. **Anisotropy:** Elongated splats properly aligned with image edges
4. **Efficiency:** Better quality-to-splat-count ratio than uniform approach

### Performance Targets
- **Processing Time:** ≤ 5x slower than current uniform approach
- **Memory Usage:** ≤ 2x current peak memory consumption
- **File Size:** Comparable to current output sizes
- **Convergence:** Stable optimization within 1000 iterations

### Technical Validation
- **Mathematical Accuracy:** Proper covariance matrix computation and eigendecomposition
- **Numerical Stability:** Robust parameter clipping and gradient computation
- **Visual Quality:** No artifacts from top-K rendering or tile boundaries
- **Reproducibility:** Consistent results across multiple runs

## Risk Mitigation

### Technical Risks
1. **Convergence Issues:** Implement robust parameter clipping and learning rate schedules
2. **Performance Impact:** Profile early and optimize critical paths (tile rendering)
3. **Quality Regression:** Comprehensive visual testing with reference images
4. **Numerical Instability:** Use stable eigendecomposition and matrix operations

### Integration Risks
1. **API Compatibility:** Maintain backward compatibility with existing extraction interface
2. **SVG Output:** Ensure anisotropic ellipses render correctly across browsers
3. **CLI Changes:** Seamless integration with existing command-line workflow
4. **Testing Coverage:** Extensive unit tests for mathematical operations

## Dependencies

### External Libraries
- **NumPy:** Core array operations and linear algebra
- **SciPy:** Advanced optimization and signal processing
- **scikit-image:** Edge detection and structure tensor computation
- **Existing Codebase:** Current Gaussian and SVG classes as foundation

### Internal Components
- **Current Gaussian class:** Extend for anisotropic properties
- **SVG rendering system:** Enhance for full ellipse support
- **CLI framework:** Add adaptive-specific parameters
- **Testing infrastructure:** Adapt for new quality metrics

## Future Enhancements

### Short-term (Next 2 months)
- **Perceptual Loss Functions:** SSIM and perceptual distance metrics
- **Advanced Initialization:** Saliency-guided and hierarchical placement
- **GPU Acceleration:** NumPy → CuPy for large images
- **Compression Optimization:** Quantized parameter storage

### Medium-term (3-6 months)
- **Real-time Optimization:** Interactive refinement during preview
- **Multi-scale Representation:** Hierarchical level-of-detail
- **Semantic Awareness:** Object-aware splat allocation
- **Style Transfer:** Artistic style-guided optimization

### Long-term (6+ months)
- **Neural Guidance:** Learned initialization networks
- **Differentiable Rasterization:** Hardware-accelerated rendering
- **Video Sequences:** Temporal coherence for animation
- **3D Extension:** Volumetric Gaussian splatting

## Conclusion

This adaptive Gaussian splatting implementation represents a fundamental upgrade from naive uniform placement to research-grade content-aware optimization. By following the Image-GS methodology with pure NumPy implementation, we achieve the quality benefits of adaptive placement while maintaining integration with our existing pipeline.

The key innovation is **treating different image regions appropriately** - elongated splats for edges, large splats for smooth areas, and dense placement only where reconstruction error demands it. This solves the rasterization issue while preserving sharp details and efficient coverage.