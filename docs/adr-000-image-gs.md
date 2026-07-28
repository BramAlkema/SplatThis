# ADR: Image-GS Content-Adaptive Image Representation via 2D Gaussians

**Status**: Accepted
**Date**: 2025-01-23
**Authors**: Based on research by Zhang et al. (SIGGRAPH 2025)
**Context**: SplatThis adaptive Gaussian splatting implementation

## Summary

This ADR documents the architectural decisions for implementing Image-GS, a content-adaptive image representation system using anisotropic 2D Gaussians for efficient image compression and real-time rendering.

## Problem Statement

Traditional image compression formats (PNG, JPEG) and existing neural image representations face several limitations:

1. **Fixed data structures** lack content adaptivity
2. **Compute-intensive implicit models** hinder real-time performance
3. **Poor random access** performance for GPU applications
4. **Limited quality control** without progressive optimization
5. **Inadequate handling** of non-uniformly distributed image features

## Decision

Implement Image-GS using explicit 2D Gaussian primitives with the following architectural choices:

### 1. Gaussian Primitive Representation

**Decision**: Use anisotropic 2D Gaussians with explicit parameterization
```
Gaussian(μ, Σ, c) where:
- μ ∈ ℝ² (mean position)
- Σ = RSS^T R^T (covariance via rotation R and scaling S)
- c ∈ ℝⁿ (color vector, n-dimensional for multi-channel support)
```

**Rationale**:
- **Explicit representation** enables fast parallel evaluation
- **Anisotropic covariance** captures directional image features
- **Factorized covariance** (R, S) ensures positive semi-definite matrices during optimization
- **Variable color dimension** supports grayscale, RGB, CMYK, and multi-channel textures

**Trade-offs**:
- ✅ Hardware-friendly parallel evaluation
- ✅ Physical intuition for parameter initialization
- ❌ More parameters than circular Gaussians
- ❌ Requires careful initialization for convergence

### 2. Tile-Based Rendering Pipeline

**Decision**: Implement tile-based rendering with top-K normalization
```python
# Tile-based evaluation with top-K selection
for pixel x in tile T_j:
    S_j^(x) = top_K(gaussians_in_tile, evaluated_at=x)
    c_r(x) = sum(G_i(x) * c_i for G_i in S_j^(x)) / sum(G_i(x) for G_i in S_j^(x))
```

**Rationale**:
- **Data locality** for GPU cache efficiency
- **Bounded computation** per pixel (K gaussians max)
- **Top-K normalization** acts as regularization and improves quality
- **Order-independent** rendering (no depth sorting needed)

**Alternatives Considered**:
- Full gaussian evaluation per pixel (too expensive)
- Fixed radius culling (less adaptive)
- Alpha blending like 3D GS (unnecessary ordering complexity)

### 3. Content-Adaptive Initialization Strategy

**Decision**: Gradient-guided probabilistic sampling
```python
P_init(x) = (1 - λ_init) * ||∇I(x)||₂ / Σ||∇I|| + λ_init / (H*W)
```

**Rationale**:
- **Content awareness** via image gradient magnitude
- **Uniform coverage** via constant term (λ_init)
- **Adaptive allocation** places more Gaussians in high-frequency regions
- **Single-pass initialization** from image analysis

**Parameters**:
- `λ_init = 0.3` balances content-adaptive vs uniform sampling
- `K = 10` top gaussians per pixel for performance/quality trade-off

### 4. Progressive Error-Guided Optimization

**Decision**: Multi-stage progressive refinement
```python
# Progressive allocation schedule
initial_gaussians = N_g / 2
add_gaussians = N_g / 8 every 0.5K steps
sampling_probability ∝ |c_r(x) - c_t(x)|  # reconstruction error
```

**Rationale**:
- **Smooth LOD hierarchy** emerges naturally from optimization
- **Error-driven allocation** focuses resources on difficult regions
- **Progressive quality control** enables rate-distortion flexibility
- **Single optimization run** produces multiple quality levels

**Alternatives Considered**:
- Fixed uniform allocation (less adaptive)
- Two-stage optimization like GaussianImage (slower)
- Saliency-based allocation (for semantic applications)

### 5. Numerical Optimization Choices

**Decision**: Inverse scale parameterization with Adam optimizer
```python
# Optimize inverse scales instead of raw scales
inv_s = 1/s  # s typically in [5,10] → inv_s in [0.1,0.2]
loss = L1_loss + 0.1 * SSIM_loss
learning_rates = {μ: 5e-4, c: 5e-3, s: 2e-3, θ: 2e-3}
```

**Rationale**:
- **Inverse scale optimization** provides smoother gradients in [0,1] range
- **Combined loss function** balances pixel accuracy (L1) and perceptual quality (SSIM)
- **Differentiated learning rates** account for parameter sensitivity
- **5K step convergence** with 95% quality reached in 400 steps

### 6. Hardware-Optimized Implementation

**Decision**: Custom CUDA kernels with optimized memory access
- **16×16 tile processing** per CUDA block
- **Float16 quantization** for all parameters
- **Coordinate normalization** to [0,1]² for resolution independence
- **Cache-friendly data layout** for tile-gaussian correspondence

**Performance Targets**:
- **0.3K MACs per pixel** (10× faster than neural methods)
- **Sub-linear scaling** with gaussian count
- **Real-time rendering** at 2K×2K resolution

## Implementation Architecture

### Core Components

1. **Gaussian Primitive Class**
   ```python
   @dataclass
   class AdaptiveGaussian2D:
       mu: np.ndarray        # Position [2]
       inv_s: np.ndarray     # Inverse scales [2]
       theta: float          # Rotation angle [0,π)
       color: np.ndarray     # Color [n_channels]
       alpha: float          # Opacity
   ```

2. **Tile Renderer**
   - Spatial subdivision and gaussian-tile mapping
   - 3σ radius computation for culling
   - Top-K selection and normalization
   - Parallel CUDA evaluation

3. **Content-Adaptive Extractor**
   - Gradient-guided initialization
   - Progressive error-based refinement
   - Structure tensor analysis for orientation

4. **Differentiable Optimization Pipeline**
   - Adam optimizer with custom learning rates
   - Combined L1 + SSIM loss
   - Gradient flow through top-K selection

### Integration Points

- **Round-trip conversions** with existing Gaussian representations
- **SVG export** for vector graphics workflows
- **Progressive quality levels** for adaptive streaming
- **Multi-channel support** for texture stacks

## Consequences

### Positive

1. **Superior Rate-Distortion**: 30.41 dB PSNR at 160 KB (vs JPEG 25.43 dB)
2. **Fast Random Access**: 0.3K MACs per pixel decode
3. **Content Adaptivity**: Automatically allocates resources to high-frequency regions
4. **Progressive Quality**: Single optimization produces smooth LOD hierarchy
5. **Hardware Friendly**: Parallel evaluation, cache-coherent memory access
6. **Flexible Applications**: Compression, restoration, semantic-aware encoding

### Negative

1. **Training Time**: Requires optimization per image (vs generalized codecs)
2. **Natural Image Limitation**: Less effective on pixel-level noise and fine details
3. **Memory vs Quality**: Requires sufficient gaussian budget for complex images
4. **Parameter Sensitivity**: Initialization and learning rates require tuning

### Risk Mitigation

1. **Convergence Robustness**: Fallback initialization strategies, gradient clipping
2. **Quality Validation**: Comprehensive metrics (PSNR, SSIM, LPIPS, FLIP)
3. **Performance Monitoring**: Render time and memory usage tracking
4. **Compatibility**: Standard format export (SVG, PNG) for downstream tools

## Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| **Implicit Neural Fields** | Compact networks | Slow evaluation, poor random access | ❌ Rejected |
| **Fixed Grid Features** | Fast evaluation | Not content-adaptive | ❌ Rejected |
| **3D Gaussian Splatting** | Proven framework | Unnecessary complexity for 2D | ❌ Rejected |
| **Vector Quantization** | Better compression | Complex optimization, slower decode | ❌ Rejected |
| **Entropy Coding** | Higher compression | Breaks data locality | ❌ Rejected |

## Implementation Notes

### Critical Design Patterns

1. **Tile-based spatial organization** for GPU memory coherence
2. **Progressive allocation** for quality-bitrate flexibility
3. **Structure tensor initialization** for content awareness
4. **Top-K normalization** for regularization and performance
5. **Inverse parameterization** for optimization stability

### Performance Characteristics

```
Image Resolution: 2K×2K
Gaussian Count: 10K-50K
Training Time: 18-26 seconds
Render Time: 3.7-4.5 ms
Memory: 160 KB typical
Compression: 30-100× vs raw
```

### Quality Metrics (at 0.366 bpp)

- **PSNR**: 32.99 ± 4.49 dB
- **MS-SSIM**: 0.966 ± 0.020
- **LPIPS**: 0.083 ± 0.057
- **FLIP**: 0.078 ± 0.029

## Future Considerations

1. **Dynamic Content**: Extension to video via temporal gaussian motion
2. **Spatial Adaptivity**: Binary space partitioning for better detail handling
3. **Semantic Integration**: Automatic saliency-guided allocation
4. **Hybrid Approaches**: Combination with neural networks for complex regions

## References

- **Primary Paper**: Zhang et al. "Image-GS: Content-Adaptive Image Representation via 2D Gaussians" SIGGRAPH 2025
- **3D Gaussian Splatting**: Kerbl et al. "3D Gaussian Splatting for Real-Time Radiance Field Rendering"
- **Neural Image Compression**: Multiple baselines (SIREN, WIRE, I-NGP, etc.)
- **Texture Compression**: Industry standards (BC1, BC7, ASTC)

---

**Status**: This ADR establishes the architectural foundation for Image-GS implementation in the SplatThis framework, providing content-adaptive 2D gaussian representation with hardware-optimized real-time rendering capabilities.