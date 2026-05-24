# CUDA Optimization Specification

**Priority**: Medium-High (Production Viability)
**Status**: Missing Critical Performance
**Impact**: 1000× performance gap vs Image-GS targets
**Estimated Effort**: 4-6 weeks

## Problem Statement

Our pure Python implementation has severe performance limitations compared to Image-GS targets:

| Metric | Image-GS Target | SplatThis Reality | Gap |
|--------|-----------------|-------------------|-----|
| Decode Speed | 0.3K MACs/pixel | ~300K ops/pixel | 1000× slower |
| Training Time | 18-26 seconds | Unknown (untested) | Likely much slower |
| Memory Usage | 160 KB typical | Unoptimized | Likely 10× larger |

## Current vs Target Implementation

| Component | Current | Target | Status |
|-----------|---------|--------|--------|
| Tile Rendering | Python loops | Custom CUDA kernels | ❌ Missing |
| Memory Layout | Standard arrays | Cache-optimized layout | ❌ Missing |
| Precision | Float32 | Float16 quantization | ❌ Missing |
| Parallelization | Sequential | Massively parallel | ❌ Missing |

## Technical Requirements

### 1. CUDA Kernel Development
```cuda
// Target: 16×16 tile processing per CUDA block
__global__ void render_tile_kernel(
    const AdaptiveGaussian2D* gaussians,
    const int* tile_gaussian_map,
    float* output_buffer,
    int tile_size,
    int top_k
) {
    // Each thread handles one pixel in the tile
    int pixel_x = blockIdx.x * blockDim.x + threadIdx.x;
    int pixel_y = blockIdx.y * blockDim.y + threadIdx.y;

    // Evaluate top-K gaussians for this pixel
    // Write result to output buffer
}
```

### 2. Memory Layout Optimization
```python
# Target: Structure of Arrays (SoA) layout for cache coherence
class CUDAGaussianBuffer:
    mu_x: torch.Tensor  # [N] all x coordinates
    mu_y: torch.Tensor  # [N] all y coordinates
    inv_s_x: torch.Tensor  # [N] all inverse scales x
    inv_s_y: torch.Tensor  # [N] all inverse scales y
    theta: torch.Tensor  # [N] all rotation angles
    color_r: torch.Tensor  # [N] all red components
    color_g: torch.Tensor  # [N] all green components
    color_b: torch.Tensor  # [N] all blue components
    alpha: torch.Tensor  # [N] all alpha values
```

### 3. Float16 Quantization
```python
def quantize_gaussians(gaussians: List[AdaptiveGaussian2D]) -> torch.Tensor:
    """Convert to half-precision for memory efficiency."""
    # All parameters stored as float16
    # Maintains quality while reducing memory by 50%
    pass
```

### 4. Tile-Gaussian Correspondence
```python
def build_tile_correspondence(gaussians: List[AdaptiveGaussian2D],
                            image_size: Tuple[int, int],
                            tile_size: int = 16) -> Dict[Tuple[int, int], List[int]]:
    """Build efficient spatial mapping for CUDA access."""
    # Precompute which gaussians affect each tile
    # Store in GPU-friendly format
    pass
```

## Implementation Plan

### Phase 1: PyTorch/CUDA Foundation (Weeks 1-2)
1. Convert core data structures to PyTorch tensors
2. Implement basic CUDA operations
3. Setup development and testing infrastructure

### Phase 2: Kernel Development (Weeks 3-4)
1. Implement tile rendering CUDA kernel
2. Add gaussian evaluation kernels
3. Optimize memory access patterns

### Phase 3: Integration (Weeks 5-6)
1. Integrate CUDA backend with existing Python interface
2. Add fallback to CPU for development/testing
3. Performance benchmarking and optimization

## Dependencies

- PyTorch with CUDA support
- CUDA toolkit (11.8+)
- Existing tile renderer architecture
- Performance benchmarking framework

## Success Criteria

- [ ] CUDA tile rendering implementation
- [ ] Float16 quantization support
- [ ] Cache-optimized memory layout
- [ ] 10× performance improvement minimum
- [ ] Target: Approach Image-GS performance (0.3K MACs/pixel)
- [ ] Memory usage within 2× of Image-GS targets
- [ ] Backward compatibility with CPU fallback

## Performance Targets

```python
# Target performance characteristics
TILE_SIZE = 16  # 16×16 tiles per CUDA block
MAX_GAUSSIANS_PER_TILE = 1024  # Memory limit per tile
TOP_K = 10  # Active gaussians per pixel
TARGET_DECODE_SPEED = "0.3K MACs/pixel"
TARGET_MEMORY = "160 KB typical"
```

## Risk Mitigation

1. **CUDA Complexity**: Start with PyTorch operations, progress to custom kernels
2. **Memory Constraints**: Implement tiling strategies for large images
3. **Precision Loss**: Careful validation of float16 quantization
4. **Development Complexity**: Maintain CPU fallback for debugging

## Validation Strategy

1. **Correctness**: Pixel-perfect comparison with CPU implementation
2. **Performance**: Benchmark against Image-GS targets
3. **Memory**: Profile GPU memory usage and optimization
4. **Quality**: Ensure no degradation in visual quality

## Related Specs

- [Progressive Optimization Pipeline](./progressive-optimization-pipeline.md)
- [Performance Benchmarking](./performance-benchmarking.md)
- [Quality Control System](./quality-control-system.md)