# ADR-001: Integration of Image-GS Techniques into SplatThis

**Status**: Proposed
**Date**: 2025-09-22
**Authors**: SplatThis Development Team

## Context

After studying the NYU-ICL/image-gs repository (SIGGRAPH 2025), we identified several advanced techniques that could significantly improve our SplatThis implementation. The current system has limitations in splat placement efficiency, optimization strategies, and quality metrics.

## Problem Statement

Our current SplatThis implementation has the following limitations:

1. **Static Allocation**: All Gaussians are allocated upfront, leading to inefficient use of computational resources
2. **Uniform Placement**: Splat placement doesn't adapt well to content complexity
3. **Simple Loss**: Only basic reconstruction error, missing perceptual quality metrics
4. **Fixed Parameters**: No progressive refinement or adaptive optimization
5. **Size Issues**: Recent fixes for splat sizing revealed need for better adaptive strategies

## Research Findings from image-gs

### Key Techniques Identified:

#### 1. Progressive Optimization
- **What**: Start with subset of Gaussians, progressively add more based on error
- **Benefit**: More efficient allocation, better convergence
- **Implementation**: `initial_ratio` parameter, error-guided addition

#### 2. Error-Guided Placement
- **What**: Use reconstruction error maps to sample new Gaussian positions
- **Benefit**: Automatic adaptive density based on content complexity
- **Implementation**: Error-weighted probability distribution for placement

#### 3. Multi-Component Loss Functions
- **What**: L1 + L2 + SSIM + LPIPS combined losses
- **Benefit**: Better optimization signals, perceptual quality
- **Implementation**: Configurable loss weights

#### 4. Separate Learning Rates
- **What**: Different rates for position, scale, rotation, features
- **Benefit**: More stable and faster convergence
- **Implementation**: Parameter-specific Adam optimizers

#### 5. Quantization Support
- **What**: Configurable bit precision for parameters
- **Benefit**: Hardware efficiency (0.3K MACs per pixel)
- **Implementation**: Pre-rendering quantization

#### 6. Tile-Based Rendering
- **What**: Efficient CUDA-based rasterization
- **Benefit**: Better performance for large images
- **Implementation**: Custom kernels

## Decision Options

### Option 1: Full Rewrite Based on image-gs
**Pros**:
- State-of-the-art performance
- Research-backed implementation
- All advanced features

**Cons**:
- Complete codebase rewrite
- CUDA dependency
- Complexity overhead
- Loss of current SVG output capability

### Option 2: Incremental Integration (Recommended)
**Pros**:
- Preserve existing SVG pipeline
- Gradual improvement
- Maintain current functionality
- Lower risk

**Cons**:
- Slower to implement all features
- May not achieve full performance potential

### Option 3: Hybrid Approach
**Pros**:
- Best of both worlds
- Dual output modes (SVG + optimized)
- Flexibility

**Cons**:
- Code complexity
- Maintenance overhead

## Decision

**We choose Option 2: Incremental Integration**

### Rationale:
1. **Preserve SVG Output**: Our SVG generation is unique and valuable for web/vector graphics
2. **Risk Management**: Incremental changes allow testing and validation
3. **Resource Efficiency**: Allows gradual improvement without major disruption
4. **Learning Opportunity**: Understand each technique before full implementation

## Implementation Plan

### Phase 1: Foundation (Immediate)
- [ ] Implement progressive Gaussian allocation
- [ ] Add error-guided splat placement
- [ ] Create reconstruction error computation

### Phase 2: Optimization (Next)
- [ ] Integrate multi-component loss functions
- [ ] Add separate learning rate scheduling
- [ ] Implement adaptive refinement

### Phase 3: Advanced Features (Future)
- [ ] Add quantization support
- [ ] Implement tile-based rendering option
- [ ] Add perceptual quality metrics (LPIPS)

### Phase 4: Performance (Long-term)
- [ ] CUDA acceleration (optional)
- [ ] Memory optimization
- [ ] Real-time rendering support

## Technical Architecture Changes

### New Components Required:
```
src/splat_this/core/
├── progressive_allocator.py     # Progressive Gaussian allocation
├── error_guided_placement.py    # Error-based placement strategies
├── multi_loss.py               # Combined loss functions
├── adaptive_optimizer.py       # Parameter-specific optimization
└── quality_metrics.py          # LPIPS, SSIM, etc.
```

### Modified Components:
```
src/splat_this/core/
├── adaptive_extract.py         # Integration with progressive allocation
├── optimized_svgout.py         # Support for quantized parameters
└── extract.py                  # Enhanced Gaussian representation
```

## Configuration Changes

### New Config Parameters:
```python
@dataclass
class ProgressiveConfig:
    initial_ratio: float = 0.3          # Start with 30% of target Gaussians
    add_gaussians_every: int = 100      # Add new Gaussians every N iterations
    max_gaussians_per_step: int = 50    # Limit additions per step
    error_threshold: float = 0.01       # Minimum error for new placement

@dataclass
class LossConfig:
    l1_weight: float = 1.0
    l2_weight: float = 0.5
    ssim_weight: float = 0.2
    lpips_weight: float = 0.1
```

## Success Metrics

### Quality Improvements:
- [ ] PSNR improvement > 2dB
- [ ] SSIM score > 0.95
- [ ] LPIPS score < 0.1
- [ ] Visual inspection confirms better reconstruction

### Efficiency Gains:
- [ ] 30% reduction in total Gaussians needed
- [ ] 50% faster convergence
- [ ] Memory usage optimization

### Compatibility:
- [ ] SVG output maintains current quality
- [ ] All existing tests pass
- [ ] Browser compatibility preserved

## Risks and Mitigations

### Risk 1: Implementation Complexity
**Mitigation**: Incremental approach, thorough testing at each phase

### Risk 2: Performance Regression
**Mitigation**: Benchmarking at each step, rollback capability

### Risk 3: SVG Compatibility
**Mitigation**: Dual testing with both old and new methods

### Risk 4: Resource Requirements
**Mitigation**: Optional features, graceful degradation

## Alternatives Considered

1. **Fork image-gs directly**: Rejected due to CUDA dependencies and SVG loss
2. **Ignore improvements**: Rejected due to significant performance potential
3. **Complete rewrite**: Rejected due to high risk and resource requirements

## References

- [NYU-ICL/image-gs Repository](https://github.com/NYU-ICL/image-gs)
- "Image-GS: Content-Adaptive Image Representation via 2D Gaussians" (SIGGRAPH 2025)
- Current SplatThis implementation analysis
- Performance benchmarking results

## Follow-up Actions

1. Create detailed implementation specs for Phase 1
2. Set up benchmarking infrastructure
3. Create test cases for progressive allocation
4. Design API for multi-component losses
5. Update project roadmap with new timeline

---

**This ADR will be updated as implementation progresses and decisions are validated.**