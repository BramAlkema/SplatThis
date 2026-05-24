# Combined L1+SSIM Loss Function Specification

**Priority**: Critical
**Status**: Wrong Implementation
**Impact**: Poor perceptual quality, no edge preservation
**Estimated Effort**: 1 week

## Problem Statement

Our current implementation uses only L2 loss, while Image-GS specification requires combined L1 + SSIM loss for optimal quality:

```python
# Current (WRONG)
loss = np.sum(error_map**2)  # L2 only

# Target (Image-GS spec)
loss = L1_loss + 0.1 * SSIM_loss
```

## Current vs Target Implementation

| Component | Current | Target | Status |
|-----------|---------|--------|--------|
| Loss Function | L2 only | L1 + 0.1 * SSIM | ❌ Wrong |
| Edge Preservation | Poor | Excellent via SSIM | ❌ Missing |
| Perceptual Quality | Limited | High via combined loss | ❌ Missing |

## Technical Requirements

### 1. L1 Loss Component
```python
def compute_l1_loss(reconstructed: np.ndarray, target: np.ndarray) -> float:
    """Pixel-accurate loss component."""
    return np.mean(np.abs(reconstructed - target))
```

### 2. SSIM Loss Component
```python
def compute_ssim_loss(reconstructed: np.ndarray, target: np.ndarray) -> float:
    """Structural similarity loss component."""
    ssim_value = structural_similarity(reconstructed, target,
                                     multichannel=True, data_range=1.0)
    return 1.0 - ssim_value  # Convert to loss
```

### 3. Combined Loss Function
```python
def compute_combined_loss(reconstructed: np.ndarray, target: np.ndarray) -> float:
    """Image-GS specified combined loss."""
    l1_loss = compute_l1_loss(reconstructed, target)
    ssim_loss = compute_ssim_loss(reconstructed, target)
    return l1_loss + 0.1 * ssim_loss
```

## Implementation Plan

### Phase 1: Core Implementation (Days 1-2)
1. Implement L1 loss computation
2. Implement SSIM loss computation
3. Combine with Image-GS weighting (0.1 factor)

### Phase 2: Integration (Days 3-4)
1. Replace existing L2 loss in optimization pipeline
2. Update gradient computation for combined loss
3. Ensure backward compatibility

### Phase 3: Validation (Days 5-7)
1. Quality comparison tests (before/after)
2. Edge preservation validation
3. Perceptual quality metrics
4. Performance impact assessment

## Dependencies

- SSIM implementation (skimage.metrics.structural_similarity)
- Existing optimization pipeline
- Quality metrics framework

## Success Criteria

- [ ] L1 loss component implemented
- [ ] SSIM loss component implemented
- [ ] Combined loss with 0.1 weighting factor
- [ ] Integration with optimization pipeline
- [ ] Improved edge preservation
- [ ] Better perceptual quality metrics
- [ ] Performance within acceptable bounds

## Expected Improvements

Based on Image-GS paper results:
- Better edge preservation
- Improved structural similarity
- Higher perceptual quality scores
- More faithful texture reproduction

## Related Specs

- [Progressive Optimization Pipeline](./progressive-optimization-pipeline.md)
- [Quality Metrics Framework](./quality-metrics-framework.md)