# Content-Adaptive Initialization Specification

**Priority**: High
**Status**: Incomplete Implementation
**Impact**: Suboptimal gaussian placement, slower convergence
**Estimated Effort**: 1 week

## Problem Statement

Our initialization strategies don't match the exact Image-GS specification for gradient-guided probabilistic sampling:

```python
# Image-GS specification
P_init(x) = (1-λ_init)*||∇I(x)||₂/Σ||∇I|| + λ_init/(H*W)

# Current implementation
# Multiple strategies but none match exact formula
```

## Current vs Target Implementation

| Component | Current | Target | Status |
|-----------|---------|--------|--------|
| Sampling Formula | Multiple approaches | Exact Image-GS formula | ⚠️ Incomplete |
| Balance Parameter | Not implemented | λ_init = 0.3 | ❌ Missing |
| Gradient Integration | Basic gradients | Normalized gradient magnitude | ⚠️ Partial |

## Technical Requirements

### 1. Exact Image-GS Initialization Formula
```python
def compute_init_probability(image: np.ndarray, lambda_init: float = 0.3) -> np.ndarray:
    """
    Compute initialization probability according to Image-GS specification.

    Args:
        image: Input image [H, W, C]
        lambda_init: Balance parameter (default 0.3)

    Returns:
        Probability map [H, W] for gaussian placement
    """
    # Compute gradient magnitude
    grad_mag = compute_gradient_magnitude(image)

    # Normalize gradient component
    grad_sum = np.sum(grad_mag)
    if grad_sum > 0:
        grad_normalized = grad_mag / grad_sum
    else:
        grad_normalized = np.zeros_like(grad_mag)

    # Uniform component
    H, W = image.shape[:2]
    uniform_prob = 1.0 / (H * W)

    # Combined probability
    P_init = (1 - lambda_init) * grad_normalized + lambda_init * uniform_prob

    return P_init
```

### 2. Gradient Magnitude Computation
```python
def compute_gradient_magnitude(image: np.ndarray) -> np.ndarray:
    """Compute L2 gradient magnitude for each pixel."""
    if len(image.shape) == 3:
        # Multi-channel: compute gradient for each channel, then combine
        grad_mag = np.zeros(image.shape[:2])
        for c in range(image.shape[2]):
            grad_y, grad_x = np.gradient(image[:, :, c])
            grad_mag += np.sqrt(grad_x**2 + grad_y**2)
        return grad_mag / image.shape[2]  # Average across channels
    else:
        # Single channel
        grad_y, grad_x = np.gradient(image)
        return np.sqrt(grad_x**2 + grad_y**2)
```

### 3. Probabilistic Sampling
```python
def sample_gaussian_positions(probability_map: np.ndarray,
                            num_gaussians: int) -> List[Tuple[float, float]]:
    """Sample gaussian positions according to probability map."""
    H, W = probability_map.shape

    # Flatten and normalize
    prob_flat = probability_map.flatten()
    prob_flat = prob_flat / np.sum(prob_flat)

    # Sample indices
    indices = np.random.choice(H * W, size=num_gaussians, p=prob_flat)

    # Convert to (y, x) coordinates
    positions = []
    for idx in indices:
        y, x = np.unravel_index(idx, (H, W))
        # Normalize to [0, 1]
        positions.append((x / W, y / H))

    return positions
```

## Implementation Plan

### Phase 1: Core Algorithm (Days 1-3)
1. Implement exact Image-GS probability formula
2. Add gradient magnitude computation
3. Integrate λ_init balance parameter

### Phase 2: Sampling Integration (Days 4-5)
1. Implement probabilistic sampling
2. Replace existing initialization strategies
3. Ensure consistent coordinate systems

### Phase 3: Validation & Tuning (Days 6-7)
1. Validate against Image-GS results
2. Parameter sensitivity analysis
3. Performance comparison with existing methods

## Dependencies

- Image gradient computation (numpy.gradient)
- Random sampling utilities
- Existing gaussian initialization framework

## Success Criteria

- [ ] Exact Image-GS formula implemented
- [ ] λ_init = 0.3 balance parameter
- [ ] Gradient-guided probabilistic sampling
- [ ] Integration with existing initialization
- [ ] Better convergence than current methods
- [ ] Validation against paper results

## Parameter Configuration

```python
# Image-GS standard parameters
LAMBDA_INIT = 0.3  # Balance between content-adaptive and uniform
TOP_K = 10         # Top gaussians per pixel
SIGMA_THRESHOLD = 3.0  # 3σ radius for culling
```

## Expected Improvements

- Faster convergence (fewer optimization steps)
- Better initial gaussian placement
- More content-aware resource allocation
- Improved final quality metrics

## Related Specs

- [Progressive Optimization Pipeline](./progressive-optimization-pipeline.md)
- [Structure Tensor Enhancement](./structure-tensor-enhancement.md)