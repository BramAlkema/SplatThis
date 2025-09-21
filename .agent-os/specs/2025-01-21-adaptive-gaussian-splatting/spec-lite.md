# Adaptive Gaussian Splatting - Quick Reference

**Created:** 2025-01-21 | **Priority:** High | **Status:** Ready for Implementation

## Problem
Current uniform splat placement causes rasterization at high density and poor edge handling.

## Solution
Implement Image-GS methodology: content-adaptive placement + anisotropic ellipses + progressive optimization.

## Key Components

### 1. AdaptiveGaussian2D
- Full covariance matrices (not just circles)
- Inverse scale optimization for stability
- Position in normalized [0,1]² coordinates

### 2. Content-Adaptive Initialization
- Gradient-guided placement (high-gradient regions get more splats)
- Structure tensor analysis for initial orientation
- Local color sampling

### 3. Progressive Optimization
- Start with ~50% budget, add splats where error is high
- SGD updates for position, scale, rotation, color
- Error-driven refinement (add splats where reconstruction fails)

### 4. Top-K Tile Rendering
- Tile-based spatial binning (16×16 tiles)
- Per-pixel top-K blending (prevents muddy overdraw)
- Order-independent weighted sum

## Implementation Flow

```python
# 1. Initialize based on image gradients
initial_splats = gradient_guided_init(image, n_splats//2)

# 2. Progressive optimization loop
for iteration in range(max_iterations):
    rendered = tile_render(splats, top_k=8)
    error_map = abs(rendered - target)

    # Update existing splat parameters
    update_parameters(splats, error_map, learning_rates)

    # Add new splats where error is high
    if len(splats) < max_splats:
        add_splats(error_map, n_add=max_splats//8)

# 3. Output anisotropic ellipses to SVG
```

## Expected Results
- **Sharp edges** with elongated splats following contours
- **Efficient backgrounds** with large circular splats
- **No rasterization** at high density
- **Clean blending** without muddy overdraw

## Technical Stack
- **Pure NumPy/SciPy** (no PyTorch needed)
- **Structure tensor** for anisotropy detection
- **Manual gradients** for parameter optimization
- **Tile rendering** for performance

## Success Criteria
- Chameleon facial features clearly preserved
- No visible gaps in smooth areas
- Proper elongated splats along edges
- ≤5x processing time vs current approach