# Adaptive Gaussian Splatting Implementation Plan

Based on Image-GS research: [https://github.com/NYU-ICL/image-gs](https://github.com/NYU-ICL/image-gs)

## Phase 1 Issue Breakdown (ADR-002)

This section converts ADR-002 Phase 1 into concrete implementation tasks.

### EPIC P1: Canonical PNG to Splat Foundation

#### Issue P1-1: Canonical Raw Splat Schema
- Status: `todo`
- Goal: Establish one authoritative in-memory and on-disk splat contract.
- Tasks:
  - Define schema fields: `x`, `y`, `sx`, `sy`, `theta`, `r`, `g`, `b`, `a`.
  - Add optional metadata fields (`source`, `score`, `layer`) behind a stable version key.
  - Document schema in API docs and example files.
- Deliverables:
  - Schema dataclass/type + serializer/deserializer.
  - Versioned JSON schema fixture for tests.
- Acceptance:
  - Invalid payloads fail fast with actionable errors.
  - Valid payloads round-trip with zero loss.

#### Issue P1-2: Parameter Constraints and Validation
- Status: `todo`
- Goal: Prevent unstable/invalid gaussians entering optimization or export.
- Tasks:
  - Enforce `sx > 0` and `sy > 0` with lower bounds.
  - Clamp `a` to `[0, 1]` and color channels to `[0, 1]` internally.
  - Add central validation function used by extractor, optimizer, and exporters.
- Deliverables:
  - Shared validation module.
  - Unit tests for boundary and failure cases.
- Acceptance:
  - No exporter accepts invalid splat parameters.
  - Validation errors identify parameter name and value.

#### Issue P1-3: Deterministic Initializer
- Status: `todo`
- Goal: Reproducible initialization under fixed seed.
- Tasks:
  - Thread `seed` through CLI and Python API.
  - Remove implicit global RNG usage in initialization path.
  - Persist seed/config snapshot with output artifacts.
- Deliverables:
  - `--seed` CLI option and matching API parameter.
  - Determinism tests on fixed input PNG.
- Acceptance:
  - Same input + seed + config produces byte-identical raw splat output.

#### Issue P1-4: Baseline Optimizer (`L1 + SSIM`)
- Status: `todo`
- Goal: Establish reliable first differentiable optimization pass.
- Tasks:
  - Implement combined loss: `loss = w_l1 * L1 + w_ssim * (1 - SSIM)`.
  - Add configurable iterations and learning-rate groups.
  - Track per-iteration metrics and stop conditions.
- Deliverables:
  - Optimizer module with config.
  - Unit/integration tests for loss decrease on reference image.
- Acceptance:
  - Median reconstruction error improves from initialization baseline.

#### Issue P1-5: Stage-Level Observability
- Status: `todo`
- Goal: Make failures diagnosable without manual debugging.
- Tasks:
  - Emit `init`, `iter-N`, and `final` artifacts.
  - Add structured logs for key metrics (loss, PSNR/SSIM, splat count).
  - Add run manifest that captures input hash, seed, config, and timings.
- Deliverables:
  - Artifact writer utilities.
  - Log schema and minimal viewer instructions.
- Acceptance:
  - Any failed run is diagnosable from artifacts/logs alone.

#### Issue P1-6: Regression and Determinism Test Harness
- Status: `todo`
- Goal: Prevent regressions while pipeline is evolving.
- Tasks:
  - Add fixed reference PNG set for deterministic checks.
  - Add threshold tests for quality and runtime guardrails.
  - Add CI target for Phase 1 checks.
- Deliverables:
  - `tests/` coverage for schema, determinism, and optimizer behavior.
  - One command to run Phase 1 validation suite.
- Acceptance:
  - CI fails on determinism drift or metric regression.

## Current vs Target Approach

### Current Limitation (Naive)
- **Uniform SLIC segmentation** regardless of image content
- **Fixed circular splats** with minimal anisotropy
- **No error-driven placement** - splats placed blindly
- **Static optimization** - no progressive refinement
- **Density vs quality trade-off** - rasterization at high counts

### Target Approach (Adaptive)
- **Content-adaptive placement** based on error maps and saliency
- **Anisotropic Gaussians** following edge orientations
- **Progressive optimization** with error-guided refinement
- **Dynamic parameter optimization** (position, shape, color, opacity)
- **Intelligent density allocation** - dense where needed, sparse elsewhere

## Implementation Strategy

### Phase 1: Adaptive Initialization
Replace uniform SLIC with intelligent placement strategies:

#### 1.1 Gradient-Based Initialization
```python
# Compute image gradients for edge detection
grad_x, grad_y = compute_gradients(image)
gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
gradient_orientation = np.arctan2(grad_y, grad_x)

# Place initial splats at high-gradient locations
high_gradient_points = find_local_maxima(gradient_magnitude)
initial_splats = [create_splat_at(point, orientation) for point in high_gradient_points]
```

#### 1.2 Saliency-Guided Initialization
```python
# Compute saliency map for importance-based placement
saliency_map = compute_saliency(image)
important_regions = saliency_map > threshold

# Allocate more splats to salient regions
splat_density = scale_by_saliency(base_density, saliency_map)
initial_splats = adaptive_placement(splat_density)
```

#### 1.3 Structure Tensor Analysis
```python
# Compute local structure for anisotropic shaping
structure_tensor = compute_structure_tensor(image, sigma=1.0)
eigenvals, eigenvecs = np.linalg.eigh(structure_tensor)

# Create anisotropic splats based on local structure
for region in image_regions:
    coherence = (eigenvals[1] - eigenvals[0]) / (eigenvals[1] + eigenvals[0])
    if coherence > anisotropy_threshold:
        # Create elongated splat along dominant direction
        splat = create_anisotropic_splat(
            orientation=eigenvecs[:, 1],
            aspect_ratio=eigenvals[1]/eigenvals[0]
        )
```

### Phase 2: Progressive Optimization

#### 2.1 Error-Guided Refinement
```python
def progressive_optimization(image, initial_splats, max_iterations=10):
    splats = initial_splats

    for iteration in range(max_iterations):
        # Render current splat configuration
        rendered = render_gaussian_splats(splats)

        # Compute reconstruction error
        error_map = compute_error(rendered, image)

        # Identify high-error regions
        high_error_regions = error_map > error_threshold

        # Add/refine splats in high-error areas
        if np.any(high_error_regions):
            new_splats = add_splats_to_regions(high_error_regions)
            splats.extend(new_splats)

        # Optimize existing splat parameters
        splats = optimize_splat_parameters(splats, error_map)

        # Early stopping if error is low enough
        if np.mean(error_map) < convergence_threshold:
            break

    return splats
```

#### 2.2 Differentiable Parameter Optimization
```python
def optimize_splat_parameters(splats, error_map, learning_rate=0.01):
    for splat in splats:
        # Compute gradients for each parameter
        grad_position = compute_position_gradient(splat, error_map)
        grad_covariance = compute_covariance_gradient(splat, error_map)
        grad_color = compute_color_gradient(splat, error_map)
        grad_alpha = compute_alpha_gradient(splat, error_map)

        # Update parameters
        splat.position += learning_rate * grad_position
        splat.covariance += learning_rate * grad_covariance
        splat.color += learning_rate * grad_color
        splat.alpha += learning_rate * grad_alpha

        # Clamp parameters to valid ranges
        splat.clamp_parameters()

    return splats
```

### Phase 3: Anisotropic Gaussian Handling

#### 3.1 Edge-Following Splats
```python
def create_edge_following_splats(edge_map, orientation_map):
    splats = []

    for edge_point in extract_edge_points(edge_map):
        local_orientation = orientation_map[edge_point]

        # Create elongated splat along edge
        splat = Gaussian(
            position=edge_point,
            covariance=create_edge_covariance(local_orientation),
            color=sample_local_color(edge_point),
            alpha=compute_edge_opacity(edge_point)
        )
        splats.append(splat)

    return splats

def create_edge_covariance(orientation, elongation_factor=3.0):
    # Create elongated covariance matrix along edge direction
    cos_theta, sin_theta = np.cos(orientation), np.sin(orientation)
    rotation_matrix = np.array([[cos_theta, -sin_theta],
                               [sin_theta, cos_theta]])

    # Elongated eigenvalues (major axis along edge)
    eigenvals = np.diag([elongation_factor, 1.0])

    # Rotated covariance matrix
    covariance = rotation_matrix @ eigenvals @ rotation_matrix.T
    return covariance
```

#### 3.2 Content-Adaptive Sizing
```python
def adaptive_splat_sizing(image_region, base_size=5.0):
    # Analyze local content complexity
    local_variance = np.var(image_region)
    edge_density = compute_edge_density(image_region)

    # Adapt size based on content
    if edge_density > high_detail_threshold:
        # Small splats for detailed areas
        size_factor = 0.5
    elif local_variance < smooth_threshold:
        # Large splats for smooth areas
        size_factor = 2.0
    else:
        # Standard size for medium complexity
        size_factor = 1.0

    return base_size * size_factor
```

### Phase 4: Intelligent Density Control

#### 4.1 Perceptual Importance Weighting
```python
def compute_perceptual_importance(image):
    # Combine multiple importance cues
    edge_importance = compute_edge_importance(image)
    saliency_importance = compute_saliency_importance(image)
    contrast_importance = compute_contrast_importance(image)

    # Weighted combination
    importance_map = (
        0.4 * edge_importance +
        0.4 * saliency_importance +
        0.2 * contrast_importance
    )

    return normalize(importance_map)

def allocate_splats_by_importance(importance_map, total_splats):
    # Allocate splats proportional to importance
    normalized_importance = importance_map / np.sum(importance_map)
    splat_allocation = (normalized_importance * total_splats).astype(int)

    return splat_allocation
```

#### 4.2 Hierarchical Refinement
```python
def hierarchical_optimization(image, levels=3):
    # Start with coarse representation
    current_splats = initialize_coarse_splats(image, level=levels)

    for level in range(levels, 0, -1):
        # Optimize at current level
        current_splats = optimize_level(current_splats, image, level)

        # Add finer detail for next level
        if level > 1:
            error_map = compute_reconstruction_error(current_splats, image)
            detail_splats = add_detail_splats(error_map, level-1)
            current_splats.extend(detail_splats)

    return current_splats
```

## Implementation Architecture

### Core Classes

```python
class AdaptiveSplatExtractor:
    def __init__(self, initialization_mode='gradient', max_iterations=10):
        self.initialization_mode = initialization_mode
        self.max_iterations = max_iterations
        self.error_threshold = 0.01
        self.convergence_threshold = 0.005

    def extract_adaptive_splats(self, image, target_count=None, quality_target=None):
        # Step 1: Intelligent initialization
        initial_splats = self.initialize_splats(image)

        # Step 2: Progressive optimization
        optimized_splats = self.progressive_optimization(image, initial_splats)

        # Step 3: Post-processing and validation
        final_splats = self.post_process_splats(optimized_splats)

        return final_splats

class AnisotropicGaussian(Gaussian):
    def __init__(self, position, covariance_matrix, color, alpha):
        self.position = position
        self.covariance = covariance_matrix  # Full 2x2 matrix
        self.color = color
        self.alpha = alpha

    @property
    def eigenvalues(self):
        return np.linalg.eigvals(self.covariance)

    @property
    def orientation(self):
        eigenvals, eigenvecs = np.linalg.eigh(self.covariance)
        return np.arctan2(eigenvecs[1, 1], eigenvecs[0, 1])

    @property
    def aspect_ratio(self):
        eigenvals = self.eigenvalues
        return max(eigenvals) / min(eigenvals)
```

## Expected Benefits

1. **Sharp Edge Preservation**: Elongated splats following edges eliminate the rasterization issue
2. **Efficient Smooth Areas**: Large splats in uniform regions avoid over-tessellation
3. **Content-Aware Quality**: More detail where the eye notices, less where it doesn't
4. **Natural Anisotropy**: Splats shaped by image structure, not forced into circles
5. **Progressive Quality**: Start fast, refine iteratively to target quality
6. **Scalable Complexity**: Adaptive allocation scales with image complexity

## Success Metrics

- **Chameleon facial features**: Sharp and clear without rasterization
- **Edge quality**: Clean, well-defined edges with appropriate elongation
- **Smooth areas**: Efficient coverage without over-splitting
- **File size efficiency**: Better quality-to-size ratio than uniform approach
- **Visual coherence**: Natural, non-pixelated appearance at all zoom levels

This adaptive approach should solve the core issues we've identified while achieving the quality standards of research-grade Gaussian splatting implementations.
