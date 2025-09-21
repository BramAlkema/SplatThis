# Technical Implementation Specification - Adaptive Gaussian Splatting

**Created:** 2025-01-21 | **Parent Spec:** @../spec.md | **Version:** 1.0

## Mathematical Foundations

### Gaussian 2D Representation

#### Covariance Matrix Construction
```python
def build_covariance_matrix(inv_s: np.ndarray, theta: float) -> np.ndarray:
    """
    Build 2x2 covariance matrix from inverse scales and rotation.

    Args:
        inv_s: [1/sx, 1/sy] inverse scale factors
        theta: rotation angle in [0, π)

    Returns:
        2x2 covariance matrix Σ
    """
    # Rotation matrix
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    R = np.array([[cos_t, -sin_t],
                  [sin_t, cos_t]])

    # Inverse scale matrix
    S_inv = np.diag(inv_s)

    # Build covariance: Σ = (R S^-1)^-1 (R S^-1)^-T
    RS_inv = R @ S_inv
    return np.linalg.inv(RS_inv @ RS_inv.T)
```

#### Gaussian Evaluation Function
```python
def evaluate_gaussian_2d(mu: np.ndarray, cov_inv: np.ndarray, point: np.ndarray) -> float:
    """
    Evaluate 2D Gaussian at given point.

    Args:
        mu: [x, y] center position
        cov_inv: 2x2 inverse covariance matrix
        point: [x, y] evaluation point

    Returns:
        Gaussian value at point
    """
    delta = point - mu
    quadratic_form = delta.T @ cov_inv @ delta
    return np.exp(-0.5 * quadratic_form)
```

### Structure Tensor Analysis

#### Local Orientation Detection
```python
def compute_structure_tensor(image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    Compute structure tensor for local orientation analysis.

    Args:
        image: Input image (H, W) or (H, W, C)
        sigma: Gaussian smoothing parameter

    Returns:
        Structure tensor field (H, W, 2, 2)
    """
    if image.ndim == 3:
        image = rgb_to_gray(image)

    # Compute gradients
    grad_x = gaussian_filter(image, sigma, order=[0, 1])
    grad_y = gaussian_filter(image, sigma, order=[1, 0])

    # Structure tensor components
    J11 = gaussian_filter(grad_x * grad_x, sigma)
    J12 = gaussian_filter(grad_x * grad_y, sigma)
    J22 = gaussian_filter(grad_y * grad_y, sigma)

    # Build tensor field
    H, W = image.shape
    tensor_field = np.zeros((H, W, 2, 2))
    tensor_field[:, :, 0, 0] = J11
    tensor_field[:, :, 0, 1] = J12
    tensor_field[:, :, 1, 0] = J12
    tensor_field[:, :, 1, 1] = J22

    return tensor_field
```

#### Edge Strength and Orientation
```python
def analyze_local_structure(tensor: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract edge strength, orientation, and coherence from structure tensor.

    Args:
        tensor: Structure tensor field (H, W, 2, 2)

    Returns:
        edge_strength: (H, W) edge magnitude
        orientation: (H, W) dominant orientation in [0, π)
        coherence: (H, W) coherence measure [0, 1]
    """
    H, W = tensor.shape[:2]
    edge_strength = np.zeros((H, W))
    orientation = np.zeros((H, W))
    coherence = np.zeros((H, W))

    for i in range(H):
        for j in range(W):
            T = tensor[i, j]
            eigenvals, eigenvecs = np.linalg.eigh(T)

            # Sort eigenvalues (largest first)
            idx = np.argsort(eigenvals)[::-1]
            lambda1, lambda2 = eigenvals[idx]

            # Edge strength (largest eigenvalue)
            edge_strength[i, j] = np.sqrt(lambda1)

            # Orientation (principal eigenvector angle)
            principal_vec = eigenvecs[:, idx[0]]
            orientation[i, j] = np.arctan2(principal_vec[1], principal_vec[0]) % np.pi

            # Coherence measure
            if lambda1 + lambda2 > 1e-10:
                coherence[i, j] = (lambda1 - lambda2) / (lambda1 + lambda2)

    return edge_strength, orientation, coherence
```

### Gradient-Based Optimization

#### Manual Gradient Computation
```python
def compute_position_gradient(gaussian: AdaptiveGaussian2D,
                            error_map: np.ndarray,
                            image_size: Tuple[int, int]) -> np.ndarray:
    """
    Compute gradient of reconstruction error w.r.t. Gaussian position.

    Args:
        gaussian: Gaussian parameters
        error_map: Per-pixel reconstruction error (H, W)
        image_size: (H, W) image dimensions

    Returns:
        Position gradient [dx, dy]
    """
    H, W = image_size

    # Convert normalized position to pixel coordinates
    px_pos = gaussian.mu * np.array([W, H])

    # Sample error in local neighborhood
    radius = 3  # pixels
    y_min, y_max = max(0, int(px_pos[1] - radius)), min(H, int(px_pos[1] + radius + 1))
    x_min, x_max = max(0, int(px_pos[0] - radius)), min(W, int(px_pos[0] + radius + 1))

    # Compute weighted gradient
    grad_x, grad_y = 0.0, 0.0
    total_weight = 0.0

    cov = gaussian.covariance_matrix
    cov_inv = np.linalg.inv(cov)

    for y in range(y_min, y_max):
        for x in range(x_min, x_max):
            pixel_pos = np.array([x, y])
            delta = pixel_pos - px_pos

            # Gaussian weight at this pixel
            weight = evaluate_gaussian_2d(np.zeros(2), cov_inv, delta)

            if weight > 1e-6:
                error = error_map[y, x]

                # Gradient contribution
                grad_contribution = -weight * error * cov_inv @ delta
                grad_x += grad_contribution[0]
                grad_y += grad_contribution[1]
                total_weight += weight

    if total_weight > 1e-10:
        grad_x /= total_weight
        grad_y /= total_weight

    # Convert back to normalized coordinates
    return np.array([grad_x / W, grad_y / H])
```

#### Scale and Rotation Gradients
```python
def compute_scale_gradient(gaussian: AdaptiveGaussian2D,
                          error_map: np.ndarray,
                          image_size: Tuple[int, int]) -> np.ndarray:
    """Compute gradient w.r.t. inverse scale parameters."""
    # Similar structure to position gradient
    # but derivatives w.r.t. inv_s parameters
    pass

def compute_rotation_gradient(gaussian: AdaptiveGaussian2D,
                            error_map: np.ndarray,
                            image_size: Tuple[int, int]) -> float:
    """Compute gradient w.r.t. rotation angle."""
    # Derivative of covariance matrix w.r.t. theta
    pass
```

### Tile-Based Rendering

#### Spatial Binning Algorithm
```python
def compute_tile_bins(gaussians: List[AdaptiveGaussian2D],
                     image_size: Tuple[int, int],
                     tile_size: int = 16) -> Dict[Tuple[int, int], List[int]]:
    """
    Bin Gaussians into spatial tiles for efficient rendering.

    Args:
        gaussians: List of Gaussian splats
        image_size: (H, W) image dimensions
        tile_size: Size of each tile in pixels

    Returns:
        Dictionary mapping tile coordinates to Gaussian indices
    """
    H, W = image_size
    tile_to_gaussians = defaultdict(list)

    for g_idx, gaussian in enumerate(gaussians):
        # Compute 3σ radius in pixels
        cov = gaussian.covariance_matrix
        eigenvals = np.linalg.eigvals(cov)
        radius_px = 3.0 * np.sqrt(max(eigenvals)) * min(H, W)

        # Convert center to pixel coordinates
        center_px = gaussian.mu * np.array([W, H])

        # Find intersecting tiles
        tile_x_min = max(0, int((center_px[0] - radius_px) // tile_size))
        tile_x_max = min((W - 1) // tile_size, int((center_px[0] + radius_px) // tile_size))
        tile_y_min = max(0, int((center_px[1] - radius_px) // tile_size))
        tile_y_max = min((H - 1) // tile_size, int((center_px[1] + radius_px) // tile_size))

        for tile_y in range(tile_y_min, tile_y_max + 1):
            for tile_x in range(tile_x_min, tile_x_max + 1):
                tile_to_gaussians[(tile_y, tile_x)].append(g_idx)

    return dict(tile_to_gaussians)
```

#### Top-K Per-Pixel Rendering
```python
def render_tile_top_k(gaussians: List[AdaptiveGaussian2D],
                     gaussian_indices: List[int],
                     tile_bounds: Tuple[int, int, int, int],
                     image_size: Tuple[int, int],
                     top_k: int = 8) -> np.ndarray:
    """
    Render a single tile using top-K Gaussian blending.

    Args:
        gaussians: All Gaussian splats
        gaussian_indices: Indices of Gaussians affecting this tile
        tile_bounds: (y_min, y_max, x_min, x_max) tile boundaries
        image_size: (H, W) full image dimensions
        top_k: Maximum Gaussians to blend per pixel

    Returns:
        Rendered tile (tile_h, tile_w, channels)
    """
    y_min, y_max, x_min, x_max = tile_bounds
    H, W = image_size
    tile_h, tile_w = y_max - y_min, x_max - x_min

    # Determine number of channels from first Gaussian
    channels = len(gaussians[0].color) if gaussians else 3
    output = np.zeros((tile_h, tile_w, channels))

    for tile_y in range(tile_h):
        for tile_x in range(tile_w):
            # Convert to global pixel coordinates
            global_x = x_min + tile_x
            global_y = y_min + tile_y
            pixel_pos = np.array([global_x, global_y])

            # Evaluate all candidate Gaussians
            values = []
            colors = []

            for g_idx in gaussian_indices:
                gaussian = gaussians[g_idx]

                # Convert Gaussian center to pixels
                center_px = gaussian.mu * np.array([W, H])

                # Evaluate Gaussian at this pixel
                cov_inv = np.linalg.inv(gaussian.covariance_matrix)
                value = evaluate_gaussian_2d(center_px, cov_inv, pixel_pos)

                if value > 1e-6:  # Skip negligible contributions
                    values.append(value)
                    colors.append(gaussian.color)

            # Blend top-K contributions
            if values:
                # Sort by value (descending)
                sorted_indices = np.argsort(values)[::-1]

                # Take top-K
                k = min(top_k, len(values))
                top_indices = sorted_indices[:k]

                # Normalize weights
                top_values = [values[i] for i in top_indices]
                total_weight = sum(top_values)

                if total_weight > 1e-10:
                    weights = [v / total_weight for v in top_values]

                    # Weighted color blend
                    final_color = np.zeros(channels)
                    for i, idx in enumerate(top_indices):
                        final_color += weights[i] * colors[idx]

                    output[tile_y, tile_x] = final_color

    return output
```

## Data Structures

### Core Classes
```python
@dataclass
class AdaptiveGaussian2D:
    """Anisotropic 2D Gaussian with full covariance matrix support."""

    # Core parameters (optimizable)
    mu: np.ndarray           # Position [x, y] in [0,1]² normalized coordinates
    inv_s: np.ndarray        # Inverse scales [1/sx, 1/sy] for stable optimization
    theta: float             # Rotation angle in [0, π)
    color: np.ndarray        # RGB or multi-channel color

    # Derived properties
    @property
    def covariance_matrix(self) -> np.ndarray:
        """Compute 2x2 covariance matrix from parameters."""
        return build_covariance_matrix(self.inv_s, self.theta)

    @property
    def aspect_ratio(self) -> float:
        """Compute anisotropy ratio from scales."""
        return max(self.inv_s) / min(self.inv_s)

    @property
    def orientation(self) -> float:
        """Return rotation angle (same as theta)."""
        return self.theta

    def clip_parameters(self) -> None:
        """Clip parameters to valid ranges."""
        self.mu = np.clip(self.mu, 0.0, 1.0)
        self.inv_s = np.clip(self.inv_s, 1e-3, 1e3)  # Prevent degenerate scales
        self.theta = self.theta % np.pi  # Keep in [0, π)
        self.color = np.clip(self.color, 0.0, 1.0)
```

### Optimization State
```python
@dataclass
class OptimizationState:
    """Track optimization progress and parameters."""

    iteration: int
    learning_rates: Dict[str, float]
    error_history: List[float]
    convergence_threshold: float
    max_iterations: int

    def is_converged(self) -> bool:
        """Check if optimization has converged."""
        if len(self.error_history) < 10:
            return False

        recent_errors = self.error_history[-10:]
        error_change = abs(recent_errors[-1] - recent_errors[0])
        return error_change < self.convergence_threshold
```

## Algorithm Implementation

### Initialization Pipeline
```python
def adaptive_initialization(image: np.ndarray,
                          target_splats: int,
                          base_scale_px: float = 5.0,
                          gradient_weight: float = 0.7) -> List[AdaptiveGaussian2D]:
    """
    Initialize Gaussians using content-adaptive placement.

    Args:
        image: Input image (H, W, C)
        target_splats: Target number of splats
        base_scale_px: Base scale in pixels
        gradient_weight: Weight for gradient-based placement [0,1]

    Returns:
        List of initialized Gaussian splats
    """
    H, W, C = image.shape

    # Compute gradient magnitude
    if C > 1:
        gray = rgb_to_gray(image)
    else:
        gray = image[:, :, 0]

    grad_x = np.gradient(gray, axis=1)
    grad_y = np.gradient(gray, axis=0)
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)

    # Create probability map
    grad_normalized = grad_magnitude / (np.max(grad_magnitude) + 1e-10)
    uniform_prob = np.ones_like(grad_normalized) / (H * W)

    prob_map = gradient_weight * grad_normalized + (1 - gradient_weight) * uniform_prob
    prob_map = prob_map / np.sum(prob_map)

    # Sample positions
    positions = sample_from_probability_map(prob_map, target_splats // 2)  # Start with half

    # Compute structure tensor for orientation
    structure_tensor = compute_structure_tensor(gray)
    edge_strength, orientation, coherence = analyze_local_structure(structure_tensor)

    # Initialize Gaussians
    gaussians = []
    for pos in positions:
        y, x = pos

        # Get local properties
        local_orientation = orientation[y, x]
        local_coherence = coherence[y, x]
        local_color = image[y, x]

        # Determine initial anisotropy
        if local_coherence > 0.3:  # High coherence → edge region
            initial_aspect = 1.0 + 2.0 * local_coherence  # Up to 3:1 aspect ratio
            sx = base_scale_px
            sy = base_scale_px / initial_aspect
        else:  # Low coherence → isotropic
            sx = sy = base_scale_px

        gaussian = AdaptiveGaussian2D(
            mu=np.array([x / W, y / H]),  # Normalize to [0,1]²
            inv_s=np.array([1/sx, 1/sy]),
            theta=local_orientation,
            color=local_color
        )

        gaussians.append(gaussian)

    return gaussians
```

### Progressive Optimization Loop
```python
def progressive_optimization(gaussians: List[AdaptiveGaussian2D],
                           target_image: np.ndarray,
                           max_splats: int,
                           max_iterations: int = 1000) -> List[AdaptiveGaussian2D]:
    """
    Optimize Gaussian parameters with progressive splat addition.

    Args:
        gaussians: Initial Gaussian list
        target_image: Target image to reconstruct
        max_splats: Maximum number of splats
        max_iterations: Maximum optimization iterations

    Returns:
        Optimized Gaussian list
    """
    H, W, C = target_image.shape

    # Learning rates
    lr_mu = 5e-4
    lr_inv_s = 2e-3
    lr_theta = 2e-3
    lr_color = 5e-3

    # Initialize renderer
    renderer = TileRenderer(tile_size=16, top_k=8)

    # Optimization state
    state = OptimizationState(
        iteration=0,
        learning_rates={'mu': lr_mu, 'inv_s': lr_inv_s, 'theta': lr_theta, 'color': lr_color},
        error_history=[],
        convergence_threshold=1e-4,
        max_iterations=max_iterations
    )

    for iteration in range(max_iterations):
        # Render current splats
        rendered = renderer.render(gaussians, (H, W))

        # Compute reconstruction error
        error_map = np.abs(rendered - target_image).mean(axis=2)
        total_error = np.mean(error_map)
        state.error_history.append(total_error)

        # Update parameters
        for gaussian in gaussians:
            # Compute gradients
            grad_mu = compute_position_gradient(gaussian, error_map, (H, W))
            grad_inv_s = compute_scale_gradient(gaussian, error_map, (H, W))
            grad_theta = compute_rotation_gradient(gaussian, error_map, (H, W))
            grad_color = compute_color_gradient(gaussian, error_map, (H, W))

            # Apply updates
            gaussian.mu -= lr_mu * grad_mu
            gaussian.inv_s -= lr_inv_s * grad_inv_s
            gaussian.theta -= lr_theta * grad_theta
            gaussian.color -= lr_color * grad_color

            # Clip parameters
            gaussian.clip_parameters()

        # Progressive splat addition
        if (iteration + 1) % 100 == 0 and len(gaussians) < max_splats:
            n_add = min(max_splats // 10, max_splats - len(gaussians))
            new_gaussians = add_splats_at_high_error(error_map, n_add, target_image)
            gaussians.extend(new_gaussians)

        # Check convergence
        if state.is_converged():
            print(f"Converged after {iteration + 1} iterations")
            break

        state.iteration = iteration + 1

    return gaussians
```

## Performance Considerations

### Memory Management
- Use NumPy views and in-place operations where possible
- Implement tile-based processing to limit memory footprint
- Cache covariance matrices and inverse computations
- Use float32 precision where appropriate

### Numerical Stability
- Optimize inverse scales instead of scales directly
- Use robust eigendecomposition with fallback for degenerate cases
- Implement parameter clipping to prevent numerical overflow
- Add regularization terms to prevent degenerate configurations

### Computational Efficiency
- Precompute tile binning and update only when splats are added
- Use spatial acceleration structures for neighbor queries
- Implement early termination for negligible Gaussian contributions
- Profile critical paths and optimize bottlenecks

This technical specification provides the mathematical foundation and algorithmic details needed to implement the adaptive Gaussian splatting system with research-grade quality and performance.