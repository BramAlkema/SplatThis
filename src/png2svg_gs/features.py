"""
Feature extraction and content-adaptive seeding.

Implements gradient-based analysis and spatial organization for Gaussian placement.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
from scipy import ndimage

logger = logging.getLogger(__name__)


def _resolve_rng(rng: Optional[np.random.Generator] = None) -> np.random.Generator:
    """Return provided RNG or create a new generator."""
    return rng if rng is not None else np.random.default_rng()


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert image to float32 grayscale."""
    if image.ndim == 3:
        return (
            0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
        ).astype(np.float32)
    return np.asarray(image, dtype=np.float32)


def _compute_gradients(
    gray: np.ndarray, method: str = "sobel"
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute image gradients for a grayscale image."""
    if method == "sobel":
        grad_x = ndimage.sobel(gray, axis=1)
        grad_y = ndimage.sobel(gray, axis=0)
    elif method == "scharr":
        scharr_x = (
            np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], dtype=np.float32) / 32.0
        )
        scharr_y = (
            np.array([[-3, -10, -3], [0, 0, 0], [3, 10, 3]], dtype=np.float32) / 32.0
        )
        grad_x = ndimage.convolve(gray, scharr_x)
        grad_y = ndimage.convolve(gray, scharr_y)
    elif method == "prewitt":
        grad_x = ndimage.prewitt(gray, axis=1)
        grad_y = ndimage.prewitt(gray, axis=0)
    else:
        raise ValueError(f"Unknown gradient method: {method}")
    return grad_x.astype(np.float32), grad_y.astype(np.float32)


def compute_gradient_magnitude(image: np.ndarray, method: str = "sobel") -> np.ndarray:
    """
    Compute gradient magnitude for content-adaptive seeding.

    Args:
        image: Input image [H, W] or [H, W, C]
        method: Gradient method ('sobel', 'scharr', 'prewitt')

    Returns:
        Gradient magnitude [H, W]
    """
    gray = _to_grayscale(image)
    grad_x, grad_y = _compute_gradients(gray, method=method)

    # Compute magnitude
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    return grad_mag


def compute_structure_field(
    image: np.ndarray,
    method: str = "sobel",
    smoothing_sigma: float = 1.0,
    anisotropy_clip: float = 10.0,
    min_coherence: float = 0.12,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute full-image principal direction and anisotropy maps.

    Returns:
        primary_dirs: [H, W, 2] unit vectors.
        anisotropy: [H, W] ratio >= 1.
    """
    gray = _to_grayscale(image)
    grad_x, grad_y = _compute_gradients(gray, method=method)

    i_xx = grad_x * grad_x
    i_yy = grad_y * grad_y
    i_xy = grad_x * grad_y

    sigma = float(max(0.0, smoothing_sigma))
    if sigma > 0.0:
        i_xx = ndimage.gaussian_filter(i_xx, sigma=sigma)
        i_yy = ndimage.gaussian_filter(i_yy, sigma=sigma)
        i_xy = ndimage.gaussian_filter(i_xy, sigma=sigma)

    trace = i_xx + i_yy
    disc = np.maximum((i_xx - i_yy) ** 2 + 4.0 * (i_xy**2), 0.0)
    root = np.sqrt(disc)
    lambda1 = 0.5 * (trace + root)
    lambda2 = 0.5 * (trace - root)
    coherence = (lambda1 - lambda2) / np.maximum(lambda1 + lambda2, 1e-6)

    angle = 0.5 * np.arctan2(2.0 * i_xy, i_xx - i_yy)
    primary_dirs = np.stack([np.cos(angle), np.sin(angle)], axis=-1).astype(np.float32)

    anisotropy = lambda1 / np.maximum(lambda2, 1e-6)
    clip_max = float(max(1.0, anisotropy_clip))
    anisotropy = np.clip(anisotropy, 1.0, clip_max).astype(np.float32)
    weak_directional = coherence < float(max(0.0, min_coherence))
    if np.any(weak_directional):
        anisotropy = anisotropy.copy()
        anisotropy[weak_directional] = 1.0
        primary_dirs = primary_dirs.copy()
        primary_dirs[weak_directional, 0] = 1.0
        primary_dirs[weak_directional, 1] = 0.0
    return primary_dirs, anisotropy


def compute_laplacian_of_gaussian(image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    Compute Laplacian of Gaussian for blob detection.

    Args:
        image: Input image [H, W] or [H, W, C]
        sigma: Gaussian blur sigma

    Returns:
        LoG response [H, W]
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
    else:
        gray = image

    # Apply Gaussian blur then Laplacian
    blurred = ndimage.gaussian_filter(gray, sigma=sigma)
    log_response = ndimage.laplace(blurred)

    return np.abs(log_response)  # Take absolute value for feature strength


def init_seeds_content_adaptive(
    image: np.ndarray,
    target_count: int,
    gradient_weight: float = 0.7,
    method: str = "sobel",
    rng: Optional[np.random.Generator] = None,
) -> List[Tuple[float, float]]:
    """
    Generate content-adaptive seed positions.

    Args:
        image: Input image [H, W, C]
        target_count: Target number of seeds
        gradient_weight: Weight for gradient vs uniform sampling (0-1)
        method: Gradient computation method

    Returns:
        List of (x, y) seed positions in image coordinates
    """
    H, W = image.shape[:2]

    # Compute gradient magnitude
    grad_mag = compute_gradient_magnitude(image, method=method)

    # Create probability map
    # High gradients → high probability
    grad_prob = grad_mag / (np.sum(grad_mag) + 1e-8)

    # Uniform probability
    uniform_prob = np.ones((H, W), dtype=np.float32) / (H * W)

    # Combined probability
    combined_prob = gradient_weight * grad_prob + (1 - gradient_weight) * uniform_prob

    # Sample positions
    seeds = sample_from_probability_map(combined_prob, target_count, rng=rng)

    logger.info(f"Generated {len(seeds)} content-adaptive seeds")
    return seeds


def sample_from_probability_map(
    prob_map: np.ndarray, num_samples: int, rng: Optional[np.random.Generator] = None
) -> List[Tuple[float, float]]:
    """
    Sample positions from probability map.

    Args:
        prob_map: Probability map [H, W]
        num_samples: Number of samples to generate

    Returns:
        List of (x, y) positions in image coordinates
    """
    H, W = prob_map.shape

    rng = _resolve_rng(rng)

    # Flatten and normalize
    prob_flat = prob_map.flatten()
    prob_flat = prob_flat / (np.sum(prob_flat) + 1e-8)

    # Sample indices
    indices = rng.choice(H * W, size=num_samples, p=prob_flat, replace=True)

    # Convert to (x, y) coordinates
    positions = []
    for idx in indices:
        y, x = np.unravel_index(idx, (H, W))
        positions.append((float(x), float(y)))

    return positions


def poisson_disk_sampling(
    width: int,
    height: int,
    min_distance: float,
    max_attempts: int = 30,
    rng: Optional[np.random.Generator] = None,
) -> List[Tuple[float, float]]:
    """
    Generate Poisson disk sampling for uniform coverage.

    Args:
        width: Image width
        height: Image height
        min_distance: Minimum distance between points
        max_attempts: Maximum attempts per iteration

    Returns:
        List of (x, y) positions
    """
    # Grid for spatial lookup
    cell_size = min_distance / np.sqrt(2)
    grid_width = int(np.ceil(width / cell_size))
    grid_height = int(np.ceil(height / cell_size))
    grid = np.full((grid_height, grid_width), -1, dtype=int)

    # Result points
    points = []
    active_list = []

    # Add initial point
    x0 = rng.uniform(0, width)
    y0 = rng.uniform(0, height)
    points.append((x0, y0))
    active_list.append(0)

    grid_x = int(x0 / cell_size)
    grid_y = int(y0 / cell_size)
    grid[grid_y, grid_x] = 0

    while active_list:
        # Pick random active point
        idx = int(rng.integers(len(active_list)))
        active_idx = active_list[idx]
        px, py = points[active_idx]

        found_valid = False

        for _ in range(max_attempts):
            # Generate candidate in annulus
            angle = rng.uniform(0, 2 * np.pi)
            radius = rng.uniform(min_distance, 2 * min_distance)

            cx = px + radius * np.cos(angle)
            cy = py + radius * np.sin(angle)

            # Check bounds
            if not (0 <= cx < width and 0 <= cy < height):
                continue

            # Check grid
            grid_x = int(cx / cell_size)
            grid_y = int(cy / cell_size)

            if _is_valid_poisson_point(
                cx,
                cy,
                points,
                grid,
                grid_x,
                grid_y,
                grid_width,
                grid_height,
                cell_size,
                min_distance,
            ):
                # Add new point
                points.append((cx, cy))
                active_list.append(len(points) - 1)
                grid[grid_y, grid_x] = len(points) - 1
                found_valid = True
                break

        if not found_valid:
            # Remove from active list
            active_list.pop(idx)

    logger.info(f"Generated {len(points)} Poisson disk samples")
    return points


def _is_valid_poisson_point(
    x: float,
    y: float,
    points: List[Tuple[float, float]],
    grid: np.ndarray,
    grid_x: int,
    grid_y: int,
    grid_width: int,
    grid_height: int,
    cell_size: float,
    min_distance: float,
) -> bool:
    """Check if point is valid for Poisson disk sampling."""
    # Check neighborhood in grid
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            gx = grid_x + dx
            gy = grid_y + dy

            if 0 <= gx < grid_width and 0 <= gy < grid_height:
                point_idx = grid[gy, gx]
                if point_idx >= 0:
                    px, py = points[point_idx]
                    distance = np.sqrt((x - px) ** 2 + (y - py) ** 2)
                    if distance < min_distance:
                        return False

    return True


def create_spatial_grid(
    positions: List[Tuple[float, float]], width: int, height: int, grid_size: int = 32
) -> dict:
    """
    Create spatial grid for fast neighbor queries.

    Args:
        positions: List of (x, y) positions
        width: Image width
        height: Image height
        grid_size: Grid cell size in pixels

    Returns:
        Dictionary mapping (grid_x, grid_y) to list of position indices
    """
    grid = {}

    for i, (x, y) in enumerate(positions):
        grid_x = int(x / grid_size)
        grid_y = int(y / grid_size)

        if (grid_x, grid_y) not in grid:
            grid[(grid_x, grid_y)] = []
        grid[(grid_x, grid_y)].append(i)

    return grid


def analyze_local_structure(
    image: np.ndarray,
    x: int,
    y: int,
    window_size: int = 7,
    anisotropy_clip: float = 4.0,
    min_coherence: float = 0.12,
    min_energy: float = 1e-4,
) -> Tuple[np.ndarray, float]:
    """
    Analyze local structure tensor for orientation estimation.

    Args:
        image: Input image [H, W, C]
        x: X coordinate
        y: Y coordinate
        window_size: Analysis window size

    Returns:
        (primary_direction, anisotropy_ratio)
    """
    H, W = image.shape[:2]

    # Define window bounds
    half_size = window_size // 2
    x1 = max(0, x - half_size)
    x2 = min(W, x + half_size + 1)
    y1 = max(0, y - half_size)
    y2 = min(H, y + half_size + 1)

    # Extract local patch
    if len(image.shape) == 3:
        patch = (
            0.299 * image[y1:y2, x1:x2, 0]
            + 0.587 * image[y1:y2, x1:x2, 1]
            + 0.114 * image[y1:y2, x1:x2, 2]
        )
    else:
        patch = image[y1:y2, x1:x2]

    if patch.size < 4:  # Too small
        return np.array([1.0, 0.0]), 1.0

    # Compute gradients
    grad_x = ndimage.sobel(patch, axis=1)
    grad_y = ndimage.sobel(patch, axis=0)

    # Structure tensor components (window-averaged).
    i_xx = float(np.mean(grad_x * grad_x))
    i_yy = float(np.mean(grad_y * grad_y))
    i_xy = float(np.mean(grad_x * grad_y))

    trace = i_xx + i_yy
    if not np.isfinite(trace) or trace <= float(max(0.0, min_energy)):
        return np.array([1.0, 0.0], dtype=np.float32), 1.0

    disc = max((i_xx - i_yy) ** 2 + 4.0 * (i_xy**2), 0.0)
    root = float(np.sqrt(disc))
    lambda1 = 0.5 * (trace + root)
    lambda2 = 0.5 * (trace - root)

    coherence = (lambda1 - lambda2) / max(trace, 1e-8)
    if coherence < float(np.clip(min_coherence, 0.0, 1.0)):
        return np.array([1.0, 0.0], dtype=np.float32), 1.0

    angle = 0.5 * np.arctan2(2.0 * i_xy, i_xx - i_yy)
    primary_direction = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)

    anisotropy = lambda1 / max(lambda2, 1e-6)
    anisotropy = float(np.clip(anisotropy, 1.0, max(1.0, anisotropy_clip)))
    return primary_direction, anisotropy


def estimate_local_color(
    image: np.ndarray, x: int, y: int, window_size: int = 5
) -> np.ndarray:
    """
    Estimate local color by averaging neighborhood.

    Args:
        image: Input image [H, W, C]
        x: X coordinate
        y: Y coordinate
        window_size: Averaging window size

    Returns:
        RGB color [3]
    """
    H, W = image.shape[:2]

    # Define window bounds
    half_size = window_size // 2
    x1 = max(0, x - half_size)
    x2 = min(W, x + half_size + 1)
    y1 = max(0, y - half_size)
    y2 = min(H, y + half_size + 1)

    # Extract patch and compute mean
    if len(image.shape) == 3:
        patch = image[y1:y2, x1:x2, :3]  # RGB only
        color = np.mean(patch.reshape(-1, patch.shape[-1]), axis=0)
    else:
        # Grayscale - convert to RGB
        patch_mean = np.mean(image[y1:y2, x1:x2])
        color = np.array([patch_mean, patch_mean, patch_mean])

    return color.astype(np.float32)
