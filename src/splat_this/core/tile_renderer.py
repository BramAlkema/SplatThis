"""Tile-based rendering framework for adaptive Gaussian splatting.

This module implements the tile-based rendering approach from Image-GS methodology,
providing efficient spatial organization and top-K blending for anisotropic Gaussians.
"""

import math
import numpy as np
import logging
from typing import List, Tuple, Dict, Optional, Set, Any
from dataclasses import dataclass
from collections import defaultdict

from .adaptive_gaussian import AdaptiveGaussian2D

logger = logging.getLogger(__name__)


@dataclass
class RenderTile:
    """A rendering tile containing spatial information and Gaussian references."""

    x: int              # Tile x coordinate (in tile units)
    y: int              # Tile y coordinate (in tile units)
    width: int          # Tile width in pixels
    height: int         # Tile height in pixels
    pixel_x_start: int  # Starting pixel x coordinate
    pixel_y_start: int  # Starting pixel y coordinate
    pixel_x_end: int    # Ending pixel x coordinate (exclusive)
    pixel_y_end: int    # Ending pixel y coordinate (exclusive)

    # Gaussians affecting this tile
    gaussian_indices: List[int] = None

    def __post_init__(self):
        """Initialize empty Gaussian list if not provided."""
        if self.gaussian_indices is None:
            self.gaussian_indices = []

    @property
    def pixel_bounds(self) -> Tuple[int, int, int, int]:
        """Return pixel bounds as (x_start, y_start, x_end, y_end)."""
        return (self.pixel_x_start, self.pixel_y_start,
                self.pixel_x_end, self.pixel_y_end)

    @property
    def center_pixel(self) -> Tuple[float, float]:
        """Return tile center in pixel coordinates."""
        center_x = (self.pixel_x_start + self.pixel_x_end) / 2.0
        center_y = (self.pixel_y_start + self.pixel_y_end) / 2.0
        return (center_x, center_y)

    def contains_pixel(self, x: int, y: int) -> bool:
        """Check if pixel coordinates are within this tile."""
        return (self.pixel_x_start <= x < self.pixel_x_end and
                self.pixel_y_start <= y < self.pixel_y_end)


@dataclass
class RenderConfig:
    """Configuration for tile-based rendering."""

    tile_size: int = 16                    # Tile size in pixels (must be power of 2)
    max_gaussians_per_tile: int = 64       # Maximum Gaussians to consider per tile
    top_k: int = 8                         # Top-K Gaussians for final blending
    sigma_threshold: float = 3.0           # Gaussian evaluation threshold (in sigmas)
    alpha_threshold: float = 0.01          # Minimum alpha for Gaussian contribution
    enable_early_termination: bool = True  # Early alpha termination during blending
    debug_mode: bool = False               # Enable debugging output

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.tile_size <= 0 or (self.tile_size & (self.tile_size - 1)) != 0:
            raise ValueError(f"Tile size must be a positive power of 2, got {self.tile_size}")
        if self.max_gaussians_per_tile <= 0:
            raise ValueError(f"Max Gaussians per tile must be positive, got {self.max_gaussians_per_tile}")
        if self.top_k <= 0 or self.top_k > self.max_gaussians_per_tile:
            raise ValueError(f"top_k must be positive and <= max_gaussians_per_tile")
        if self.sigma_threshold <= 0:
            raise ValueError(f"Sigma threshold must be positive, got {self.sigma_threshold}")
        if not (0 < self.alpha_threshold < 1):
            raise ValueError(f"Alpha threshold must be in (0,1), got {self.alpha_threshold}")


class TileRenderer:
    """Tile-based renderer for adaptive Gaussian splats."""

    def __init__(self, image_size: Tuple[int, int], config: Optional[RenderConfig] = None):
        """
        Initialize tile-based renderer.

        Args:
            image_size: (height, width) of output image
            config: Rendering configuration
        """
        self.image_height, self.image_width = image_size
        self.config = config or RenderConfig()

        # Initialize tile grid
        self.tiles_x = math.ceil(self.image_width / self.config.tile_size)
        self.tiles_y = math.ceil(self.image_height / self.config.tile_size)
        self.total_tiles = self.tiles_x * self.tiles_y

        # Create tile grid
        self.tiles = self._create_tile_grid()

        # Cache for 3σ radii
        self._sigma_radius_cache = {}

        logger.info(f"Initialized TileRenderer: {self.image_width}x{self.image_height}, "
                   f"{self.tiles_x}x{self.tiles_y} tiles ({self.total_tiles} total)")

    def _create_tile_grid(self) -> List[List[RenderTile]]:
        """Create 2D grid of rendering tiles."""
        tiles = []

        for tile_y in range(self.tiles_y):
            tile_row = []

            for tile_x in range(self.tiles_x):
                # Compute pixel bounds
                pixel_x_start = tile_x * self.config.tile_size
                pixel_y_start = tile_y * self.config.tile_size
                pixel_x_end = min(pixel_x_start + self.config.tile_size, self.image_width)
                pixel_y_end = min(pixel_y_start + self.config.tile_size, self.image_height)

                # Actual tile dimensions (may be smaller at edges)
                width = pixel_x_end - pixel_x_start
                height = pixel_y_end - pixel_y_start

                tile = RenderTile(
                    x=tile_x, y=tile_y,
                    width=width, height=height,
                    pixel_x_start=pixel_x_start, pixel_y_start=pixel_y_start,
                    pixel_x_end=pixel_x_end, pixel_y_end=pixel_y_end
                )

                tile_row.append(tile)

            tiles.append(tile_row)

        return tiles

    def compute_3sigma_radius_px(self, gaussian: AdaptiveGaussian2D) -> float:
        """
        Compute (and cache) a conservative pixel-space 3σ radius for the given Gaussian.
        We transform covariance from normalised coords to pixel coords by scaling the eigenvectors
        with (width, height) before measuring extent. We then take the oriented-bounding radius
        as max(major_extent, minor_extent) to remain conservative for tiles.
        """
        # Cache key keeps behaviour stable and dedupes repeated queries.
        key = getattr(gaussian, "cache_key", None)
        if key is None:
            key = (id(gaussian), self.image_height, self.image_width, float(self.config.sigma_threshold))

        cached = self._sigma_radius_cache.get(key)
        if cached is not None:
            return cached

        # Guard rails
        height = int(max(0, self.image_height))
        width = int(max(0, self.image_width))
        if height == 0 or width == 0:
            self._sigma_radius_cache[key] = 0.0
            return 0.0

        # Expected: gaussian.covariance_matrix is 2x2 in normalised image coords [0,1].
        Sigma = np.asarray(gaussian.covariance_matrix, dtype=float)
        # Defensive clamp to PSD-ish if tiny negatives creep in
        Sigma = 0.5 * (Sigma + Sigma.T)

        try:
            evals, evecs = np.linalg.eigh(Sigma)
        except np.linalg.LinAlgError:
            # Worst-case fallback: use geometric mean of dimensions for better aspect ratio handling
            # This avoids the bias toward the smaller dimension that min(height, width) would cause
            geometric_mean_dim = float(np.sqrt(height * width))
            r = max(0.0, 3.0 * self.config.sigma_threshold * geometric_mean_dim)
            self._sigma_radius_cache[key] = r
            return r

        # Ensure non-negative eigenvalues (numerical noise)
        evals = np.maximum(evals, 0.0)

        # Sort ascending -> last is principal
        order = np.argsort(evals)
        evals = evals[order]
        evecs = evecs[:, order]

        # Scale eigenvectors to pixel coordinates
        S = np.array([width, height], dtype=float)

        # For each axis i, pixel-space direction length
        # std along axis i in normalised coords = sqrt(evals[i])
        # pixel extent for 1σ along that axis = sqrt(evals[i]) * || S * v_i ||
        extents_1sigma = []
        for i in (0, 1):
            v = evecs[:, i]
            v_px = v * S  # element-wise scale: (vx*W, vy*H)
            dir_len_px = float(np.sqrt(v_px[0] * v_px[0] + v_px[1] * v_px[1]))
            sigma_i = float(np.sqrt(evals[i]))
            extents_1sigma.append(sigma_i * dir_len_px)

        # Conservative oriented-bounding radius = 3σ * max of axis extents
        radius_px = 3.0 * max(extents_1sigma)

        # Preserve sigma_threshold semantics: if a threshold scales 3σ, apply it multiplicatively
        # (keeps previous call sites working the same way if they expected thresholds < 1 to cull)
        radius_px *= float(self.config.sigma_threshold)

        # Clamp to non-negative
        radius_px = max(0.0, float(radius_px))

        self._sigma_radius_cache[key] = radius_px
        return radius_px

    def assign_gaussians_to_tiles(self, gaussians: List[AdaptiveGaussian2D]) -> None:
        """
        Assign Gaussians to tiles based on their spatial extent.

        Args:
            gaussians: List of adaptive Gaussians to assign
        """
        # Clear existing assignments
        for tile_row in self.tiles:
            for tile in tile_row:
                tile.gaussian_indices.clear()

        # Clear cache for fresh computation
        self._sigma_radius_cache.clear()

        # Process each Gaussian
        for gaussian_idx, gaussian in enumerate(gaussians):
            self._assign_gaussian_to_tiles(gaussian_idx, gaussian)

        # Log assignment statistics
        if self.config.debug_mode:
            self._log_assignment_stats(gaussians)

    def _assign_gaussian_to_tiles(self, gaussian_idx: int, gaussian: AdaptiveGaussian2D) -> None:
        """Assign a single Gaussian to relevant tiles."""
        # Compute Gaussian center in pixel coordinates
        center_x_px = gaussian.mu[0] * self.image_width
        center_y_px = gaussian.mu[1] * self.image_height

        # Compute 3σ radius
        radius_px = self.compute_3sigma_radius_px(gaussian)

        # Determine tile range
        min_tile_x = max(0, int((center_x_px - radius_px) // self.config.tile_size))
        max_tile_x = min(self.tiles_x - 1, int((center_x_px + radius_px) // self.config.tile_size))
        min_tile_y = max(0, int((center_y_px - radius_px) // self.config.tile_size))
        max_tile_y = min(self.tiles_y - 1, int((center_y_px + radius_px) // self.config.tile_size))

        # Assign to relevant tiles
        for tile_y in range(min_tile_y, max_tile_y + 1):
            for tile_x in range(min_tile_x, max_tile_x + 1):
                tile = self.tiles[tile_y][tile_x]

                # Check if Gaussian actually influences this tile
                if self._gaussian_influences_tile(gaussian, tile, center_x_px, center_y_px, radius_px):
                    tile.gaussian_indices.append(gaussian_idx)

    def _gaussian_influences_tile(self, gaussian: AdaptiveGaussian2D, tile: RenderTile,
                                center_x_px: float, center_y_px: float, radius_px: float) -> bool:
        """Check if Gaussian significantly influences tile."""
        # Quick radius check
        tile_center_x, tile_center_y = tile.center_pixel
        distance = np.sqrt((center_x_px - tile_center_x)**2 + (center_y_px - tile_center_y)**2)

        if distance > radius_px + np.sqrt(tile.width**2 + tile.height**2) / 2:
            return False

        # More precise check: evaluate Gaussian at tile corners
        corners = [
            (tile.pixel_x_start, tile.pixel_y_start),
            (tile.pixel_x_end - 1, tile.pixel_y_start),
            (tile.pixel_x_start, tile.pixel_y_end - 1),
            (tile.pixel_x_end - 1, tile.pixel_y_end - 1)
        ]

        for corner_x, corner_y in corners:
            # Convert to normalized coordinates
            norm_x = corner_x / self.image_width
            norm_y = corner_y / self.image_height

            # Evaluate Gaussian
            value = gaussian.evaluate_at(np.array([norm_x, norm_y]))

            if value * gaussian.alpha > self.config.alpha_threshold:
                return True

        return False

    def render_tile(self, tile: RenderTile, gaussians: List[AdaptiveGaussian2D]) -> np.ndarray:
        """
        Render a single tile using top-K blending.

        Args:
            tile: Tile to render
            gaussians: List of all Gaussians

        Returns:
            Rendered tile as RGBA array (height, width, 4)
        """
        # Initialize output
        output = np.zeros((tile.height, tile.width, 4), dtype=np.float32)

        # Get relevant Gaussians for this tile
        tile_gaussians = [(idx, gaussians[idx]) for idx in tile.gaussian_indices]

        if not tile_gaussians:
            return output

        # Limit to max Gaussians per tile
        if len(tile_gaussians) > self.config.max_gaussians_per_tile:
            # Sort by distance to tile center and take closest
            tile_center_x, tile_center_y = tile.center_pixel
            tile_center_norm = np.array([tile_center_x / self.image_width,
                                       tile_center_y / self.image_height])

            def distance_to_tile(item):
                idx, gaussian = item
                return np.linalg.norm(gaussian.mu - tile_center_norm)

            tile_gaussians.sort(key=distance_to_tile)
            tile_gaussians = tile_gaussians[:self.config.max_gaussians_per_tile]

        # Render each pixel in the tile
        for local_y in range(tile.height):
            for local_x in range(tile.width):
                # Convert to global pixel coordinates
                global_x = tile.pixel_x_start + local_x
                global_y = tile.pixel_y_start + local_y

                # Render pixel using top-K blending
                pixel_color = self._render_pixel_top_k(
                    global_x, global_y, tile_gaussians
                )

                output[local_y, local_x] = pixel_color

        return output

    def _render_pixel_top_k(self, pixel_x: int, pixel_y: int,
                          tile_gaussians: List[Tuple[int, AdaptiveGaussian2D]]) -> np.ndarray:
        """Render single pixel using top-K Gaussian blending."""
        # Convert pixel to normalized coordinates
        norm_x = pixel_x / self.image_width
        norm_y = pixel_y / self.image_height
        pixel_pos = np.array([norm_x, norm_y])

        # Evaluate all Gaussians at this pixel
        gaussian_contributions = []

        for idx, gaussian in tile_gaussians:
            # Evaluate Gaussian value
            value = gaussian.evaluate_at(pixel_pos)
            contribution = value * gaussian.alpha

            if contribution > self.config.alpha_threshold:
                gaussian_contributions.append((contribution, gaussian))

        # Sort by contribution (highest first)
        gaussian_contributions.sort(key=lambda x: x[0], reverse=True)

        # Take top-K for blending
        top_k_contributions = gaussian_contributions[:self.config.top_k]

        if not top_k_contributions:
            return np.array([0.0, 0.0, 0.0, 0.0])  # Transparent pixel

        # Alpha blending (front-to-back)
        final_color = np.array([0.0, 0.0, 0.0])
        accumulated_alpha = 0.0

        for contribution, gaussian in top_k_contributions:
            # Current alpha with accumulated transparency
            current_alpha = contribution * (1.0 - accumulated_alpha)

            if current_alpha < self.config.alpha_threshold:
                if self.config.enable_early_termination:
                    break
                continue

            # Blend color
            final_color += current_alpha * gaussian.color[:3]
            accumulated_alpha += current_alpha

            # Early termination if nearly opaque
            if self.config.enable_early_termination and accumulated_alpha > 0.99:
                break

        return np.array([final_color[0], final_color[1], final_color[2], accumulated_alpha])

    def render_full_image(self, gaussians: List[AdaptiveGaussian2D]) -> np.ndarray:
        """
        Render complete image using tile-based approach.

        Args:
            gaussians: List of adaptive Gaussians to render

        Returns:
            Rendered image as RGBA array (height, width, 4)
        """
        logger.info(f"Rendering {len(gaussians)} Gaussians to {self.image_width}x{self.image_height} image")

        # Assign Gaussians to tiles
        self.assign_gaussians_to_tiles(gaussians)

        # Initialize output image
        output = np.zeros((self.image_height, self.image_width, 4), dtype=np.float32)

        # Render each tile
        tiles_rendered = 0
        for tile_row in self.tiles:
            for tile in tile_row:
                # Render tile
                tile_output = self.render_tile(tile, gaussians)

                # Copy to main output
                output[tile.pixel_y_start:tile.pixel_y_end,
                      tile.pixel_x_start:tile.pixel_x_end] = tile_output

                tiles_rendered += 1

                if self.config.debug_mode and tiles_rendered % 100 == 0:
                    logger.debug(f"Rendered {tiles_rendered}/{self.total_tiles} tiles")

        logger.info(f"Rendering complete: {tiles_rendered} tiles processed")
        return output

    def get_tile_at_pixel(self, pixel_x: int, pixel_y: int) -> Optional[RenderTile]:
        """Get tile containing the specified pixel."""
        tile_x = pixel_x // self.config.tile_size
        tile_y = pixel_y // self.config.tile_size

        if 0 <= tile_x < self.tiles_x and 0 <= tile_y < self.tiles_y:
            return self.tiles[tile_y][tile_x]
        return None

    def get_rendering_stats(self) -> Dict[str, Any]:
        """Get rendering statistics and debugging information."""
        total_assignments = sum(len(tile.gaussian_indices)
                              for tile_row in self.tiles
                              for tile in tile_row)

        non_empty_tiles = sum(1 for tile_row in self.tiles
                            for tile in tile_row
                            if tile.gaussian_indices)

        max_gaussians_in_tile = max((len(tile.gaussian_indices)
                                   for tile_row in self.tiles
                                   for tile in tile_row), default=0)

        avg_gaussians_per_tile = total_assignments / self.total_tiles if self.total_tiles > 0 else 0
        avg_gaussians_per_nonempty_tile = (total_assignments / non_empty_tiles
                                         if non_empty_tiles > 0 else 0)

        return {
            'image_size': (self.image_height, self.image_width),
            'tile_size': self.config.tile_size,
            'tiles_grid': (self.tiles_y, self.tiles_x),
            'total_tiles': self.total_tiles,
            'non_empty_tiles': non_empty_tiles,
            'total_gaussian_assignments': total_assignments,
            'max_gaussians_in_tile': max_gaussians_in_tile,
            'avg_gaussians_per_tile': avg_gaussians_per_tile,
            'avg_gaussians_per_nonempty_tile': avg_gaussians_per_nonempty_tile,
            'cache_size': len(self._sigma_radius_cache)
        }

    def _log_assignment_stats(self, gaussians: List[AdaptiveGaussian2D]) -> None:
        """Log Gaussian-to-tile assignment statistics."""
        stats = self.get_rendering_stats()

        logger.debug(f"Tile assignment statistics:")
        logger.debug(f"  Total Gaussians: {len(gaussians)}")
        logger.debug(f"  Non-empty tiles: {stats['non_empty_tiles']}/{stats['total_tiles']}")
        logger.debug(f"  Total assignments: {stats['total_gaussian_assignments']}")
        logger.debug(f"  Max Gaussians in tile: {stats['max_gaussians_in_tile']}")
        logger.debug(f"  Avg Gaussians per tile: {stats['avg_gaussians_per_tile']:.2f}")
        logger.debug(f"  Avg Gaussians per non-empty tile: {stats['avg_gaussians_per_nonempty_tile']:.2f}")


def create_tile_renderer(image_size: Tuple[int, int],
                        tile_size: int = 16,
                        top_k: int = 8,
                        debug_mode: bool = False) -> TileRenderer:
    """
    Convenience function to create tile renderer with common settings.

    Args:
        image_size: (height, width) of output image
        tile_size: Size of rendering tiles in pixels
        top_k: Number of top Gaussians to blend per pixel
        debug_mode: Enable debug logging

    Returns:
        Configured TileRenderer instance
    """
    config = RenderConfig(
        tile_size=tile_size,
        top_k=top_k,
        debug_mode=debug_mode
    )

    return TileRenderer(image_size, config)