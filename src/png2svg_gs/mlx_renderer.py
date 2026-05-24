"""Optional MLX batched renderer for Apple Silicon experiments.

This module intentionally stays out of the default import path. Import it only
in environments where `mlx` is installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np

from .splat import GaussianSplat, render_importance_for_raw

try:  # pragma: no cover - exercised only in MLX-enabled environments.
    import mlx.core as mx
except Exception:  # pragma: no cover - import guard for optional dependency.
    mx = None  # type: ignore[assignment]


ArrayLike = Union[np.ndarray, Any]


@dataclass(frozen=True)
class MlxTilePlan:
    """Static tile/splat lookup table for one image geometry."""

    indices: Any
    mask: Any
    order: Any
    tiles_x: int
    tiles_y: int
    max_active: int
    tile_size: int


def is_mlx_available() -> bool:
    """Return whether MLX imported successfully."""

    return mx is not None


def _require_mlx() -> Any:
    if mx is None:
        raise RuntimeError(
            "MLX is not installed. Install `mlx` to use mlx-batched rendering."
        )
    return mx


def splats_to_numpy_table(splats: Sequence[GaussianSplat]) -> np.ndarray:
    """Convert splats to the canonical float32 table [N, 11]."""

    if not splats:
        return np.zeros((0, 11), dtype=np.float32)

    rows = np.empty((len(splats), 11), dtype=np.float32)
    for idx, splat in enumerate(splats):
        raw = splat.to_raw_splat()
        rows[idx, 0] = float(raw.x)
        rows[idx, 1] = float(raw.y)
        rows[idx, 2] = float(raw.sx)
        rows[idx, 3] = float(raw.sy)
        rows[idx, 4] = float(raw.theta)
        rows[idx, 5] = 0.0
        rows[idx, 6] = float(raw.r)
        rows[idx, 7] = float(raw.g)
        rows[idx, 8] = float(raw.b)
        rows[idx, 9] = float(raw.a)
        rows[idx, 10] = render_importance_for_raw(raw)
    return rows


def _as_numpy_table(table: ArrayLike) -> np.ndarray:
    if isinstance(table, np.ndarray):
        out = table.astype(np.float32, copy=False)
    else:
        out = np.asarray(table, dtype=np.float32)
    if out.ndim != 2 or out.shape[1] != 11:
        raise ValueError("splat table must have shape [N, 11]")
    return out


class MlxBatchedGaussianRenderer:
    """Forward MLX renderer using batched tiles.

    The renderer mirrors `TorchBatchedGaussianRenderer` alpha-over and weighted
    blend math closely enough for parity benchmarking. Tile bins are static per
    render plan and are currently built on CPU from the supplied table.
    """

    def __init__(
        self,
        width: int,
        height: int,
        tile_size: int = 16,
        batch_tile_count: int = 16,
        blend_mode: str = "alpha-over",
        background_color: Optional[Sequence[float]] = None,
        culling_sigma: float = 3.0,
        max_active_splats_per_tile: Optional[int] = None,
    ):
        _require_mlx()
        self.width = int(width)
        self.height = int(height)
        self.tile_size = int(max(1, tile_size))
        self.batch_tile_count = int(max(1, batch_tile_count))
        self.blend_mode = str(blend_mode).strip().lower()
        if self.blend_mode not in {"alpha-over", "weighted"}:
            raise ValueError(f"Unsupported blend mode: {blend_mode}")
        self.culling_sigma = float(max(1.0, culling_sigma))
        if max_active_splats_per_tile is None:
            self.max_active_splats_per_tile = None
        else:
            self.max_active_splats_per_tile = int(max(1, max_active_splats_per_tile))
        if background_color is None:
            background = np.zeros(3, dtype=np.float32)
        else:
            background = np.asarray(background_color, dtype=np.float32).reshape(-1)
            if background.size != 3:
                raise ValueError("background_color must have exactly 3 values")
            background = np.clip(background, 0.0, 1.0)
        self.background = mx.array(background.astype(np.float32))  # type: ignore[union-attr]
        self._black_background = bool(np.max(np.abs(background)) <= 1e-8)

    def build_plan(self, table: ArrayLike) -> MlxTilePlan:
        """Build a static tile plan from current splat geometry."""

        mlx = _require_mlx()
        table_np = _as_numpy_table(table)
        tiles_x = (self.width + self.tile_size - 1) // self.tile_size
        tiles_y = (self.height + self.tile_size - 1) // self.tile_size
        num_tiles = tiles_x * tiles_y

        if table_np.shape[0] == 0:
            return MlxTilePlan(
                indices=mlx.zeros((num_tiles, 0), dtype=mlx.int32),
                mask=mlx.zeros((num_tiles, 0), dtype=mlx.float32),
                order=mlx.zeros((0,), dtype=mlx.int32),
                tiles_x=tiles_x,
                tiles_y=tiles_y,
                max_active=0,
                tile_size=self.tile_size,
            )

        order_np = np.argsort(table_np[:, 10], kind="stable").astype(np.int32)
        sorted_table = table_np[order_np]

        radius = self.culling_sigma * np.maximum(sorted_table[:, 2], sorted_table[:, 3])
        x_min = np.clip(
            np.floor((sorted_table[:, 0] - radius) / self.tile_size).astype(np.int64),
            0,
            tiles_x - 1,
        )
        x_max = np.clip(
            np.floor((sorted_table[:, 0] + radius) / self.tile_size).astype(np.int64),
            0,
            tiles_x - 1,
        )
        y_min = np.clip(
            np.floor((sorted_table[:, 1] - radius) / self.tile_size).astype(np.int64),
            0,
            tiles_y - 1,
        )
        y_max = np.clip(
            np.floor((sorted_table[:, 1] + radius) / self.tile_size).astype(np.int64),
            0,
            tiles_y - 1,
        )

        # Vectorized tile assignment: build (splat_idx, tile_idx) pairs via a
        # repeat-and-broadcast over each splat's bounding tile-range, then
        # group by tile via stable argsort.
        widths = (x_max - x_min + 1).astype(np.int64)
        heights = (y_max - y_min + 1).astype(np.int64)
        counts = widths * heights
        total_pairs = int(counts.sum())
        if total_pairs == 0:
            max_active = 0
        else:
            splat_ids = np.repeat(
                np.arange(sorted_table.shape[0], dtype=np.int64), counts
            )
            # Per-pair local offset within each splat's [width*height] bbox grid.
            within = np.arange(total_pairs, dtype=np.int64) - np.repeat(
                np.concatenate(([0], np.cumsum(counts[:-1]))), counts
            )
            widths_per_pair = np.repeat(widths, counts)
            ty_offsets = within // widths_per_pair
            tx_offsets = within - ty_offsets * widths_per_pair
            ty = np.repeat(y_min, counts) + ty_offsets
            tx = np.repeat(x_min, counts) + tx_offsets
            tile_ids = ty * tiles_x + tx
            # Group pairs by tile_id, stable to preserve importance order.
            sort_idx = np.argsort(tile_ids, kind="stable")
            tile_ids_sorted = tile_ids[sort_idx]
            splat_ids_sorted = splat_ids[sort_idx]
            tile_counts = np.bincount(tile_ids_sorted, minlength=num_tiles)
            max_active = int(tile_counts.max())
            if self.max_active_splats_per_tile is not None:
                max_active = min(max_active, self.max_active_splats_per_tile)

        if max_active <= 0:
            indices_np = np.zeros((num_tiles, 0), dtype=np.int32)
            mask_np = np.zeros((num_tiles, 0), dtype=np.float32)
        else:
            indices_np = np.zeros((num_tiles, max_active), dtype=np.int32)
            mask_np = np.zeros((num_tiles, max_active), dtype=np.float32)
            # Per-pair "slot index" within each tile (0..count-1), capped at max_active.
            tile_starts = np.concatenate(([0], np.cumsum(tile_counts[:-1])))
            slot_idx = np.arange(total_pairs, dtype=np.int64) - np.repeat(
                tile_starts, tile_counts
            )
            keep = slot_idx < max_active
            kept_tiles = tile_ids_sorted[keep]
            kept_splats = splat_ids_sorted[keep]
            kept_slots = slot_idx[keep]
            indices_np[kept_tiles, kept_slots] = kept_splats.astype(np.int32)
            mask_np[kept_tiles, kept_slots] = 1.0

        return MlxTilePlan(
            indices=mlx.array(indices_np),
            mask=mlx.array(mask_np),
            order=mlx.array(order_np),
            tiles_x=tiles_x,
            tiles_y=tiles_y,
            max_active=max_active,
            tile_size=self.tile_size,
        )

    def render(self, table: ArrayLike, plan: Optional[MlxTilePlan] = None) -> Any:
        """Render a canonical splat table to an MLX image [H, W, 3]."""

        mlx = _require_mlx()
        table_mx = mlx.array(table) if isinstance(table, np.ndarray) else table
        if plan is None:
            plan = self.build_plan(table)
        if plan.max_active == 0:
            return mlx.broadcast_to(
                mlx.reshape(self.background, (1, 1, 3)),
                (self.height, self.width, 3),
            )

        sorted_table = table_mx[plan.order]
        local_y, local_x = mlx.meshgrid(
            mlx.arange(self.tile_size, dtype=mlx.float32),
            mlx.arange(self.tile_size, dtype=mlx.float32),
            indexing="ij",
        )
        local = mlx.stack([local_x, local_y], axis=-1)
        num_tiles = plan.tiles_x * plan.tiles_y
        tile_ids_all = mlx.arange(num_tiles, dtype=mlx.int32)
        outputs = []

        for start in range(0, num_tiles, self.batch_tile_count):
            end = min(start + self.batch_tile_count, num_tiles)
            ids = tile_ids_all[start:end]
            outputs.append(self._render_tile_batch(ids, local, sorted_table, plan))

        tiles = mlx.concatenate(outputs, axis=0)
        padded = mlx.reshape(
            tiles,
            (plan.tiles_y, plan.tiles_x, self.tile_size, self.tile_size, 3),
        )
        padded = mlx.transpose(padded, (0, 2, 1, 3, 4))
        image = mlx.reshape(
            padded,
            (plan.tiles_y * self.tile_size, plan.tiles_x * self.tile_size, 3),
        )
        return image[: self.height, : self.width, :]

    def _render_tile_batch(
        self,
        ids: Any,
        local: Any,
        sorted_table: Any,
        plan: MlxTilePlan,
    ) -> Any:
        mlx = _require_mlx()
        batch_size = ids.shape[0]
        tile_y = ids // plan.tiles_x
        tile_x = ids - tile_y * plan.tiles_x
        origins = mlx.stack(
            [
                tile_x.astype(mlx.float32) * self.tile_size,
                tile_y.astype(mlx.float32) * self.tile_size,
            ],
            axis=-1,
        )
        coords = mlx.expand_dims(local, 0) + mlx.reshape(origins, (batch_size, 1, 1, 2))

        active_idx = plan.indices[ids]
        active_mask = plan.mask[ids]
        active = sorted_table[active_idx]
        mu = active[:, :, 0:2]
        sx = mlx.maximum(active[:, :, 2], 1e-4)
        sy = mlx.maximum(active[:, :, 3], 1e-4)
        theta = active[:, :, 4]
        colors = active[:, :, 6:9]
        alphas = active[:, :, 9]

        delta = mlx.expand_dims(coords, 3) - mlx.reshape(
            mu,
            (batch_size, 1, 1, plan.max_active, 2),
        )
        dx = delta[..., 0]
        dy = delta[..., 1]
        cos_t = mlx.reshape(mlx.cos(theta), (batch_size, 1, 1, plan.max_active))
        sin_t = mlx.reshape(mlx.sin(theta), (batch_size, 1, 1, plan.max_active))
        u = cos_t * dx + sin_t * dy
        v = -sin_t * dx + cos_t * dy
        inv_sx2 = 1.0 / mlx.square(mlx.reshape(sx, (batch_size, 1, 1, plan.max_active)))
        inv_sy2 = 1.0 / mlx.square(mlx.reshape(sy, (batch_size, 1, 1, plan.max_active)))
        weights = mlx.exp(-0.5 * (u * u * inv_sx2 + v * v * inv_sy2))
        weights = weights * mlx.reshape(
            active_mask, (batch_size, 1, 1, plan.max_active)
        )

        if self.blend_mode == "weighted":
            return self._render_weighted_batch(
                weights, colors, alphas, batch_size, plan.max_active
            )
        return self._render_alpha_over_batch(
            weights, colors, alphas, batch_size, plan.max_active
        )

    def _render_weighted_batch(
        self,
        weights: Any,
        colors: Any,
        alphas: Any,
        batch_size: int,
        max_active: int,
    ) -> Any:
        mlx = _require_mlx()
        weighted = weights * mlx.reshape(alphas, (batch_size, 1, 1, max_active))
        total_weight = mlx.sum(weighted, axis=-1, keepdims=True)
        weighted_colors = mlx.expand_dims(weighted, -1) * mlx.reshape(
            colors,
            (batch_size, 1, 1, max_active, 3),
        )
        normalized = mlx.sum(weighted_colors, axis=3) / mlx.maximum(total_weight, 1e-8)
        if self._black_background:
            return mlx.clip(normalized, 0.0, 1.0)
        coverage = mlx.clip(total_weight, 0.0, 1.0)
        background = mlx.reshape(self.background, (1, 1, 1, 3))
        return mlx.clip(coverage * normalized + (1.0 - coverage) * background, 0.0, 1.0)

    def _render_alpha_over_batch(
        self,
        weights: Any,
        colors: Any,
        alphas: Any,
        batch_size: int,
        max_active: int,
    ) -> Any:
        mlx = _require_mlx()
        density = mlx.maximum(
            weights * mlx.reshape(alphas, (batch_size, 1, 1, max_active)), 0.0
        )
        alpha_layers = 1.0 - mlx.exp(-density)
        one_minus = mlx.clip(1.0 - alpha_layers, 1e-6, 1.0)
        seed = mlx.ones(
            (batch_size, self.tile_size, self.tile_size, 1),
            dtype=mlx.float32,
        )
        prefix = mlx.cumprod(mlx.concatenate([seed, one_minus], axis=-1), axis=-1)[
            ..., :-1
        ]
        contributions = prefix * alpha_layers
        output = mlx.sum(
            mlx.expand_dims(contributions, -1)
            * mlx.reshape(colors, (batch_size, 1, 1, max_active, 3)),
            axis=3,
        )
        remaining = mlx.prod(one_minus, axis=-1, keepdims=True)
        background = mlx.reshape(self.background, (1, 1, 1, 3))
        return mlx.clip(output + remaining * background, 0.0, 1.0)


__all__ = [
    "MlxBatchedGaussianRenderer",
    "MlxTilePlan",
    "is_mlx_available",
    "splats_to_numpy_table",
]
