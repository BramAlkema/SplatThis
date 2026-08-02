"""Optional MLX batched renderer for Apple Silicon experiments.

This module intentionally stays out of the default import path. Import it only
in environments where `mlx` is installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Union

import numpy as np
import numpy.typing as npt

from .export_common import PPTX_GRADIENT_ALPHA_SCALE, PPTX_PROXY_MODES
from .mlx_runtime import is_mlx_available, require_mlx
from .splat import GaussianSplat, render_importance_for_raw

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


def _require_mlx() -> Any:
    return require_mlx("MLX batched rendering")


def mlx_linear_to_srgb(x: Any) -> Any:
    """Differentiable linear-RGB -> sRGB (gamma encode), values in [0,1].

    Mirrors `torch_linear_to_srgb` in renderer.py so MLX-trained splats see
    the same compositing math as torch-trained ones when compositing_space="srgb".
    """
    mlx = _require_mlx()
    x_clipped = mlx.clip(x, 0.0, 1.0)
    safe = mlx.maximum(x_clipped, 1e-12)
    return mlx.where(
        x_clipped <= 0.0031308,
        12.92 * x_clipped,
        1.055 * mlx.power(safe, 1.0 / 2.4) - 0.055,
    )


def mlx_srgb_to_linear(x: Any) -> Any:
    """Differentiable sRGB -> linear-RGB (gamma decode), values in [0,1]."""
    mlx = _require_mlx()
    x_clipped = mlx.clip(x, 0.0, 1.0)
    return mlx.where(
        x_clipped <= 0.04045,
        x_clipped / 12.92,
        mlx.power((x_clipped + 0.055) / 1.055, 2.4),
    )


def splats_to_numpy_table(splats: Sequence[GaussianSplat]) -> npt.NDArray[Any]:
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


def _as_numpy_table(table: ArrayLike) -> npt.NDArray[Any]:
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
        batch_tile_count: int = 8,
        blend_mode: str = "alpha-over",
        normalized_top_k: int = 10,
        background_color: Optional[Sequence[float]] = None,
        culling_sigma: float = 3.0,
        max_active_splats_per_tile: Optional[int] = None,
        compositing_space: str = "linear",
        pptx_proxy: str = "none",
        pptx_alpha_scale: float = 0.25,
        pptx_gradient_alpha_scale: float = PPTX_GRADIENT_ALPHA_SCALE,
        pptx_sigma_scale: float = 0.92,
    ):
        mlx = _require_mlx()
        self.width = int(width)
        self.height = int(height)
        self.tile_size = int(max(1, tile_size))
        self.batch_tile_count = int(max(1, batch_tile_count))
        self.blend_mode = str(blend_mode).strip().lower()
        if self.blend_mode not in {"alpha-over", "weighted", "normalized-topk"}:
            raise ValueError(f"Unsupported blend mode: {blend_mode}")
        self.normalized_top_k = int(max(1, normalized_top_k))
        self.compositing_space = str(compositing_space).strip().lower()
        if self.compositing_space not in {"linear", "srgb"}:
            raise ValueError(f"Unsupported compositing space: {compositing_space}")
        # PPTX soft-edge proxy: mirror _PPTXSoftEdgeProxyRenderer in
        # converter.py. PowerPoint renders ellipses brighter and slightly
        # softer than a true Gaussian; without this transform a PPTX export
        # of the trained splats looks washed out. With pptx_proxy="softedge",
        # render() applies sigma *= pptx_sigma_scale and rewrites alpha to the
        # value that produces center_opacity = (1 - exp(-alpha)) * pptx_alpha_scale.
        if pptx_proxy not in PPTX_PROXY_MODES:
            raise ValueError(
                f"Unsupported pptx proxy mode: {pptx_proxy!r}; "
                f"expected one of {', '.join(sorted(PPTX_PROXY_MODES))}"
            )
        self.pptx_proxy = pptx_proxy
        self.pptx_alpha_scale = float(np.clip(pptx_alpha_scale, 1e-4, 1.0))
        self.pptx_gradient_alpha_scale = float(
            np.clip(pptx_gradient_alpha_scale, 1e-4, 1.0)
        )
        self._pptx_gradient_gain: Any = None
        self.pptx_sigma_scale = float(np.clip(pptx_sigma_scale, 0.25, 3.0))
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
        self.background = mlx.array(background.astype(np.float32))
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

        # With the soft-edge proxy the render-time transform scales sigma by
        # pptx_sigma_scale AFTER planning, so the culling radius must account
        # for it here or scales > 1 under-cull and clip splat footprints.
        # Only the soft-edge proxy rescales sigma; the gradient proxy
        # touches alpha alone, so its footprint needs no plan padding.
        sigma_plan_scale = (
            float(self.pptx_sigma_scale) if self.pptx_proxy == "softedge" else 1.0
        )
        radius = (
            self.culling_sigma
            * sigma_plan_scale
            * np.maximum(sorted_table[:, 2], sorted_table[:, 3])
        )
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
            # Pairs are ordered back-to-front (ascending importance) within
            # each tile; on overload keep the LAST max_active entries so the
            # back-most splats are dropped, not the front-most ones.
            overflow = np.repeat(np.maximum(tile_counts - max_active, 0), tile_counts)
            new_slots = slot_idx - overflow
            keep = new_slots >= 0
            kept_tiles = tile_ids_sorted[keep]
            kept_splats = splat_ids_sorted[keep]
            kept_slots = new_slots[keep]
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

    def _apply_pptx_gradient_transform(self, table_mx: Any) -> Any:
        """Mirror _PPTXGradientProxyRenderer in proxies.py for the MLX path.

        Scales the alpha column so the renderer's ``1 - exp(-a * G)`` matches
        the gradient emitter's stop curve. Pure MLX ops, so it composes with
        mx.compile. Parity with the torch proxy is pinned by
        tests/unit/test_mlx_renderer.py.
        """

        mlx = _require_mlx()
        # One fused multiply by a cached gain row rather than slicing the
        # table apart and concatenating it back on every render.
        if self._pptx_gradient_gain is None:
            gain = np.ones((1, 11), dtype=np.float32)
            gain[0, 9] = self.pptx_gradient_alpha_scale
            self._pptx_gradient_gain = mlx.array(gain)
        return table_mx * self._pptx_gradient_gain

    def _apply_pptx_softedge_transform(self, table_mx: Any) -> Any:
        """Mirror _PPTXSoftEdgeProxyRenderer in converter.py for the MLX path.

        Scales sigma columns and rewrites alpha so that center_opacity ==
        (1 - exp(-raw_alpha)) * pptx_alpha_scale. Differentiable and pure
        MLX ops, so it composes with mx.compile.
        """

        mlx = _require_mlx()
        sigma_scaled = mlx.maximum(table_mx[:, 2:4] * self.pptx_sigma_scale, 1e-4)
        raw_alpha = mlx.clip(table_mx[:, 9], 0.0, 1.0)
        center_opacity = mlx.clip(
            (1.0 - mlx.exp(-raw_alpha)) * self.pptx_alpha_scale,
            0.0,
            1.0 - 1e-5,
        )
        # -log1p(-x) = -log(1 - x); MLX has no log1p but log(1 - x) is safe
        # for center_opacity clamped < 1.
        effective_alpha = -mlx.log(1.0 - center_opacity)
        return mlx.concatenate(
            [
                table_mx[:, 0:2],
                sigma_scaled,
                table_mx[:, 4:9],
                mlx.expand_dims(effective_alpha, -1),
                table_mx[:, 10:11],
            ],
            axis=1,
        )

    def render(self, table: ArrayLike, plan: Optional[MlxTilePlan] = None) -> Any:
        """Render a canonical splat table to an MLX image [H, W, 3].

        In compositing_space="srgb" mode, colors and background are encoded
        linear->sRGB before the alpha-over math and the output is decoded
        sRGB->linear so external interfaces stay linear-RGB. Mirrors the
        torch path in renderer.py:305-317.

        With pptx_proxy="softedge" the table is first run through the
        sigma/alpha proxy transform that mirrors PowerPoint's soft-edge
        rendering; see _apply_pptx_softedge_transform.
        """

        mlx = _require_mlx()
        table_mx = mlx.array(table) if isinstance(table, np.ndarray) else table
        if plan is None:
            plan = self.build_plan(table)

        if self.pptx_proxy == "softedge":
            table_mx = self._apply_pptx_softedge_transform(table_mx)
        elif self.pptx_proxy == "gradient":
            table_mx = self._apply_pptx_gradient_transform(table_mx)

        srgb_mode = self.compositing_space == "srgb"
        saved_background = self.background
        if srgb_mode:
            encoded_colors = mlx_linear_to_srgb(table_mx[:, 6:9])
            table_mx = mlx.concatenate(
                [table_mx[:, :6], encoded_colors, table_mx[:, 9:]], axis=1
            )
            self.background = mlx_linear_to_srgb(saved_background)

        try:
            if plan.max_active == 0:
                rendered = mlx.broadcast_to(
                    mlx.reshape(self.background, (1, 1, 3)),
                    (self.height, self.width, 3),
                )
                return mlx_srgb_to_linear(rendered) if srgb_mode else rendered

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
            rendered = image[: self.height, : self.width, :]
            return mlx_srgb_to_linear(rendered) if srgb_mode else rendered
        finally:
            self.background = saved_background

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
        if self.blend_mode == "normalized-topk":
            return self._render_normalized_topk_batch(
                weights, colors, batch_size, plan.max_active
            )
        return self._render_alpha_over_batch(
            weights, colors, alphas, batch_size, plan.max_active
        )

    def _render_normalized_topk_batch(
        self,
        weights: Any,
        colors: Any,
        batch_size: int,
        max_active: int,
    ) -> Any:
        """Image-GS teacher equation: normalize the K strongest responses.

        Membership selection is discrete, as in the Torch reference. Gradients
        continue through the selected Gaussian responses and colors; alpha and
        drawing order do not participate in the equation.
        """

        mlx = _require_mlx()
        k = min(self.normalized_top_k, max_active)
        split = max_active - k
        selected_indices = mlx.argpartition(weights, split, axis=-1)[..., -k:]
        selected_weights = mlx.take_along_axis(weights, selected_indices, axis=-1)
        color_table = mlx.broadcast_to(
            mlx.reshape(colors, (batch_size, 1, 1, max_active, 3)),
            (
                batch_size,
                self.tile_size,
                self.tile_size,
                max_active,
                3,
            ),
        )
        color_indices = mlx.broadcast_to(
            mlx.expand_dims(selected_indices, -1),
            (*selected_indices.shape, 3),
        )
        selected_colors = mlx.take_along_axis(color_table, color_indices, axis=3)
        total_weight = mlx.sum(selected_weights, axis=-1, keepdims=True)
        color_sum = mlx.sum(
            mlx.expand_dims(selected_weights, -1) * selected_colors,
            axis=3,
        )
        normalized = color_sum / mlx.maximum(total_weight, 1e-8)
        background = mlx.reshape(self.background, (1, 1, 1, 3))
        return mlx.where(total_weight > 1e-8, normalized, background)

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
