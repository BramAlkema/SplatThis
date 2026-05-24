"""Experimental MLX optimization stage runner."""

from __future__ import annotations

import logging
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .mlx_losses import MlxLossConfig, make_loss_fn
from .mlx_optimizer import (
    MlxAdam,
    MlxSplatParams,
    clone_tree,
    constrain_trainable_tree,
    table_to_splats,
    tree_to_numpy_table,
)
from .mlx_renderer import MlxBatchedGaussianRenderer, MlxTilePlan, splats_to_numpy_table
from .optimizer import DEFAULT_LEARNING_RATES
from .splat import GaussianSplat

try:  # pragma: no cover - exercised in MLX-enabled environments.
    import mlx.core as mx
except Exception:  # pragma: no cover - optional dependency guard.
    mx = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


def is_mlx_available() -> bool:
    return mx is not None


def _require_mlx() -> Any:
    if mx is None:
        raise RuntimeError(
            "MLX is not installed. Install `mlx` to use MLX stage optimization."
        )
    return mx


def _array_scalar(value: Any) -> float:
    return float(np.asarray(value))


@dataclass(frozen=True)
class MlxRendererConfig:
    tile_size: int = 16
    batch_tile_count: int = 16
    blend_mode: str = "alpha-over"
    background_color: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    culling_sigma: float = 3.0
    max_active_splats_per_tile: Optional[int] = None
    compositing_space: str = "linear"


@dataclass(frozen=True)
class MlxStageConfig:
    renderer: MlxRendererConfig = field(default_factory=MlxRendererConfig)
    loss: MlxLossConfig = field(default_factory=MlxLossConfig)
    trainable_groups: Tuple[str, ...] = ("color", "alpha")
    grad_clip_norm: Optional[float] = 1.0
    tile_plan_mode: str = "static"
    tile_plan_rebuild_interval: int = 10
    progress_interval: int = 0


@dataclass
class MlxStageResult:
    splats: Sequence[GaussianSplat]
    rendered_linear_rgb: np.ndarray
    metrics: Dict[str, Any]


def _summarize_times(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"avg_iter_sec": 0.0, "median_iter_sec": 0.0}
    return {
        "avg_iter_sec": float(statistics.mean(values)),
        "median_iter_sec": float(statistics.median(values)),
    }


def optimize_stage_mlx(
    splats: Sequence[GaussianSplat],
    target_linear_rgb: np.ndarray,
    width: int,
    height: int,
    num_iters: int,
    *,
    config: Optional[MlxStageConfig] = None,
    learning_rates: Optional[Mapping[str, float]] = None,
    spatial_weight_map: Optional[np.ndarray] = None,
    should_stop: Optional[Callable[[], bool]] = None,
    verbose: bool = False,
) -> MlxStageResult:
    """Run one MLX optimization stage.

    Static tile plans are exact only for fixed geometry, so moving parameter
    groups require periodic plan rebuilds.
    """

    mlx = _require_mlx()
    stage_config = config or MlxStageConfig()
    plan_mode = str(stage_config.tile_plan_mode).strip().lower().replace("_", "-")
    if plan_mode not in {"static", "periodic"}:
        raise ValueError(
            f"Unsupported MLX tile plan mode: {stage_config.tile_plan_mode}"
        )
    moving_groups = {"position", "scale", "theta"}.intersection(
        stage_config.trainable_groups
    )
    if plan_mode == "static" and moving_groups:
        raise ValueError(
            "Static MLX tile plans only support color/alpha optimization; "
            f"got moving group(s): {', '.join(sorted(moving_groups))}"
        )

    target_np = np.asarray(target_linear_rgb, dtype=np.float32)
    if target_np.ndim != 3 or target_np.shape[-1] < 3:
        raise ValueError("target_linear_rgb must have shape [H, W, 3]")
    target_np = np.clip(target_np[: int(height), : int(width), :3], 0.0, 1.0)
    loss_name = str(stage_config.loss.name).strip().lower().replace("_", "-")
    use_spatial_weights = loss_name.startswith("weighted")
    weights = None
    if use_spatial_weights and spatial_weight_map is not None:
        weights_np = np.asarray(spatial_weight_map, dtype=np.float32)
        if weights_np.shape != (int(height), int(width)):
            raise ValueError("spatial_weight_map shape must match target HxW")
        weights = mlx.array(np.clip(weights_np, 0.0, None))

    if not splats:
        rendered = np.zeros((int(height), int(width), 3), dtype=np.float32)
        return MlxStageResult(
            splats=[],
            rendered_linear_rgb=rendered,
            metrics={
                "optimizer_backend": "mlx",
                "start_loss": 0.0,
                "end_loss": 0.0,
                "best_loss": 0.0,
                "iterations": 0,
                "elapsed_sec": 0.0,
                "tile_plan_mode": plan_mode,
            },
        )

    table_np = splats_to_numpy_table(splats)
    params = MlxSplatParams.from_table(table_np)
    trainable = params.trainable_tree(stage_config.trainable_groups)
    trainable = constrain_trainable_tree(
        trainable, image_width=width, image_height=height
    )
    target = mlx.array(target_np)
    renderer = MlxBatchedGaussianRenderer(
        width=width,
        height=height,
        tile_size=stage_config.renderer.tile_size,
        batch_tile_count=stage_config.renderer.batch_tile_count,
        blend_mode=stage_config.renderer.blend_mode,
        background_color=stage_config.renderer.background_color,
        culling_sigma=stage_config.renderer.culling_sigma,
        max_active_splats_per_tile=stage_config.renderer.max_active_splats_per_tile,
        compositing_space=stage_config.renderer.compositing_space,
    )
    plan: Optional[MlxTilePlan] = None
    plan_build_sec = 0.0
    plan_builds_total = 0
    plan_rebuilds_in_loop = 0
    plan_rebuild_sec = 0.0
    plan_rebuild_interval = int(max(1, stage_config.tile_plan_rebuild_interval))

    def rebuild_plan(
        tree: Mapping[str, Any], *, count_as_training_rebuild: bool
    ) -> None:
        nonlocal plan, plan_build_sec, plan_builds_total, plan_rebuilds_in_loop, plan_rebuild_sec
        table = params.as_table(tree)
        mlx.eval(table)
        current_table_np = np.asarray(table, dtype=np.float32)
        plan_t0 = time.perf_counter()
        plan = renderer.build_plan(current_table_np)
        elapsed = time.perf_counter() - plan_t0
        plan_build_sec += elapsed
        plan_builds_total += 1
        if count_as_training_rebuild:
            plan_rebuilds_in_loop += 1
            plan_rebuild_sec += elapsed

    rebuild_plan(trainable, count_as_training_rebuild=False)
    loss_fn = make_loss_fn(stage_config.loss)
    lr_dict: Dict[str, float] = {
        **{k: float(v) for k, v in DEFAULT_LEARNING_RATES.items()},
        **{k: float(v) for k, v in (learning_rates or {}).items()},
    }

    tile_size_const = int(renderer.tile_size)
    tiles_x_const = (int(width) + tile_size_const - 1) // tile_size_const
    tiles_y_const = (int(height) + tile_size_const - 1) // tile_size_const
    image_width_const = int(width)
    image_height_const = int(height)
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    grad_clip_norm = stage_config.grad_clip_norm

    # Compiled train_step: render + loss + grad + Adam + constrain, fused.
    # The compile cache keys on (plan.indices.shape[1] == max_active); pin
    # `max_active_splats_per_tile` if you want zero recompiles across rebuilds.
    def _loss_with_plan(
        tree: Dict[str, Any],
        plan_indices: Any,
        plan_mask: Any,
        plan_order: Any,
    ) -> Any:
        max_active = int(plan_indices.shape[1])
        plan_inner = MlxTilePlan(
            indices=plan_indices,
            mask=plan_mask,
            order=plan_order,
            tiles_x=tiles_x_const,
            tiles_y=tiles_y_const,
            max_active=max_active,
            tile_size=tile_size_const,
        )
        rendered = renderer.render(params.as_table(tree), plan=plan_inner)
        return loss_fn(rendered, target, weights)

    _value_and_grad = mlx.value_and_grad(_loss_with_plan)

    def _train_step(
        tree: Dict[str, Any],
        m: Dict[str, Any],
        v: Dict[str, Any],
        step_count: Any,
        plan_indices: Any,
        plan_mask: Any,
        plan_order: Any,
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Any, Any, Any, Any]:
        loss, grads = _value_and_grad(tree, plan_indices, plan_mask, plan_order)
        total_sq = mlx.array(0.0, dtype=mlx.float32)
        for grad in grads.values():
            total_sq = total_sq + mlx.sum(grad * grad)
        grad_norm = mlx.sqrt(total_sq)
        if grad_clip_norm is None:
            clip_factor = mlx.array(1.0, dtype=mlx.float32)
        else:
            clip_factor = mlx.minimum(
                mlx.array(1.0, dtype=mlx.float32),
                float(grad_clip_norm) / (grad_norm + 1e-6),
            )
        step_next = step_count + 1
        step_f = step_next.astype(mlx.float32)
        bias1 = 1.0 - mlx.power(mlx.array(beta1, dtype=mlx.float32), step_f)
        bias2 = 1.0 - mlx.power(mlx.array(beta2, dtype=mlx.float32), step_f)
        new_m: Dict[str, Any] = {}
        new_v: Dict[str, Any] = {}
        new_tree: Dict[str, Any] = {}
        for key, value in tree.items():
            grad_clipped = grads[key] * clip_factor
            new_m[key] = beta1 * m[key] + (1.0 - beta1) * grad_clipped
            new_v[key] = beta2 * v[key] + (1.0 - beta2) * (grad_clipped * grad_clipped)
            m_hat = new_m[key] / bias1
            v_hat = new_v[key] / bias2
            lr = lr_dict.get(key, 0.0)
            new_tree[key] = value - lr * m_hat / (mlx.sqrt(v_hat) + eps)
        new_tree = constrain_trainable_tree(
            new_tree,
            image_width=image_width_const,
            image_height=image_height_const,
        )
        return new_tree, new_m, new_v, step_next, loss, grad_norm, clip_factor

    compiled_train_step = mlx.compile(_train_step)

    start_loss_arr = _loss_with_plan(trainable, plan.indices, plan.mask, plan.order)
    mlx.eval(start_loss_arr)
    start_loss = _array_scalar(start_loss_arr)
    best_loss_arr = start_loss_arr
    best_tree = clone_tree(trainable)
    m_state: Dict[str, Any] = {k: mlx.zeros_like(v) for k, v in trainable.items()}
    v_state: Dict[str, Any] = {k: mlx.zeros_like(v) for k, v in trainable.items()}
    step_count_arr = mlx.array(0, dtype=mlx.int32)
    loss_arr = start_loss_arr
    grad_norm_arr = mlx.array(0.0, dtype=mlx.float32)
    clip_factor_arr = mlx.array(1.0, dtype=mlx.float32)
    iter_times: List[float] = []
    last_logged_loss = start_loss
    last_logged_grad_norm = 0.0
    last_logged_clip_factor = 1.0
    stopped_for_time_budget = False

    stage_t0 = time.perf_counter()
    iterations_run = 0
    progress_interval = int(max(0, stage_config.progress_interval))
    for iteration in range(max(0, int(num_iters))):
        if should_stop is not None and should_stop():
            stopped_for_time_budget = True
            if verbose:
                logger.info(
                    "  MLX time budget exhausted at iteration %s/%s",
                    iteration,
                    num_iters,
                )
            break
        iter_t0 = time.perf_counter()
        (
            trainable,
            m_state,
            v_state,
            step_count_arr,
            loss_arr,
            grad_norm_arr,
            clip_factor_arr,
        ) = compiled_train_step(
            trainable,
            m_state,
            v_state,
            step_count_arr,
            plan.indices,
            plan.mask,
            plan.order,
        )
        # Track best entirely MLX-side; no host sync per iter.
        is_better = loss_arr < best_loss_arr
        best_loss_arr = mlx.where(is_better, loss_arr, best_loss_arr)
        best_tree = {
            key: mlx.where(is_better, trainable[key], best_tree[key])
            for key in trainable
        }
        if plan_mode == "periodic" and (iteration + 1) % plan_rebuild_interval == 0:
            rebuild_plan(trainable, count_as_training_rebuild=True)
        iter_elapsed = time.perf_counter() - iter_t0
        iter_times.append(iter_elapsed)
        iterations_run = iteration + 1

        should_log = bool(
            verbose
            and progress_interval
            and (
                iteration == 0
                or (iteration + 1) % progress_interval == 0
                or iteration + 1 == num_iters
            )
        )
        if should_log:
            mlx.eval(loss_arr, grad_norm_arr, clip_factor_arr, best_loss_arr)
            last_logged_loss = _array_scalar(loss_arr)
            last_logged_grad_norm = _array_scalar(grad_norm_arr)
            last_logged_clip_factor = _array_scalar(clip_factor_arr)
            avg_iter = statistics.mean(iter_times)
            logger.info(
                "  MLX iteration %s/%s: loss=%.6f best=%.6f iter=%.3fs avg=%.3fs",
                iteration + 1,
                num_iters,
                last_logged_loss,
                _array_scalar(best_loss_arr),
                iter_elapsed,
                avg_iter,
            )

    # Single final sync for end-of-stage scalars + best_tree materialization.
    # Must precede elapsed_sec capture: in static/deferred-eval modes the loop
    # body queues lazy ops and the real GPU work happens here.
    mlx.eval(
        loss_arr,
        grad_norm_arr,
        clip_factor_arr,
        best_loss_arr,
        *best_tree.values(),
    )
    elapsed_sec = time.perf_counter() - stage_t0
    end_loss = _array_scalar(loss_arr)
    best_loss = _array_scalar(best_loss_arr)
    grad_norm = (
        _array_scalar(grad_norm_arr) if iterations_run > 0 else last_logged_grad_norm
    )
    clip_factor = (
        _array_scalar(clip_factor_arr)
        if iterations_run > 0
        else last_logged_clip_factor
    )
    best_table_np = tree_to_numpy_table(params, best_tree)
    if plan_mode == "periodic":
        rebuild_plan(best_tree, count_as_training_rebuild=False)
    if plan is None:
        raise RuntimeError("MLX tile plan has not been built")
    best_rendered = renderer.render(params.as_table(best_tree), plan=plan)
    mlx.eval(best_rendered)
    rendered_np = np.asarray(best_rendered, dtype=np.float32)
    summary = _summarize_times(iter_times)
    metrics: Dict[str, Any] = {
        "optimizer_backend": "mlx",
        "loss_profile": stage_config.loss.name,
        "spatial_weighted": bool(weights is not None),
        "trainable_groups": list(stage_config.trainable_groups),
        "start_loss": float(start_loss),
        "end_loss": float(end_loss),
        "best_loss": float(best_loss),
        "iterations": int(iterations_run),
        "elapsed_sec": float(elapsed_sec),
        "avg_iter_sec": summary["avg_iter_sec"],
        "median_iter_sec": summary["median_iter_sec"],
        "last_grad_norm": float(grad_norm),
        "last_clip_factor": float(clip_factor),
        "lr_decays": 0,
        "stopped_for_time_budget": bool(stopped_for_time_budget),
        "tile_plan_mode": plan_mode,
        "tile_plan_build_sec": float(plan_build_sec),
        "tile_plan_rebuild_interval": int(plan_rebuild_interval),
        "tile_plan_rebuilds": int(plan_rebuilds_in_loop),
        "tile_plan_builds_total": int(plan_builds_total),
        "tile_plan_rebuild_sec": float(plan_rebuild_sec),
        "tile_plan_tiles": int(plan.tiles_x * plan.tiles_y),
        "tile_plan_max_active": int(plan.max_active),
        "renderer_tile_size": int(stage_config.renderer.tile_size),
        "renderer_batch_tile_count": int(stage_config.renderer.batch_tile_count),
        "renderer_compositing_space": str(stage_config.renderer.compositing_space),
        "mlx_compile_enabled": True,
    }
    return MlxStageResult(
        splats=table_to_splats(best_table_np, templates=splats),
        rendered_linear_rgb=rendered_np,
        metrics=metrics,
    )


__all__ = [
    "MlxRendererConfig",
    "MlxStageConfig",
    "MlxStageResult",
    "is_mlx_available",
    "optimize_stage_mlx",
]
