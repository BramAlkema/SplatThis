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
from .mlx_renderer import MlxBatchedGaussianRenderer, splats_to_numpy_table
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
    )
    plan = None
    plan_build_sec = 0.0
    plan_rebuilds = 0
    plan_rebuild_sec = 0.0
    plan_rebuild_interval = int(max(1, stage_config.tile_plan_rebuild_interval))

    def rebuild_plan(tree: Mapping[str, Any]) -> None:
        nonlocal plan, plan_build_sec, plan_rebuilds, plan_rebuild_sec
        table = params.as_table(tree)
        mlx.eval(table)
        current_table_np = np.asarray(table, dtype=np.float32)
        plan_t0 = time.perf_counter()
        plan = renderer.build_plan(current_table_np)
        elapsed = time.perf_counter() - plan_t0
        plan_build_sec += elapsed
        if plan_rebuilds > 0:
            plan_rebuild_sec += elapsed
        plan_rebuilds += 1

    rebuild_plan(trainable)
    loss_fn = make_loss_fn(stage_config.loss)
    adam = MlxAdam(
        trainable,
        learning_rates=learning_rates,
        grad_clip_norm=stage_config.grad_clip_norm,
    )

    def loss_for_tree(tree: Dict[str, Any]) -> Any:
        if plan is None:
            raise RuntimeError("MLX tile plan has not been built")
        rendered = renderer.render(params.as_table(tree), plan=plan)
        return loss_fn(rendered, target, weights)

    start_loss_arr = loss_for_tree(trainable)
    mlx.eval(start_loss_arr)
    start_loss = _array_scalar(start_loss_arr)
    best_loss = start_loss
    end_loss = start_loss
    best_tree = clone_tree(trainable)
    iter_times: List[float] = []
    grad_norm = 0.0
    clip_factor = 1.0
    stopped_for_time_budget = False

    value_and_grad = mlx.value_and_grad(loss_for_tree)
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
        loss, grads = value_and_grad(trainable)
        trainable, opt_stats = adam.step(trainable, grads)
        trainable = constrain_trainable_tree(
            trainable,
            image_width=width,
            image_height=height,
        )
        if plan_mode == "periodic" and (iteration + 1) % plan_rebuild_interval == 0:
            rebuild_plan(trainable)
        mlx.eval(
            loss, opt_stats["grad_norm"], opt_stats["clip_factor"], *trainable.values()
        )
        iter_elapsed = time.perf_counter() - iter_t0
        iter_times.append(iter_elapsed)
        iterations_run = iteration + 1

        end_loss = _array_scalar(loss)
        grad_norm = _array_scalar(opt_stats["grad_norm"])
        clip_factor = _array_scalar(opt_stats["clip_factor"])
        if end_loss < best_loss:
            best_loss = end_loss
            best_tree = clone_tree(trainable)
            mlx.eval(*best_tree.values())

        if (
            verbose
            and progress_interval
            and (
                iteration == 0
                or (iteration + 1) % progress_interval == 0
                or iteration + 1 == num_iters
            )
        ):
            avg_iter = statistics.mean(iter_times)
            logger.info(
                "  MLX iteration %s/%s: loss=%.6f best=%.6f iter=%.3fs avg=%.3fs",
                iteration + 1,
                num_iters,
                end_loss,
                best_loss,
                iter_elapsed,
                avg_iter,
            )

    elapsed_sec = time.perf_counter() - stage_t0
    best_table_np = tree_to_numpy_table(params, best_tree)
    if plan_mode == "periodic":
        rebuild_plan(best_tree)
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
        "tile_plan_rebuilds": int(plan_rebuilds),
        "tile_plan_rebuild_sec": float(plan_rebuild_sec),
        "tile_plan_tiles": int(plan.tiles_x * plan.tiles_y),
        "tile_plan_max_active": int(plan.max_active),
        "renderer_tile_size": int(stage_config.renderer.tile_size),
        "renderer_batch_tile_count": int(stage_config.renderer.batch_tile_count),
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
