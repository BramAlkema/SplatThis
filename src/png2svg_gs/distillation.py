"""Experimental normalized-top-K teacher to alpha-over student fitting.

This module is intentionally not wired into the production converter. It
exists to make the teacher/student hypothesis measurable before another
permanent pipeline stage or CLI contract is introduced.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Literal, Sequence

import numpy as np
import torch

from .optimizer import SplatParams, build_optimizer
from .renderer import L1SSIMLoss, create_renderer, splats_to_tensor, tensor_to_splats
from .splat import GaussianSplat


@dataclass(frozen=True)
class DistillationArm:
    splats: List[GaussianSplat]
    rendered_linear_rgb: np.ndarray
    start_loss: float
    end_loss: float
    iterations: int
    elapsed_sec: float = 0.0


@dataclass(frozen=True)
class DistillationMvpResult:
    direct: DistillationArm
    teacher: DistillationArm
    student: DistillationArm


HandoffMode = Literal["full", "color-only"]


def _fit(
    initial_splats: Sequence[GaussianSplat],
    target_linear_rgb: np.ndarray,
    *,
    blend_mode: str,
    iterations: int,
    normalized_top_k: int,
    teacher_guide: np.ndarray | None = None,
    teacher_weight: float = 0.25,
    decay_teacher_weight: bool = True,
    exportability_weight: float = 0.0,
    device: str = "cpu",
    renderer_backend: str = "torch",
    tile_size: int = 32,
    batch_tile_count: int = 16,
    background_linear_rgb: np.ndarray | None = None,
) -> DistillationArm:
    fit_t0 = time.perf_counter()
    target_np = np.asarray(target_linear_rgb, dtype=np.float32)[..., :3]
    height, width = target_np.shape[:2]
    torch_device = torch.device(device)
    target = torch.from_numpy(target_np).to(torch_device)
    guide = (
        None
        if teacher_guide is None
        else torch.from_numpy(np.asarray(teacher_guide, dtype=np.float32)).to(
            torch_device
        )
    )
    background = (
        np.zeros(3, dtype=np.float32)
        if background_linear_rgb is None
        else np.clip(
            np.asarray(background_linear_rgb, dtype=np.float32).reshape(3),
            0.0,
            1.0,
        )
    )

    renderer = create_renderer(
        backend=renderer_backend,
        width=width,
        height=height,
        device=torch_device,
        tile_size=max(1, int(tile_size)),
        batch_tile_count=max(1, int(batch_tile_count)),
        blend_mode=blend_mode,
        normalized_top_k=normalized_top_k,
        background_color=background,
        compositing_space="srgb" if blend_mode == "alpha-over" else "linear",
    )
    export_renderer = None
    if exportability_weight > 0.0:
        export_renderer = create_renderer(
            backend=renderer_backend,
            width=width,
            height=height,
            device=torch_device,
            tile_size=max(1, int(tile_size)),
            batch_tile_count=max(1, int(batch_tile_count)),
            blend_mode="alpha-over",
            background_color=background,
            compositing_space="srgb",
        )
    params = SplatParams(
        splats_to_tensor(list(initial_splats), device=torch_device)
    ).to(torch_device)
    optimizer = build_optimizer(params)
    loss_fn = L1SSIMLoss(
        l1_weight=1.0,
        ssim_weight=0.2,
        gradient_weight=0.05,
        color_space="oklab",
    ).to(torch_device)

    def objective(
        rendered: torch.Tensor,
        table: torch.Tensor,
        progress: float,
    ) -> torch.Tensor:
        source_loss = loss_fn(rendered, target)
        loss = source_loss
        if guide is not None:
            guide_mix = float(np.clip(teacher_weight, 0.0, 1.0))
            if decay_teacher_weight:
                guide_mix *= max(0.0, 1.0 - float(np.clip(progress, 0.0, 1.0)))
            loss = (1.0 - guide_mix) * source_loss + guide_mix * loss_fn(
                rendered, guide
            )
        if export_renderer is not None:
            # The normalized teacher has no meaningful opacity. Evaluate the
            # same geometry/color as a maximally opaque source-over student so
            # teacher optimization cannot win solely through infinitesimal
            # Gaussian tails that native vector shapes cannot reproduce.
            opaque_table = torch.cat(
                [
                    table[:, :9],
                    torch.ones_like(table[:, 9:10]),
                    table[:, 10:11],
                ],
                dim=1,
            )
            loss = loss + float(exportability_weight) * loss_fn(
                export_renderer(opaque_table), target
            )
        return loss

    with torch.no_grad():
        initial_table = params.as_tensor()
        start_loss = float(loss_fn(renderer(initial_table), target).item())
    best_loss = start_loss
    best_snapshot = params.snapshot()

    iteration_count = max(0, int(iterations))
    for iteration in range(iteration_count):
        optimizer.zero_grad(set_to_none=True)
        table = params.as_tensor()
        rendered = renderer(table)
        progress = float(iteration) / float(max(iteration_count - 1, 1))
        loss = objective(rendered, table, progress=progress)
        loss.backward()
        source_loss_value = float(loss_fn(rendered.detach(), target).item())
        if source_loss_value < best_loss:
            best_loss = source_loss_value
            best_snapshot = params.snapshot()
        torch.nn.utils.clip_grad_norm_(params.parameters(), max_norm=1.0)
        optimizer.step()
        params.apply_constraints(width, height)

    with torch.no_grad():
        final_table = params.as_tensor()
        final_loss = float(loss_fn(renderer(final_table), target).item())
        if final_loss < best_loss:
            best_loss = final_loss
            best_snapshot = params.snapshot()

    params.restore(best_snapshot)
    renderer.clear_tile_bin_cache()
    with torch.no_grad():
        rendered_np = renderer(params.as_tensor()).detach().cpu().numpy()

    return DistillationArm(
        splats=tensor_to_splats(params.as_tensor().detach()),
        rendered_linear_rgb=np.asarray(rendered_np, dtype=np.float32),
        start_loss=float(start_loss),
        end_loss=float(best_loss),
        iterations=max(0, int(iterations)),
        elapsed_sec=float(time.perf_counter() - fit_t0),
    )


def _fit_mlx(
    initial_splats: Sequence[GaussianSplat],
    target_linear_rgb: np.ndarray,
    *,
    blend_mode: str,
    iterations: int,
    normalized_top_k: int,
    teacher_guide: np.ndarray | None = None,
    teacher_weight: float = 0.25,
    decay_teacher_weight: bool = True,
    exportability_weight: float = 0.0,
    tile_size: int = 32,
    batch_tile_count: int = 16,
    background_linear_rgb: np.ndarray | None = None,
    tile_plan_rebuild_interval: int = 1,
) -> DistillationArm:
    """MLX equivalent of `_fit`, including the distillation objective.

    This stays in the experimental module instead of widening the production
    stage API with teacher-specific guide/exportability semantics.
    """

    try:
        import mlx.core as mx
    except Exception as exc:  # pragma: no cover - optional dependency guard.
        raise RuntimeError("MLX is not installed; cannot run MLX distillation") from exc

    from .mlx_losses import MlxLossConfig, make_loss_fn
    from .mlx_optimizer import (
        MlxSplatParams,
        clone_tree,
        constrain_trainable_tree,
        table_to_splats,
        tree_to_numpy_table,
    )
    from .mlx_renderer import (
        MlxBatchedGaussianRenderer,
        MlxTilePlan,
        splats_to_numpy_table,
    )
    from .optimizer import DEFAULT_LEARNING_RATES

    fit_t0 = time.perf_counter()
    target_np = np.clip(
        np.asarray(target_linear_rgb, dtype=np.float32)[..., :3], 0.0, 1.0
    )
    height, width = target_np.shape[:2]
    background = (
        np.zeros(3, dtype=np.float32)
        if background_linear_rgb is None
        else np.clip(
            np.asarray(background_linear_rgb, dtype=np.float32).reshape(3),
            0.0,
            1.0,
        )
    )
    guide = (
        None
        if teacher_guide is None
        else mx.array(np.clip(np.asarray(teacher_guide, dtype=np.float32), 0.0, 1.0))
    )

    table_np = splats_to_numpy_table(initial_splats)
    params = MlxSplatParams.from_table(table_np)
    trainable = constrain_trainable_tree(
        params.trainable_tree(),
        image_width=width,
        image_height=height,
    )
    target = mx.array(target_np)
    renderer = MlxBatchedGaussianRenderer(
        width=width,
        height=height,
        tile_size=max(1, int(tile_size)),
        batch_tile_count=max(1, int(batch_tile_count)),
        blend_mode=blend_mode,
        normalized_top_k=normalized_top_k,
        background_color=background,
        compositing_space="srgb" if blend_mode == "alpha-over" else "linear",
    )
    export_renderer = None
    if exportability_weight > 0.0:
        export_renderer = MlxBatchedGaussianRenderer(
            width=width,
            height=height,
            tile_size=max(1, int(tile_size)),
            batch_tile_count=max(1, int(batch_tile_count)),
            blend_mode="alpha-over",
            background_color=background,
            compositing_space="srgb",
        )
    loss_fn = make_loss_fn(
        MlxLossConfig(
            name="oklab-l1-ssim",
            l1_weight=1.0,
            ssim_weight=0.2,
            gradient_weight=0.05,
        )
    )

    plan = renderer.build_plan(table_np)
    rebuild_interval = max(1, int(tile_plan_rebuild_interval))
    tiles_x = (width + renderer.tile_size - 1) // renderer.tile_size
    tiles_y = (height + renderer.tile_size - 1) // renderer.tile_size

    def rebuild_plan(tree: Dict[str, object]) -> None:
        nonlocal plan
        current = params.as_table(tree)
        mx.eval(current)
        plan = renderer.build_plan(np.asarray(current, dtype=np.float32))

    def objective(
        tree: Dict[str, object],
        plan_indices,
        plan_mask,
        plan_order,
        progress,
    ):
        plan_inner = MlxTilePlan(
            indices=plan_indices,
            mask=plan_mask,
            order=plan_order,
            tiles_x=tiles_x,
            tiles_y=tiles_y,
            max_active=int(plan_indices.shape[1]),
            tile_size=renderer.tile_size,
        )
        table = params.as_table(tree)
        rendered = renderer.render(table, plan=plan_inner)
        source_loss = loss_fn(rendered, target, None)
        optimized_loss = source_loss
        if guide is not None:
            guide_mix = mx.array(
                float(np.clip(teacher_weight, 0.0, 1.0)),
                dtype=mx.float32,
            )
            if decay_teacher_weight:
                guide_mix = guide_mix * mx.maximum(0.0, 1.0 - progress)
            guide_loss = loss_fn(rendered, guide, None)
            optimized_loss = (1.0 - guide_mix) * source_loss + guide_mix * guide_loss
        if export_renderer is not None:
            opaque_table = mx.concatenate(
                [
                    table[:, :9],
                    mx.ones_like(table[:, 9:10]),
                    table[:, 10:11],
                ],
                axis=1,
            )
            export_rendered = export_renderer.render(opaque_table, plan=plan_inner)
            optimized_loss = optimized_loss + float(exportability_weight) * loss_fn(
                export_rendered, target, None
            )
        # The optimized objective may change as teacher guidance decays. As in
        # Torch, checkpoint selection therefore uses source_loss only.
        return optimized_loss, source_loss

    value_and_grad = mx.value_and_grad(objective)
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8

    def train_step(
        tree,
        first_moment,
        second_moment,
        step_count,
        plan_indices,
        plan_mask,
        plan_order,
        progress,
    ):
        (optimized_loss, source_loss), grads = value_and_grad(
            tree,
            plan_indices,
            plan_mask,
            plan_order,
            progress,
        )
        total_sq = mx.array(0.0, dtype=mx.float32)
        for grad in grads.values():
            total_sq = total_sq + mx.sum(grad * grad)
        grad_norm = mx.sqrt(total_sq)
        clip_factor = mx.minimum(
            mx.array(1.0, dtype=mx.float32),
            1.0 / (grad_norm + 1e-6),
        )
        next_step = step_count + 1
        step_float = next_step.astype(mx.float32)
        bias1 = 1.0 - mx.power(mx.array(beta1, dtype=mx.float32), step_float)
        bias2 = 1.0 - mx.power(mx.array(beta2, dtype=mx.float32), step_float)
        new_first = {}
        new_second = {}
        new_tree = {}
        for key, value in tree.items():
            clipped_grad = grads[key] * clip_factor
            new_first[key] = beta1 * first_moment[key] + (1.0 - beta1) * clipped_grad
            new_second[key] = (
                beta2 * second_moment[key] + (1.0 - beta2) * clipped_grad * clipped_grad
            )
            first_hat = new_first[key] / bias1
            second_hat = new_second[key] / bias2
            new_tree[key] = value - (
                float(DEFAULT_LEARNING_RATES[key])
                * first_hat
                / (mx.sqrt(second_hat) + eps)
            )
        new_tree = constrain_trainable_tree(
            new_tree,
            image_width=width,
            image_height=height,
        )
        return (
            new_tree,
            new_first,
            new_second,
            next_step,
            optimized_loss,
            source_loss,
        )

    compiled_train_step = mx.compile(train_step)
    progress_zero = mx.array(0.0, dtype=mx.float32)
    _, start_source_arr = objective(
        trainable,
        plan.indices,
        plan.mask,
        plan.order,
        progress_zero,
    )
    mx.eval(start_source_arr)
    start_loss = float(np.asarray(start_source_arr))
    best_source_arr = start_source_arr
    best_tree = clone_tree(trainable)
    first_moment = {key: mx.zeros_like(value) for key, value in trainable.items()}
    second_moment = {key: mx.zeros_like(value) for key, value in trainable.items()}
    step_count = mx.array(0, dtype=mx.int32)
    iterations_run = max(0, int(iterations))

    for iteration in range(iterations_run):
        pre_step_tree = trainable
        progress = mx.array(
            float(iteration) / float(max(iterations_run - 1, 1)),
            dtype=mx.float32,
        )
        (
            trainable,
            first_moment,
            second_moment,
            step_count,
            _optimized_loss,
            source_loss_arr,
        ) = compiled_train_step(
            trainable,
            first_moment,
            second_moment,
            step_count,
            plan.indices,
            plan.mask,
            plan.order,
            progress,
        )
        is_better = source_loss_arr < best_source_arr
        best_source_arr = mx.where(is_better, source_loss_arr, best_source_arr)
        best_tree = {
            key: mx.where(is_better, pre_step_tree[key], best_tree[key])
            for key in trainable
        }
        if (iteration + 1) % rebuild_interval == 0:
            rebuild_plan(trainable)

    if iterations_run > 0:
        progress_one = mx.array(1.0, dtype=mx.float32)
        _, final_source_arr = objective(
            trainable,
            plan.indices,
            plan.mask,
            plan.order,
            progress_one,
        )
        final_better = final_source_arr < best_source_arr
        best_source_arr = mx.where(final_better, final_source_arr, best_source_arr)
        best_tree = {
            key: mx.where(final_better, trainable[key], best_tree[key])
            for key in trainable
        }

    mx.eval(best_source_arr, *best_tree.values())
    best_table = tree_to_numpy_table(params, best_tree)
    best_plan = renderer.build_plan(best_table)
    best_rendered = renderer.render(mx.array(best_table), plan=best_plan)
    mx.eval(best_rendered)

    return DistillationArm(
        splats=table_to_splats(best_table, templates=initial_splats),
        rendered_linear_rgb=np.asarray(best_rendered, dtype=np.float32),
        start_loss=start_loss,
        end_loss=float(np.asarray(best_source_arr)),
        iterations=iterations_run,
        elapsed_sec=float(time.perf_counter() - fit_t0),
    )


def run_distillation_mvp(
    initial_splats: Sequence[GaussianSplat],
    target_linear_rgb: np.ndarray,
    *,
    teacher_iterations: int = 40,
    student_iterations: int = 40,
    normalized_top_k: int = 10,
    teacher_weight: float = 0.25,
    decay_teacher_weight: bool = True,
    teacher_exportability_weight: float = 0.0,
    handoff_mode: HandoffMode = "full",
    device: str = "cpu",
    renderer_backend: str = "torch",
    tile_size: int = 32,
    batch_tile_count: int = 16,
    background_linear_rgb: np.ndarray | None = None,
    optimization_backend: str = "torch",
    mlx_tile_plan_rebuild_interval: int = 1,
) -> DistillationMvpResult:
    """Run equal-iteration direct, teacher, and distilled-student arms."""

    backend = str(optimization_backend).strip().lower().replace("_", "-")
    if backend not in {"torch", "mlx"}:
        raise ValueError(f"Unsupported distillation backend: {optimization_backend}")
    fit = _fit_mlx if backend == "mlx" else _fit
    common = {
        "normalized_top_k": normalized_top_k,
        "tile_size": tile_size,
        "batch_tile_count": batch_tile_count,
        "background_linear_rgb": background_linear_rgb,
    }
    if backend == "torch":
        common.update(
            {
                "device": device,
                "renderer_backend": renderer_backend,
            }
        )
    else:
        common["tile_plan_rebuild_interval"] = int(
            max(1, mlx_tile_plan_rebuild_interval)
        )

    direct = fit(
        initial_splats,
        target_linear_rgb,
        blend_mode="alpha-over",
        iterations=int(teacher_iterations) + int(student_iterations),
        **common,
    )
    teacher = fit(
        initial_splats,
        target_linear_rgb,
        blend_mode="normalized-topk",
        iterations=teacher_iterations,
        exportability_weight=teacher_exportability_weight,
        **common,
    )

    from .mlx_optimizer import table_to_splats
    from .mlx_renderer import splats_to_numpy_table

    teacher_table = splats_to_numpy_table(teacher.splats)
    initial_table = splats_to_numpy_table(initial_splats)
    if handoff_mode == "full":
        # A literal teacher copy starts source-over with holes. Preserve at
        # least the initialization's scale footprint and begin at maximum
        # native alpha; the student can shrink both during adaptation.
        student_table = teacher_table.copy()
        student_table[:, 2:4] = np.maximum(teacher_table[:, 2:4], initial_table[:, 2:4])
        student_table[:, 9] = 1.0
    elif handoff_mode == "color-only":
        # Retain geometry, opacity, and order already designed for coverage;
        # borrow only the teacher's locally optimized color vocabulary.
        student_table = initial_table.copy()
        student_table[:, 6:9] = teacher_table[:, 6:9]
    else:
        raise ValueError(f"Unsupported handoff mode: {handoff_mode}")
    student_initial = table_to_splats(student_table, templates=initial_splats)
    student = fit(
        student_initial,
        target_linear_rgb,
        blend_mode="alpha-over",
        iterations=student_iterations,
        teacher_guide=teacher.rendered_linear_rgb,
        teacher_weight=teacher_weight,
        decay_teacher_weight=decay_teacher_weight,
        **common,
    )

    return DistillationMvpResult(direct=direct, teacher=teacher, student=student)


def summarize_mvp_metrics(
    result: DistillationMvpResult, target_linear_rgb: np.ndarray
) -> Dict[str, Dict[str, float]]:
    from .io import compute_quality_metrics

    return {
        name: compute_quality_metrics(
            target_linear_rgb, getattr(result, name).rendered_linear_rgb
        )
        for name in ("direct", "teacher", "student")
    }


__all__ = [
    "DistillationArm",
    "DistillationMvpResult",
    "HandoffMode",
    "run_distillation_mvp",
    "summarize_mvp_metrics",
]
