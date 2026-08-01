"""Torch/MLX optimization and checkpoint evaluation."""

from __future__ import annotations

import copy
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import psutil
import torch

from .adaptive_compute import (
    CanvasCheckpoint,
    OnlineAdaptiveDecision,
    evaluate_online_checkpoints,
)
from .engine_state import ConversionEngineState
from .export_common import PPTX_SOFT_EDGE_ALPHA_SCALE, PPTX_SOFT_EDGE_K_SIGMA_SCALE
from .mlx_losses import MlxLossConfig
from .mlx_stage import MlxRendererConfig, MlxStageConfig, optimize_stage_mlx
from .optimizer import SplatParams, build_optimizer
from .quality import compute_quality_metrics
from .renderer import splats_to_tensor, tensor_to_splats
from .splat import GaussianSplat

logger = logging.getLogger(__name__)


class ConversionOptimizationMixin(ConversionEngineState):
    """Fits splat parameters with Torch or MLX and grades checkpoints."""

    def _optimize_splats(  # noqa: C901
        self,
        image: np.ndarray,
        splats: List[GaussianSplat],
        rng: np.random.Generator,
        verbose: bool = True,
        artifacts_dir: Optional[Path] = None,
        structure_primary: Optional[np.ndarray] = None,
        structure_anisotropy: Optional[np.ndarray] = None,
        monotonic_stage_selection: bool = False,
    ) -> Tuple[List[GaussianSplat], List[Dict[str, Any]]]:
        """Progressive optimization of splats."""
        height, width = image.shape[:2]
        target = torch.from_numpy(image[:, :, :3]).to(self.device)
        edge_map = self._build_edge_map(image)

        memory_before = psutil.virtual_memory().percent
        if memory_before > 85:
            logger.warning(
                "High memory usage detected: %.1f%% - reducing splat count",
                memory_before,
            )
            self.max_splats = min(self.max_splats, max(1, len(splats) // 2))

        renderer = self._create_training_renderer(width=width, height=height)
        loss_fn = self._create_training_loss(target=target, width=width, height=height)
        if verbose:
            cache_stats = self._renderer_cache_stats(renderer)
            if cache_stats:
                logger.info(
                    "Renderer: tile=%s cache_interval=%s padding=%.1f",
                    cache_stats.get("tile_size"),
                    cache_stats.get("rebuild_interval"),
                    float(cache_stats.get("padding", 0.0)),
                )

        current_splats = splats.copy()
        stage_metrics: List[Dict[str, Any]] = []
        best_canvas_splats: Optional[List[GaussianSplat]] = None
        best_canvas_metrics: Optional[Dict[str, float]] = None
        best_canvas_label: Optional[str] = None
        final_canvas_label: Optional[str] = None
        canvas_checkpoints: List[CanvasCheckpoint] = []
        adaptive_last_decision: Optional[OnlineAdaptiveDecision] = None
        adaptive_stop_decision: Optional[OnlineAdaptiveDecision] = None
        adaptive_enabled = bool(
            monotonic_stage_selection and self.adaptive_compute_config.enabled
        )

        def consider_canvas_checkpoint(
            label: str,
            candidate_splats: List[GaussianSplat],
            metric: Dict[str, float],
        ) -> None:
            nonlocal best_canvas_splats, best_canvas_metrics, best_canvas_label
            if not monotonic_stage_selection:
                return
            if best_canvas_metrics is None or self._prefer_canvas_checkpoint(
                candidate=metric,
                candidate_count=len(candidate_splats),
                incumbent=best_canvas_metrics,
                incumbent_count=(
                    len(best_canvas_splats) if best_canvas_splats is not None else 0
                ),
            ):
                best_canvas_splats = copy.deepcopy(candidate_splats)
                best_canvas_metrics = dict(metric)
                best_canvas_label = label

        residual_detail_enabled = bool(
            self.refinement_config.get("residual_detail_enabled", False)
        )
        residual_reserve_fraction = float(
            np.clip(
                self.refinement_config.get("residual_detail_reserve_fraction", 0.0),
                0.0,
                0.40,
            )
        )
        residual_time_reserve_sec = (
            float(
                max(
                    0.0,
                    self.refinement_config.get("residual_detail_time_reserve_sec", 0.0),
                )
            )
            if residual_detail_enabled
            else 0.0
        )
        reserved_slots = (
            int(round(float(self.max_splats) * residual_reserve_fraction))
            if residual_detail_enabled
            else 0
        )
        main_budget = max(1, self.max_splats - max(0, reserved_slots))

        for stage_idx, num_iters in enumerate(self.stages):
            if self._time_budget_exhausted():
                if verbose:
                    logger.info(
                        "Training budget exhausted before stage %s/%s",
                        stage_idx + 1,
                        len(self.stages),
                    )
                break
            remaining_before_stage = self._time_budget_seconds_remaining()
            if (
                residual_time_reserve_sec > 0.0
                and remaining_before_stage is not None
                and remaining_before_stage <= residual_time_reserve_sec
            ):
                if verbose:
                    logger.info(
                        "Stopping main stages before stage %s/%s with %.1fs reserved for residual detail",
                        stage_idx + 1,
                        len(self.stages),
                        remaining_before_stage,
                    )
                break
            if verbose:
                logger.info(
                    "Stage %s/%s: %s iterations, %s splats",
                    stage_idx + 1,
                    len(self.stages),
                    num_iters,
                    len(current_splats),
                )

            current_splats, stage_metric, stage_rendered = self._optimize_stage(
                splats=current_splats,
                target=target,
                renderer=renderer,
                loss_fn=loss_fn,
                num_iters=num_iters,
                verbose=verbose,
            )

            quality, _, coverage_map = self._compute_quality_metrics_cached(
                splats=current_splats,
                target=target,
                renderer=renderer,
                loss_fn=loss_fn,
                precomputed_rendered=stage_rendered,
            )
            stage_metric.update(quality)
            stage_metric["stage"] = stage_idx + 1
            stage_metric["splat_count"] = len(current_splats)
            if monotonic_stage_selection:
                deployed_quality = self._score_canvas_runtime_model(
                    current_splats, image
                )
                stage_metric.update(
                    {
                        f"deployed_{key}": float(value)
                        for key, value in deployed_quality.items()
                    }
                )
                consider_canvas_checkpoint(
                    f"stage-{stage_idx + 1}",
                    current_splats,
                    deployed_quality,
                )
                final_canvas_label = f"stage-{stage_idx + 1}"
                canvas_checkpoints.append(
                    CanvasCheckpoint(
                        label=final_canvas_label,
                        ssim_srgb=float(deployed_quality["ssim_srgb"]),
                        psnr_srgb=float(deployed_quality["psnr_srgb"]),
                        splat_count=len(current_splats),
                        elapsed_sec=float(stage_metric.get("elapsed_sec", 0.0)),
                    )
                )
                if adaptive_enabled:
                    adaptive_last_decision = evaluate_online_checkpoints(
                        canvas_checkpoints,
                        self.adaptive_compute_config,
                    )
                    stage_metric["adaptive_compute_decision"] = (
                        adaptive_last_decision.as_dict()
                    )
                    if adaptive_last_decision.stop:
                        adaptive_stop_decision = adaptive_last_decision
            remaining = self._time_budget_seconds_remaining()
            if remaining is not None:
                stage_metric["time_budget_remaining_sec"] = max(0.0, float(remaining))
                stage_metric["time_budget_exhausted"] = bool(
                    self._time_budget_exhausted()
                )
            if verbose:
                logger.info(
                    "Stage %s/%s done in %.2fs: loss %.6f -> %.6f, SSIM_sRGB=%.4f, coverage=%.3f",
                    stage_idx + 1,
                    len(self.stages),
                    float(stage_metric.get("elapsed_sec", 0.0)),
                    float(stage_metric.get("start_loss", 0.0)),
                    float(
                        stage_metric.get("best_loss", stage_metric.get("end_loss", 0.0))
                    ),
                    float(stage_metric.get("ssim_srgb", 0.0)),
                    float(stage_metric.get("coverage", 0.0)),
                )
            stage_metrics.append(stage_metric)
            self._write_stage_artifact(
                artifacts_dir,
                f"iter-{stage_idx + 1}",
                current_splats,
                stage_metric,
            )
            if adaptive_stop_decision is not None:
                if verbose:
                    logger.info(
                        "Adaptive compute stopped after stage %s/%s: "
                        "selected %s at SSIM_sRGB=%.4f",
                        stage_idx + 1,
                        len(self.stages),
                        adaptive_stop_decision.selected.label,
                        adaptive_stop_decision.selected.ssim_srgb,
                    )
                break

            coverage_after_densify: Optional[np.ndarray] = None
            remaining_after_stage = self._time_budget_seconds_remaining()
            in_residual_time_reserve = (
                residual_time_reserve_sec > 0.0
                and remaining_after_stage is not None
                and remaining_after_stage <= residual_time_reserve_sec
            )
            if (
                stage_idx < len(self.stages) - 1
                and not self._time_budget_exhausted()
                and not in_residual_time_reserve
            ):
                before_densify = len(current_splats)
                densify_t0 = time.perf_counter()
                current_splats, coverage_after_densify = self._add_error_driven_splats(
                    splats=current_splats,
                    image=image,
                    target=target,
                    renderer=renderer,
                    rng=rng,
                    edge_map=edge_map,
                    stage_idx=stage_idx,
                    precomputed_rendered=stage_rendered,
                    precomputed_coverage_map=coverage_map,
                    structure_primary=structure_primary,
                    structure_anisotropy=structure_anisotropy,
                    max_splats_cap=main_budget,
                )
                if verbose:
                    logger.info(
                        "Densify after stage %s: +%s splats in %.2fs (%s total)",
                        stage_idx + 1,
                        len(current_splats) - before_densify,
                        time.perf_counter() - densify_t0,
                        len(current_splats),
                    )
            elif in_residual_time_reserve and verbose:
                logger.info(
                    "Skipping main-stage densification to reserve %.1fs for residual detail",
                    remaining_after_stage,
                )

            if len(current_splats) > main_budget:
                current_splats = self._prune_splats(
                    current_splats,
                    main_budget,
                    target=target,
                    renderer=renderer,
                    precomputed_coverage_map=coverage_after_densify,
                )

        residual_metrics: List[Dict[str, Any]]
        if adaptive_stop_decision is not None:
            if verbose and residual_detail_enabled:
                logger.info(
                    "Skipping residual detail because the adaptive quality target "
                    "was met."
                )
            residual_metrics = []
        elif self._time_budget_exhausted():
            if verbose:
                logger.info(
                    "Skipping residual detail pass because training budget is exhausted."
                )
            residual_metrics = []
        else:
            current_splats, residual_metrics = self._run_residual_detail_passes(
                splats=current_splats,
                image=image,
                target=target,
                renderer=renderer,
                loss_fn=loss_fn,
                rng=rng,
                edge_map=edge_map,
                verbose=verbose,
            )
        if monotonic_stage_selection and residual_metrics:
            deployed_quality = self._score_canvas_runtime_model(current_splats, image)
            residual_metrics[-1].update(
                {
                    f"deployed_{key}": float(value)
                    for key, value in deployed_quality.items()
                }
            )
            consider_canvas_checkpoint(
                "residual-final",
                current_splats,
                deployed_quality,
            )
            final_canvas_label = "residual-final"
        for metric in residual_metrics:
            stage_metrics.append(metric)
            pass_idx = int(metric.get("residual_pass", len(stage_metrics)))
            self._write_stage_artifact(
                artifacts_dir,
                f"residual-{pass_idx}",
                current_splats,
                metric,
            )

        if adaptive_enabled:
            observed_stages = len(canvas_checkpoints)
            stopped_early = adaptive_stop_decision is not None
            decision = adaptive_stop_decision or adaptive_last_decision
            stage_metrics.append(
                {
                    "stage": -4,
                    "stage_type": "canvas_adaptive_compute",
                    "mode": "online-observed-only",
                    "uses_future_evidence": False,
                    "enabled": True,
                    "stopped_early": stopped_early,
                    "reason": (
                        decision.reason if decision is not None else "no-checkpoint"
                    ),
                    "stop_after_checkpoint": (
                        adaptive_stop_decision.current.label
                        if adaptive_stop_decision is not None
                        else None
                    ),
                    "selected_checkpoint": (
                        decision.selected.label if decision is not None else None
                    ),
                    "checkpoints_observed": observed_stages,
                    "scheduled_main_stages": len(self.stages),
                    "skipped_main_stages": (
                        max(0, len(self.stages) - observed_stages)
                        if stopped_early
                        else 0
                    ),
                    "skipped_main_stage_iterations": (
                        int(sum(self.stages[observed_stages:])) if stopped_early else 0
                    ),
                    "skipped_residual_detail": bool(
                        stopped_early and residual_detail_enabled
                    ),
                    "policy": self.adaptive_compute_config.as_dict(),
                }
            )

        if monotonic_stage_selection and best_canvas_splats is not None:
            selected_count = len(best_canvas_splats)
            current_count = len(current_splats)
            selection_decision = (
                "keep-final"
                if best_canvas_label == final_canvas_label
                else "revert-to-best-checkpoint"
            )
            stage_metrics.append(
                {
                    "stage": -3,
                    "stage_type": "canvas_monotonic_stage_selection",
                    "decision": selection_decision,
                    "selected_checkpoint": best_canvas_label,
                    "selected_splat_count": selected_count,
                    "final_candidate_splat_count": current_count,
                    "selected_ssim_srgb": float(
                        best_canvas_metrics.get("ssim_srgb", 0.0)
                    ),
                    "selected_psnr_srgb": float(
                        best_canvas_metrics.get("psnr_srgb", 0.0)
                    ),
                }
            )
            current_splats = best_canvas_splats

        return current_splats, stage_metrics

    def _optimize_stage(  # noqa: C901
        self,
        splats: List[GaussianSplat],
        target: torch.Tensor,
        renderer: torch.nn.Module,
        loss_fn: torch.nn.Module,
        num_iters: int,
        verbose: bool,
    ) -> Tuple[List[GaussianSplat], Dict[str, Any], torch.Tensor]:
        """Optimize splats for one stage using SplatParams + Adam param_groups.

        Each splat parameter group is a separate nn.Parameter so per-group
        learning rates (position / scale / theta / color / alpha) have their
        textbook meaning instead of the old post-step delta-rescale hack.
        """
        if self.optimizer_backend == "mlx":
            return self._optimize_stage_mlx(
                splats=splats,
                target=target,
                num_iters=num_iters,
                verbose=verbose,
            )

        if not splats:
            empty = torch.zeros(
                (int(target.shape[0]), int(target.shape[1]), 3),
                dtype=torch.float32,
                device=self.device,
            )
            return (
                splats,
                {"start_loss": 0.0, "end_loss": 0.0, "best_loss": 0.0, "iterations": 0},
                empty,
            )

        initial_tensor = splats_to_tensor(splats, device=self.device)
        params = SplatParams(initial_tensor).to(self.device)
        optimizer = build_optimizer(params, self.learning_rates)
        stage_start = time.perf_counter()
        default_progress_interval = max(
            1, min(10, int(np.ceil(max(1, num_iters) / 6.0)))
        )
        progress_interval = int(
            max(
                1,
                self.refinement_config.get(
                    "progress_log_interval", default_progress_interval
                ),
            )
        )
        renderer_cache_before = int(
            self._renderer_cache_stats(renderer).get("rebuilds", 0)
        )

        with torch.no_grad():
            start_loss = float(loss_fn(renderer(params.as_tensor()), target).item())

        best_loss = start_loss
        end_loss = start_loss
        best_snapshot = params.snapshot()
        iterations_run = 0

        schedule_enabled = bool(self.schedule_config.get("enabled", True))
        check_interval = int(max(1, self.schedule_config.get("check_interval", 50)))
        patience_checks = int(max(1, self.schedule_config.get("patience_checks", 3)))
        decay_ratio = float(max(1.0, self.schedule_config.get("decay_ratio", 2.0)))
        max_decays = int(max(0, self.schedule_config.get("max_decays", 2)))
        min_delta = float(max(0.0, self.schedule_config.get("min_delta", 1e-4)))

        no_improve_checks = 0
        decay_count = 0
        best_at_last_check = best_loss

        image_height = int(target.shape[0])
        image_width = int(target.shape[1])
        stopped_for_time_budget = False

        for iteration in range(max(0, num_iters)):
            if self._time_budget_exhausted():
                stopped_for_time_budget = True
                if verbose:
                    logger.info(
                        "  Time budget exhausted at iteration %s/%s",
                        iteration,
                        num_iters,
                    )
                break
            iterations_run = iteration + 1
            iter_t0 = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            rendered = renderer(params.as_tensor())
            loss = loss_fn(rendered, target)
            loss.backward()
            loss_value = float(loss.item())
            # Snapshot BEFORE optimizer.step(): `loss` was measured on the
            # current params, so the best snapshot must capture them pre-step.
            if loss_value < best_loss:
                best_loss = loss_value
                best_snapshot = params.snapshot()
            # Clip gradient norm across all trainable splat params.
            torch.nn.utils.clip_grad_norm_(
                [
                    params.position,
                    params.scale,
                    params.theta,
                    params.color,
                    params.alpha,
                ],
                max_norm=1.0,
            )
            optimizer.step()
            params.apply_constraints(image_width, image_height)

            iter_elapsed = time.perf_counter() - iter_t0
            end_loss = loss_value

            should_log_progress = verbose and (
                iteration == 0
                or (iteration + 1) % progress_interval == 0
                or iteration + 1 == num_iters
            )
            if should_log_progress:
                elapsed = time.perf_counter() - stage_start
                avg_iter = elapsed / max(iterations_run, 1)
                eta = max(0.0, avg_iter * max(0, num_iters - iterations_run))
                remaining = self._time_budget_seconds_remaining()
                budget_text = (
                    ""
                    if remaining is None
                    else f", budget_left={max(0.0, remaining):.1f}s"
                )
                cache_stats = self._renderer_cache_stats(renderer)
                cache_text = ""
                if cache_stats:
                    cache_text = f", bin_rebuilds={cache_stats.get('rebuilds', 0)}"
                logger.info(
                    "  Iteration %s/%s: loss=%.6f best=%.6f iter=%.2fs avg=%.2fs eta=%.1fs%s%s",
                    iteration + 1,
                    num_iters,
                    loss_value,
                    best_loss,
                    iter_elapsed,
                    avg_iter,
                    eta,
                    budget_text,
                    cache_text,
                )

            if schedule_enabled and (iteration + 1) % check_interval == 0:
                if best_loss < best_at_last_check - min_delta:
                    best_at_last_check = best_loss
                    no_improve_checks = 0
                else:
                    no_improve_checks += 1
                    if no_improve_checks >= patience_checks:
                        if decay_count >= max_decays:
                            if verbose:
                                logger.info(
                                    "  Early stop at iteration %s/%s after %s LR decays",
                                    iteration + 1,
                                    num_iters,
                                    decay_count,
                                )
                            break
                        for param_group in optimizer.param_groups:
                            param_group["lr"] /= decay_ratio
                        decay_count += 1
                        no_improve_checks = 0
                        if verbose:
                            logger.info(
                                "  LR decay %s/%s at iteration %s/%s (ratio=%.2f)",
                                decay_count,
                                max_decays,
                                iteration + 1,
                                num_iters,
                                decay_ratio,
                            )

        # The loop's best-tracking only ever measures pre-step params, so the
        # final post-step params are unmeasured; on a still-descending loss
        # tail they are the true best. One extra forward closes that gap.
        if iterations_run > 0:
            with torch.no_grad():
                final_loss = float(loss_fn(renderer(params.as_tensor()), target).item())
            end_loss = final_loss
            if final_loss < best_loss:
                best_loss = final_loss
                best_snapshot = params.snapshot()

        # Restore the best-loss snapshot.
        params.restore(best_snapshot)
        self._clear_renderer_cache(renderer)
        with torch.no_grad():
            best_rendered = renderer(params.as_tensor()).detach()
        elapsed_sec = time.perf_counter() - stage_start
        renderer_cache_after = int(
            self._renderer_cache_stats(renderer).get("rebuilds", renderer_cache_before)
        )

        optimized_splats = self._copy_splat_layers(
            splats,
            tensor_to_splats(params.as_tensor().detach()),
        )
        return (
            optimized_splats,
            {
                "start_loss": start_loss,
                "end_loss": end_loss,
                "best_loss": best_loss,
                "iterations": int(iterations_run),
                "lr_decays": int(decay_count),
                "stopped_for_time_budget": bool(stopped_for_time_budget),
                "elapsed_sec": float(elapsed_sec),
                "avg_iter_sec": float(elapsed_sec / max(iterations_run, 1)),
                "progress_log_interval": int(progress_interval),
                "renderer_tile_bin_rebuilds": int(
                    renderer_cache_after - renderer_cache_before
                ),
            },
            best_rendered,
        )

    def _optimize_stage_mlx(
        self,
        splats: List[GaussianSplat],
        target: torch.Tensor,
        num_iters: int,
        verbose: bool,
    ) -> Tuple[List[GaussianSplat], Dict[str, Any], torch.Tensor]:
        """Optimize one stage with the experimental MLX stage runner."""
        height = int(target.shape[0])
        width = int(target.shape[1])
        if not splats:
            empty = torch.zeros(
                (height, width, 3), dtype=torch.float32, device=self.device
            )
            return (
                splats,
                {
                    "optimizer_backend": "mlx",
                    "start_loss": 0.0,
                    "end_loss": 0.0,
                    "best_loss": 0.0,
                    "iterations": 0,
                },
                empty,
            )

        tile_size = int(
            np.clip(self.refinement_config.get("renderer_tile_size", 16), 4, 128)
        )
        # The release benchmark on three real 384 px corpus checkpoints found
        # eight tiles consistently faster than 16/32/128 for a forward+backward
        # pass (12-47% depending on content).  Keep this MLX-specific default
        # separate from the Torch renderer's larger batch default.
        batch_tile_count = int(
            max(1, self.refinement_config.get("renderer_batch_tile_count", 8))
        )
        max_active_raw = self.refinement_config.get(
            "renderer_max_active_splats_per_tile"
        )
        max_active_splats_per_tile = (
            None if max_active_raw in (None, "", 0) else int(max_active_raw)
        )
        default_progress_interval = max(
            1, min(10, int(np.ceil(max(1, num_iters) / 6.0)))
        )
        progress_interval = int(
            max(
                1,
                self.refinement_config.get(
                    "progress_log_interval", default_progress_interval
                ),
            )
        )
        # Mirror the torch path (_create_training_renderer): when the
        # training_export_target is svg or pptx-softedge, composite in sRGB
        # so the trained splats match what the governing browser produces when
        # the SVG is rendered. Pure linear-light pixel-runtime training keeps
        # compositing_space="linear". For pptx-softedge, also apply the
        # sigma/alpha proxy transform that compensates for PowerPoint's
        # brighter-than-Gaussian soft-edge rendering.
        mlx_compositing_space = (
            "srgb"
            if self.training_export_target in {"svg", "pptx-softedge"}
            else self.compositing_space
        )
        pptx_softedge_mode = self._use_pptx_proxy_training()
        # Mirror the torch training renderer: SVG/PPTX targets deploy via
        # source-over, so force alpha-over alongside the sRGB compositing.
        mlx_blend_mode = (
            "alpha-over"
            if self.training_export_target in {"svg", "pptx-softedge"}
            else self.blend_mode
        )
        stage_config = MlxStageConfig(
            renderer=MlxRendererConfig(
                tile_size=tile_size,
                batch_tile_count=batch_tile_count,
                blend_mode=mlx_blend_mode,
                background_color=tuple(
                    float(v) for v in self._background_linear_rgb[:3]
                ),
                max_active_splats_per_tile=max_active_splats_per_tile,
                compositing_space=mlx_compositing_space,
                pptx_softedge_mode=pptx_softedge_mode,
                pptx_alpha_scale=float(
                    self.refinement_config.get(
                        "pptx_proxy_train_alpha_scale", PPTX_SOFT_EDGE_ALPHA_SCALE
                    )
                ),
                pptx_sigma_scale=float(
                    self.refinement_config.get(
                        "pptx_proxy_train_sigma_scale", PPTX_SOFT_EDGE_K_SIGMA_SCALE
                    )
                ),
            ),
            loss=MlxLossConfig(
                name=self.mlx_loss,
                l1_weight=float(self.loss_weights.get("l1_weight", 1.0)),
                ssim_weight=float(self.loss_weights.get("ssim_weight", 0.2)),
                gradient_weight=float(self.loss_weights.get("gradient_weight", 0.0)),
            ),
            trainable_groups=self.mlx_trainable_groups,
            tile_plan_mode=self.mlx_tile_plan,
            tile_plan_rebuild_interval=self.mlx_tile_plan_rebuild_interval,
            progress_interval=progress_interval,
            schedule=dict(self.schedule_config),
        )
        target_np = target.detach().cpu().numpy()
        spatial_weight_map = None
        if (
            self._use_mlx_spatial_weights()
            and self._region_weight_map is not None
            and self._region_weight_map.shape == (height, width)
        ):
            spatial_weight_map = self._region_weight_map
        result = optimize_stage_mlx(
            splats=splats,
            target_linear_rgb=target_np,
            width=width,
            height=height,
            num_iters=num_iters,
            config=stage_config,
            learning_rates=self.learning_rates,
            spatial_weight_map=spatial_weight_map,
            should_stop=self._time_budget_exhausted,
            verbose=verbose,
        )
        rendered = torch.from_numpy(result.rendered_linear_rgb).to(
            device=self.device,
            dtype=target.dtype,
        )
        return list(result.splats), dict(result.metrics), rendered

    def _compute_quality_metrics(
        self,
        splats: List[GaussianSplat],
        target: torch.Tensor,
        renderer: torch.nn.Module,
        loss_fn: torch.nn.Module,
    ) -> Dict[str, float]:
        """Compute stage-level quality metrics."""
        metrics, _, _ = self._compute_quality_metrics_cached(
            splats=splats,
            target=target,
            renderer=renderer,
            loss_fn=loss_fn,
        )
        return metrics

    def _compute_quality_metrics_cached(
        self,
        splats: List[GaussianSplat],
        target: torch.Tensor,
        renderer: torch.nn.Module,
        loss_fn: torch.nn.Module,
        precomputed_rendered: Optional[torch.Tensor] = None,
        precomputed_coverage_map: Optional[np.ndarray] = None,
    ) -> Tuple[Dict[str, float], torch.Tensor, np.ndarray]:
        """Compute quality metrics while optionally reusing rendered and coverage maps."""
        height, width = int(target.shape[0]), int(target.shape[1])
        if not splats:
            empty_render = torch.zeros(
                (height, width, 3), dtype=target.dtype, device=target.device
            )
            empty_coverage = np.zeros((height, width), dtype=np.float32)
            return (
                {
                    "l1": 0.0,
                    "mse": 0.0,
                    "psnr": 0.0,
                    "ssim": 0.0,
                    "psnr_srgb": 0.0,
                    "ssim_srgb": 0.0,
                    "coverage": 0.0,
                },
                empty_render,
                empty_coverage,
            )

        if precomputed_rendered is None:
            with torch.no_grad():
                rendered = renderer(
                    splats_to_tensor(splats, device=self.device)
                ).detach()
        else:
            rendered = precomputed_rendered.detach()

        # Use the honest shared metric: standard windowed SSIM plus perceptual
        # (sRGB-display) variants. The old path used L1SSIMLoss._global_ssim, a
        # global single-window SSIM that over-reports, and omitted the
        # psnr_srgb/ssim_srgb keys the acceptance gate checks, so the internal
        # diagnostic metrics were incomplete when browser capture was absent.
        with torch.no_grad():
            target_np = target.detach().cpu().numpy()
            rendered_np = rendered.detach().cpu().numpy()
        metrics = compute_quality_metrics(target_np[..., :3], rendered_np[..., :3])

        if precomputed_coverage_map is not None and precomputed_coverage_map.shape == (
            height,
            width,
        ):
            coverage_map = precomputed_coverage_map
        else:
            coverage_map = self._build_alpha_coverage_map(
                splats=splats, width=width, height=height
            )
        coverage = self._compute_coverage_ratio(coverage_map)
        metrics["coverage"] = coverage
        return (
            metrics,
            rendered,
            coverage_map,
        )
