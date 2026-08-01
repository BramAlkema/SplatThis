"""Explicit load/analyze/fit phases used by the conversion coordinator."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional

import numpy as np
import numpy.typing as npt

from .artifact_backends import ArtifactBackend
from .domain import SplatScene
from .features import compute_structure_field
from .pipeline import RunContext
from .storage import load_png

if TYPE_CHECKING:
    from .conversion_engine import ConversionEngine

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreparedInput:
    """Image analysis consumed by initialization and densification."""

    image: npt.NDArray[Any]
    width: int
    height: int
    structure_primary: Optional[npt.NDArray[Any]]
    structure_anisotropy: Optional[npt.NDArray[Any]]


def _install_guidance(
    converter: "ConversionEngine", guidance: Optional[Dict[str, Any]]
) -> None:
    if guidance is None:
        return
    converter._region_weight_map = guidance["weight_map"]
    converter._region_saliency_map = guidance.get("saliency_map")
    converter._region_detail_priority_map = guidance.get("detail_priority_map")
    converter._region_background_penalty_map = guidance.get("background_penalty_map")
    converter._region_foreground_mask = guidance["foreground_mask"]
    converter._region_background_safe_mask = guidance["background_safe_mask"]
    converter._region_edge_band_mask = guidance["edge_band_mask"]
    converter._background_linear_rgb = guidance["background_linear_rgb"]


def build_run_manifest(
    *,
    converter: "ConversionEngine",
    context: RunContext,
    backend: ArtifactBackend,
    resolved_target_size: tuple[int, int],
) -> Dict[str, Any]:
    """Create the stable manifest envelope before mutable budget resolution."""

    output_format = context.request.output_format
    return {
        "input_path": context.request.input_path,
        "input_sha256": converter._sha256_file(context.request.input_path),
        "config_sha256": context.config.fingerprint(),
        "seed": context.run_seed,
        "config": {
            "requested_max_splats": converter.requested_max_splats,
            "max_splats": converter.max_splats,
            "k_sigma": converter.k_sigma,
            "stages": list(converter.stages),
            "target_size": converter.target_size,
            "resolved_target_size": resolved_target_size,
            "resolution_scale": converter.resolution_scale,
            "gradient_method": converter.gradient_method,
            "init_random_ratio": converter.init_random_ratio,
            "init_gradient_weight": converter.init_gradient_weight,
            "device": str(converter.device),
            "renderer_backend": converter.renderer_backend,
            "resolved_renderer_backend": converter.resolved_renderer_backend,
            "optimizer_backend": converter.optimizer_backend,
            "mlx_loss": (
                converter.mlx_loss if converter.optimizer_backend == "mlx" else None
            ),
            "mlx_spatial_weighting_enabled": (
                converter.mlx_spatial_weighting_enabled
                if converter.optimizer_backend == "mlx"
                else None
            ),
            "mlx_tile_plan": (
                converter.mlx_tile_plan
                if converter.optimizer_backend == "mlx"
                else None
            ),
            "mlx_tile_plan_rebuild_interval": (
                converter.mlx_tile_plan_rebuild_interval
                if converter.optimizer_backend == "mlx"
                else None
            ),
            "mlx_trainable_groups": (
                list(converter.mlx_trainable_groups)
                if converter.optimizer_backend == "mlx"
                else None
            ),
            "blend_mode": converter.blend_mode,
            "output_format": output_format,
            "training_export_target": converter.training_export_target,
            "pptx_export_mode": backend.pptx_export_mode,
            "pptx_splat_style": converter.pptx_splat_style,
            "pptx_painter_order": converter.pptx_painter_order,
            "quality_profile": converter.quality_profile,
            "loss_weights": converter.loss_weights,
            "learning_rates": converter.learning_rates,
            "refinement_config": converter.refinement_config,
            "schedule_config": converter.schedule_config,
            "region_weighting_enabled": converter.region_weighting_enabled,
            "svg_export_recipe": converter.svg_export_recipe,
            "svg_gradient_quality": converter.svg_gradient_quality,
            "svg_painter_order": converter.svg_painter_order,
            "svg_compositor_gate": converter.svg_compositor_gate,
            "time_budget": converter.time_budget,
            "time_budget_plan": converter.time_budget_plan,
            "platform_splat_cap": converter._platform_splat_cap,
            "apple_silicon_splat_cap": converter.apple_silicon_splat_cap,
            "layered_saliency": converter.layered_saliency,
            "adaptive_compute": converter.adaptive_compute_config.as_dict(),
        },
        "stages": [],
        "timings_sec": context.timings,
    }


def prepare_input(
    *,
    converter: "ConversionEngine",
    context: RunContext,
    manifest: Dict[str, Any],
    resolved_target_size: tuple[int, int],
) -> PreparedInput:
    """Load the image and resolve guidance, budget, and structure fields."""

    request = context.request
    timings = context.timings
    if request.verbose:
        logger.info(
            "Run start: input=%s output_format=%s seed=%s requested_splats=%s device=%s",
            request.input_path,
            request.output_format,
            context.run_seed,
            converter.requested_max_splats,
            converter.device,
        )
        logger.info(
            "Loading PNG: %s (target_size=%s, resolution_scale=%.2f)",
            request.input_path,
            resolved_target_size,
            converter.resolution_scale,
        )
    phase_started = time.perf_counter()
    image = load_png(request.input_path, target_size=resolved_target_size)
    timings["load_png"] = float(time.perf_counter() - phase_started)
    height, width = image.shape[:2]
    if request.verbose:
        logger.info("Loaded %sx%s image in %.2fs", width, height, timings["load_png"])

    converter._image_width = width
    converter._image_height = height
    converter._background_linear_rgb = converter._estimate_background_color(image)
    converter._region_weight_map = None
    converter._region_saliency_map = None
    converter._region_detail_priority_map = None
    converter._region_background_penalty_map = None
    converter._region_foreground_mask = None
    converter._region_background_safe_mask = None
    converter._region_edge_band_mask = None

    guidance: Optional[Dict[str, Any]] = None
    if converter._needs_region_guidance():
        phase_started = time.perf_counter()
        guidance = converter._compute_region_guidance(image)
        timings["region_guidance"] = float(time.perf_counter() - phase_started)
        manifest["config"]["region_guidance"] = guidance["summary"]
        if request.verbose:
            summary = guidance["summary"]
            logger.info(
                "Region guidance in %.2fs: foreground=%.1f%% edge=%.1f%% "
                "detail_mean=%.4f detail_p95=%.4f",
                timings["region_guidance"],
                100.0 * float(summary.get("foreground_ratio", 0.0)),
                100.0 * float(summary.get("edge_band_ratio", 0.0)),
                float(
                    summary.get(
                        "detail_priority_mean", summary.get("saliency_mean", 0.0)
                    )
                ),
                float(
                    summary.get("detail_priority_p95", summary.get("saliency_p95", 0.0))
                ),
            )

    if converter.time_budget is not None:
        phase_started = time.perf_counter()
        plan = converter._apply_time_budget_plan(width, height, guidance)
        timings["time_budget_plan"] = float(time.perf_counter() - phase_started)
        converter.time_budget_plan = plan
        converter._time_budget_deadline = context.start_time + float(
            plan["target_seconds"]
        )
        manifest["config"].update(
            {
                "max_splats": converter.max_splats,
                "stages": list(converter.stages),
                "refinement_config": converter.refinement_config,
                "time_budget_plan": plan,
                "target_runtime_sec": float(plan["target_seconds"]),
                "time_budget_deadline_enabled": True,
            }
        )
        if "max_splats" in converter.acceptance_criteria:
            converter.acceptance_criteria["max_splats"] = float(converter.max_splats)
        if "max_runtime_sec" in converter.acceptance_criteria:
            converter.acceptance_criteria["max_runtime_sec"] = float(
                plan["target_seconds"]
            )
        if request.verbose:
            logger.info(
                "Applied %s budget: max_splats=%s, stages=%s, "
                "saliency_multiplier=%.2f",
                plan["label"],
                converter.max_splats,
                converter.stages,
                plan["saliency_multiplier"],
            )
    else:
        converter._time_budget_deadline = None
        manifest["config"]["time_budget_deadline_enabled"] = False

    _install_guidance(converter, guidance)

    structure_enabled = bool(
        converter.refinement_config.get("structure_precompute_enabled", False)
    )
    smoothing_sigma = float(
        max(0.0, converter.refinement_config.get("structure_smoothing_sigma", 0.0))
    )
    anisotropy_clip = float(
        max(1.0, converter.refinement_config.get("structure_anisotropy_clip", 10.0))
    )
    min_coherence = float(
        np.clip(
            converter.refinement_config.get("structure_min_coherence", 0.12),
            0.0,
            1.0,
        )
    )
    structure_primary: Optional[npt.NDArray[Any]] = None
    structure_anisotropy: Optional[npt.NDArray[Any]] = None
    if structure_enabled:
        phase_started = time.perf_counter()
        structure_primary, structure_anisotropy = compute_structure_field(
            image=image,
            method=converter.gradient_method,
            smoothing_sigma=smoothing_sigma,
            anisotropy_clip=anisotropy_clip,
            min_coherence=min_coherence,
        )
        timings["structure_precompute"] = float(time.perf_counter() - phase_started)
    manifest["config"].update(
        {
            "structure_smoothing_sigma": smoothing_sigma,
            "structure_anisotropy_clip": anisotropy_clip,
            "structure_min_coherence": min_coherence,
            "structure_precompute_enabled": structure_enabled,
            "background_linear_rgb": [
                float(converter._background_linear_rgb[0]),
                float(converter._background_linear_rgb[1]),
                float(converter._background_linear_rgb[2]),
            ],
        }
    )
    return PreparedInput(
        image=image,
        width=width,
        height=height,
        structure_primary=structure_primary,
        structure_anisotropy=structure_anisotropy,
    )


def fit_scene(
    *,
    converter: "ConversionEngine",
    context: RunContext,
    manifest: Dict[str, Any],
    prepared: PreparedInput,
    backend: ArtifactBackend,
) -> SplatScene:
    """Initialize, optimize, refine, and artifact-gate one splat scene."""

    request = context.request
    timings = context.timings
    image = prepared.image
    artifacts_path = context.artifacts_path

    phase_started = time.perf_counter()
    splats = converter._initialize_splats(
        image,
        rng=context.rng,
        structure_primary=prepared.structure_primary,
        structure_anisotropy=prepared.structure_anisotropy,
    )
    timings["initialize_splats"] = float(time.perf_counter() - phase_started)
    converter._write_stage_artifact(
        artifacts_path, "init", splats, {"count": len(splats)}
    )

    phase_started = time.perf_counter()
    splats, stage_metrics = converter._optimize_splats(
        image=image,
        splats=splats,
        rng=context.rng,
        verbose=request.verbose,
        artifacts_dir=artifacts_path,
        structure_primary=prepared.structure_primary,
        structure_anisotropy=prepared.structure_anisotropy,
        monotonic_stage_selection=(
            backend.monotonic_canvas_selection
            and bool(
                converter.refinement_config.get(
                    "canvas_monotonic_stage_selection_enabled", True
                )
            )
        ),
    )
    timings["optimize_splats"] = float(time.perf_counter() - phase_started)
    manifest["stages"].extend(stage_metrics)

    # Recorded whether or not it fired. A run whose population was halved for
    # host-memory reasons is otherwise indistinguishable from one that simply
    # converged smaller, and a fixed seed no longer explains the output.
    guard = getattr(converter, "_memory_guard", None)
    if guard is not None:
        manifest["memory_guard"] = dict(guard)

    # Whether this run's own bytes are reproducible, travelling with the run
    # rather than living in the README. MLX orders float32 reductions on the
    # Metal device nondeterministically, so a repeated seeded run differs by
    # about one ULP in splat parameters -- far below any quality threshold, but
    # enough to move a rounded attribute and change the artifact hash. Anyone
    # diffing artifacts or comparing `artifact_sha256` across runs needs to know
    # that from the manifest, not from prose they may never have read.
    manifest["artifact_hash_stable"] = converter.optimizer_backend != "mlx"
    if not manifest["artifact_hash_stable"]:
        manifest["artifact_hash_stable_note"] = (
            "MLX orders float32 reductions nondeterministically; repeated seeded "
            "runs agree on metrics to ~9 significant figures but are not "
            "byte-identical. Use --optimizer-backend torch for stable hashes."
        )

    phase_started = time.perf_counter()
    optimized_splats = list(splats)
    postprocessed = converter._postprocess_splats(
        splats=splats, image=image, rng=context.rng
    )
    if backend.monotonic_canvas_selection and bool(
        converter.refinement_config.get("canvas_monotonic_postprocess_enabled", True)
    ):
        splats, gate = converter._select_monotonic_canvas_postprocess(
            optimized_splats=optimized_splats,
            postprocessed_splats=postprocessed,
            image=image,
        )
        manifest["canvas_postprocess_gate"] = gate
    else:
        splats = postprocessed
    timings["postprocess_splats"] = float(time.perf_counter() - phase_started)

    svg_iters = int(
        max(0, converter.refinement_config.get("svg_proxy_postfit_iters", 0))
    )
    if backend.postfit_family == "svg" and svg_iters > 0:
        phase_started = time.perf_counter()
        splats, metric = converter._postfit_splats_for_svg_proxy(
            splats=splats,
            image=image,
            width=prepared.width,
            height=prepared.height,
            num_iters=svg_iters,
            verbose=request.verbose,
        )
        timings["svg_proxy_postfit"] = float(time.perf_counter() - phase_started)
        manifest["stages"].append(metric)
        converter._write_stage_artifact(artifacts_path, "svg-postfit", splats, metric)

    pptx_iters = int(
        max(0, converter.refinement_config.get("pptx_proxy_postfit_iters", 0))
    )
    if (
        backend.postfit_family == "pptx"
        and pptx_iters > 0
        and not backend.is_blur_target(converter)
    ):
        phase_started = time.perf_counter()
        splats, metric = converter._postfit_splats_for_pptx_proxy(
            splats=splats,
            image=image,
            width=prepared.width,
            height=prepared.height,
            num_iters=pptx_iters,
            verbose=request.verbose,
        )
        timings["pptx_proxy_postfit"] = float(time.perf_counter() - phase_started)
        manifest["stages"].append(metric)
        converter._write_stage_artifact(artifacts_path, "pptx-postfit", splats, metric)

    blur_iters = int(
        max(0, converter.refinement_config.get("blur_proxy_postfit_iters", 0))
    )
    if backend.is_blur_target(converter) and blur_iters > 0:
        phase_started = time.perf_counter()
        splats, metric = converter._postfit_splats_for_blur_proxy(
            splats=splats,
            image=image,
            width=prepared.width,
            height=prepared.height,
            num_iters=blur_iters,
            verbose=request.verbose,
        )
        timings["blur_proxy_postfit"] = float(time.perf_counter() - phase_started)
        manifest["stages"].append(metric)
        converter._write_stage_artifact(artifacts_path, "blur-postfit", splats, metric)

    if converter.fidelity_config.mode != "off":
        if backend.supports_fidelity_stage:
            phase_started = time.perf_counter()
            splats, fragment = converter._run_fidelity_stage(
                splats=splats,
                image=image,
                width=prepared.width,
                height=prepared.height,
                artifacts_path=artifacts_path,
                seed=request.seed,
                verbose=request.verbose,
            )
            timings["fidelity_stage"] = float(time.perf_counter() - phase_started)
            manifest["fidelity_stage"] = fragment
        else:
            manifest["fidelity_stage"] = {
                "enabled": False,
                "mode": converter.fidelity_config.mode,
                "reason": (
                    f"unsupported-target:{request.output_format} "
                    "(ADR-003 phase 1 implements svg only)"
                ),
            }

    converter._write_stage_artifact(
        artifacts_path, "final", splats, {"count": len(splats)}
    )
    return SplatScene(
        width=prepared.width,
        height=prepared.height,
        splats=splats,
        background_linear_rgb=converter._background_linear_rgb,
        compositing_space=converter._deployed_compositing_space(),
    )
