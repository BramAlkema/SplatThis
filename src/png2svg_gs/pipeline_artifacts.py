"""Emit, evaluate, report, and finalize a fitted splat scene."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import numpy as np
import torch

from .artifact_backends import ArtifactBackend
from .domain import ArtifactEvaluation, ArtifactPayload, SplatScene
from .pipeline import RunContext
from .pipeline_phases import PreparedInput
from .quality import compute_quality_metrics
from .renderer import render_splats_numpy
from .reporting import save_side_by_side_html
from .roundtrip import validate_export_roundtrip
from .storage import save_linear_rgb_png, save_splats_json

if TYPE_CHECKING:
    from .converter import PNG2SVGConverter

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvaluationBundle:
    internal_metrics: Dict[str, Any]
    final_metrics: Dict[str, Any]
    export_quality: Dict[str, Any]
    artifact_evaluation: ArtifactEvaluation
    acceptance: Dict[str, Any]
    preview_linear: np.ndarray


def _emit_and_write(
    *,
    converter: "PNG2SVGConverter",
    context: RunContext,
    manifest: Dict[str, Any],
    prepared: PreparedInput,
    scene: SplatScene,
    backend: ArtifactBackend,
) -> ArtifactPayload:
    started = time.perf_counter()
    payload = backend.emit(
        converter=converter,
        request=context.request,
        scene=scene,
        target_linear_rgb=prepared.image,
        artifacts_path=context.artifacts_path,
    )
    for key, value in payload.metadata.items():
        if key == "config":
            manifest["config"].update(value)
        elif key == "svg_compositor_gate":
            manifest[key] = value
    manifest["artifact_backend"] = {
        "format": payload.output_format,
        "media_type": payload.media_type,
    }
    context.timings["generate_output"] = float(time.perf_counter() - started)

    if context.request.output_path:
        started = time.perf_counter()
        manifest.update(
            backend.write(
                converter=converter,
                request=context.request,
                scene=scene,
                payload=payload,
            )
        )
        if context.request.save_json:
            save_splats_json(
                list(scene.splats),
                str(Path(context.request.output_path).with_suffix(".json")),
            )
        context.timings["write_output"] = float(time.perf_counter() - started)
    return payload


def _evaluate_scene(
    *,
    converter: "PNG2SVGConverter",
    context: RunContext,
    prepared: PreparedInput,
    scene: SplatScene,
    backend: ArtifactBackend,
) -> EvaluationBundle:
    splats = list(scene.splats)
    elapsed = time.perf_counter() - context.start_time
    target = torch.from_numpy(prepared.image[:, :, :3]).to(converter.device)
    started = time.perf_counter()
    internal = converter._compute_quality_metrics(
        splats,
        target,
        converter._create_training_renderer(scene.width, scene.height),
        converter._create_training_loss(
            target=target, width=scene.width, height=scene.height
        ),
    )
    context.timings["internal_metrics"] = float(time.perf_counter() - started)
    internal.update(runtime_sec=float(elapsed), splat_count=float(len(splats)))

    started = time.perf_counter()
    preview = render_splats_numpy(
        splats,
        scene.width,
        scene.height,
        background_linear_rgb=scene.background_linear_rgb,
        compositing_space=scene.compositing_space,
    )
    context.timings["proxy_render"] = float(time.perf_counter() - started)
    started = time.perf_counter()
    proxy_metrics = compute_quality_metrics(prepared.image[:, :, :3], preview)
    context.timings["proxy_metrics"] = float(time.perf_counter() - started)
    proxy_quality = {
        "available": True,
        "method": "proxy-render",
        "used_fallback": True,
        "metrics": proxy_metrics,
    }

    governing = backend.requires_governing_render(converter)
    if governing and context.request.output_path:
        started = time.perf_counter()
    evaluation = backend.evaluate(
        converter=converter,
        request=context.request,
        target_linear_rgb=prepared.image[:, :, :3],
        fallback_linear_rgb=preview,
        proxy_quality=proxy_quality,
    )
    if governing and context.request.output_path:
        context.timings[f"{context.request.output_format}_export_quality"] = float(
            time.perf_counter() - started
        )

    export_quality = dict(evaluation.quality)
    final_metrics = (
        dict(export_quality.get("metrics") or {})
        if evaluation.acceptance_eligible
        else dict(internal)
    )
    final_metrics.update(runtime_sec=float(elapsed), splat_count=float(len(splats)))
    if "coverage" not in final_metrics and "coverage" in internal:
        final_metrics["coverage"] = internal["coverage"]

    criteria = (
        dict(context.request.acceptance_criteria)
        if context.request.acceptance_overridden
        else converter.acceptance_criteria.copy()
    )
    acceptance = converter._evaluate_acceptance(final_metrics, criteria)
    if governing:
        acceptance["checks"]["governing_browser_render"] = bool(
            evaluation.acceptance_eligible
        )
        acceptance["pass"] = bool(acceptance["pass"] and evaluation.acceptance_eligible)
        if not evaluation.acceptance_eligible:
            acceptance["reason"] = "governing-browser-render-unavailable"
    return EvaluationBundle(
        internal_metrics=internal,
        final_metrics=final_metrics,
        export_quality=export_quality,
        artifact_evaluation=evaluation,
        acceptance=acceptance,
        preview_linear=preview,
    )


def _write_diagnostics(
    *,
    converter: "PNG2SVGConverter",
    context: RunContext,
    scene: SplatScene,
    backend: ArtifactBackend,
    evaluation: EvaluationBundle,
) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    roundtrip: Optional[Dict[str, Any]] = None
    if context.request.validate_roundtrip:
        started = time.perf_counter()
        roundtrip = validate_export_roundtrip(
            splats=list(scene.splats),
            width=scene.width,
            height=scene.height,
            k_sigma=converter.k_sigma,
        )
        context.timings["roundtrip_validation"] = float(time.perf_counter() - started)

    preview_path = context.request.preview_png_path
    if preview_path is None and context.request.side_by_side_html:
        source = Path(context.request.output_path or context.request.side_by_side_html)
        preview_path = str(source.with_name(f"{source.stem}_splat_proxy.png"))
    if preview_path:
        started = time.perf_counter()
        save_linear_rgb_png(evaluation.preview_linear, preview_path)
        context.timings["splat_proxy_png"] = float(time.perf_counter() - started)
    if context.request.side_by_side_html:
        started = time.perf_counter()
        export_metrics = evaluation.export_quality.get("metrics") or {}
        save_side_by_side_html(
            output_path=context.request.side_by_side_html,
            source_png_path=context.request.input_path,
            svg_path=(
                context.request.output_path
                if backend.svg_comparison_artifact and context.request.output_path
                else ""
            ),
            preview_png_path=preview_path,
            title="PNG2Splat Side-by-Side",
            metrics={
                "output_format": context.request.output_format,
                "internal_psnr": evaluation.internal_metrics.get("psnr"),
                "internal_ssim": evaluation.internal_metrics.get("ssim"),
                "export_method": evaluation.export_quality.get("method"),
                "export_psnr": export_metrics.get("psnr"),
                "export_ssim": export_metrics.get("ssim"),
                "runtime_sec": time.perf_counter() - context.start_time,
                "splats": len(scene.splats),
            },
        )
        context.timings["side_by_side_html"] = float(time.perf_counter() - started)
    return preview_path, roundtrip


def _finalize_manifest(
    *,
    converter: "PNG2SVGConverter",
    context: RunContext,
    manifest: Dict[str, Any],
    scene: SplatScene,
    evaluation: EvaluationBundle,
    preview_path: Optional[str],
    roundtrip: Optional[Dict[str, Any]],
) -> None:
    elapsed = time.perf_counter() - context.start_time
    evaluation.internal_metrics["runtime_sec"] = float(elapsed)
    evaluation.final_metrics["runtime_sec"] = float(elapsed)
    remaining = converter._time_budget_seconds_remaining()
    context.timings["total_wall"] = float(elapsed)
    manifest.update(
        {
            "total_time_sec": elapsed,
            "time_budget_remaining_sec": (
                None if remaining is None else max(0.0, float(remaining))
            ),
            "time_budget_exhausted": bool(converter._time_budget_exhausted()),
            "final_splat_count": len(scene.splats),
            "layered_saliency": converter._layer_summary(list(scene.splats)),
            "final_metrics": evaluation.final_metrics,
            "internal_metrics": evaluation.internal_metrics,
            "export_quality": evaluation.export_quality,
            "acceptance_metric_source": evaluation.artifact_evaluation.metric_source,
            "artifact_evaluation": (evaluation.artifact_evaluation.as_manifest_dict()),
            "artifacts": {
                "primary": {
                    "path": context.request.output_path,
                    "format": context.request.output_format,
                    "is_deployed_artifact": bool(context.request.output_path),
                },
                "splat_proxy": {
                    "path": preview_path,
                    "render_kind": "internal-splat-proxy",
                    "renderer": "render_splats_numpy",
                    "is_deployed_artifact": False,
                },
            },
            "acceptance": evaluation.acceptance,
        }
    )
    if roundtrip is not None:
        manifest["roundtrip_validation"] = roundtrip
    started = time.perf_counter()
    converter._write_manifest(context.artifacts_path, manifest)
    context.timings["write_manifest"] = float(time.perf_counter() - started)
    converter._time_budget_deadline = None


def emit_evaluate_and_finalize(
    *,
    converter: "PNG2SVGConverter",
    context: RunContext,
    manifest: Dict[str, Any],
    prepared: PreparedInput,
    scene: SplatScene,
    backend: ArtifactBackend,
) -> str:
    """Run the deployment half of the pipeline and return emitted content."""

    payload = _emit_and_write(
        converter=converter,
        context=context,
        manifest=manifest,
        prepared=prepared,
        scene=scene,
        backend=backend,
    )
    evaluation = _evaluate_scene(
        converter=converter,
        context=context,
        prepared=prepared,
        scene=scene,
        backend=backend,
    )
    preview_path, roundtrip = _write_diagnostics(
        converter=converter,
        context=context,
        scene=scene,
        backend=backend,
        evaluation=evaluation,
    )
    _finalize_manifest(
        converter=converter,
        context=context,
        manifest=manifest,
        scene=scene,
        evaluation=evaluation,
        preview_path=preview_path,
        roundtrip=roundtrip,
    )
    if context.request.verbose:
        logger.info(
            "Conversion completed in %.2fs: splats=%s acceptance=%s source=%s",
            context.timings["total_wall"],
            len(scene.splats),
            evaluation.acceptance.get("pass"),
            evaluation.artifact_evaluation.metric_source,
        )
    return payload.content
