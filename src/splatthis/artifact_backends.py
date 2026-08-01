"""Deployment-target backends for emission, persistence, and evaluation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Mapping

import numpy as np

from .artifact_evaluation import (
    evaluate_css_export_quality,
    evaluate_native_canvas_export_quality,
    evaluate_pixel_runtime_export_quality,
    evaluate_svg_export_quality,
)
from .browser_export import generate_css_splat_html, generate_native_canvas_html
from .config import SUPPORTED_OUTPUT_FORMATS, ConversionRequest
from .domain import ArtifactEvaluation, ArtifactPayload, EvidenceLevel, SplatScene
from .pixel_runtime import (
    generate_parallax_pixel_runtime_html,
    generate_webgl_pixel_runtime_html,
)
from .pptx_export import save_pptx_with_drawingml_content
from .storage import atomic_write_text

if TYPE_CHECKING:
    from .conversion_engine import ConversionEngine


class ArtifactBackend(ABC):
    """One deployment target's complete artifact lifecycle."""

    output_format: str
    media_type: str
    default_training_target: str = "pixel-runtime"
    postfit_family: str | None = None
    supports_fidelity_stage: bool = False
    svg_comparison_artifact: bool = False
    pptx_export_mode: str | None = None

    @property
    def monotonic_canvas_selection(self) -> bool:
        return False

    def is_blur_target(self, converter: "ConversionEngine") -> bool:
        return False

    def requires_governing_render(self, converter: "ConversionEngine") -> bool:
        return False

    @abstractmethod
    def emit(
        self,
        *,
        converter: "ConversionEngine",
        request: ConversionRequest,
        scene: SplatScene,
        target_linear_rgb: np.ndarray,
        artifacts_path: Path | None,
    ) -> ArtifactPayload:
        """Emit the primary target exactly once."""

    def write(
        self,
        *,
        converter: "ConversionEngine",
        request: ConversionRequest,
        scene: SplatScene,
        payload: ArtifactPayload,
    ) -> Mapping[str, Any]:
        if request.output_path is None:
            return {}
        atomic_write_text(request.output_path, payload.content)
        return {}

    def evaluate(
        self,
        *,
        converter: "ConversionEngine",
        request: ConversionRequest,
        target_linear_rgb: np.ndarray,
        fallback_linear_rgb: np.ndarray,
        proxy_quality: Mapping[str, Any],
    ) -> ArtifactEvaluation:
        quality = dict(proxy_quality)
        return ArtifactEvaluation(
            evidence_level=EvidenceLevel.PROXY,
            render_kind="internal-proxy",
            renderer="internal-splat-renderer",
            metric_source="internal",
            quality=quality,
            acceptance_eligible=False,
            artifact_path=(
                None if request.output_path is None else Path(request.output_path)
            ),
        )

    @staticmethod
    def _browser_evaluation(
        *,
        request: ConversionRequest,
        proxy_quality: Mapping[str, Any],
        evaluate: Any,
        success_kind: str,
        unavailable_kind: str,
        target_linear_rgb: np.ndarray,
        fallback_linear_rgb: np.ndarray,
    ) -> ArtifactEvaluation:
        quality = dict(proxy_quality)
        if request.output_path is not None:
            quality = evaluate(
                target_linear_rgb=target_linear_rgb,
                fallback_linear_rgb=fallback_linear_rgb,
            )
        method = str(quality.get("method", ""))
        eligible = bool(
            quality.get("available")
            and not method.startswith("proxy")
            and quality.get("metrics") is not None
        )
        renderer = (
            method
            if eligible
            else str(quality.get("governing_method", method or "unavailable"))
        )
        return ArtifactEvaluation(
            evidence_level=(
                EvidenceLevel.DEPLOYED if eligible else EvidenceLevel.UNAVAILABLE
            ),
            render_kind=success_kind if eligible else unavailable_kind,
            renderer=renderer,
            metric_source="export" if eligible else "unavailable",
            quality=quality,
            acceptance_eligible=eligible,
            artifact_path=(
                None if request.output_path is None else Path(request.output_path)
            ),
        )


class SvgArtifactBackend(ArtifactBackend):
    output_format = "svg"
    media_type = "image/svg+xml"
    default_training_target = "svg"
    postfit_family = "svg"
    supports_fidelity_stage = True
    svg_comparison_artifact = True

    def is_blur_target(self, converter: "ConversionEngine") -> bool:
        return converter.svg_export_recipe == "blur"

    def requires_governing_render(self, converter: "ConversionEngine") -> bool:
        return True

    def emit(
        self,
        *,
        converter: "ConversionEngine",
        request: ConversionRequest,
        scene: SplatScene,
        target_linear_rgb: np.ndarray,
        artifacts_path: Path | None,
    ) -> ArtifactPayload:
        metadata: Dict[str, Any] = {}
        if converter.svg_compositor_gate:
            content, gate = converter._select_svg_compositor(
                splats=list(scene.splats),
                image=target_linear_rgb,
                width=scene.width,
                height=scene.height,
                artifacts_path=artifacts_path,
            )
            metadata["svg_compositor_gate"] = gate
            if gate.get("available"):
                selected = gate["selection"]["selected"]
                metadata["config"] = {
                    "selected_svg_painter_order": selected["painter_order"],
                    "selected_svg_gradient_quality": selected["gradient_quality"],
                }
            else:
                metadata["config"] = {
                    "selected_svg_painter_order": gate["fallback_painter_order"],
                    "selected_svg_gradient_quality": gate["fallback_gradient_quality"],
                }
        else:
            content = converter._generate_svg(
                list(scene.splats), scene.width, scene.height
            )
            metadata["svg_compositor_gate"] = {
                "enabled": False,
                "selected_painter_order": converter.svg_painter_order,
                "selected_gradient_quality": converter.svg_gradient_quality,
            }
            metadata["config"] = {
                "selected_svg_painter_order": converter.svg_painter_order,
                "selected_svg_gradient_quality": converter.svg_gradient_quality,
            }
        return ArtifactPayload(
            output_format=self.output_format,
            content=content,
            media_type=self.media_type,
            metadata=metadata,
        )

    def write(
        self,
        *,
        converter: "ConversionEngine",
        request: ConversionRequest,
        scene: SplatScene,
        payload: ArtifactPayload,
    ) -> Mapping[str, Any]:
        metadata = dict(
            super().write(
                converter=converter, request=request, scene=scene, payload=payload
            )
        )
        if request.output_path is not None and converter.svg_optimize:
            from .svg_export import optimize_svg_file

            metadata["svg_optimize"] = optimize_svg_file(
                request.output_path, precision=converter.svg_optimize_precision
            )
        return metadata

    def evaluate(
        self,
        *,
        converter: "ConversionEngine",
        request: ConversionRequest,
        target_linear_rgb: np.ndarray,
        fallback_linear_rgb: np.ndarray,
        proxy_quality: Mapping[str, Any],
    ) -> ArtifactEvaluation:
        return self._browser_evaluation(
            request=request,
            proxy_quality=proxy_quality,
            evaluate=lambda **values: evaluate_svg_export_quality(
                svg_path=request.output_path, **values
            ),
            success_kind="svg-rasterization",
            unavailable_kind="svg-browser-unavailable",
            target_linear_rgb=target_linear_rgb,
            fallback_linear_rgb=fallback_linear_rgb,
        )


class DrawingMLArtifactBackend(ArtifactBackend):
    output_format = "drawingml"
    media_type = (
        "application/vnd.openxmlformats-officedocument.presentationml.slide+xml"
    )

    def emit(
        self,
        *,
        converter: "ConversionEngine",
        request: ConversionRequest,
        scene: SplatScene,
        target_linear_rgb: np.ndarray,
        artifacts_path: Path | None,
    ) -> ArtifactPayload:
        return ArtifactPayload(
            output_format=self.output_format,
            content=converter._generate_drawingml(
                list(scene.splats), scene.width, scene.height
            ),
            media_type=self.media_type,
        )


class PptxArtifactBackend(DrawingMLArtifactBackend):
    output_format = "pptx"
    media_type = (
        "application/vnd.openxmlformats-officedocument.presentationml.presentation"
    )
    postfit_family = "pptx"
    pptx_export_mode = "drawingml-splats"

    def is_blur_target(self, converter: "ConversionEngine") -> bool:
        return converter.pptx_splat_style == "blur"

    def write(
        self,
        *,
        converter: "ConversionEngine",
        request: ConversionRequest,
        scene: SplatScene,
        payload: ArtifactPayload,
    ) -> Mapping[str, Any]:
        if request.output_path is not None:
            save_pptx_with_drawingml_content(
                slide_xml=payload.content,
                width=scene.width,
                height=scene.height,
                output_path=request.output_path,
                splat_count=len(scene.splats),
            )
        return {}

    def evaluate(
        self,
        *,
        converter: "ConversionEngine",
        request: ConversionRequest,
        target_linear_rgb: np.ndarray,
        fallback_linear_rgb: np.ndarray,
        proxy_quality: Mapping[str, Any],
    ) -> ArtifactEvaluation:
        return ArtifactEvaluation(
            evidence_level=EvidenceLevel.PROXY,
            render_kind="pptx-proxy",
            renderer="internal-splat-renderer",
            metric_source="internal",
            quality=dict(proxy_quality),
            acceptance_eligible=False,
            artifact_path=(
                None if request.output_path is None else Path(request.output_path)
            ),
        )


class CanvasArtifactBackend(ArtifactBackend):
    output_format = "canvas"
    media_type = "text/html"
    default_training_target = "svg"

    def requires_governing_render(self, converter: "ConversionEngine") -> bool:
        return True

    def emit(
        self,
        *,
        converter: "ConversionEngine",
        request: ConversionRequest,
        scene: SplatScene,
        target_linear_rgb: np.ndarray,
        artifacts_path: Path | None,
    ) -> ArtifactPayload:
        parallax_strength = float(
            converter.refinement_config.get("canvas_parallax_strength", 0.0)
        )
        return ArtifactPayload(
            output_format=self.output_format,
            content=generate_native_canvas_html(
                list(scene.splats),
                scene.width,
                scene.height,
                background_linear_rgb=scene.background_linear_rgb,
                title=Path(request.input_path).stem,
                parallax_strength=parallax_strength,
                k_sigma=converter.k_sigma,
            ),
            media_type=self.media_type,
            metadata={"parallax_strength": parallax_strength},
        )

    def evaluate(
        self,
        *,
        converter: "ConversionEngine",
        request: ConversionRequest,
        target_linear_rgb: np.ndarray,
        fallback_linear_rgb: np.ndarray,
        proxy_quality: Mapping[str, Any],
    ) -> ArtifactEvaluation:
        return self._browser_evaluation(
            request=request,
            proxy_quality=proxy_quality,
            evaluate=lambda **values: evaluate_native_canvas_export_quality(
                html_path=request.output_path, **values
            ),
            success_kind="canvas-api-browser-capture",
            unavailable_kind="canvas-api-browser-unavailable",
            target_linear_rgb=target_linear_rgb,
            fallback_linear_rgb=fallback_linear_rgb,
        )


class CssArtifactBackend(ArtifactBackend):
    output_format = "css"
    media_type = "text/html"
    default_training_target = "svg"

    def requires_governing_render(self, converter: "ConversionEngine") -> bool:
        return True

    def emit(
        self,
        *,
        converter: "ConversionEngine",
        request: ConversionRequest,
        scene: SplatScene,
        target_linear_rgb: np.ndarray,
        artifacts_path: Path | None,
    ) -> ArtifactPayload:
        parallax_strength = float(
            converter.refinement_config.get("css_parallax_strength", 0.0)
        )
        hover_grid_size = int(
            converter.refinement_config.get("css_hover_grid_size", 10)
        )
        return ArtifactPayload(
            output_format=self.output_format,
            content=generate_css_splat_html(
                list(scene.splats),
                scene.width,
                scene.height,
                background_linear_rgb=scene.background_linear_rgb,
                title=Path(request.input_path).stem,
                parallax_strength=parallax_strength,
                hover_grid_size=hover_grid_size,
                k_sigma=converter.k_sigma,
            ),
            media_type=self.media_type,
            metadata={
                "parallax_strength": parallax_strength,
                "hover_grid_size": hover_grid_size,
            },
        )

    def evaluate(
        self,
        *,
        converter: "ConversionEngine",
        request: ConversionRequest,
        target_linear_rgb: np.ndarray,
        fallback_linear_rgb: np.ndarray,
        proxy_quality: Mapping[str, Any],
    ) -> ArtifactEvaluation:
        return self._browser_evaluation(
            request=request,
            proxy_quality=proxy_quality,
            evaluate=lambda **values: evaluate_css_export_quality(
                html_path=request.output_path, **values
            ),
            success_kind="css-browser-capture",
            unavailable_kind="css-browser-unavailable",
            target_linear_rgb=target_linear_rgb,
            fallback_linear_rgb=fallback_linear_rgb,
        )


class PixelRuntimeArtifactBackend(ArtifactBackend):
    output_format = "pixel-runtime"
    media_type = "text/html"

    @property
    def monotonic_canvas_selection(self) -> bool:
        return True

    @staticmethod
    def _parallax_strength(converter: "ConversionEngine") -> float:
        return float(
            converter.refinement_config.get(
                "pixel_runtime_parallax_strength",
                converter.refinement_config.get("canvas_parallax_strength", 0.0),
            )
        )

    def requires_governing_render(self, converter: "ConversionEngine") -> bool:
        return self._parallax_strength(converter) <= 0.0

    def emit(
        self,
        *,
        converter: "ConversionEngine",
        request: ConversionRequest,
        scene: SplatScene,
        target_linear_rgb: np.ndarray,
        artifacts_path: Path | None,
    ) -> ArtifactPayload:
        parallax_strength = self._parallax_strength(converter)
        if parallax_strength > 0.0:
            content = generate_parallax_pixel_runtime_html(
                list(scene.splats),
                scene.width,
                scene.height,
                background_linear_rgb=scene.background_linear_rgb,
                title=Path(request.input_path).stem,
                parallax_strength=parallax_strength,
            )
        else:
            content = generate_webgl_pixel_runtime_html(
                list(scene.splats),
                scene.width,
                scene.height,
                background_linear_rgb=scene.background_linear_rgb,
                title=Path(request.input_path).stem,
                compositing_space=scene.compositing_space,
            )
        return ArtifactPayload(
            output_format=self.output_format,
            content=content,
            media_type=self.media_type,
            metadata={"parallax_strength": parallax_strength},
        )

    def evaluate(
        self,
        *,
        converter: "ConversionEngine",
        request: ConversionRequest,
        target_linear_rgb: np.ndarray,
        fallback_linear_rgb: np.ndarray,
        proxy_quality: Mapping[str, Any],
    ) -> ArtifactEvaluation:
        if self.requires_governing_render(converter):
            return self._browser_evaluation(
                request=request,
                proxy_quality=proxy_quality,
                evaluate=lambda **values: evaluate_pixel_runtime_export_quality(
                    html_path=request.output_path, **values
                ),
                success_kind="pixel-runtime-browser-capture",
                unavailable_kind="pixel-runtime-browser-unavailable",
                target_linear_rgb=target_linear_rgb,
                fallback_linear_rgb=fallback_linear_rgb,
            )
        return ArtifactEvaluation(
            evidence_level=EvidenceLevel.PARITY_MODEL,
            render_kind="pixel-runtime-model",
            renderer="internal-splat-renderer",
            metric_source="internal",
            quality=dict(proxy_quality),
            acceptance_eligible=False,
            artifact_path=(
                None if request.output_path is None else Path(request.output_path)
            ),
        )


_BACKENDS: Dict[str, ArtifactBackend] = {
    backend.output_format: backend
    for backend in (
        SvgArtifactBackend(),
        DrawingMLArtifactBackend(),
        PptxArtifactBackend(),
        CanvasArtifactBackend(),
        CssArtifactBackend(),
        PixelRuntimeArtifactBackend(),
    )
}

if frozenset(_BACKENDS) != SUPPORTED_OUTPUT_FORMATS:  # pragma: no cover
    raise RuntimeError("artifact backend registry and supported formats diverged")


def get_artifact_backend(output_format: str) -> ArtifactBackend:
    """Resolve a validated deployment backend."""

    try:
        return _BACKENDS[str(output_format).strip().lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported output format: {output_format}") from exc
