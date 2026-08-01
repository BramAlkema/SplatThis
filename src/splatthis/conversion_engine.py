"""Composition root for the internal PNG-to-splat conversion engine.

The public API lives in :mod:`splatthis.converter`.  Keeping the numerical
engine in this explicitly internal module prevents its implementation details
from becoming the package's architectural entry point.
"""

from typing import Dict, Optional

from .artifact_backends import get_artifact_backend
from .config import ConversionRequest
from .engine_artifacts import ConversionArtifactMixin
from .engine_configuration import ConversionConfigurationMixin
from .engine_densification import ConversionDensificationMixin
from .engine_guidance import ConversionGuidanceMixin
from .engine_initialization import ConversionInitializationMixin
from .engine_optimization import ConversionOptimizationMixin
from .engine_postfit import ConversionPostfitMixin
from .pipeline import ConversionPipeline, RunContext
from .pipeline_artifacts import emit_evaluate_and_finalize
from .pipeline_phases import build_run_manifest, fit_scene, prepare_input


class ConversionEngine(
    ConversionConfigurationMixin,
    ConversionInitializationMixin,
    ConversionOptimizationMixin,
    ConversionDensificationMixin,
    ConversionPostfitMixin,
    ConversionArtifactMixin,
    ConversionGuidanceMixin,
):
    """Composes configuration, fitting and deployment responsibilities."""

    def convert(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        save_json: bool = False,
        verbose: bool = True,
        output_format: str = "svg",
        seed: Optional[int] = None,
        artifacts_dir: Optional[str] = None,
        acceptance_criteria: Optional[Dict[str, float]] = None,
        validate_roundtrip: bool = False,
        side_by_side_html: Optional[str] = None,
        preview_png_path: Optional[str] = None,
    ) -> str:
        """Convert through an isolated per-run pipeline."""
        request = ConversionRequest(
            input_path=input_path,
            output_path=output_path,
            save_json=save_json,
            verbose=verbose,
            output_format=output_format,
            seed=seed,
            artifacts_dir=artifacts_dir,
            acceptance_criteria=acceptance_criteria or {},
            acceptance_overridden=bool(acceptance_criteria),
            validate_roundtrip=validate_roundtrip,
            side_by_side_html=side_by_side_html,
            preview_png_path=preview_png_path,
        )
        return ConversionPipeline(self).run(request)

    def _convert_impl(
        self,
        *,
        request: ConversionRequest,
        context: RunContext,
    ) -> str:
        """Coordinate the explicit prepare, fit, and deployment phases."""
        input_path = request.input_path
        output_format = request.output_format
        backend = get_artifact_backend(output_format)
        if not self._training_export_target_explicit:
            self.training_export_target = backend.default_training_target
        if self.adaptive_compute_config.enabled:
            if not backend.monotonic_canvas_selection:
                raise ValueError(
                    "adaptive compute currently supports only pixel-runtime output"
                )
            if self.training_export_target != "pixel-runtime":
                raise ValueError(
                    "adaptive compute requires training_export_target='pixel-runtime'"
                )
            if not bool(
                self.refinement_config.get(
                    "canvas_monotonic_stage_selection_enabled", True
                )
            ):
                raise ValueError(
                    "adaptive compute requires monotonic Canvas stage selection"
                )

        resolved_target_size = self._resolve_target_size(input_path)

        manifest = build_run_manifest(
            converter=self,
            context=context,
            backend=backend,
            resolved_target_size=resolved_target_size,
        )

        prepared = prepare_input(
            converter=self,
            context=context,
            manifest=manifest,
            resolved_target_size=resolved_target_size,
        )
        scene = fit_scene(
            converter=self,
            context=context,
            manifest=manifest,
            prepared=prepared,
            backend=backend,
        )
        return emit_evaluate_and_finalize(
            converter=self,
            context=context,
            manifest=manifest,
            prepared=prepared,
            scene=scene,
            backend=backend,
        )
