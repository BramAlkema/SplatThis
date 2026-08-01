"""Artifact generation, fidelity selection and acceptance helpers."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from .color import srgb_to_linear
from .engine_state import ConversionEngineState
from .export_common import (
    SVG_BROWSER_COMPAT_RECIPE,
    SVG_SCRIPTED_MATRIX_RECIPE,
    _sort_splats_for_export,
)
from .pptx_export import generate_drawingml_slide_content
from .splat import GaussianSplat
from .storage import atomic_write_text, save_splats_json

logger = logging.getLogger(__name__)


class ConversionArtifactMixin(ConversionEngineState):
    """Owns legacy artifact strategies used by the deployment pipeline."""

    def _generate_svg(
        self,
        splats: List[GaussianSplat],
        width: int,
        height: int,
        *,
        gradient_quality: Optional[str] = None,
        painter_order: Optional[str] = None,
    ) -> str:
        """Generate SVG content."""
        from .svg_export import generate_svg_content

        palette_size = self.refinement_config.get("svg_palette_size")
        ordered_splats = _sort_splats_for_export(splats)
        return generate_svg_content(
            ordered_splats,
            width,
            height,
            self.k_sigma,
            background_linear_rgb=self._background_linear_rgb,
            export_recipe=self.svg_export_recipe,
            foreground_mask=self._region_foreground_mask,
            background_safe_mask=self._region_background_safe_mask,
            edge_band_mask=self._region_edge_band_mask,
            palette_size=None if palette_size is None else int(palette_size),
            gradient_quality=(
                self.svg_gradient_quality
                if gradient_quality is None
                else gradient_quality
            ),
            painter_order=(
                self.svg_painter_order if painter_order is None else painter_order
            ),
        )

    def _select_svg_compositor(
        self,
        *,
        splats: List[GaussianSplat],
        image: np.ndarray,
        width: int,
        height: int,
        artifacts_path: Optional[Path],
    ) -> Tuple[str, Dict[str, Any]]:
        """Accept or revert SVG order/stop candidates using real Chromium.

        The historical forward DOM order is the incumbent for monotonicity,
        not the semantic ideal. Correct back-to-front standard and high-stop
        candidates may replace it only when the full guarded metric vector,
        compressed size, and browser latency pass the artifact gate.
        """

        import gzip
        import tempfile

        from .browser_capture import get_shared_svg_renderer
        from .fidelity.analysis import analyze_residual
        from .fidelity.metrics import compute_fidelity_metrics
        from .storage import atomic_write_text
        from .svg_recipe_gate import SvgRecipeGatePolicy, select_recipe_candidate

        candidate_specs = [
            ("legacy-standard", "standard", "legacy"),
            ("corrected-standard", "standard", "back-to-front"),
        ]
        if self.svg_export_recipe in {
            "standard",
            SVG_BROWSER_COMPAT_RECIPE,
            SVG_SCRIPTED_MATRIX_RECIPE,
        }:
            candidate_specs.append(("corrected-high", "high", "back-to-front"))

        contents = {
            name: self._generate_svg(
                splats,
                width,
                height,
                gradient_quality=quality,
                painter_order=order,
            )
            for name, quality, order in candidate_specs
        }
        requested_content = self._generate_svg(splats, width, height)
        temporary: Optional[tempfile.TemporaryDirectory[str]] = None
        if artifacts_path is None:
            temporary = tempfile.TemporaryDirectory(prefix="splatthis-svg-gate-")
            candidate_dir = Path(temporary.name)
        else:
            candidate_dir = artifacts_path / "svg-compositor-candidates"
            candidate_dir.mkdir(parents=True, exist_ok=True)

        try:
            renderer = get_shared_svg_renderer()
            repeats = int(
                max(1, self.refinement_config.get("svg_compositor_gate_repeats", 1))
            )
            rendered_by_name: Dict[str, np.ndarray] = {}
            captures: Dict[str, Any] = {}
            for name, _, _ in candidate_specs:
                svg_path = candidate_dir / f"{name}.svg"
                png_path = candidate_dir / f"{name}.png"
                atomic_write_text(svg_path, contents[name])
                capture = renderer.capture(
                    svg_path,
                    png_path,
                    width=width,
                    height=height,
                    repeats=repeats,
                )
                with Image.open(png_path) as raster:
                    srgb = np.asarray(raster.convert("RGB"), dtype=np.float32) / 255.0
                rendered_by_name[name] = srgb_to_linear(srgb)
                captures[name] = capture

            baseline_name = candidate_specs[0][0]
            fixed_rois = analyze_residual(
                image[:, :, :3],
                rendered_by_name[baseline_name],
                roi_size=min(64, height, width),
                roi_count=8,
            ).fixed_rois
            measurements = []
            for name, quality, order in candidate_specs:
                content = contents[name]
                metrics = compute_fidelity_metrics(
                    image[:, :, :3],
                    rendered_by_name[name],
                    fixed_rois=fixed_rois,
                    splat_count=len(splats),
                    file_size_bytes=len(
                        gzip.compress(content.encode("utf-8"), compresslevel=9)
                    ),
                    render_method=renderer.renderer_label,
                ).as_dict()
                measurements.append(
                    {
                        "recipe": name,
                        "painter_order": order,
                        "gradient_quality": quality,
                        "raw_size_bytes": len(content.encode("utf-8")),
                        "render_time_sec": captures[name].capture_time_ms / 1000.0,
                        "pixel_stable": captures[name].pixel_stable,
                        **{
                            key: (
                                None
                                if isinstance(value, float) and not np.isfinite(value)
                                else value
                            )
                            for key, value in metrics.items()
                        },
                    }
                )

            policy = SvgRecipeGatePolicy(
                max_size_growth_fraction=1.0,
                max_render_time_growth_fraction=0.50,
                max_ssim_regression=0.002,
                max_ms_ssim_regression=0.003,
                max_lpips_regression=0.005,
                maximum_median_size_growth_fraction=1.0,
                maximum_median_render_time_growth_fraction=0.25,
            )
            selection = select_recipe_candidate(
                measurements[0], measurements[1:], policy
            )
            selected_name = str(selection["selected_recipe"])
            return contents[selected_name], {
                "enabled": True,
                "available": True,
                "renderer": renderer.renderer_label,
                "policy": policy.as_dict(),
                "fixed_rois": [list(roi) for roi in fixed_rois],
                "candidates": measurements,
                "selection": selection,
            }
        except Exception as exc:
            logger.warning(
                "SVG compositor gate unavailable; using requested corrected "
                "export (%s: %s)",
                type(exc).__name__,
                exc,
            )
            return requested_content, {
                "enabled": True,
                "available": False,
                "reason": f"{type(exc).__name__}: {exc}",
                "fallback_painter_order": self.svg_painter_order,
                "fallback_gradient_quality": self.svg_gradient_quality,
            }
        finally:
            if temporary is not None:
                temporary.cleanup()

    def _run_fidelity_stage(
        self,
        *,
        splats: List[GaussianSplat],
        image: np.ndarray,
        width: int,
        height: int,
        artifacts_path: Optional[Path],
        seed: Optional[int],
        verbose: bool,
    ) -> Tuple[List[GaussianSplat], Dict[str, Any]]:
        """ADR-003 stage: candidates emitted through the exact export path.

        The evaluator emits candidates via _generate_svg, so recipe and
        region-mask semantics are identical to the final export. State
        isolation is inherited from the convert() snapshot wrapper.
        """
        import tempfile

        from .fidelity import (
            FidelityCandidate,
            FidelityEvaluator,
            FidelityStage,
            build_operators,
            write_fidelity_report,
        )

        if artifacts_path is not None:
            work_dir = str(Path(artifacts_path) / "fidelity")
            keep_artifacts = True
            cleanup_dir = None
        else:
            cleanup_dir = tempfile.TemporaryDirectory(prefix="fidelity-")
            work_dir = cleanup_dir.name
            keep_artifacts = False

        try:
            saliency_mask = None
            if self._region_foreground_mask is not None:
                saliency_mask = np.asarray(self._region_foreground_mask, dtype=bool)
            evaluator = FidelityEvaluator(
                target_linear_rgb=image[:, :, :3],
                background_linear_rgb=self._background_linear_rgb,
                compositing_space=self._deployed_compositing_space(),
                emit_svg=lambda s: self._generate_svg(s, width, height),
                work_dir=work_dir,
                config=self.fidelity_config,
                saliency_mask=saliency_mask,
                keep_candidate_artifacts=keep_artifacts,
            )
            stage = FidelityStage(
                config=self.fidelity_config,
                evaluator=evaluator,
                operators=build_operators(self.fidelity_config),
            )
            baseline = FidelityCandidate(name="baseline", splats=tuple(splats))
            result = stage.run(baseline)
            fragment = write_fidelity_report(
                work_dir, result, self.fidelity_config, seed
            )
            return list(result.winner.splats), fragment
        finally:
            if cleanup_dir is not None:
                cleanup_dir.cleanup()

    def _generate_drawingml(
        self, splats: List[GaussianSplat], width: int, height: int
    ) -> str:
        """Generate DrawingML slide XML content."""
        ordered_splats = _sort_splats_for_export(splats)
        return generate_drawingml_slide_content(
            ordered_splats,
            width,
            height,
            self.k_sigma,
            background_linear_rgb=self._background_linear_rgb,
            splat_style=self.pptx_splat_style,
            painter_order=self.pptx_painter_order,
        )

    def _write_stage_artifact(
        self,
        artifacts_dir: Optional[Path],
        stage_name: str,
        splats: List[GaussianSplat],
        metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Write per-stage debug artifacts."""
        if artifacts_dir is None:
            return
        raw_path = artifacts_dir / f"{stage_name}.raw.json"
        save_splats_json(splats, str(raw_path))

        if metrics is not None:
            metrics_path = artifacts_dir / f"{stage_name}.metrics.json"
            atomic_write_text(
                metrics_path,
                json.dumps(metrics, indent=2, sort_keys=True),
            )

    def _write_manifest(
        self, artifacts_dir: Optional[Path], manifest: Dict[str, Any]
    ) -> None:
        """Write run manifest if artifact directory is configured."""
        if artifacts_dir is None:
            return
        manifest_path = artifacts_dir / "run_manifest.json"
        atomic_write_text(
            manifest_path,
            json.dumps(manifest, indent=2, sort_keys=True),
        )

    def _evaluate_acceptance(
        self, metrics: Dict[str, float], criteria: Dict[str, float]
    ) -> Dict[str, Any]:
        """Evaluate pass/fail against acceptance criteria."""
        checks: Dict[str, bool] = {}

        if "min_psnr" in criteria:
            checks["psnr"] = float(metrics.get("psnr", 0.0)) >= float(
                criteria["min_psnr"]
            )
        if "min_ssim" in criteria:
            checks["ssim"] = float(metrics.get("ssim", 0.0)) >= float(
                criteria["min_ssim"]
            )
        # Perceptual (sRGB-display) gates: what the eye actually sees.
        if "min_psnr_srgb" in criteria:
            checks["psnr_srgb"] = float(metrics.get("psnr_srgb", 0.0)) >= float(
                criteria["min_psnr_srgb"]
            )
        if "min_ssim_srgb" in criteria:
            checks["ssim_srgb"] = float(metrics.get("ssim_srgb", 0.0)) >= float(
                criteria["min_ssim_srgb"]
            )
        if "max_runtime_sec" in criteria:
            checks["runtime_sec"] = float(metrics.get("runtime_sec", 0.0)) <= float(
                criteria["max_runtime_sec"]
            )
        if "max_splats" in criteria:
            checks["splat_count"] = float(metrics.get("splat_count", 0.0)) <= float(
                criteria["max_splats"]
            )

        return {
            "pass": bool(all(checks.values())) if checks else True,
            "checks": checks,
            "thresholds": criteria,
            "measured": {
                "psnr": float(metrics.get("psnr", 0.0)),
                "ssim": float(metrics.get("ssim", 0.0)),
                "psnr_srgb": float(metrics.get("psnr_srgb", 0.0)),
                "ssim_srgb": float(metrics.get("ssim_srgb", 0.0)),
                "runtime_sec": float(metrics.get("runtime_sec", 0.0)),
                "splat_count": float(metrics.get("splat_count", 0.0)),
            },
        }
