"""
Main PNG→SVG converter class.

Orchestrates the full pipeline from PNG input to SVG/DrawingML output.
"""

import hashlib
import json
import logging
import platform
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import psutil
import torch
from PIL import Image

from .features import (
    analyze_local_structure,
    compute_structure_field,
    compute_gradient_magnitude,
    estimate_local_color,
    init_seeds_content_adaptive,
    poisson_disk_sampling,
)
from .io import (
    compute_quality_metrics,
    evaluate_svg_export_quality,
    generate_drawingml_slide_content,
    load_png,
    render_splats_preview_png,
    save_pptx_with_splat_png,
    save_side_by_side_html,
    save_drawingml,
    save_splats_json,
    save_svg,
    validate_export_roundtrip,
)
from .optimizer import SplatOptimizer
from .renderer import (
    L1SSIMLoss,
    create_renderer,
    render_splats_numpy,
    resolve_renderer_backend,
    splats_to_tensor,
    tensor_to_splats,
)
from .splat import GaussianSplat, create_anisotropic_splat, create_isotropic_splat

logger = logging.getLogger(__name__)


class PNG2SVGConverter:
    """Main converter class for PNG→SVG Gaussian splatting pipeline."""

    def __init__(
        self,
        max_splats: int = 1000,
        k_sigma: float = 2.5,
        stages: Optional[List[int]] = None,
        target_size: Optional[Tuple[int, int]] = None,
        gradient_method: str = "sobel",
        device: str = "cpu",
        seed: Optional[int] = None,
        quality_profile: str = "balanced",
        resolution_scale: float = 1.0,
        loss_weights: Optional[Dict[str, float]] = None,
        learning_rates: Optional[Dict[str, float]] = None,
        refinement_config: Optional[Dict[str, float]] = None,
        schedule_config: Optional[Dict[str, float]] = None,
        acceptance_criteria: Optional[Dict[str, float]] = None,
        init_random_ratio: float = 0.2,
        init_gradient_weight: float = 0.7,
        renderer_backend: str = "auto",
        blend_mode: str = "weighted",
        compositing_space: str = "linear",
        loss_color_space: str = "oklab",
    ):
        self.max_splats = max_splats
        self.k_sigma = k_sigma
        self.stages = stages or [200, 150, 100, 50]
        self.target_size = target_size
        self.gradient_method = gradient_method
        self.device = torch.device(device)
        self.renderer_backend = renderer_backend
        self.resolved_renderer_backend = resolve_renderer_backend(
            renderer_backend,
            self.device,
        )
        self.blend_mode = str(blend_mode).strip().lower()
        # Compositing space for the optimizer's forward render. "linear" is
        # physically correct and fits cleanly; "srgb" matches how browsers blend
        # overlapping SVG shapes. Empirically these reach the same final SVG
        # quality (sRGB matches the browser but optimizes worse), so default to
        # "linear". The flag is retained for renderers that need exact match.
        self.compositing_space = str(compositing_space).strip().lower()
        # Color space for the reconstruction loss. "oklab" optimizes perceptual
        # (lightness/chroma) error instead of linear-RGB MSE.
        self.loss_color_space = str(loss_color_space).strip().lower()
        self.seed = seed
        self.quality_profile = quality_profile
        self.resolution_scale = float(max(1.0, resolution_scale))
        self.init_random_ratio = float(np.clip(init_random_ratio, 0.0, 1.0))
        self.init_gradient_weight = float(np.clip(init_gradient_weight, 0.0, 1.0))

        profile_defaults = self._get_profile_defaults(quality_profile)

        # Phase 1 baseline: L1 + SSIM.
        self.loss_weights = loss_weights or profile_defaults["loss_weights"].copy()

        # Parameter-group learning rates.
        self.learning_rates = profile_defaults["learning_rates"].copy()
        if learning_rates:
            self.learning_rates.update(learning_rates)

        self.refinement_config = profile_defaults["refinement"].copy()
        if refinement_config:
            self.refinement_config.update(refinement_config)
        self.schedule_config = profile_defaults["schedule"].copy()
        if schedule_config:
            self.schedule_config.update(schedule_config)
        # Acceptance floors are meaningful "not-garbage" gates, not the old
        # vacuous 0.02 SSIM. Perceptual (sRGB-display) gates reflect what the eye
        # sees; linear gates reflect the optimizer's own space.
        self.acceptance_criteria = acceptance_criteria or {
            "min_psnr": 15.0,
            "min_ssim": 0.50,
            "min_psnr_srgb": 12.0,
            "min_ssim_srgb": 0.50,
            "max_runtime_sec": 60.0,
            "max_splats": float(self.max_splats),
        }

        self._image_width = 1000
        self._image_height = 1000
        self._background_linear_rgb = np.zeros(3, dtype=np.float32)

        if "arm" in platform.processor().lower():
            self.max_splats = min(self.max_splats, 2000)
            logger.info("Apple Silicon detected - limiting max_splats to %s", self.max_splats)

        logger.info(
            "Initialized PNG2SVG converter: max_splats=%s, stages=%s, device=%s, backend=%s->%s, blend=%s, seed=%s, profile=%s, resolution_scale=%.2f, init_random_ratio=%.2f",
            self.max_splats,
            self.stages,
            device,
            self.renderer_backend,
            self.resolved_renderer_backend,
            self.blend_mode,
            self.seed,
            self.quality_profile,
            self.resolution_scale,
            self.init_random_ratio,
        )

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
        """
        Convert PNG to SVG or DrawingML.

        Args:
            input_path: Path to input PNG.
            output_path: Path for output file (optional).
            save_json: Whether to save splats as canonical raw JSON.
            verbose: Whether to log progress.
            output_format: Output format ("svg", "drawingml", or "pptx").
            seed: Optional run seed overriding converter seed.
            artifacts_dir: Optional directory for stage artifacts + run manifest.
            acceptance_criteria: Optional run-level acceptance thresholds override.
            validate_roundtrip: Whether to run raw->export round-trip validation.
            side_by_side_html: Optional output HTML path for side-by-side comparison page.
            preview_png_path: Optional output PNG path for preview render used in reports.

        Returns:
            Generated vector content as string.
        """
        if output_format not in {"svg", "drawingml", "pptx"}:
            raise ValueError(f"Unsupported output format: {output_format}")

        run_seed = self.seed if seed is None else seed
        rng = np.random.default_rng(run_seed) if run_seed is not None else np.random.default_rng()
        if run_seed is not None:
            torch.manual_seed(int(run_seed))

        start_time = time.time()
        artifacts_path: Optional[Path] = None
        if artifacts_dir:
            artifacts_path = Path(artifacts_dir)
            artifacts_path.mkdir(parents=True, exist_ok=True)
        resolved_target_size = self._resolve_target_size(input_path)

        manifest: Dict[str, Any] = {
            "input_path": str(input_path),
            "input_sha256": self._sha256_file(input_path),
            "seed": run_seed,
            "config": {
                "max_splats": self.max_splats,
                "k_sigma": self.k_sigma,
                "stages": list(self.stages),
                "target_size": self.target_size,
                "resolved_target_size": resolved_target_size,
                "resolution_scale": self.resolution_scale,
                "gradient_method": self.gradient_method,
                "init_random_ratio": self.init_random_ratio,
                "init_gradient_weight": self.init_gradient_weight,
                "device": str(self.device),
                "renderer_backend": self.renderer_backend,
                "resolved_renderer_backend": self.resolved_renderer_backend,
                "blend_mode": self.blend_mode,
                "output_format": output_format,
                "quality_profile": self.quality_profile,
                "loss_weights": self.loss_weights,
                "learning_rates": self.learning_rates,
                "refinement_config": self.refinement_config,
                "schedule_config": self.schedule_config,
            },
            "stages": [],
        }

        if verbose:
            logger.info(
                "Loading PNG: %s (target_size=%s, resolution_scale=%.2f)",
                input_path,
                resolved_target_size,
                self.resolution_scale,
            )
        image = load_png(input_path, target_size=resolved_target_size)
        height, width = image.shape[:2]
        self._image_width = width
        self._image_height = height
        self._background_linear_rgb = self._estimate_background_color(image)
        structure_enabled = bool(self.refinement_config.get("structure_precompute_enabled", False))
        structure_smoothing_sigma = float(max(0.0, self.refinement_config.get("structure_smoothing_sigma", 0.0)))
        structure_anisotropy_clip = float(max(1.0, self.refinement_config.get("structure_anisotropy_clip", 10.0)))
        structure_min_coherence = float(
            np.clip(self.refinement_config.get("structure_min_coherence", 0.12), 0.0, 1.0)
        )
        structure_primary: Optional[np.ndarray] = None
        structure_anisotropy: Optional[np.ndarray] = None
        if structure_enabled:
            structure_primary, structure_anisotropy = compute_structure_field(
                image=image,
                method=self.gradient_method,
                smoothing_sigma=structure_smoothing_sigma,
                anisotropy_clip=structure_anisotropy_clip,
                min_coherence=structure_min_coherence,
            )
            if verbose:
                logger.info("Using precomputed structure maps for init/densify guidance.")
        manifest["config"]["structure_smoothing_sigma"] = structure_smoothing_sigma
        manifest["config"]["structure_anisotropy_clip"] = structure_anisotropy_clip
        manifest["config"]["structure_min_coherence"] = structure_min_coherence
        manifest["config"]["structure_precompute_enabled"] = structure_enabled
        manifest["config"]["background_linear_rgb"] = [
            float(self._background_linear_rgb[0]),
            float(self._background_linear_rgb[1]),
            float(self._background_linear_rgb[2]),
        ]

        if verbose:
            logger.info("Initializing splats...")
        splats = self._initialize_splats(
            image,
            rng=rng,
            structure_primary=structure_primary,
            structure_anisotropy=structure_anisotropy,
        )
        self._write_stage_artifact(artifacts_path, "init", splats, {"count": len(splats)})

        if verbose:
            logger.info("Starting optimization with %s initial splats...", len(splats))
        splats, stage_metrics = self._optimize_splats(
            image=image,
            splats=splats,
            rng=rng,
            verbose=verbose,
            artifacts_dir=artifacts_path,
            structure_primary=structure_primary,
            structure_anisotropy=structure_anisotropy,
        )
        manifest["stages"].extend(stage_metrics)

        if verbose:
            logger.info("Post-processing splats...")
        splats = self._postprocess_splats(splats=splats, image=image, rng=rng)
        self._write_stage_artifact(artifacts_path, "final", splats, {"count": len(splats)})

        if output_format == "drawingml":
            if verbose:
                logger.info("Generating DrawingML with %s splats...", len(splats))
            output_content = self._generate_drawingml(splats, width, height)
        elif output_format == "pptx":
            if verbose:
                logger.info("Preparing PPTX package with rendered splat slide...")
            output_content = ""
        else:
            if verbose:
                logger.info("Generating SVG with %s splats...", len(splats))
            output_content = self._generate_svg(splats, width, height)

        if output_path:
            if output_format == "drawingml":
                save_drawingml(splats, width, height, output_path, k_sigma=self.k_sigma)
                if verbose:
                    logger.info("Saved DrawingML: %s", output_path)
            elif output_format == "pptx":
                save_pptx_with_splat_png(
                    splats=splats,
                    width=width,
                    height=height,
                    output_path=output_path,
                    background_linear_rgb=self._background_linear_rgb,
                )
                if verbose:
                    logger.info("Saved PPTX: %s", output_path)
            else:
                save_svg(
                    splats,
                    width,
                    height,
                    output_path,
                    k_sigma=self.k_sigma,
                    background_linear_rgb=self._background_linear_rgb,
                )
                if verbose:
                    logger.info("Saved SVG: %s", output_path)

            if save_json:
                json_path = str(Path(output_path).with_suffix(".json"))
                save_splats_json(splats, json_path)
                if verbose:
                    logger.info("Saved JSON: %s", json_path)

        total_time = time.time() - start_time
        target = torch.from_numpy(image[:, :, :3]).to(self.device)
        final_renderer = create_renderer(
            backend=self.renderer_backend,
            width=width,
            height=height,
            device=self.device,
            blend_mode=self.blend_mode,
            compositing_space=self.compositing_space,
        )
        final_loss_fn = L1SSIMLoss(**self.loss_weights, color_space=self.loss_color_space).to(self.device)
        internal_metrics = self._compute_quality_metrics(splats, target, final_renderer, final_loss_fn)
        internal_metrics["runtime_sec"] = float(total_time)
        internal_metrics["splat_count"] = float(len(splats))

        # Export-proxy metrics always available from CPU preview render.
        preview_linear = render_splats_numpy(splats, width, height)
        export_quality: Dict[str, Any] = {
            "available": True,
            "method": "proxy-render",
            "used_fallback": True,
            "metrics": compute_quality_metrics(image[:, :, :3], preview_linear),
        }

        if output_format == "svg" and output_path:
            svg_quality = evaluate_svg_export_quality(
                target_linear_rgb=image[:, :, :3],
                svg_path=output_path,
                fallback_linear_rgb=preview_linear,
            )
            if svg_quality.get("available"):
                export_quality = svg_quality

        export_method = str(export_quality.get("method", ""))
        use_export_for_acceptance = bool(
            export_quality.get("available")
            and not export_method.startswith("proxy")
            and (export_quality.get("metrics") is not None)
        )
        acceptance_source_metrics = (
            dict(export_quality.get("metrics") or {})
            if use_export_for_acceptance
            else dict(internal_metrics)
        )
        final_metrics = acceptance_source_metrics
        final_metrics["runtime_sec"] = float(total_time)
        final_metrics["splat_count"] = float(len(splats))
        # Preserve internal-only diagnostics (e.g. coverage) regardless of which
        # render the acceptance metrics came from.
        if "coverage" not in final_metrics and "coverage" in internal_metrics:
            final_metrics["coverage"] = internal_metrics["coverage"]
        # Explicit acceptance_criteria fully replace the defaults: a caller that
        # specifies criteria specifies the whole gate (so partial criteria don't
        # silently inherit the default perceptual gates).
        effective_acceptance = (
            dict(acceptance_criteria) if acceptance_criteria else self.acceptance_criteria.copy()
        )
        acceptance_result = self._evaluate_acceptance(final_metrics, effective_acceptance)
        roundtrip_result: Optional[Dict[str, Any]] = None
        if validate_roundtrip:
            roundtrip_result = validate_export_roundtrip(
                splats=splats,
                width=width,
                height=height,
                k_sigma=self.k_sigma,
            )

        # Optional preview and side-by-side artifacts.
        preview_path = preview_png_path
        if preview_path is None and output_path:
            preview_path = str(Path(output_path).with_name(f"{Path(output_path).stem}_preview.png"))
        if preview_path:
            render_splats_preview_png(
                splats=splats,
                width=width,
                height=height,
                output_path=preview_path,
            )
        if side_by_side_html:
            side_metrics = {
                "output_format": output_format,
                "internal_psnr": internal_metrics.get("psnr"),
                "internal_ssim": internal_metrics.get("ssim"),
                "export_method": export_quality.get("method"),
                "export_psnr": (export_quality.get("metrics") or {}).get("psnr"),
                "export_ssim": (export_quality.get("metrics") or {}).get("ssim"),
                "runtime_sec": total_time,
                "splats": len(splats),
            }
            save_side_by_side_html(
                output_path=side_by_side_html,
                source_png_path=input_path,
                svg_path=output_path if output_format == "svg" and output_path else "",
                preview_png_path=preview_path,
                title="PNG2Splat Side-by-Side",
                metrics=side_metrics,
            )

        manifest["total_time_sec"] = total_time
        manifest["final_splat_count"] = len(splats)
        manifest["final_metrics"] = final_metrics
        manifest["internal_metrics"] = internal_metrics
        manifest["export_quality"] = export_quality
        manifest["acceptance_metric_source"] = "export" if use_export_for_acceptance else "internal"
        manifest["acceptance"] = acceptance_result
        if roundtrip_result is not None:
            manifest["roundtrip_validation"] = roundtrip_result

        self._write_manifest(artifacts_path, manifest)

        if verbose:
            logger.info("Conversion completed in %.2fs", total_time)

        return output_content

    def _initialize_splats(
        self,
        image: np.ndarray,
        rng: np.random.Generator,
        structure_primary: Optional[np.ndarray] = None,
        structure_anisotropy: Optional[np.ndarray] = None,
    ) -> List[GaussianSplat]:
        """
        Initialize splats with a guaranteed-coverage base layer plus detail layer.

        The base layer is stratified over the full canvas to avoid early empty regions.
        The detail layer is content-adaptive and edge-biased.
        """
        height, width = image.shape[:2]
        initial_count = min(self.max_splats // 2, 1200)
        if initial_count <= 0:
            return []

        base_fraction = float(np.clip(self.refinement_config.get("base_layer_fraction", 0.35), 0.10, 0.80))
        base_count = max(4, int(round(initial_count * base_fraction)))
        detail_count = max(1, initial_count - base_count)

        base_positions = self._make_stratified_positions(
            width=width,
            height=height,
            count=base_count,
            rng=rng,
            jitter_ratio=0.65,
        )

        adaptive_count = max(1, int(round(detail_count * (1.0 - self.init_random_ratio))))
        random_count = max(0, detail_count - adaptive_count)

        seed_positions = init_seeds_content_adaptive(
            image=image,
            target_count=adaptive_count,
            gradient_weight=self.init_gradient_weight,
            method=self.gradient_method,
            rng=rng,
        )
        random_positions: List[Tuple[float, float]] = []
        if random_count > 0:
            random_x = rng.uniform(0.0, float(width), size=random_count)
            random_y = rng.uniform(0.0, float(height), size=random_count)
            random_positions = [(float(x), float(y)) for x, y in zip(random_x, random_y)]

        poisson_count = max(1, detail_count // 5)
        min_distance = max(2.0, min(width, height) / max(np.sqrt(max(detail_count, 1.0)), 1.0))
        poisson_positions = poisson_disk_sampling(
            width=width,
            height=height,
            min_distance=min_distance,
            rng=rng,
        )[:poisson_count]

        all_positions = base_positions + seed_positions + random_positions + poisson_positions
        splats: List[GaussianSplat] = []

        base_sigma = float(
            np.clip(
                np.sqrt((float(width) * float(height)) / max(base_count, 1)) * 0.85,
                self.refinement_config.get("sigma_min", 1.0),
                self.refinement_config.get("coverage_sigma_max", 8.0),
            )
        )
        base_alpha = float(np.clip(self.refinement_config.get("base_layer_alpha", 0.42), 0.08, 0.95))
        sigma_minor_min = float(self.refinement_config.get("sigma_minor_min", 0.35))

        for idx, (x, y) in enumerate(all_positions):
            x_int = int(np.clip(x, 0, width - 1))
            y_int = int(np.clip(y, 0, height - 1))

            if (
                structure_primary is not None
                and structure_anisotropy is not None
                and structure_primary.shape[:2] == (height, width)
                and structure_primary.shape[-1] == 2
                and structure_anisotropy.shape == (height, width)
            ):
                primary_direction = structure_primary[y_int, x_int]
                anisotropy = float(structure_anisotropy[y_int, x_int])
            else:
                primary_direction, anisotropy = self._analyze_local_structure(image, x_int, y_int)
            color = estimate_local_color(image, x_int, y_int)

            is_base = idx < len(base_positions)
            if is_base:
                sigma = base_sigma
                alpha = base_alpha
            else:
                sigma = float(np.clip(base_sigma * 0.65, self.refinement_config.get("sigma_min", 1.0), 6.0))
                alpha = float(np.clip(base_alpha + 0.18, 0.15, 0.95))

            init_anisotropy_threshold = float(
                max(1.0, self.refinement_config.get("init_anisotropy_threshold", 1.55))
            )
            if (not is_base) and anisotropy >= init_anisotropy_threshold:
                angle = float(np.arctan2(primary_direction[1], primary_direction[0]))
                cos_a, sin_a = float(np.cos(angle)), float(np.sin(angle))
                rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
                sigma_major = sigma
                sigma_minor = max(
                    sigma_major / min(float(anisotropy), float(self.refinement_config.get("local_structure_anisotropy_clip", 4.0))),
                    sigma_minor_min,
                )
                splat = create_anisotropic_splat(
                    center=np.array([x, y], dtype=np.float32),
                    eigenvals=np.array([sigma_major**2, sigma_minor**2], dtype=np.float32),
                    eigenvecs=rotation_matrix,
                    color=color,
                    alpha=alpha,
                )
            else:
                splat = create_isotropic_splat(
                    center=np.array([x, y], dtype=np.float32),
                    sigma=sigma,
                    color=color,
                    alpha=alpha,
                )

            # Base layer should sit behind detail splats.
            splat.importance = 0.1 if is_base else 0.35
            splats.append(splat)

        logger.info(
            "Initialized %s splats (%s base + %s detail)",
            len(splats),
            len(base_positions),
            len(splats) - len(base_positions),
        )
        return splats

    def _make_stratified_positions(
        self,
        width: int,
        height: int,
        count: int,
        rng: np.random.Generator,
        jitter_ratio: float = 0.5,
    ) -> List[Tuple[float, float]]:
        """Generate approximately uniform stratified points over image space."""
        if count <= 0:
            return []

        aspect = float(width) / max(float(height), 1.0)
        cols = max(1, int(np.ceil(np.sqrt(float(count) * aspect))))
        rows = max(1, int(np.ceil(float(count) / float(cols))))
        cell_w = float(width) / float(cols)
        cell_h = float(height) / float(rows)
        jitter = float(np.clip(jitter_ratio, 0.0, 1.0))

        positions: List[Tuple[float, float]] = []
        for row in range(rows):
            for col in range(cols):
                if len(positions) >= count:
                    break
                cx = (float(col) + 0.5) * cell_w
                cy = (float(row) + 0.5) * cell_h
                jx = (rng.random() - 0.5) * jitter * cell_w
                jy = (rng.random() - 0.5) * jitter * cell_h
                x = float(np.clip(cx + jx, 0.0, max(float(width) - 1.0, 0.0)))
                y = float(np.clip(cy + jy, 0.0, max(float(height) - 1.0, 0.0)))
                positions.append((x, y))
            if len(positions) >= count:
                break
        return positions

    def _build_edge_map(self, image: np.ndarray) -> np.ndarray:
        """Build normalized edge-energy map used by densification."""
        grad_mag = compute_gradient_magnitude(image, method=self.gradient_method)
        return self._normalize_map(grad_mag)

    def _analyze_local_structure(self, image: np.ndarray, x: int, y: int) -> Tuple[np.ndarray, float]:
        """Analyze local orientation with conservative anisotropy gating."""
        return analyze_local_structure(
            image=image,
            x=x,
            y=y,
            window_size=int(max(3, self.refinement_config.get("structure_local_window", 7))),
            anisotropy_clip=float(max(1.0, self.refinement_config.get("local_structure_anisotropy_clip", 4.0))),
            min_coherence=float(np.clip(self.refinement_config.get("local_structure_min_coherence", 0.12), 0.0, 1.0)),
            min_energy=float(max(0.0, self.refinement_config.get("local_structure_min_energy", 1e-4))),
        )

    def _optimize_splats(
        self,
        image: np.ndarray,
        splats: List[GaussianSplat],
        rng: np.random.Generator,
        verbose: bool = True,
        artifacts_dir: Optional[Path] = None,
        structure_primary: Optional[np.ndarray] = None,
        structure_anisotropy: Optional[np.ndarray] = None,
    ) -> Tuple[List[GaussianSplat], List[Dict[str, Any]]]:
        """Progressive optimization of splats."""
        height, width = image.shape[:2]
        target = torch.from_numpy(image[:, :, :3]).to(self.device)
        edge_map = self._build_edge_map(image)

        memory_before = psutil.virtual_memory().percent
        if memory_before > 85:
            logger.warning(
                "High memory usage detected: %.1f%% - reducing splat count", memory_before
            )
            self.max_splats = min(self.max_splats, max(1, len(splats) // 2))

        renderer = create_renderer(
            backend=self.renderer_backend,
            width=width,
            height=height,
            device=self.device,
            blend_mode=self.blend_mode,
            compositing_space=self.compositing_space,
        )
        loss_fn = L1SSIMLoss(**self.loss_weights, color_space=self.loss_color_space).to(self.device)
        optimizer_helper = SplatOptimizer(learning_rates=self.learning_rates)

        current_splats = splats.copy()
        stage_metrics: List[Dict[str, Any]] = []
        residual_detail_enabled = bool(self.refinement_config.get("residual_detail_enabled", False))
        residual_reserve_fraction = float(
            np.clip(self.refinement_config.get("residual_detail_reserve_fraction", 0.0), 0.0, 0.40)
        )
        reserved_slots = int(round(float(self.max_splats) * residual_reserve_fraction)) if residual_detail_enabled else 0
        main_budget = max(1, self.max_splats - max(0, reserved_slots))

        for stage_idx, num_iters in enumerate(self.stages):
            if verbose:
                logger.info("Stage %s/%s: %s iterations", stage_idx + 1, len(self.stages), num_iters)

            current_splats, stage_metric, stage_rendered = self._optimize_stage(
                splats=current_splats,
                target=target,
                renderer=renderer,
                loss_fn=loss_fn,
                optimizer_helper=optimizer_helper,
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
            stage_metrics.append(stage_metric)
            self._write_stage_artifact(
                artifacts_dir,
                f"iter-{stage_idx + 1}",
                current_splats,
                stage_metric,
            )

            coverage_after_densify: Optional[np.ndarray] = None
            if stage_idx < len(self.stages) - 1:
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

            if len(current_splats) > main_budget:
                current_splats = self._prune_splats(
                    current_splats,
                    main_budget,
                    target=target,
                    renderer=renderer,
                    precomputed_coverage_map=coverage_after_densify,
                )

        current_splats, residual_metrics = self._run_residual_detail_passes(
            splats=current_splats,
            image=image,
            target=target,
            renderer=renderer,
            loss_fn=loss_fn,
            optimizer_helper=optimizer_helper,
            rng=rng,
            edge_map=edge_map,
            verbose=verbose,
        )
        for metric in residual_metrics:
            stage_metrics.append(metric)
            pass_idx = int(metric.get("residual_pass", len(stage_metrics)))
            self._write_stage_artifact(
                artifacts_dir,
                f"residual-{pass_idx}",
                current_splats,
                metric,
            )

        return current_splats, stage_metrics

    def _optimize_stage(
        self,
        splats: List[GaussianSplat],
        target: torch.Tensor,
        renderer: torch.nn.Module,
        loss_fn: L1SSIMLoss,
        optimizer_helper: SplatOptimizer,
        num_iters: int,
        verbose: bool,
    ) -> Tuple[List[GaussianSplat], Dict[str, Any], torch.Tensor]:
        """Optimize splats for one stage."""
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

        splats_tensor = splats_to_tensor(splats, device=self.device)
        splats_tensor.requires_grad_(True)
        optimizer = optimizer_helper.create_optimizer(splats_tensor)

        with torch.no_grad():
            start_loss = float(loss_fn(renderer(splats_tensor), target).item())

        best_loss = start_loss
        end_loss = start_loss
        best_tensor = splats_tensor.detach().clone()
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

        for iteration in range(max(0, num_iters)):
            iterations_run = iteration + 1
            optimizer.zero_grad()
            rendered = renderer(splats_tensor)
            loss = loss_fn(rendered, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([splats_tensor], max_norm=1.0)

            optimizer_helper.step_with_constraints(optimizer, splats_tensor)
            with torch.no_grad():
                self._clamp_splat_parameters(splats_tensor)

            loss_value = float(loss.item())
            end_loss = loss_value
            if loss_value < best_loss:
                best_loss = loss_value
                best_tensor = splats_tensor.detach().clone()

            if verbose and (iteration + 1) % 50 == 0:
                logger.info("  Iteration %s/%s: loss = %.6f", iteration + 1, num_iters, loss_value)

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

        with torch.no_grad():
            best_rendered = renderer(best_tensor).detach()

        return tensor_to_splats(best_tensor), {
            "start_loss": start_loss,
            "end_loss": end_loss,
            "best_loss": best_loss,
            "iterations": int(iterations_run),
            "lr_decays": int(decay_count),
        }, best_rendered

    def _compute_quality_metrics(
        self,
        splats: List[GaussianSplat],
        target: torch.Tensor,
        renderer: torch.nn.Module,
        loss_fn: L1SSIMLoss,
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
        loss_fn: L1SSIMLoss,
        precomputed_rendered: Optional[torch.Tensor] = None,
        precomputed_coverage_map: Optional[np.ndarray] = None,
    ) -> Tuple[Dict[str, float], torch.Tensor, np.ndarray]:
        """Compute quality metrics while optionally reusing rendered and coverage maps."""
        height, width = int(target.shape[0]), int(target.shape[1])
        if not splats:
            empty_render = torch.zeros((height, width, 3), dtype=target.dtype, device=target.device)
            empty_coverage = np.zeros((height, width), dtype=np.float32)
            return (
                {
                    "l1": 0.0, "mse": 0.0, "psnr": 0.0, "ssim": 0.0,
                    "psnr_srgb": 0.0, "ssim_srgb": 0.0, "coverage": 0.0,
                },
                empty_render,
                empty_coverage,
            )

        if precomputed_rendered is None:
            with torch.no_grad():
                rendered = renderer(splats_to_tensor(splats, device=self.device)).detach()
        else:
            rendered = precomputed_rendered.detach()

        # Use the honest shared metric: standard windowed SSIM plus perceptual
        # (sRGB-display) variants. The old path used L1SSIMLoss._global_ssim, a
        # global single-window SSIM that over-reports, and omitted the
        # psnr_srgb/ssim_srgb keys the acceptance gate checks -- so on machines
        # without an SVG rasterizer the perceptual gates read 0.0 and always
        # failed even good runs.
        with torch.no_grad():
            target_np = target.detach().cpu().numpy()
            rendered_np = rendered.detach().cpu().numpy()
        metrics = compute_quality_metrics(target_np[..., :3], rendered_np[..., :3])

        if precomputed_coverage_map is not None and precomputed_coverage_map.shape == (height, width):
            coverage_map = precomputed_coverage_map
        else:
            coverage_map = self._build_alpha_coverage_map(splats=splats, width=width, height=height)
        coverage = self._compute_coverage_ratio(coverage_map)
        metrics["coverage"] = coverage
        return (
            metrics,
            rendered,
            coverage_map,
        )

    def _clamp_splat_parameters(self, splats_tensor: torch.Tensor) -> None:
        """Clamp splat parameters to valid ranges."""
        with torch.no_grad():
            splats_tensor[:, 0].clamp_(0, float(max(self._image_width - 1, 0)))
            splats_tensor[:, 1].clamp_(0, float(max(self._image_height - 1, 0)))
            splats_tensor[:, 2].clamp_(min=1e-4)
            splats_tensor[:, 3].clamp_(min=1e-4)
            splats_tensor[:, 4].remainder_(2.0 * torch.pi)
            splats_tensor[:, 6:9].clamp_(0, 1)
            splats_tensor[:, 9].clamp_(0, 1)

    def _add_error_driven_splats(
        self,
        splats: List[GaussianSplat],
        image: np.ndarray,
        target: torch.Tensor,
        renderer: torch.nn.Module,
        rng: np.random.Generator,
        edge_map: Optional[np.ndarray] = None,
        stage_idx: int = 0,
        precomputed_rendered: Optional[torch.Tensor] = None,
        precomputed_coverage_map: Optional[np.ndarray] = None,
        structure_primary: Optional[np.ndarray] = None,
        structure_anisotropy: Optional[np.ndarray] = None,
        max_splats_cap: Optional[int] = None,
    ) -> Tuple[List[GaussianSplat], Optional[np.ndarray]]:
        """Add new splats using residual, uncovered-opacity, and edge cues."""
        cap = int(self.max_splats if max_splats_cap is None else np.clip(max_splats_cap, 0, self.max_splats))
        if len(splats) >= cap:
            return splats, precomputed_coverage_map

        if precomputed_rendered is None:
            splats_tensor = splats_to_tensor(splats, device=self.device)
            with torch.no_grad():
                rendered = renderer(splats_tensor)
        else:
            rendered = precomputed_rendered
        with torch.no_grad():
            residual_map = target - rendered
            error_map = torch.mean(residual_map ** 2, dim=-1)
        error_np = error_map.cpu().numpy()
        residual_np = residual_map.cpu().numpy()
        error_norm = self._normalize_map(error_np)
        height, width = image.shape[:2]
        if edge_map is None or edge_map.shape != (height, width):
            edge_map = self._build_edge_map(image)

        if precomputed_coverage_map is not None and precomputed_coverage_map.shape == (height, width):
            coverage_map = precomputed_coverage_map
        else:
            coverage_map = self._build_alpha_coverage_map(
                splats=splats,
                width=width,
                height=height,
            )
        uncovered_map = np.clip(1.0 - coverage_map, 0.0, 1.0).astype(np.float32)

        coverage_ratio = self._compute_coverage_ratio(coverage_map)
        target_coverage = float(np.clip(self.refinement_config.get("coverage_target", 0.985), 0.0, 1.0))
        coverage_deficit = max(target_coverage - coverage_ratio, 0.0)

        weight_error = float(max(self.refinement_config.get("densify_weight_error", 0.50), 0.0))
        weight_uncovered = float(max(self.refinement_config.get("densify_weight_uncovered", 0.40), 0.0))
        weight_edge = float(max(self.refinement_config.get("densify_weight_edge", 0.10), 0.0))
        weight_sum = max(weight_error + weight_uncovered + weight_edge, 1e-8)
        sampling_map = (
            (weight_error / weight_sum) * error_norm
            + (weight_uncovered / weight_sum) * uncovered_map
            + (weight_edge / weight_sum) * edge_map
        )
        sampling_map = np.clip(sampling_map, 0.0, 1.0).astype(np.float32)
        if float(np.sum(sampling_map)) <= 1e-12:
            sampling_map = np.maximum(error_norm, uncovered_map)

        base_percentile = float(np.clip(self.refinement_config["densify_percentile"], 0.0, 100.0))
        stage_scale = max(len(self.stages) - stage_idx, 1) / max(len(self.stages), 1)
        adaptive_percentile = float(np.clip(base_percentile - 35.0 * coverage_deficit * stage_scale, 45.0, 99.8))

        densify_fraction = float(np.clip(self.refinement_config["densify_fraction"], 0.01, 1.0))
        deficit_boost = 1.0 + float(self.refinement_config.get("coverage_densify_boost", 2.0)) * coverage_deficit
        max_new = min(
            cap - len(splats),
            int(np.ceil(len(splats) * densify_fraction * deficit_boost)),
        )
        if max_new <= 0:
            return splats, coverage_map

        x_indices, y_indices, sample_weights = self._sample_candidate_positions(
            score_map=sampling_map,
            percentile=adaptive_percentile,
            max_samples=max_new,
            rng=rng,
        )
        if len(x_indices) == 0:
            return splats, coverage_map

        new_splats: List[GaussianSplat] = []
        residual_color_gain = float(self.refinement_config.get("residual_color_gain", 0.75))
        sigma_minor_min = float(self.refinement_config.get("sigma_minor_min", 0.35))
        sigma_min = float(self.refinement_config.get("sigma_min", 0.45))
        sigma_max = float(self.refinement_config.get("sigma_max", 4.0))
        sigma_scale = float(self.refinement_config.get("sigma_scale", 2.0))
        sigma_fill_max = float(max(self.refinement_config.get("coverage_sigma_max", sigma_max * 1.8), sigma_max))
        for idx, (x, y) in enumerate(zip(x_indices, y_indices)):
            base_color = estimate_local_color(image, x, y)
            residual_rgb = residual_np[y, x, :3].astype(np.float32)
            color = np.clip(base_color + residual_color_gain * residual_rgb, 0.0, 1.0).astype(np.float32)
            if not np.isfinite(color).all():
                color = base_color

            detail_need = float(error_norm[y, x])
            fill_need = float(uncovered_map[y, x])
            edge_need = float(edge_map[y, x])

            sigma_detail = float(np.clip(sigma_max - sigma_scale * detail_need, sigma_min, sigma_max))
            sigma = float(
                np.clip(
                    (1.0 - fill_need) * sigma_detail + fill_need * sigma_fill_max,
                    sigma_min,
                    sigma_fill_max,
                )
            )
            alpha = float(
                np.clip(
                    self.refinement_config["alpha_base"]
                    + self.refinement_config["alpha_scale"] * (0.55 * detail_need + 0.45 * fill_need),
                    self.refinement_config["alpha_min"],
                    self.refinement_config["alpha_max"],
                )
            )
            x_center = float(np.clip(x + rng.uniform(-0.5, 0.5), 0.0, width - 1.0))
            y_center = float(np.clip(y + rng.uniform(-0.5, 0.5), 0.0, height - 1.0))

            local_structure_edge_threshold = float(
                np.clip(self.refinement_config.get("structure_local_edge_threshold", 0.18), 0.0, 1.0)
            )
            local_structure_detail_threshold = float(
                np.clip(self.refinement_config.get("structure_local_detail_threshold", 0.22), 0.0, 1.0)
            )
            prefer_local_structure = bool(
                edge_need >= local_structure_edge_threshold
                or detail_need >= local_structure_detail_threshold
            )
            if (
                structure_primary is not None
                and structure_anisotropy is not None
                and structure_primary.shape[:2] == (height, width)
                and structure_primary.shape[-1] == 2
                and structure_anisotropy.shape == (height, width)
                and not prefer_local_structure
            ):
                primary_direction = structure_primary[y, x]
                anisotropy = float(structure_anisotropy[y, x])
            else:
                primary_direction, anisotropy = self._analyze_local_structure(image, x, y)
            anisotropy_threshold = float(max(1.0, self.refinement_config.get("densify_anisotropy_threshold", 1.30)))
            anisotropy_edge_threshold = float(
                np.clip(self.refinement_config.get("densify_anisotropy_edge_threshold", 0.14), 0.0, 1.0)
            )
            strong_edge_threshold = float(
                np.clip(self.refinement_config.get("densify_strong_edge_threshold", 0.38), 0.0, 1.0)
            )
            make_anisotropic = (
                anisotropy >= anisotropy_threshold and edge_need >= anisotropy_edge_threshold
            ) or (
                anisotropy >= max(1.0, anisotropy_threshold - 0.08) and edge_need >= strong_edge_threshold
            )
            if make_anisotropic:
                angle = float(np.arctan2(primary_direction[1], primary_direction[0]))
                cos_a, sin_a = float(np.cos(angle)), float(np.sin(angle))
                rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
                sigma_major = sigma * (1.0 + 0.5 * fill_need)
                anisotropy_cap = max(
                    1.0,
                    min(float(anisotropy), float(self.refinement_config.get("local_structure_anisotropy_clip", 4.0))),
                )
                sigma_minor = max(sigma_major / anisotropy_cap, sigma_minor_min)
                if edge_need > 0.5 and fill_need < 0.5:
                    sigma_minor = max(sigma_minor * 0.75, sigma_minor_min)
                new_splat = (
                    create_anisotropic_splat(
                        center=np.array([x_center, y_center], dtype=np.float32),
                        eigenvals=np.array([sigma_major**2, sigma_minor**2], dtype=np.float32),
                        eigenvecs=rotation_matrix,
                        color=color,
                        alpha=alpha,
                    )
                )
            else:
                new_splat = (
                    create_isotropic_splat(
                        center=np.array([x_center, y_center], dtype=np.float32),
                        sigma=sigma,
                        color=color,
                        alpha=alpha,
                    )
                )
            new_splat.importance = float(np.clip(sample_weights[idx], 0.0, 1.0))
            new_splats.append(new_splat)

        logger.info(
            "Added %s splats (coverage %.1f%% -> target %.1f%%)",
            len(new_splats),
            coverage_ratio * 100.0,
            target_coverage * 100.0,
        )
        if not new_splats:
            return splats, coverage_map

        # Incremental coverage update: apply only newly inserted splats to current transmittance.
        transmittance = np.clip(1.0 - coverage_map, 0.0, 1.0).astype(np.float32, copy=True)
        self._apply_splats_to_transmittance(
            transmittance=transmittance,
            splats=new_splats,
            width=width,
            height=height,
        )
        updated_coverage = np.clip(1.0 - transmittance, 0.0, 1.0).astype(np.float32)
        return splats + new_splats, updated_coverage

    def _run_residual_detail_passes(
        self,
        splats: List[GaussianSplat],
        image: np.ndarray,
        target: torch.Tensor,
        renderer: torch.nn.Module,
        loss_fn: L1SSIMLoss,
        optimizer_helper: SplatOptimizer,
        rng: np.random.Generator,
        edge_map: np.ndarray,
        verbose: bool,
    ) -> Tuple[List[GaussianSplat], List[Dict[str, Any]]]:
        """Run late residual-focused densification with small isotropic splats."""
        if not bool(self.refinement_config.get("residual_detail_enabled", False)):
            return splats, []

        passes = int(max(1, self.refinement_config.get("residual_detail_passes", 1)))
        residual_metrics: List[Dict[str, Any]] = []
        current_splats = splats
        height, width = image.shape[:2]

        for pass_idx in range(passes):
            if len(current_splats) >= self.max_splats:
                break

            with torch.no_grad():
                if current_splats:
                    rendered = renderer(splats_to_tensor(current_splats, device=self.device))
                else:
                    rendered = torch.zeros(
                        (height, width, 3),
                        dtype=target.dtype,
                        device=target.device,
                    )
                residual_map = target - rendered
                error_map = torch.mean(residual_map ** 2, dim=-1)

            error_norm = self._normalize_map(error_map.cpu().numpy())
            residual_np = residual_map.cpu().numpy()

            edge_weight = float(np.clip(self.refinement_config.get("residual_detail_edge_weight", 0.30), 0.0, 1.0))
            score_map = self._normalize_map(error_norm * (1.0 + edge_weight * edge_map))
            percentile = float(np.clip(self.refinement_config.get("residual_detail_percentile", 90.0), 0.0, 100.0))
            fraction = float(np.clip(self.refinement_config.get("residual_detail_fraction", 0.12), 0.01, 1.0))
            max_new = min(
                self.max_splats - len(current_splats),
                int(np.ceil(max(1, len(current_splats)) * fraction)),
            )
            if max_new <= 0:
                break

            x_indices, y_indices, sample_weights = self._sample_candidate_positions(
                score_map=score_map,
                percentile=percentile,
                max_samples=max_new,
                rng=rng,
            )
            if len(x_indices) == 0:
                break

            sigma_min = float(max(0.10, self.refinement_config.get("residual_detail_sigma_min", 0.28)))
            sigma_max = float(max(sigma_min, self.refinement_config.get("residual_detail_sigma_max", 1.20)))
            alpha_min = float(np.clip(self.refinement_config.get("residual_detail_alpha_min", 0.16), 0.0, 1.0))
            alpha_max = float(np.clip(self.refinement_config.get("residual_detail_alpha_max", 0.70), alpha_min, 1.0))
            residual_color_gain = float(self.refinement_config.get("residual_detail_color_gain", 0.95))

            new_splats: List[GaussianSplat] = []
            for idx, (x, y) in enumerate(zip(x_indices, y_indices)):
                base_color = estimate_local_color(image, x, y)
                residual_rgb = residual_np[y, x, :3].astype(np.float32)
                color = np.clip(base_color + residual_color_gain * residual_rgb, 0.0, 1.0).astype(np.float32)
                if not np.isfinite(color).all():
                    color = base_color

                detail_need = float(error_norm[y, x])
                sigma = float(np.clip(sigma_max - (sigma_max - sigma_min) * detail_need, sigma_min, sigma_max))
                alpha = float(np.clip(alpha_min + (alpha_max - alpha_min) * (0.30 + 0.70 * detail_need), alpha_min, alpha_max))
                x_center = float(np.clip(x + rng.uniform(-0.35, 0.35), 0.0, width - 1.0))
                y_center = float(np.clip(y + rng.uniform(-0.35, 0.35), 0.0, height - 1.0))

                splat = create_isotropic_splat(
                    center=np.array([x_center, y_center], dtype=np.float32),
                    sigma=sigma,
                    color=color,
                    alpha=alpha,
                )
                splat.importance = float(np.clip(0.65 + 0.35 * sample_weights[idx], 0.0, 1.0))
                new_splats.append(splat)

            if not new_splats:
                break

            if verbose:
                logger.info("Residual detail pass %s: adding %s small splats", pass_idx + 1, len(new_splats))

            current_splats = current_splats + new_splats
            residual_iters = int(max(0, self.refinement_config.get("residual_detail_iters", 8)))
            current_splats, stage_metric, stage_rendered = self._optimize_stage(
                splats=current_splats,
                target=target,
                renderer=renderer,
                loss_fn=loss_fn,
                optimizer_helper=optimizer_helper,
                num_iters=residual_iters,
                verbose=verbose,
            )

            quality, _, _ = self._compute_quality_metrics_cached(
                splats=current_splats,
                target=target,
                renderer=renderer,
                loss_fn=loss_fn,
                precomputed_rendered=stage_rendered,
            )
            stage_metric.update(quality)
            stage_metric["stage"] = -1
            stage_metric["stage_type"] = "residual_detail"
            stage_metric["residual_pass"] = pass_idx + 1
            stage_metric["splat_count"] = len(current_splats)
            residual_metrics.append(stage_metric)

        return current_splats, residual_metrics

    def _prune_splats(
        self,
        splats: List[GaussianSplat],
        max_count: int,
        target: Optional[torch.Tensor] = None,
        renderer: Optional[torch.nn.Module] = None,
        precomputed_coverage_map: Optional[np.ndarray] = None,
    ) -> List[GaussianSplat]:
        """Prune splats by utility score: residual support + gap filling + alpha."""
        if len(splats) <= max_count:
            return splats

        if target is None or renderer is None:
            splats_sorted = sorted(splats, key=lambda s: s.alpha, reverse=True)
            pruned = splats_sorted[:max_count]
            logger.info("Pruned from %s to %s splats", len(splats), len(pruned))
            return pruned

        with torch.no_grad():
            rendered = renderer(splats_to_tensor(splats, device=self.device))
            error_map = torch.mean((rendered - target) ** 2, dim=-1).cpu().numpy()
        error_norm = self._normalize_map(error_map)
        height, width = error_norm.shape
        if precomputed_coverage_map is not None and precomputed_coverage_map.shape == (height, width):
            coverage_map = precomputed_coverage_map
        else:
            coverage_map = self._build_alpha_coverage_map(splats=splats, width=width, height=height)
        uncovered_map = np.clip(1.0 - coverage_map, 0.0, 1.0).astype(np.float32)

        combined_scores: List[Tuple[float, GaussianSplat]] = []
        w_alpha = float(self.refinement_config.get("prune_weight_contribution", 0.45))
        w_residual = float(self.refinement_config.get("prune_weight_residual", 0.35))
        w_uncovered = float(max(self.refinement_config.get("prune_weight_uncovered", 0.20), 0.0))
        weight_sum = max(w_alpha + w_residual + w_uncovered, 1e-8)
        sample_radius_scale = float(max(self.refinement_config.get("prune_sample_radius", 1.4), 0.8))
        for splat in splats:
            raw = splat.to_raw_splat()
            cx = int(np.clip(round(float(raw.x)), 0, width - 1))
            cy = int(np.clip(round(float(raw.y)), 0, height - 1))
            rx = max(1, int(np.ceil(sample_radius_scale * float(raw.sx))))
            ry = max(1, int(np.ceil(sample_radius_scale * float(raw.sy))))
            x0 = max(0, cx - rx)
            x1 = min(width, cx + rx + 1)
            y0 = max(0, cy - ry)
            y1 = min(height, cy + ry + 1)

            local_error = float(np.mean(error_norm[y0:y1, x0:x1])) if x0 < x1 and y0 < y1 else 0.0
            local_uncovered = float(np.mean(uncovered_map[y0:y1, x0:x1])) if x0 < x1 and y0 < y1 else 0.0
            alpha_score = float(np.clip(splat.alpha, 0.0, 1.0))
            keep_score = (
                (w_alpha / weight_sum) * alpha_score
                + (w_residual / weight_sum) * local_error
                + (w_uncovered / weight_sum) * local_uncovered
            )
            combined_scores.append((keep_score, splat))

        combined_scores.sort(key=lambda item: item[0], reverse=True)
        pruned = [splat for _, splat in combined_scores[:max_count]]
        logger.info("Pruned from %s to %s splats", len(splats), len(pruned))
        return pruned

    def _postprocess_splats(
        self,
        splats: List[GaussianSplat],
        image: np.ndarray,
        rng: np.random.Generator,
    ) -> List[GaussianSplat]:
        """Post-process splats and backfill persistent uncovered regions."""
        splats = [s for s in splats if s.alpha > 0.03]
        if not splats:
            return splats

        height, width = image.shape[:2]
        coverage_map = self._build_alpha_coverage_map(splats=splats, width=width, height=height)
        coverage_ratio = self._compute_coverage_ratio(coverage_map)
        min_final_coverage = float(np.clip(self.refinement_config.get("coverage_target", 0.985), 0.0, 1.0))

        # If we are saturated at max_splats, reclaim budget from low-value splats.
        if coverage_ratio < min_final_coverage and len(splats) >= self.max_splats:
            edge_map = self._build_edge_map(image)
            reallocate_fraction = float(
                np.clip(self.refinement_config.get("reallocate_for_coverage_fraction", 0.08), 0.0, 0.30)
            )
            reallocate_budget = int(min(len(splats) // 4, max(1, np.ceil(len(splats) * reallocate_fraction))))
            ranked: List[Tuple[float, int]] = []
            for idx, splat in enumerate(splats):
                x = int(np.clip(round(float(splat.mu[0])), 0, width - 1))
                y = int(np.clip(round(float(splat.mu[1])), 0, height - 1))
                local_uncovered = float(np.clip(1.0 - coverage_map[y, x], 0.0, 1.0))
                edge_value = float(edge_map[y, x])
                alpha_value = float(np.clip(splat.alpha, 0.0, 1.0))
                keep_score = 0.40 * alpha_value + 0.40 * local_uncovered + 0.20 * edge_value
                ranked.append((keep_score, idx))
            ranked.sort(key=lambda pair: pair[0])
            drop_indices = {idx for _, idx in ranked[:reallocate_budget]}
            if drop_indices:
                splats = [s for idx, s in enumerate(splats) if idx not in drop_indices]
                coverage_map = self._build_alpha_coverage_map(splats=splats, width=width, height=height)
                coverage_ratio = self._compute_coverage_ratio(coverage_map)

        final_fill_budget = int(
            max(
                0,
                min(
                    self.max_splats - len(splats),
                    np.ceil(self.max_splats * float(self.refinement_config.get("final_fill_fraction", 0.10))),
                ),
            )
        )

        if coverage_ratio < min_final_coverage and final_fill_budget > 0:
            uncovered = np.clip(1.0 - coverage_map, 0.0, 1.0).astype(np.float32)
            threshold = float(np.percentile(uncovered, 80.0))
            candidate_mask = uncovered >= threshold
            y_indices, x_indices = np.where(candidate_mask)
            if len(x_indices) > 0:
                sample_count = int(min(final_fill_budget, len(x_indices)))
                weights = uncovered[y_indices, x_indices].astype(np.float64)
                if float(weights.sum()) > 1e-12:
                    weights = weights / float(weights.sum())
                else:
                    weights = None
                sampled_idx = rng.choice(len(x_indices), size=sample_count, replace=False, p=weights)
                sigma_fill = float(
                    np.clip(
                        self.refinement_config.get("coverage_sigma_max", 6.0),
                        self.refinement_config.get("sigma_min", 0.5),
                        20.0,
                    )
                )
                alpha_fill = float(
                    np.clip(
                        self.refinement_config.get("coverage_alpha_fill", self.refinement_config.get("alpha_base", 0.3)),
                        self.refinement_config.get("alpha_min", 0.05),
                        self.refinement_config.get("alpha_max", 0.95),
                    )
                )
                for idx in sampled_idx:
                    x = int(x_indices[idx])
                    y = int(y_indices[idx])
                    x_center = float(np.clip(x + rng.uniform(-0.5, 0.5), 0.0, width - 1.0))
                    y_center = float(np.clip(y + rng.uniform(-0.5, 0.5), 0.0, height - 1.0))
                    color = estimate_local_color(image, x, y)
                    splat = create_isotropic_splat(
                        center=np.array([x_center, y_center], dtype=np.float32),
                        sigma=sigma_fill,
                        color=color,
                        alpha=alpha_fill,
                    )
                    splat.importance = 0.05
                    splats.append(splat)

            coverage_map = self._build_alpha_coverage_map(splats=splats, width=width, height=height)
            coverage_ratio = self._compute_coverage_ratio(coverage_map)

        logger.info(
            "Post-processing: %s splats remaining (coverage=%.1f%%)",
            len(splats),
            coverage_ratio * 100.0,
        )
        return splats

    def _generate_svg(self, splats: List[GaussianSplat], width: int, height: int) -> str:
        """Generate SVG content."""
        from .io import generate_svg_content

        return generate_svg_content(
            splats,
            width,
            height,
            self.k_sigma,
            background_linear_rgb=self._background_linear_rgb,
        )

    def _generate_drawingml(self, splats: List[GaussianSplat], width: int, height: int) -> str:
        """Generate DrawingML slide XML content."""
        return generate_drawingml_slide_content(splats, width, height, self.k_sigma)

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
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2, sort_keys=True)

    def _write_manifest(self, artifacts_dir: Optional[Path], manifest: Dict[str, Any]) -> None:
        """Write run manifest if artifact directory is configured."""
        if artifacts_dir is None:
            return
        manifest_path = artifacts_dir / "run_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)

    def _evaluate_acceptance(
        self, metrics: Dict[str, float], criteria: Dict[str, float]
    ) -> Dict[str, Any]:
        """Evaluate pass/fail against acceptance criteria."""
        checks: Dict[str, bool] = {}

        if "min_psnr" in criteria:
            checks["psnr"] = float(metrics.get("psnr", 0.0)) >= float(criteria["min_psnr"])
        if "min_ssim" in criteria:
            checks["ssim"] = float(metrics.get("ssim", 0.0)) >= float(criteria["min_ssim"])
        # Perceptual (sRGB-display) gates: what the eye actually sees.
        if "min_psnr_srgb" in criteria:
            checks["psnr_srgb"] = float(metrics.get("psnr_srgb", 0.0)) >= float(criteria["min_psnr_srgb"])
        if "min_ssim_srgb" in criteria:
            checks["ssim_srgb"] = float(metrics.get("ssim_srgb", 0.0)) >= float(criteria["min_ssim_srgb"])
        if "max_runtime_sec" in criteria:
            checks["runtime_sec"] = float(metrics.get("runtime_sec", 0.0)) <= float(criteria["max_runtime_sec"])
        if "max_splats" in criteria:
            checks["splat_count"] = float(metrics.get("splat_count", 0.0)) <= float(criteria["max_splats"])

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

    def _normalize_map(self, values: np.ndarray) -> np.ndarray:
        """Normalize array to [0, 1]."""
        min_v = float(np.min(values))
        max_v = float(np.max(values))
        if max_v <= min_v + 1e-12:
            return np.zeros_like(values, dtype=np.float32)
        return ((values - min_v) / (max_v - min_v)).astype(np.float32)

    def _estimate_background_color(self, image: np.ndarray) -> np.ndarray:
        """
        Estimate a stable background color from border pixels in linear RGB.

        This avoids SVG transparency defaulting to white when the optimizer
        implicitly relies on a non-white canvas.
        """
        if image.ndim != 3 or image.shape[2] < 3:
            return np.zeros(3, dtype=np.float32)

        rgb = np.asarray(image[:, :, :3], dtype=np.float32)
        height, width = rgb.shape[:2]
        border = max(1, int(round(0.04 * float(min(height, width)))))

        top = rgb[:border, :, :].reshape(-1, 3)
        bottom = rgb[max(height - border, 0):, :, :].reshape(-1, 3)
        left = rgb[:, :border, :].reshape(-1, 3)
        right = rgb[:, max(width - border, 0):, :].reshape(-1, 3)
        border_pixels = np.concatenate([top, bottom, left, right], axis=0)

        if image.shape[2] >= 4:
            alpha = np.asarray(image[:, :, 3], dtype=np.float32)
            top_a = alpha[:border, :].reshape(-1)
            bottom_a = alpha[max(height - border, 0):, :].reshape(-1)
            left_a = alpha[:, :border].reshape(-1)
            right_a = alpha[:, max(width - border, 0):].reshape(-1)
            border_alpha = np.concatenate([top_a, bottom_a, left_a, right_a], axis=0)
            valid = border_alpha > 0.02
            if np.any(valid):
                border_pixels = border_pixels[valid]

        if border_pixels.size == 0:
            border_pixels = rgb.reshape(-1, 3)
        border_std = float(np.mean(np.std(border_pixels, axis=0)))
        max_uniform_std = float(self.refinement_config.get("background_uniformity_std_max", 0.18))
        if border_std > max_uniform_std:
            return np.zeros(3, dtype=np.float32)
        background = np.median(border_pixels, axis=0).astype(np.float32)
        if not np.isfinite(background).all():
            return np.zeros(3, dtype=np.float32)
        return np.clip(background, 0.0, 1.0)

    def _sample_candidate_positions(
        self,
        score_map: np.ndarray,
        percentile: float,
        max_samples: int,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample top-scoring coordinates with probability proportional to score."""
        if max_samples <= 0:
            return (
                np.empty((0,), dtype=np.int32),
                np.empty((0,), dtype=np.int32),
                np.empty((0,), dtype=np.float32),
            )

        threshold = float(np.percentile(score_map, percentile))
        mask = score_map >= threshold
        y_indices, x_indices = np.where(mask)

        if len(x_indices) == 0:
            flat = score_map.reshape(-1)
            if flat.size == 0:
                return (
                    np.empty((0,), dtype=np.int32),
                    np.empty((0,), dtype=np.int32),
                    np.empty((0,), dtype=np.float32),
                )
            topk = min(max_samples, flat.size)
            top_idx = np.argpartition(flat, -topk)[-topk:]
            y_indices, x_indices = np.unravel_index(top_idx, score_map.shape)

        sample_count = min(int(max_samples), len(x_indices))
        if sample_count <= 0:
            return (
                np.empty((0,), dtype=np.int32),
                np.empty((0,), dtype=np.int32),
                np.empty((0,), dtype=np.float32),
            )

        weights = score_map[y_indices, x_indices].astype(np.float64)
        if float(weights.sum()) > 1e-12:
            weights = weights / float(weights.sum())
        else:
            weights = None

        selected = rng.choice(len(x_indices), size=sample_count, replace=False, p=weights)
        selected_x = x_indices[selected].astype(np.int32)
        selected_y = y_indices[selected].astype(np.int32)
        selected_scores = score_map[selected_y, selected_x].astype(np.float32)
        return selected_x, selected_y, selected_scores

    def _build_alpha_coverage_map(self, splats: List[GaussianSplat], width: int, height: int) -> np.ndarray:
        """Build alpha coverage map where 1 means fully covered by accumulated opacity."""
        transmittance = np.ones((height, width), dtype=np.float32)
        self._apply_splats_to_transmittance(
            transmittance=transmittance,
            splats=splats,
            width=width,
            height=height,
        )
        coverage = 1.0 - transmittance
        return np.clip(coverage, 0.0, 1.0).astype(np.float32)

    def _apply_splats_to_transmittance(
        self,
        transmittance: np.ndarray,
        splats: List[GaussianSplat],
        width: int,
        height: int,
    ) -> None:
        """Apply splat alpha-over attenuation into a transmittance map in place."""
        footprint_sigma = float(max(self.refinement_config.get("coverage_footprint_sigma", 3.0), 1.0))
        for splat in splats:
            raw = splat.to_raw_splat()
            cx = float(np.clip(raw.x, 0.0, width - 1.0))
            cy = float(np.clip(raw.y, 0.0, height - 1.0))
            sx = max(float(raw.sx), 1e-4)
            sy = max(float(raw.sy), 1e-4)
            theta = float(raw.theta)

            radius_x = max(1, int(np.ceil(footprint_sigma * sx)))
            radius_y = max(1, int(np.ceil(footprint_sigma * sy)))
            x0 = max(0, int(np.floor(cx - radius_x)))
            x1 = min(width, int(np.ceil(cx + radius_x + 1)))
            y0 = max(0, int(np.floor(cy - radius_y)))
            y1 = min(height, int(np.ceil(cy + radius_y + 1)))
            if x0 >= x1 or y0 >= y1:
                continue

            xs = np.arange(x0, x1, dtype=np.float32)
            ys = np.arange(y0, y1, dtype=np.float32)
            gx, gy = np.meshgrid(xs, ys)
            dx = gx - cx
            dy = gy - cy
            cos_t = float(np.cos(theta))
            sin_t = float(np.sin(theta))
            u = cos_t * dx + sin_t * dy
            v = -sin_t * dx + cos_t * dy
            quadratic = (u * u) / (sx * sx) + (v * v) / (sy * sy)
            density = float(max(0.0, splat.alpha)) * np.exp(-0.5 * quadratic)
            layer_alpha = 1.0 - np.exp(-density)
            transmittance[y0:y1, x0:x1] *= np.clip(1.0 - layer_alpha.astype(np.float32), 0.0, 1.0)

    def _compute_coverage_ratio(self, coverage_map: np.ndarray) -> float:
        """Compute covered-pixel ratio under configured alpha threshold."""
        threshold = float(np.clip(self.refinement_config.get("coverage_threshold", 0.03), 0.0, 1.0))
        return float(np.mean(coverage_map >= threshold))

    def _build_contribution_map(self, splats: List[GaussianSplat], width: int, height: int) -> np.ndarray:
        """Backward-compatible alias for the alpha coverage map."""
        return self._build_alpha_coverage_map(splats=splats, width=width, height=height)

    def _resolve_target_size(self, input_path: str) -> Tuple[int, int]:
        """Resolve effective target size after applying resolution scale."""
        if self.target_size is not None:
            base_w, base_h = self.target_size
        else:
            with Image.open(input_path) as img:
                base_w, base_h = img.size

        scaled_w = max(1, int(round(base_w * self.resolution_scale)))
        scaled_h = max(1, int(round(base_h * self.resolution_scale)))
        return (scaled_w, scaled_h)

    def _get_profile_defaults(self, profile: str) -> Dict[str, Dict[str, Any]]:
        """Return tuned defaults for quality profile."""
        profiles: Dict[str, Dict[str, Dict[str, Any]]] = {
            "m4-fast-loop": {
                "learning_rates": {
                    "position": 0.0095,
                    "covariance": 0.0040,
                    "color": 0.016,
                    "alpha": 0.0080,
                },
                "loss_weights": {"l1_weight": 1.0, "ssim_weight": 0.18},
                "refinement": {
                    "densify_percentile": 90.0,
                    "densify_fraction": 0.18,
                    "base_layer_fraction": 0.45,
                    "base_layer_alpha": 0.34,
                    "sigma_min": 1.4,
                    "sigma_max": 4.0,
                    "sigma_scale": 2.2,
                    "sigma_minor_min": 0.45,
                    "coverage_sigma_max": 5.5,
                    "coverage_alpha_fill": 0.32,
                    "coverage_threshold": 0.035,
                    "coverage_target": 0.92,
                    "coverage_footprint_sigma": 3.0,
                    "coverage_densify_boost": 1.2,
                    "reallocate_for_coverage_fraction": 0.04,
                    "densify_weight_error": 0.60,
                    "densify_weight_uncovered": 0.30,
                    "densify_weight_edge": 0.10,
                    "alpha_base": 0.30,
                    "alpha_scale": 0.40,
                    "alpha_min": 0.20,
                    "alpha_max": 0.85,
                    "prune_weight_contribution": 0.75,
                    "prune_weight_residual": 0.25,
                    "prune_weight_uncovered": 0.10,
                    "prune_sample_radius": 1.3,
                    "residual_color_gain": 0.60,
                    "final_fill_fraction": 0.06,
                    "structure_precompute_enabled": True,
                    "structure_smoothing_sigma": 0.0,
                    "structure_anisotropy_clip": 10.0,
                    "structure_min_coherence": 0.12,
                    "structure_local_window": 7,
                    "local_structure_anisotropy_clip": 4.0,
                    "local_structure_min_coherence": 0.12,
                    "local_structure_min_energy": 1e-4,
                    "init_anisotropy_threshold": 1.55,
                    "densify_anisotropy_threshold": 1.30,
                    "densify_anisotropy_edge_threshold": 0.14,
                    "densify_strong_edge_threshold": 0.38,
                    "residual_detail_enabled": False,
                    "residual_detail_reserve_fraction": 0.00,
                    "residual_detail_passes": 1,
                    "residual_detail_percentile": 90.0,
                    "residual_detail_fraction": 0.10,
                    "residual_detail_sigma_min": 0.35,
                    "residual_detail_sigma_max": 1.40,
                    "residual_detail_alpha_min": 0.14,
                    "residual_detail_alpha_max": 0.65,
                    "residual_detail_iters": 4,
                    "residual_detail_edge_weight": 0.25,
                    "residual_detail_color_gain": 0.90,
                },
                "schedule": {
                    "enabled": True,
                    "check_interval": 25,
                    "patience_checks": 1,
                    "decay_ratio": 2.0,
                    "max_decays": 1,
                    "min_delta": 4e-4,
                },
            },
            "fast": {
                "learning_rates": {
                    "position": 0.009,
                    "covariance": 0.004,
                    "color": 0.016,
                    "alpha": 0.008,
                },
                "loss_weights": {"l1_weight": 1.0, "ssim_weight": 0.18},
                "refinement": {
                    "densify_percentile": 90.0,
                    "densify_fraction": 0.18,
                    "base_layer_fraction": 0.45,
                    "base_layer_alpha": 0.34,
                    "sigma_min": 1.4,
                    "sigma_max": 4.0,
                    "sigma_scale": 2.2,
                    "sigma_minor_min": 0.45,
                    "coverage_sigma_max": 5.5,
                    "coverage_alpha_fill": 0.32,
                    "coverage_threshold": 0.035,
                    "coverage_target": 0.92,
                    "coverage_footprint_sigma": 3.0,
                    "coverage_densify_boost": 1.2,
                    "reallocate_for_coverage_fraction": 0.04,
                    "densify_weight_error": 0.60,
                    "densify_weight_uncovered": 0.30,
                    "densify_weight_edge": 0.10,
                    "alpha_base": 0.30,
                    "alpha_scale": 0.40,
                    "alpha_min": 0.20,
                    "alpha_max": 0.85,
                    "prune_weight_contribution": 0.75,
                    "prune_weight_residual": 0.25,
                    "prune_weight_uncovered": 0.10,
                    "prune_sample_radius": 1.3,
                    "residual_color_gain": 0.60,
                    "final_fill_fraction": 0.06,
                    "structure_precompute_enabled": False,
                    "structure_smoothing_sigma": 0.0,
                    "structure_anisotropy_clip": 10.0,
                    "structure_min_coherence": 0.12,
                    "structure_local_window": 7,
                    "local_structure_anisotropy_clip": 4.0,
                    "local_structure_min_coherence": 0.12,
                    "local_structure_min_energy": 1e-4,
                    "init_anisotropy_threshold": 1.55,
                    "densify_anisotropy_threshold": 1.30,
                    "densify_anisotropy_edge_threshold": 0.14,
                    "densify_strong_edge_threshold": 0.38,
                    "residual_detail_enabled": False,
                    "residual_detail_reserve_fraction": 0.00,
                    "residual_detail_passes": 1,
                    "residual_detail_percentile": 90.0,
                    "residual_detail_fraction": 0.10,
                    "residual_detail_sigma_min": 0.35,
                    "residual_detail_sigma_max": 1.40,
                    "residual_detail_alpha_min": 0.14,
                    "residual_detail_alpha_max": 0.65,
                    "residual_detail_iters": 4,
                    "residual_detail_edge_weight": 0.25,
                    "residual_detail_color_gain": 0.90,
                },
                "schedule": {
                    "enabled": True,
                    "check_interval": 50,
                    "patience_checks": 2,
                    "decay_ratio": 2.0,
                    "max_decays": 1,
                    "min_delta": 2e-4,
                },
            },
            "balanced": {
                "learning_rates": {
                    "position": 0.01,
                    "covariance": 0.005,
                    "color": 0.02,
                    "alpha": 0.01,
                },
                "loss_weights": {"l1_weight": 1.0, "ssim_weight": 0.2},
                "refinement": {
                    "densify_percentile": 85.0,
                    "densify_fraction": 0.25,
                    "base_layer_fraction": 0.40,
                    "base_layer_alpha": 0.40,
                    "sigma_min": 1.25,
                    "sigma_max": 4.0,
                    "sigma_scale": 2.5,
                    "sigma_minor_min": 0.40,
                    "coverage_sigma_max": 6.5,
                    "coverage_alpha_fill": 0.36,
                    "coverage_threshold": 0.03,
                    "coverage_target": 0.965,
                    "coverage_footprint_sigma": 3.0,
                    "coverage_densify_boost": 1.8,
                    "reallocate_for_coverage_fraction": 0.06,
                    "densify_weight_error": 0.50,
                    "densify_weight_uncovered": 0.40,
                    "densify_weight_edge": 0.10,
                    "alpha_base": 0.35,
                    "alpha_scale": 0.45,
                    "alpha_min": 0.20,
                    "alpha_max": 0.90,
                    "prune_weight_contribution": 0.65,
                    "prune_weight_residual": 0.35,
                    "prune_weight_uncovered": 0.20,
                    "prune_sample_radius": 1.4,
                    "residual_color_gain": 0.75,
                    "final_fill_fraction": 0.09,
                    "structure_precompute_enabled": False,
                    "structure_smoothing_sigma": 0.0,
                    "structure_anisotropy_clip": 10.0,
                    "structure_min_coherence": 0.12,
                    "structure_local_window": 7,
                    "local_structure_anisotropy_clip": 4.0,
                    "local_structure_min_coherence": 0.12,
                    "local_structure_min_energy": 1e-4,
                    "init_anisotropy_threshold": 1.55,
                    "densify_anisotropy_threshold": 1.30,
                    "densify_anisotropy_edge_threshold": 0.14,
                    "densify_strong_edge_threshold": 0.38,
                    "residual_detail_enabled": False,
                    "residual_detail_reserve_fraction": 0.00,
                    "residual_detail_passes": 1,
                    "residual_detail_percentile": 90.0,
                    "residual_detail_fraction": 0.10,
                    "residual_detail_sigma_min": 0.35,
                    "residual_detail_sigma_max": 1.30,
                    "residual_detail_alpha_min": 0.14,
                    "residual_detail_alpha_max": 0.68,
                    "residual_detail_iters": 6,
                    "residual_detail_edge_weight": 0.28,
                    "residual_detail_color_gain": 0.92,
                },
                "schedule": {
                    "enabled": True,
                    "check_interval": 50,
                    "patience_checks": 3,
                    "decay_ratio": 2.0,
                    "max_decays": 2,
                    "min_delta": 1e-4,
                },
            },
            "max-fidelity": {
                "learning_rates": {
                    "position": 0.0075,
                    "covariance": 0.0055,
                    "color": 0.016,
                    "alpha": 0.010,
                },
                "loss_weights": {"l1_weight": 1.0, "ssim_weight": 0.24},
                "refinement": {
                    "densify_percentile": 74.0,
                    "densify_fraction": 0.40,
                    "base_layer_fraction": 0.32,
                    "base_layer_alpha": 0.44,
                    "sigma_min": 0.45,
                    "sigma_max": 3.0,
                    "sigma_scale": 2.1,
                    "sigma_minor_min": 0.30,
                    "coverage_sigma_max": 11.0,
                    "coverage_alpha_fill": 0.50,
                    "coverage_threshold": 0.025,
                    "coverage_target": 0.985,
                    "coverage_footprint_sigma": 3.2,
                    "coverage_densify_boost": 2.4,
                    "reallocate_for_coverage_fraction": 0.10,
                    "densify_weight_error": 0.35,
                    "densify_weight_uncovered": 0.55,
                    "densify_weight_edge": 0.10,
                    "alpha_base": 0.22,
                    "alpha_scale": 0.62,
                    "alpha_min": 0.10,
                    "alpha_max": 0.92,
                    "prune_weight_contribution": 0.35,
                    "prune_weight_residual": 0.65,
                    "prune_weight_uncovered": 0.45,
                    "prune_sample_radius": 1.5,
                    "residual_color_gain": 1.00,
                    "final_fill_fraction": 0.12,
                    "structure_precompute_enabled": False,
                    "structure_smoothing_sigma": 0.0,
                    "structure_anisotropy_clip": 10.0,
                    "structure_min_coherence": 0.12,
                    "structure_local_window": 7,
                    "local_structure_anisotropy_clip": 3.6,
                    "local_structure_min_coherence": 0.14,
                    "local_structure_min_energy": 1e-4,
                    "init_anisotropy_threshold": 1.60,
                    "densify_anisotropy_threshold": 1.35,
                    "densify_anisotropy_edge_threshold": 0.16,
                    "densify_strong_edge_threshold": 0.42,
                    "residual_detail_enabled": True,
                    "residual_detail_reserve_fraction": 0.08,
                    "residual_detail_passes": 1,
                    "residual_detail_percentile": 90.0,
                    "residual_detail_fraction": 0.18,
                    "residual_detail_sigma_min": 0.28,
                    "residual_detail_sigma_max": 1.20,
                    "residual_detail_alpha_min": 0.16,
                    "residual_detail_alpha_max": 0.72,
                    "residual_detail_iters": 8,
                    "residual_detail_edge_weight": 0.30,
                    "residual_detail_color_gain": 0.95,
                },
                "schedule": {
                    "enabled": True,
                    "check_interval": 50,
                    "patience_checks": 3,
                    "decay_ratio": 1.6,
                    "max_decays": 3,
                    "min_delta": 5e-5,
                },
            },
        }
        if profile not in profiles:
            raise ValueError(f"Unknown quality profile: {profile}")
        return profiles[profile]

    def _sha256_file(self, path: str) -> str:
        """Compute SHA256 of input file."""
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                digest.update(chunk)
        return digest.hexdigest()
