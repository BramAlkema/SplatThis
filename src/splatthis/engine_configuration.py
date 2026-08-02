"""Engine settings, budget policy, renderer and loss construction."""

from __future__ import annotations

import logging
import platform
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import torch

from .adaptive_compute import resolve_online_adaptive_config
from .budgets import TIME_BUDGET_ALIASES, TIME_BUDGET_PRESETS
from .engine_state import ConversionEngineState
from .export_common import (
    DEFAULT_PPTX_SPLAT_STYLE,
    PPTX_PAINTER_ORDER_BACK_TO_FRONT,
    PPTX_PROXY_TRAINING_TARGETS,
    PPTX_SOFT_EDGE_ALPHA_SCALE,
    PPTX_SOFT_EDGE_K_SIGMA_SCALE,
    SRGB_TRAINING_TARGETS,
    SVG_BROWSER_COMPAT_RECIPE,
    SVG_PAINTER_ORDER_BACK_TO_FRONT,
    SVG_PALETTE_QUANTIZED_RECIPE,
    SVG_SCRIPTED_MATRIX_RECIPE,
    TRAINING_TARGET_ALIASES,
    _normalize_pptx_painter_order,
    _normalize_svg_export_recipe,
    _normalize_svg_gradient_quality,
    _normalize_svg_painter_order,
)
from .proxies import (
    _PPTXGradientProxyRenderer,
    _PPTXProxyLoss,
    _PPTXSoftEdgeProxyRenderer,
)
from .renderer import L1SSIMLoss, create_renderer, resolve_renderer_backend
from .splat import LAYER_DETAIL, LAYER_EDGE, SPLAT_LAYER_NAMES, GaussianSplat

logger = logging.getLogger(__name__)


class ConversionConfigurationMixin(ConversionEngineState):
    """Normalizes settings and constructs target-aware training primitives."""

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
        refinement_config: Optional[Dict[str, Any]] = None,
        schedule_config: Optional[Dict[str, Any]] = None,
        acceptance_criteria: Optional[Dict[str, float]] = None,
        init_random_ratio: float = 0.2,
        init_gradient_weight: float = 0.7,
        renderer_backend: str = "auto",
        optimizer_backend: str = "torch",
        blend_mode: str = "weighted",
        compositing_space: str = "linear",
        loss_color_space: str = "oklab",
        time_budget: Optional[str] = None,
        apple_silicon_splat_cap: Optional[int] = 2000,
        layered_saliency: bool = False,
        pptx_splat_style: str = DEFAULT_PPTX_SPLAT_STYLE,
        pptx_painter_order: str = PPTX_PAINTER_ORDER_BACK_TO_FRONT,
    ):
        self.requested_max_splats = int(max_splats)
        self.max_splats = int(max_splats)
        # Host-memory safety valve, in percent. Above this the optimizer halves
        # the population rather than risk being OOM-killed. Set to None to
        # disable it and make a seeded run independent of machine load; the
        # decision is recorded in the run manifest either way.
        self.memory_guard_percent: Optional[float] = 85.0
        self.k_sigma = k_sigma
        profile_defaults = self._get_profile_defaults(quality_profile)
        self.stages = list(
            stages or profile_defaults.get("stages", [200, 150, 100, 50])
        )
        self.target_size = target_size
        self.gradient_method = gradient_method
        self._configure_backend(
            device=device,
            renderer_backend=renderer_backend,
            optimizer_backend=optimizer_backend,
        )
        self.blend_mode = str(blend_mode).strip().lower()
        # Compositing space for the optimizer's forward render. "linear" is
        # physically correct; "srgb" matches overlapping browser shapes.
        self.compositing_space = str(compositing_space).strip().lower()
        self.loss_color_space = str(loss_color_space).strip().lower()
        self.seed = seed
        self.quality_profile = quality_profile
        self.resolution_scale = float(max(1.0, resolution_scale))
        self.init_random_ratio = float(np.clip(init_random_ratio, 0.0, 1.0))
        self.init_gradient_weight = float(np.clip(init_gradient_weight, 0.0, 1.0))
        self.time_budget = self._normalize_time_budget(time_budget)
        self.layered_saliency = bool(layered_saliency)
        self.pptx_splat_style = str(pptx_splat_style).strip().lower().replace("_", "-")
        self.pptx_painter_order = _normalize_pptx_painter_order(pptx_painter_order)
        self.time_budget_plan: Optional[Dict[str, Any]] = None
        self._time_budget_deadline: Optional[float] = None
        self._platform_splat_cap: Optional[Dict[str, Any]] = None
        self.apple_silicon_splat_cap = (
            None
            if apple_silicon_splat_cap is None or int(apple_silicon_splat_cap) <= 0
            else int(apple_silicon_splat_cap)
        )

        self.loss_weights = loss_weights or profile_defaults["loss_weights"].copy()
        self.learning_rates = profile_defaults["learning_rates"].copy()
        if learning_rates:
            self.learning_rates.update(learning_rates)
        self.learning_rates = self._normalize_learning_rates(self.learning_rates)

        self._configure_refinement(profile_defaults, refinement_config)
        self.schedule_config = profile_defaults["schedule"].copy()
        if schedule_config:
            self.schedule_config.update(schedule_config)
        # Acceptance is a *quality* gate. Wall-clock deliberately is not part of
        # it by default: the shipped presets train for minutes (the SVG corpus
        # median at 2k splats is 4.2), so a 60-second ceiling failed correct,
        # in-spec runs -- 8 of 70 stored manifests failed on runtime alone,
        # including full-quality runs reaching SSIM 0.91. A gate that is false
        # for good work teaches everyone to ignore `acceptance.pass`.
        #
        # Runtime is still measured and recorded under `acceptance.measured`,
        # and `max_runtime_sec` remains honoured when a caller passes it
        # explicitly or when --time-budget sets one. It simply no longer decides
        # whether a conversion was acceptable.
        self.acceptance_criteria = acceptance_criteria or {
            "min_psnr": 15.0,
            "min_ssim": 0.50,
            "min_psnr_srgb": 12.0,
            "min_ssim_srgb": 0.50,
            "max_splats": float(self.max_splats),
        }

        self._initialize_runtime_state()
        self._apply_platform_splat_cap()
        self._log_configuration(device)

    def _configure_backend(
        self,
        *,
        device: str,
        renderer_backend: str,
        optimizer_backend: str,
    ) -> None:
        """Resolve compute backends and fail over before a long run starts."""
        self.device = torch.device(device)
        self.renderer_backend = renderer_backend
        self.optimizer_backend = self._normalize_optimizer_backend(optimizer_backend)
        if self.optimizer_backend == "mlx":
            from .mlx_runtime import is_mlx_available, is_mlx_imported

            if not is_mlx_available():
                if is_mlx_imported():
                    reason = (
                        "the installed MLX runtime has no Metal device; this "
                        "usually means the session is headless, sandboxed, or "
                        "virtualized"
                    )
                else:
                    reason = "the optional MLX package is not installed"
                # Fail over before a long run starts. Torch is a required
                # dependency and is the supported cross-platform backend.
                logger.warning(
                    "Optimizer backend 'mlx' requested but %s; falling back to "
                    "'torch'. Use an Apple-Silicon Metal session for MLX.",
                    reason,
                )
                self.optimizer_backend = "torch"
        self.resolved_renderer_backend = resolve_renderer_backend(
            renderer_backend,
            self.device,
        )

    def _configure_refinement(
        self,
        profile_defaults: Dict[str, Any],
        refinement_config: Optional[Dict[str, Any]],
    ) -> None:
        """Resolve export, fidelity and MLX refinement policy."""
        self.refinement_config = profile_defaults["refinement"].copy()
        self._training_export_target_explicit = bool(
            refinement_config and "training_export_target" in refinement_config
        )
        if refinement_config:
            self.refinement_config.update(refinement_config)
        self.adaptive_compute_config = resolve_online_adaptive_config(
            self.refinement_config
        )
        self.region_weighting_enabled = bool(
            self.refinement_config.get("region_weighting_enabled", False)
        )
        # Canonicalize aliases up front ("browser" -> "browser-compatible",
        # "palette" -> "palette-quantized", ...) and fail fast on invalid
        # recipes here instead of erroring after training at save time.
        self.svg_export_recipe = _normalize_svg_export_recipe(
            self.refinement_config.get("svg_export_recipe", "standard")
        )
        self.svg_gradient_quality = _normalize_svg_gradient_quality(
            self.refinement_config.get("svg_gradient_quality", "standard")
        )
        self.svg_painter_order = _normalize_svg_painter_order(
            self.refinement_config.get(
                "svg_painter_order", SVG_PAINTER_ORDER_BACK_TO_FRONT
            )
        )
        self.svg_compositor_gate = bool(
            self.refinement_config.get(
                "svg_compositor_gate", self.quality_profile == "max-fidelity"
            )
        )
        self.svg_optimize = bool(self.refinement_config.get("svg_optimize", False))
        self.svg_optimize_precision = int(
            self.refinement_config.get("svg_optimize_precision", 2)
        )
        # ADR-003 fidelity stage: resolve (and validate) the mode up front.
        from .fidelity import resolve_fidelity_config

        self.fidelity_config = resolve_fidelity_config(self.refinement_config)
        self.training_export_target = self._normalize_training_export_target(
            self.refinement_config.get("training_export_target", "pixel-runtime")
        )
        self.mlx_loss = (
            str(self.refinement_config.get("mlx_loss", "oklab-l1-ssim"))
            .strip()
            .lower()
            .replace("_", "-")
        )
        self.mlx_tile_plan = (
            str(self.refinement_config.get("mlx_tile_plan", "static"))
            .strip()
            .lower()
            .replace("_", "-")
        )
        if self.mlx_tile_plan not in {"static", "periodic"}:
            raise ValueError(f"Unsupported MLX tile plan: {self.mlx_tile_plan}")
        self.mlx_tile_plan_rebuild_interval = int(
            max(1, self.refinement_config.get("mlx_tile_plan_rebuild_interval", 10))
        )
        self.mlx_trainable_groups = self._normalize_mlx_trainable_groups(
            self.refinement_config.get("mlx_trainable_groups", "color,alpha")
        )
        if self.optimizer_backend == "mlx" and self.mlx_tile_plan == "static":
            moving = {"position", "scale", "theta"}.intersection(
                self.mlx_trainable_groups
            )
            if moving:
                raise ValueError(
                    "optimizer_backend='mlx' currently supports only color/alpha with static tile plans; "
                    f"got moving group(s): {', '.join(sorted(moving))}"
                )
        self.mlx_spatial_weighting_enabled = self._use_mlx_spatial_weights()

    def _initialize_runtime_state(self) -> None:
        """Initialize image-sized state that is replaced for every run."""
        self._image_width = 1000
        self._image_height = 1000
        self._background_linear_rgb = np.zeros(3, dtype=np.float32)
        self._region_weight_map: Optional[npt.NDArray[Any]] = None
        self._region_saliency_map: Optional[npt.NDArray[Any]] = None
        self._region_detail_priority_map: Optional[npt.NDArray[Any]] = None
        self._region_background_penalty_map: Optional[npt.NDArray[Any]] = None
        self._region_foreground_mask: Optional[npt.NDArray[Any]] = None
        self._region_background_safe_mask: Optional[npt.NDArray[Any]] = None
        self._region_edge_band_mask: Optional[npt.NDArray[Any]] = None

    def _apply_platform_splat_cap(self) -> None:
        """Apply the explicit Apple-Silicon safety cap, if configured."""
        if (
            "arm" in platform.processor().lower()
            and self.apple_silicon_splat_cap is not None
        ):
            before_cap = self.max_splats
            self.max_splats = min(self.max_splats, self.apple_silicon_splat_cap)
            self._platform_splat_cap = {
                "platform": "apple-silicon",
                "cap": int(self.apple_silicon_splat_cap),
                "requested_max_splats": int(before_cap),
                "applied": bool(before_cap != self.max_splats),
            }
            if before_cap != self.max_splats:
                # WARNING, not INFO: the default CLI log level would hide an
                # actual clamp of a user-requested count.
                logger.warning(
                    "Apple Silicon splat cap: limiting max_splats from %s to %s "
                    "(pass --no-apple-silicon-splat-cap or a higher "
                    "--apple-silicon-splat-cap to override)",
                    before_cap,
                    self.max_splats,
                )
            else:
                logger.info(
                    "Apple Silicon detected - max_splats %s within cap",
                    self.max_splats,
                )

    def _log_configuration(self, device: str) -> None:
        """Log the resolved configuration once construction is complete."""
        logger.info(
            "Initialized PNG2SVG converter: max_splats=%s, stages=%s, device=%s, backend=%s->%s, optimizer=%s, blend=%s, seed=%s, profile=%s, resolution_scale=%.2f, init_random_ratio=%.2f, time_budget=%s, layered_saliency=%s",
            self.max_splats,
            self.stages,
            device,
            self.renderer_backend,
            self.resolved_renderer_backend,
            self.optimizer_backend,
            self.blend_mode,
            self.seed,
            self.quality_profile,
            self.resolution_scale,
            self.init_random_ratio,
            self.time_budget,
            self.layered_saliency,
        )

    @staticmethod
    def _normalize_learning_rates(learning_rates: Dict[str, float]) -> Dict[str, float]:
        """Map the legacy 'covariance' LR key onto scale+theta; reject typos.

        Profiles historically set {"covariance": ...} from the pre-refactor
        parameter layout; build_optimizer/mlx read only position/scale/theta/
        color/alpha, so the key was silently dead and scale/theta trained at
        defaults regardless of profile intent.
        """
        valid = {"position", "scale", "theta", "color", "alpha"}
        normalized: Dict[str, float] = {}
        for key, value in learning_rates.items():
            if key == "covariance":
                logger.warning(
                    "learning_rates key 'covariance' is deprecated; applying "
                    "%.5f to both 'scale' and 'theta' (set them explicitly).",
                    float(value),
                )
                normalized.setdefault("scale", float(value))
                normalized.setdefault("theta", float(value))
            elif key in valid:
                normalized[key] = float(value)
            else:
                raise ValueError(
                    f"Unknown learning_rates key: {key!r} "
                    f"(expected one of {sorted(valid)} or legacy 'covariance')"
                )
        return normalized

    @staticmethod
    def _normalize_optimizer_backend(value: Any) -> str:
        normalized = str(value).strip().lower().replace("_", "-")
        if normalized in {"", "torch", "pytorch"}:
            return "torch"
        if normalized in {"mlx", "mlx-batched"}:
            return "mlx"
        raise ValueError(f"Unsupported optimizer backend: {value}")

    @staticmethod
    def _normalize_mlx_trainable_groups(value: Any) -> Tuple[str, ...]:
        if isinstance(value, str):
            raw_items = [part.strip() for part in value.split(",")]
        elif isinstance(value, (list, tuple)):
            raw_items = [str(part).strip() for part in value]
        else:
            raise ValueError("mlx_trainable_groups must be a comma string or sequence")
        groups = tuple(item.replace("-", "_") for item in raw_items if item)
        valid = {"position", "scale", "theta", "color", "alpha"}
        invalid = [item for item in groups if item not in valid]
        if invalid:
            raise ValueError(
                f"Unsupported MLX trainable group(s): {', '.join(invalid)}"
            )
        return groups or ("color", "alpha")

    def _use_mlx_spatial_weights(self) -> bool:
        # Every MLX loss profile applies spatial weights to its L1 term when
        # given, so honor the profile's region_weighting_enabled switch
        # instead of gating on the "weighted-" loss-name prefix (which
        # silently dropped the region map for the default l1-ssim losses).
        return self.optimizer_backend == "mlx" and (
            self.region_weighting_enabled or self.mlx_loss.startswith("weighted")
        )

    # SVG recipes whose emitters consume region-guidance masks (safe-background
    # tests, alpha caps, precompensation). Recipe names here are canonical —
    # svg_export_recipe is normalized in __init__.
    _GUIDANCE_SVG_RECIPES = frozenset(
        {
            SVG_BROWSER_COMPAT_RECIPE,
            SVG_SCRIPTED_MATRIX_RECIPE,
            SVG_PALETTE_QUANTIZED_RECIPE,
        }
    )

    def _deployed_compositing_space(self) -> str:
        """Compositing space of the DEPLOYED artifact the model was trained for.

        Mirrors the forcing in _create_training_renderer/_optimize_stage_mlx:
        SVG/PPTX-softedge targets train and deploy in sRGB compositing, so
        validation/preview renders must composite there too or they misreport
        deployed fidelity.
        """
        if self.training_export_target in SRGB_TRAINING_TARGETS:
            return "srgb"
        return self.compositing_space

    def _needs_region_guidance(self) -> bool:
        """Single source of truth for whether a run computes region guidance."""
        return bool(
            self.time_budget is not None
            or self.region_weighting_enabled
            or self.layered_saliency
            or self.training_export_target in PPTX_PROXY_TRAINING_TARGETS
            or self._use_mlx_spatial_weights()
            or int(max(0, self.refinement_config.get("svg_proxy_postfit_iters", 0))) > 0
            or int(max(0, self.refinement_config.get("pptx_proxy_postfit_iters", 0)))
            > 0
            or self.svg_export_recipe in self._GUIDANCE_SVG_RECIPES
        )

    def _normalize_time_budget(self, time_budget: Optional[str]) -> Optional[str]:
        """Normalize time-budget labels accepted by the CLI/API."""
        if time_budget is None:
            return None
        key = str(time_budget).strip().lower().replace("_", "-")
        key = TIME_BUDGET_ALIASES.get(key, key)
        if key not in TIME_BUDGET_PRESETS:
            valid = ", ".join(sorted(TIME_BUDGET_PRESETS))
            raise ValueError(
                f"Unknown time budget: {time_budget!r}. Expected one of: {valid}"
            )
        return key

    def _apply_time_budget_plan(
        self,
        width: int,
        height: int,
        guidance: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Resolve a budget preset into stage schedule, splat cap, and residual settings."""
        if self.time_budget is None:
            return {}
        preset = TIME_BUDGET_PRESETS[self.time_budget]
        area = max(1, int(width) * int(height))
        megapixels = area / 1_000_000.0
        saliency_multiplier, saliency_summary = self._estimate_saliency_multiplier(
            width=width,
            height=height,
            guidance=guidance,
        )
        raw_splats = int(
            round(
                megapixels * float(preset["splats_per_megapixel"]) * saliency_multiplier
            )
        )
        min_splats = int(preset["min_splats"])
        preset_cap = preset.get("max_splats")
        requested_ceiling = max(1, int(self.max_splats))
        budget_ceiling = requested_ceiling
        if preset_cap is not None:
            budget_ceiling = min(budget_ceiling, int(preset_cap))
        selected_splats = int(max(1, min(budget_ceiling, max(min_splats, raw_splats))))

        self.max_splats = selected_splats
        self.stages = [int(v) for v in preset["stages"]]
        refinement_overrides = dict(preset.get("refinement_overrides") or {})
        self.refinement_config.update(refinement_overrides)
        for key in (
            "residual_detail_enabled",
            "residual_detail_reserve_fraction",
            "residual_detail_passes",
            "residual_detail_iters",
        ):
            self.refinement_config[key] = preset[key]

        initial_count = self._initial_splat_count()
        base_fraction = float(
            np.clip(self.refinement_config.get("base_layer_fraction", 0.35), 0.10, 0.80)
        )
        base_count = max(4, int(round(initial_count * base_fraction)))
        native_base_sigma = float(np.sqrt(area / max(base_count, 1)) * 0.85)
        coverage_multiplier = float(
            max(0.0, preset.get("coverage_sigma_cap_multiplier", 0.0))
        )
        dynamic_coverage_sigma_max = max(
            float(self.refinement_config.get("coverage_sigma_max", 0.0)),
            native_base_sigma * coverage_multiplier,
        )
        self.refinement_config["coverage_sigma_max"] = float(dynamic_coverage_sigma_max)

        return {
            "preset": self.time_budget,
            "label": str(preset["label"]),
            "target_seconds": float(preset["target_seconds"]),
            "image_pixels": int(area),
            "image_megapixels": float(megapixels),
            "requested_ceiling": int(requested_ceiling),
            "preset_ceiling": None if preset_cap is None else int(preset_cap),
            "selected_max_splats": int(selected_splats),
            "raw_recommended_splats": int(raw_splats),
            "splats_per_megapixel": float(preset["splats_per_megapixel"]),
            "saliency_multiplier": float(saliency_multiplier),
            "saliency_summary": saliency_summary,
            "stages": list(self.stages),
            "initial_splat_estimate": int(initial_count),
            "base_layer_estimate": int(base_count),
            "base_layer_fraction": float(base_fraction),
            "dynamic_coverage_sigma_max": float(dynamic_coverage_sigma_max),
            "native_base_sigma": float(native_base_sigma),
            "refinement_overrides": refinement_overrides,
            "residual_detail_enabled": bool(
                self.refinement_config["residual_detail_enabled"]
            ),
            "residual_detail_iters": int(
                self.refinement_config["residual_detail_iters"]
            ),
        }

    def _estimate_saliency_multiplier(
        self,
        width: int,
        height: int,
        guidance: Optional[Dict[str, Any]],
    ) -> Tuple[float, Dict[str, float]]:
        """Estimate how much content density should push the budget up or down."""
        area = float(max(1, int(width) * int(height)))
        summary = dict((guidance or {}).get("summary") or {})

        def ratio(key: str) -> float:
            return float(np.clip(float(summary.get(key, 0.0)) / area, 0.0, 1.0))

        foreground_ratio = ratio("foreground_pixels")
        edge_ratio = ratio("edge_band_pixels")
        background_ratio = ratio("background_safe_pixels")
        saliency_mean = float(np.clip(summary.get("saliency_mean", 0.0), 0.0, 1.0))
        saliency_p95 = float(
            np.clip(summary.get("saliency_p95", saliency_mean), 0.0, 1.0)
        )
        mean_weight = 0.70
        weight_map = (guidance or {}).get("weight_map")
        if isinstance(weight_map, np.ndarray) and weight_map.size:
            mean_weight = float(np.clip(np.mean(weight_map), 0.0, 10.0))

        raw_multiplier = (
            0.72
            + 0.85 * foreground_ratio
            + 0.65 * edge_ratio
            - 0.30 * background_ratio
            + 0.20 * (mean_weight - 0.70)
            + 0.25 * saliency_mean
            + 0.15 * saliency_p95
        )
        multiplier = float(np.clip(raw_multiplier, 0.55, 2.10))
        return multiplier, {
            "foreground_ratio": float(foreground_ratio),
            "edge_band_ratio": float(edge_ratio),
            "background_safe_ratio": float(background_ratio),
            "saliency_mean": float(saliency_mean),
            "saliency_p95": float(saliency_p95),
            "mean_region_weight": float(mean_weight),
            "raw_multiplier": float(raw_multiplier),
        }

    def _initial_splat_count(self) -> int:
        """Resolve the initial population size before staged densification.

        The historical fixed cap of 1200 splats kept interactive runs stable,
        but it also made larger native-photo budgets start from the same blurry
        basis as smaller runs. Long-budget and ROI workflows can raise the cap
        through refinement_config while legacy profiles keep the old default.
        """
        if self.max_splats <= 0:
            return 0

        fraction = float(
            np.clip(
                self.refinement_config.get("initial_splat_fraction", 0.50), 0.05, 1.0
            )
        )
        requested = max(1, int(round(float(self.max_splats) * fraction)))
        cap_raw = self.refinement_config.get("initial_splat_cap", 1200)
        try:
            cap = int(cap_raw)
        except (TypeError, ValueError):
            cap = 1200
        if cap > 0:
            requested = min(requested, cap)
        return int(min(self.max_splats, requested))

    def _time_budget_seconds_remaining(self) -> Optional[float]:
        """Return training-budget seconds remaining, if a budget is active."""
        if self._time_budget_deadline is None:
            return None
        return float(self._time_budget_deadline - time.perf_counter())

    def _time_budget_exhausted(self) -> bool:
        """Whether the active training budget has been exhausted."""
        remaining = self._time_budget_seconds_remaining()
        return remaining is not None and remaining <= 0.0

    def _assign_splat_layer(
        self,
        splat: GaussianSplat,
        layer: int,
        local_importance: float,
    ) -> None:
        """Assign layered draw metadata while preserving legacy importance when disabled."""
        if self.layered_saliency:
            importance = float(np.clip(local_importance, 0.0, 0.999))
            splat.layer = int(layer)
            splat.importance = float(int(layer) + importance)
        else:
            splat.importance = float(np.clip(local_importance, 0.0, 1.0))

    def _saliency_at(self, x: int, y: int) -> float:
        """Return continuous saliency at an image-space pixel."""
        saliency = self._sampling_prior_map()
        if saliency is None:
            return 0.0
        if saliency.ndim != 2 or saliency.size == 0:
            return 0.0
        height, width = saliency.shape
        xx = int(np.clip(x, 0, max(width - 1, 0)))
        yy = int(np.clip(y, 0, max(height - 1, 0)))
        return float(np.clip(saliency[yy, xx], 0.0, 1.0))

    def _sampling_prior_map(self) -> Optional[npt.NDArray[Any]]:
        """Return the saliency-like map used for sampling and layer importance."""
        if (
            bool(
                self.refinement_config.get(
                    "background_suppressed_saliency_use_for_sampling", False
                )
            )
            and self._region_detail_priority_map is not None
            and self._region_detail_priority_map.ndim == 2
        ):
            return self._region_detail_priority_map
        return self._region_saliency_map

    def _saliency_layer_for_pixel(
        self, x: int, y: int, default_layer: int
    ) -> Tuple[int, float]:
        """Resolve layered draw band and local importance from region guidance."""
        saliency = self._saliency_at(x, y)
        layer = int(default_layer)
        if self.layered_saliency:
            mask_shape = None
            if (
                self._region_saliency_map is not None
                and self._region_saliency_map.ndim == 2
            ):
                mask_shape = self._region_saliency_map.shape
            elif (
                self._region_edge_band_mask is not None
                and self._region_edge_band_mask.ndim == 2
            ):
                mask_shape = self._region_edge_band_mask.shape
            elif (
                self._region_foreground_mask is not None
                and self._region_foreground_mask.ndim == 2
            ):
                mask_shape = self._region_foreground_mask.shape

            if mask_shape is not None:
                height, width = mask_shape
                xx = int(np.clip(x, 0, max(width - 1, 0)))
                yy = int(np.clip(y, 0, max(height - 1, 0)))
            else:
                xx = int(x)
                yy = int(y)

            if (
                self._region_edge_band_mask is not None
                and mask_shape is not None
                and self._region_edge_band_mask.shape == mask_shape
                and bool(self._region_edge_band_mask[yy, xx])
            ):
                layer = LAYER_EDGE
            elif (
                self._region_foreground_mask is not None
                and mask_shape is not None
                and self._region_foreground_mask.shape == mask_shape
                and bool(self._region_foreground_mask[yy, xx])
            ):
                layer = max(layer, LAYER_DETAIL)
        local_importance = float(np.clip(0.10 + 0.89 * saliency, 0.0, 0.999))
        return layer, local_importance

    def _apply_saliency_sampling_bias(
        self, score_map: npt.NDArray[Any], strength: float
    ) -> npt.NDArray[Any]:
        """Multiply a score map by a continuous saliency prior when available."""
        score = np.asarray(score_map, dtype=np.float32)
        prior_map = self._sampling_prior_map()
        if (
            prior_map is None
            or prior_map.shape != score.shape
            or float(strength) <= 0.0
        ):
            return score
        saliency = np.asarray(prior_map, dtype=np.float32)
        gamma = float(
            max(0.10, self.refinement_config.get("saliency_sampling_gamma", 0.75))
        )
        prior = np.power(np.clip(saliency, 0.0, 1.0), gamma).astype(np.float32)
        biased = np.clip(score, 0.0, None) * (1.0 + float(strength) * prior)
        additive = float(
            max(0.0, self.refinement_config.get("saliency_sampling_additive", 0.0))
        )
        if additive > 0.0:
            biased = biased + additive * float(np.max(score)) * prior
        biased = np.clip(biased, 0.0, None).astype(np.float32)
        if float(np.max(biased) - np.min(biased)) <= 1e-8:
            return biased
        return self._normalize_map(biased)

    def _copy_splat_layers(
        self,
        source: List[GaussianSplat],
        optimized: List[GaussianSplat],
    ) -> List[GaussianSplat]:
        """Restore non-optimized layer metadata after tensor optimizer round-trips."""
        if not self.layered_saliency:
            return optimized
        for src, dst in zip(source, optimized):
            dst.layer = src.layer
            dst.importance = src.importance
        return optimized

    @staticmethod
    def _normalize_training_export_target(value: Any) -> str:
        normalized = str(value).strip().lower().replace("_", "-")
        try:
            return TRAINING_TARGET_ALIASES[normalized]
        except KeyError:
            raise ValueError(f"Unsupported training export target: {value}") from None

    def _use_pptx_proxy_training(self) -> bool:
        return self.training_export_target == "pptx-softedge"

    def _create_training_renderer(self, width: int, height: int) -> torch.nn.Module:
        compositing_space = (
            "srgb"
            if self.training_export_target in SRGB_TRAINING_TARGETS
            else self.compositing_space
        )
        blend_mode = self.blend_mode
        if (
            self.training_export_target in SRGB_TRAINING_TARGETS
            and blend_mode != "alpha-over"
        ):
            # The SVG/PPTX emitters model per-splat source-over opacity
            # (1 - exp(-a*G)); training a "weighted" forward model against
            # them optimizes an objective no exporter can reproduce. Force
            # alpha-over exactly as compositing_space is forced above.
            logger.info(
                "training_export_target=%s: forcing blend_mode='alpha-over' "
                "(was %r) to match export compositing semantics.",
                self.training_export_target,
                blend_mode,
            )
            blend_mode = "alpha-over"
        tile_size = int(
            np.clip(self.refinement_config.get("renderer_tile_size", 16), 4, 128)
        )
        tile_bin_rebuild_interval = int(
            max(1, self.refinement_config.get("renderer_tile_bin_rebuild_interval", 1))
        )
        tile_bin_padding = float(
            max(0.0, self.refinement_config.get("renderer_tile_bin_padding", 0.0))
        )
        batch_tile_count = int(
            max(1, self.refinement_config.get("renderer_batch_tile_count", 32))
        )
        max_active_raw = self.refinement_config.get(
            "renderer_max_active_splats_per_tile"
        )
        max_active_splats_per_tile = (
            None if max_active_raw in (None, "", 0) else int(max_active_raw)
        )
        base_renderer = create_renderer(
            backend=self.renderer_backend,
            width=width,
            height=height,
            device=self.device,
            tile_size=tile_size,
            blend_mode=blend_mode,
            background_color=self._background_linear_rgb,
            compositing_space=compositing_space,
            tile_bin_rebuild_interval=tile_bin_rebuild_interval,
            tile_bin_padding=tile_bin_padding,
            batch_tile_count=batch_tile_count,
            max_active_splats_per_tile=max_active_splats_per_tile,
        )
        if self.training_export_target == "pptx-gradient":
            return _PPTXGradientProxyRenderer(base_renderer=base_renderer).to(
                self.device
            )
        if self.training_export_target != "pptx-softedge":
            return base_renderer
        return _PPTXSoftEdgeProxyRenderer(
            base_renderer=base_renderer,
            alpha_scale=float(
                self.refinement_config.get(
                    "pptx_proxy_train_alpha_scale", PPTX_SOFT_EDGE_ALPHA_SCALE
                )
            ),
            sigma_scale=float(
                self.refinement_config.get(
                    "pptx_proxy_train_sigma_scale", PPTX_SOFT_EDGE_K_SIGMA_SCALE
                )
            ),
        ).to(self.device)

    @staticmethod
    def _base_renderer(renderer: torch.nn.Module) -> torch.nn.Module:
        """Unwrap proxy renderers for runtime cache diagnostics."""
        current = renderer
        while hasattr(current, "base_renderer"):
            current = getattr(current, "base_renderer")
        return current

    def _renderer_cache_stats(self, renderer: torch.nn.Module) -> Dict[str, Any]:
        base = self._base_renderer(renderer)
        if hasattr(base, "tile_bin_cache_stats"):
            return dict(base.tile_bin_cache_stats())
        return {}

    def _clear_renderer_cache(self, renderer: torch.nn.Module) -> None:
        base = self._base_renderer(renderer)
        if hasattr(base, "clear_tile_bin_cache"):
            base.clear_tile_bin_cache()

    def _create_training_loss(
        self, target: torch.Tensor, width: int, height: int
    ) -> torch.nn.Module:
        spatial_weights = self._loss_weight_tensor(width=width, height=height)
        base_loss = L1SSIMLoss(
            **self.loss_weights,
            color_space=self.loss_color_space,
            spatial_weight_map=spatial_weights,
        ).to(self.device)
        if not self._use_pptx_proxy_training():
            return base_loss
        return _PPTXProxyLoss(
            target_linear_rgb=target,
            base_loss=base_loss,
            spatial_weight_map=spatial_weights,
            contrast_weight=float(
                self.refinement_config.get("pptx_proxy_train_contrast_weight", 0.35)
            ),
            saturation_weight=float(
                self.refinement_config.get("pptx_proxy_train_saturation_weight", 0.18)
            ),
            gradient_weight=float(
                self.refinement_config.get("pptx_proxy_train_gradient_weight", 0.10)
            ),
        ).to(self.device)

    def _layer_summary(self, splats: List[GaussianSplat]) -> Dict[str, Any]:
        """Summarize splat layer counts for manifests and debugging."""
        counts: Dict[int, int] = {}
        unassigned = 0
        for splat in splats:
            raw = splat.to_raw_splat()
            if raw.layer is None:
                unassigned += 1
                continue
            layer = int(raw.layer)
            counts[layer] = counts.get(layer, 0) + 1

        layers = [
            {
                "id": layer,
                "name": SPLAT_LAYER_NAMES.get(layer, f"layer-{layer}"),
                "count": count,
            }
            for layer, count in sorted(counts.items())
        ]
        return {
            "enabled": bool(self.layered_saliency),
            "layers": layers,
            "unassigned": int(unassigned),
        }
