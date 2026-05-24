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
    compute_gradient_magnitude,
    compute_structure_field,
    estimate_local_color,
    init_seeds_content_adaptive,
    poisson_disk_sampling,
)
from .io import (
    PPTX_GRADIENT_ALPHA_SCALE,
    PPTX_SOFT_EDGE_ALPHA_SCALE,
    PPTX_SOFT_EDGE_K_SIGMA_SCALE,
    SVG_BACKGROUND_ALPHA_CAP,
    compute_quality_metrics,
    evaluate_svg_export_quality,
    generate_canvas_html,
    generate_drawingml_slide_content,
    load_png,
    render_splats_preview_png,
    save_drawingml,
    save_pptx_with_splats,
    save_side_by_side_html,
    save_splats_json,
    save_svg,
    validate_export_roundtrip,
)
from .mlx_losses import MlxLossConfig
from .mlx_stage import MlxRendererConfig, MlxStageConfig, optimize_stage_mlx
from .optimizer import SplatParams, build_optimizer
from .renderer import (
    L1SSIMLoss,
    create_renderer,
    render_splats_numpy,
    resolve_renderer_backend,
    splats_to_tensor,
    tensor_to_splats,
    torch_linear_rgb_to_oklab,
    torch_linear_to_srgb,
)
from .splat import (
    LAYER_BASE,
    LAYER_DETAIL,
    LAYER_EDGE,
    LAYER_MASS,
    SPLAT_LAYER_NAMES,
    GaussianSplat,
    create_anisotropic_splat,
    create_isotropic_splat,
)

logger = logging.getLogger(__name__)


class _PPTXSoftEdgeProxyRenderer(torch.nn.Module):
    """Approximate native PPTX soft-edge ellipses with a differentiable renderer."""

    def __init__(
        self,
        base_renderer: torch.nn.Module,
        alpha_scale: float = PPTX_SOFT_EDGE_ALPHA_SCALE,
        sigma_scale: float = PPTX_SOFT_EDGE_K_SIGMA_SCALE,
    ):
        super().__init__()
        self.base_renderer = base_renderer
        self.alpha_scale = float(np.clip(alpha_scale, 1e-4, 1.0))
        self.sigma_scale = float(np.clip(sigma_scale, 0.25, 3.0))

    def forward(self, splats_tensor: torch.Tensor) -> torch.Tensor:
        scaled_sigma = torch.clamp(splats_tensor[:, 2:4] * self.sigma_scale, min=1e-4)
        raw_alpha = torch.clamp(splats_tensor[:, 9], 0.0, 1.0)
        center_opacity = torch.clamp(
            (1.0 - torch.exp(-raw_alpha)) * self.alpha_scale, 0.0, 1.0 - 1e-5
        )
        effective_alpha = -torch.log1p(-center_opacity)
        fitted = torch.cat(
            [
                splats_tensor[:, 0:2],
                scaled_sigma,
                splats_tensor[:, 4:9],
                effective_alpha.unsqueeze(-1),
                splats_tensor[:, 10:11],
            ],
            dim=-1,
        )
        return self.base_renderer(fitted)


class _PPTXProxyLoss(torch.nn.Module):
    """Perceptual loss for PPTX-soft-edge proxy training."""

    def __init__(
        self,
        target_linear_rgb: torch.Tensor,
        base_loss: L1SSIMLoss,
        spatial_weight_map: Optional[torch.Tensor] = None,
        contrast_weight: float = 0.35,
        saturation_weight: float = 0.18,
        gradient_weight: float = 0.10,
    ):
        super().__init__()
        self.base_loss = base_loss
        self.contrast_weight = float(max(0.0, contrast_weight))
        self.saturation_weight = float(max(0.0, saturation_weight))
        self.gradient_weight = float(max(0.0, gradient_weight))
        weights = (
            torch.ones(
                target_linear_rgb.shape[:2],
                dtype=target_linear_rgb.dtype,
                device=target_linear_rgb.device,
            )
            if spatial_weight_map is None
            else spatial_weight_map.to(
                target_linear_rgb.device, dtype=target_linear_rgb.dtype
            )
        )
        weights = weights / torch.clamp(torch.mean(weights), min=1e-6)
        target_srgb = torch_linear_to_srgb(target_linear_rgb)
        target_luma = self._srgb_luminance(target_srgb)
        target_sat = self._srgb_saturation(target_srgb)
        self.register_buffer("weights", weights)
        self.register_buffer("target_luma", target_luma)
        self.register_buffer("target_sat", target_sat)
        self.register_buffer(
            "target_luma_std", self._weighted_std(target_luma, weights).detach()
        )
        self.register_buffer(
            "target_sat_mean", self._weighted_mean(target_sat, weights).detach()
        )
        self.register_buffer(
            "target_sat_std", self._weighted_std(target_sat, weights).detach()
        )

    def forward(self, rendered: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self.base_loss(rendered, target)
        if (
            self.contrast_weight <= 0.0
            and self.saturation_weight <= 0.0
            and self.gradient_weight <= 0.0
        ):
            return loss

        rendered_srgb = torch_linear_to_srgb(rendered)
        rendered_luma = self._srgb_luminance(rendered_srgb)
        rendered_sat = self._srgb_saturation(rendered_srgb)
        contrast_loss = torch.abs(
            self._weighted_std(rendered_luma, self.weights) - self.target_luma_std
        )
        saturation_loss = torch.abs(
            self._weighted_mean(rendered_sat, self.weights) - self.target_sat_mean
        ) + 0.5 * torch.abs(
            self._weighted_std(rendered_sat, self.weights) - self.target_sat_std
        )
        gradient_loss = self._luminance_gradient_l1(rendered_luma, self.target_luma)
        return (
            loss
            + self.contrast_weight * contrast_loss
            + self.saturation_weight * saturation_loss
            + self.gradient_weight * gradient_loss
        )

    @staticmethod
    def _srgb_luminance(values: torch.Tensor) -> torch.Tensor:
        return (
            0.2126 * values[..., 0] + 0.7152 * values[..., 1] + 0.0722 * values[..., 2]
        )

    @staticmethod
    def _srgb_saturation(values: torch.Tensor) -> torch.Tensor:
        maxc = torch.max(values, dim=-1).values
        minc = torch.min(values, dim=-1).values
        return torch.where(
            maxc > 1e-6,
            (maxc - minc) / torch.clamp(maxc, min=1e-6),
            torch.zeros_like(maxc),
        )

    @staticmethod
    def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.sum(values * weights) / torch.clamp(torch.sum(weights), min=1e-8)

    @classmethod
    def _weighted_std(cls, values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        mean = cls._weighted_mean(values, weights)
        variance = cls._weighted_mean((values - mean).pow(2), weights)
        return torch.sqrt(torch.clamp(variance, min=1e-8))

    def _luminance_gradient_l1(
        self, rendered_luma: torch.Tensor, target_luma: torch.Tensor
    ) -> torch.Tensor:
        dx = torch.abs(
            (rendered_luma[:, 1:] - rendered_luma[:, :-1])
            - (target_luma[:, 1:] - target_luma[:, :-1])
        )
        dy = torch.abs(
            (rendered_luma[1:, :] - rendered_luma[:-1, :])
            - (target_luma[1:, :] - target_luma[:-1, :])
        )
        wx = 0.5 * (self.weights[:, 1:] + self.weights[:, :-1])
        wy = 0.5 * (self.weights[1:, :] + self.weights[:-1, :])
        return self._weighted_mean(dx, wx) + self._weighted_mean(dy, wy)


TIME_BUDGET_ALIASES = {
    "smoke": "1m",
    "1min": "1m",
    "1minute": "1m",
    "5min": "5m",
    "5minute": "5m",
    "10min": "10m",
    "10minute": "10m",
    "20min": "20m",
    "20minute": "20m",
    "30min": "30m",
    "30minute": "30m",
    "photo10k": "photo-native-10k",
    "native10k": "photo-native-10k",
    "native-10k": "photo-native-10k",
    "photo-10k": "photo-native-10k",
    "photo20k": "photo-native-20k",
    "native20k": "photo-native-20k",
    "native-20k": "photo-native-20k",
    "photo-20k": "photo-native-20k",
}

TIME_BUDGET_PRESETS: Dict[str, Dict[str, Any]] = {
    # Budget presets are calibrated for the CPU/Apple-Silicon path by using the
    # native 1500x1000 photo run as the high-end anchor. They are still heuristic:
    # actual runtime depends on backend, image content, and memory pressure.
    "1m": {
        "label": "1m smoke",
        "target_seconds": 60.0,
        "stages": [2],
        "splats_per_megapixel": 340.0,
        "min_splats": 96,
        "max_splats": 512,
        "coverage_sigma_cap_multiplier": 1.00,
        "refinement_overrides": {
            "base_layer_fraction": 0.80,
            "base_layer_alpha": 0.54,
            "saliency_init_fraction": 0.18,
            "saliency_random_fraction": 0.15,
            "densify_weight_saliency": 0.25,
            "residual_detail_saliency_weight": 0.35,
            "region_weight_saliency_boost": 0.35,
            "saliency_sampling_gamma": 0.85,
        },
        "residual_detail_enabled": False,
        "residual_detail_reserve_fraction": 0.0,
        "residual_detail_passes": 0,
        "residual_detail_iters": 0,
    },
    "5m": {
        "label": "5m",
        "target_seconds": 300.0,
        "stages": [18, 12, 6],
        "splats_per_megapixel": 900.0,
        "min_splats": 256,
        "max_splats": 1400,
        "coverage_sigma_cap_multiplier": 0.90,
        "refinement_overrides": {
            "base_layer_fraction": 0.60,
            "base_layer_alpha": 0.48,
            "saliency_init_fraction": 0.30,
            "saliency_random_fraction": 0.30,
            "densify_weight_saliency": 0.45,
            "residual_detail_saliency_weight": 0.55,
            "region_weight_saliency_boost": 0.50,
            "saliency_sampling_gamma": 0.80,
        },
        "residual_detail_enabled": True,
        "residual_detail_reserve_fraction": 0.04,
        "residual_detail_passes": 1,
        "residual_detail_iters": 2,
    },
    "10m": {
        "label": "10m",
        "target_seconds": 600.0,
        "stages": [30, 20, 10],
        "splats_per_megapixel": 1350.0,
        "min_splats": 512,
        "max_splats": 2000,
        "coverage_sigma_cap_multiplier": 0.80,
        "refinement_overrides": {
            "base_layer_fraction": 0.48,
            "base_layer_alpha": 0.44,
            "saliency_init_fraction": 0.38,
            "saliency_random_fraction": 0.42,
            "densify_weight_saliency": 0.65,
            "residual_detail_saliency_weight": 0.75,
            "region_weight_saliency_boost": 0.65,
            "saliency_sampling_gamma": 0.75,
        },
        "residual_detail_enabled": True,
        "residual_detail_reserve_fraction": 0.06,
        "residual_detail_passes": 1,
        "residual_detail_iters": 4,
    },
    "20m": {
        "label": "20m",
        "target_seconds": 1200.0,
        "stages": [32, 12],
        "splats_per_megapixel": 1900.0,
        "min_splats": 640,
        "max_splats": 3200,
        "coverage_sigma_cap_multiplier": 0.74,
        "refinement_overrides": {
            "base_layer_fraction": 0.32,
            "base_layer_alpha": 0.43,
            "initial_splat_fraction": 0.72,
            "initial_splat_cap": 2400,
            "edge_init_fraction": 0.60,
            "edge_init_percentile": 64.0,
            "edge_init_sigma_min": 0.42,
            "edge_init_sigma_max": 1.15,
            "edge_init_sigma_major_scale": 2.50,
            "edge_init_sigma_major_max": 3.00,
            "edge_init_alpha_min": 0.34,
            "edge_init_alpha_max": 0.78,
            "edge_init_saliency_weight": 0.90,
            "saliency_init_fraction": 0.42,
            "saliency_random_fraction": 0.48,
            "densify_weight_saliency": 0.75,
            "residual_detail_saliency_weight": 0.85,
            "region_weight_saliency_boost": 0.72,
            "saliency_sampling_gamma": 0.72,
            "residual_detail_fraction": 0.30,
            "residual_detail_percentile": 92.0,
            "residual_detail_sigma_min": 0.16,
            "residual_detail_sigma_max": 0.90,
            "residual_detail_edge_weight": 1.10,
            "residual_detail_edge_fraction": 0.68,
            "residual_detail_edge_percentile": 70.0,
            "residual_detail_edge_sigma_min": 0.06,
            "residual_detail_edge_sigma_max": 0.42,
            "residual_detail_edge_sigma_major_max": 0.95,
            "residual_detail_edge_alpha_boost": 0.10,
            "residual_detail_edge_color_gain": 1.18,
            "residual_detail_edge_saliency_weight": 1.05,
            "residual_detail_edge_aspect": 2.40,
            "residual_detail_color_gain": 1.08,
            "residual_detail_time_reserve_sec": 420.0,
        },
        "residual_detail_enabled": True,
        "residual_detail_reserve_fraction": 0.07,
        "residual_detail_passes": 1,
        "residual_detail_iters": 6,
    },
    "30m": {
        "label": "30m",
        "target_seconds": 1800.0,
        "stages": [80, 60, 40, 20],
        "splats_per_megapixel": 2500.0,
        "min_splats": 768,
        "max_splats": None,
        "coverage_sigma_cap_multiplier": 0.70,
        "refinement_overrides": {
            "base_layer_fraction": 0.30,
            "base_layer_alpha": 0.42,
            "initial_splat_fraction": 0.68,
            "initial_splat_cap": 3600,
            "edge_init_fraction": 0.52,
            "edge_init_percentile": 64.0,
            "edge_init_sigma_min": 0.42,
            "edge_init_sigma_max": 1.20,
            "edge_init_sigma_major_scale": 2.40,
            "edge_init_sigma_major_max": 3.10,
            "edge_init_alpha_min": 0.34,
            "edge_init_alpha_max": 0.78,
            "edge_init_saliency_weight": 0.85,
            "saliency_init_fraction": 0.46,
            "saliency_random_fraction": 0.52,
            "densify_weight_saliency": 0.85,
            "residual_detail_saliency_weight": 0.95,
            "region_weight_saliency_boost": 0.80,
            "saliency_sampling_gamma": 0.70,
            "residual_detail_edge_weight": 1.00,
            "residual_detail_edge_fraction": 0.58,
            "residual_detail_edge_percentile": 72.0,
            "residual_detail_edge_sigma_min": 0.07,
            "residual_detail_edge_sigma_max": 0.50,
            "residual_detail_edge_sigma_major_max": 1.10,
            "residual_detail_edge_alpha_boost": 0.08,
            "residual_detail_edge_color_gain": 1.12,
            "residual_detail_edge_saliency_weight": 0.95,
            "residual_detail_edge_aspect": 2.20,
        },
        "residual_detail_enabled": True,
        "residual_detail_reserve_fraction": 0.08,
        "residual_detail_passes": 1,
        "residual_detail_iters": 8,
    },
    "photo-native-10k": {
        "label": "photo native 10k",
        "target_seconds": 1200.0,
        "stages": [36, 20, 10],
        "splats_per_megapixel": 6500.0,
        "min_splats": 10000,
        "max_splats": 10000,
        "coverage_sigma_cap_multiplier": 0.78,
        "refinement_overrides": {
            # Start dense enough for native photos, but keep most of that
            # budget in stable color/shape support. The previous edge-heavy
            # 10k run proved that too many tangent strokes improve metrics
            # while making faces visibly noisy.
            "base_layer_fraction": 0.34,
            "base_layer_alpha": 0.38,
            "initial_splat_fraction": 0.64,
            "initial_splat_cap": 6400,
            "edge_init_fraction": 0.24,
            "edge_init_percentile": 62.0,
            "edge_init_sigma_min": 0.55,
            "edge_init_sigma_max": 1.60,
            "edge_init_sigma_major_scale": 1.70,
            "edge_init_sigma_major_max": 3.20,
            "edge_init_alpha_min": 0.18,
            "edge_init_alpha_max": 0.52,
            "edge_init_saliency_weight": 0.75,
            "edge_init_anisotropy_threshold": 1.20,
            "saliency_init_fraction": 0.58,
            "saliency_init_percentile": 48.0,
            "saliency_random_fraction": 0.60,
            "saliency_random_percentile": 45.0,
            "densify_fraction": 0.36,
            "densify_percentile": 64.0,
            "densify_weight_saliency": 1.00,
            "densify_weight_error": 0.40,
            "densify_weight_uncovered": 0.38,
            "densify_weight_edge": 0.12,
            "region_weight_saliency_boost": 0.92,
            "saliency_sampling_gamma": 0.62,
            "background_suppressed_saliency_enabled": True,
            "background_suppressed_saliency_use_for_sampling": True,
            "background_suppressed_saliency_use_for_weights": False,
            "background_suppressed_saliency_edge_gate": 0.80,
            "background_suppressed_saliency_focus_gate": 0.80,
            "background_suppressed_saliency_penalty_strength": 0.65,
            "renderer_tile_size": 24,
            "residual_detail_fraction": 0.14,
            "residual_detail_percentile": 76.0,
            "residual_detail_sigma_min": 0.22,
            "residual_detail_sigma_max": 1.05,
            "residual_detail_alpha_min": 0.12,
            "residual_detail_alpha_max": 0.58,
            "residual_detail_edge_weight": 0.45,
            "residual_detail_edge_fraction": 0.24,
            "residual_detail_edge_percentile": 62.0,
            "residual_detail_edge_sigma_min": 0.12,
            "residual_detail_edge_sigma_max": 0.58,
            "residual_detail_edge_sigma_major_max": 1.10,
            "residual_detail_edge_alpha_boost": 0.04,
            "residual_detail_edge_color_gain": 1.04,
            "residual_detail_edge_saliency_weight": 0.70,
            "residual_detail_edge_aspect": 1.80,
            "residual_detail_color_gain": 1.02,
            "coverage_sigma_max": 22.0,
        },
        "residual_detail_enabled": True,
        "residual_detail_reserve_fraction": 0.06,
        "residual_detail_passes": 2,
        "residual_detail_iters": 6,
    },
    "photo-native-20k": {
        "label": "photo native 20k",
        "target_seconds": 1800.0,
        "stages": [24, 12, 6],
        "splats_per_megapixel": 13000.0,
        "min_splats": 20000,
        "max_splats": 20000,
        "coverage_sigma_cap_multiplier": 0.74,
        "refinement_overrides": {
            # The 20k path is meant to test native-photo ceiling behavior, not
            # spend most of its budget on late edge confetti.
            "base_layer_fraction": 0.32,
            "base_layer_alpha": 0.38,
            "initial_splat_fraction": 1.00,
            "initial_splat_cap": 20000,
            "edge_init_fraction": 0.22,
            "edge_init_percentile": 62.0,
            "edge_init_sigma_min": 0.55,
            "edge_init_sigma_max": 1.60,
            "edge_init_sigma_major_scale": 1.70,
            "edge_init_sigma_major_max": 3.20,
            "edge_init_alpha_min": 0.18,
            "edge_init_alpha_max": 0.52,
            "edge_init_saliency_weight": 0.75,
            "edge_init_anisotropy_threshold": 1.20,
            "saliency_init_fraction": 0.58,
            "saliency_init_percentile": 48.0,
            "saliency_random_fraction": 0.60,
            "saliency_random_percentile": 45.0,
            "densify_fraction": 0.18,
            "densify_percentile": 66.0,
            "densify_weight_saliency": 1.00,
            "densify_weight_error": 0.40,
            "densify_weight_uncovered": 0.34,
            "densify_weight_edge": 0.12,
            "region_weight_saliency_boost": 0.92,
            "saliency_sampling_gamma": 0.62,
            "background_suppressed_saliency_enabled": True,
            "background_suppressed_saliency_use_for_sampling": True,
            "background_suppressed_saliency_use_for_weights": False,
            "background_suppressed_saliency_edge_gate": 0.80,
            "background_suppressed_saliency_focus_gate": 0.80,
            "background_suppressed_saliency_penalty_strength": 0.65,
            "renderer_tile_size": 24,
            "residual_detail_fraction": 0.08,
            "residual_detail_percentile": 78.0,
            "residual_detail_sigma_min": 0.18,
            "residual_detail_sigma_max": 0.92,
            "residual_detail_alpha_min": 0.10,
            "residual_detail_alpha_max": 0.48,
            "residual_detail_edge_weight": 0.38,
            "residual_detail_edge_fraction": 0.20,
            "residual_detail_edge_percentile": 64.0,
            "residual_detail_edge_sigma_min": 0.10,
            "residual_detail_edge_sigma_max": 0.50,
            "residual_detail_edge_sigma_major_max": 1.00,
            "residual_detail_edge_alpha_boost": 0.03,
            "residual_detail_edge_color_gain": 1.03,
            "residual_detail_edge_saliency_weight": 0.70,
            "residual_detail_edge_aspect": 1.70,
            "residual_detail_color_gain": 1.01,
            "coverage_sigma_max": 24.0,
        },
        "residual_detail_enabled": True,
        "residual_detail_reserve_fraction": 0.05,
        "residual_detail_passes": 1,
        "residual_detail_iters": 4,
    },
}


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
        pptx_splat_style: str = "soft-edge",
    ):
        self.requested_max_splats = int(max_splats)
        self.max_splats = int(max_splats)
        self.k_sigma = k_sigma
        self.stages = stages or [200, 150, 100, 50]
        self.target_size = target_size
        self.gradient_method = gradient_method
        self.device = torch.device(device)
        self.renderer_backend = renderer_backend
        self.optimizer_backend = self._normalize_optimizer_backend(optimizer_backend)
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
        self.time_budget = self._normalize_time_budget(time_budget)
        self.layered_saliency = bool(layered_saliency)
        self.pptx_splat_style = str(pptx_splat_style).strip().lower().replace("_", "-")
        self.time_budget_plan: Optional[Dict[str, Any]] = None
        self._time_budget_deadline: Optional[float] = None
        self._platform_splat_cap: Optional[Dict[str, Any]] = None
        self.apple_silicon_splat_cap = (
            None
            if apple_silicon_splat_cap is None or int(apple_silicon_splat_cap) <= 0
            else int(apple_silicon_splat_cap)
        )

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
        self.region_weighting_enabled = bool(
            self.refinement_config.get("region_weighting_enabled", False)
        )
        self.svg_export_recipe = (
            str(self.refinement_config.get("svg_export_recipe", "standard"))
            .strip()
            .lower()
        )
        self.training_export_target = self._normalize_training_export_target(
            self.refinement_config.get("training_export_target", "canvas")
        )
        self.mlx_loss = (
            str(self.refinement_config.get("mlx_loss", "linear-l1"))
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
        self.mlx_spatial_weighting_enabled = bool(
            self.optimizer_backend == "mlx" and self.mlx_loss.startswith("weighted")
        )
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
        self._region_weight_map: Optional[np.ndarray] = None
        self._region_saliency_map: Optional[np.ndarray] = None
        self._region_detail_priority_map: Optional[np.ndarray] = None
        self._region_background_penalty_map: Optional[np.ndarray] = None
        self._region_foreground_mask: Optional[np.ndarray] = None
        self._region_background_safe_mask: Optional[np.ndarray] = None
        self._region_edge_band_mask: Optional[np.ndarray] = None

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
            logger.info(
                "Apple Silicon detected - limiting max_splats to %s", self.max_splats
            )

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
        return bool(
            self.optimizer_backend == "mlx" and self.mlx_loss.startswith("weighted")
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

    def _sampling_prior_map(self) -> Optional[np.ndarray]:
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
        self, score_map: np.ndarray, strength: float
    ) -> np.ndarray:
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
        if normalized in {"", "canvas", "native", "renderer", "linear"}:
            return "canvas"
        if normalized in {"svg", "svg-browser", "browser-svg"}:
            return "svg"
        if normalized in {
            "pptx",
            "pptx-soft",
            "pptx-softedge",
            "pptx-soft-edge",
            "powerpoint",
        }:
            return "pptx-softedge"
        raise ValueError(f"Unsupported training export target: {value}")

    def _use_pptx_proxy_training(self) -> bool:
        return self.training_export_target == "pptx-softedge"

    def _create_training_renderer(self, width: int, height: int) -> torch.nn.Module:
        compositing_space = (
            "srgb"
            if self.training_export_target in {"svg", "pptx-softedge"}
            else self.compositing_space
        )
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
            blend_mode=self.blend_mode,
            background_color=self._background_linear_rgb,
            compositing_space=compositing_space,
            tile_bin_rebuild_interval=tile_bin_rebuild_interval,
            tile_bin_padding=tile_bin_padding,
            batch_tile_count=batch_tile_count,
            max_active_splats_per_tile=max_active_splats_per_tile,
        )
        if not self._use_pptx_proxy_training():
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
        if output_format not in {"svg", "drawingml", "pptx", "canvas"}:
            raise ValueError(f"Unsupported output format: {output_format}")

        run_seed = self.seed if seed is None else seed
        rng = (
            np.random.default_rng(run_seed)
            if run_seed is not None
            else np.random.default_rng()
        )
        if run_seed is not None:
            torch.manual_seed(int(run_seed))

        start_time = time.perf_counter()
        run_timer = start_time
        timings: Dict[str, float] = {}
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
                "requested_max_splats": self.requested_max_splats,
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
                "optimizer_backend": self.optimizer_backend,
                "mlx_loss": self.mlx_loss if self.optimizer_backend == "mlx" else None,
                "mlx_spatial_weighting_enabled": (
                    self.mlx_spatial_weighting_enabled
                    if self.optimizer_backend == "mlx"
                    else None
                ),
                "mlx_tile_plan": (
                    self.mlx_tile_plan if self.optimizer_backend == "mlx" else None
                ),
                "mlx_tile_plan_rebuild_interval": (
                    self.mlx_tile_plan_rebuild_interval
                    if self.optimizer_backend == "mlx"
                    else None
                ),
                "mlx_trainable_groups": (
                    list(self.mlx_trainable_groups)
                    if self.optimizer_backend == "mlx"
                    else None
                ),
                "blend_mode": self.blend_mode,
                "output_format": output_format,
                "training_export_target": self.training_export_target,
                "pptx_export_mode": (
                    "drawingml-splats" if output_format == "pptx" else None
                ),
                "pptx_splat_style": self.pptx_splat_style,
                "quality_profile": self.quality_profile,
                "loss_weights": self.loss_weights,
                "learning_rates": self.learning_rates,
                "refinement_config": self.refinement_config,
                "schedule_config": self.schedule_config,
                "region_weighting_enabled": self.region_weighting_enabled,
                "svg_export_recipe": self.svg_export_recipe,
                "time_budget": self.time_budget,
                "time_budget_plan": self.time_budget_plan,
                "platform_splat_cap": self._platform_splat_cap,
                "apple_silicon_splat_cap": self.apple_silicon_splat_cap,
                "layered_saliency": self.layered_saliency,
            },
            "stages": [],
            "timings_sec": timings,
        }

        if verbose:
            logger.info(
                "Run start: input=%s output_format=%s seed=%s requested_splats=%s device=%s",
                input_path,
                output_format,
                run_seed,
                self.requested_max_splats,
                self.device,
            )
            logger.info(
                "Loading PNG: %s (target_size=%s, resolution_scale=%.2f)",
                input_path,
                resolved_target_size,
                self.resolution_scale,
            )
        phase_t0 = time.perf_counter()
        image = load_png(input_path, target_size=resolved_target_size)
        timings["load_png"] = float(time.perf_counter() - phase_t0)
        height, width = image.shape[:2]
        if verbose:
            logger.info(
                "Loaded %sx%s image in %.2fs", width, height, timings["load_png"]
            )
        self._image_width = width
        self._image_height = height
        self._background_linear_rgb = self._estimate_background_color(image)
        self._region_weight_map = None
        self._region_saliency_map = None
        self._region_detail_priority_map = None
        self._region_background_penalty_map = None
        self._region_foreground_mask = None
        self._region_background_safe_mask = None
        self._region_edge_band_mask = None
        guidance: Optional[Dict[str, Any]] = None
        needs_region_guidance = (
            self.time_budget is not None
            or self.region_weighting_enabled
            or self.layered_saliency
            or self._use_pptx_proxy_training()
            or self._use_mlx_spatial_weights()
            or int(max(0, self.refinement_config.get("svg_proxy_postfit_iters", 0))) > 0
            or int(max(0, self.refinement_config.get("pptx_proxy_postfit_iters", 0)))
            > 0
            or self.svg_export_recipe
            in {
                "browser",
                "browser_compatible",
                "browser-compatible",
                "scripted",
                "scripted-matrix",
                "scripted-standard",
            }
        )
        if needs_region_guidance:
            phase_t0 = time.perf_counter()
            guidance = self._compute_region_guidance(image)
            timings["region_guidance"] = float(time.perf_counter() - phase_t0)
            manifest["config"]["region_guidance"] = guidance["summary"]
            if verbose:
                summary = guidance["summary"]
                logger.info(
                    "Region guidance in %.2fs: foreground=%.1f%% edge=%.1f%% detail_mean=%.4f detail_p95=%.4f",
                    timings["region_guidance"],
                    100.0 * float(summary.get("foreground_ratio", 0.0)),
                    100.0 * float(summary.get("edge_band_ratio", 0.0)),
                    float(
                        summary.get(
                            "detail_priority_mean", summary.get("saliency_mean", 0.0)
                        )
                    ),
                    float(
                        summary.get(
                            "detail_priority_p95", summary.get("saliency_p95", 0.0)
                        )
                    ),
                )
        if self.time_budget is not None:
            phase_t0 = time.perf_counter()
            plan = self._apply_time_budget_plan(
                width=width, height=height, guidance=guidance
            )
            timings["time_budget_plan"] = float(time.perf_counter() - phase_t0)
            self.time_budget_plan = plan
            self._time_budget_deadline = start_time + float(plan["target_seconds"])
            manifest["config"]["max_splats"] = self.max_splats
            manifest["config"]["stages"] = list(self.stages)
            manifest["config"]["refinement_config"] = self.refinement_config
            manifest["config"]["time_budget_plan"] = plan
            manifest["config"]["target_runtime_sec"] = float(plan["target_seconds"])
            manifest["config"]["time_budget_deadline_enabled"] = True
            if "max_splats" in self.acceptance_criteria:
                self.acceptance_criteria["max_splats"] = float(self.max_splats)
            if "max_runtime_sec" in self.acceptance_criteria:
                self.acceptance_criteria["max_runtime_sec"] = float(
                    plan["target_seconds"]
                )
            if verbose:
                logger.info(
                    "Applied %s budget: max_splats=%s, stages=%s, saliency_multiplier=%.2f",
                    plan["label"],
                    self.max_splats,
                    self.stages,
                    plan["saliency_multiplier"],
                )
                logger.info(
                    "Budget plan timing %.2fs: initial=%s base=%s target_runtime=%.0fs",
                    timings["time_budget_plan"],
                    plan.get("initial_splat_estimate"),
                    plan.get("base_layer_estimate"),
                    plan.get("target_seconds", 0.0),
                )
        else:
            self._time_budget_deadline = None
            manifest["config"]["time_budget_deadline_enabled"] = False
        if guidance is not None and (
            self.time_budget is not None
            or self.region_weighting_enabled
            or self.layered_saliency
            or self._use_pptx_proxy_training()
            or self._use_mlx_spatial_weights()
            or int(max(0, self.refinement_config.get("svg_proxy_postfit_iters", 0))) > 0
            or int(max(0, self.refinement_config.get("pptx_proxy_postfit_iters", 0)))
            > 0
            or self.svg_export_recipe
            in {
                "browser",
                "browser_compatible",
                "browser-compatible",
                "scripted",
                "scripted-matrix",
                "scripted-standard",
            }
        ):
            self._region_weight_map = guidance["weight_map"]
            self._region_saliency_map = guidance.get("saliency_map")
            self._region_detail_priority_map = guidance.get("detail_priority_map")
            self._region_background_penalty_map = guidance.get("background_penalty_map")
            self._region_foreground_mask = guidance["foreground_mask"]
            self._region_background_safe_mask = guidance["background_safe_mask"]
            self._region_edge_band_mask = guidance["edge_band_mask"]
            self._background_linear_rgb = guidance["background_linear_rgb"]
        structure_enabled = bool(
            self.refinement_config.get("structure_precompute_enabled", False)
        )
        structure_smoothing_sigma = float(
            max(0.0, self.refinement_config.get("structure_smoothing_sigma", 0.0))
        )
        structure_anisotropy_clip = float(
            max(1.0, self.refinement_config.get("structure_anisotropy_clip", 10.0))
        )
        structure_min_coherence = float(
            np.clip(
                self.refinement_config.get("structure_min_coherence", 0.12), 0.0, 1.0
            )
        )
        structure_primary: Optional[np.ndarray] = None
        structure_anisotropy: Optional[np.ndarray] = None
        if structure_enabled:
            phase_t0 = time.perf_counter()
            structure_primary, structure_anisotropy = compute_structure_field(
                image=image,
                method=self.gradient_method,
                smoothing_sigma=structure_smoothing_sigma,
                anisotropy_clip=structure_anisotropy_clip,
                min_coherence=structure_min_coherence,
            )
            timings["structure_precompute"] = float(time.perf_counter() - phase_t0)
            if verbose:
                logger.info(
                    "Using precomputed structure maps for init/densify guidance in %.2fs.",
                    timings["structure_precompute"],
                )
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
        phase_t0 = time.perf_counter()
        splats = self._initialize_splats(
            image,
            rng=rng,
            structure_primary=structure_primary,
            structure_anisotropy=structure_anisotropy,
        )
        timings["initialize_splats"] = float(time.perf_counter() - phase_t0)
        self._write_stage_artifact(
            artifacts_path, "init", splats, {"count": len(splats)}
        )
        if verbose:
            logger.info(
                "Initialized %s splats in %.2fs",
                len(splats),
                timings["initialize_splats"],
            )

        if verbose:
            logger.info("Starting optimization with %s initial splats...", len(splats))
        phase_t0 = time.perf_counter()
        splats, stage_metrics = self._optimize_splats(
            image=image,
            splats=splats,
            rng=rng,
            verbose=verbose,
            artifacts_dir=artifacts_path,
            structure_primary=structure_primary,
            structure_anisotropy=structure_anisotropy,
        )
        timings["optimize_splats"] = float(time.perf_counter() - phase_t0)
        manifest["stages"].extend(stage_metrics)
        if verbose:
            logger.info(
                "Optimization completed in %.2fs with %s splats",
                timings["optimize_splats"],
                len(splats),
            )

        if verbose:
            logger.info("Post-processing splats...")
        phase_t0 = time.perf_counter()
        splats = self._postprocess_splats(splats=splats, image=image, rng=rng)
        timings["postprocess_splats"] = float(time.perf_counter() - phase_t0)
        if verbose:
            logger.info(
                "Post-processing completed in %.2fs", timings["postprocess_splats"]
            )
        svg_postfit_iters = int(
            max(0, self.refinement_config.get("svg_proxy_postfit_iters", 0))
        )
        if output_format == "svg" and svg_postfit_iters > 0:
            if verbose:
                logger.info(
                    "Post-fitting splats for SVG proxy (%s iterations)...",
                    svg_postfit_iters,
                )
            phase_t0 = time.perf_counter()
            splats, svg_postfit_metric = self._postfit_splats_for_svg_proxy(
                splats=splats,
                image=image,
                width=width,
                height=height,
                num_iters=svg_postfit_iters,
                verbose=verbose,
            )
            timings["svg_proxy_postfit"] = float(time.perf_counter() - phase_t0)
            manifest["stages"].append(svg_postfit_metric)
            self._write_stage_artifact(
                artifacts_path, "svg-postfit", splats, svg_postfit_metric
            )
            if verbose:
                logger.info(
                    "SVG proxy post-fit completed in %.2fs",
                    timings["svg_proxy_postfit"],
                )
        pptx_postfit_iters = int(
            max(0, self.refinement_config.get("pptx_proxy_postfit_iters", 0))
        )
        if output_format == "pptx" and pptx_postfit_iters > 0:
            if verbose:
                logger.info(
                    "Post-fitting splats for PPTX proxy (%s iterations)...",
                    pptx_postfit_iters,
                )
            phase_t0 = time.perf_counter()
            splats, pptx_postfit_metric = self._postfit_splats_for_pptx_proxy(
                splats=splats,
                image=image,
                width=width,
                height=height,
                num_iters=pptx_postfit_iters,
                verbose=verbose,
            )
            timings["pptx_proxy_postfit"] = float(time.perf_counter() - phase_t0)
            manifest["stages"].append(pptx_postfit_metric)
            self._write_stage_artifact(
                artifacts_path, "pptx-postfit", splats, pptx_postfit_metric
            )
            if verbose:
                logger.info(
                    "PPTX proxy post-fit completed in %.2fs",
                    timings["pptx_proxy_postfit"],
                )
        self._write_stage_artifact(
            artifacts_path, "final", splats, {"count": len(splats)}
        )

        phase_t0 = time.perf_counter()
        if output_format == "drawingml":
            if verbose:
                logger.info("Generating DrawingML with %s splats...", len(splats))
            output_content = self._generate_drawingml(splats, width, height)
        elif output_format == "pptx":
            if verbose:
                logger.info(
                    "Preparing PPTX package with native DrawingML splat shapes..."
                )
            output_content = self._generate_drawingml(splats, width, height)
        elif output_format == "canvas":
            if verbose:
                logger.info("Generating canvas HTML with %s splats...", len(splats))
            output_content = generate_canvas_html(
                splats,
                width,
                height,
                background_linear_rgb=self._background_linear_rgb,
                title=Path(input_path).stem,
            )
        else:
            if verbose:
                logger.info("Generating SVG with %s splats...", len(splats))
            output_content = self._generate_svg(splats, width, height)
        timings["generate_output"] = float(time.perf_counter() - phase_t0)
        if verbose:
            logger.info(
                "Generated %s output in %.2fs",
                output_format,
                timings["generate_output"],
            )

        if output_path:
            phase_t0 = time.perf_counter()
            if output_format == "drawingml":
                save_drawingml(
                    splats,
                    width,
                    height,
                    output_path,
                    k_sigma=self.k_sigma,
                    background_linear_rgb=self._background_linear_rgb,
                    splat_style=self.pptx_splat_style,
                )
                if verbose:
                    logger.info("Saved DrawingML: %s", output_path)
            elif output_format == "pptx":
                save_pptx_with_splats(
                    splats=splats,
                    width=width,
                    height=height,
                    output_path=output_path,
                    k_sigma=self.k_sigma,
                    background_linear_rgb=self._background_linear_rgb,
                    splat_style=self.pptx_splat_style,
                )
                if verbose:
                    logger.info("Saved PPTX: %s", output_path)
            elif output_format == "canvas":
                Path(output_path).write_text(output_content, encoding="utf-8")
                if verbose:
                    logger.info("Saved canvas HTML: %s", output_path)
            else:
                save_svg(
                    splats,
                    width,
                    height,
                    output_path,
                    k_sigma=self.k_sigma,
                    background_linear_rgb=self._background_linear_rgb,
                    export_recipe=self.svg_export_recipe,
                    foreground_mask=self._region_foreground_mask,
                    background_safe_mask=self._region_background_safe_mask,
                    edge_band_mask=self._region_edge_band_mask,
                )
                if verbose:
                    logger.info("Saved SVG: %s", output_path)

            if save_json:
                json_path = str(Path(output_path).with_suffix(".json"))
                save_splats_json(splats, json_path)
                if verbose:
                    logger.info("Saved JSON: %s", json_path)
            timings["write_output"] = float(time.perf_counter() - phase_t0)
            if verbose:
                logger.info("Wrote output artifacts in %.2fs", timings["write_output"])

        total_time = time.perf_counter() - start_time
        target = torch.from_numpy(image[:, :, :3]).to(self.device)
        phase_t0 = time.perf_counter()
        final_renderer = self._create_training_renderer(width=width, height=height)
        final_loss_fn = self._create_training_loss(
            target=target, width=width, height=height
        )
        internal_metrics = self._compute_quality_metrics(
            splats, target, final_renderer, final_loss_fn
        )
        timings["internal_metrics"] = float(time.perf_counter() - phase_t0)
        if verbose:
            logger.info(
                "Internal metrics in %.2fs: SSIM_sRGB=%.4f PSNR_sRGB=%.2f coverage=%.3f",
                timings["internal_metrics"],
                float(internal_metrics.get("ssim_srgb", 0.0)),
                float(internal_metrics.get("psnr_srgb", 0.0)),
                float(internal_metrics.get("coverage", 0.0)),
            )
        internal_metrics["runtime_sec"] = float(total_time)
        internal_metrics["splat_count"] = float(len(splats))

        # Export-proxy metrics always available from CPU preview render.
        phase_t0 = time.perf_counter()
        preview_linear = render_splats_numpy(
            splats,
            width,
            height,
            background_linear_rgb=self._background_linear_rgb,
        )
        timings["proxy_render"] = float(time.perf_counter() - phase_t0)
        phase_t0 = time.perf_counter()
        proxy_metrics = compute_quality_metrics(image[:, :, :3], preview_linear)
        timings["proxy_metrics"] = float(time.perf_counter() - phase_t0)
        export_quality: Dict[str, Any] = {
            "available": True,
            "method": "proxy-render",
            "used_fallback": True,
            "metrics": proxy_metrics,
        }
        if verbose:
            logger.info(
                "Proxy render+metrics in %.2fs + %.2fs: SSIM_sRGB=%.4f PSNR_sRGB=%.2f",
                timings["proxy_render"],
                timings["proxy_metrics"],
                float(proxy_metrics.get("ssim_srgb", 0.0)),
                float(proxy_metrics.get("psnr_srgb", 0.0)),
            )

        if (
            output_format == "svg"
            and output_path
            and self.svg_export_recipe
            not in {
                "scripted",
                "scripted-matrix",
                "scripted-standard",
            }
        ):
            phase_t0 = time.perf_counter()
            svg_quality = evaluate_svg_export_quality(
                target_linear_rgb=image[:, :, :3],
                svg_path=output_path,
                fallback_linear_rgb=preview_linear,
            )
            timings["svg_export_quality"] = float(time.perf_counter() - phase_t0)
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
            dict(acceptance_criteria)
            if acceptance_criteria
            else self.acceptance_criteria.copy()
        )
        acceptance_result = self._evaluate_acceptance(
            final_metrics, effective_acceptance
        )
        roundtrip_result: Optional[Dict[str, Any]] = None
        if validate_roundtrip:
            phase_t0 = time.perf_counter()
            roundtrip_result = validate_export_roundtrip(
                splats=splats,
                width=width,
                height=height,
                k_sigma=self.k_sigma,
            )
            timings["roundtrip_validation"] = float(time.perf_counter() - phase_t0)

        # Optional preview and side-by-side artifacts.
        preview_path = preview_png_path
        if preview_path is None and output_path:
            preview_path = str(
                Path(output_path).with_name(f"{Path(output_path).stem}_preview.png")
            )
        if preview_path:
            phase_t0 = time.perf_counter()
            render_splats_preview_png(
                splats=splats,
                width=width,
                height=height,
                output_path=preview_path,
                background_linear_rgb=self._background_linear_rgb,
            )
            timings["preview_png"] = float(time.perf_counter() - phase_t0)
            if verbose:
                logger.info(
                    "Rendered preview PNG in %.2fs: %s",
                    timings["preview_png"],
                    preview_path,
                )
        if side_by_side_html:
            phase_t0 = time.perf_counter()
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
            timings["side_by_side_html"] = float(time.perf_counter() - phase_t0)

        total_time = time.perf_counter() - start_time
        internal_metrics["runtime_sec"] = float(total_time)
        final_metrics["runtime_sec"] = float(total_time)
        remaining_budget = self._time_budget_seconds_remaining()
        timings["total_wall"] = float(time.perf_counter() - run_timer)
        manifest["total_time_sec"] = total_time
        manifest["time_budget_remaining_sec"] = (
            None if remaining_budget is None else max(0.0, float(remaining_budget))
        )
        manifest["time_budget_exhausted"] = bool(self._time_budget_exhausted())
        manifest["final_splat_count"] = len(splats)
        manifest["layered_saliency"] = self._layer_summary(splats)
        manifest["final_metrics"] = final_metrics
        manifest["internal_metrics"] = internal_metrics
        manifest["export_quality"] = export_quality
        manifest["acceptance_metric_source"] = (
            "export" if use_export_for_acceptance else "internal"
        )
        manifest["acceptance"] = acceptance_result
        if roundtrip_result is not None:
            manifest["roundtrip_validation"] = roundtrip_result

        phase_t0 = time.perf_counter()
        self._write_manifest(artifacts_path, manifest)
        timings["write_manifest"] = float(time.perf_counter() - phase_t0)
        self._time_budget_deadline = None

        if verbose:
            logger.info(
                "Conversion completed in %.2fs: splats=%s acceptance=%s source=%s",
                timings["total_wall"],
                len(splats),
                acceptance_result.get("pass"),
                manifest["acceptance_metric_source"],
            )
            logger.info(
                "Timings: load=%.2fs guidance=%.2fs init=%.2fs optimize=%.2fs output=%.2fs metrics=%.2fs total=%.2fs",
                timings.get("load_png", 0.0),
                timings.get("region_guidance", 0.0),
                timings.get("initialize_splats", 0.0),
                timings.get("optimize_splats", 0.0),
                timings.get("generate_output", 0.0) + timings.get("write_output", 0.0),
                timings.get("internal_metrics", 0.0)
                + timings.get("proxy_render", 0.0)
                + timings.get("proxy_metrics", 0.0),
                timings.get("total_wall", 0.0),
            )

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
        initial_count = self._initial_splat_count()
        if initial_count <= 0:
            return []

        base_fraction = float(
            np.clip(self.refinement_config.get("base_layer_fraction", 0.35), 0.10, 0.80)
        )
        base_count = max(4, int(round(initial_count * base_fraction)))
        detail_count = max(1, initial_count - base_count)

        base_positions = self._make_stratified_positions(
            width=width,
            height=height,
            count=base_count,
            rng=rng,
            jitter_ratio=0.65,
        )

        adaptive_count = max(
            1, int(round(detail_count * (1.0 - self.init_random_ratio)))
        )
        random_count = max(0, detail_count - adaptive_count)
        sampling_prior = self._sampling_prior_map()
        edge_count = 0
        edge_map: Optional[np.ndarray] = None
        edge_positions: List[Tuple[float, float]] = []
        edge_init_fraction = float(
            np.clip(self.refinement_config.get("edge_init_fraction", 0.0), 0.0, 0.85)
        )
        if edge_init_fraction > 0.0:
            edge_count = min(
                adaptive_count, int(round(adaptive_count * edge_init_fraction))
            )
            if edge_count > 0:
                edge_map = self._build_edge_map(image)
                edge_score = np.asarray(edge_map, dtype=np.float32)
                if sampling_prior is not None and sampling_prior.shape == (
                    height,
                    width,
                ):
                    edge_score = self._apply_saliency_sampling_bias(
                        edge_score,
                        strength=float(
                            self.refinement_config.get(
                                "edge_init_saliency_weight", 0.70
                            )
                        ),
                    )
                edge_positions = self._sample_map_positions(
                    score_map=edge_score,
                    count=edge_count,
                    rng=rng,
                    percentile=float(
                        self.refinement_config.get("edge_init_percentile", 68.0)
                    ),
                    jitter=0.28,
                )
                edge_count = len(edge_positions)
        saliency_count = 0
        if sampling_prior is not None and sampling_prior.shape == (height, width):
            saliency_fraction = float(
                np.clip(
                    self.refinement_config.get("saliency_init_fraction", 0.35),
                    0.0,
                    0.85,
                )
            )
            saliency_count = min(
                max(0, adaptive_count - edge_count),
                int(round(adaptive_count * saliency_fraction)),
            )
        content_adaptive_count = max(0, adaptive_count - edge_count - saliency_count)

        seed_positions = (
            init_seeds_content_adaptive(
                image=image,
                target_count=content_adaptive_count,
                gradient_weight=self.init_gradient_weight,
                method=self.gradient_method,
                rng=rng,
            )
            if content_adaptive_count > 0
            else []
        )
        saliency_positions: List[Tuple[float, float]] = []
        if saliency_count > 0 and sampling_prior is not None:
            saliency_positions = self._sample_map_positions(
                score_map=sampling_prior,
                count=saliency_count,
                rng=rng,
                percentile=float(
                    self.refinement_config.get("saliency_init_percentile", 62.0)
                ),
                jitter=0.45,
            )
        random_positions: List[Tuple[float, float]] = []
        if random_count > 0:
            random_saliency_fraction = float(
                np.clip(
                    self.refinement_config.get("saliency_random_fraction", 0.35),
                    0.0,
                    1.0,
                )
            )
            random_saliency_count = (
                min(random_count, int(round(random_count * random_saliency_fraction)))
                if sampling_prior is not None
                else 0
            )
            if random_saliency_count > 0 and sampling_prior is not None:
                random_positions.extend(
                    self._sample_map_positions(
                        score_map=sampling_prior,
                        count=random_saliency_count,
                        rng=rng,
                        percentile=float(
                            self.refinement_config.get(
                                "saliency_random_percentile", 55.0
                            )
                        ),
                        jitter=0.65,
                    )
                )
            uniform_count = random_count - random_saliency_count
            if uniform_count > 0:
                random_x = rng.uniform(0.0, float(width), size=uniform_count)
                random_y = rng.uniform(0.0, float(height), size=uniform_count)
                random_positions.extend(
                    (float(x), float(y)) for x, y in zip(random_x, random_y)
                )

        filled_count = (
            len(base_positions)
            + len(seed_positions)
            + len(edge_positions)
            + len(saliency_positions)
            + len(random_positions)
        )
        poisson_count = max(
            0, min(initial_count - filled_count, max(1, detail_count // 5))
        )
        if poisson_count > 0:
            min_distance = max(
                2.0, min(width, height) / max(np.sqrt(max(detail_count, 1.0)), 1.0)
            )
            poisson_positions = poisson_disk_sampling(
                width=width,
                height=height,
                min_distance=min_distance,
                rng=rng,
            )[:poisson_count]
        else:
            poisson_positions = []

        all_positions = (
            base_positions
            + seed_positions
            + edge_positions
            + saliency_positions
            + random_positions
            + poisson_positions
        )
        splats: List[GaussianSplat] = []
        edge_start = len(base_positions) + len(seed_positions)
        edge_end = edge_start + len(edge_positions)

        base_sigma = float(
            np.clip(
                np.sqrt((float(width) * float(height)) / max(base_count, 1)) * 0.85,
                self.refinement_config.get("sigma_min", 1.0),
                self.refinement_config.get("coverage_sigma_max", 8.0),
            )
        )
        base_alpha = float(
            np.clip(self.refinement_config.get("base_layer_alpha", 0.42), 0.08, 0.95)
        )
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
                primary_direction, anisotropy = self._analyze_local_structure(
                    image, x_int, y_int
                )
            color = estimate_local_color(image, x_int, y_int)

            is_base = idx < len(base_positions)
            is_edge_init = edge_start <= idx < edge_end
            if is_base:
                sigma = base_sigma
                alpha = base_alpha
            elif is_edge_init:
                edge_need = (
                    float(edge_map[y_int, x_int]) if edge_map is not None else 1.0
                )
                edge_sigma_min = float(
                    max(0.10, self.refinement_config.get("edge_init_sigma_min", 0.45))
                )
                edge_sigma_max = float(
                    max(
                        edge_sigma_min,
                        self.refinement_config.get("edge_init_sigma_max", 1.25),
                    )
                )
                sigma = float(
                    np.clip(
                        edge_sigma_max - (edge_sigma_max - edge_sigma_min) * edge_need,
                        edge_sigma_min,
                        edge_sigma_max,
                    )
                )
                edge_alpha_min = float(
                    np.clip(
                        self.refinement_config.get("edge_init_alpha_min", 0.30),
                        0.0,
                        1.0,
                    )
                )
                edge_alpha_max = float(
                    np.clip(
                        self.refinement_config.get("edge_init_alpha_max", 0.72),
                        edge_alpha_min,
                        1.0,
                    )
                )
                alpha = float(
                    np.clip(
                        edge_alpha_min
                        + (edge_alpha_max - edge_alpha_min) * (0.35 + 0.65 * edge_need),
                        edge_alpha_min,
                        edge_alpha_max,
                    )
                )
            else:
                sigma = float(
                    np.clip(
                        base_sigma * 0.65,
                        self.refinement_config.get("sigma_min", 1.0),
                        6.0,
                    )
                )
                alpha = float(np.clip(base_alpha + 0.18, 0.15, 0.95))

            init_anisotropy_threshold = float(
                max(1.0, self.refinement_config.get("init_anisotropy_threshold", 1.55))
            )
            edge_init_anisotropy_threshold = float(
                max(
                    1.0,
                    self.refinement_config.get("edge_init_anisotropy_threshold", 1.15),
                )
            )
            if is_edge_init and anisotropy >= edge_init_anisotropy_threshold:
                angle = float(
                    np.arctan2(primary_direction[1], primary_direction[0])
                    + (0.5 * np.pi)
                )
                cos_a, sin_a = float(np.cos(angle)), float(np.sin(angle))
                rotation_matrix = np.array(
                    [[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32
                )
                sigma_major_scale = float(
                    max(
                        1.0,
                        self.refinement_config.get("edge_init_sigma_major_scale", 2.20),
                    )
                )
                sigma_major_max = float(
                    max(
                        sigma,
                        self.refinement_config.get("edge_init_sigma_major_max", 3.00),
                    )
                )
                sigma_major = float(
                    np.clip(sigma * sigma_major_scale, sigma, sigma_major_max)
                )
                sigma_minor = max(sigma, sigma_minor_min)
                splat = create_anisotropic_splat(
                    center=np.array([x, y], dtype=np.float32),
                    eigenvals=np.array(
                        [sigma_major**2, sigma_minor**2], dtype=np.float32
                    ),
                    eigenvecs=rotation_matrix,
                    color=color,
                    alpha=alpha,
                )
            elif (not is_base) and anisotropy >= init_anisotropy_threshold:
                angle = float(np.arctan2(primary_direction[1], primary_direction[0]))
                cos_a, sin_a = float(np.cos(angle)), float(np.sin(angle))
                rotation_matrix = np.array(
                    [[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32
                )
                sigma_major = sigma
                sigma_minor = max(
                    sigma_major
                    / min(
                        float(anisotropy),
                        float(
                            self.refinement_config.get(
                                "local_structure_anisotropy_clip", 4.0
                            )
                        ),
                    ),
                    sigma_minor_min,
                )
                splat = create_anisotropic_splat(
                    center=np.array([x, y], dtype=np.float32),
                    eigenvals=np.array(
                        [sigma_major**2, sigma_minor**2], dtype=np.float32
                    ),
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

            if is_base:
                layer, local_importance = LAYER_BASE, 0.10
            elif is_edge_init:
                layer, local_importance = self._saliency_layer_for_pixel(
                    x_int, y_int, LAYER_EDGE
                )
                edge_need = (
                    float(edge_map[y_int, x_int]) if edge_map is not None else 1.0
                )
                local_importance = float(max(local_importance, 0.78 + 0.21 * edge_need))
            else:
                layer, local_importance = self._saliency_layer_for_pixel(
                    x_int, y_int, LAYER_MASS
                )
            self._assign_splat_layer(splat, layer, local_importance)
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

    def _sample_map_positions(
        self,
        score_map: np.ndarray,
        count: int,
        rng: np.random.Generator,
        percentile: float,
        jitter: float,
    ) -> List[Tuple[float, float]]:
        """Sample image positions from a continuous saliency/score map."""
        if count <= 0:
            return []
        score = np.asarray(score_map, dtype=np.float32)
        if score.ndim != 2 or score.size == 0:
            return []
        x_indices, y_indices, _ = self._sample_candidate_positions(
            score_map=score,
            percentile=float(np.clip(percentile, 0.0, 100.0)),
            max_samples=int(count),
            rng=rng,
        )
        height, width = score.shape
        jitter_amount = float(np.clip(jitter, 0.0, 1.0))
        positions: List[Tuple[float, float]] = []
        for x, y in zip(x_indices, y_indices):
            x_center = float(
                np.clip(
                    float(x) + rng.uniform(-jitter_amount, jitter_amount),
                    0.0,
                    width - 1.0,
                )
            )
            y_center = float(
                np.clip(
                    float(y) + rng.uniform(-jitter_amount, jitter_amount),
                    0.0,
                    height - 1.0,
                )
            )
            positions.append((x_center, y_center))
        return positions

    def _build_edge_map(self, image: np.ndarray) -> np.ndarray:
        """Build normalized edge-energy map used by densification."""
        grad_mag = compute_gradient_magnitude(image, method=self.gradient_method)
        return self._normalize_map(grad_mag)

    def _analyze_local_structure(
        self, image: np.ndarray, x: int, y: int
    ) -> Tuple[np.ndarray, float]:
        """Analyze local orientation with conservative anisotropy gating."""
        return analyze_local_structure(
            image=image,
            x=x,
            y=y,
            window_size=int(
                max(3, self.refinement_config.get("structure_local_window", 7))
            ),
            anisotropy_clip=float(
                max(
                    1.0,
                    self.refinement_config.get("local_structure_anisotropy_clip", 4.0),
                )
            ),
            min_coherence=float(
                np.clip(
                    self.refinement_config.get("local_structure_min_coherence", 0.12),
                    0.0,
                    1.0,
                )
            ),
            min_energy=float(
                max(0.0, self.refinement_config.get("local_structure_min_energy", 1e-4))
            ),
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

        if self._time_budget_exhausted():
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

            loss_value = float(loss.item())
            iter_elapsed = time.perf_counter() - iter_t0
            end_loss = loss_value
            if loss_value < best_loss:
                best_loss = loss_value
                best_snapshot = params.snapshot()

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
        # MLX renderer prefers a larger tile batch than the torch renderer because
        # mx.compile fuses fewer-but-bigger batches more effectively. Sweet spot
        # on a 400px M-series run was ~128 (40% faster vs 16); above ~256 memory
        # pressure dominates.
        batch_tile_count = int(
            max(1, self.refinement_config.get("renderer_batch_tile_count", 128))
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
        stage_config = MlxStageConfig(
            renderer=MlxRendererConfig(
                tile_size=tile_size,
                batch_tile_count=batch_tile_count,
                blend_mode=self.blend_mode,
                background_color=tuple(
                    float(v) for v in self._background_linear_rgb[:3]
                ),
                max_active_splats_per_tile=max_active_splats_per_tile,
            ),
            loss=MlxLossConfig(self.mlx_loss),
            trainable_groups=self.mlx_trainable_groups,
            tile_plan_mode=self.mlx_tile_plan,
            tile_plan_rebuild_interval=self.mlx_tile_plan_rebuild_interval,
            progress_interval=progress_interval,
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
        # psnr_srgb/ssim_srgb keys the acceptance gate checks -- so on machines
        # without an SVG rasterizer the perceptual gates read 0.0 and always
        # failed even good runs.
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
        cap = int(
            self.max_splats
            if max_splats_cap is None
            else np.clip(max_splats_cap, 0, self.max_splats)
        )
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
            error_map = torch.mean(residual_map**2, dim=-1)
        error_np = error_map.cpu().numpy()
        residual_np = residual_map.cpu().numpy()
        error_norm = self._normalize_map(error_np)
        height, width = image.shape[:2]
        if edge_map is None or edge_map.shape != (height, width):
            edge_map = self._build_edge_map(image)

        if precomputed_coverage_map is not None and precomputed_coverage_map.shape == (
            height,
            width,
        ):
            coverage_map = precomputed_coverage_map
        else:
            coverage_map = self._build_alpha_coverage_map(
                splats=splats,
                width=width,
                height=height,
            )
        uncovered_map = np.clip(1.0 - coverage_map, 0.0, 1.0).astype(np.float32)

        coverage_ratio = self._compute_coverage_ratio(coverage_map)
        target_coverage = float(
            np.clip(self.refinement_config.get("coverage_target", 0.985), 0.0, 1.0)
        )
        coverage_deficit = max(target_coverage - coverage_ratio, 0.0)

        weight_error = float(
            max(self.refinement_config.get("densify_weight_error", 0.50), 0.0)
        )
        weight_uncovered = float(
            max(self.refinement_config.get("densify_weight_uncovered", 0.40), 0.0)
        )
        weight_edge = float(
            max(self.refinement_config.get("densify_weight_edge", 0.10), 0.0)
        )
        weight_sum = max(weight_error + weight_uncovered + weight_edge, 1e-8)
        sampling_map = (
            (weight_error / weight_sum) * error_norm
            + (weight_uncovered / weight_sum) * uncovered_map
            + (weight_edge / weight_sum) * edge_map
        )
        sampling_map = self._apply_saliency_sampling_bias(
            sampling_map,
            strength=float(self.refinement_config.get("densify_weight_saliency", 0.45)),
        )
        sampling_map = np.clip(sampling_map, 0.0, 1.0).astype(np.float32)
        if float(np.sum(sampling_map)) <= 1e-12:
            sampling_map = np.maximum(error_norm, uncovered_map)

        base_percentile = float(
            np.clip(self.refinement_config["densify_percentile"], 0.0, 100.0)
        )
        stage_scale = max(len(self.stages) - stage_idx, 1) / max(len(self.stages), 1)
        adaptive_percentile = float(
            np.clip(base_percentile - 35.0 * coverage_deficit * stage_scale, 45.0, 99.8)
        )

        densify_fraction = float(
            np.clip(self.refinement_config["densify_fraction"], 0.01, 1.0)
        )
        deficit_boost = (
            1.0
            + float(self.refinement_config.get("coverage_densify_boost", 2.0))
            * coverage_deficit
        )
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
        residual_color_gain = float(
            self.refinement_config.get("residual_color_gain", 0.75)
        )
        sigma_minor_min = float(self.refinement_config.get("sigma_minor_min", 0.35))
        sigma_min = float(self.refinement_config.get("sigma_min", 0.45))
        sigma_max = float(self.refinement_config.get("sigma_max", 4.0))
        sigma_scale = float(self.refinement_config.get("sigma_scale", 2.0))
        sigma_fill_max = float(
            max(
                self.refinement_config.get("coverage_sigma_max", sigma_max * 1.8),
                sigma_max,
            )
        )
        for idx, (x, y) in enumerate(zip(x_indices, y_indices)):
            base_color = estimate_local_color(image, x, y)
            residual_rgb = residual_np[y, x, :3].astype(np.float32)
            color = np.clip(
                base_color + residual_color_gain * residual_rgb, 0.0, 1.0
            ).astype(np.float32)
            if not np.isfinite(color).all():
                color = base_color

            detail_need = float(error_norm[y, x])
            fill_need = float(uncovered_map[y, x])
            edge_need = float(edge_map[y, x])

            sigma_detail = float(
                np.clip(sigma_max - sigma_scale * detail_need, sigma_min, sigma_max)
            )
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
                    + self.refinement_config["alpha_scale"]
                    * (0.55 * detail_need + 0.45 * fill_need),
                    self.refinement_config["alpha_min"],
                    self.refinement_config["alpha_max"],
                )
            )
            x_center = float(np.clip(x + rng.uniform(-0.5, 0.5), 0.0, width - 1.0))
            y_center = float(np.clip(y + rng.uniform(-0.5, 0.5), 0.0, height - 1.0))

            local_structure_edge_threshold = float(
                np.clip(
                    self.refinement_config.get("structure_local_edge_threshold", 0.18),
                    0.0,
                    1.0,
                )
            )
            local_structure_detail_threshold = float(
                np.clip(
                    self.refinement_config.get(
                        "structure_local_detail_threshold", 0.22
                    ),
                    0.0,
                    1.0,
                )
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
                primary_direction, anisotropy = self._analyze_local_structure(
                    image, x, y
                )
            anisotropy_threshold = float(
                max(
                    1.0,
                    self.refinement_config.get("densify_anisotropy_threshold", 1.30),
                )
            )
            anisotropy_edge_threshold = float(
                np.clip(
                    self.refinement_config.get(
                        "densify_anisotropy_edge_threshold", 0.14
                    ),
                    0.0,
                    1.0,
                )
            )
            strong_edge_threshold = float(
                np.clip(
                    self.refinement_config.get("densify_strong_edge_threshold", 0.38),
                    0.0,
                    1.0,
                )
            )
            make_anisotropic = (
                anisotropy >= anisotropy_threshold
                and edge_need >= anisotropy_edge_threshold
            ) or (
                anisotropy >= max(1.0, anisotropy_threshold - 0.08)
                and edge_need >= strong_edge_threshold
            )
            if make_anisotropic:
                angle = float(np.arctan2(primary_direction[1], primary_direction[0]))
                cos_a, sin_a = float(np.cos(angle)), float(np.sin(angle))
                rotation_matrix = np.array(
                    [[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32
                )
                sigma_major = sigma * (1.0 + 0.5 * fill_need)
                anisotropy_cap = max(
                    1.0,
                    min(
                        float(anisotropy),
                        float(
                            self.refinement_config.get(
                                "local_structure_anisotropy_clip", 4.0
                            )
                        ),
                    ),
                )
                sigma_minor = max(sigma_major / anisotropy_cap, sigma_minor_min)
                if edge_need > 0.5 and fill_need < 0.5:
                    sigma_minor = max(sigma_minor * 0.75, sigma_minor_min)
                new_splat = create_anisotropic_splat(
                    center=np.array([x_center, y_center], dtype=np.float32),
                    eigenvals=np.array(
                        [sigma_major**2, sigma_minor**2], dtype=np.float32
                    ),
                    eigenvecs=rotation_matrix,
                    color=color,
                    alpha=alpha,
                )
            else:
                new_splat = create_isotropic_splat(
                    center=np.array([x_center, y_center], dtype=np.float32),
                    sigma=sigma,
                    color=color,
                    alpha=alpha,
                )
            layer, local_importance = self._saliency_layer_for_pixel(x, y, LAYER_DETAIL)
            local_importance = float(
                max(
                    local_importance,
                    0.20 + 0.79 * float(np.clip(sample_weights[idx], 0.0, 1.0)),
                )
            )
            self._assign_splat_layer(new_splat, layer, local_importance)
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
        transmittance = np.clip(1.0 - coverage_map, 0.0, 1.0).astype(
            np.float32, copy=True
        )
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
        loss_fn: torch.nn.Module,
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
            pass_t0 = time.perf_counter()
            if self._time_budget_exhausted():
                break
            if len(current_splats) >= self.max_splats:
                break

            with torch.no_grad():
                if current_splats:
                    rendered = renderer(
                        splats_to_tensor(current_splats, device=self.device)
                    )
                else:
                    rendered = torch.zeros(
                        (height, width, 3),
                        dtype=target.dtype,
                        device=target.device,
                    )
                residual_map = target - rendered
                error_map = torch.mean(residual_map**2, dim=-1)

            error_norm = self._normalize_map(error_map.cpu().numpy())
            residual_np = residual_map.cpu().numpy()

            edge_weight = float(
                np.clip(
                    self.refinement_config.get("residual_detail_edge_weight", 0.30),
                    0.0,
                    2.0,
                )
            )
            score_map = self._normalize_map(error_norm * (1.0 + edge_weight * edge_map))
            score_map = self._apply_saliency_sampling_bias(
                score_map,
                strength=float(
                    self.refinement_config.get("residual_detail_saliency_weight", 0.55)
                ),
            )
            percentile = float(
                np.clip(
                    self.refinement_config.get("residual_detail_percentile", 90.0),
                    0.0,
                    100.0,
                )
            )
            fraction = float(
                np.clip(
                    self.refinement_config.get("residual_detail_fraction", 0.12),
                    0.01,
                    1.0,
                )
            )
            max_new = min(
                self.max_splats - len(current_splats),
                int(np.ceil(max(1, len(current_splats)) * fraction)),
            )
            if max_new <= 0:
                break

            edge_fraction = float(
                np.clip(
                    self.refinement_config.get("residual_detail_edge_fraction", 0.45),
                    0.0,
                    0.95,
                )
            )
            edge_count = min(max_new, int(round(max_new * edge_fraction)))
            residual_count = max_new - edge_count
            edge_gamma = float(
                max(
                    0.20, self.refinement_config.get("residual_detail_edge_gamma", 0.70)
                )
            )
            edge_error_floor = float(
                np.clip(
                    self.refinement_config.get(
                        "residual_detail_edge_error_floor", 0.20
                    ),
                    0.0,
                    1.0,
                )
            )
            edge_score = self._normalize_map(
                np.power(np.clip(edge_map, 0.0, 1.0), edge_gamma).astype(np.float32)
                * np.clip(edge_error_floor + error_norm, 0.0, None)
            )
            edge_score = self._apply_saliency_sampling_bias(
                edge_score,
                strength=float(
                    self.refinement_config.get(
                        "residual_detail_edge_saliency_weight", 0.85
                    )
                ),
            )
            edge_percentile = float(
                np.clip(
                    self.refinement_config.get(
                        "residual_detail_edge_percentile", max(55.0, percentile - 18.0)
                    ),
                    0.0,
                    100.0,
                )
            )

            candidates: List[Tuple[int, int, float, bool]] = []
            seen: set[Tuple[int, int]] = set()

            def add_candidates(
                x_values: np.ndarray,
                y_values: np.ndarray,
                weights: np.ndarray,
                *,
                is_edge: bool,
            ) -> None:
                for x_raw, y_raw, weight_raw in zip(x_values, y_values, weights):
                    x = int(np.clip(int(x_raw), 0, width - 1))
                    y = int(np.clip(int(y_raw), 0, height - 1))
                    key = (x, y)
                    if key in seen:
                        continue
                    seen.add(key)
                    candidates.append(
                        (x, y, float(np.clip(weight_raw, 0.0, 1.0)), bool(is_edge))
                    )
                    if len(candidates) >= max_new:
                        break

            if edge_count > 0 and float(np.max(edge_score)) > 1e-8:
                edge_x, edge_y, edge_weights = self._sample_candidate_positions(
                    score_map=edge_score,
                    percentile=edge_percentile,
                    max_samples=edge_count,
                    rng=rng,
                )
                add_candidates(edge_x, edge_y, edge_weights, is_edge=True)

            if residual_count > 0 and len(candidates) < max_new:
                res_x, res_y, res_weights = self._sample_candidate_positions(
                    score_map=score_map,
                    percentile=percentile,
                    max_samples=residual_count,
                    rng=rng,
                )
                add_candidates(res_x, res_y, res_weights, is_edge=False)

            if len(candidates) < max_new:
                fill_x, fill_y, fill_weights = self._sample_candidate_positions(
                    score_map=score_map,
                    percentile=max(0.0, percentile - 12.0),
                    max_samples=max_new - len(candidates),
                    rng=rng,
                )
                add_candidates(fill_x, fill_y, fill_weights, is_edge=False)

            if not candidates:
                break

            sigma_min = float(
                max(0.10, self.refinement_config.get("residual_detail_sigma_min", 0.28))
            )
            sigma_max = float(
                max(
                    sigma_min,
                    self.refinement_config.get("residual_detail_sigma_max", 1.20),
                )
            )
            edge_sigma_min = float(
                max(
                    0.04,
                    self.refinement_config.get(
                        "residual_detail_edge_sigma_min", sigma_min * 0.60
                    ),
                )
            )
            edge_sigma_max = float(
                max(
                    edge_sigma_min,
                    self.refinement_config.get(
                        "residual_detail_edge_sigma_max", min(sigma_max, 0.70)
                    ),
                )
            )
            edge_sigma_major_max = float(
                max(
                    edge_sigma_max,
                    self.refinement_config.get(
                        "residual_detail_edge_sigma_major_max", edge_sigma_max * 1.8
                    ),
                )
            )
            alpha_min = float(
                np.clip(
                    self.refinement_config.get("residual_detail_alpha_min", 0.16),
                    0.0,
                    1.0,
                )
            )
            alpha_max = float(
                np.clip(
                    self.refinement_config.get("residual_detail_alpha_max", 0.70),
                    alpha_min,
                    1.0,
                )
            )
            residual_color_gain = float(
                self.refinement_config.get("residual_detail_color_gain", 0.95)
            )
            edge_color_gain = float(
                self.refinement_config.get(
                    "residual_detail_edge_color_gain", residual_color_gain
                )
            )
            edge_alpha_boost = float(
                max(
                    0.0,
                    self.refinement_config.get(
                        "residual_detail_edge_alpha_boost", 0.06
                    ),
                )
            )
            edge_make_threshold = float(
                np.clip(
                    self.refinement_config.get(
                        "residual_detail_edge_make_threshold", 0.20
                    ),
                    0.0,
                    1.0,
                )
            )
            edge_anisotropic = bool(
                self.refinement_config.get("residual_detail_edge_anisotropic", True)
            )
            edge_anisotropy_threshold = float(
                max(
                    1.0,
                    self.refinement_config.get(
                        "residual_detail_edge_anisotropy_threshold", 1.20
                    ),
                )
            )
            edge_aspect = float(
                max(1.0, self.refinement_config.get("residual_detail_edge_aspect", 2.0))
            )

            new_splats: List[GaussianSplat] = []
            edge_candidates_used = 0
            anisotropic_edge_splats = 0
            for x, y, sample_weight, sampled_from_edge_pool in candidates:
                base_color = estimate_local_color(image, x, y)
                residual_rgb = residual_np[y, x, :3].astype(np.float32)
                edge_need = float(edge_map[y, x])
                is_edge_candidate = bool(
                    sampled_from_edge_pool or edge_need >= edge_make_threshold
                )
                color_gain = (
                    edge_color_gain if is_edge_candidate else residual_color_gain
                )
                color = np.clip(
                    base_color + color_gain * residual_rgb, 0.0, 1.0
                ).astype(np.float32)
                if not np.isfinite(color).all():
                    color = base_color

                detail_need = float(error_norm[y, x])
                if is_edge_candidate:
                    edge_candidates_used += 1
                    edge_detail_need = float(
                        np.clip(0.65 * detail_need + 0.35 * edge_need, 0.0, 1.0)
                    )
                    sigma = float(
                        np.clip(
                            edge_sigma_max
                            - (edge_sigma_max - edge_sigma_min) * edge_detail_need,
                            edge_sigma_min,
                            edge_sigma_max,
                        )
                    )
                    alpha = float(
                        np.clip(
                            alpha_min
                            + (alpha_max - alpha_min) * (0.35 + 0.65 * edge_detail_need)
                            + edge_alpha_boost,
                            alpha_min,
                            alpha_max,
                        )
                    )
                else:
                    sigma = float(
                        np.clip(
                            sigma_max - (sigma_max - sigma_min) * detail_need,
                            sigma_min,
                            sigma_max,
                        )
                    )
                    alpha = float(
                        np.clip(
                            alpha_min
                            + (alpha_max - alpha_min) * (0.30 + 0.70 * detail_need),
                            alpha_min,
                            alpha_max,
                        )
                    )
                x_center = float(
                    np.clip(x + rng.uniform(-0.35, 0.35), 0.0, width - 1.0)
                )
                y_center = float(
                    np.clip(y + rng.uniform(-0.35, 0.35), 0.0, height - 1.0)
                )

                splat: GaussianSplat
                if is_edge_candidate and edge_anisotropic:
                    primary_direction, anisotropy = self._analyze_local_structure(
                        image, x, y
                    )
                    if float(anisotropy) >= edge_anisotropy_threshold:
                        angle = float(
                            np.arctan2(primary_direction[1], primary_direction[0])
                            + (0.5 * np.pi)
                        )
                        cos_a, sin_a = float(np.cos(angle)), float(np.sin(angle))
                        rotation_matrix = np.array(
                            [[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32
                        )
                        sigma_major = float(
                            np.clip(sigma * edge_aspect, sigma, edge_sigma_major_max)
                        )
                        sigma_minor = float(max(edge_sigma_min, sigma / edge_aspect))
                        splat = create_anisotropic_splat(
                            center=np.array([x_center, y_center], dtype=np.float32),
                            eigenvals=np.array(
                                [sigma_major**2, sigma_minor**2], dtype=np.float32
                            ),
                            eigenvecs=rotation_matrix,
                            color=color,
                            alpha=alpha,
                        )
                        anisotropic_edge_splats += 1
                    else:
                        splat = create_isotropic_splat(
                            center=np.array([x_center, y_center], dtype=np.float32),
                            sigma=sigma,
                            color=color,
                            alpha=alpha,
                        )
                else:
                    splat = create_isotropic_splat(
                        center=np.array([x_center, y_center], dtype=np.float32),
                        sigma=sigma,
                        color=color,
                        alpha=alpha,
                    )
                layer, local_importance = self._saliency_layer_for_pixel(
                    x,
                    y,
                    LAYER_EDGE if is_edge_candidate else LAYER_DETAIL,
                )
                local_importance = float(
                    max(local_importance, 0.65 + 0.35 * sample_weight)
                )
                if is_edge_candidate:
                    local_importance = float(
                        max(local_importance, 0.80 + 0.19 * edge_need)
                    )
                self._assign_splat_layer(splat, layer, local_importance)
                new_splats.append(splat)

            if not new_splats:
                break

            if verbose:
                logger.info(
                    "Residual detail pass %s: adding %s small splats (%s edge candidates)",
                    pass_idx + 1,
                    len(new_splats),
                    edge_candidates_used,
                )

            current_splats = current_splats + new_splats
            residual_iters = int(
                max(0, self.refinement_config.get("residual_detail_iters", 8))
            )
            current_splats, stage_metric, stage_rendered = self._optimize_stage(
                splats=current_splats,
                target=target,
                renderer=renderer,
                loss_fn=loss_fn,
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
            stage_metric["residual_detail_added"] = len(new_splats)
            stage_metric["residual_detail_edge_candidates"] = int(edge_candidates_used)
            stage_metric["residual_detail_anisotropic_edge_splats"] = int(
                anisotropic_edge_splats
            )
            stage_metric["residual_detail_edge_fraction"] = float(edge_fraction)
            stage_metric["residual_detail_edge_percentile"] = float(edge_percentile)
            stage_metric["residual_detail_elapsed_sec"] = float(
                time.perf_counter() - pass_t0
            )
            remaining = self._time_budget_seconds_remaining()
            if remaining is not None:
                stage_metric["time_budget_remaining_sec"] = max(0.0, float(remaining))
                stage_metric["time_budget_exhausted"] = bool(
                    self._time_budget_exhausted()
                )
            residual_metrics.append(stage_metric)
            if verbose:
                logger.info(
                    "Residual detail pass %s done in %.2fs: SSIM_sRGB=%.4f, splats=%s",
                    pass_idx + 1,
                    stage_metric["residual_detail_elapsed_sec"],
                    float(stage_metric.get("ssim_srgb", 0.0)),
                    len(current_splats),
                )

        return current_splats, residual_metrics

    def _postfit_splats_for_svg_proxy(
        self,
        splats: List[GaussianSplat],
        image: np.ndarray,
        width: int,
        height: int,
        num_iters: int,
        verbose: bool,
    ) -> Tuple[List[GaussianSplat], Dict[str, Any]]:
        """Post-fit color/alpha against a browser-like SVG compositing proxy."""
        if not splats or num_iters <= 0:
            return splats, {
                "stage": -2,
                "stage_type": "svg_proxy_postfit",
                "iterations": 0,
                "splat_count": len(splats),
            }

        base = splats_to_tensor(splats, device=self.device)
        target_linear = torch.from_numpy(image[:, :, :3]).to(self.device)
        target_srgb = torch_linear_to_srgb(target_linear)
        renderer = create_renderer(
            backend=self.renderer_backend,
            width=width,
            height=height,
            device=self.device,
            tile_size=int(
                np.clip(self.refinement_config.get("renderer_tile_size", 16), 4, 128)
            ),
            blend_mode="alpha-over",
            background_color=self._background_linear_rgb,
            compositing_space="srgb",
            tile_bin_rebuild_interval=int(
                max(
                    1,
                    self.refinement_config.get("renderer_tile_bin_rebuild_interval", 1),
                )
            ),
            tile_bin_padding=float(
                max(0.0, self.refinement_config.get("renderer_tile_bin_padding", 0.0))
            ),
            batch_tile_count=int(
                max(1, self.refinement_config.get("renderer_batch_tile_count", 32))
            ),
            max_active_splats_per_tile=(
                None
                if self.refinement_config.get("renderer_max_active_splats_per_tile")
                in (None, "", 0)
                else int(
                    self.refinement_config.get("renderer_max_active_splats_per_tile")
                )
            ),
        )

        safe_mask_np = np.zeros(len(splats), dtype=bool)
        if self._region_background_safe_mask is not None:
            for idx, splat in enumerate(splats):
                x = int(np.clip(round(float(splat.mu[0])), 0, width - 1))
                y = int(np.clip(round(float(splat.mu[1])), 0, height - 1))
                is_safe = bool(self._region_background_safe_mask[y, x])
                if self._region_foreground_mask is not None and bool(
                    self._region_foreground_mask[y, x]
                ):
                    is_safe = False
                if self._region_edge_band_mask is not None and bool(
                    self._region_edge_band_mask[y, x]
                ):
                    is_safe = False
                safe_mask_np[idx] = is_safe
        safe_mask = torch.from_numpy(safe_mask_np).to(self.device)

        init_color = torch.clamp(base[:, 6:9], 1e-4, 1.0 - 1e-4)
        init_alpha = torch.clamp(base[:, 9], 1e-4, 1.0 - 1e-4)
        color_logits = torch.nn.Parameter(torch.logit(init_color))
        alpha_logits = torch.nn.Parameter(torch.logit(init_alpha).unsqueeze(-1))
        optimizer = torch.optim.Adam(
            [
                {
                    "params": [color_logits],
                    "lr": float(
                        self.refinement_config.get("svg_proxy_postfit_color_lr", 0.035)
                    ),
                },
                {
                    "params": [alpha_logits],
                    "lr": float(
                        self.refinement_config.get("svg_proxy_postfit_alpha_lr", 0.020)
                    ),
                },
            ]
        )

        init_effective_alpha = torch.where(
            safe_mask,
            init_alpha * float(SVG_BACKGROUND_ALPHA_CAP),
            init_alpha,
        )
        best_loss = float("inf")
        best_color: Optional[torch.Tensor] = None
        best_alpha: Optional[torch.Tensor] = None
        start_time = time.time()
        iterations_run = 0
        final_l1 = 0.0
        final_mse = 0.0

        for iteration in range(int(num_iters)):
            if self._time_budget_exhausted():
                break
            iterations_run = iteration + 1
            optimizer.zero_grad(set_to_none=True)
            color = torch.sigmoid(color_logits)
            raw_alpha = torch.sigmoid(alpha_logits).squeeze(-1)
            effective_alpha = torch.where(
                safe_mask,
                raw_alpha * float(SVG_BACKGROUND_ALPHA_CAP),
                raw_alpha,
            )

            fitted = base.clone()
            fitted[:, 6:9] = color
            fitted[:, 9] = effective_alpha
            rendered_srgb = torch_linear_to_srgb(renderer(fitted))
            l1 = torch.mean(torch.abs(rendered_srgb - target_srgb))
            mse = torch.mean((rendered_srgb - target_srgb) ** 2)
            color_reg = torch.mean(torch.abs(color - init_color))
            alpha_reg = torch.mean(torch.abs(effective_alpha - init_effective_alpha))
            safe_alpha_mean = (
                torch.mean(effective_alpha[safe_mask])
                if bool(torch.any(safe_mask))
                else torch.tensor(0.0, device=self.device)
            )
            loss = (
                l1
                + 0.35 * mse
                + float(
                    self.refinement_config.get("svg_proxy_postfit_color_reg", 0.012)
                )
                * color_reg
                + float(
                    self.refinement_config.get("svg_proxy_postfit_alpha_reg", 0.008)
                )
                * alpha_reg
                + float(
                    self.refinement_config.get(
                        "svg_proxy_postfit_safe_alpha_reg", 0.005
                    )
                )
                * safe_alpha_mean
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_([color_logits, alpha_logits], max_norm=1.0)
            optimizer.step()

            loss_value = float(loss.item())
            final_l1 = float(l1.item())
            final_mse = float(mse.item())
            if loss_value < best_loss:
                best_loss = loss_value
                best_color = color.detach().clone()
                best_alpha = effective_alpha.detach().clone()
            if verbose and (iteration + 1) % 20 == 0:
                logger.info(
                    "  SVG proxy post-fit %s/%s: loss=%.6f l1=%.6f",
                    iteration + 1,
                    num_iters,
                    loss_value,
                    final_l1,
                )

        if best_color is None or best_alpha is None:
            return splats, {
                "stage": -2,
                "stage_type": "svg_proxy_postfit",
                "iterations": int(iterations_run),
                "splat_count": len(splats),
                "best_loss": float(best_loss),
                "runtime_sec": float(time.time() - start_time),
            }

        output_tensor = base.clone()
        output_tensor[:, 6:9] = best_color
        output_tensor[:, 9] = best_alpha
        fitted_splats = self._copy_splat_layers(
            splats,
            tensor_to_splats(output_tensor.detach()),
        )
        return fitted_splats, {
            "stage": -2,
            "stage_type": "svg_proxy_postfit",
            "iterations": int(iterations_run),
            "splat_count": len(fitted_splats),
            "best_loss": float(best_loss),
            "final_l1_srgb": float(final_l1),
            "final_mse_srgb": float(final_mse),
            "safe_background_splats": int(np.count_nonzero(safe_mask_np)),
            "runtime_sec": float(time.time() - start_time),
        }

    def _postfit_splats_for_pptx_proxy(
        self,
        splats: List[GaussianSplat],
        image: np.ndarray,
        width: int,
        height: int,
        num_iters: int,
        verbose: bool,
    ) -> Tuple[List[GaussianSplat], Dict[str, Any]]:
        """Post-fit color/alpha against a PowerPoint soft-edge approximation."""
        if not splats or num_iters <= 0:
            return splats, {
                "stage": -2,
                "stage_type": "pptx_proxy_postfit",
                "iterations": 0,
                "splat_count": len(splats),
            }

        base = splats_to_tensor(splats, device=self.device)
        target_linear = torch.from_numpy(image[:, :, :3]).to(self.device)
        target_srgb = torch_linear_to_srgb(target_linear)
        renderer = create_renderer(
            backend=self.renderer_backend,
            width=width,
            height=height,
            device=self.device,
            tile_size=int(
                np.clip(self.refinement_config.get("renderer_tile_size", 16), 4, 128)
            ),
            blend_mode="alpha-over",
            background_color=self._background_linear_rgb,
            compositing_space="srgb",
            tile_bin_rebuild_interval=int(
                max(
                    1,
                    self.refinement_config.get("renderer_tile_bin_rebuild_interval", 1),
                )
            ),
            tile_bin_padding=float(
                max(0.0, self.refinement_config.get("renderer_tile_bin_padding", 0.0))
            ),
            batch_tile_count=int(
                max(1, self.refinement_config.get("renderer_batch_tile_count", 32))
            ),
            max_active_splats_per_tile=(
                None
                if self.refinement_config.get("renderer_max_active_splats_per_tile")
                in (None, "", 0)
                else int(
                    self.refinement_config.get("renderer_max_active_splats_per_tile")
                )
            ),
        )

        if self._region_weight_map is not None:
            pixel_weights = torch.from_numpy(
                self._region_weight_map.astype(np.float32)
            ).to(self.device)
        else:
            pixel_weights = torch.ones(
                (height, width), dtype=torch.float32, device=self.device
            )
        pixel_weights = pixel_weights / torch.clamp(torch.mean(pixel_weights), min=1e-6)
        pixel_weights3 = pixel_weights.unsqueeze(-1)

        def weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
            return torch.sum(values * weights) / torch.clamp(
                torch.sum(weights), min=1e-8
            )

        def weighted_std(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
            mean = weighted_mean(values, weights)
            variance = weighted_mean((values - mean).pow(2), weights)
            return torch.sqrt(torch.clamp(variance, min=1e-8))

        def srgb_luminance(values: torch.Tensor) -> torch.Tensor:
            return (
                0.2126 * values[..., 0]
                + 0.7152 * values[..., 1]
                + 0.0722 * values[..., 2]
            )

        def srgb_saturation(values: torch.Tensor) -> torch.Tensor:
            maxc = torch.max(values, dim=-1).values
            minc = torch.min(values, dim=-1).values
            return torch.where(
                maxc > 1e-6,
                (maxc - minc) / torch.clamp(maxc, min=1e-6),
                torch.zeros_like(maxc),
            )

        def luminance_gradient_l1(
            rendered_luma: torch.Tensor, target_luma: torch.Tensor
        ) -> torch.Tensor:
            dx = torch.abs(
                (rendered_luma[:, 1:] - rendered_luma[:, :-1])
                - (target_luma[:, 1:] - target_luma[:, :-1])
            )
            dy = torch.abs(
                (rendered_luma[1:, :] - rendered_luma[:-1, :])
                - (target_luma[1:, :] - target_luma[:-1, :])
            )
            wx = 0.5 * (pixel_weights[:, 1:] + pixel_weights[:, :-1])
            wy = 0.5 * (pixel_weights[1:, :] + pixel_weights[:-1, :])
            return weighted_mean(dx, wx) + weighted_mean(dy, wy)

        target_luma = srgb_luminance(target_srgb)
        target_sat = srgb_saturation(target_srgb)
        target_luma_std = weighted_std(target_luma, pixel_weights).detach()
        target_sat_mean = weighted_mean(target_sat, pixel_weights).detach()
        target_sat_std = weighted_std(target_sat, pixel_weights).detach()

        init_color = torch.clamp(base[:, 6:9], 1e-4, 1.0 - 1e-4)
        init_alpha = torch.clamp(base[:, 9], 1e-4, 1.0 - 1e-4)
        color_logits = torch.nn.Parameter(torch.logit(init_color))
        alpha_logits = torch.nn.Parameter(torch.logit(init_alpha).unsqueeze(-1))
        optimizer = torch.optim.Adam(
            [
                {
                    "params": [color_logits],
                    "lr": float(
                        self.refinement_config.get("pptx_proxy_postfit_color_lr", 0.040)
                    ),
                },
                {
                    "params": [alpha_logits],
                    "lr": float(
                        self.refinement_config.get("pptx_proxy_postfit_alpha_lr", 0.030)
                    ),
                },
            ]
        )

        default_alpha_scale = (
            PPTX_GRADIENT_ALPHA_SCALE
            if self.pptx_splat_style == "gradient"
            else PPTX_SOFT_EDGE_ALPHA_SCALE
        )
        default_sigma_scale = (
            1.0 if self.pptx_splat_style == "gradient" else PPTX_SOFT_EDGE_K_SIGMA_SCALE
        )
        alpha_scale = float(
            self.refinement_config.get(
                "pptx_proxy_postfit_alpha_scale", default_alpha_scale
            )
        )
        sigma_scale = float(
            self.refinement_config.get(
                "pptx_proxy_postfit_sigma_scale", default_sigma_scale
            )
        )
        alpha_scale = float(np.clip(alpha_scale, 1e-4, 1.0))
        sigma_scale = float(np.clip(sigma_scale, 0.25, 3.0))

        def pptx_effective_alpha(raw_alpha: torch.Tensor) -> torch.Tensor:
            center_opacity = (
                1.0 - torch.exp(-torch.clamp(raw_alpha, 0.0, 1.0))
            ) * alpha_scale
            center_opacity = torch.clamp(center_opacity, 0.0, 1.0 - 1e-5)
            return -torch.log1p(-center_opacity)

        init_effective_alpha = pptx_effective_alpha(init_alpha)
        best_loss = float("inf")
        best_color: Optional[torch.Tensor] = None
        best_alpha: Optional[torch.Tensor] = None
        best_luma_std = 0.0
        best_sat_mean = 0.0
        start_time = time.time()
        iterations_run = 0
        final_l1 = 0.0
        final_mse = 0.0
        final_gradient_l1 = 0.0

        for iteration in range(int(num_iters)):
            if self._time_budget_exhausted():
                break
            iterations_run = iteration + 1
            optimizer.zero_grad(set_to_none=True)
            color = torch.sigmoid(color_logits)
            raw_alpha = torch.sigmoid(alpha_logits).squeeze(-1)
            effective_alpha = pptx_effective_alpha(raw_alpha)

            fitted = base.clone()
            fitted[:, 2:4] = torch.clamp(fitted[:, 2:4] * sigma_scale, min=1e-4)
            fitted[:, 6:9] = color
            fitted[:, 9] = effective_alpha
            rendered_srgb = torch_linear_to_srgb(renderer(fitted))
            diff = rendered_srgb - target_srgb
            l1 = torch.sum(torch.abs(diff) * pixel_weights3) / torch.clamp(
                torch.sum(pixel_weights3) * 3.0, min=1e-8
            )
            mse = torch.sum(diff.pow(2) * pixel_weights3) / torch.clamp(
                torch.sum(pixel_weights3) * 3.0, min=1e-8
            )

            rendered_luma = srgb_luminance(rendered_srgb)
            rendered_sat = srgb_saturation(rendered_srgb)
            rendered_luma_std = weighted_std(rendered_luma, pixel_weights)
            rendered_sat_mean = weighted_mean(rendered_sat, pixel_weights)
            rendered_sat_std = weighted_std(rendered_sat, pixel_weights)
            contrast_loss = torch.abs(rendered_luma_std - target_luma_std)
            saturation_loss = torch.abs(
                rendered_sat_mean - target_sat_mean
            ) + 0.5 * torch.abs(rendered_sat_std - target_sat_std)
            gradient_l1 = luminance_gradient_l1(rendered_luma, target_luma)
            color_reg = torch.mean(torch.abs(color - init_color))
            alpha_reg = torch.mean(torch.abs(effective_alpha - init_effective_alpha))
            loss = (
                l1
                + 0.35 * mse
                + float(
                    self.refinement_config.get(
                        "pptx_proxy_postfit_contrast_weight", 0.35
                    )
                )
                * contrast_loss
                + float(
                    self.refinement_config.get(
                        "pptx_proxy_postfit_saturation_weight", 0.18
                    )
                )
                * saturation_loss
                + float(
                    self.refinement_config.get(
                        "pptx_proxy_postfit_gradient_weight", 0.10
                    )
                )
                * gradient_l1
                + float(
                    self.refinement_config.get("pptx_proxy_postfit_color_reg", 0.010)
                )
                * color_reg
                + float(
                    self.refinement_config.get("pptx_proxy_postfit_alpha_reg", 0.006)
                )
                * alpha_reg
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_([color_logits, alpha_logits], max_norm=1.0)
            optimizer.step()

            loss_value = float(loss.item())
            final_l1 = float(l1.item())
            final_mse = float(mse.item())
            final_gradient_l1 = float(gradient_l1.item())
            if loss_value < best_loss:
                best_loss = loss_value
                best_color = color.detach().clone()
                best_alpha = raw_alpha.detach().clone()
                best_luma_std = float(rendered_luma_std.item())
                best_sat_mean = float(rendered_sat_mean.item())
            if verbose and (iteration + 1) % 20 == 0:
                logger.info(
                    "  PPTX proxy post-fit %s/%s: loss=%.6f l1=%.6f contrast=%.6f sat=%.6f",
                    iteration + 1,
                    num_iters,
                    loss_value,
                    final_l1,
                    float(contrast_loss.item()),
                    float(saturation_loss.item()),
                )

        if best_color is None or best_alpha is None:
            return splats, {
                "stage": -2,
                "stage_type": "pptx_proxy_postfit",
                "iterations": int(iterations_run),
                "splat_count": len(splats),
                "best_loss": float(best_loss),
                "runtime_sec": float(time.time() - start_time),
            }

        output_tensor = base.clone()
        output_tensor[:, 6:9] = best_color
        output_tensor[:, 9] = best_alpha
        fitted_splats = self._copy_splat_layers(
            splats,
            tensor_to_splats(output_tensor.detach()),
        )
        return fitted_splats, {
            "stage": -2,
            "stage_type": "pptx_proxy_postfit",
            "iterations": int(iterations_run),
            "splat_count": len(fitted_splats),
            "best_loss": float(best_loss),
            "final_l1_srgb": float(final_l1),
            "final_mse_srgb": float(final_mse),
            "final_gradient_l1_srgb": float(final_gradient_l1),
            "target_luminance_std_srgb": float(target_luma_std.item()),
            "proxy_luminance_std_srgb": float(best_luma_std),
            "target_saturation_mean_srgb": float(target_sat_mean.item()),
            "proxy_saturation_mean_srgb": float(best_sat_mean),
            "alpha_scale": float(alpha_scale),
            "sigma_scale": float(sigma_scale),
            "pptx_splat_style": self.pptx_splat_style,
            "runtime_sec": float(time.time() - start_time),
        }

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
        if precomputed_coverage_map is not None and precomputed_coverage_map.shape == (
            height,
            width,
        ):
            coverage_map = precomputed_coverage_map
        else:
            coverage_map = self._build_alpha_coverage_map(
                splats=splats, width=width, height=height
            )
        uncovered_map = np.clip(1.0 - coverage_map, 0.0, 1.0).astype(np.float32)

        combined_scores: List[Tuple[float, GaussianSplat]] = []
        w_alpha = float(self.refinement_config.get("prune_weight_contribution", 0.45))
        w_residual = float(self.refinement_config.get("prune_weight_residual", 0.35))
        w_uncovered = float(
            max(self.refinement_config.get("prune_weight_uncovered", 0.20), 0.0)
        )
        weight_sum = max(w_alpha + w_residual + w_uncovered, 1e-8)
        sample_radius_scale = float(
            max(self.refinement_config.get("prune_sample_radius", 1.4), 0.8)
        )
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

            local_error = (
                float(np.mean(error_norm[y0:y1, x0:x1])) if x0 < x1 and y0 < y1 else 0.0
            )
            local_uncovered = (
                float(np.mean(uncovered_map[y0:y1, x0:x1]))
                if x0 < x1 and y0 < y1
                else 0.0
            )
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
        coverage_map = self._build_alpha_coverage_map(
            splats=splats, width=width, height=height
        )
        coverage_ratio = self._compute_coverage_ratio(coverage_map)
        min_final_coverage = float(
            np.clip(self.refinement_config.get("coverage_target", 0.985), 0.0, 1.0)
        )

        # If we are saturated at max_splats, reclaim budget from low-value splats.
        if coverage_ratio < min_final_coverage and len(splats) >= self.max_splats:
            edge_map = self._build_edge_map(image)
            reallocate_fraction = float(
                np.clip(
                    self.refinement_config.get(
                        "reallocate_for_coverage_fraction", 0.08
                    ),
                    0.0,
                    0.30,
                )
            )
            reallocate_budget = int(
                min(
                    len(splats) // 4, max(1, np.ceil(len(splats) * reallocate_fraction))
                )
            )
            ranked: List[Tuple[float, int]] = []
            for idx, splat in enumerate(splats):
                x = int(np.clip(round(float(splat.mu[0])), 0, width - 1))
                y = int(np.clip(round(float(splat.mu[1])), 0, height - 1))
                local_uncovered = float(np.clip(1.0 - coverage_map[y, x], 0.0, 1.0))
                edge_value = float(edge_map[y, x])
                alpha_value = float(np.clip(splat.alpha, 0.0, 1.0))
                keep_score = (
                    0.40 * alpha_value + 0.40 * local_uncovered + 0.20 * edge_value
                )
                ranked.append((keep_score, idx))
            ranked.sort(key=lambda pair: pair[0])
            drop_indices = {idx for _, idx in ranked[:reallocate_budget]}
            if drop_indices:
                splats = [s for idx, s in enumerate(splats) if idx not in drop_indices]
                coverage_map = self._build_alpha_coverage_map(
                    splats=splats, width=width, height=height
                )
                coverage_ratio = self._compute_coverage_ratio(coverage_map)

        final_fill_budget = int(
            max(
                0,
                min(
                    self.max_splats - len(splats),
                    np.ceil(
                        self.max_splats
                        * float(self.refinement_config.get("final_fill_fraction", 0.10))
                    ),
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
                sampled_idx = rng.choice(
                    len(x_indices), size=sample_count, replace=False, p=weights
                )
                sigma_fill = float(
                    np.clip(
                        self.refinement_config.get("coverage_sigma_max", 6.0),
                        self.refinement_config.get("sigma_min", 0.5),
                        20.0,
                    )
                )
                alpha_fill = float(
                    np.clip(
                        self.refinement_config.get(
                            "coverage_alpha_fill",
                            self.refinement_config.get("alpha_base", 0.3),
                        ),
                        self.refinement_config.get("alpha_min", 0.05),
                        self.refinement_config.get("alpha_max", 0.95),
                    )
                )
                for idx in sampled_idx:
                    x = int(x_indices[idx])
                    y = int(y_indices[idx])
                    x_center = float(
                        np.clip(x + rng.uniform(-0.5, 0.5), 0.0, width - 1.0)
                    )
                    y_center = float(
                        np.clip(y + rng.uniform(-0.5, 0.5), 0.0, height - 1.0)
                    )
                    color = estimate_local_color(image, x, y)
                    splat = create_isotropic_splat(
                        center=np.array([x_center, y_center], dtype=np.float32),
                        sigma=sigma_fill,
                        color=color,
                        alpha=alpha_fill,
                    )
                    self._assign_splat_layer(splat, LAYER_BASE, 0.05)
                    splats.append(splat)

            coverage_map = self._build_alpha_coverage_map(
                splats=splats, width=width, height=height
            )
            coverage_ratio = self._compute_coverage_ratio(coverage_map)

        logger.info(
            "Post-processing: %s splats remaining (coverage=%.1f%%)",
            len(splats),
            coverage_ratio * 100.0,
        )
        return splats

    def _generate_svg(
        self, splats: List[GaussianSplat], width: int, height: int
    ) -> str:
        """Generate SVG content."""
        from .io import generate_svg_content

        return generate_svg_content(
            splats,
            width,
            height,
            self.k_sigma,
            background_linear_rgb=self._background_linear_rgb,
            export_recipe=self.svg_export_recipe,
            foreground_mask=self._region_foreground_mask,
            background_safe_mask=self._region_background_safe_mask,
            edge_band_mask=self._region_edge_band_mask,
        )

    def _generate_drawingml(
        self, splats: List[GaussianSplat], width: int, height: int
    ) -> str:
        """Generate DrawingML slide XML content."""
        return generate_drawingml_slide_content(
            splats,
            width,
            height,
            self.k_sigma,
            background_linear_rgb=self._background_linear_rgb,
            splat_style=self.pptx_splat_style,
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
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2, sort_keys=True)

    def _write_manifest(
        self, artifacts_dir: Optional[Path], manifest: Dict[str, Any]
    ) -> None:
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

    def _normalize_map(self, values: np.ndarray) -> np.ndarray:
        """Normalize array to [0, 1]."""
        min_v = float(np.min(values))
        max_v = float(np.max(values))
        if max_v <= min_v + 1e-12:
            return np.zeros_like(values, dtype=np.float32)
        return ((values - min_v) / (max_v - min_v)).astype(np.float32)

    def _normalize_percentile_map(
        self,
        values: np.ndarray,
        lower: float = 1.0,
        upper: float = 99.0,
    ) -> np.ndarray:
        """Normalize to [0, 1] after clipping extreme percentiles."""
        arr = np.asarray(values, dtype=np.float32)
        lo = float(np.percentile(arr, np.clip(lower, 0.0, 100.0)))
        hi = float(np.percentile(arr, np.clip(upper, 0.0, 100.0)))
        if hi <= lo + 1e-12:
            return np.zeros_like(arr, dtype=np.float32)
        return np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)

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
        bottom = rgb[max(height - border, 0) :, :, :].reshape(-1, 3)
        left = rgb[:, :border, :].reshape(-1, 3)
        right = rgb[:, max(width - border, 0) :, :].reshape(-1, 3)
        border_pixels = np.concatenate([top, bottom, left, right], axis=0)

        if image.shape[2] >= 4:
            alpha = np.asarray(image[:, :, 3], dtype=np.float32)
            top_a = alpha[:border, :].reshape(-1)
            bottom_a = alpha[max(height - border, 0) :, :].reshape(-1)
            left_a = alpha[:, :border].reshape(-1)
            right_a = alpha[:, max(width - border, 0) :].reshape(-1)
            border_alpha = np.concatenate([top_a, bottom_a, left_a, right_a], axis=0)
            valid = border_alpha > 0.02
            if np.any(valid):
                border_pixels = border_pixels[valid]

        if border_pixels.size == 0:
            border_pixels = rgb.reshape(-1, 3)
        border_std = float(np.mean(np.std(border_pixels, axis=0)))
        max_uniform_std = float(
            self.refinement_config.get("background_uniformity_std_max", 0.18)
        )
        if border_std > max_uniform_std:
            return np.zeros(3, dtype=np.float32)
        background = np.median(border_pixels, axis=0).astype(np.float32)
        if not np.isfinite(background).all():
            return np.zeros(3, dtype=np.float32)
        return np.clip(background, 0.0, 1.0)

    def _estimate_border_median_color(self, image: np.ndarray) -> np.ndarray:
        """Estimate border median without rejecting non-uniform photo borders."""
        if image.ndim != 3 or image.shape[2] < 3:
            return np.zeros(3, dtype=np.float32)
        rgb = np.asarray(image[:, :, :3], dtype=np.float32)
        height, width = rgb.shape[:2]
        border = max(1, int(round(0.04 * float(min(height, width)))))
        border_pixels = np.concatenate(
            [
                rgb[:border, :, :].reshape(-1, 3),
                rgb[max(height - border, 0) :, :, :].reshape(-1, 3),
                rgb[:, :border, :].reshape(-1, 3),
                rgb[:, max(width - border, 0) :, :].reshape(-1, 3),
            ],
            axis=0,
        )
        if image.shape[2] >= 4:
            alpha = np.asarray(image[:, :, 3], dtype=np.float32)
            border_alpha = np.concatenate(
                [
                    alpha[:border, :].reshape(-1),
                    alpha[max(height - border, 0) :, :].reshape(-1),
                    alpha[:, :border].reshape(-1),
                    alpha[:, max(width - border, 0) :].reshape(-1),
                ],
                axis=0,
            )
            valid = border_alpha > 0.02
            if np.any(valid):
                border_pixels = border_pixels[valid]
        if border_pixels.size == 0:
            border_pixels = rgb.reshape(-1, 3)
        background = np.median(border_pixels, axis=0).astype(np.float32)
        if not np.isfinite(background).all():
            return np.zeros(3, dtype=np.float32)
        return np.clip(background, 0.0, 1.0)

    def _compute_background_suppressed_priority(
        self,
        lightness: np.ndarray,
        saliency: np.ndarray,
        foreground: np.ndarray,
        edge_strength: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Build a pure image-statistics detail prior that discounts border background texture."""
        from scipy import ndimage as ndi
        from skimage.filters import gaussian, sobel
        from skimage.morphology import (
            binary_closing,
            binary_dilation,
            binary_opening,
            disk,
            remove_small_objects,
        )

        light = np.asarray(lightness, dtype=np.float32)
        sal = np.asarray(saliency, dtype=np.float32)
        fg = np.asarray(foreground, dtype=bool)
        edge = np.asarray(edge_strength, dtype=np.float32)
        height, width = sal.shape
        total_pixels = max(1, int(height * width))

        grad = self._normalize_percentile_map(sobel(light), lower=1.0, upper=99.0)
        lap = np.abs(ndi.laplace(gaussian(light, sigma=0.6))).astype(np.float32)
        lap_local = self._normalize_percentile_map(
            gaussian(lap, sigma=1.2), lower=1.0, upper=99.5
        )
        focus = self._normalize_map(0.55 * lap_local + 0.45 * grad)
        focus = self._normalize_map(gaussian(focus, sigma=2.0))

        low_saliency_pct = float(
            np.clip(
                self.refinement_config.get(
                    "background_suppressed_saliency_low_percentile", 68.0
                ),
                0.0,
                100.0,
            )
        )
        low_focus_pct = float(
            np.clip(
                self.refinement_config.get(
                    "background_suppressed_focus_low_percentile", 62.0
                ),
                0.0,
                100.0,
            )
        )
        low_saliency = sal < float(np.percentile(sal, low_saliency_pct))
        low_focus = focus < float(np.percentile(focus, low_focus_pct))
        background_corridor = np.asarray(low_saliency | low_focus, dtype=bool)

        border_seed = np.zeros((height, width), dtype=bool)
        border = max(1, int(round(0.035 * float(min(height, width)))))
        border_seed[:border, :] = True
        border_seed[max(height - border, 0) :, :] = True
        border_seed[:, :border] = True
        border_seed[:, max(width - border, 0) :] = True
        border_background = ndi.binary_propagation(
            border_seed, mask=background_corridor
        )
        border_background = binary_dilation(
            np.asarray(border_background, dtype=bool), disk(3)
        )
        background_penalty = self._normalize_map(
            gaussian(border_background.astype(np.float32), sigma=5.0)
        )

        saliency_cut = float(
            np.percentile(
                sal,
                np.clip(
                    self.refinement_config.get(
                        "background_suppressed_subject_saliency_percentile", 72.0
                    ),
                    0.0,
                    100.0,
                ),
            )
        )
        edge_cut = float(
            np.percentile(
                edge,
                np.clip(
                    self.refinement_config.get(
                        "background_suppressed_subject_edge_percentile", 80.0
                    ),
                    0.0,
                    100.0,
                ),
            )
        )
        focus_cut = float(
            np.percentile(
                focus,
                np.clip(
                    self.refinement_config.get(
                        "background_suppressed_subject_focus_percentile", 55.0
                    ),
                    0.0,
                    100.0,
                ),
            )
        )
        candidate = np.asarray(
            (sal >= saliency_cut) | ((edge >= edge_cut) & (focus >= focus_cut)) | fg,
            dtype=bool,
        )
        candidate = binary_closing(candidate, disk(2))
        candidate = binary_opening(candidate, disk(1))
        candidate = remove_small_objects(
            candidate, min_size=max(8, int(total_pixels * 0.0025))
        )
        candidate = np.asarray(candidate, dtype=bool)

        labels, label_count = ndi.label(candidate)
        keep = np.zeros((height, width), dtype=bool)
        min_component_area = max(24, int(total_pixels * 0.0025))
        border_keep_score = float(
            np.clip(
                self.refinement_config.get(
                    "background_suppressed_border_component_keep_score", 0.42
                ),
                0.0,
                1.0,
            )
        )
        component_score_map = self._normalize_map(
            0.45 * sal + 0.35 * focus + 0.20 * edge
        )
        for label_id in range(1, int(label_count) + 1):
            component = labels == label_id
            area = int(np.count_nonzero(component))
            if area < min_component_area:
                continue
            touches_border = bool(
                np.any(component[:border, :])
                or np.any(component[max(height - border, 0) :, :])
                or np.any(component[:, :border])
                or np.any(component[:, max(width - border, 0) :])
            )
            component_score = (
                float(np.mean(component_score_map[component])) if area else 0.0
            )
            if touches_border and component_score < border_keep_score:
                continue
            keep[component] = True

        if not np.any(keep) and np.any(fg):
            keep = np.asarray(fg, dtype=bool)

        center_score = self._normalize_map(gaussian(keep.astype(np.float32), sigma=8.0))
        subject_prior = self._normalize_map(
            0.55 * keep.astype(np.float32)
            + 0.30 * center_score
            + 0.15 * fg.astype(np.float32)
        )
        if float(np.max(subject_prior) - np.min(subject_prior)) <= 1e-8:
            subject_prior = self._normalize_map(
                0.65 * fg.astype(np.float32) + 0.35 * sal
            )

        penalty_strength = float(
            np.clip(
                self.refinement_config.get(
                    "background_suppressed_saliency_penalty_strength", 0.65
                ),
                0.0,
                1.0,
            )
        )
        edge_gate = float(
            np.clip(
                self.refinement_config.get(
                    "background_suppressed_saliency_edge_gate", 0.80
                ),
                0.0,
                1.0,
            )
        )
        focus_gate = float(
            np.clip(
                self.refinement_config.get(
                    "background_suppressed_saliency_focus_gate", 0.80
                ),
                0.0,
                1.0,
            )
        )
        background_keep = np.clip(
            1.0 - penalty_strength * background_penalty, 0.0, 1.0
        ).astype(np.float32)
        gated_edges = self._normalize_map(
            edge
            * background_keep
            * (1.0 - edge_gate + edge_gate * subject_prior)
            * (1.0 - focus_gate + focus_gate * focus)
        )
        gated_saliency = self._normalize_map(
            (0.40 * sal + 0.35 * focus + 0.25 * subject_prior) * background_keep
        )
        detail_priority = self._normalize_map(
            (0.45 * gated_saliency + 0.55 * gated_edges) * (0.35 + 0.65 * subject_prior)
        )
        if float(np.max(detail_priority) - np.min(detail_priority)) <= 1e-8:
            detail_priority = self._normalize_map(
                (0.65 * sal + 0.35 * edge) * background_keep
            )

        return {
            "focus_map": focus.astype(np.float32),
            "background_penalty_map": background_penalty.astype(np.float32),
            "subject_mask": keep,
            "subject_prior_map": subject_prior.astype(np.float32),
            "gated_edge_map": gated_edges.astype(np.float32),
            "gated_saliency_map": gated_saliency.astype(np.float32),
            "detail_priority_map": detail_priority.astype(np.float32),
        }

    def _compute_region_guidance(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Build foreground/background/edge masks plus a spatial loss weight map.

        Edges and salient foreground stay at full weight; safe background gets
        a lower weight so flat regions do not dominate optimization or sampling.
        """
        from skimage.feature import canny
        from skimage.filters import gaussian, threshold_otsu
        from skimage.morphology import (
            binary_closing,
            binary_erosion,
            binary_opening,
            disk,
            remove_small_objects,
        )

        if image.ndim != 3 or image.shape[2] < 3:
            height, width = image.shape[:2]
            ones = np.ones((height, width), dtype=np.float32)
            zeros = np.zeros((height, width), dtype=bool)
            zero_map = np.zeros((height, width), dtype=np.float32)
            return {
                "weight_map": ones,
                "saliency_map": zero_map,
                "detail_priority_map": zero_map,
                "focus_map": zero_map,
                "background_penalty_map": zero_map,
                "gated_edge_map": zero_map,
                "gated_saliency_map": zero_map,
                "subject_prior_map": zero_map,
                "subject_mask": zeros,
                "foreground_mask": zeros,
                "background_safe_mask": zeros,
                "edge_band_mask": zeros,
                "background_linear_rgb": np.zeros(3, dtype=np.float32),
                "summary": {
                    "total_pixels": int(height * width),
                    "foreground_pixels": 0,
                    "background_safe_pixels": 0,
                    "edge_band_pixels": 0,
                    "foreground_ratio": 0.0,
                    "background_safe_ratio": 0.0,
                    "edge_band_ratio": 0.0,
                    "weight_min": 1.0,
                    "weight_max": 1.0,
                    "detail_priority_mean": 0.0,
                    "background_penalty_mean": 0.0,
                    "focus_mean": 0.0,
                    "subject_mask_ratio": 0.0,
                },
            }

        rgb = np.asarray(image[:, :, :3], dtype=np.float32)
        height, width = rgb.shape[:2]
        background = self._estimate_border_median_color(image)

        with torch.no_grad():
            lightness = torch_linear_rgb_to_oklab(torch.from_numpy(rgb)).numpy()[
                :, :, 0
            ]

        edge_binary = canny(lightness, sigma=1.4)
        edge_strength = self._normalize_map(
            gaussian(edge_binary.astype(np.float32), sigma=2.0)
        )

        color_distance = self._normalize_map(
            np.linalg.norm(rgb - background.reshape(1, 1, 3), axis=-1)
        )
        dog = self._normalize_map(
            np.abs(gaussian(lightness, sigma=1.0) - gaussian(lightness, sigma=8.0))
        )
        edge_density = self._normalize_map(
            gaussian(edge_binary.astype(np.float32), sigma=6.0)
        )
        saliency = self._normalize_map(
            0.45 * color_distance + 0.35 * dog + 0.20 * edge_density
        )

        if float(np.max(saliency) - np.min(saliency)) <= 1e-8:
            foreground = np.zeros((height, width), dtype=bool)
        else:
            foreground = saliency > float(threshold_otsu(saliency))
            foreground = binary_closing(foreground, disk(3))
            foreground = binary_opening(foreground, disk(2))
            foreground = remove_small_objects(
                foreground,
                min_size=max(1, int(width * height * 0.005)),
            )
            foreground = np.asarray(foreground, dtype=bool)

        background_safe = np.asarray(binary_erosion(~foreground, disk(4)), dtype=bool)
        edge_band = np.asarray(edge_strength > 0.05, dtype=bool)

        base_weight = float(self.refinement_config.get("region_weight_base", 0.70))
        background_weight = float(
            self.refinement_config.get("region_weight_background", 0.25)
        )
        foreground_weight = float(
            self.refinement_config.get("region_weight_foreground", 1.00)
        )
        edge_weight = float(self.refinement_config.get("region_weight_edge", 1.00))
        saliency_boost = float(
            max(0.0, self.refinement_config.get("region_weight_saliency_boost", 0.55))
        )
        saliency_gamma = float(
            max(0.25, self.refinement_config.get("region_weight_saliency_gamma", 0.80))
        )
        weights = np.full(
            (height, width), np.clip(base_weight, 0.0, 10.0), dtype=np.float32
        )
        weights[background_safe & ~edge_band] = np.clip(background_weight, 0.0, 10.0)
        weights[foreground] = np.clip(foreground_weight, 0.0, 10.0)
        weights[edge_band] = np.clip(edge_weight, 0.0, 10.0)
        if saliency_boost > 0.0:
            saliency_prior = np.power(
                np.clip(saliency, 0.0, 1.0), saliency_gamma
            ).astype(np.float32)
            weights = weights * (1.0 + saliency_boost * saliency_prior)

        if bool(
            self.refinement_config.get("background_suppressed_saliency_enabled", False)
        ):
            suppressed = self._compute_background_suppressed_priority(
                lightness=lightness,
                saliency=saliency,
                foreground=foreground,
                edge_strength=edge_strength,
            )
            if bool(
                self.refinement_config.get(
                    "background_suppressed_saliency_use_for_weights", False
                )
            ):
                detail_priority = suppressed["detail_priority_map"]
                background_penalty = suppressed["background_penalty_map"]
                weights = weights * (0.35 + 0.65 * detail_priority)
                weights = weights * np.clip(1.0 - 0.35 * background_penalty, 0.0, 1.0)
                weights = np.clip(weights, 0.0, 10.0).astype(np.float32)
        else:
            suppressed = {
                "detail_priority_map": saliency.astype(np.float32),
                "focus_map": np.zeros((height, width), dtype=np.float32),
                "background_penalty_map": np.zeros((height, width), dtype=np.float32),
                "gated_edge_map": edge_strength.astype(np.float32),
                "gated_saliency_map": saliency.astype(np.float32),
                "subject_prior_map": foreground.astype(np.float32),
                "subject_mask": foreground,
            }

        total_pixels = max(1, int(width * height))
        foreground_pixels = int(np.count_nonzero(foreground))
        background_safe_pixels = int(np.count_nonzero(background_safe))
        edge_band_pixels = int(np.count_nonzero(edge_band))
        subject_mask = np.asarray(suppressed["subject_mask"], dtype=bool)
        detail_priority_map = np.asarray(
            suppressed["detail_priority_map"], dtype=np.float32
        )
        background_penalty_map = np.asarray(
            suppressed["background_penalty_map"], dtype=np.float32
        )
        focus_map = np.asarray(suppressed["focus_map"], dtype=np.float32)

        return {
            "weight_map": weights.astype(np.float32),
            "saliency_map": saliency.astype(np.float32),
            "detail_priority_map": detail_priority_map,
            "focus_map": focus_map,
            "background_penalty_map": background_penalty_map,
            "gated_edge_map": np.asarray(
                suppressed["gated_edge_map"], dtype=np.float32
            ),
            "gated_saliency_map": np.asarray(
                suppressed["gated_saliency_map"], dtype=np.float32
            ),
            "subject_prior_map": np.asarray(
                suppressed["subject_prior_map"], dtype=np.float32
            ),
            "subject_mask": subject_mask,
            "foreground_mask": foreground,
            "background_safe_mask": background_safe,
            "edge_band_mask": edge_band,
            "background_linear_rgb": background,
            "summary": {
                "total_pixels": int(total_pixels),
                "foreground_pixels": foreground_pixels,
                "background_safe_pixels": background_safe_pixels,
                "edge_band_pixels": edge_band_pixels,
                "foreground_ratio": float(foreground_pixels / total_pixels),
                "background_safe_ratio": float(background_safe_pixels / total_pixels),
                "edge_band_ratio": float(edge_band_pixels / total_pixels),
                "weight_min": float(np.min(weights)),
                "weight_max": float(np.max(weights)),
                "weight_mean": float(np.mean(weights)),
                "saliency_min": float(np.min(saliency)),
                "saliency_max": float(np.max(saliency)),
                "saliency_mean": float(np.mean(saliency)),
                "saliency_p90": float(np.percentile(saliency, 90)),
                "saliency_p95": float(np.percentile(saliency, 95)),
                "detail_priority_mean": float(np.mean(detail_priority_map)),
                "detail_priority_p90": float(np.percentile(detail_priority_map, 90)),
                "detail_priority_p95": float(np.percentile(detail_priority_map, 95)),
                "background_penalty_mean": float(np.mean(background_penalty_map)),
                "focus_mean": float(np.mean(focus_map)),
                "subject_mask_pixels": int(np.count_nonzero(subject_mask)),
                "subject_mask_ratio": float(
                    np.count_nonzero(subject_mask) / total_pixels
                ),
                "background_linear_rgb": [
                    float(background[0]),
                    float(background[1]),
                    float(background[2]),
                ],
            },
        }

    def _loss_weight_tensor(self, width: int, height: int) -> Optional[torch.Tensor]:
        """Return region weighting as a tensor when enabled and shape-compatible."""
        if not self.region_weighting_enabled or self._region_weight_map is None:
            return None
        if self._region_weight_map.shape != (int(height), int(width)):
            return None
        return torch.from_numpy(
            self._region_weight_map.astype(np.float32, copy=False)
        ).to(self.device)

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

        if (
            self.region_weighting_enabled
            and self._region_weight_map is not None
            and self._region_weight_map.shape == score_map.shape
        ):
            score_map = (
                np.asarray(score_map, dtype=np.float32) * self._region_weight_map
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

        selected = rng.choice(
            len(x_indices), size=sample_count, replace=False, p=weights
        )
        selected_x = x_indices[selected].astype(np.int32)
        selected_y = y_indices[selected].astype(np.int32)
        selected_scores = score_map[selected_y, selected_x].astype(np.float32)
        return selected_x, selected_y, selected_scores

    def _build_alpha_coverage_map(
        self, splats: List[GaussianSplat], width: int, height: int
    ) -> np.ndarray:
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
        footprint_sigma = float(
            max(self.refinement_config.get("coverage_footprint_sigma", 3.0), 1.0)
        )
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

            dx = np.arange(x0, x1, dtype=np.float32).reshape(1, -1) - cx
            dy = np.arange(y0, y1, dtype=np.float32).reshape(-1, 1) - cy
            cos_t = float(np.cos(theta))
            sin_t = float(np.sin(theta))
            u = cos_t * dx + sin_t * dy
            v = -sin_t * dx + cos_t * dy
            quadratic = (u * u) / (sx * sx) + (v * v) / (sy * sy)
            density = float(max(0.0, splat.alpha)) * np.exp(-0.5 * quadratic)
            layer_alpha = 1.0 - np.exp(-density)
            transmittance[y0:y1, x0:x1] *= np.clip(
                1.0 - layer_alpha.astype(np.float32), 0.0, 1.0
            )

    def _compute_coverage_ratio(self, coverage_map: np.ndarray) -> float:
        """Compute covered-pixel ratio under configured alpha threshold."""
        threshold = float(
            np.clip(self.refinement_config.get("coverage_threshold", 0.03), 0.0, 1.0)
        )
        return float(np.mean(coverage_map >= threshold))

    def _build_contribution_map(
        self, splats: List[GaussianSplat], width: int, height: int
    ) -> np.ndarray:
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
                "loss_weights": {
                    "l1_weight": 1.0,
                    "ssim_weight": 0.24,
                    "gradient_weight": 0.08,
                },
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
                    "edge_init_fraction": 0.30,
                    "edge_init_percentile": 68.0,
                    "edge_init_sigma_min": 0.45,
                    "edge_init_sigma_max": 1.25,
                    "edge_init_sigma_major_scale": 2.20,
                    "edge_init_sigma_major_max": 3.00,
                    "edge_init_alpha_min": 0.30,
                    "edge_init_alpha_max": 0.72,
                    "edge_init_saliency_weight": 0.70,
                    "edge_init_anisotropy_threshold": 1.15,
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
                    "residual_detail_edge_fraction": 0.45,
                    "residual_detail_edge_percentile": 76.0,
                    "residual_detail_edge_gamma": 0.70,
                    "residual_detail_edge_error_floor": 0.20,
                    "residual_detail_edge_sigma_min": 0.10,
                    "residual_detail_edge_sigma_max": 0.70,
                    "residual_detail_edge_sigma_major_max": 1.25,
                    "residual_detail_edge_alpha_boost": 0.06,
                    "residual_detail_edge_color_gain": 1.00,
                    "residual_detail_edge_saliency_weight": 0.85,
                    "residual_detail_edge_make_threshold": 0.20,
                    "residual_detail_edge_anisotropic": True,
                    "residual_detail_edge_anisotropy_threshold": 1.20,
                    "residual_detail_edge_aspect": 2.00,
                    "residual_detail_color_gain": 0.95,
                    "region_weighting_enabled": True,
                    "region_weight_base": 0.70,
                    "region_weight_background": 0.25,
                    "region_weight_foreground": 1.00,
                    "region_weight_edge": 1.00,
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
