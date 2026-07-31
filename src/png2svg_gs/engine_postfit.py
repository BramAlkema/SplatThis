"""Target-aware post-fit, pruning and monotonic selection."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from .export_common import (
    PPTX_GRADIENT_ALPHA_SCALE,
    PPTX_SOFT_EDGE_ALPHA_SCALE,
    PPTX_SOFT_EDGE_K_SIGMA_SCALE,
    SVG_BACKGROUND_ALPHA_CAP,
)
from .features import estimate_local_color
from .proxies import _PPTXProxyLoss
from .quality import compute_quality_metrics
from .renderer import (
    L1SSIMLoss,
    create_renderer,
    render_pixel_runtime_numpy,
    splats_to_tensor,
    tensor_to_splats,
    torch_linear_to_srgb,
)
from .splat import LAYER_BASE, GaussianSplat, create_isotropic_splat

logger = logging.getLogger(__name__)


class ConversionPostfitMixin:
    """Refines and prunes a fitted splat population for deployment."""

    def _build_postfit_renderer(self, width: int, height: int):
        """Standard Gaussian renderer with sRGB compositing — used by both
        SVG-proxy and blur-proxy postfit. Matches what SVG with gradient
        stops AND PPTX-blur output approximate at the per-splat level."""
        cfg = self.refinement_config
        return create_renderer(
            backend=self.renderer_backend,
            width=width,
            height=height,
            device=self.device,
            tile_size=int(np.clip(cfg.get("renderer_tile_size", 16), 4, 128)),
            blend_mode="alpha-over",
            background_color=self._background_linear_rgb,
            compositing_space="srgb",
            tile_bin_rebuild_interval=int(
                max(1, cfg.get("renderer_tile_bin_rebuild_interval", 1))
            ),
            tile_bin_padding=float(max(0.0, cfg.get("renderer_tile_bin_padding", 0.0))),
            batch_tile_count=int(max(1, cfg.get("renderer_batch_tile_count", 32))),
            max_active_splats_per_tile=(
                None
                if cfg.get("renderer_max_active_splats_per_tile") in (None, "", 0)
                else int(cfg["renderer_max_active_splats_per_tile"])
            ),
        )

    def _compute_safe_background_mask(
        self,
        splats: List[GaussianSplat],
        width: int,
        height: int,
    ) -> np.ndarray:
        """Per-splat boolean: is this splat parked in a 'safe background'
        region where we should cap its alpha? Only used by the SVG proxy
        postfit to suppress backdrop bleed-through."""
        mask = np.zeros(len(splats), dtype=bool)
        if self._region_background_safe_mask is None:
            return mask
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
            mask[idx] = is_safe
        return mask

    def _run_color_alpha_postfit(  # noqa: C901
        self,
        splats: List[GaussianSplat],
        image: np.ndarray,
        width: int,
        height: int,
        num_iters: int,
        verbose: bool,
        *,
        stage_type: str,
        color_lr: float,
        alpha_lr: float,
        color_reg_weight: float,
        alpha_reg_weight: float,
        mse_weight: float = 0.35,
        safe_alpha_cap: Optional[float] = None,
        safe_alpha_reg_weight: float = 0.0,
        log_label: str = "post-fit",
    ) -> Tuple[List[GaussianSplat], Dict[str, Any]]:
        """Core color+alpha postfit loop. Both SVG-proxy and blur-proxy
        postfit are thin wrappers around this.

        `safe_alpha_cap` enables the SVG-style backdrop-suppression policy:
        splats in 'safe background' regions get their effective alpha
        clamped to (raw_alpha × safe_alpha_cap). When None, no policy is
        applied (the blur proxy uses this — its compositor doesn't need
        the suppression).
        """
        if not splats or num_iters <= 0:
            return splats, {
                "stage": -2,
                "stage_type": stage_type,
                "iterations": 0,
                "splat_count": len(splats),
            }

        base = splats_to_tensor(splats, device=self.device)
        target_linear = torch.from_numpy(image[:, :, :3]).to(self.device)
        target_srgb = torch_linear_to_srgb(target_linear)
        renderer = self._build_postfit_renderer(width, height)

        use_safe_mask = safe_alpha_cap is not None
        if use_safe_mask:
            safe_mask_np = self._compute_safe_background_mask(splats, width, height)
        else:
            safe_mask_np = np.zeros(len(splats), dtype=bool)
        safe_mask = torch.from_numpy(safe_mask_np).to(self.device)

        init_color = torch.clamp(base[:, 6:9], 1e-4, 1.0 - 1e-4)
        init_alpha = torch.clamp(base[:, 9], 1e-4, 1.0 - 1e-4)
        color_logits = torch.nn.Parameter(torch.logit(init_color))
        alpha_logits = torch.nn.Parameter(torch.logit(init_alpha).unsqueeze(-1))
        optimizer = torch.optim.Adam(
            [
                {"params": [color_logits], "lr": color_lr},
                {"params": [alpha_logits], "lr": alpha_lr},
            ]
        )

        def apply_safe_cap(a: torch.Tensor) -> torch.Tensor:
            if not use_safe_mask:
                return a
            return torch.where(safe_mask, a * float(safe_alpha_cap), a)

        init_effective_alpha = apply_safe_cap(init_alpha)
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
            effective_alpha = apply_safe_cap(raw_alpha)

            fitted = base.clone()
            fitted[:, 6:9] = color
            fitted[:, 9] = effective_alpha
            rendered_srgb = torch_linear_to_srgb(renderer(fitted))
            l1 = torch.mean(torch.abs(rendered_srgb - target_srgb))
            mse = torch.mean((rendered_srgb - target_srgb) ** 2)
            color_reg = torch.mean(torch.abs(color - init_color))
            alpha_reg = torch.mean(torch.abs(effective_alpha - init_effective_alpha))
            loss = (
                l1
                + mse_weight * mse
                + color_reg_weight * color_reg
                + alpha_reg_weight * alpha_reg
            )
            if (
                use_safe_mask
                and safe_alpha_reg_weight > 0.0
                and bool(torch.any(safe_mask))
            ):
                loss = loss + safe_alpha_reg_weight * torch.mean(
                    effective_alpha[safe_mask]
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [color_logits, alpha_logits],
                max_norm=1.0,
            )
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
                    "  %s %s/%s: loss=%.6f l1=%.6f",
                    log_label,
                    iteration + 1,
                    num_iters,
                    loss_value,
                    final_l1,
                )

        result_meta = {
            "stage": -2,
            "stage_type": stage_type,
            "iterations": int(iterations_run),
            "splat_count": len(splats),
            "best_loss": float(best_loss),
            "runtime_sec": float(time.time() - start_time),
        }
        if best_color is None or best_alpha is None:
            return splats, result_meta

        output_tensor = base.clone()
        output_tensor[:, 6:9] = best_color
        output_tensor[:, 9] = best_alpha
        fitted_splats = self._copy_splat_layers(
            splats,
            tensor_to_splats(output_tensor.detach()),
        )
        result_meta.update(
            {
                "splat_count": len(fitted_splats),
                "final_l1_srgb": float(final_l1),
                "final_mse_srgb": float(final_mse),
            }
        )
        if use_safe_mask:
            result_meta["safe_background_splats"] = int(np.count_nonzero(safe_mask_np))
        return fitted_splats, result_meta

    def _postfit_splats_for_svg_proxy(
        self,
        splats: List[GaussianSplat],
        image: np.ndarray,
        width: int,
        height: int,
        num_iters: int,
        verbose: bool,
    ) -> Tuple[List[GaussianSplat], Dict[str, Any]]:
        """Post-fit color/alpha against a browser-like SVG compositing proxy.

        Includes safe-background alpha suppression (SVG_BACKGROUND_ALPHA_CAP)
        to keep background-region splats from bleeding through the gradient
        primitive's "stained-glass" rendering."""
        cfg = self.refinement_config
        return self._run_color_alpha_postfit(
            splats,
            image,
            width,
            height,
            num_iters,
            verbose,
            stage_type="svg_proxy_postfit",
            color_lr=float(cfg.get("svg_proxy_postfit_color_lr", 0.035)),
            alpha_lr=float(cfg.get("svg_proxy_postfit_alpha_lr", 0.020)),
            color_reg_weight=float(cfg.get("svg_proxy_postfit_color_reg", 0.012)),
            alpha_reg_weight=float(cfg.get("svg_proxy_postfit_alpha_reg", 0.008)),
            safe_alpha_cap=float(SVG_BACKGROUND_ALPHA_CAP),
            safe_alpha_reg_weight=float(
                cfg.get("svg_proxy_postfit_safe_alpha_reg", 0.005)
            ),
            log_label="SVG proxy post-fit",
        )

    def _postfit_splats_for_blur_proxy(
        self,
        splats: List[GaussianSplat],
        image: np.ndarray,
        width: int,
        height: int,
        num_iters: int,
        verbose: bool,
    ) -> Tuple[List[GaussianSplat], Dict[str, Any]]:
        """Post-fit color/alpha against a Gaussian-convolution proxy.

        The blur recipe emits each splat as conv(small_disk, Gaussian σ),
        which the standard Gaussian renderer already models exactly. No
        safe-mask alpha suppression — the blur compositor doesn't show the
        stained-glass backdrop bleed-through that the gradient recipe does.
        Lighter regularization than SVG postfit since there's no aggressive
        backdrop-cap policy to balance against."""
        cfg = self.refinement_config
        return self._run_color_alpha_postfit(
            splats,
            image,
            width,
            height,
            num_iters,
            verbose,
            stage_type="blur_proxy_postfit",
            color_lr=float(cfg.get("blur_proxy_postfit_color_lr", 0.030)),
            alpha_lr=float(cfg.get("blur_proxy_postfit_alpha_lr", 0.015)),
            color_reg_weight=float(cfg.get("blur_proxy_postfit_color_reg", 0.004)),
            alpha_reg_weight=float(cfg.get("blur_proxy_postfit_alpha_reg", 0.002)),
            mse_weight=0.30,
            log_label="blur proxy post-fit",
        )

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
        renderer = self._build_postfit_renderer(width, height)

        # One authoritative implementation of the weighted-stat math lives in
        # _PPTXProxyLoss (proxies.py); reuse it as a stats helper instead of
        # re-implementing weighted mean/std, luma, saturation, and the
        # weighted gradient term as drifting local closures. The wrapped
        # base_loss is never invoked here (weights are pre-normalized the
        # same way: w / mean(w)).
        spatial_weight_map = (
            torch.from_numpy(self._region_weight_map.astype(np.float32)).to(self.device)
            if self._region_weight_map is not None
            else None
        )
        stats = _PPTXProxyLoss(
            target_linear_rgb=target_linear,
            base_loss=L1SSIMLoss(ssim_weight=0.0),
            spatial_weight_map=spatial_weight_map,
        ).to(self.device)
        pixel_weights = stats.weights
        pixel_weights3 = pixel_weights.unsqueeze(-1)
        weighted_mean = _PPTXProxyLoss._weighted_mean
        weighted_std = _PPTXProxyLoss._weighted_std
        srgb_luminance = _PPTXProxyLoss._srgb_luminance
        srgb_saturation = _PPTXProxyLoss._srgb_saturation
        luminance_gradient_l1 = stats._luminance_gradient_l1

        target_luma = stats.target_luma
        target_luma_std = stats.target_luma_std
        target_sat_mean = stats.target_sat_mean
        target_sat_std = stats.target_sat_std

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

    def _select_monotonic_canvas_postprocess(
        self,
        optimized_splats: List[GaussianSplat],
        postprocessed_splats: List[GaussianSplat],
        image: np.ndarray,
    ) -> Tuple[List[GaussianSplat], Dict[str, Any]]:
        """Accept canvas post-processing only when deployed-model quality holds.

        Low-alpha splats can be individually weak but collectively essential,
        especially for dense dark images.  The historical fixed alpha cutoff
        could therefore destroy a converged solution after its final metrics
        had already improved.  Score both populations with the exact NumPy
        counterpart of the emitted canvas and revert material regressions.
        """

        before = self._score_canvas_runtime_model(optimized_splats, image)
        candidate = self._score_canvas_runtime_model(postprocessed_splats, image)
        max_ssim_regression = float(
            max(
                0.0,
                self.refinement_config.get(
                    "canvas_postprocess_max_ssim_regression", 5e-4
                ),
            )
        )
        max_psnr_regression = float(
            max(
                0.0,
                self.refinement_config.get(
                    "canvas_postprocess_max_psnr_regression", 0.10
                ),
            )
        )
        accepted = bool(
            float(candidate["ssim_srgb"])
            >= float(before["ssim_srgb"]) - max_ssim_regression
            and float(candidate["psnr_srgb"])
            >= float(before["psnr_srgb"]) - max_psnr_regression
        )
        selected = postprocessed_splats if accepted else optimized_splats
        return selected, {
            "enabled": True,
            "accepted": accepted,
            "decision": "accept" if accepted else "revert",
            "before_count": len(optimized_splats),
            "candidate_count": len(postprocessed_splats),
            "selected_count": len(selected),
            "max_ssim_srgb_regression": max_ssim_regression,
            "max_psnr_srgb_regression": max_psnr_regression,
            "before": {
                "ssim_srgb": float(before["ssim_srgb"]),
                "psnr_srgb": float(before["psnr_srgb"]),
            },
            "candidate": {
                "ssim_srgb": float(candidate["ssim_srgb"]),
                "psnr_srgb": float(candidate["psnr_srgb"]),
            },
        }

    def _score_canvas_runtime_model(
        self,
        splats: List[GaussianSplat],
        image: np.ndarray,
    ) -> Dict[str, float]:
        """Score splats with the exact NumPy counterpart of emitted canvas JS."""

        height, width = image.shape[:2]
        rendered = render_pixel_runtime_numpy(
            splats,
            width=width,
            height=height,
            background_linear_rgb=self._background_linear_rgb,
            compositing_space=self._deployed_compositing_space(),
        )
        return compute_quality_metrics(
            np.asarray(image[:, :, :3], dtype=np.float32),
            rendered,
        )

    def _prefer_canvas_checkpoint(
        self,
        *,
        candidate: Dict[str, float],
        candidate_count: int,
        incumbent: Dict[str, float],
        incumbent_count: int,
    ) -> bool:
        """Return whether a deployed-model checkpoint should replace the best.

        A denser checkpoint must buy a material SSIM gain and may not materially
        regress PSNR. Within the SSIM tolerance, only a smaller checkpoint can
        win. This prevents a loss-optimized later stage from silently replacing
        a better and cheaper deployed canvas.
        """

        min_ssim_gain = float(
            max(
                0.0,
                self.refinement_config.get("canvas_stage_min_ssim_gain", 5e-4),
            )
        )
        max_ssim_regression = float(
            max(
                0.0,
                self.refinement_config.get("canvas_stage_max_ssim_regression", 5e-4),
            )
        )
        max_psnr_regression = float(
            max(
                0.0,
                self.refinement_config.get("canvas_stage_max_psnr_regression", 0.10),
            )
        )
        candidate_ssim = float(candidate.get("ssim_srgb", 0.0))
        incumbent_ssim = float(incumbent.get("ssim_srgb", 0.0))
        candidate_psnr = float(candidate.get("psnr_srgb", 0.0))
        incumbent_psnr = float(incumbent.get("psnr_srgb", 0.0))
        psnr_safe = candidate_psnr >= incumbent_psnr - max_psnr_regression
        if candidate_ssim >= incumbent_ssim + min_ssim_gain:
            return psnr_safe
        return bool(
            candidate_count < incumbent_count
            and candidate_ssim >= incumbent_ssim - max_ssim_regression
            and psnr_safe
        )
