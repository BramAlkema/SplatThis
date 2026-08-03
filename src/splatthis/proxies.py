"""PPTX soft-edge proxy renderer and loss (extracted from converter.py).

PowerPoint renders soft-edge ellipses brighter and softer than a true
Gaussian; these torch modules mirror that behavior so training can target
the deployed PPTX appearance.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from .export_common import (
    PPTX_GRADIENT_ALPHA_SCALE,
    PPTX_SOFT_EDGE_ALPHA_SCALE,
    PPTX_SOFT_EDGE_K_SIGMA_SCALE,
)
from .renderer import L1SSIMLoss, torch_linear_to_srgb


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
        effective_alpha = pptx_effective_alpha(
            splats_tensor[:, 9], splat_style="soft-edge", alpha_scale=self.alpha_scale
        )
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
        rendered: torch.Tensor = self.base_renderer(fitted)
        return rendered


def pptx_effective_alpha(
    raw_alpha: torch.Tensor,
    *,
    splat_style: str,
    alpha_scale: float,
) -> torch.Tensor:
    """Map a splat's alpha to the value the plain Gaussian renderer needs.

    The renderer computes ``1 - exp(-a * G(r))``, so ``a`` must be whatever
    reproduces the primitive PowerPoint will actually draw:

    ``gradient``
        The emitter writes stops of ``1 - exp(-scale * alpha * G(r))``
        (``pptx_export``), so ``a = alpha * scale``.
    ``soft-edge``
        PowerPoint renders soft edges brighter than a Gaussian; the centre
        opacity is ``(1 - exp(-alpha)) * scale``, inverted back to an alpha.

    Both laws agree as alpha approaches zero, which is why using one for the
    other went unnoticed: the divergence reaches 27% only at alpha 1.0.
    Shared so the training proxies and the post-fit stage cannot drift apart.
    """
    clamped = torch.clamp(raw_alpha, 0.0, 1.0)
    if splat_style == "gradient":
        return clamped * alpha_scale
    center_opacity = torch.clamp(
        (1.0 - torch.exp(-clamped)) * alpha_scale, 0.0, 1.0 - 1e-5
    )
    return -torch.log1p(-center_opacity)


class _PPTXGradientProxyRenderer(torch.nn.Module):
    """Approximate native PPTX gradient-fill ellipses with the base renderer.

    The gradient emitter writes stops of ``1 - exp(-scale * alpha * G(r))``
    and PowerPoint interpolates them linearly, compositing the result
    alpha-over in display sRGB. The base renderer already computes
    ``1 - exp(-a * G(r))``, so scaling the alpha column by the same ``scale``
    reproduces the deployed opacity curve exactly at every stop.

    Between stops the deck is piecewise-linear where this proxy is smooth.
    The PPTX path emits a fixed ``SVG_GRADIENT_STOPS`` ramp rather than
    calling ``_adaptive_gradient_stops``, so that residual is not bounded by
    ``SVG_GRADIENT_STOP_MAX_ERROR`` -- it is simply small at eight stops, and
    is the known approximation this proxy makes.

    Unlike the soft-edge proxy this needs no sigma scaling: the emitted
    ellipse spans the same footprint the renderer integrates over.
    """

    #: Declared because nn.Module.__getattr__ is typed as returning Module,
    #: so a registered buffer otherwise reads as a Module rather than a Tensor.
    alpha_gain: torch.Tensor

    def __init__(
        self,
        base_renderer: torch.nn.Module,
        alpha_scale: float = PPTX_GRADIENT_ALPHA_SCALE,
    ):
        super().__init__()
        self.base_renderer = base_renderer
        self.alpha_scale = float(np.clip(alpha_scale, 1e-4, 1.0))
        # A row of ones with the alpha column set to the scale: one fused
        # multiply per forward instead of slicing the table apart and
        # concatenating it back every training iteration. SplatParams
        # already constrains alpha to [0, 1] after each optimizer step.
        gain = torch.ones(1, 11)
        gain[0, 9] = self.alpha_scale
        self.register_buffer("alpha_gain", gain)

    def forward(self, splats_tensor: torch.Tensor) -> torch.Tensor:
        rendered: torch.Tensor = self.base_renderer(splats_tensor * self.alpha_gain)
        return rendered


class _PPTXProxyLoss(torch.nn.Module):
    """Perceptual loss for PPTX-soft-edge proxy training."""

    # Same reason as above: these are buffers, not submodules.
    weights: torch.Tensor
    target_luma: torch.Tensor
    target_sat: torch.Tensor
    target_luma_std: torch.Tensor
    target_sat_mean: torch.Tensor
    target_sat_std: torch.Tensor

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
        loss: torch.Tensor = self.base_loss(rendered, target)
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
