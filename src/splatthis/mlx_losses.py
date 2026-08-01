"""Optional MLX loss helpers for Apple Silicon optimizer experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from .mlx_runtime import is_mlx_available, require_mlx


def _require_mlx() -> Any:
    return require_mlx("MLX losses")


def linear_l1_loss(rendered: Any, target: Any, weights: Optional[Any] = None) -> Any:
    """Mean absolute error in linear RGB."""

    mlx = _require_mlx()
    diff = mlx.abs(rendered - target)
    if weights is None:
        return mlx.mean(diff)
    w = mlx.expand_dims(weights, -1)
    return mlx.sum(diff * w) / mlx.maximum(mlx.sum(w) * diff.shape[-1], 1e-8)


def linear_rgb_to_oklab(rgb: Any) -> Any:
    """MLX port of the existing differentiable linear-RGB -> OKLab transform."""

    mlx = _require_mlx()
    rgb = mlx.clip(rgb, 0.0, 1.0)
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]
    lms_l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    lms_m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    lms_s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
    l_ = mlx.power(mlx.maximum(lms_l, 1e-8), 1.0 / 3.0)
    m_ = mlx.power(mlx.maximum(lms_m, 1e-8), 1.0 / 3.0)
    s_ = mlx.power(mlx.maximum(lms_s, 1e-8), 1.0 / 3.0)
    lab_l = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
    lab_a = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
    lab_b = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_
    return mlx.stack([lab_l, lab_a, lab_b], axis=-1)


def oklab_l1_loss(rendered: Any, target: Any, weights: Optional[Any] = None) -> Any:
    """Mean absolute error in OKLab."""

    return linear_l1_loss(
        linear_rgb_to_oklab(rendered), linear_rgb_to_oklab(target), weights
    )


def mlx_global_ssim(x: Any, y: Any, *, c1: float = 0.01**2, c2: float = 0.03**2) -> Any:
    """Global SSIM over spatial dimensions, averaged across channels.

    Mirrors the torch `L1SSIMLoss._global_ssim` in renderer.py:1253. "Global"
    = single mean/variance over the entire HxW image per channel, not the
    windowed SSIM variant. Cheap, fully differentiable, and matches what the
    torch path uses for the structural component of its combined loss.
    Returns a scalar in [-1, 1].
    """

    mlx = _require_mlx()
    mu_x = mlx.mean(x, axis=(0, 1))
    mu_y = mlx.mean(y, axis=(0, 1))
    x_centered = x - mlx.reshape(mu_x, (1, 1, -1))
    y_centered = y - mlx.reshape(mu_y, (1, 1, -1))
    sigma_x = mlx.mean(x_centered * x_centered, axis=(0, 1))
    sigma_y = mlx.mean(y_centered * y_centered, axis=(0, 1))
    sigma_xy = mlx.mean(x_centered * y_centered, axis=(0, 1))
    numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
    ssim_per_channel = numerator / mlx.maximum(denominator, 1e-8)
    return mlx.clip(mlx.mean(ssim_per_channel), -1.0, 1.0)


def l1_ssim_loss(
    rendered: Any,
    target: Any,
    weights: Optional[Any] = None,
    *,
    l1_weight: float = 1.0,
    ssim_weight: float = 0.2,
) -> Any:
    """Linear-RGB L1 + (1 - global SSIM), matching torch L1SSIMLoss defaults.

    The structural component preserves local color/contrast relationships in
    addition to pixel-wise error, so MLX-trained splats look closer to what
    the torch path produces on the same input. Default ssim_weight=0.2
    matches torch's L1SSIMLoss default.
    """

    l1 = linear_l1_loss(rendered, target, weights)
    ssim = mlx_global_ssim(rendered, target)
    return float(l1_weight) * l1 + float(ssim_weight) * (1.0 - ssim)


def _linear_to_srgb(x: Any) -> Any:
    """Differentiable linear-RGB -> sRGB, mirroring torch_linear_to_srgb."""

    mlx = _require_mlx()
    x = mlx.clip(x, 0.0, 1.0)
    return mlx.where(
        x <= 0.0031308,
        12.92 * x,
        1.055 * mlx.power(mlx.maximum(x, 1e-12), 1.0 / 2.4) - 0.055,
    )


def luminance_gradient_l1(
    rendered: Any, target: Any, weights: Optional[Any] = None
) -> Any:
    """MLX port of torch L1SSIMLoss._luminance_gradient_l1 (renderer.py).

    Compares display-luminance finite differences so optimization preserves
    local edges; identical math and edge-weight averaging to the torch term.
    """

    mlx = _require_mlx()
    rendered_srgb = _linear_to_srgb(rendered)
    target_srgb = _linear_to_srgb(target)

    def luma(x):
        return 0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]

    rendered_luma = luma(rendered_srgb)
    target_luma = luma(target_srgb)
    losses = []

    if rendered.shape[1] >= 2:
        dx = mlx.abs(
            (rendered_luma[:, 1:] - rendered_luma[:, :-1])
            - (target_luma[:, 1:] - target_luma[:, :-1])
        )
        if weights is None:
            losses.append(mlx.mean(dx))
        else:
            wx = 0.5 * (weights[:, 1:] + weights[:, :-1])
            losses.append(mlx.sum(dx * wx) / mlx.maximum(mlx.sum(wx), 1e-8))

    if rendered.shape[0] >= 2:
        dy = mlx.abs(
            (rendered_luma[1:, :] - rendered_luma[:-1, :])
            - (target_luma[1:, :] - target_luma[:-1, :])
        )
        if weights is None:
            losses.append(mlx.mean(dy))
        else:
            wy = 0.5 * (weights[1:, :] + weights[:-1, :])
            losses.append(mlx.sum(dy * wy) / mlx.maximum(mlx.sum(wy), 1e-8))

    if not losses:
        return mlx.array(0.0)
    total = losses[0]
    for extra in losses[1:]:
        total = total + extra
    return total / float(len(losses))


def oklab_l1_ssim_loss(
    rendered: Any,
    target: Any,
    weights: Optional[Any] = None,
    *,
    l1_weight: float = 1.0,
    ssim_weight: float = 0.2,
    gradient_weight: float = 0.0,
) -> Any:
    """MLX mirror of the torch default objective: L1SSIMLoss(color_space="oklab").

    L1 in OKLab (spatially weighted), global SSIM on the OKLab L channel only
    (a/b are small and signed, so the c1/c2 constants tuned for [0,1] ranges
    would make SSIM on them near-meaningless — same rationale as the torch
    implementation), plus the optional sRGB-luminance gradient term.
    """

    r_lab = linear_rgb_to_oklab(rendered)
    t_lab = linear_rgb_to_oklab(target)
    l1 = linear_l1_loss(r_lab, t_lab, weights)
    ssim = mlx_global_ssim(r_lab[..., 0:1], t_lab[..., 0:1])
    total = float(l1_weight) * l1 + float(ssim_weight) * (1.0 - ssim)
    if float(gradient_weight) > 0.0:
        total = total + float(gradient_weight) * luminance_gradient_l1(
            rendered, target, weights
        )
    return total


@dataclass(frozen=True)
class MlxLossConfig:
    # Default mirrors the torch default objective (L1SSIMLoss with
    # color_space="oklab") so both backends optimize the same thing.
    name: str = "oklab-l1-ssim"
    l1_weight: float = 1.0
    ssim_weight: float = 0.2
    gradient_weight: float = 0.0


def make_loss_fn(config: MlxLossConfig):
    """Return an MLX loss callable for the requested loss profile."""

    normalized = str(config.name).strip().lower().replace("_", "-")
    if normalized == "linear-l1":
        return linear_l1_loss
    if normalized in {"oklab-l1", "weighted-oklab-l1"}:
        return oklab_l1_loss
    if normalized in {"l1-ssim", "linear-l1-ssim"}:
        l1_weight = float(config.l1_weight)
        ssim_weight = float(config.ssim_weight)

        def _fn(rendered, target, weights=None):
            return l1_ssim_loss(
                rendered,
                target,
                weights,
                l1_weight=l1_weight,
                ssim_weight=ssim_weight,
            )

        return _fn
    if normalized == "oklab-l1-ssim":
        l1_weight = float(config.l1_weight)
        ssim_weight = float(config.ssim_weight)
        gradient_weight = float(config.gradient_weight)

        def _fn(rendered, target, weights=None):
            return oklab_l1_ssim_loss(
                rendered,
                target,
                weights,
                l1_weight=l1_weight,
                ssim_weight=ssim_weight,
                gradient_weight=gradient_weight,
            )

        return _fn
    raise ValueError(f"Unsupported MLX loss profile: {config.name}")


__all__ = [
    "MlxLossConfig",
    "is_mlx_available",
    "l1_ssim_loss",
    "linear_l1_loss",
    "linear_rgb_to_oklab",
    "luminance_gradient_l1",
    "make_loss_fn",
    "mlx_global_ssim",
    "oklab_l1_loss",
    "oklab_l1_ssim_loss",
]
