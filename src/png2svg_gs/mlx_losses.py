"""Optional MLX loss helpers for Apple Silicon optimizer experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

try:  # pragma: no cover - exercised in MLX-enabled environments.
    import mlx.core as mx
except Exception:  # pragma: no cover - optional dependency guard.
    mx = None  # type: ignore[assignment]


def is_mlx_available() -> bool:
    return mx is not None


def _require_mlx() -> Any:
    if mx is None:
        raise RuntimeError("MLX is not installed. Install `mlx` to use MLX losses.")
    return mx


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
    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
    l_ = mlx.power(mlx.maximum(l, 1e-8), 1.0 / 3.0)
    m_ = mlx.power(mlx.maximum(m, 1e-8), 1.0 / 3.0)
    s_ = mlx.power(mlx.maximum(s, 1e-8), 1.0 / 3.0)
    lab_l = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
    lab_a = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
    lab_b = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_
    return mlx.stack([lab_l, lab_a, lab_b], axis=-1)


def oklab_l1_loss(rendered: Any, target: Any, weights: Optional[Any] = None) -> Any:
    """Mean absolute error in OKLab."""

    return linear_l1_loss(
        linear_rgb_to_oklab(rendered), linear_rgb_to_oklab(target), weights
    )


@dataclass(frozen=True)
class MlxLossConfig:
    name: str = "linear-l1"


def make_loss_fn(config: MlxLossConfig):
    """Return an MLX loss callable for the requested loss profile."""

    normalized = str(config.name).strip().lower().replace("_", "-")
    if normalized == "linear-l1":
        return linear_l1_loss
    if normalized in {"oklab-l1", "weighted-oklab-l1"}:
        return oklab_l1_loss
    raise ValueError(f"Unsupported MLX loss profile: {config.name}")


__all__ = [
    "MlxLossConfig",
    "is_mlx_available",
    "linear_l1_loss",
    "linear_rgb_to_oklab",
    "make_loss_fn",
    "oklab_l1_loss",
]
