"""Optional MLX splat parameter state and Adam optimizer."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from .optimizer import DEFAULT_LEARNING_RATES
from .splat import GaussianSplat, RawSplat

try:  # pragma: no cover - exercised in MLX-enabled environments.
    import mlx.core as mx
except Exception:  # pragma: no cover - optional dependency guard.
    mx = None  # type: ignore[assignment]


TRAINABLE_KEYS = ("position", "scale", "theta", "color", "alpha")


def is_mlx_available() -> bool:
    return mx is not None


def _require_mlx() -> Any:
    if mx is None:
        raise RuntimeError(
            "MLX is not installed. Install `mlx` to use MLX optimization."
        )
    return mx


def _normalize_trainable(keys: Optional[Sequence[str]]) -> tuple[str, ...]:
    if keys is None:
        return TRAINABLE_KEYS
    normalized = tuple(str(key).strip().lower().replace("-", "_") for key in keys)
    unsupported = [key for key in normalized if key not in TRAINABLE_KEYS]
    if unsupported:
        raise ValueError(
            f"Unsupported MLX trainable parameter group(s): {', '.join(unsupported)}"
        )
    return normalized


@dataclass(frozen=True)
class MlxSplatParams:
    """MLX representation of the canonical [N, 11] splat parameter table."""

    position: Any
    scale: Any
    theta: Any
    color: Any
    alpha: Any
    importance: Any

    @classmethod
    def from_table(cls, table: Any) -> "MlxSplatParams":
        mlx = _require_mlx()
        data = mlx.array(np.asarray(table, dtype=np.float32))
        if len(data.shape) != 2 or data.shape[1] != 11:
            raise ValueError(f"table must have shape [N, 11], got {tuple(data.shape)}")
        return cls(
            position=data[:, 0:2],
            scale=data[:, 2:4],
            theta=data[:, 4],
            color=data[:, 6:9],
            alpha=data[:, 9],
            importance=data[:, 10],
        )

    @property
    def num_splats(self) -> int:
        return int(self.position.shape[0])

    def trainable_tree(self, keys: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        names = _normalize_trainable(keys)
        return {name: getattr(self, name) for name in names}

    def frozen_tree(
        self, trainable_keys: Optional[Sequence[str]] = None
    ) -> Dict[str, Any]:
        names = set(_normalize_trainable(trainable_keys))
        return {
            name: getattr(self, name) for name in TRAINABLE_KEYS if name not in names
        }

    def as_table(self, trainable: Optional[Mapping[str, Any]] = None) -> Any:
        mlx = _require_mlx()
        values = {
            "position": self.position,
            "scale": self.scale,
            "theta": self.theta,
            "color": self.color,
            "alpha": self.alpha,
        }
        if trainable is not None:
            values.update(trainable)
        reserved = mlx.zeros((self.num_splats, 1), dtype=mlx.float32)
        return mlx.concatenate(
            [
                values["position"],
                values["scale"],
                mlx.expand_dims(values["theta"], -1),
                reserved,
                values["color"],
                mlx.expand_dims(values["alpha"], -1),
                mlx.expand_dims(self.importance, -1),
            ],
            axis=1,
        )


def constrain_trainable_tree(
    tree: Mapping[str, Any],
    *,
    image_width: int,
    image_height: int,
) -> Dict[str, Any]:
    """Clamp trainable groups to the same ranges as the torch optimizer."""

    mlx = _require_mlx()
    constrained = dict(tree)
    if "position" in constrained:
        pos = constrained["position"]
        x = mlx.clip(pos[:, 0], 0.0, float(max(image_width - 1, 0)))
        y = mlx.clip(pos[:, 1], 0.0, float(max(image_height - 1, 0)))
        constrained["position"] = mlx.stack([x, y], axis=1)
    if "scale" in constrained:
        constrained["scale"] = mlx.maximum(constrained["scale"], 1e-4)
    if "theta" in constrained:
        constrained["theta"] = mlx.remainder(constrained["theta"], 2.0 * math.pi)
    if "color" in constrained:
        constrained["color"] = mlx.clip(constrained["color"], 0.0, 1.0)
    if "alpha" in constrained:
        constrained["alpha"] = mlx.clip(constrained["alpha"], 0.0, 1.0)
    return constrained


def clone_tree(tree: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a detached MLX tree snapshot."""

    mlx = _require_mlx()
    return {key: mlx.stop_gradient(value) for key, value in tree.items()}


def tree_to_numpy_table(params: MlxSplatParams, tree: Mapping[str, Any]) -> np.ndarray:
    mlx = _require_mlx()
    table = params.as_table(tree)
    mlx.eval(table)
    return np.asarray(table, dtype=np.float32)


def table_to_splats(
    table: np.ndarray, templates: Optional[Sequence[GaussianSplat]] = None
) -> List[GaussianSplat]:
    """Convert a canonical table back to splats, preserving template layers when present."""

    splats: List[GaussianSplat] = []
    for idx, row in enumerate(np.asarray(table, dtype=np.float32)):
        layer = None
        if templates is not None and idx < len(templates):
            layer = templates[idx].to_raw_splat().layer
        raw = RawSplat(
            x=float(row[0]),
            y=float(row[1]),
            sx=max(float(row[2]), 1e-4),
            sy=max(float(row[3]), 1e-4),
            theta=float(np.remainder(row[4], 2.0 * np.pi)),
            r=float(row[6]),
            g=float(row[7]),
            b=float(row[8]),
            a=float(row[9]),
            importance=float(row[10]),
            layer=layer,
        )
        splats.append(GaussianSplat.from_raw_splat(raw))
    return splats


class MlxAdam:
    """Small functional Adam implementation over a dict of MLX arrays."""

    def __init__(
        self,
        initial_tree: Mapping[str, Any],
        learning_rates: Optional[Mapping[str, float]] = None,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        grad_clip_norm: Optional[float] = 1.0,
    ):
        mlx = _require_mlx()
        self.learning_rates = {**DEFAULT_LEARNING_RATES, **dict(learning_rates or {})}
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.grad_clip_norm = grad_clip_norm
        self.step_count = 0
        self.m = {key: mlx.zeros_like(value) for key, value in initial_tree.items()}
        self.v = {key: mlx.zeros_like(value) for key, value in initial_tree.items()}

    def _global_grad_norm(self, grads: Mapping[str, Any]) -> Any:
        mlx = _require_mlx()
        total = mlx.array(0.0, dtype=mlx.float32)
        for grad in grads.values():
            total = total + mlx.sum(grad * grad)
        return mlx.sqrt(total)

    def step(
        self, tree: Mapping[str, Any], grads: Mapping[str, Any]
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        mlx = _require_mlx()
        self.step_count += 1
        grad_norm = self._global_grad_norm(grads)
        if self.grad_clip_norm is None:
            clip_factor = mlx.array(1.0, dtype=mlx.float32)
        else:
            clip_factor = mlx.minimum(
                1.0, float(self.grad_clip_norm) / (grad_norm + 1e-6)
            )

        next_tree: Dict[str, Any] = {}
        for key, value in tree.items():
            grad = grads[key] * clip_factor
            self.m[key] = self.beta1 * self.m[key] + (1.0 - self.beta1) * grad
            self.v[key] = self.beta2 * self.v[key] + (1.0 - self.beta2) * (grad * grad)
            m_hat = self.m[key] / (1.0 - self.beta1**self.step_count)
            v_hat = self.v[key] / (1.0 - self.beta2**self.step_count)
            lr = float(self.learning_rates[key])
            next_tree[key] = value - lr * m_hat / (mlx.sqrt(v_hat) + self.eps)

        return next_tree, {"grad_norm": grad_norm, "clip_factor": clip_factor}


__all__ = [
    "MlxAdam",
    "MlxSplatParams",
    "TRAINABLE_KEYS",
    "clone_tree",
    "constrain_trainable_tree",
    "is_mlx_available",
    "table_to_splats",
    "tree_to_numpy_table",
]
