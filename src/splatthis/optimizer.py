"""Splat parameter container + Adam optimizer with real per-group learning rates.

This module replaces the older SplatOptimizer-on-a-monolithic-tensor design,
whose "per-group learning rates" were a post-Adam delta rescaling. With Adam's
m_hat/sqrt(v_hat) normalization, a constant per-slice gradient or delta scale
ends up indistinguishable from the base LR -- so the old knobs were no-ops.

Here each parameter group is a separate `nn.Parameter` and Adam is built with
proper `param_groups`, so per-group learning rates have their textbook meaning.

Parameter layout of the [N, 11] tensor consumed by the renderer:
    [0:2]=position, [2:4]=scale, [4]=theta, [5]=reserved(0),
    [6:9]=color, [9]=alpha, [10]=importance.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


DEFAULT_LEARNING_RATES: Dict[str, float] = {
    "position": 0.0075,
    "scale": 0.0055,
    "theta": 0.0055,
    "color": 0.016,
    "alpha": 0.01,
}


class SplatParams(nn.Module):
    """Holds the trainable splat parameters as five independent nn.Parameters
    so `torch.optim.Adam` can drive each group with a real learning rate.

    `importance` is kept as a frozen buffer; it isn't optimized in this phase.
    """

    def __init__(self, initial: torch.Tensor):
        super().__init__()
        if initial.dim() != 2 or initial.shape[1] != 11:
            raise ValueError(f"initial must be [N, 11], got {tuple(initial.shape)}")
        data = initial.detach().clone()
        # Each group is a distinct leaf Parameter -> Adam tracks its own state.
        self.position = nn.Parameter(data[:, 0:2].contiguous())
        self.scale = nn.Parameter(data[:, 2:4].contiguous())
        self.theta = nn.Parameter(data[:, 4].contiguous())
        self.color = nn.Parameter(data[:, 6:9].contiguous())
        self.alpha = nn.Parameter(data[:, 9].contiguous())
        # Frozen rows (no grad).
        self.register_buffer("importance", data[:, 10].contiguous())

    @property
    def num_splats(self) -> int:
        return int(self.position.shape[0])

    def as_tensor(self) -> torch.Tensor:
        """Reassemble the [N, 11] tensor the renderer expects.

        Differentiable -- gradients on the output flow back into each parameter.
        """
        n = self.num_splats
        reserved = torch.zeros(
            n, 1, dtype=self.position.dtype, device=self.position.device
        )
        return torch.cat(
            [
                self.position,  # [N, 2]
                self.scale,  # [N, 2]
                self.theta.unsqueeze(-1),  # [N, 1]
                reserved,  # [N, 1]
                self.color,  # [N, 3]
                self.alpha.unsqueeze(-1),  # [N, 1]
                self.importance.unsqueeze(-1),  # [N, 1]
            ],
            dim=1,
        )

    @torch.no_grad()
    def apply_constraints(self, image_width: int, image_height: int) -> None:
        """In-place clamps to keep parameters in valid ranges after each step."""
        self.position[:, 0].clamp_(0.0, float(max(image_width - 1, 0)))
        self.position[:, 1].clamp_(0.0, float(max(image_height - 1, 0)))
        self.scale.clamp_(min=1e-4)
        self.theta.remainder_(2.0 * torch.pi)
        self.color.clamp_(0.0, 1.0)
        self.alpha.clamp_(0.0, 1.0)

    @torch.no_grad()
    def snapshot(self) -> Dict[str, torch.Tensor]:
        """Return a deep copy of all parameter tensors (for early-stop revert)."""
        return {
            "position": self.position.detach().clone(),
            "scale": self.scale.detach().clone(),
            "theta": self.theta.detach().clone(),
            "color": self.color.detach().clone(),
            "alpha": self.alpha.detach().clone(),
            "importance": self.importance.detach().clone(),
        }

    @torch.no_grad()
    def restore(self, snapshot: Dict[str, torch.Tensor]) -> None:
        """Restore parameter tensors from a snapshot."""
        self.position.data.copy_(snapshot["position"])
        self.scale.data.copy_(snapshot["scale"])
        self.theta.data.copy_(snapshot["theta"])
        self.color.data.copy_(snapshot["color"])
        self.alpha.data.copy_(snapshot["alpha"])
        self.importance.copy_(snapshot["importance"])


def build_optimizer(
    params: SplatParams, learning_rates: Optional[Dict[str, float]] = None
) -> torch.optim.Adam:
    """Build a torch.optim.Adam with one param_group per splat parameter, so the
    per-group learning rates are real (not the old post-step delta rescaling).
    """
    lr = {**DEFAULT_LEARNING_RATES, **(learning_rates or {})}
    return torch.optim.Adam(
        [
            {"params": [params.position], "lr": float(lr["position"])},
            {"params": [params.scale], "lr": float(lr["scale"])},
            {"params": [params.theta], "lr": float(lr["theta"])},
            {"params": [params.color], "lr": float(lr["color"])},
            {"params": [params.alpha], "lr": float(lr["alpha"])},
        ]
    )


@dataclass
class TrainStepResult:
    loss: float
    rendered: torch.Tensor


def train_step(
    params: SplatParams,
    optimizer: torch.optim.Adam,
    renderer: nn.Module,
    loss_fn: nn.Module,
    target: torch.Tensor,
    image_width: int,
    image_height: int,
) -> TrainStepResult:
    """One training iteration: forward, loss, backward, step, constrain."""
    optimizer.zero_grad(set_to_none=True)
    rendered = renderer(params.as_tensor())
    loss = loss_fn(rendered, target)
    loss.backward()
    optimizer.step()
    params.apply_constraints(image_width, image_height)
    return TrainStepResult(loss=float(loss.detach().item()), rendered=rendered.detach())
