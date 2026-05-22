"""Optimization utilities for Gaussian splats."""

import torch
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class SplatOptimizer:
    """
    Advanced optimizer for Gaussian splats.

    Wraps Adam optimizer while applying parameter-group learning rates by
    scaling gradients per parameter slice on each step.
    """

    def __init__(self, learning_rates: Dict[str, float] = None):
        """
        Initialize splat optimizer.

        Args:
            learning_rates: Learning rates for different parameter types
        """
        self.learning_rates = learning_rates or {
            'position': 0.01,
            'covariance': 0.005,
            'color': 0.02,
            'alpha': 0.01
        }
        self.base_lr = float(self.learning_rates["position"])

    def create_optimizer(self, splat_tensor: torch.Tensor) -> torch.optim.Optimizer:
        """
        Create optimizer with parameter-specific learning rates.

        Args:
            splat_tensor: Tensor of splat parameters

        Returns:
            Configured optimizer
        """
        return torch.optim.Adam([splat_tensor], lr=self.base_lr)

    def step_with_constraints(self, optimizer: torch.optim.Optimizer,
                            splat_tensor: torch.Tensor) -> None:
        """
        Optimization step with parameter constraints.

        Args:
            optimizer: PyTorch optimizer
            splat_tensor: Tensor to optimize
        """
        # Snapshot before the step so we can apply per-group learning rates to
        # the actual Adam update. Scaling the *gradient* before Adam (the old
        # approach) is a no-op: Adam normalizes by m_hat/sqrt(v_hat), which
        # cancels any constant per-slice gradient scaling. Scaling the resulting
        # update delta instead gives a true effective lr of base_lr * ratio per
        # group, with shared moment estimates.
        with torch.no_grad():
            previous = splat_tensor.detach().clone()
        optimizer.step()
        with torch.no_grad():
            self._apply_group_learning_rates(splat_tensor, previous)
            self._apply_constraints(splat_tensor)

    def _apply_constraints(self, splat_tensor: torch.Tensor) -> None:
        """Apply parameter constraints."""
        # Position in valid frame.
        splat_tensor[:, 0:2].clamp_(0)

        # Scale+rotation parameterization.
        splat_tensor[:, 2].clamp_(min=1e-4)
        splat_tensor[:, 3].clamp_(min=1e-4)
        splat_tensor[:, 4].remainder_(2.0 * torch.pi)

        # Colors and alpha in [0, 1]
        splat_tensor[:, 6:10].clamp_(0, 1)

    def _apply_group_learning_rates(
        self, splat_tensor: torch.Tensor, previous: torch.Tensor
    ) -> None:
        """
        Rescale the just-applied Adam update per parameter group so each group's
        effective learning rate is base_lr * ratio.

        Parameter layout:
        [0:2]=position, [2:5]=scale+rotation, [6:9]=color, [9]=alpha, [10]=importance.
        """
        if self.base_lr <= 0:
            return

        ratios = {
            "position": self.learning_rates.get("position", self.base_lr) / self.base_lr,
            "covariance": self.learning_rates.get("covariance", self.base_lr) / self.base_lr,
            "color": self.learning_rates.get("color", self.base_lr) / self.base_lr,
            "alpha": self.learning_rates.get("alpha", self.base_lr) / self.base_lr,
        }

        def rescale(cols, ratio):
            splat_tensor[:, cols] = previous[:, cols] + (splat_tensor[:, cols] - previous[:, cols]) * ratio

        rescale(slice(0, 2), ratios["position"])
        rescale(slice(2, 5), ratios["covariance"])
        rescale(slice(6, 9), ratios["color"])
        rescale(slice(9, 10), ratios["alpha"])
        # Keep importance frozen by default in this phase (ratio 0 -> no update).
        splat_tensor[:, 10] = previous[:, 10]
