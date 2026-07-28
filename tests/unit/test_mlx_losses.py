"""Numerical parity tests: MLX loss profiles vs the torch reference losses.

The MLX backend is the CLI default, so its objective must match what the
torch reference path optimizes — this is the guard that keeps the two
backends training toward the same target.
"""

import numpy as np
import pytest
import torch

from png2svg_gs.mlx_losses import MlxLossConfig, is_mlx_available, make_loss_fn
from png2svg_gs.renderer import L1SSIMLoss

pytestmark = pytest.mark.skipif(not is_mlx_available(), reason="MLX is not installed")


def _images(seed: int = 7, h: int = 31, w: int = 29):
    rng = np.random.default_rng(seed)
    rendered = rng.random((h, w, 3), dtype=np.float32)
    target = np.clip(
        rendered + rng.normal(0.0, 0.15, size=(h, w, 3)).astype(np.float32), 0.0, 1.0
    )
    weights = rng.random((h, w), dtype=np.float32)
    return rendered, target, weights


def _mlx_scalar(value) -> float:
    import mlx.core as mx

    mx.eval(value)
    return float(value.item())


@pytest.mark.parametrize("use_weights", [False, True])
@pytest.mark.parametrize("gradient_weight", [0.0, 0.08])
def test_oklab_l1_ssim_matches_torch_default_objective(use_weights, gradient_weight):
    """MLX 'oklab-l1-ssim' == torch L1SSIMLoss(color_space='oklab') numerically."""
    import mlx.core as mx

    rendered, target, weights = _images()

    torch_loss = L1SSIMLoss(
        l1_weight=1.0,
        ssim_weight=0.2,
        gradient_weight=gradient_weight,
        color_space="oklab",
        spatial_weight_map=torch.from_numpy(weights) if use_weights else None,
    )
    expected = float(
        torch_loss(torch.from_numpy(rendered), torch.from_numpy(target)).item()
    )

    mlx_fn = make_loss_fn(
        MlxLossConfig(
            name="oklab-l1-ssim",
            l1_weight=1.0,
            ssim_weight=0.2,
            gradient_weight=gradient_weight,
        )
    )
    actual = _mlx_scalar(
        mlx_fn(
            mx.array(rendered),
            mx.array(target),
            mx.array(weights) if use_weights else None,
        )
    )

    assert actual == pytest.approx(expected, abs=2e-5)


@pytest.mark.parametrize("use_weights", [False, True])
def test_linear_l1_ssim_matches_torch_linear_objective(use_weights):
    """MLX 'l1-ssim' == torch L1SSIMLoss(color_space='linear') numerically."""
    import mlx.core as mx

    rendered, target, weights = _images(seed=11)

    torch_loss = L1SSIMLoss(
        l1_weight=1.0,
        ssim_weight=0.2,
        color_space="linear",
        spatial_weight_map=torch.from_numpy(weights) if use_weights else None,
    )
    expected = float(
        torch_loss(torch.from_numpy(rendered), torch.from_numpy(target)).item()
    )

    mlx_fn = make_loss_fn(MlxLossConfig(name="l1-ssim", ssim_weight=0.2))
    actual = _mlx_scalar(
        mlx_fn(
            mx.array(rendered),
            mx.array(target),
            mx.array(weights) if use_weights else None,
        )
    )

    assert actual == pytest.approx(expected, abs=2e-5)
