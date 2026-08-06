import numpy as np
import torch

from splatthis.optimizer import SplatParams, build_optimizer
from splatthis.renderer import render_splats_numpy, splats_to_tensor, tensor_to_splats
from splatthis.splat import create_isotropic_splat


def test_tensor_and_renderer_contract():
    splat = create_isotropic_splat(
        center=np.array([4, 4]),
        sigma=2,
        color=np.array([1, 0, 0]),
        alpha=0.8,
    )
    tensor = splats_to_tensor([splat])
    restored = tensor_to_splats(tensor)
    rendered = render_splats_numpy(restored, 9, 9)

    assert tensor.shape == (1, 11)
    assert rendered.shape == (9, 9, 3)
    assert rendered[4, 4, 0] > rendered[0, 0, 0]


def test_adam_uses_real_parameter_groups():
    params = SplatParams(torch.zeros((2, 11), dtype=torch.float32))
    optimizer = build_optimizer(params)
    assert len(optimizer.param_groups) == 5
    assert all(len(group["params"]) == 1 for group in optimizer.param_groups)
