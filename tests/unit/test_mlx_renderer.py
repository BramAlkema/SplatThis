from typing import List

import numpy as np
import pytest
import torch

from png2svg_gs.mlx_renderer import (
    MlxBatchedGaussianRenderer,
    is_mlx_available,
    splats_to_numpy_table,
)
from png2svg_gs.renderer import create_renderer, splats_to_tensor
from png2svg_gs.splat import GaussianSplat, RawSplat


def _sample_splats() -> List[GaussianSplat]:
    return [
        GaussianSplat.from_raw_splat(
            RawSplat(
                x=5.0,
                y=5.0,
                sx=3.0,
                sy=2.0,
                theta=0.2,
                r=0.90,
                g=0.10,
                b=0.05,
                a=0.75,
                importance=0.1,
            )
        ),
        GaussianSplat.from_raw_splat(
            RawSplat(
                x=12.0,
                y=8.0,
                sx=4.0,
                sy=1.5,
                theta=1.0,
                r=0.10,
                g=0.80,
                b=0.20,
                a=0.55,
                importance=0.5,
            )
        ),
        GaussianSplat.from_raw_splat(
            RawSplat(
                x=18.0,
                y=15.0,
                sx=5.0,
                sy=3.0,
                theta=2.2,
                r=0.15,
                g=0.25,
                b=0.95,
                a=0.65,
                importance=0.9,
            )
        ),
    ]


def test_splats_to_numpy_table_uses_layered_render_order() -> None:
    splat = GaussianSplat.from_raw_splat(
        RawSplat(
            x=1.0,
            y=2.0,
            sx=3.0,
            sy=4.0,
            theta=0.0,
            r=0.1,
            g=0.2,
            b=0.3,
            a=0.4,
            importance=0.2,
            layer=3,
        )
    )

    table = splats_to_numpy_table([splat])

    assert table.shape == (1, 11)
    assert table.dtype == np.float32
    assert table[0, 10] == pytest.approx(3.2)


def test_mlx_renderer_import_guard_when_mlx_is_absent() -> None:
    if is_mlx_available():
        pytest.skip("MLX is available in this environment")

    with pytest.raises(RuntimeError, match="MLX is not installed"):
        MlxBatchedGaussianRenderer(width=8, height=8)


@pytest.mark.skipif(not is_mlx_available(), reason="MLX is not installed")
def test_mlx_batched_renderer_matches_torch_reference() -> None:
    import mlx.core as mx

    splats = _sample_splats()
    width = 23
    height = 19
    background = [0.05, 0.08, 0.11]

    reference = create_renderer(
        backend="torch-batched",
        width=width,
        height=height,
        device=torch.device("cpu"),
        tile_size=8,
        blend_mode="alpha-over",
        background_color=background,
        batch_tile_count=3,
    )(splats_to_tensor(splats))

    renderer = MlxBatchedGaussianRenderer(
        width=width,
        height=height,
        tile_size=8,
        batch_tile_count=3,
        blend_mode="alpha-over",
        background_color=background,
    )
    table = splats_to_numpy_table(splats)
    image = renderer.render(table, plan=renderer.build_plan(table))
    mx.eval(image)

    assert np.allclose(
        np.asarray(image), reference.detach().numpy(), atol=1e-5, rtol=1e-5
    )
