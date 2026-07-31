import json
from typing import List, Optional, Sequence

import numpy as np
import pytest
import torch

from png2svg_gs.io import generate_pixel_runtime_html, linear_to_srgb
from png2svg_gs.mlx_renderer import (
    MlxBatchedGaussianRenderer,
    is_mlx_available,
    splats_to_numpy_table,
)
from png2svg_gs.mlx_stage import MlxRendererConfig
from png2svg_gs.renderer import (
    create_renderer,
    prepare_pixel_runtime_data,
    render_pixel_runtime_numpy,
    render_splats_numpy,
    splats_to_tensor,
)
from png2svg_gs.splat import GaussianSplat, RawSplat

WIDTH = 23
HEIGHT = 19
BACKGROUND = [0.05, 0.08, 0.11]
TILE_SIZE = 8
BATCH_TILE_COUNT = 3


def test_mlx_renderer_config_uses_benchmarked_small_tile_batch() -> None:
    assert MlxRendererConfig().batch_tile_count == 8


def test_pixel_runtime_data_is_the_payload_serialized_into_html() -> None:
    splats = _sample_splats()
    background = np.asarray(BACKGROUND, dtype=np.float32)
    rows, serialized_background, srgb_mode = prepare_pixel_runtime_data(
        splats,
        background_linear_rgb=background,
        compositing_space="linear",
    )
    html = generate_pixel_runtime_html(
        splats,
        width=WIDTH,
        height=HEIGHT,
        background_linear_rgb=background,
        compositing_space="linear",
    )
    payload = html.split("const SPLATS = ", 1)[1].split(";", 1)[0]

    assert json.loads(payload) == rows
    assert srgb_mode is False
    for channel in serialized_background:
        assert f"{float(channel):.6f}" in html


def test_pixel_runtime_renderer_returns_an_exact_8bit_srgb_framebuffer() -> None:
    rendered = render_pixel_runtime_numpy(
        _sample_splats(),
        width=WIDTH,
        height=HEIGHT,
        background_linear_rgb=np.asarray(BACKGROUND, dtype=np.float32),
    )
    display_codes = linear_to_srgb(rendered) * 255.0

    assert rendered.dtype == np.float32
    assert rendered.shape == (HEIGHT, WIDTH, 3)
    assert np.allclose(display_codes, np.round(display_codes), atol=2e-4)


def test_pixel_runtime_renderer_validates_dimensions() -> None:
    with pytest.raises(ValueError, match="positive integers"):
        render_pixel_runtime_numpy([], width=0, height=10)


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


def test_numpy_rotated_footprints_match_tiled_renderer() -> None:
    """A rotated major axis must not be clipped by an unrotated sx/sy bbox."""

    splats = [
        GaussianSplat.from_raw_splat(
            RawSplat(
                x=11.0,
                y=9.0,
                sx=6.0,
                sy=0.8,
                theta=np.pi / 2.0,
                r=0.9,
                g=0.2,
                b=0.1,
                a=0.8,
                importance=0.5,
            )
        )
    ]

    reference = _render_torch_batched(splats)
    rendered = render_splats_numpy(
        splats,
        width=WIDTH,
        height=HEIGHT,
        background_linear_rgb=np.asarray(BACKGROUND, dtype=np.float32),
    )

    _assert_images_close(rendered, reference)


def _render_torch_batched(
    splats: Sequence[GaussianSplat],
    blend_mode: str = "alpha-over",
    compositing_space: str = "linear",
    max_active_splats_per_tile: Optional[int] = None,
    normalized_top_k: int = 10,
) -> np.ndarray:
    renderer = create_renderer(
        backend="torch-batched",
        width=WIDTH,
        height=HEIGHT,
        device=torch.device("cpu"),
        tile_size=TILE_SIZE,
        blend_mode=blend_mode,
        normalized_top_k=normalized_top_k,
        background_color=BACKGROUND,
        compositing_space=compositing_space,
        batch_tile_count=BATCH_TILE_COUNT,
        max_active_splats_per_tile=max_active_splats_per_tile,
    )
    return renderer(splats_to_tensor(list(splats))).detach().numpy()


def _render_mlx(
    splats: Sequence[GaussianSplat],
    blend_mode: str = "alpha-over",
    compositing_space: str = "linear",
    max_active_splats_per_tile: Optional[int] = None,
    pptx_softedge_mode: bool = False,
    normalized_top_k: int = 10,
) -> np.ndarray:
    import mlx.core as mx

    renderer = MlxBatchedGaussianRenderer(
        width=WIDTH,
        height=HEIGHT,
        tile_size=TILE_SIZE,
        batch_tile_count=BATCH_TILE_COUNT,
        blend_mode=blend_mode,
        normalized_top_k=normalized_top_k,
        background_color=BACKGROUND,
        compositing_space=compositing_space,
        max_active_splats_per_tile=max_active_splats_per_tile,
        pptx_softedge_mode=pptx_softedge_mode,
    )
    table = splats_to_numpy_table(list(splats))
    image = renderer.render(table, plan=renderer.build_plan(table))
    mx.eval(image)
    return np.asarray(image)


def _assert_images_close(mlx_image: np.ndarray, torch_image: np.ndarray) -> None:
    assert mlx_image.shape == torch_image.shape
    assert np.allclose(mlx_image, torch_image, atol=1e-5, rtol=1e-5)


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

    with pytest.raises(RuntimeError, match="MLX is not installed|no Metal device"):
        MlxBatchedGaussianRenderer(width=8, height=8)


@pytest.mark.skipif(not is_mlx_available(), reason="MLX is not installed")
@pytest.mark.parametrize("compositing_space", ["linear", "srgb"])
@pytest.mark.parametrize("blend_mode", ["alpha-over", "weighted"])
def test_mlx_batched_renderer_matches_torch_reference_matrix(
    blend_mode: str, compositing_space: str
) -> None:
    splats = _sample_splats()

    reference = _render_torch_batched(
        splats, blend_mode=blend_mode, compositing_space=compositing_space
    )
    image = _render_mlx(
        splats, blend_mode=blend_mode, compositing_space=compositing_space
    )

    _assert_images_close(image, reference)


@pytest.mark.skipif(not is_mlx_available(), reason="MLX is not installed")
def test_mlx_normalized_topk_matches_torch_reference() -> None:
    splats = _sample_splats()

    reference = _render_torch_batched(
        splats,
        blend_mode="normalized-topk",
        normalized_top_k=2,
    )
    image = _render_mlx(
        splats,
        blend_mode="normalized-topk",
        normalized_top_k=2,
    )

    _assert_images_close(image, reference)


@pytest.mark.skipif(not is_mlx_available(), reason="MLX is not installed")
def test_mlx_normalized_topk_has_finite_gradients() -> None:
    import mlx.core as mx

    splats = _sample_splats()
    table = mx.array(splats_to_numpy_table(splats))
    renderer = MlxBatchedGaussianRenderer(
        width=WIDTH,
        height=HEIGHT,
        tile_size=TILE_SIZE,
        batch_tile_count=BATCH_TILE_COUNT,
        blend_mode="normalized-topk",
        normalized_top_k=2,
        background_color=BACKGROUND,
    )
    plan = renderer.build_plan(np.asarray(table))

    def objective(values):
        return mx.mean(renderer.render(values, plan=plan))

    gradients = mx.grad(objective)(table)
    mx.eval(gradients)

    assert np.isfinite(np.asarray(gradients)).all()


@pytest.mark.skipif(not is_mlx_available(), reason="MLX is not installed")
def test_mlx_pptx_softedge_mode_matches_torch_proxy_renderer() -> None:
    from png2svg_gs.converter import _PPTXSoftEdgeProxyRenderer

    splats = _sample_splats()

    base = create_renderer(
        backend="torch-batched",
        width=WIDTH,
        height=HEIGHT,
        device=torch.device("cpu"),
        tile_size=TILE_SIZE,
        blend_mode="alpha-over",
        background_color=BACKGROUND,
        compositing_space="srgb",
        batch_tile_count=BATCH_TILE_COUNT,
    )
    proxy = _PPTXSoftEdgeProxyRenderer(base_renderer=base)
    reference = proxy(splats_to_tensor(splats)).detach().numpy()

    mlx_renderer = MlxBatchedGaussianRenderer(
        width=WIDTH,
        height=HEIGHT,
        tile_size=TILE_SIZE,
        batch_tile_count=BATCH_TILE_COUNT,
        blend_mode="alpha-over",
        background_color=BACKGROUND,
        compositing_space="srgb",
        pptx_softedge_mode=True,
    )
    # The MLX defaults must line up with the converter proxy defaults, or the
    # parity below would silently compare different soft-edge transforms.
    assert mlx_renderer.pptx_alpha_scale == pytest.approx(proxy.alpha_scale)
    assert mlx_renderer.pptx_sigma_scale == pytest.approx(proxy.sigma_scale)

    import mlx.core as mx

    table = splats_to_numpy_table(splats)
    image = mlx_renderer.render(table, plan=mlx_renderer.build_plan(table))
    mx.eval(image)

    _assert_images_close(np.asarray(image), reference)


@pytest.mark.skipif(not is_mlx_available(), reason="MLX is not installed")
def test_mlx_tied_importances_match_torch_backends() -> None:
    # Overlapping splats sharing one importance value: only a stable sort keeps
    # their input order identical across backends, so this pins the stable
    # argsort in every renderer.
    colors = [
        (0.95, 0.05, 0.05),
        (0.05, 0.90, 0.10),
        (0.10, 0.15, 0.95),
        (0.85, 0.80, 0.10),
    ]
    splats = [
        GaussianSplat.from_raw_splat(
            RawSplat(
                x=10.0 + 0.7 * idx,
                y=8.0 + 0.5 * idx,
                sx=4.0,
                sy=3.0,
                theta=0.3 * idx,
                r=r,
                g=g,
                b=b,
                a=0.7,
                importance=0.5,
            )
        )
        for idx, (r, g, b) in enumerate(colors)
    ]

    torch_batched = _render_torch_batched(splats)
    torch_reference = (
        create_renderer(
            backend="torch",
            width=WIDTH,
            height=HEIGHT,
            device=torch.device("cpu"),
            tile_size=TILE_SIZE,
            blend_mode="alpha-over",
            background_color=BACKGROUND,
        )(splats_to_tensor(splats))
        .detach()
        .numpy()
    )
    mlx_image = _render_mlx(splats)

    _assert_images_close(mlx_image, torch_batched)
    _assert_images_close(mlx_image, torch_reference)
    assert np.allclose(torch_batched, torch_reference, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not is_mlx_available(), reason="MLX is not installed")
def test_mlx_edge_straddling_and_offcanvas_splats_match_torch() -> None:
    splats = [
        # Sigma comparable to the canvas, center hugging the top-left corner.
        GaussianSplat.from_raw_splat(
            RawSplat(
                x=1.0,
                y=1.5,
                sx=18.0,
                sy=14.0,
                theta=0.4,
                r=0.85,
                g=0.30,
                b=0.10,
                a=0.60,
                importance=0.2,
            )
        ),
        # Center off-canvas, but the 3-sigma footprint reaches into the image.
        GaussianSplat.from_raw_splat(
            RawSplat(
                x=-4.0,
                y=9.0,
                sx=5.0,
                sy=4.0,
                theta=1.2,
                r=0.10,
                g=0.55,
                b=0.90,
                a=0.80,
                importance=0.8,
            )
        ),
    ]

    reference = _render_torch_batched(splats)
    image = _render_mlx(splats)

    _assert_images_close(image, reference)
    # The off-canvas splat must actually contribute on-canvas, otherwise this
    # test is vacuous: its footprint dominates the left edge mid-height.
    assert image[9, 0, 2] > BACKGROUND[2] + 0.05


@pytest.mark.skipif(not is_mlx_available(), reason="MLX is not installed")
def test_mlx_theta_beyond_two_pi_matches_torch() -> None:
    splats = [
        GaussianSplat.from_raw_splat(
            RawSplat(
                x=11.0,
                y=9.0,
                sx=5.0,
                sy=1.5,
                theta=7.0,
                r=0.80,
                g=0.20,
                b=0.60,
                a=0.70,
                importance=0.4,
            )
        ),
        GaussianSplat.from_raw_splat(
            RawSplat(
                x=8.0,
                y=12.0,
                sx=3.0,
                sy=2.0,
                theta=7.0,
                r=0.15,
                g=0.75,
                b=0.25,
                a=0.60,
                importance=0.6,
            )
        ),
    ]

    reference = _render_torch_batched(splats)
    image = _render_mlx(splats)

    _assert_images_close(image, reference)


@pytest.mark.skipif(not is_mlx_available(), reason="MLX is not installed")
def test_mlx_tile_overload_truncation_keeps_front_most_splat() -> None:
    # Two co-centered splats, one tile slot: truncation must drop the BACK
    # (lower-importance, red) splat and keep the FRONT (higher-importance,
    # blue) one in both backends.
    back_red = GaussianSplat.from_raw_splat(
        RawSplat(
            x=11.0,
            y=9.0,
            sx=4.0,
            sy=4.0,
            theta=0.0,
            r=0.95,
            g=0.05,
            b=0.05,
            a=1.0,
            importance=0.2,
        )
    )
    front_blue = GaussianSplat.from_raw_splat(
        RawSplat(
            x=11.0,
            y=9.0,
            sx=4.0,
            sy=4.0,
            theta=0.0,
            r=0.05,
            g=0.05,
            b=0.95,
            a=1.0,
            importance=0.8,
        )
    )
    splats = [back_red, front_blue]

    reference = _render_torch_batched(splats, max_active_splats_per_tile=1)
    image = _render_mlx(splats, max_active_splats_per_tile=1)

    _assert_images_close(image, reference)

    for rendered in (image, reference):
        center = rendered[9, 11]
        # Blue must dominate at the shared center: alpha=1.0 gives a layer
        # opacity of 1 - exp(-1) ~ 0.63, so blue lands well above 0.5 while
        # red stays at the (kept) blue splat's tiny red component.
        assert center[2] > 0.5
        assert center[0] < 0.1
        assert center[2] > center[0]
