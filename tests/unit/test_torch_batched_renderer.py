import torch

from png2svg_gs.renderer import create_renderer, resolve_renderer_backend


def _sample_splats() -> torch.Tensor:
    rows = torch.tensor(
        [
            [5.0, 5.0, 3.0, 2.0, 0.2, 0.0, 0.90, 0.10, 0.05, 0.75, 0.1],
            [12.0, 8.0, 4.0, 1.5, 1.0, 0.0, 0.10, 0.80, 0.20, 0.55, 0.5],
            [18.0, 15.0, 5.0, 3.0, 2.2, 0.0, 0.15, 0.25, 0.95, 0.65, 0.9],
            [2.0, 16.0, 2.0, 4.0, 0.6, 0.0, 0.95, 0.85, 0.20, 0.45, 0.3],
        ],
        dtype=torch.float32,
    )
    return rows


def test_resolve_torch_batched_backend_on_cpu() -> None:
    assert (
        resolve_renderer_backend("torch-batched", torch.device("cpu"))
        == "torch-batched"
    )
    assert (
        resolve_renderer_backend("torch_batched", torch.device("cpu"))
        == "torch-batched"
    )


def test_torch_batched_renderer_matches_reference_alpha_over() -> None:
    splats = _sample_splats()
    kwargs = {
        "width": 23,
        "height": 19,
        "device": torch.device("cpu"),
        "tile_size": 8,
        "blend_mode": "alpha-over",
        "background_color": [0.05, 0.08, 0.11],
    }

    reference = create_renderer(backend="torch", **kwargs)(splats)
    batched = create_renderer(
        backend="torch-batched",
        batch_tile_count=3,
        **kwargs,
    )(splats)

    assert torch.allclose(batched, reference, atol=1e-6, rtol=1e-6)


def test_torch_batched_renderer_matches_reference_weighted() -> None:
    splats = _sample_splats()
    kwargs = {
        "width": 23,
        "height": 19,
        "device": torch.device("cpu"),
        "tile_size": 8,
        "blend_mode": "weighted",
        "background_color": [0.05, 0.08, 0.11],
    }

    reference = create_renderer(backend="torch", **kwargs)(splats)
    batched = create_renderer(
        backend="torch-batched",
        batch_tile_count=2,
        **kwargs,
    )(splats)

    assert torch.allclose(batched, reference, atol=1e-6, rtol=1e-6)


def test_normalized_topk_matches_manual_per_pixel_equation() -> None:
    splats = torch.tensor(
        [
            [1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.1, 0.1],
            [2.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.9, 0.2],
            [4.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.5, 0.3],
        ],
        dtype=torch.float32,
    )
    renderer = create_renderer(
        backend="torch",
        width=1,
        height=1,
        device=torch.device("cpu"),
        tile_size=1,
        blend_mode="normalized-topk",
        normalized_top_k=2,
        background_color=[0.2, 0.2, 0.2],
    )

    rendered = renderer(splats)[0, 0]
    red_weight = torch.exp(torch.tensor(-0.5))
    green_weight = torch.exp(torch.tensor(-2.0))
    expected = torch.tensor(
        [
            red_weight / (red_weight + green_weight),
            green_weight / (red_weight + green_weight),
            0.0,
        ]
    )

    assert torch.allclose(rendered, expected, atol=1e-6, rtol=1e-6)


def test_normalized_topk_ignores_alpha_and_order() -> None:
    splats = _sample_splats()
    renderer = create_renderer(
        backend="torch",
        width=23,
        height=19,
        device=torch.device("cpu"),
        tile_size=8,
        blend_mode="normalized-topk",
        normalized_top_k=3,
        background_color=[0.05, 0.08, 0.11],
    )

    reference = renderer(splats)
    changed = splats.flip(0).clone()
    changed[:, 9] = torch.tensor([0.01, 0.25, 0.75, 1.0])
    changed[:, 10] = torch.tensor([0.9, 0.1, 0.7, 0.3])

    assert torch.allclose(renderer(changed), reference, atol=1e-6, rtol=1e-6)


def test_torch_batched_renderer_matches_reference_normalized_topk() -> None:
    splats = _sample_splats()
    kwargs = {
        "width": 23,
        "height": 19,
        "device": torch.device("cpu"),
        "tile_size": 8,
        "blend_mode": "normalized-topk",
        "normalized_top_k": 3,
        "background_color": [0.05, 0.08, 0.11],
    }

    reference = create_renderer(backend="torch", **kwargs)(splats)
    batched = create_renderer(
        backend="torch-batched",
        batch_tile_count=2,
        **kwargs,
    )(splats)

    assert torch.allclose(batched, reference, atol=1e-6, rtol=1e-6)


def test_normalized_topk_backward_is_finite() -> None:
    splats = _sample_splats().requires_grad_(True)
    renderer = create_renderer(
        backend="torch-batched",
        width=23,
        height=19,
        device=torch.device("cpu"),
        tile_size=8,
        blend_mode="normalized-topk",
        normalized_top_k=3,
        background_color=[0.05, 0.08, 0.11],
        batch_tile_count=4,
    )

    renderer(splats).mean().backward()

    assert splats.grad is not None
    assert torch.isfinite(splats.grad).all()


def test_torch_batched_renderer_backward() -> None:
    splats = _sample_splats().requires_grad_(True)
    renderer = create_renderer(
        backend="torch-batched",
        width=23,
        height=19,
        device=torch.device("cpu"),
        tile_size=8,
        blend_mode="alpha-over",
        background_color=[0.05, 0.08, 0.11],
        batch_tile_count=4,
    )

    loss = renderer(splats).mean()
    loss.backward()

    assert splats.grad is not None
    assert torch.isfinite(splats.grad).all()
