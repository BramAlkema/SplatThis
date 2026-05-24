import numpy as np
import pytest

from png2svg_gs.cli import build_parser
from png2svg_gs.converter import PNG2SVGConverter
from png2svg_gs.mlx_optimizer import MlxSplatParams, table_to_splats
from png2svg_gs.mlx_stage import MlxStageConfig, is_mlx_available, optimize_stage_mlx
from png2svg_gs.splat import GaussianSplat, RawSplat


def test_table_to_splats_preserves_template_layer_without_mlx() -> None:
    template = GaussianSplat.from_raw_splat(
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
            importance=0.5,
            layer=3,
        )
    )
    table = np.array(
        [[2.0, 3.0, 4.0, 5.0, 0.25, 0.0, 0.2, 0.3, 0.4, 0.6, 0.7]], dtype=np.float32
    )

    splat = table_to_splats(table, templates=[template])[0]
    raw = splat.to_raw_splat()

    assert raw.layer == 3
    assert raw.x == pytest.approx(2.0)
    assert raw.a == pytest.approx(0.6)


def test_mlx_param_guard_when_mlx_is_absent() -> None:
    if is_mlx_available():
        pytest.skip("MLX is available in this environment")

    with pytest.raises(RuntimeError, match="MLX is not installed"):
        MlxSplatParams.from_table(np.zeros((1, 11), dtype=np.float32))


def test_mlx_stage_guard_when_mlx_is_absent() -> None:
    if is_mlx_available():
        pytest.skip("MLX is available in this environment")

    with pytest.raises(RuntimeError, match="MLX is not installed"):
        optimize_stage_mlx(
            [],
            np.zeros((2, 2, 3), dtype=np.float32),
            2,
            2,
            1,
            config=MlxStageConfig(),
        )


def test_cli_accepts_mlx_optimizer_flags() -> None:
    args = build_parser().parse_args(
        [
            "input.png",
            "--optimizer-backend",
            "mlx",
            "--mlx-loss",
            "linear-l1",
            "--mlx-tile-plan",
            "periodic",
            "--mlx-tile-plan-rebuild-interval",
            "2",
            "--mlx-trainable-groups",
            "position,scale,theta,color,alpha",
        ]
    )

    assert args.optimizer_backend == "mlx"
    assert args.mlx_loss == "linear-l1"
    assert args.mlx_tile_plan == "periodic"
    assert args.mlx_tile_plan_rebuild_interval == 2
    assert args.mlx_trainable_groups == "position,scale,theta,color,alpha"


def test_converter_rejects_geometry_groups_for_static_mlx_plan() -> None:
    with pytest.raises(ValueError, match="color/alpha with static tile plans"):
        PNG2SVGConverter(
            optimizer_backend="mlx",
            refinement_config={"mlx_trainable_groups": "position,color"},
        )


def test_converter_accepts_geometry_groups_for_periodic_mlx_plan() -> None:
    converter = PNG2SVGConverter(
        optimizer_backend="mlx",
        refinement_config={
            "mlx_tile_plan": "periodic",
            "mlx_tile_plan_rebuild_interval": 2,
            "mlx_trainable_groups": "position,scale,theta,color,alpha",
        },
    )

    assert converter.mlx_tile_plan == "periodic"
    assert converter.mlx_tile_plan_rebuild_interval == 2
    assert converter.mlx_trainable_groups == (
        "position",
        "scale",
        "theta",
        "color",
        "alpha",
    )


def test_converter_enables_spatial_weights_for_weighted_mlx_loss() -> None:
    converter = PNG2SVGConverter(
        optimizer_backend="mlx",
        refinement_config={
            "mlx_loss": "weighted-oklab-l1",
            "mlx_tile_plan": "periodic",
            "mlx_trainable_groups": "position,scale,theta,color,alpha",
        },
    )

    assert converter.mlx_spatial_weighting_enabled is True
    assert converter._use_mlx_spatial_weights() is True


@pytest.mark.skipif(not is_mlx_available(), reason="MLX is not installed")
def test_mlx_color_alpha_stage_decreases_loss() -> None:
    from png2svg_gs.mlx_stage import MlxRendererConfig

    target = np.zeros((16, 16, 3), dtype=np.float32)
    target[:, :, 0] = 0.8
    splats = [
        GaussianSplat.from_raw_splat(
            RawSplat(
                x=8.0,
                y=8.0,
                sx=20.0,
                sy=20.0,
                theta=0.0,
                r=0.1,
                g=0.1,
                b=0.1,
                a=0.4,
                importance=0.0,
            )
        )
    ]
    result = optimize_stage_mlx(
        splats,
        target,
        16,
        16,
        3,
        config=MlxStageConfig(
            renderer=MlxRendererConfig(tile_size=8, batch_tile_count=2),
            trainable_groups=("color", "alpha"),
        ),
    )

    assert result.metrics["best_loss"] < result.metrics["start_loss"]


@pytest.mark.skipif(not is_mlx_available(), reason="MLX is not installed")
def test_mlx_periodic_stage_allows_geometry_groups() -> None:
    from png2svg_gs.mlx_stage import MlxRendererConfig

    target = np.zeros((16, 16, 3), dtype=np.float32)
    target[:, :, 0] = 0.8
    splats = [
        GaussianSplat.from_raw_splat(
            RawSplat(
                x=6.0,
                y=6.0,
                sx=10.0,
                sy=10.0,
                theta=0.0,
                r=0.1,
                g=0.1,
                b=0.1,
                a=0.4,
                importance=0.0,
            )
        )
    ]
    result = optimize_stage_mlx(
        splats,
        target,
        16,
        16,
        2,
        config=MlxStageConfig(
            renderer=MlxRendererConfig(tile_size=8, batch_tile_count=2),
            trainable_groups=("position", "scale", "theta", "color", "alpha"),
            tile_plan_mode="periodic",
            tile_plan_rebuild_interval=1,
        ),
    )

    assert result.metrics["iterations"] == 2
    assert result.metrics["tile_plan_rebuilds"] >= 2
    assert result.metrics["best_loss"] <= result.metrics["start_loss"]
