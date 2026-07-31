import numpy as np
import pytest

from png2svg_gs.distillation import run_distillation_mvp
from png2svg_gs.mlx_renderer import is_mlx_available
from png2svg_gs.splat import create_isotropic_splat


def test_distillation_mvp_runs_all_three_arms_and_reduces_student_loss() -> None:
    target = np.zeros((12, 12, 3), dtype=np.float32)
    target[:, :6, 0] = 0.9
    target[:, 6:, 2] = 0.8
    initial = [
        create_isotropic_splat(
            center=np.array([3.0, 6.0]),
            sigma=4.0,
            color=np.array([0.6, 0.2, 0.2]),
            alpha=0.6,
        ),
        create_isotropic_splat(
            center=np.array([9.0, 6.0]),
            sigma=4.0,
            color=np.array([0.2, 0.2, 0.6]),
            alpha=0.6,
        ),
    ]

    result = run_distillation_mvp(
        initial,
        target,
        teacher_iterations=2,
        student_iterations=3,
        normalized_top_k=2,
    )

    assert result.direct.iterations == 5
    assert result.teacher.iterations == 2
    assert result.student.iterations == 3
    assert result.student.rendered_linear_rgb.shape == target.shape
    assert result.student.end_loss < result.student.start_loss


@pytest.mark.parametrize("handoff_mode", ["full", "color-only"])
def test_distillation_handoff_and_exportability_variants_run(
    handoff_mode: str,
) -> None:
    target = np.full((8, 8, 3), 0.35, dtype=np.float32)
    initial = [
        create_isotropic_splat(
            center=np.array([4.0, 4.0]),
            sigma=3.0,
            color=np.array([0.2, 0.3, 0.4]),
            alpha=0.5,
        )
    ]

    result = run_distillation_mvp(
        initial,
        target,
        teacher_iterations=1,
        student_iterations=1,
        normalized_top_k=1,
        teacher_exportability_weight=0.2,
        handoff_mode=handoff_mode,
        background_linear_rgb=np.array([0.1, 0.2, 0.3], dtype=np.float32),
    )

    assert np.isfinite(result.teacher.rendered_linear_rgb).all()
    assert np.isfinite(result.student.rendered_linear_rgb).all()


def test_distillation_rejects_unknown_handoff_mode() -> None:
    target = np.zeros((4, 4, 3), dtype=np.float32)
    initial = [
        create_isotropic_splat(
            center=np.array([2.0, 2.0]),
            sigma=2.0,
            color=np.array([0.2, 0.3, 0.4]),
            alpha=0.5,
        )
    ]

    with pytest.raises(ValueError, match="Unsupported handoff mode"):
        run_distillation_mvp(
            initial,
            target,
            teacher_iterations=0,
            student_iterations=0,
            handoff_mode="unknown",  # type: ignore[arg-type]
        )


@pytest.mark.skipif(not is_mlx_available(), reason="MLX is not installed")
def test_mlx_distillation_runs_all_three_arms() -> None:
    target = np.zeros((12, 12, 3), dtype=np.float32)
    target[:, :6, 0] = 0.9
    target[:, 6:, 2] = 0.8
    initial = [
        create_isotropic_splat(
            center=np.array([3.0, 6.0]),
            sigma=4.0,
            color=np.array([0.6, 0.2, 0.2]),
            alpha=0.6,
        ),
        create_isotropic_splat(
            center=np.array([9.0, 6.0]),
            sigma=4.0,
            color=np.array([0.2, 0.2, 0.6]),
            alpha=0.6,
        ),
    ]

    result = run_distillation_mvp(
        initial,
        target,
        teacher_iterations=2,
        student_iterations=3,
        normalized_top_k=2,
        optimization_backend="mlx",
        tile_size=8,
        batch_tile_count=2,
    )

    assert result.direct.iterations == 5
    assert result.teacher.iterations == 2
    assert result.student.iterations == 3
    assert result.student.end_loss < result.student.start_loss
    assert all(
        arm.elapsed_sec >= 0.0
        for arm in (result.direct, result.teacher, result.student)
    )
