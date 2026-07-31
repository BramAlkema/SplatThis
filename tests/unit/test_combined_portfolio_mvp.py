from pathlib import Path

import numpy as np

from png2svg_gs.io import load_splats_json, save_splats_json
from png2svg_gs.splat import GaussianSplat, RawSplat
from tools.combined_portfolio_mvp import (
    _build_foreground_hybrid,
    metric_deltas,
    passes_guard,
    quality_score,
)


def _metrics(**overrides):
    values = {
        "ssim_srgb": 0.80,
        "ms_ssim_luma": 0.79,
        "lpips": 0.20,
        "psnr_srgb": 24.0,
        "delta_e_ok_mean": 0.05,
        "delta_e_ok_p95": 0.12,
        "edge_chamfer": 2.0,
        "edge_gradient_l1": 0.08,
        "worst_roi_error": 0.10,
        "file_size_bytes": 1000,
    }
    values.update(overrides)
    return values


def test_guard_rejects_ssim_win_that_is_only_blur():
    baseline = _metrics()
    candidate = _metrics(
        ssim_srgb=0.82,
        ms_ssim_luma=0.78,
        lpips=0.21,
    )

    accepted, failures = passes_guard(baseline, candidate)

    assert not accepted
    assert failures == ["ms-ssim", "lpips"]


def test_balanced_improvement_passes_and_has_positive_score():
    baseline = _metrics()
    candidate = _metrics(
        ssim_srgb=0.81,
        ms_ssim_luma=0.80,
        lpips=0.19,
        delta_e_ok_p95=0.11,
        edge_gradient_l1=0.07,
        worst_roi_error=0.09,
        file_size_bytes=900,
    )

    accepted, failures = passes_guard(baseline, candidate)

    assert accepted
    assert failures == []
    assert quality_score(baseline, candidate) > 0.0
    assert metric_deltas(baseline, candidate)["file_size_bytes"] == -100.0


def test_foreground_hybrid_uses_student_locally_and_preserves_direct_order(
    tmp_path: Path,
):
    direct = [
        GaussianSplat.from_raw_splat(
            RawSplat(
                x=0,
                y=0,
                sx=1,
                sy=1,
                theta=0,
                r=0.1,
                g=0.1,
                b=0.1,
                a=0.4,
                importance=0.2,
            )
        ),
        GaussianSplat.from_raw_splat(
            RawSplat(
                x=1,
                y=1,
                sx=1,
                sy=1,
                theta=0,
                r=0.2,
                g=0.2,
                b=0.2,
                a=0.5,
                importance=0.8,
            )
        ),
    ]
    student = [
        GaussianSplat.from_raw_splat(
            RawSplat(
                x=0,
                y=0,
                sx=2,
                sy=2,
                theta=0,
                r=0.9,
                g=0.1,
                b=0.1,
                a=0.7,
                importance=0.9,
            )
        ),
        GaussianSplat.from_raw_splat(
            RawSplat(
                x=1,
                y=1,
                sx=2,
                sy=2,
                theta=0,
                r=0.1,
                g=0.9,
                b=0.1,
                a=0.8,
                importance=0.1,
            )
        ),
    ]
    direct_raw = tmp_path / "direct.raw.json"
    student_raw = tmp_path / "student.raw.json"
    save_splats_json(direct, str(direct_raw))
    save_splats_json(student, str(student_raw))
    mask = np.array([[False, False], [False, True]])

    _, hybrid_path, _ = _build_foreground_hybrid(
        name="hybrid",
        foreground_population=("student", student_raw, []),
        background_population=("direct", direct_raw, []),
        foreground_mask=mask,
        output_dir=tmp_path,
    )

    hybrid = load_splats_json(str(hybrid_path))
    assert hybrid[0].to_raw_splat().r == direct[0].to_raw_splat().r
    assert hybrid[1].to_raw_splat().g == student[1].to_raw_splat().g
    assert hybrid[1].importance == direct[1].importance
