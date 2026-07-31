"""Retrospective adaptive Canvas checkpoint and scaling policies."""

import json
from pathlib import Path

import pytest

from png2svg_gs.adaptive_compute import (
    DEFAULT_CHROME_PSNR_SAFETY_MARGIN,
    DEFAULT_CHROME_SSIM_SAFETY_MARGIN,
    AdaptiveComputePolicy,
    CanvasBudgetPoint,
    CanvasCheckpoint,
    OnlineAdaptiveConfig,
    ScalingPolicy,
    evaluate_online_checkpoints,
    resolve_online_adaptive_config,
    retrospective_scale_decision,
    simulate_adaptive_checkpoints,
)


def _checkpoint(label, ssim, psnr, splats, seconds):
    return CanvasCheckpoint(
        label=label,
        ssim_srgb=ssim,
        psnr_srgb=psnr,
        splat_count=splats,
        elapsed_sec=seconds,
    )


def test_stage_replay_stops_at_quality_target_and_reports_opportunity_cost():
    checkpoints = [
        _checkpoint("stage-1", 0.95, 35.0, 1000, 10.0),
        _checkpoint("stage-2", 0.981, 40.0, 1600, 20.0),
        _checkpoint("stage-3", 0.985, 41.0, 2200, 30.0),
    ]
    result = simulate_adaptive_checkpoints(
        checkpoints, AdaptiveComputePolicy(target_ssim_srgb=0.98)
    )

    assert result.stop_reason == "quality-target"
    assert result.selected.label == "stage-2"
    assert result.checkpoints_observed == 2
    assert result.saved_stage_sec == pytest.approx(30.0)
    assert result.ssim_opportunity_cost == pytest.approx(0.004)


def test_stage_replay_reverts_regression_and_stops_before_later_work():
    checkpoints = [
        _checkpoint("stage-1", 0.80, 30.0, 1000, 10.0),
        _checkpoint("stage-2", 0.82, 31.0, 1600, 20.0),
        _checkpoint("stage-3", 0.81, 30.5, 2200, 30.0),
        _checkpoint("residual", 0.83, 32.0, 2500, 40.0),
    ]
    result = simulate_adaptive_checkpoints(
        checkpoints,
        AdaptiveComputePolicy(
            min_checkpoints=3,
            target_ssim_srgb=None,
            plateau_min_ssim_gain=0.0,
            plateau_min_psnr_gain=0.0,
        ),
    )

    assert result.stop_reason == "regression-revert"
    assert result.selected.label == "stage-2"
    assert result.full_run_best.label == "residual"
    assert result.saved_stage_sec == pytest.approx(40.0)
    assert result.ssim_opportunity_cost == pytest.approx(0.01)


def test_stage_replay_detects_plateau():
    checkpoints = [
        _checkpoint("stage-1", 0.80, 30.0, 1000, 10.0),
        _checkpoint("stage-2", 0.801, 30.05, 1500, 20.0),
        _checkpoint("stage-3", 0.82, 31.0, 2000, 30.0),
    ]
    result = simulate_adaptive_checkpoints(
        checkpoints,
        AdaptiveComputePolicy(target_ssim_srgb=None),
    )

    assert result.stop_reason == "plateau"
    assert result.checkpoints_observed == 2
    assert result.saved_stage_sec == pytest.approx(30.0)


def test_stage_replay_can_disable_retrospective_plateau_stopping():
    checkpoints = [
        _checkpoint("stage-1", 0.80, 30.0, 1000, 10.0),
        _checkpoint("stage-2", 0.79, 29.0, 1500, 20.0),
        _checkpoint("stage-3", 0.82, 31.0, 2000, 30.0),
    ]
    result = simulate_adaptive_checkpoints(
        checkpoints,
        AdaptiveComputePolicy(
            target_ssim_srgb=0.98,
            stop_on_regression=False,
            stop_on_plateau=False,
        ),
    )

    assert result.stop_reason == "curve-exhausted"
    assert result.checkpoints_observed == 3
    assert result.saved_stage_sec == 0.0


def test_online_controller_uses_only_observed_checkpoints_and_hard_target():
    config = OnlineAdaptiveConfig(
        enabled=True,
        min_checkpoints=2,
        target_ssim_srgb=0.98,
        target_psnr_srgb=40.0,
    )
    first = evaluate_online_checkpoints(
        [_checkpoint("stage-1", 0.99, 41.0, 1000, 10.0)],
        config,
    )
    second = evaluate_online_checkpoints(
        [
            _checkpoint("stage-1", 0.99, 41.0, 1000, 10.0),
            _checkpoint("stage-2", 0.985, 40.5, 1500, 20.0),
        ],
        config,
    )

    assert not first.stop
    assert first.reason == "minimum-checkpoints"
    assert second.stop
    assert second.reason == "quality-target"
    assert second.selected.label == "stage-1"
    assert second.as_dict()["uses_future_evidence"] is False
    assert second.as_dict()["effective_model_threshold"] == {
        "ssim_srgb": pytest.approx(0.98),
        "psnr_srgb": pytest.approx(40.0),
    }
    assert second.as_dict()["policy"]["runtime_scorer_pixel_exact"] is True


def test_online_controller_applies_chrome_safety_margin_to_stop_threshold():
    config = OnlineAdaptiveConfig(
        enabled=True,
        min_checkpoints=2,
        target_ssim_srgb=0.98,
        chrome_ssim_safety_margin=0.0012,
    )
    below_margin = evaluate_online_checkpoints(
        [
            _checkpoint("stage-1", 0.9805, 40.0, 1000, 10.0),
            _checkpoint("stage-2", 0.9811, 40.1, 1500, 20.0),
        ],
        config,
    )
    above_margin = evaluate_online_checkpoints(
        [
            _checkpoint("stage-1", 0.9805, 40.0, 1000, 10.0),
            _checkpoint("stage-2", 0.9813, 40.1, 1500, 20.0),
        ],
        config,
    )

    assert not below_margin.stop
    assert below_margin.reason == "quality-target-not-met"
    assert above_margin.stop
    assert above_margin.as_dict()["requested_chrome_target"]["ssim_srgb"] == 0.98


def test_online_controller_does_not_stop_on_plateau_or_regression():
    checkpoints = [
        _checkpoint("stage-1", 0.80, 30.0, 1000, 10.0),
        _checkpoint("stage-2", 0.79, 29.0, 1500, 20.0),
    ]
    decision = evaluate_online_checkpoints(
        checkpoints,
        OnlineAdaptiveConfig(enabled=True, target_ssim_srgb=0.98),
    )

    assert not decision.stop
    assert decision.reason == "quality-target-not-met"
    assert decision.selected.label == "stage-1"


def test_online_config_resolution_validates_safe_contract():
    config = resolve_online_adaptive_config(
        {
            "adaptive_compute_enabled": True,
            "adaptive_compute_min_checkpoints": 3,
            "adaptive_compute_target_ssim_srgb": 0.99,
        }
    )
    assert config.enabled
    assert config.min_checkpoints == 3
    assert config.target_ssim_srgb == pytest.approx(0.99)
    assert config.chrome_ssim_safety_margin == pytest.approx(
        DEFAULT_CHROME_SSIM_SAFETY_MARGIN
    )
    assert config.chrome_psnr_safety_margin == pytest.approx(
        DEFAULT_CHROME_PSNR_SAFETY_MARGIN
    )

    with pytest.raises(ValueError, match="between 0 and 1"):
        resolve_online_adaptive_config(
            {
                "adaptive_compute_enabled": True,
                "adaptive_compute_target_ssim_srgb": 1.01,
            }
        )
    with pytest.raises(ValueError, match="requires an SSIM or PSNR"):
        resolve_online_adaptive_config(
            {
                "adaptive_compute_enabled": True,
                "adaptive_compute_target_ssim_srgb": None,
                "adaptive_compute_target_psnr_srgb": None,
            }
        )
    with pytest.raises(ValueError, match="plus Chrome safety margin"):
        resolve_online_adaptive_config(
            {
                "adaptive_compute_enabled": True,
                "adaptive_compute_target_ssim_srgb": 0.999,
                "adaptive_compute_chrome_ssim_margin": 0.002,
            }
        )
    with pytest.raises(ValueError, match="must be non-negative"):
        resolve_online_adaptive_config(
            {
                "adaptive_compute_chrome_ssim_margin": -0.001,
            }
        )


def _budget_point(image, budget, ssim, lpips, runtime):
    return CanvasBudgetPoint(
        image=image,
        requested_budget=budget,
        ssim_srgb=ssim,
        lpips=lpips,
        runtime_sec=runtime,
        artifact_bytes=budget * 100,
        final_splats=budget - 100,
    )


def test_budget_scaling_oracle_stops_at_target_or_scales_for_observed_gain():
    target_stop = retrospective_scale_decision(
        _budget_point("easy", 2000, 0.96, 0.10, 100.0),
        _budget_point("easy", 4000, 0.97, 0.08, 220.0),
        ScalingPolicy(target_ssim_srgb=0.95, target_lpips=0.15),
    )
    hard_scale = retrospective_scale_decision(
        _budget_point("hard", 2000, 0.70, 0.40, 100.0),
        _budget_point("hard", 4000, 0.76, 0.30, 220.0),
        ScalingPolicy(target_ssim_srgb=0.95),
    )

    assert not target_stop["scale"]
    assert target_stop["reason"] == "quality-target"
    assert hard_scale["scale"]
    assert hard_scale["reason"] == "observed-quality-gain"
    assert hard_scale["mode"] == "retrospective-oracle"


def test_budget_scaling_requires_same_image_and_increasing_budget():
    policy = ScalingPolicy()
    with pytest.raises(ValueError, match="same image"):
        retrospective_scale_decision(
            _budget_point("a", 2000, 0.7, 0.3, 100.0),
            _budget_point("b", 4000, 0.8, 0.2, 200.0),
            policy,
        )
    with pytest.raises(ValueError, match="larger requested budget"):
        retrospective_scale_decision(
            _budget_point("a", 4000, 0.7, 0.3, 100.0),
            _budget_point("a", 2000, 0.8, 0.2, 200.0),
            policy,
        )


def test_versioned_online_mvp_evidence_is_loadable():
    repo = Path(__file__).resolve().parents[2]
    evidence = json.loads((repo / "data" / "adaptive-online-mvp.json").read_text())

    assert evidence["schema"] == "splatthis.adaptive-online-mvp/1"
    assert evidence["scope"]["general_speedup_claim"] is False
    runs = {run["id"]: run for run in evidence["runs"]}
    stopped = runs["colorwheel-target-0.979"]
    assert stopped["controller"]["stopped_early"] is True
    assert stopped["controller"]["skipped_main_stages"] == 1
    assert stopped["controller"]["skipped_residual_detail"] is True
