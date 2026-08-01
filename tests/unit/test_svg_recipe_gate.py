"""Guarded selection of target-specific SVG export recipes."""

import json
from pathlib import Path

import pytest

from splatthis.svg_recipe_gate import (
    SvgRecipeGatePolicy,
    evaluate_recipe_candidate,
    select_recipe_candidate,
)

REPO = Path(__file__).resolve().parents[2]


def _measurement(recipe: str, **overrides):
    metrics = {
        "recipe": recipe,
        "ssim_srgb": 0.70,
        "ms_ssim_luma": 0.69,
        "psnr_srgb": 30.0,
        "lpips": 0.30,
        "delta_e_ok_mean": 0.05,
        "delta_e_ok_p95": 0.12,
        "edge_chamfer": 2.0,
        "edge_gradient_l1": 0.08,
        "worst_roi_error": 0.10,
        "file_size_bytes": 1000,
        "render_time_sec": 0.1,
    }
    metrics.update(overrides)
    return metrics


def test_recipe_gate_accepts_balanced_perceptual_gain() -> None:
    baseline = _measurement("standard")
    candidate = _measurement(
        "blur",
        ssim_srgb=0.702,
        ms_ssim_luma=0.692,
        lpips=0.29,
        delta_e_ok_p95=0.115,
        edge_gradient_l1=0.075,
        file_size_bytes=1050,
    )

    decision = evaluate_recipe_candidate(baseline, candidate, SvgRecipeGatePolicy())

    assert decision["accepted"]
    assert decision["meaningful_gains"]["lpips"]
    assert decision["balanced_quality_score"] > 0.0


def test_recipe_gate_rejects_resource_and_protected_metric_regressions() -> None:
    baseline = _measurement("standard")
    candidate = _measurement(
        "palette-quantized",
        ssim_srgb=0.71,
        lpips=0.31,
        worst_roi_error=0.11,
        file_size_bytes=1200,
        render_time_sec=0.2,
    )

    decision = evaluate_recipe_candidate(baseline, candidate, SvgRecipeGatePolicy())

    assert not decision["accepted"]
    assert {"file-size", "render-time", "lpips", "worst-roi"}.issubset(
        decision["failures"]
    )


def test_recipe_selector_reverts_or_picks_highest_balanced_score() -> None:
    baseline = _measurement("standard")
    rejected = _measurement("blur", file_size_bytes=2000, lpips=0.28)
    palette = _measurement(
        "palette-quantized", ssim_srgb=0.704, lpips=0.292, file_size_bytes=900
    )
    blur = _measurement("blur", ssim_srgb=0.702, lpips=0.295)

    selected = select_recipe_candidate(
        baseline, [rejected, palette, blur], SvgRecipeGatePolicy()
    )
    reverted = select_recipe_candidate(baseline, [rejected], SvgRecipeGatePolicy())

    assert selected["accepted_candidate"]
    assert selected["selected_recipe"] == "palette-quantized"
    assert reverted["accepted_candidate"] is False
    assert reverted["selected"] == baseline
    assert selected["selected"]["file_size_bytes"] == pytest.approx(900)


def test_versioned_recipe_gate_records_predeclared_chrome_go() -> None:
    evidence = json.loads((REPO / "data" / "svg-recipe-gate-mvp.json").read_text())

    assert evidence["schema"] == "splatthis.svg-recipe-gate-evidence/2"
    assert evidence["method"]["renderer"].startswith("Chromium ")
    assert evidence["method"]["device_scale_factor"] == 1
    assert evidence["coverage"]["expected_images"] == 21
    assert evidence["coverage"]["completed_images"] == 21
    assert evidence["coverage"]["failed_images"] == 0
    assert evidence["coverage"]["artifacts_rasterized"] == 63
    assert evidence["coverage"]["measured_captures"] == 189
    assert evidence["coverage"]["pixel_stable_artifacts"] == 63
    assert evidence["policy"]["minimum_accepted_images"] == 5
    assert evidence["outcome"]["accepted_images"] == 7
    assert evidence["outcome"]["candidate_acceptance"] == {
        "palette-quantized": 7,
        "blur": 0,
    }
    assert evidence["outcome"]["go_no_go"] == "integrate-recipe-selector"
    assert evidence["decision"]["next_operator"] == (
        "default-off-browser-recipe-selection"
    )
