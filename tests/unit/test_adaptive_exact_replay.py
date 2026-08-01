"""Exact raw-checkpoint replay used by the adaptive Canvas simulator."""

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from splatthis.adaptive_compute import AdaptiveComputePolicy, CanvasCheckpoint
from splatthis.io import save_splats_json
from splatthis.splat import create_isotropic_splat

REPO = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "simulate_adaptive_canvas", REPO / "tools" / "simulate_adaptive_canvas.py"
)
assert SPEC is not None and SPEC.loader is not None
simulate_adaptive_canvas = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = simulate_adaptive_canvas
SPEC.loader.exec_module(simulate_adaptive_canvas)


def _checkpoint(label: str, ssim: float, elapsed_sec: float) -> CanvasCheckpoint:
    return CanvasCheckpoint(
        label=label,
        ssim_srgb=ssim,
        psnr_srgb=30.0 + ssim,
        splat_count=1,
        elapsed_sec=elapsed_sec,
    )


def test_raw_stage_loader_exact_rescores_checkpoints_without_historical_metrics(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.png"
    Image.new("RGB", (16, 16), (0, 0, 0)).save(source_path)
    splats = [
        create_isotropic_splat(
            center=np.array([8.0, 8.0]),
            sigma=2.0,
            color=np.array([1.0, 0.2, 0.1]),
            alpha=0.6,
        )
    ]
    for stage in range(1, 4):
        save_splats_json(splats, str(tmp_path / f"iter-{stage}.raw.json"))

    manifest_path = tmp_path / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "config": {
                    "resolved_target_size": [16, 16],
                    "background_linear_rgb": [0.0, 0.0, 0.0],
                    "training_export_target": "canvas",
                    "compositing_space": "linear",
                },
                "stages": [
                    {
                        "stage": 1,
                        "elapsed_sec": 1.0,
                        "deployed_ssim_srgb": 0.5,
                    },
                    {"stage": 2, "elapsed_sec": 2.0},
                    {"stage": 3, "elapsed_sec": 3.0},
                ],
            }
        )
    )

    checkpoints, comparisons = simulate_adaptive_canvas._load_deployed_checkpoints(
        manifest_path, source_path
    )

    assert [checkpoint.label for checkpoint in checkpoints] == [
        "stage-1",
        "stage-2",
        "stage-3",
    ]
    assert len(comparisons) == 3
    assert comparisons[0]["historical_continuous_ssim_srgb"] == 0.5
    assert comparisons[1]["historical_continuous_ssim_srgb"] is None
    assert comparisons[2]["historical_minus_exact_ssim"] is None
    assert comparisons[2]["ssim_srgb"] == pytest.approx(checkpoints[2].ssim_srgb)


def test_full_curve_summary_models_only_the_online_hard_target() -> None:
    curves: dict[str, tuple[Path, list[CanvasCheckpoint], list[dict[str, Any]]]] = {
        "example": (
            Path("run_manifest.json"),
            [
                _checkpoint("stage-1", 0.8, 1.0),
                _checkpoint("stage-2", 0.9, 2.0),
                _checkpoint("stage-3", 0.95, 3.0),
            ],
            [],
        )
    }
    policy = AdaptiveComputePolicy(
        target_ssim_srgb=0.0,
        stop_on_regression=False,
        stop_on_plateau=False,
    )

    summary = simulate_adaptive_canvas._summarize_stage_replay(
        curves,
        policy,
        expected_image_count=1,
        minimum_useful_saving_fraction=0.05,
    )

    assert summary["mode"] == "online-hard-target-observed-only"
    assert summary["uses_plateau_stop"] is False
    assert summary["uses_regression_stop"] is False
    assert summary["early_stop_count"] == 1
    assert summary["saved_stage_sec"] == pytest.approx(3.0)
    assert summary["saved_stage_fraction"] == pytest.approx(0.5)
    assert summary["compute_gate_met"] is True
    assert summary["go_no_go"] == "continue-to-fresh-ab"


def test_versioned_exact_replay_records_full_corpus_no_go() -> None:
    evidence = json.loads((REPO / "data" / "adaptive-exact-replay.json").read_text())

    assert evidence["source"]["image_count"] == 21
    assert evidence["source"]["raw_checkpoint_count"] == 84
    assert evidence["source"]["missing_curve_count"] == 0
    assert [item["target_ssim_srgb"] for item in evidence["target_comparison"]] == [
        0.98,
        0.979,
    ]
    assert all(
        item["saved_stage_fraction"] == pytest.approx(0.012637920332276651)
        for item in evidence["target_comparison"]
    )
    assert all(
        item["go_no_go"] == "do-not-expand" for item in evidence["target_comparison"]
    )
    assert evidence["decision"]["next_slice"] == "algorithmic-fidelity-improvement"
