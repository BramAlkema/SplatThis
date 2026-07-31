"""Canvas checkpoint model-to-browser parity calibration helpers."""

import json
import sys
from pathlib import Path

import pytest

from png2svg_gs.adaptive_compute import (
    CANVAS_RUNTIME_CALIBRATION_CHECKPOINTS,
    CANVAS_RUNTIME_SCORER,
    DEFAULT_CHROME_PSNR_SAFETY_MARGIN,
    DEFAULT_CHROME_SSIM_SAFETY_MARGIN,
)
from png2svg_gs.canvas_parity import (
    CanvasParityObservation,
    ceil_margin,
    summarize_canvas_parity,
)
from tools.calibrate_canvas_checkpoint_parity import DEFAULT_CAPTURE_PYTHON


def _observation(image, checkpoint, *, ssim_bias, psnr_bias, parity=0.9999):
    return CanvasParityObservation(
        image=image,
        checkpoint=checkpoint,
        model_ssim_srgb=0.98,
        browser_ssim_srgb=0.98 - ssim_bias,
        model_psnr_srgb=40.0,
        browser_psnr_srgb=40.0 - psnr_bias,
        pixel_parity_ssim_srgb=parity,
        pixel_mae_linear=0.0001,
        pixel_max_abs_linear=0.01,
    )


def test_summary_rounds_worst_positive_overstatement_up():
    summary = summarize_canvas_parity(
        [
            _observation("a", "iter-1", ssim_bias=0.00021, psnr_bias=0.011),
            _observation("b", "iter-2", ssim_bias=0.00031, psnr_bias=0.021),
        ]
    )

    assert summary["observation_count"] == 2
    assert summary["image_count"] == 2
    assert summary["ssim_overstatement"]["max"] == pytest.approx(0.00031)
    assert summary["recommended_ssim_safety_margin"] == pytest.approx(0.0004)
    assert summary["recommended_psnr_safety_margin"] == pytest.approx(0.03)


def test_summary_does_not_turn_browser_understatement_into_negative_margin():
    summary = summarize_canvas_parity(
        [_observation("a", "iter-1", ssim_bias=-0.0002, psnr_bias=-0.1)]
    )

    assert summary["recommended_ssim_safety_margin"] == 0.0
    assert summary["recommended_psnr_safety_margin"] == 0.0


def test_ceil_margin_validates_quantum():
    assert ceil_margin(0.0002, 0.0001) == pytest.approx(0.0002)
    with pytest.raises(ValueError, match="quantum must be positive"):
        ceil_margin(0.1, 0.0)
    with pytest.raises(ValueError, match="at least one"):
        summarize_canvas_parity([])


def test_checkpoint_calibrator_defaults_to_current_virtual_environment():
    assert DEFAULT_CAPTURE_PYTHON == Path(sys.executable)


def test_versioned_parity_evidence_matches_controller_defaults():
    repo = Path(__file__).resolve().parents[2]
    evidence = json.loads((repo / "data" / "canvas-checkpoint-parity.json").read_text())

    assert evidence["schema"] == "splatthis.canvas-checkpoint-parity-evidence/2"
    assert evidence["coverage"]["measured_checkpoints"] == (
        CANVAS_RUNTIME_CALIBRATION_CHECKPOINTS
    )
    assert evidence["coverage"]["capture_failures"] == 0
    assert evidence["calibration"]["pixel_exact_checkpoints"] == (
        CANVAS_RUNTIME_CALIBRATION_CHECKPOINTS
    )
    assert evidence["method"]["runtime_scorer"] == CANVAS_RUNTIME_SCORER
    assert evidence["calibration"]["recommended_ssim_safety_margin"] == (
        DEFAULT_CHROME_SSIM_SAFETY_MARGIN
    )
    assert evidence["calibration"]["recommended_psnr_safety_margin_db"] == (
        DEFAULT_CHROME_PSNR_SAFETY_MARGIN
    )
