"""Target-specific artifact-noise calibration and shared gate helpers."""

import json
from pathlib import Path

import pytest

from splatthis.artifact_gates import (
    ArtifactGateCalibration,
    calibrate_artifact_observations,
    metric_gain,
)


def _observation(target, artifact, repeat, **metrics):
    return {
        "target": target,
        "artifact_id": artifact,
        "repeat": repeat,
        "metrics": metrics,
    }


def test_calibration_measures_spread_within_unchanged_artifacts():
    observations = [
        _observation(
            "pixel-runtime",
            "pixel-runtime:a",
            0,
            ssim_srgb=0.900,
            lpips=0.200,
        ),
        _observation(
            "pixel-runtime",
            "pixel-runtime:a",
            1,
            ssim_srgb=0.902,
            lpips=0.199,
        ),
        _observation(
            "pixel-runtime",
            "pixel-runtime:b",
            0,
            ssim_srgb=0.700,
            lpips=0.400,
        ),
        _observation(
            "pixel-runtime",
            "pixel-runtime:b",
            1,
            ssim_srgb=0.704,
            lpips=0.397,
        ),
    ]
    calibration = calibrate_artifact_observations(
        observations,
        required_repeats=2,
        noise_multiplier=2.0,
        expected_targets=("pixel-runtime",),
    )
    pixel_runtime = calibration.targets["pixel-runtime"]
    ssim = pixel_runtime.metrics["ssim_srgb"]

    assert pixel_runtime.complete
    assert pixel_runtime.artifact_count == 2
    assert ssim.median_span == pytest.approx(0.003)
    assert ssim.p95_span == pytest.approx(0.0039)
    assert ssim.max_span == pytest.approx(0.004)
    assert ssim.recommended_min_delta == pytest.approx(0.0078)


def test_expected_missing_artifact_keeps_target_incomplete():
    observations = [
        _observation("svg", "svg:a", 0, ssim_srgb=0.8),
        _observation("svg", "svg:a", 1, ssim_srgb=0.8),
    ]
    calibration = calibrate_artifact_observations(
        observations,
        required_repeats=2,
        expected_targets=("svg",),
        expected_artifacts={"svg": ("svg:a", "svg:missing")},
    )
    svg = calibration.targets["svg"]

    assert svg.artifact_count == 2
    assert svg.complete_artifact_count == 1
    assert not svg.complete


def test_rare_observed_variation_is_not_lost_when_p95_is_zero():
    observations = []
    expected = []
    for artifact_index in range(21):
        artifact = f"pptx:{artifact_index}"
        expected.append(artifact)
        observations.extend(
            [
                _observation("pptx", artifact, 0, ssim_srgb=0.8),
                _observation(
                    "pptx",
                    artifact,
                    1,
                    ssim_srgb=0.8 + (0.001 if artifact_index == 20 else 0.0),
                ),
            ]
        )
    calibration = calibrate_artifact_observations(
        observations,
        required_repeats=2,
        noise_multiplier=2.0,
        expected_targets=("pptx",),
        expected_artifacts={"pptx": expected},
    )
    estimate = calibration.targets["pptx"].metrics["ssim_srgb"]

    assert estimate.p95_span == pytest.approx(0.0)
    assert estimate.max_span == pytest.approx(0.001)
    assert estimate.recommended_min_delta == pytest.approx(0.001)


def test_gate_uses_stricter_of_policy_and_renderer_noise():
    observations = [
        _observation("pptx", "pptx:a", 0, ssim_srgb=0.80, lpips=0.30),
        _observation("pptx", "pptx:a", 1, ssim_srgb=0.81, lpips=0.28),
    ]
    calibration = calibrate_artifact_observations(
        observations,
        required_repeats=2,
        noise_multiplier=2.0,
        expected_targets=("pptx",),
    )

    assert calibration.effective_delta("pptx", "ssim_srgb", 0.005) == pytest.approx(
        0.02
    )
    assert not calibration.meaningful_gain(
        target="pptx",
        metric="ssim_srgb",
        incumbent=0.80,
        candidate=0.815,
        policy_delta=0.005,
    )
    assert calibration.meaningful_gain(
        target="pptx",
        metric="lpips",
        incumbent=0.30,
        candidate=0.25,
        policy_delta=0.001,
    )
    assert not calibration.meaningful_gain(
        target="pixel-runtime",
        metric="ssim_srgb",
        incumbent=0.80,
        candidate=0.80,
    )
    assert calibration.protects_baseline(
        target="pptx",
        metric="ssim_srgb",
        baseline=0.80,
        candidate=0.785,
        policy_tolerance=0.005,
    )


def test_calibration_round_trip_and_metric_directions():
    calibration = calibrate_artifact_observations(
        [
            _observation("svg", "svg:a", 0, ssim_srgb=0.8),
            _observation("svg", "svg:a", 1, ssim_srgb=0.8),
        ],
        required_repeats=2,
        expected_targets=("svg",),
    )
    restored = ArtifactGateCalibration.from_dict(calibration.as_dict())

    assert restored.as_dict() == calibration.as_dict()
    assert metric_gain("ssim_srgb", incumbent=0.8, candidate=0.9) == pytest.approx(0.1)
    assert metric_gain("lpips", incumbent=0.3, candidate=0.2) == pytest.approx(0.1)
    with pytest.raises(ValueError, match="unknown quality metric"):
        metric_gain("made-up", incumbent=0.0, candidate=1.0)


def test_versioned_release_calibration_is_loadable_and_complete():
    root = Path(__file__).resolve().parents[2]
    data = json.loads((root / "data" / "artifact-gates.json").read_text())
    calibration = ArtifactGateCalibration.from_dict(data)

    assert set(calibration.targets) == {"pixel-runtime", "svg", "pptx"}
    assert all(target.complete for target in calibration.targets.values())
    assert calibration.targets["pptx"].artifact_count == 21
    assert data["provenance"]["svg_renderer"].startswith("Playwright Chromium ")
