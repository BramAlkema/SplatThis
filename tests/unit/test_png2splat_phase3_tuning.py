"""Phase 3 tests: profile tuning and exporter round-trip validation."""

import json
from pathlib import Path

import numpy as np
from PIL import Image

from png2svg_gs.converter import PNG2SVGConverter
from png2svg_gs.io import validate_export_roundtrip
from png2svg_gs.splat import create_isotropic_splat


def test_quality_profiles_provide_distinct_tuning_defaults():
    """Fast and max-fidelity profiles should differ in LR/refinement defaults."""
    m4_loop = PNG2SVGConverter(quality_profile="m4-fast-loop")
    fast = PNG2SVGConverter(quality_profile="fast")
    balanced = PNG2SVGConverter(quality_profile="balanced")
    max_fid = PNG2SVGConverter(quality_profile="max-fidelity")

    assert m4_loop.learning_rates != balanced.learning_rates
    assert m4_loop.schedule_config["check_interval"] < fast.schedule_config["check_interval"]
    assert m4_loop.refinement_config["coverage_target"] <= fast.refinement_config["coverage_target"]
    assert m4_loop.refinement_config["structure_precompute_enabled"] is True
    assert fast.refinement_config["structure_precompute_enabled"] is False
    assert balanced.refinement_config["structure_precompute_enabled"] is False
    assert max_fid.refinement_config["structure_precompute_enabled"] is False
    assert max_fid.refinement_config["residual_detail_enabled"] is True
    assert fast.refinement_config["residual_detail_enabled"] is False
    assert balanced.refinement_config["residual_detail_enabled"] is False
    assert fast.learning_rates != balanced.learning_rates
    assert max_fid.learning_rates != balanced.learning_rates
    assert fast.refinement_config["densify_percentile"] > max_fid.refinement_config["densify_percentile"]
    assert max_fid.refinement_config["densify_fraction"] > fast.refinement_config["densify_fraction"]


def test_export_roundtrip_validation_reports_consistent_counts():
    """Round-trip validator should preserve splat count and pass tolerances."""
    splats = [
        create_isotropic_splat(center=np.array([8.0, 8.0]), sigma=2.0, color=np.array([1.0, 0.2, 0.1]), alpha=0.8),
        create_isotropic_splat(center=np.array([16.0, 10.0]), sigma=1.5, color=np.array([0.1, 0.6, 0.9]), alpha=0.5),
    ]
    result = validate_export_roundtrip(splats, width=32, height=24, k_sigma=2.5)
    assert result["pass"]
    assert result["svg_ellipse_count"] == 2
    assert result["drawingml_shape_count"] == 2


def test_converter_manifest_includes_roundtrip_validation_when_enabled(tmp_path: Path):
    """Converter should emit roundtrip_validation block in run manifest."""
    image = np.zeros((24, 24, 3), dtype=np.uint8)
    image[:, :12, 0] = 255
    image[:, 12:, 2] = 255
    input_path = tmp_path / "input.png"
    output_path = tmp_path / "output.svg"
    artifacts_path = tmp_path / "artifacts"
    Image.fromarray(image).save(input_path)

    converter = PNG2SVGConverter(
        max_splats=48,
        stages=[2, 1],
        target_size=(24, 24),
        seed=33,
        quality_profile="max-fidelity",
        device="cpu",
    )
    converter.convert(
        input_path=str(input_path),
        output_path=str(output_path),
        save_json=True,
        verbose=False,
        seed=33,
        artifacts_dir=str(artifacts_path),
        validate_roundtrip=True,
    )

    manifest = json.loads((artifacts_path / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["config"]["quality_profile"] == "max-fidelity"
    assert "roundtrip_validation" in manifest
    assert manifest["roundtrip_validation"]["pass"]


def test_converter_accepts_alpha_over_blend_mode(tmp_path: Path):
    """Converter should run with alpha-over blend mode and record it in manifest."""
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    image[:, 5:15, 1] = 220
    input_path = tmp_path / "input_blend.png"
    output_path = tmp_path / "output_blend.svg"
    artifacts_path = tmp_path / "artifacts_blend"
    Image.fromarray(image).save(input_path)

    converter = PNG2SVGConverter(
        max_splats=40,
        stages=[1],
        target_size=(20, 20),
        seed=11,
        device="cpu",
        blend_mode="alpha-over",
    )
    converter.convert(
        input_path=str(input_path),
        output_path=str(output_path),
        save_json=True,
        verbose=False,
        seed=11,
        artifacts_dir=str(artifacts_path),
    )

    manifest = json.loads((artifacts_path / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["config"]["blend_mode"] == "alpha-over"
