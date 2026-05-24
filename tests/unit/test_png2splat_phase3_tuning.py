"""Phase 3 tests: profile tuning and exporter round-trip validation."""

import json
from pathlib import Path

import numpy as np
from PIL import Image

from png2svg_gs.cli import DEFAULT_APPLE_SILICON_SPLAT_CAP, _resolve_cli_resource_limits
from png2svg_gs.converter import PNG2SVGConverter
from png2svg_gs.io import validate_export_roundtrip
from png2svg_gs.splat import LAYER_DETAIL, LAYER_EDGE, create_isotropic_splat


def test_quality_profiles_provide_distinct_tuning_defaults():
    """Fast and max-fidelity profiles should differ in LR/refinement defaults."""
    m4_loop = PNG2SVGConverter(quality_profile="m4-fast-loop")
    fast = PNG2SVGConverter(quality_profile="fast")
    balanced = PNG2SVGConverter(quality_profile="balanced")
    max_fid = PNG2SVGConverter(quality_profile="max-fidelity")

    assert m4_loop.learning_rates != balanced.learning_rates
    assert (
        m4_loop.schedule_config["check_interval"]
        < fast.schedule_config["check_interval"]
    )
    assert (
        m4_loop.refinement_config["coverage_target"]
        <= fast.refinement_config["coverage_target"]
    )
    assert m4_loop.refinement_config["structure_precompute_enabled"] is True
    assert fast.refinement_config["structure_precompute_enabled"] is False
    assert balanced.refinement_config["structure_precompute_enabled"] is False
    assert max_fid.refinement_config["structure_precompute_enabled"] is False
    assert max_fid.refinement_config["residual_detail_enabled"] is True
    assert fast.refinement_config["residual_detail_enabled"] is False
    assert balanced.refinement_config["residual_detail_enabled"] is False
    assert max_fid.loss_weights["gradient_weight"] > 0.0
    assert fast.learning_rates != balanced.learning_rates
    assert max_fid.learning_rates != balanced.learning_rates
    assert (
        fast.refinement_config["densify_percentile"]
        > max_fid.refinement_config["densify_percentile"]
    )
    assert (
        max_fid.refinement_config["densify_fraction"]
        > fast.refinement_config["densify_fraction"]
    )
    assert max_fid.svg_export_recipe == "standard"


def test_export_roundtrip_validation_reports_consistent_counts():
    """Round-trip validator should preserve splat count and pass tolerances."""
    splats = [
        create_isotropic_splat(
            center=np.array([8.0, 8.0]),
            sigma=2.0,
            color=np.array([1.0, 0.2, 0.1]),
            alpha=0.8,
        ),
        create_isotropic_splat(
            center=np.array([16.0, 10.0]),
            sigma=1.5,
            color=np.array([0.1, 0.6, 0.9]),
            alpha=0.5,
        ),
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

    manifest = json.loads(
        (artifacts_path / "run_manifest.json").read_text(encoding="utf-8")
    )
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

    manifest = json.loads(
        (artifacts_path / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["config"]["blend_mode"] == "alpha-over"


def test_time_budget_plan_scales_splats_with_native_pixels_and_saliency():
    """The budget prepass should give larger native images a larger splat budget."""

    def guidance(
        width: int, height: int, *, foreground: float, edge: float, background: float
    ):
        area = width * height
        return {
            "summary": {
                "foreground_pixels": int(area * foreground),
                "edge_band_pixels": int(area * edge),
                "background_safe_pixels": int(area * background),
                "saliency_mean": 0.35,
                "saliency_p95": 0.80,
            },
            "weight_map": np.full((2, 2), 0.85, dtype=np.float32),
        }

    small = PNG2SVGConverter(
        max_splats=5000, time_budget="30m", apple_silicon_splat_cap=None
    )
    small_plan = small._apply_time_budget_plan(
        width=400,
        height=267,
        guidance=guidance(400, 267, foreground=0.30, edge=0.25, background=0.55),
    )
    native = PNG2SVGConverter(
        max_splats=5000, time_budget="30m", apple_silicon_splat_cap=None
    )
    native_plan = native._apply_time_budget_plan(
        width=1500,
        height=1000,
        guidance=guidance(1500, 1000, foreground=0.30, edge=0.25, background=0.55),
    )

    assert small_plan["selected_max_splats"] < native_plan["selected_max_splats"]
    assert native_plan["selected_max_splats"] > 2000
    assert native_plan["preset_ceiling"] is None
    assert native.stages == [80, 60, 40, 20]
    assert native.refinement_config["saliency_init_fraction"] > 0.40
    assert native.refinement_config["densify_weight_saliency"] > 0.80


def test_20m_budget_sits_between_10m_and_30m():
    """The 20m preset should bridge the interactive and long-run budgets."""

    def guidance(width: int, height: int):
        area = width * height
        return {
            "summary": {
                "foreground_pixels": int(area * 0.30),
                "edge_band_pixels": int(area * 0.25),
                "background_safe_pixels": int(area * 0.55),
                "saliency_mean": 0.35,
                "saliency_p95": 0.80,
            },
            "weight_map": np.full((2, 2), 0.85, dtype=np.float32),
        }

    budget_10m = PNG2SVGConverter(
        max_splats=5000, time_budget="10m", apple_silicon_splat_cap=None
    )
    budget_20m = PNG2SVGConverter(
        max_splats=5000, time_budget="20m", apple_silicon_splat_cap=None
    )
    budget_30m = PNG2SVGConverter(
        max_splats=5000, time_budget="30m", apple_silicon_splat_cap=None
    )

    plan_10m = budget_10m._apply_time_budget_plan(
        width=1500, height=1000, guidance=guidance(1500, 1000)
    )
    plan_20m = budget_20m._apply_time_budget_plan(
        width=1500, height=1000, guidance=guidance(1500, 1000)
    )
    plan_30m = budget_30m._apply_time_budget_plan(
        width=1500, height=1000, guidance=guidance(1500, 1000)
    )

    assert (
        plan_10m["selected_max_splats"]
        < plan_20m["selected_max_splats"]
        < plan_30m["selected_max_splats"]
    )
    assert budget_20m.stages == [32, 12]
    assert (
        budget_10m.refinement_config["densify_weight_saliency"]
        < budget_20m.refinement_config["densify_weight_saliency"]
    )
    assert (
        budget_20m.refinement_config["densify_weight_saliency"]
        < budget_30m.refinement_config["densify_weight_saliency"]
    )
    assert budget_20m.refinement_config["residual_detail_time_reserve_sec"] > 0.0
    assert plan_20m["initial_splat_estimate"] > 1200
    assert budget_20m.refinement_config["initial_splat_cap"] > 1200
    assert budget_20m.refinement_config["edge_init_fraction"] > 0.50
    assert budget_20m.refinement_config["residual_detail_edge_fraction"] > 0.60
    assert (
        budget_20m.refinement_config["residual_detail_edge_sigma_max"]
        < budget_20m.refinement_config["residual_detail_sigma_max"]
    )


def test_initial_splat_population_cap_is_configurable_for_large_runs():
    """Large native/ROI runs should be able to start above the legacy 1200 cap."""
    image = np.zeros((32, 32, 3), dtype=np.float32)
    image[8:24, 8:24] = np.array([0.8, 0.2, 0.1], dtype=np.float32)

    default = PNG2SVGConverter(
        max_splats=4000, stages=[1], seed=41, device="cpu", apple_silicon_splat_cap=None
    )
    expanded = PNG2SVGConverter(
        max_splats=4000,
        stages=[1],
        seed=41,
        device="cpu",
        apple_silicon_splat_cap=None,
        refinement_config={"initial_splat_fraction": 0.80, "initial_splat_cap": 3000},
    )

    assert default._initial_splat_count() == 1200
    assert expanded._initial_splat_count() == 3000

    default_splats = default._initialize_splats(image, rng=np.random.default_rng(41))
    expanded_splats = expanded._initialize_splats(image, rng=np.random.default_rng(41))
    assert len(expanded_splats) > len(default_splats)
    assert len(default_splats) <= default._initial_splat_count()
    assert len(expanded_splats) <= expanded._initial_splat_count()


def test_photo_native_10k_budget_is_mass_first_not_edge_heavy():
    """The native-photo 10k preset should target high count without edge-confetti defaults."""

    area = 1500 * 1000
    guidance = {
        "summary": {
            "foreground_pixels": int(area * 0.30),
            "edge_band_pixels": int(area * 0.25),
            "background_safe_pixels": int(area * 0.55),
            "saliency_mean": 0.35,
            "saliency_p95": 0.80,
        },
        "weight_map": np.full((2, 2), 0.85, dtype=np.float32),
    }

    converter = PNG2SVGConverter(
        max_splats=10000,
        time_budget="photo-10k",
        apple_silicon_splat_cap=None,
    )
    plan = converter._apply_time_budget_plan(width=1500, height=1000, guidance=guidance)

    assert plan["preset"] == "photo-native-10k"
    assert plan["selected_max_splats"] == 10000
    assert plan["initial_splat_estimate"] > 5000
    assert converter.stages == [36, 20, 10]
    assert (
        converter.refinement_config["base_layer_fraction"]
        > converter.refinement_config["edge_init_fraction"]
    )
    assert converter.refinement_config["edge_init_fraction"] < 0.30
    assert converter.refinement_config["residual_detail_edge_fraction"] < 0.30
    assert (
        converter.refinement_config["densify_weight_edge"]
        < converter.refinement_config["densify_weight_saliency"]
    )
    assert converter.refinement_config["background_suppressed_saliency_enabled"] is True
    assert (
        converter.refinement_config["background_suppressed_saliency_use_for_sampling"]
        is True
    )
    assert converter.refinement_config["renderer_tile_size"] == 24


def test_cli_exact_photo_presets_lift_default_interactive_ceiling():
    """CLI defaults should not collapse explicit photo-native presets to 2k."""

    max_splats, cap = _resolve_cli_resource_limits(
        time_budget="photo-10k",
        splats=None,
        apple_silicon_splat_cap=None,
    )
    assert max_splats == 10000
    assert cap is None

    max_splats, cap = _resolve_cli_resource_limits(
        time_budget="photo-20k",
        splats=None,
        apple_silicon_splat_cap=None,
    )
    assert max_splats == 20000
    assert cap is None

    max_splats, cap = _resolve_cli_resource_limits(
        time_budget="10m",
        splats=None,
        apple_silicon_splat_cap=None,
    )
    assert max_splats == 2000
    assert cap == DEFAULT_APPLE_SILICON_SPLAT_CAP


def test_cli_exact_photo_presets_still_respect_explicit_user_ceiling_and_cap():
    """An explicit --splats or --apple-silicon-splat-cap remains a hard user limit."""

    max_splats, cap = _resolve_cli_resource_limits(
        time_budget="photo-10k",
        splats=3500,
        apple_silicon_splat_cap=2500,
    )
    assert max_splats == 3500
    assert cap == 2500


def test_photo_native_20k_budget_uses_benchmarked_tile_size_and_exact_initial_cap():
    """The 20k native-photo preset should be selectable without the 10k ceiling."""

    area = 1500 * 1000
    guidance = {
        "summary": {
            "foreground_pixels": int(area * 0.30),
            "edge_band_pixels": int(area * 0.25),
            "background_safe_pixels": int(area * 0.55),
            "saliency_mean": 0.35,
            "saliency_p95": 0.80,
        },
        "weight_map": np.full((2, 2), 0.85, dtype=np.float32),
    }

    converter = PNG2SVGConverter(
        max_splats=20000,
        time_budget="photo-20k",
        apple_silicon_splat_cap=None,
    )
    plan = converter._apply_time_budget_plan(width=1500, height=1000, guidance=guidance)

    assert plan["preset"] == "photo-native-20k"
    assert plan["selected_max_splats"] == 20000
    assert plan["initial_splat_estimate"] == 20000
    assert converter.stages == [24, 12, 6]
    assert converter.refinement_config["renderer_tile_size"] == 24
    assert (
        converter.refinement_config["background_suppressed_saliency_use_for_sampling"]
        is True
    )
    assert converter.refinement_config["edge_init_fraction"] < 0.30
    renderer = converter._create_training_renderer(width=64, height=64)
    assert renderer.tile_size == 24
    assert renderer.tile_bin_cache_stats()["rebuild_interval"] == 1


def test_region_guidance_exposes_continuous_saliency_and_layers_initial_splats():
    """Saliency guidance should bias placement and draw-order layers."""
    image = np.zeros((32, 32, 3), dtype=np.float32)
    image[:, :] = np.array([0.04, 0.04, 0.04], dtype=np.float32)
    image[8:24, 8:24] = np.array([0.85, 0.12, 0.08], dtype=np.float32)
    image[13:19, 13:19] = np.array([0.10, 0.75, 0.95], dtype=np.float32)

    converter = PNG2SVGConverter(
        max_splats=64,
        stages=[1],
        target_size=(32, 32),
        seed=23,
        device="cpu",
        layered_saliency=True,
    )
    guidance = converter._compute_region_guidance(image)
    converter._region_saliency_map = guidance["saliency_map"]
    converter._region_foreground_mask = guidance["foreground_mask"]
    converter._region_edge_band_mask = guidance["edge_band_mask"]

    assert guidance["saliency_map"].shape == (32, 32)
    assert guidance["summary"]["saliency_p95"] > guidance["summary"]["saliency_mean"]

    splats = converter._initialize_splats(image, rng=np.random.default_rng(23))
    layers = {splat.layer for splat in splats if splat.layer is not None}
    assert LAYER_DETAIL in layers or LAYER_EDGE in layers

    saliency_positions = converter._sample_map_positions(
        guidance["saliency_map"],
        count=8,
        rng=np.random.default_rng(24),
        percentile=70.0,
        jitter=0.0,
    )
    assert saliency_positions
    assert all(6 <= x <= 25 and 6 <= y <= 25 for x, y in saliency_positions)


def test_background_suppressed_saliency_reduces_border_background_priority():
    """The detail prior should prefer central subject detail over border-connected texture."""
    image = np.zeros((64, 64, 3), dtype=np.float32)
    image[:, :] = np.array([0.18, 0.23, 0.10], dtype=np.float32)
    for idx in range(0, 64, 4):
        image[:, idx : idx + 2] += np.array([0.05, 0.04, 0.01], dtype=np.float32)
        image[idx : idx + 2, :] += np.array([0.03, 0.05, 0.01], dtype=np.float32)
    image = np.clip(image, 0.0, 1.0)

    image[20:46, 18:48] = np.array([0.82, 0.66, 0.48], dtype=np.float32)
    image[26:32, 24:30] = np.array([0.08, 0.08, 0.07], dtype=np.float32)
    image[26:32, 37:43] = np.array([0.08, 0.08, 0.07], dtype=np.float32)
    image[38:41, 28:39] = np.array([0.95, 0.92, 0.82], dtype=np.float32)

    converter = PNG2SVGConverter(
        max_splats=128,
        stages=[1],
        target_size=(64, 64),
        seed=31,
        device="cpu",
        refinement_config={
            "background_suppressed_saliency_enabled": True,
            "background_suppressed_saliency_use_for_sampling": True,
        },
    )
    guidance = converter._compute_region_guidance(image)
    priority = guidance["detail_priority_map"]
    penalty = guidance["background_penalty_map"]

    center_priority = float(np.mean(priority[24:42, 24:42]))
    border_priority = float(
        np.mean(
            np.concatenate(
                [
                    priority[:8, :].reshape(-1),
                    priority[-8:, :].reshape(-1),
                    priority[:, :8].reshape(-1),
                    priority[:, -8:].reshape(-1),
                ]
            )
        )
    )
    center_penalty = float(np.mean(penalty[24:42, 24:42]))
    border_penalty = float(
        np.mean(
            np.concatenate(
                [
                    penalty[:8, :].reshape(-1),
                    penalty[-8:, :].reshape(-1),
                    penalty[:, :8].reshape(-1),
                    penalty[:, -8:].reshape(-1),
                ]
            )
        )
    )

    assert (
        guidance["summary"]["detail_priority_p95"]
        > guidance["summary"]["detail_priority_mean"]
    )
    assert center_priority > border_priority + 0.05
    assert border_penalty > center_penalty

    converter._region_saliency_map = guidance["saliency_map"]
    converter._region_detail_priority_map = priority
    biased = converter._apply_saliency_sampling_bias(
        np.ones((64, 64), dtype=np.float32), strength=1.0
    )
    assert float(np.mean(biased[24:42, 24:42])) > float(np.mean(biased[:8, :]))


def test_saliency_sampling_bias_preserves_salient_ranking():
    """Continuous saliency should bias otherwise similar candidate maps."""
    converter = PNG2SVGConverter(
        max_splats=16,
        stages=[1],
        target_size=(8, 8),
        seed=7,
        device="cpu",
        refinement_config={
            "saliency_sampling_gamma": 0.70,
            "saliency_sampling_additive": 0.20,
        },
    )
    score_map = np.full((8, 8), 0.5, dtype=np.float32)
    saliency = np.zeros((8, 8), dtype=np.float32)
    saliency[2:6, 2:6] = 1.0
    converter._region_saliency_map = saliency

    biased = converter._apply_saliency_sampling_bias(score_map, strength=0.80)

    assert biased.shape == score_map.shape
    assert float(biased[3, 3]) > float(biased[0, 0])
    assert int(np.argmax(biased)) in {
        row * 8 + col for row in range(2, 6) for col in range(2, 6)
    }


def test_time_budget_smoke_records_resolved_plan_in_manifest(tmp_path: Path):
    """A budgeted conversion should record the resolved cap, stages, and prepass ratios."""
    image = np.zeros((18, 18, 3), dtype=np.uint8)
    image[4:14, 4:14, 0] = 220
    image[7:11, 7:11, 2] = 255
    input_path = tmp_path / "budget_input.png"
    output_path = tmp_path / "budget_output.html"
    artifacts_path = tmp_path / "budget_artifacts"
    Image.fromarray(image).save(input_path)

    converter = PNG2SVGConverter(
        max_splats=24,
        target_size=(18, 18),
        seed=17,
        quality_profile="fast",
        device="cpu",
        time_budget="smoke",
        apple_silicon_splat_cap=None,
    )
    converter.convert(
        input_path=str(input_path),
        output_path=str(output_path),
        output_format="canvas",
        verbose=False,
        seed=17,
        artifacts_dir=str(artifacts_path),
    )

    manifest = json.loads(
        (artifacts_path / "run_manifest.json").read_text(encoding="utf-8")
    )
    plan = manifest["config"]["time_budget_plan"]
    assert manifest["config"]["time_budget"] == "1m"
    assert manifest["config"]["stages"] == [2]
    assert manifest["config"]["max_splats"] == 24
    assert plan["label"] == "1m smoke"
    assert plan["selected_max_splats"] == 24
    assert "region_guidance" in manifest["config"]
    assert manifest["final_splat_count"] <= 24
    assert manifest["timings_sec"]["load_png"] >= 0.0
    assert manifest["timings_sec"]["initialize_splats"] >= 0.0
    assert manifest["timings_sec"]["optimize_splats"] >= 0.0
    assert (
        manifest["timings_sec"]["total_wall"]
        >= manifest["timings_sec"]["optimize_splats"]
    )
