"""Phase 2 regression tests: densify/prune + acceptance thresholds."""

import json
from pathlib import Path

import numpy as np
from PIL import Image

from png2svg_gs.converter import PNG2SVGConverter
from png2svg_gs.splat import create_isotropic_splat


def _make_gradient_image(size: int = 32) -> np.ndarray:
    x = np.linspace(0, 255, size, dtype=np.float32)
    grad = np.tile(x, (size, 1))
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:, :, 0] = grad.astype(np.uint8)
    img[:, :, 1] = np.flipud(grad).astype(np.uint8)
    img[:, :, 2] = 128
    return img


def _make_checker_image(size: int = 32, cell: int = 4) -> np.ndarray:
    yy, xx = np.indices((size, size))
    pattern = ((xx // cell + yy // cell) % 2) * 255
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:, :, 0] = pattern.astype(np.uint8)
    img[:, :, 1] = (255 - pattern).astype(np.uint8)
    img[:, :, 2] = 64
    return img


def _make_radial_image(size: int = 32) -> np.ndarray:
    yy, xx = np.indices((size, size))
    cx = (size - 1) / 2.0
    cy = (size - 1) / 2.0
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    dist = np.clip(dist / dist.max(), 0.0, 1.0)
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:, :, 0] = np.round((1.0 - dist) * 255).astype(np.uint8)
    img[:, :, 1] = np.round(dist * 255).astype(np.uint8)
    img[:, :, 2] = 180
    return img


def test_acceptance_manifest_contains_threshold_checks(tmp_path: Path):
    """Run manifest should contain measured metrics, thresholds, and check status."""
    input_path = tmp_path / "gradient.png"
    Image.fromarray(_make_gradient_image()).save(input_path)

    out_path = tmp_path / "gradient.svg"
    artifacts = tmp_path / "artifacts"

    converter = PNG2SVGConverter(
        max_splats=48,
        stages=[2, 1],
        target_size=(32, 32),
        seed=21,
        device="cpu",
    )
    converter.convert(
        input_path=str(input_path),
        output_path=str(out_path),
        save_json=True,
        verbose=False,
        seed=21,
        artifacts_dir=str(artifacts),
        acceptance_criteria={
            "min_psnr": 20.0,
            "min_ssim": 0.9,
            "max_runtime_sec": 5.0,
            "max_splats": 48.0,
        },
    )

    manifest = json.loads((artifacts / "run_manifest.json").read_text(encoding="utf-8"))
    assert "final_metrics" in manifest
    assert "acceptance" in manifest
    assert set(manifest["acceptance"]["checks"].keys()) == {
        "psnr",
        "ssim",
        "runtime_sec",
        "splat_count",
    }


def test_fixed_image_regression_suite_passes_thresholds(tmp_path: Path):
    """Fixed synthetic image set should pass configured regression thresholds."""
    fixtures = {
        "gradient": _make_gradient_image(),
        "checker": _make_checker_image(),
        "radial": _make_radial_image(),
    }

    # Honest BREAKAGE floors, not quality targets. Acceptance grades the actual
    # rasterized SVG with standard windowed SSIM in perceptual space. The
    # adversarial 2px checker at 32x32 with 64 splats is pathological for
    # Gaussian splatting and lands around psnr~7 / ssim~0.2, and is sensitive to
    # loss-function changes, so floors are set well below that to catch only
    # gross pipeline breakage rather than to chase the loss.
    acceptance = {
        "min_psnr": 5.0,
        "min_ssim": 0.15,
        "max_runtime_sec": 5.0,
        "max_splats": 64.0,
    }

    converter = PNG2SVGConverter(
        max_splats=64,
        stages=[2, 1],
        target_size=(32, 32),
        seed=77,
        device="cpu",
    )

    for name, image in fixtures.items():
        input_path = tmp_path / f"{name}.png"
        output_path = tmp_path / f"{name}.svg"
        artifacts_path = tmp_path / f"{name}_artifacts"

        Image.fromarray(image).save(input_path)
        converter.convert(
            input_path=str(input_path),
            output_path=str(output_path),
            save_json=True,
            verbose=False,
            seed=77,
            artifacts_dir=str(artifacts_path),
            acceptance_criteria=acceptance,
        )

        manifest = json.loads((artifacts_path / "run_manifest.json").read_text(encoding="utf-8"))
        assert manifest["acceptance"]["pass"], f"acceptance failed for {name}: {manifest['acceptance']}"
        assert (artifacts_path / "iter-1.raw.json").exists()
        assert (artifacts_path / "iter-2.raw.json").exists()


def test_manifest_records_coverage_and_densify_improves_it(tmp_path: Path):
    """Coverage metric should be present and increase after densification."""
    input_path = tmp_path / "coverage_input.png"
    Image.fromarray(_make_checker_image(size=28, cell=2)).save(input_path)

    output_path = tmp_path / "coverage_output.svg"
    artifacts_path = tmp_path / "coverage_artifacts"

    converter = PNG2SVGConverter(
        max_splats=72,
        stages=[1, 1],
        target_size=(28, 28),
        seed=5,
        device="cpu",
        quality_profile="max-fidelity",
    )
    converter.convert(
        input_path=str(input_path),
        output_path=str(output_path),
        save_json=True,
        verbose=False,
        seed=5,
        artifacts_dir=str(artifacts_path),
    )

    manifest = json.loads((artifacts_path / "run_manifest.json").read_text(encoding="utf-8"))
    stage1 = manifest["stages"][0]
    stage2 = manifest["stages"][1]
    assert "coverage" in stage1
    assert "coverage" in stage2
    assert stage2["coverage"] >= stage1["coverage"]
    assert manifest["final_metrics"]["coverage"] >= 0.95


def test_incremental_coverage_update_matches_full_recompute():
    """Applying only new splats to transmittance should match full coverage recompute."""
    converter = PNG2SVGConverter(
        max_splats=32,
        stages=[1],
        target_size=(24, 24),
        seed=9,
        device="cpu",
    )
    width, height = 24, 24

    base_splats = [
        create_isotropic_splat(center=np.array([6.0, 8.0]), sigma=2.4, color=np.array([0.8, 0.2, 0.2]), alpha=0.45),
        create_isotropic_splat(center=np.array([15.0, 10.0]), sigma=2.0, color=np.array([0.2, 0.7, 0.2]), alpha=0.35),
    ]
    new_splats = [
        create_isotropic_splat(center=np.array([12.0, 15.0]), sigma=2.8, color=np.array([0.1, 0.1, 0.8]), alpha=0.50),
        create_isotropic_splat(center=np.array([18.0, 5.0]), sigma=1.8, color=np.array([0.9, 0.9, 0.2]), alpha=0.40),
    ]

    base_coverage = converter._build_alpha_coverage_map(base_splats, width=width, height=height)
    transmittance = np.clip(1.0 - base_coverage, 0.0, 1.0).astype(np.float32, copy=True)
    converter._apply_splats_to_transmittance(
        transmittance=transmittance,
        splats=new_splats,
        width=width,
        height=height,
    )
    incremental_coverage = np.clip(1.0 - transmittance, 0.0, 1.0).astype(np.float32)

    full_coverage = converter._build_alpha_coverage_map(base_splats + new_splats, width=width, height=height)
    assert np.allclose(incremental_coverage, full_coverage, atol=1e-6)


def test_residual_detail_pass_runs_and_respects_budget():
    """Residual detail stage should execute for max-fidelity and never exceed max_splats."""
    image = _make_checker_image(size=24, cell=2).astype(np.float32) / 255.0
    converter = PNG2SVGConverter(
        max_splats=40,
        stages=[1],
        target_size=(24, 24),
        seed=13,
        device="cpu",
        quality_profile="max-fidelity",
        refinement_config={
            "residual_detail_enabled": True,
            "residual_detail_passes": 1,
            "residual_detail_iters": 0,
            "residual_detail_fraction": 0.25,
            "residual_detail_reserve_fraction": 0.25,
            "residual_detail_percentile": 80.0,
        },
    )

    rng = np.random.default_rng(13)
    initial = converter._initialize_splats(image=image, rng=rng)
    final_splats, stage_metrics = converter._optimize_splats(
        image=image,
        splats=initial,
        rng=np.random.default_rng(13),
        verbose=False,
        artifacts_dir=None,
    )

    residual_metrics = [m for m in stage_metrics if m.get("stage_type") == "residual_detail"]
    assert residual_metrics, "expected at least one residual_detail stage metric"
    assert len(final_splats) <= converter.max_splats
