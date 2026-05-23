"""Phase 1 tests for canonical png2splat pipeline behavior."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from png2svg_gs.converter import PNG2SVGConverter
from png2svg_gs.features import analyze_local_structure, compute_structure_field, init_seeds_content_adaptive
from png2svg_gs.io import load_splats_json, save_splats_json
from png2svg_gs.renderer import L1SSIMLoss, splats_to_tensor, tensor_to_splats
from png2svg_gs.splat import GaussianSplat, RawSplat, create_isotropic_splat


def test_raw_splat_validation_rejects_invalid_scale():
    """Raw schema must reject non-positive scales."""
    with pytest.raises(ValueError, match="sx must be"):
        RawSplat(
            x=1.0,
            y=2.0,
            sx=0.0,
            sy=1.0,
            theta=0.0,
            r=0.1,
            g=0.2,
            b=0.3,
            a=0.5,
        )


def test_gaussian_raw_roundtrip_preserves_core_parameters():
    """Gaussian <-> raw schema roundtrip should preserve key values."""
    splat = create_isotropic_splat(center=np.array([12.0, 9.0]), sigma=3.0, color=np.array([0.2, 0.4, 0.6]), alpha=0.7)
    splat.layer = 2
    raw = splat.to_raw_splat()
    restored = GaussianSplat.from_raw_splat(raw)

    assert np.allclose(restored.mu, splat.mu, atol=1e-5)
    assert np.allclose(restored.color[:3], splat.color[:3], atol=1e-5)
    assert restored.alpha == pytest.approx(splat.alpha, abs=1e-5)
    assert raw.layer == 2
    assert restored.layer == 2
    assert raw.sx > 0
    assert raw.sy > 0


def test_tensor_parameterization_uses_scale_rotation_and_roundtrips():
    """Tensor format should store sx/sy/theta and reconstruct covariance correctly."""
    splat = create_isotropic_splat(
        center=np.array([8.0, 6.0]),
        sigma=3.0,
        color=np.array([0.3, 0.5, 0.7]),
        alpha=0.8,
    )
    tensor = splats_to_tensor([splat])
    assert float(tensor[0, 2]) == pytest.approx(3.0, abs=1e-5)
    assert float(tensor[0, 3]) == pytest.approx(3.0, abs=1e-5)
    assert np.isfinite(float(tensor[0, 4]))

    restored = tensor_to_splats(tensor)[0]
    assert np.allclose(restored.mu, splat.mu, atol=1e-5)
    assert np.allclose(restored.sigma, splat.sigma, atol=1e-5)


def test_save_and_load_splats_json_canonical(tmp_path: Path):
    """Canonical raw JSON should serialize and deserialize successfully."""
    splats = [
        create_isotropic_splat(center=np.array([4.0, 5.0]), sigma=2.0, color=np.array([1.0, 0.0, 0.0]), alpha=0.9),
        create_isotropic_splat(center=np.array([8.0, 9.0]), sigma=1.5, color=np.array([0.0, 1.0, 0.0]), alpha=0.6),
    ]
    splats[0].layer = 0
    splats[1].layer = 2
    out_path = tmp_path / "splats.json"
    save_splats_json(splats, str(out_path))

    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["schema"] == "png2splat.raw/1"
    assert data["num_splats"] == 2
    assert data["layers"] == [
        {"count": 1, "id": 0, "name": "base"},
        {"count": 1, "id": 2, "name": "detail"},
    ]
    assert {"x", "y", "sx", "sy", "theta", "r", "g", "b", "a"}.issubset(data["splats"][0])

    loaded = load_splats_json(str(out_path))
    assert len(loaded) == 2
    assert np.allclose(loaded[0].mu, splats[0].mu, atol=1e-5)
    assert loaded[0].layer == 0
    assert loaded[1].layer == 2


def test_seeded_initializer_is_deterministic():
    """Content-adaptive seeding should be deterministic with a fixed RNG seed."""
    image = np.linspace(0.0, 1.0, 32 * 32 * 3, dtype=np.float32).reshape(32, 32, 3)
    seeds_a = init_seeds_content_adaptive(image, target_count=24, rng=np.random.default_rng(123))
    seeds_b = init_seeds_content_adaptive(image, target_count=24, rng=np.random.default_rng(123))
    assert seeds_a == seeds_b


def test_structure_field_has_valid_shapes_and_ranges():
    """Structure field precompute should return normalized directions and anisotropy >= 1."""
    image = np.linspace(0.0, 1.0, 24 * 20 * 3, dtype=np.float32).reshape(24, 20, 3)
    dirs, anis = compute_structure_field(image, method="sobel", smoothing_sigma=1.0, anisotropy_clip=7.0)

    assert dirs.shape == (24, 20, 2)
    assert anis.shape == (24, 20)
    assert np.all(np.isfinite(dirs))
    assert np.all(np.isfinite(anis))
    assert float(np.min(anis)) >= 1.0
    assert float(np.max(anis)) <= 7.0 + 1e-6


def test_local_structure_on_uniform_patch_is_isotropic():
    """Flat regions should not trigger high-anisotropy orientation."""
    image = np.full((21, 21, 3), 0.5, dtype=np.float32)
    direction, anisotropy = analyze_local_structure(image, x=10, y=10)
    assert np.allclose(direction, np.array([1.0, 0.0], dtype=np.float32), atol=1e-6)
    assert anisotropy == pytest.approx(1.0, abs=1e-6)


def test_l1_ssim_loss_is_zero_for_identical_images():
    """L1+SSIM loss should be near zero for identical tensors."""
    loss_fn = L1SSIMLoss(l1_weight=1.0, ssim_weight=0.2)
    image = torch.rand(12, 12, 3)
    loss = loss_fn(image, image)
    assert float(loss.item()) == pytest.approx(0.0, abs=1e-7)


def test_converter_writes_artifacts_and_is_seed_reproducible(tmp_path: Path):
    """Two seeded runs should produce identical raw JSON output."""
    image = np.zeros((24, 24, 3), dtype=np.uint8)
    image[:, :12, 0] = 255
    image[:, 12:, 1] = 255
    input_path = tmp_path / "input.png"
    Image.fromarray(image).save(input_path)

    out1 = tmp_path / "run1.svg"
    out2 = tmp_path / "run2.svg"
    artifacts1 = tmp_path / "artifacts1"
    artifacts2 = tmp_path / "artifacts2"

    converter = PNG2SVGConverter(
        max_splats=32,
        stages=[1],
        target_size=(24, 24),
        seed=7,
        device="cpu",
    )

    converter.convert(
        input_path=str(input_path),
        output_path=str(out1),
        save_json=True,
        verbose=False,
        seed=7,
        artifacts_dir=str(artifacts1),
    )
    converter.convert(
        input_path=str(input_path),
        output_path=str(out2),
        save_json=True,
        verbose=False,
        seed=7,
        artifacts_dir=str(artifacts2),
    )

    json1 = Path(str(out1.with_suffix(".json"))).read_text(encoding="utf-8")
    json2 = Path(str(out2.with_suffix(".json"))).read_text(encoding="utf-8")
    assert json1 == json2

    manifest = json.loads((artifacts1 / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["seed"] == 7
    assert (artifacts1 / "init.raw.json").exists()
    assert (artifacts1 / "iter-1.raw.json").exists()
    assert (artifacts1 / "final.raw.json").exists()
