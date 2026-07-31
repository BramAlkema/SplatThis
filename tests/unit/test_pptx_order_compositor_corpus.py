import json
from pathlib import Path

import pytest

from tools.pptx_order_compositor_corpus import (
    _aggregate,
    _corpus_images,
    _ensure_selected_artifact,
)


def _metrics(recipe: str, ssim: float, lpips: float) -> dict[str, object]:
    return {
        "recipe": recipe,
        "ssim_srgb": ssim,
        "ms_ssim_luma": ssim,
        "psnr_srgb": 20.0,
        "lpips": lpips,
        "delta_e_ok_mean": 0.05,
        "delta_e_ok_p95": 0.10,
        "edge_chamfer": 2.0,
        "edge_gradient_l1": 0.03,
        "worst_roi_error": 0.08,
        "file_size_bytes": 1000,
        "render_time_sec": 1.0,
    }


def test_corpus_images_accepts_name_keyed_manifest(tmp_path: Path) -> None:
    manifest = {
        "images": {
            "alpha": {
                "name": "alpha",
                "path": "images/alpha.png",
                "content_class": "test",
            },
            "beta": {
                "name": "beta",
                "path": "images/beta.png",
                "content_class": "test",
            },
        }
    }
    (tmp_path / "corpus.json").write_text(json.dumps(manifest), encoding="utf-8")

    assert [item["name"] for item in _corpus_images(tmp_path, "beta")] == ["beta"]


def test_aggregate_records_per_image_selection_and_medians() -> None:
    images = [{"name": "alpha", "content_class": "test"}]
    report = {
        "width": 10,
        "height": 8,
        "splat_count": 12,
        "pptx_splat_style": "gradient",
        "results": [
            _metrics("legacy-order", 0.70, 0.30),
            _metrics("corrected-order", 0.80, 0.20),
        ],
        "selection": {
            "selected_recipe": "corrected-order",
            "decisions": [{"failures": []}],
        },
        "capture": {
            "legacy-order": {"captures": [{}]},
            "corrected-order": {"captures": [{}]},
        },
    }

    summary = _aggregate(images, [report])

    assert summary["selection_counts"] == {
        "corrected-order": 1,
        "legacy-order": 0,
    }
    assert summary["median_delta_corrected_minus_legacy"]["ssim_srgb"] == pytest.approx(
        0.1
    )
    assert summary["per_image"][0]["captures_successful"] is True


def test_resumed_run_materializes_selected_native_deck(tmp_path: Path) -> None:
    (tmp_path / "legacy-order.pptx").write_bytes(b"legacy")
    (tmp_path / "corrected-order.pptx").write_bytes(b"corrected")
    report = {"selection": {"selected_recipe": "corrected-order"}}

    selected = _ensure_selected_artifact(tmp_path, report)

    assert selected.read_bytes() == b"corrected"
    assert report["selected_artifact"]["recipe"] == "corrected-order"
    assert len(report["selected_artifact"]["sha256"]) == 64
