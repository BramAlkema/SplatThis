"""In-process smoke tests for the splatlify CLI (png2svg_gs.cli.main)."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import pytest
from PIL import Image

from png2svg_gs import __version__
from png2svg_gs.cli import main
from png2svg_gs.mlx_stage import is_mlx_available


def _write_fixture_image(tmp_path):
    """Write a small 16x16 RGB PNG with some structure (not a flat color)."""
    rng = np.random.default_rng(0)
    pixels = rng.integers(0, 256, size=(16, 16, 3), dtype=np.uint8)
    # Add a deterministic gradient so the image has clear low-frequency content.
    ramp = np.linspace(0, 255, 16, dtype=np.uint8)
    pixels[:, :, 0] = ramp[np.newaxis, :]
    path = tmp_path / "fixture.png"
    Image.fromarray(pixels, mode="RGB").save(path)
    return path


def _run_cli(args):
    """Invoke main() in-process; normalize SystemExit to a return code."""
    try:
        return main(args)
    except SystemExit as exc:  # pragma: no cover - main() returns int today
        return exc.code


def test_cli_smoke_torch_svg(tmp_path):
    img = _write_fixture_image(tmp_path)
    out_svg = tmp_path / "nested" / "out.svg"
    code = _run_cli(
        [
            str(img),
            "-o",
            str(out_svg),
            "--optimizer-backend",
            "torch",
            "--stages",
            "2,1",
            "--splats",
            "16",
        ]
    )
    assert code in (0, None)
    assert out_svg.is_file()
    root = ET.parse(out_svg).getroot()
    assert root.tag.endswith("svg")


def test_cli_version(capsys):
    assert _run_cli(["--version"]) == 0
    assert capsys.readouterr().out.strip() == f"splatlify {__version__}"


def test_cli_rejects_invalid_resource_limit():
    with pytest.raises(SystemExit) as exc:
        main(["source.png", "--splats", "0"])
    assert exc.value.code == 2


def test_cli_rejects_invalid_adaptive_target():
    with pytest.raises(SystemExit) as exc:
        main(["source.png", "--adaptive-target-ssim-srgb", "1.01"])
    assert exc.value.code == 2


def test_cli_rejects_negative_adaptive_chrome_margin():
    with pytest.raises(SystemExit) as exc:
        main(["source.png", "--adaptive-chrome-ssim-margin", "-0.001"])
    assert exc.value.code == 2


def test_cli_reports_corrupt_image_without_traceback(tmp_path, capsys):
    corrupt = tmp_path / "corrupt.png"
    corrupt.write_bytes(b"not an image")

    assert _run_cli([str(corrupt), "--optimizer-backend", "torch"]) == 1
    captured = capsys.readouterr()
    assert captured.err.startswith("error:")
    assert "Traceback" not in captured.err


@pytest.mark.skipif(not is_mlx_available(), reason="MLX is not installed")
def test_cli_smoke_mlx_svg(tmp_path):
    img = _write_fixture_image(tmp_path)
    out_svg = tmp_path / "out_mlx.svg"
    code = _run_cli(
        [
            str(img),
            "-o",
            str(out_svg),
            "--optimizer-backend",
            "mlx",
            "--stages",
            "2,1",
            "--splats",
            "16",
        ]
    )
    assert code in (0, None)
    assert out_svg.is_file()
    root = ET.parse(out_svg).getroot()
    assert root.tag.endswith("svg")


def test_cli_smoke_torch_pptx(tmp_path):
    img = _write_fixture_image(tmp_path)
    out_pptx = tmp_path / "out.pptx"
    artifacts = tmp_path / "pptx-artifacts"
    code = _run_cli(
        [
            str(img),
            "-o",
            str(out_pptx),
            "--format",
            "pptx",
            "--optimizer-backend",
            "torch",
            "--stages",
            "2,1",
            "--splats",
            "16",
            "--pptx-painter-order",
            "back-to-front",
            "--artifacts-dir",
            str(artifacts),
        ]
    )
    assert code in (0, None)
    assert out_pptx.is_file()
    with zipfile.ZipFile(out_pptx) as zf:
        names = zf.namelist()
    assert "[Content_Types].xml" in names
    assert any(name.startswith("ppt/slides/") for name in names)
    manifest = json.loads((artifacts / "run_manifest.json").read_text())
    assert manifest["config"]["pptx_painter_order"] == "back-to-front"


def test_cli_smoke_scriptless_css_compositor(tmp_path):
    img = _write_fixture_image(tmp_path)
    out_html = tmp_path / "out.html"
    artifacts = tmp_path / "artifacts"
    code = _run_cli(
        [
            str(img),
            "-o",
            str(out_html),
            "--format",
            "css",
            "--optimizer-backend",
            "torch",
            "--stages",
            "2,1",
            "--splats",
            "16",
            "--layered-saliency",
            "--css-parallax-strength",
            "8",
            "--artifacts-dir",
            str(artifacts),
        ]
    )

    assert code in (0, None)
    html = out_html.read_text(encoding="utf-8")
    assert 'data-compositor="css-splats"' in html
    assert 'data-grid="10"' in html
    assert "<script" not in html.lower()
    assert "<canvas" not in html.lower()
    manifest = json.loads((artifacts / "run_manifest.json").read_text())
    assert manifest["config"]["training_export_target"] == "svg"
    assert manifest["artifact_evaluation"]["render_kind"] in {
        "css-browser-capture",
        "css-browser-unavailable",
    }


def test_cli_smoke_native_canvas_compositor(tmp_path):
    img = _write_fixture_image(tmp_path)
    out_html = tmp_path / "out.html"
    artifacts = tmp_path / "artifacts"
    code = _run_cli(
        [
            str(img),
            "-o",
            str(out_html),
            "--format",
            "canvas",
            "--optimizer-backend",
            "torch",
            "--stages",
            "2,1",
            "--splats",
            "16",
            "--artifacts-dir",
            str(artifacts),
        ]
    )

    assert code in (0, None)
    html = out_html.read_text(encoding="utf-8")
    assert 'data-compositor="canvas-api-splats"' in html
    assert "createRadialGradient" in html
    assert "putImageData" not in html
    manifest = json.loads((artifacts / "run_manifest.json").read_text())
    assert manifest["config"]["training_export_target"] == "svg"
    assert manifest["artifact_evaluation"]["render_kind"] in {
        "canvas-api-browser-capture",
        "canvas-api-browser-unavailable",
    }


def test_cli_smoke_pixel_runtime_adaptive_compute(tmp_path):
    img = _write_fixture_image(tmp_path)
    out_html = tmp_path / "out.html"
    artifacts = tmp_path / "artifacts"
    code = _run_cli(
        [
            str(img),
            "-o",
            str(out_html),
            "--format",
            "pixel-runtime",
            "--optimizer-backend",
            "torch",
            "--stages",
            "1,1,1",
            "--splats",
            "16",
            "--adaptive-compute",
            "--adaptive-target-ssim-srgb",
            "0",
            "--adaptive-min-checkpoints",
            "2",
            "--artifacts-dir",
            str(artifacts),
        ]
    )

    assert code in (0, None)
    assert out_html.is_file()
    html = out_html.read_text(encoding="utf-8")
    assert 'data-compositor="pixel-runtime"' in html
    assert "gl.RGBA32F" in html
    assert "worker-offscreen" in html
    assert "putImageData" in html
    manifest = json.loads((artifacts / "run_manifest.json").read_text())
    adaptive = next(
        stage
        for stage in manifest["stages"]
        if stage.get("stage_type") == "canvas_adaptive_compute"
    )
    assert adaptive["stopped_early"] is True
    assert adaptive["skipped_main_stages"] == 1
    assert adaptive["policy"]["chrome_ssim_safety_margin"] == 0.0
    assert adaptive["policy"]["effective_model_ssim_threshold"] == 0.0
    assert adaptive["policy"]["runtime_scorer"] == "canvas-image-data-byte-v1"
    assert manifest["artifact_evaluation"]["render_kind"] in {
        "pixel-runtime-browser-capture",
        "pixel-runtime-browser-unavailable",
    }
