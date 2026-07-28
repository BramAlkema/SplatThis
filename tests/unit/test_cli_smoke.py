"""In-process smoke tests for the splatlify CLI (png2svg_gs.cli.main)."""

from __future__ import annotations

import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import pytest
from PIL import Image

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
    out_svg = tmp_path / "out.svg"
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
        ]
    )
    assert code in (0, None)
    assert out_pptx.is_file()
    with zipfile.ZipFile(out_pptx) as zf:
        names = zf.namelist()
    assert "[Content_Types].xml" in names
    assert any(name.startswith("ppt/slides/") for name in names)
