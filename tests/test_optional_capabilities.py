import importlib.util
import subprocess
import tomllib
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from splatthis.browser_capture import read_svg_pixel_size
from splatthis.io import render_splats_preview_png, save_svg
from splatthis.population_embed import (
    decode_population,
    encode_population,
    load_population,
    population_from_png,
    population_from_pptx,
)
from splatthis.pptx_export import save_pptx_with_splats
from splatthis.splat import GaussianSplat, RawSplat


def _splats() -> list[GaussianSplat]:
    return [
        GaussianSplat.from_raw_splat(
            RawSplat(
                x=20 + index,
                y=18,
                sx=4,
                sy=2,
                theta=0.3,
                r=0.2,
                g=0.5,
                b=0.8,
                a=0.7,
                importance=0.2 + index / 10,
            )
        )
        for index in range(2)
    ]


def _assert_population_matches(actual: list[GaussianSplat]) -> None:
    expected = _splats()
    assert len(actual) == len(expected)
    for left, right in zip(actual, expected):
        np.testing.assert_allclose(left.mu, right.mu)
        np.testing.assert_allclose(left.sigma, right.sigma, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(left.color, right.color)
        assert left.alpha == pytest.approx(right.alpha)


def test_optional_capabilities_do_not_expand_default_dependencies():
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    names = {requirement.split(">=")[0] for requirement in project["dependencies"]}
    extras = project["optional-dependencies"]

    assert names == {"numpy", "Pillow", "torch"}
    assert extras["capture"] == ["playwright>=1.55"]
    assert extras["steg"] == ["stego-lsb>=1.4"]
    assert extras["mlx"][0].startswith("mlx>=0.15;")


def test_population_envelope_round_trip():
    _assert_population_matches(decode_population(encode_population(_splats())))


def test_svg_and_pptx_population_carriers_round_trip(tmp_path):
    svg = tmp_path / "scene.svg"
    pptx = tmp_path / "scene.pptx"
    save_svg(_splats(), 64, 48, str(svg), embed_population=True)
    save_pptx_with_splats(_splats(), 64, 48, str(pptx), embed_population=True)

    _assert_population_matches(load_population(str(svg)))
    _assert_population_matches(population_from_pptx(str(pptx)))


@pytest.mark.skipif(
    importlib.util.find_spec("stego_lsb") is None,
    reason="stego-lsb optional extra is not installed",
)
def test_png_text_and_pixel_carriers_survive_opposite_transformations(tmp_path):
    carrier = tmp_path / "carrier.png"
    pixels_only = tmp_path / "pixels-only.png"
    render_splats_preview_png(
        _splats(),
        96,
        96,
        str(carrier),
        embed_splats=_splats(),
        embed_in_pixels=True,
    )

    _assert_population_matches(population_from_png(str(carrier)))
    with Image.open(carrier) as image:
        image.convert("RGB").save(pixels_only)
    _assert_population_matches(population_from_png(str(pixels_only)))


def test_browser_capture_geometry_is_dependency_free(tmp_path):
    svg = tmp_path / "scene.svg"
    svg.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="64px" height="48" '
        'viewBox="0 0 64 48"/>'
    )

    assert read_svg_pixel_size(svg) == (64, 48)


def test_powerpoint_osa_capture_has_no_package_dependency(tmp_path, monkeypatch):
    from splatthis import powerpoint_osa

    source = tmp_path / "scene.pptx"
    output = tmp_path / "capture.png"
    source.write_bytes(b"pptx fixture")

    def fake_run(command, **options):
        assert command[:4] == [
            "/usr/bin/osascript",
            "-l",
            "AppleScript",
            "-e",
        ]
        assert command[-3] == str(source)
        Image.new("RGB", (80, 60), "red").save(command[-2], format="PNG")
        return subprocess.CompletedProcess(command, 0, stdout="10,20,80,60\n")

    monkeypatch.setattr(powerpoint_osa, "_require_macos_tools", lambda: None)
    monkeypatch.setattr(powerpoint_osa.subprocess, "run", fake_run)

    result = powerpoint_osa.capture_pptx_with_powerpoint(source, output)

    assert result["schema"] == "splatthis.powerpoint-osa-capture/1"
    assert result["width"] == 80
    assert result["height"] == 60
    assert output.is_file()


def test_cli_routes_pptx_capture_to_powerpoint_osa(tmp_path, monkeypatch):
    from splatthis import cli, powerpoint_osa

    source = tmp_path / "source.png"
    output = tmp_path / "scene.pptx"
    capture = tmp_path / "capture.png"
    Image.new("RGB", (8, 6), "blue").save(source)
    observed = {}

    class FakeConverter:
        def __init__(self, **options):
            pass

        def convert(self, input_path, output_path, **options):
            observed["format"] = options["output_format"]

    def fake_capture(pptx_path, output_path):
        observed["capture"] = (Path(pptx_path), Path(output_path))

    monkeypatch.setattr(cli, "PNG2SVGConverter", FakeConverter)
    monkeypatch.setattr(powerpoint_osa, "capture_pptx_with_powerpoint", fake_capture)

    assert (
        cli.main(
            [
                str(source),
                "--format",
                "pptx",
                "--output",
                str(output),
                "--capture",
                str(capture),
            ]
        )
        == 0
    )
    assert observed == {"format": "pptx", "capture": (output, capture)}


def test_mlx_request_falls_back_cleanly_without_an_executable_device(monkeypatch):
    from splatthis import mlx_runtime
    from splatthis.converter import SplatConverter

    monkeypatch.setattr(mlx_runtime, "is_mlx_available", lambda: False)
    monkeypatch.setattr(mlx_runtime, "is_mlx_imported", lambda: True)

    converter = SplatConverter(max_splats=4, stages=[1], optimizer_backend="mlx")

    assert converter.requested_optimizer_backend == "mlx"
    assert converter.optimizer_backend == "torch"


def test_cli_exposes_optional_capabilities():
    from splatthis.cli import build_parser

    args = build_parser().parse_args(
        [
            "source.png",
            "--optimizer-backend",
            "mlx",
            "--capture",
            "capture.png",
            "--embed-population",
            "--embed-population-in-pixels",
            "--preview",
            "preview.png",
        ]
    )

    assert args.optimizer_backend == "mlx"
    assert args.capture == Path("capture.png")
    assert args.embed_population is True
    assert args.embed_population_in_pixels is True
