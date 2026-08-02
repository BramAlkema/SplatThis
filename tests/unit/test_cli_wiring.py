"""The flag-to-config plumbing in ``cli._run_conversion``.

argparse validates that a flag parses; nothing validates that it *arrives*.
A flag whose value never reaches the converter fails silently, producing a
default run under a non-default command line. These tests stub the converter
and assert on what actually shows up in its constructor and ``convert`` call.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

import pytest
from PIL import Image

import splatthis.cli as cli

RunCli = Callable[..., Tuple[Dict[str, Any], Dict[str, Any]]]


@pytest.fixture()
def run_cli(tmp_path, monkeypatch) -> RunCli:
    png = tmp_path / "in.png"
    Image.new("RGB", (8, 8), (120, 30, 200)).save(png)
    captured: Dict[str, Dict[str, Any]] = {}

    class CaptureConverter:
        def __init__(self, **kwargs: Any) -> None:
            captured["init"] = kwargs

        def convert(self, **kwargs: Any) -> None:
            captured["convert"] = kwargs

    monkeypatch.setattr(cli, "PNG2SVGConverter", CaptureConverter)

    def run(*flags: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        assert cli.main([str(png), *flags]) == 0
        return captured["init"], captured["convert"]

    return run


def test_default_run_wires_the_documented_defaults(run_cli: RunCli) -> None:
    init, convert = run_cli()
    assert init["quality_profile"] == "max-fidelity"
    assert init["seed"] == 0
    # "auto" resolves to sRGB browser-gradient training for the svg target.
    assert init["refinement_config"] == {"training_export_target": "svg"}
    assert convert["output_format"] == "svg"
    assert convert["output_path"].endswith(".svg")


def test_svg_flags_reach_the_refinement_config(run_cli: RunCli) -> None:
    init, _ = run_cli(
        "--svg-recipe",
        "palette-quantized",
        "--svg-gradient-quality",
        "high",
        "--svg-painter-order",
        "legacy",
        "--no-svg-compositor-gate",
        "--fidelity-stage",
        "max",
        "--svg-optimize",
    )
    config = init["refinement_config"]
    assert config["svg_export_recipe"] == "palette-quantized"
    assert config["svg_gradient_quality"] == "high"
    assert config["svg_painter_order"] == "legacy"
    assert config["svg_compositor_gate"] is False
    assert config["fidelity_stage"] == "max"
    assert config["svg_optimize"] is True
    assert config["svg_optimize_precision"] == 2


@pytest.mark.parametrize(
    ("flags", "expected"),
    [
        # The pixel-runtime training model is the implicit default and is
        # therefore never written into the config.
        (("--training-export-target", "canvas"), None),
        (("--training-export-target", "pixel-runtime"), None),
        (("--format", "pptx"), None),
        (("--training-export-target", "browser-gradient"), "svg"),
        (("--format", "css"), "svg"),
    ],
)
def test_training_export_target_aliases_resolve(
    run_cli: RunCli, flags: Tuple[str, ...], expected: Any
) -> None:
    init, _ = run_cli(*flags)
    # An all-defaults refinement config is passed as None, not as {}.
    config = init["refinement_config"] or {}
    assert config.get("training_export_target") == expected


def test_css_and_adaptive_flags_arrive_typed(run_cli: RunCli) -> None:
    init, convert = run_cli(
        "--format",
        "css",
        "--css-parallax-strength",
        "28",
        "--css-hover-grid-size",
        "5",
        "--adaptive-compute",
        "--adaptive-target-ssim-srgb",
        "0.97",
        "--adaptive-min-checkpoints",
        "3",
    )
    config = init["refinement_config"]
    assert config["css_parallax_strength"] == pytest.approx(28.0)
    assert config["css_hover_grid_size"] == 5
    assert config["adaptive_compute_enabled"] is True
    assert config["adaptive_compute_target_ssim_srgb"] == pytest.approx(0.97)
    assert config["adaptive_compute_min_checkpoints"] == 3
    assert convert["output_path"].endswith(".html")


def test_constructor_passthroughs(run_cli: RunCli) -> None:
    init, convert = run_cli(
        "--format",
        "pptx",
        "--profile",
        "balanced",
        "--seed",
        "7",
        "--splats",
        "500",
        "--pptx-painter-order",
        "back-to-front",
        "--pptx-splat-style",
        "soft-edge",
    )
    assert init["quality_profile"] == "balanced"
    assert init["seed"] == 7
    assert init["max_splats"] == 500
    assert init["pptx_painter_order"] == "back-to-front"
    assert init["pptx_splat_style"] == "soft-edge"
    assert convert["output_format"] == "pptx"
    assert convert["seed"] == 7
