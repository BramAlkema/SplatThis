"""One real conversion, end to end, with a numerical floor.

Every other test in this suite isolates a unit. That is why 87% line coverage
did not notice that the packaged SVG templates were absent from every fresh
checkout: no test ever ran the pipeline the way a user runs it, from source
image to written artifact.

This test does. It is deliberately the slowest test in the suite.

**Evidence level: PROXY.** The score asserted here comes from the internal
NumPy render, not from a governing Chromium capture, so it is a regression
guard on the conversion pipeline and *not* a deployed-fidelity claim. Per
``docs/ARCHITECTURE.md`` a proxy metric may never approve a browser artifact;
it may only detect that the pipeline moved. Deployed fidelity is measured by
the corpus tooling against real browsers, and its floors live in
``data/artifact-gates.json``. Do not merge these two things.

The backend is pinned to Torch on purpose. MLX orders float32 reductions on
the Metal device nondeterministically, and CI installs MLX only on macOS
ARM64 — an unpinned backend would test three different things across the
matrix and be flaky on exactly one of them.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from splatthis.cli import main

REPO = Path(__file__).resolve().parents[2]
SOURCE = REPO / "docs" / "demo" / "source.png"

# Provenance for these floors, measured at the exact invocation below on
# Torch CPU, GitHub Actions runners, 2026-08-01:
#
#     macOS 3.13    ssim_srgb 0.5167345048   psnr_srgb 18.387884
#     ubuntu 3.13   ssim_srgb 0.5165994102   psnr_srgb 18.388382
#     windows 3.13  ssim_srgb 0.5165853723   psnr_srgb 18.388220
#
# Cross-platform spread is 1.5e-4 (different BLAS kernels); on a single
# platform three repeats were bit-identical. The margin below is 44x that
# spread, and the regression it is meant to catch is far larger still:
# reverting the splat-orientation convention drops ssim_srgb to 0.460974,
# which is 375x the spread.
#
# The floor also sits deliberately above the pipeline's own acceptance
# minimum of 0.50 — a floor equal to that one would assert nothing the
# pipeline does not already enforce for itself.
MIN_SSIM_SRGB = 0.510
MIN_PSNR_SRGB = 18.0

# The population is budget- and prune-driven, so it is a range, not a constant.
MIN_SPLATS = 100
MAX_SPLATS = 200

SVG_NS = "{http://www.w3.org/2000/svg}"


@pytest.fixture(scope="module")
def conversion(tmp_path_factory):
    """Run the pipeline once; both assertions below read the same artifact.

    Module-scoped deliberately. This is the only test that performs a real
    conversion, and under coverage instrumentation it dominates the suite's
    runtime — running it per-test would double that for no added signal.
    """
    assert SOURCE.is_file(), "demo source image must be tracked in the repo"

    workdir = tmp_path_factory.mktemp("e2e")
    out_svg = workdir / "out.svg"
    artifacts = workdir / "artifacts"

    code = main(
        [
            str(SOURCE),
            "-o",
            str(out_svg),
            "--seed",
            "42",
            "--splats",
            "200",
            "--max-edge",
            "128",
            "--stages",
            "3,2",
            # Pinned: see module docstring.
            "--optimizer-backend",
            "torch",
            # This test asserts seeded pipeline quality, not the ambient host
            # memory policy. The safety valve is covered independently and can
            # halve the population on a busy CI runner.
            "--no-memory-guard",
            "--artifacts-dir",
            str(artifacts),
        ]
    )
    assert code in (0, None)
    assert out_svg.is_file()

    manifest = json.loads((artifacts / "run_manifest.json").read_text())
    return out_svg, manifest


@pytest.mark.slow
def test_real_conversion_meets_quality_floor(conversion, capsys):
    """The pipeline must not silently lose reconstruction quality."""
    _, manifest = conversion
    measured = manifest["export_quality"]["metrics"]

    # Always surface the numbers, so a CI log records the per-platform value
    # even on a pass. Tightening the floor later needs that provenance.
    with capsys.disabled():
        print(
            f"\n  [e2e] ssim_srgb={measured['ssim_srgb']:.10f} "
            f"psnr_srgb={measured['psnr_srgb']:.6f}"
        )

    assert measured["ssim_srgb"] >= MIN_SSIM_SRGB, (
        f"SSIM_sRGB {measured['ssim_srgb']:.6f} fell below {MIN_SSIM_SRGB}. "
        "This is a pipeline regression, not noise."
    )
    assert measured["psnr_srgb"] >= MIN_PSNR_SRGB


@pytest.mark.slow
def test_real_conversion_emits_a_self_contained_svg(conversion):
    """The artifact must be renderable on its own, with no external fetches."""
    out_svg, _ = conversion

    root = ET.parse(out_svg).getroot()
    assert root.tag == f"{SVG_NS}svg"

    ellipses = root.findall(f".//{SVG_NS}ellipse")
    gradients = root.findall(f".//{SVG_NS}radialGradient")

    assert MIN_SPLATS <= len(ellipses) <= MAX_SPLATS
    # Per-splat baked gradients: a shared currentColor gradient does not render
    # in rsvg, so one gradient per ellipse is the contract, not an accident.
    assert len(gradients) == len(ellipses)

    markup = out_svg.read_text(encoding="utf-8")
    assert "currentColor" not in markup
    for external in ("http://", "https://", 'xlink:href="http'):
        assert external not in markup.replace(SVG_NS.strip("{}"), "")
