"""Published expectations must be readable, honest, and hard to misread.

The module exists so a downstream project does not re-derive these numbers. It
has been restructured once already, because its first version published
compositor-only figures that read like quality claims. The tests below pin the
distinction that caused it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from splatthis import expected_fidelity
from splatthis.expectations import SEED_NOISE_FLOOR_SSIM, SUPPORTED_FORMATS


def test_expectations_data_lives_inside_the_package() -> None:
    """Resolved from the package, not the repository."""
    import splatthis

    assert (
        Path(splatthis.__file__).parent / "data" / "compositor-fidelity.json"
    ).is_file()


@pytest.mark.parametrize("fmt", SUPPORTED_FORMATS)
def test_every_format_publishes_a_compositor_figure(fmt: str) -> None:
    band = expected_fidelity(fmt)
    assert 0.0 <= band.compositor_lpips < 1.0
    assert 0.0 < band.compositor_ssim <= 1.0
    assert band.corpus_images >= 21
    assert band.summary


def test_deployed_and_compositor_are_not_the_same_number() -> None:
    """Conflating them is the mistake this module was restructured to prevent.

    Compositor loss is roughly an order of magnitude smaller than deployed
    loss, because deployed fidelity carries the fit and compositor fidelity
    does not. Quoting the compositor figure as quality overstates the artifact
    by that whole margin.
    """
    band = expected_fidelity("svg")
    assert band.deployed_lpips is not None
    assert band.deployed_lpips > band.compositor_lpips * 5


def test_declarative_emitters_are_indistinguishable_end_to_end() -> None:
    """The finding that matters, asserted so it cannot quietly regress.

    Against the original image, svg, svg-high and css differ by 0.001 median
    LPIPS and 0.011 SSIM -- the latter below the 0.029 seed noise floor. Their
    compositor figures differ far more, which is exactly why compositor numbers
    must not be presented as quality.
    """
    svg, high, css = (
        expected_fidelity("svg"),
        expected_fidelity("svg-high"),
        expected_fidelity("css"),
    )

    for other in (high, css):
        assert not svg.is_distinguishable_from(other)

    deployed = [b.deployed_lpips for b in (svg, high, css)]
    assert max(deployed) - min(deployed) < 0.01  # type: ignore[type-var]


def test_the_runtime_is_a_parity_model_not_a_compositor() -> None:
    """It evaluates the splat formula rather than approximating it.

    The deployed figure exists since August 2026 -- re-emitted from current
    code, captured in governing Chrome, and matching the historical ledger --
    and, like every deployed figure, it is dominated by fitting error rather
    than the effectively lossless emitter.
    """
    runtime = expected_fidelity("pixel-runtime")
    assert runtime.compositor_lpips < 0.001
    assert runtime.deployed_lpips is not None
    assert runtime.deployed_lpips > runtime.compositor_lpips * 100


def test_seed_noise_floor_is_the_published_one() -> None:
    assert SEED_NOISE_FLOOR_SSIM == pytest.approx(0.029)


@pytest.mark.parametrize("fmt", ["SVG", " svg ", "svg"])
def test_format_lookup_is_forgiving_about_case_and_space(fmt: str) -> None:
    assert expected_fidelity(fmt).output_format == "svg"


def test_unmeasured_formats_raise_rather_than_guess() -> None:
    for fmt in ("pptx", "canvas"):
        with pytest.raises(ValueError, match="no published fidelity"):
            expected_fidelity(fmt)


def test_docstring_examples_execute_and_pin_the_published_numbers() -> None:
    """The module docstring quotes exact medians; run it so they stay pinned.

    Nothing else in the suite asserts an exact published value -- by design,
    since the registry changes when re-measured -- so the doctest is the one
    place a silent registry edit that contradicts the prose gets caught.
    """
    import doctest

    import splatthis.expectations as module

    result = doctest.testmod(module)
    assert result.attempted > 0, "the docstring examples were not collected"
    assert result.failed == 0
