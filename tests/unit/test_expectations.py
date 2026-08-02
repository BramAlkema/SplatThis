"""Published expectations must be readable by an installed consumer.

The point of this module is that a downstream project does not have to re-derive
compositor loss from the corpus. That only holds if the data ships inside the
package: a path resolved relative to the repository root works in a checkout and
raises FileNotFoundError from a wheel, which is the failure mode this repository
has already shipped once with its SVG templates.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from splatthis import compositor_fidelity
from splatthis.expectations import SUPPORTED_FORMATS


def test_expectations_data_lives_inside_the_package() -> None:
    """Resolved from the package, not the repository."""
    import splatthis

    package_dir = Path(splatthis.__file__).parent
    assert (package_dir / "data" / "compositor-fidelity.json").is_file()


@pytest.mark.parametrize("fmt", SUPPORTED_FORMATS)
def test_published_band_is_internally_consistent(fmt: str) -> None:
    band = compositor_fidelity(fmt)

    assert band.minimum <= band.p10 <= band.median <= band.maximum
    assert 0.0 < band.minimum <= band.maximum <= 1.0
    assert band.corpus_images >= 21
    assert band.summary


def test_the_emitter_separates_not_the_format_family() -> None:
    """What costs fidelity is how the falloff is expressed, not SVG-vs-CSS.

    An earlier version of this test asserted that SVG and CSS agree, because
    both were emitting the Gaussian as opacity baked into gradient colour
    stops. Once CSS switched to an alpha mask over a solid fill -- so the
    browser stops interpolating colour and opacity together -- it gained about
    0.19 median SSIM and left SVG behind on all 21 corpus images.

    SVG's radialGradient cannot express a mask, so the technique does not
    transfer. The ordering is therefore a property of what each format can say,
    not of vector-versus-DOM.
    """
    svg = compositor_fidelity("svg")
    css = compositor_fidelity("css")
    runtime = compositor_fidelity("pixel-runtime")

    assert css.median - svg.median > 0.15, "the mask technique should dominate"
    assert runtime.median > css.median > svg.median


def test_emitters_differ_by_far_more_than_their_own_spread() -> None:
    """The choice of emitter dominates, and that is the actionable result.

    The pixel runtime evaluates the splat formula directly, so it is a parity
    model of the reference renderer. SVG approximates a Gaussian with a radial
    gradient and pays about a quarter of SSIM for it. A consumer supplying its
    own splats should pick on this axis before tuning anything else.
    """
    svg = compositor_fidelity("svg")
    runtime = compositor_fidelity("pixel-runtime")

    assert runtime.median > 0.99, "the runtime should be effectively lossless"
    assert svg.median < 0.85
    assert runtime.median - svg.median > 0.2


def test_compositor_loss_is_not_content_predictable() -> None:
    """The published guidance, asserted rather than left in prose.

    End-to-end fidelity correlates strongly with content gradient, but that
    dependence is almost entirely the fitting stage. A consumer supplying its
    own splats inherits only the emitter term, which is broadly uniform -- so it
    should budget the band rather than a per-content-class estimate. A 7-image
    slice of the same corpus reports -0.854 and would suggest the opposite.
    """
    band = compositor_fidelity("svg")

    assert abs(band.content_gradient_correlation) < 0.7
    assert band.is_content_predictable() is False


@pytest.mark.parametrize("fmt", ["SVG", " svg ", "svg"])
def test_format_lookup_is_forgiving_about_case_and_space(fmt: str) -> None:
    assert compositor_fidelity(fmt).output_format == "svg"


def test_unmeasured_formats_raise_rather_than_guess() -> None:
    """An unmeasured expectation is exactly what this module exists to prevent."""
    for fmt in ("pptx", "css", "canvas", "pixel-runtime"):
        if fmt in SUPPORTED_FORMATS:
            continue
        with pytest.raises(ValueError, match="no published compositor fidelity"):
            compositor_fidelity(fmt)
