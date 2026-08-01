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


def test_published_band_is_internally_consistent() -> None:
    band = compositor_fidelity("svg")

    assert band.minimum <= band.p10 <= band.median <= band.maximum
    assert 0.0 < band.minimum < 1.0
    assert band.maximum < 1.0, "a lossless emitter would make this module pointless"
    assert band.corpus_images >= 21


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
