"""Published fidelity expectations, measured on the governing corpus.

The public entry point is :func:`expected_fidelity`. It is deliberately not
called ``fidelity``: ``splatthis.fidelity`` is the ADR-003 accept-or-revert
subpackage, and a function of that name is shadowed by the module as soon as
anything imports it -- which surfaces only when the whole suite runs.

Two questions are kept apart here because they have very different answers, and
conflating them is how this module's first version came to publish figures that
looked like quality claims and were not.

**Deployed fidelity** compares the emitted artifact to the *original image*.
This is what a user of this tool gets, and it is dominated by fitting error.

**Compositor fidelity** compares the emitted artifact to the internal reference
render of the *same splats*. It isolates the emitter, and is the relevant number
only for a caller that supplies its own splats -- projecting 3D Gaussians
through EWA, say -- and therefore has no fitting error.

The gap between them is the point::

    >>> band = expected_fidelity("svg")
    >>> band.deployed_lpips, band.compositor_lpips
    (0.2433, 0.031)

Against the original the declarative emitters are indistinguishable: median
LPIPS 0.2433 / 0.2439 / 0.2429 for svg / svg-high / css, a spread of 0.001,
with an SSIM spread of 0.011 that sits below the 0.029 seed noise floor.
Compositor-only figures make the same emitters look far more different than
they are.

**Read LPIPS, not SSIM.** On this content SSIM has almost no dynamic range:
against the reference render of one chameleon population, a 2-pixel Gaussian
blur scores 0.9290 and the source photograph -- a different image entirely --
scores 0.9053, against 0.9336 for the actual SVG. LPIPS separates the same
cases as 0.1268 / 0.1456 / 0.0411.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

_DATA = Path(__file__).resolve().parent / "data" / "compositor-fidelity.json"

#: Formats with a published measurement. Others raise rather than guess.
SUPPORTED_FORMATS: Tuple[str, ...] = ("svg", "svg-high", "css", "pixel-runtime")

#: Below this, two emitters are not distinguishable on a single seeded run.
#: Six images at three seeds; see ``paper/report.md`` section 5.2.
SEED_NOISE_FLOOR_SSIM = 0.029


@dataclass(frozen=True)
class Fidelity:
    """Published expectation for one output format.

    ``deployed_*`` is against the original image and is what a user sees.
    ``compositor_*`` is against the reference render of the same splats and
    isolates the emitter. A consumer supplying its own splats wants the second;
    everyone else wants the first.
    """

    output_format: str
    deployed_lpips: Optional[float]
    deployed_lpips_p90: Optional[float]
    deployed_ssim: Optional[float]
    compositor_lpips: float
    compositor_ssim: float
    corpus_images: int
    summary: str

    def is_distinguishable_from(self, other: "Fidelity") -> bool:
        """Whether two formats differ by more than a seeded run's own noise.

        False for every pair of declarative emitters measured so far: their
        deployed SSIM spread is 0.011 against a 0.029 floor, so choosing
        between them on fidelity is choosing on noise.
        """
        if self.deployed_ssim is None or other.deployed_ssim is None:
            return True
        return abs(self.deployed_ssim - other.deployed_ssim) > SEED_NOISE_FLOOR_SSIM


@lru_cache(maxsize=None)
def _load() -> Dict[str, Any]:
    if not _DATA.is_file():  # pragma: no cover - packaging guard
        raise FileNotFoundError(f"missing published expectations: {_DATA}")
    data: Dict[str, Any] = json.loads(_DATA.read_text(encoding="utf-8"))
    return data


def expected_fidelity(output_format: str = "svg") -> Fidelity:
    """Return the published fidelity expectation for ``output_format``.

    Raises:
        ValueError: for a format with no published measurement. Callers get an
            error rather than a plausible-looking guess, because an unmeasured
            expectation is the thing this module exists to prevent.
    """
    normalized = output_format.strip().lower()
    if normalized not in SUPPORTED_FORMATS:
        raise ValueError(
            f"no published fidelity for {output_format!r}; "
            f"measured formats are {', '.join(SUPPORTED_FORMATS)}"
        )
    entry = _load()["formats"].get(normalized)
    if entry is None:  # pragma: no cover - guarded by SUPPORTED_FORMATS
        raise ValueError(f"published data does not cover {normalized!r}")
    expectation = entry["expectation"]
    deployed = expectation.get("deployed") or {}
    compositor = expectation["compositor"]
    return Fidelity(
        output_format=normalized,
        deployed_lpips=deployed.get("lpips_median"),
        deployed_lpips_p90=deployed.get("lpips_p90"),
        deployed_ssim=deployed.get("ssim_srgb_median"),
        compositor_lpips=float(compositor["lpips_median"]),
        compositor_ssim=float(compositor["ssim_srgb_median"]),
        corpus_images=int(entry["corpus_images"]),
        summary=str(expectation["summary"]),
    )


def compositor_fidelity(output_format: str = "svg") -> Fidelity:
    """Deprecated alias for :func:`expected_fidelity`.

    The original name implied the compositor figure was the headline. It is
    not: it describes the emitter in isolation, which matters only to a caller
    supplying its own splats.
    """
    return expected_fidelity(output_format)


#: Deprecated alias for :class:`Fidelity`, kept so 0.2.5-era imports survive
#: the rename. The dataclass fields changed with the deployed/compositor
#: split, so downstream attribute access may still need updating.
CompositorFidelity = Fidelity
