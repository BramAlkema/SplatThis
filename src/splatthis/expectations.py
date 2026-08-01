"""Published fidelity expectations, for consumers that supply their own splats.

A caller that projects 3D Gaussians through EWA rather than fitting a bitmap
gets an exact splat population and inherits only the *emitter's* loss. It must
not budget against this project's headline source-fidelity numbers, which are
dominated by fitting error it does not have.

This module publishes the emitter term on its own, measured by rendering one
identical splat population two ways -- through the internal reference renderer,
and through the deployed compositor in its governing browser -- so the only
difference between them is the emitter.

    >>> from splatthis import compositor_fidelity
    >>> band = compositor_fidelity("svg")
    >>> band.median, band.p10
    (0.754, 0.6524)
    >>> compositor_fidelity("pixel-runtime").median
    0.9993

Deriving this by reading the corpus is possible but easy to get wrong. Two
mistakes are worth naming, because both were made while producing it: rendering
the reference in the wrong compositing space (SVG-target models composite in
sRGB, not linear), and quoting a subset -- a 7-image slice of the same corpus
reports a content correlation of -0.854 against the full population's -0.470.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Tuple

_DATA = Path(__file__).resolve().parent / "data" / "compositor-fidelity.json"

#: Formats this measurement covers. Others raise rather than guess.
SUPPORTED_FORMATS: Tuple[str, ...] = ("svg", "pixel-runtime")


@dataclass(frozen=True)
class CompositorFidelity:
    """Expected agreement between a deployed compositor and the reference render.

    All figures are SSIM_sRGB against the internal renderer drawing the *same*
    splats, so 1.0 would mean the emitter is lossless.
    """

    output_format: str
    minimum: float
    p10: float
    median: float
    maximum: float
    content_gradient_correlation: float
    corpus_images: int
    content_dependence: str
    summary: str

    def is_content_predictable(self) -> bool:
        """Whether content frequency meaningfully predicts this loss.

        False for every format measured so far: compositor loss is broadly
        uniform, and the strong content dependence in end-to-end fidelity
        belongs to the fitting stage. A consumer that supplies its own splats
        should therefore budget the band, not a per-content-class estimate.
        """
        return abs(self.content_gradient_correlation) >= 0.7


@lru_cache(maxsize=None)
def _load() -> Dict[str, Any]:
    if not _DATA.is_file():  # pragma: no cover - packaging guard
        raise FileNotFoundError(f"missing published expectations: {_DATA}")
    data: Dict[str, Any] = json.loads(_DATA.read_text(encoding="utf-8"))
    return data


def compositor_fidelity(output_format: str = "svg") -> CompositorFidelity:
    """Return the published compositor-only fidelity band for ``output_format``.

    Raises:
        ValueError: for a format with no published measurement. Callers get an
            error rather than a plausible-looking guess, because an unmeasured
            expectation is the thing this module exists to prevent.
    """
    normalized = output_format.strip().lower()
    if normalized not in SUPPORTED_FORMATS:
        raise ValueError(
            f"no published compositor fidelity for {output_format!r}; "
            f"measured formats are {', '.join(SUPPORTED_FORMATS)}"
        )
    entry = _load()["formats"].get(normalized)
    if entry is None:  # pragma: no cover - guarded by SUPPORTED_FORMATS
        raise ValueError(f"published data does not cover {normalized!r}")
    expectation = entry["expectation"]
    return CompositorFidelity(
        output_format=normalized,
        minimum=float(expectation["ssim_srgb_min"]),
        p10=float(expectation["ssim_srgb_p10"]),
        median=float(expectation["ssim_srgb_median"]),
        maximum=float(expectation["ssim_srgb_max"]),
        content_gradient_correlation=float(expectation["content_gradient_correlation"]),
        corpus_images=int(entry["corpus_images"]),
        content_dependence=str(expectation["content_dependence"]),
        summary=str(expectation["summary"]),
    )
