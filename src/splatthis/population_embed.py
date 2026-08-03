"""Embed the splat population inside the artifact that was fitted from it.

A deployed SVG or deck is the *result* of a fit; the population that produced
it normally lives in a sidecar ``final.raw.json`` that never travels with the
file. Embedding it makes the artifact self-contained in three useful ways:

- **Re-targetable.** Anyone holding the SVG can emit the PPTX, CSS or pixel
  runtime from it without the source image or the artifacts directory.
- **Warm-startable.** A later run can resume from the fit rather than
  re-seeding, which is where nearly all of a repeat run's wall clock goes.
- **Comparable.** The population is the fit's own answer at a stated splat
  budget, so another algorithm can load it, beat it, and be checked against
  the same artifact -- which is the point of publishing it at all.

That last use is why the payload is deliberately self-describing rather than
a pickle of our classes. The envelope names its fields, dtype and layout, so
decoding needs base64 and gzip and nothing from this project::

    import base64, gzip, json, numpy as np
    env = json.loads(<the metadata text>)
    buf = gzip.decompress(base64.b64decode(env["data"]))
    arr = np.frombuffer(buf, dtype=env["dtype"]).reshape(-1, len(env["fields"]))
    # columns are env["fields"], in that order

Storing floats as float32 rather than JSON numbers is what makes this cheap:
a 1,595-splat population is 456 KB as canonical JSON and 59 KB here, about
3.7% of a typical SVG.
"""

from __future__ import annotations

import base64
import gzip
import json
from typing import Any, Dict, List

import numpy as np

from .splat import GaussianSplat, RawSplat

#: Column order of the encoded array. Fixed: readers index by position, so
#: appending is safe and reordering is a schema break.
POPULATION_FIELDS = (
    "x",
    "y",
    "sx",
    "sy",
    "theta",
    "r",
    "g",
    "b",
    "a",
    "importance",
)

POPULATION_SCHEMA = "splatthis.population/1"

#: Detects SVG input by its namespace rather than its root tag, so this
#: module stays free of vector markup (test_module_boundaries).
SVG_NAMESPACE = "http://www.w3.org/2000/svg"

#: Where the population lives inside an OOXML package. Deliberately a
#: plain, unreferenced part outside ppt/: PowerPoint ignores parts it has
#: no relationship to, so the deck opens normally, and a Save As that
#: rewrites the package simply drops it rather than corrupting anything.
PPTX_POPULATION_PART = "splatthis/population.json"


def encode_population(splats: List[GaussianSplat]) -> str:
    """Return a self-describing JSON envelope for ``splats``.

    The envelope is plain text so it can sit in SVG ``<metadata>`` or an
    OOXML package part without escaping; the bulk is a gzipped float32 array.
    """
    rows = []
    for splat in splats:
        raw = splat.to_raw_splat().to_dict()
        rows.append([float(raw[field]) for field in POPULATION_FIELDS])
    array = np.asarray(rows, dtype=np.float32)
    payload = base64.b64encode(gzip.compress(array.tobytes(), 9)).decode("ascii")
    envelope: Dict[str, Any] = {
        "schema": POPULATION_SCHEMA,
        "count": len(splats),
        "fields": list(POPULATION_FIELDS),
        "dtype": "float32",
        "layout": "row-major",
        "encoding": "gzip+base64",
        "note": (
            "Gaussian splat population this artifact was fitted from. "
            "Decode: base64 -> gzip -> float32 array of shape (count, "
            "len(fields)). Angles in radians, positions and sigmas in pixels "
            "of the artifact's own coordinate system, colour linear-sRGB in "
            "[0,1]. Published so the fit can be reproduced, re-targeted, or "
            "beaten at the same splat budget."
        ),
        "data": payload,
    }
    return json.dumps(envelope, separators=(",", ":"))


def decode_population(envelope_text: str) -> List[GaussianSplat]:
    """Rebuild splats from :func:`encode_population` output.

    Raises:
        ValueError: on an unknown schema or a payload whose length disagrees
            with its declared shape -- a silently truncated population would
            warm-start from a fit that never existed.
    """
    envelope = json.loads(envelope_text)
    schema = str(envelope.get("schema", ""))
    if not schema.startswith("splatthis.population/"):
        raise ValueError(f"not a splatthis population envelope: {schema!r}")
    fields = list(envelope["fields"])
    buffer = gzip.decompress(base64.b64decode(envelope["data"]))
    array = np.frombuffer(buffer, dtype=np.dtype(envelope["dtype"]))
    if array.size != int(envelope["count"]) * len(fields):
        raise ValueError(
            f"population payload is {array.size} values; envelope declares "
            f"{envelope['count']} x {len(fields)}"
        )
    array = array.reshape(int(envelope["count"]), len(fields))
    return [
        GaussianSplat.from_raw_splat(
            RawSplat.from_dict({name: float(value) for name, value in zip(fields, row)})
        )
        for row in array
    ]


def svg_metadata_element(splats: List[GaussianSplat]) -> str:
    """Return an SVG ``<metadata>`` block carrying the population.

    ``<metadata>`` is the standards-defined home for this; a comment would be
    stripped by ``svgo`` (which ``--svg-optimize`` runs) and cannot contain
    ``--``. Renderers ignore the element, so the drawn output is unchanged.
    The markup itself lives in the packaged template set, like every other
    fragment this project emits.
    """
    from .template_assets import render_template

    return render_template(
        "svg/population_metadata.svg", envelope=encode_population(splats)
    )


def population_from_svg(svg_text: str) -> List[GaussianSplat]:
    """Extract an embedded population from SVG markup."""
    import re

    match = re.search(
        r"<splatthis:population[^>]*>(.*?)</splatthis:population>", svg_text, re.S
    )
    if not match:
        raise ValueError("no embedded splatthis population in this SVG")
    return decode_population(match.group(1))


def load_population(path: str) -> List[GaussianSplat]:
    """Load a population from an embedded-population SVG or a raw JSON file.

    Accepts either artifact this project can produce: an SVG written with
    ``--embed-population``, or the canonical ``*.raw.json`` that
    ``--save-json`` and the artifacts directory already emit. Callers get
    splats without needing to know which.
    """
    from pathlib import Path as _Path

    from .storage import load_splats_json

    if str(path).lower().endswith(".pptx"):
        return population_from_pptx(path)
    text = _Path(path).read_text(encoding="utf-8", errors="ignore")
    if "<splatthis:population" in text:
        return population_from_svg(text)
    if SVG_NAMESPACE in text[:2048]:
        # Falling through to the JSON loader here would surface a bare
        # "Expecting value: line 1 column 1", which says nothing about the
        # actual problem: this SVG was written without --embed-population.
        raise ValueError(
            f"no embedded splatthis population in {path}; re-export it with "
            f"--embed-population, or pass a population JSON instead"
        )
    return load_splats_json(path)


def pptx_population_part(splats: List[GaussianSplat]) -> tuple:
    """Return the ``(name, text)`` package part carrying ``splats``."""
    return (PPTX_POPULATION_PART, encode_population(splats))


def population_from_pptx(path: str) -> List[GaussianSplat]:
    """Extract an embedded population from a .pptx package."""
    import zipfile

    with zipfile.ZipFile(path) as package:
        if PPTX_POPULATION_PART not in package.namelist():
            raise ValueError(
                f"no embedded splatthis population in {path}; re-export it "
                f"with --embed-population"
            )
        return decode_population(package.read(PPTX_POPULATION_PART).decode("utf-8"))
