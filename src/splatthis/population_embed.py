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
from typing import Any, Dict, List, Optional

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

#: PNG text-chunk keyword. PNG carries arbitrary text in tEXt/zTXt
#: chunks, which every decoder is required to skip over, so a preview or
#: capture can carry the fit that produced it without becoming a
#: different image.
PNG_POPULATION_KEY = "splatthis:population"


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

    lowered = str(path).lower()
    if lowered.endswith(".pptx"):
        return population_from_pptx(path)
    if lowered.endswith(".png"):
        return population_from_png(path)
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


def png_population_chunk(splats: List[GaussianSplat]) -> Any:
    """Return a ``PngInfo`` carrying ``splats``, for ``Image.save(pnginfo=...)``.

    The payload goes in a compressed text chunk. PNG readers that do not know
    the keyword are required to ignore it, so the decoded pixels are
    unchanged -- a preview stays a preview, and gains the ability to say what
    it was rendered from.
    """
    from PIL import PngImagePlugin

    info = PngImagePlugin.PngInfo()
    # zTXt: the envelope is already gzip+base64, but the chunk-level deflate
    # still pays for itself on the base64 alphabet.
    info.add_text(PNG_POPULATION_KEY, encode_population(splats), zip=True)
    return info


def population_from_png(path: str) -> List[GaussianSplat]:
    """Extract an embedded population from a PNG.

    Reads the text chunk when it is there and falls back to the low bits of
    the pixels when it is not -- which is exactly the case the pixel carrier
    exists for, since stripping the chunk is what optimisers do. Callers do
    not need to know which carrier a given file still has.
    """
    from PIL import Image

    with Image.open(path) as image:
        envelope = (image.text or {}).get(PNG_POPULATION_KEY)
        if envelope:
            return decode_population(envelope)
        try:
            return population_from_pixels(image)
        except ImportError:
            hidden = " (install the 'stego-lsb' extra to also check pixels)"
        except ValueError:
            hidden = ""
    raise ValueError(
        f"no embedded splatthis population in {path}; the PNG carries no "
        f"{PNG_POPULATION_KEY!r} text chunk and nothing hidden in its "
        f"pixels{hidden}"
    )


#: Bit depths tried when hiding or recovering, smallest first: a large image
#: pays one bit per channel, a smaller one pays two. Recovery walks the same
#: ladder, so the depth needs no header.
#:
#: Deliberately stops at 2. Depth 4 does fit more -- a 200x200 checkerboard
#: only takes the payload at 4 -- but it costs 0.19 SSIM and 0.047 delta-E,
#: which is visible damage. Escalating there automatically would turn "this
#: does not fit" into "the picture was quietly mangled", so the ladder ends
#: and the caller gets an error naming the chunk carrier instead. Depth 4
#: stays reachable by passing ``bits_per_channel=4`` explicitly.
STEG_BIT_DEPTHS = (1, 2)

#: What recovery tries, which is deliberately *wider* than what embedding
#: chooses. The asymmetry is the whole point: refusing to write depth 4
#: automatically protects the picture, but refusing to read it would strand
#: any file written with an explicit ``bits_per_channel=4`` -- a documented
#: write path with no reader. Being conservative about damage and liberal
#: about recovery are not the same policy, so they do not share a constant.
STEG_READ_DEPTHS = (1, 2, 4)


def _require_stego_lsb() -> Any:
    try:
        from stego_lsb import LSBSteg
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "in-pixel embedding needs the optional 'stego-lsb' package: "
            "pip install 'splatthis[steg]'"
        ) from exc
    return LSBSteg


def steg_capacity_bytes(image: Any, bits_per_channel: int = 1) -> int:
    """Payload bytes this PIL image can carry at the given depth.

    Net of the length prefix ``stego-lsb`` writes ahead of the payload, so
    this is what a caller can actually hand to
    :func:`embed_population_in_pixels`, not the raw bit budget.
    """
    steg = _require_stego_lsb()
    channels = len(image.getbands())
    total = int(steg.max_bits_to_hide(image, bits_per_channel, channels)) // 8
    prefix = int(steg.bytes_in_max_file_size(image, bits_per_channel, channels))
    return max(0, total - prefix)


def embed_population_in_pixels(
    image: Any,
    splats: List[GaussianSplat],
    bits_per_channel: Optional[int] = None,
) -> Any:
    """Hide a population in the low bits of an RGB PIL image.

    The chunk carriers are all *removable*, and measurably so: against a PNG
    holding both, ``oxipng --strip safe``, ``exiftool -all=`` and even a plain
    re-save through PIL each drop the text chunk while the pixels come
    through intact. Pixels are not metadata to be stripped -- they are the
    image -- which is the property the chunk cannot offer.

    The converse also holds, so the two carriers are complements rather than
    alternatives: a resize destroys the low bits and preserves the chunk.
    Neither survives a lossy re-encode. ``docs/embedded-populations.md``
    carries the full matrix.

    Requires an ``RGB`` image, and returns a new one -- ``stego-lsb`` writes
    through ``putdata()`` on whatever it is handed. Other modes are refused
    rather than converted: ``RGBA`` would hide bits in the alpha channel,
    where any later composite destroys them, and ``L``/``P`` crash inside the
    library. Bit packing is delegated to ``stego-lsb`` (MIT) rather than
    hand-rolled, since an off-by-one there corrupts payloads silently.
    """
    steg = _require_stego_lsb()
    if image.mode != "RGB":
        raise ValueError(
            f"in-pixel embedding needs an RGB image, got {image.mode!r}; "
            f"call image.convert('RGB') first (note that this discards any "
            f"alpha channel, which is why it is not done for you)"
        )
    payload = encode_population(splats).encode("utf-8")
    depths = STEG_BIT_DEPTHS if bits_per_channel is None else (int(bits_per_channel),)
    for depth in depths:
        if steg_capacity_bytes(image, depth) >= len(payload):
            return steg.hide_message_in_image(image.copy(), payload, depth)
    raise ValueError(
        f"a {image.size[0]}x{image.size[1]} image cannot carry {len(payload)} "
        f"bytes at {'/'.join(str(d) for d in depths)} bit(s) per channel "
        f"(capacity {steg_capacity_bytes(image, depths[-1])} bytes); use the "
        f"PNG text chunk carrier, which has no capacity limit, or pass "
        f"bits_per_channel=4 and accept the visible damage"
    )


def population_from_pixels(image: Any) -> List[GaussianSplat]:
    """Recover a population hidden by :func:`embed_population_in_pixels`.

    The depth is not recorded in the image; every depth in
    :data:`STEG_READ_DEPTHS` is tried and the one whose payload parses as a
    population envelope wins. Reading at the wrong depth yields a length
    prefix the library rejects as exceeding the image, or bytes that fail
    the envelope's own schema check -- so a mangled population cannot be
    returned silently.

    Accepts any mode convertible to RGB: a carrier that some tool has since
    promoted to RGBA still holds its bits in the colour channels.
    """
    steg = _require_stego_lsb()
    if image.mode != "RGB":
        image = image.convert("RGB")
    for depth in STEG_READ_DEPTHS:
        try:
            payload = steg.recover_message_from_image(image, depth)
            return decode_population(payload.decode("utf-8"))
        except (ValueError, UnicodeDecodeError, OSError, KeyError):
            # Exactly the ways a wrong depth fails: an over-long length
            # prefix, non-UTF-8 bytes, a corrupt gzip member, or an envelope
            # missing a key. Anything else is a real bug and should surface.
            continue
    raise ValueError("no splatthis population hidden in these pixels")
