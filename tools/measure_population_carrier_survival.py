#!/usr/bin/env python3
"""Measure which population carriers survive which image-handling tools.

An embedded population is only useful if it is still there when the file
arrives. The PNG text chunk is metadata, and metadata is what sanitisers,
optimisers and re-savers remove; the in-pixel carrier is the image itself.
This runs the actual tools against one PNG carrying both and reports what
still decodes, rather than reasoning about what should.

The result that motivated the second carrier: the chunk is lost not only to
deliberate stripping but to *any* tool that opens the file and writes it
back. The result that keeps the first: a resize destroys low bits and
leaves the chunk alone. They fail on opposite inputs.

Tools that are not installed are reported as skipped, never as passes.

Usage::

    PYTHONPATH=src python tools/measure_population_carrier_survival.py
    PYTHONPATH=src python tools/measure_population_carrier_survival.py \
        --image chameleon --markdown
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, List, Optional, Tuple

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from PIL import Image  # noqa: E402

from splatthis.population_embed import (  # noqa: E402
    PNG_POPULATION_KEY,
    embed_population_in_pixels,
    png_population_chunk,
    population_from_pixels,
)
from splatthis.population_embed import decode_population  # noqa: E402
from splatthis.storage import load_splats_json  # noqa: E402

SOURCES = REPO / "result" / "corpus" / "images"
RUNS = REPO / "result" / "corpus" / "runs"
WORK = REPO / "tmp" / "carrier-survival"


def _chunk_survives(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            envelope = (image.text or {}).get(PNG_POPULATION_KEY)
        return bool(envelope) and bool(decode_population(envelope))
    except Exception:
        return False


def _pixels_survive(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            return bool(population_from_pixels(image))
    except Exception:
        return False


def _copy_then(cmd: List[str]) -> Callable[[Path, Path], None]:
    """Tools that rewrite in place need the file copied to the target first."""

    def run(src: Path, dst: Path) -> None:
        shutil.copy(src, dst)
        subprocess.run([*cmd, str(dst)], check=False, capture_output=True, timeout=180)

    return run


def _magick(extra: Optional[List[str]] = None) -> Callable[[Path, Path], None]:
    def run(src: Path, dst: Path) -> None:
        subprocess.run(
            ["magick", str(src), *(extra or []), str(dst)],
            check=False,
            capture_output=True,
            timeout=180,
        )

    return run


def _pil_resave(src: Path, dst: Path) -> None:
    with Image.open(src) as image:
        image.save(dst)


def _jpeg_roundtrip(src: Path, dst: Path) -> None:
    interim = dst.with_suffix(".jpg")
    with Image.open(src) as image:
        image.convert("RGB").save(interim, quality=95)
    with Image.open(interim) as image:
        image.save(dst)


#: (label, required executable or None, runner)
ATTACKS: Tuple[Tuple[str, Optional[str], Callable[[Path, Path], None]], ...] = (
    (
        "oxipng -o4 --strip safe",
        "oxipng",
        _copy_then(["oxipng", "-o4", "--strip", "safe", "-q"]),
    ),
    (
        "oxipng -o4 --strip all",
        "oxipng",
        _copy_then(["oxipng", "-o4", "--strip", "all", "-q"]),
    ),
    ("optipng -o2", "optipng", _copy_then(["optipng", "-o2", "-quiet"])),
    (
        "exiftool -all=",
        "exiftool",
        _copy_then(["exiftool", "-all=", "-overwrite_original"]),
    ),
    ("ImageMagick re-encode", "magick", _magick()),
    ("PIL re-save (no pnginfo)", None, _pil_resave),
    ("ImageMagick -resize 50%", "magick", _magick(["-resize", "50%"])),
    ("JPEG q95 round-trip", None, _jpeg_roundtrip),
)


def _run_attacks(base: Path) -> List[Tuple[str, Optional[bool], Optional[bool]]]:
    """Apply every available tool to ``base`` and report what still decodes."""
    rows: List[Tuple[str, Optional[bool], Optional[bool]]] = [
        ("nothing", _chunk_survives(base), _pixels_survive(base))
    ]
    for index, (label, executable, run) in enumerate(ATTACKS):
        if executable and not shutil.which(executable):
            rows.append((label, None, None))
            continue
        target = base.with_name(f"attacked-{index}.png")
        target.unlink(missing_ok=True)
        run(base, target)
        if not target.is_file():
            rows.append((label, None, None))
            continue
        rows.append((label, _chunk_survives(target), _pixels_survive(target)))
    return rows


def _cell(value: Optional[bool], markdown: bool) -> str:
    if value is None:
        return "not installed"
    if markdown:
        return "survives" if value else "**lost**"
    return "survives" if value else "LOST"


def _report(
    rows: List[Tuple[str, Optional[bool], Optional[bool]]], markdown: bool
) -> None:
    if markdown:
        print("| what it went through | `zTXt` chunk | in-pixels |")
        print("|---|---|---|")
        for label, chunk, pixels in rows:
            print(f"| {label} | {_cell(chunk, True)} | {_cell(pixels, True)} |")
    else:
        print(f"{'what it went through':28s}{'zTXt chunk':>14s}{'in-pixels':>14s}")
        for label, chunk, pixels in rows:
            print(f"{label:28s}{_cell(chunk, False):>14s}{_cell(pixels, False):>14s}")

    tested = [r for r in rows[1:] if r[1] is not None]
    chunk_only = sum(1 for _, c, p in tested if c and not p)
    pixel_only = sum(1 for _, c, p in tested if p and not c)
    print(
        f"\n{len(tested)} tools run: {pixel_only} where only the pixels "
        f"survived, {chunk_only} where only the chunk did. Both carriers earn "
        f"their place while each of those stays non-zero."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="astronaut")
    parser.add_argument(
        "--markdown", action="store_true", help="emit the docs table instead"
    )
    args = parser.parse_args()

    source = SOURCES / f"{args.image}.png"
    population = RUNS / f"{args.image}_svg_s0_art" / "final.raw.json"
    if not source.is_file() or not population.is_file():
        print(f"error: no corpus image + run for {args.image!r}", file=sys.stderr)
        return 2

    WORK.mkdir(parents=True, exist_ok=True)
    splats = load_splats_json(str(population))
    with Image.open(source) as opened:
        image = opened.convert("RGB")

    base = WORK / "both.png"
    embed_population_in_pixels(image, splats).save(
        base, pnginfo=png_population_chunk(splats)
    )
    print(
        f"{args.image}: {image.size[0]}x{image.size[1]}, {len(splats)} splats, "
        f"carrier {base.stat().st_size / 1024:.0f} KB\n"
    )

    _report(_run_attacks(base), args.markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
