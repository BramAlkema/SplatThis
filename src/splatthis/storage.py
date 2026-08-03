"""Atomic persistence for source images, splat data, and preview frames."""

from __future__ import annotations

import json
import logging
import os
import stat
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
from PIL import Image

from .color import linear_to_srgb, srgb_to_linear
from .export_common import _layer_name
from .splat import RAW_SPLAT_SCHEMA_VERSION, GaussianSplat, RawSplat

logger = logging.getLogger(__name__)


@contextmanager
def atomic_output_path(output_path: str | Path) -> Iterator[Path]:
    """Yield a sibling temporary path and atomically replace the destination."""

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    mode = stat.S_IMODE(target.stat().st_mode) if target.exists() else 0o644
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=str(target.parent),
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        yield temporary
        with temporary.open("rb+") as stream:
            os.fsync(stream.fileno())
        temporary.chmod(mode)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_text(
    output_path: str | Path,
    content: str,
    *,
    encoding: str = "utf-8",
) -> None:
    """Write a text artifact without exposing a partial destination file."""

    with atomic_output_path(output_path) as temporary:
        temporary.write_text(content, encoding=encoding)


def load_png(
    path: str,
    target_size: Optional[Tuple[int, int]] = None,
    linearize_srgb: bool = True,
) -> npt.NDArray[Any]:
    """Load a PNG as normalized float32 RGB(A), optionally in linear RGB."""

    try:
        img = Image.open(path)
        logger.info("Loaded %s×%s image: %s", img.size[0], img.size[1], path)
        if img.mode == "P":
            img = img.convert("RGBA")
        elif img.mode in ["L", "LA"]:
            img = img.convert("RGB")
        elif img.mode != "RGBA":
            img = img.convert("RGB")

        if target_size is not None:
            img = img.resize(target_size, Image.Resampling.LANCZOS)
            logger.info("Resized to %s×%s", target_size[0], target_size[1])

        img_array = np.array(img, dtype=np.float32) / 255.0
        if linearize_srgb and img_array.shape[-1] >= 3:
            img_array[..., :3] = srgb_to_linear(img_array[..., :3])
            logger.info("Applied sRGB → linear RGB conversion")
        logger.info(
            "Final image shape: %s, range: [%.3f, %.3f]",
            img_array.shape,
            img_array.min(),
            img_array.max(),
        )
        return img_array
    except Exception as exc:
        logger.error("Failed to load PNG %s: %s", path, exc)
        raise


def save_splats_json(splats: List[GaussianSplat], output_path: str) -> None:
    """Save splats to the canonical raw JSON schema."""

    raw_splats = [splat.to_raw_splat().to_dict() for splat in splats]
    payload = {
        "schema": RAW_SPLAT_SCHEMA_VERSION,
        "num_splats": len(raw_splats),
        "splats": raw_splats,
    }
    layer_counts: Dict[int, int] = {}
    for item in raw_splats:
        layer = item.get("layer")
        if layer is not None:
            layer_counts[int(layer)] = layer_counts.get(int(layer), 0) + 1
    if layer_counts:
        payload["layers"] = [
            {"id": layer, "name": _layer_name(layer), "count": count}
            for layer, count in sorted(layer_counts.items())
        ]
    try:
        atomic_write_text(output_path, json.dumps(payload, indent=2, sort_keys=True))
        logger.info("Saved %s splats to JSON: %s", len(splats), output_path)
    except Exception as exc:
        logger.error("Failed to save JSON %s: %s", output_path, exc)
        raise


def load_splats_json(input_path: str) -> List[GaussianSplat]:
    """Load canonical or legacy splat JSON."""

    try:
        with open(input_path, "r", encoding="utf-8") as stream:
            payload = json.load(stream)
    except Exception as exc:
        logger.error("Failed to read JSON %s: %s", input_path, exc)
        raise

    splat_items = payload.get("splats", [])
    if payload.get("schema") == RAW_SPLAT_SCHEMA_VERSION:
        return [
            GaussianSplat.from_raw_splat(RawSplat.from_dict(item))
            for item in splat_items
        ]

    splats: List[GaussianSplat] = []
    for item in splat_items:
        if {"mu", "sigma", "color", "alpha"}.issubset(item):
            splats.append(
                GaussianSplat(
                    mu=np.array(item["mu"], dtype=np.float32),
                    sigma=np.array(item["sigma"], dtype=np.float32),
                    color=np.array(item["color"], dtype=np.float32),
                    alpha=float(item["alpha"]),
                    importance=float(item.get("importance", 0.0)),
                )
            )
        else:
            splats.append(GaussianSplat.from_raw_splat(RawSplat.from_dict(item)))
    return splats


def save_linear_rgb_png(
    rendered_linear_rgb: npt.NDArray[Any],
    output_path: str,
    scale: float = 1.0,
    embed_splats: Optional[List[GaussianSplat]] = None,
    embed_in_pixels: bool = False,
) -> str:
    """Write an HxWx3 linear-RGB framebuffer as an atomic display-sRGB PNG.

    ``embed_splats`` adds the population to a compressed PNG text chunk, so a
    shared render can say what it was fitted from. Decoders that do not know
    the keyword must skip the chunk, so the pixels are unchanged.

    ``embed_in_pixels`` additionally hides the population in the low bits of
    the image. The two carriers are complements, not alternatives, and both
    are written when both are asked for: stripping metadata (``oxipng``,
    ``exiftool``, any tool that re-saves) kills the chunk and leaves the
    pixels, while resizing kills the pixels and leaves the chunk. This costs
    picture quality -- little on photographic content, more on smooth
    low-entropy images -- so it stays opt-in.
    """

    rendered = np.asarray(rendered_linear_rgb, dtype=np.float32)
    if rendered.ndim != 3 or rendered.shape[2] != 3:
        raise ValueError("rendered_linear_rgb must have shape HxWx3")
    height, width = rendered.shape[:2]
    if width <= 0 or height <= 0:
        raise ValueError("rendered_linear_rgb must be non-empty")
    render_width = max(1, int(round(float(width) * float(scale))))
    render_height = max(1, int(round(float(height) * float(scale))))
    rendered_srgb = linear_to_srgb(np.clip(rendered, 0.0, 1.0))
    image = Image.fromarray((rendered_srgb * 255.0).astype(np.uint8), mode="RGB")
    if render_width != width or render_height != height:
        image = image.resize((render_width, render_height), Image.Resampling.LANCZOS)
    save_kwargs = {}
    if embed_splats:
        from .population_embed import embed_population_in_pixels, png_population_chunk

        save_kwargs["pnginfo"] = png_population_chunk(embed_splats)
        if embed_in_pixels:
            # After any resize above: resampling destroys the low bits.
            image = embed_population_in_pixels(image, embed_splats)
    with atomic_output_path(output_path) as temporary:
        image.save(temporary, format="PNG", **save_kwargs)
    return output_path


def render_splats_preview_png(
    splats: List[GaussianSplat],
    width: int,
    height: int,
    output_path: str,
    scale: float = 1.0,
    background_linear_rgb: Optional[npt.NDArray[Any]] = None,
    compositing_space: str = "linear",
) -> str:
    """Render in-memory splats to a diagnostic proxy PNG."""

    from .renderer import render_splats_numpy

    rendered = render_splats_numpy(
        splats,
        width,
        height,
        background_linear_rgb=background_linear_rgb,
        compositing_space=compositing_space,
    )
    return save_linear_rgb_png(rendered, output_path=output_path, scale=scale)
