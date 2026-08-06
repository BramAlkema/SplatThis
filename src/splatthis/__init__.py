"""Fit images with 2D Gaussian splats and export portable artifacts."""

from ._version import __version__
from .browser_capture import capture_artifact_to_png
from .browser_export import generate_css_splat_html, generate_native_canvas_html
from .converter import PNG2SVGConverter, SplatConverter
from .email_export import generate_css_email_message, save_css_email
from .io import load_png, load_splats_json, save_splats_json, save_svg
from .population_embed import (
    decode_population,
    embed_population_in_pixels,
    encode_population,
    load_population,
    population_from_pixels,
)
from .powerpoint_osa import capture_pptx_with_powerpoint
from .pptx_export import save_pptx_with_splats
from .splat import GaussianSplat, RawSplat

__all__ = [
    "__version__",
    "GaussianSplat",
    "PNG2SVGConverter",
    "RawSplat",
    "SplatConverter",
    "capture_artifact_to_png",
    "capture_pptx_with_powerpoint",
    "decode_population",
    "embed_population_in_pixels",
    "encode_population",
    "generate_css_email_message",
    "generate_css_splat_html",
    "generate_native_canvas_html",
    "load_png",
    "load_population",
    "load_splats_json",
    "save_css_email",
    "save_pptx_with_splats",
    "save_splats_json",
    "save_svg",
    "population_from_pixels",
]
