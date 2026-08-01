"""Static SVG splat emitters and SVG artifact optimization."""

from __future__ import annotations

import logging
import math
import os
import shutil
import subprocess
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt

from .color import linear_to_srgb
from .export_common import (
    DEFAULT_EXPORT_ORDER,
    ELLIPSE_OVERLAP_BOOST,
    MIN_ELLIPSE_RADIUS_PX,
    SVG_BACKGROUND_ALPHA_CAP,
    SVG_BLUR_CORE_K_SIGMA,
    SVG_BLUR_RECIPE,
    SVG_BLUR_SIGMA_BUCKETS,
    SVG_BROWSER_COMPAT_RECIPE,
    SVG_FEATHER_EXTENT,
    SVG_GRADIENT_QUALITY_HIGH,
    SVG_GRADIENT_QUALITY_STANDARD,
    SVG_GRADIENT_STOPS,
    SVG_GRADIENT_STOPS_HIGH,
    SVG_PAINTER_ORDER_BACK_TO_FRONT,
    SVG_PALETTE_QUANTIZED_DEFAULT_SIZE,
    SVG_PALETTE_QUANTIZED_RECIPE,
    SVG_PRECOMP_ALPHA_THRESHOLD,
    SVG_PRECOMP_MAX_SRGB,
    SVG_SCRIPTED_MATRIX_RECIPE,
    RegionMasks,
    _adaptive_gradient_stops,
    _normalize_svg_export_recipe,
    _normalize_svg_gradient_quality,
    _normalize_svg_painter_order,
    _sort_splats_for_export,
    _svg_gradient_settings,
    _svg_painter_indices,
)
from .splat import GaussianSplat
from .storage import atomic_write_text
from .template_assets import render_template

logger = logging.getLogger(__name__)

# Rounding stop-opacity below 2 decimals is measurably lossy: at precision 1
# the 64-value opacity vocabulary of a 1750-splat chameleon collapses to 7,
# ~1300 extra Gaussian tails snap to fully transparent, and LPIPS regresses
# 0.0063. At 2 decimals the optimization is free (identical LPIPS/SSIM to four
# decimals) and still shrinks the file, because the real savings come from hex
# colors, whitespace and ellipse->circle — not from rounding.
SVG_OPTIMIZE_MIN_SAFE_PRECISION = 2


def optimize_svg_file(
    path: str,
    precision: int = SVG_OPTIMIZE_MIN_SAFE_PRECISION,
    timeout: float = 300.0,
) -> Dict[str, Any]:
    """Shrink an emitted SVG in place with `svgo`, if it is installed.

    The original file is only replaced once the optimized result has been
    parsed and sanity-checked, so a missing, failing or misbehaving svgo can
    never damage good output. Returns a report for the run manifest.
    """
    report: Dict[str, Any] = {
        "applied": False,
        "precision": int(precision),
        "bytes_before": None,
        "bytes_after": None,
    }
    if not os.path.exists(path):
        report["reason"] = "input-missing"
        return report

    before = os.path.getsize(path)
    report["bytes_before"] = before

    binary = shutil.which("svgo")
    if binary is None:
        report["reason"] = "svgo-not-installed"
        logger.warning(
            "--svg-optimize requested but `svgo` is not on PATH; leaving the "
            "SVG unoptimized (install with `npm i -g svgo`)."
        )
        return report

    if int(precision) < SVG_OPTIMIZE_MIN_SAFE_PRECISION:
        logger.warning(
            "svg optimize precision=%s is below the measured-safe minimum of "
            "%s; stop-opacity quantization at this level visibly degrades "
            "splat output.",
            precision,
            SVG_OPTIMIZE_MIN_SAFE_PRECISION,
        )

    with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        proc = subprocess.run(
            [binary, "--precision", str(int(precision)), "-i", path, "-o", tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if proc.returncode != 0:
            report["reason"] = f"svgo-failed:{proc.returncode}"
            logger.warning(
                "svgo failed (%s); keeping the original SVG:\n%s",
                proc.returncode,
                (proc.stderr or "").strip()[:400],
            )
            return report

        after = os.path.getsize(tmp_path)
        if after == 0:
            report["reason"] = "svgo-empty-output"
            logger.warning("svgo produced an empty file; keeping the original SVG.")
            return report

        # Never trust the optimizer blindly: the result must still parse and
        # must still contain every shape we emitted.
        import xml.etree.ElementTree as ET

        try:
            ET.parse(tmp_path)
        except ET.ParseError as exc:
            report["reason"] = f"svgo-invalid-xml:{exc}"
            logger.warning("svgo output is not well-formed XML; keeping the original.")
            return report

        def _shapes(p: str) -> int:
            root = ET.parse(p).getroot()
            shape_names = {"ellipse", "circle", "path"}
            return sum(
                element.tag.rsplit("}", 1)[-1] in shape_names for element in root.iter()
            )

        if _shapes(tmp_path) < _shapes(path):
            report["reason"] = "svgo-dropped-shapes"
            logger.warning("svgo dropped shapes; keeping the original SVG.")
            return report

        if after >= before:
            report["reason"] = "no-size-win"
            report["bytes_after"] = after
            return report

        shutil.move(tmp_path, path)
        tmp_path = None  # moved
        report.update(
            applied=True,
            bytes_after=after,
            saved_bytes=before - after,
            saved_pct=round(100.0 * (before - after) / max(before, 1), 2),
        )
        logger.info(
            "Optimized SVG with svgo (precision=%s): %.0f KB -> %.0f KB (-%.1f%%)",
            precision,
            before / 1024,
            after / 1024,
            report["saved_pct"],
        )
        return report
    except subprocess.TimeoutExpired:
        report["reason"] = "svgo-timeout"
        logger.warning("svgo timed out after %ss; keeping the original SVG.", timeout)
        return report
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def save_svg(
    splats: List[GaussianSplat],
    width: int,
    height: int,
    output_path: str,
    k_sigma: float = 2.5,
    sort_by_area: bool = False,
    sort_mode: str = DEFAULT_EXPORT_ORDER,
    background_linear_rgb: Optional[npt.NDArray[Any]] = None,
    export_recipe: str = "standard",
    foreground_mask: Optional[npt.NDArray[Any]] = None,
    background_safe_mask: Optional[npt.NDArray[Any]] = None,
    edge_band_mask: Optional[npt.NDArray[Any]] = None,
    gradient_quality: str = SVG_GRADIENT_QUALITY_STANDARD,
    painter_order: str = SVG_PAINTER_ORDER_BACK_TO_FRONT,
) -> None:
    """
    Save splats as SVG file.

    Args:
        splats: List of Gaussian splats
        width: Image width in pixels
        height: Image height in pixels
        output_path: Output SVG file path
        k_sigma: Sigma multiplier for ellipse size (2-3 for 95-99.7% coverage)
        sort_by_area: Legacy flag to sort by area descending
        sort_mode: Export order: importance|area|input
        background_linear_rgb: Optional background color in linear RGB [0,1]
    """
    ordered_splats = _sort_splats_for_export(
        splats=splats,
        sort_mode=sort_mode,
        sort_by_area=sort_by_area,
    )

    # Generate SVG content
    svg_content = generate_svg_content(
        ordered_splats,
        width,
        height,
        k_sigma,
        background_linear_rgb=background_linear_rgb,
        export_recipe=export_recipe,
        foreground_mask=foreground_mask,
        background_safe_mask=background_safe_mask,
        edge_band_mask=edge_band_mask,
        gradient_quality=gradient_quality,
        painter_order=painter_order,
    )

    try:
        atomic_write_text(output_path, svg_content)
        logger.info(f"Saved SVG with {len(ordered_splats)} splats to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save SVG {output_path}: {e}")
        raise


def generate_svg_content(
    splats: List[GaussianSplat],
    width: int,
    height: int,
    k_sigma: float = 2.5,
    background_linear_rgb: Optional[npt.NDArray[Any]] = None,
    export_recipe: str = "standard",
    foreground_mask: Optional[npt.NDArray[Any]] = None,
    background_safe_mask: Optional[npt.NDArray[Any]] = None,
    edge_band_mask: Optional[npt.NDArray[Any]] = None,
    palette_size: Optional[int] = None,
    gradient_quality: str = SVG_GRADIENT_QUALITY_STANDARD,
    painter_order: str = SVG_PAINTER_ORDER_BACK_TO_FRONT,
) -> str:
    """
    Generate SVG content from splats.

    Args:
        splats: List of Gaussian splats
        width: Image width
        height: Image height
        k_sigma: Sigma multiplier for ellipse size
        background_linear_rgb: Optional background color in linear RGB [0,1]
        export_recipe: "standard" or "browser-compatible". The browser recipe
            feathers gradients, pre-compensates dark opaque splats against the
            background, and caps alpha in safe background regions.
        gradient_quality: "standard" or stricter adaptive "high" gradients.
        painter_order: Correct "back-to-front" SVG paint order or "legacy".

    Returns:
        Complete SVG document as string
    """
    normalized_recipe = _normalize_svg_export_recipe(export_recipe)
    normalized_gradient_quality = _normalize_svg_gradient_quality(gradient_quality)
    normalized_painter_order = _normalize_svg_painter_order(painter_order)
    if normalized_recipe == SVG_SCRIPTED_MATRIX_RECIPE:
        return generate_scripted_svg_content(
            splats=splats,
            width=width,
            height=height,
            k_sigma=k_sigma,
            background_linear_rgb=background_linear_rgb,
            foreground_mask=foreground_mask,
            background_safe_mask=background_safe_mask,
            edge_band_mask=edge_band_mask,
            gradient_quality=normalized_gradient_quality,
            painter_order=normalized_painter_order,
        )
    if normalized_recipe == SVG_PALETTE_QUANTIZED_RECIPE:
        return generate_palette_quantized_svg_content(
            splats=splats,
            width=width,
            height=height,
            k_sigma=k_sigma,
            background_linear_rgb=background_linear_rgb,
            foreground_mask=foreground_mask,
            background_safe_mask=background_safe_mask,
            edge_band_mask=edge_band_mask,
            palette_size=(
                SVG_PALETTE_QUANTIZED_DEFAULT_SIZE
                if palette_size is None
                else int(palette_size)
            ),
            painter_order=normalized_painter_order,
        )
    if normalized_recipe == SVG_BLUR_RECIPE:
        return generate_blur_svg_content(
            splats=splats,
            width=width,
            height=height,
            k_sigma=k_sigma,
            background_linear_rgb=background_linear_rgb,
            painter_order=normalized_painter_order,
        )
    use_browser_recipe = normalized_recipe == SVG_BROWSER_COMPAT_RECIPE

    masks = RegionMasks(
        width,
        height,
        foreground=foreground_mask,
        background_safe=background_safe_mask,
        edge_band=edge_band_mask,
    )

    bg_linear: Optional[npt.NDArray[Any]] = None
    bg_srgb: Optional[npt.NDArray[Any]] = None
    background_rect_line: Optional[str] = None
    if background_linear_rgb is not None:
        bg = np.asarray(background_linear_rgb, dtype=np.float32).reshape(-1)
        if bg.size != 3:
            raise ValueError("background_linear_rgb must have exactly 3 components")
        bg = np.clip(bg, 0.0, 1.0)
        bg_linear = bg
        bg_srgb = linear_to_srgb(bg)
        bg_r = int(np.clip(np.round(bg_srgb[0] * 255), 0, 255))
        bg_g = int(np.clip(np.round(bg_srgb[1] * 255), 0, 255))
        bg_b = int(np.clip(np.round(bg_srgb[2] * 255), 0, 255))
        background_rect_line = render_template(
            "svg/background.svg",
            width=width,
            height=height,
            fill=f"rgb({bg_r},{bg_g},{bg_b})",
            class_attr=' class="background"',
        ).rstrip("\n")

    gradient_blocks: List[str] = []

    def _browser_compensated_color(
        splat: GaussianSplat, alpha: float
    ) -> npt.NDArray[Any]:
        color_linear = np.clip(np.array(splat.color[:3], dtype=np.float32), 0.0, 1.0)
        color_srgb = linear_to_srgb(color_linear)
        if (
            not use_browser_recipe
            or bg_linear is None
            or bg_srgb is None
            or float(splat.alpha) < SVG_PRECOMP_ALPHA_THRESHOLD
            or float(np.max(color_srgb) * 255.0) > SVG_PRECOMP_MAX_SRGB
        ):
            return color_srgb

        # Browsers blend SVG stops in display space. Solve the stop color that
        # gives the same center-over-background result as linear alpha-over.
        paint_alpha = 1.0 - math.exp(-float(np.clip(alpha, 0.0, 1.0)))
        target_srgb = linear_to_srgb(
            paint_alpha * color_linear + (1.0 - paint_alpha) * bg_linear
        )
        if paint_alpha <= 1e-6:
            return color_srgb
        return np.clip(
            (target_srgb - (1.0 - paint_alpha) * bg_srgb) / paint_alpha, 0.0, 1.0
        )

    gradient_footprint = ELLIPSE_OVERLAP_BOOST * k_sigma
    feather_extent = SVG_FEATHER_EXTENT if use_browser_recipe else 1.0
    inner_end = 1.0 / feather_extent
    # Density-aware stop-error threshold: sparse scenes tolerate fewer stops
    # per splat; dense scenes need more because per-splat 2-stop ramps stack
    # into visible "unsmoothed" artifacts.
    stop_error, max_gradient_stops, opacity_precision = _svg_gradient_settings(
        normalized_gradient_quality, len(splats)
    )

    # Per-splat radial gradients approximate gaussian opacity in exported SVG.
    for i, splat in enumerate(splats):
        gradient_id = f"splat_grad_{i}"
        alpha = float(np.clip(splat.alpha, 0.0, 1.0))
        if use_browser_recipe and masks.in_safe_background(splat):
            alpha = min(alpha, SVG_BACKGROUND_ALPHA_CAP)

        rgb_srgb = _browser_compensated_color(splat, alpha)
        r = int(np.clip(np.round(rgb_srgb[0] * 255), 0, 255))
        g = int(np.clip(np.round(rgb_srgb[1] * 255), 0, 255))
        b = int(np.clip(np.round(rgb_srgb[2] * 255), 0, 255))
        color = f"rgb({r},{g},{b})"
        # True-Gaussian gradient stops with adaptive count: reproduce the
        # renderer's per-splat alpha-over opacity 1 - exp(-a * exp(-0.5 * r^2))
        # using only as many stops as needed to keep the piecewise-linear
        # interpolation within `stop_error` of the true curve. The threshold
        # is density-aware (see _density_aware_stop_error): looser for sparse
        # scenes, tighter for dense ones so 2-stop linear ramps don't pile up
        # into visible artifacts.
        adaptive_stops = _adaptive_gradient_stops(
            alpha,
            gradient_footprint,
            inner_end,
            max_error=stop_error,
            max_stops=max_gradient_stops,
            opacity_precision=opacity_precision,
        )
        offset_precision = 2 if opacity_precision > 2 else 1
        stop_lines = [
            render_template(
                "svg/standard_stop.svg",
                offset=f"{offset * 100:.{offset_precision}f}",
                color=color,
                opacity=f"{opacity:.{opacity_precision}f}",
            ).rstrip("\n")
            for offset, opacity in adaptive_stops
        ]
        if use_browser_recipe:
            mid_fade = (inner_end + 1.0) / 2.0
            stop_lines.append(
                render_template(
                    "svg/standard_stop.svg",
                    offset=f"{mid_fade * 100:.1f}",
                    color=color,
                    opacity="0",
                ).rstrip("\n")
            )
            stop_lines.append(
                render_template(
                    "svg/standard_stop.svg",
                    offset="100.0",
                    color=color,
                    opacity="0",
                ).rstrip("\n")
            )
        gradient_blocks.append(
            render_template(
                "svg/standard_gradient.svg",
                gradient_id=gradient_id,
                stops="\n".join(stop_lines),
            ).rstrip("\n")
        )

    ellipse_lines: List[str] = []
    for i in _svg_painter_indices(len(splats), normalized_painter_order):
        splat = splats[i]
        ellipse_element = splat_to_svg_ellipse(
            splat=splat,
            k_sigma=k_sigma,
            element_id=f"splat_{i}",
            gradient_id=f"splat_grad_{i}",
            radius_scale=feather_extent,
        )
        ellipse_lines.append(f"  {ellipse_element}")

    gradients = "\n".join(gradient_blocks)
    if gradients:
        gradients += "\n"
    background = ""
    if background_rect_line is not None:
        background = f"{background_rect_line}\n\n"
    return render_template(
        "svg/standard_document.svg",
        width=width,
        height=height,
        gradients=gradients,
        background=background,
        ellipses="\n".join(ellipse_lines),
    ).rstrip("\n")


def generate_scripted_svg_content(
    splats: List[GaussianSplat],
    width: int,
    height: int,
    k_sigma: float = 2.5,
    background_linear_rgb: Optional[npt.NDArray[Any]] = None,
    foreground_mask: Optional[npt.NDArray[Any]] = None,
    background_safe_mask: Optional[npt.NDArray[Any]] = None,
    edge_band_mask: Optional[npt.NDArray[Any]] = None,
    gradient_quality: str = SVG_GRADIENT_QUALITY_STANDARD,
    painter_order: str = SVG_PAINTER_ORDER_BACK_TO_FRONT,
) -> str:
    """
    Generate a compact browser SVG that stores splats as a numeric matrix.

    The SVG source contains one data row per splat plus a small script. On load,
    the script expands the rows into normal SVG radial gradients and matrix-
    transformed unit ellipses. This keeps the source small and gzip-friendly
    while matching the browser-compatible static SVG rendering in browsers that
    execute inline SVG scripts.
    """

    masks = RegionMasks(
        width,
        height,
        foreground=foreground_mask,
        background_safe=background_safe_mask,
        edge_band=edge_band_mask,
    )

    bg_linear = np.ones(3, dtype=np.float32)
    if background_linear_rgb is not None:
        bg = np.asarray(background_linear_rgb, dtype=np.float32).reshape(-1)
        if bg.size != 3:
            raise ValueError("background_linear_rgb must have exactly 3 components")
        bg_linear = np.clip(bg, 0.0, 1.0)
    bg_srgb = linear_to_srgb(bg_linear)
    bg_rgb = tuple(int(np.clip(np.round(c * 255), 0, 255)) for c in bg_srgb)

    def _scripted_color_and_alpha(
        splat: GaussianSplat,
    ) -> Tuple[Tuple[int, int, int], float]:
        alpha = float(np.clip(splat.alpha, 0.0, 1.0))
        if masks.in_safe_background(splat):
            alpha = min(alpha, SVG_BACKGROUND_ALPHA_CAP)

        color_linear = np.clip(np.array(splat.color[:3], dtype=np.float32), 0.0, 1.0)
        color_srgb = linear_to_srgb(color_linear)
        if (
            float(splat.alpha) >= SVG_PRECOMP_ALPHA_THRESHOLD
            and float(np.max(color_srgb) * 255.0) <= SVG_PRECOMP_MAX_SRGB
        ):
            paint_alpha = 1.0 - math.exp(-alpha)
            if paint_alpha > 1e-6:
                target_srgb = linear_to_srgb(
                    paint_alpha * color_linear + (1.0 - paint_alpha) * bg_linear
                )
                color_srgb = np.clip(
                    (target_srgb - (1.0 - paint_alpha) * bg_srgb) / paint_alpha,
                    0.0,
                    1.0,
                )

        rgb = tuple(int(np.clip(np.round(c * 255), 0, 255)) for c in color_srgb)
        return rgb, alpha

    def _matrix_row(splat: GaussianSplat) -> str:
        eigenvals, eigenvecs = splat.eigendecomposition()
        rx = max(
            MIN_ELLIPSE_RADIUS_PX,
            SVG_FEATHER_EXTENT
            * ELLIPSE_OVERLAP_BOOST
            * k_sigma
            * np.sqrt(max(float(eigenvals[0]), 1e-8)),
        )
        ry = max(
            MIN_ELLIPSE_RADIUS_PX,
            SVG_FEATHER_EXTENT
            * ELLIPSE_OVERLAP_BOOST
            * k_sigma
            * np.sqrt(max(float(eigenvals[1]), 1e-8)),
        )
        theta = float(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
        cos_t = float(np.cos(theta))
        sin_t = float(np.sin(theta))
        # SVG matrix(a b c d e f) maps the unit circle to the rotated ellipse.
        a = rx * cos_t
        b = rx * sin_t
        c = -ry * sin_t
        d = ry * cos_t
        e = float(splat.mu[0])
        f = float(splat.mu[1])
        rgb, alpha = _scripted_color_and_alpha(splat)
        values = [
            f"{a:.2f}",
            f"{b:.2f}",
            f"{c:.2f}",
            f"{d:.2f}",
            f"{e:.2f}",
            f"{f:.2f}",
            str(rgb[0]),
            str(rgb[1]),
            str(rgb[2]),
            f"{alpha:.4f}",
        ]
        return ",".join(values)

    # Script-created ellipses obey the same painter's-order rules as static
    # SVG. Store rows back-to-front so the first front-to-back splat is
    # appended last and therefore remains visually front-most.
    rows = ";".join(
        _matrix_row(splats[i]) for i in _svg_painter_indices(len(splats), painter_order)
    )
    gradient_footprint = ELLIPSE_OVERLAP_BOOST * k_sigma
    inner_end = 1.0 / SVG_FEATHER_EXTENT
    normalized_gradient_quality = _normalize_svg_gradient_quality(gradient_quality)
    scripted_stops = (
        SVG_GRADIENT_STOPS_HIGH
        if normalized_gradient_quality == SVG_GRADIENT_QUALITY_HIGH
        else SVG_GRADIENT_STOPS
    )
    scripted_opacity_precision = (
        4 if normalized_gradient_quality == SVG_GRADIENT_QUALITY_HIGH else 2
    )
    script = render_template(
        "svg/scripted_runtime.js",
        stops=scripted_stops,
        opacity_precision=scripted_opacity_precision,
        footprint=f"{gradient_footprint:.8f}",
        inner_end=f"{inner_end:.8f}",
    ).strip()

    return render_template(
        "svg/scripted_document.svg",
        width=width,
        height=height,
        background=f"rgb({bg_rgb[0]},{bg_rgb[1]},{bg_rgb[2]})",
        rows=rows,
        script=script,
    ).rstrip("\n")


def generate_palette_quantized_svg_content(
    splats: List[GaussianSplat],
    width: int,
    height: int,
    k_sigma: float = 2.5,
    background_linear_rgb: Optional[npt.NDArray[Any]] = None,
    foreground_mask: Optional[npt.NDArray[Any]] = None,
    background_safe_mask: Optional[npt.NDArray[Any]] = None,
    edge_band_mask: Optional[npt.NDArray[Any]] = None,
    palette_size: int = SVG_PALETTE_QUANTIZED_DEFAULT_SIZE,
    painter_order: str = SVG_PAINTER_ORDER_BACK_TO_FRONT,
) -> str:
    """Compact SVG that quantizes splat colors into a shared palette.

    Generates one radial gradient per palette color in the definitions (with the
    palette color baked into every stop) and references it per-splat via
    ``fill="url(#p{label})"``. Per-element ``opacity="..."`` scales the
    Gaussian profile to the splat's trained alpha. The governing Chromium
    target supports these standards-based radial gradients.

    The naive "one gradient per splat" `standard` recipe writes ~400 bytes
    per splat in gradient defs alone. This recipe writes one gradient
    block per palette color (~300 bytes * N) plus a thin ~100-byte ellipse
    per splat. At 40k splats / 128 palette colors that's ~4 MB vs ~16 MB
    for the standard recipe.

    The earlier "shared-currentcolor" attempt failed because per SVG spec
    ``currentColor`` inside a paint server resolves at the gradient's
    DEFINITION context, not the reference context. Color quantization
    sidesteps the spec issue by baking real colors into shared gradients.

    Trade-offs:
    - Color quantization introduces banding when the palette is too small;
      defaults to 128 colors which is visually clean for photographic input.
      Tune via refinement_config['svg_palette_size'].
    - The shared Gaussian stop profile uses an alpha-independent shape and
      relies on per-element opacity to scale to the splat's alpha. Exact at
      stop t=0 (when element_opacity = 1-exp(-alpha)); slight underestimate
      in the falloff at high alpha. Visually negligible for typical content.
    """
    from scipy.cluster.vq import kmeans2 as _kmeans2

    masks = RegionMasks(
        width,
        height,
        foreground=foreground_mask,
        background_safe=background_safe_mask,
        edge_band=edge_band_mask,
    )

    bg_srgb_str: Optional[str] = None
    if background_linear_rgb is not None:
        bg = np.asarray(background_linear_rgb, dtype=np.float32).reshape(-1)
        if bg.size != 3:
            raise ValueError("background_linear_rgb must have exactly 3 components")
        bg_srgb = linear_to_srgb(np.clip(bg, 0.0, 1.0))
        bg_r, bg_g, bg_b = (int(np.clip(np.round(c * 255), 0, 255)) for c in bg_srgb)
        bg_srgb_str = f"rgb({bg_r},{bg_g},{bg_b})"

    bg_lin_arr: Optional[npt.NDArray[Any]] = None
    bg_srgb_arr: Optional[npt.NDArray[Any]] = None
    if background_linear_rgb is not None:
        bg_lin_arr = np.clip(
            np.asarray(background_linear_rgb, dtype=np.float32).reshape(-1)[:3],
            0.0,
            1.0,
        )
        bg_srgb_arr = linear_to_srgb(bg_lin_arr)

    def _precompensated_srgb(splat: GaussianSplat) -> npt.NDArray[Any]:
        """Same dark-splat display-space solve as the browser recipe.

        Browsers blend stops in display space; solve the stop color whose
        center-over-background result matches linear alpha-over. Applied
        BEFORE clustering so the palette is built from deployable colors.
        """
        color_linear = np.clip(np.array(splat.color[:3], dtype=np.float32), 0.0, 1.0)
        color_srgb = linear_to_srgb(color_linear)
        if (
            bg_lin_arr is None
            or bg_srgb_arr is None
            or float(splat.alpha) < SVG_PRECOMP_ALPHA_THRESHOLD
            or float(np.max(color_srgb) * 255.0) > SVG_PRECOMP_MAX_SRGB
        ):
            return color_srgb
        paint_alpha = 1.0 - math.exp(-float(np.clip(splat.alpha, 0.0, 1.0)))
        if paint_alpha <= 1e-6:
            return color_srgb
        target_srgb = linear_to_srgb(
            paint_alpha * color_linear + (1.0 - paint_alpha) * bg_lin_arr
        )
        return np.clip(
            (target_srgb - (1.0 - paint_alpha) * bg_srgb_arr) / paint_alpha, 0.0, 1.0
        )

    # Palette-quantize the splats' sRGB colors via k-means. Use a fixed RNG
    # seed so the same input produces the same SVG byte-for-byte.
    splat_colors_srgb = np.empty((len(splats), 3), dtype=np.float64)
    for i, splat in enumerate(splats):
        splat_colors_srgb[i] = _precompensated_srgb(splat)
    actual_palette_size = int(max(1, min(int(palette_size), len(splats))))
    if actual_palette_size >= len(splats):
        # No clustering needed; each splat is its own palette entry.
        centroids = splat_colors_srgb
        labels = np.arange(len(splats), dtype=np.int64)
    else:
        rng = np.random.default_rng(42)
        try:
            centroids, labels = _kmeans2(
                splat_colors_srgb,
                actual_palette_size,
                minit="++",
                seed=rng,
            )
        except TypeError:
            # Older scipy: kmeans2 took an int seed, not a Generator.
            centroids, labels = _kmeans2(
                splat_colors_srgb,
                actual_palette_size,
                minit="++",
                seed=42,
            )
        labels = np.asarray(labels, dtype=np.int64)
        # k-means can converge to empty clusters; replace those centroids with
        # the mean of any orphans they would have hosted. With "++" init this is
        # rare but worth guarding against.
        centroids = np.clip(centroids, 0.0, 1.0)

    # Palette-shared gradient stops use a Gaussian falloff in opacity space.
    # The palette color is baked into stop-color; the per-element opacity
    # then scales the whole splat to its trained alpha. (An alpha-bucketed
    # exact-profile variant was measured LPIPS-neutral on real content while
    # costing ~75% more bytes, so the alpha-independent profile stays.)
    footprint = ELLIPSE_OVERLAP_BOOST * k_sigma
    n_stops = 5
    stop_t = np.linspace(0.0, 1.0, n_stops)
    stop_op = np.exp(-0.5 * (stop_t * footprint) ** 2)

    gradient_blocks: List[str] = []
    for i, centroid in enumerate(centroids):
        r = int(np.clip(np.round(centroid[0] * 255), 0, 255))
        g = int(np.clip(np.round(centroid[1] * 255), 0, 255))
        b = int(np.clip(np.round(centroid[2] * 255), 0, 255))
        color_str = f"rgb({r},{g},{b})"
        stop_lines: List[str] = []
        for t, op in zip(stop_t, stop_op):
            stop_lines.append(
                render_template(
                    "svg/palette_stop.svg",
                    offset=f"{t * 100:.1f}",
                    color=color_str,
                    opacity=f"{float(op):.4f}",
                ).rstrip("\n")
            )
        gradient_blocks.append(
            render_template(
                "svg/palette_gradient.svg",
                index=i,
                stops="\n".join(stop_lines),
            ).rstrip("\n")
        )

    ellipse_lines: List[str] = []
    for splat_index in _svg_painter_indices(len(splats), painter_order):
        splat = splats[splat_index]
        label = labels[splat_index]
        alpha = float(np.clip(splat.alpha, 0.0, 1.0))
        if masks.in_safe_background(splat):
            alpha = min(alpha, SVG_BACKGROUND_ALPHA_CAP)
        # Per-element opacity scales the shared Gaussian profile so the
        # center pixel reaches the true alpha-over center opacity.
        element_opacity = 1.0 - math.exp(-alpha)
        if element_opacity <= 0.0:
            continue

        eigenvals, eigenvecs = splat.eigendecomposition()
        rx = max(
            MIN_ELLIPSE_RADIUS_PX,
            ELLIPSE_OVERLAP_BOOST
            * k_sigma
            * float(np.sqrt(max(float(eigenvals[0]), 1e-8))),
        )
        ry = max(
            MIN_ELLIPSE_RADIUS_PX,
            ELLIPSE_OVERLAP_BOOST
            * k_sigma
            * float(np.sqrt(max(float(eigenvals[1]), 1e-8))),
        )
        rotation_rad = float(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
        rotation_deg = float(np.degrees(rotation_rad))
        cx = float(splat.mu[0])
        cy = float(splat.mu[1])

        transform_attr = ""
        if abs(rotation_deg) > 0.1:
            transform_attr = (
                f' transform="rotate({rotation_deg:.1f} {cx:.1f} {cy:.1f})"'
            )

        ellipse_lines.append(
            render_template(
                "svg/palette_ellipse.svg",
                cx=f"{cx:.1f}",
                cy=f"{cy:.1f}",
                rx=f"{rx:.2f}",
                ry=f"{ry:.2f}",
                opacity=f"{element_opacity:.4f}",
                transform_attr=transform_attr,
                label=int(label),
            ).rstrip("\n")
        )

    gradients = "\n".join(gradient_blocks)
    if gradients:
        gradients += "\n"
    background = ""
    if bg_srgb_str is not None:
        background = (
            render_template(
                "svg/palette_background.svg",
                width=width,
                height=height,
                fill=bg_srgb_str,
            ).rstrip("\n")
            + "\n\n"
        )
    return render_template(
        "svg/palette_document.svg",
        width=width,
        height=height,
        palette_size=actual_palette_size,
        gradients=gradients,
        background=background,
        ellipses="\n".join(ellipse_lines),
    ).rstrip("\n")


def generate_blur_svg_content(
    splats: List[GaussianSplat],
    width: int,
    height: int,
    k_sigma: float = 2.5,
    background_linear_rgb: Optional[npt.NDArray[Any]] = None,
    painter_order: str = SVG_PAINTER_ORDER_BACK_TO_FRONT,
) -> str:
    """Generate SVG using `<feGaussianBlur>` per splat instead of gradient stops.

    Each splat becomes a small flat-fill ellipse passed through a shared
    Gaussian blur filter keyed on quantized sigma. The blur produces a true
    Gaussian falloff, which the gradient-stop recipes can only approximate.
    """
    if not splats:
        return _empty_svg_document(width, height, background_linear_rgb)

    bg_rect_line: Optional[str] = None
    if background_linear_rgb is not None:
        bg_srgb = linear_to_srgb(np.array(background_linear_rgb[:3], dtype=np.float32))
        r = int(np.clip(np.round(bg_srgb[0] * 255), 0, 255))
        g = int(np.clip(np.round(bg_srgb[1] * 255), 0, 255))
        b = int(np.clip(np.round(bg_srgb[2] * 255), 0, 255))
        bg_rect_line = render_template(
            "svg/background.svg",
            width=int(width),
            height=int(height),
            fill=f"rgb({r},{g},{b})",
            class_attr="",
        ).rstrip("\n")

    # Build sigma buckets (geometric quantization). Cache the per-splat
    # eigendecomposition so the emit loop below doesn't redo it.
    decomps: List[Tuple[float, float, npt.NDArray[Any]]] = []
    for s in splats:
        eigenvals, eigenvecs = s.eigendecomposition()
        decomps.append(
            (
                float(np.sqrt(max(float(eigenvals[0]), 1e-8))),
                float(np.sqrt(max(float(eigenvals[1]), 1e-8))),
                eigenvecs,
            )
        )
    all_sigma = np.array([sx for sx, _, _ in decomps] + [sy for _, sy, _ in decomps])
    sigma_min = float(max(all_sigma.min(), 0.25))
    sigma_max = float(max(all_sigma.max(), sigma_min * 1.001))
    n_buckets = int(SVG_BLUR_SIGMA_BUCKETS)
    bucket_edges = np.geomspace(sigma_min, sigma_max, n_buckets + 1)
    bucket_centers = np.sqrt(bucket_edges[:-1] * bucket_edges[1:])

    def _bucket(sigma_val: float) -> int:
        return int(
            np.clip(
                np.searchsorted(bucket_edges, sigma_val, side="right") - 1,
                0,
                n_buckets - 1,
            )
        )

    # Shared filters. Each filter is anisotropic-capable (stdDeviation = "sx sy")
    # but we use a single bucket per (sx, sy) pair; the actual stdDeviation
    # carries both components from the bucket center pair. The simpler choice:
    # one filter per sigma bucket, used isotropically for that bucket, and
    # absorb minor anisotropy in the ellipse aspect ratio.
    filter_blocks: List[str] = []
    for i, center in enumerate(bucket_centers):
        # Filter region: the blurred core has support out to k·σ_axis + 3·σ_blur,
        # and for anisotropic splats σ_blur (≈ σ_geo) can be ~2× the minor
        # axis — the old ±2·bbox region hard-clipped those tails at ~2σ-minor.
        # ±3.5·bbox (x=-300%, width=700%) covers anisotropy up to the 4.0
        # structure-tensor clip with margin; filter regions are cheap.
        filter_blocks.append(
            render_template(
                "svg/blur_filter.svg",
                index=i,
                std_deviation=f"{center:.3f}",
            ).rstrip("\n")
        )

    # Per-splat small ellipse referencing the bucketed filter.
    ellipse_lines: List[str] = []
    for splat_index in _svg_painter_indices(len(splats), painter_order):
        splat = splats[splat_index]
        sigma_x, sigma_y, eigenvecs = decomps[splat_index]
        sigma_geo = float(np.sqrt(max(sigma_x * sigma_y, 1e-8)))
        idx = _bucket(sigma_geo)
        cx, cy = float(splat.mu[0]), float(splat.mu[1])
        rx = max(MIN_ELLIPSE_RADIUS_PX, SVG_BLUR_CORE_K_SIGMA * sigma_x)
        ry = max(MIN_ELLIPSE_RADIUS_PX, SVG_BLUR_CORE_K_SIGMA * sigma_y)
        rotation_rad = float(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
        rotation_deg = float(np.degrees(rotation_rad))
        transform_attr = (
            f' transform="rotate({rotation_deg:.2f} {cx:.2f} {cy:.2f})"'
            if abs(rotation_deg) > 0.05
            else ""
        )
        rgb_srgb = linear_to_srgb(np.array(splat.color[:3], dtype=np.float32))
        ri = int(np.clip(np.round(rgb_srgb[0] * 255), 0, 255))
        gi = int(np.clip(np.round(rgb_srgb[1] * 255), 0, 255))
        bi = int(np.clip(np.round(rgb_srgb[2] * 255), 0, 255))
        # Mass-fraction compensation: a uniform disk of radius k·σ convolved
        # with Gaussian σ has peak fill_opacity·(1 − e^(−k²/2)). The renderer's
        # per-splat peak opacity is 1 − e^(−α) (alpha-over layer alpha), not α
        # itself, so compensate that target by 1/mf. Values above the
        # mf ceiling clip — a structural limit of the flat-core approximation.
        mass_fraction = 1.0 - np.exp(-0.5 * SVG_BLUR_CORE_K_SIGMA**2)
        peak_opacity = 1.0 - np.exp(-float(np.clip(splat.alpha, 0.0, 1.0)))
        alpha = float(np.clip(peak_opacity / max(mass_fraction, 1e-6), 0.0, 1.0))
        ellipse_lines.append(
            render_template(
                "svg/blur_ellipse.svg",
                cx=f"{cx:.2f}",
                cy=f"{cy:.2f}",
                rx=f"{rx:.2f}",
                ry=f"{ry:.2f}",
                transform_attr=transform_attr,
                fill=f"rgb({ri},{gi},{bi})",
                opacity=f"{alpha:.4f}",
                filter_index=idx,
            ).rstrip("\n")
        )

    filters = "\n".join(filter_blocks)
    if filters:
        filters += "\n"
    background = "" if bg_rect_line is None else f"{bg_rect_line}\n"
    return render_template(
        "svg/blur_document.svg",
        width=int(width),
        height=int(height),
        filters=filters,
        background=background,
        ellipses="\n".join(ellipse_lines),
    ).rstrip("\n")


def _empty_svg_document(
    width: int,
    height: int,
    background_linear_rgb: Optional[npt.NDArray[Any]],
) -> str:
    bg_line = ""
    if background_linear_rgb is not None:
        bg_srgb = linear_to_srgb(np.array(background_linear_rgb[:3], dtype=np.float32))
        r = int(np.clip(np.round(bg_srgb[0] * 255), 0, 255))
        g = int(np.clip(np.round(bg_srgb[1] * 255), 0, 255))
        b = int(np.clip(np.round(bg_srgb[2] * 255), 0, 255))
        bg_line = (
            render_template(
                "svg/background.svg",
                width=int(width),
                height=int(height),
                fill=f"rgb({r},{g},{b})",
                class_attr="",
            ).rstrip("\n")
            + "\n"
        )
    return render_template(
        "svg/empty_document.svg",
        width=int(width),
        height=int(height),
        background=bg_line,
    ).rstrip("\n")


def splat_to_svg_ellipse(
    splat: GaussianSplat,
    k_sigma: float = 2.5,
    element_id: Optional[str] = None,
    gradient_id: Optional[str] = None,
    radius_scale: float = 1.0,
) -> str:
    """
    Convert Gaussian splat to SVG ellipse element.

    Args:
        splat: Gaussian splat
        k_sigma: Sigma multiplier for ellipse size
        element_id: Optional element ID

    Returns:
        SVG ellipse element string
    """
    # Get eigendecomposition
    eigenvals, eigenvecs = splat.eigendecomposition()

    # Compute ellipse parameters
    # Semi-axes lengths (k * σ where σ = sqrt(eigenvalue))
    rx = max(
        MIN_ELLIPSE_RADIUS_PX,
        float(max(radius_scale, 0.0))
        * ELLIPSE_OVERLAP_BOOST
        * k_sigma
        * np.sqrt(eigenvals[0]),
    )
    ry = max(
        MIN_ELLIPSE_RADIUS_PX,
        float(max(radius_scale, 0.0))
        * ELLIPSE_OVERLAP_BOOST
        * k_sigma
        * np.sqrt(eigenvals[1]),
    )

    # Rotation angle (from first eigenvector)
    # Note: SVG rotation is in degrees, positive clockwise
    rotation_rad = np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0])
    rotation_deg = np.degrees(rotation_rad)

    # Center position
    cx, cy = splat.mu

    # Color fallback for environments that do not resolve referenced gradients.
    rgb_srgb = linear_to_srgb(np.array(splat.color[:3], dtype=np.float32))
    color_int = tuple(int(np.clip(np.round(c * 255), 0, 255)) for c in rgb_srgb)
    fallback_color = f"rgb({color_int[0]},{color_int[1]},{color_int[2]})"

    # Build ellipse element
    id_attr = f' id="{element_id}"' if element_id else ""
    transform_attr = (
        f' transform="rotate({rotation_deg:.2f} {cx:.2f} {cy:.2f})"'
        if abs(rotation_deg) > 0.1
        else ""
    )

    fill_attr = f"url(#{gradient_id})" if gradient_id else fallback_color
    alpha_attr = (
        ""
        if gradient_id
        else f' fill-opacity="{float(np.clip(splat.alpha, 0.0, 1.0)):.3f}"'
    )
    return render_template(
        "svg/ellipse.svg",
        id_attr=id_attr,
        cx=f"{cx:.2f}",
        cy=f"{cy:.2f}",
        rx=f"{rx:.2f}",
        ry=f"{ry:.2f}",
        fill=fill_attr,
        fallback_fill=fallback_color,
        alpha_attr=alpha_attr,
        transform_attr=transform_attr,
    ).rstrip("\n")


def estimate_svg_size(splats: List[GaussianSplat]) -> int:
    """
    Estimate SVG file size in bytes.

    Args:
        splats: List of splats

    Returns:
        Estimated file size in bytes
    """
    # Rough estimate: ~120 bytes per ellipse + overhead
    bytes_per_ellipse = 120
    overhead_bytes = 500  # Headers, etc.

    return len(splats) * bytes_per_ellipse + overhead_bytes
