"""Cross-export integrity checks kept outside exporter dependencies."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any, Dict, List

import numpy as np

from .splat import GaussianSplat, RawSplat


def validate_export_roundtrip(
    splats: List[GaussianSplat],
    width: int,
    height: int,
    k_sigma: float = 2.5,
    atol: float = 1e-4,
) -> Dict[str, Any]:
    """Round-trip canonical splats and validate both vector exporters."""

    from .pptx_export import generate_drawingml_slide_content
    from .svg_export import generate_svg_content

    def element_count(content: str, local_name: str) -> int:
        root = ET.fromstring(content)
        return sum(
            element.tag.rsplit("}", 1)[-1] == local_name for element in root.iter()
        )

    reconstructed = [
        GaussianSplat.from_raw_splat(RawSplat.from_dict(s.to_raw_splat().to_dict()))
        for s in splats
    ]
    svg_content = generate_svg_content(reconstructed, width, height, k_sigma)
    dml_content = generate_drawingml_slide_content(
        reconstructed, width, height, k_sigma
    )
    svg_count = element_count(svg_content, "ellipse")
    svg_gradient_count = element_count(svg_content, "radialGradient")
    dml_count = element_count(dml_content, "sp")
    if not splats:
        return {
            "pass": True,
            "num_splats": 0,
            "max_mu_delta": 0.0,
            "max_color_delta": 0.0,
            "max_alpha_delta": 0.0,
            "svg_ellipse_count": svg_count,
            "svg_gradient_count": svg_gradient_count,
            "drawingml_shape_count": dml_count,
        }

    mu_delta = max(
        float(np.max(np.abs(original.mu[:2] - restored.mu[:2])))
        for original, restored in zip(splats, reconstructed)
    )
    color_delta = max(
        float(np.max(np.abs(original.color[:3] - restored.color[:3])))
        for original, restored in zip(splats, reconstructed)
    )
    alpha_delta = max(
        abs(float(original.alpha) - float(restored.alpha))
        for original, restored in zip(splats, reconstructed)
    )
    passed = (
        mu_delta <= atol
        and color_delta <= atol
        and alpha_delta <= atol
        and svg_count == len(splats)
        and svg_gradient_count == len(splats)
        and dml_count == len(splats)
    )
    return {
        "pass": bool(passed),
        "num_splats": len(splats),
        "max_mu_delta": float(mu_delta),
        "max_color_delta": float(color_delta),
        "max_alpha_delta": float(alpha_delta),
        "svg_ellipse_count": int(svg_count),
        "svg_gradient_count": int(svg_gradient_count),
        "drawingml_shape_count": int(dml_count),
        "atol": float(atol),
    }
