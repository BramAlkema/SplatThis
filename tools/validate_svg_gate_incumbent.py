#!/usr/bin/env python3
"""Re-run the SVG compositor gate study under the corrected-standard incumbent.

The July 2026 corpus study (`data/svg-compositor-corpus.json`) measured the
gate with the legacy order as incumbent and motivated moving the incumbent to
the default corrected-standard emitter. That change shipped on a structural
argument -- the gate's floor becomes the default output -- without a fresh
measurement. This script closes that gap: for each of the 21 stored seed-0
populations it emits the three gate candidates with the shipped emitter,
captures each in governing Chromium, freezes worst-error ROIs from the
incumbent exactly as the engine does, scores the full fidelity metric vector,
and runs the engine's own selection policy. The result is written to
``data/svg-compositor-corpus-v2.json`` with per-image evidence.

Usage::

    PYTHONPATH=src python tools/validate_svg_gate_incumbent.py
"""

from __future__ import annotations

import gzip
import json
import statistics
import sys
import tempfile
from datetime import date
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from splatthis.browser_capture import get_shared_svg_renderer  # noqa: E402
from splatthis.color import srgb_to_linear  # noqa: E402
from splatthis.export_common import _sort_splats_for_export  # noqa: E402
from splatthis.fidelity.analysis import analyze_residual  # noqa: E402
from splatthis.fidelity.metrics import compute_fidelity_metrics  # noqa: E402
from splatthis.io import load_png  # noqa: E402
from splatthis.storage import load_splats_json  # noqa: E402
from splatthis.svg_export import generate_svg_content  # noqa: E402
from splatthis.svg_recipe_gate import (  # noqa: E402
    SvgRecipeGatePolicy,
    select_recipe_candidate,
)

RUNS = REPO / "result" / "corpus" / "runs"
SOURCES = REPO / "result" / "corpus" / "images"
OUTPUT = REPO / "data" / "svg-compositor-corpus-v2.json"

#: Incumbent first -- the shipped default emitter -- mirroring
#: engine_artifacts._run_svg_compositor_gate.
CANDIDATE_SPECS = (
    ("corrected-standard", "standard", "back-to-front"),
    ("legacy-standard", "standard", "legacy"),
    ("corrected-high", "high", "back-to-front"),
)

#: Identical to the engine's gate policy.
POLICY = SvgRecipeGatePolicy(
    max_size_growth_fraction=1.0,
    max_render_time_growth_fraction=0.50,
    max_ssim_regression=0.002,
    max_ms_ssim_regression=0.003,
    max_lpips_regression=0.005,
    maximum_median_size_growth_fraction=1.0,
    maximum_median_render_time_growth_fraction=0.25,
)


def _median(values: List[float]) -> float:
    return statistics.median(values)


def study_image(image: str, renderer: Any, work_dir: Path) -> Dict[str, Any]:
    run = RUNS / f"{image}_svg_s0_art"
    splats = load_splats_json(str(run / "final.raw.json"))
    manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    background = np.asarray(
        manifest["config"]["background_linear_rgb"], dtype=np.float32
    )
    source = np.asarray(
        load_png(str(SOURCES / f"{image}.png"))[..., :3], dtype=np.float32
    )
    height, width = source.shape[:2]

    ordered = _sort_splats_for_export(splats)
    contents = {
        name: generate_svg_content(
            ordered,
            width,
            height,
            2.5,
            background_linear_rgb=background,
            export_recipe="standard",
            gradient_quality=quality,
            painter_order=order,
        )
        for name, quality, order in CANDIDATE_SPECS
    }

    rendered_by_name: Dict[str, np.ndarray] = {}
    captures: Dict[str, Any] = {}
    for name, _, _ in CANDIDATE_SPECS:
        svg_path = work_dir / f"{image}-{name}.svg"
        png_path = work_dir / f"{image}-{name}.png"
        svg_path.write_text(contents[name], encoding="utf-8")
        capture = renderer.capture(
            svg_path, png_path, width=width, height=height, repeats=1
        )
        from PIL import Image

        with Image.open(png_path) as raster:
            srgb = np.asarray(raster.convert("RGB"), dtype=np.float32) / 255.0
        rendered_by_name[name] = srgb_to_linear(srgb)
        captures[name] = capture

    incumbent_name = CANDIDATE_SPECS[0][0]
    fixed_rois = analyze_residual(
        source,
        rendered_by_name[incumbent_name],
        roi_size=min(64, height, width),
        roi_count=8,
    ).fixed_rois

    measurements = []
    for name, quality, order in CANDIDATE_SPECS:
        content = contents[name]
        metrics = compute_fidelity_metrics(
            source,
            rendered_by_name[name],
            fixed_rois=fixed_rois,
            splat_count=len(splats),
            file_size_bytes=len(
                gzip.compress(content.encode("utf-8"), compresslevel=9)
            ),
            render_method=renderer.renderer_label,
        ).as_dict()
        measurements.append(
            {
                "recipe": name,
                "painter_order": order,
                "gradient_quality": quality,
                "raw_size_bytes": len(content.encode("utf-8")),
                "render_time_sec": captures[name].capture_time_ms / 1000.0,
                "pixel_stable": captures[name].pixel_stable,
                **{
                    key: (
                        None
                        if isinstance(value, float) and not np.isfinite(value)
                        else value
                    )
                    for key, value in metrics.items()
                },
            }
        )

    selection = select_recipe_candidate(measurements[0], measurements[1:], POLICY)
    return {
        "image": image,
        "splats": len(splats),
        "size": [width, height],
        "selected": str(selection["selected_recipe"]),
        "candidates": {
            m["recipe"]: {
                "ssim_srgb": m.get("ssim_srgb"),
                "lpips": m.get("lpips"),
                "gzip_size_bytes": m.get("file_size_bytes"),
                "capture_time_ms": m["render_time_sec"] * 1000.0,
            }
            for m in measurements
        },
    }


def main() -> int:
    images = sorted(p.stem for p in SOURCES.glob("*.png"))
    if len(images) != 21:
        print(f"error: expected 21 corpus images, found {len(images)}", file=sys.stderr)
        return 2
    renderer = get_shared_svg_renderer()
    results = []
    with tempfile.TemporaryDirectory(prefix="svg-gate-v2-") as tmp:
        work_dir = Path(tmp)
        for index, image in enumerate(images, 1):
            print(f"[{index}/{len(images)}] {image} ...", flush=True)
            results.append(study_image(image, renderer, work_dir))

    variants = [name for name, _, _ in CANDIDATE_SPECS]
    medians = {
        name: {
            metric: _median([r["candidates"][name][metric] for r in results])
            for metric in ("ssim_srgb", "lpips", "gzip_size_bytes", "capture_time_ms")
        }
        for name in variants
    }
    selected_counts: Dict[str, int] = {}
    for r in results:
        selected_counts[r["selected"]] = selected_counts.get(r["selected"], 0) + 1
    gate_medians = {
        metric: _median([r["candidates"][r["selected"]][metric] for r in results])
        for metric in ("ssim_srgb", "lpips", "gzip_size_bytes", "capture_time_ms")
    }

    payload = {
        "schema": "splatthis.svg-compositor-corpus/2",
        "date": date.today().isoformat(),
        "browser": renderer.renderer_label,
        "incumbent": CANDIDATE_SPECS[0][0],
        "policy": POLICY.as_dict(),
        "scope": {
            "images": len(images),
            "seed": 0,
            "capture_repeats": 1,
            "populations": "stored seed-0 svg populations, re-emitted by the shipped emitter",
        },
        "seed0_medians": {
            **{name.replace("-", "_"): medians[name] for name in variants},
            "artifact_gate_selection": {
                "selected_counts": selected_counts,
                **gate_medians,
            },
        },
        "per_image": results,
        "note": (
            "Validation of the incumbent change shipped in 0.2.6: the gate "
            "races corrected-standard (the default emitter) as incumbent, "
            "with legacy and corrected-high as challengers. Compare "
            "data/svg-compositor-corpus.json (July 2026, legacy incumbent)."
        ),
    }
    OUTPUT.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(f"\nwrote {OUTPUT.relative_to(REPO)}")
    print("selected:", selected_counts)
    print(
        "gate medians: "
        f"ssim {gate_medians['ssim_srgb']:.4f}, lpips {gate_medians['lpips']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
