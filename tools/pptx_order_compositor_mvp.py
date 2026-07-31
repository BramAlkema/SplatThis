#!/usr/bin/env python3
"""Compare legacy and corrected native-PPTX painter order in PowerPoint."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import shutil
import statistics
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from full_corpus_mvp import _capture_powerpoint_slideshow

from png2svg_gs.fidelity.analysis import analyze_residual
from png2svg_gs.fidelity.metrics import compute_fidelity_metrics
from png2svg_gs.io import (
    _sort_splats_for_export,
    atomic_output_path,
    atomic_write_text,
    load_png,
    load_splats_json,
    save_pptx_with_splats,
)
from png2svg_gs.svg_recipe_gate import SvgRecipeGatePolicy, select_recipe_candidate


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--splats-json", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--selected-output",
        type=Path,
        default=None,
        help="Winning native deck (default: OUTPUT_DIR/selected.pptx)",
    )
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    return args


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _finite(value: Any) -> Any:
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _atomic_copy(source: Path, destination: Path) -> None:
    """Copy a selected binary artifact without exposing a partial deck."""

    with atomic_output_path(destination) as temporary:
        shutil.copyfile(source, temporary)


def _capture_variant(
    pptx: Path,
    output_dir: Path,
    name: str,
    *,
    width: int,
    height: int,
    repeats: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    capture_dir = output_dir / "captures"
    capture_dir.mkdir(parents=True, exist_ok=True)
    captures = []
    for repeat in range(repeats):
        capture = capture_dir / f"{name}-{repeat + 1}.png"
        started = time.perf_counter()
        returncode, message = _capture_powerpoint_slideshow(
            pptx, capture, width, height
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        atomic_write_text(capture.with_suffix(".log"), message)
        if returncode or not capture.exists():
            raise RuntimeError(
                f"PowerPoint capture failed for {name} repeat {repeat + 1}: "
                f"{message.strip()}"
            )
        captures.append(
            {
                "path": str(capture),
                "sha256": _sha256(capture),
                "capture_time_ms": elapsed_ms,
            }
        )
    hashes = {item["sha256"] for item in captures}
    rendered = np.asarray(load_png(str(capture_dir / f"{name}-1.png"))[:, :, :3])
    return rendered, {
        "captures": captures,
        "pixel_stable": len(hashes) == 1,
        "capture_time_ms": statistics.median(
            float(item["capture_time_ms"]) for item in captures
        ),
    }


def _write_overview(
    output_dir: Path,
    source: Path,
    results: Sequence[dict[str, Any]],
    selection: dict[str, Any],
    selected_artifact: Path,
    *,
    width: int,
    height: int,
) -> None:
    baseline = next(result for result in results if result["recipe"] == "legacy-order")
    cards = [
        f'<figure><img src="{html.escape(os.path.relpath(source, output_dir))}">'
        "<figcaption>Source</figcaption></figure>"
    ]
    for result in results:
        name = str(result["recipe"])
        cards.append(
            f'<figure><img src="captures/{html.escape(name)}-1.png">'
            f"<figcaption>{html.escape(name)}"
            f"<small>SSIM {float(result['ssim_srgb']):.6f} · "
            f"Δ {float(result['ssim_srgb']) - float(baseline['ssim_srgb']):+.6f}<br>"
            f"LPIPS {float(result['lpips']):.6f} · "
            f"Δ {float(result['lpips']) - float(baseline['lpips']):+.6f}<br>"
            f"{int(result['file_size_bytes']) / 1000.0:.0f} KB · "
            f"capture {float(result['render_time_sec']) * 1000.0:.0f} ms · "
            f"stable {str(result['pixel_stable']).lower()}</small>"
            f'<a href="{html.escape(name)}.pptx">Open native PPTX</a>'
            "</figcaption></figure>"
        )
    document = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>PPTX painter-order MVP</title>
<style>
:root{{color-scheme:dark;font:15px system-ui,sans-serif}}
body{{margin:0;padding:24px;background:#111;color:#eee}}
h1,p,.grid{{max-width:1300px;margin:0 auto 18px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:18px}}
figure{{margin:0;padding:12px;background:#1c1c1c;border:1px solid #333;border-radius:10px}}
img{{display:block;width:100%;max-width:{width}px;aspect-ratio:{width}/{height};margin:auto}}
figcaption{{margin-top:10px;font-weight:650}}small{{display:block;color:#aaa;font-weight:400}}
a{{color:#8cc8ff}}
</style></head><body><h1>PPTX painter-order MVP</h1>
<p>Selected: <strong>{html.escape(str(selection['selected_recipe']))}</strong>.
Complete native-size frames captured by Microsoft PowerPoint.
<a href="{html.escape(os.path.relpath(selected_artifact, output_dir))}">Open selected native PPTX</a></p>
<div class="grid">{''.join(cards)}</div></body></html>
"""
    atomic_write_text(output_dir / "index.html", document)


def main() -> int:
    args = _parse_args()
    source = args.source.resolve()
    splats_path = args.splats_json.resolve()
    manifest_path = args.manifest.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    target = np.asarray(load_png(str(source))[:, :, :3], dtype=np.float32)
    height, width = target.shape[:2]
    background = np.asarray(
        manifest["config"]["background_linear_rgb"], dtype=np.float32
    )
    k_sigma = float(manifest["config"].get("k_sigma", 2.5))
    splat_style = str(manifest["config"].get("pptx_splat_style", "gradient"))
    front_to_back = _sort_splats_for_export(load_splats_json(str(splats_path)))
    variants = {
        "legacy-order": "legacy",
        "corrected-order": "back-to-front",
    }

    rendered: dict[str, np.ndarray] = {}
    capture_data: dict[str, dict[str, Any]] = {}
    artifacts: dict[str, Path] = {}
    for name, painter_order in variants.items():
        pptx = output_dir / f"{name}.pptx"
        save_pptx_with_splats(
            front_to_back,
            width,
            height,
            str(pptx),
            k_sigma=k_sigma,
            sort_mode="input",
            background_linear_rgb=background,
            splat_style=splat_style,
            painter_order=painter_order,
        )
        artifacts[name] = pptx
        rendered[name], capture_data[name] = _capture_variant(
            pptx,
            output_dir,
            name,
            width=width,
            height=height,
            repeats=args.repeats,
        )

    fixed_rois = analyze_residual(
        target,
        rendered["legacy-order"],
        roi_size=min(64, height, width),
        roi_count=8,
    ).fixed_rois
    results = []
    for name, painter_order in variants.items():
        metrics = compute_fidelity_metrics(
            target,
            rendered[name],
            fixed_rois=fixed_rois,
            splat_count=len(front_to_back),
            file_size_bytes=artifacts[name].stat().st_size,
            render_method="Microsoft PowerPoint slideshow",
        ).as_dict()
        results.append(
            {
                "recipe": name,
                "painter_order": painter_order,
                "render_time_sec": capture_data[name]["capture_time_ms"] / 1000.0,
                "pixel_stable": capture_data[name]["pixel_stable"],
                **{key: _finite(value) for key, value in metrics.items()},
            }
        )

    policy = SvgRecipeGatePolicy(
        max_size_growth_fraction=0.01,
        max_render_time_growth_fraction=0.50,
        maximum_median_size_growth_fraction=0.01,
        maximum_median_render_time_growth_fraction=0.50,
    )
    selection = select_recipe_candidate(results[0], results[1:], policy)
    selected_name = str(selection["selected_recipe"])
    selected_output = (
        args.selected_output.resolve()
        if args.selected_output is not None
        else output_dir / "selected.pptx"
    )
    _atomic_copy(artifacts[selected_name], selected_output)
    report = {
        "schema": "splatthis.pptx-order-compositor-mvp/2",
        "source": str(source),
        "splats_json": str(splats_path),
        "manifest": str(manifest_path),
        "width": width,
        "height": height,
        "splat_count": len(front_to_back),
        "pptx_splat_style": splat_style,
        "fixed_rois": [list(roi) for roi in fixed_rois],
        "policy": policy.as_dict(),
        "results": results,
        "capture": capture_data,
        "selection": selection,
        "selected_artifact": {
            "recipe": selected_name,
            "path": str(selected_output),
            "sha256": _sha256(selected_output),
        },
    }
    atomic_write_text(output_dir / "results.json", json.dumps(report, indent=2))
    _write_overview(
        output_dir,
        source,
        results,
        selection,
        selected_output,
        width=width,
        height=height,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
