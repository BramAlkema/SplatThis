#!/usr/bin/env python3
"""Sweep residual edge-stroke candidates against an actual SVG artifact."""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

from png2svg_gs.io import (
    _try_rasterize_svg_to_linear_rgb,
    compute_quality_metrics,
    load_png,
)
from png2svg_gs.mixed_primitives import (
    edge_paths_to_svg_group,
    edge_strokes_to_svg_group,
    inject_edge_paths_into_pptx,
    inject_svg_before_close,
    propose_residual_edge_paths,
    propose_residual_edge_strokes,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("baseline_svg", type=Path)
    parser.add_argument("--baseline-pptx", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("./tmp/mixed-mvp"))
    parser.add_argument("--counts", default="8,16,32")
    parser.add_argument("--lengths", default="3,5,7")
    parser.add_argument("--widths", default="0.6,1.0,1.4")
    parser.add_argument("--opacities", default="0.45,0.65,0.85")
    parser.add_argument("--primitive", choices=["paths", "strokes"], default="paths")
    parser.add_argument("--min-ssim-gain", type=float, default=0.005)
    parser.add_argument("--max-psnr-regression", type=float, default=0.1)
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Reuse one scratch SVG during the sweep and retain only the winner.",
    )
    return parser


def _csv_numbers(value: str, cast):
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def _svg_size(svg_path: Path) -> tuple[int, int]:
    root = ET.fromstring(svg_path.read_text())
    view_box = root.attrib.get("viewBox", "").split()
    if len(view_box) == 4:
        return int(round(float(view_box[2]))), int(round(float(view_box[3])))
    return int(round(float(root.attrib["width"]))), int(
        round(float(root.attrib["height"]))
    )


def main() -> int:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    width, height = _svg_size(args.baseline_svg)
    target = load_png(str(args.source), target_size=(width, height))[..., :3]
    baseline_render, baseline_renderer = _try_rasterize_svg_to_linear_rgb(
        str(args.baseline_svg), width, height
    )
    if baseline_render is None:
        raise RuntimeError(f"could not rasterize baseline SVG: {baseline_renderer}")
    baseline_metrics = compute_quality_metrics(target, baseline_render)
    baseline_content = args.baseline_svg.read_text()

    records = []
    winner = None
    winner_content = None
    scratch_path = args.output_dir / "candidate.svg"
    for opacity in _csv_numbers(args.opacities, float):
        for length in _csv_numbers(args.lengths, float):
            for stroke_width in _csv_numbers(args.widths, float):
                max_count = max(_csv_numbers(args.counts, int), default=0)
                if args.primitive == "paths":
                    primitives = propose_residual_edge_paths(
                        target,
                        baseline_render,
                        max_paths=max_count,
                        path_length=length,
                        width=stroke_width,
                        opacity=opacity,
                    )
                else:
                    primitives = propose_residual_edge_strokes(
                        target,
                        baseline_render,
                        max_strokes=max_count,
                        length=length,
                        width=stroke_width,
                        opacity=opacity,
                    )
                for count in _csv_numbers(args.counts, int):
                    selected = primitives[:count]
                    fragment = (
                        edge_paths_to_svg_group(selected)
                        if args.primitive == "paths"
                        else edge_strokes_to_svg_group(selected)
                    )
                    candidate_content = inject_svg_before_close(
                        baseline_content, fragment
                    )
                    label = f"c{count}-l{length:g}-w{stroke_width:g}-a{opacity:g}"
                    candidate_path = (
                        scratch_path
                        if args.compact
                        else args.output_dir / f"{label}.svg"
                    )
                    candidate_path.write_text(candidate_content)
                    rendered, renderer = _try_rasterize_svg_to_linear_rgb(
                        str(candidate_path), width, height
                    )
                    if rendered is None:
                        continue
                    metrics = compute_quality_metrics(target, rendered)
                    ssim_gain = float(
                        metrics["ssim_srgb"] - baseline_metrics["ssim_srgb"]
                    )
                    psnr_regression = float(
                        baseline_metrics["psnr_srgb"] - metrics["psnr_srgb"]
                    )
                    accepted = bool(
                        ssim_gain >= args.min_ssim_gain
                        and psnr_regression <= args.max_psnr_regression
                    )
                    record = {
                        "label": label,
                        "path": None if args.compact else str(candidate_path),
                        "stroke_count": len(selected),
                        "primitive": args.primitive,
                        "length": length,
                        "width": stroke_width,
                        "opacity": opacity,
                        "bytes": candidate_path.stat().st_size,
                        "renderer": renderer,
                        "metrics": metrics,
                        "ssim_gain": ssim_gain,
                        "psnr_regression": psnr_regression,
                        "accepted": accepted,
                    }
                    records.append(record)
                    complexity_key = (
                        record["stroke_count"],
                        record["width"],
                        record["bytes"],
                        -record["ssim_gain"],
                    )
                    if accepted and (
                        winner is None
                        or complexity_key
                        < (
                            winner["stroke_count"],
                            winner["width"],
                            winner["bytes"],
                            -winner["ssim_gain"],
                        )
                    ):
                        winner = record
                        winner_content = candidate_content

    if args.compact and scratch_path.exists():
        scratch_path.unlink()
    if winner is not None and args.compact:
        winner_svg = args.output_dir / "winner-mixed.svg"
        if winner_content is None:
            raise RuntimeError("accepted a compact candidate without SVG content")
        winner_svg.write_text(winner_content)
        winner["path"] = str(winner_svg)

    result = {
        "source": str(args.source),
        "baseline_svg": str(args.baseline_svg),
        "size": [width, height],
        "baseline_renderer": baseline_renderer,
        "baseline_bytes": args.baseline_svg.stat().st_size,
        "baseline_metrics": baseline_metrics,
        "primitive": args.primitive,
        "candidate_count": len(records),
        "winner": winner,
        "decision": {
            "accepted": winner is not None,
            "reason": (
                f"mixed edge {args.primitive} cleared SVG gates"
                if winner is not None
                else f"no mixed edge-{args.primitive} candidate cleared SVG gates"
            ),
            "drawingml_required": winner is not None,
        },
        "candidates": records,
    }
    if (
        winner is not None
        and args.primitive == "paths"
        and args.baseline_pptx is not None
    ):
        winner_paths = propose_residual_edge_paths(
            target,
            baseline_render,
            max_paths=int(winner["stroke_count"]),
            path_length=float(winner["length"]),
            width=float(winner["width"]),
            opacity=float(winner["opacity"]),
        )[: int(winner["stroke_count"])]
        pptx_output = args.output_dir / "winner-mixed.pptx"
        segment_count = inject_edge_paths_into_pptx(
            args.baseline_pptx,
            pptx_output,
            winner_paths,
            width=width,
            height=height,
        )
        result["winner_pptx"] = {
            "path": str(pptx_output),
            "bytes": pptx_output.stat().st_size,
            "path_count": len(winner_paths),
            "native_segment_shapes": segment_count,
            "actual_powerpoint": None,
        }
    output = args.output_dir / "comparison.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "candidates"}, indent=2))
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
