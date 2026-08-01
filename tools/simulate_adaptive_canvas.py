#!/usr/bin/env python3
"""Replay adaptive Canvas policies over existing full-frame corpus evidence.

Two evidence sets are reported separately:

1. raw stage checkpoints rescored with the byte-exact deployed Canvas model;
2. final 2k/4k Chrome artifact pairs.

The second result is explicitly a retrospective oracle. It measures available
headroom and cost, but cannot predict an unseen higher-budget run.
"""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from splatthis.adaptive_compute import (
    CANVAS_RUNTIME_SCORER,
    AdaptiveComputePolicy,
    CanvasBudgetPoint,
    CanvasCheckpoint,
    ScalingPolicy,
    retrospective_scale_decision,
    simulate_adaptive_checkpoints,
)
from splatthis.artifact_gates import ArtifactGateCalibration
from splatthis.io import compute_quality_metrics, load_png, load_splats_json
from splatthis.renderer import render_pixel_runtime_numpy

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = REPO / "result" / "corpus"
DEFAULT_OUTPUT = REPO / "tmp" / "adaptive-canvas-simulation"
DEFAULT_GATES = REPO / "data" / "artifact-gates.json"


def _json_lines(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
        if isinstance(value, dict):
            records.append(value)
    return records


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _latest_canvas_budget_points(
    records: Sequence[Mapping[str, Any]],
    *,
    selected_images: Optional[set[str]],
) -> Dict[Tuple[str, int], CanvasBudgetPoint]:
    selected: Dict[Tuple[str, int], CanvasBudgetPoint] = {}
    for record in records:
        image = record.get("image")
        budget = record.get("splats_requested")
        if (
            record.get("format") not in {"canvas", "pixel-runtime"}
            or record.get("render_kind")
            not in {"canvas-pixel-buffer", "pixel-runtime-buffer"}
            or not isinstance(image, str)
            or not isinstance(budget, int)
            or (selected_images is not None and image not in selected_images)
        ):
            continue
        required = (
            "ssim_srgb",
            "lpips",
            "runtime_sec",
            "artifact_bytes",
            "splats_final",
        )
        if any(record.get(name) is None for name in required):
            continue
        selected[(image, budget)] = CanvasBudgetPoint(
            image=image,
            requested_budget=int(budget),
            ssim_srgb=float(record["ssim_srgb"]),
            lpips=float(record["lpips"]),
            runtime_sec=float(record["runtime_sec"]),
            artifact_bytes=int(record["artifact_bytes"]),
            final_splats=int(record["splats_final"]),
        )
    return selected


def _candidate_manifest_paths(
    records: Sequence[Mapping[str, Any]],
    root: Path,
    *,
    selected_images: Optional[set[str]],
) -> Dict[str, list[Path]]:
    paths: Dict[str, list[Path]] = {}
    for record in reversed(records):
        image = record.get("image")
        artifacts = record.get("artifacts_path")
        if (
            record.get("format") not in {"canvas", "pixel-runtime"}
            or not isinstance(image, str)
            or not isinstance(artifacts, str)
            or (selected_images is not None and image not in selected_images)
        ):
            continue
        path = root / artifacts / "run_manifest.json"
        if path.exists() and path not in paths.setdefault(image, []):
            paths[image].append(path)
    return paths


def _runtime_compositing_space(config: Mapping[str, Any]) -> str:
    training_target = str(config.get("training_export_target", "canvas"))
    if training_target in {"svg", "pptx-softedge"}:
        return "srgb"
    return str(config.get("compositing_space", "linear"))


def _load_deployed_checkpoints(
    path: Path,
    source_path: Path,
) -> Tuple[list[CanvasCheckpoint], list[dict[str, Any]]]:
    """Rescore canonical raw checkpoints at the exact Canvas byte boundary."""

    manifest = json.loads(path.read_text())
    config = manifest.get("config", {})
    resolved_size = config.get("resolved_target_size")
    target_size = (
        (int(resolved_size[0]), int(resolved_size[1]))
        if isinstance(resolved_size, list) and len(resolved_size) == 2
        else None
    )
    target = load_png(str(source_path), target_size=target_size)[..., :3]
    height, width = target.shape[:2]
    background = np.asarray(
        config.get("background_linear_rgb", [0.0, 0.0, 0.0]),
        dtype=np.float32,
    )
    compositing_space = _runtime_compositing_space(config)
    checkpoints: list[CanvasCheckpoint] = []
    comparisons: list[dict[str, Any]] = []
    for stage in manifest.get("stages", []):
        if stage.get("elapsed_sec") is None:
            continue
        stage_type = stage.get("stage_type")
        if stage_type == "residual_detail":
            residual_pass = int(stage.get("residual_pass", 1))
            raw_label = f"residual-{residual_pass}"
            label = "residual-final"
        elif stage_type is None:
            stage_index = int(stage.get("stage", 0))
            if stage_index <= 0:
                continue
            raw_label = f"iter-{stage_index}"
            label = f"stage-{stage_index}"
        else:
            continue
        raw_path = path.parent / f"{raw_label}.raw.json"
        if not raw_path.exists():
            continue
        splats = load_splats_json(str(raw_path))
        rendered = render_pixel_runtime_numpy(
            splats,
            width=width,
            height=height,
            background_linear_rgb=background,
            compositing_space=compositing_space,
        )
        quality = compute_quality_metrics(target, rendered)
        checkpoints.append(
            CanvasCheckpoint(
                label=label,
                ssim_srgb=float(quality["ssim_srgb"]),
                psnr_srgb=float(quality["psnr_srgb"]),
                splat_count=len(splats),
                elapsed_sec=float(stage["elapsed_sec"]),
            )
        )
        historical_ssim = stage.get("deployed_ssim_srgb")
        historical_psnr = stage.get("deployed_psnr_srgb")
        comparisons.append(
            {
                "label": label,
                "raw_artifact": str(raw_path),
                "splat_count": len(splats),
                "ssim_srgb": float(quality["ssim_srgb"]),
                "psnr_srgb": float(quality["psnr_srgb"]),
                "historical_continuous_ssim_srgb": (
                    None if historical_ssim is None else float(historical_ssim)
                ),
                "historical_continuous_psnr_srgb": (
                    None if historical_psnr is None else float(historical_psnr)
                ),
                "historical_minus_exact_ssim": (
                    None
                    if historical_ssim is None
                    else float(historical_ssim) - float(quality["ssim_srgb"])
                ),
            }
        )
    return checkpoints, comparisons


def _latest_checkpoint_curves(
    records: Sequence[Mapping[str, Any]],
    root: Path,
    image_meta: Mapping[str, Mapping[str, Any]],
    *,
    selected_images: Optional[set[str]],
) -> Dict[
    str,
    Tuple[Path, list[CanvasCheckpoint], list[dict[str, Any]]],
]:
    curves = {}
    for image, paths in _candidate_manifest_paths(
        records, root, selected_images=selected_images
    ).items():
        for path in paths:
            source_path = root / str(image_meta[image]["path"])
            checkpoints, comparisons = _load_deployed_checkpoints(path, source_path)
            if len(checkpoints) >= 2:
                curves[image] = (path, checkpoints, comparisons)
                break
    return curves


def _load_calibration(path: Path) -> Optional[ArtifactGateCalibration]:
    if not path.exists():
        return None
    return ArtifactGateCalibration.from_dict(json.loads(path.read_text()))


def _median(values: Iterable[float]) -> float:
    materialized = list(values)
    return float(statistics.median(materialized)) if materialized else 0.0


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO))
    except ValueError:
        return str(resolved)


def _target_values(primary: float, comparisons: Optional[str]) -> list[float]:
    values = [float(primary)]
    if comparisons:
        values.extend(
            float(part.strip()) for part in comparisons.split(",") if part.strip()
        )
    unique: list[float] = []
    for value in values:
        if not 0.0 <= value <= 1.0:
            raise ValueError("checkpoint SSIM targets must be between 0 and 1")
        if value not in unique:
            unique.append(value)
    return unique


def _summarize_stage_replay(
    curves: Mapping[
        str,
        Tuple[Path, list[CanvasCheckpoint], list[dict[str, Any]]],
    ],
    policy: AdaptiveComputePolicy,
    *,
    expected_image_count: int,
    minimum_useful_saving_fraction: float,
) -> dict[str, Any]:
    stage_results = []
    for image, (manifest_path, checkpoints, comparisons) in sorted(curves.items()):
        result = simulate_adaptive_checkpoints(checkpoints, policy)
        stage_results.append(
            {
                "image": image,
                "manifest": _display_path(manifest_path),
                "rescored_checkpoints": comparisons,
                **result.as_dict(),
            }
        )
    observed_stage_sec = sum(
        float(result["observed_stage_sec"]) for result in stage_results
    )
    full_stage_sec = sum(float(result["full_stage_sec"]) for result in stage_results)
    saved_stage_sec = max(0.0, full_stage_sec - observed_stage_sec)
    saved_fraction = saved_stage_sec / full_stage_sec if full_stage_sec else 0.0
    ssim_costs = [float(result["ssim_opportunity_cost"]) for result in stage_results]
    psnr_costs = [float(result["psnr_opportunity_cost"]) for result in stage_results]
    compute_gate_met = saved_fraction >= minimum_useful_saving_fraction
    return {
        "target_ssim_srgb": policy.target_ssim_srgb,
        "mode": "online-hard-target-observed-only",
        "uses_plateau_stop": False,
        "uses_regression_stop": False,
        "curve_count": len(stage_results),
        "missing_curve_count": expected_image_count - len(stage_results),
        "checkpoints_rescored": sum(
            len(result["rescored_checkpoints"]) for result in stage_results
        ),
        "early_stop_count": sum(
            int(result["checkpoints_observed"]) < int(result["checkpoints_available"])
            for result in stage_results
        ),
        "observed_stage_sec": observed_stage_sec,
        "full_stage_sec": full_stage_sec,
        "saved_stage_sec": saved_stage_sec,
        "saved_stage_fraction": saved_fraction,
        "median_ssim_opportunity_cost": _median(ssim_costs),
        "mean_ssim_opportunity_cost": (
            sum(ssim_costs) / len(ssim_costs) if ssim_costs else 0.0
        ),
        "max_ssim_opportunity_cost": max(ssim_costs, default=0.0),
        "median_psnr_opportunity_cost": _median(psnr_costs),
        "max_psnr_opportunity_cost": max(psnr_costs, default=0.0),
        "minimum_useful_saving_fraction": minimum_useful_saving_fraction,
        "compute_gate_met": compute_gate_met,
        "go_no_go": ("continue-to-fresh-ab" if compute_gate_met else "do-not-expand"),
        "results": stage_results,
    }


def _report_markdown(summary: Mapping[str, Any]) -> str:
    stage_replays = summary["stage_replays"]
    stage = stage_replays[0]
    scale = summary["budget_scaling_oracle"]
    lines = [
        "# Adaptive Canvas simulation",
        "",
        "This is a replay over observed checkpoints and artifacts. It is not yet",
        "an online predictor.",
        "",
        "## Byte-exact online hard-target replay",
        "",
        "Every raw checkpoint is rescored at the emitted 8-bit Canvas boundary.",
        "The replay stops only on the explicit target, like the online controller.",
        "",
        "| Target | Early stops | Saved stage time | Saving | Max SSIM cost | Verdict |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for replay in stage_replays:
        lines.append(
            f"| {replay['target_ssim_srgb']:.6f} | "
            f"{replay['early_stop_count']}/{replay['curve_count']} | "
            f"{replay['saved_stage_sec']:.1f} s | "
            f"{replay['saved_stage_fraction']:.1%} | "
            f"{replay['max_ssim_opportunity_cost']:.6f} | "
            f"{replay['go_no_go']} |"
        )
    lines.extend(
        [
            "",
            f"### Primary target {stage['target_ssim_srgb']:.6f}",
            "",
            "| Image | Stop | Selected | Seen | Saved stage time | SSIM cost |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for result in stage["results"]:
        lines.append(
            f"| {result['image']} | {result['stop_reason']} | "
            f"{result['selected']['label']} | "
            f"{result['checkpoints_observed']}/"
            f"{result['checkpoints_available']} | "
            f"{result['saved_stage_sec']:.1f} s | "
            f"{result['ssim_opportunity_cost']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## 2k to 4k retrospective oracle",
            "",
            "The scale decisions below use the already-known 4k result. They",
            "measure whether a policy envelope is plausible; they must not be",
            "used as claims about prediction accuracy.",
            "",
            f"- Paired images: {scale['pair_count']}",
            f"- Would scale: {scale['scale_count']}",
            f"- Oracle runtime versus fixed 4k: "
            f"{scale['selected_runtime_sec']:.1f} s versus "
            f"{scale['fixed_high_runtime_sec']:.1f} s",
            f"- Runtime saving: {scale['saved_runtime_fraction']:.1%}",
            f"- Median SSIM opportunity cost: "
            f"{scale['median_ssim_opportunity_cost']:.6f}",
            "",
            "| Image | Decision | Reason | 2k→4k SSIM | LPIPS gain | Extra time |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for decision in scale["decisions"]:
        lines.append(
            f"| {decision['image']} | "
            f"{'scale' if decision['scale'] else 'stop'} | "
            f"{decision['reason']} | {decision['ssim_gain']:+.5f} | "
            f"{decision['lpips_gain']:+.5f} | "
            f"{decision['runtime_delta_sec']:+.1f} s |"
        )
    lines.append("")
    return "\n".join(lines)


def _parse_csv(value: Optional[str]) -> Optional[set[str]]:
    if value is None:
        return None
    return {part.strip() for part in value.split(",") if part.strip()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--artifact-gates", type=Path, default=DEFAULT_GATES)
    parser.add_argument("--only", help="comma-separated corpus image names")
    parser.add_argument("--target-ssim", type=float, default=0.98)
    parser.add_argument(
        "--compare-target-ssim",
        default="0.979",
        help="comma-separated additional online hard targets (default: 0.979)",
    )
    parser.add_argument(
        "--minimum-useful-saving-fraction",
        type=float,
        default=0.05,
        help="compute go/no-go floor for expanding to fresh A/B runs (default: 0.05)",
    )
    parser.add_argument("--scale-target-ssim", type=float, default=0.95)
    parser.add_argument("--scale-target-lpips", type=float, default=0.15)
    parser.add_argument("--scale-min-ssim-gain", type=float, default=0.005)
    parser.add_argument("--scale-min-lpips-gain", type=float, default=0.001)
    args = parser.parse_args()
    if not 0.0 <= args.minimum_useful_saving_fraction <= 1.0:
        parser.error("--minimum-useful-saving-fraction must be between 0 and 1")
    try:
        target_values = _target_values(args.target_ssim, args.compare_target_ssim)
    except ValueError as exc:
        parser.error(str(exc))

    root = args.corpus_root.resolve()
    output_dir = args.output_dir.resolve()
    selected_images = _parse_csv(args.only)
    corpus = json.loads((root / "corpus.json").read_text())
    image_meta = corpus["images"]
    corpus_images = set(image_meta)
    if selected_images is not None:
        unknown = sorted(selected_images - corpus_images)
        if unknown:
            parser.error(f"unknown corpus images: {', '.join(unknown)}")
        expected_images = selected_images
    else:
        expected_images = corpus_images

    records = _json_lines(root / "results.jsonl")
    calibration = _load_calibration(args.artifact_gates.resolve())
    ssim_noise = (
        calibration.recommended_delta("canvas", "ssim_srgb")
        if calibration is not None
        else 0.0
    )
    psnr_noise = (
        calibration.recommended_delta("canvas", "psnr_srgb")
        if calibration is not None
        else 0.0
    )
    lpips_noise = (
        calibration.recommended_delta("canvas", "lpips")
        if calibration is not None
        else 0.0
    )

    curves = _latest_checkpoint_curves(
        records,
        root,
        image_meta,
        selected_images=selected_images,
    )
    checkpoint_policies = [
        AdaptiveComputePolicy(
            target_ssim_srgb=target,
            checkpoint_min_ssim_gain=max(0.0005, ssim_noise),
            max_ssim_regression=max(0.0005, ssim_noise),
            max_psnr_regression=max(0.10, psnr_noise),
            plateau_min_ssim_gain=0.0,
            plateau_min_psnr_gain=0.0,
            stop_on_regression=False,
            stop_on_plateau=False,
        )
        for target in target_values
    ]
    stage_replays = [
        _summarize_stage_replay(
            curves,
            policy,
            expected_image_count=len(expected_images),
            minimum_useful_saving_fraction=args.minimum_useful_saving_fraction,
        )
        for policy in checkpoint_policies
    ]

    historical_deltas = [
        float(item["historical_minus_exact_ssim"])
        for _, _, comparisons in curves.values()
        for item in comparisons
        if item["historical_minus_exact_ssim"] is not None
    ]

    scaling_policy = ScalingPolicy(
        target_ssim_srgb=args.scale_target_ssim,
        target_lpips=args.scale_target_lpips,
        min_ssim_gain=max(args.scale_min_ssim_gain, ssim_noise),
        min_lpips_gain=max(args.scale_min_lpips_gain, lpips_noise),
    )
    budget_points = _latest_canvas_budget_points(
        records, selected_images=selected_images
    )
    scaling_decisions = []
    selected_runtime = 0.0
    fixed_high_runtime = 0.0
    ssim_costs = []
    for image in sorted(expected_images):
        lower = budget_points.get((image, 2000))
        higher = budget_points.get((image, 4000))
        if lower is None or higher is None:
            continue
        decision = retrospective_scale_decision(lower, higher, scaling_policy)
        selected_runtime += (
            higher.runtime_sec if decision["scale"] else lower.runtime_sec
        )
        fixed_high_runtime += higher.runtime_sec
        selected_ssim = higher.ssim_srgb if decision["scale"] else lower.ssim_srgb
        opportunity_cost = max(0.0, higher.ssim_srgb - selected_ssim)
        decision["ssim_opportunity_cost"] = opportunity_cost
        ssim_costs.append(opportunity_cost)
        scaling_decisions.append(decision)

    summary = {
        "schema": "splatthis.adaptive-canvas-simulation/2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "evidence": {
            "stage_replay": "raw checkpoints rescored with byte-exact Canvas model",
            "stage_runtime_scorer": CANVAS_RUNTIME_SCORER,
            "budget_scaling": "Chrome canvas.toDataURL deployed artifacts",
            "predictive_claim": False,
        },
        "selection": sorted(expected_images),
        "artifact_gate_calibration": (
            str(args.artifact_gates.resolve()) if calibration is not None else None
        ),
        "effective_noise_floors": {
            "ssim_srgb": ssim_noise,
            "psnr_srgb": psnr_noise,
            "lpips": lpips_noise,
        },
        "checkpoint_policies": [
            dict(policy.__dict__) for policy in checkpoint_policies
        ],
        "scaling_policy": dict(scaling_policy.__dict__),
        "checkpoint_rescoring": {
            "runtime_scorer": CANVAS_RUNTIME_SCORER,
            "checkpoint_count": sum(len(value[1]) for value in curves.values()),
            "historical_score_count": len(historical_deltas),
            "historical_minus_exact_ssim": {
                "min": min(historical_deltas, default=0.0),
                "median": _median(historical_deltas),
                "max": max(historical_deltas, default=0.0),
            },
        },
        "stage_replay": stage_replays[0],
        "stage_replays": stage_replays,
        "budget_scaling_oracle": {
            "pair_count": len(scaling_decisions),
            "missing_pair_count": len(expected_images) - len(scaling_decisions),
            "scale_count": sum(bool(item["scale"]) for item in scaling_decisions),
            "selected_runtime_sec": selected_runtime,
            "fixed_high_runtime_sec": fixed_high_runtime,
            "saved_runtime_sec": max(0.0, fixed_high_runtime - selected_runtime),
            "saved_runtime_fraction": (
                max(0.0, fixed_high_runtime - selected_runtime) / fixed_high_runtime
                if fixed_high_runtime
                else 0.0
            ),
            "median_ssim_opportunity_cost": _median(ssim_costs),
            "mean_ssim_opportunity_cost": (
                sum(ssim_costs) / len(ssim_costs) if ssim_costs else 0.0
            ),
            "max_ssim_opportunity_cost": max(ssim_costs, default=0.0),
            "images_with_ssim_opportunity_cost": sum(cost > 0.0 for cost in ssim_costs),
            "decisions": scaling_decisions,
        },
    }
    _write_json(output_dir / "summary.json", summary)
    (output_dir / "report.md").write_text(_report_markdown(summary))
    print(f"wrote {output_dir / 'summary.json'}")
    print(f"wrote {output_dir / 'report.md'}")
    print(
        f"{len(curves)} exact-rescored stage curves; "
        f"{len(scaling_decisions)} paired 2k/4k artifacts"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
