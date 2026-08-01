"""Deterministic accept-or-revert policy for target-specific SVG recipes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional, Sequence

HIGHER_IS_BETTER = ("ssim_srgb", "ms_ssim_luma", "psnr_srgb")
LOWER_IS_BETTER = (
    "lpips",
    "delta_e_ok_mean",
    "delta_e_ok_p95",
    "edge_chamfer",
    "edge_gradient_l1",
    "worst_roi_error",
)


@dataclass(frozen=True)
class SvgRecipeGatePolicy:
    """Predeclared candidate and corpus gates for the bounded recipe MVP."""

    max_size_growth_fraction: float = 0.10
    max_render_time_growth_fraction: float = 0.50
    max_ssim_regression: float = 0.002
    max_ms_ssim_regression: float = 0.003
    max_lpips_regression: float = 0.005
    max_delta_e_p95_regression: float = 0.005
    max_edge_chamfer_regression: float = 0.5
    max_edge_gradient_regression: float = 0.01
    max_worst_roi_regression_fraction: float = 0.03
    min_ssim_gain: float = 0.002
    min_ms_ssim_gain: float = 0.002
    min_lpips_gain: float = 0.005
    min_delta_e_p95_gain: float = 0.005
    min_edge_gradient_gain: float = 0.005
    minimum_accepted_images: int = 5
    maximum_median_size_growth_fraction: float = 0.10
    maximum_median_render_time_growth_fraction: float = 0.20

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _number(metrics: Mapping[str, Any], key: str) -> Optional[float]:
    value = metrics.get(key)
    if value is None:
        return None
    return float(value)


def metric_deltas(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Optional[float]]:
    """Return candidate-minus-baseline deltas for every guarded metric."""

    result: dict[str, Optional[float]] = {}
    for key in (*HIGHER_IS_BETTER, *LOWER_IS_BETTER):
        before = _number(baseline, key)
        after = _number(candidate, key)
        result[key] = None if before is None or after is None else after - before
    before_bytes = _number(baseline, "file_size_bytes")
    after_bytes = _number(candidate, "file_size_bytes")
    result["file_size_bytes"] = (
        None
        if before_bytes is None or after_bytes is None
        else after_bytes - before_bytes
    )
    before_render = _number(baseline, "render_time_sec")
    after_render = _number(candidate, "render_time_sec")
    result["render_time_sec"] = (
        None
        if before_render is None or after_render is None
        else after_render - before_render
    )
    return result


def balanced_quality_score(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> float:
    """Rank already-guarded candidates without letting one metric dominate."""

    delta = metric_deltas(baseline, candidate)

    def value(key: str) -> float:
        item = delta.get(key)
        return 0.0 if item is None else float(item)

    return float(
        1.50 * value("ssim_srgb")
        + 1.00 * value("ms_ssim_luma")
        - 0.70 * value("lpips")
        - 0.15 * value("delta_e_ok_p95")
        - 0.02 * value("edge_chamfer")
        - 0.30 * value("edge_gradient_l1")
        - 0.25 * value("worst_roi_error")
    )


def evaluate_recipe_candidate(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    policy: SvgRecipeGatePolicy,
) -> dict[str, Any]:
    """Apply hard guards, then require a meaningful balanced quality gain."""

    failures: list[str] = []
    baseline_bytes = float(baseline["file_size_bytes"])
    candidate_bytes = float(candidate["file_size_bytes"])
    if candidate_bytes > baseline_bytes * (1.0 + policy.max_size_growth_fraction):
        failures.append("file-size")
    baseline_render = float(baseline["render_time_sec"])
    candidate_render = float(candidate["render_time_sec"])
    if candidate_render > baseline_render * (
        1.0 + policy.max_render_time_growth_fraction
    ):
        failures.append("render-time")

    comparisons = (
        (
            "ssim",
            "ssim_srgb",
            -policy.max_ssim_regression,
            "minimum",
        ),
        (
            "ms-ssim",
            "ms_ssim_luma",
            -policy.max_ms_ssim_regression,
            "minimum",
        ),
        (
            "lpips",
            "lpips",
            policy.max_lpips_regression,
            "maximum",
        ),
        (
            "delta-e-p95",
            "delta_e_ok_p95",
            policy.max_delta_e_p95_regression,
            "maximum",
        ),
        (
            "edge-chamfer",
            "edge_chamfer",
            policy.max_edge_chamfer_regression,
            "maximum",
        ),
        (
            "edge-gradient",
            "edge_gradient_l1",
            policy.max_edge_gradient_regression,
            "maximum",
        ),
    )
    deltas = metric_deltas(baseline, candidate)
    for label, key, threshold, direction in comparisons:
        delta = deltas[key]
        if delta is None:
            continue
        if direction == "minimum" and delta < threshold:
            failures.append(label)
        if direction == "maximum" and delta > threshold:
            failures.append(label)

    baseline_roi = float(baseline["worst_roi_error"])
    candidate_roi = float(candidate["worst_roi_error"])
    if candidate_roi > baseline_roi * (1.0 + policy.max_worst_roi_regression_fraction):
        failures.append("worst-roi")

    meaningful_gains = {
        "ssim": (deltas["ssim_srgb"] or 0.0) >= policy.min_ssim_gain,
        "ms-ssim": (deltas["ms_ssim_luma"] or 0.0) >= policy.min_ms_ssim_gain,
        "lpips": -(deltas["lpips"] or 0.0) >= policy.min_lpips_gain,
        "delta-e-p95": -(deltas["delta_e_ok_p95"] or 0.0)
        >= policy.min_delta_e_p95_gain,
        "edge-gradient": -(deltas["edge_gradient_l1"] or 0.0)
        >= policy.min_edge_gradient_gain,
    }
    score = balanced_quality_score(baseline, candidate)
    if not any(meaningful_gains.values()):
        failures.append("gain-below-threshold")
    if score <= 0.0:
        failures.append("non-positive-balanced-score")

    return {
        "accepted": not failures,
        "failures": failures,
        "meaningful_gains": meaningful_gains,
        "balanced_quality_score": score,
        "deltas": deltas,
    }


def select_recipe_candidate(
    baseline: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    policy: SvgRecipeGatePolicy,
) -> dict[str, Any]:
    """Select the strongest passing recipe or return the standard baseline."""

    decisions = []
    accepted = []
    for candidate in candidates:
        decision = {
            "recipe": str(candidate["recipe"]),
            **evaluate_recipe_candidate(baseline, candidate, policy),
        }
        decisions.append(decision)
        if decision["accepted"]:
            accepted.append((candidate, decision))

    if not accepted:
        return {
            "selected_recipe": str(baseline["recipe"]),
            "accepted_candidate": False,
            "selected": dict(baseline),
            "decisions": decisions,
        }

    winner, _ = max(
        accepted,
        key=lambda item: (
            float(item[1]["balanced_quality_score"]),
            -int(item[0]["file_size_bytes"]),
            str(item[0]["recipe"]),
        ),
    )
    return {
        "selected_recipe": str(winner["recipe"]),
        "accepted_candidate": True,
        "selected": dict(winner),
        "decisions": decisions,
    }
