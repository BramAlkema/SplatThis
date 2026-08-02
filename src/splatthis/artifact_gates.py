"""Shared deployed-artifact noise calibration and gate helpers.

The calibration records how much an unchanged artifact's measured metrics move
when the same target renderer captures it repeatedly. Algorithm-specific gates
remain free to be stricter, but they must never treat a delta below this
measurement floor as a proven gain.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

# The calibrated browser framebuffer belongs to the historical ImageData
# software renderer, now named ``pixel-runtime``. CSS was calibrated August
# 2026 against the corpus-gallery builds emitted from the measured
# populations; native Canvas still needs its own repeat-render calibration
# before it can be added here.
ARTIFACT_TARGETS = ("pixel-runtime", "svg", "css", "pptx")

# True means higher values are better.
QUALITY_METRIC_DIRECTIONS: Dict[str, bool] = {
    "ssim_srgb": True,
    "ms_ssim_luma": True,
    "psnr_srgb": True,
    "lpips": False,
    "delta_e_ok_mean": False,
    "delta_e_ok_p95": False,
    "edge_chamfer": False,
    "edge_gradient_l1": False,
    "worst_roi_error": False,
}

ObservationGroups = Dict[Tuple[str, str], list[Mapping[str, Any]]]


@dataclass(frozen=True)
class MetricNoiseEstimate:
    """Repeat-render spread for one target and metric."""

    metric: str
    higher_is_better: bool
    artifact_count: int
    median_span: float
    p95_span: float
    max_span: float
    recommended_min_delta: float

    def as_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MetricNoiseEstimate":
        return cls(
            metric=str(data["metric"]),
            higher_is_better=bool(data["higher_is_better"]),
            artifact_count=int(data["artifact_count"]),
            median_span=float(data["median_span"]),
            p95_span=float(data["p95_span"]),
            max_span=float(data["max_span"]),
            recommended_min_delta=float(data["recommended_min_delta"]),
        )


@dataclass(frozen=True)
class TargetGateCalibration:
    """Calibration completeness and metric floors for one deployed target."""

    target: str
    required_repeats: int
    artifact_count: int
    complete_artifact_count: int
    observation_count: int
    complete: bool
    metrics: Mapping[str, MetricNoiseEstimate]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "target": self.target,
            "required_repeats": self.required_repeats,
            "artifact_count": self.artifact_count,
            "complete_artifact_count": self.complete_artifact_count,
            "observation_count": self.observation_count,
            "complete": self.complete,
            "metrics": {
                name: estimate.as_dict()
                for name, estimate in sorted(self.metrics.items())
            },
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TargetGateCalibration":
        raw_metrics = data.get("metrics", {})
        if not isinstance(raw_metrics, Mapping):
            raise ValueError("target calibration metrics must be a mapping")
        return cls(
            target=str(data["target"]),
            required_repeats=int(data["required_repeats"]),
            artifact_count=int(data["artifact_count"]),
            complete_artifact_count=int(data["complete_artifact_count"]),
            observation_count=int(data["observation_count"]),
            complete=bool(data["complete"]),
            metrics={
                str(name): MetricNoiseEstimate.from_dict(value)
                for name, value in raw_metrics.items()
                if isinstance(value, Mapping)
            },
        )


@dataclass(frozen=True)
class ArtifactGateCalibration:
    """Versioned target-specific measurement floors."""

    required_repeats: int
    noise_multiplier: float
    targets: Mapping[str, TargetGateCalibration]
    schema: str = "splatthis.artifact-gates/1"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "required_repeats": self.required_repeats,
            "noise_multiplier": self.noise_multiplier,
            "targets": {
                target: calibration.as_dict()
                for target, calibration in sorted(self.targets.items())
            },
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ArtifactGateCalibration":
        schema = str(data.get("schema", ""))
        if schema != "splatthis.artifact-gates/1":
            raise ValueError(f"unsupported artifact gate schema: {schema!r}")
        raw_targets = data.get("targets", {})
        if not isinstance(raw_targets, Mapping):
            raise ValueError("artifact gate targets must be a mapping")
        return cls(
            schema=schema,
            required_repeats=int(data["required_repeats"]),
            noise_multiplier=float(data["noise_multiplier"]),
            targets={
                str(target): TargetGateCalibration.from_dict(value)
                for target, value in raw_targets.items()
                if isinstance(value, Mapping)
            },
        )

    def recommended_delta(self, target: str, metric: str) -> float:
        target_calibration = self.targets.get(target)
        if target_calibration is None:
            return 0.0
        estimate = target_calibration.metrics.get(metric)
        return 0.0 if estimate is None else float(estimate.recommended_min_delta)

    def effective_delta(
        self, target: str, metric: str, policy_delta: float = 0.0
    ) -> float:
        """Return the stricter of policy intent and measured renderer noise."""

        return max(
            0.0,
            float(policy_delta),
            self.recommended_delta(target, metric),
        )

    def meaningful_gain(
        self,
        *,
        target: str,
        metric: str,
        incumbent: float,
        candidate: float,
        policy_delta: float = 0.0,
    ) -> bool:
        delta = metric_gain(metric, incumbent=incumbent, candidate=candidate)
        return bool(
            delta > 0.0 and delta >= self.effective_delta(target, metric, policy_delta)
        )

    def protects_baseline(
        self,
        *,
        target: str,
        metric: str,
        baseline: float,
        candidate: float,
        policy_tolerance: float = 0.0,
    ) -> bool:
        regression = -metric_gain(metric, incumbent=baseline, candidate=candidate)
        return bool(
            regression <= self.effective_delta(target, metric, policy_tolerance)
        )


def metric_gain(metric: str, *, incumbent: float, candidate: float) -> float:
    """Signed improvement for a known metric; positive is always better."""

    try:
        higher_is_better = QUALITY_METRIC_DIRECTIONS[metric]
    except KeyError as exc:
        raise ValueError(f"unknown quality metric: {metric!r}") from exc
    if higher_is_better:
        return float(candidate) - float(incumbent)
    return float(incumbent) - float(candidate)


def calibrate_artifact_observations(
    observations: Sequence[Mapping[str, Any]],
    *,
    required_repeats: int = 5,
    noise_multiplier: float = 2.0,
    expected_targets: Iterable[str] = ARTIFACT_TARGETS,
    expected_artifacts: Optional[Mapping[str, Sequence[str]]] = None,
) -> ArtifactGateCalibration:
    """Derive conservative metric floors from repeated artifact observations.

    Each observation must contain ``target``, ``artifact_id``, ``repeat``, and
    a ``metrics`` mapping. Spans are computed within one unchanged artifact;
    differences between source images or algorithms are never counted as
    renderer noise.
    """

    if required_repeats < 2:
        raise ValueError("required_repeats must be at least 2")
    if noise_multiplier < 1.0:
        raise ValueError("noise_multiplier must be at least 1")

    grouped = _group_observations(observations)
    target_names = tuple(dict.fromkeys(str(target) for target in expected_targets))
    target_results = {
        target: _calibrate_target(
            target=target,
            grouped=grouped,
            required_repeats=required_repeats,
            noise_multiplier=noise_multiplier,
            expected_ids=(
                tuple(dict.fromkeys(expected_artifacts.get(target, ())))
                if expected_artifacts is not None
                else ()
            ),
        )
        for target in target_names
    }

    return ArtifactGateCalibration(
        required_repeats=required_repeats,
        noise_multiplier=float(noise_multiplier),
        targets=target_results,
    )


def _group_observations(
    observations: Sequence[Mapping[str, Any]],
) -> ObservationGroups:
    grouped: ObservationGroups = {}
    for observation in observations:
        target = str(observation.get("target", ""))
        if target not in ARTIFACT_TARGETS:
            raise ValueError(f"unsupported artifact target: {target!r}")
        artifact_id = str(observation.get("artifact_id", ""))
        if not artifact_id:
            raise ValueError("artifact observation is missing artifact_id")
        repeat = int(observation.get("repeat", -1))
        if repeat < 0:
            raise ValueError("artifact observation repeat must be non-negative")
        metrics = observation.get("metrics")
        if not isinstance(metrics, Mapping):
            raise ValueError("artifact observation metrics must be a mapping")
        grouped.setdefault((target, artifact_id), []).append(metrics)
    return grouped


def _calibrate_target(
    *,
    target: str,
    grouped: ObservationGroups,
    required_repeats: int,
    noise_multiplier: float,
    expected_ids: Sequence[str],
) -> TargetGateCalibration:
    if expected_ids:
        artifact_groups = [
            grouped.get((target, artifact_id), []) for artifact_id in expected_ids
        ]
    else:
        artifact_groups = [
            repeats
            for (group_target, _), repeats in grouped.items()
            if group_target == target
        ]
    complete_count = sum(
        len(repeats) >= required_repeats for repeats in artifact_groups
    )
    estimates = {
        metric: estimate
        for metric, higher_is_better in QUALITY_METRIC_DIRECTIONS.items()
        if (
            estimate := _estimate_metric_noise(
                metric=metric,
                higher_is_better=higher_is_better,
                artifact_groups=artifact_groups,
                noise_multiplier=noise_multiplier,
            )
        )
        is not None
    }
    return TargetGateCalibration(
        target=target,
        required_repeats=required_repeats,
        artifact_count=len(artifact_groups),
        complete_artifact_count=complete_count,
        observation_count=sum(len(repeats) for repeats in artifact_groups),
        complete=bool(artifact_groups and complete_count == len(artifact_groups)),
        metrics=estimates,
    )


def _estimate_metric_noise(
    *,
    metric: str,
    higher_is_better: bool,
    artifact_groups: Sequence[Sequence[Mapping[str, Any]]],
    noise_multiplier: float,
) -> Optional[MetricNoiseEstimate]:
    spans = []
    for repeats in artifact_groups:
        values = [
            float(metrics[metric])
            for metrics in repeats
            if _is_finite_number(metrics.get(metric))
        ]
        if len(values) >= 2:
            spans.append(max(values) - min(values))
    if not spans:
        return None
    span_array = np.asarray(spans)
    p95_span = float(np.percentile(span_array, 95))
    max_span = float(max(spans))
    return MetricNoiseEstimate(
        metric=metric,
        higher_is_better=higher_is_better,
        artifact_count=len(spans),
        median_span=float(np.median(span_array)),
        p95_span=p95_span,
        max_span=max_span,
        recommended_min_delta=float(max(noise_multiplier * p95_span, max_span)),
    )


def _is_finite_number(value: Any) -> bool:
    if value is None or isinstance(value, bool):
        return False
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def load_artifact_gate_calibration(
    data: Optional[Mapping[str, Any]],
) -> Optional[ArtifactGateCalibration]:
    """Load a calibration mapping while allowing an explicitly absent file."""

    if data is None:
        return None
    return ArtifactGateCalibration.from_dict(data)
