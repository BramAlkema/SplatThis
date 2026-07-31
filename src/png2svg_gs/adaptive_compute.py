"""Canvas adaptive-compute policies.

The online controller stops only on an absolute target observed in the current
run. The retrospective helpers can measure whether broader stopping or scaling
rules would have saved work, but do not predict an unseen 4k or 8k result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

# Full-frame model-to-Chrome calibration, data/canvas-checkpoint-parity.json.
DEFAULT_CHROME_SSIM_SAFETY_MARGIN = 0.0
DEFAULT_CHROME_PSNR_SAFETY_MARGIN = 0.0
CANVAS_RUNTIME_SCORER = "canvas-image-data-byte-v1"
CANVAS_RUNTIME_CALIBRATION_CHECKPOINTS = 48


@dataclass(frozen=True)
class CanvasCheckpoint:
    label: str
    ssim_srgb: float
    psnr_srgb: float
    splat_count: int
    elapsed_sec: float


@dataclass(frozen=True)
class AdaptiveComputePolicy:
    """Policy parameters for replaying an observed checkpoint curve."""

    min_checkpoints: int = 2
    target_ssim_srgb: Optional[float] = 0.98
    target_psnr_srgb: Optional[float] = None
    checkpoint_min_ssim_gain: float = 0.0005
    max_ssim_regression: float = 0.0005
    max_psnr_regression: float = 0.10
    plateau_min_ssim_gain: float = 0.002
    plateau_min_psnr_gain: float = 0.10
    min_ssim_gain_per_second: float = 0.0
    stop_on_regression: bool = True
    stop_on_plateau: bool = True

    def __post_init__(self) -> None:
        if self.min_checkpoints < 1:
            raise ValueError("min_checkpoints must be positive")
        for name in (
            "checkpoint_min_ssim_gain",
            "max_ssim_regression",
            "max_psnr_regression",
            "plateau_min_ssim_gain",
            "plateau_min_psnr_gain",
            "min_ssim_gain_per_second",
        ):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True)
class OnlineAdaptiveConfig:
    """Conservative online policy that only stops on an absolute quality target.

    Plateau, regression, and higher-budget prediction remain retrospective
    experiments. The online controller sees only completed pixel-runtime checkpoints
    and cannot use a future result to justify a stop.
    """

    enabled: bool = False
    min_checkpoints: int = 2
    target_ssim_srgb: Optional[float] = 0.98
    target_psnr_srgb: Optional[float] = None
    chrome_ssim_safety_margin: float = DEFAULT_CHROME_SSIM_SAFETY_MARGIN
    chrome_psnr_safety_margin: float = DEFAULT_CHROME_PSNR_SAFETY_MARGIN
    checkpoint_min_ssim_gain: float = 0.0005
    max_ssim_regression: float = 0.0005
    max_psnr_regression: float = 0.10

    def __post_init__(self) -> None:
        if self.min_checkpoints < 1:
            raise ValueError("adaptive_compute_min_checkpoints must be positive")
        if self.target_ssim_srgb is not None and not (
            0.0 <= self.target_ssim_srgb <= 1.0
        ):
            raise ValueError(
                "adaptive_compute_target_ssim_srgb must be between 0 and 1"
            )
        if self.target_psnr_srgb is not None and self.target_psnr_srgb < 0.0:
            raise ValueError("adaptive_compute_target_psnr_srgb must be non-negative")
        if self.chrome_ssim_safety_margin < 0.0:
            raise ValueError("adaptive_compute_chrome_ssim_margin must be non-negative")
        if self.chrome_psnr_safety_margin < 0.0:
            raise ValueError("adaptive_compute_chrome_psnr_margin must be non-negative")
        for name in (
            "checkpoint_min_ssim_gain",
            "max_ssim_regression",
            "max_psnr_regression",
        ):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if (
            self.enabled
            and self.target_ssim_srgb is None
            and self.target_psnr_srgb is None
        ):
            raise ValueError("adaptive compute requires an SSIM or PSNR quality target")
        if (
            self.enabled
            and self.effective_model_ssim_threshold is not None
            and self.effective_model_ssim_threshold > 1.0
        ):
            raise ValueError(
                "adaptive SSIM target plus Chrome safety margin must not exceed 1"
            )

    @property
    def effective_model_ssim_threshold(self) -> Optional[float]:
        if self.target_ssim_srgb is None:
            return None
        return float(self.target_ssim_srgb + self.chrome_ssim_safety_margin)

    @property
    def effective_model_psnr_threshold(self) -> Optional[float]:
        if self.target_psnr_srgb is None:
            return None
        return float(self.target_psnr_srgb + self.chrome_psnr_safety_margin)

    def checkpoint_policy(self) -> AdaptiveComputePolicy:
        """Return the shared checkpoint-selection policy used by the converter."""

        return AdaptiveComputePolicy(
            min_checkpoints=self.min_checkpoints,
            target_ssim_srgb=self.target_ssim_srgb,
            target_psnr_srgb=self.target_psnr_srgb,
            checkpoint_min_ssim_gain=self.checkpoint_min_ssim_gain,
            max_ssim_regression=self.max_ssim_regression,
            max_psnr_regression=self.max_psnr_regression,
            plateau_min_ssim_gain=0.0,
            plateau_min_psnr_gain=0.0,
            min_ssim_gain_per_second=0.0,
            stop_on_regression=False,
            stop_on_plateau=False,
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            **self.__dict__,
            "effective_model_ssim_threshold": self.effective_model_ssim_threshold,
            "effective_model_psnr_threshold": self.effective_model_psnr_threshold,
            "runtime_scorer": CANVAS_RUNTIME_SCORER,
            "runtime_scorer_pixel_exact": True,
            "runtime_scorer_calibration_checkpoints": (
                CANVAS_RUNTIME_CALIBRATION_CHECKPOINTS
            ),
            "runtime_scorer_calibration": "data/canvas-checkpoint-parity.json",
        }


@dataclass(frozen=True)
class OnlineAdaptiveDecision:
    """Decision made from the pixel-runtime checkpoints observed so far."""

    stop: bool
    reason: str
    checkpoints_observed: int
    current: CanvasCheckpoint
    selected: CanvasCheckpoint
    config: OnlineAdaptiveConfig

    def as_dict(self) -> Dict[str, Any]:
        return {
            "mode": "online-observed-only",
            "uses_future_evidence": False,
            "stop": self.stop,
            "reason": self.reason,
            "checkpoints_observed": self.checkpoints_observed,
            "current": dict(self.current.__dict__),
            "selected": dict(self.selected.__dict__),
            "requested_chrome_target": {
                "ssim_srgb": self.config.target_ssim_srgb,
                "psnr_srgb": self.config.target_psnr_srgb,
            },
            "effective_model_threshold": {
                "ssim_srgb": self.config.effective_model_ssim_threshold,
                "psnr_srgb": self.config.effective_model_psnr_threshold,
            },
            "policy": self.config.as_dict(),
        }


@dataclass(frozen=True)
class AdaptiveSimulationResult:
    selected: CanvasCheckpoint
    stop_checkpoint: CanvasCheckpoint
    full_run_best: CanvasCheckpoint
    stop_reason: str
    checkpoints_observed: int
    checkpoints_available: int
    observed_stage_sec: float
    full_stage_sec: float
    saved_stage_sec: float
    ssim_opportunity_cost: float
    psnr_opportunity_cost: float
    decisions: Tuple[Dict[str, Any], ...]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "selected": dict(self.selected.__dict__),
            "stop_checkpoint": dict(self.stop_checkpoint.__dict__),
            "full_run_best": dict(self.full_run_best.__dict__),
            "stop_reason": self.stop_reason,
            "checkpoints_observed": self.checkpoints_observed,
            "checkpoints_available": self.checkpoints_available,
            "observed_stage_sec": self.observed_stage_sec,
            "full_stage_sec": self.full_stage_sec,
            "saved_stage_sec": self.saved_stage_sec,
            "ssim_opportunity_cost": self.ssim_opportunity_cost,
            "psnr_opportunity_cost": self.psnr_opportunity_cost,
            "decisions": list(self.decisions),
        }


@dataclass(frozen=True)
class CanvasBudgetPoint:
    image: str
    requested_budget: int
    ssim_srgb: float
    lpips: float
    runtime_sec: float
    artifact_bytes: int
    final_splats: int


@dataclass(frozen=True)
class ScalingPolicy:
    """Retrospective 2k-to-4k scale decision thresholds."""

    target_ssim_srgb: float = 0.95
    target_lpips: Optional[float] = 0.15
    min_ssim_gain: float = 0.005
    min_lpips_gain: float = 0.001


def resolve_online_adaptive_config(
    refinement_config: Mapping[str, Any],
) -> OnlineAdaptiveConfig:
    """Resolve and validate the converter's default-off online policy."""

    enabled_value = refinement_config.get("adaptive_compute_enabled", False)
    if not isinstance(enabled_value, bool):
        raise ValueError("adaptive_compute_enabled must be a boolean")
    target_ssim = refinement_config.get("adaptive_compute_target_ssim_srgb", 0.98)
    target_psnr = refinement_config.get("adaptive_compute_target_psnr_srgb")
    return OnlineAdaptiveConfig(
        enabled=enabled_value,
        min_checkpoints=int(
            refinement_config.get("adaptive_compute_min_checkpoints", 2)
        ),
        target_ssim_srgb=(None if target_ssim is None else float(target_ssim)),
        target_psnr_srgb=(None if target_psnr is None else float(target_psnr)),
        chrome_ssim_safety_margin=float(
            refinement_config.get(
                "adaptive_compute_chrome_ssim_margin",
                DEFAULT_CHROME_SSIM_SAFETY_MARGIN,
            )
        ),
        chrome_psnr_safety_margin=float(
            refinement_config.get(
                "adaptive_compute_chrome_psnr_margin",
                DEFAULT_CHROME_PSNR_SAFETY_MARGIN,
            )
        ),
        checkpoint_min_ssim_gain=float(
            refinement_config.get("canvas_stage_min_ssim_gain", 0.0005)
        ),
        max_ssim_regression=float(
            refinement_config.get("canvas_stage_max_ssim_regression", 0.0005)
        ),
        max_psnr_regression=float(
            refinement_config.get("canvas_stage_max_psnr_regression", 0.10)
        ),
    )


def evaluate_online_checkpoints(
    checkpoints: Sequence[CanvasCheckpoint],
    config: OnlineAdaptiveConfig,
) -> OnlineAdaptiveDecision:
    """Stop only when an observed, selected checkpoint meets the hard target."""

    if not checkpoints:
        raise ValueError("at least one checkpoint is required")
    policy = config.checkpoint_policy()
    selected = _best_checkpoint(checkpoints, policy)
    if not config.enabled:
        reason = "disabled"
        stop = False
    elif len(checkpoints) < config.min_checkpoints:
        reason = "minimum-checkpoints"
        stop = False
    else:
        ssim_met = (
            config.effective_model_ssim_threshold is None
            or selected.ssim_srgb >= config.effective_model_ssim_threshold
        )
        psnr_met = (
            config.effective_model_psnr_threshold is None
            or selected.psnr_srgb >= config.effective_model_psnr_threshold
        )
        stop = bool(ssim_met and psnr_met)
        reason = "quality-target" if stop else "quality-target-not-met"
    return OnlineAdaptiveDecision(
        stop=stop,
        reason=reason,
        checkpoints_observed=len(checkpoints),
        current=checkpoints[-1],
        selected=selected,
        config=config,
    )


def simulate_adaptive_checkpoints(
    checkpoints: Sequence[CanvasCheckpoint],
    policy: AdaptiveComputePolicy,
) -> AdaptiveSimulationResult:
    """Replay a policy over observed checkpoints and report saved stage work."""

    if not checkpoints:
        raise ValueError("at least one checkpoint is required")
    for checkpoint in checkpoints:
        if checkpoint.elapsed_sec < 0.0:
            raise ValueError("checkpoint elapsed_sec must be non-negative")

    full_run_best = _best_checkpoint(checkpoints, policy)
    best = checkpoints[0]
    stop_checkpoint = checkpoints[-1]
    stop_reason = "curve-exhausted"
    observed = len(checkpoints)
    decisions = []

    for index, checkpoint in enumerate(checkpoints):
        accepted, reason = _prefer_checkpoint(checkpoint, best, policy)
        previous_best = best
        if index == 0 or accepted:
            best = checkpoint
        ssim_gain = checkpoint.ssim_srgb - previous_best.ssim_srgb
        psnr_gain = checkpoint.psnr_srgb - previous_best.psnr_srgb
        gain_rate = (
            ssim_gain / checkpoint.elapsed_sec if checkpoint.elapsed_sec > 0.0 else 0.0
        )
        decision = {
            "index": index,
            "checkpoint": checkpoint.label,
            "accepted": bool(index == 0 or accepted),
            "accept_reason": "initial" if index == 0 else reason,
            "selected": best.label,
            "ssim_gain_vs_previous_best": float(ssim_gain),
            "psnr_gain_vs_previous_best": float(psnr_gain),
            "ssim_gain_per_second": float(gain_rate),
            "stop": False,
            "stop_reason": None,
        }
        decisions.append(decision)

        seen = index + 1
        if seen < policy.min_checkpoints:
            continue
        reason_to_stop = _stop_reason(
            checkpoint=checkpoint,
            selected=best,
            accepted=bool(index == 0 or accepted),
            ssim_gain=ssim_gain,
            psnr_gain=psnr_gain,
            gain_rate=gain_rate,
            policy=policy,
        )
        if reason_to_stop is not None:
            stop_checkpoint = checkpoint
            stop_reason = reason_to_stop
            observed = seen
            decisions[-1]["stop"] = True
            decisions[-1]["stop_reason"] = reason_to_stop
            break

    observed_seconds = float(
        sum(checkpoint.elapsed_sec for checkpoint in checkpoints[:observed])
    )
    full_seconds = float(sum(checkpoint.elapsed_sec for checkpoint in checkpoints))
    return AdaptiveSimulationResult(
        selected=best,
        stop_checkpoint=stop_checkpoint,
        full_run_best=full_run_best,
        stop_reason=stop_reason,
        checkpoints_observed=observed,
        checkpoints_available=len(checkpoints),
        observed_stage_sec=observed_seconds,
        full_stage_sec=full_seconds,
        saved_stage_sec=max(0.0, full_seconds - observed_seconds),
        ssim_opportunity_cost=max(0.0, full_run_best.ssim_srgb - best.ssim_srgb),
        psnr_opportunity_cost=max(0.0, full_run_best.psnr_srgb - best.psnr_srgb),
        decisions=tuple(decisions),
    )


def retrospective_scale_decision(
    lower: CanvasBudgetPoint,
    higher: CanvasBudgetPoint,
    policy: ScalingPolicy,
) -> Dict[str, Any]:
    """Describe an oracle 2k-to-4k decision from two already-observed points."""

    if lower.image != higher.image:
        raise ValueError("budget points must describe the same image")
    if lower.requested_budget >= higher.requested_budget:
        raise ValueError("higher budget point must have a larger requested budget")

    ssim_gain = float(higher.ssim_srgb - lower.ssim_srgb)
    lpips_gain = float(lower.lpips - higher.lpips)
    quality_target_met = lower.ssim_srgb >= policy.target_ssim_srgb and (
        policy.target_lpips is None or lower.lpips <= policy.target_lpips
    )
    if quality_target_met:
        scale = False
        reason = "quality-target"
    elif ssim_gain >= policy.min_ssim_gain and lpips_gain >= policy.min_lpips_gain:
        scale = True
        reason = "observed-quality-gain"
    else:
        scale = False
        reason = "observed-gain-below-threshold"
    return {
        "image": lower.image,
        "mode": "retrospective-oracle",
        "from_budget": lower.requested_budget,
        "to_budget": higher.requested_budget,
        "scale": scale,
        "reason": reason,
        "ssim_gain": ssim_gain,
        "lpips_gain": lpips_gain,
        "runtime_delta_sec": float(higher.runtime_sec - lower.runtime_sec),
        "artifact_delta_bytes": int(higher.artifact_bytes - lower.artifact_bytes),
        "final_splat_delta": int(higher.final_splats - lower.final_splats),
    }


def _best_checkpoint(
    checkpoints: Sequence[CanvasCheckpoint],
    policy: AdaptiveComputePolicy,
) -> CanvasCheckpoint:
    best = checkpoints[0]
    for checkpoint in checkpoints[1:]:
        accepted, _ = _prefer_checkpoint(checkpoint, best, policy)
        if accepted:
            best = checkpoint
    return best


def _prefer_checkpoint(
    candidate: CanvasCheckpoint,
    incumbent: CanvasCheckpoint,
    policy: AdaptiveComputePolicy,
) -> Tuple[bool, str]:
    psnr_safe = candidate.psnr_srgb >= incumbent.psnr_srgb - policy.max_psnr_regression
    if candidate.ssim_srgb >= incumbent.ssim_srgb + policy.checkpoint_min_ssim_gain:
        return (psnr_safe, "material-ssim-gain" if psnr_safe else "psnr-gate")
    if (
        candidate.splat_count < incumbent.splat_count
        and candidate.ssim_srgb >= incumbent.ssim_srgb - policy.max_ssim_regression
        and psnr_safe
    ):
        return True, "smaller-equivalent"
    if candidate.ssim_srgb < incumbent.ssim_srgb - policy.max_ssim_regression:
        return False, "ssim-regression"
    if not psnr_safe:
        return False, "psnr-regression"
    return False, "gain-below-checkpoint-threshold"


def _stop_reason(
    *,
    checkpoint: CanvasCheckpoint,
    selected: CanvasCheckpoint,
    accepted: bool,
    ssim_gain: float,
    psnr_gain: float,
    gain_rate: float,
    policy: AdaptiveComputePolicy,
) -> Optional[str]:
    target_ssim_met = (
        policy.target_ssim_srgb is None or selected.ssim_srgb >= policy.target_ssim_srgb
    )
    target_psnr_met = (
        policy.target_psnr_srgb is None or selected.psnr_srgb >= policy.target_psnr_srgb
    )
    target_enabled = (
        policy.target_ssim_srgb is not None or policy.target_psnr_srgb is not None
    )
    if target_enabled and target_ssim_met and target_psnr_met:
        return "quality-target"
    if (
        policy.stop_on_regression
        and not accepted
        and (
            checkpoint.ssim_srgb < selected.ssim_srgb - policy.max_ssim_regression
            or checkpoint.psnr_srgb < selected.psnr_srgb - policy.max_psnr_regression
        )
    ):
        return "regression-revert"
    if (
        policy.stop_on_plateau
        and ssim_gain < policy.plateau_min_ssim_gain
        and psnr_gain < policy.plateau_min_psnr_gain
    ):
        return "plateau"
    if (
        policy.min_ssim_gain_per_second > 0.0
        and gain_rate < policy.min_ssim_gain_per_second
    ):
        return "low-marginal-return"
    return None
