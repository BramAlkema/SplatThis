"""Typed configuration for the fidelity stage (ADR-003)."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, Literal, Optional

FidelityMode = Literal["off", "balanced", "max"]

VALID_FIDELITY_MODES = ("off", "balanced", "max")


@dataclass(frozen=True)
class FidelityConfig:
    """Bounded budgets and acceptance thresholds for the fidelity stage.

    Thresholds are benchmark parameters, not universal constants (ADR-003);
    these defaults are starting values pending the Phase-0 noise-floor study.

    Deviation from the ADR sketch: the ADR's acceptance example used
    ``delta_e_gain >= 0.25``, which is unit-confused for OKLab (L spans
    [0, 1]; a just-noticeable difference is ~0.02). ``min_delta_e_gain``
    defaults to 0.005 in native OKLab distance instead.
    """

    mode: FidelityMode = "off"
    max_passes: int = 4
    max_candidates_per_pass: int = 12
    max_added_splats: int = 0
    max_file_size_bytes: Optional[int] = None
    min_lpips_gain: float = 0.001
    min_delta_e_gain: float = 0.005
    max_ssim_regression: float = 0.002
    max_edge_regression: float = 0.002
    max_worst_roi_regression_fraction: float = 0.01
    roi_size: int = 64
    roi_count: int = 8
    # Proxy fast-reject: skip the emit+rasterize cost when the cheap proxy
    # render already regresses SSIM this many times the hard gate.
    proxy_reject_ssim_factor: float = 4.0


_BALANCED = FidelityConfig(mode="balanced", max_passes=2, max_candidates_per_pass=8)
_MAX = FidelityConfig(mode="max", max_passes=4, max_candidates_per_pass=12)


def resolve_fidelity_config(refinement_config: Dict[str, Any]) -> FidelityConfig:
    """Resolve the stage config from refinement_config['fidelity_stage'].

    Accepts the mode string; unknown modes raise (fail fast at construction,
    mirroring the converter's recipe normalization rule).
    """
    mode = str(refinement_config.get("fidelity_stage", "off")).strip().lower()
    if mode not in VALID_FIDELITY_MODES:
        raise ValueError(
            f"Unsupported fidelity stage mode: {mode!r} "
            f"(expected one of {', '.join(VALID_FIDELITY_MODES)})"
        )
    if mode == "off":
        return FidelityConfig(mode="off")
    base = _BALANCED if mode == "balanced" else _MAX
    overrides = {}
    max_bytes = refinement_config.get("fidelity_max_file_size_bytes")
    if max_bytes is not None:
        overrides["max_file_size_bytes"] = int(max_bytes)
    return replace(base, **overrides) if overrides else base
