"""Monotonic candidate loop and acceptance gate (ADR-003).

Evaluation is intentionally outside each operator: an operator cannot
redefine success to favor its own output.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Protocol, Sequence, Tuple

from ..splat import GaussianSplat
from .analysis import ResidualAnalysis
from .config import FidelityConfig
from .metrics import FidelityMetrics

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FidelityCandidate:
    name: str
    splats: Tuple[GaussianSplat, ...]
    recipe_overrides: Dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class FidelityResult:
    winner: FidelityCandidate
    baseline_metrics: FidelityMetrics
    final_metrics: FidelityMetrics
    decisions: Tuple[Dict[str, Any], ...]
    passes_run: int
    candidates_evaluated: int
    stop_reason: str


class CandidateOperator(Protocol):
    name: str

    def propose(
        self,
        best: FidelityCandidate,
        analysis: ResidualAnalysis,
        limit: int,
    ) -> Sequence[FidelityCandidate]: ...


def accept_candidate(
    *,
    baseline: FidelityMetrics,
    incumbent: FidelityMetrics,
    candidate: FidelityMetrics,
    config: FidelityConfig,
) -> Tuple[bool, str]:
    """Pareto-like gate: meaningful primary gain, no important regressions.

    Hard floors compare against the BASELINE (monotonic no-regression
    contract); gains compare against the INCUMBENT best. NaN metrics (e.g.
    lpips without the optional dependency) compare False and therefore never
    count as a gain and never trip a hard gate.
    """
    if candidate.render_method.startswith("proxy"):
        return False, "no deployed-artifact render"
    if config.max_file_size_bytes is not None:
        if candidate.file_size_bytes > config.max_file_size_bytes:
            return False, "file-size budget exceeded"
    if candidate.splat_count > baseline.splat_count + config.max_added_splats:
        return False, "splat budget exceeded"
    if candidate.ssim_srgb < baseline.ssim_srgb - config.max_ssim_regression:
        return False, "SSIM hard gate"
    if candidate.edge_chamfer > baseline.edge_chamfer + config.max_edge_regression:
        return False, "edge hard gate"
    if candidate.worst_roi_error > baseline.worst_roi_error * (
        1.0 + config.max_worst_roi_regression_fraction
    ):
        return False, "worst-ROI hard gate"

    lpips_gain = incumbent.lpips - candidate.lpips
    salient_gain = incumbent.salient_lpips - candidate.salient_lpips
    delta_e_gain = incumbent.delta_e_ok_p95 - candidate.delta_e_ok_p95

    meaningful_gain = bool(
        lpips_gain >= config.min_lpips_gain
        or salient_gain >= config.min_lpips_gain
        or delta_e_gain >= config.min_delta_e_gain
    )
    if meaningful_gain:
        return True, "measured fidelity gain"
    return False, "gain below threshold"


class FidelityStage:
    """Runs bounded operator proposals through the accept-or-revert gate."""

    def __init__(
        self,
        config: FidelityConfig,
        evaluator: "FidelityEvaluatorLike",
        operators: Sequence[CandidateOperator],
    ):
        self.config = config
        self.evaluator = evaluator
        self.operators = tuple(operators)

    def run(self, baseline: FidelityCandidate) -> FidelityResult:
        best = baseline
        best_metrics = self.evaluator.evaluate(best, label="baseline")
        baseline_metrics = best_metrics
        decisions = []
        candidates_evaluated = 0
        passes_run = 0
        stop_reason = "max-passes"

        for pass_index in range(self.config.max_passes):
            passes_run = pass_index + 1
            analysis = self.evaluator.analyze(best)
            improved = False

            for operator in self.operators:
                proposals = operator.propose(
                    best, analysis, limit=self.config.max_candidates_per_pass
                )
                for candidate in proposals:
                    metrics = self.evaluator.evaluate(
                        candidate,
                        label=f"pass{pass_index:02d}-{candidate.name}",
                        baseline=baseline_metrics,
                    )
                    candidates_evaluated += 1
                    accepted, reason = accept_candidate(
                        baseline=baseline_metrics,
                        incumbent=best_metrics,
                        candidate=metrics,
                        config=self.config,
                    )
                    decisions.append(
                        {
                            "pass": pass_index,
                            "operator": operator.name,
                            "candidate": candidate.name,
                            "accepted": accepted,
                            "reason": reason,
                            "metrics": metrics.as_dict(),
                        }
                    )
                    if accepted:
                        best, best_metrics = candidate, metrics
                        improved = True

            if not improved:
                stop_reason = "no-accepted-candidate"
                break

        return FidelityResult(
            winner=best,
            baseline_metrics=baseline_metrics,
            final_metrics=best_metrics,
            decisions=tuple(decisions),
            passes_run=passes_run,
            candidates_evaluated=candidates_evaluated,
            stop_reason=stop_reason,
        )


class FidelityEvaluatorLike(Protocol):
    def evaluate(
        self,
        candidate: FidelityCandidate,
        *,
        label: str,
        baseline: FidelityMetrics | None = None,
    ) -> FidelityMetrics: ...

    def analyze(self, candidate: FidelityCandidate) -> ResidualAnalysis: ...
