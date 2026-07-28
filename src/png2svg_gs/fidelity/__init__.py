"""Fidelity stage (ADR-003): bounded, accept-or-revert artifact-level polish.

Every fidelity operation is speculative: candidates are evaluated against the
DEPLOYED artifact (emitted + rasterized SVG) and kept only when measurably
better under hard regression gates. See docs/adr-003-fidelity-roadmap.md.
"""

from .config import FidelityConfig, resolve_fidelity_config
from .metrics import FidelityMetrics, compute_fidelity_metrics
from .analysis import ResidualAnalysis, analyze_residual, select_fixed_rois
from .evaluator import FidelityEvaluator
from .stage import CandidateOperator, FidelityCandidate, FidelityResult, FidelityStage
from .operators import RecolorOperator, build_operators
from .report import write_fidelity_report

__all__ = [
    "CandidateOperator",
    "FidelityCandidate",
    "FidelityConfig",
    "FidelityEvaluator",
    "FidelityMetrics",
    "FidelityResult",
    "FidelityStage",
    "RecolorOperator",
    "ResidualAnalysis",
    "analyze_residual",
    "build_operators",
    "compute_fidelity_metrics",
    "resolve_fidelity_config",
    "select_fixed_rois",
    "write_fidelity_report",
]
