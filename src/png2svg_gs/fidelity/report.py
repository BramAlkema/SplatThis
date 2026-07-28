"""Decision trace + manifest fragment for the fidelity stage (ADR-003)."""

from __future__ import annotations

import json
import os
from typing import Any, Dict

from .config import FidelityConfig
from .stage import FidelityResult


def write_fidelity_report(
    work_dir: str, result: FidelityResult, config: FidelityConfig, seed: Any
) -> Dict[str, Any]:
    """Write decisions.jsonl + metrics files; return the manifest fragment."""
    os.makedirs(work_dir, exist_ok=True)

    with open(os.path.join(work_dir, "decisions.jsonl"), "w") as f:
        for decision in result.decisions:
            f.write(json.dumps(decision, default=str) + "\n")
    with open(os.path.join(work_dir, "baseline-metrics.json"), "w") as f:
        json.dump(result.baseline_metrics.as_dict(), f, indent=2, default=str)
    with open(os.path.join(work_dir, "final-metrics.json"), "w") as f:
        json.dump(result.final_metrics.as_dict(), f, indent=2, default=str)

    accepted = [d for d in result.decisions if d["accepted"]]
    return {
        "enabled": True,
        "mode": config.mode,
        "seed": seed,
        "baseline_metrics": result.baseline_metrics.as_dict(),
        "final_metrics": result.final_metrics.as_dict(),
        "passes_run": result.passes_run,
        "candidates_evaluated": result.candidates_evaluated,
        "accepted_operations": sorted({d["operator"] for d in accepted}),
        "accepted_candidates": [d["candidate"] for d in accepted],
        "rejected_operations": len(result.decisions) - len(accepted),
        "actual_artifact_rasterizer": result.final_metrics.render_method,
        "hybrid_residual": False,
        "stop_reason": result.stop_reason,
        "winner_is_baseline": result.winner.name == "baseline",
    }
