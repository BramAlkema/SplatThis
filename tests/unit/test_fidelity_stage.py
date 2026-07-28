"""ADR-003 fidelity stage: monotonic gate, no-op guarantee, determinism."""

import json
from dataclasses import replace

import numpy as np
import pytest
from PIL import Image

from png2svg_gs.converter import PNG2SVGConverter
from png2svg_gs.fidelity import (
    FidelityCandidate,
    FidelityConfig,
    FidelityStage,
    RecolorOperator,
    analyze_residual,
    compute_fidelity_metrics,
    resolve_fidelity_config,
    select_fixed_rois,
)
from png2svg_gs.fidelity.stage import accept_candidate
from png2svg_gs.splat import create_isotropic_splat


def _metrics(**overrides):
    base = dict(
        lpips=0.40,
        psnr_srgb=20.0,
        ssim_srgb=0.70,
        ms_ssim_luma=0.75,
        delta_e_ok_mean=0.05,
        delta_e_ok_p95=0.15,
        edge_chamfer=2.0,
        edge_gradient_l1=0.02,
        salient_lpips=0.45,
        worst_roi_error=0.20,
        splat_count=100,
        file_size_bytes=50_000,
        render_method="rsvg-convert",
    )
    base.update(overrides)
    from png2svg_gs.fidelity.metrics import FidelityMetrics

    return FidelityMetrics(**base)


CONFIG = FidelityConfig(mode="max")


def test_gate_rejects_proxy_renders():
    baseline = _metrics()
    candidate = _metrics(lpips=0.30, render_method="proxy-numpy")
    ok, reason = accept_candidate(
        baseline=baseline, incumbent=baseline, candidate=candidate, config=CONFIG
    )
    assert not ok and "deployed" in reason


def test_gate_hard_floors_use_baseline():
    baseline = _metrics()
    # Big LPIPS gain but SSIM regressed beyond the hard gate vs baseline.
    candidate = _metrics(lpips=0.30, ssim_srgb=0.70 - 0.01)
    ok, reason = accept_candidate(
        baseline=baseline, incumbent=baseline, candidate=candidate, config=CONFIG
    )
    assert not ok and reason == "SSIM hard gate"

    candidate = _metrics(lpips=0.30, edge_chamfer=2.0 + 0.01)
    ok, reason = accept_candidate(
        baseline=baseline, incumbent=baseline, candidate=candidate, config=CONFIG
    )
    assert not ok and reason == "edge hard gate"

    candidate = _metrics(lpips=0.30, worst_roi_error=0.21)
    ok, reason = accept_candidate(
        baseline=baseline, incumbent=baseline, candidate=candidate, config=CONFIG
    )
    assert not ok and reason == "worst-ROI hard gate"


def test_gate_accepts_meaningful_lpips_gain_and_rejects_noise():
    baseline = _metrics()
    ok, _ = accept_candidate(
        baseline=baseline,
        incumbent=baseline,
        candidate=_metrics(lpips=0.40 - 0.002),
        config=CONFIG,
    )
    assert ok
    ok, reason = accept_candidate(
        baseline=baseline,
        incumbent=baseline,
        candidate=_metrics(lpips=0.40 - 0.0005),
        config=CONFIG,
    )
    assert not ok and reason == "gain below threshold"


def test_gate_handles_nan_lpips_via_delta_e():
    """Without lpips installed the gate falls back to the delta-E gain."""
    nan = float("nan")
    baseline = _metrics(lpips=nan, salient_lpips=nan)
    ok, _ = accept_candidate(
        baseline=baseline,
        incumbent=baseline,
        candidate=_metrics(lpips=nan, salient_lpips=nan, delta_e_ok_p95=0.14),
        config=CONFIG,
    )
    assert ok
    ok, reason = accept_candidate(
        baseline=baseline,
        incumbent=baseline,
        candidate=_metrics(lpips=nan, salient_lpips=nan, delta_e_ok_p95=0.1495),
        config=CONFIG,
    )
    assert not ok


def test_gate_enforces_budgets():
    baseline = _metrics()
    config = replace(CONFIG, max_file_size_bytes=60_000)
    ok, reason = accept_candidate(
        baseline=baseline,
        incumbent=baseline,
        candidate=_metrics(lpips=0.30, file_size_bytes=70_000),
        config=config,
    )
    assert not ok and reason == "file-size budget exceeded"
    ok, reason = accept_candidate(
        baseline=baseline,
        incumbent=baseline,
        candidate=_metrics(lpips=0.30, splat_count=101),
        config=CONFIG,
    )
    assert not ok and reason == "splat budget exceeded"


class _FakeEvaluator:
    """Deterministic evaluator: metrics keyed by candidate name."""

    def __init__(self, metrics_by_name):
        self.metrics_by_name = metrics_by_name
        self.evaluated = []

    def evaluate(self, candidate, *, label, baseline=None):
        self.evaluated.append(candidate.name)
        return self.metrics_by_name[candidate.name]

    def analyze(self, candidate):
        return None  # operators in these tests ignore analysis


class _FixedOperator:
    def __init__(self, name, candidates):
        self.name = name
        self._candidates = candidates
        self.calls = 0

    def propose(self, best, analysis, limit):
        self.calls += 1
        # One round of proposals only; nothing new on later passes.
        return self._candidates if self.calls == 1 else []


def test_stage_keeps_winner_and_traces_all_decisions():
    baseline = FidelityCandidate(name="baseline", splats=())
    better = FidelityCandidate(name="better", splats=())
    worse = FidelityCandidate(name="worse", splats=())
    evaluator = _FakeEvaluator(
        {
            "baseline": _metrics(),
            "better": _metrics(lpips=0.38),
            "worse": _metrics(ssim_srgb=0.60),
        }
    )
    stage = FidelityStage(
        config=CONFIG,
        evaluator=evaluator,
        operators=[_FixedOperator("fixed", [worse, better])],
    )
    result = stage.run(baseline)
    assert result.winner.name == "better"
    assert result.final_metrics.lpips == pytest.approx(0.38)
    assert result.baseline_metrics.lpips == pytest.approx(0.40)
    assert [d["candidate"] for d in result.decisions] == ["worse", "better"]
    assert [d["accepted"] for d in result.decisions] == [False, True]
    assert result.stop_reason == "no-accepted-candidate"


def test_stage_with_no_operators_returns_baseline():
    baseline = FidelityCandidate(name="baseline", splats=())
    evaluator = _FakeEvaluator({"baseline": _metrics()})
    stage = FidelityStage(config=CONFIG, evaluator=evaluator, operators=[])
    result = stage.run(baseline)
    assert result.winner is baseline
    assert result.candidates_evaluated == 0
    assert result.decisions == ()


def test_select_fixed_rois_deterministic_and_spread():
    rng = np.random.default_rng(0)
    error = rng.random((100, 120)).astype(np.float32)
    error[10:20, 10:20] += 5.0
    error[70:80, 90:100] += 4.0
    rois_a = select_fixed_rois(error, size=16, count=4)
    rois_b = select_fixed_rois(error, size=16, count=4)
    assert rois_a == rois_b
    assert len(rois_a) == 4
    y0, x0, y1, x1 = rois_a[0]
    assert (y1 - y0, x1 - x0) == (16, 16)
    # The two hot spots must be covered by the first two ROIs.
    assert any(y0 <= 15 < y1 and x0 <= 15 < x1 for (y0, x0, y1, x1) in rois_a[:2])
    assert any(y0 <= 75 < y1 and x0 <= 95 < x1 for (y0, x0, y1, x1) in rois_a[:2])


def test_metrics_identical_images_are_perfect():
    rng = np.random.default_rng(1)
    img = rng.random((48, 40, 3)).astype(np.float32)
    m = compute_fidelity_metrics(img, img, render_method="rsvg-convert")
    assert m.ssim_srgb == pytest.approx(1.0, abs=1e-5)
    assert m.delta_e_ok_p95 == pytest.approx(0.0, abs=1e-5)
    assert m.edge_chamfer == pytest.approx(0.0, abs=1e-6)
    assert m.psnr_srgb > 60


def test_recolor_operator_targets_worst_roi():
    # Localized defect so the worst ROI lands on the splat's neighborhood:
    # the center square should be red but renders blue.
    target = np.full((64, 64, 3), 0.5, dtype=np.float32)
    rendered = target.copy()
    target[20:44, 20:44] = [0.8, 0.0, 0.0]
    rendered[20:44, 20:44] = [0.0, 0.0, 0.8]
    analysis = analyze_residual(target, rendered, roi_size=32, roi_count=2)
    splat = create_isotropic_splat(
        center=np.array([32.0, 32.0]),
        sigma=10.0,
        color=np.array([0.0, 0.0, 0.8]),
        alpha=0.9,
    )
    best = FidelityCandidate(name="baseline", splats=(splat,))
    proposals = RecolorOperator().propose(best, analysis, limit=4)
    assert proposals, "expected at least one recolor proposal"
    new_color = np.asarray(proposals[0].splats[0].color[:3])
    # Bounded step, shifted toward red and away from blue.
    assert new_color[0] > 0.1
    assert new_color[2] < 0.75


def test_resolve_fidelity_config_validates_mode():
    assert resolve_fidelity_config({}).mode == "off"
    assert resolve_fidelity_config({"fidelity_stage": "balanced"}).max_passes == 2
    with pytest.raises(ValueError):
        resolve_fidelity_config({"fidelity_stage": "extreme"})


def _tiny_image(tmp_path):
    rng = np.random.default_rng(7)
    img_path = tmp_path / "tiny.png"
    Image.fromarray(rng.uniform(0, 255, size=(24, 24, 3)).astype(np.uint8)).save(
        img_path
    )
    return img_path


def _converter(**refinement):
    return PNG2SVGConverter(
        max_splats=12,
        stages=[2, 1],
        optimizer_backend="torch",
        refinement_config=refinement or None,
    )


def test_noop_fidelity_stage_cannot_change_output(tmp_path):
    """ADR-003 Phase-1 exit gate: balanced (no operators) is byte-identical."""
    img = _tiny_image(tmp_path)

    out_off = tmp_path / "off.svg"
    _converter().convert(str(img), output_path=str(out_off), seed=5, verbose=False)

    out_shell = tmp_path / "shell.svg"
    _converter(fidelity_stage="balanced").convert(
        str(img), output_path=str(out_shell), seed=5, verbose=False
    )

    assert out_off.read_bytes() == out_shell.read_bytes()


def test_fidelity_stage_writes_manifest_and_artifacts(tmp_path):
    img = _tiny_image(tmp_path)
    artifacts = tmp_path / "artifacts"
    out = tmp_path / "out.svg"
    _converter(fidelity_stage="balanced").convert(
        str(img),
        output_path=str(out),
        seed=5,
        verbose=False,
        artifacts_dir=str(artifacts),
    )
    manifest = json.loads((artifacts / "run_manifest.json").read_text())
    fragment = manifest["fidelity_stage"]
    assert fragment["enabled"] is True
    assert fragment["mode"] == "balanced"
    assert fragment["winner_is_baseline"] is True
    assert fragment["candidates_evaluated"] == 0
    # Baseline was rendered from the deployed artifact, not a proxy.
    assert not str(fragment["baseline_metrics"]["render_method"]).startswith("proxy")
    fidelity_dir = artifacts / "fidelity"
    assert (fidelity_dir / "decisions.jsonl").exists()
    assert (fidelity_dir / "baseline-metrics.json").exists()
    assert (fidelity_dir / "final-metrics.json").exists()
