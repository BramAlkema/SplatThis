"""The splat-orientation convention, pinned directly.

``CLAUDE.md`` calls this convention non-negotiable, and the project's own notes
record a wrong orientation as the single largest historical quality regression.
Until now nothing tested it: no test in the suite referenced
``edge_tangent_angle`` at all, and ``features.py`` sat at 57% coverage.

These tests replace the intent of four abandoned September 2025 pull requests
(#1 and #3, "gaussian rotation axis selection" / "optimized gaussian
orientation"). Those targeted ``src/splat_this/``, a package that has since
been retired, so their patches could not be applied — but the concern they
raised was real and was never covered afterwards.

The end-to-end floor in ``test_end_to_end_regression.py`` also detects a broken
orientation, via a 0.056 SSIM drop. That is a backstop, not a diagnosis: it
says the pipeline moved. These tests say which convention broke.
"""

from __future__ import annotations

import numpy as np
import pytest

from splatthis.features import edge_tangent_angle


def _unit(angle: float) -> np.ndarray:
    return np.array([np.cos(angle), np.sin(angle)], dtype=np.float64)


@pytest.mark.parametrize(
    "gradient_angle_deg",
    [0.0, 30.0, 45.0, 90.0, 135.0, 180.0, 270.0, -45.0],
)
def test_major_axis_is_perpendicular_to_the_gradient(gradient_angle_deg):
    """The whole convention in one assertion.

    The structure tensor's dominant eigenvector points *across* the edge. A
    splat that smears along the edge must therefore sit 90 degrees from it.
    Dropping that rotation is exactly the historical defect.
    """
    gradient = _unit(np.deg2rad(gradient_angle_deg))

    major_axis = _unit(edge_tangent_angle(gradient))

    # Perpendicular <=> zero dot product. Sign/wrap-around is irrelevant here,
    # which is the point: an axis is a direction, not an arrow.
    assert np.isclose(float(np.dot(major_axis, gradient)), 0.0, atol=1e-12)


def test_orientation_is_invariant_to_gradient_sign():
    """A gradient and its negation describe the same edge.

    The eigenvector's sign is arbitrary — solvers may return either — so an
    orientation that flips with it would make seeding depend on LAPACK
    internals rather than on image content.
    """
    gradient = _unit(np.deg2rad(37.0))

    forward = _unit(edge_tangent_angle(gradient))
    reversed_ = _unit(edge_tangent_angle(-gradient))

    # Same axis, possibly opposite arrow: |cos| between them is 1.
    assert np.isclose(abs(float(np.dot(forward, reversed_))), 1.0, atol=1e-12)


def test_axis_aligned_gradient_gives_the_expected_quarter_turn():
    """A concrete, hand-checkable case, so a sign error cannot hide in algebra."""
    # Gradient along +x (a vertical edge) -> splat elongated along y.
    assert np.isclose(edge_tangent_angle(np.array([1.0, 0.0])), np.pi / 2)
    # Gradient along +y (a horizontal edge) -> splat elongated along x.
    assert np.isclose(edge_tangent_angle(np.array([0.0, 1.0])), np.pi)


def test_every_anisotropic_seeding_site_uses_the_helper():
    """The convention must not be re-derived inline anywhere.

    ``CLAUDE.md``: "any anisotropic splat creation must go through
    features.edge_tangent_angle()". A second, open-coded ``arctan2 + pi/2``
    is how the two implementations drifted apart the first time.
    """
    import ast
    import pathlib

    import splatthis

    package = pathlib.Path(splatthis.__file__).parent
    offenders = []

    for source in package.rglob("*.py"):
        if source.name == "features.py":
            continue  # the helper itself
        tree = ast.parse(source.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            # Look for `arctan2(...) + <something>` — the open-coded form.
            if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Add):
                continue
            left = node.left
            if (
                isinstance(left, ast.Call)
                and isinstance(left.func, ast.Attribute)
                and left.func.attr == "arctan2"
            ):
                offenders.append(f"{source.relative_to(package)}:{node.lineno}")

    assert offenders == [], (
        "anisotropic orientation computed inline instead of via "
        f"features.edge_tangent_angle(): {offenders}"
    )
