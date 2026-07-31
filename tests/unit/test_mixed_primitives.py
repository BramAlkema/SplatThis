import zipfile

import numpy as np
import pytest

from png2svg_gs.io import save_pptx_with_splats
from png2svg_gs.mixed_primitives import (
    edge_paths_to_svg_group,
    edge_strokes_to_svg_group,
    inject_edge_paths_into_pptx,
    inject_svg_before_close,
    propose_residual_edge_paths,
    propose_residual_edge_strokes,
)
from png2svg_gs.splat import create_isotropic_splat


def test_residual_edge_strokes_are_bounded_deterministic_and_native_svg() -> None:
    target = np.zeros((16, 16, 3), dtype=np.float32)
    target[:, 8:, :] = 1.0
    rendered = np.full_like(target, 0.25)

    first = propose_residual_edge_strokes(
        target, rendered, max_strokes=4, length=5.0, width=1.0, opacity=0.6
    )
    second = propose_residual_edge_strokes(
        target, rendered, max_strokes=4, length=5.0, width=1.0, opacity=0.6
    )

    assert first == second
    assert 0 < len(first) <= 4
    assert all(stroke.score > 0 for stroke in first)
    fragment = edge_strokes_to_svg_group(first)
    assert fragment.count("<line ") == len(first)
    assert "<image" not in fragment
    assert 'stroke-linecap="round"' in fragment


def test_edge_stroke_proposals_validate_shape() -> None:
    target = np.zeros((4, 4, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="identical shapes"):
        propose_residual_edge_strokes(target, np.zeros((3, 4, 3), dtype=np.float32))


def test_svg_fragment_is_inserted_before_close() -> None:
    svg = '<svg xmlns="http://www.w3.org/2000/svg"></svg>'
    result = inject_svg_before_close(svg, '  <g id="correction"/>')

    assert result.index('id="correction"') < result.index("</svg>")
    assert result.count("</svg>") == 1


def test_residual_edge_paths_are_connected_native_paths() -> None:
    target = np.zeros((24, 24, 3), dtype=np.float32)
    yy, xx = np.ogrid[:24, :24]
    target[(xx - 12) ** 2 + (yy - 12) ** 2 <= 7**2] = 1.0
    rendered = np.full_like(target, 0.2)

    paths = propose_residual_edge_paths(
        target,
        rendered,
        max_paths=3,
        path_length=10.0,
        width=0.8,
        opacity=0.6,
    )
    fragment = edge_paths_to_svg_group(paths)

    assert 0 < len(paths) <= 3
    assert all(len(path.points) >= 2 for path in paths)
    assert fragment.count("<path ") == len(paths)
    assert 'stroke-linejoin="round"' in fragment


def test_edge_paths_are_injected_as_native_drawingml_shapes(tmp_path) -> None:
    baseline = tmp_path / "baseline.pptx"
    output = tmp_path / "mixed.pptx"
    splat = create_isotropic_splat(
        center=np.array([8.0, 8.0]),
        sigma=4.0,
        color=np.array([0.2, 0.3, 0.4]),
        alpha=0.6,
    )
    save_pptx_with_splats([splat], 16, 16, str(baseline))
    target = np.zeros((16, 16, 3), dtype=np.float32)
    target[:, 8:, :] = 1.0
    paths = propose_residual_edge_paths(target, np.full_like(target, 0.2), max_paths=2)

    segment_count = inject_edge_paths_into_pptx(
        baseline, output, paths, width=16, height=16
    )

    assert segment_count > 0
    with zipfile.ZipFile(output) as package:
        slide = package.read("ppt/slides/slide1.xml").decode()
    assert slide.count('name="Edge Path ') == segment_count
    assert '<a:prstGeom prst="roundRect">' in slide
