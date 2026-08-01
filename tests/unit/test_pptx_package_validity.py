"""OPC package validity for generated PPTX files.

These are the checks that decide whether PowerPoint opens a deck cleanly or
throws the repair dialog. They were previously only run by hand against the
external `openxml-audit` tool; the structural half is reimplemented here so
it runs in CI on every platform, and the authoritative external gate is
still exercised when that tool happens to be installed.
"""

import shutil
import subprocess
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import List, Set

import numpy as np
import pytest

from splatthis.io import px_to_emu, save_pptx_with_splat_png, save_pptx_with_splats
from splatthis.splat import GaussianSplat, create_isotropic_splat

REL_NS = "{http://schemas.openxmlformats.org/package/2006/relationships}Relationship"
CT_NS = "http://schemas.openxmlformats.org/package/2006/content-types"
R_ID = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
R_EMBED = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed"

# Parts PowerPoint requires; a package missing any of these triggers repair.
REQUIRED_PARTS = {
    "[Content_Types].xml",
    "_rels/.rels",
    "docProps/core.xml",
    "docProps/app.xml",
    "ppt/presentation.xml",
    "ppt/_rels/presentation.xml.rels",
    "ppt/slides/slide1.xml",
    "ppt/slides/_rels/slide1.xml.rels",
    "ppt/slideLayouts/slideLayout1.xml",
    "ppt/slideLayouts/_rels/slideLayout1.xml.rels",
    "ppt/slideMasters/slideMaster1.xml",
    "ppt/slideMasters/_rels/slideMaster1.xml.rels",
    "ppt/theme/theme1.xml",
    # presProps/viewProps are optional per spec but their absence is a known
    # PowerPoint repair trigger, so they are required here deliberately.
    "ppt/presProps.xml",
    "ppt/viewProps.xml",
}


def _splats(n: int = 6) -> List[GaussianSplat]:
    rng = np.random.default_rng(5)
    return [
        create_isotropic_splat(
            center=rng.uniform(4, 28, size=2),
            sigma=float(rng.uniform(1.5, 4.0)),
            color=rng.uniform(0.1, 0.9, size=3),
            alpha=float(rng.uniform(0.3, 1.0)),
        )
        for _ in range(n)
    ]


def _assert_parts_present(names: Set[str]) -> None:
    missing = REQUIRED_PARTS - names
    assert not missing, f"missing required parts: {sorted(missing)}"


def _assert_xml_well_formed(zf: zipfile.ZipFile, names: Set[str]) -> None:
    for name in names:
        if name.endswith((".xml", ".rels")):
            try:
                ET.fromstring(zf.read(name))
            except ET.ParseError as exc:
                pytest.fail(f"{name} is not well-formed XML: {exc}")


def _assert_content_types_cover_parts(zf: zipfile.ZipFile, names: Set[str]) -> None:
    types = ET.fromstring(zf.read("[Content_Types].xml"))
    defaults = {
        d.get("Extension", "").lower() for d in types.findall(f"{{{CT_NS}}}Default")
    }
    overrides = {o.get("PartName") for o in types.findall(f"{{{CT_NS}}}Override")}
    for name in names:
        if name == "[Content_Types].xml":
            continue
        ext = name.rsplit(".", 1)[-1].lower()
        assert (
            ext in defaults or f"/{name}" in overrides
        ), f"{name} has no content type (Default .{ext} or Override)"


def _resolve_rel_target(base: str, target: str) -> str:
    resolved = str(Path(base + target).as_posix()).lstrip("/")
    while "/../" in resolved:
        head, tail = resolved.split("/../", 1)
        resolved = str(Path(head).parent.joinpath(tail).as_posix())
    return resolved


def _assert_rel_targets_resolve(zf: zipfile.ZipFile, names: Set[str]) -> None:
    for name in names:
        if not name.endswith(".rels"):
            continue
        base = name.rsplit("_rels/", 1)[0]
        for rel in ET.fromstring(zf.read(name)).findall(REL_NS):
            if rel.get("TargetMode") == "External":
                continue
            resolved = _resolve_rel_target(base, rel.get("Target", ""))
            assert (
                resolved in names
            ), f"{name} -> {rel.get('Id')} targets missing part {resolved}"


def _assert_no_dangling_rel_ids(zf: zipfile.ZipFile) -> None:
    for part in ("ppt/presentation.xml", "ppt/slides/slide1.xml"):
        head, tail = part.rsplit("/", 1)
        declared = {
            r.get("Id")
            for r in ET.fromstring(zf.read(f"{head}/_rels/{tail}.rels")).findall(REL_NS)
        }
        used = {
            el.get(attr)
            for el in ET.fromstring(zf.read(part)).iter()
            for attr in (R_ID, R_EMBED)
            if el.get(attr)
        }
        dangling = used - declared
        assert not dangling, f"{part} references undeclared rels {sorted(dangling)}"


def assert_valid_opc_package(path: Path) -> zipfile.ZipFile:
    """Structural OPC checks: the repair-dialog triggers, in order of nastiness."""
    assert path.exists() and path.stat().st_size > 0

    zf = zipfile.ZipFile(path)
    assert zf.testzip() is None, "corrupt zip entry"
    names: Set[str] = set(zf.namelist())

    _assert_parts_present(names)
    _assert_xml_well_formed(zf, names)
    _assert_content_types_cover_parts(zf, names)
    _assert_rel_targets_resolve(zf, names)
    _assert_no_dangling_rel_ids(zf)
    return zf


def _shape_count(zf: zipfile.ZipFile) -> int:
    return zf.read("ppt/slides/slide1.xml").decode("utf-8").count("<p:sp>")


@pytest.mark.parametrize("splat_style", ["gradient", "soft-edge", "blur"])
@pytest.mark.parametrize("painter_order", ["legacy", "back-to-front"])
def test_native_shape_package_is_valid(
    tmp_path: Path, splat_style: str, painter_order: str
):
    """Every splat style must emit a package PowerPoint can open."""
    out = tmp_path / f"{splat_style}-{painter_order}.pptx"
    splats = _splats()
    save_pptx_with_splats(
        splats,
        width=32,
        height=32,
        output_path=str(out),
        background_linear_rgb=np.array([0.1, 0.1, 0.12], dtype=np.float32),
        splat_style=splat_style,
        painter_order=painter_order,
    )
    zf = assert_valid_opc_package(out)
    # One shape per splat, plus the background plate.
    assert _shape_count(zf) == len(splats) + 1
    # Native-shape decks must carry no raster media.
    assert not [n for n in zf.namelist() if n.startswith("ppt/media/")]


def test_png_backed_package_is_valid(tmp_path: Path):
    """The raster-preview writer must also produce a clean package."""
    out = tmp_path / "raster.pptx"
    save_pptx_with_splat_png(_splats(), width=32, height=32, output_path=str(out))
    zf = assert_valid_opc_package(out)
    media = [n for n in zf.namelist() if n.startswith("ppt/media/")]
    assert media, "raster variant should embed exactly one image"


def test_border_splats_keep_negative_offsets_end_to_end(tmp_path: Path):
    """Splats overlapping the left/top edge must survive the whole writer.

    Regression guard for the px_to_emu clamp that displaced every
    border-overlapping splat inward while keeping its full extent. The
    unit-level check covers the XML generator; this one covers the packaged
    file a user actually opens.
    """
    out = tmp_path / "border.pptx"
    edge = create_isotropic_splat(
        center=np.array([2.0, 3.0]),
        sigma=8.0,
        color=np.array([0.0, 1.0, 0.0]),
        alpha=0.8,
    )
    # 128px keeps emu_scale at 1.0 so the expected offsets below are exact;
    # the sub-96px scaling path is covered by its own tests.
    save_pptx_with_splats([edge], width=128, height=128, output_path=str(out))
    zf = assert_valid_opc_package(out)

    slide = zf.read("ppt/slides/slide1.xml").decode("utf-8")
    import re

    offsets = [
        (int(m.group(1)), int(m.group(2)))
        for m in re.finditer(r'<a:off x="(-?\d+)" y="(-?\d+)"/>', slide)
    ]
    assert any(
        x < 0 and y < 0 for x, y in offsets
    ), f"border splat was clamped into the slide: {offsets}"
    # rx = ELLIPSE_OVERLAP_BOOST(1.15) * k_sigma(2.5) * sigma(8) = 23px
    assert (px_to_emu(2.0 - 23.0), px_to_emu(3.0 - 23.0)) in offsets


def test_empty_splat_list_still_yields_valid_package(tmp_path: Path):
    """Degenerate input must not produce a corrupt deck."""
    out = tmp_path / "empty.pptx"
    save_pptx_with_splats([], width=32, height=32, output_path=str(out))
    assert_valid_opc_package(out)


@pytest.mark.skipif(
    shutil.which("openxml-audit") is None,
    reason="openxml-audit not installed; structural checks above still ran",
)
def test_openxml_audit_reports_no_findings(tmp_path: Path):
    """Authoritative external gate: 'PowerPoint expected to open cleanly'."""
    out = tmp_path / "audited.pptx"
    save_pptx_with_splats(_splats(12), width=48, height=48, output_path=str(out))
    proc = subprocess.run(
        ["openxml-audit", str(out)], capture_output=True, text=True, timeout=120
    )
    combined = proc.stdout + proc.stderr
    assert proc.returncode == 0, f"openxml-audit failed:\n{combined}"
    assert "Errors: 0" in combined, f"openxml-audit found problems:\n{combined}"


@pytest.mark.parametrize(
    "width,height,expect_scaled",
    [
        (476, 502, False),
        (1000, 800, False),
        (96, 96, False),
        (48, 48, True),
        (32, 64, True),
    ],
)
def test_slide_size_always_meets_ooxml_minimum(width, height, expect_scaled):
    """`sldSz` must never fall below one inch, at any input size.

    A sub-96px canvas would otherwise emit cx/cy under 914400 EMU, which
    openxml-audit reports as a schema violation ("may open, repair, or
    reject"). Normal-size images must be left at scale 1.0 exactly, so this
    guard cannot silently resize ordinary output.
    """
    from splatthis.io import MIN_SLIDE_EMU, pptx_emu_scale

    scale = pptx_emu_scale(width, height)
    assert (scale > 1.0) is expect_scaled
    if not expect_scaled:
        assert scale == 1.0

    assert px_to_emu(width, scale) >= MIN_SLIDE_EMU
    assert px_to_emu(height, scale) >= MIN_SLIDE_EMU


def test_small_canvas_scaling_is_uniform():
    """The minimum-size guard must scale everything by one factor, not warp it.

    Every shape offset on a sub-96px canvas must equal its unscaled value
    times the canvas scale factor — otherwise geometry drifts relative to
    the slide box and the composition shifts.
    """
    import re

    from splatthis.io import generate_drawingml_slide_content, pptx_emu_scale

    def offsets(xml):
        return [
            (int(a), int(b))
            for a, b in re.findall(r'<a:off x="(-?\d+)" y="(-?\d+)"/>', xml)
        ]

    splats = _splats(5)
    scale = pptx_emu_scale(48, 48)
    assert scale > 1.0, "48px must trigger the guard for this test to mean anything"

    scaled = offsets(
        generate_drawingml_slide_content(splats, width=48, height=48, k_sigma=2.5)
    )
    # Same splats on a canvas large enough to need no scaling.
    unscaled = offsets(
        generate_drawingml_slide_content(splats, width=128, height=128, k_sigma=2.5)
    )
    assert len(scaled) == len(unscaled)

    for (sx, sy), (ux, uy) in zip(scaled, unscaled):
        if (ux, uy) == (0, 0):  # slide-sized group/background boxes
            continue
        assert sx == pytest.approx(ux * scale, abs=2)
        assert sy == pytest.approx(uy * scale, abs=2)
