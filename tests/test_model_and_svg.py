import xml.etree.ElementTree as ET

import numpy as np

from splatthis.io import load_splats_json, save_splats_json, save_svg
from splatthis.splat import GaussianSplat, RawSplat


def _splat() -> GaussianSplat:
    return GaussianSplat.from_raw_splat(
        RawSplat(
            x=8,
            y=7,
            sx=3,
            sy=1.5,
            theta=0.4,
            r=0.2,
            g=0.5,
            b=0.8,
            a=0.75,
            importance=0.3,
        )
    )


def test_raw_json_roundtrip(tmp_path):
    path = tmp_path / "splats.json"
    save_splats_json([_splat()], str(path))
    loaded = load_splats_json(str(path))

    assert len(loaded) == 1
    assert np.allclose(loaded[0].mu, [8, 7])
    assert np.allclose(loaded[0].color, [0.2, 0.5, 0.8])


def test_svg_is_static_and_self_contained(tmp_path):
    path = tmp_path / "nested" / "out.svg"
    save_svg([_splat()], 16, 12, str(path), background_linear_rgb=np.zeros(3))

    root = ET.parse(path).getroot()
    namespace = "{http://www.w3.org/2000/svg}"
    assert root.tag == f"{namespace}svg"
    assert len(root.findall(f".//{namespace}ellipse")) == 1
    assert len(root.findall(f".//{namespace}radialGradient")) == 1

    markup = path.read_text(encoding="utf-8")
    assert "<script" not in markup
    assert "http://" not in markup.replace("http://www.w3.org/2000/svg", "")
    assert "https://" not in markup
