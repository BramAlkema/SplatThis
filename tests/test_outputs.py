from email import policy
from email.parser import BytesParser
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest

from splatthis.browser_export import (
    generate_css_splat_html,
    generate_native_canvas_html,
)
from splatthis.email_export import (
    GMAIL_CLIP_BYTES,
    generate_css_email_message,
    save_css_email,
)
from splatthis.pptx_export import save_pptx_with_splats
from splatthis.splat import GaussianSplat, RawSplat


def _splat(index: int = 0) -> GaussianSplat:
    return GaussianSplat.from_raw_splat(
        RawSplat(
            x=8 + index,
            y=7,
            sx=3,
            sy=1.5,
            theta=0.4,
            r=0.2,
            g=0.5,
            b=0.8,
            a=0.75,
            importance=0.3 + index / 100,
        )
    )


def test_css_html_is_scriptless_and_self_contained():
    html = generate_css_splat_html([_splat()], 16, 12, np.zeros(3))

    assert 'data-compositor="css-splats"' in html
    assert "radial-gradient" in html
    assert "mask-image" in html
    assert "<script" not in html
    assert "<canvas" not in html
    assert "<svg" not in html
    assert "https://" not in html


def test_canvas_html_draws_native_gradients_without_pixel_payload():
    html = generate_native_canvas_html([_splat()], 16, 12, np.zeros(3))

    assert 'data-compositor="canvas-api-splats"' in html
    assert "createElement('canvas')" in html
    assert "createRadialGradient" in html
    assert "ImageData" not in html
    assert "data:image" not in html
    assert "https://" not in html


def test_pptx_contains_editable_shapes_without_embedded_bitmap(tmp_path):
    output = tmp_path / "out.pptx"
    save_pptx_with_splats(
        [_splat()],
        16,
        12,
        str(output),
        background_linear_rgb=np.zeros(3),
    )

    with ZipFile(output) as archive:
        names = set(archive.namelist())
        slide = archive.read("ppt/slides/slide1.xml").decode("utf-8")

    assert "[Content_Types].xml" in names
    assert "ppt/presentation.xml" in names
    assert "ppt/slides/slide1.xml" in names
    assert not any(name.startswith("ppt/media/") for name in names)
    assert "Splat " in slide
    assert "radialGradient" not in slide


def test_eml_contains_email_safe_css_and_plain_fallback(tmp_path):
    output = tmp_path / "out.eml"
    save_css_email(
        [_splat()],
        16,
        12,
        str(output),
        background_linear_rgb=np.zeros(3),
        subject="Test splats",
        sender="from@example.com",
        recipient="to@example.com",
    )

    message = BytesParser(policy=policy.default).parsebytes(output.read_bytes())
    html = message.get_body(preferencelist=("html",)).get_content()
    plain = message.get_body(preferencelist=("plain",)).get_content()

    assert message["Subject"] == "Test splats"
    assert message.get_content_type() == "multipart/alternative"
    assert "plain-text fallback" in plain
    assert 'data-compositor="css-splats-email"' in html
    assert "radial-gradient" in html
    assert "mask-image" not in html
    assert "position:absolute" not in html
    assert "<script" not in html
    assert "https://" not in html
    assert len(html.encode("utf-8")) < GMAIL_CLIP_BYTES


def test_default_email_population_stays_below_clipping_guard():
    raw = generate_css_email_message(
        [_splat(index) for index in range(285)],
        320,
        180,
        background_linear_rgb=np.zeros(3),
    )
    message = BytesParser(policy=policy.default).parsebytes(raw)
    html = message.get_body(preferencelist=("html",)).get_content()

    assert len(html.encode("utf-8")) < GMAIL_CLIP_BYTES


@pytest.mark.parametrize(
    ("output_format", "suffix"),
    [
        ("svg", ".svg"),
        ("pptx", ".pptx"),
        ("canvas", ".html"),
        ("css", ".html"),
        ("eml", ".eml"),
    ],
)
def test_cli_routes_each_format_to_the_shared_converter(
    tmp_path, monkeypatch, output_format, suffix
):
    from splatthis import cli

    source = tmp_path / "source.png"
    source.write_bytes((Path(__file__).parent / "assets" / "source.png").read_bytes())
    captured = {}

    class FakeConverter:
        def __init__(self, **options):
            captured["init"] = options

        def convert(self, input_path, output_path, **options):
            captured["input"] = input_path
            captured["output"] = output_path
            captured["convert"] = options
            return output_path

    monkeypatch.setattr(cli, "PNG2SVGConverter", FakeConverter)

    assert cli.main([str(source), "--format", output_format]) == 0
    assert Path(captured["output"]).suffix == suffix
    assert captured["convert"]["output_format"] == output_format
