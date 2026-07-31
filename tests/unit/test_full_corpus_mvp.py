import importlib.util
from pathlib import Path

from PIL import Image

MODULE_PATH = Path(__file__).resolve().parents[2] / "tools" / "full_corpus_mvp.py"
SPEC = importlib.util.spec_from_file_location("full_corpus_mvp", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_powerpoint_slide_crop_box_uses_queried_retina_surface() -> None:
    crop = MODULE._powerpoint_slide_crop_box(
        screen_size=(2940, 1912),
        window_bounds=(0, 0, 1470, 956),
        menu_bar_height=33,
        slide_size=(384, 256),
    )

    assert crop == (84, 64, 2856, 1912)


def test_powerpoint_slide_crop_box_centers_wide_slide_vertically() -> None:
    crop = MODULE._powerpoint_slide_crop_box(
        screen_size=(1920, 1080),
        window_bounds=(0, 0, 1920, 1080),
        menu_bar_height=0,
        slide_size=(16, 9),
    )

    assert crop == (0, 0, 1920, 1080)


def test_powerpoint_matte_crop_box_removes_thin_asymmetric_margin() -> None:
    image = Image.new("RGB", (384, 256), (0, 0, 0))
    image.paste((120, 70, 30), (6, 8, 378, 256))

    crop = MODULE._powerpoint_matte_crop_box(image, (384, 256))

    assert crop == (6, 8, 378, 256)
