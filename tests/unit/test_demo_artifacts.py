"""Keep checked-in demo renders tied to their deployed artifacts."""

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity

REPO = Path(__file__).resolve().parents[2]
DEMO = REPO / "docs" / "demo"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_pixel_runtime_demo_render_matches_recorded_html_and_source() -> None:
    provenance = json.loads((DEMO / "canvas_render.json").read_text())
    assert provenance["render_kind"] == "pixel-runtime-buffer"
    assert provenance["is_deployed_artifact"] is True
    assert _sha256(DEMO / "canvas.html") == provenance["canvas_html_sha256"]
    assert _sha256(DEMO / "source.png") == provenance["source_png_sha256"]
    assert _sha256(DEMO / "canvas_render.png") == provenance["render_png_sha256"]

    source = np.asarray(Image.open(DEMO / "source.png").convert("RGB")) / 255.0
    rendered = np.asarray(Image.open(DEMO / "canvas_render.png").convert("RGB")) / 255.0
    assert rendered.shape == source.shape
    ssim = structural_similarity(source, rendered, channel_axis=2, data_range=1.0)
    assert ssim >= 0.88
