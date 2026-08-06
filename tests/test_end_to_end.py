import json
import xml.etree.ElementTree as ET
from pathlib import Path

from splatthis.cli import main

SOURCE = Path(__file__).parent / "assets" / "source.png"
MIN_SSIM_SRGB = 0.51
MIN_PSNR_SRGB = 18.0


def test_real_conversion_preserves_quality_and_svg_contract(tmp_path):
    output = tmp_path / "out.svg"
    artifacts = tmp_path / "artifacts"

    assert (
        main(
            [
                str(SOURCE),
                "-o",
                str(output),
                "--seed",
                "42",
                "--splats",
                "200",
                "--max-edge",
                "128",
                "--stages",
                "3,2",
                "--profile",
                "max-fidelity",
                "--artifacts-dir",
                str(artifacts),
            ]
        )
        == 0
    )

    manifest = json.loads((artifacts / "run_manifest.json").read_text())
    metrics = manifest["internal_metrics"]
    assert metrics["ssim_srgb"] >= MIN_SSIM_SRGB
    assert metrics["psnr_srgb"] >= MIN_PSNR_SRGB
    assert 100 <= manifest["final_splat_count"] <= 200

    root = ET.parse(output).getroot()
    namespace = "{http://www.w3.org/2000/svg}"
    ellipses = root.findall(f".//{namespace}ellipse")
    gradients = root.findall(f".//{namespace}radialGradient")
    assert len(ellipses) == manifest["final_splat_count"]
    assert len(gradients) == len(ellipses)
