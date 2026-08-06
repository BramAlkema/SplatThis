import tomllib
from pathlib import Path

import pytest

from splatthis import __version__
from splatthis.cli import main


def test_version(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code == 0
    assert capsys.readouterr().out.strip() == f"splatthis {__version__}"


def test_missing_input_is_a_clean_error(tmp_path, capsys):
    missing = tmp_path / "missing.png"
    assert main([str(missing)]) == 2
    captured = capsys.readouterr()
    assert captured.err == f"error: input not found: {missing}\n"


def test_default_output_is_svg(tmp_path):
    source = Path(__file__).parent / "assets" / "source.png"
    assert source.suffix == ".png"
    assert source.with_suffix(".svg").suffix == ".svg"


def test_runtime_dependency_budget_stays_small():
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    names = {requirement.split(">=")[0] for requirement in project["dependencies"]}

    assert project["version"] == __version__
    assert names == {"numpy", "Pillow", "torch"}
