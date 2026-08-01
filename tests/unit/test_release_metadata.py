"""Release metadata must stay synchronized with the importable package."""

import tomllib
from pathlib import Path

from splatthis import __version__

REPO = Path(__file__).resolve().parents[2]


def test_package_version_matches_pyproject() -> None:
    metadata = tomllib.loads((REPO / "pyproject.toml").read_text())
    assert metadata["project"]["version"] == __version__


def test_capture_extra_owns_the_playwright_dependency() -> None:
    metadata = tomllib.loads((REPO / "pyproject.toml").read_text())
    assert any(
        dependency.startswith("playwright>=")
        for dependency in metadata["project"]["optional-dependencies"]["capture"]
    )
    assert "rasterize" not in metadata["project"]["optional-dependencies"]


def test_release_documents_exist() -> None:
    for filename in ("LICENSE", "CHANGELOG.md", "CONTRIBUTING.md", "README.md"):
        path = REPO / filename
        assert path.is_file()
        assert path.stat().st_size > 0


def test_py_typed_marker_is_present_and_shipped() -> None:
    """The PEP 561 marker must exist *and* be packaged.

    A marker that is not declared in package-data is silently absent from the
    wheel, and every consumer type-checking against the installed package sees
    ``Any`` for this API. That failure is invisible from inside the repo,
    where the source tree is on the path regardless.
    """
    import splatthis

    assert (Path(splatthis.__file__).parent / "py.typed").is_file()

    metadata = tomllib.loads((REPO / "pyproject.toml").read_text())
    assert "py.typed" in metadata["tool"]["setuptools"]["package-data"]["splatthis"]
    assert "src/splatthis/py.typed" in (REPO / "MANIFEST.in").read_text()
