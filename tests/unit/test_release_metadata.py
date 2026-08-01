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


def test_every_packaged_file_is_tracked_by_git() -> None:
    """A packaged asset present only in the working tree is invisible locally.

    ``git add`` obeys ``.gitignore`` silently, so a packaged file matching an
    ignore rule is simply never committed. Tests still pass, wheels built from
    a dirty tree still work, and only a fresh checkout — a CI run, or a release
    build — is missing it. This happened to all 19 packaged SVG templates,
    which the blanket ``*.svg`` rule swallowed.
    """
    import subprocess

    package_dir = REPO / "src" / "splatthis"

    on_disk = {
        path.relative_to(REPO).as_posix()
        for path in package_dir.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }

    result = subprocess.run(
        ["git", "ls-files", "src/splatthis"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return  # not a git checkout (e.g. running from an unpacked sdist)

    tracked = {line for line in result.stdout.splitlines() if line}
    assert on_disk - tracked == set(), "packaged files missing from git"


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
