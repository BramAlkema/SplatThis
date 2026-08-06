"""Atomic file publication used by the binary and XML exporters."""

from __future__ import annotations

import os
import stat
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


@contextmanager
def atomic_output_path(output_path: str | Path) -> Iterator[Path]:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    mode = stat.S_IMODE(target.stat().st_mode) if target.exists() else 0o644
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        yield temporary
        with temporary.open("rb+") as stream:
            os.fsync(stream.fileno())
        temporary.chmod(mode)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_text(
    output_path: str | Path, content: str, *, encoding: str = "utf-8"
) -> None:
    with atomic_output_path(output_path) as temporary:
        temporary.write_text(content, encoding=encoding)
