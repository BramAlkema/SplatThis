"""Cached package-template loading with explicit token substitution."""

from __future__ import annotations

import re
from functools import lru_cache
from importlib.resources import files
from typing import Any

_TOKEN = re.compile(r"@@([A-Z0-9_]+)@@")


@lru_cache(maxsize=None)
def load_template(name: str) -> str:
    """Read one UTF-8 template from the installed package once."""

    path = files("splatthis.templates").joinpath(name)
    return path.read_text(encoding="utf-8")


def render_template(name: str, /, **values: Any) -> str:
    """Replace explicit template tokens and reject missing substitutions."""

    rendered = load_template(name)
    for key, value in values.items():
        rendered = rendered.replace(f"@@{key.upper()}@@", str(value))
    missing = sorted(set(_TOKEN.findall(rendered)))
    if missing:
        raise ValueError(f"Missing template values for {name}: {', '.join(missing)}")
    return rendered


def render_template_lines(name: str, /, **values: Any) -> list[str]:
    """Render a fragment template as lines for existing list-based emitters."""

    return render_template(name, **values).splitlines()
