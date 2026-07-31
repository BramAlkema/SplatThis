"""Shared MLX import and Metal-device availability checks.

Importing :mod:`mlx.core` is not sufficient to prove that MLX can execute.
Headless, virtualized, or sandboxed macOS sessions can have the package
installed while exposing no Metal device.  Keep that distinction in one place
so the CLI can fall back to Torch before starting a long optimization run.
"""

from __future__ import annotations

from typing import Any

try:  # pragma: no cover - import depends on the host platform.
    import mlx.core as mx
except Exception:  # pragma: no cover - optional dependency guard.
    mx = None  # type: ignore[assignment]


def is_mlx_imported() -> bool:
    """Return whether the optional MLX package imported successfully."""

    return mx is not None


def is_mlx_available() -> bool:
    """Return whether MLX can execute on an available Metal device."""

    if mx is None:
        return False
    metal = getattr(mx, "metal", None)
    probe = getattr(metal, "is_available", None)
    if probe is None:
        # Older MLX releases did not expose the probe.  Preserve compatibility
        # and let the first operation provide the platform-specific error.
        return True
    try:
        return bool(probe())
    except Exception:
        return False


def require_mlx(feature: str) -> Any:
    """Return ``mlx.core`` or raise an actionable environment error."""

    if mx is None:
        raise RuntimeError(
            f"MLX is not installed. Install `splat-this[mlx]` to use {feature}."
        )
    if not is_mlx_available():
        raise RuntimeError(
            "MLX is installed but no Metal device is available. "
            f"{feature} requires an Apple-Silicon Metal session; use "
            "`--optimizer-backend torch` in headless, sandboxed, or virtualized "
            "environments."
        )
    return mx
