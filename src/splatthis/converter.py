"""Stable public converter API."""

from . import proxies as _proxies
from .conversion_engine import ConversionEngine

__all__ = ["PNG2SVGConverter"]

_PPTXSoftEdgeProxyRenderer = _proxies._PPTXSoftEdgeProxyRenderer


class PNG2SVGConverter(ConversionEngine):
    """Public PNG-to-splat converter with a backward-compatible API."""
