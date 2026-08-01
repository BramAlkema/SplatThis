"""Human-readable artifact comparison reports."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from .storage import atomic_write_text
from .template_assets import render_template


def save_side_by_side_html(
    output_path: str,
    source_png_path: str,
    svg_path: str,
    preview_png_path: Optional[str] = None,
    title: str = "PNG2Splat Side-by-Side",
    metrics: Optional[Dict[str, Any]] = None,
) -> None:
    """Save a source, deployed-SVG, and proxy-preview comparison page."""

    output_dir = Path(output_path).resolve().parent

    def relative_reference(path_value: Optional[str]) -> str:
        if not path_value:
            return ""
        path = Path(path_value).resolve()
        try:
            return path.relative_to(output_dir).as_posix()
        except Exception:
            return path.as_posix()

    source_ref = relative_reference(source_png_path)
    svg_ref = relative_reference(svg_path)
    preview_ref = relative_reference(preview_png_path)
    svg_view = (
        f'<img src="{svg_ref}" alt="SVG Splats" />'
        if svg_ref
        else '<div style="padding:16px;color:#9ba5bc;">No SVG artifact for this run.</div>'
    )
    preview_view = (
        f'<img src="{preview_ref}" alt="Splat Preview PNG" />'
        if preview_ref
        else '<div style="padding:16px;color:#9ba5bc;">No preview PNG generated.</div>'
    )

    rows: List[str] = []
    for key, value in (metrics or {}).items():
        if isinstance(value, dict):
            rows.append(f"<tr><td colspan='2'><strong>{key}</strong></td></tr>")
            rows.extend(
                f"<tr><td>{key}.{sub_key}</td><td>{sub_value}</td></tr>"
                for sub_key, sub_value in value.items()
            )
        else:
            rows.append(f"<tr><td>{key}</td><td>{value}</td></tr>")
    metrics_table = (
        "<table>" + "".join(rows) + "</table>"
        if rows
        else "<p>No metrics recorded.</p>"
    )
    atomic_write_text(
        output_path,
        render_template(
            "reporting/side_by_side.html",
            title=title,
            source_ref=source_ref,
            svg_view=svg_view,
            preview_view=preview_view,
            metrics_table=metrics_table,
        ),
    )
