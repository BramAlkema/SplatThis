"""Package the email-safe CSS compositor as a self-contained EML message."""

from __future__ import annotations

from email import policy
from email.message import EmailMessage
from email.utils import formatdate, make_msgid
from pathlib import Path
from typing import Optional

import numpy as np
import numpy.typing as npt

from .browser_export import generate_css_splat_html
from .color import linear_to_srgb
from .splat import GaussianSplat
from .storage import atomic_output_path

GMAIL_CLIP_BYTES = 102 * 1024
DEFAULT_EMAIL_SPLAT_LIMIT = 285


def _scene_fragment(document: str) -> str:
    start = document.index('<div id="scene"')
    end = document.rindex("</div>") + len("</div>")
    return document[start:end]


def _background_css(background_linear_rgb: Optional[npt.NDArray[np.floating]]) -> str:
    if background_linear_rgb is None:
        return "#20222a"
    linear = np.asarray(background_linear_rgb, dtype=np.float32).reshape(-1)
    if linear.size != 3:
        raise ValueError("background_linear_rgb must have exactly 3 components")
    srgb = linear_to_srgb(np.clip(linear, 0.0, 1.0))
    return "#{:02x}{:02x}{:02x}".format(
        *(int(round(float(channel) * 255)) for channel in srgb)
    )


def build_email_html(scene: str, width: int, height: int, backdrop: str) -> str:
    """Wrap an inline-CSS scene with a Word/Outlook fallback."""
    return (
        "<!doctype html>\n"
        '<html><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<meta name="color-scheme" content="light dark">'
        '<meta name="supported-color-schemes" content="light dark">'
        "<style>:root{color-scheme:light dark;supported-color-schemes:light dark}"
        "@media (prefers-color-scheme:dark){.page{background:#1c1c1e!important}"
        ".caption{color:#9a9aa4!important}}</style>"
        "<title>Gaussian splats, drawn by your mail client</title></head>\n"
        '<body class="page" style="margin:0;padding:24px 0;background:#f4f4f6">'
        '<table role="presentation" width="100%" cellpadding="0" cellspacing="0" '
        'border="0" style="border-collapse:collapse"><tr>'
        '<td align="center" style="padding:0">'
        f'<table role="presentation" width="{width}" cellpadding="0" '
        'cellspacing="0" border="0" style="border-collapse:collapse">'
        "<tr><td>"
        "<!--[if mso]>"
        f'<table role="presentation" width="{width}" cellpadding="0" '
        'cellspacing="0" border="0"><tr>'
        f'<td height="{height}" align="center" valign="middle" '
        f'bgcolor="{backdrop}" style="color:#ffffff;font-family:Arial,sans-serif;'
        'font-size:14px">This picture uses CSS gradients, which Word-based '
        "Outlook cannot display.</td></tr></table>"
        "<![endif]-->"
        "<!--[if !mso]><!-->"
        f"{scene}"
        "<!--<![endif]-->"
        "</td></tr>"
        '<tr><td class="caption" style="padding:12px 2px 0;font-family:'
        "-apple-system,BlinkMacSystemFont,Segoe UI,Arial,sans-serif;font-size:12px;"
        'line-height:1.5;color:#5b5b66">Every shape above is an HTML element '
        "with an inline CSS radial gradient. No image was downloaded and no script ran."
        "</td></tr></table></td></tr></table></body></html>\n"
    )


def generate_css_email_message(
    splats: list[GaussianSplat],
    width: int,
    height: int,
    *,
    background_linear_rgb: Optional[npt.NDArray[np.floating]] = None,
    subject: str = "Gaussian splats, drawn by your mail client",
    sender: str = "splatthis@localhost",
    recipient: str = "you@example.com",
    max_splats: int = DEFAULT_EMAIL_SPLAT_LIMIT,
) -> bytes:
    """Return an RFC 5322 message containing an inline, script-free CSS scene."""
    selected = list(splats)
    if max_splats > 0:
        selected = selected[:max_splats]
    document = generate_css_splat_html(
        selected,
        width=width,
        height=height,
        background_linear_rgb=background_linear_rgb,
        email_safe=True,
    )
    html = build_email_html(
        _scene_fragment(document),
        int(width),
        int(height),
        _background_css(background_linear_rgb),
    )
    if len(html.encode("utf-8")) > GMAIL_CLIP_BYTES:
        raise ValueError(
            "email HTML exceeds Gmail's 102 KB clipping threshold; lower max_splats"
        )

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = sender
    message["To"] = recipient
    message["Date"] = formatdate(localtime=True)
    message["Message-ID"] = make_msgid(domain="splatthis.local")
    message.set_content(
        f"This message renders {len(selected)} Gaussian splats as CSS gradients. "
        "Your client is showing the plain-text fallback.\n"
    )
    message.add_alternative(html, subtype="html")
    return message.as_bytes(policy=policy.SMTP)


def save_css_email(
    splats: list[GaussianSplat],
    width: int,
    height: int,
    output_path: str,
    **message_options: object,
) -> None:
    raw = generate_css_email_message(
        splats,
        width,
        height,
        **message_options,
    )
    with atomic_output_path(Path(output_path)) as temporary:
        temporary.write_bytes(raw)
