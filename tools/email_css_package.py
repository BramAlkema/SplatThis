#!/usr/bin/env python3
"""Package an email-safe CSS splat build as a ready-to-open .eml message.

``tools/email_css_mvp.py`` answers whether the build fits and what it costs.
This answers the next question: how it actually reaches an inbox.

No SMTP is involved. A ``.eml`` is a complete RFC 5322 message on disk, and
Apple Mail, Outlook and Thunderbird all open one by double-clicking, so the
same file that would be sent can be inspected in the real clients first --
which is the only way to check rendering, since no local capture tool models
a mail client.

Two things the browser measurement does not cover:

**Outlook on Windows renders through Microsoft Word.** It has no CSS
gradients, which is expected, but it also ignores ``position:absolute`` --
so without intervention 300 absolutely positioned splats reflow into a
single tall column and the message is worse than useless. The scene is
therefore wrapped so Word never sees it, and a plain coloured block with a
line of text takes its place.

**Gmail clips at roughly 102 KB.** That budget covers the whole message, not
just the splats, so the packaged size is reported against it rather than
assumed.

Usage::

    PYTHONPATH=src python tools/email_css_package.py --open
    PYTHONPATH=src python tools/email_css_package.py \
        --population tmp/email-css/art300/final.raw.json --splats 290
"""

from __future__ import annotations

import argparse
import json
import sys
from email.message import EmailMessage
from email.utils import formatdate, make_msgid
from pathlib import Path
from typing import Optional

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from splatthis.browser_export import generate_css_splat_html  # noqa: E402
from splatthis.color import linear_to_srgb  # noqa: E402
from splatthis.storage import load_splats_json  # noqa: E402

GMAIL_CLIP_BYTES = 102 * 1024
WORK = REPO / "tmp" / "email-css"


def _scene_fragment(html: str) -> str:
    """The scene element on its own, without the document wrapper."""
    start = html.index('<div id="scene"')
    end = html.rindex("</div>") + len("</div>")
    return html[start:end]


def _background_css(population: Path) -> str:
    manifest = population.parent / "run_manifest.json"
    default = "#20222a"
    if not manifest.is_file():
        return default
    value = (
        json.loads(manifest.read_text(encoding="utf-8"))
        .get("config", {})
        .get("background_linear_rgb")
    )
    if value is None:
        return default
    srgb = linear_to_srgb(np.clip(np.asarray(value, dtype=np.float32), 0.0, 1.0))
    return "#{:02x}{:02x}{:02x}".format(*(int(round(float(c) * 255)) for c in srgb))


def build_email_html(scene: str, width: int, height: int, backdrop: str) -> str:
    """Wrap the scene in email-grade markup with an Outlook fallback.

    The conditional comments are the standard pair: Word-based Outlook reads
    the ``[if mso]`` block and nothing else, every other client reads the
    ``[if !mso]`` block. Splitting them this way is what stops Outlook from
    reflowing hundreds of absolutely positioned elements into a column.
    """
    return (
        "<!doctype html>\n"
        '<html><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>Gaussian splats, drawn by your mail client</title></head>\n"
        '<body style="margin:0;padding:24px 0;background:#f4f4f6">\n'
        '<table role="presentation" width="100%" cellpadding="0" cellspacing="0" '
        'border="0" style="border-collapse:collapse"><tr>'
        '<td align="center" style="padding:0">\n'
        f'<table role="presentation" width="{width}" cellpadding="0" '
        'cellspacing="0" border="0" style="border-collapse:collapse">\n'
        "<tr><td>\n"
        # Outlook / Word: never show it the scene.
        "<!--[if mso]>\n"
        f'<table role="presentation" width="{width}" cellpadding="0" '
        'cellspacing="0" border="0"><tr>'
        f'<td height="{height}" align="center" valign="middle" '
        f'bgcolor="{backdrop}" '
        'style="color:#ffffff;font-family:Arial,sans-serif;font-size:14px">'
        "This message draws its picture with CSS gradients, which Outlook "
        "renders through Word and cannot display."
        "</td></tr></table>\n"
        "<![endif]-->\n"
        # Everything else: the real thing.
        "<!--[if !mso]><!-->\n"
        f"{scene}\n"
        "<!--<![endif]-->\n"
        "</td></tr>\n"
        '<tr><td style="padding:12px 2px 0;font-family:-apple-system,'
        "BlinkMacSystemFont,Segoe UI,Arial,sans-serif;font-size:12px;"
        'line-height:1.5;color:#5b5b66">'
        "Every shape above is a DOM element with a CSS radial gradient. "
        "No image was downloaded and no script ran."
        "</td></tr>\n"
        "</table>\n</td></tr></table>\n</body></html>\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population", default=str(WORK / "art300" / "final.raw.json"))
    parser.add_argument(
        "--splats",
        type=int,
        default=285,
        help=(
            "cap the population; 0 keeps all of it. The default leaves about "
            "3.5 KB under Gmail's clip for real copy -- 296 is the bare "
            "maximum and 300 already overruns once the wrapper is added"
        ),
    )
    parser.add_argument("--width", type=int, default=364)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--to", default="you@example.com")
    parser.add_argument("--sender", default="splatthis@localhost")
    parser.add_argument(
        "--subject", default="Gaussian splats, drawn by your mail client"
    )
    parser.add_argument("--out", default=str(WORK / "splats.eml"))
    parser.add_argument(
        "--open", action="store_true", help="open the .eml in the default mail client"
    )
    args = parser.parse_args()

    population = Path(args.population)
    if not population.is_file():
        print(f"error: no population at {population}", file=sys.stderr)
        return 2

    splats = load_splats_json(str(population))
    if args.splats:
        splats = splats[: args.splats]
    backdrop = _background_css(population)

    manifest = population.parent / "run_manifest.json"
    background_linear: Optional[np.ndarray] = None
    if manifest.is_file():
        value = (
            json.loads(manifest.read_text(encoding="utf-8"))
            .get("config", {})
            .get("background_linear_rgb")
        )
        if value is not None:
            background_linear = np.asarray(value, dtype=np.float32)

    scene = _scene_fragment(
        generate_css_splat_html(
            splats,
            width=args.width,
            height=args.height,
            background_linear_rgb=background_linear,
            email_safe=True,
        )
    )
    html = build_email_html(scene, args.width, args.height, backdrop)

    message = EmailMessage()
    message["Subject"] = args.subject
    message["From"] = args.sender
    message["To"] = args.to
    message["Date"] = formatdate(localtime=True)
    message["Message-ID"] = make_msgid(domain="splatthis.local")
    message.set_content(
        "This message renders "
        f"{len(splats)} Gaussian splats as CSS gradients. "
        "Your client is showing the plain-text part, which cannot.\n"
    )
    message.add_alternative(html, subtype="html")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    raw = message.as_bytes()
    out.write_bytes(raw)

    html_bytes = len(html.encode("utf-8"))
    print(f"wrote {out}")
    print(f"  splats            {len(splats)}")
    print(f"  HTML source       {html_bytes / 1024:6.1f} KB")
    print(f"  full message      {len(raw) / 1024:6.1f} KB  (headers + both parts)")
    verdict = "fits" if html_bytes <= GMAIL_CLIP_BYTES else "OVER -- Gmail will clip"
    print(f"  vs Gmail's 102 KB {verdict}")
    print(
        "\nOpen it in the clients that matter -- no capture tool models a mail\n"
        "renderer, so this is the only way to know. Outlook/Word gets the\n"
        "fallback block by design."
    )
    if args.open:
        import subprocess

        subprocess.run(["open", str(out)], check=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
