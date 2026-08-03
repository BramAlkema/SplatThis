#!/usr/bin/env python3
"""Upload a .eml into a Gmail mailbox so it can be judged in Gmail itself.

Gmail's web UI has no import button, and forwarding the file as an
attachment is a weak test: attached messages render in a preview pane rather
than through the normal message path, so the sanitiser and the clipping
behaviour are not necessarily the ones a real message meets.

IMAP ``APPEND`` puts the message in the mailbox as if it had arrived, which
is what makes it a fair test of the two things this build actually needs
from Gmail: whether the CSS survives its allowlist, and whether it clips.

Credentials come from the environment, never from arguments -- a password on
the command line lands in shell history and in ``ps``::

    GMAIL_ADDRESS=you@gmail.com GMAIL_APP_PASSWORD=... \
        PYTHONPATH=src python tools/email_imap_append.py

Gmail requires an app-specific password (Google Account -> Security ->
2-Step Verification -> App passwords); an account password will be refused.
Nothing is sent to anyone: APPEND writes to your own mailbox only.
"""

from __future__ import annotations

import argparse
import imaplib
import os
import ssl
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_EML = REPO / "tmp" / "email-css" / "splats.eml"


def _load_dotenv(path: Path) -> None:
    """Fill in missing values from .env without overriding the environment."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eml", default=str(DEFAULT_EML))
    parser.add_argument("--mailbox", default="INBOX")
    parser.add_argument("--host", default="imap.gmail.com")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="check the file and the credentials without uploading",
    )
    args = parser.parse_args()

    _load_dotenv(REPO / ".env")
    address = os.environ.get("GMAIL_ADDRESS")
    password = os.environ.get("GMAIL_APP_PASSWORD")

    message = Path(args.eml)
    if not message.is_file():
        print(f"error: no message at {message}", file=sys.stderr)
        return 2
    raw = message.read_bytes()
    print(f"{message.name}: {len(raw) / 1024:.1f} KB")

    if not address or not password:
        print(
            "\nSet GMAIL_ADDRESS and GMAIL_APP_PASSWORD (in the environment or\n"
            ".env) to upload. Gmail needs an app-specific password:\n"
            "  Google Account -> Security -> 2-Step Verification -> App passwords\n"
            "\nWithout them, drag the file onto a Gmail mailbox in Apple Mail\n"
            "instead -- same IMAP APPEND, no credentials to store.",
            file=sys.stderr,
        )
        return 2

    if args.dry_run:
        print(f"dry run: would append to {args.mailbox} as {address}")
        return 0

    print(f"appending to {args.mailbox} as {address} ...")
    connection = imaplib.IMAP4_SSL(args.host, ssl_context=ssl.create_default_context())
    try:
        connection.login(address, password)
        status, detail = connection.append(
            args.mailbox, "", imaplib.Time2Internaldate(time.time()), raw
        )
    finally:
        try:
            connection.logout()
        except Exception:  # pragma: no cover - best-effort close
            pass

    if status != "OK":
        print(f"error: APPEND failed: {status} {detail!r}", file=sys.stderr)
        return 1
    print(
        f"done. Open {args.mailbox} in Gmail's web UI and check two things:\n"
        "  1. do the splats render, or does the CSS allowlist strip them\n"
        "  2. is there a 'Message clipped' link at the bottom"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
