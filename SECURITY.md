# Security Policy

## Supported versions

SplatThis is pre-1.0 and published from `main`. Only the latest release on
PyPI receives fixes; there are no maintenance branches for older versions.

| Version | Supported |
|---|---|
| latest `0.2.x` | yes |
| anything earlier | no — upgrade |

## Reporting a vulnerability

Report privately through GitHub's
[security advisory form](https://github.com/BramAlkema/SplatThis/security/advisories/new),
not as a public issue. That opens a channel visible only to the maintainers.

Please include what you did, what happened, and the version
(`splatthis --version`). A reproducing image or `.pptx` helps more than a
description.

Expect an acknowledgement within a week. This is a single-maintainer research
project, not a funded product — if a fix requires work, it will be scheduled
openly rather than promised to a date.

## What is in scope

SplatThis reads untrusted images and writes SVG, HTML, PPTX and PNG. The
interesting boundaries are:

- **Image decoding** — an input that crashes, hangs, or reads out of bounds.
  Pillow and NumPy do the decoding, so upstream issues belong upstream; the
  handling around them is ours.
- **Generated markup** — SVG and HTML output embeds colours, dimensions and
  document titles. An input that escapes its context and injects markup or
  script into a generated artifact is a real finding. Vector markup lives in
  packaged templates under `src/splatthis/templates/`, and
  `tests/unit/test_module_boundaries.py` enforces that it stays there.
- **OOXML generation** — a `.pptx` is a ZIP. Path traversal in the package, or
  output that triggers PowerPoint's repair dialog, counts.
- **Archive and path handling** — anything that writes outside the directory
  the user pointed at.

## What is not in scope

- Rendering artifacts, colour drift, or quality regressions. Those matter, but
  they are correctness bugs; open a normal issue.
- Denial of service via deliberately enormous inputs or splat budgets. The
  converter is compute-bound by design and exposes `--splats`, `--max-edge`
  and `--time-budget` precisely so the caller sets the ceiling.
- Vulnerabilities in Torch, MLX, Pillow, or Chromium themselves. Report those
  to their maintainers; if SplatThis pins an affected version, tell us and we
  will bump it.

## Running untrusted input

The pipeline itself does no network I/O. The `capture` extra drives a local
Chrome through Playwright to grade browser artifacts, and the PPTX capture
tooling drives Microsoft PowerPoint — both open generated files in real
applications. If you are converting untrusted images at scale, run without the
`capture` extra so nothing generated is opened in a browser or Office.
