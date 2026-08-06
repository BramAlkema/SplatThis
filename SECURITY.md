# Security policy

Only the latest `0.3.x` release receives fixes. Report vulnerabilities through
GitHub's private security advisory form, not a public issue:

https://github.com/BramAlkema/SplatThis/security/advisories/new

SplatThis reads images through Pillow and writes SVG, PPTX, HTML, EML, and
optional JSON or PNG diagnostics. The default path performs no network I/O,
browser automation, Office automation, or archive extraction to disk. Canvas
HTML contains a local script that renders only the embedded numeric population.

Optional browser capture launches an installed Chrome/Chromium through
Playwright without network access. Optional population carriers may place a
compressed derivative of the source image in SVG, PPTX, or PNG metadata, and
the `steg` extra can alter PNG low bits to preserve that payload.

Optional PPTX capture launches Microsoft PowerPoint through macOS OSA and uses
the system screenshot utility. It needs Automation, Accessibility, and Screen
Recording permissions, brings PowerPoint to the foreground, and closes the
presentation it opened without saving. Use it only in a trusted desktop
session; it does not add a Python dependency.

In scope:

- unsafe handling of image paths or output paths;
- generated markup or mail headers that can escape their fixed structures;
- invalid or unsafe OOXML package paths;
- unsafe browser executable or capture output handling;
- malformed embedded population envelopes or excessive decompression;
- unexpected pixel damage outside the explicitly requested LSB carrier;
- crashes or resource exhaustion that bypass the explicit resolution and
  splat-count limits.

Dependency vulnerabilities in Pillow, NumPy, or Torch should be reported
upstream as well. If SplatThis prevents use of a fixed release, report that
here so the constraint can be changed.
