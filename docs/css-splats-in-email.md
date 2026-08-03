# CSS splats in email

Measured 2026-08-03/04 on chameleon, governing native-size Playwright
Chromium captures, plus one real Gmail render and one real Apple Mail
render. Tools: `tools/email_css_mvp.py` (fit, emit, score),
`tools/email_css_package.py` (package as `.eml`),
`tools/email_imap_append.py` (get it into Gmail).

Mail clients block remote images by default, which is the usual reason an
email arrives blank. A CSS splat build downloads nothing and runs nothing —
it is DOM elements with background gradients — so the question is whether
that survives a mail client. Two constraints decide it, and neither is the
gradients.

## Gmail's size clip sets the splat budget

Gmail truncates a message beyond roughly **102 KB of HTML source**, measured
on the source rather than the transfer encoding: base64 buys about 5 KB of
slack, quoted-printable none. Certain special characters trigger it
regardless of size.

The shipping CSS recipe costs 534 B/splat, so it fits 194 splats — and the
recipe leans on three things mail clients do not have. `email_safe=True` on
`generate_css_splat_html` drops all three:

| shipping recipe | email-safe | why |
|---|---|---|
| shared `<style>` block | every declaration inlined | Gmail's app strips `<style>` for non-Gmail accounts |
| `mask-image` over a solid fill | colour folded into the gradient's own stops | no mail client has CSS masking |
| `color(srgb-linear …)` | legacy `rgb()` | CSS Color 4 is too new even for browsers |

That costs 348 B/splat, so **299 splats** fit — 285 by default, leaving
about 3.5 KB for the copy a real message carries.

Dropping the mask is the interesting one, because the shipping recipe uses
it deliberately: painting colour through a gradient makes the browser
interpolate colour and opacity together, which darkens each splat's skirt.
It does not measurably, because every stop carries the same `rgb` and varies
only alpha, so nothing interpolates toward black.

| recipe | splats | HTML | fits Gmail | SSIM | LPIPS |
|---|---:|---:|:---:|---:|---:|
| shipping | 1,615 | 842 KB | no | 0.875 | 0.146 |
| shipping | 300 | 157 KB | no | 0.746 | 0.389 |
| email-safe | 300 | 102 KB | yes | 0.744 | 0.386 |

So the recipe is quality-neutral — LPIPS −0.0029, SSIM −0.0017, both inside
noise — at 35% fewer bytes. At 300 splats the image is soft but correct.

## Gmail's CSS allowlist sets the layout

The first packaged message arrived in Gmail as a bare backdrop with nothing
on it. Diffing what was sent against what Gmail rendered:

| | declarations |
|---|---|
| **kept** | `border-radius`, `width`, `height`, `background:radial-gradient` |
| **stripped** | `position`, `left`, `top`, `transform` |

The gradients were never at risk. The failure is second-order: with
`position` gone, an inline `<i>` also stops honouring `width` and `height`,
so all 285 splats collapsed to zero size.

The fix uses only what survives. Block `<div>`s in normal flow, offset by
`margin-left` and a `margin-top` delta from the previous element's bottom
edge. Sibling margins collapse against a zero `margin-bottom` to exactly
the requested value, negative included, so arbitrary 2D placement is
expressible in margins alone. `rotate()` stays, minus the `translate` the
absolute version needed: a client that honours transforms gets the fitted
orientation, one that strips it still gets the splat in the right place.

**Rounding has to be tracked, not just emitted.** Each margin is relative to
the previous element, so error compounds down the chain instead of staying
under half a pixel per splat. Emitting rounded margins while tracking ideal
positions cost 0.085 SSIM before the emitter started tracking the rounded
position it had actually written.

Scored with Gmail's four declarations stripped, which reproduces its output:

| | SSIM | LPIPS |
|---|---:|---:|
| transform kept (Apple Mail, Chromium) | 0.7442 | 0.3882 |
| transform stripped (Gmail) | 0.6670 | 0.5328 |

0.7442 matches the absolute layout's 0.7441, so nothing was lost where
transforms work. Gmail goes from rendering nothing to rendering a
recognisable image — but not a good one. Without rotation every anisotropic
splat becomes a horizontal bar, visible as streaking. The splats land in the
right places with the right colours; only orientation is lost.

## Apple Mail renders it, and then repaints it

Apple Mail draws all 285 gradients. In dark mode it also applies its own
partial inversion to any message that has not declared a colour scheme, and
a picture made entirely of background colours is made entirely of the thing
it inverts — so it arrives washed out rather than cleanly flipped.

`<meta name="color-scheme">` plus `supported-color-schemes` says those
colours are deliberate, which they are: they are a fit, not a theme. Only
the page chrome follows `prefers-color-scheme`; the scene keeps its fitted
backdrop in both, since re-tinting it would make the message a different
image from the one that was measured. Both chrome colours stay inline too,
so a stripped `<style>` costs the dark variant and nothing else. The
backdrop is off-white rather than `#ffffff`, because pure white and pure
black are what clients invert hardest whatever the declared scheme.

## Outlook would be worse than degraded

Outlook on Windows renders through Microsoft Word. It has no CSS gradients,
which is expected — but it also ignores `position:absolute`, so several
hundred positioned splats reflow into one enormous vertical column. The
scene therefore sits behind the standard `<!--[if !mso]><!-->` conditional
and Word gets a plain coloured block with a line of text.

## Getting one into a client

`tools/email_css_package.py` writes a complete RFC 5322 `.eml`. That is
deliberate rather than an SMTP send: Apple Mail, Outlook and Thunderbird all
open one by double-clicking, so rendering can be checked in the clients
themselves, which no local capture models. Verified by pulling the
`text/html` part back out and re-rendering it — unchanged by the
quoted-printable round-trip.

Gmail has no import button and will not take a `.eml` directly. Forwarding
it as an attachment is a weak test, since attached messages render in a
preview pane rather than through the normal message path, so neither the
allowlist nor the clipping behaviour is the one a real message meets. IMAP
`APPEND` puts it in the mailbox as if it had arrived: either drag the file
onto a Gmail mailbox in Apple Mail, or run `tools/email_imap_append.py`
with an app-specific password.

## Known limits

- **Gmail needs an isotropic fit to look right.** Rotation is unavailable
  there at any price, so the way to recover the 0.077 SSIM is to fit
  circular splats, which need no rotation. That is a training change, not
  an emitter one, and it is untested.
- Outlook/Word cannot render this at all; the fallback block is the answer.
- No local tool models a mail client, so every number above that is not a
  Chromium capture came from opening the message by hand. The Gmail column
  is a *simulation* — Chromium with Gmail's four declarations removed —
  validated against one real Gmail render, not a capture of Gmail itself.
- Only chameleon was measured. The budget is bytes-per-splat, which is
  content-independent, but the quality at 285 splats is not.
- Sizes assume the scene is the whole message. Real copy, a header and an
  unsubscribe link all come out of the same 102 KB.
