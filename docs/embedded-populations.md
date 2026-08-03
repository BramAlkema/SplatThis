# Embedded populations

A deployed SVG or deck is the *result* of a fit. The population that produced
it — the Gaussians themselves — normally lives in a sidecar that never travels
with the file. `--embed-population` puts it inside the artifact, and
`--init-from` reads it back.

Three things follow from that. The artifact becomes **re-targetable** (emit a
deck from an SVG without the source image), **warm-startable** (a refit begins
where the last one finished), and **comparable** — the population is this
project's own answer at a stated splat budget, published so another fitter can
load the same file, try to beat it, and be checked against it.

## Command line

```bash
# Write the population into the artifact.
splatthis input.png -o out.svg  --format svg  --embed-population
splatthis input.png -o out.pptx --format pptx --embed-population

# Also hide it in the preview PNG's pixels, where metadata strippers
# cannot reach it. Needs the optional extra: pip install 'splatthis[steg]'
splatthis input.png -o out.svg --embed-population --embed-population-in-pixels

# Re-target an artifact to another format. No source image needed; the splat
# budget is pinned to the loaded population automatically.
splatthis input.png -o deck.pptx --format pptx --init-from out.svg --stages 1

# Warm-start a refit from a previous fit.
splatthis input.png -o better.svg --format svg \
  --init-from out.svg --stages 500,250,125
```

`--init-from` accepts an embedded-population SVG, `.pptx` or `.png`, or
the canonical `*.raw.json` that `--save-json` and `--artifacts-dir`
already write.

## Python API

```python
from splatthis import (
    encode_population,      # List[GaussianSplat] -> envelope text
    decode_population,      # envelope text -> List[GaussianSplat]
    load_population,        # path (.svg|.pptx|.png|.json) -> List[GaussianSplat]
    population_from_svg,    # SVG markup -> List[GaussianSplat]
    population_from_pptx,   # .pptx path -> List[GaussianSplat]
    population_from_png,    # .png path -> List[GaussianSplat] (chunk, else pixels)
    png_population_chunk,   # -> PngInfo for Image.save(pnginfo=...)
    embed_population_in_pixels,  # (RGB Image, splats) -> new carrier Image
    population_from_pixels,      # PIL Image -> List[GaussianSplat]
    steg_capacity_bytes,         # (Image, bits) -> payload bytes it can hold
    pptx_population_part,   # -> (part_name, text) for the packager
    POPULATION_FIELDS,      # column order of the encoded array
    POPULATION_SCHEMA,      # "splatthis.population/1"
)

splats = load_population("out.svg")
```

## Where it lives in the artifact

**SVG** — an SVG-standard `<metadata>` element holding the envelope. Renderers
ignore it, so the drawn markup is byte-identical with and without embedding
(asserted across all four SVG recipes). A comment would have been the obvious
alternative and is the wrong tool: `svgo`, which `--svg-optimize` runs, strips
comments, and a comment cannot contain `--`.

**PNG** — two carriers, and they are complements rather than alternatives.
The default is a compressed `zTXt` text chunk keyed `splatthis:population`;
PNG decoders must skip chunks they do not recognise, so the decoded pixels
stay byte-identical. This is what makes a *render* self-describing: a
preview or a shared screenshot can state the fit it came from. When
`--embed-population` is set, the pipeline's preview PNG carries it
automatically.

`--embed-population-in-pixels` adds a second copy in the low bits of the
pixels themselves. A chunk is metadata and metadata gets removed; pixels
are the image. See [Surviving sanitisation](#surviving-sanitisation) for
what each one actually withstands, and [Cost](#cost) for what the second
one charges you.

**PPTX** — an unreferenced package part at `splatthis/population.json`. OOXML
packages are ZIPs, so this needs no XML surgery. PowerPoint ignores parts
nothing relates to; the deck opens normally, verified by a real slideshow
capture rather than by structural validation alone. Note that a Save As which
rewrites the package will drop the part — the population survives
distribution, not editing.

## The envelope

Deliberately self-describing rather than a dump of this project's classes, so
that decoding it requires nothing from this project:

```python
import base64, gzip, json, numpy as np

envelope = json.loads(metadata_text)          # or the .pptx part's bytes
buffer = gzip.decompress(base64.b64decode(envelope["data"]))
splats = np.frombuffer(buffer, dtype=envelope["dtype"])
splats = splats.reshape(envelope["count"], len(envelope["fields"]))
# columns are envelope["fields"], in that order
```

| key | meaning |
|---|---|
| `schema` | `splatthis.population/1` |
| `count` | number of splats |
| `fields` | column names, in array order |
| `dtype` | `float32` |
| `layout` | `row-major` |
| `encoding` | `gzip+base64` |
| `data` | the payload |

Units: positions and sigmas in pixels of the artifact's own coordinate system,
`theta` in radians, colour linear-sRGB in `[0, 1]`, `importance` fixing
composite order (ascending, back to front).

## Surviving sanitisation

Measured on one 384×384 PNG carrying both, by running each tool and asking
whether the population still decodes:

| what it went through | `zTXt` chunk | in-pixels |
|---|---|---|
| nothing | survives | survives |
| `oxipng -o4 --strip safe` | **lost** | survives |
| `oxipng -o4 --strip all` | **lost** | survives |
| `exiftool -all=` | **lost** | survives |
| re-saved through PIL | **lost** | survives |
| ImageMagick re-encode | survives | survives |
| ImageMagick `-resize 50%` | survives | **lost** |
| JPEG q95 round-trip | **lost** | **lost** |

Two things fall out of this. Metadata is lost far more easily than expected
— not just to deliberate stripping, but to *any* tool that opens the file
and writes it back, which most pipelines do somewhere. And the two carriers
fail on opposite inputs: geometry changes destroy low bits and preserve
chunks; metadata handling does the reverse. So both are written when both
are asked for, and `population_from_png()` reads whichever is left without
the caller needing to know. Nothing survives a lossy re-encode, and nothing
here is a security property — the payload is hidden from a stripper, not
from an adversary.

## Cost

Storing floats as floats rather than JSON numbers is what makes this cheap:

| encoding | 1,595 splats |
|---|---:|
| canonical JSON | 456 KB |
| minified JSON | 316 KB |
| **gzipped float32** | **59 KB** |

Per carrier, for a 1,595-splat population: about **4%** of a typical SVG,
**19%** of a 236 KB PNG preview, and a much larger share of a 120-160 KB
deck. Size is one reason all of them are off by default; the other is that
the population is a derivative of the source image travelling inside a
file people share.

### What the in-pixel carrier charges

Two separate bills, and the surprise is which one is larger.

Fidelity, source versus carrier, scored the way this project scores
anything — 1 bit per channel where the payload fits, 2 where it does not:

| image | size | bits | LPIPS | SSIM | ΔE mean |
|---|---|---:|---:|---:|---:|
| astronaut | 384×384 | 2 | 0.0014 | 0.9906 | 0.0035 |
| brick | 384×384 | 2 | 0.0012 | 0.9914 | 0.0026 |
| camera | 384×384 | 1 | 0.0014 | 0.9959 | 0.0022 |
| chameleon | 364×384 | 2 | 0.0039 | 0.9880 | 0.0030 |
| cell | 320×384 | 2 | 0.0856 | 0.9803 | 0.0038 |

On photographs this is nothing: 0.001–0.004 LPIPS against a deployed
figure near 0.42. `cell` is the case worth understanding rather than
footnoting — at 4× zoom it shows no banding and no structure, just uniform
grain, but it is a dark, nearly flat field, so ±3/255 is a large *relative*
perturbation and LPIPS is right to charge for it. Expect the cost to track
how little local contrast an image has, not how large it is.

File size is the bigger bill, because LSB noise is incompressible and PNG
was compressing those bits before:

| image | clean | +chunk | +pixels |
|---|---:|---:|---:|
| astronaut | 235 KB | 280 KB (1.19×) | 243 KB (1.03×) |
| chameleon | 166 KB | 216 KB (1.30×) | 196 KB (1.18×) |
| camera | 121 KB | 156 KB (1.29×) | 209 KB (1.73×) |
| cell | 44 KB | 90 KB (2.07×) | 120 KB (2.74×) |
| checkerboard | 1 KB | 32 KB (49.9×) | — |

On busy images the pixel carrier can undercut the chunk, since the payload
partly displaces entropy PNG was already paying for. On compressible ones
it is worse, and on a synthetic flat image the payload simply dwarfs the
picture — a 1 KB checkerboard cannot sensibly carry 41 KB by any route.

The depth ladder therefore stops at 2 bits. A 200×200 checkerboard only
takes the payload at 4 bits, which costs 0.19 SSIM and 0.047 ΔE: visible
damage. Escalating there automatically would turn "this does not fit" into
"your picture was quietly mangled", so `embed_population_in_pixels` raises
and names the chunk carrier instead. Depth 4 stays reachable by passing
`bits_per_channel=4` explicitly.

## Known limits

- `--stages 0` is rejected by the schedule validator, so a genuinely
  zero-training re-target cannot be expressed yet; `--stages 1` is the
  practical equivalent.
- The warm-start benefit is real but not cleanly isolated. Warm-starting
  astronaut at half the default schedule reached SSIM 0.7586 in 3.3 minutes
  against a cold 0.7030 in 6.6 — but densification still ran, so the warm run
  finished with 1,913 splats against 1,617, and some of that gain is the
  splat-count lever rather than the warm start. Pinning `--splats` would
  settle it. (Re-targets pin automatically; refits do not, since growing the
  population is usually the point.)
- A PowerPoint Save As drops the embedded part.
- Any tool that strips PNG ancillary chunks drops the population from a
  preview -- and as the table above shows, so does merely re-saving the
  file. `--embed-population-in-pixels` is the answer to that; nothing
  answers a lossy re-encode.
- The in-pixel carrier needs an RGB image. `RGBA` is refused rather than
  converted, because hiding bits in an alpha channel loses them to the
  next composite, and `L`/`P` crash inside the packing library.
- LSB embedding is delegated to `stego-lsb` (MIT), not hand-rolled. An
  off-by-one in bit packing corrupts payloads silently, which is the
  worst failure mode available here.
- The PPTX part must be declared in `[Content_Types].xml`. It is, but the
  first version was not, and PowerPoint offered to repair the deck --
  while `openxml-audit` reported zero findings and the AppleScript
  capture harness returned success with a rendered image. Neither check
  detects this class of fault; a human opening the file did.
