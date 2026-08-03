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
    load_population,        # path (.svg | .pptx | .json) -> List[GaussianSplat]
    population_from_svg,    # SVG markup -> List[GaussianSplat]
    population_from_pptx,   # .pptx path -> List[GaussianSplat]
    population_from_png,    # .png path -> List[GaussianSplat]
    png_population_chunk,   # -> PngInfo for Image.save(pnginfo=...)
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

**PNG** — a compressed `zTXt` text chunk keyed `splatthis:population`.
PNG decoders are required to skip chunks they do not recognise, so the
decoded pixels are byte-identical. This is the carrier that makes a
*render* self-describing: a preview or a shared screenshot can state the
fit it came from. When `--embed-population` is set, the pipeline's
preview PNG carries it automatically.

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

## Cost

Storing floats as floats rather than JSON numbers is what makes this cheap:

| encoding | 1,595 splats |
|---|---:|
| canonical JSON | 456 KB |
| minified JSON | 316 KB |
| **gzipped float32** | **59 KB** |

Per carrier, for a 1,595-splat population: about **4%** of a typical SVG,
**19%** of a 236 KB PNG preview, and a much larger share of a 120-160 KB
deck. Size is one reason all three are off by default; the other is that
the population is a derivative of the source image travelling inside a
file people share.

That is about **4% of a typical SVG** and a much larger fraction of a small
deck — PPTX artifacts are around 120–160 KB, so embedding is a far bigger
relative cost there. Both are off by default, and not only for size: the
population is a derivative of the source image, travelling inside a file
people share. That is a disclosure the user should choose.

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
- Any tool that strips PNG ancillary chunks (many optimisers do) drops
  the population from a preview.
- The PPTX part must be declared in `[Content_Types].xml`. It is, but the
  first version was not, and PowerPoint offered to repair the deck --
  while `openxml-audit` reported zero findings and the AppleScript
  capture harness returned success with a rendered image. Neither check
  detects this class of fault; a human opening the file did.
