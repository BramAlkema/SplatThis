# SVG & DrawingML Gaussian / Feather Trick Catalog

A reference for which native primitives in each format we could use to
render our trained Gaussian splats with higher fidelity than the current
"piecewise-linear gradient stops" approximation. Written 2026-05-24 after
hitting an SSIM ceiling around 0.55–0.83 on splat-heavy content.

The headline insight: **both formats have a true Gaussian blur primitive**.
We could approximate each splat as a small flat-color shape blurred by
the format's native blur instead of approximating the Gaussian by hand
with 3-5 gradient stops. Anywhere this is feasible, the per-splat
fidelity goes from "piecewise linear" to "true Gaussian", which directly
attacks the artifact we're seeing.

The catalog also covers feathering, masking, and compositing tricks we
haven't yet evaluated.

---

## SVG primitives that affect opacity / feathering

| Primitive | Where it lives | What it produces | Splat applicability |
|---|---|---|---|
| `<radialGradient>` with stops | `<defs>` | Piecewise-linear opacity curve along radius. | **Current path.** Cheap, universal. Hard ceiling on Gaussian fidelity. |
| `<linearGradient>` with stops | `<defs>` | Piecewise-linear opacity along an axis. | Not directly useful for radial splats. |
| **`<feGaussianBlur stdDeviation="σx σy"/>`** in `<filter>` | `<defs>` | **True Gaussian blur of input pixels.** Anisotropic supported. | **Highly promising.** A flat-color ellipse + filter = true Gaussian. Filter cost is the concern. Need shared filters via sigma quantization (~32 buckets). |
| `<feSpecularLighting>` / `<feDiffuseLighting>` | inside `<filter>` | Light-source lighting model. | Not splat-relevant. |
| `<feComponentTransfer>` with `<feFuncA>` | inside `<filter>` | Arbitrary lookup-table opacity curve. | Could shape a true-Gaussian curve over a square. Combined with `<feGaussianBlur>` is overkill. |
| `<feConvolveMatrix kernelMatrix="...">` | inside `<filter>` | Custom convolution kernel. | We could write the literal Gaussian kernel. Slower than `feGaussianBlur` and same result. |
| `<feDropShadow>` | inside `<filter>` | Soft drop shadow (blur + offset + color). | Not splat-relevant. |
| `<feMorphology operator="dilate|erode">` | inside `<filter>` | Expand or contract shape. | Could pre-grow shapes before blurring; usually unnecessary. |
| `<feOffset>` | inside `<filter>` | Translate the input. | Not splat-relevant by itself. |
| `<feMerge>` | inside `<filter>` | Composite multiple inputs in order. | Composing stages of a filter graph. |
| `<feComposite operator="...">` | inside `<filter>` | Alpha-over / in / out / xor / arithmetic compositing. | If we move all splats inside one filter, we control the inter-splat blend math here. |
| `<feFlood>` | inside `<filter>` | Solid-color fill. | Plus a small alpha-mask shape, then blur, is one way to emit a Gaussian. |
| `<feImage>` | inside `<filter>` | Reference an external/internal SVG image. | Could host the splat-data table as a single image. |
| `<filter color-interpolation-filters="linearRGB">` | filter attribute | Forces filter math to happen in linear-light, not sRGB display. | **Useful.** Pairs cleanly with the linear-light optimizer training, avoiding the train→deploy gap. |
| `<mask>` with gradient | `<defs>` | Per-element opacity map from any drawable content. | Could reference one shared mask (a Gaussian texture) per splat instance with translation/scale via `<use>`. Performance unknown at 20k shapes. |
| `<clipPath>` | `<defs>` | Hard boundary. | Useful as an outer cull to prevent splat bleed beyond bbox. |
| `<pattern>` | `<defs>` | Repeating fill. | Not splat-relevant directly; could host a Gaussian tile. |
| `stop-opacity` / `fill-opacity` / `opacity` | element attributes | Three opacity tiers (stop, fill, element). | All used in current path. |
| `mix-blend-mode` (CSS) | element style | sRGB blend modes: multiply / screen / overlay / soft-light / etc. | Each splat is currently `mix-blend-mode: normal`. Other modes might let us approximate linear-light blend with creative use of `multiply` + pre-baked color spaces. Speculative. |

### SVG filter-based Gaussian splat recipe (untried)

```svg
<defs>
  <!-- One shared filter per quantized sigma bucket -->
  <filter id="g16" color-interpolation-filters="linearRGB">
    <feGaussianBlur stdDeviation="16"/>
  </filter>
  <filter id="g8" color-interpolation-filters="linearRGB">
    <feGaussianBlur stdDeviation="8"/>
  </filter>
  <!-- ... ~32 buckets ... -->
</defs>

<g color-interpolation-filters="linearRGB">
  <!-- Tiny flat-color shape, blurred by the format primitive -->
  <ellipse cx="X" cy="Y" rx="1" ry="1" fill="rgb(R,G,B)"
           opacity="ALPHA" filter="url(#g8)" />
</g>
```

The blur output IS a Gaussian (mathematically; it's what blur means). Cost:
each filter is a per-shape compositing pass. Browsers may struggle at high
shape counts — needs benchmarking. With sigma quantization the filters are
shared across many shapes, helping browser caching.

**Anisotropic splats:** `stdDeviation="SIGMA_X SIGMA_Y"`. Plus a rotation
on the wrapping `<g transform="rotate(...)">`.

---

## DrawingML primitives that affect opacity / feathering

Inside `<a:effectLst>` (ordering matters; svg2ooxml/effect_fragments.py
canonicalizes the order):

| Primitive | What it produces | Splat applicability |
|---|---|---|
| **`<a:blur rad="EMU"/>`** | **True Gaussian blur of the shape.** rad in EMU. | **Highly promising — DrawingML equivalent of the SVG filter trick.** Solid-fill ellipse + blur = true Gaussian. Per-shape rad means we don't need to share filters; just emit the right value. |
| `<a:fillOverlay>` | Overlay another fill on top. | Composing per-splat color over a base layer. |
| `<a:glow rad="..." clr="...">` | Glow extending beyond shape (outer). | Could simulate bright-highlight splats with no fill, just glow. |
| `<a:innerShdw rad="..." dir="..." dist="..."/>` | Inner shadow. | Carves softness inward from edge — opposite of what we want. |
| `<a:outerShdw rad="..." .../>` | Outer shadow. | Similar to glow, less symmetric. |
| `<a:prstShdw prst="...">` | Preset shadows. | Limited library; not parameterizable. |
| `<a:reflection .../>` | Mirror reflection below shape. | Not splat-relevant. |
| `<a:softEdge rad="EMU"/>` | **Edge-feather only.** Inner ring stays solid; outer `rad` ring fades. | **Current path for soft-edge style.** Confirmed via real PowerPoint capture: not a true Gaussian, just a feathered hard shape. The "pink mist" we hit comes from the solid interior. |

Inside `<a:gradFill>`:

| Subprimitive | What it produces | Splat applicability |
|---|---|---|
| `<a:gsLst>` with `<a:gs pos="..."><a:srgbClr><a:alpha>` | Per-stop alpha, baked color. | **Current path for gradient style** — piecewise-linear opacity curve. |
| `<a:path path="circle"/>` vs `<a:path path="shape"/>` | Gradient origin / shape behavior. Empirically `path="shape"` matches Gaussian better than `path="circle"`. | We already use `path="shape"`. |
| `<a:fillToRect>` | Constrain the gradient region within the shape. | Tuning knob if we re-touch gradient style. |
| `<a:lin ang="..." scaled="0|1"/>` | Linear gradient. | Could express axis-aligned features. |
| `<a:tileRect>` | Pattern tile control. | Not splat-relevant. |

Color modifiers (any `<a:srgbClr>` accepts these as children):

- `<a:alpha val="..."/>` — per-color alpha (we use this).
- `<a:alphaModFix amt="..."/>` — fixed alpha modifier.
- `<a:tint val="..."/>` / `<a:shade val="..."/>` — color transforms.
- `<a:lumMod val="..."/>` / `<a:lumOff val="..."/>` — luminance modifiers.
- `<a:satMod val="..."/>` / `<a:satOff val="..."/>` — saturation modifiers.
- `<a:hueMod val="..."/>` / `<a:hueOff val="..."/>` — hue modifiers.

### DrawingML blur-based Gaussian splat recipe (untried)

```xml
<p:sp>
  <p:spPr>
    <a:xfrm><a:off x="X" y="Y"/><a:ext cx="W" cy="H"/></a:xfrm>
    <a:prstGeom prst="ellipse"><a:avLst/></a:prstGeom>
    <a:solidFill>
      <a:srgbClr val="RRGGBB"><a:alpha val="ALPHA_UNITS"/></a:srgbClr>
    </a:solidFill>
    <a:ln><a:noFill/></a:ln>
    <a:effectLst>
      <a:blur rad="SIGMA_EMU" grow="1"/>
    </a:effectLst>
  </p:spPr>
  ...
</p:sp>
```

`<a:blur rad="..." grow="0|1"/>`: `rad` is the std deviation in EMU; `grow=1`
expands the bounding region so the blur doesn't get clipped. This produces
a Gaussian-shaped opacity profile directly, instead of approximating it
with a gradient ramp.

**Untested concerns:**
- PowerPoint's `<a:blur>` math may not be a pure Gaussian — could be a
  fast box filter approximation. Needs visual confirmation via the
  svg2ooxml/tools/ppt_research/ capture rig.
- LibreOffice (soffice) may implement blur differently — confirmed
  irrelevant per `feedback-no-soffice-for-pptx-validation` memory; the
  test is in real PowerPoint.

---

## Linear-light compositing in each format

| Format | Linear-light support |
|---|---|
| Canvas runtime (our JS) | **Yes**, we compute it explicitly. This is why canvas SSIM hits 0.98. |
| SVG | Only inside `<filter color-interpolation-filters="linearRGB">`. Inter-element compositing is sRGB. So filter-based splat rendering can be linear-light, but if we have one filter per splat the inter-splat blend reverts to sRGB. To get fully-linear blending we'd need to compose all splats inside ONE filter graph, which doesn't scale to 20k shapes. |
| DrawingML | **No public linear-light path.** PowerPoint blends in sRGB display space throughout. The sRGB-aware training (`compositing_space="srgb"`) is our only lever. |

---

## Promising experiments to try, in priority order

The order below reflects "cheapest experiment per expected quality gain":

1. **SVG with `<feGaussianBlur>` + flat ellipses** (replaces gradient stops)
   - One filter per sigma bucket (~32 filters).
   - Anisotropic via `stdDeviation="σx σy"`.
   - `color-interpolation-filters="linearRGB"` inside the filter.
   - Benchmark browser render time at 18k shapes.
   - Expected: closes most of the SVG-vs-canvas gap. SSIM 0.83 → ~0.92+.

2. **DrawingML with `<a:blur>` + solid-fill ellipses** (replaces gradFill stops)
   - Each splat: solid fill + `<a:blur rad="sigma_emu" grow="1"/>`.
   - Verify in real PowerPoint capture rig.
   - Confirm whether PowerPoint's blur is true Gaussian or box-filter.
   - Expected: closes most of the PPTX-vs-canvas gap. SSIM 0.75 → ~0.88+.

3. **Edge-layer-specific opacity damping** (lower-risk targeted fix)
   - Apply 0.3-0.5x alpha scale to splats with layer=3 at export time.
   - Doesn't require changing primitives. Compatible with current gradient
     stops or the new blur recipes.
   - Expected: removes the visible "leaves/blades" artifact specifically.

4. **`color-interpolation-filters="linearRGB"` on the gradient path** (free)
   - Add the attribute to existing radialGradients with no other change.
   - May not move SVG SSIM at all because gradient stops are interpolated
     in stop-space, not compositing-space. Worth verifying empirically.

5. **One mega-filter compositing all splats in linear-light** (theoretical)
   - Move all 18k splats inside a single `<filter>` graph using
     `<feFlood>` + `<feComposite>` chains.
   - Filter graphs at 18k nodes — performance and browser limits unknown.
   - The "right" SVG solution if it scales. Probably doesn't.

6. **DrawingML `<a:glow>` for highlight-style edge splats** (semantic)
   - Re-render edge-layer splats as no-fill + glow instead of fill +
     softEdge/blur. May give a different artistic look that matches edge
     splats' role (sharp highlights) better than the current blob.

7. **SVG `<mask>` with a shared Gaussian texture** (heavy)
   - Define one shared `<mask>` containing a pre-rendered Gaussian.
   - Each splat is a flat-color rect with that mask applied + transform.
   - 20k masked elements — performance unknown.

---

## What svg2ooxml has that's relevant

- `src/svg2ooxml/filters/primitives/gaussian_blur.py` — they emit
  DrawingML `<a:blur>` from SVG `<feGaussianBlur>`. Read this to see
  their handling of stdDeviation → EMU conversion and edge cases.
- `src/svg2ooxml/filters/primitives/component_transfer.py` — their LUT
  opacity primitives. Useful if we ever want a custom opacity curve
  beyond a Gaussian.
- `src/svg2ooxml/drawingml/effect_fragments.py` — the canonical effect
  ordering for `<a:effectLst>`. We need to write effects in this order
  or PowerPoint may reject the file.
- `src/svg2ooxml/filters/palette.py` — color quantization helpers.
  Likely overlaps with our `_density_aware_stop_error` + k-means SVG
  palette work.

---

## Memory references

- [[svg-rendering-not-bottleneck]] — SVG export was the original
  bottleneck; this catalog identifies the next levers to attack it.
- [[feedback-no-soffice-for-pptx-validation]] — PowerPoint capture rig
  is required for visual validation of DrawingML experiments.
- [[reference-pptx-render-truth]] — soffice / LibreOffice has rendering
  bugs; always validate in real PowerPoint.
- [[feedback-train-in-deployment-color-space]] — sRGB training already
  helps; the catalog above is the next layer of fidelity work.
