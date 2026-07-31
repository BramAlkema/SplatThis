# PowerPoint hover-parallax MVP

**Status:** proposed, not yet validated in PowerPoint
**Date:** 2026-07-30

## Chosen interaction

Build the parallax effect as one native PowerPoint slide with discrete
mouse-over positions. This is intentionally not continuous pointer tracking
and does not require VBA, a content add-in, or a matrix of linked slides.

The slide contains:

- one transparent 10 x 10 hotspot grid over the artwork;
- four to six depth-layer groups;
- one precomputed target pose per hotspot;
- native PresentationML timing triggered by `onMouseOver`; and
- a short eased transition to each target pose.

The artwork is not duplicated for every hotspot. One hundred hotspots only add
interaction shapes and timing instructions; they all animate the same small
set of depth-layer groups.

For a 400 x 400 image, each hotspot covers approximately 40 x 40 source pixels.
For non-square artwork, the grid should be adjusted so hotspot cells remain
approximately square rather than forcing a literal 10 x 10 layout.

## Interaction semantics

Entering a hotspot sets the target parallax pose associated with that cell.
The pose remains active until another hotspot is entered.

Individual cells must not recenter on `onMouseOut`: moving from one adjacent
cell to another would otherwise briefly animate through the center. Instead:

- the center cell maps to the neutral pose;
- optional transparent reset strips around the artwork map to the neutral
  pose; and
- leaving the slide may leave the last pose held until the slide is exited.

Initial transition duration should be 60-100 ms. The timing tree must replace
or supersede an active transition when the pointer crosses cells quickly; it
must not queue every intermediate pose.

For normalized hotspot coordinates `u, v` in `[-1, 1]`, a layer pose can start
with:

```text
dx(layer) = u * max_shift_x * depth_weight(layer)
dy(layer) = v * max_shift_y * depth_weight(layer)
scale(layer) = 1 + radial_amount * scale_weight(layer)
```

The exact direction, amplitudes, easing, scale, and shadow response are visual
parameters and must be tuned in a real PowerPoint slideshow.

## Vector and splat artwork

SVG-derived or native DrawingML artwork should be divided into a few coherent
depth groups. PowerPoint animates the group transforms, not every splat.
Thousands of splats may remain editable inside those groups without producing
thousands of animation targets.

Useful starting groups are:

1. far background;
2. near background;
3. main subject;
4. foreground;
5. optional highlights; and
6. optional shadows.

Depth should come from semantic/depth segmentation rather than saliency alone.
Highlights and shadows may move slightly differently from their owning layer
to suggest volume rather than flat cardboard translation.

## Bitmap artwork

The same interaction works for bitmaps and may be cheaper for PowerPoint to
render. A flat bitmap must first be converted into transparent depth plates:

1. infer or provide a depth map and semantic segmentation;
2. extract the subject and foreground as alpha PNGs;
3. inpaint content hidden behind nearer objects;
4. add overscan so motion does not expose empty edges; and
5. optionally isolate highlights and shadows.

The slide still contains only one image object per depth plate. The 100
hotspots animate those shared image objects; they do not require 100 copies of
the bitmap layers.

This is layered 2.5D. Native PowerPoint can translate and scale image objects,
but it cannot continuously deform one flat bitmap according to a depth map.

## MVP sequence

1. Generate a minimal one-slide file with three obvious test layers and a
   small hotspot set.
2. Inject a correct native `p:timing` tree with `onMouseOver` triggers
   directly, without depending on the unfinished general animation converter.
3. Verify hover triggering, held end states, transition replacement, and reset
   behavior in real Microsoft PowerPoint on macOS.
4. Increase the interaction layer to 10 x 10 and stress-test rapid diagonal
   pointer movement.
5. Apply the proven timing structure to Coffee with four to six real depth
   groups.
6. Capture the slideshow with the existing real-PowerPoint screenshot tooling
   and compare the neutral frame against the non-animated artifact.

## Acceptance gates

- no slide navigation or transition flashes;
- no recenter flash between adjacent hotspots;
- no queued-animation tail after fast pointer movement;
- neutral pose visually matches the original static slide;
- no exposed transparent seams or clipped layer edges;
- acceptable interaction latency in PowerPoint for macOS;
- normal static editing remains possible; and
- file size and slideshow render cost are reported separately for vector and
  bitmap variants.

If 10 x 10 generates visible event churn, first shorten the transition and
verify replacement semantics. Reducing the grid to 8 x 8 is a fallback, not
the initial target.
