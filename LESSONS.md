# Lessons

What this project has cost to learn, written down so it is only paid once.
Every number here was measured, not recalled.

---

## 1. Measure the distribution before you choose the threshold

Nearly every visual bug in this project has been a plausible-looking constant
that was wrong by an amount only a histogram could reveal. Not one of them was
findable by reading the code.

| Constant | Guessed | Actual distribution | Result on screen |
|---|---|---|---|
| `RIVER_WET_LO` | 8.4 | median land wetness is **8.45** | 54% of all land drawn as river |
| billow window offset | 0.62 | true mean of the field is **0.8455** | planet 100% overcast; ground went black under its own cloud shadow |
| `LAKE_ON_LO` | 0.5 m | 13.9% of land is ≥1 m basin (Earth: 1.8%) | an eighth of every continent underwater |
| `VEG_MAX_SLOPE` | 0.55 | that is **63°** in the 1−cos metric | trees growing out of cliffs |
| `RELIEF_SLOPE_*` | fitted at 512 | median land slope moved 0.0034 → 0.0048 at 1024 | 22% of land at full amplitude instead of 13% |

The fix, every time, was a twenty-line script that printed percentiles. Write
that script *before* picking the number.

**Corollary — put the measurement in the comment.** A constant with its
distribution written next to it can be re-fitted by the next person when the
bake resolution changes. A bare number cannot.

**Corollary — measure the right subset.** The first anisotropy test on the
baked elevation returned "only 6% grid-locked" and nearly killed a correct
hypothesis. Hillslopes are most of the land area and they diluted the signal;
the *channels* were 100% locked. Aggregate statistics hide the feature you are
looking for whenever that feature is a small fraction of the domain.

---

## 2. A screenshot cannot falsify a confounded observable

Four wrong conclusions this project, all from looking instead of measuring:

- **"The mesh is 800 m below `heightAt`."** It was 1.7 m. I counted metric-grid
  lines belonging to a different camera state and never cross-checked. Two
  turns of work went into chasing band limits and `RIDGE_MEAN` for a bug that
  did not exist. Rendering distance-to-pixel as colour bands settled it in one
  frame.
- **"The cloud shadow isn't working."** It was. Auto-exposure compensates a
  uniform multiplier *exactly*, so forcing `cloudLit = 0.30` produced a
  pixel-identical image. Only rendering the field itself to a debug channel
  showed `cloudLit ≈ 0.18` under the cumulus.
- **"The tree model is done."** The canopy had never rasterised once. I had
  looked down from 42 m, seen crowns, and concluded the geometry worked — those
  crowns were mid-band billboards.
- **"Catastrophic 2 fps."** The browser pane throttles its compositor. Wall
  clock in that pane is not a frame time; `renderer.info.render.timestamp` is.

**Rule:** when the thing you want to see is downstream of an adaptive system
(auto-exposure, LOD selection, a compositor, a fallback material), you cannot
test it by looking at the final image. Add a debug mode that returns the
quantity directly. `ShadeMode` exists for this; use it early, not as a last
resort.

---

## 3. Resolution is a wall. Compute the arithmetic first.

The single most expensive lesson here. Three separate features hit it:

**Rivers.** The global bake stores 9 km per texel. A continental trunk river is
1–3 km wide. That is one to two orders of magnitude below what the data can
place, and no reconstruction recovers a feature that is not in the field. Three
approaches were built and measured:

1. *Threshold the wetness field* → texel-shaped blobs. Widening until stable
   gave a river 10 km across.
2. *Reconstruct the axis from the transverse wetness gradient* → the estimator
   is built from a 9 km stencil, so it is noisiest at exactly the scale the
   channel lives at. The river broke into a chain of ponds.
3. *Bake a Dijkstra distance field and read it directly* → path correct and
   contiguous, but a 250 m half-width against a 9 km texel means the mask flips
   inside a fraction of one bilinear cell. Blue rectangles.

The right answer was to stop and remove the water, keeping what *is* resolvable
at 9 km: the carved valleys, the lakes, the riparian tint.

**Grass.** Blades were built at a physically correct 3–8 mm. At that width a
blade is under a pixel beyond four metres, so 46 000 of them resolved to
nothing. Real-time grass is stylised wide for a reason.

**Trees at 50 km.** At 693 px/radian, the largest tree on the planet subtends
**0.40 px**. There is no level of detail that fixes that — it is aliasing with
extra steps. Individual sprites top out near 6 km; past that it has to be
cluster impostors or the terrain's own albedo.

**Rule:** before building the feature, compute how many pixels or texels it
occupies at the ranges it must work at. If the answer is under one, change the
representation or don't draw it.

---

## 4. Sampling theory is not optional

The "overly noisy ground" was a detail-normal octave at **2.5 pixels**
wavelength — below the display's Nyquist frequency. It could never resolve; it
could only alias. It read as film grain over the whole planet and crawled with
the camera. Moving the finest octave to 8 px turned noise into texture.

Related, and already load-bearing in this codebase:

- Octaves are faded out by mesh vertex spacing, because a vertex landing on an
  arbitrary phase of an undersampled octave makes the surface boil as it morphs
  (`BAND_FADE_LO/HI`).
- The octave ceiling is *derived*, not chosen: noise is evaluated at `dir * F`
  with `|dir| = 1`, so the lattice fraction quantises to `F·2⁻²³`. That fixes
  `DEFAULT_OCTAVES = 9`. It was hard-coded at 17, and octaves 10–16 were
  returning pure quantisation noise for eight hash evaluations each.
- Dropping an octave must *low-pass* the surface, never rescale it, or the
  ground rises as you approach and the coastline moves (SPEC I2). That is why
  `RIDGE_MEAN` is subtracted per octave.

---

## 5. Grid algorithms leave grid signatures

Two artifacts visible from 573 km, both from the solver rather than the shader:

- **D8 single-flow routing.** Every channel picks one of eight directions, so a
  river drawn from it is a polyline of exact 0°/45°/90° segments. Fixed with a
  multiple-flow-direction accumulation for the *rendered* field; the erosion
  keeps D8, because stream power is a channel process and the Braun–Willett
  O(n) solve needs a receiver tree.
- **Priority-Flood + ε.** The filler raises basins by 1 mm per cell in
  8-neighbour BFS order, and BFS parent chains on a grid are straight rays along
  the eight directions. Route flow down that and a basin fills with perfectly
  straight rivers meeting at right angles — rectangles drawn on the continent.
  Fixed by keeping the basin in the terrain and emitting the filled level as a
  lake surface, which puts the degenerate drainage underwater where it belongs.

**Rule:** any field that will be *looked at* needs its own treatment. The field
the solver needs and the field the renderer draws are not the same object.

---

## 6. Every consumer of a shared field must apply every transform

The coastline warp displaces the bake lookup by 0.62 of a texel — **5.6 km**,
over which elevation moves hundreds of metres. Applied in the terrain shader
only, the vegetation and grass scatters were standing on a different planet.
Four call sites had to be updated: `terrainMesh`, `heightCPU`, `quadtree`'s
tile records, and `grass`.

The same warp, applied *globally*, slid inland terrain sideways under the
camera and low-altitude views went black. It had to be faded out by `|bakeH|`
so it only touches the shoreline — which is also the only place it buys
anything, since inland the amplification already supplies detail at every
scale.

**Rule:** when adding a transform to a shared input, grep for every reader
before writing the code, and gate the transform to the region that needs it.

---

## 7. Pipeline ordering is semantics

`lakeDepth = max(0, water − z)` was computed *after* `carveChannels`, which
lowers `z` by up to 60 m along every channel. Result: the entire drainage
network registered as a 60 m deep lake and rendered as blue ponds down every
valley. Moving three lines fixed it.

The correction was also exactly computable from channels already in the atlas,
so it was patched in place rather than paid for with another 19-minute bake.
**When a pipeline bug has a closed-form correction, patch the artifact.**

---

## 8. Do not hand-write what a data structure already knows

- **Indirect draw `indexCount` written as the literal `18`.** Correct while
  every band was quads; silently wrong the moment the near band became a
  462-index tree. Six triangles of the bole rasterised and the entire canopy
  sleeve never reached the raster. Nothing about the geometry, winding or
  shader was at fault. It is now
  `(argsAttr.array)[b * ARGS] = geo.getIndex()!.count`.
- **Shadow cascade selection by radial distance.** A cascade *covers* a square
  centred 0.65 radii ahead of the camera; selecting by `|rel| < radius` picks
  cascades whose map the point falls outside of. Now selected by footprint
  coverage, which makes the sphere/box mismatch impossible to express.
- **Shadow map cleared to `0xffffff`.** The map stores signed distance along the
  light *in metres*, so white means "there is an occluder one metre away". Every
  point with `dot(rel, sunDir) < 1` came back shadowed — and that set is a
  half-space, whose boundary is a plane, which is why the artifact was a hard
  straight-edged wedge covering half the world. R32F holds a real sentinel;
  `−1e9` costs nothing.
- **Band selection nested three deep by hand.** Now built from `VEG_BANDS`, so
  adding a band is one line instead of a nested `select` nobody notices is
  still three deep.

---

## 9. Camera-relative architecture: never hash a position

`inst.xyz` is camera-relative — that is the entire point of the precision
design (SPEC I4). Hashing it for per-instance identity means every tree
re-rolls its species and its yaw as the camera moves. Trees rotated on the spot
and a broadleaf became a conifer when you walked toward it.

Two fixes, both worth reusing:

- **Trees:** the scatter already has a stable hash built from the tile's global
  cell index. Carrying it to the draw cost nothing — it rides in the low bits of
  the scale. 12.5 cm of size resolution buys 16 bits of identity, and a tree has
  no preferred height to that precision.
- **Grass:** the grid is camera-centred but **world-snapped**. The camera's
  face-uv position is quantised to the blade spacing in f64 on the CPU; the
  integer part seeds the hash and the fraction positions the grid. Without the
  snap every blade is reseeded every frame and the field boils.

---

## 10. Damp toward an absolute, not by a factor

The amplification damped trunk valleys by ×0.28. In mountainous country the
unmodified amplitude is `150 + 4000·relief`, so 0.28 of it is still **380 m** of
ridged relief laid across the valley floor. A waterline a few metres under that
floor cannot make a river; the water pools in whichever hollows fall below it.

A floodplain is flat in absolute terms. Mixing toward a fixed 18 m rather than
scaling is what made a valley floor a valley floor whatever the surrounding
relief.

**Rule:** when a term must produce a *specific* physical scale, mix toward that
scale. Multiplying by a fraction preserves whatever variance was already there.

---

## 11. What actually bought realism, per unit of effort

Ranked by visual return:

1. **Cloud shadows on the ground** — one ray/sphere intersection and one shared
   field. A landscape under a broken deck is mostly *in shadow*, and that moving
   shadow is what makes a sky read as weather rather than wallpaper.
2. **Synoptic-scale cloud organisation** — one low-frequency noise gives two
   things: its value sets where systems and clear air are, and its gradient
   rotated a quarter turn about the vertical gives a circulating flow to warp
   the domain along. That shear turns uniform cells into spiral arms and fronts.
   Strength matters: at 0.16 the arms grew longer than the cells and the planet
   became swirled paint; 0.085 organises without dissolving.
3. **Billow sign.** `|n|` puts creases at the bottom and rounded cells between
   them — cumulus. `1 − |n|` puts ridges along the noise's zero set, which is a
   family of *curves*, so the planet came out wrapped in a white reticulated
   web. One character.
4. **Dithering the biome axis, not the colour.** Perturbing `moist` with two
   noise octaves breaks every boundary contour into a mixture. Applied to the
   axis, a point takes the colour of ground slightly drier or wetter than it is
   — which is what patchiness physically is — and never a colour from elsewhere
   in the diagram.
5. **Keeping basins as lakes** instead of filling them flat.

---

## 12. A mirror that does not mirror

`shaderNoiseCPU.ts` exists for one reason: to reproduce the shader's gradient
noise bit for bit, so the camera stands on the surface the GPU draws. Its last
three lines were

```js
h[0] = (x + Math.imul(y, z)) >>> 0;
h[1] = (y + Math.imul(z, x)) >>> 0;
h[2] = (z + Math.imul(x, y)) >>> 0;
```

and the WGSL they mirror is

```wgsl
p.x = p.x + p.y * p.z;
p.y = p.y + p.z * p.x;   // reads the p.x written above
p.z = p.z + p.x * p.y;   // reads both
```

The shader's round is **sequential**; writing into an output array makes the
CPU's **parallel**. `x` agrees either way — it is the first assignment — and
`y` and `z` disagree at *every lattice point in the domain*. Two of three
gradient components were an unrelated field.

Measured cost: `heightAt` and the shader were **625–893 m apart** in mountains,
so walk mode put the camera 844 m in the air, the LOD selector measured against
a planet nobody was drawing, and every scatter that anchors to the CPU field
was standing on the wrong surface. After the fix, 0.05–1.18 m across three
unrelated sites.

The two rounds are textually identical apart from where they store the result.
Reading the file will not find this. Diffing the two implementations
statement-by-statement will, and so will a test that evaluates both on the same
input and compares — which is four lines and did not exist.

**Rule:** a function that exists to duplicate another one needs a test that
runs both. "Kept deliberately in lockstep" is a comment, not a mechanism.

---

## 13. Ablation is not diagnosis

Two artefacts sat on the open list for a long time, both blamed on geometry:
holes along coastlines seen from above, and a picket fence of bright slivers at
patch borders over flat ground at grazing angles. Turning the patch skirt off
made both disappear, which looked like proof and was not — it removed the
geometry that was being mis-rejected, not the thing rejecting it.

Both were the **logarithmic depth buffer**. three's WebGPU path evaluates log
depth per *vertex* and lets the rasteriser interpolate it; log depth is not
linear in screen space, so a triangle's interior gets a wrong depth, worst
where the depth gradient across the triangle is steepest. That is exactly a
coastline from above (sea level to hillside in one triangle) and flat ground at
a grazing angle (hundreds of metres of range in one triangle). Fragments lose
the depth test against nothing and the clear colour shows through.

Four wrong fixes were built and measured first — relief-scaled skirt depth, a
raised depth cap, an 8x deeper skirt, and a one-cell lateral apron. The apron
measured *worse* (0.158% → 0.165%) and every one of them was reverted.

`sim.audit()`, same coastline, every patch at one level:

| lodFactor | 1.2 | 2.2 | 4.4 |
|---|---|---|---|
| log depth on | 0.116% | 0.158% | 0.292% |
| log depth off | 0.000% | 0.000% | 0.000% |

What settled it in one frame: set the clear colour to magenta. The specks
turned magenta, so they were holes and not shading, and every shading
hypothesis died at once.

**Rule:** "the symptom went away" identifies a *participant*, not a cause. Ask
what else changed, and prefer a test that distinguishes categories — is this
geometry or shading? — over a test that removes one suspect.

**Corollary:** the frame's own configuration is part of the renderer. Depth
format, near/far, MSAA and colour space are as capable of drawing an artefact
as any shader, and none of them are in the shader you are reading.

---

## 14. Check the instrument before the subject

Both tools written to catch exactly these bugs were themselves wrong, which is
why neither bug was caught.

- **`auditCracks` forced a 1280x720 frustum onto a patch selection made for the
  window's aspect**, without re-selecting. On any non-16:9 window it counted
  everything outside the selection as a hole — ~5% on a mesh that had 0.15%.
  It agreed on a 16:9 window, which is why it went unnoticed for so long.
- **`mirror.ts` measured the *flooded* surface against bathymetry.** With the
  sea drawn over it, 72% of samples read exactly 0 m, the hypsographic curve
  lost its whole bathymetric half, and "amplification must be zero-mean"
  became a statement about the height of sea level above the abyssal plain:
  2669 m against a 25 m tolerance. It had been failing and being explained away
  as expected. Unflooded, it reads −8.2 m.

Neither tool was measuring the planet. One was measuring the frustum and the
other the sea.

**Rule:** when a check has been failing for a while and the failure has an
accepted explanation, that is the moment to re-derive what it computes. A
tolerance that gets explained instead of met has stopped being a test.

---

## 15. Process

- **`npm run check:shaders` before every commit, without exception.** Backticks
  inside WGSL template literals silently truncate the shader and produce a 500.
  This has been self-inflicted at least five times. The checker catches it in a
  second; skipping it costs a browser round trip and a confused diagnosis.
- **Bake iteration cost dominates everything.** 41 s at 512, 801 s at 1024 —
  and the 20× is not the 4× the cell count implies, because the LEM needs more
  iterations to converge at finer spacing. One careless Dijkstra that seeded
  every ocean cell (4.5 M heap entries) pushed a run past 35 minutes. Restrict
  the frontier before committing to a long run.
- **Keep a fast repositioning helper in the page.** Most of the wall clock in a
  debugging session goes on getting the camera somewhere useful. Store the spot
  in `sessionStorage` so it survives the reloads a shader change requires.
- **Removing a feature is a legitimate outcome.** After three measured failures
  on rivers, shipping blue rectangles would have been worse than shipping
  nothing. The valleys, lakes and riparian tint still read as drainage.

---

## 16. Still open

- The forest is dense enough that the ground is invisible wherever it grows —
  no glades, no gaps, cover saturating at `FOREST_DENSITY = 0.9`. This is
  unrealistic on its own *and* it blocks any verification of the grass.
- **Mesas and buttes.** Tried and reverted: a terrace operator on absolute
  elevation, so hard beds outcrop along a contour. It is mean-preserving and
  costs almost nothing, and it cannot work. Applied to the full height it
  quantises the metre-scale noise and the ground comes out as vertical flutes;
  applied to the band-limited landform it does nothing, because a terrace only
  bites where the landform already has relief comparable to the bed — and where
  it does, that country is a mountain range, not a plateau. The A/B with the
  term off was indistinguishable.
  A butte is a *resistant remnant*: the caprock has to survive an erosion
  episode that removed everything around it. That is a property of the LEM, not
  of a post-hoc height function, and it belongs in the bake — hardness as a
  third solved field alongside elevation and drainage.
- Lakes come out at a fraction of Earth's 1.8% of land once thresholded deep
  enough to avoid the shallow-basin flood. The LEM simply does not leave many
  large closed basins.
