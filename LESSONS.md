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

**The general form of that rule is to leave the lattice entirely.** Both fixes
above are patches applied while still on the grid. The drainage *network* is now
lifted off it instead: the receiver tree is decomposed into main-stem strands,
each strand is corner-cut into a smooth curve, and the distance field is
measured to the curve. A spline through a D8 path is not a D8 path, so the
angularity is gone by construction rather than by threshold — **98% of turning
in corners over 20° → 0.1%**, with total curvature unchanged, because
corner-cutting redistributes bend rather than removing it. The cells still
supply the topology, which is the part the global solve got right and no local
function could recover.

D∞ routing in the erosion is a separate and complementary change, and it is
worth being clear that it does *not* fix the drawn angularity: it makes the
*share* of flow between neighbours continuous, which is what the stream-power
solve wants, but the neighbours it splits between are still the same eight.

One trap on the way. An exact distance to a curve does not quantise the way a
Dijkstra over cell links does — a cell centre sits anywhere from 0 to half a
cell off the curve — so a carve profile narrower than a cell flickered *along*
the channel, left pits the LEM never filled, and the re-routed network
collapsed: Strahler order 3 went from 167 segments to 9 and orders 4–5
disappeared. The carve is floored at the cell size now, because what a 9 km grid
can represent is the valley, not the channel (§3 again).

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

  **That fix was half the bug, and the write-up above hid the other half for two
  milestones.** The same artefact came back — a dark region following the camera
  at every altitude under 10 km — and the clear was not to blame this time: read
  back, the sentinel is exactly `−1e9` where nothing drew. What is wrong is the
  *written* texels. Across one row of the far cascade: 1566 written, **minimum
  0, not one negative, 1206 of them exactly 0**, while positives pass through
  untouched (max 4590). Something in three's WebGPU output path clamps at zero —
  not tone mapping, which changes nothing when disabled. Half of any scene is on
  the far side of the camera from the sun, so half the map stored 0 instead of a
  large negative, and `frag + bias >= stored` failed over a half-space bounded
  by a plane through the camera. The same shape, from the other end of the same
  comparison.

  The payload is unsigned now: both sides add `SHADOW_DEPTH_OFFSET`, derived
  from the largest distance any point inside a cascade can reach. **A float
  render target carrying signed values through an output chain you do not
  control is a contract you cannot enforce** — and the debugging cost of a
  quantity that is silently half-destroyed is enormous, because every hypothesis
  about lighting, bias, geometry and cascade selection stays alive.

  Three probes died before the fourth found it: a 2000 m bias (unchanged), the
  patch skirt removed (unchanged), the cascade centring reverted (unchanged).
  All three were guesses at a mechanism. What settled it in one step was
  rendering the shadow factor on its own — it was flat zero over the entire near
  field, which no lighting explanation survives — and then reading the texels
  back. §2 says to add a debug mode early rather than as a last resort; this is
  the fourth time that has been the cheapest move available and the third time
  it was not taken first. `ShadeMode 8` now returns the shadow factor.
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
- **The static checks pay for themselves inside the session that writes them.**
  The checker gained a WGSL reserved-word pass — the language reserves ~130
  words that are not keywords and have no legal use, several of which are the
  obvious name for something in a renderer. It caught a `ref` local within
  minutes of being written, and the backtick half caught three more prose
  backticks in the same afternoon, one of them in the comment explaining the
  reserved-word rule. Nothing else in the toolchain reads WGSL before the GPU
  does, and the GPU's answer is a blank screen.
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

## 16. A guard is not a model

`sunDepth_` integrates the optical depth toward the sun as `H·exp(-h/H)/mu`,
and floors `mu` at 0.035 so the grazing case stays finite. The comment says the
error is hidden by how little light arrives. That is true at the horizon and
false everywhere past it: below the horizon the floor does not approximate a
dim sun, it **invents** one, permanently 2° up, over the whole night side.

Measured before touching it — the honest way to state the bug — the night
hemisphere from orbit had **84.6% of the frame reading "bright" against the day
side's 84.5%**. The dark side was rendering indistinguishably from the lit one,
and it had been doing so since the atmosphere was written.

The same shape of error, three times in one file:

| guard | added for | became |
|---|---|---|
| `max(mu, 0.035)` | a finite integral at grazing incidence | a sun that never sets |
| `0.045 +` on the sky ambient | shaded ground stays blue after sunset | skylight with no sun |
| `0.10 +` on cloud lighting | a deck is lit from below, terminator is a gradient | a lit deck at midnight |

Each is correct inside the range it was reasoned about and each was applied
outside it. **A term added to keep an expression well-behaved is still a
physical claim everywhere it evaluates.** The fix in every case was to gate it
by the condition it silently assumed — here, that the sun is up — and the gate
is worth deriving rather than guessing: the set point depends on altitude, so a
cloud 8 km up keeps the sun 2.7° after the ground has lost it and a sky sample
at 80 km keeps it for 9°.

**Corollary — say when a number is a rendering choice.** Gating all three
correctly makes the night side black, on the ground too, where auto-exposure
opens to 4.0 and has nothing to open up for. The floor that fixes that is not
starlight (10⁻⁷ of daylight) and not moonlight (10⁻⁶); it stands in for
scotopic dark adaptation, which the tone curve cannot express. It is documented
as that. A magic number with an honest name is maintainable; one dressed up as
physics is not.

---

## 17. An approximation is only valid over the scale it was justified for

Two bugs, one shape, and in both cases the original comment states the
assumption plainly — then a later caller violates it.

- **Aerial perspective sampled the sun transmittance at the path midpoint**,
  "one evaluation instead of a march, which is plenty over the tens of
  kilometres this ever covers". The same function shades the planet from orbit,
  where the path is 9000 km and the midpoint sits 4500 km up — a point that has
  the sun above its horizon at any phase short of the shadow cone. With the
  clouds hidden the dark hemisphere came out a flat sky-blue disc, brighter than
  the terminator. The air is not at the midpoint: density falls as `exp(-h/H)`,
  so essentially all of it is within a scale height of the *lower* end.
- **Shadow cascades were centred at `forward · radius · 0.65`**, near the
  camera. Right at eye height, where the ground is two metres away; at 8 km the
  ground is 8 km below and cascades 0 and 1 — 540 m and 2.6 km there — do not
  reach it at all. The ground under the camera sat on the last cascade's
  footprint edge and slid in and out of shadow. Measured over a 6–12 km sweep:
  a drift of ~0.3 units per 250 m step, then a **jump of 3.71 at 8750 m**.

**Rule:** when a helper's justification names a length scale, that scale is part
of its signature. Grep the call sites before reusing it, and prefer a form that
degrades correctly — sampling at the density-weighted end collapses back to the
midpoint when the two ends are at the same altitude, so the short-path case is
unchanged and the orbital case is fixed by the same line.

---

## 18. Fix the class, not the instance

The near ground was carpeted in hard-edged quads several metres across. Not the
shadows and not the mesh: the normal was clean, slope mode was clean, and
turning shadows off changed nothing. Only shade mode 5 showed it, because it was
in the albedo alone.

The metre-scale grain read
`noised_(up · RADIUS / max(mPerPx·8, 0.30))`, which at eye height is a frequency
of 21.2 M against a `MAX_NOISE_FREQ` of 336 k — **sixty-three times past it**,
where one f32 ULP spans two whole lattice cells. What reached the screen was the
lattice, not a field.

Twenty lines below, a sibling ladder carried this comment:

> Floored by f32, not by ambition. This read 0.35 m, which is 54x past
> MAX_NOISE_FREQ — under one step of fractional position per lattice cell, so
> the "detail" was the hash aliasing against itself rather than a field.

The identical bug had already been found, understood, written up, and fixed —
in the same function — and the sibling was left alone. The local-lattice escape
hatch that makes metre-scale detail reachable at all was also already there.

The same pattern, benignly, in the atmosphere: eight separate sites composed
`transmit_(sunDepth_(...))` by hand, so the missing occlusion had to be fixed
eight times or not at all. It is one `sunLight_()` now.

**Rule:** when a bug turns out to be an instance of a rule the codebase already
knows (the f32 wall, the band limit, the Nyquist floor), the fix is not the
instance. Grep for every other place the rule applies, and if the same
expression is being composed by hand at N sites, make it one function first.

---

## 19. A discontinuity multiplied by time is a tear that grows

The cloud deck's zonal wind was a translation of the noise domain,
`t · rate · select(1, -1, lat < 0.35)` — easterlies one side of the boundary,
westerlies the other. Two problems that are really one: a `select` is a
discontinuity, and the offset accumulates, so the two bands slide apart
*without bound*. Half an hour of sim time in, they were six planet-widths out of
step and the boundary was a hard circle of latitude with unrelated weather
either side.

This class of bug is dangerous because **it starts invisible**. At t = 0 the
tear is exactly zero and every screenshot taken shortly after a reload looks
perfect. It only appears after the thing has been left running — which is the
one thing a debugging session never does.

Advection is a rotation about the axis now, which is continuous on a sphere by
construction, cannot tear however long it runs, and is what a zonal wind
physically is. The profile is smooth as well, so the reversal between the trades
and the westerlies is a shear zone — where fronts actually form — rather than a
cut.

**Rule:** a hard switch on a smooth field is a known mistake (§13). A hard
switch on a quantity that *accumulates* is worse, and needs a different test:
run it for an hour, not a frame.

### The same bug came back, one derivative up

The fix above bounded the *magnitude* with a tanh:

    slip = SHEAR_MAX · tanh(t · band · ZONAL / SHEAR_MAX)

`band` went **inside** the tanh, so the saturation applied to latitude as well
as to time. As t grows, `tanh(t · band · k) -> sign(band)`, and the smooth
latitude profile collapses into a step: the offset swings from -SHEAR_MAX to
+SHEAR_MAX across a latitude interval that narrows *without bound*. That is 40
degrees of longitude across a joint that keeps sharpening, and it draws a
hairline seam along the two circles where band crosses zero — about ±21.7° —
with the cloud visibly chopped and offset either side.

So the tear never went away. Bounding the value simply moved the unbounded
growth into the *gradient*, where it was harder to see and produced a thinner,
more convincing-looking artefact. Same signature as before, too: invisible on a
fresh load, knife-sharp within a few minutes.

The bound belongs on the time factor alone — `SHEAR_MAX · band · tanh(t·k)` —
leaving `band` the smooth multiplier it was always meant to be.

Two further things fell out of it, and both are the interesting part:

- Fixing it exposed a constant that had never actually done anything.
  `CLOUD_SHEAR_MAX` documents a shear strain of 1.5 drawing a cell to 1.8:1.
  While the profile was a step, the shear was *zero either side and infinite at
  the joint*, so that number was never applied. Switching it on for the first
  time revealed it was also double what the comment described, because `band`
  runs -1 to +1 and the note sized the full swing. A constant whose effect is
  bypassed by a bug is not a tuned constant; it is an untested guess wearing a
  comment.
- The seam had been reported twice as "a seam along a line of longitude". It is
  a line of *latitude*. A misread of the symptom sent the first investigation at
  the advection's longitude term, which was innocent both times.

**Rule:** when you bound something to stop it growing, check what you moved the
growth *into*. Magnitude, gradient and curvature are all things that can run
away, and a tanh only ever bounds the first.

---

## 20. How you compute the number is part of the measurement

§14 says to check the instrument. Two more, and both were instruments that
looked fine and were measuring the wrong thing.

- **Horton's bifurcation ratio read 10.96 against Earth's 3–5**, and had been
  reported as OUT for as long as the bake existed. It was the *mean of the
  consecutive ratios*, and on a whole planet the top Strahler order is always a
  handful of segments: 4.57, 4.43, 6.61, and then 113/4 = **28.25**, which moved
  the answer on its own. Fitted the way the ratios are actually defined — a
  log-linear regression over orders with enough segments to estimate one — the
  network reads **Rb 5.04 at R² 0.997**. Horton's law holds almost exactly. The
  model was never outside Earth's range; the statistic was.
- **The channel-angularity check measured the cube-sphere warp.** The first
  version histogrammed segment *bearings* against the face axes, on the theory
  that a lattice network can only run at multiples of 45°. It cannot: the
  tangent warp makes cells non-square away from a face centre, so a diagonal
  step is 43° here and 52° there. It read 39% where an unsmoothed network should
  have read 100%. Total curvature per unit length has no such problem — it is a
  property of the curve and not of any frame — and it separated the two cases
  cleanly: **98% of turning in corners over 20°, against 0.1% after smoothing**,
  with the total curvature unchanged.

**Rule:** prefer a statistic that is a property of the object over one that is a
property of the coordinates you happened to measure it in. And when a quantity
is defined as the slope of a line, fit the line — do not average the steps, and
report R² next to it, because a ratio quoted from points that are not on a line
is not describing anything.

---

## 21. Normalise for what you are *not* changing

The octave ladder gained a per-biome gain and hillslope crossover, so a dune
field, a badland and an alpine ridge stop being the same shape at three sizes.
The detail is `Σ wᵢaᵢ(rᵢ² − mean) / norm`, and the obvious normaliser is `Σaᵢ`,
which is what was there.

That would have made the biome table an *amplitude* control as well as a
spectral one. The octaves are near-independent, so the RMS goes as
`√(Σaᵢ²)/Σaᵢ`, which moves with the gain — measured across the candidate
range, **−13% to +15%**, and in the wrong direction: the smoothest spectra would
have delivered the *most* total relief. How much relief a place has is already
the business of `ampAt_` and the bake's own slope, and two controls fighting
over one quantity is how a fitted constant stops meaning anything.

Normalising by `√(Σaᵢ²)`, rescaled to the reference ladder's own ratio, holds
the RMS to **±0.6%** across the whole table and leaves the reference spectrum
bit-identical to what `Σaᵢ` gave.

**Rule:** when adding a per-point parameter to a normalised sum, work out which
moments of the result it moves. Fix the ones you did not mean to change, and
check the fix by measuring the moment rather than by looking at it.


---

## 22. When two causes share a symptom, build the thing that tells them apart

A shadow-map cascade cannot distinguish *"nothing occludes this"* from *"I was
not given the occluder"*. Both read as the clear value, both come back lit, and
on screen both produce a straight-edged quadrilateral on the ground. The
*opposite* failure — a wrong or too-coarse caster — produces a straight-edged
quadrilateral too, just dark. Three separate sessions have now chased
quadrilaterals through the bias, the cascade selection, the caster frustum
depth, the patch skirt and the cross-fade, by reasoning backwards from the
shaded image. That reasoning has been wrong nearly every time, because the
image genuinely does not contain the distinguishing information.

The instrument is ten minutes of work: draw the map. `src/shadowDebug.ts`
renders a cascade to a screen overlay with the sentinel colour-coded, which is
exact rather than heuristic because `SHADOW_DEPTH_OFFSET` already guarantees
every written value is positive. Magenta means nothing was recorded; grey means
something was. A lit artefact over magenta is a missing caster. A dark artefact
over grey is a wrong caster.

Pointed at a view that had been resisting explanation, it showed a quarter of
cascade 2 magenta with a patch-shaped edge, and two real bugs fell out
immediately:

- **The shadow pass culled its casters against the main camera's frustum.**
  `select()` builds its frustum from the matrices it is handed, and the shadow
  pass was handing it the camera's. Terrain beside and behind the camera casts
  into the map; culling it left the map cleared there.
- **The distance cap is a sphere and a cascade is a box.** `radii[2] · 1.5`
  does not reach a corner that sits at `radii[2] · (0.65 + √2)`. The corners
  simply had no casters selected. This is the *same* sphere/box mismatch §13
  records fixing on the sampling side, still present on the selection side —
  fix the class, not the instance (§18).

What it cost to not have the instrument: a symptom-level fix (override the near
cascade with the far one) that cleared the artefact under test, passed a
one-viewpoint check, and introduced a dark quadrilateral somewhere else. It had
to be reverted. The real fix was two lines in the selection and made the
symptom-level fix unnecessary.

**Rule:** when two opposite causes produce the same symptom, no amount of
staring at the symptom will separate them. Build the smallest thing that reads
the intermediate state directly, *before* forming the third hypothesis — not
after the second one fails.

---

## 23. Still open

- **The dark quadrilaterals are not fixed.** The two selection bugs in §22
  fixed the *lit* class and the 3 km lake artefact with it. Flat ground under a
  low sun still shows large dark bands with straight edges and right-angle
  corners — reproduce with `sim.tour('valley', 700, 28, 130)`, which is a plain
  with essentially no relief and should have almost no shadow at all. Filling
  the map barely changed them, so they are *not* missing casters.
  **Next step:** press `M` to put cascade 2 on screen and read the texels under
  a band, now that the ramp is calibrated to the light frustum depth
  (`radius + CASTER_DEPTH`) rather than to one radius — the earlier reading
  clipped most of the map to flat black and white and hid exactly the structure
  this is for. Grey under the band means a real occluder is recorded and the
  question is *which* geometry wrote it; a step in the ramp means a cliff in
  the map that is not a cliff in the world. Ruled out already, each tested and
  reverted: caster frustum depth (`CASTER_DEPTH` 4500 → 20 000), patch skirts
  in the shadow pass, and LOD patch boundaries.
- **Rivers are hidden again.** `DRAW_RIVERS = 0`. The curve fitting fixed the
  *shape* — the network is smooth and measurably so — but it cannot conjure
  resolution that was never there, and a 1–3 km channel against a 9 km cell
  still reads as a painted line up close rather than as water. The LEM's
  valleys stay, and they are the part that reads well at every altitude. This
  is the second time removing rivers has been the right call (§15); the
  machinery is intact behind one constant.
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
- **Auto-exposure does not know what is in frame.** It meters analytically from
  the sun, the albedo under the camera and the altitude, which is what makes it
  lag-free and reproducible (and worth keeping for that). But a bright overcast
  sky filling half the view is not in the model, so a dark forest floor under it
  comes out nearly black. The fix is not a framebuffer readback — that costs the
  reproducibility the whole design is for — it is to include the sky's own
  contribution to the frame in the analytic meter.
- **Rb sits at 5.04, just above Earth's typical 3–5.** Measured honestly now
  (§20), with R² 0.997, and the same estimator gives 5.09 on the pre-D∞ bake, so
  it is the model's own value. Natural basins run 2–8, so this is not wrong —
  but if it is ever worth moving, the lever is the channel-head support area,
  which sets how many first-order streams the network is credited with.
