# world_sim

Planetary terrain, orbit to ground, fully procedural. TypeScript + three.js on
WebGPU. Architecture and roadmap live in [SPEC.md](SPEC.md).

**Status.** Cube-sphere quadtree with CDLOD and camera-relative precision; a
global bake solving plate tectonics, stream-power erosion and drainage over
1.57 M cells — **in the browser**, on first visit, then cached; runtime
amplification whose spectrum varies by biome; climate, biomes, a cloud deck with
ground shadows, GPU-driven forest and grass. Rivers are carried as curves in the
bake but not drawn — see LESSONS §24.

## Running it

```bash
npm install
npm run dev        # http://localhost:5173
```

That is the whole of it. **The planet is solved in your browser** on first
visit — plate tectonics, then stream-power erosion and drainage over the whole
sphere — behind a progress screen, and kept in IndexedDB afterwards. About a
minute, once per seed.

`src/bake/` never had a Node dependency; only the CLI wrapper did. So the same
solver runs in a Web Worker (`src/bake/worker.ts`), and the atlas goes straight
into a texture without ever becoming a file.

An offline bake still exists, and `public/planet/` is still checked for and used
when it matches — it is how the *reference* planet the baselines describe is
produced, and it saves first-time visitors the minute:

```bash
npm run bake -- --write   # 45 s at 512, writes public/planet/ (12.7 MB)
```

### A different planet

Add `?seed=` to the URL. Anything works — numbers are used directly, words are
hashed — and the planet is built and kept the first time you ask for it:

```
http://localhost:5173/?seed=4242
http://localhost:5173/?seed=hello
```

To move the *default* planet, which is the one the checked-in baselines
describe, bake it offline and reissue them:

```bash
npm run bake -- --seed 12345 --write
npm run sites                        # reissue the site table in src/tour.ts
npm run realism -- --update          # reissue the baseline for the new world
```

The seed drives the plate layout and their motions, and everything else —
erosion, drainage, climate, biomes — follows from those, so a new seed is a new
world under the same physics. The two reissue steps matter: the realism suite
judges the planet at 24 fixed places chosen *for this seed*, and a site labelled
`desert-flat` on one planet is somewhere else entirely on another. `npm run
bake` prints the reminder when the seed is not the default. To keep a seed, set
`DEFAULT_TECTONICS.seed` in [src/bake/plates.ts](src/bake/plates.ts).

### Publishing to GitHub Pages

`.github/workflows/pages.yml` builds the site in CI and commits the output to
the `docs/` folder on `main`, so GitHub Pages can deploy directly from the
branch. Push to `main`, then set **Settings → Pages → Source** to **Deploy from
a branch**, select `main`, and set the folder to **`/docs`**. The build takes a
few minutes; the first Pages deployment can take another minute to propagate.

Two things worth knowing before you point anyone at the link:

- **It needs WebGPU.** Chrome/Edge 113+, Safari 18+, Opera 99+, or any current
  Chromium-based browser. Firefox will show the "Could not start" panel, which
  is the app telling the truth rather than a broken deploy.

  Presence of the API is not the same as a working one, which is what the
  startup check actually asks. Chromium forks all ship `navigator.gpu`, so the
  old `'gpu' in navigator` test passed and the failure surfaced later as
  three's internal `Unable to create WebGPU adapter.` with a stack trace —
  advising a reader on a perfectly capable browser to install Chrome. The check
  now requests an adapter, and when that comes back null it names the real
  cause: hardware acceleration switched off, or a blocklisted GPU. On Opera and
  Opera GX it prints the `opera://` settings paths rather than the `chrome://`
  ones.
- **First visit costs either a 12.7 MB download or a minute of CPU** — the
  prebuilt planet if the seed is the default, a browser bake otherwise. Both are
  cached afterwards.

### Checks

```bash
npm run verify     # typecheck + shaders + scaler + precision + mirror + biomes + realism
npm run realism    # the fast gate: 24 fixed sites vs a stored baseline, ~4 s
npm run scaler     # adaptive resolution scaler, both clocks
npm run slopes     # slope distribution vs sampling scale
```

### Quality tiers

The calibrated numbers throughout this project were chosen on an M-series
laptop at 1440p, and a phone is not a small one of those. `src/quality.ts`
picks a tier at load and every render-side knob follows from it — pixel ratio,
MSAA, octaves, LOD factor and max level, shadow map size and cascade count,
vegetation density and buffer size, grass.

| | low (phones) | medium (tablets) | high (desktop) |
|---|---:|---:|---:|
| pixel ratio cap | 1 | 1.5 | 2 |
| MSAA | off | off | on |
| octaves | 9 | 12 | 14 |
| LOD factor · max level | 1.5 · 17 | 1.8 · 18 | 2.2 · 19 |
| shadow map · cascades | 1024 · 2 | 1024 · 2 | 2048 · 3 |
| vegetation density · slots | 0.45 · 150 k | 0.7 · 200 k | 1 · 400 k |
| grass | off | on | on |

The adapter is requested with `powerPreference: 'high-performance'`. Nothing
was passing a preference before, and on a laptop carrying both an integrated
and a discrete GPU an undefined preference may return the integrated one —
which swamps every row of that table put together.

Pixel ratio is the first lever because it is the largest: the fragment shader
carries atmosphere, biome blending and a 3×3 PCF tap, and a phone reporting
`devicePixelRatio` 3 was rasterising four times the fragments a cap of 1 asks
for. On a 1280×720 viewport at dpr 1.5 the tiers measure 1280×720 against
high's 1920×1080 — 2.25× fewer fragments before anything else is counted.

Patch counts follow the LOD factor as its square, measured at
`broadleaf-steep`, cracks audited at each stop:

| above ground | low | high |
|---:|---:|---:|
| 3 m | 381 | 795 |
| 200 m | 342 | 594 |
| 4 km | 136 | 248 |

The tier is a starting point, not a cap: `[` `]` `,` `.` `-` `=` still move all
of it at runtime, up to the desktop ceiling.

**These numbers are reasoned from cost ratios, not measured on a handset** —
which is a real difference from the rest of the constants here, and the reason
the overrides below exist. What does not depend on the guesses is the adaptive
scaler: it steps resolution down when frames overrun and back up when they do
not, with asymmetric hysteresis so the reallocation hitch is never mistaken for
the overrun that justifies another.

It steers by **GPU time, not wall-clock frame time**, and that is not a
refinement. Wall-clock time under vsync is quantised to the refresh interval: a
healthy frame on a 60 Hz phone measures 16.7 ms whether the GPU spent 4 ms or
15 ms on it. Against an absolute 15 ms budget that reads as a permanent
near-overrun, and "comfortably under budget" — the test for headroom — becomes
unreachable, because the interval cannot fall below the refresh however cheap
the frame gets. The first version of this scaler steered by wall clock and
could therefore drop resolution and never restore it. GPU time is the quantity
resolution actually controls and it is not quantised. A wall-clock path
survives for browsers without timestamp queries, with thresholds in multiples
of the measured refresh interval rather than absolute milliseconds, clamped to
60 Hz so a machine that never once hits its refresh rate cannot adopt a dropped
frame as its baseline.

`npm run scaler` is the regression: ten cases over both clocks, including the
drop-then-recover that the wall-clock version failed.

```
?quality=low|medium|high   force a tier on any device
?scale=0.75                pin the render scale, bypassing the scaler
```

The HUD prints the tier, why it was chosen, and the live scale.

## Controls

| | |
|---|---|
| `W` `A` `S` `D` | fly — speed scales with altitude |
| `Space` / `Ctrl` | up / down · `Shift` boost ×8 |
| drag | look · wheel adjusts speed multiplier |
| `L` | jump to the most rugged land on the planet |
| `T` / `R` | drop to surface / reset to orbit |
| `F` | ground-follow at eye height |
| `V` / `B` | vegetation on·off / band debug colours |
| `;` `'` | vegetation density · `N` read instance counts |
| `O` / `P` | sun elevation / azimuth (hold Shift to reverse) |
| `H` | shadows on·off — **off by default**, see below · `I` invert vertical look |
| `1`–`7` | shading: natural, LOD, slope, normals, cover, albedo, climate |
| `G` | metric grid: off → 100 m → 10 m → 1 m → 10 cm |
| `[` `]` | LOD factor · `,` `.` octaves · `-` `=` max level |
| `` ` `` | the readout — hidden by default |

## What M1 establishes

**The precision architecture.** The naive planetary vertex shader computes
`normalize(warp(uv)) * R - camPos` in f32, differencing two values near
6.4 × 10⁶ where the ULP is 0.76 m. It cannot represent the surface it draws.
This build never forms the full-magnitude vector on the GPU: the per-patch
anchor arrives with its subtraction already done on the CPU in f64, and the
within-patch offset is reconstructed analytically — tangent addition for the
warp delta, a cancellation-free expansion for the direction delta. Details and
derivations in [src/shaders/terrain.ts](src/shaders/terrain.ts).

Measured by `npm run precision`, against an f64 ground truth:

| level | sample spacing | this build | naive f32 |
|---:|---:|---:|---:|
| 0 | 36.0 km | 304 mm | 733 mm |
| 10 | 35.2 m | 1.02 mm | 904 mm |
| 16 | 550 mm | 16.1 µm | 904 mm — **broken** |
| 19 | 68.7 mm | **2.0 µm** | 720 mm — **broken** |

Error holds at ~3 × 10⁻⁵ of the sample spacing at *every* level, which is what
correct relative precision looks like. Naive f32 sits at a fixed ~0.8 m floor
and exceeds its own sample spacing from level 16 down.

**Crack-free LOD.** `sim.audit()` in the console renders to an offscreen target
cleared to a colour the terrain cannot produce and counts background pixels
enclosed by terrain. Currently 0 at every altitude from 3 m to 8000 km — at the
low tier's coarser LOD factor as well as the desktop one, which is the case
that could have opened seams and does not.

That takes two things, not one. The skirt closes vertical mismatches — the
two-level jumps a single-parity morph cannot bridge. `PATCH_BLEED` closes
lateral ones: neighbouring patches agree about their shared edge analytically
but not *bitwise*, because each reconstructs it from its own anchor in f32, and
a rasteriser leaves a pixel of background wherever two nearly-identical edges
diverge. Patches therefore overlap by 1e-5 of their own size — under a
hundredth of a pixel at any level. Without it the seams split into hairlines
near the ground: 0.54% of terrain pixels at 49 m. See LESSONS §23.

**GPU-driven vegetation.** The CPU uploads ~24 tiles × 20 floats per frame and
dispatches. Placement, gating, LOD binning and the draw counts all happen on
the GPU; the indirect draws read their instance count from the buffer the
compute pass wrote, and nothing is ever read back. Cost is O(candidate cells),
independent of how many instances survive.

Cost breakdown at 55 m altitude, 1440p, measured by toggling each system:

| | frame | triangles |
|---|---:|---:|
| terrain only | 5.6 ms | 1.83 M |
| + shadows | 9.5 ms | 5.00 M |
| + vegetation | 14.3 ms | 5.00 M + fill |
| **after the fixes below** | **4.9 ms** | **2.89 M** |

The shadow pass was 75% of all geometry — the full horizon selection was being
drawn into a cascade covering 61 m — and it is vertex-bound, because every
shadow vertex pays the same 17-octave terrain shader as a display vertex.

215 k instances is above SPEC §8's target of ~1.9 × 10⁵ canopy trees within a
1 km radius, so the browser is not the constraint here. Band split at full
density is 0 / 6.3 k / 208.6 k — the far band covers 96% of the scattered area,
which is why band capacities are equal rather than tapered.

**Seamless transitions.** The forest used to end at a visible circle. It now
dissolves, because the scatter and the terrain shading share one canopy-cover
field — the ground tint and where plants actually stand are literally the same
function, so instances can fade into the tint without revealing an edge.
Instances thin over the outer half of the range, and sub-pixel quads fade out
rather than alias.

**Atmosphere.** Single-scattering Rayleigh + Mie, shared by terrain, foliage
and the sky dome so the horizon is continuous by construction. Aerial
perspective is not decoration: distant LOD changes and the vegetation fade sit
behind progressively more air, which is why those transitions are invisible in
a photograph and obvious without it. Shading is now roughly physical —
reflectances lit by sun irradiance, through an ACES curve.

**Cascaded shadows.** Three cascades, camera-relative, sized from altitude and
faded out above 14 km where they stop contributing. The camera-relative
architecture makes this unusually simple: every mesh already emits positions
relative to the camera, so a light camera in that same space needs no world
transform, and shadow matrices never hit the precision loss that normally
forces cascade re-centring hacks. Depth is linear distance along the light in
an R32F target rather than a hardware depth texture, so the test is a plain
comparison. Terrain and foliage share one pass; each caster keeps its own
vertex node graph, so the surface in the shadow map is exactly the surface
drawn. Costs ~5 ms at ground level, nothing from orbit.

**Measured realism, not eyeballed.** `npm run hypsometry` compares the
elevation distribution against Earth's and can solve for better parameters
(`-- --solve`). Land fraction 29.2% vs Earth's 29.2%; land median 850 m vs
797 m; 4.1% above 3100 m vs 4.9%.

## Measurements

Descent over one ground track, terrain at 2501 m, 1280×720:

| above ground | patches | nodes visited | LOD | cracks |
|---:|---:|---:|---|---:|
| 8000 km | 34 | 54 | 1–2 | 0 |
| 400 km | 104 | — | 3–6 | 0 |
| 4 km | 300 | — | 5–13 | 0 |
| 900 m | 468 | 790 | 5–15 | 0 |
| 200 m | 658 | — | 5–17 | 0 |
| 3 m | 878 | — | 5–19 | 0 |

Frame time 8–13 ms at 1440p on an M-series laptop; CPU patch selection
0.1–0.8 ms of that. Patch count plateaus once max LOD is reached, which is the
expected behaviour for a log-scale LOD.

## Known limits

**Detail bottoms out at ~35 m features.** Not an octave-count problem. The
field is evaluated at `dir * F` from a unit vector, so the fractional position
inside a noise cell is quantised to `F · 2⁻²³`; once that exceeds ~4% of a cell
the field visibly steps. No octave count fixes it — the limit is intrinsic to
evaluating a global function from a unit vector in f32. Detail below ~35 m
arrives at M5/M6 by sampling per-tile data instead, which sidesteps the problem
entirely (SPEC.md §6). Press `.` past the ceiling to watch it break.

The ceiling is now *derived* from `AMP_F0` rather than hard-coded
(`maxOctavesFor` in [src/planet.ts](src/planet.ts)). It had been a literal 17,
which was right for M1's base frequency of ~3 and silently wrong once M3 raised
the start to 260 to sit under the bake's Nyquist: at 260 the real ceiling is 10
octaves, so seven of them ran at frequencies where one ULP of the direction
vector moves whole lattice cells — 3.5 of them by the last. They cost eight
hash evaluations each per vertex and returned quantisation noise, and because
they were also the steepest octaves they were most of what made mountains
render as vertical facets.

**Vegetation and ground still differ in texture at the handoff.** Tone now
matches, and the fade has no hard edge, but instanced crowns have relief and
inter-crown shadowing while the tinted ground is smooth. Closing that needs the
canopy normal and height baked into the terrain virtual texture (SPEC §8), not
more tuning.

**Water has no geometry.** The surface is still exactly the sphere; what
changed is that it is no longer shaded as one. A wave field perturbs the
*normal* — see `waveBlock` in shaders/terrain.ts — which buys the sun glitter
path, the break-up of the sky reflection and whitecaps, none of which need the
surface to actually move. What it does not buy is silhouette: a swell never
occludes what is behind it, and at eye height on open water that is a real
absence. Displacement needs the water to be its own mesh, which it is not,
because shading it as part of the terrain is what lets the shoreline resolve
per pixel instead of along a seam.

The octaves are rungs of the same LOCAL_PERIOD lattice the ground detail uses,
so the field wraps seamlessly, and each rung advects at its own deep-water
phase speed — c = sqrt(gλ/2π), so the 128 m swell runs at 14 m/s and the
half-metre chop at 0.9. Octaves finer than a pixel are not dropped; their mean
square slope is added to the specular roughness in quadrature, which is what
turns the sun's reflection from a tight blob up close into a glitter path at
range without ever aliasing.

Not modelled: anisotropy in the sample coordinate (it would rotate the period
lattice out of alignment and seam every 4096 m, so the wind direction is
applied to the gradient instead), refraction of the seabed, caustics, and
foam persistence — whitecaps appear and vanish with the slope that made them
rather than decaying.

**Shadows are off by default.** The cascades draw a hard-edged trapezoid of
false shadow on flat ground at low altitude — the footprint of a cascade's own
orthographic box, its interior uniformly darkened rather than showing any
silhouette, which puts the caster mesh nearer the light than the receiver
across the whole region. `H` turns them on; the pass itself is intact and
behaves above the altitude where the trapezoid appears. Diagnosis so far: not
the cloud deck (present with cover at 0), not missing casters (those clear to
`NO_OCCLUDER` and fail *lit*), and not plain depth-slope bias (adding it
changed nothing, though the clamp may have been too tight).

**Shadow quality is basic.** 3×3 PCF, no contact hardening, no filtering that
adapts to cascade. Good enough that relief and canopy read correctly; not good
enough to look at closely.

**Vegetation is billboard quads, not plants.** The point of this stage was the
substrate, not the art: one camera-facing quad per instance with a procedural
crown mask. Geometry for the near band, octahedral impostors for the mid band,
and the >1 km bake into the terrain virtual texture are all still to come. The
forest currently stops dead at 1100 m because that last handoff does not exist
yet.

**The height field is a placeholder.** Continents plus ridged relief, tuned to
match Earth's hypsometric curve. It has no drainage, no erosion, no climate.
Rivers and everything that follows from them arrive at M3/M4.

**Two-level LOD jumps are covered, not prevented.** Near the ground the LOD
distance varies by roughly one patch width per patch, so a two-level jump can
appear at a quadtree boundary, which a single-parity morph cannot bridge.
Prevention needs either cross-face neighbour queries for 2:1 balance or
`lodFactor > 4.25` (4× the patch count). Skirts cover the gaps instead, at 13%
more triangles. Revisit if skirts ever become visible.

**WebGPU allows 8 vertex buffers per pipeline and the geometry uses all 8.**
Adding a ninth fails pipeline creation with no visual clue beyond a black
screen. Tile data at M5 will need packing or a storage buffer.

## Layout

```
src/
  planet.ts       constants; every calibrated number lives here
  cubesphere.ts   face bases, tangent warp, face↔direction
  quadtree.ts     CDLOD selection, frustum + horizon culling
  terrainMesh.ts  shared patch grid, skirt ring, instance plumbing
  controls.ts     orbit-to-ground camera, f64 world position
  quality.ts      device tier, and the render scale that tracks the frame clock
  heightCPU.ts    CPU mirror of the height field, for ground-following
  devAudit.ts     offscreen crack counter
  vegetation.ts   GPU scatter, atomic band binning, indirect draw
  shaders/
    terrain.ts    WGSL: precision reconstruction, noise, shading
    vegetation.ts WGSL: scatter candidate, billboard, foliage shading
tools/
  precision.ts    f32 vs f64 vertex error, per level
  scaler.ts       adaptive resolution, against vsync-quantised frame time
  mirror.ts       what the renderer draws, vs Earth's curve; CPU/GPU agreement
  slopes.ts       slope distribution vs sampling scale; the convergence check
  bakeSweep.ts    tectonic parameters against the hypsographic curve
```

`npm run slopes` is the one to reach for after touching the amplification. It
reports the slope distribution at 300 m down to 3 m sampling, and the number
that matters is not the level but whether the increments *shrink*: an fBm with
H < 1 has no slope limit, so it gets steeper without bound the closer you look,
and a hypsometric curve can be perfect while the surface between those
elevations is a field of vertical facets.

### Checking realism

Two halves of one suite, over the same 24 fixed sites — every biome at both the
flat and the steep end of its relief range, plus a shoreline and a trunk valley
floor. The site table is `src/tour.ts`.

```bash
npm run realism              # check against tools/realism.baseline.json
npm run realism -- --update  # accept the current numbers as the baseline
```

It measures, per site, the median slope at 300 m / 30 m / 3 m, the ratio
between the outer two — which is the signature the per-biome octave spectrum
controls — the relief over a 5 km transect, and the spectrum the biome table
asks for; then land fraction, the amplification's zero mean, and the biome
shares planet-wide. Four seconds, CPU only, exits non-zero on drift.

What a number cannot check is whether it *looks* like anywhere, so the same
list is a camera tour in the running app:

```js
sim.tour()                 // list the sites
sim.tour('chaparral-steep')
sim.tour(17, 12000)        // the same site from 12 km
sim.tour.next()            // step through, wrapping
```

The sun is aimed in each site's own horizon frame, so the light is identical
relative to the camera at every stop — otherwise one azimuth means a different
thing at each site and half the tour comes out backlit.

`sim` is exposed on `window` for driving from the console:
`sim.controls`, `sim.terrain`, `sim.selector`, `sim.vegetation`,
`await sim.audit()`, `await sim.vegCounts()`.

## Next

Two independent fronts, in rough priority order:

1. **HZB occlusion culling** on the vegetation substrate. Canopy occludes
   almost everything behind it, so this is the largest remaining win — but
   scene arrangement decides whether it pays, so measure before assuming.
2. **M2/M3** — tile store and streaming, then the global bake. Until B2 runs
   there is no drainage, and drainage is what makes terrain read as real.

See SPEC.md §12 for the full order.
