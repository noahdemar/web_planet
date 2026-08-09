# world_sim

Planetary terrain, orbit to ground, fully procedural. TypeScript + three.js on
WebGPU. Architecture and roadmap live in [SPEC.md](SPEC.md).

**Status: M1 complete, plus the GPU-driven vegetation substrate.** Cube-sphere
quadtree, CDLOD, camera-relative precision, a placeholder analytic height
field, and a fully GPU-driven forest — scatter, LOD binning and draw counts all
on the GPU, no per-instance CPU work. No streaming and no bake yet; those are
M2/M3.

```bash
npm install
npm run dev        # http://localhost:5173
npm run verify     # typecheck + precision + hypsometry
```

Needs WebGPU (Chrome/Edge 113+, Safari 18+). WebGL2 is not a fallback — the
bake pipeline at M3 requires compute shaders.

## Controls

| | |
|---|---|
| `W` `A` `S` `D` | fly — speed scales with altitude |
| `Space` / `Shift` | up / down · `Ctrl` boost ×8 |
| drag | look · wheel adjusts speed multiplier |
| `L` | jump to the most rugged land on the planet |
| `T` / `R` | drop to surface / reset to orbit |
| `F` | ground-follow at eye height |
| `V` / `B` | vegetation on·off / band debug colours |
| `;` `'` | vegetation density · `N` read instance counts |
| `O` / `P` | sun elevation / azimuth (hold Shift to reverse) |
| `I` | invert vertical look |
| `1`–`6` | shading: natural, LOD, slope, normals, cover, albedo |
| `G` | metric grid: off → 100 m → 10 m → 1 m → 10 cm |
| `[` `]` | LOD factor · `,` `.` octaves · `-` `=` max level |

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
enclosed by terrain. Currently 0 at every altitude from 3 m to 8000 km.

**GPU-driven vegetation.** The CPU uploads ~24 tiles × 20 floats per frame and
dispatches. Placement, gating, LOD binning and the draw counts all happen on
the GPU; the indirect draws read their instance count from the buffer the
compute pass wrote, and nothing is ever read back. Cost is O(candidate cells),
independent of how many instances survive.

| density | candidates | instances | frame @1440p |
|---|---:|---:|---:|
| 0.42 (default) | 377 k | 90.5 k | 4.6 ms · 216 fps |
| 1.0 | 377 k | **214.9 k** | 11.7 ms · 86 fps |

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

**Detail bottoms out at ~19 m features.** Not an octave-count problem. The
field is evaluated at `dir * F` from a unit vector, so the fractional position
inside a noise cell is quantised to `F · 2⁻²³`; past 17 octaves that exceeds
~4% of a cell and the field visibly steps. No octave count fixes it — the limit
is intrinsic to evaluating a global function from a unit vector in f32. Detail
below ~20 m arrives at M5/M6 by sampling per-tile data instead, which sidesteps
the problem entirely (SPEC.md §6). Press `.` past 17 to watch it break.

**Vegetation and ground still differ in texture at the handoff.** Tone now
matches, and the fade has no hard edge, but instanced crowns have relief and
inter-crown shadowing while the tinted ground is smooth. Closing that needs the
canopy normal and height baked into the terrain virtual texture (SPEC §8), not
more tuning.

**No clouds, no water surface detail, no shadows.** The reference images imply
all three. Clouds and shadow cascades are M9; water is currently a Fresnel
term over a flat sphere, with no waves and no shoreline.

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
  heightCPU.ts    CPU mirror of the height field, for ground-following
  devAudit.ts     offscreen crack counter
  vegetation.ts   GPU scatter, atomic band binning, indirect draw
  shaders/
    terrain.ts    WGSL: precision reconstruction, noise, shading
    vegetation.ts WGSL: scatter candidate, billboard, foliage shading
tools/
  precision.ts    f32 vs f64 vertex error, per level
  hypsometry.ts   elevation distribution vs Earth; parameter solver
```

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
