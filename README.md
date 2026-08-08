# world_sim

Planetary terrain, orbit to ground, fully procedural. TypeScript + three.js on
WebGPU. Architecture and roadmap live in [SPEC.md](SPEC.md).

**Status: M1 complete** — cube-sphere quadtree, CDLOD, camera-relative
precision, and a placeholder analytic height field. No streaming, no bake, no
vegetation yet; those are M2 onward.

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
| `1`–`4` | shading: natural, LOD level, slope, normals |
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
  shaders/
    terrain.ts    WGSL: precision reconstruction, noise, shading
tools/
  precision.ts    f32 vs f64 vertex error, per level
  hypsometry.ts   elevation distribution vs Earth; parameter solver
```

`sim` is exposed on `window` for driving from the console:
`sim.controls`, `sim.terrain`, `sim.selector`, `await sim.audit()`.

## Next

M2 is the tile store, streaming, and parent-fallback — no hitch on a 500 m/s
traverse, still no hitch with 200 ms of forced disk latency. See SPEC.md §12.
