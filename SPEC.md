# Planetary Terrain — Architecture Spec

**Goal:** one continuous world from orbit (6.4 × 10⁶ m) to blades of grass (10⁻² m), realistic enough to survive scrutiny at both ends, at 60 Hz.

**Status:** v0.3 — M0/M1 implemented and measured. See [README.md](README.md).

**Decisions locked:** continuous orbit-to-ground is a pillar · fictional planet, fully procedural · single player · 5 GB install ceiling · **TypeScript + three.js on WebGPU**.

---

## 0. The thesis

Nine orders of magnitude separate a planet from a pine needle. No single technique spans that. Two obvious plans both fail:

- **"Noise all the way down."** Fails on realism. Real terrain is the residue of *processes*, and the dominant ones (tectonic uplift, fluvial incision, glaciation) are **non-local**. A river's course depends on its entire upstream basin. This is why pure procedural terrain always reads as procedural.
- **"Bake the whole planet."** Fails on tractability, and on your 5 GB budget. Earth at 1 m/sample is ~1 PB.

The resolution:

> **Solve globally at coarse resolution, where geomorphic processes are non-local and the grid is small enough to afford. Amplify locally, where detail is statistically stationary and must be free.**

The crossover sits where geomorphology becomes stationary — the scale of a single hillslope, **30–300 m**.

---

## 1. "Can we do this entirely procedurally?"

Two different questions hide in that sentence. The answers differ.

### "No hand-authored content — everything from a seed?" — **Yes, completely.**

You ship a generator, not a planet. Nothing about your world is authored: tectonics, drainage, climate, biomes, and every tree position derive from one integer seed. Your install contains code, shaders, and a library of *species and materials* — no terrain data at all. §9 shows this fits 5 GB with room to spare.

### "A pure function `h(p)` evaluated per-frame, zero precomputation?" — **No. And this is a theorem, not an engineering limitation.**

Realistic terrain requires `h(p)` to depend on the drainage basin containing `p`. Identifying that basin requires flow routing across the whole basin. Basins reach continental scale — the Amazon is 7 × 10⁶ km². So a genuinely "pure" `h(p)` would have to evaluate `h` over ~10⁶ km² of neighbours before it could return a single value.

At which point you have done the bake. You just didn't cache it.

> **Precomputation here is not a compromise. It is the memoization of an unavoidable computation.**

The underlying reason: a river network is a **global constraint-satisfaction problem** — every point drains to exactly one outlet, monotonically. Local functions cannot satisfy global constraints. It's the same reason a noise function can't emit a solved maze.

Anyone selling you "fully runtime-procedural realistic terrain" has either precomputed something and not told you, or has terrain that doesn't drain.

### What this actually costs you: one bounded, hierarchical cascade

The saving grace is that the non-locality is **scale-bounded**. Detail at 10 m does not depend on the far side of the continent — it depends on the *discharge and base level inherited from its parent*, which is a handful of numbers. So the precomputation is a finite recursive cascade, not an infinite regress:

| Stage | Grid | Domain | When | Cost |
|---|---|---|---|---|
| **Global LEM** | 4 km | Whole sphere, one domain | Once, first run | ~30 s – 3 min |
| **Regional LEM** | ~500 m | Per super-tile, inherits parent | Lazy, on approach | ~50 ms/tile |
| **Local LEM** | ~30 m | Per tile, inherits parent | Lazy, on approach | ~5 ms/tile |
| **Amplification** | ≤ 4 m | Per tile, pure function | Runtime, per-frame | ≤ 1.5 ms/frame |

Only the top row must be eager, because only the top row is genuinely global. Everything below is **local given its parent**, and therefore lazy, cacheable, evictable, and re-derivable.

That is what "entirely procedural" means in practice, and it is the whole architecture.

---

## 2. Invariants

Not guidelines. Each violation produces a bug class that is very expensive to retrofit.

**I1 — Determinism.** Every world quantity is a pure function of `(position, seed, parent data)`. No visit-order dependence.
*Single-player relaxation:* cross-machine determinism is **not** required — GPU floating-point differs across vendors and nobody can tell that two planets differ by a metre. But *within* a machine it is absolute. Consequence: **bake once, persist, never re-derive.** A driver update must not silently reshape a player's world.

**I2 — Scale consistency.** Refinement adds detail with ≈zero mean over the parent cell. For levels `i < j`: `|h_i(p) − h_j(p)| < ε(i)`, `ε` shrinking monotonically. Approaching a mountain must never move the mountain.

**I3 — Anchoring.** Placed objects compute ground height from a **fixed reference LOD**, never the resident one. Otherwise they float and sink as you approach.

**I4 — Precision hygiene.** No `f32` holds a planet-absolute coordinate. At 6371 km, `f32` ULP is **≈0.76 m**; `f64` is 1.4 nm. World space is `f64`/`i64`; `f32` only in tile-local or camera-relative frames, converted **before** GPU upload.

**I5 — No hitches.** Missing data renders the parent LOD — never a hole, never a stall. Amplification has a hard per-frame tile cap.

---

## 3. Spatial index — cube-sphere quadtree

**Cube-sphere with tangent warp.** Not HEALPix, not a naive normalized cube.

- Naive `normalize(cube_point)` gives **1.30×** area distortion corner-to-centre. Tangent warp `u' = tan(u·π/4)` cuts it to **≈1.05×**, invisible, and costs one `tan` in a rarely-called function.
- Near-uniform cell area also matters for the LEM: flow accumulation is only physical if cells have comparable area.
- HEALPix is exactly equal-area but has 12 diamond faces; neighbour topology and GPU atlasing get materially worse. Not worth 5%.
- **Sphere, not ellipsoid** — fictional planet, so the ellipsoid buys nothing and complicates every projection.

**Geometry** (R = 6371 km): surface 5.101 × 10¹⁴ m² → face area 8.502 × 10¹³ m² → **face edge ≈ 9221 km**.

**Tile = 256² samples + 2-texel border** (258²), so filtering and normals are correct without neighbour lookups. Budget for the border from day one; retrofitting it is miserable.

GSD = 9.221 × 10⁶ / (2^L · 256):

| Level | GSD | Role |
|---:|---:|---|
| 5 | 1126 m | |
| 7 | 281 m | Regional LEM output |
| 11 | 17.6 m | Local LEM output |
| 13 | 4.40 m | **Object anchor LOD (I3)** |
| 16 | 0.55 m | Amplification |
| 19 | 6.9 cm | Amplification floor |

Tile ID packs into `u64`: `face:3 | level:5 | x:28 | y:28`. Morton order for streaming and cache locality.

**Cross-face topology:** the global LEM runs directly on the cube-sphere, so flow routing needs a neighbour table that crosses face seams. The 8 cube corners are 3-valent — special-case them; they are 8 cells out of 3 × 10⁷.

---

## 4. Global bake — runs once, at first launch

### B0 — Tectonics
Voronoi plates on the sphere; classify boundaries convergent / divergent / transform. Derive **uplift rate**, **lithology**, **crustal age**. ~50 km resolution. Cheap, and it gives everything downstream a reason to be where it is.

### B1 — Base relief
Multi-fractal at 4 km, shaped by B0's uplift and lithology. This is an initial condition for B2, not a deliverable — don't over-invest.

### B2 — Global landscape evolution ← **the realism step**

Stream-power incision over the **entire planet as one domain**:

1. Flow routing (D8/D∞) → receiver graph
2. Depression routing (carving, not jumping — jumping leaves flow discontinuities)
3. Implicit stream-power incision `∂z/∂t = U − K·A^m·S^n` (Braun–Willett `O(n)`)
4. Hillslope diffusion + sediment deposition
5. Iterate ~200 steps

**Concrete cost.** [FastFlow (CGF 2024)](https://www-sop.inria.fr/reves/Basilic/2024/JKGFC24/) reports a 2048² grid, 200 iterations, **under 10 seconds on an RTX A6000** — ≈8.4 × 10⁷ cell-iterations/s, with `O(log n)` flow routing and `O(log²n)` depression routing.

Extrapolating:

| Global grid | Cells | VRAM (10 × f32) | A6000 | Modern high-end | Verdict |
|---|---:|---:|---:|---:|---|
| **4 km** | 3.2 × 10⁷ | 1.3 GB | ~75 s | **~30 s** | **Default** |
| 2 km | 1.3 × 10⁸ | 5.1 GB | ~6 min | ~2 min | Optional "high detail" |

**4 km is the default.** It resolves continental drainage, divides, ranges, and basins — everything genuinely non-local. Smaller catchments are handled by the regional stage, which inherits from it. A ~30-second first-run cost, presented as the planet forming, is defensible UX. 2 km is a quality setting for players with ≥12 GB VRAM.

*Reference implementations:* FastFlow for the GPU routing kernels; [Cordonnier et al. 2016](https://www.cs.purdue.edu/cgvlab/www/publications/cordonnier2016large/) for the uplift-plus-stream-power formulation, which is the canonical graphics treatment of exactly this stage.

### B3 — Hydrology graph ← **highest-value artifact in the pipeline**

Vectorize channels above a discharge threshold into a tree. Per node: position, discharge `Q`, width (`w ≈ a·Q^0.5`), depth, bank slope, sinuosity class, base level. ~10⁶–10⁷ nodes, **100–300 MB**.

A resampled heightfield gives *approximately* continuous rivers. A vector graph gives *exactly* continuous rivers at every LOD, for free. Rivers are the most diagnostic feature of realistic terrain and where noise fails most visibly.

### B4 — Climate
Insolation from latitude and axial tilt; **advective orographic precipitation** over B2's topography; temperature lapse rate; seasonality. 10–20 km. One afternoon of work, and rain shadows are a top-tier realism cue — deserts land where deserts belong, and the biome layer inherits the reason.

**Total eager bake: B0–B4, ~30–60 s, output ~400 MB compressed.**

---

## 5. Lazy stages — generated on approach, cached, evictable

### B5 — Regional LEM (→ L7, 281 m)
Per super-tile, with **discharge and base level inherited from B2/B3** as boundary conditions. Same solver, small domain. Neighbouring tiles agree because they agree with their shared parent, not because they were computed together.

### B6 — Local LEM (→ L11, 17.6 m)
Same recursion, one level down. This is where hillslope-scale channels and small catchments appear.

### B7 — Biome fields
Not a hard Whittaker lookup — that's the generic-procgen look. Inputs: B4 climate + topographic wetness index + slope + aspect + lithology + disturbance history (fire, windthrow, avalanche). Output: **fuzzy membership**, 2–4 dominant weights per cell, not one label. Transitions blend; species mixes vary continuously.

---

## 6. Runtime amplification (L11 → L19)

```
h(p) = h_parent(p)                              //  bicubic from tile atlas
     + Σ_k A_k(ctx) · W_k(ctx) · noise_k(p)     //  context-modulated detail
     + carve_channels(p, hydro_graph)           //  exact, from B3 vectors
     + landform_ops(p, ctx)                     //  scree, terraces, dunes, ...
```

`ctx = { lithology, slope, curvature, flow_accum, dist_to_channel, altitude, climate, biome_weights }` — all inherited, all cheap.

**The realism is in the modulation, not the noise.** Uniform fBm looks synthetic at any octave count; the same fBm driven by lithology and drainage context looks like geology.

Rules:
- Octave amplitude ≤ ~0.5 × wavelength × local slope tolerance, or you get spikes that read as noise.
- Each band ≈zero mean over its parent cell (**I2**).
- Channel carving must be **idempotent and clamped** — carve *toward* a target profile, never by a delta. Deltas compound across LOD and destroy I2.

**Landform operators**, in value order:
1. Slope-limited talus/scree relaxation — the highest return of anything here
2. Ridge sharpening along the negative-curvature skeleton
3. Lithologic terracing
4. Glacial U-section where paleo-ice is flagged
5. Coastal wave-cut platforms and beach profiles keyed to fetch
6. Dune fields keyed to wind × sand availability

One compute dispatch per tile → heightmap atlas. Hard cap **N tiles/frame** (start at 4).

---

## 7. Orbit-to-ground specifics

These only matter because you chose continuous traversal. Each is load-bearing.

**Depth precision.** Reverse-Z with an infinite far plane, `float` depth buffer. Non-negotiable across this range — the conventional setup collapses entirely.

**Atmosphere.** Precomputed multiple-scattering LUTs (Bruneton-style). The single biggest visual payoff for orbit views, and it must be physically parameterised or the ground-level sky won't match the orbital limb.

**Ocean.** FFT wave spectrum near, analytic BRDF far, with a distance-based handoff. The horizon must curve correctly — at 6371 km it's visible from ~2 km up.

**Clouds.** Volumetric near, 2D layer from orbit, with a handoff. Expensive; budget it explicitly or it eats the frame.

**Speed-adaptive streaming.** Prefetch along the velocity vector, with lookahead proportional to speed. At orbital velocity you cross tiles faster than disk can serve them — the parent-fallback rule (**I5**) is what keeps this from being a hole in the world.

**LOD hysteresis.** Split and merge thresholds must differ, or hovering at a boundary thrashes the amplification cache. Start at 15% separation.

**Shadow cascades.** Standard CSM assumes a bounded view distance. You need a custom scheme that degrades to no shadows above ~10 km altitude, where they contribute nothing anyway.

---

## 8. Vegetation — the infinite forest

Start from density, because it dictates the architecture. Over a 1 km view radius (314 ha) in temperate forest:

| Layer | Density | In view |
|---|---:|---:|
| Canopy trees | ~600 /ha | 1.9 × 10⁵ |
| Understory shrubs | ~5000 /ha | 1.6 × 10⁶ |
| Ground cover | ~100 /m² | 3.1 × 10⁸ |

You cannot instance 3 × 10⁸ of anything. **LOD must change representation, not mesh density:**

| Distance | Representation | Live budget |
|---|---|---:|
| 0–30 m | Full geometry, per-leaf | ~2 K |
| 30–150 m | Reduced geometry, card leaves | ~15 K |
| 150–1000 m | Octahedral impostors, 1 quad, GPU-culled | ~300 K |
| > 1000 m | **Baked into the terrain virtual texture** | 0 |

Ground cover: geometry to ~20 m, shell/parallax to ~60 m, texture beyond.

**The critical handoff is at 1 km: canopy stops being objects and becomes a material property of the terrain.** Bake per-tile canopy albedo, roughness, and normal into the terrain VT and the forest reads correctly from orbit at zero instance cost. Bake canopy *height* too, so shadowing and wind stay plausible.

### Do we need Nanite? — no, and mostly it is the wrong tool

Worth settling explicitly, because losing Nanite Foliage was the stated cost of not using Unreal (§10).

Nanite solves one problem: *an authored mesh with millions of triangles, and nobody wants to hand-author its LODs.* It does that with a cluster DAG — split into ~128-triangle clusters, group, simplify, re-split, with locked shared boundaries so any cut through the DAG stays watertight. This project has that problem in exactly one place.

| Content | Verdict |
|---|---|
| **Terrain** | Already solved, and better. A heightfield CDLOD *is* a cluster-LOD scheme: topology is implicit, boundaries are watertight by construction, and the morph is continuous. No DAG, no simplification error metric, no build step. Measured 0 crack pixels (M1). Nanite's machinery exists to solve problems a heightfield does not have |
| **Vegetation** | Nanite's triangle path actively **fails** here. Aggregate geometry — leaves, grass — is disjoint shells with open edges; simplification deletes whole elements and the canopy thins out. That is precisely why Epic shipped **Nanite Voxels** in 5.7, voxelising into 4×4×4 bricks to preserve silhouette, rather than decimating. The ladder above is the same insight reached from the other side: **change representation, don't decimate** |
| **Rocks, cliffs, hard-surface props** | The one genuine use case. Revisit if and when scanned detail meshes actually appear |

**What people actually credit to Nanite is the GPU-driven pipeline underneath it,** and that part is separable, much cheaper, and needed here regardless:

- compute culling, instance → cluster, frustum plus two-phase occlusion against a hierarchical depth buffer
- indirect draw with GPU-determined counts — no CPU draw-call cost, no readback
- screen-space-error LOD selection, continuous, so nothing pops

That is what removes popping and makes cost independent of scene complexity. It is also exactly the machinery the deterministic scatter above already needs. **Build the substrate, not the DAG.**

*WebGPU note:* a full Nanite port is possible — [nanite-webgpu](https://github.com/Scthe/nanite-webgpu) does meshlet DAG, GPU culling, software raster and impostors in a browser. But WGSL has **no 64-bit atomics** (proposed as `atomic-64-min-max`, not shipped; native-only in wgpu), so Nanite's core trick — packing depth and payload into one `atomicMax<u64>` — is unavailable. The workaround is packing u16 depth plus 2×u8 into a 32-bit atomic, which costs z-fighting and rules out textured software rasterisation. That loss lands squarely on the use case we do not have.

### Deterministic scatter — O(1) per instance

```
for each cell c in the species-spacing grid over the tile:
    r = hash(c.xy, face, species_id, world_seed)
    p = c.centre + blue_noise_offset[r & 63] * cell_size
    if density_field(p, species) < r.f01:         continue
    if slope(p) > species.max_slope:              continue
    if !altitude/moisture/biome_gates(p):         continue
    emit(p, h_anchor(p), scale = f(r, site_quality), rot = r, variant = r)
```

- `h_anchor` samples the **fixed anchor LOD (L13, 4.4 m)** — never the resident LOD (**I3**).
- Multi-species: fixed priority order with exclusion radii. Still order-independent, because the order is a constant.
- Compute shader per tile → indirect-draw instance buffer. Regenerated only on tile load.

### Cheap realism, high payoff

- **Size distribution must be inverse-J** — many small stems, few large. Uniform or Gaussian sizing is exactly what makes procedural forests read as planted orchards. This single change outperforms any amount of asset quality.
- **Clumping:** density × low-frequency noise × disturbance history → glades, gaps, thickets.
- **Edge effects:** density and species mix shift within ~30 m of a forest boundary.
- **Deadfall, snags, stumps** at a few percent — disproportionate lived-in payoff.
- Canopy gap-size distribution should follow a power law.

---

## 9. The 5 GB budget — it binds on art, not terrain

Because the planet is procedural, **you ship zero terrain data.** The constraint lands entirely on the species and material library.

**Install (target 5 GB):**

| Item | Size |
|---|---:|
| Engine, game code, shader libraries | 1.2 GB |
| Canopy species (8–10 × 3 variants, Nanite + textures) | 1.1 GB |
| Understory (~20 species) | 300 MB |
| Ground cover (~30) | 150 MB |
| Terrain materials (16–24 layers, 2K BC7) | 250 MB |
| Rocks / detail meshes | 300 MB |
| Audio | 400 MB |
| UI, fonts, misc | 100 MB |
| **Total** | **≈3.8 GB** |

**≈1.2 GB margin.** Two levers if it tightens:
- **Impostor atlases are generated at cook or first run** from the source meshes — near-zero ship cost. Never author them.
- **Procedural variation beats asset count.** UE's Procedural Vegetation Editor generates Nanite-ready plants from graphs; combined with runtime variation (scale, bend, hue, branch pruning, seasonal tint), a handful of base species yields apparent diversity that a large authored library can't match per megabyte.

**Generated cache — outside the install, user data, capped and evictable:**

| Item | Size |
|---|---:|
| Global LEM + climate (B0–B4) | ~400 MB |
| Hydrology graph (B3) | 100–300 MB |
| Regional/local tiles (B5–B6), LRU | 2 GB cap |
| Amplified tiles, LRU | 500 MB cap |

Treat it like a save file: written once, never silently invalidated (**I1**).

---

## 10. Stack — TypeScript + three.js on WebGPU

Chosen over Unreal because there is already working Three.js experience here,
and because the architecture is compute-shader-shaped rather than
engine-feature-shaped: the parts that matter (cube-sphere quadtree, CDLOD,
precision reconstruction, the bake pipeline) are all things you write yourself
in any stack.

The apparent cost was Nanite Foliage at §8. On inspection that cost is small:
Nanite's triangle-cluster LOD is the wrong tool for aggregate geometry, which
is why Epic replaced it with voxels for foliage, and the terrain already has a
better-suited continuous LOD. What is worth taking from Nanite is the GPU-driven
substrate — compute culling, indirect draw, screen-space-error LOD — which is
stack-independent and on the roadmap anyway. See §8, "Do we need Nanite?".

- **WebGPU, not WebGL2.** Non-negotiable: WebGL2 has no compute shaders, and
  B2/B5/B6 and the scatter at §8 all need them.
- **No f64 in WGSL, ever.** WGSL has no f64 at all. This turns out not to
  matter, because §I4 already forbids f32 planet-absolute coordinates on the
  GPU. JS numbers are f64, so world-space CPU maths is correct for free, and
  the GPU only ever sees camera-relative quantities.
- **The cube-sphere partitions data into six faces**, which keeps every buffer
  well under WebGPU's default 128 MB storage-binding limit without special
  handling.
- **Portable to native later.** `wgpu` (Rust) is the same API and shading
  language; the WGSL would port close to verbatim if the browser's ceiling
  becomes the binding constraint.

**Measured limits found while building M1** — these are stack facts, not
guesses, and they shape M2 onward:

| Limit | Value | Consequence |
|---|---|---|
| Vertex buffers per pipeline | **8** | Already all 8. Tile data at M5 must pack or use storage buffers. Exceeding it fails pipeline creation with no visual clue beyond a black screen |
| f32 noise from a unit vector | **~19 m features** (17 octaves) | Detail below this cannot come from a global function — it must come from per-tile data (§6). Empirically confirms the amplification design |
| `wgslFn` inlining | per-stage | Two wgslFn sharing a helper name collide; helpers are emitted with distinct suffixes |

*Not chosen:* Unreal 5.8 (Lumen is the real loss; the planetary core would have
been custom anyway and World Partition works against a sphere), Bevy/Rust (no Nanite-class foliage either, and less iteration speed
here), Godot (weaker on all counts for this).

Third-party, offline only:

| Project | Role | License |
|---|---|---|
| [Hesiod](https://github.com/otto-link/Hesiod) / [HighMap](https://github.com/otto-link/HighMap) | Prototyping the B5/B6 operator stack | **GPL-3.0** ⚠ see below |
| FastFlow | GPU flow/depression routing kernels (B2) | verify |
| fastscapelib | CPU reference LEM, for validating B2 | verify |

### ⚠ The GPL boundary

**Hesiod and HighMap are GPL-3.0.** Linking HighMap into your runtime makes
your runtime GPL-3.0. But **the GPL binds the program, not the data it emits.**

> **Hesiod belongs on your workstation, as an interactive sandbox for designing
> the amplification and landform stack. It never ships. The license boundary is
> the filesystem.**

Since the runtime generates everything, Hesiod's value is not producing shipped
assets — it is letting you design and tune §6's operator chain interactively
before porting it to WGSL. That is a real accelerant, and it is clean.

**Action item:** verify HighMap's actual `LICENSE` file. The README states
GPL-3.0 but repository metadata surfaced LGPL-2.1; LGPL-2.1 would permit
dynamic linking and change the calculus.

## 11. Frame budget

1440p / 60 Hz, mid-high GPU (16.7 ms):

| System | Budget |
|---|---:|
| Terrain geometry | 2.5 ms |
| Vegetation | 4.0 ms |
| Amplification / VT update | 1.5 ms amortized |
| Shadows | 2.0 ms |
| Atmosphere / clouds | 1.5 ms |
| Shading + post | 5.0 ms |
| **Headroom** | **0.2 ms** |

Already tight. Live debug HUD from M1, not a reckoning at the end.

**VRAM:** heightmap atlas (4096² × 16, R16) 512 MB · albedo/normal VT 1 GB · instance buffers 256 MB · streaming staging 256 MB.

---

## 12. Build order

Each milestone independently verifiable. **Do not proceed past a failing verification.**

| # | Milestone | Verification |
|---|---|---|
| ~~**M0**~~ | ~~Precision spike~~ | ✅ **2.0 µm at L19** vs 720 mm naive; error holds at 3×10⁻⁵ of sample spacing at every level. `npm run precision` |
| ~~**M1**~~ | ~~Cube-sphere quadtree, CDLOD, analytic noise~~ | ✅ Orbit → ground, **0 crack pixels at every altitude** (`sim.audit()`), 8–13 ms at 1440p, 34–878 patches |
| **M2** | Tile cache, streaming, parent-fallback, speed-adaptive prefetch | 500 m/s traverse hitch-free; still hitch-free with 200 ms forced disk latency |
| **M3** | Global bake B0–B4 at 4 km | Completes in < 60 s. Drainage density, Horton ratios, hypsometric curve within Earth range |
| **M4** | Hydrology graph + runtime carving | A river is continuous and monotonically downhill, source to sea, at every LOD |
| **M5** | Regional + local LEM (B5–B6), lazy and cached | Tile boundaries agree without cross-tile communication |
| **M6** | Amplification stack | Sample `h` at L11 and L19 over 10⁶ random points; assert \|Δ\| < ε (**I2**) |
| **M7** | Biomes + scatter, one species | ◐ GPU-driven substrate done: deterministic scatter, atomic band binning, indirect draw. **215 k instances at 11.7 ms**, no CPU per-instance cost. Biomes and the anchoring check still open |
| **M8** | Vegetation LOD ladder + VT canopy bake | Frame budget holds under dense-forest worst case |
| **M9** | Atmosphere, ocean, clouds | Orbital limb and ground-level sky agree |
| **M10** | Multi-species, disturbance, edges, understory | Visual review against reference photography |

**M0 and M1 are complete** — see [README.md](README.md) for measurements.
Three findings from building them are worth carrying forward:

- **The elevation envelope is load-bearing.** `MIN_ELEVATION` must stay below
  the field's true minimum: it is the occluder for horizon culling, and setting
  it too high silently disables culling near the ground, where 100× of the
  traversal cost lives.
- **LOD distance and morph distance must use the same reference sphere.**
  Measuring the morph against sea level while selecting against the camera's
  ground radius saturates the morph at every deep level; standing on 2.5 km of
  terrain it produced cracks across the whole near field.
- **Bounding volumes are wrong for LOD distance.** A patch's bounding sphere is
  dominated by the global elevation span, which collapses the metric to zero
  within ~6.6 km and forces maximum subdivision over a huge area. Measure to
  the surface, not the volume.

---

## 13. Measuring "realistic"

"Highly realistic" must be measurable or it becomes an endless eyeball loop.

| Metric | Expected |
|---|---|
| Hypsometric curve | Match Earth within tolerance |
| Drainage density (km channel / km²) | Match by climate zone |
| Slope distribution | Roughly exponential, **not** Gaussian |
| Horton–Strahler bifurcation ratio | 3–5 |
| Elevation power spectrum | β ≈ 2 over 4+ decades |
| Tree DBH distribution | Inverse-J |
| Canopy gap-size distribution | Power law |

The power spectrum is the sharpest tool. Pure fBm gives a perfectly straight log-log line. **Real terrain deviates at the fluvial rollover** — where river incision takes over from hillslope processes. Matching that deviation, rather than a straight line, is a large fraction of what "realistic" means.

---

## 14. Explicitly out of scope

- **Runtime hydraulic/droplet erosion.** Non-deterministic across LOD (violates I1, I2), expensive, and worse-looking than vector-graph carving.
- **Ellipsoid geodesy.** Fictional planet; sphere.
- **HEALPix.** Not worth 5% area distortion.
- **Shipping any baked terrain.** The generator is smaller than its output by four orders of magnitude.
- **Voxels / caves / overhangs in v1.** Separate system, own streaming and meshing. Design the heightfield so it *can* be added; don't add it. (UE 5.8's Mesh Terrain handles overhangs but is planar-oriented — not your path.)
- **Literal infinity.** 19 quadtree levels is 7 cm resolution over 5 × 10¹⁴ m². Not the limiting factor on anything.
- **Multiplayer.** Single-player is what relaxes I1 to per-machine. Adding a server later means re-deriving the bake deterministically across vendors — a hard, separate problem. Decide now if that's ever likely.

---

## Open questions

1. **Target hardware floor?** Decides the 4 km vs 2 km default and the whole frame budget. A 6 GB-VRAM floor changes several tables.
2. **First-run bake — acceptable as a one-time 30–60 s "planet forming" screen,** or does it need a progressive path into play?
3. **Planetary parameters:** radius, gravity, axial tilt, rotation period, sea-level fraction. These feed B0–B4 directly. Earth-like unless you want otherwise.
4. **Is multiplayer ever plausible?** Cheap to preserve now, expensive to retrofit (§14).
5. **Team size and timeline?** The §12 ordering assumes you can carry ~1 person-year of custom rendering before M8.
