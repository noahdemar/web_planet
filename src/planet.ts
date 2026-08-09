/**
 * Planet parameters. See SPEC.md §3.
 *
 * All lengths in metres. All world-space arithmetic happens in JS numbers,
 * which are IEEE f64 — that is deliberate and load-bearing (SPEC.md I4).
 */

export const RADIUS = 6_371_000;

/** Terrain elevation envelope, relative to the reference sphere. */
export const MAX_ELEVATION = 9_000;
export const MIN_ELEVATION = -4_200;

/** Cube-sphere face edge length at the surface: sqrt(4πR² / 6). */
export const FACE_EDGE = Math.sqrt((4 * Math.PI * RADIUS * RADIUS) / 6);

/**
 * Sea level, expressed as a threshold on the continentalness field rather
 * than as an elevation — that is what actually controls the land fraction.
 * Calibrated by `npm run hypsometry` to put land near Earth's 29%.
 */
export const SEA_LEVEL = -0.0617;

/** Half-width of the land/ocean transition in continentalness units. */
export const SEA_BAND = 0.18;

/**
 * Elevation weights, in metres. Tuned against Earth's hypsometric curve by
 * `npm run hypsometry`; the shader carries the same three numbers.
 */
export const RELIEF_BASE = 700; // broad continental undulation
export const RELIEF_PEAK = 9000; // ridged relief, after the power below
export const RELIEF_POWER = 7; // skews relief toward lowland
/**
 * fBm gain for the ridged relief. With lacunarity 2.07, gain g gives
 * amplitude ∝ λ^H for H = ln(1/g)/ln(2.07); g = 0.5 yields H ≈ 0.95, which is
 * self-similar but far too gentle — octave 12 then carries ~4 m of relief
 * across a 735 m wavelength. Real landscapes run nearer H ≈ 0.6.
 */
export const RELIEF_GAIN = 0.62;
export const RELIEF_LACUNARITY = 2.07;
export const OCEAN_DEPTH = -5400; // basin floor where the land mask is zero

/** Deepest quadtree level. L19 ≈ 6.9 cm ground sample distance at 256²/tile. */
export const MAX_LEVEL = 19;

/**
 * Mountain octaves for the M1 placeholder field.
 *
 * The ceiling is not performance, it is f32. Noise is evaluated at `dir * F`
 * with |dir| = 1, so the fractional position inside a lattice cell is
 * quantised to F·2⁻²³. Past 17 octaves (F ≈ 3.3e5, features ≈ 19 m) that
 * quantisation exceeds ~4% of a cell and the field visibly steps.
 *
 * No octave count fixes this — the limit is intrinsic to evaluating a global
 * function from a unit vector in f32. Detail below ~20 m arrives at M5/M6 by
 * sampling per-tile data instead, which sidesteps the problem entirely
 * (SPEC.md §6).
 */
export const DEFAULT_OCTAVES = 17;

/**
 * Vertices per patch edge. 33 → 32 segments, 2048 triangles per patch.
 * Must be odd so the CDLOD parent grid is a strict subset (SPEC.md §6).
 */
export const PATCH_VERTS = 33;
export const PATCH_SEGS = PATCH_VERTS - 1;

/** Upper bound on simultaneously drawn patches; sizes the instance buffers. */
export const MAX_PATCHES = 4096;

/**
 * LOD range for level L is LOD_FACTOR × (surface edge length at L).
 * A node subdivides while the camera is nearer than this. Larger = more detail.
 */
export const DEFAULT_LOD_FACTOR = 2.2;

/**
 * Fraction of a node's active range at which CDLOD morphing begins.
 *
 * Not a free parameter. A level-L patch is a quadrant of a parent whose
 * nearest point lies within range[L-1], so its own farthest vertex can reach
 *
 *     (1 + √2 / lodFactor) · range[L-1].
 *
 * At the shared edge with a coarser neighbour the L patch is fully morphed
 * onto the L-1 grid, so the neighbour must still be *unmorphed* out to that
 * same distance — otherwise the finer patch presents an L-1 grid while the
 * coarser one has already started collapsing toward L-2, leaving a T-junction.
 * The neighbour begins morphing at 2·MORPH_START·range[L-1], hence:
 *
 *     MORPH_START ≥ (1 + √2 / lodFactor) / 2
 *
 * Larger lodFactor relaxes this. Getting it wrong does not fail loudly: it
 * produces a scatter of single-pixel holes that grows with the number of LOD
 * transitions on screen. `sim.audit()` counts them.
 */
export function morphStartFor(lodFactor: number): number {
  return Math.min(0.93, Math.max(0.55, (1 + Math.SQRT2 / lodFactor) / 2 + 0.02));
}

/** Surface edge length of one patch at level L. */
export function edgeLengthAt(level: number): number {
  return FACE_EDGE / Math.pow(2, level);
}

/** Ground sample distance at level L, given 256 samples per tile edge. */
export function gsdAt(level: number): number {
  return edgeLengthAt(level) / 256;
}

/* ── Vegetation (SPEC.md §8) ─────────────────────────────────────────── */

/**
 * Quadtree level the scatter runs on. L14 → 562.8 m tiles, which is small
 * enough that a tile's instances share a coherent LOD band and large enough
 * that a 1 km view radius needs only ~14 of them.
 */
export const VEG_LEVEL = 14;

/** Scatter cells per tile edge. 128 → 4.4 m spacing ≈ 520 stems/ha. */
export const VEG_CELLS = 128;

/**
 * Tiles resident at once. Must comfortably exceed the number within
 * VEG_TILE_RANGE (≈15 at L14, plus boundary), because the traversal emits in
 * quadtree order, not by distance — overflow silently keeps an arbitrary
 * subset and empties the bands nearest the camera.
 */
export const MAX_VEG_TILES = 32;

/** Distance beyond which canopy becomes a terrain material, not objects. */
export const VEG_RANGE = 1100;

/**
 * Representation bands. LOD here changes what an instance *is*, not how many
 * triangles it has — decimation is the wrong tool for aggregate geometry
 * (SPEC.md §8, "Do we need Nanite?").
 */
export const VEG_BANDS = [
  { name: 'near', maxDist: 45 },
  { name: 'mid', maxDist: 220 },
  { name: 'far', maxDist: VEG_RANGE },
] as const;

/**
 * Instances per band. Equal across bands on purpose: it makes the shader's
 * base offset plain arithmetic (`band * capacity`) instead of a branch chain.
 *
 * Only the far band ever approaches this — it covers the 220–1100 m annulus,
 * which is 96% of the scattered area. At 400k per band the instance buffer is
 * 19 MB, which is nothing, and the cap stops being what limits the forest.
 */
export const VEG_BAND_CAPACITY = 400_000;
export const VEG_CAPACITY = VEG_BAND_CAPACITY * VEG_BANDS.length;

/** Growth limits, metres. */
export const VEG_MIN_ELEVATION = 2;
export const VEG_TREELINE = 2600;
export const VEG_MAX_SLOPE = 0.55;

/**
 * Baseline canopy closure before clumping and the growth gates. Multiplies the
 * shared cover field, so it scales instance count and ground tint together.
 */
export const FOREST_DENSITY = 0.9;

/**
 * Where instances begin dissolving. A long tail matters more than a late one:
 * fading over the outer half of the range makes the forest *thin*, which the
 * eye reads as distance, whereas a short fade near the limit reads as an edge
 * however smooth it is. They must reach zero by VEG_RANGE, where the terrain's
 * canopy tint carries the forest alone.
 */
export const VEG_FADE_START = VEG_RANGE * 0.45;

/**
 * Tile selection radius. Only a small margin over VEG_RANGE is needed, not a
 * tile diagonal: tiles are selected by their *nearest* point, so any tile
 * holding an instance within VEG_RANGE necessarily has a nearest point within
 * VEG_RANGE too. The margin only covers the sphere approximation the selector
 * uses for distance. Overshooting here is expensive — tile count grows with
 * the square of this — and it is what starves the near bands.
 */
export const VEG_TILE_RANGE = VEG_RANGE + 120;

/**
 * Quads thinner than this many pixels only alias; fade them out instead.
 * Alpha-to-coverage resolves the crown *outline*, but a whole crown narrower
 * than a few pixels still crawls between frames however it is filtered.
 */
export const VEG_MIN_PIXELS = 4.5;
