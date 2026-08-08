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
