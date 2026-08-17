/**
 * Planet parameters. See SPEC.md §3.
 *
 * All lengths in metres. All world-space arithmetic happens in JS numbers,
 * which are IEEE f64 — that is deliberate and load-bearing (SPEC.md I4).
 *
 * Everything here describes the *planet* and is the same on every device. The
 * one exception is VEG_BAND_CAPACITY, which sizes a buffer rather than
 * describing the world — see the note on it.
 */

import { QUALITY } from './quality.js';

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

/**
 * Cells per cube-face edge in the global bake.
 *
 * Lives here rather than in bake/ because the *runtime* depends on it: AMP_F0
 * is derived from it, and DEFAULT_OCTAVES from that. Changing it and re-baking
 * moves the whole amplification band to match, with nothing else to touch.
 *
 *    256 → 0.39 M cells, 36 km spacing,   10 s to bake,  3.2 MB
 *    512 → 1.57 M cells, 18 km spacing,   66 s to bake, 12.7 MB
 *   1024 → 6.29 M cells,  9 km spacing,  433 s to bake, 50.5 MB
 *
 * 512 is the default because the bake runs *in the browser* now (see
 * planetSource.ts), and a minute of arithmetic behind a progress bar is a
 * first visit while seven minutes is not. The cost is real and measured: land
 * median falls 833 m → 530 m against Earth's 797, because the coarser solve
 * simply resolves less relief. Drainage structure is untouched — Horton's Rb
 * is 5.07 at R² 1.000 against 5.04 at 1024 — so what is lost is height, not
 * hydrology, and the amplification is zero-mean so it cannot give it back.
 * Raising the uplift to recover it is a bakeSweep job and has not been done.
 *
 * The 20× is not the 4× you would expect from the cell count: the landscape
 * evolution model is 91% of the run and needs more iterations to converge at
 * finer spacing, so it scales far worse than linearly. That is the number to
 * weigh against the sharper coastlines and drainage, and it is worth knowing
 * that most of the *visible* globe-scale gain came free from MIN_SELECT_LEVEL
 * rather than from here.
 */
export const BAKE_RES = 512;

/** Deepest quadtree level. L19 ≈ 6.9 cm ground sample distance at 256²/tile. */
export const MAX_LEVEL = 19;


/* ── Amplification (SPEC.md §6) ──────────────────────────────────────────
 *
 * Detail finer than the bake's 9 km cell. The shader and the CPU mirror both
 * read these from here rather than each carrying their own copy: the two
 * previously agreed by comment ("must match AMP_F0 in shaders/terrain.ts"),
 * which is exactly the kind of agreement that stops holding.
 */

/**
 * Frequency at which amplification starts, in cycles per unit direction —
 * wavelength is RADIUS/f.
 *
 * Derived from the bake's own resolution, not chosen. The bake resolves cells
 * of FACE_EDGE/BAKE_RES, so its Nyquist wavelength is two of those and the
 * band from there to twice that is already attenuated by the resample.
 * Starting a little inside the attenuated band puts detail back where the
 * resample took it out rather than double-counting relief the bake already
 * has.
 *
 * Tying it to BAKE_RES is what makes the bake resolution a free choice: at 512
 * this is 260 and the amplification starts at a 24 km wavelength; at 1024 it
 * is 520 and starts at 12 km. Change BAKE_RES, re-bake, and the octave
 * ceiling follows too, because DEFAULT_OCTAVES is derived from this.
 */
export const AMP_F0 = Math.round(1.47 * ((RADIUS * BAKE_RES) / (2 * FACE_EDGE)));

/**
 * Mean of (1 − |noise|)² for the shader's gradient noise, measured by
 * tools/ridgeMean.ts. Subtracted per octave so switching octaves on and off
 * with distance cannot move the coastline.
 */
export const RIDGE_MEAN = 0.7452;

/**
 * Mean of the billow term, 2|noise| − 1. Measured the same way RIDGE_MEAN is,
 * and cross-checked by reproducing RIDGE_MEAN to five figures in the same run.
 */
export const BILLOW_MEAN = -0.7123;

/**
 * Where the ladder changes the *shape* of its octaves, in octave index.
 *
 * Every octave used to be ridged — (1 − |n|)² — which is why the terrain read
 * as one crinkle repeated at every size rather than as a landscape. Real
 * relief does not work that way: a massif at 30 km is a rounded mass, and it
 * is only at the scale of spurs and gullies that ground develops sharp crests
 * with flat ground between them. A ladder whose rungs are all the same shape
 * can have exactly the right spectrum and still look like a texture, because
 * shape is what tells you how big something is.
 *
 * So the coarse rungs are billow — rounded humps — and the fine ones ridged,
 * crossing over across the first few octaves. Both are functions of |n| alone,
 * so the analytic gradient stays exact: they differ only in a factor that the
 * chain rule carries through unchanged, and each carries its own measured mean
 * so the blend is still zero-mean at every band limit (I2).
 */
export const SHAPE_SHARP_LO = 0.5;
export const SHAPE_SHARP_HI = 3.5;

/**
 * Largest noise frequency f32 can carry.
 *
 * Noise is evaluated at `dir * F` with |dir| = 1, so the fractional position
 * inside a lattice cell is quantised to F·2⁻²³. Past ~4% of a cell the field
 * visibly steps: vertices land on essentially arbitrary phases of the octave,
 * and because the finest octaves are also the steepest, the surface breaks
 * into vertical facets rather than merely losing detail.
 */
export const MAX_NOISE_FREQ = 0.04 * (1 << 23);

/**
 * Finest wavelength any global function of a unit direction may use, metres.
 *
 * MAX_NOISE_FREQ is the limit for a noise *value*: at that frequency f32
 * leaves ~50 steps of fractional position inside a lattice cell, which the
 * height field tolerates because the octave ceiling lands it below the limit.
 * A *derivative* does not tolerate it. `noised_().yzw` is built from the same
 * quantised fraction and its error is proportionally larger, so a term that
 * perturbs the normal turns 50 steps into a visible staircase — measured as a
 * hard sawtooth comb along every shading gradient, at exactly the 19 m
 * wavelength the detail normal was floored at.
 *
 * A quarter of the limit gives ~200 steps per cell, which is smooth. The floor
 * used to be the bare literal `RADIUS / 336000`, sitting precisely on the value
 * limit with no headroom at all and no note that it was doing so.
 *
 * Anything finer than this cannot come from a global function and has to come
 * from per-tile data with a local origin (SPEC.md §6, M5) — the same wall
 * DEFAULT_OCTAVES hits.
 */
export const MIN_NOISE_LAMBDA = RADIUS / (MAX_NOISE_FREQ * 0.25);

/**
 * Repeat period of the near-field detail lattice, metres.
 *
 * The way under the f32 wall. MIN_NOISE_LAMBDA is a hard floor only for a
 * *global* function of a unit direction: the coordinate is dir·(RADIUS/λ), it
 * reaches 10^7 for sub-metre λ, and f32 has nothing left for the fraction
 * inside a lattice cell.
 *
 * Nothing forces the near field to be a global function. Evaluated instead in
 * metres about an origin snapped to this period, the coordinate never exceeds
 * PERIOD/2 + the camera-relative distance — five thousand or so — which leaves
 * ten clear bits of lattice fraction at a quarter-metre wavelength. That is
 * four thousand times the precision the global form has there.
 *
 * The price is that the field repeats every 4 km. Detecting that means
 * comparing centimetre-scale gravel between two points 4 km apart, so it is
 * not a price. The snap must be a whole number of lattice cells or the field
 * would jump as the camera crosses a boundary, which is why the period is a
 * power of two and the rungs below are the period divided by powers of two —
 * then wrapping the cell index is a mask, and the seam is exact rather than
 * approximate.
 */
export const LOCAL_PERIOD = 4096;

/**
 * Usable octaves for an fBm starting at `f0`.
 *
 * Derived, not chosen. This was a hard-coded 17, which was right for the M1
 * placeholder's base frequency of ~3 and silently wrong the moment M3 raised
 * the start to AMP_F0 = 260 to sit under the bake's Nyquist. At 260 the
 * ceiling is octave 9; octaves 10–16 were running at frequencies where one ULP
 * of the direction vector moves whole lattice cells — by octave 16, 3.5 of
 * them. They cost eight hash evaluations each per vertex and returned
 * quantisation noise.
 *
 * Detail below the resulting ~35 m floor arrives at M5/M6 by sampling per-tile
 * data instead, which sidesteps the f32 limit entirely (SPEC.md §6).
 */
export function maxOctavesFor(f0: number, lacunarity: number = RELIEF_LACUNARITY): number {
  return Math.floor(Math.log(MAX_NOISE_FREQ / f0) / Math.log(lacunarity)) + 1;
}

/**
 * Derived, never chosen.
 *
 * This was the literal 14, and four of those octaves were past the ceiling
 * maxOctavesFor computes for AMP_F0 = 260, which is 10. Past it the field is
 * not detail, it is quantisation: one f32 ULP of the direction vector moves
 * the sample by 4.5% of a lattice cell at octave 10 and by 40% at octave 13,
 * so the position snaps to a coarse sub-lattice and the noise returns the same
 * value across axis-aligned bands. That is the rectilinear tread pattern
 * visible on flat, bright ground from a few kilometres up — not a texture, an
 * aliasing artefact of asking the lattice for detail it cannot address.
 *
 * The README already stated the mechanism and the 4%-of-a-cell threshold; the
 * constant simply drifted away from it. Deriving it means it cannot drift
 * again: change AMP_F0 or the lacunarity and the ceiling follows.
 *
 * It is also four octaves of eight hash evaluations per vertex, paid twice
 * because the shadow pass runs the same vertex shader — so removing them is a
 * straight performance win alongside the fix.
 *
 * `,` and `.` still move the count at runtime, so the broken range is still
 * there to look at, which is what the README means by pressing `.` past the
 * ceiling to watch it break.
 */
export const DEFAULT_OCTAVES = maxOctavesFor(AMP_F0);

/**
 * Wavelength below which relief stops being fractal, metres.
 *
 * Above this, amplitude falls as λ^H and slope therefore *grows* by
 * lacunarity^(1−H) ≈ 1.28 per octave. Over the scale range the amplification
 * spans that compounds to a factor of nine, and it is why the steepest terrain
 * rendered as vertical stripes — measured median slope ran 12° at 100 m
 * sampling, 36° at 10 m and 61° at 1 m, still climbing with no limit in sight.
 *
 * Real landscapes do not do that. Below the hillslope crossover — the scale at
 * which mass wasting outruns fluvial incision, tens to a couple of hundred
 * metres — slope is capped near the angle of repose and relief falls as λ¹.
 * Switching the gain to 1/lacunarity past this wavelength reproduces it
 * exactly: H = 1, so every finer octave contributes the *same* slope instead
 * of more, and the slope distribution converges rather than diverging.
 *
 * It is a per-octave constant, so the analytic gradient stays exact and the
 * band-limit invariant (SPEC.md I2) is untouched — dropping an octave still
 * low-passes the surface rather than rescaling it.
 *
 * This pair is now the *reference* spectrum rather than the only one: the
 * biome table carries a gain and a crossover per class, and these are what a
 * temperate broadleaf landscape gets — the country the ladder was fitted to.
 * See BIOMES in biome.ts, and `spectrumOctave` below.
 */
/**
 * Below this wavelength the ladder stops behaving like a fractal surface and
 * starts behaving like a hillslope.
 */
export const HILLSLOPE_WAVELENGTH = 700;
/**
 * Gain past the hillslope crossover.
 *
 * It was 1/lacunarity, which is exactly the value that makes every finer
 * octave contribute the *same* slope — H = 1.00. That was chosen to stop slope
 * growing without bound as octaves are added, and it does. What it also does
 * is guarantee that four or five sub-700 m octaves each add identical
 * roughness, and they stack into a uniform orange-peel crinkle with no
 * landform in it at all. Mountains came out as a field of same-sized bumps.
 *
 * Real ground is smoother than that below the drainage scale, and for a
 * physical reason: soil creep is diffusive, so it erases short-wavelength
 * relief faster than long. A hillslope is close to planar between its gullies.
 * 0.42 puts H at 1.19 there — slope now *falls* with each finer octave instead
 * of holding, so the finest rungs texture the surface rather than defining it.
 */
export const HILLSLOPE_GAIN = 0.42;

/**
 * There used to be a HILLSLOPE_F here — the crossover as a frequency, for
 * consumers that compared it against the octave's own frequency. Nothing wants
 * that now and it is deliberately gone rather than left as a convenience: the
 * ladder walks octave *offsets* (see `spectrumOctave`), and a frequency
 * comparison is the hard `select` that the per-point crossover cannot use.
 * Leaving it exported would be leaving the old semantics one import away.
 */
export const LOG2_LACUNARITY = Math.log2(RELIEF_LACUNARITY);

/**
 * The crossover, expressed as a position on the octave ladder: how many
 * octaves below AMP_F0 a wavelength of `lambda` sits. 0 is the first octave,
 * and the reference 700 m lands at 3.93.
 *
 * Two reasons this and not the frequency. The shader walks the ladder by
 * index, so an octave offset is one add per octave against a divide and a
 * comparison — and more importantly it is the right space to *blend* in. A
 * crossover is a scale, so the mean of 220 m and 1800 m should be 630 m and
 * not 1010 m; averaging octave offsets is averaging log wavelength, which
 * gives the former.
 */
export function spectrumOctave(lambda: number): number {
  return Math.log2((AMP_F0 * lambda) / RADIUS);
}

/**
 * Half-width of the crossover, in octaves.
 *
 * The switch used to be a `select` on frequency, which was exact when there
 * was one crossover for the whole planet: the ladder is a function of the
 * octave index alone, so a step in it is a step in nothing observable. With a
 * per-point crossover it stops being harmless — the octave at which the gain
 * drops flips along a contour of the biome blend, every finer octave changes
 * amplitude by 22% across that line, and the result is a step in the ground
 * tens of metres high following a climate boundary. Exactly the mistake
 * LESSONS §13 is about, in the one place it had been safe.
 *
 * Softening it over an octave either side removes the step and costs almost
 * nothing: measured against the hard select at the reference spectrum, the
 * detail RMS moves 0.6% and the mean slope 6%. Real landscapes do not have a
 * sharp spectral break there either.
 */
export const HILLSLOPE_SOFT = 1.0;

/**
 * Sub-grid relief amplitude, metres: a floor everywhere plus a term scaled by
 * the relief the bake already resolves.
 *
 * Large because the ridged sum they multiply is small — summing the octaves at
 * gain RELIEF_GAIN and dividing by the unweighted normaliser leaves a signal
 * of roughly ±0.1.
 */
/* ── Sub-grid valley network (SPEC.md §6) ────────────────────────────────
 *
 * The single largest gap between this planet and a satellite image, and it is
 * a gap in *structure* rather than in spectrum.
 *
 * The bake solves drainage globally, which is the one thing a local function
 * cannot do — but it can only resolve channels down to its own cell. At
 * BAKE_RES 1024 that is a 9 km cell and a measured drainage density of
 * 0.010 km/km², against Earth's 0.5–5. The bake's own diagnostic prints the
 * verdict: "sub-grid density is the runtime's job".
 *
 * Raising BAKE_RES does not fix it. Density ceiling is 1/cell, so even an
 * 8192 solve — 403 M cells — tops out at 0.89 km/km², the very bottom of
 * Earth's range, and 16384 is not a bake anybody is going to run. The
 * amplification has to invent the network, and the amplification it had could
 * not: ridged fBm with an erosion term makes a surface that is *weathered*,
 * with slopes that look eroded, but its valleys do not connect to anything.
 * Real terrain is organised by drainage at every scale, and connectivity is
 * the property that reads as real.
 *
 * The trick that makes this affordable is that the zero set of a smooth noise
 * field is already a connected, branching curve network, and the distance to
 * it is available in closed form to first order as |n| / |∇n|. So a valley
 * network costs one noise evaluation per level — the same as an octave — and
 * carries an analytic gradient, which this field must have.
 *
 * Four levels at these frequencies give a cumulative 2.45 km/km², inside
 * Earth's range and 250x what the bake resolves.
 */

/** Levels of sub-grid network. Each doubles the density of the one above. */
export const VALLEY_LEVELS = 4;
/**
 * Frequency of the coarsest level, cycles per unit direction.
 *
 * Twice AMP_F0, so the network starts just inside the band the bake has
 * stopped resolving rather than competing with channels the bake already
 * carved. At BAKE_RES 1024 that is a 6.1 km valley spacing, refining to
 * 770 m at the fourth level.
 */
export const VALLEY_F0 = AMP_F0 * 2;
export const VALLEY_LACUNARITY = 2;
/**
 * Trench half-width, in lattice cells of the level's own noise.
 *
 * Also the width the bias measurement is taken at — see tools/valleyMean.ts.
 * Valley spacing is about one lattice cell, so this is a valley floor about
 * an eighth of the distance to the next divide, which is what a mature
 * fluvial landscape looks like from above.
 */
export const VALLEY_WIDTH = 0.06;
/**
 * Depth of the coarsest trench, as a fraction of the local amplification
 * amplitude, and the falloff per level.
 *
 * A fraction of `ampAt_` rather than an absolute depth, so the network
 * inherits everything that field already knows: it collapses on floodplains,
 * fades at the shoreline, and scales with resolved relief. A plain stays a
 * plain and a range gets a gorge.
 */
export const VALLEY_DEPTH = 0.30;
export const VALLEY_DEPTH_GAIN = 0.62;
/**
 * Pixel window for the *shading* half of the network, in pixels across the
 * trench. Full weight at VALLEY_PIX_HI, gone by VALLEY_PIX_LO.
 *
 * Deliberately far more permissive than BAND_FADE, and the difference is the
 * difference between displacing a surface and shading one. BAND_FADE is fitted
 * to a morphing CDLOD grid, where a vertex landing on an arbitrary phase makes
 * the surface boil (LESSONS §4) — it has to be conservative. Nothing moves in
 * the fragment stage, sampling is one per pixel by construction, and the only
 * question is whether the trench is wide enough to draw. Below a pixel it must
 * go, because a sub-pixel trench is aliasing and nothing else; at two and a
 * half it is a line you can see.
 *
 * This is what puts the drainage on the ground at orbital altitude. At 70 km
 * a pixel is 72 m and the four levels are 10, 5, 2.5 and 1.3 pixels across, so
 * the whole network is renderable — while the mesh, at a vertex every kilometre
 * or so, can displace none of it.
 */
export const VALLEY_PIX_LO = 0.8;
export const VALLEY_PIX_HI = 2.5;

/**
 * Mean of the trench profile, per unit depth. Measured, not derived —
 * `npm run valleymean`. Subtracted so a level fading in cannot lower the
 * ground, which near sea level would move the coastline (I2).
 */
export const VALLEY_MEAN = 0.2006;

export const AMP_BASE = 150;
/**
 * Retuned with BAKE_RES: at 1024 the bake carries relief the amplification
 * used to have to invent, so leaving this at 6200 double-loaded it and put the
 * rugged median at 33° with 11% of the surface past 60° — back toward the
 * cliffs the hillslope crossover exists to prevent.
 */
/**
 * Retuned again, and this time downward.
 *
 * Cutting this was the wrong lever and it is back where it was. The mountains
 * were not too *tall*, they were too *rough* — the two are different problems
 * and only one of them lives here. Amplitude scales every octave at once, so
 * dropping it took the massifs down along with the crinkle and the result read
 * as flattened, which is worse than noisy: a range with no height is not a
 * range at all.
 *
 * Where the noise actually came from is the spectrum, not the amplitude — see
 * HILLSLOPE_GAIN. Steepening the roll-off below the hillslope scale moves
 * energy out of the finest rungs and leaves the coarse ones carrying the
 * landform, which is the shape of the fix: same relief, distributed the way
 * real ground distributes it.
 */
export const AMP_RELIEF = 4000;

/**
 * Baked-slope window over which sub-grid roughness ramps in.
 *
 * Fitted to the bake, by `npm run slopes`. The previous window of
 * [0.006, 0.085] was fitted to nothing: the baked land-slope distribution runs
 * median 0.0034, p90 0.0134, max 0.0909, so its lower edge sat above the 75th
 * percentile and its upper edge above the 99.95th. 78% of land therefore got
 * `relief` = 0 and an amplitude of ~51 m, which is why the largest continent's
 * interior carried 14 m of relief over a 20 km transect — a flat green plain
 * to the horizon, which is the exact failure AMP_RELIEF was raised to prevent.
 * Raising the amplitude could never fix it, because the gate in front of the
 * amplitude never opened.
 *
 * The window is narrow because the distribution is: median 0.0034 against a
 * p99 of 0.056, so anything wide enough to reach the tail leaves the bulk of
 * land pinned at zero. Placing the half-way point near the 78th percentile
 * spreads `relief` across the land that actually exists — 13% saturates, which
 * is about the fraction of Earth's land that is mountainous.
 *
 * Refit again at 512, by percentile rather than by eye. The 1024 window
 * [0.0017, 0.0155] spanned p14 to p83 of the baked land-slope distribution; at
 * 512 the same percentiles are [0.0015, 0.0111], because a coarser solve
 * resolves gentler gradients — median 0.00378 against 0.00502. Holding the
 * *percentiles* rather than the values is what keeps the same share of land in
 * the ramp: 30% below relief 0.05 and 19% saturated, against 36% and 18% at
 * 1024.
 *
 * Refit when BAKE_RES changes: a finer bake resolves steeper gradients, so the
 * whole distribution shifts. At 512 the land median was 0.0034 and the window
 * was [0.0012, 0.011]; at 1024 the median is 0.0048, and leaving the window
 * alone put 22% of land at full amplitude instead of 13% and took the rugged
 * median from 15° to 28°. Scaled by the same 1.41 the distribution moved.
 */
export const RELIEF_SLOPE_LO = 0.0015;
export const RELIEF_SLOPE_HI = 0.0111;

/**
 * ── Standing water ───────────────────────────────────────────────────────
 *
 * Where a river exists, as log₁₀ of upstream drainage area.
 *
 * Fitted to the measured distribution, not guessed — and the guess was badly
 * wrong. Multiple-flow-direction accumulation spreads area over the whole
 * hillslope, so the *median* land cell reads 8.45; a lower bound of 8.4 called
 * 54% of the planet's land a river and drew it as one. On this bake:
 *
 *   wet > 9.0    15.2% of land        wet > 10.5   0.66%
 *   wet > 10.0    2.1%                wet > 10.9   0.23%
 *
 * Earth's rivers cover roughly 0.3–0.6% of land, which puts the network edge
 * near 10.2 and a continental trunk near 11.8 (the observed maximum is 12.34).
 * Re-measure if the bake resolution or the MFD exponent changes: these are
 * percentiles of a distribution, not physical constants.
 *
 * Below the lower bound the waterline drops out of reach and no amount of
 * terrain can flood, which is how the network terminates without a branch.
 */
export const RIVER_WET_LO = 10.3;
export const RIVER_WET_HI = 11.6;

/**
 * Where the waterline sits relative to the bake's valley floor, metres, at the
 * smallest channel the bake resolves and at a continental trunk.
 *
 * A bake cell is 9 km across and the Amazon is 3 km wide, so the data does not
 * contain a channel and no amount of thresholding will extract one. What it
 * does contain is the valley floor. Put the waterline near that and the
 * *amplification* decides where the water goes — the flooded set is the low
 * line of the surface actually being drawn, so it is sinuous, it agrees with
 * the terrain at every zoom level by construction, and it widens downstream on
 * its own because the valley-floor damping in `height_` widens with drainage
 * area.
 *
 * The sign is what makes a trunk a river and a headwater a stream. Above the
 * valley floor the whole corridor floods, so the water is continuous — which a
 * trunk river has to be. Below it, only the low excursions of the ridged field
 * flood, so the water beads into pools and wet hollows — which is what a
 * seasonal headwater actually looks like, and it fails gracefully because a
 * broken small stream reads as ponds rather than as a broken river.
 *
 * Magnitudes are set against the amplification: a lowland trunk valley has
 * `valley` saturated, so its amplitude is 150 × 0.28 = 42 m and the ridged
 * field's excursions are roughly ±13 m. A headwater keeps most of the full
 * 150 m, so −14 floods only its deepest hollows.
 */
export const RIVER_HEAD_LO = -4;
export const RIVER_HEAD_HI = 4;

/**
 * Residual relief on a floodplain, metres.
 *
 * The amplification used to damp trunk valleys by a *factor* — ×0.28 — which
 * is wrong in the one place it matters. In mountainous country the unmodified
 * amplitude is (150 + 4000·relief), so 0.28 of it is still 380 m of ridged
 * relief laid across the valley floor, and no waterline a few metres under
 * that floor can produce a river: the water pools in whichever hollows happen
 * to fall below it and the network breaks into disconnected lakes, which is
 * exactly what it did.
 *
 * A floodplain is flat in absolute terms, not relative ones. Mixing toward a
 * fixed 10 m rather than scaling means the valley floor is genuinely a valley
 * floor whatever the surrounding relief, the river is continuous, and its
 * width comes from how fast `valley` falls off — which is the drainage area,
 * which is the right answer.
 */
export const FLOODPLAIN_AMP = 18;

/* ── Plains swell ────────────────────────────────────────────────────────
 *
 * ampAt_ sets its relief from smoothstep(RELIEF_SLOPE_LO, ..., slope), so
 * ground the bake left flat gets amplified least and flatness is
 * self-reinforcing. Measured on the shipped bake (`npm run plains`), the flat
 * sixth of the land carried a median of 4.6 m of relief over a 2 km window.
 * Real plains over that window run 5-15 m on an active floodplain, 10-50 m on
 * till and loess, 20-60 m on the Great Plains. Two kilometres of ground that
 * rises and falls by four metres is a billiard table, and no amount of fine
 * detail fixes it, because the missing signal is not texture — it is *form*.
 *
 * The mistake to avoid is reaching for the existing ladder. Raising AMP_BASE
 * or dropping RELIEF_SLOPE_LO multiplies every octave, so the plains gain as
 * much crest-scale noise as they do form, which is the failure already
 * rejected in the mountains. This is a separate, deliberately low-gain stack
 * that lives only in the band a plain is shaped in and dies out before it
 * reaches the octaves that read as roughness.
 *
 * LAMBDA is the coarsest wavelength. 22 km is the scale of the swells you
 * actually cross on a plain — long enough that cresting one changes what you
 * can see, short enough that four octaves at this lacunarity reach 2.5 km and
 * so put relief inside the window the flatness was measured in.
 *
 * GAIN is the whole character of the term. At 0.5 the stack loses half its
 * amplitude per octave against a lacunarity of 2.07, so relief falls very
 * nearly as wavelength: equal slope contribution per octave, no accumulation
 * toward the fine end, and the result reads as swell rather than as noise laid
 * over a flat plate. Raising it is what would make plains look busy.
 *
 * Plain gradient noise, summed signed — no billow, no ridge. It is therefore
 * zero-mean by construction and needs no bias term of the kind BILLOW_MEAN and
 * RIDGE_MEAN exist to remove, which matters here more than anywhere: plains sit
 * near sea level, and a term with a mean fading in as you approach would walk
 * the coastline (I2).
 */
export const PLAINS_LAMBDA = 22_000;
export const PLAINS_F0 = RADIUS / PLAINS_LAMBDA;
export const PLAINS_OCTAVES = 6;
export const PLAINS_GAIN = 0.62;

/**
 * Amplitude of the swell, metres — calibrated against `npm run plains`, and
 * larger than it looks because it is the peak of the stack before the gates.
 *
 * Measured on flat ground (median local relief), before and after:
 *
 *              2 km        10 km
 *     before    5.1         19.8
 *     after    15.2         43.4
 *
 * against real plains of 10-50 m over 2 km and 30-100 m over 10 km. The ratio
 * between the two windows was already right at 3:1 before any of this — what
 * was missing was scale, not spectrum, which is why the fix is an amplitude
 * and not a re-tuned gain.
 *
 * The shore gate costs a factor of about 2.6 on low ground: its median over
 * flat land is 0.378, because plains sit below SHORE_FLAT_HI. That is why the
 * number here is a few hundred metres rather than a few tens, and it is not an
 * accident worth removing — coastal plains really are flatter than inland
 * ones, and the alternative is letting a fading octave walk the coastline.
 *
 * Mountains pay nothing: the gate is the complement of ampAt_'s relief term,
 * so the hilly class is bit-identical before and after.
 */
export const PLAINS_AMP = 500;

/* ── Aeolian: dune fields ────────────────────────────────────────────────
 *
 * The second of the plains processes. A plain is flat because something
 * deposited it, and what deposited it decides its texture; the swell above is
 * the form all plains share, and this is the one that only dry ones have.
 *
 * Gated on climate rather than on the bake, which is why height_ now takes
 * clim at all. An erg needs sand supply, wind, and no water to fix it — so
 * arid *and* warm, on top of the same flat-land gate the swell uses. On this
 * planet that lands the fields in the subtropical dry belts, which is where
 * the Sahara, the Rub al Khali, the Namib and the Simpson all are, for the
 * same reason: the descending limb of the Hadley cell.
 *
 * STRETCH is the whole visual signature. Dunes are not lumps, they are long
 * parallel crests transverse to the prevailing wind, and the aspect ratio is
 * the thing the eye reads as "desert" before it reads anything else. The
 * anisotropy is a fixed diagonal scaling of the noise domain, the same trick
 * the cirrus layer uses: it makes the field vary more slowly along the polar
 * axis, so crests run roughly north-south under the easterly trades. Crude
 * next to a real sand-transport model, exactly right for the cost, and — this
 * is the part that matters — a *linear* map, so the gradient stays exact
 * rather than needing chain-rule terms for a rotating wind field.
 *
 * Ridged, so the crests are sharp and the interdune corridors flat, which is
 * what a dune field actually looks like from above. Ridged means a nonzero
 * mean, so RIDGE_MEAN is subtracted per octave exactly as height_ does it —
 * without that, every octave arriving as the mesh refines would lift the
 * desert floor and, near sea level, walk the coastline (I2).
 *
 * 2.5 km is the coarsest crest spacing: draa scale, the mega-dunes that
 * organise an erg. Three octaves reach 580 m, which is an ordinary dune.
 */
export const AEOLIAN_LAMBDA = 2500;
export const AEOLIAN_F0 = RADIUS / AEOLIAN_LAMBDA;
export const AEOLIAN_OCTAVES = 3;
export const AEOLIAN_GAIN = 0.5;
export const AEOLIAN_STRETCH = 4.5;

/** Dune height, metres. Draa run 50-150 m; simple dunes 10-30 m. */
export const AEOLIAN_AMP = 0;

/**
 * Moisture window. Below DRY_LO is full erg, above DRY_HI no sand at all.
 *
 * Fitted to coverage, not chosen. The first window (0.30/0.46) put a sand
 * weight over 12% of land; the hot-desert extent it implied was right at
 * ~14%, but a hot desert is not all sand sea — reg and hamada are most of it,
 * and Earth's ergs are about 6% of land. Tightening to this lands at 7.4%.
 */
export const AEOLIAN_DRY_LO = 0.14;
export const AEOLIAN_DRY_HI = 0.30;

/** Temperature window — sand seas are hot deserts, not tundra. */
export const AEOLIAN_WARM_LO = 0.45;
export const AEOLIAN_WARM_HI = 0.65;

/**
 * Wetness window over which the amplification collapses to FLOODPLAIN_AMP.
 *
 * Deliberately narrower and lower than the river window. The flattening has to
 * be *complete* before the waterline rises above the valley floor, or the
 * water floods only the hollows of a surface that is still 100 m rough and the
 * river beads into a chain of ponds — which is what it did at every wider
 * setting tried. Saturating at 10.8 puts full flattening on the 0.23% of land
 * with the largest drainage, which is the corridor the trunk rivers run in.
 */
export const VALLEY_WET_LO = 9.3;
export const VALLEY_WET_HI = 10.1;

/**
 * ── The channel ──────────────────────────────────────────────────────────
 *
 * A bake cell is 9 km across and the Amazon is 3 km wide, so the corridor the
 * bake resolves is the *valley*, not the river. Flooding the corridor gives a
 * dead-flat sheet 9 km wide, which from the ground is a lake with no far bank.
 * The channel has to be reconstructed sub-cell, and it has to come out
 * connected — a river that beads into ponds is worse than no river.
 *
 * The reconstruction is a piece of geometry rather than a piece of noise. Near
 * its axis the wetness field is a smooth ridge, so to second order
 *
 *     wet(x) ≈ wetAxis − ½κx²      ⟹   |∇wet| = κx,   so   x = |∇wet| / κ
 *
 * with x the distance to the axis — and no need to know where the axis is or
 * what value it takes there. The gradient comes out of the same four atlas
 * taps the terrain already does for the elevation gradient, so it is free.
 *
 * κ is *not* measured. Estimating it from the same 9 km stencil gives a second
 * difference dominated by its own noise, and dividing by it made the channel
 * flicker on and off along its length — the river came out as a chain of
 * ponds, which is the one failure mode worth designing against. A transverse
 * valley profile is about one and a half cells wide and drops about a decade
 * of drainage area across it, which fixes κ ≈ 2Δ/L² and makes the distance
 * simply proportional to the gradient. |∇wet| is smooth, it vanishes exactly
 * on the ridge, and it is continuous along it — which is all the channel
 * needs.
 *
 * Width is *not* hydraulic geometry, and trying to make it so was the mistake.
 * W ≈ 0.002·√Q puts a large river at a 200 m half-width, which is 2% of a bake
 * cell — two orders below anything a distance field built from that cell can
 * place. Asking for it produced a channel that existed only where the gradient
 * happened to pass within 2% of its own zero crossing, which is to say a
 * scattering of ponds.
 *
 * So the channel is sized to what the data can carry: a few hundred metres to
 * a couple of kilometres of half-width, growing with drainage area. That is
 * the river *and its floodplain* rather than the wetted channel alone, which
 * for a continental trunk is honest — the Amazon's is tens of kilometres —
 * and for anything smaller is the best this resolution supports. Narrower
 * water than this belongs to a later milestone with per-tile data, not to a
 * global 9 km field.
 */
export const CHANNEL_HALF_LO = 60;
export const CHANNEL_HALF_HI = 900;
/** W = k*sqrt(A). A 10^11 m² basin lands near 250 m of half-width. */
export const CHANNEL_WIDTH_K = 0.0008;
/**
 * Metres of distance-to-axis per decade-per-metre of *transverse* wetness
 * gradient.
 *
 * Transverse matters. Drainage area grows downstream, so ∇wet has a large
 * along-channel component that does not vanish anywhere — using the full
 * magnitude, the measured minimum of the distance field along a real trunk
 * valley was 2.7 km, and since that is wider than the channel the channel
 * never formed at all. Projecting out the flow direction — which the baked
 * elevation gradient already gives — leaves the part that does vanish on the
 * axis.
 *
 * Calibrated against a measured transect rather than derived: across a trunk
 * valley wet falls about 0.33 of a decade in the first 1.5 km, which fixes the
 * transverse curvature at ~2.9e-7 and, allowing for the smoothing a 9 km
 * central difference applies, puts the channel at roughly ±1.5 km here.
 */
export const CHANNEL_DIST_K = 1.5e7;
/** How deep the channel is cut below the floodplain, metres. */
/**
 * Depth of the carved channel, metres.
 *
 * Halved from 45, and the width bounds with it. The valleys were reading as a
 * dense mat of dark lines over every continent rather than as the few large
 * rivers you can actually pick out from orbit — on the Blue Marble you can
 * count the visible drainages on two hands. Depth and width both feed the
 * riparian tint, so cutting them thins the network in the same stroke.
 */
export const CHANNEL_DEPTH = 22;
/**
 * Fraction of the incision that would fill with water — unused.
 *
 * Kept as a record: river water is not drawn. See the long note in
 * waterLevel_ in shaders/terrain.ts for the three approaches that were built
 * and why all three fail at a 9 km bake resolution. The carve itself stays,
 * because a valley *is* resolvable at 9 km and reads as drainage without it.
 */
export const CHANNEL_FILL = 0.72;

/**
 * Wetness withdrawn from low-gradient ground, in log₁₀ of drainage area, and
 * the baked-slope window it fades across.
 *
 * The drainage the bake hands out is grid-shaped wherever the ground is flat,
 * and it is worth being clear about why that is not fixable downstream. Flow
 * on a plain has no gradient to follow, so the router is deciding between
 * neighbours on differences of millimetres — the filler's ε, the last digits
 * of the erosion — and whatever structure those have is what the water gets.
 * On a slope the real gradient buries all of it. On a plain it *is* the
 * signal, and it radiates from spill points in straight lines because that is
 * the shape of a flood front on a lattice.
 *
 * So rather than keep trying to make the plain's drainage right, this stops
 * drawing the part of it that cannot be. Subtracting from log-area on flat
 * ground keeps the trunks — a river with 10¹¹ m² behind it survives losing
 * 0.9 of a decade — and takes out the fan of minor tributaries that carries
 * all of the artefact and none of the information. Which is also what a plain
 * looks like: one defined channel, and drainage either side of it too diffuse
 * to see.
 *
 * It fades on the *baked* slope, so it costs nothing and it moves with the
 * terrain rather than with the camera.
 */
export const FLAT_WET_CUT = 1.9;
export const FLAT_WET_LO = 0.0020;
export const FLAT_WET_HI = 0.0140;

/**
 * Depth at which a filled basin counts as a lake, metres, with a soft edge so
 * the shoreline ramps across the bilinear interpolant rather than stepping at
 * a texel.
 *
 * Fitted to the bake, not to LAKE_MIN_DEPTH. The LEM leaves a great many
 * shallow depressions — measured on this bake:
 *
 *   > 1 m   13.9% of land       > 20 m   5.6%
 *   > 4 m   12.2%               > 40 m   0.7%
 *   > 10 m   9.5%               > 80 m   0.0%
 *
 * At the old 0.5 m that was an eighth of every continent under water, drawn as
 * blue texel-shaped ponds in every valley. Earth's lakes are about 1.8% of
 * land, which puts the threshold near 30 m. That is a deep cut — it keeps only
 * basins that would hold a real lake — and the shallow ones are better served
 * by the riparian moisture term, which already greens them.
 *
 * The window is wide on purpose. A lake here is one or two 9 km texels across,
 * and a narrow threshold on a bilinearly interpolated field cuts those into
 * hard texel-shaped polygons. Spreading the transition over 30 m of depth
 * turns that edge into a shoreline that shelves, which is both what a lake
 * margin looks like and the only way to get a curve out of data this coarse.
 */
export const LAKE_ON_LO = 22;
export const LAKE_ON_HI = 52;

/**
 * Octave fade window, in mesh samples per wavelength.
 *
 * An octave is worthless below 2 samples per wavelength and is faded out
 * there; the question is how *late* it should reach full strength. The
 * surface depends on the camera because this window does, so every metre of
 * relief that fades in as you descend is a metre of terrain that was not
 * there before — and near sea level it changes the land/water topology, which
 * is what makes islands and lakes appear out of nothing.
 *
 * Measured over a 200 km → 1 km descent in 8% altitude steps, on a 20 km
 * coastal box, the worst single step moves:
 *
 *   [1, 2.5]   12.6 m mean, 0.93% of the ground flips land/water
 *   [1, 4]     10.3 m mean, 0.64%
 *   [1, 6]     15.9 m mean, 1.12%
 *
 * Wider is not monotonically better, which is the non-obvious part: spreading
 * the fade drags *coarser* octaves into the transition, and their amplitude is
 * larger, so past [1, 4] the extra amplitude outweighs the gentler ramp.
 */
/**
 * Pixel-side fade for the fragment ladder, in units of `pixLam`.
 *
 * Separate from BAND_FADE because the two windows guard different things, and
 * conflating them cost three octaves of sharpness everywhere.
 *
 * BAND_FADE guards a *displaced* octave on a morphing CDLOD grid, where a
 * vertex landing on an arbitrary phase makes the surface boil (LESSONS §4).
 * The fragment ladder displaces nothing, samples once per pixel by
 * construction, and only has to stay above Nyquist.
 *
 * The old code reused BAND_FADE against a `pixLam` that already carried a 4x
 * factor, so the safety margin was applied twice: an octave reached full
 * weight only at *sixteen* pixels per wavelength, when the comment right above
 * it claimed four. At 1793 km that put full weight at a 30 km wavelength and
 * rendered the first amplification octave at 0.12 — which is why a desert from
 * orbit was a smooth blank. Onset at 3 px and full at 6 px is still three times
 * Nyquist and four times sharper.
 */
export const PIX_LAMBDA = 3.0;
export const PIX_FADE_LO = 1.0;
export const PIX_FADE_HI = 2.0;

export const BAND_FADE_LO = 1.0;
export const BAND_FADE_HI = 4.0;

/**
 * Amplification near sea level, as a fraction of full.
 *
 * The coastline is the one place where a small change in height changes the
 * *topology* of what you see rather than its shape, so it is the one place
 * worth spending detail to stabilise. Collapsing the amplification within a
 * few hundred metres of sea level pins the shoreline to the bake, which does
 * not depend on the camera, and islands stop materialising as you approach.
 *
 * It costs nothing where it matters: relief on ground above 400 m is
 * unchanged at 165 m RMS either way. And it is what coasts actually look
 * like — coastal plains and continental shelves are the flattest large
 * landforms there are.
 *
 * With BAND_FADE at [1, 4] this takes the worst descent step from 10.3 m and
 * 0.64% down to 5.6 m and 0.54%.
 */
export const SHORE_FLAT_HI = 300;
export const SHORE_FLAT_FLOOR = 0.08;

/**
 * Vertices per patch edge. 33 → 32 segments, 2048 triangles per patch.
 * Must be odd so the CDLOD parent grid is a strict subset (SPEC.md §6).
 */
/**
 * Water waves — a normal field, not geometry.
 *
 * The water surface here is not its own mesh: it is the terrain mesh shaded as
 * water wherever the waterline sits above the ground, which is what lets the
 * shoreline resolve per pixel instead of along a seam. So displacing vertices
 * the way a Gerstner ocean does is not available, and it is also not where the
 * realism is. Almost everything that reads as "water" at any distance beyond
 * arm's reach is *specular*: how the sky breaks up in it and how the sun
 * scatters into a glitter path. Both are functions of the surface normal, so a
 * normal field buys nearly all of it at none of the structural cost.
 *
 * The wavelengths are rungs of the LOCAL_PERIOD lattice — see localNoiseBlock.
 * That is what keeps the field seamless when the camera crosses a period
 * boundary, and it is why the octaves step by 4 rather than 2: they have to be
 * whole power-of-two divisions of the period or the wrap stops being exact.
 * WAVE_CELLS0 = 32 puts the longest rung at 4096/32 = 128 m, a plausible ocean
 * swell, and five octaves reach 0.5 m ripples.
 */
/**
 * Strength of the erosion feedback in the amplification fBm.
 *
 * Each octave is divided by `1 + EROSION_K * |sum of gradients so far|^2`, so
 * fine detail is suppressed wherever the coarser octaves have already produced
 * a steep face. That single term is the difference between a ridged fBm and an
 * *eroded* one: without it every octave contributes at full strength
 * everywhere, which is uniform lumpiness in all directions — the cauliflower
 * the 50 km survey shows. With it, detail survives on flats and shoulders and
 * is stripped from steep ground, which produces smooth valley floors, sharp
 * ridgelines, and structure that aligns to the slope instead of blobbing.
 *
 * The accumulator is deliberately *not* the same one the normal uses. That one
 * carries the frequency factor and so grows by a factor of two an octave,
 * reaching ~1e5 by the last rung; squaring it would make the divisor explode
 * and flatten the planet. This one drops the frequency, which leaves it O(1) —
 * the sum of the amplitude ladder — and therefore scale-free and safe to tune.
 *
 * Measured, against the pre-erosion baseline: at 1.15 it halves the planet
 * (broadleaf-steep 5 km relief 1076 -> 511 m, steppe-steep slope@3m 0.40 ->
 * 0.04), which is flattening rather than eroding. 0.15 costs 10-20% of relief
 * and keeps the slope growth ratios close to where they were — 3.45 -> 3.2 at
 * broadleaf-steep — while still stripping the fine octaves off steep faces,
 * which is the structure this exists for.
 *
 * The relief that is lost is a normaliser artefact, not a property of erosion:
 * mNorm is built from the *undamped* amplitude ladder, so damping mSum lowers
 * the total instead of redistributing it. Folding the eroded amplitudes into
 * the normaliser would let this run harder at no cost in relief, and is the
 * obvious next move on it.
 */
export const EROSION_K = 0.15;

/* ── Landform-aware amplification ────────────────────────────────────────
 *
 * The amplification ladder used to run the same spectrum everywhere and let
 * `ampAt_` scale it. That is why a mountain read as a field of same-sized
 * bumps: switch the amplification off and the bake underneath has massifs,
 * ridge systems and a trunk valley; switch it on and all of it disappears
 * under uniform crinkle. Additive isotropic noise laid over a landform reads
 * as noise, however carefully its amplitude is tuned — and tuning amplitude is
 * the wrong lever anyway, because it takes the mountain down with the crinkle.
 *
 * What changes here is *character*, not amount: how rough the ground is at
 * sub-kilometre scales now depends on where in the landform it sits, using
 * fields the bake already exports and every caller of height_ already passes.
 *
 * The dominant term is the facet. Steep ground is bedrock, scree or cliff, and
 * those are close to planar at a hundred metres — a mountain face is a *face*.
 * Gentle ground is soil and is dissected by its own small drainage, so it is
 * rough at exactly the scales the steep ground is smooth. The old ladder had
 * this backwards: RELIEF_SLOPE ramps the amplitude *up* with slope, so the
 * steepest ground got the most fine detail, which is where the crinkle was
 * loudest and where it did the most damage to the landform.
 */

/**
 * Baked slope over which ground becomes a facet. Fitted to the measured
 * distribution — land median 0.0034, p90 0.0134, p99 0.056 — so this catches
 * the steep tail and leaves ordinary hill country alone.
 */
export const FACET_SLOPE_LO = 0.008;
export const FACET_SLOPE_HI = 0.045;
/** Share of the fine-octave energy removed on a full facet. */
export const FACET_K = 0.78;
/**
 * Octave indices over which "fine" ramps in. At AMP_F0 the ladder runs
 * 12.25, 5.92, 2.86, 1.38, 0.67, 0.32, 0.16 km, so this damps from about
 * 1.4 km down — which is the band the crinkle lives in and well below
 * anything carrying a landform.
 */
export const FACET_OCT_LO = 2.5;
export const FACET_OCT_HI = 6.0;

export const WAVE_CELLS0 = 32;
export const WAVE_OCTAVES = 5;

/**
 * RMS slope of the longest rung, and the per-octave falloff.
 *
 * Slope rather than amplitude because slope is the only thing a normal field
 * can express, and it is the quantity that stays roughly constant across the
 * saturated part of a wind-sea spectrum. Open ocean sits near 0.1–0.2 RMS
 * total; too much reads as a storm at every latitude, too little as varnish.
 */
export const WAVE_SLOPE = 0.105;
export const WAVE_FALLOFF = 0.82;

/**
 * How much steeper the field is along the wind than across it.
 *
 * Applied to the *gradient*, never to the sample coordinate. Stretching the
 * coordinate is the usual way to get anisotropy and it cannot be used here:
 * the period lattice is axis-aligned, and any transform that is not a uniform
 * scale rotates the 4096 m wrap out of alignment with the cell grid and puts a
 * seam on every period boundary. Shaping the gradient afterwards costs one
 * multiply-add, keeps the wrap exact, and gives the same visual result — crests
 * reading as though they run across the wind.
 */
export const WAVE_ANISO = 0.55;

/**
 * Depth over which waves reach full height, metres.
 *
 * Standing in for fetch. Real waves shoal and break as the bottom comes up, and
 * a puddle in a valley floor has no swell on it whatever the wind does — so
 * tying amplitude to depth suppresses waves on ponds and shallows without
 * needing to know which body of water is which.
 */
export const WAVE_DEEP = 22;

/**
 * Base roughness of the water microsurface, before wave slope is added.
 *
 * This is the roughness of what the wave field cannot resolve at any distance —
 * capillary ripple below the finest rung. Sub-pixel wave slope is added to it
 * in quadrature, which is what turns the sun's reflection from a single blob up
 * close into a long glitter path at range instead of aliasing away.
 */
export const WATER_ROUGH = 0.055;

/**
 * Surface slope at which water starts to break into whitecaps.
 *
 * A threshold rather than a proportion, because whitecapping is a breaking
 * criterion: water is either steep enough to break or it is not, and foam that
 * fades smoothly in with steepness reads as dirt on the lens. Sits above the
 * field's RMS slope so caps are occasional — the state of a moderate sea, not
 * a storm at every latitude.
 */
export const WAVE_CAP = 0.2;

export const PATCH_VERTS = 33;
export const PATCH_SEGS = PATCH_VERTS - 1;

/**
 * Fractional overlap between neighbouring patches — the lateral half of the
 * skirt. Each patch is drawn `1 + PATCH_BLEED` times its own half-size, so
 * adjacent patches overlap instead of meeting exactly.
 *
 * Two patches sharing an edge agree about it *analytically*, and they must:
 * the whole precision architecture exists so a vertex reconstructed from one
 * patch's anchor lands where the neighbour's does. What they cannot do is
 * agree *bitwise*. Each side reconstructs the shared vertex from its own
 * anchor, its own warped centre and its own tangent-addition delta, all in
 * f32, so the two results differ by a few ULPs — microns of ground, and
 * nothing at all against the ~3e-5 of sample spacing tools/precision.ts
 * measures.
 *
 * A rasteriser does not care that the disagreement is microscopic. Its fill
 * rule guarantees seamless coverage only for edges that are *identical*; two
 * edges that differ in the last bits are two edges, and along the stretch
 * where they diverge outward the pixel centres between them belong to neither
 * triangle. That is a one-pixel hole per seam, following the patch boundary,
 * and it is why the seam is visible as a hairline crack near the ground where
 * the mesh is finest and the seams are longest on screen. sim.audit() counted
 * 0.54% of terrain pixels at 49 m. The skirt only ever hid part of it — 8×,
 * 16× and 64× the skirt depth measure 6992, 1567 and 0 crack pixels, which is
 * a curtain being drawn further across the symptom rather than a cause being
 * removed. A skirt is depth; this gap is width.
 *
 * The overlap is a *fraction of the patch*, so it stays the same fraction of
 * a pixel at every level: a patch never spans more than ~10^3 px, which puts
 * the overlap under 10^-2 px everywhere. Coincident surfaces cannot z-fight
 * across a sliver that thin, and both patches evaluate the same height field
 * at the same direction inside it, so the overlap is not merely invisible —
 * it is the same surface drawn twice.
 *
 * 1e-5 is two orders of magnitude above the f32 reconstruction noise (~1e-7
 * relative) and two below a pixel. Measured, at the site the artefact was
 * found: 0.54213% of terrain pixels enclosing background before, 0.00000%
 * after, from 3 m to 8000 km.
 */
export const PATCH_BLEED = 1e-5;

/**
 * Shallowest level the selector may stop at, whatever the distance.
 *
 * Geometric LOD error is not the only sampling constraint: the climate,
 * elevation and normal all reach the fragment stage as varyings and the albedo
 * puts hard thresholds on them. No varying may be stretched further than the
 * data behind it, so the floor is the level whose vertex spacing matches the
 * bake cell.
 *
 * That cell is FACE_EDGE / BAKE_RES = 9.00 km, and level 5 is
 * FACE_EDGE / 2^5 / PATCH_SEGS = 9.00 km. This was 4 — 18.01 km, exactly twice
 * the cell — with a comment asserting it was the cell size, which it had been
 * when BAKE_RES was 512. The consequence was visible only from orbit and only
 * as an absence: the mesh could not carry the bake's own relief, so continents
 * were shaded off an 18 km normal and the finest thing on them was the biome
 * dither. Everything between 9 km and 46 km was simply missing, which is what
 * "splotchy and low resolution" was.
 *
 * Derived rather than written down, so re-baking at a different resolution
 * moves it. See the note in quadtree.ts.
 */
export const MIN_SELECT_LEVEL = Math.round(
  Math.log2(FACE_EDGE / PATCH_SEGS / (FACE_EDGE / BAKE_RES)),
);

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
export const MAX_VEG_TILES = 384;

/**
 * vec4s per vegetation tile record. Five of geometry and precision data, then
 * three carrying the M3 bake sampled at the tile centre — elevation and its
 * gradient, wetness and lake depth, and the distance to the drainage axis with
 * *its* gradient — so the scatter can reconstruct the baked surface linearly
 * instead of fetching the cube map per candidate cell.
 *
 * The distance-to-axis gradient is not optional. It drives a channel cut
 * CHANNEL_DEPTH metres deep, so holding it constant across a tile puts every
 * stem near a river up to 45 m above or below the ground it is standing on.
 */
export const VEG_TILE_VEC4 = 8;

/**
 * Distance beyond which canopy becomes a terrain material, not objects.
 *
 * 6 km is the practical ceiling for *individual* trees, and it is set by
 * arithmetic rather than by taste. At this display's 693 pixels per radian:
 *
 *        8 m tree   20 m tree   29 m tree
 *   1.8 km   3.1 px     7.7 px     11.2 px
 *   6 km     0.9 px     2.3 px      3.4 px
 *   16 km    0.4 px     0.9 px      1.3 px
 *   50 km    0.1 px     0.3 px      0.4 px
 *
 * So at 50 km even the largest tree on the planet is four tenths of a pixel.
 * Drawing one there is not a level of detail, it is aliasing with extra steps —
 * and the projected-size cull would reject it anyway. What actually carries
 * the forest past here is the terrain's own canopy tint, which is already
 * driven by the same cover function the scatter accepts against, so the
 * handover is a change of representation and not of appearance.
 *
 * Reaching 50 km with objects needs *cluster* impostors: one quad per stand of
 * a few hundred trees, sized to the stand rather than the stem. That is a
 * different feature — it needs its own scatter, its own accept rule and its
 * own art — and it is the honest next step if the far view is not enough.
 */
export const VEG_RANGE = 6000;

/**
 * Representation bands. LOD here changes what an instance *is*, not how many
 * triangles it has — decimation is the wrong tool for aggregate geometry
 * (SPEC.md §8, "Do we need Nanite?").
 *
 * Instances are scanned models now (see treeAssets.ts), and that changes the
 * shape of this table in one way that matters: a model band can only draw one
 * species, because a band is one indirect draw and an indirect draw is one
 * geometry. So the two geometry distances are each split in two, conifer and
 * broadleaf, and the scatter bins on distance *and* species. The impostor
 * bands are species-agnostic — the species picks a row of the atlas, not a
 * draw.
 *
 * The geometry distances came down from 130 m, and that is the price of real
 * models: a procedural tree was 512 triangles and the scanned near LOD is
 * 2.2–4.8k. 45 m and 110 m keep the geometry bands at roughly the triangle
 * budget the old 130 m band had. What buys the loss back is that the impostor
 * now carries the model's own silhouette from eight baked yaws rather than a
 * procedural blob, so the handover is far less visible than it was.
 */
export const VEG_MODEL_NEAR = 45;
export const VEG_MODEL_MID = 110;

/**
 * Distance thresholds, before the species split. The first
 * VEG_MODEL_LODS entries are geometry; the rest are impostors.
 */
export const VEG_DIST_BANDS = [VEG_MODEL_NEAR, VEG_MODEL_MID, 1800, VEG_RANGE];
/** How many of those distances draw real geometry. */
export const VEG_MODEL_LODS = 2;
/** Species the models cover: 0 conifer, 1 broadleaf. Also atlas rows. */
export const VEG_SPECIES = 2;
/**
 * Bands drawing real geometry. The rest are impostors — and the split matters
 * outside this file, because only the geometry bands cast shadows.
 */
export const VEG_MODEL_BANDS = VEG_MODEL_LODS * VEG_SPECIES;

export const VEG_BANDS = [
  { name: 'near-conifer', maxDist: VEG_MODEL_NEAR, model: true, lod: 0, species: 0 },
  { name: 'near-broadleaf', maxDist: VEG_MODEL_NEAR, model: true, lod: 0, species: 1 },
  { name: 'mid-conifer', maxDist: VEG_MODEL_MID, model: true, lod: 1, species: 0 },
  { name: 'mid-broadleaf', maxDist: VEG_MODEL_MID, model: true, lod: 1, species: 1 },
  { name: 'far', maxDist: 1800, model: false, lod: -1, species: -1 },
  // The last band exists purely to reach 6 km. Same impostor quad as 'far'; it
  // gets its own draw and its own capacity so the 1.8–6 km annulus — 89% of the
  // scattered area — does not crowd the band that has to hold up at 500 m.
  { name: 'distant', maxDist: VEG_RANGE, model: false, lod: -1, species: -1 },
] as const;

/**
 * Instances per band — sized per band, not shared equally.
 *
 * Equal capacities used to be the point: the shader's base offset was
 * `band * capacity`, plain arithmetic instead of a lookup. That stopped paying
 * when the geometry distances were split by species. Four of the six bands now
 * hold real models within 110 m, and a dense taiga at eye height puts about
 * 170, 160, 940 and 710 instances in them — against a reservation of 400 000
 * slots each. Nine-tenths of a 38 MB buffer was held for bands that never used
 * a percent of it, while the two bands that carry the actual landscape sat
 * pinned at their ceiling.
 *
 * That ceiling was not a cosmetic problem. Past capacity the scatter keeps
 * whichever instances happened to win the atomic, and thread order on a GPU is
 * not defined — so the *set* of distant trees was reshuffling as the camera
 * moved. Trees appeared to wander and blink at range, and no amount of hashing
 * the placement could fix it, because the placement was never what varied.
 *
 * So: a small fixed allowance for the geometry bands, and the whole remaining
 * budget split between the impostor bands, which is where the instances are.
 * Same total memory as before.
 */
const VEG_GEOM_SLOTS = 16_384;
const VEG_TOTAL_SLOTS = QUALITY.vegBandCapacity * VEG_BANDS.length;

/** Slots per band, in band order. */
export const VEG_BAND_SLOTS: readonly number[] = VEG_BANDS.map((b) =>
  b.model
    ? VEG_GEOM_SLOTS
    : Math.floor(
        (VEG_TOTAL_SLOTS - VEG_GEOM_SLOTS * VEG_MODEL_BANDS) /
          (VEG_BANDS.length - VEG_MODEL_BANDS),
      ),
);

/** Where each band's slice of the instance buffer starts. */
export const VEG_BAND_BASE: readonly number[] = VEG_BAND_SLOTS.reduce<number[]>(
  (acc, _slots, i) => {
    acc.push(i === 0 ? 0 : acc[i - 1] + VEG_BAND_SLOTS[i - 1]);
    return acc;
  },
  [],
);

export const VEG_CAPACITY = VEG_BAND_SLOTS.reduce((a, b) => a + b, 0);

/** Growth limits, metres. */
export const VEG_MIN_ELEVATION = 2;
/**
 * Slope at which canopy has thinned to nothing, as 1 − cos(angle).
 *
 * This was 0.55, which is 63°, with the thinning only starting at 0.30 = 46°.
 * So the forest stood at full density on 46° ground and did not give up until
 * the slope was steeper than any soil will hold — trees growing out of cliffs.
 *
 * Real forests thin from about 25° and are gone by 40°, which is roughly where
 * regolith stops staying put; above that you get bare rock, scree and the
 * occasional stem in a crack, none of which a density function should be
 * placing. The pair below is 25° to 40°.
 *
 *   20°  0.060      35°  0.181
 *   25°  0.094      40°  0.234
 *   30°  0.134      45°  0.293
 */
/**
 * Cover below which nothing is planted at all.
 *
 * The cover function tails off smoothly to zero, which is right for a *tint*
 * and wrong for *objects*: at cover 0.02 the scatter still accepts one
 * candidate in fifty, and on the bright sand of a coastal strip each of those
 * is a lone dark stem two pixels across. From orbit that reads as a fringe of
 * black speckles along every shoreline — the marginal-moisture band happens to
 * follow the coast, so the artefact does too.
 *
 * A floor turns the tail into an edge. Below it the terrain's canopy tint
 * carries the thinning on its own, which is what a tint is for.
 */
export const VEG_MIN_COVER = 0.075;

/**
 * Slope gates for what will grow, as `1 - cos(theta)` rather than a gradient.
 *
 * That measure is what the shaders already have in hand — `1 - dot(n, dir)`,
 * one dot product from the normal — but it reads as far gentler than it is,
 * which is how the old numbers survived: 0.234 looks like a fifth of a slope
 * and is actually **40 degrees**, and grass was gated at 0.28-0.55, which is
 * 44 degrees thinning out at 63. Sixty-three degrees is not a hillside, it is
 * a cliff face, and the meadow drawn up it was the most obvious thing wrong
 * with any steep ground in the build.
 *
 * Stated in degrees, so the next person to touch them knows what they are
 * choosing:
 *
 *   trees   full canopy to 22 deg, gone by 33
 *   grass   full cover  to 20 deg, gone by 32
 *
 * Both are the shared closure — see canopyClosure in shaders/terrain.ts — so
 * the ground tint and where plants actually stand move together. Changing one
 * without the other is how you get white slopes with trees standing on them.
 */
export const VEG_SLOPE_FULL = 0.073;
export const VEG_MAX_SLOPE = 0.161;

/**
 * The same, for ground cover. Slightly tighter than the canopy: a stand of
 * trees on a steep slope is a real thing that holds itself up, and a lawn on
 * one is not.
 */
export const GRASS_SLOPE_FULL = 0.060;
export const GRASS_MAX_SLOPE = 0.152;

/* ── Climate (SPEC.md §8) ────────────────────────────────────────────────
 *
 * Two fields, temperature and moisture, which are the axes of a Whittaker
 * diagram — between them they decide desert from steppe from forest from
 * tundra from ice. Everything the surface looks like at continental scale
 * comes from here.
 *
 * They exist because the albedo had no large-scale structure at all: it was a
 * function of elevation, slope and a 7 km canopy-clump noise, so the coarsest
 * thing in it was 7 km. Above about 100 km altitude every clump octave is
 * band-limited away, `forestClump` collapsed to a constant, and the whole
 * planet went one flat olive tone — which is most of why continents read as
 * undetailed from orbit. Earth from orbit is almost entirely biome contrast.
 */

/**
 * Latitude falloff for insolation, as `cos(latitude)^(2·LAT_EXP)`.
 *
 * Fitted to Earth's mean annual temperature, normalised so the equator is 1
 * and the pole is 0. `lat` is |sin latitude| — the form a unit direction gives
 * for free — so cos² is `1 − lat²` and the whole profile is one pow:
 *
 *   latitude   0°    20°   30°   40°   50°   60°   70°   80°   90°
 *   Earth     1.000 0.942 0.846 0.731 0.596 0.442 0.250 0.096 0.000
 *   this      1.000 0.928 0.841 0.726 0.588 0.435 0.276 0.122 0.000
 *
 * This replaces `1 − 1.047·lat^4.6`, which was flat where Earth is not: it
 * still read 0.96 at 30° and 0.79 at 45°, so anything gated on "tropical"
 * reached to 55° and the planet had twice Earth's savanna and a third of its
 * steppe. A power of |sin| is the wrong family of curve for this — it is
 * nearly constant across the whole mid-latitude band and then falls off a
 * cliff, which is the opposite of how insolation behaves.
 */
export const LAT_EXP = 0.6;

/**
 * Lapse rate in temperature units per metre. 6.5 K/km over the ~55 K span the
 * 0..1 temperature covers. This is what makes the treeline emerge instead of
 * being a hard-coded elevation: it lowers with latitude on its own, which the
 * old fixed VEG_TREELINE of 2600 m could not do.
 */
export const LAPSE = 0.000165;

/**
 * Sky radiance on the night side, as a fraction of the daytime sky term.
 *
 * Not physics, and it should not pretend to be. A moonless night sky is about
 * 10⁻⁷ of a sunlit surface; rendered honestly at that ratio the night side is
 * a black screen at any exposure, which is what happened the moment the sky
 * ambient was correctly gated to the sun. Nor is it moonlight — full moonlight
 * is ~10⁻⁶ of sunlight and would be just as invisible.
 *
 * What it stands in for is the dark adaptation the eye does and the renderer
 * cannot: scotopic vision compresses six orders of magnitude into something a
 * person describes as "dim", so a moonlit field looks perhaps a hundredth as
 * bright as noon rather than a millionth. This is that compression, applied as
 * a floor because the tone curve has no other way to express it.
 *
 * Measured, at 0.020: the ground under a sun 12° below the horizon reads 8-bit
 * 30/44/35 at the ground-level night exposure of 4.0 — legible, unmistakably
 * night, and enough to walk by. From orbit, where the exposure ceiling is 0.6,
 * the night hemisphere reads black over the whole disc with only the limb
 * showing. That second number is the one that matters: this floor is what the
 * dark side used to blow out from, and it now costs nothing there.
 */
export const NIGHT_SKY = 0.020;
export const MOON_LIGHT = 0.012;

/**
 * Rigid rotation of the cloud deck, radians per second.
 *
 * The whole deck turns at this rate. A rigid rotation is an isometry — it can
 * run forever without distorting anything — so this is the part that is allowed
 * to accumulate.
 */
/* ── Canopy colour ───────────────────────────────────────────────────────
 *
 * The canopy used to *replace* the ground: one colour, blended in at 0.92, so
 * anywhere forested went to a near-uniform dark green and everything the
 * biome, the meso ladder and the landform had built underneath it was thrown
 * away. From orbit that is most of what made continents read as flat — real
 * forest is not one colour, and you can still see the land through it.
 */

/** Share of the underlying ground colour a closed canopy keeps. */
export const CANOPY_KEEP = 0.38;
/**
 * Clearing rungs, metres. Roughly 2, 6, 18 and 50 km.
 *
 * Forest is patchy at every scale — burns, blowdown, bog, farmland, altitude —
 * and a cover field that only varies with climate has none of that. These
 * modulate the *density* rather than the colour, so a clearing shows whatever
 * is underneath it rather than a lighter shade of tree.
 */
export const CANOPY_CLEAR_LAM = [50_000, 18_000, 6_000, 2_000] as const;
/** How deeply the clearing field can cut cover. */
export const CANOPY_CLEAR_K = 0.55;

export const CLOUD_SPIN = 0.002;

/**
 * Differential zonal wind on top of that, radians per second, and the total
 * slip it is allowed to accumulate.
 *
 * Bands have to move at different speeds or the deck reads as a painted ball.
 * But a *differential* rotation is a shear, and shear accumulates: at 0.002
 * rad/s the trades and the westerlies slip 40 radians apart in 10000 s, which
 * is six and a half relative turns of the planet. Anything spanning the
 * reversal is stretched by that factor, and the deck comes out as long zonal
 * filaments — swirled paint, which is the failure the flow-warp strength was
 * already tuned to avoid, arriving by a different route.
 *
 * Real bands do slip forever and real cloud does not smear, because cloud
 * *regenerates*: a system lives days, not the age of the planet, so it only
 * ever experiences a few days' worth of shear. Cross-fading two ages of the
 * field would model that honestly and would double the cost of the eight
 * billow octaves and everything that samples them, including the ground
 * shadow.
 *
 * Capping the accumulated slip is the cheap stand-in and it buys the same
 * thing: bands still slide apart at the right rate to begin with, then hold a
 * fixed offset instead of winding up. 0.35 rad over the ~0.23 rad width of the
 * reversal is a shear strain of 1.5, which draws a cell out to about 1.8:1 —
 * a front, which is what that zone should look like, rather than a filament.
 */
export const CLOUD_ZONAL = 0.002;
/**
 * Half the peak-to-peak differential, because `band` runs -1 to +1: the
 * equator lags by this and the mid latitudes lead by it. The note above sizes
 * the strain from 0.35 rad across the reversal, which is the *full* swing, so
 * this is half of it. At 0.35 the swing was 0.7 and the strain 3.0 — cells
 * drawn out past 3:1, which is filament, not front.
 *
 * It never showed as filament before, because until the tanh above was fixed
 * the profile was a step: the shear was zero either side and infinite at the
 * joint, so the strain this number controls was never actually applied.
 */
export const CLOUD_SHEAR_MAX = 0.175;

/**
 * Draw the drainage network as a river at all.
 *
 * Off. The valleys stay — those are the LEM's, they are baked into the
 * elevation, and they are the part of the drainage that reads well at every
 * altitude. What this switches off is the *river*: the narrow trench the
 * runtime cuts along the drainage axis (channel_) and the green gallery
 * corridor drawn along it (see the river corridor block in shadeTerrain).
 *
 * Both are reconstructions of a feature the data cannot hold. LESSONS §3 is
 * the long version: a continental trunk river is 1-3 km wide against a 9 km
 * bake cell, so its position is known to about a cell and its width to nothing
 * at all. The curve fitting in bake/channels.ts fixed the *shape* — the network
 * is smooth now, and measurably so — but it cannot conjure resolution that was
 * never there, and up close the result still reads as a painted line rather
 * than as water.
 *
 * Kept as a switch rather than deleted because nothing about the machinery is
 * wrong and the bake still produces the distance field either way. Set it back
 * to 1 to see it. Removing a feature is a legitimate outcome (LESSONS §15) and
 * this is the second time it has been the right call for rivers.
 */
export const DRAW_RIVERS = 0;

/**
 * Cloud deck altitude, metres.
 *
 * One shell, not a volume. Real cloud is 3D and raymarching it is the single
 * most expensive thing a planet renderer can do; a shell at the altitude where
 * most cloud actually lives gets the silhouette, the terminator and the
 * shadowing for one alpha-blended sphere. The cost is that you cannot fly
 * *through* it convincingly, which is a fair trade at this stage.
 */
export const CLOUD_ALT = 6500;

/**
 * Synoptic scale: the frequency of the low-frequency field that decides where
 * weather systems *are*, and how hard it gates the cumulus below it.
 *
 * This is the knob that sets how many storms the planet has. A gradient-noise
 * field of frequency f on the unit sphere has on the order of 4*pi*f^2
 * features, so the count goes as the square and small changes here are not
 * small changes on screen: f = 2.1 is roughly fifty-five systems, which reads
 * as an even sprinkle of small cells at every longitude — weather-textured,
 * but not weather. Earth at any given moment has a handful of large organised
 * systems with genuinely empty ocean between them.
 *
 * 1.3 is measured, not guessed — `npm run storms` counts the storm centres on
 * the actual field, and it is the value that lands on five. The old 2.1 counts
 * twenty, eight of them strong. Do not reach for this by intuition: the first
 * attempt here was 0.62, on the arithmetic that 4*pi*f^2 = 5, and it gives one
 * supersystem covering a third of the planet rather than five storms. The
 * arithmetic counts features of the noise; what reads as a storm is a *peak*,
 * and peaks are denser than features because the field has structure between
 * its lattice cells. Measure it.
 *
 * Five systems at f = 1.3 are about 1/f radians across — some 4,900 km, which
 * is larger than an Earth synoptic system and is the point: fewer and bigger
 * was the requirement.
 *
 * GAIN is what makes fewer systems read as *larger* rather than merely
 * sparser. The synoptic term is an offset into the cumulus threshold, so
 * raising it widens the swing between "inside a system" and "outside one":
 * the cloud piles into the systems and the space between them opens to clear
 * air. At the old 0.085 the field only tinted the coverage and a lower
 * frequency would have given five faint patches on a uniform deck.
 */
export const CLOUD_SYN_FREQ = 1.3;
export const CLOUD_SYN_FREQ2 = 3.25;
export const CLOUD_SYN_GAIN = 0.17;
export const CLOUD_SYN_GAIN_CI = 0.15;

/**
 * Vertical extent of the cumulus layer, metres — the base sits at CLOUD_ALT
 * and the tops reach this far above it.
 *
 * The deck is one alpha shell, so it has no thickness of its own and reads as
 * a texture painted on glass. A cumulus is not flat: it is roughly as tall as
 * it is wide, its top is displaced from its base by parallax as soon as you
 * are not looking straight down it, and that displacement is most of what
 * tells the eye it is an object rather than a stain.
 *
 * 2200 m is a fair-weather cumulus — base near the condensation level, top a
 * couple of kilometres above. Deep convection goes far higher, but a towering
 * cumulonimbus needs an anvil and a real march to look like anything.
 */
export const CLOUD_THICK = 2200;

/**
 * Samples through the cloud slab, and the octave discount each one takes.
 *
 * The deck was one alpha shell with a parallax trick standing in for depth.
 * That gets the silhouette leaning the right way but it cannot give a cumulus
 * an *interior*: opacity came from squaring the coverage, so a cloud was as
 * opaque at its edge as a real one is in its middle, and the base shadow was a
 * proxy on thickness rather than light actually failing to reach the bottom.
 *
 * Marching the slab fixes both, and it is affordable for one reason: the
 * coverage field is a function of direction, and over a 2.2 km slab seen from
 * 6.5 km the direction barely changes except at the limb. So the march does
 * not need the full octave stack at every step — it passes a slightly inflated
 * pixel footprint, which is the same band limit the field already honours.
 *
 * It ended up at 1.0 — no inflation at all — and the reason is worth keeping.
 * The march lights the cloud that the *shell* draws, so the two have to be
 * looking at the same field. Give the march a coarser footprint and it keeps
 * fewer octaves, so it sees the deck as the smooth round lumps the coarse
 * octaves make on their own. The shading then says "nothing above this point"
 * across a whole lump while the shell says "cloud here", and the disagreement
 * is drawn: hard-edged bright discs stacked inside every cloud mass, which is
 * not a subtle artefact and is not what any amount of tuning elsewhere would
 * have fixed.
 *
 * So the inflation is kept as a knob and set to nothing. Any saving it buys
 * comes straight out of the agreement between the light and the cloud.
 *
 * Ten steps is where the banding stops being visible against the dither the
 * field already has. Past about sixteen there is nothing left to gain.
 */
export const CLOUD_MARCH = 6;

/* ── Aurora ──────────────────────────────────────────────────────────────
 *
 * Cheated, and the cheat is that it is drawn on a shell rather than as a
 * volume. What it is not is a screen overlay: an aurora seen from orbit sits
 * *above the limb* and wraps the pole as an oval, and a screen-space glow
 * cannot do either. One sphere intersection puts it in the right place for
 * nothing, and the same intersection is what makes it disappear correctly
 * behind the planet when you are on the wrong side of it.
 *
 * The oval is a band of geomagnetic latitude, not a cap. Emission peaks around
 * 67 degrees on Earth and falls away both poleward and equatorward, which is
 * why the pole itself is usually dark — the single most recognisable thing
 * about an auroral oval from space and free to reproduce here.
 */

/** Emission altitude, metres. The green line sits near 110 km. */
export const AURORA_ALT = 115_000;
/** Centre of the oval as sin(latitude), and its half-width in the same. */
export const AURORA_BAND = 0.92;
export const AURORA_WIDTH = 0.075;
/** Overall strength. */
export const AURORA_GAIN = 0.22;
/**
 * Floor on the grazing term, which caps the limb brightening.
 *
 * Emission scales with path length through the layer, so a ray running along
 * the sheet picks up far more than one crossing it — that is why an aurora is
 * a faint glow overhead and a bright wall seen edge-on, and it is worth
 * keeping. But the reciprocal runs away: at 0.09 the limb was eleven times the
 * face-on value, which is a physical relationship pushed well past where the
 * thin-layer approximation behind it still holds, and it is what made the band
 * on the limb read as a solid stripe rather than as light.
 */
export const AURORA_GRAZE_MIN = 0.15;
/**
 * Sun elevation over which the aurora fades out.
 *
 * It is always there; it is only ever *visible* against a dark sky, and it has
 * to be gone before the sky brightens or it reads as a coloured smear on a
 * blue day. Measured at the emitting point, not at the camera, so the terminator
 * cuts it the way it actually cuts it.
 */
export const AURORA_NIGHT = -0.09;
export const CLOUD_MARCH_PX = 1.0;
/** Extinction per unit coverage per metre of path. */
export const CLOUD_SIGMA = 0.0;

/* ── Lightning, seen from orbit ──────────────────────────────────────────
 *
 * Separate from the bolts in weather.ts, and it has to be. Those are drawn as
 * line geometry a few kilometres from the camera with a screen flash behind
 * them, which is right at ground level and invisible from anywhere else — a
 * bolt is a couple of hundred metres across and there is nothing to draw by
 * the time you are in orbit.
 *
 * What you *do* see from orbit is real and famous: storm complexes flickering
 * inside the cloud, each flash lighting a patch of deck tens of kilometres
 * across for a fraction of a second. That is not a bolt, it is the cloud
 * acting as a diffuser, so it is drawn the way it actually appears — as light
 * added inside the deck, gated on there being deck to light.
 */

/** Storm-cell size, metres. A mesoscale convective complex, not one cell. */
export const BOLT_CELL = 55_000;
/**
 * Mean seconds between flashes *in one active cell*, and the visible decay.
 *
 * This is the knob that decides how rare lightning looks, and the first
 * attempt had it three orders of magnitude too fast. 3.4 seconds per cell
 * sounds sparse until you count the cells: at orbital range the visible disc
 * holds on the order of ten thousand of them, a fair fraction of which pass
 * the convection gate, so a few seconds each adds up to a continuous
 * scintillation over the whole night side rather than a storm here and there.
 *
 * The rate you actually see is (active cells in view) / period, so it has to
 * be set against the cell count and not against intuition about one storm.
 * Roughly a thousand cells clear the coverage gate at orbit; BOLT_ACTIVE keeps
 * a twenty-fifth of those, and half an hour apiece puts the visible rate near
 * two a minute. It is altitude-dependent by construction — fewer cells are in
 * frame lower down — which is also true of the real thing.
 */
export const BOLT_PERIOD = 10;
/** Slightly longer than a real return stroke, so a rare flash is catchable. */
export const BOLT_DECAY = 0.18;
/**
 * Fraction of cells that ever fire at all.
 *
 * Storms are not uniformly distributed through the convective band — most deep
 * cloud is not electrically active at any given moment. Gating on a hash keeps
 * the same cells lit for the whole session rather than making every cell
 * flicker occasionally, which is both closer to the truth and much rarer.
 */
export const BOLT_ACTIVE = 0.05;
/** Radius of the lit patch, as a fraction of the cell. */
export const BOLT_GLOW = 0.17;
/** Emissive strength. */
export const BOLT_GAIN = 2.6;
/**
 * Coverage window a cell needs before it is allowed to flash.
 *
 * Lightning is a deep-convection phenomenon, so it belongs to the thick cores
 * and not to the thin margins. Keying it to the same coverage the deck is
 * drawn from means the storms light up and the fair-weather cumulus does not,
 * with no separate storm field to keep in step.
 */
export const BOLT_COVER_LO = 0.45;
export const BOLT_COVER_HI = 0.78;

/** Coarsest moisture octave. RADIUS/3.1 ≈ 2000 km — provinces, not weather. */
export const BIOME_F0 = 3.1;
export const BIOME_LACUNARITY = 2.6;
export const BIOME_GAIN = 0.6;

/**
 * Value `forestClump` takes when every one of its octaves has been
 * band-limited away.
 *
 * The clump field is divided by this so that a fully faded clump multiplies
 * the climate-driven canopy by exactly 1 — the detail vanishes and leaves the
 * biome, instead of replacing it with a constant. Previously nothing divided
 * it out, so at distance the ground everywhere took `smoothstep(-0.42, 0.26,
 * 0)` = 0.673 of full canopy tint regardless of climate, terrain or latitude.
 */
export const CLUMP_MEDIAN = 0.6731;

/**
 * How hard the clump modulates the climate's closure.
 *
 * At 1.0 the clump swings closure over its full 0..1.49 range, so glades open
 * onto bare ground and stands close completely — and since canopy reflects
 * about a quarter of what dry ground does, that turns a 3 km noise field into
 * a high-contrast mottle covering entire continents. It was the single most
 * visible thing from 20–100 km, and reads as camouflage rather than forest.
 * Compressing it toward 1 keeps the glades without letting a mid-frequency
 * noise field dominate the planet's appearance.
 *
 * The fallback stays exact at any strength: when the clump fades to
 * CLUMP_MEDIAN the quotient is 1, and mixing toward 1 leaves it at 1.
 */
export const CLUMP_STRENGTH = 0.62;

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
export const VEG_MIN_PIXELS = 6.0;

/* ── Ground clutter (SPEC.md §8) ─────────────────────────────────────────
 *
 * Grass blades, on the pattern momentchan/false-earth uses: real Bezier
 * geometry generated by a compute pass into a packed instance buffer, drawn
 * indirectly, with a PCG hash for jitter and a *world-snapped* grid index for
 * the seed so a blade keeps its identity as the camera moves. The two
 * departures from that reference are that the surface underneath is this
 * project's own amplified bake rather than a heightfield, and that the grid is
 * camera-centred on a sphere rather than on a plane.
 *
 * Blades are cheap individually and ruinous collectively, so the range is
 * short and everything past it is the terrain's own albedo — the same division
 * the tree bands already use, one step further in.
 */

/** Radius of the grass field, metres. Past this the ground colour carries it. */
/**
 * Radius of the grass field, metres.
 *
 * Density was the thing to protect: 65 536 blades over a 34 m disc is 18 per
 * m², which is not a sward but scattered tufts on bare ground, and it read as
 * sprinkles. The answer then was to shrink the disc.
 *
 * The answer now is to grow the grid instead, because the two rejection tests
 * in grassSample changed what a long range costs. Inside 45% of the range the
 * field is still at full density; past that the hashed radial dissolve thins
 * it to a third and takes it to nothing, so the added area is cheap and the
 * boundary is never a visible circle. Doubling the grid buys twice the reach
 * at the same density where density is actually looked at.
 */
export const GRASS_RANGE = 105;

/**
 * Radius held at full density, metres. Everything from here to GRASS_RANGE is
 * the hashed dissolve thinning toward nothing.
 *
 * An absolute distance, deliberately, where it used to be `GRASS_RANGE * 0.45`
 * inside the shader. Tying it to the range meant the dense disc grew with the
 * square of any reach you asked for: taking the range from 42 m to 85 m
 * quadrupled the full-density area and put the frame over 700 ms. The distance
 * at which a viewer stops resolving one blade from the next is a property of
 * blade size and screen resolution, not of how far the field happens to
 * extend, so it belongs here as a number in metres.
 */
export const GRASS_FULL = 24;

/**
 * Cells per side of the camera-centred blade grid, and their spacing.
 *
 * The grid is what actually bounds the range: it is camera-centred and finite,
 * so no blade can exist outside it however large GRASS_RANGE is set. 1024² at
 * 0.17 m covers a 174 m square, half-extent 87 m, which clears GRASS_RANGE on
 * the axes where the square is thinnest.
 *
 * A megabyte of candidates is a big dispatch, and it is still the cheaper half
 * of the pass: rejection happens in the first few instructions for the large
 * majority of them, while everything that survives costs a full height-field
 * evaluation and five segments of geometry.
 */
export const GRASS_GRID = 512;
export const GRASS_SPACING = 0.42;

/** Segments per blade. Four gives a curve; the tip is a triangle. */
export const GRASS_SEGMENTS = 3;

/** Blade height range, metres. */
export const GRASS_H_LO = 0.22;
export const GRASS_H_HI = 0.78;
/**
 * Blade width at the base, as a fraction of height.
 *
 * Real grass is 3–8 mm across, and at that width a blade is under a pixel
 * beyond about four metres — the first version was physically right and
 * invisible, drawing 46 000 blades that resolved to nothing. Stylised wider,
 * as false-earth and every other real-time field does, so a blade holds a
 * couple of pixels out to the edge of the range.
 */
export const GRASS_WIDTH = 0.19;

/** Upper bound on live blades, for the instance buffer. */
/**
 * Slots for accepted blades.
 *
 * Half the candidate count, not all of it. With the view cone and the radial
 * dissolve in place (see grassSample) the accepted fraction is well under a
 * third even looking along the long axis of the field, and the buffer is a
 * vec4 per slot — 512² would be 16 MB of mostly-never-written storage.
 * Overflow is dropped silently by the atomic, which is the right failure: a
 * missing blade at the edge of a 42 m field is invisible.
 */
export const GRASS_CAPACITY = (GRASS_GRID * GRASS_GRID) / 2;

/* ── Coastline warp ──────────────────────────────────────────────────────
 *
 * The shoreline is the zero crossing of a field stored at 9 km per texel and
 * reconstructed bilinearly. A bilinear patch's zero contour is a hyperbola per
 * texel, so the coast is a chain of texel-sized arcs meeting at texel corners —
 * blocky at exactly the scale of the data, and no amount of amplification
 * fixes it because SHORE_FLAT_FLOOR deliberately damps amplification near sea
 * level to keep islands from materialising as you approach.
 *
 * Warping the *lookup* rather than the value fixes it without giving that up.
 * The coast becomes the zero set of bakeH(warp(dir)), which is as crinkled as
 * the warp is, at every scale the warp carries — and because the warp is a
 * fixed function of direction with no band limit in it, the coastline does not
 * depend on the camera and cannot move on approach. Elevation statistics are
 * untouched: it is the same field, resampled.
 *
 * Amplitude is a bit under one texel. More would start moving headlands
 * around rather than roughening them; less is invisible.
 */
export const COAST_WARP_AMP = 0.62 * (FACE_EDGE / BAKE_RES);

/**
 * Octaves in the coastline warp.
 *
 * Three reached 5.5 km, which is still coarser than the 9 km grid it is
 * meant to disguise, so from 1000 km the coast read as a chain of rounded
 * polygons — the zero crossing of a bilinear field on a square grid, which is
 * exactly what it was. Five reach 1.2 km. Amplitude halves per octave, so the
 * fine ones cost little and buy the fractal edge a real coastline has.
 *
 * The amplitude went with it, 0.30 → 0.62 of a texel: a displacement smaller
 * than half a cell cannot break up a cell-sized feature however many octaves
 * it has. It stays gated to the shoreline by COAST_WARP_FADE.
 */
export const COAST_WARP_OCTAVES = 5;
/**
 * Elevation over which the warp fades out, metres.
 *
 * The warp must be *local to the coast*. Applied globally it displaces the
 * bake lookup by kilometres everywhere, which does not roughen mountains, it
 * moves them — and low-altitude views went black because the camera ended up
 * inside terrain that had slid sideways under it. It is also the one place the
 * warp buys anything: inland, the amplification already supplies detail at
 * every scale, and only the shoreline is a hard threshold on the raw bake.
 */
export const COAST_WARP_FADE = 450;
/**
 * Share of the warp that survives inland, where the coast fade has run out.
 *
 * The warp was gated to the shoreline on the grounds that inland terrain
 * should sit where the bake put it. That is right about elevation and wrong
 * about everything the *grid* leaks: the bake solves flow on a cube lattice
 * with eight neighbours, so its channels can only leave a cell along eight
 * directions. Inland, with no warp, that is exactly what gets drawn — straight
 * runs meeting at right angles and a comb of parallel tributaries, which from
 * 1000 km, where a 9 km cell is nine pixels, reads as a low-resolution
 * drawing rather than as a river system.
 *
 * The same displacement that turns a chain of bilinear arcs into a coastline
 * turns a D8 staircase into a meander, and it is the same three kilometres of
 * lookup offset either way. It cannot slide under the camera — it is a fixed
 * function of direction — and elevation, wetness and channel distance are all
 * fetched through the one warped direction, so they stay mutually consistent:
 * a river moves, and its valley and its floodplain move with it.
 *
 * Not the full amount, because inland the warp is buying disguise rather than
 * shape, and the bake's elevation is worth staying close to.
 */
export const COAST_WARP_FLOOR = 0.55;
/** Coarsest warp octave, cycles per radian. Three octaves down from here. */
export const COAST_WARP_F0 = (RADIUS / (FACE_EDGE / BAKE_RES)) * 0.35;
