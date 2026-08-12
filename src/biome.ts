/**
 * The biome classification, and the one place its numbers live.
 *
 * Earth's land is not evenly divided and the division is not arbitrary — it is
 * the output of a few physical fields. This table is the target the classifier
 * is fitted to (`npm run biomes` measures it):
 *
 *   1  Deserts & xeric shrublands        ~19.5%   Sahara, Arabia, Gobi, Outback
 *   2  Taiga / boreal forest             ~11.5%   Siberia, Canada, Scandinavia
 *   3  Tropical & subtropical savanna    ~10.5%   Sahel, Cerrado, N. Australia
 *   4  Temperate grassland & steppe       ~9.5%   Eurasian steppe, Prairies
 *   5  Tropical rainforest                ~8.0%   Amazon, Congo, SE Asia
 *   6  Tundra                             ~7.5%   Arctic coast, N. Canada
 *   7  Temperate broadleaf forest         ~6.5%   E. North America, W. Europe
 *   8  Tropical dry & seasonal forest     ~3.0%   Central India, Indochina
 *   9  Mediterranean scrubland            ~2.0%   Med. basin, California, Chile
 *  10  Montane grassland & shrubland      ~2.0%   Tibet, Andes puna, E. Africa
 *
 * That is 80%. The rest of Earth's land is permanent ice (~10%, almost all of
 * it Antarctica and Greenland), bare rock, wetlands and open water, so those
 * are classes here too and carry the residual.
 *
 * ── why there is a third climate axis ─────────────────────────────────────
 *
 * Two axes cannot produce this table. Rainforest, tropical dry forest and
 * savanna can all sit at the same annual temperature and similar annual
 * rainfall; what separates them is whether that rain arrives all year or in one
 * season. Mediterranean scrub is the extreme case — its annual total is
 * unremarkable and its *distribution* is the entire biome. So the classifier
 * takes `season`: the strength of the dry season, 0 for even rainfall and 1 for
 * a pronounced one.
 *
 * It comes out of the same circulation that already sets `moist`. The ITCZ
 * migrates with the sun, so the belt it crosses twice a year is wet twice and
 * dry in between — that is the savanna band at 8–20°, and it is why savanna
 * lies between rainforest and desert rather than anywhere else. The subtropical
 * high does the same thing at 30–40° on its poleward edge, giving the wet
 * winter and dry summer that makes chaparral.
 *
 * ── soft membership, not a switch ─────────────────────────────────────────
 *
 * Each biome is a product of smooth windows over (temp, moist, season,
 * elevation), and the drawn colour is the weighted mean. A hard `if` on a
 * smooth field is a step function sampled once per pixel, which is the mistake
 * the shoreline made (LESSONS §13) — and biome boundaries on Earth are
 * genuinely hundreds of kilometres of mixture, so the blend is also the more
 * truthful model. `classify` takes the argmax of the same weights, and exists
 * so the area fractions above can be measured.
 *
 * Both the WGSL and the CPU classifier are generated from `BIOMES` below, so
 * there is one copy of every threshold. See LESSONS §12 for why that matters.
 */

import {
  BIOME_F0, BIOME_GAIN, BIOME_LACUNARITY, HILLSLOPE_WAVELENGTH, LAPSE, LAT_EXP,
  RELIEF_GAIN, spectrumOctave,
} from './planet.js';
import { shaderNoise } from './shaderNoiseCPU.js';

/**
 * Frequency of the field that breaks the seasonality belts out of perfect
 * zonal stripes. One octave at ~2700 km, which is continental scale — the
 * difference between a monsoon coast and the dry interior at the same latitude.
 */
export const SEASON_F0 = 2.35;

/** Half-width of the transition on each axis, in that axis's own units. */
const SOFT = { t: 0.055, m: 0.06, s: 0.08, h: 450, b: 0.07 };

export interface Biome {
  key: string;
  name: string;
  /** Target share of land area, per the table above. 0 for residual classes. */
  target: number;
  /** Reflectance, not a screen colour — everything is lit before display. */
  colour: readonly [number, number, number];
  /** Windows on temperature, moisture, seasonality, elevation. */
  t?: readonly [number, number];
  m?: readonly [number, number];
  s?: readonly [number, number];
  h?: readonly [number, number];
  /**
   * Sea-level temperature, i.e. latitude before the lapse rate. Montane
   * grassland is the biome that needs it: it is defined by being high *and*
   * not polar, and the lapsed temperature cannot tell a 4 km subtropical
   * plateau from Arctic coast — both land near zero. Tibet, the Andean puna
   * and the East African highlands all sit above 0.8 here.
   */
  b?: readonly [number, number];
  /** Relative weight, for breaking ties where two windows genuinely overlap. */
  bias?: number;
  /**
   * Strength of the metre-scale mottle, 0..1. Not decoration: it is how much
   * of the surface is loose material with its own colour. Desert pavement is
   * gravel over sand and reads at 1:1 contrast from a metre away; a rainforest
   * floor is uniform litter under closed canopy and has almost none.
   */
  grain: number;
  /** Tallest stem this biome supports, metres. 0 where nothing grows. */
  treeMax: number;
  /** Fraction of stems that are needle-leaved rather than broad. */
  conifer: number;
  /**
   * fBm gain for the amplification's octave ladder — the spectral slope of the
   * landscape, and the single most "procedural" thing left about the surface
   * before this existed. One gain everywhere means a dune field, a badland and
   * an alpine ridge are the same shape at every scale, differing only in how
   * far the whole ladder is scaled up.
   *
   * With lacunarity 2.07 a gain g gives amplitude proportional to lambda^H for
   * H = ln(1/g)/ln(2.07), so lower is smoother: 0.50 is H = 0.95 and nearly
   * scale-free in slope, 0.72 is H = 0.45 and rougher than Brownian motion.
   * RELIEF_GAIN, 0.62, is H = 0.66 and is what temperate fluvial country gets.
   *
   * These are *not* an amplitude control. The ladder is normalised so its
   * variance is the same whatever the gain (see the l2 normaliser in height_),
   * and how much relief a place has stays the business of ampAt_ and the
   * bake's own slope. Without that the numbers below would quietly move total
   * relief by −13% to +15%, and in the wrong direction — the smoothest spectra
   * would deliver the most.
   */
  gain: number;
  /**
   * Wavelength at which relief stops being fractal here, metres.
   *
   * Below it slope is capped and relief falls as lambda^1; above it the ladder
   * is fractal (see HILLSLOPE_WAVELENGTH). Physically it is the scale at which
   * mass wasting outruns fluvial incision, and what sets it is how much the
   * ground resists being rilled — which is mostly vegetation and soil depth,
   * i.e. exactly what a biome is.
   *
   * Measured by `npm run slopes`. Slope contribution of the finest octave
   * (36 m), per unit of delivered relief, x100: ice 3.9, desert 4.4,
   * broadleaf 9.8, steppe 20.6, chaparral 26.5 — while at 12.3 km the desert
   * has *more* than the chaparral does (3.50 against 2.93). That inversion is
   * the whole point: it is a different shape, not a different size.
   *
   * On the planet rather than in the ladder, the same tool measures median
   * slope growing from a 300 m step to a 3 m step by 1.6x in ice and 3.1x in
   * chaparral. With one spectrum for the whole planet that column reads 1.9x
   * to 2.4x and the spread is sampling noise.
   */
  hillslope: number;
}

/**
 * Windows are open-ended where the axis does not bound the biome: -9 and 9
 * are outside every normalised axis, and elevation uses metres.
 *
 * The colours are held close to the old palette where a class already existed,
 * because those were fitted against satellite imagery and the point of this
 * change is the *distribution*, not a repaint.
 *
 * ── the shape of the ground, not just its colour ───────────────────────────
 *
 * `gain` and `hillslope` set the spectrum of the sub-grid relief. The ordering
 * is a single physical statement — how well the ground resists being rilled —
 * and it runs from bare mobile sand, where nothing at all survives below the
 * bedform scale, to bare seasonally-drenched silt, where dissection runs to
 * the finest octave there is:
 *
 *   smooth, low frequency   ice, desert, tundra, savanna    glacial scour,
 *                                                           aeolian mantling,
 *                                                           solifluction,
 *                                                           deep etchplain
 *   the reference           rainforest, taiga, broadleaf     soil-mantled
 *                                                            fluvial hillslopes
 *   dissected               dryforest, montane, steppe,       thin cover, frost
 *                           chaparral                         shatter, gullying
 *
 * The badlands are the interesting end and they are *not* in the desert row.
 * Badlands need enough rain to run off and too little vegetation to hold the
 * regolith, which is the semi-arid middle — the Dakota badlands are short-grass
 * prairie and the Spanish ones are scrub, so they belong to steppe and
 * chaparral. Hyper-arid country is the opposite: sand mantles everything and a
 * sand sea is smooth at every scale below the dune.
 *
 * Where this lands on the ground is the product of the spectrum and ampAt_'s
 * amplitude, and the two come from independent fields. Flat steppe gets fine
 * dissection at 150 m of amplitude, which is a plain with drainage texture on
 * it; steppe on a slope gets the same spectrum at 4 km of amplitude, which is
 * a badland. Neither had to be asked for.
 */
export const BIOMES: readonly Biome[] = [
  // ── the ten ───────────────────────────────────────────────────────────
  //
  // Windows are fitted, not chosen. Measured over this planet's land:
  //
  //   axis     p05    p25    p50    p75    p95
  //   temp    0.000  0.202  0.491  0.760  0.933
  //   moist   0.029  0.305  0.483  0.642  0.849
  //   season  0.000  0.021  0.176  0.600  0.876
  //   elev      67    287    798   1368   3180 m
  //
  // The temperature bands partition at ice 0.045 / tundra 0.17 / boreal 0.38 /
  // temperate 0.60 / tropical, and the moisture bands at desert 0.24 /
  // grassland 0.46 / forest. Everything else is a refinement inside one of
  // those cells.
  {
    key: 'desert', name: 'Desert & xeric shrubland', target: 19.5,
    colour: [0.40, 0.33, 0.22],
    t: [0.17, 9], m: [-9, 0.31],
    grain: 1.0, treeMax: 4, conifer: 0.1,
    gain: 0.52, hillslope: 1800,    // sand and gravel mantle everything below the bedform scale
  },
  {
    key: 'taiga', name: 'Taiga / boreal forest', target: 11.5,
    colour: [0.055, 0.070, 0.042],
    t: [0.18, 0.35], m: [0.34, 9],
    grain: 0.42, treeMax: 21, conifer: 0.95,
    gain: 0.60, hillslope: 900,     // till and peat over low-gradient shield
  },
  {
    key: 'savanna', name: 'Tropical savanna', target: 10.5,
    colour: [0.215, 0.195, 0.105],
    t: [0.60, 9], m: [0.24, 0.45],
    grain: 0.72, treeMax: 10, conifer: 0.05,
    gain: 0.56, hillslope: 1100,    // etchplain: broad low surfaces, inselbergs, deep saprolite
  },
  {
    key: 'steppe', name: 'Temperate grassland & steppe', target: 9.5,
    colour: [0.26, 0.235, 0.135],
    t: [0.24, 0.66], m: [0.18, 0.52],
    grain: 0.62, treeMax: 4, conifer: 0.15,
    gain: 0.68, hillslope: 300,     // semi-arid runoff on sparse cover — the badland end
  },
  {
    key: 'rainforest', name: 'Tropical rainforest', target: 8.0,
    colour: [0.055, 0.088, 0.038],
    t: [0.56, 9], m: [0.42, 9], s: [-9, 0.57],
    grain: 0.26, treeMax: 36, conifer: 0.02,
    gain: 0.58, hillslope: 1200,    // deep saprolite under closed canopy, convex hillslopes
  },
  {
    key: 'tundra', name: 'Tundra', target: 7.5,
    colour: [0.19, 0.185, 0.155],
    t: [0.03, 0.165],
    grain: 0.55, treeMax: 1.6, conifer: 0.3,
    gain: 0.55, hillslope: 1500,    // solifluction and permafrost creep smooth the fine scales
  },
  {
    key: 'broadleaf', name: 'Temperate broadleaf forest', target: 6.5,
    colour: [0.085, 0.115, 0.055],
    t: [0.41, 0.57], m: [0.58, 9],
    grain: 0.4, treeMax: 28, conifer: 0.25,
    gain: 0.62, hillslope: 700,     // the reference: temperate soil-mantled fluvial country
  },
  {
    key: 'dryforest', name: 'Tropical dry & seasonal forest', target: 3.0,
    colour: [0.135, 0.140, 0.070],
    t: [0.60, 9], m: [0.52, 0.72], s: [0.52, 9], bias: 1.15,
    grain: 0.62, treeMax: 17, conifer: 0.05,
    gain: 0.64, hillslope: 500,     // monsoon incision through a thinner cover
  },
  {
    key: 'chaparral', name: 'Mediterranean scrubland', target: 2.0,
    colour: [0.175, 0.165, 0.100],
    t: [0.44, 0.66], m: [0.30, 0.52], s: [0.56, 9], bias: 1.25,
    grain: 0.78, treeMax: 5, conifer: 0.15,
    gain: 0.70, hillslope: 240,     // wet-winter downpours on scrub — the other badland
  },
  {
    // Defined by elevation, so it has to outrank the climate cell it sits in —
    // a 4 km plateau at 20° latitude is puna, not broadleaf forest, whatever
    // its lapsed temperature says. 3400 m is where this planet has ~2% of its
    // land, which is the share Earth's montane grasslands hold; on a less
    // mountainous bake the number moves and this is the constant to re-fit.
    key: 'montane', name: 'Montane grassland & shrubland', target: 2.0,
    colour: [0.205, 0.195, 0.145],
    t: [0.02, 9], b: [0.50, 9], h: [2400, 9000], bias: 1.5,
    grain: 0.82, treeMax: 2.4, conifer: 0.4,
    gain: 0.66, hillslope: 400,     // bare rock and frost shatter; rough to the block scale
  },

  // ── residual: the other ~20% of Earth's land ──────────────────────────
  {
    key: 'ice', name: 'Permanent ice', target: 10.0,
    colour: [0.68, 0.71, 0.76],
    t: [-9, 0.022],
    grain: 0.22, treeMax: 0, conifer: 0.0,
    gain: 0.50, hillslope: 2500,    // glacial scour leaves nothing below the flowline scale
  },
];

export const BIOME_KEYS: readonly string[] = BIOMES.map((b) => b.key);

const clamp01 = (x: number): number => (x < 0 ? 0 : x > 1 ? 1 : x);
const sstep = (e0: number, e1: number, x: number): number => {
  const t = clamp01((x - e0) / (e1 - e0));
  return t * t * (3 - 2 * t);
};

/** Smooth membership of [lo, hi], with `soft` of ramp on each edge. */
function band(x: number, lo: number, hi: number, soft: number): number {
  return sstep(lo - soft, lo + soft, x) * (1 - sstep(hi - soft, hi + soft, x));
}

/**
 * Dry-season strength at this latitude, in [0, 1].
 *
 * `lat` is |sin(latitude)|, the form the rest of the climate uses. The two
 * Gaussians are the two circulation features that create a dry season:
 *
 *   ITCZ margin      the belt the convergence zone crosses but does not sit
 *                    over, ~8–20°. Wet high sun, dry low sun. This is what
 *                    puts savanna between rainforest and desert.
 *   subtropical high  its poleward edge, ~30–42°, which retreats in winter and
 *                    returns in summer: wet winter, dry summer, chaparral.
 *
 * `swing` breaks the zonal symmetry. Real seasonality is not a function of
 * latitude alone — monsoon coasts are far more seasonal than the same latitude
 * inland, and a west coast is drier in summer than an east coast — and a purely
 * zonal field would draw both belts as perfect stripes right around the planet.
 * One low-frequency octave is enough to make them regional.
 */
export function seasonalityAt(lat: number, swing: number): number {
  const deg = (Math.asin(Math.min(1, lat)) * 180) / Math.PI;
  const itcz = Math.exp(-(((deg - 15) / 9) ** 2));
  const subtropical = Math.exp(-(((deg - 36) / 8) ** 2));
  return clamp01(itcz * 0.95 + subtropical * 0.85 + swing * 0.22);
}

/**
 * CPU mirror of `climate_` and `season_`, for the calibration tool.
 *
 * Only the offline tools use this — the renderer evaluates the same fields in
 * the shader. It is here rather than in the tool so the mirror sits beside the
 * WGSL it mirrors, which is the whole of what LESSONS §12 asks for.
 *
 * `wet` is the bake's log10 upstream drainage area at this direction.
 */
export function climateAt(dir: readonly [number, number, number], wet: number): BiomeSample & { lat: number } {
  const lat = Math.abs(dir[1]);
  const temp = Math.pow(Math.max(1 - lat * lat, 0), LAT_EXP);

  let amp = 1;
  let frq = BIOME_F0;
  let sum = 0;
  let norm = 0;
  for (let i = 0; i < 3; i++) {
    const o = i * 19.13 + 5;
    sum += amp * shaderNoise(dir[0] * frq + o, dir[1] * frq + o, dir[2] * frq + o);
    norm += amp;
    amp *= BIOME_GAIN;
    frq *= BIOME_LACUNARITY;
  }
  const province = sum / norm;
  const q = (lat - 0.45) / 0.17;
  const subtropics = Math.exp(-q * q);
  const riparian = sstep(8.6, 10.2, wet);
  const moist = clamp01(0.58 + province * 1.6 - subtropics * 0.55 + riparian * 0.07);

  const swing = shaderNoise(
    dir[0] * SEASON_F0 + 61.7, dir[1] * SEASON_F0 + 61.7, dir[2] * SEASON_F0 + 61.7,
  );
  return { temp, base: temp, moist, season: seasonalityAt(lat, swing), elevation: 0, lat };
}

/** Temperature at elevation, mirroring tempAt_ in shaders/terrain.ts. */
export function tempAtCPU(base: number, h: number): number {
  return clamp01(base - Math.max(h, 0) * LAPSE);
}

export interface BiomeSample {
  /** Temperature after the lapse rate. */
  temp: number;
  /** Temperature before it — latitude alone. */
  base: number;
  moist: number;
  season: number;
  elevation: number;
}

/** Membership weight of every biome, in `BIOMES` order. */
export function biomeWeights(s: BiomeSample): number[] {
  return BIOMES.map((b) => {
    let w = b.bias ?? 1;
    if (b.t) w *= band(s.temp, b.t[0], b.t[1], SOFT.t);
    if (b.m) w *= band(s.moist, b.m[0], b.m[1], SOFT.m);
    if (b.s) w *= band(s.season, b.s[0], b.s[1], SOFT.s);
    if (b.h) w *= band(s.elevation, b.h[0], b.h[1], SOFT.h);
    if (b.b) w *= band(s.base, b.b[0], b.b[1], SOFT.b);
    return w;
  });
}

/** Total weight above which the blend is trusted on its own. */
const SPECTRUM_CLAIM = 0.12;

/**
 * The spectrum of the sub-grid relief here: (fBm gain, crossover octave).
 *
 * The second component is a position on the octave ladder rather than a
 * wavelength — see `spectrumOctave`. Blending there is blending log
 * wavelength, which is the right mean for a scale, and it saves the shader a
 * logarithm per vertex.
 *
 * Falls back to the reference spectrum where no biome claims the point, which
 * is the same ~2% of land the classifier leaves as bare rock. A zero gain
 * would be a flat, unnormalisable ladder, so this is a fallback and not a
 * clamp.
 */
export const REFERENCE_SPECTRUM: readonly [number, number] = [
  RELIEF_GAIN, spectrumOctave(HILLSLOPE_WAVELENGTH),
];

export function spectrumAt(s: BiomeSample): [number, number] {
  const w = biomeWeights(s);
  let tot = 0;
  let g = 0;
  let x = 0;
  for (let i = 0; i < w.length; i++) {
    tot += w[i];
    g += w[i] * BIOMES[i].gain;
    x += w[i] * spectrumOctave(BIOMES[i].hillslope);
  }
  const inv = 1 / Math.max(tot, 1e-4);
  const claimed = sstep(0, SPECTRUM_CLAIM, tot);
  return [
    REFERENCE_SPECTRUM[0] + (g * inv - REFERENCE_SPECTRUM[0]) * claimed,
    REFERENCE_SPECTRUM[1] + (x * inv - REFERENCE_SPECTRUM[1]) * claimed,
  ];
}

/** Index of the dominant biome, or -1 where nothing claims the point. */
export function classify(s: BiomeSample): number {
  const w = biomeWeights(s);
  let best = -1;
  let bestW = 1e-4;
  for (let i = 0; i < w.length; i++) {
    if (w[i] > bestW) {
      bestW = w[i];
      best = i;
    }
  }
  return best;
}

/** WGSL float literal that always parses as one. */
const f = (v: number): string => (Number.isInteger(v) ? `${v}.0` : `${v}`);

/**
 * The same table as WGSL.
 *
 * Emitted from `BIOMES` rather than written out, so a threshold cannot be
 * changed on one side only — the failure mode LESSONS §12 is about. Returns
 * the weighted mean colour and the total weight; the caller decides what to do
 * where nothing claims the point (bare ground, in practice).
 */
export function biomeBlock(sfx: string): string {
  const rows = BIOMES.map((b) => {
    const terms: string[] = [];
    if (b.bias !== undefined) terms.push(f(b.bias));
    if (b.t) terms.push(`bnd_${sfx}(temp, ${f(b.t[0])}, ${f(b.t[1])}, ${f(SOFT.t)})`);
    if (b.m) terms.push(`bnd_${sfx}(moist, ${f(b.m[0])}, ${f(b.m[1])}, ${f(SOFT.m)})`);
    if (b.s) terms.push(`bnd_${sfx}(season, ${f(b.s[0])}, ${f(b.s[1])}, ${f(SOFT.s)})`);
    if (b.h) terms.push(`bnd_${sfx}(h, ${f(b.h[0])}, ${f(b.h[1])}, ${f(SOFT.h)})`);
    if (b.b) terms.push(`bnd_${sfx}(base, ${f(b.b[0])}, ${f(b.b[1])}, ${f(SOFT.b)})`);
    const c = b.colour.map(f).join(', ');
    return `  w = ${terms.join(' * ') || '1.0'};\n` +
           `  acc = acc + w * vec4<f32>(${c}, ${f(b.grain)});\n` +
           `  veg = veg + w * vec2<f32>(${f(b.treeMax)}, ${f(b.conifer)});\n` +
           `  spc = spc + w * vec2<f32>(${f(b.gain)}, ${f(spectrumOctave(b.hillslope))});\n` +
           `  tot = tot + w;   // ${b.key}`;
  }).join('\n');

  // Every one of the three blends declares all three accumulators so `rows`
  // can be one string; the two a given function does not return are dead and
  // the compiler removes them. Writing three row sets instead would be three
  // places for a threshold to drift, which is the thing this file exists to
  // prevent.
  const decl = `  var acc = vec4<f32>(0.0);
  var veg = vec2<f32>(0.0);
  var spc = vec2<f32>(0.0);
  var tot = 0.0;
  var w = 0.0;`;

  // Named constants for the colours the shading still needs by hand — the
  // meso mottle shifts toward savanna or broadleaf, bare rock bleaches toward
  // desert, and snow uses the ice colour. Emitted from the same table so the
  // palette has one home.
  const consts = BIOMES.map(
    (b) => `const bc_${b.key}_${sfx} = vec3<f32>(${b.colour.map(f).join(', ')});`,
  ).join('\n');

  return /* wgsl */ `
${consts}

fn bnd_${sfx}(x: f32, lo: f32, hi: f32, soft: f32) -> f32 {
  return smoothstep(lo - soft, lo + soft, x) * (1.0 - smoothstep(hi - soft, hi + soft, x));
}

/**
 * Dry-season strength — see seasonalityAt in biome.ts, which this mirrors.
 * lat is |dir.y|; asin turns it into degrees, which is the unit the two
 * circulation features are naturally described in.
 */
fn season_${sfx}(lat: f32, swing: f32) -> f32 {
  let deg = asin(min(lat, 1.0)) * 57.29577951308232;
  let itcz = exp(-pow((deg - 15.0) / 9.0, 2.0));
  let subtropical = exp(-pow((deg - 36.0) / 8.0, 2.0));
  return clamp(itcz * 0.95 + subtropical * 0.85 + swing * 0.22, 0.0, 1.0);
}

/**
 * (albedo.rgb, grain strength).
 *
 * The vegetation half of the same blend is biomeVeg_. They are separate
 * functions rather than one struct because no call site needs both: the ground
 * shading wants colour and grain, the scatter wants stem height and species,
 * and splitting them means neither pays for the other's weights.
 */
fn biome_${sfx}(temp: f32, base: f32, moist: f32, season: f32, h: f32) -> vec4<f32> {
${decl}
${rows}
  let inv = 1.0 / max(tot, 1e-4);
  return acc * inv;
}

/** (tallest stem in metres, conifer fraction), blended by the same weights. */
fn biomeVeg_${sfx}(temp: f32, base: f32, moist: f32, season: f32, h: f32) -> vec2<f32> {
${decl}
${rows}
  return veg / max(tot, 1e-4);
}

/**
 * (fBm gain, crossover octave) for the amplification's ladder — the spectrum
 * of the ground rather than its colour. Mirrors spectrumAt in biome.ts.
 *
 * Falls back to the reference spectrum where nothing claims the point: a zero
 * gain is not a smooth landscape, it is an unnormalisable ladder.
 */
fn biomeSpectrum_${sfx}(temp: f32, base: f32, moist: f32, season: f32, h: f32) -> vec2<f32> {
${decl}
${rows}
  let fallback = vec2<f32>(${f(REFERENCE_SPECTRUM[0])}, ${f(REFERENCE_SPECTRUM[1])});
  return mix(fallback, spc / max(tot, 1e-4), smoothstep(0.0, ${f(SPECTRUM_CLAIM)}, tot));
}
`;
}
