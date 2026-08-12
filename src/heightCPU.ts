/**
 * CPU mirror of the WGSL height field in shaders/terrain.ts.
 *
 * Value only — the GPU needs the gradient for normals, the CPU only needs the
 * elevation for camera ground-following. Kept deliberately in lockstep with
 * the shader: if you change one, change both. tools/mirror.ts measures the
 * disagreement, so "in lockstep" is a number rather than an intention.
 *
 * The two will not agree bit-for-bit — this evaluates in f64, the shader's
 * inputs are f32, and the baked term is bilinear here against the GPU's
 * seamless cube filtering — but they agree to well inside the camera's
 * clearance margin.
 *
 * The gradient noise below is *not* the shader's. It is used by the bake
 * (plates.ts, relief.ts) for plate warping and base relief, where nothing has
 * to match the GPU. The amplification further down uses shaderNoiseCPU.ts,
 * which does.
 */

import { normalize, type V3 } from './math/vec3d.js';
import {
  AMP_BASE,
  AMP_F0,
  AMP_RELIEF,
  BAND_FADE_HI,
  BAND_FADE_LO,
  CHANNEL_DEPTH,
  COAST_WARP_AMP,
  COAST_WARP_F0,
  COAST_WARP_FADE,
  COAST_WARP_OCTAVES,
  CHANNEL_HALF_HI,
  CHANNEL_HALF_LO,
  CHANNEL_WIDTH_K,
  FACE_EDGE,
  FLOODPLAIN_AMP,
  HILLSLOPE_GAIN,
  HILLSLOPE_SOFT,
  LAKE_ON_HI,
  LAKE_ON_LO,
  LOG2_LACUNARITY,
  RADIUS,
  RELIEF_GAIN,
  RELIEF_LACUNARITY,
  RELIEF_SLOPE_HI,
  RELIEF_SLOPE_LO,
  RIDGE_MEAN,
  SHORE_FLAT_FLOOR,
  SHORE_FLAT_HI,
  VALLEY_WET_HI,
  VALLEY_WET_LO,
} from './planet.js';
import { shaderNoise } from './shaderNoiseCPU.js';
import { REFERENCE_SPECTRUM, climateAt, spectrumAt, tempAtCPU } from './biome.js';
import { sampleSurface, type PlanetSurface } from './planetData.js';

const U32 = 4294967296;

/** PCG3D. Must match hash33_* in the shader statement-for-statement. */
function hash33(ix: number, iy: number, iz: number, out: Float64Array): void {
  let px = ix >>> 0;
  let py = iy >>> 0;
  let pz = iz >>> 0;

  px = (Math.imul(px, 1664525) + 1013904223) >>> 0;
  py = (Math.imul(py, 1664525) + 1013904223) >>> 0;
  pz = (Math.imul(pz, 1664525) + 1013904223) >>> 0;

  px = (px + Math.imul(py, pz)) >>> 0;
  py = (py + Math.imul(pz, px)) >>> 0;
  pz = (pz + Math.imul(px, py)) >>> 0;

  px = (px ^ (px >>> 16)) >>> 0;
  py = (py ^ (py >>> 16)) >>> 0;
  pz = (pz ^ (pz >>> 16)) >>> 0;

  px = (px + Math.imul(py, pz)) >>> 0;
  py = (py + Math.imul(pz, px)) >>> 0;
  pz = (pz + Math.imul(px, py)) >>> 0;

  out[0] = -1 + (2 * px) / U32;
  out[1] = -1 + (2 * py) / U32;
  out[2] = -1 + (2 * pz) / U32;
}

const g0 = new Float64Array(3);

/** Gradient noise, value only. */
export function noise3(x: number, y: number, z: number): number {
  const px = Math.floor(x);
  const py = Math.floor(y);
  const pz = Math.floor(z);
  const fx = x - px;
  const fy = y - py;
  const fz = z - pz;

  const ux = fx * fx * fx * (fx * (fx * 6 - 15) + 10);
  const uy = fy * fy * fy * (fy * (fy * 6 - 15) + 10);
  const uz = fz * fz * fz * (fz * (fz * 6 - 15) + 10);

  const corner = (cx: number, cy: number, cz: number): number => {
    hash33(px + cx, py + cy, pz + cz, g0);
    return g0[0] * (fx - cx) + g0[1] * (fy - cy) + g0[2] * (fz - cz);
  };

  const va = corner(0, 0, 0);
  const vb = corner(1, 0, 0);
  const vc = corner(0, 1, 0);
  const vd = corner(1, 1, 0);
  const ve = corner(0, 0, 1);
  const vf = corner(1, 0, 1);
  const vg = corner(0, 1, 1);
  const vh = corner(1, 1, 1);

  const k0 = va;
  const k1 = vb - va;
  const k2 = vc - va;
  const k3 = ve - va;
  const k4 = va - vb - vc + vd;
  const k5 = va - vc - ve + vg;
  const k6 = va - vb - ve + vf;
  const k7 = -va + vb + vc - vd + ve - vf - vg + vh;

  return (
    k0 +
    k1 * ux +
    k2 * uy +
    k3 * uz +
    k4 * ux * uy +
    k5 * uy * uz +
    k6 * uz * ux +
    k7 * ux * uy * uz
  );
}

const sstep = (e0: number, e1: number, x: number): number => {
  const t = Math.max(0, Math.min(1, (x - e0) / (e1 - e0)));
  return t * t * (3 - 2 * t);
};

/**
 * The baked surface the CPU mirror sits on. Set once at startup by main.ts;
 * `heightAt` throws without it rather than quietly returning a field the GPU
 * is not drawing.
 */
let surface: PlanetSurface | null = null;

export function setPlanetSurface(s: PlanetSurface): void {
  surface = s;
}

/**
 * Elevation in metres above the reference sphere, for a unit direction.
 *
 * Must stay in lockstep with `height_N` in shaders/terrain.ts. The camera
 * stands on what this returns, so a disagreement of a few metres is a camera
 * that floats or sinks — which is why the amplification here uses the shader's
 * own hash (shaderNoiseCPU.ts) rather than the one above.
 *
 * `bandLimit` defaults to the shader's finest, because the CPU is asked for
 * the surface the player is standing on, which is always the fully detailed
 * one.
 */
/**
 * Coastline warp, mirroring `coastWarp` in shaders/terrain.ts exactly.
 *
 * Exported because *every* consumer of the bake has to apply it. The warp is
 * 0.62 of a texel, which is 5.6 km, and elevation moves hundreds of metres
 * over that — so a scatter that samples the unwarped direction is standing on
 * a different planet from the one being drawn.
 *
 * Any drift between the two moves the ground out from under the camera, so
 * this is written to match the shader line for line rather than idiomatically.
 */
export function warpForCoast(dir: V3, bakeH: number): V3 {
  const wgt = 1 - sstep(0, COAST_WARP_FADE, Math.abs(bakeH));
  if (wgt <= 0) return dir;
  const ax: V3 = Math.abs(dir[1]) > 0.95 ? [1, 0, 0] : [0, 1, 0];
  const e0 = normalize([
    ax[1] * dir[2] - ax[2] * dir[1],
    ax[2] * dir[0] - ax[0] * dir[2],
    ax[0] * dir[1] - ax[1] * dir[0],
  ]);
  const e1: V3 = [
    dir[1] * e0[2] - dir[2] * e0[1],
    dir[2] * e0[0] - dir[0] * e0[2],
    dir[0] * e0[1] - dir[1] * e0[0],
  ];
  let amp = 1;
  let frq = COAST_WARP_F0;
  let u = 0;
  let v = 0;
  let norm = 0;
  for (let i = 0; i < COAST_WARP_OCTAVES; i++) {
    const a = i * 3.71 + 1.5;
    const b = i * 3.71 + 61.3;
    u += amp * shaderNoise(dir[0] * frq + a, dir[1] * frq + a, dir[2] * frq + a);
    v += amp * shaderNoise(dir[0] * frq + b, dir[1] * frq + b, dir[2] * frq + b);
    norm += amp;
    amp *= 0.5;
    frq *= 2.17;
  }
  const k = (COAST_WARP_AMP * wgt) / (norm * RADIUS);
  return normalize([
    dir[0] + (e0[0] * u + e1[0] * v) * k,
    dir[1] + (e0[1] * u + e1[1] * v) * k,
    dir[2] + (e0[2] * u + e1[2] * v) * k,
  ]);
}

export function heightAt(
  dir: V3,
  octaves: number,
  hscale = 1,
  bandLimit = 1e9,
  flood = true,
): number {
  if (!surface) throw new Error('heightAt: call setPlanetSurface() first');

  // Same coastline warp the shader applies to its lookup — see COAST_WARP_AMP.
  // Mirrored rather than skipped: this function is what the camera stands on
  // and what the LOD selector measures against, so any term the shader has and
  // this one does not is a divergence between where the ground is and where
  // the ground is drawn.
  const probe = sampleSurface(surface, dir[0], dir[1], dir[2]).elevation;
  const wdir = warpForCoast(dir, probe);
  const { elevation: bakeH, wetness, lakeDepth, channelDist } =
    sampleSurface(surface, wdir[0], wdir[1], wdir[2]);

  // Baked slope by the same one-cell central difference the shader uses. The
  // tangent frame here is arbitrary — unlike the shader there is no face basis
  // to hand — so pick the axis least aligned with dir and orthogonalise.
  const e = FACE_EDGE / surface.size / RADIUS;
  const ax = Math.abs(dir[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
  let t1: V3 = [
    ax[0] - dir[0] * (ax[0] * dir[0] + ax[1] * dir[1] + ax[2] * dir[2]),
    ax[1] - dir[1] * (ax[0] * dir[0] + ax[1] * dir[1] + ax[2] * dir[2]),
    ax[2] - dir[2] * (ax[0] * dir[0] + ax[1] * dir[1] + ax[2] * dir[2]),
  ];
  t1 = normalize(t1);
  const t2: V3 = normalize([
    dir[1] * t1[2] - dir[2] * t1[1],
    dir[2] * t1[0] - dir[0] * t1[2],
    dir[0] * t1[1] - dir[1] * t1[0],
  ]);
  const along = (t: V3, sgn: number): { elevation: number; wetness: number } => {
    const d = normalize([
      wdir[0] + t[0] * e * sgn,
      wdir[1] + t[1] * e * sgn,
      wdir[2] + t[2] * e * sgn,
    ]);
    return sampleSurface(surface!, d[0], d[1], d[2]);
  };
  const xp = along(t1, 1), xm = along(t1, -1);
  const yp = along(t2, 1), ym = along(t2, -1);
  const gx = (xp.elevation - xm.elevation) / (2 * e);
  const gy = (yp.elevation - ym.elevation) / (2 * e);
  const slope = Math.hypot(gx, gy) / RADIUS;

  // Distance to the nearest channel, straight from the bake. See
  // carveChannels: it is measured, not reconstructed.
  const distAxis = channelDist;

  const relief = sstep(RELIEF_SLOPE_LO, RELIEF_SLOPE_HI, slope);
  const landW = sstep(-350, 40, bakeH);
  const valley = sstep(VALLEY_WET_LO, VALLEY_WET_HI, wetness);
  // Matches the shore term in the shader: the coastline is pinned to the bake
  // so it cannot move with the camera.
  const shore = SHORE_FLAT_FLOOR + (1 - SHORE_FLAT_FLOOR) * sstep(0, SHORE_FLAT_HI, Math.abs(bakeH));
  const land = (AMP_BASE + AMP_RELIEF * relief) * (0.22 + 0.78 * landW) * shore;
  const amp = land + (FLOODPLAIN_AMP - land) * valley;

  // The direction the shader actually has: it reconstructs this in f32, so the
  // mirror must start from the same value or every octave is offset.
  const fdx = Math.fround(dir[0]);
  const fdy = Math.fround(dir[1]);
  const fdz = Math.fround(dir[2]);

  // The spectrum of the ground here, from the biome table — see the gain field
  // on Biome, and spectrumBlock in shaders/terrain.ts for why it is evaluated
  // against the *baked* elevation rather than the one being computed.
  //
  // The shader evaluates its climate at the patch's band limit and this one at
  // full resolution. They agree wherever it matters: the amplification's first
  // octave does not switch on until a band limit of 560, by which point the
  // climate's finest octave (frequency 21) has been saturated for an order of
  // magnitude.
  const c = climateAt(dir, wetness);
  const [sGain, sCross] = spectrumAt({
    temp: tempAtCPU(c.temp, bakeH),
    base: c.temp,
    moist: c.moist,
    season: c.season,
    elevation: bakeH,
  });

  let mAmp = 1;
  let mFrq = AMP_F0;
  let mSum = 0;
  let mSq = 0;
  let mBias = 0;
  // The reference ladder, walked alongside — see the normaliser in height_.
  let rAmp = 1;
  let rSum = 0;
  let rSq = 0;
  let mOct = sCross;
  let rOct = REFERENCE_SPECTRUM[1];
  for (let i = 0; i < octaves; i++) {
    const w = sstep(BAND_FADE_LO, BAND_FADE_HI, bandLimit / mFrq);
    if (w > 0.002) {
      const o = i * 7.77;
      // Rounded to f32 *before* the hash, because the shader has no choice but
      // to evaluate this in f32 and the lattice cell must be the same one.
      //
      // This is the whole CPU/GPU height divergence. At octave 8 the argument
      // reaches 1.75e5, where one f32 ULP is 0.0156 against a lattice cell of
      // 1.0 — so a percent or two of samples floor into a *different cell* on
      // the GPU than on the CPU, and a different cell means a completely
      // uncorrelated value for that octave, not a slightly different one. Nine
      // octaves of that, multiplied by an amplitude that reaches 4150 m in
      // mountainous country, is metres of error on a plain and hundreds of
      // metres on a ridge — which is precisely how the error was observed to
      // scale. The camera stands on this function, so in the mountains it
      // stood inside the hill.
      //
      // fround costs nothing and makes the two agree bit for bit.
      const n = shaderNoise(
        Math.fround(Math.fround(fdx * mFrq) + o),
        Math.fround(Math.fround(fdy * mFrq) + o),
        Math.fround(Math.fround(fdz * mFrq) + o),
      );
      const r = 1 - Math.abs(n);
      mSum += w * mAmp * r * r;
      mBias += w * mAmp * RIDGE_MEAN;
    }
    mSq += mAmp * mAmp;
    rSum += rAmp;
    rSq += rAmp * rAmp;
    // Matches the mix() in the shader: the gain falls toward HILLSLOPE_GAIN
    // across the crossover, which is a per-point position on the ladder.
    mAmp *= sGain + (HILLSLOPE_GAIN - sGain) * sstep(-HILLSLOPE_SOFT, HILLSLOPE_SOFT, mOct);
    rAmp *= RELIEF_GAIN + (HILLSLOPE_GAIN - RELIEF_GAIN) * sstep(-HILLSLOPE_SOFT, HILLSLOPE_SOFT, rOct);
    mOct += LOG2_LACUNARITY;
    rOct += LOG2_LACUNARITY;
    mFrq *= RELIEF_LACUNARITY;
  }
  // Variance-preserving, as in height_ — see the normaliser note there.
  const mNorm = (Math.sqrt(mSq) * rSum) / Math.sqrt(rSq);

  // The reconstructed channel, cut into the valley floor. Mirrors channel_().
  const chOn = sstep(VALLEY_WET_LO, VALLEY_WET_HI, wetness);
  const halfW = Math.min(
    COAST_WARP_AMP,
  COAST_WARP_F0,
  COAST_WARP_FADE,
  COAST_WARP_OCTAVES,
  CHANNEL_HALF_HI,
    Math.max(CHANNEL_HALF_LO, CHANNEL_WIDTH_K * Math.sqrt(10 ** wetness)),
  );
  const chM = chOn > 0 ? 1 - sstep(0, halfW * 2.5, distAxis) : 0;
  const incision = CHANNEL_DEPTH * chOn * chM * chM;

  const h0 = (bakeH - incision + ((mSum - mBias) / mNorm) * amp) * hscale;

  const h = h0;

  // Standing water. The GPU draws the higher of the ground and the waterline
  // (see waterLevel_ in shaders/terrain.ts), so this has to as well: the camera
  // follows this function, and without it the ground-follow mode walks along
  // the riverbed while the render shows a river surface a few metres above.
  //
  // `flood` is off only for the offline tools. The hypsographic curve and the
  // amplification bias are properties of the *solid* surface: with the sea laid
  // over it, 72% of the planet reads as exactly 0 m, the curve loses its whole
  // bathymetric half, and "amplification must be zero-mean" becomes a statement
  // about the difference between sea level and the abyssal plain.
  if (!flood) return h;
  const lakeOn = sstep(LAKE_ON_LO, LAKE_ON_HI, lakeDepth);
  const waterZ = Math.max(
    0,
    bakeH + lakeDepth - 20000 * (1 - lakeOn),
    // No river water — see the note in waterLevel_ in shaders/terrain.ts.
    // The carved valley stays; only the water in it is gone.
  );
  return Math.max(h, waterZ * hscale);
}
