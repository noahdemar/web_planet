/**
 * CPU mirror of the WGSL height field in shaders/terrain.ts.
 *
 * Value only — the GPU needs the gradient for normals, the CPU only needs the
 * elevation for camera ground-following. Kept deliberately in lockstep with
 * the shader: if you change one, change both.
 *
 * The two will not agree bit-for-bit — this evaluates in f64, the shader in
 * f32 — but they agree to within a few centimetres, which is far inside the
 * camera's clearance margin. At M5 this is replaced by a read of the tile
 * cache, and the duplication disappears.
 */

import { type V3 } from './math/vec3d.js';
import {
  OCEAN_DEPTH,
  RELIEF_BASE,
  RELIEF_PEAK,
  RELIEF_GAIN,
  RELIEF_LACUNARITY,
  RELIEF_POWER,
  SEA_BAND,
  SEA_LEVEL,
} from './planet.js';

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
function noise3(x: number, y: number, z: number): number {
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
 * The two scale-separated fields the elevation is built from.
 *
 * Neither depends on sea level, which is what makes calibration cheap: the
 * tool evaluates these once and then sweeps sea level as pure arithmetic.
 */
export interface Components {
  /** Continentalness — decides land from ocean. */
  cont: number;
  /** Ridged relief in [0, 1]. */
  ridge: number;
}

export function componentsAt(
  dir: V3,
  octaves: number,
  gain = RELIEF_GAIN,
): Components {
  let cAmp = 1;
  let cFrq = 1.15;
  let cSum = 0;
  let cNorm = 0;
  for (let i = 0; i < 4; i++) {
    const o = i * 19.37;
    cSum += cAmp * noise3(dir[0] * cFrq + o, dir[1] * cFrq + o, dir[2] * cFrq + o);
    cNorm += cAmp;
    cAmp *= 0.5;
    cFrq *= 2.03;
  }

  let mAmp = 1;
  let mFrq = 2.9;
  let mSum = 0;
  let mNorm = 0;
  for (let i = 0; i < octaves; i++) {
    const o = i * 7.77;
    const n = noise3(dir[0] * mFrq + o, dir[1] * mFrq + o, dir[2] * mFrq + o);
    const r = 1 - Math.abs(n);
    mSum += mAmp * r * r;
    mNorm += mAmp;
    mAmp *= gain;
    mFrq *= RELIEF_LACUNARITY;
  }

  return { cont: cSum / cNorm, ridge: mSum / mNorm };
}

/** The knobs `tools/hypsometry.ts` searches over. */
export interface ReliefParams {
  seaLevel: number;
  base: number;
  peak: number;
  power: number;
  oceanDepth: number;
  band: number;
}

export const DEFAULT_RELIEF: ReliefParams = {
  seaLevel: SEA_LEVEL,
  base: RELIEF_BASE,
  peak: RELIEF_PEAK,
  power: RELIEF_POWER,
  oceanDepth: OCEAN_DEPTH,
  band: SEA_BAND,
};

/**
 * Compose the elevation. Must stay in lockstep with `height_*` in
 * shaders/terrain.ts — this is the function `tools/hypsometry.ts` calibrates.
 *
 * Ridged fBm has a mean near 0.56, so used raw it lifts the whole planet above
 * the snow line. Raising it to a power skews the distribution toward lowland,
 * which is what Earth's hypsometric curve looks like (SPEC.md §13).
 */
export function compose(c: Components, p = DEFAULT_RELIEF, hscale = 1): number {
  const mp = Math.pow(c.ridge, p.power);
  const land = sstep(p.seaLevel - p.band, p.seaLevel + p.band, c.cont);
  return (
    (c.cont * p.base + mp * land * land * p.peak + (1 - land) * p.oceanDepth) * hscale
  );
}

/** Elevation in metres above the reference sphere, for a unit direction. */
export function heightAt(
  dir: V3,
  octaves: number,
  hscale = 1,
  seaLevel = SEA_LEVEL,
): number {
  const p = seaLevel === SEA_LEVEL ? DEFAULT_RELIEF : { ...DEFAULT_RELIEF, seaLevel };
  return compose(componentsAt(dir, octaves), p, hscale);
}

