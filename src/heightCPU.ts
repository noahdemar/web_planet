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
import { FACE_EDGE, RADIUS, RELIEF_GAIN, RELIEF_LACUNARITY } from './planet.js';
import { AMP_BASE, AMP_F0, AMP_RELIEF, RIDGE_MEAN, shaderNoise } from './shaderNoiseCPU.js';
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
export function heightAt(
  dir: V3,
  octaves: number,
  hscale = 1,
  bandLimit = 1e9,
): number {
  if (!surface) throw new Error('heightAt: call setPlanetSurface() first');

  const { elevation: bakeH, wetness } = sampleSurface(surface, dir[0], dir[1], dir[2]);

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
  const along = (t: V3, sgn: number): number => {
    const d = normalize([dir[0] + t[0] * e * sgn, dir[1] + t[1] * e * sgn, dir[2] + t[2] * e * sgn]);
    return sampleSurface(surface!, d[0], d[1], d[2]).elevation;
  };
  const gx = (along(t1, 1) - along(t1, -1)) / (2 * e);
  const gy = (along(t2, 1) - along(t2, -1)) / (2 * e);
  const slope = Math.hypot(gx, gy) / RADIUS;

  const relief = sstep(0.006, 0.085, slope);
  const landW = sstep(-350, 40, bakeH);
  const valley = sstep(7.5, 10.5, wetness);
  const amp = (AMP_BASE + AMP_RELIEF * relief) * (0.22 + 0.78 * landW) * (1 - 0.72 * valley);

  let mAmp = 1;
  let mFrq = AMP_F0;
  let mSum = 0;
  let mNorm = 0;
  let mBias = 0;
  for (let i = 0; i < octaves; i++) {
    const w = sstep(1, 2.5, bandLimit / mFrq);
    if (w > 0.002) {
      const o = i * 7.77;
      const n = shaderNoise(dir[0] * mFrq + o, dir[1] * mFrq + o, dir[2] * mFrq + o);
      const r = 1 - Math.abs(n);
      mSum += w * mAmp * r * r;
      mBias += w * mAmp * RIDGE_MEAN;
    }
    mNorm += mAmp;
    mAmp *= RELIEF_GAIN;
    mFrq *= RELIEF_LACUNARITY;
  }

  return (bakeH + ((mSum - mBias) / mNorm) * amp) * hscale;
}
