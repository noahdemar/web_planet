/**
 * Cube-sphere with tangent warp. See SPEC.md §3.
 *
 * A face point is  P = warp(u)·U + warp(v)·V + W  for (u,v) ∈ [-1,1]²,
 * and the surface direction is normalize(P).
 *
 * warp(x) = tan(x·π/4) reduces corner-to-centre area distortion from
 * 1.30× (naive normalize) to ≈1.05×, which also keeps cell areas
 * near-uniform enough for flow accumulation to be physical later.
 */

import { type V3, cross, dot, len, normalize, scale } from './math/vec3d.js';

export const QUARTER_PI = Math.PI / 4;

export const warp = (x: number): number => Math.tan(x * QUARTER_PI);

/** Face basis: U and V span the face, W is the outward axis. */
export interface FaceBasis {
  readonly U: V3;
  readonly V: V3;
  readonly W: V3;
}

/**
 * Six faces, wound so that (U × V) · W > 0 for every face. That gives
 * consistent triangle winding across the whole sphere, so backface
 * culling behaves identically on all six roots.
 */
export const FACES: readonly FaceBasis[] = [
  { U: [0, 0, -1], V: [0, 1, 0], W: [1, 0, 0] }, // +X
  { U: [0, 0, 1], V: [0, 1, 0], W: [-1, 0, 0] }, // -X
  { U: [1, 0, 0], V: [0, 0, -1], W: [0, 1, 0] }, // +Y
  { U: [1, 0, 0], V: [0, 0, 1], W: [0, -1, 0] }, // -Y
  { U: [1, 0, 0], V: [0, 1, 0], W: [0, 0, 1] }, // +Z
  { U: [-1, 0, 0], V: [0, 1, 0], W: [0, 0, -1] }, // -Z
];

// Verified once at module load: a wrong sign here produces inside-out
// faces that are maddening to diagnose from the rendered image alone.
for (const f of FACES) {
  if (dot(cross(f.U, f.V), f.W) <= 0) {
    throw new Error('cubesphere: face basis is left-handed');
  }
}

/** Unwarped cube point for face-local (u,v) — warp is applied by the caller. */
export function cubePoint(face: number, wu: number, wv: number): V3 {
  const { U, V, W } = FACES[face];
  return [
    U[0] * wu + V[0] * wv + W[0],
    U[1] * wu + V[1] * wv + W[1],
    U[2] * wu + V[2] * wv + W[2],
  ];
}

/** Surface direction (unit vector) for face-local (u,v) ∈ [-1,1]². */
export function faceToDirection(face: number, u: number, v: number): V3 {
  return normalize(cubePoint(face, warp(u), warp(v)));
}

/** Length of the unwarped cube point — needed by the shader's delta expansion. */
export function cubePointLength(face: number, wu: number, wv: number): number {
  return len(cubePoint(face, wu, wv));
}

/**
 * Which face a direction belongs to, plus its face-local (u,v).
 * Used to seed traversal from the camera's sub-planet point.
 */
export function directionToFace(d: V3): { face: number; u: number; v: number } {
  const ax = Math.abs(d[0]);
  const ay = Math.abs(d[1]);
  const az = Math.abs(d[2]);

  let face: number;
  if (ax >= ay && ax >= az) face = d[0] > 0 ? 0 : 1;
  else if (ay >= az) face = d[1] > 0 ? 2 : 3;
  else face = d[2] > 0 ? 4 : 5;

  const { U, V, W } = FACES[face];
  const w = dot(d, W);
  const inv = 1 / w;
  // Invert the tangent warp to recover the face-local coordinates.
  const u = Math.atan(dot(d, U) * inv) / QUARTER_PI;
  const v = Math.atan(dot(d, V) * inv) / QUARTER_PI;
  return { face, u, v };
}

/** Surface position at a given elevation above the reference sphere. */
export function surfacePoint(
  face: number,
  u: number,
  v: number,
  radius: number,
): V3 {
  return scale(faceToDirection(face, u, v), radius);
}
