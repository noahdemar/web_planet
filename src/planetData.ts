/**
 * Load the M3 bake and expose it to both the GPU and the CPU.
 *
 * The bake takes ~40 s (tools/bake.ts) and is deterministic from a seed, so it
 * runs offline and ships as a 12.6 MB asset rather than being recomputed at
 * every launch. That is well inside the 5 GB budget in SPEC.md §9 and it is
 * the difference between a two-second start and a forty-second one.
 *
 * Two consumers, one file:
 *
 *   GPU  a seamless cube map, sampled per vertex by the terrain shader
 *   CPU  the same data, for ground-following and any tool that needs the
 *        surface without a renderer
 *
 * They must agree. The CPU side deliberately re-implements the same lookup the
 * shader does — major axis, face coordinates, atlas offset, bilinear — instead
 * of going back to the solve grid, so a disagreement can only come from the
 * arithmetic, not from a different field.
 */

import {
  ClampToEdgeWrapping,
  DataTexture,
  DataUtils,
  HalfFloatType,
  LinearFilter,
  RGBAFormat,
} from 'three';
import { ATLAS_COLS, ATLAS_PAD } from './bake/cubemap.js';

export interface PlanetSurface {
  /** Texels per face edge, excluding the border. */
  size: number;
  width: number;
  height: number;
  data: Uint16Array;
  texture: DataTexture;
  /** Solve-grid resolution the bake ran at, for reporting. */
  solveRes: number;
  seed: number;
}

export interface Meta {
  size: number;
  width: number;
  height: number;
  solveRes: number;
  seed: number;
  format: string;
}

/**
 * Build the surface from raw bytes. Split out from the fetch so offline tools
 * can exercise exactly the same decode path the runtime uses — a mirror check
 * that reconstructed the data differently would not be checking anything.
 */
export function surfaceFromBuffer(meta: Meta, buf: ArrayBuffer): PlanetSurface {
  const expect = meta.width * meta.height * 4 * 2;
  if (buf.byteLength !== expect) {
    throw new Error(`surface.f16 is ${buf.byteLength} bytes, expected ${expect}`);
  }
  const data = new Uint16Array(buf);

  const texture = new DataTexture(data, meta.width, meta.height, RGBAFormat, HalfFloatType);
  texture.minFilter = LinearFilter;
  texture.magFilter = LinearFilter;
  texture.wrapS = ClampToEdgeWrapping;
  texture.wrapT = ClampToEdgeWrapping;
  texture.generateMipmaps = false;
  // Raw data, not an image: three must not flip it, or every face is mirrored
  // about its horizontal axis.
  texture.flipY = false;
  texture.needsUpdate = true;

  return {
    size: meta.size,
    width: meta.width,
    height: meta.height,
    data,
    texture,
    solveRes: meta.solveRes,
    seed: meta.seed,
  };
}

export async function loadPlanetSurface(base = 'planet'): Promise<PlanetSurface> {
  const meta: Meta = await fetch(`${base}/meta.json`).then((r) => {
    if (!r.ok) throw new Error(`${base}/meta.json missing — run: npm run bake -- --write`);
    return r.json();
  });
  const buf = await fetch(`${base}/surface.f16`).then((r) => {
    if (!r.ok) throw new Error(`${base}/surface.f16 missing — run: npm run bake -- --write`);
    return r.arrayBuffer();
  });
  return surfaceFromBuffer(meta, buf);
}

/** Face index and face coordinates for a direction, per the cube-map rules. */
function faceCoords(x: number, y: number, z: number): [number, number, number] {
  const ax = Math.abs(x);
  const ay = Math.abs(y);
  const az = Math.abs(z);
  let face: number;
  let sc: number;
  let tc: number;
  let ma: number;
  if (ax >= ay && ax >= az) {
    face = x > 0 ? 0 : 1;
    ma = ax;
    sc = x > 0 ? -z : z;
    tc = -y;
  } else if (ay >= az) {
    face = y > 0 ? 2 : 3;
    ma = ay;
    sc = x;
    tc = y > 0 ? z : -z;
  } else {
    face = z > 0 ? 4 : 5;
    ma = az;
    sc = z > 0 ? x : -x;
    tc = -y;
  }
  return [face, (sc / ma + 1) / 2, (tc / ma + 1) / 2];
}

/**
 * Bilinear sample of the baked surface. Returns (elevation m, wetness).
 *
 * Reads the same padded atlas the GPU does, with the same bilinear weights, so
 * the two agree to the precision of the arithmetic — including across face
 * boundaries, where the border ring carries the neighbour's data.
 */
export function sampleSurface(
  s: PlanetSurface,
  x: number,
  y: number,
  z: number,
): { elevation: number; wetness: number } {
  const [face, u, v] = faceCoords(x, y, z);
  const n = s.size;
  const cell = n + 2 * ATLAS_PAD;
  const ox = (face % ATLAS_COLS) * cell + ATLAS_PAD;
  const oy = Math.floor(face / ATLAS_COLS) * cell + ATLAS_PAD;
  const fx = u * n - 0.5;
  const fy = v * n - 0.5;
  const i0 = Math.floor(fx);
  const j0 = Math.floor(fy);
  const tx = fx - i0;
  const ty = fy - j0;

  const at = (i: number, j: number, c: number): number => {
    // Only the border ring is outside [0, n); clamping there stays inside the
    // padded region, which holds the neighbouring face's data.
    const ii = Math.min(n, Math.max(-1, i));
    const jj = Math.min(n, Math.max(-1, j));
    return DataUtils.fromHalfFloat(s.data[((oy + jj) * s.width + ox + ii) * 4 + c]);
  };
  const lerp2 = (c: number): number =>
    at(i0, j0, c) * (1 - tx) * (1 - ty) +
    at(i0 + 1, j0, c) * tx * (1 - ty) +
    at(i0, j0 + 1, c) * (1 - tx) * ty +
    at(i0 + 1, j0 + 1, c) * tx * ty;

  return { elevation: lerp2(0), wetness: lerp2(1) };
}
