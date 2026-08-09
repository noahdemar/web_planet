/**
 * Resample the bake onto a padded cube atlas.
 *
 * The solve grid uses the tangent warp (SPEC.md §3) because that keeps cell
 * areas within 1.05× across a face, and the landscape evolution model weights
 * everything by cell area. Cube-face lookup at runtime is gnomonic: given a
 * direction it forms u = x/|z| with no warp. So the two cannot be the same
 * array, and something has to resample.
 *
 * Why an atlas rather than a hardware cube map: TSL's `cubeTexture` applies
 * its own transform to the lookup vector before sampling — a handedness flip
 * and an object matrix — and neither is under this file's control. Six faces
 * side by side in one 2D texture, with the direction-to-face mapping written
 * out explicitly in WGSL, is the same cost and leaves nothing implicit.
 *
 * Each face carries a one-texel border filled from *outside* its own bounds.
 * That costs 0.4% of the texture and it is what makes cross-face filtering
 * exact: bilinear at a face edge blends this face's last texel with the
 * neighbouring face's first, which is what a seamless cube map does in
 * hardware. Without it there would be a visible crack along every face
 * boundary — 1600 km of seam, six times over.
 *
 * The border needs no adjacency table. `cubeTexelDirection` with a coordinate
 * outside [0,1] returns a direction that has simply left the face, and the
 * solve-grid lookup finds whichever face it landed on.
 *
 * Channels are (elevation m, wetness, 0, 1) in half float. Half float has an
 * 11-bit mantissa, so an 8000 m summit quantises to ~8 m — but the value is
 * quantised *before* interpolation, not after, so the reconstructed surface is
 * still continuous and the slope error is 8 m over an 18 km cell. Low ground,
 * where a metre would be noticeable, is far more precise: floating point puts
 * its resolution where the values are small.
 */

import { DataUtils } from 'three';
import { cubePoint, directionToFace, warp } from '../cubesphere.js';
import { type Grid } from './grid.js';

/** Faces per atlas row, and rows. Six faces in a 3×2 grid. */
export const ATLAS_COLS = 3;
export const ATLAS_ROWS = 2;
/** Border texels on each side of every face. */
export const ATLAS_PAD = 1;

/**
 * Direction for a cube-face coordinate.
 *
 * s and t are in [0,1] inside the face and may run outside it for the border,
 * where the returned direction belongs to a neighbour. The axis assignment is
 * arbitrary but must match `faceCoords` in src/planetData.ts and `cubeFace_`
 * in the shader exactly — all three are the same mapping written three times.
 */
export function cubeTexelDirection(face: number, s: number, t: number): [number, number, number] {
  const a = 2 * s - 1;
  const b = 2 * t - 1;
  let v: [number, number, number];
  switch (face) {
    case 0: v = [1, -b, -a]; break; // +X
    case 1: v = [-1, -b, a]; break; // −X
    case 2: v = [a, 1, b]; break; // +Y
    case 3: v = [a, -1, -b]; break; // −Y
    case 4: v = [a, -b, 1]; break; // +Z
    default: v = [-a, -b, -1]; break; // −Z
  }
  const l = Math.hypot(v[0], v[1], v[2]);
  return [v[0] / l, v[1] / l, v[2] / l];
}

/**
 * Bilinear sample of a per-cell field on the warped solve grid.
 *
 * Out-of-range cells are resolved *geometrically* rather than clamped. That
 * distinction is the whole seam: clamping replicates the outermost ring, so
 * both faces reconstruct the last half-cell as flat and they flatten it to
 * different values. Measured, that was a 381 m step across 130 m of ground
 * along a face edge — a straight-line cliff long enough to read as a rift
 * valley from orbit.
 *
 * The fix is the same trick buildGrid uses for its neighbour table: a cell
 * coordinate outside the face still names a real point on the sphere, so
 * build its direction and ask which face it actually belongs to.
 */
function sampleGrid(grid: Grid, field: ArrayLike<number>, d: [number, number, number]): number {
  const { face, u, v } = directionToFace(d);
  const res = grid.res;
  const x = ((u + 1) / 2) * res - 0.5;
  const y = ((v + 1) / 2) * res - 0.5;
  const i0 = Math.floor(x);
  const j0 = Math.floor(y);
  const fx = x - i0;
  const fy = y - j0;

  const at = (i: number, j: number): number => {
    if (i >= 0 && i < res && j >= 0 && j < res) {
      return field[(face * res + j) * res + i];
    }
    const cu = ((i + 0.5) / res) * 2 - 1;
    const cv = ((j + 0.5) / res) * 2 - 1;
    const p = cubePoint(face, warp(cu), warp(cv));
    const l = Math.hypot(p[0], p[1], p[2]);
    const nd = directionToFace([p[0] / l, p[1] / l, p[2] / l]);
    const ni = Math.min(res - 1, Math.max(0, Math.floor(((nd.u + 1) / 2) * res)));
    const nj = Math.min(res - 1, Math.max(0, Math.floor(((nd.v + 1) / 2) * res)));
    return field[(nd.face * res + nj) * res + ni];
  };

  return (
    at(i0, j0) * (1 - fx) * (1 - fy) +
    at(i0 + 1, j0) * fx * (1 - fy) +
    at(i0, j0 + 1) * (1 - fx) * fy +
    at(i0 + 1, j0 + 1) * fx * fy
  );
}

export interface AtlasData {
  /** Texels per face edge, excluding the border. */
  faceSize: number;
  width: number;
  height: number;
  /** RGBA half float, width·height·4. */
  data: Uint16Array;
}

export function toAtlas(
  grid: Grid,
  elevation: Float32Array,
  wetness: Float32Array,
  faceSize = grid.res,
): AtlasData {
  const cell = faceSize + 2 * ATLAS_PAD;
  const width = cell * ATLAS_COLS;
  const height = cell * ATLAS_ROWS;
  const data = new Uint16Array(width * height * 4);
  const half = DataUtils.toHalfFloat;
  const one = half(1);

  for (let f = 0; f < 6; f++) {
    const ox = (f % ATLAS_COLS) * cell;
    const oy = Math.floor(f / ATLAS_COLS) * cell;
    for (let j = -ATLAS_PAD; j < faceSize + ATLAS_PAD; j++) {
      for (let i = -ATLAS_PAD; i < faceSize + ATLAS_PAD; i++) {
        const d = cubeTexelDirection(f, (i + 0.5) / faceSize, (j + 0.5) / faceSize);
        const px = ox + ATLAS_PAD + i;
        const py = oy + ATLAS_PAD + j;
        const o = (py * width + px) * 4;
        data[o] = half(sampleGrid(grid, elevation, d));
        data[o + 1] = half(sampleGrid(grid, wetness, d));
        data[o + 2] = 0;
        data[o + 3] = one;
      }
    }
  }

  return { faceSize, width, height, data };
}
