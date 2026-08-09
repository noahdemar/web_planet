/**
 * The global bake grid. See SPEC.md §4.
 *
 * One cell array covering the whole sphere as six cube-sphere faces, so the
 * landscape evolution model runs over a single connected domain. That is the
 * whole point of baking globally: a river's course depends on its entire
 * upstream basin, and a basin does not respect tile boundaries (SPEC.md §1).
 *
 * Cross-face neighbours are found by *geometry* rather than a hand-written
 * adjacency table: step outside a face in face-uv, build the cube point
 * anyway, normalise it, and ask which face that direction belongs to. The
 * cube-sphere map is continuous across seams, so this is exact, and it cannot
 * develop the sign and winding bugs an adjacency table invites.
 */

import { FACE_EDGE } from '../planet.js';
import { cubePoint, directionToFace, warp, QUARTER_PI } from '../cubesphere.js';
import { type V3, normalize } from '../math/vec3d.js';

export interface Grid {
  /** Cells per face edge. */
  readonly res: number;
  /** Total cells: 6 · res². */
  readonly count: number;
  /** Approximate surface spacing between adjacent cells, metres. */
  readonly spacing: number;
  /**
   * Smallest cardinal neighbour distance anywhere on the grid. The tangent
   * warp makes cells near a face centre wider than cells near an edge, so an
   * explicit scheme's stability limit is set by this, not by `spacing`.
   */
  readonly minSpacing: number;
  /** Unit direction of each cell centre, packed xyz. */
  readonly dirs: Float32Array;
  /**
   * Eight neighbours per cell, D8 order. Every entry is a valid cell index —
   * seams are resolved at construction, so no consumer needs a boundary case.
   */
  readonly nbr: Int32Array;
  /** Great-circle distance to each neighbour, metres. Varies with face warp. */
  readonly nbrDist: Float32Array;
}

const OFFSETS: readonly (readonly [number, number])[] = [
  [-1, -1], [0, -1], [1, -1],
  [-1, 0], [1, 0],
  [-1, 1], [0, 1], [1, 1],
];

/** Face-local coordinate of a cell centre, in [-1, 1]. */
function cellUV(res: number, i: number, j: number): [number, number] {
  return [((i + 0.5) / res) * 2 - 1, ((j + 0.5) / res) * 2 - 1];
}

function dirOf(face: number, u: number, v: number): V3 {
  return normalize(cubePoint(face, warp(u), warp(v)));
}

/** Cell index for a direction, by inverting the cube-sphere map. */
export function cellAt(grid: Grid, d: V3): number {
  const { face, u, v } = directionToFace(d);
  const res = grid.res;
  const i = Math.min(res - 1, Math.max(0, Math.floor(((u + 1) / 2) * res)));
  const j = Math.min(res - 1, Math.max(0, Math.floor(((v + 1) / 2) * res)));
  return (face * res + j) * res + i;
}

export function buildGrid(res: number): Grid {
  const count = 6 * res * res;
  const dirs = new Float32Array(count * 3);
  const nbr = new Int32Array(count * 8);
  const nbrDist = new Float32Array(count * 8);
  const spacing = FACE_EDGE / res;

  for (let f = 0; f < 6; f++) {
    for (let j = 0; j < res; j++) {
      for (let i = 0; i < res; i++) {
        const c = (f * res + j) * res + i;
        const [u, v] = cellUV(res, i, j);
        const d = dirOf(f, u, v);
        dirs[c * 3] = d[0];
        dirs[c * 3 + 1] = d[1];
        dirs[c * 3 + 2] = d[2];
      }
    }
  }

  const step = 2 / res;
  for (let f = 0; f < 6; f++) {
    for (let j = 0; j < res; j++) {
      for (let i = 0; i < res; i++) {
        const c = (f * res + j) * res + i;
        const [u, v] = cellUV(res, i, j);
        const d: V3 = [dirs[c * 3], dirs[c * 3 + 1], dirs[c * 3 + 2]];

        for (let k = 0; k < 8; k++) {
          const [di, dj] = OFFSETS[k];
          const nu = u + di * step;
          const nv = v + dj * step;
          let ni: number;

          if (nu >= -1 && nu <= 1 && nv >= -1 && nv <= 1) {
            ni = (f * res + (j + dj)) * res + (i + di);
          } else {
            // Off the edge: the cube point is still well defined, so let the
            // geometry say which face and cell it lands in.
            const nd = dirOf(f, nu, nv);
            ni = cellAt({ res } as unknown as Grid, nd);
          }

          nbr[c * 8 + k] = ni;
          const nd: V3 = [dirs[ni * 3], dirs[ni * 3 + 1], dirs[ni * 3 + 2]];
          const dot = Math.max(-1, Math.min(1, d[0] * nd[0] + d[1] * nd[1] + d[2] * nd[2]));
          // Great-circle distance, not the uv step: the tangent warp makes
          // cells near a face centre wider than cells near its edge, and using
          // a constant would bias every slope the flow router computes.
          nbrDist[c * 8 + k] = Math.acos(dot) * 6_371_000;
        }
      }
    }
  }

  let minSpacing = Infinity;
  for (let c = 0; c < count; c++) {
    for (const k of [1, 3, 4, 6]) {
      const d = nbrDist[c * 8 + k];
      if (d > 0 && d < minSpacing) minSpacing = d;
    }
  }

  return { res, count, spacing, minSpacing, dirs, nbr, nbrDist };
}

/** Cell direction, written into `out` to avoid allocating in hot loops. */
export function cellDir(grid: Grid, c: number, out: V3): V3 {
  out[0] = grid.dirs[c * 3];
  out[1] = grid.dirs[c * 3 + 1];
  out[2] = grid.dirs[c * 3 + 2];
  return out;
}

export { QUARTER_PI };
