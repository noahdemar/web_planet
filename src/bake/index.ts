/**
 * M3 — the global bake. See SPEC.md §4 and §12.
 *
 * Runs once, produces the planet's structural layer, and hands the runtime two
 * fields per cube face:
 *
 *   elevation    metres relative to sea level
 *   wetness      log₁₀ of upstream drainage area — where the water is
 *
 * Everything the runtime draws below the bake's cell size is amplification on
 * top of these (SPEC.md §6). The point of baking is not to store detail: at
 * this resolution there is barely any. The point is that drainage is a global
 * property and *cannot* be recovered from a local function, so it is solved
 * once, globally, and then used as the boundary condition everywhere else.
 */

import { buildGrid, type Grid } from './grid.js';
import { DEFAULT_TECTONICS, buildTectonics, type TectonicsParams } from './plates.js';
import { DEFAULT_LEM, runLEM, type LemParams } from './lem.js';
import { DEFAULT_RELIEF, addBaseRelief, type ReliefParams } from './relief.js';

/** Cells per cube-face edge. 512 → 1.57 M cells at ~18 km spacing. */
export const BAKE_RES = 512;

export interface BakeParams {
  res: number;
  tectonics: TectonicsParams;
  relief: ReliefParams;
  lem: LemParams;
}

export const DEFAULT_BAKE: BakeParams = {
  res: BAKE_RES,
  tectonics: DEFAULT_TECTONICS,
  relief: DEFAULT_RELIEF,
  lem: DEFAULT_LEM,
};

export interface Baked {
  res: number;
  /** Metres relative to sea level, 6·res² in face-major order. */
  elevation: Float32Array;
  /** log₁₀(drainage area in m²), clamped to [0, 14]. */
  wetness: Float32Array;
  /** Kept for diagnostics and for the runtime's coastline logic. */
  continental: Uint8Array;
  timings: Record<string, number>;
}

export type Progress = (stage: string, t: number) => void;

export function bake(params = DEFAULT_BAKE, onProgress?: Progress): Baked {
  const t0 = Date.now();
  const grid: Grid = buildGrid(params.res);
  const tGrid = Date.now();
  onProgress?.('grid', 1);

  const tec = buildTectonics(grid, params.tectonics);
  addBaseRelief(grid, tec.base, tec.uplift, params.relief);
  const tTec = Date.now();
  onProgress?.('tectonics', 1);

  const lem = runLEM(grid, tec.base, tec.uplift, tec.erodibility, params.lem, 0, (t) =>
    onProgress?.('erosion', t),
  );
  const tLem = Date.now();

  const wetness = new Float32Array(grid.count);
  for (let c = 0; c < grid.count; c++) {
    // log area, not area: drainage spans ten orders of magnitude and every
    // downstream consumer (river width, valley depth, moisture) is logarithmic
    // in it anyway.
    wetness[c] = Math.min(14, Math.log10(Math.max(1, lem.area[c])));
  }
  onProgress?.('done', 1);

  return {
    res: params.res,
    elevation: lem.z,
    wetness,
    continental: tec.continental,
    timings: {
      grid: tGrid - t0,
      tectonics: tTec - tGrid,
      erosion: tLem - tTec,
      total: Date.now() - t0,
    },
  };
}

export { buildGrid, buildTectonics, runLEM };
export type { Grid };
