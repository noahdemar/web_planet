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
import {
  DEFAULT_CHANNELS,
  DEFAULT_LEM,
  accumulateMFD,
  carveChannels,
  cellAreas,
  runLEM,
  type LemParams,
} from './lem.js';
import { DEFAULT_RELIEF, addBaseRelief, type ReliefParams } from './relief.js';

import { BAKE_RES } from '../planet.js';
export { BAKE_RES };

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
  /**
   * Standing-water *depth*, metres — zero on dry ground.
   *
   * Depth rather than surface elevation because this is what gets bilinearly
   * magnified 11x at orbital range: interpolating a depth that falls to zero
   * gives a shoreline that ramps, while interpolating a surface elevation
   * against ground that is 200 m higher gives a step, and neither value is
   * meaningful in between.
   */
  lakeDepth: Float32Array;
  /**
   * Distance to the nearest channel or coast, metres, capped. The runtime
   * places rivers from this rather than reconstructing an axis — see
   * carveChannels.
   */
  channelDist: Float32Array;
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

  // Drainage area for display comes from MFD, not from the D8 tree the erosion
  // ran on: see accumulateMFD. `lem.area` is still the right input for stream
  // power and is what the LEM used; this is the field the renderer draws.
  // On lem.water, not lem.z: the basins are back in lem.z now (that is the
  // point — a lake is a basin, not a plateau), and flow cannot route out of a
  // depression. lem.water is the same surface with them filled, which is what
  // every cell downstream of a lake needs in order to receive its area at all.
  const mfd = accumulateMFD(grid, lem.water, cellAreas(grid));

  // Lake depth *before* the carve, and the order is not cosmetic. lakeDepth is
  // water level minus ground, and carveChannels lowers the ground by up to
  // CHANNEL_DEPTH along every channel — so computing it afterwards turned the
  // entire drainage network into a 60 m deep lake, which rendered as a system
  // of blue texel-shaped ponds following every valley.
  const lakeDepth = new Float32Array(grid.count);
  for (let c = 0; c < grid.count; c++) lakeDepth[c] = Math.max(0, lem.water[c] - lem.z[c]);

  // Then carve the network into the surface and measure the distance out from
  // it. After the LEM, so it cuts the channel the model actually routed rather
  // than one imposed on top of it.
  const channelDist = carveChannels(grid, lem.z, mfd, DEFAULT_CHANNELS);
  const wetness = new Float32Array(grid.count);
  for (let c = 0; c < grid.count; c++) {
    // log area, not area: drainage spans ten orders of magnitude and every
    // downstream consumer (river width, valley depth, moisture) is logarithmic
    // in it anyway.
    wetness[c] = Math.min(14, Math.log10(Math.max(1, mfd[c])));
  }
  onProgress?.('done', 1);

  return {
    res: params.res,
    elevation: lem.z,
    wetness,
    lakeDepth,
    channelDist,
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
