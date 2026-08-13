/**
 * The bake, in a Web Worker.
 *
 * Nothing in src/bake/ ever touched Node — its only external import is three,
 * for the half-float conversion — so the solver runs here unchanged. What was
 * offline was the *file*, not the computation.
 *
 * It has to be a worker rather than an await on the main thread: this is one
 * uninterrupted minute of arithmetic, and on the main thread that is a minute
 * of frozen tab with no way to draw a progress bar. `bake` already takes an
 * onProgress callback, which is the hook this was always going to need.
 *
 * The atlas goes back as a transferable, so the 12 MB is moved rather than
 * copied and the main thread can hand it straight to a DataTexture.
 */

import { bake, DEFAULT_BAKE } from './index.js';
import { toAtlas, ATLAS_PAD } from './cubemap.js';
import { BAKE_RES } from '../planet.js';

export interface BakeRequest {
  seed: number;
}

export type BakeMessage =
  | { kind: 'progress'; stage: string; t: number }
  | { kind: 'done'; meta: unknown; data: Uint16Array }
  | { kind: 'error'; message: string };

self.onmessage = (e: MessageEvent<BakeRequest>) => {
  const post = (m: BakeMessage, transfer?: Transferable[]): void => {
    (self as unknown as Worker).postMessage(m, transfer ?? []);
  };
  try {
    const params = {
      ...DEFAULT_BAKE,
      res: BAKE_RES,
      tectonics: { ...DEFAULT_BAKE.tectonics, seed: e.data.seed },
    };

    // Throttled: the solver calls this every LEM step and postMessage is not
    // free. A percent is as fine as a progress bar can usefully be.
    let last = -1;
    const out = bake(params, (stage, t) => {
      const pc = Math.floor(t * 100);
      if (pc !== last || t === 1) {
        last = pc;
        post({ kind: 'progress', stage, t });
      }
    });

    const cube = toAtlas(out.grid, out.elevation, out.wetness, out.lakeDepth, out.channelDist);
    post(
      {
        kind: 'done',
        meta: {
          size: cube.faceSize,
          width: cube.width,
          height: cube.height,
          pad: ATLAS_PAD,
          layout: '3x2 cube atlas, one-texel border per face',
          channels: ['elevation_m', 'wetness_log10_area', 'lake_depth_m', 'channel_dist_m'],
          format: 'rgba16float',
          solveRes: params.res,
          seed: params.tectonics.seed,
        },
        data: cube.data,
      },
      [cube.data.buffer],
    );
  } catch (err) {
    post({ kind: 'error', message: err instanceof Error ? err.message : String(err) });
  }
};
