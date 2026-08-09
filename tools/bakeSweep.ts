/**
 * Sweep the bake's tectonic parameters against Earth's hypsographic curve.
 *
 * The curve error is a single scalar — the fraction of the planet's surface
 * sitting in the wrong elevation band — so the parameters can simply be
 * searched. This is the same reason tools/hypsometry.ts has a solver: hand
 * tuning a coupled tectonics-plus-erosion model by looking at renders is how
 * you end up with a planet that is 38% continental shelf and nobody notices.
 *
 *   npm run bake:sweep
 *   npm run bake:sweep -- --res 160 --stage lem
 */

import { buildGrid } from '../src/bake/grid.ts';
import { buildTectonics, DEFAULT_TECTONICS } from '../src/bake/plates.ts';
import { cellAreas, runLEM, DEFAULT_LEM } from '../src/bake/lem.ts';

const argv = process.argv.slice(2);
const arg = (k: string, d: number): number => {
  const i = argv.indexOf(`--${k}`);
  return i >= 0 ? Number(argv[i + 1]) : d;
};
const str = (k: string, d: string): string => {
  const i = argv.indexOf(`--${k}`);
  return i >= 0 ? argv[i + 1] : d;
};

const RES = arg('res', 128);
const STAGE = str('stage', 'tectonics');

const EARTH_CURVE: readonly (readonly [number, number, number])[] = [
  [5000, Infinity, 0.001], [4000, 5000, 0.005], [3000, 4000, 0.011],
  [2000, 3000, 0.022], [1000, 2000, 0.048], [500, 1000, 0.058],
  [200, 500, 0.060], [0, 200, 0.087], [-200, 0, 0.053], [-1000, -200, 0.031],
  [-2000, -1000, 0.030], [-3000, -2000, 0.050], [-4000, -3000, 0.121],
  [-5000, -4000, 0.216], [-6000, -5000, 0.184], [-Infinity, -6000, 0.023],
];

const grid = buildGrid(RES);
const areas = cellAreas(grid);
let total = 0;
for (let c = 0; c < grid.count; c++) total += areas[c];

/**
 * Half the L1 distance: the fraction of the surface in the wrong band.
 *
 * `oceanOnly` matters. Before the LEM runs, all land is a flat 190 m plateau,
 * so every land band is maximally wrong and the total error is dominated by
 * something tectonics does not control. Scoring the sub-sea bands alone
 * isolates what this stage actually decides — bathymetry — which is the point
 * of sweeping it separately from the erosion parameters.
 */
function curveError(z: Float32Array, oceanOnly: boolean): number {
  let e = 0;
  let mass = 0;
  for (const [lo, hi, want] of EARTH_CURVE) {
    if (oceanOnly && lo >= 0) continue;
    mass += want;
    let a = 0;
    for (let c = 0; c < grid.count; c++) if (z[c] > lo && z[c] <= hi) a += areas[c];
    e += Math.abs(a / total - want);
  }
  return e / 2 / mass;
}

interface Trial {
  plates: number;
  plateSpeed: number;
  continentalFraction: number;
  err: number;
  land: number;
  oceanMedian: number;
  high: number;
}

const results: Trial[] = [];
const plateSet = [15, 20, 26, 32];
const speedSet = [0.06, 0.09, 0.13, 0.18];
const contSet = [0.42, 0.48];

for (const plates of plateSet) {
  for (const plateSpeed of speedSet) {
    for (const continentalFraction of contSet) {
      const tec = buildTectonics(grid, {
        ...DEFAULT_TECTONICS,
        plates,
        plateSpeed,
        continentalFraction,
      });
      // `tectonics` scores the pre-erosion surface, which is 40× faster and
      // is where every bathymetric term is decided anyway. `lem` scores the
      // real output — use it to confirm the winner, not to search.
      const z =
        STAGE === 'lem'
          ? runLEM(grid, tec.base, tec.uplift, tec.erodibility, DEFAULT_LEM, 0).z
          : tec.base;

      const err = curveError(z, STAGE !== 'lem');
      let landA = 0;
      const ocean: number[] = [];
      let high = -Infinity;
      for (let c = 0; c < grid.count; c++) {
        if (z[c] > 0) landA += areas[c];
        else ocean.push(z[c]);
        if (z[c] > high) high = z[c];
      }
      ocean.sort((a, b) => a - b);
      results.push({
        plates,
        plateSpeed,
        continentalFraction,
        err,
        land: landA / total,
        oceanMedian: ocean[Math.floor(ocean.length / 2)],
        high,
      });
      process.stdout.write(
        `  plates ${String(plates).padStart(2)}  speed ${plateSpeed.toFixed(3)}  ` +
          `cont ${continentalFraction.toFixed(2)}  →  err ${(err * 100).toFixed(1)}%  ` +
          `land ${(results.at(-1)!.land * 100).toFixed(1)}%  ` +
          `oceanMed ${results.at(-1)!.oceanMedian.toFixed(0)} m  ` +
          `high ${high.toFixed(0)} m\n`,
      );
    }
  }
}

results.sort((a, b) => a.err - b.err);
process.stdout.write(`\nbest (${STAGE}, res ${RES}):\n`);
for (const r of results.slice(0, 5)) {
  process.stdout.write(
    `  err ${(r.err * 100).toFixed(1)}%   plates ${r.plates}  speed ${r.plateSpeed}  ` +
      `cont ${r.continentalFraction}   land ${(r.land * 100).toFixed(1)}%  ` +
      `oceanMed ${r.oceanMedian.toFixed(0)} m\n`,
  );
}
