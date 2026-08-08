/**
 * Hypsometry calibration and check. SPEC.md §13 asks for "realistic" to be a
 * measured target rather than an eyeball loop — this is the first measurement.
 *
 *   npm run hypsometry            check the committed parameters
 *   npm run hypsometry -- --solve search for better ones
 *
 * Components are cached, so the parameter search is pure arithmetic over
 * 60k pre-evaluated samples and runs in about a second.
 */

import {
  type Components,
  type ReliefParams,
  DEFAULT_RELIEF,
  componentsAt,
  compose,
} from '../src/heightCPU.ts';
import { DEFAULT_OCTAVES } from '../src/planet.ts';
import type { V3 } from '../src/math/vec3d.ts';

const N = 60_000;
const OCTAVES = DEFAULT_OCTAVES;

/** Earth reference values. */
const EARTH = {
  landFraction: 0.292,
  landMedian: 797,
  landP90: 2200,
  landP99: 4600,
  landAbove3100: 0.049,
  oceanMedian: -3700,
  median: -2440,
};

/** Fibonacci sphere — near-uniform in area, and deterministic. */
function samples(n: number): V3[] {
  const out: V3[] = [];
  for (let i = 0; i < n; i++) {
    const z = 1 - (2 * i + 1) / n;
    const r = Math.sqrt(Math.max(0, 1 - z * z));
    const th = i * 2.399963229728653; // golden angle
    out.push([r * Math.cos(th), r * Math.sin(th), z]);
  }
  return out;
}

const comps: Components[] = samples(N).map((p) => componentsAt(p, OCTAVES));

const q = (a: number[], p: number) => a[Math.floor(p * (a.length - 1))];

interface Report {
  landFraction: number;
  median: number;
  landMedian: number;
  landP90: number;
  landP99: number;
  landAbove3100: number;
  oceanMedian: number;
  min: number;
  max: number;
}

function evaluate(p: ReliefParams): Report {
  const h = comps.map((c) => compose(c, p)).sort((a, b) => a - b);
  const land = h.filter((x) => x > 0);
  const ocean = h.filter((x) => x <= 0);
  return {
    landFraction: land.length / N,
    median: q(h, 0.5),
    landMedian: land.length ? q(land, 0.5) : 0,
    landP90: land.length ? q(land, 0.9) : 0,
    landP99: land.length ? q(land, 0.99) : 0,
    landAbove3100: land.length ? land.filter((x) => x > 3100).length / land.length : 0,
    oceanMedian: ocean.length ? q(ocean, 0.5) : 0,
    min: h[0],
    max: h[N - 1],
  };
}

/** Bisect sea level for the target land fraction — monotone decreasing. */
function solveSeaLevel(p: ReliefParams): number {
  let lo = -1;
  let hi = 1;
  for (let i = 0; i < 44; i++) {
    const mid = (lo + hi) / 2;
    const cand = { ...p, seaLevel: mid };
    let n = 0;
    for (const c of comps) if (compose(c, cand) > 0) n++;
    if (n / N > EARTH.landFraction) lo = mid;
    else hi = mid;
  }
  return +((lo + hi) / 2).toFixed(4);
}

/** Relative error against the land quantiles that define the curve's shape. */
function cost(r: Report): number {
  const rel = (a: number, b: number) => Math.abs(a - b) / Math.abs(b);
  return (
    rel(r.landMedian, EARTH.landMedian) * 1.5 +
    rel(r.landP90, EARTH.landP90) +
    rel(r.landP99, EARTH.landP99) +
    rel(r.oceanMedian, EARTH.oceanMedian) +
    Math.abs(r.landAbove3100 - EARTH.landAbove3100) * 6
  );
}

function print(label: string, p: ReliefParams, r: Report): void {
  const m = (x: number) => `${Math.round(x)} m`.padStart(9);
  const pc = (x: number) => `${(x * 100).toFixed(1)}%`.padStart(9);
  console.log(`
  ${label}
  seaLevel ${p.seaLevel}   base ${p.base}   peak ${p.peak}   power ${p.power}   oceanDepth ${p.oceanDepth}
  (SEA_BAND ${p.band}, ${OCTAVES} octaves, ${N.toLocaleString()} samples)

                              this planet       Earth
  land fraction              ${pc(r.landFraction)}      ${pc(EARTH.landFraction)}
  median elevation           ${m(r.median)}   ${m(EARTH.median)}
  min / max                  ${m(r.min)} /${m(r.max)}
  land median                ${m(r.landMedian)}   ${m(EARTH.landMedian)}
  land p90                   ${m(r.landP90)}   ${m(EARTH.landP90)}
  land p99                   ${m(r.landP99)}   ${m(EARTH.landP99)}
  land above 3100 m          ${pc(r.landAbove3100)}      ${pc(EARTH.landAbove3100)}
  ocean median               ${m(r.oceanMedian)}   ${m(EARTH.oceanMedian)}
  fit cost                   ${cost(r).toFixed(4)}`);
}

const current = { ...DEFAULT_RELIEF };
print('committed parameters', current, evaluate(current));

if (process.argv.includes('--solve')) {
  let best: { p: ReliefParams; r: Report; c: number } | null = null;

  for (const power of [3, 4, 5, 6, 7]) {
    for (const peak of [4000, 5000, 6000, 7000, 8000, 9000, 11000, 14000]) {
      for (const base of [700, 900, 1150, 1400]) {
        for (const oceanDepth of [-3600, -4200, -4800, -5400]) {
          const p: ReliefParams = { ...DEFAULT_RELIEF, power, peak, base, oceanDepth };
          // Sea level is not free: it is pinned by the land-fraction target.
          p.seaLevel = solveSeaLevel(p);
          const r = evaluate(p);
          const c = cost(r);
          if (!best || c < best.c) best = { p, r, c };
        }
      }
    }
  }

  if (best) {
    print('best found', best.p, best.r);
    console.log(`
  → planet.ts
      export const SEA_LEVEL   = ${best.p.seaLevel};
      export const RELIEF_BASE = ${best.p.base};
      export const RELIEF_PEAK = ${best.p.peak};
      export const RELIEF_POWER = ${best.p.power};
      export const OCEAN_DEPTH = ${best.p.oceanDepth};
`);
  }
} else {
  console.log('\n  (pass --solve to search for better parameters)\n');
}
