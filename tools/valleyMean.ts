/**
 * Measure the mean of the sub-grid valley carve, per unit depth.
 *
 * The valley network in src/shaders/terrain.ts cuts trenches along the zero
 * set of a noise field. Carving only ever removes material, so the term has a
 * large positive mean — unlike the ridged octaves, whose mean is a
 * second-order correction, this one is the whole point of the term and lowers
 * the ground by a fixed fraction of its depth everywhere.
 *
 * That matters for exactly the reason RIDGE_MEAN matters, only more so. Levels
 * fade in as the mesh refines, so a level arriving with an unremoved mean drops
 * the ground as you approach — and near sea level, dropping the ground moves
 * the coastline. SPEC.md I2 forbids it, and it is invisible in a screenshot.
 *
 * The estimate is dimensionless. The trench profile is written against the
 * *noise argument*, where distance to the zero set is |n| / |∇n| and the half
 * width is VALLEY_WIDTH lattice cells, so one measurement serves every level.
 *
 * It samples src/shaderNoiseCPU.ts, the exact mirror of the shader's PCG3D
 * noise — the same reason ridgeMean.ts does, and the same trap if you reach
 * for heightCPU's hash instead.
 *
 *   npm run valleymean
 */

import { shaderNoiseD as noiseD } from '../src/shaderNoiseCPU.ts';
import { VALLEY_WIDTH } from '../src/planet.ts';

const N = 4_000_000;
let sum = 0;
let sumSq = 0;
// Same low-discrepancy walk as ridgeMean: deterministic and non-clustering.
const P1 = 0.7548776662466927;
const P2 = 0.5698402909980532;
const P3 = 0.4301597090019478;
for (let i = 1; i <= N; i++) {
  const x = ((i * P1) % 1) * 512 - 256;
  const y = ((i * P2) % 1) * 512 - 256;
  const z = ((i * P3) % 1) * 512 - 256;
  const nd = noiseD(x, y, z);
  const gl = Math.max(Math.hypot(nd[1], nd[2], nd[3]), 1e-6);
  // First-order distance to the zero set, in lattice cells.
  const dist = Math.abs(nd[0]) / gl;
  const t = dist / VALLEY_WIDTH;
  const v = Math.exp(-t * t);
  sum += v;
  sumSq += v * v;
}

const mean = sum / N;
const sd = Math.sqrt(sumSq / N - mean * mean);
const stderr = sd / Math.sqrt(N);

process.stdout.write(
  `E[exp(-(d/w)²)] = ${mean.toFixed(5)}  ±${(2 * stderr).toFixed(5)} (95%)   ` +
    `over ${(N / 1e6).toFixed(0)} M samples, VALLEY_WIDTH ${VALLEY_WIDTH}\n` +
    `areal coverage of the trenches: ${(100 * mean).toFixed(2)}%\n` +
    `set VALLEY_MEAN in src/planet.ts to ${mean.toFixed(4)}\n`,
);
