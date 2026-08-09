/**
 * Measure the mean of the ridged octave term (1 − |noise|)².
 *
 * The amplification in src/shaders/terrain.ts subtracts this per octave. If
 * the constant is wrong, every octave that fades in with distance adds a small
 * DC offset, and since octaves switch on as you approach, the ground rises as
 * you fly toward it — which near sea level moves the coastline. That is
 * exactly the class of error SPEC.md I2 exists to forbid, and it is invisible
 * in a screenshot.
 *
 * It samples src/shaderNoiseCPU.ts, which is the exact mirror of the shader's
 * PCG3D noise. heightCPU.ts has a *different* hash, used only by the bake, and
 * a mean measured from that one would look rigorous and be wrong.
 *
 *   npm run ridgemean
 */

import { shaderNoise as noise } from '../src/shaderNoiseCPU.ts';

const N = 4_000_000;
let sum = 0;
let sumSq = 0;
// A low-discrepancy walk rather than Math.random: deterministic, and it does
// not cluster, so the estimate converges faster and is reproducible.
const P1 = 0.7548776662466927;
const P2 = 0.5698402909980532;
const P3 = 0.4301597090019478;
for (let i = 1; i <= N; i++) {
  const x = ((i * P1) % 1) * 512 - 256;
  const y = ((i * P2) % 1) * 512 - 256;
  const z = ((i * P3) % 1) * 512 - 256;
  const r = 1 - Math.abs(noise(x, y, z));
  const v = r * r;
  sum += v;
  sumSq += v * v;
}

const mean = sum / N;
const sd = Math.sqrt(sumSq / N - mean * mean);
const stderr = sd / Math.sqrt(N);

process.stdout.write(
  `E[(1 - |noise|)²] = ${mean.toFixed(5)}  ±${(2 * stderr).toFixed(5)} (95%)   ` +
    `over ${(N / 1e6).toFixed(0)} M samples\n` +
    `set RIDGE_MEAN in src/shaders/terrain.ts to ${mean.toFixed(4)}\n`,
);
