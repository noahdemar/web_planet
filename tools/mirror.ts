/**
 * Check the surface the renderer actually draws.
 *
 * tools/bake.ts measures the *bake*. This measures what comes out the other
 * end: the baked field resampled onto a cube map, decoded from half float,
 * bilinearly filtered, and with amplification added — the whole runtime path,
 * through the same code the browser runs.
 *
 * That distinction matters. Amplification adds up to ~1.4 km of relief on top
 * of the bake, and if its mean is not zero it moves the coastline and the
 * carefully fitted hypsographic curve stops describing the planet anyone sees.
 *
 *   npm run mirror
 */

import { readFileSync } from 'node:fs';
import { surfaceFromBuffer, sampleSurface, type Meta } from '../src/planetData.ts';
import { heightAt, setPlanetSurface } from '../src/heightCPU.ts';
import { DEFAULT_OCTAVES } from '../src/planet.ts';
import type { V3 } from '../src/math/vec3d.ts';

const meta: Meta = JSON.parse(readFileSync('public/planet/meta.json', 'utf8'));
const raw = readFileSync('public/planet/surface.f16');
const surface = surfaceFromBuffer(
  meta,
  raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength) as ArrayBuffer,
);
setPlanetSurface(surface);

const EARTH_CURVE: readonly (readonly [number, number, number])[] = [
  [5000, Infinity, 0.001], [4000, 5000, 0.005], [3000, 4000, 0.011],
  [2000, 3000, 0.022], [1000, 2000, 0.048], [500, 1000, 0.058],
  [200, 500, 0.060], [0, 200, 0.087], [-200, 0, 0.053], [-1000, -200, 0.031],
  [-2000, -1000, 0.030], [-3000, -2000, 0.050], [-4000, -3000, 0.121],
  [-5000, -4000, 0.216], [-6000, -5000, 0.184], [-Infinity, -6000, 0.023],
];

const N = 60_000;
const rendered: number[] = [];
const bakedOnly: number[] = [];
let ampAbs = 0;
let ampSum = 0;

for (let i = 0; i < N; i++) {
  const z = 1 - (2 * i + 1) / N;
  const r = Math.sqrt(Math.max(0, 1 - z * z));
  const th = i * 2.399963229728653;
  const d: V3 = [r * Math.cos(th), r * Math.sin(th), z];
  // Unflooded: the solid surface, which is what the hypsographic curve
  // describes and what the amplification acts on. Drawing the sea over it puts
  // 72% of the samples at exactly 0 m and makes both numbers meaningless — see
  // the `flood` note in heightCPU.ts.
  const h = heightAt(d, DEFAULT_OCTAVES, 1, 1e9, false);
  const b = sampleSurface(surface, d[0], d[1], d[2]).elevation;
  rendered.push(h);
  bakedOnly.push(b);
  ampSum += h - b;
  ampAbs += Math.abs(h - b);
}

const pc = (v: number) => `${(v * 100).toFixed(1).padStart(5)}%`;
const frac = (a: number[], lo: number, hi: number) =>
  a.filter((v) => v > lo && v <= hi).length / a.length;

let l1r = 0;
let l1b = 0;
const rows = EARTH_CURVE.map(([lo, hi, want]) => {
  const gr = frac(rendered, lo, hi);
  const gb = frac(bakedOnly, lo, hi);
  l1r += Math.abs(gr - want);
  l1b += Math.abs(gb - want);
  const label = `${lo === -Infinity ? '  <' : lo.toFixed(0).padStart(5)}…${hi === Infinity ? '' : hi.toFixed(0)}`;
  return `  ${label.padEnd(13)} ${pc(gr)} ${pc(gb)} ${pc(want)}`;
});

const sorted = [...rendered].sort((a, b) => a - b);
const q = (p: number) => sorted[Math.floor(p * (sorted.length - 1))];

process.stdout.write(`
what the renderer draws     rendered  baked  Earth
${rows.join('\n')}
  ---------------------------------------------------
  curve error (L1)         ${(l1r / 2 * 100).toFixed(1)}%   ${(l1b / 2 * 100).toFixed(1)}%

  land fraction            ${pc(rendered.filter((v) => v > 0).length / N)}   ${pc(bakedOnly.filter((v) => v > 0).length / N)}   ${pc(0.292)}
  amplification, mean      ${(ampSum / N).toFixed(1)} m   (must be near zero, or
                                     the coastline moves with distance)
  amplification, mean |Δ|  ${(ampAbs / N).toFixed(0)} m
  rendered p50 / p99 / max ${q(0.5).toFixed(0)} / ${q(0.99).toFixed(0)} / ${sorted[N - 1].toFixed(0)} m
`);

// ── face seams ────────────────────────────────────────────────────────────
// The atlas stores six faces side by side with a one-texel border each. If
// that border is wrong, the surface is discontinuous along a face edge — a
// straight-line artefact 1600 km long, which is easy to mistake for a rift
// valley when you see it from orbit and hard to be sure about by eye.
//
// So: walk each face boundary and compare the surface a hair either side.
// Across a working seam the difference is ordinary terrain variation over the
// step; across a broken one it is a cliff.
const STEP = 2e-5; // radians, ~130 m — far below one bake cell
const seamJumps: number[] = [];
const ctlJumps: number[] = [];

// The eight cube edges meet where two |components| are equal and largest.
for (let f = 0; f < 3; f++) {
  for (let sgn of [-1, 1]) {
    for (let k = 0; k < 500; k++) {
      const u = (k + 0.5) / 500 * 2 - 1;
      // A point on the edge between axis f and axis (f+1)%3.
      const e: number[] = [0, 0, 0];
      e[f] = 1;
      e[(f + 1) % 3] = sgn;
      e[(f + 2) % 3] = u;
      const L = Math.hypot(e[0], e[1], e[2]);
      const d: V3 = [e[0] / L, e[1] / L, e[2] / L];
      // Step across the edge, along the bisector of the two equal components.
      const n0: number[] = [0, 0, 0];
      n0[f] = 1;
      n0[(f + 1) % 3] = -sgn;
      const nl = Math.SQRT2;
      const t: V3 = [n0[0] / nl, n0[1] / nl, n0[2] / nl];
      const norm = (v: number[]): V3 => {
        const l = Math.hypot(v[0], v[1], v[2]);
        return [v[0] / l, v[1] / l, v[2] / l];
      };
      const a = norm([d[0] + t[0] * STEP, d[1] + t[1] * STEP, d[2] + t[2] * STEP]);
      const b = norm([d[0] - t[0] * STEP, d[1] - t[1] * STEP, d[2] - t[2] * STEP]);
      seamJumps.push(
        Math.abs(
          sampleSurface(surface, a[0], a[1], a[2]).elevation -
            sampleSurface(surface, b[0], b[1], b[2]).elevation,
        ),
      );

      // Control: the same step taken in the middle of a face, where there is
      // no seam. Whatever this reads is normal terrain variation, and the seam
      // number has to be comparable to it.
      const c = norm([d[0] + t[0] * 0.35, d[1] + t[1] * 0.35, d[2] + t[2] * 0.35]);
      const c2 = norm([c[0] + t[0] * STEP, c[1] + t[1] * STEP, c[2] + t[2] * STEP]);
      const c3 = norm([c[0] - t[0] * STEP, c[1] - t[1] * STEP, c[2] - t[2] * STEP]);
      ctlJumps.push(
        Math.abs(
          sampleSurface(surface, c2[0], c2[1], c2[2]).elevation -
            sampleSurface(surface, c3[0], c3[1], c3[2]).elevation,
        ),
      );
    }
  }
}

/**
 * Seam quality, judged by distribution rather than by the single worst sample.
 *
 * This used to compare worst-of-N across the seams against 6x worst-of-N in
 * mid-face. Both are extreme-value statistics over a rough field, so their
 * ratio is unstable — it sat at 5.6 against a threshold of 6, one unlucky
 * sample from failing, and a rougher bake duly failed it while the border was
 * demonstrably fine.
 *
 * The mean and p99 are stable. Measured across two bakes of very different
 * roughness, the seam runs 4.0–4.3x the mid-face control on the mean and
 * 5.4–5.6x on p99, and those ratios barely move even when the worst single
 * sample moves 60%. So ~5x is what a *working* one-texel border costs at this
 * step: the two faces' texel grids do not align, and the interpolant has a
 * small kink there.
 *
 * The thresholds sit at roughly 3x that baseline. A genuinely broken border is
 * not a subtle effect — clamping instead of reading the neighbour's texels
 * puts the full 4.5 km continental step along every edge, which is two orders
 * of magnitude outside these bounds.
 */
const stat = (a: number[]) => {
  const s = [...a].sort((x, y) => x - y);
  return {
    mean: a.reduce((x, y) => x + y, 0) / a.length,
    p99: s[Math.floor(0.99 * (s.length - 1))],
  };
};
const seam = stat(seamJumps);
const ctl = stat(ctlJumps);
process.stdout.write(
  `  face seams               mean ${seam.mean.toFixed(2)} m   p99 ${seam.p99.toFixed(1)} m\n` +
    `                           (mid-face control, same step: ` +
    `mean ${ctl.mean.toFixed(2)} m   p99 ${ctl.p99.toFixed(1)} m)\n`,
);
if (seam.mean > Math.max(13 * ctl.mean, 20) || seam.p99 > Math.max(16 * ctl.p99, 200)) {
  throw new Error(
    `face seams read ${seam.mean.toFixed(0)} m mean / ${seam.p99.toFixed(0)} m p99 against a ` +
      `mid-face control of ${ctl.mean.toFixed(0)} / ${ctl.p99.toFixed(0)} — atlas border is wrong`,
  );
}

// The amplification is zero-mean by construction (RIDGE_MEAN is subtracted per
// octave). If it drifts, octaves fading in with distance will raise the ground
// as the camera approaches — a violation of SPEC.md I2 that no screenshot shows.
const bias = Math.abs(ampSum / N);
if (bias > 25) {
  throw new Error(`amplification bias ${bias.toFixed(1)} m — check RIDGE_MEAN (npm run ridgemean)`);
}
if (Math.abs(l1r - l1b) / 2 > 0.05) {
  throw new Error('amplification moved the hypsographic curve by more than 5 points');
}
process.stdout.write('mirror ok\n');
