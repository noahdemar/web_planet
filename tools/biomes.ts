/**
 * What fraction of this planet's land each biome covers, against Earth.
 *
 * The counterpart to tools/mirror.ts: that one checks the *shape* of the land
 * against Earth's hypsometric curve, this one checks what grows on it. Both
 * exist for the same reason — a planet can look plausible in every screenshot
 * and still have twice as much desert as Earth, and no amount of flying around
 * will tell you.
 *
 * Areas are equal-weighted because the sample set is a Fibonacci sphere, which
 * is equal-area by construction.
 *
 *   npm run biomes
 */

import { readFileSync } from 'node:fs';
import { surfaceFromBuffer, sampleSurface, type Meta } from '../src/planetData.ts';
import { heightAt, setPlanetSurface } from '../src/heightCPU.ts';
import { BIOMES, classify, climateAt, tempAtCPU } from '../src/biome.ts';
import { DEFAULT_OCTAVES } from '../src/planet.ts';
import type { V3 } from '../src/math/vec3d.ts';

const meta: Meta = JSON.parse(readFileSync('public/planet/meta.json', 'utf8'));
const raw = readFileSync('public/planet/surface.f16');
const surface = surfaceFromBuffer(
  meta,
  raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength) as ArrayBuffer,
);
setPlanetSurface(surface);

const N = Number(process.argv[2] ?? 400_000);

const count = new Array<number>(BIOMES.length).fill(0);
let land = 0;
let unclaimed = 0;

for (let i = 0; i < N; i++) {
  const z = 1 - (2 * i + 1) / N;
  const r = Math.sqrt(Math.max(0, 1 - z * z));
  const th = i * 2.399963229728653;
  const d: V3 = [r * Math.cos(th), r * Math.sin(th), z];

  const bake = sampleSurface(surface, d[0], d[1], d[2]);
  // Land is what the renderer draws as land: the solid surface above the
  // waterline, not the bake's own sign.
  const h = heightAt(d, DEFAULT_OCTAVES, 1, 1e9, false);
  if (h <= 0) continue;
  land++;

  const c = climateAt(d, bake.wetness);
  const idx = classify({
    temp: tempAtCPU(c.temp, h),
    base: c.temp,
    moist: c.moist,
    season: c.season,
    elevation: h,
  });
  if (idx < 0) unclaimed++;
  else count[idx]++;
}

const pc = (v: number) => `${(v * 100).toFixed(1)}%`;
const bar = (v: number, t: number) => {
  const n = Math.round(v * 100);
  const m = Math.round(t * 100);
  let s = '';
  for (let i = 0; i < Math.max(n, m, 1); i++) s += i < Math.min(n, m) ? '█' : i < n ? '+' : '·';
  return s;
};

console.log(`\nland samples ${land} of ${N} (${pc(land / N)} of the sphere)\n`);
console.log('biome                            here   Earth    delta');
console.log('─────────────────────────────────────────────────────────────');

// Earth's ten biomes plus permanent ice are ~90% of its land; the rest is bare
// rock, wetland and open water, which this classifier does not model and the
// renderer draws as overlays on top of a biome. Comparing raw shares against a
// classifier that claims ~100% would build in a fixed 10-point error, so both
// sides are normalised to the set actually being modelled.
const totalTarget = BIOMES.reduce((a, b) => a + b.target, 0);

let l1 = 0;
const order = BIOMES.map((b, i) => ({ b, i })).sort((a, c) => c.b.target - a.b.target);
for (const { b, i } of order) {
  const got = count[i] / land;
  const want = b.target / totalTarget;
  l1 += Math.abs(got - want);
  const delta = (got - want) * 100;
  const sign = delta >= 0 ? '+' : '−';
  console.log(
    `${b.name.padEnd(30)} ${pc(got).padStart(6)} ${pc(want).padStart(7)}  ` +
      `${sign}${Math.abs(delta).toFixed(1).padStart(4)}  ${bar(got, want)}`,
  );
}
console.log('─────────────────────────────────────────────────────────────');
console.log(`${'unclaimed (bare rock, water)'.padEnd(30)} ${pc(unclaimed / land).padStart(6)}`);
console.log(`\ntotal absolute error  ${(l1 * 100).toFixed(1)} points\n`);

// The target is not a hard test — the planet is not Earth and is not meant to
// be. It is a drift alarm: this ran at 18.7 points before the classifier was
// fitted, and a change that pushes it back there has broken the distribution
// whatever it did for one screenshot.
if (l1 * 100 > 30) {
  throw new Error(
    `biome distribution is ${(l1 * 100).toFixed(1)} points from Earth's — ` +
      'something has moved a climate threshold',
  );
}
console.log('biomes ok');
