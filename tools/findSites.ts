/**
 * Reissue the site table in src/tour.ts against the current bake.
 *
 * The realism suite judges the planet at 24 fixed places — every biome at both
 * the flat and the steep end of its relief range, plus a shoreline and a trunk
 * valley floor. Those places are *data*, not something to be found by flying
 * around: a site chosen because it was memorable is chosen for the opposite of
 * being representative, and a baseline whose sites move with the thing they
 * measure is not a baseline at all.
 *
 * So they are picked once, deterministically, by sweeping a Fibonacci sphere
 * and taking the extremes of each class. Re-run this whenever the bake changes
 * — a new seed, a new resolution, a change to the LEM — because the biome
 * classification moves with it and a site labelled `desert-flat` may not be
 * desert any more. Then re-run `npm run realism -- --update`.
 *
 *   npm run sites
 */

import { readFileSync, writeFileSync } from 'node:fs';
import { surfaceFromBuffer, sampleSurface, type Meta } from '../src/planetData.ts';
import { heightAt, setPlanetSurface } from '../src/heightCPU.ts';
import { BIOMES, classify, climateAt, tempAtCPU } from '../src/biome.ts';
import {
  DEFAULT_OCTAVES, FACE_EDGE, RADIUS, RELIEF_SLOPE_HI, RELIEF_SLOPE_LO,
} from '../src/planet.ts';
import type { V3 } from '../src/math/vec3d.ts';

const TOUR = 'src/tour.ts';
const N = Number(process.argv[2] ?? 400_000);

const meta: Meta = JSON.parse(readFileSync('public/planet/meta.json', 'utf8'));
const raw = readFileSync('public/planet/surface.f16');
const surface = surfaceFromBuffer(
  meta,
  raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength) as ArrayBuffer,
);
setPlanetSurface(surface);

const norm = (v: V3): V3 => {
  const m = Math.hypot(v[0], v[1], v[2]);
  return [v[0] / m, v[1] / m, v[2] / m];
};
const sstep = (a: number, b: number, x: number): number => {
  const t = Math.max(0, Math.min(1, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
};

/** Baked slope, by the same one-cell central difference the shader uses. */
function bakedSlope(d: V3): number {
  const e = FACE_EDGE / surface.size / RADIUS;
  const ax: V3 = Math.abs(d[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
  const p = ax[0] * d[0] + ax[1] * d[1] + ax[2] * d[2];
  const t1 = norm([ax[0] - d[0] * p, ax[1] - d[1] * p, ax[2] - d[2] * p]);
  const t2 = norm([
    d[1] * t1[2] - d[2] * t1[1],
    d[2] * t1[0] - d[0] * t1[2],
    d[0] * t1[1] - d[1] * t1[0],
  ]);
  const at = (t: V3, s: number): number => {
    const q = norm([d[0] + t[0] * e * s, d[1] + t[1] * e * s, d[2] + t[2] * e * s]);
    return sampleSurface(surface, q[0], q[1], q[2]).elevation;
  };
  return Math.hypot((at(t1, 1) - at(t1, -1)) / (2 * e), (at(t2, 1) - at(t2, -1)) / (2 * e)) / RADIUS;
}

interface Cand { d: V3; h: number; relief: number; wet: number }

const flat = new Map<string, Cand>();
const steep = new Map<string, Cand>();
let coast: Cand | null = null;
let valley: Cand | null = null;

for (let i = 0; i < N; i++) {
  const z = 1 - (2 * i + 1) / N;
  const r = Math.sqrt(Math.max(0, 1 - z * z));
  const th = i * 2.399963229728653;
  const d: V3 = [r * Math.cos(th), r * Math.sin(th), z];
  const bake = sampleSurface(surface, d[0], d[1], d[2]);
  const h = heightAt(d, DEFAULT_OCTAVES);
  if (h <= 0) continue;

  const relief = sstep(RELIEF_SLOPE_LO, RELIEF_SLOPE_HI, bakedSlope(d));
  const c = climateAt(d, bake.wetness);
  const idx = classify({
    temp: tempAtCPU(c.temp, h), base: c.temp, moist: c.moist,
    season: c.season, elevation: h,
  });
  if (idx < 0) continue;
  const key = BIOMES[idx].key;
  const cand: Cand = { d, h, relief, wet: bake.wetness };

  // The flat end: as close to no sub-grid relief as this biome gets, but still
  // properly inland, so the site is a plain and not a beach.
  if (relief < 0.08 && h > 60) {
    const prev = flat.get(key);
    if (!prev || Math.abs(relief - 0.02) < Math.abs(prev.relief - 0.02)) flat.set(key, cand);
  }
  // The steep end: fully saturated relief, and the highest such point, which is
  // where the amplification is working hardest.
  if (relief > 0.85) {
    const prev = steep.get(key);
    if (!prev || h > prev.h) steep.set(key, cand);
  }
  if (bake.elevation > 0 && bake.elevation < 40 && (!coast || bake.elevation < coast.h)) {
    coast = { ...cand, h: bake.elevation };
  }
  if (bake.wetness > 10.2 && relief < 0.35 && (!valley || bake.wetness > valley.wet)) {
    valley = cand;
  }
}

const fmt = (v: V3): string => `[${v.map((x) => x.toFixed(6)).join(', ')}]`;
const rows: string[] = [];
for (const b of BIOMES) {
  for (const [tag, m] of [['flat', flat], ['steep', steep]] as const) {
    const c = m.get(b.key);
    if (!c) {
      process.stderr.write(`  no ${tag} site for ${b.key} — it may not exist on this planet\n`);
      continue;
    }
    rows.push(
      `  { key: '${b.key}-${tag}', name: ${JSON.stringify(`${b.name} — ${tag}`)},\n` +
        `    dir: ${fmt(c.d)}, ground: ${Math.round(c.h)}, relief: ${c.relief.toFixed(2)} },`,
    );
  }
}
if (coast) {
  rows.push(`  { key: 'coast', name: 'Shoreline',\n    dir: ${fmt(coast.d)}, ground: ${Math.round(coast.h)}, relief: ${coast.relief.toFixed(2)} },`);
}
if (valley) {
  rows.push(`  { key: 'valley', name: 'Trunk valley floor',\n    dir: ${fmt(valley.d)}, ground: ${Math.round(valley.h)}, relief: ${valley.relief.toFixed(2)} },`);
}

const src = readFileSync(TOUR, 'utf8');
const open = 'export const SITES: readonly Site[] = [\n';
const i0 = src.indexOf(open);
if (i0 < 0) throw new Error(`${TOUR}: could not find the SITES array to rewrite`);
const i1 = src.indexOf('\n];', i0);
writeFileSync(TOUR, src.slice(0, i0 + open.length) + rows.join('\n') + src.slice(i1));

process.stdout.write(
  `\n  ${rows.length} sites written to ${TOUR} (seed ${meta.seed ?? 'unknown'}, ${N} samples)\n` +
    '  Now: npm run realism -- --update\n\n',
);
