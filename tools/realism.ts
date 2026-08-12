/**
 * The fast realism gate.
 *
 * mirror.ts, slopes.ts and biomes.ts each answer one deep question and take
 * their time about it. This answers a shallower one across the whole planet
 * and is meant to be run every time: *has anything moved?* It walks the fixed
 * site table in src/tour.ts, measures the handful of numbers that describe
 * what the ground looks like at each, and diffs them against a stored
 * baseline.
 *
 * What it is for is regressions, not absolute truth. A tolerance here does not
 * say "this is realistic", it says "this is what it was when a human last
 * looked and agreed it was" — which is the only claim a number can honestly
 * make about realism, and the reason the visual half (sim.tour) exists beside
 * it.
 *
 *   npm run realism            check against the baseline
 *   npm run realism -- --update   accept the current numbers as the baseline
 *
 * Deliberately CPU-only. It runs the same heightAt the camera stands on, so
 * it catches anything in the height field, the amplification spectrum, the
 * bake sampling or the climate — everything except shading, which is what
 * sim.tour is for.
 */

import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import { surfaceFromBuffer, sampleSurface, type Meta } from '../src/planetData.ts';
import { heightAt, setPlanetSurface } from '../src/heightCPU.ts';
import { BIOMES, classify, climateAt, spectrumAt, tempAtCPU } from '../src/biome.ts';
import { DEFAULT_OCTAVES, RADIUS } from '../src/planet.ts';
import { SITES } from '../src/tour.ts';
import type { V3 } from '../src/math/vec3d.ts';

const BASELINE = 'tools/realism.baseline.json';
const update = process.argv.includes('--update');

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

function tangents(d: V3): [V3, V3] {
  const ax: V3 = Math.abs(d[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
  const p = ax[0] * d[0] + ax[1] * d[1] + ax[2] * d[2];
  const t1 = norm([ax[0] - d[0] * p, ax[1] - d[1] * p, ax[2] - d[2] * p]);
  const t2 = norm([
    d[1] * t1[2] - d[2] * t1[1],
    d[2] * t1[0] - d[0] * t1[2],
    d[0] * t1[1] - d[1] * t1[0],
  ]);
  return [t1, t2];
}

const step = (d: V3, t: V3, s: number): V3 =>
  norm([d[0] + t[0] * s, d[1] + t[1] * s, d[2] + t[2] * s]);

const median = (a: number[]): number => {
  const s = [...a].sort((x, y) => x - y);
  return s[s.length >> 1];
};

/**
 * The scales the slope is sampled at.
 *
 * Three decades, because the diagnostic is the *ratio* between them — that is
 * what the per-biome spectrum controls, and a single scale cannot see it.
 */
const SCALES = [300, 30, 3];

/** A small patch of ground around a site, so one unlucky lattice cell cannot move a number. */
const PATCH = 32;
const PATCH_SPREAD = 900; // metres

interface SiteStats {
  ground: number;
  /** Median slope at each of SCALES, as a gradient. */
  slope: number[];
  /** Slope growth from the coarsest scale to the finest — the spectral signature. */
  growth: number;
  /** Peak-to-trough relief over a 5 km transect, metres. */
  relief5k: number;
  /** Blended fBm gain and crossover octave the biome table asks for here. */
  spectrum: number[];
}

function measure(site: (typeof SITES)[number]): SiteStats {
  const [a1, a2] = tangents(site.dir);
  // A ring of sample points around the site rather than the point itself.
  const pts: V3[] = [];
  for (let i = 0; i < PATCH; i++) {
    const th = (i / PATCH) * Math.PI * 2;
    const r = (PATCH_SPREAD * (0.35 + 0.65 * ((i * 7) % PATCH) / PATCH)) / RADIUS;
    pts.push(
      step(step(site.dir, a1, Math.cos(th) * r), a2, Math.sin(th) * r),
    );
  }

  const slope = SCALES.map((s) => {
    const out: number[] = [];
    for (const d of pts) {
      const h0 = heightAt(d, DEFAULT_OCTAVES);
      for (const t of tangents(d)) {
        out.push(Math.abs(heightAt(step(d, t, s / RADIUS), DEFAULT_OCTAVES) - h0) / s);
      }
    }
    return median(out);
  });

  // Relief across 5 km, which is the scale a landscape reads at from the air.
  const hs: number[] = [];
  for (let i = 0; i < 60; i++) {
    hs.push(heightAt(step(site.dir, a1, (i * (5000 / 59)) / RADIUS), DEFAULT_OCTAVES));
  }

  const bake = sampleSurface(surface, site.dir[0], site.dir[1], site.dir[2]);
  const h = heightAt(site.dir, DEFAULT_OCTAVES);
  const c = climateAt(site.dir, bake.wetness);
  const spec = spectrumAt({
    temp: tempAtCPU(c.temp, bake.elevation),
    base: c.temp,
    moist: c.moist,
    season: c.season,
    elevation: bake.elevation,
  });

  return {
    ground: h,
    slope,
    growth: slope[2] / Math.max(slope[0], 1e-9),
    relief5k: Math.max(...hs) - Math.min(...hs),
    spectrum: spec,
  };
}

/* ── planet-wide invariants, cheap versions of what mirror/biomes do fully ── */

function global(): Record<string, number> {
  // Sized to the three-minute budget rather than the minimum that works: the
  // whole run is a few seconds, and a land fraction measured on 30k samples
  // wobbles by more than the tolerance it is checked against.
  const N = 250_000;
  let land = 0;
  let ampSum = 0;
  const counts = new Array<number>(BIOMES.length).fill(0);
  let claimed = 0;
  for (let i = 0; i < N; i++) {
    const z = 1 - (2 * i + 1) / N;
    const r = Math.sqrt(Math.max(0, 1 - z * z));
    const th = i * 2.399963229728653;
    const d: V3 = [r * Math.cos(th), r * Math.sin(th), z];
    const bake = sampleSurface(surface, d[0], d[1], d[2]);
    // Unflooded: the amplification's zero-mean property is a statement about
    // the solid surface, not about the height of the sea (LESSONS §14).
    const h = heightAt(d, DEFAULT_OCTAVES, 1, 1e9, false);
    ampSum += h - bake.elevation;
    if (h <= 0) continue;
    land++;
    const c = climateAt(d, bake.wetness);
    const idx = classify({
      temp: tempAtCPU(c.temp, h), base: c.temp, moist: c.moist,
      season: c.season, elevation: h,
    });
    if (idx >= 0) {
      counts[idx]++;
      claimed++;
    }
  }
  const out: Record<string, number> = {
    landFraction: land / N,
    amplificationMean: ampSum / N,
    unclaimed: 1 - claimed / land,
  };
  BIOMES.forEach((b, i) => {
    out[`biome.${b.key}`] = counts[i] / land;
  });
  return out;
}

/* ── run ──────────────────────────────────────────────────────────────────── */

const t0 = Date.now();
const sites: Record<string, SiteStats> = {};
for (const s of SITES) sites[s.key] = measure(s);
const globals = global();
const current = { sites, globals };

if (update || !existsSync(BASELINE)) {
  writeFileSync(BASELINE, `${JSON.stringify(current, null, 2)}\n`);
  console.log(`\n  baseline written to ${BASELINE} (${SITES.length} sites)`);
  console.log(`  ${((Date.now() - t0) / 1000).toFixed(1)} s\n`);
  process.exit(0);
}

const base = JSON.parse(readFileSync(BASELINE, 'utf8')) as typeof current;

/**
 * Tolerances.
 *
 * Wide enough that they are not a second implementation of the field, narrow
 * enough to catch a spectrum or an amplitude that has moved. `growth` is the
 * tight one on purpose: it is the number the per-biome ladder exists to
 * control, it is dimensionless, and it is the first thing a change to the
 * octave ladder moves.
 */
const TOL = {
  ground: 0.02,      // relative
  slope: 0.12,
  growth: 0.08,
  relief5k: 0.15,
  spectrum: 0.01,
  landFraction: 0.004, // absolute
  amplificationMean: 12, // metres, absolute — the zero-mean invariant
  biome: 0.012,      // absolute share of land
  unclaimed: 0.01,
};

const fails: string[] = [];
const rel = (a: number, b: number): number => Math.abs(a - b) / Math.max(Math.abs(b), 1e-6);

console.log(`\n  ${SITES.length} sites, ${DEFAULT_OCTAVES} octaves\n`);
console.log('  site                     ground   slope 300m/30m/3m   growth   5km relief');
console.log('  ' + '-'.repeat(76));
for (const s of SITES) {
  const c = sites[s.key];
  const b = base.sites[s.key];
  if (!b) {
    fails.push(`  ${s.key}: not in the baseline — re-run with --update if the table changed`);
    continue;
  }
  const bad: string[] = [];
  if (rel(c.ground, b.ground) > TOL.ground) bad.push(`ground ${b.ground.toFixed(0)}→${c.ground.toFixed(0)} m`);
  c.slope.forEach((v, i) => {
    if (rel(v, b.slope[i]) > TOL.slope) bad.push(`slope@${SCALES[i]}m ${b.slope[i].toFixed(4)}→${v.toFixed(4)}`);
  });
  if (rel(c.growth, b.growth) > TOL.growth) bad.push(`growth ${b.growth.toFixed(2)}→${c.growth.toFixed(2)}`);
  if (rel(c.relief5k, b.relief5k) > TOL.relief5k) bad.push(`5km relief ${b.relief5k.toFixed(0)}→${c.relief5k.toFixed(0)} m`);
  c.spectrum.forEach((v, i) => {
    if (Math.abs(v - b.spectrum[i]) > TOL.spectrum) bad.push(`spectrum[${i}] ${b.spectrum[i].toFixed(3)}→${v.toFixed(3)}`);
  });

  const deg = (g: number) => ((Math.atan(g) * 180) / Math.PI).toFixed(1);
  console.log(
    `  ${(bad.length ? '✗ ' : '  ') + s.key.padEnd(23)}${c.ground.toFixed(0).padStart(6)} m   ` +
      `${c.slope.map((v) => `${deg(v)}°`.padStart(6)).join(' ')}   ${c.growth.toFixed(2).padStart(5)}x  ` +
      `${c.relief5k.toFixed(0).padStart(7)} m`,
  );
  if (bad.length) fails.push(`  ${s.key}: ${bad.join(', ')}`);
}

console.log('\n  planet-wide');
console.log('  ' + '-'.repeat(76));
const gcheck = (k: string, tol: number, absolute: boolean, fmt: (v: number) => string) => {
  const c = globals[k];
  const b = base.globals[k];
  const off = absolute ? Math.abs(c - b) > tol : rel(c, b) > tol;
  console.log(`  ${(off ? '✗ ' : '  ') + k.padEnd(28)}${fmt(c).padStart(10)}   baseline ${fmt(b)}`);
  if (off) fails.push(`  ${k}: ${fmt(b)} → ${fmt(c)}`);
};
gcheck('landFraction', TOL.landFraction, true, (v) => `${(v * 100).toFixed(2)}%`);
gcheck('amplificationMean', TOL.amplificationMean, true, (v) => `${v.toFixed(1)} m`);
gcheck('unclaimed', TOL.unclaimed, true, (v) => `${(v * 100).toFixed(2)}%`);
for (const b of BIOMES) {
  gcheck(`biome.${b.key}`, TOL.biome, true, (v) => `${(v * 100).toFixed(2)}%`);
}

console.log(`\n  ${((Date.now() - t0) / 1000).toFixed(1)} s`);
if (fails.length) {
  console.error(`\n  ${fails.length} regression(s) against ${BASELINE}:\n`);
  console.error(fails.join('\n'));
  console.error('\n  If the change was intended: npm run realism -- --update\n');
  process.exitCode = 1;
} else {
  console.log('  realism ok\n');
}
