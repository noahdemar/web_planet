/**
 * Measure the slope of the surface the renderer actually draws.
 *
 * tools/mirror.ts checks the *elevation* distribution against Earth's. This
 * checks the derivative, which is what you actually look at: a hypsometric
 * curve can be perfect while the surface between those elevations is a field
 * of vertical facets, and that is exactly the state this tool was written to
 * catch.
 *
 * The diagnostic is not the absolute number, it is whether the number
 * *converges* as the sampling gets finer. An fBm with H < 1 has no slope
 * limit: amplitude falls as λ^H but the run falls as λ¹, so slope grows by
 * lacunarity^(1−H) with every octave and the terrain gets steeper without
 * bound the closer you look. Real landscapes flatten out below the hillslope
 * crossover instead, because material at more than the angle of repose does
 * not stay there. A run whose median slope keeps climbing to the finest row is
 * reporting a broken field however good the elevations look.
 *
 *   npm run slopes
 */

import { readFileSync } from 'node:fs';
import { surfaceFromBuffer, sampleSurface, type Meta } from '../src/planetData.ts';
import { heightAt, setPlanetSurface } from '../src/heightCPU.ts';
import {
  AMP_BASE,
  AMP_F0,
  AMP_RELIEF,
  DEFAULT_OCTAVES,
  FACE_EDGE,
  HILLSLOPE_GAIN,
  HILLSLOPE_SOFT,
  HILLSLOPE_WAVELENGTH,
  LOG2_LACUNARITY,
  RADIUS,
  RELIEF_GAIN,
  RELIEF_LACUNARITY,
  RELIEF_SLOPE_HI,
  RELIEF_SLOPE_LO,
  spectrumOctave,
} from '../src/planet.ts';
import { BIOMES, classify, climateAt, tempAtCPU } from '../src/biome.ts';
import type { V3 } from '../src/math/vec3d.ts';

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

/** A unit tangent at `d`, deterministic and free of the pole degeneracy. */
function tangent(d: V3): [V3, V3] {
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

/** Baked slope, by the same one-cell central difference the shader uses. */
function bakedSlope(d: V3): number {
  const e = FACE_EDGE / surface.size / RADIUS;
  const [t1, t2] = tangent(d);
  const at = (t: V3, sgn: number): number => {
    const q = step(d, t, e * sgn);
    return sampleSurface(surface, q[0], q[1], q[2]).elevation;
  };
  return Math.hypot((at(t1, 1) - at(t1, -1)) / (2 * e), (at(t2, 1) - at(t2, -1)) / (2 * e)) / RADIUS;
}

const pct = (sorted: number[], p: number): number => sorted[Math.floor(p * (sorted.length - 1))];
const deg = (s: number): string => `${((Math.atan(s) * 180) / Math.PI).toFixed(1)}°`;

/* ── the spectrum this field is asking for ──────────────────────────────── */

console.log(`AMP_F0 ${AMP_F0}  lacunarity ${RELIEF_LACUNARITY}  reference gain ${RELIEF_GAIN}`);
console.log(
  `octaves ${DEFAULT_OCTAVES} (derived)  reference crossover ${HILLSLOPE_WAVELENGTH} m` +
    ` (octave ${spectrumOctave(HILLSLOPE_WAVELENGTH).toFixed(2)}, soft +-${HILLSLOPE_SOFT})` +
    `  gain past it ${HILLSLOPE_GAIN.toFixed(4)}`,
);
console.log(`relief window [${RELIEF_SLOPE_LO}, ${RELIEF_SLOPE_HI}]  amp ${AMP_BASE}..${AMP_BASE + AMP_RELIEF} m\n`);

/**
 * The amplitude ladder for a spectrum, and the normaliser height_ divides by.
 *
 * Mirrors the loop in `height_`: the gain falls toward HILLSLOPE_GAIN across
 * the crossover, and the normaliser is sqrt(sum a^2) rescaled to the reference
 * ladder's own sum/l2 ratio so the detail's variance does not move with the
 * spectrum.
 */
function ladder(gain: number, lamHill: number): { amp: number[]; norm: number } {
  const walk = (g: number, cross: number): number[] => {
    const out: number[] = [];
    let amp = 1;
    let oct = cross;
    for (let i = 0; i < DEFAULT_OCTAVES; i++) {
      out.push(amp);
      const t = Math.max(0, Math.min(1, (oct + HILLSLOPE_SOFT) / (2 * HILLSLOPE_SOFT)));
      amp *= g + (HILLSLOPE_GAIN - g) * (t * t * (3 - 2 * t));
      oct += LOG2_LACUNARITY;
    }
    return out;
  };
  const l1 = (a: number[]) => a.reduce((s, v) => s + v, 0);
  const l2 = (a: number[]) => Math.sqrt(a.reduce((s, v) => s + v * v, 0));
  const amp = walk(gain, spectrumOctave(lamHill));
  const ref = walk(RELIEF_GAIN, spectrumOctave(HILLSLOPE_WAVELENGTH));
  return { amp, norm: (l2(amp) * l1(ref)) / l2(ref) };
}

{
  const { amp: amps, norm } = ladder(RELIEF_GAIN, HILLSLOPE_WAVELENGTH);
  let frq = AMP_F0;
  console.log('the reference ladder (temperate broadleaf; every other biome scales this)');
  console.log('oct         F  wavelength  amp@max     slope           cells/ULP');
  for (let i = 0; i < DEFAULT_OCTAVES; i++) {
    const lam = RADIUS / frq;
    const a = ((AMP_BASE + AMP_RELIEF) * amps[i]) / norm;
    const q = 1.19e-7 * frq;
    console.log(
      `${String(i).padStart(3)} ${frq.toExponential(2).padStart(9)} ` +
        `${(lam > 1000 ? `${(lam / 1000).toFixed(1)} km` : `${lam.toFixed(1)} m`).padStart(10)} ` +
        `${a.toFixed(1).padStart(9)} m ${(a / lam).toFixed(2).padStart(7)} ` +
        `${deg(a / lam).padStart(7)} ${q.toFixed(3).padStart(8)}` +
        `${lam <= HILLSLOPE_WAVELENGTH ? '  hillslope' : ''}`,
    );
    frq *= RELIEF_LACUNARITY;
  }
  console.log('');
}

/* ── the spectrum each biome asks for, before any of it touches the planet ── */

{
  console.log('per-biome spectrum, and what it does to the ladder:');
  console.log(
    '  biome                          gain     H  crossover' +
      '   slope@12km  slope@36m   relief',
  );
  const refL = ladder(RELIEF_GAIN, HILLSLOPE_WAVELENGTH);
  const slopeAt = (l: { amp: number[]; norm: number }, i: number): number => {
    const lam = RADIUS / (AMP_F0 * RELIEF_LACUNARITY ** i);
    return (l.amp[i] / l.norm / lam) * 1000 * 100;
  };
  for (const b of BIOMES) {
    const l = ladder(b.gain, b.hillslope);
    const H = Math.log(1 / b.gain) / Math.log(RELIEF_LACUNARITY);
    // Total variance of the ladder relative to the reference: this is the
    // number the l2 normaliser exists to hold at 1.000. If it drifts, the
    // table has become an amplitude control (see the gain field on Biome).
    const relief =
      Math.sqrt(l.amp.reduce((s, v) => s + v * v, 0)) / l.norm /
      (Math.sqrt(refL.amp.reduce((s, v) => s + v * v, 0)) / refL.norm);
    console.log(
      `  ${b.name.padEnd(30)} ${b.gain.toFixed(2)}  ${H.toFixed(2)}  ` +
        `${String(b.hillslope).padStart(5)} m  ${slopeAt(l, 0).toFixed(2).padStart(9)}  ` +
        `${slopeAt(l, DEFAULT_OCTAVES - 1).toFixed(2).padStart(9)}   ${relief.toFixed(3)}x`,
    );
  }
  console.log('  (slope columns are x100 per unit of delivered relief, at 1000 m of amplitude)\n');
}

/* ── what the bake's own slopes look like, i.e. what relief must be fitted to ─ */

const N_LAND = 60_000;
const land: { d: V3; slope: number }[] = [];
for (let i = 0; i < N_LAND; i++) {
  const z = 1 - (2 * i + 1) / N_LAND;
  const r = Math.sqrt(Math.max(0, 1 - z * z));
  const th = i * 2.399963229728653;
  const d: V3 = [r * Math.cos(th), r * Math.sin(th), z];
  if (sampleSurface(surface, d[0], d[1], d[2]).elevation <= 0) continue;
  land.push({ d, slope: bakedSlope(d) });
}
const bakedSorted = land.map((l) => l.slope).sort((a, b) => a - b);
const sstep = (e0: number, e1: number, x: number): number => {
  const t = Math.max(0, Math.min(1, (x - e0) / (e1 - e0)));
  return t * t * (3 - 2 * t);
};

console.log(`baked land slope, ${land.length} samples:`);
for (const p of [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]) {
  const s = pct(bakedSorted, p);
  console.log(
    `  p${String(Math.round(p * 100)).padStart(3)}  ${s.toFixed(5)}   ` +
      `relief -> ${sstep(RELIEF_SLOPE_LO, RELIEF_SLOPE_HI, s).toFixed(3)}`,
  );
}
const reliefOf = (s: number) => sstep(RELIEF_SLOPE_LO, RELIEF_SLOPE_HI, s);
console.log(
  `  land with relief < 0.05: ${((100 * bakedSorted.filter((s) => reliefOf(s) < 0.05).length) / bakedSorted.length).toFixed(1)}%` +
    `   > 0.95: ${((100 * bakedSorted.filter((s) => reliefOf(s) > 0.95).length) / bakedSorted.length).toFixed(1)}%\n`,
);

/* ── the rendered surface, sampled at a ladder of scales ─────────────────── */

const STEPS = [300, 100, 30, 10, 3];
const WALK = 160; // samples per transect

/** Slope percentiles along transects through a set of points. */
function profile(points: V3[], stepM: number): number[] {
  const out: number[] = [];
  for (const d of points) {
    const [t1] = tangent(d);
    let prev = heightAt(d, DEFAULT_OCTAVES);
    for (let i = 1; i < WALK; i++) {
      const q = step(d, t1, (i * stepM) / RADIUS);
      const h = heightAt(q, DEFAULT_OCTAVES);
      out.push(Math.abs(h - prev) / stepM);
      prev = h;
    }
  }
  return out.sort((a, b) => a - b);
}

/** Land split by how much sub-grid roughness the amplification will give it. */
const rugged = land.filter((l) => reliefOf(l.slope) > 0.5).map((l) => l.d);
const typical = land.filter((l) => reliefOf(l.slope) <= 0.5).map((l) => l.d);
const pick = (a: V3[], n: number): V3[] =>
  a.length <= n ? a : Array.from({ length: n }, (_, i) => a[Math.floor((i * a.length) / n)]);

for (const [name, pts] of [
  ['typical land', pick(typical, 24)],
  ['rugged land', pick(rugged, 24)],
] as const) {
  if (pts.length === 0) {
    console.log(`${name}: no samples\n`);
    continue;
  }
  console.log(`${name} (${pts.length} transects x ${WALK} samples):`);
  console.log('  sampling    median      p90       p99   >35°   >60°');
  const medians: number[] = [];
  for (const s of STEPS) {
    const p = profile(pts, s);
    const med = pct(p, 0.5);
    medians.push(med);
    const over = (t: number) => `${((100 * p.filter((v) => v > t).length) / p.length).toFixed(0)}%`;
    console.log(
      `  ${`${s} m`.padStart(7)}  ${deg(med).padStart(7)}  ${deg(pct(p, 0.9)).padStart(7)}  ` +
        `${deg(pct(p, 0.99)).padStart(7)}  ${over(0.7).padStart(5)}  ${over(1.73).padStart(5)}`,
    );
  }
  // Convergence is about the *increments*, not the level. Octaves of equal
  // slope still grow the total as sqrt(k), so the median is expected to creep
  // upward; what must not happen is the increments growing, which is what a
  // field with no slope limit does — its per-octave slope rises geometrically,
  // so on a log-spaced ladder each row adds more than the one above it.
  // Averaged in pairs because a single increment is noisy at this sample count.
  const d = medians.slice(1).map((m, i) => m - medians[i]);
  const coarse = (d[0] + d[1]) / 2;
  const fine = (d[d.length - 2] + d[d.length - 1]) / 2;
  console.log(
    fine > 0.7 * coarse && medians[medians.length - 1] > 0.03
      ? `  DIVERGING — increments ${coarse.toFixed(3)} -> ${fine.toFixed(3)}, slope is unbounded\n`
      : `  converged — increments ${coarse.toFixed(3)} -> ${fine.toFixed(3)}\n`,
  );
}

/* ── does the spectrum actually reach the ground it was written for? ──────
 *
 * The ladder table above is arithmetic. This is the planet: sample land of
 * each biome and walk transects across it at the same ladder of scales. The
 * claim being tested is narrow and falsifiable — a desert and a chaparral of
 * the *same* baked relief should differ in how their slope grows as the
 * sampling gets finer, and should not differ much in how much total relief
 * they have. If the two columns move together, the l2 normaliser is not doing
 * its job and the table is secretly an amplitude control.
 */

{
  const byBiome = new Map<number, V3[]>();
  for (const l of land) {
    const bake = sampleSurface(surface, l.d[0], l.d[1], l.d[2]);
    const h = heightAt(l.d, DEFAULT_OCTAVES);
    if (h <= 0) continue;
    const c = climateAt(l.d, bake.wetness);
    const idx = classify({
      temp: tempAtCPU(c.temp, h),
      base: c.temp,
      moist: c.moist,
      season: c.season,
      elevation: h,
    });
    if (idx < 0) continue;
    // Only ground the amplification actually acts on: on a floodplain the
    // amplitude collapses to FLOODPLAIN_AMP and every biome looks the same,
    // which would dilute the signal to nothing (LESSONS §1, "measure the
    // right subset").
    if (reliefOf(l.slope) < 0.15) continue;
    (byBiome.get(idx) ?? byBiome.set(idx, []).get(idx)!).push(l.d);
  }

  // Local differences, not transects. A 160-sample walk at 300 m spans 48 km,
  // which crosses several biomes — so the coarse row of a per-biome transect
  // is really a planet average, and dividing by it hides exactly the signal
  // being looked for. One step from the sample point keeps every measurement
  // inside the biome that was classified, at every scale.
  const localSlopes = (pts: V3[], stepM: number): number[] => {
    const out: number[] = [];
    for (const d of pts) {
      const h0 = heightAt(d, DEFAULT_OCTAVES);
      for (const t of tangent(d)) {
        out.push(Math.abs(heightAt(step(d, t, stepM / RADIUS), DEFAULT_OCTAVES) - h0) / stepM);
      }
    }
    return out.sort((a, b) => a - b);
  };

  console.log('slope on the ground, by biome (land with relief > 0.15, local steps):');
  console.log('  biome                         300 m    30 m     3 m    growth    samples');
  const rows: [string, number][] = [];
  for (const [idx, pts] of [...byBiome].sort((a, b) => b[1].length - a[1].length)) {
    const use = pick(pts, 500);
    if (use.length < 40) continue;
    const med = [300, 30, 3].map((s) => pct(localSlopes(use, s), 0.5));
    rows.push([BIOMES[idx].key, med[2] / med[0]]);
    console.log(
      `  ${BIOMES[idx].name.padEnd(30)}${deg(med[0]).padStart(6)}  ${deg(med[1]).padStart(6)}  ` +
        `${deg(med[2]).padStart(6)}   ${(med[2] / med[0]).toFixed(2).padStart(6)}x  ` +
        `${String(use.length * 2).padStart(9)}`,
    );
  }
  // The whole point is that this column is *not* flat. A single spectrum makes
  // every biome grow at the same rate; the ordering below is what the table
  // buys, and it is the thing to look at after changing a gain.
  const g = rows.map(([, v]) => v).sort((a, b) => a - b);
  console.log(
    `  3 m / 300 m slope growth spans ${g[0].toFixed(2)}x to ${g[g.length - 1].toFixed(2)}x ` +
      `across ${rows.length} biomes\n`,
  );
}

/* ── relief actually delivered, which is the other half of "does it read" ── */

console.log('relief over a transect (full field minus bake, and total):');
for (const [name, pts] of [
  ['typical land', pick(typical, 40)],
  ['rugged land', pick(rugged, 40)],
] as const) {
  for (const km of [20, 5]) {
    const totals: number[] = [];
    for (const d of pts) {
      const [t1] = tangent(d);
      const hs: number[] = [];
      for (let i = 0; i < 120; i++) {
        const q = step(d, t1, (i * ((km * 1000) / 119)) / RADIUS);
        hs.push(heightAt(q, DEFAULT_OCTAVES));
      }
      totals.push(Math.max(...hs) - Math.min(...hs));
    }
    totals.sort((a, b) => a - b);
    console.log(
      `  ${name.padEnd(13)} ${String(km).padStart(2)} km:  ` +
        `median ${pct(totals, 0.5).toFixed(0).padStart(5)} m   p90 ${pct(totals, 0.9).toFixed(0).padStart(5)} m`,
    );
  }
}
