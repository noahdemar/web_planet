/**
 * How flat is the flat ground?
 *
 * Local relief — the height range over a 2 km window — bucketed by the baked
 * slope that ampAt_ keys its amplitude on. This is the instrument for the
 * plains problem: "the plains look flat" is a claim about a number, and the
 * number is not one you can read off any constant, because the drawn relief is
 * the amplitude times a normalised noise sum and only the finest octaves reach
 * inside a 2 km window at all.
 *
 * Measured on the shipped bake:
 *
 *     flat     15.2% of land   p50   4.6 m
 *     gentle                   p50  42.6 m
 *     hilly                    p50 240.5 m
 *
 * For scale, real plains over the same window: Amazon floodplain 5-15 m, till
 * and loess plains 10-50 m, the Great Plains 20-60 m. So the gentle and hilly
 * classes are about right and the flat class is roughly an order of magnitude
 * too smooth — 4.6 m over two kilometres is a billiard table.
 *
 * The cause is structural rather than a mistuned constant. ampAt_ sets relief
 * from smoothstep(RELIEF_SLOPE_LO, ..., slope), so ground that the bake left
 * flat is amplified least, and flatness is self-reinforcing. Real plains are
 * not flat because they have no slope; they are flat because they were built
 * by deposition, and each depositional process leaves its own texture.
 *
 *   npm run plains
 */

import { readFileSync } from 'node:fs';
import { heightAt, setPlanetSurface } from '../src/heightCPU.js';
import { sampleSurface, surfaceFromBuffer, type Meta, type PlanetSurface } from '../src/planetData.js';
import { DEFAULT_OCTAVES, FACE_EDGE, RADIUS } from '../src/planet.js';

type V3 = [number, number, number];
const norm = (v: V3): V3 => { const l = Math.hypot(...v); return [v[0]/l, v[1]/l, v[2]/l]; };

const meta: Meta = JSON.parse(readFileSync('public/planet/meta.json', 'utf8'));
const raw = readFileSync('public/planet/surface.f16');
const surface: PlanetSurface = surfaceFromBuffer(meta, raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength) as ArrayBuffer);
setPlanetSurface(surface);

const P1 = 0.7548776662466927;
const bucket: Record<string, number[]> = { flat: [], gentle: [], hilly: [] };
const wide: Record<string, number[]> = { flat: [], gentle: [], hilly: [] };
let land = 0, flatArea = 0;

for (let i = 1; i <= 30000; i++) {
  const y = 1 - (2 * (i + 0.5)) / 30000;
  const r = Math.sqrt(Math.max(0, 1 - y * y));
  const th = ((i * P1) % 1) * Math.PI * 2;
  const p: V3 = [Math.cos(th) * r, y, Math.sin(th) * r];
  const h0 = heightAt(p, DEFAULT_OCTAVES, 1);
  if (h0 <= 1) continue;
  land++;

  // Baked slope, the thing ampAt keys on.
  const e = FACE_EDGE / surface.size / RADIUS;
  const ax: V3 = Math.abs(p[0]) < 0.9 ? [1,0,0] : [0,1,0];
  const d0 = ax[0]*p[0]+ax[1]*p[1]+ax[2]*p[2];
  const t1 = norm([ax[0]-p[0]*d0, ax[1]-p[1]*d0, ax[2]-p[2]*d0]);
  const t2 = norm([p[1]*t1[2]-p[2]*t1[1], p[2]*t1[0]-p[0]*t1[2], p[0]*t1[1]-p[1]*t1[0]]);
  const sb = (t: V3, s: number) => sampleSurface(surface, ...(norm([p[0]+t[0]*e*s, p[1]+t[1]*e*s, p[2]+t[2]*e*s]) as V3)).elevation;
  const slope = Math.hypot((sb(t1,1)-sb(t1,-1))/(2*e), (sb(t2,1)-sb(t2,-1))/(2*e)) / RADIUS;

  // Local relief: height range over a window, at two scales. 2 km is what you
  // see standing on it; 10 km is the regional form you cross.
  const relief = (radiusM: number): number => {
    const step = radiusM / 4 / RADIUS;
    let lo = Infinity, hi = -Infinity;
    for (let a = 0; a < 8; a++) {
      for (let k = 1; k <= 4; k++) {
        const an = (a / 8) * Math.PI * 2;
        const d = norm([
          p[0] + (t1[0]*Math.cos(an) + t2[0]*Math.sin(an)) * step * k,
          p[1] + (t1[1]*Math.cos(an) + t2[1]*Math.sin(an)) * step * k,
          p[2] + (t1[2]*Math.cos(an) + t2[2]*Math.sin(an)) * step * k,
        ]);
        const h = heightAt(d, DEFAULT_OCTAVES, 1);
        if (h < lo) lo = h; if (h > hi) hi = h;
      }
    }
    return hi - lo;
  };
  const rel = relief(1000);
  const relW = relief(5000);
  const key = slope < 0.0015 ? 'flat' : slope < 0.0111 ? 'gentle' : 'hilly';
  bucket[key]!.push(rel);
  wide[key]!.push(relW);
  if (key === 'flat') flatArea++;
}

const q = (a: number[], t: number) => { const s=[...a].sort((x,y)=>x-y); return s[Math.floor(s.length*t)] ?? 0; };
process.stdout.write(`\n  local relief by baked slope class\n\n`);
for (const [k, v] of Object.entries(bucket)) {
  const w = wide[k]!;
  process.stdout.write(`    ${k.padEnd(7)} n=${String(v.length).padStart(5)}   ` +
    `2 km  p50 ${q(v,0.5).toFixed(1).padStart(6)}  p90 ${q(v,0.9).toFixed(1).padStart(6)}` +
    `   10 km  p50 ${q(w,0.5).toFixed(1).padStart(6)}  p90 ${q(w,0.9).toFixed(1).padStart(6)} m\n`);
}
