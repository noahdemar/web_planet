/**
 * Do plants stand on the ground the terrain draws?
 *
 * This is invariant I3, and it has been broken twice by changes that looked
 * unrelated to vegetation. Both times the symptom was the same — trees hanging
 * in the air — and both times the cause was that the *terrain* and the
 * *scatter* were handed different views of the bake.
 *
 * They are different by design. The terrain samples the atlas per vertex. The
 * scatter cannot: it runs one thread per candidate cell over hundreds of
 * thousands of candidates, so it reads the bake once at the centre of a
 * VEG_LEVEL tile and reconstructs it linearly across the tile —
 *
 *     bakeH    = h(c) + grad(c) · dd        first order about the centre
 *     wet      = wet(c)                     constant across the tile
 *     distAxis = dist(c) + gradDist(c) · dd
 *
 * — and the height field is then evaluated from *those*. A plant is placed on
 * the result. So the gap between the two evaluations is, quite literally, how
 * far off the ground a tree sits.
 *
 * That gap is fine while it stays small, and small is not automatic: it is
 * small only because every term the height field applies is a smooth function
 * of fields that barely change across 563 m. Add a term that is *sharp* in one
 * of them — a slope window, a wetness threshold — and the reconstruction stops
 * tracking. That is what happened both times:
 *
 *   FLAT_WET_CUT   applied to `wet` on the terrain side only, so the two
 *                  paths disagreed about where a river was.
 *   FACET_*        keyed on baked slope through a narrow smoothstep, which
 *                  the tile centre resolves differently from a vertex.
 *
 * Neither shows up in `npm run mirror`, which checks the CPU against itself
 * and against seams. This measures the thing that actually moved.
 *
 *   npm run anchor
 */

import { readFileSync } from 'node:fs';
import { heightAt, setPlanetSurface, warpForCoast } from '../src/heightCPU.ts';
import { sampleSurface, surfaceFromBuffer, type Meta, type PlanetSurface } from '../src/planetData.ts';
import {
  DEFAULT_OCTAVES,
  FACE_EDGE,
  RADIUS,
  VEG_LEVEL,
  VEG_MAX_SLOPE,
} from '../src/planet.ts';

type V3 = [number, number, number];

const norm = (v: V3): V3 => {
  const l = Math.hypot(v[0], v[1], v[2]);
  return [v[0] / l, v[1] / l, v[2] / l];
};

/**
 * Thresholds, metres — fitted to the measured distribution, not chosen.
 *
 * This reports two numbers, because there are two different errors here and
 * only one of them is a regression risk.
 *
 * The **total** is the honest gap between the two surfaces: p50 0.86, p90 8.4,
 * p99 33.1, worst 256.6 over 4,438 plantable land points. A typical stem sits
 * about a metre off the drawn ground. That is a real property of the current
 * build and is larger than the code implies — quadtree.ts justifies one value
 * and one gradient per tile on the grounds that the bake "is very nearly
 * linear over" 563 m, which is true of the field but not of the gradient
 * paired with it: that is a secant across two 18 km cells, while the atlas is
 * bilinear with a different local slope. Shortening the differencing step does
 * not fix it — it re-evaluates the coastline warp, a displacement of
 * kilometres, over a few hundred metres and divides by that span; measured,
 * that took p99 from 33 m to 241 m. The cell-wide smoothing is load-bearing.
 * Closing this properly means sampling the bake per candidate, which is the
 * cost the tile record exists to avoid. Recorded, not asserted away.
 *
 * The **sharp-term channel** hands the elevation over exactly, so what remains
 * is only what the coarsely-carried fields cost — wetness, which is constant
 * across the whole tile, and slope, which comes from the centre's gradient.
 * That is the mechanism behind both I3 failures this project has had, and it
 * is what gets gated. Its measured p50 is 0.355 m and p90 8.01 m.
 *
 * Sensitivity is verified, not assumed. Re-introducing the FLAT_WET_CUT bug —
 * withdrawing wetness on one side only — moves the sharp channel from
 * 0.355 / 8.0 to 4.60 / 334 m, a 13x jump in the median. Note that no *constant*
 * reproduces it: both paths now apply that withdrawal inside height_, so the
 * failure is an asymmetry between code paths and only shows up as one. The
 * gates sit between the two, with ~3x headroom below and ~4x above.
 *
 * If you have genuinely improved the reconstruction, lower these. If you are
 * raising them, that is the test doing its job — find out what got sharp.
 */
const TOL_SHARP_P50 = 1.0;
const TOL_SHARP_P90 = 25.0;
/** The total gap, guarded loosely — it is dominated by the tile secant above. */
const TOL_P50 = 2.0;
const TOL_P99 = 60.0;

const SAMPLES = 20_000;

const meta: Meta = JSON.parse(readFileSync('public/planet/meta.json', 'utf8'));
const raw = readFileSync('public/planet/surface.f16');
const surface: PlanetSurface = surfaceFromBuffer(
  meta,
  raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength) as ArrayBuffer,
);
setPlanetSurface(surface);

/** Half-size of a VEG_LEVEL tile, as a direction offset. */
const tileHalf = FACE_EDGE / 2 ** VEG_LEVEL / 2 / RADIUS;

/**
 * The bake fields the scatter would reconstruct at `p` from a tile centred at
 * `c`: value and gradient sampled once at the centre, extrapolated linearly.
 *
 * Sampled straight out of the atlas, not through heightAt — the tile record
 * carries the *baked* elevation and its gradient, not the amplified surface,
 * and the amplification is what heightAt is going to add on top of this.
 */
function reconstruct(
  c: V3,
  p: V3,
): { elevation: number; exact: number; wetness: number; slope: number } {
  // One bake cell, the spacing the selector differences over. Not a free
  // choice: a shorter step re-evaluates the coastline warp — a displacement of
  // kilometres — over a few hundred metres and divides by that span, which
  // turns warp variation into a huge spurious gradient. Measured: dropping to
  // 0.03 cells took p99 from 30 m to 241 m. The cell-wide secant's smoothing
  // is load-bearing.
  const e = FACE_EDGE / surface.size / RADIUS;
  // Tangent to the cube face, as the selector's U is. Which face `c` is on is
  // decided by its largest component, and any axis not that one spans it.
  const m = [Math.abs(c[0]), Math.abs(c[1]), Math.abs(c[2])];
  const major = m[0] >= m[1] && m[0] >= m[2] ? 0 : m[1] >= m[2] ? 1 : 2;
  const ax: V3 = major === 0 ? [0, 1, 0] : [1, 0, 0];
  const d0 = ax[0] * c[0] + ax[1] * c[1] + ax[2] * c[2];
  const t1 = norm([ax[0] - c[0] * d0, ax[1] - c[1] * d0, ax[2] - c[2] * d0]);
  const t2 = norm([
    c[1] * t1[2] - c[2] * t1[1],
    c[2] * t1[0] - c[0] * t1[2],
    c[0] * t1[1] - c[1] * t1[0],
  ]);

  // Through the same coastline warp heightAt applies to its own lookup, or
  // this compares a warped surface against an unwarped one and reports the
  // warp — up to several kilometres of displacement near a coast — as float.
  const sample = (v: V3) => {
    const probe = sampleSurface(surface, v[0], v[1], v[2]).elevation;
    const w = warpForCoast(v, probe);
    return sampleSurface(surface, w[0], w[1], w[2]);
  };
  const at = (v: V3): number => sample(v).elevation;
  const step = (t: V3, sg: number): V3 =>
    norm([c[0] + t[0] * e * sg, c[1] + t[1] * e * sg, c[2] + t[2] * e * sg]);
  const g1 = (at(step(t1, 1)) - at(step(t1, -1))) / (2 * e);
  const g2 = (at(step(t2, 1)) - at(step(t2, -1))) / (2 * e);

  const dd: V3 = [p[0] - c[0], p[1] - c[1], p[2] - c[2]];
  const u1 = dd[0] * t1[0] + dd[1] * t1[1] + dd[2] * t1[2];
  const u2 = dd[0] * t2[0] + dd[1] * t2[1] + dd[2] * t2[2];

  const centre = sample(c);
  return {
    // First order about the centre, as vegSample does with t5.
    elevation: centre.elevation + g1 * u1 + g2 * u2,
    // The same field sampled at p instead of extrapolated to it, through the
    // identical warp — what the terrain vertex gets.
    exact: sample(p).elevation,
    // Constant across the tile, exactly as t6.x is.
    wetness: centre.wetness,
    // From the centre's gradient, so a tile straddling a slope break resolves
    // it differently from a vertex sitting on one — which is the divergence
    // the FACET_* window turned into visible float.
    slope: Math.hypot(g1, g2) / RADIUS,
  };
}

let worst = 0;
let worstDir: V3 = [0, 1, 0];
let over = 0;
let sum = 0;
let n = 0;
const all: number[] = [];
const allSharp: number[] = [];
let worstSharp = 0;

// Low-discrepancy directions, so the sample is reproducible and does not clump.
const P1 = 0.7548776662466927;
const P2 = 0.5698402909980532;
for (let i = 1; i <= SAMPLES; i++) {
  const y = 1 - (2 * (i + 0.5)) / SAMPLES;
  const r = Math.sqrt(Math.max(0, 1 - y * y));
  const th = ((i * P1) % 1) * Math.PI * 2;
  const p: V3 = [Math.cos(th) * r, y, Math.sin(th) * r];

  // Land only: nothing is planted on the sea floor.
  const exact = heightAt(p, DEFAULT_OCTAVES, 1);
  if (exact <= 1) continue;

  // And only where a plant would actually be placed. The scatter hard-rejects
  // anything steeper than VEG_MAX_SLOPE, so cliffs and mountain faces never
  // carry a stem — and they are precisely where the tile reconstruction is
  // least accurate. Measuring them would report a float that nothing can
  // exhibit, and would drown the terrain that does carry plants.
  const axs: V3 = Math.abs(p[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
  const ds = axs[0] * p[0] + axs[1] * p[1] + axs[2] * p[2];
  const s1 = norm([axs[0] - p[0] * ds, axs[1] - p[1] * ds, axs[2] - p[2] * ds]);
  const s2 = norm([
    p[1] * s1[2] - p[2] * s1[1],
    p[2] * s1[0] - p[0] * s1[2],
    p[0] * s1[1] - p[1] * s1[0],
  ]);
  // One scatter cell, the spacing stems are actually placed at.
  const ds0 = 4.4 / RADIUS;
  const hAt = (t: V3, sg: number): number =>
    heightAt(norm([p[0] + t[0] * ds0 * sg, p[1] + t[1] * ds0 * sg, p[2] + t[2] * ds0 * sg]),
      DEFAULT_OCTAVES, 1);
  const gs1 = (hAt(s1, 1) - hAt(s1, -1)) / (2 * ds0 * RADIUS);
  const gs2 = (hAt(s2, 1) - hAt(s2, -1)) / (2 * ds0 * RADIUS);
  const tanT = Math.hypot(gs1, gs2);
  // The scatter's own measure: 1 - cos(theta) of the surface normal from up.
  if (1 - 1 / Math.sqrt(1 + tanT * tanT) > VEG_MAX_SLOPE) continue;

  // A tile centre offset from p by up to half a tile, as a real one would be.
  // Offset in the tangent plane at p, bounded by the tile half-size — an
  // instance is somewhere inside its tile, never further.
  const jx = ((i * P2) % 1) * 2 - 1;
  const jy = ((i * P1 * 7.13) % 1) * 2 - 1;
  const axp: V3 = Math.abs(p[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
  const dp = axp[0] * p[0] + axp[1] * p[1] + axp[2] * p[2];
  const e1 = norm([axp[0] - p[0] * dp, axp[1] - p[1] * dp, axp[2] - p[2] * dp]);
  const e2 = norm([
    p[1] * e1[2] - p[2] * e1[1],
    p[2] * e1[0] - p[0] * e1[2],
    p[0] * e1[1] - p[1] * e1[0],
  ]);
  const c = norm([
    p[0] + tileHalf * (e1[0] * jx + e2[0] * jy),
    p[1] + tileHalf * (e1[1] * jx + e2[1] * jy),
    p[2] + tileHalf * (e1[2] * jx + e2[2] * jy),
  ]);

  const rec = reconstruct(c, p);
  const planted = heightAt(p, DEFAULT_OCTAVES, 1, 1e9, true, {
    elevation: rec.elevation,
    wetness: rec.wetness,
    slope: rec.slope,
  });
  if (!Number.isFinite(planted) || !Number.isFinite(exact)) {
    throw new Error(`anchoring: non-finite height at ${p.join(', ')}`);
  }

  // The same placement with the *elevation* handed over exactly, so only the
  // coarsely-carried fields differ. See the two channels above.
  const sharp = heightAt(p, DEFAULT_OCTAVES, 1, 1e9, true, {
    elevation: rec.exact,
    wetness: rec.wetness,
    slope: rec.slope,
  });
  const dSharp = Math.abs(sharp - exact);
  allSharp.push(dSharp);
  if (dSharp > worstSharp) worstSharp = dSharp;

  const d = Math.abs(planted - exact);
  all.push(d);
  sum += d;
  n++;
  if (d > TOL_P50) over++;
  if (d > worst) {
    worst = d;
    worstDir = p;
  }
}

const mean = sum / Math.max(n, 1);
all.sort((a, b) => a - b);
allSharp.sort((a, b) => a - b);
const q = (a: number[], t: number): number => a[Math.min(a.length - 1, Math.floor(a.length * t))] ?? 0;
const pc = (t: number): number => q(all, t);
const ps = (t: number): number => q(allSharp, t);

process.stdout.write(
  '\n  anchoring — the surface a plant is placed on against the surface drawn\n\n' +
    `    land samples          ${n.toLocaleString()}\n` +
    `    mean |Δ|              ${mean.toFixed(3)} m\n` +
    `    p50 / p90             ${pc(0.5).toFixed(3)} / ${pc(0.9).toFixed(3)} m\n` +
    `    p99 / p99.9           ${pc(0.99).toFixed(3)} / ${pc(0.999).toFixed(3)} m\n` +
    `    worst |Δ|             ${worst.toFixed(3)} m\n` +
    `    over ${TOL_P50} m              ${over} (${((100 * over) / Math.max(n, 1)).toFixed(2)}%)\n` +
    `    worst at              ${worstDir.map((v) => v.toFixed(6)).join(', ')}\n\n` +
    '    sharp-term channel — elevation handed over exactly, so this is only\n' +
    '    what the coarsely-carried fields (wetness, slope) cost\n\n' +
    `    p50 / p90             ${ps(0.5).toFixed(4)} / ${ps(0.9).toFixed(4)} m\n` +
    `    p99 / worst           ${ps(0.99).toFixed(4)} / ${worstSharp.toFixed(4)} m\n\n`,
);

const fail: string[] = [];
if (ps(0.5) > TOL_SHARP_P50) fail.push(`sharp p50 ${ps(0.5).toFixed(3)} m > ${TOL_SHARP_P50}`);
if (ps(0.9) > TOL_SHARP_P90) fail.push(`sharp p90 ${ps(0.9).toFixed(3)} m > ${TOL_SHARP_P90}`);
if (pc(0.5) > TOL_P50) fail.push(`total p50 ${pc(0.5).toFixed(3)} m > ${TOL_P50}`);
if (pc(0.99) > TOL_P99) fail.push(`total p99 ${pc(0.99).toFixed(3)} m > ${TOL_P99}`);

if (fail.length > 0) {
  throw new Error(
    `anchoring: ${fail.join('; ')}\n` +
      '  Plants stand further off the drawn ground than they did. Something the\n' +
      '  height field applies has become sharp in a bake field that the scatter\n' +
      '  only carries coarsely across a tile — wetness is constant over all\n' +
      '  563 m of it, and slope comes from the centre. Check any new slope or\n' +
      '  wetness window, and check that every caller of height_ is handed the\n' +
      '  same untransformed inputs: the two failures this has caught before were\n' +
      '  both a transform applied on the terrain side only. See the notes above.',
  );
}
process.stdout.write('  anchor ok\n');
