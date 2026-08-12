/**
 * Run the global bake offline and check it against Earth.
 *
 * SPEC.md §4 says the bake is done when it completes in under a minute and
 * the result sits inside Earth's range on hypsometry, drainage density and
 * Horton's laws. Those are the three numbers this prints. Eyeballing a
 * height field cannot distinguish "has drainage" from "has noise that looks
 * a bit like drainage" — the Horton ratios can.
 *
 *   npm run bake              default parameters
 *   npm run bake -- --res 256 faster iteration
 *   npm run bake -- --write   emit the runtime asset
 */

import { writeFileSync, mkdirSync } from 'node:fs';
import { bake, DEFAULT_BAKE } from '../src/bake/index.ts';
import { buildGrid } from '../src/bake/grid.ts';
import { cellAreas, routeFlow } from '../src/bake/lem.ts';
import { cellAt } from '../src/bake/grid.ts';
import { toAtlas, ATLAS_COLS, ATLAS_PAD, type AtlasData } from '../src/bake/cubemap.ts';
import { DataUtils } from 'three';

const argv = process.argv.slice(2);
const arg = (k: string, d: number): number => {
  const i = argv.indexOf(`--${k}`);
  return i >= 0 ? Number(argv[i + 1]) : d;
};

const RES = arg('res', DEFAULT_BAKE.res);

/**
 * CPU emulation of hardware cube-map sampling: pick the major axis, form the
 * face coordinates, read the nearest texel. Used only to verify the written
 * asset — if this disagrees with the solve grid, the faces are laid out wrong.
 */
function sampleCube(cube: AtlasData, d: [number, number, number], rotate = false): number {
  const ax = Math.abs(d[0]);
  const ay = Math.abs(d[1]);
  const az = Math.abs(d[2]);
  let face: number;
  let sc: number;
  let tc: number;
  let ma: number;
  if (ax >= ay && ax >= az) {
    face = d[0] > 0 ? 0 : 1;
    ma = ax;
    sc = d[0] > 0 ? -d[2] : d[2];
    tc = -d[1];
  } else if (ay >= az) {
    face = d[1] > 0 ? 2 : 3;
    ma = ay;
    sc = d[0];
    tc = d[1] > 0 ? d[2] : -d[2];
  } else {
    face = d[2] > 0 ? 4 : 5;
    ma = az;
    sc = d[2] > 0 ? d[0] : -d[0];
    tc = -d[1];
  }
  let s = (sc / ma + 1) / 2;
  let t = (tc / ma + 1) / 2;
  if (rotate) {
    const s0 = s;
    s = t;
    t = 1 - s0;
  }
  const n = cube.faceSize;
  const cell = n + 2 * ATLAS_PAD;
  const ox = (face % ATLAS_COLS) * cell + ATLAS_PAD;
  const oy = Math.floor(face / ATLAS_COLS) * cell + ATLAS_PAD;
  const i = Math.min(n - 1, Math.max(0, Math.floor(s * n)));
  const j = Math.min(n - 1, Math.max(0, Math.floor(t * n)));
  return DataUtils.fromHalfFloat(cube.data[((oy + j) * cube.width + ox + i) * 4]);
}

const params = { ...DEFAULT_BAKE, res: RES };

/**
 * Earth's hypsographic curve: fraction of total surface area in each
 * elevation band, shallow-to-deep. This replaces the handful of scalars the
 * older tool compared against — one of which (ocean "median" −3700 m) was
 * actually Earth's *mean* ocean depth, so the model was being tuned toward a
 * target 600 m off. A full curve cannot be wrong in that quiet way: the bands
 * have to sum to 1 and the land bands have to sum to the land fraction, and
 * both are checked below.
 */
const EARTH_CURVE: readonly (readonly [number, number, number])[] = [
  // [lower bound, upper bound, fraction of Earth's surface]
  [5000, Infinity, 0.001],
  [4000, 5000, 0.005],
  [3000, 4000, 0.011],
  [2000, 3000, 0.022],
  [1000, 2000, 0.048],
  [500, 1000, 0.058],
  [200, 500, 0.060],
  [0, 200, 0.087],
  [-200, 0, 0.053],
  [-1000, -200, 0.031],
  [-2000, -1000, 0.030],
  [-3000, -2000, 0.050],
  [-4000, -3000, 0.121],
  [-5000, -4000, 0.216],
  [-6000, -5000, 0.184],
  [-Infinity, -6000, 0.023],
];

const EARTH = {
  landFraction: 0.292,
  landMedian: 797,
  landP90: 2200,
  landP99: 4600,
  landAbove3100: 0.049,
  /** Median, not mean: half the ocean floor is deeper than this. */
  oceanMedian: -4320,
  median: -3360,
  /** Channels per kilometre of basin, humid temperate. Field range 0.5–5. */
  drainageDensity: [0.5, 5],
  /** Horton bifurcation ratio. Field range 3–5. */
  bifurcation: [3, 5],
  /** Horton length ratio. Field range 1.5–3.5. */
  lengthRatio: [1.5, 3.5],
};

{
  const sum = EARTH_CURVE.reduce((a, [, , f]) => a + f, 0);
  const land = EARTH_CURVE.filter(([lo]) => lo >= 0).reduce((a, [, , f]) => a + f, 0);
  if (Math.abs(sum - 1) > 5e-3 || Math.abs(land - EARTH.landFraction) > 5e-3) {
    throw new Error(`EARTH_CURVE inconsistent: sums to ${sum}, land ${land}`);
  }
}

process.stdout.write(`baking ${RES}²·6 = ${(6 * RES * RES / 1e6).toFixed(2)} M cells\n`);
let lastStage = '';
const out = bake(params, (stage, t) => {
  if (stage !== lastStage || t === 1) {
    process.stdout.write(`\r  ${stage.padEnd(10)} ${(t * 100).toFixed(0).padStart(3)}%   `);
    lastStage = stage;
  }
});
process.stdout.write('\n');

const grid = buildGrid(RES);
const areas = cellAreas(grid);
const z = out.elevation;
const N = grid.count;

// --- hypsometry, area-weighted -------------------------------------------
let total = 0;
for (let c = 0; c < N; c++) total += areas[c];

const order = Array.from({ length: N }, (_, i) => i).sort((a, b) => z[a] - z[b]);
const quantile = (p: number): number => {
  let acc = 0;
  const want = p * total;
  for (const c of order) {
    acc += areas[c];
    if (acc >= want) return z[c];
  }
  return z[order[N - 1]];
};

const fracAbove = (h: number): number => {
  let a = 0;
  for (let c = 0; c < N; c++) if (z[c] > h) a += areas[c];
  return a / total;
};
const fracBetween = (lo: number, hi: number): number => {
  let a = 0;
  for (let c = 0; c < N; c++) if (z[c] > lo && z[c] <= hi) a += areas[c];
  return a / total;
};

// Spread over 1.5 M elements, so no Math.max(...z) — that overflows the
// argument stack well before this size.
let zMin = Infinity;
let zMax = -Infinity;
for (let c = 0; c < N; c++) {
  if (z[c] < zMin) zMin = z[c];
  if (z[c] > zMax) zMax = z[c];
}

const landFraction = fracAbove(0);
let landArea = 0;
let landAbove3100 = 0;
const landZ: number[] = [];
const oceanZ: number[] = [];
for (let c = 0; c < N; c++) {
  if (z[c] > 0) {
    landArea += areas[c];
    if (z[c] > 3100) landAbove3100 += areas[c];
    landZ.push(z[c]);
  } else oceanZ.push(z[c]);
}
landZ.sort((a, b) => a - b);
oceanZ.sort((a, b) => a - b);
const q = (a: number[], p: number) => (a.length ? a[Math.floor(p * (a.length - 1))] : 0);

const m = (v: number) => `${v >= 0 ? ' ' : ''}${v.toFixed(0).padStart(6)} m`;
const pc = (v: number) => `${(v * 100).toFixed(1).padStart(5)}%`;
const rng = (v: number, [lo, hi]: number[]) =>
  `${v.toFixed(2).padStart(6)}   ${lo}–${hi}   ${v >= lo && v <= hi ? 'ok' : 'OUT'}`;

// Full-curve comparison. L1 over the bands is the single number to minimise:
// it is the fraction of the planet's surface that sits in the wrong elevation
// band, so 0.08 literally means "8% of the surface is at the wrong height".
let l1 = 0;
const bandRows = EARTH_CURVE.map(([lo, hi, want]) => {
  const got = fracBetween(lo, hi);
  l1 += Math.abs(got - want);
  const bar = '#'.repeat(Math.round(got * 200)).padEnd(24).slice(0, 24);
  const mark = Math.abs(got - want) > 0.035 ? ' <<' : '';
  const label = `${lo === -Infinity ? '  <' : lo.toFixed(0).padStart(5)}…${hi === Infinity ? '' : hi.toFixed(0)}`;
  return `  ${label.padEnd(13)} ${pc(got)} ${pc(want)}  ${bar}${mark}`;
});

process.stdout.write(`
hypsography            this  Earth
${bandRows.join('\n')}
  ------------------------------------------------
  curve error (L1)          ${(l1 / 2 * 100).toFixed(1)}% of the surface in the wrong band

                              this      Earth
  land fraction              ${pc(landFraction)}     ${pc(EARTH.landFraction)}
  median elevation           ${m(quantile(0.5))}   ${m(EARTH.median)}
  land median                ${m(q(landZ, 0.5))}   ${m(EARTH.landMedian)}
  land p90                   ${m(q(landZ, 0.9))}   ${m(EARTH.landP90)}
  land p99                   ${m(q(landZ, 0.99))}   ${m(EARTH.landP99)}
  land above 3100 m          ${pc(landAbove3100 / Math.max(landArea, 1))}     ${pc(EARTH.landAbove3100)}
  ocean median               ${m(q(oceanZ, 0.5))}   ${m(EARTH.oceanMedian)}
  highest                    ${m(zMax)}
  deepest                    ${m(zMin)}
`);

// --- drainage network -----------------------------------------------------
// A cell is a channel where its drainage area exceeds a threshold; that is the
// standard support-area definition. Strahler order then comes from the
// receiver tree, and Horton's ratios from the order statistics.
const flow = routeFlow(grid, Float32Array.from(z), areas, 0);

// Support area for a channel head. A real threshold is ~1 km², but a cell here
// is hundreds of km² — so the threshold has to be a *multiple of cell area* or
// every single land cell counts as a first-order stream and Horton's ratios
// measure the grid instead of the network.
const meanCell = total / N;
const CHANNEL_AREA = Math.max(1e9, 25 * meanCell);

const isChannel = new Uint8Array(N);
for (let c = 0; c < N; c++) isChannel[c] = z[c] > 0 && flow.area[c] >= CHANNEL_AREA ? 1 : 0;

// Strahler order, computed upstream-to-downstream over the stack.
const order2 = new Int32Array(N);
const maxDonor = new Int32Array(N);
const donorCount = new Int32Array(N);
for (let i = N - 1; i >= 0; i--) {
  const c = flow.stack[i];
  if (!isChannel[c]) continue;
  order2[c] = donorCount[c] === 0 ? 1 : donorCount[c] > 1 ? maxDonor[c] + 1 : maxDonor[c];
  const r = flow.receiver[c];
  if (r !== c && isChannel[r]) {
    if (order2[c] > maxDonor[r]) {
      maxDonor[r] = order2[c];
      donorCount[r] = 1;
    } else if (order2[c] === maxDonor[r]) donorCount[r]++;
  }
}

// Segment counts and mean lengths per order. A segment runs until the order
// changes, which is exactly the Strahler link definition.
const nSeg = new Map<number, number>();
const lenSum = new Map<number, number>();
const segLen = new Float64Array(N);
for (let i = 0; i < N; i++) {
  const c = flow.stack[i];
  if (!isChannel[c]) continue;
  const r = flow.receiver[c];
  const carry = r !== c && isChannel[r] && order2[r] === order2[c] ? segLen[r] : 0;
  segLen[c] = carry + flow.length[c];
}
for (let c = 0; c < N; c++) {
  if (!isChannel[c]) continue;
  // A segment ends at a cell whose donors of the same order number zero.
  let sameOrderDonor = false;
  for (let k = 0; k < 8; k++) {
    const n = grid.nbr[c * 8 + k];
    if (isChannel[n] && flow.receiver[n] === c && order2[n] === order2[c]) sameOrderDonor = true;
  }
  if (sameOrderDonor) continue;
  const o = order2[c];
  nSeg.set(o, (nSeg.get(o) ?? 0) + 1);
  lenSum.set(o, (lenSum.get(o) ?? 0) + segLen[c]);
}

let chanLen = 0;
for (let c = 0; c < N; c++) if (isChannel[c]) chanLen += flow.length[c];
const density = chanLen / 1000 / (landArea / 1e6); // km of channel per km²

const orders = [...nSeg.keys()].sort((a, b) => a - b);
let bifSum = 0;
let bifN = 0;
let lenRatioSum = 0;
let lenRatioN = 0;
process.stdout.write('drainage network\n  order   segments   mean length\n');
for (const o of orders) {
  const n = nSeg.get(o)!;
  const l = lenSum.get(o)! / n / 1000;
  process.stdout.write(`  ${String(o).padStart(5)}   ${String(n).padStart(8)}   ${l.toFixed(0).padStart(8)} km\n`);
  const prev = nSeg.get(o - 1);
  if (prev && n > 0) {
    bifSum += prev / n;
    bifN++;
    lenRatioSum += l / (lenSum.get(o - 1)! / prev / 1000);
    lenRatioN++;
  }
}

// Drainage density has a hard ceiling of 1/spacing: you cannot fit more
// channel into a cell than one channel per cell. Comparing the raw number to
// Earth's 0.5–5 km/km² would only measure the grid, so report the ceiling too
// and treat the *scale-invariant* Horton ratios as the real test.
const ceiling = 1000 / grid.spacing;

process.stdout.write(`
                              this   Earth
  drainage density           ${density.toFixed(3)}   0.5–5    km/km², ceiling at this
                                             resolution ${ceiling.toFixed(3)} — sub-grid
                                             density is the runtime's job
  bifurcation ratio Rb       ${rng(bifN ? bifSum / bifN : 0, EARTH.bifurcation)}
  length ratio Rl            ${rng(lenRatioN ? lenRatioSum / lenRatioN : 0, EARTH.lengthRatio)}

timings (ms)  ${Object.entries(out.timings).map(([k, v]) => `${k} ${v}`).join('   ')}
`);

// --- is the network still a staircase? ------------------------------------
//
// The question the curve fitting exists to answer, and the only one a picture
// of a river cannot settle. A cell-to-cell path turns 45 degrees every cell or
// two; a curve through those same cells turns by however much the valley bends.
// Total curvature per 100 km separates the two and belongs to the curve rather
// than to any frame — see cornerStats.
process.stdout.write(`
drainage geometry
  strands                    ${out.strandCount}
  vertices                   ${out.strandVertices}
  median segment             ${(out.medianSegment / 1000).toFixed(2)} km
  turning                    ${out.turnPer100km.toFixed(0)} deg per 100 km
  of it in corners > 20 deg  ${(out.sharpShare * 100).toFixed(1)}%
`);

if (argv.includes('--write')) {
  mkdirSync('public/planet', { recursive: true });

  // Round-trip check before writing: sample the cube map back at a few
  // thousand directions and compare against the solve grid. This catches the
  // failure that would otherwise be found by eye — a face written with the
  // wrong in-plane rotation, which looks fine in isolation and produces a
  // 1600 km discontinuity along one face edge.
  //
  // The statistic is the *mean*, not the worst case. Worst case is dominated
  // by half-texel offsets in terrain that has 2 km of relief per cell, so it
  // is large even when everything is correct; a mis-oriented face instead
  // compares unrelated parts of the planet and shows up in the mean.
  const cube = toAtlas(grid, out.elevation, out.wetness, out.lakeDepth, out.channelDist);

  const roundTrip = (rotate: boolean): { mean: number; worst: number } => {
    let worst = 0;
    let sum = 0;
    const CHECKS = 4000;
    for (let k = 0; k < CHECKS; k++) {
      const zz = 1 - (2 * k + 1) / CHECKS;
      const rr = Math.sqrt(Math.max(0, 1 - zz * zz));
      const th = k * 2.399963229728653;
      const d: [number, number, number] = [rr * Math.cos(th), rr * Math.sin(th), zz];
      const want = out.elevation[cellAt(grid, d)];
      const got = sampleCube(cube, d, rotate);
      const e = Math.abs(got - want);
      sum += e;
      if (e > worst) worst = e;
    }
    return { mean: sum / CHECKS, worst };
  };

  const ok = roundTrip(false);
  // Control: the same check with every face rotated a quarter turn. If this
  // does not fail loudly then the check has no power and proves nothing.
  const bad = roundTrip(true);
  process.stdout.write(
    `cube round-trip   mean |Δ| ${ok.mean.toFixed(1)} m   worst ${ok.worst.toFixed(0)} m` +
      `   (rotated-face control: ${bad.mean.toFixed(0)} m)\n`,
  );
  if (ok.mean > 200) {
    throw new Error(`cube faces are mis-oriented: mean |Δ| ${ok.mean.toFixed(0)} m`);
  }
  if (bad.mean < 4 * ok.mean) {
    throw new Error('round-trip check cannot detect a rotated face — the check is useless');
  }

  const bytes = Buffer.from(cube.data.buffer, cube.data.byteOffset, cube.data.byteLength);
  writeFileSync('public/planet/surface.f16', bytes);
  writeFileSync(
    'public/planet/meta.json',
    JSON.stringify(
      {
        size: cube.faceSize,
        width: cube.width,
        height: cube.height,
        pad: ATLAS_PAD,
        layout: '3x2 cube atlas, one-texel border per face',
        channels: ['elevation_m', 'wetness_log10_area', 'lake_depth_m', 'channel_dist_m'],
        format: 'rgba16float',
        solveRes: RES,
        seed: params.tectonics.seed,
      },
      null,
      2,
    ),
  );
  process.stdout.write(`wrote public/planet/surface.f16 — ${(bytes.length / 1e6).toFixed(1)} MB\n`);
}
