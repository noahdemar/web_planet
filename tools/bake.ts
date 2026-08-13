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
 * Which planet to build.
 *
 * Everything downstream is a deterministic function of this and the resolution:
 * plate seeds and their motions come straight from it, and the erosion,
 * drainage, climate and biomes are all consequences of the tectonics. So a new
 * seed is a new world with the same physics, and the same seed is always the
 * same world — which is what makes the realism baseline (npm run realism) and
 * the site table in src/tour.ts meaningful at all. Both are tied to a seed and
 * have to be reissued when it changes; the run prints the reminder.
 *
 *   npm run bake -- --seed 12345 --write
 */
const SEED = arg('seed', DEFAULT_BAKE.tectonics.seed);

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

const params = {
  ...DEFAULT_BAKE,
  res: RES,
  tectonics: { ...DEFAULT_BAKE.tectonics, seed: SEED },
};

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
  /**
   * Horton's ratios, *typical* values — not hard bounds, which is a
   * distinction this file used to get wrong by calling them the field range.
   * Horton (1945) and Strahler (1957) put most natural basins at Rb 3–5 and
   * Rl 1.5–3.5, but real basins run Rb 2–8: strong structural control makes a
   * network elongated and pushes it high, and nothing about a value of 5.1 says
   * a drainage network is wrong.
   *
   * So the band is reported with slack, and the number to read alongside it is
   * R². These ratios are the slope of a straight line through log-counts; a
   * ratio quoted from points that do not lie on a line is not measuring
   * Horton's law at all, and that is the failure this check exists to catch.
   */
  bifurcation: [3, 5],
  lengthRatio: [1.5, 3.5],
};

{
  const sum = EARTH_CURVE.reduce((a, [, , f]) => a + f, 0);
  const land = EARTH_CURVE.filter(([lo]) => lo >= 0).reduce((a, [, , f]) => a + f, 0);
  if (Math.abs(sum - 1) > 5e-3 || Math.abs(land - EARTH.landFraction) > 5e-3) {
    throw new Error(`EARTH_CURVE inconsistent: sums to ${sum}, land ${land}`);
  }
}

process.stdout.write(`baking ${RES}²·6 = ${(6 * RES * RES / 1e6).toFixed(2)} M cells, seed ${SEED}\n`);
let lastStage = '';
const out = bake(params, (stage, t) => {
  if (stage !== lastStage || t === 1) {
    process.stdout.write(`\r  ${stage.padEnd(10)} ${(t * 100).toFixed(0).padStart(3)}%   `);
    lastStage = stage;
  }
});
process.stdout.write('\n');

const grid = out.grid;
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
/**
 * A value against a typical band, with slack.
 *
 * Ten per cent, because the band is a rule of thumb rather than a tolerance —
 * flagging OUT at 5.07 against 3–5 says nothing about the planet and only
 * trains the reader to ignore the line.
 */
const SLACK = 0.1;
const rng = (v: number, [lo, hi]: number[]) => {
  const inside = v >= lo && v <= hi;
  const near = v >= lo * (1 - SLACK) && v <= hi * (1 + SLACK);
  const tag = inside ? 'ok' : near ? (v > hi ? 'high' : 'low') : 'OUT';
  return `${v.toFixed(2).padStart(6)}   ${lo}–${hi}   ${tag.padEnd(4)}`;
};

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

/**
 * Fewest segments an order needs before its count is worth fitting to.
 *
 * The relative standard error of a count goes as 1/sqrt(N), so an order with
 * four segments carries a 50% error and an order with a hundred carries ten.
 * The old estimator averaged the consecutive ratios with no such filter, and
 * on a whole planet the top Strahler order is always a handful of segments —
 * there are only so many continent-sized basins. At 1024 that meant averaging
 * 4.57, 4.43, 6.61 and 113/4 = 28.25, and the last of those, computed from
 * four segments, moved the reported ratio from 4.9 to 10.96 on its own. The
 * model was never outside Earth's range; the statistic was.
 */
const MIN_SEGMENTS = 30;

/**
 * Horton's ratios by log-linear regression, which is how they are defined.
 *
 * Horton's laws say the counts fall geometrically with order and the lengths
 * rise geometrically, so the ratio is the slope of ln(quantity) against order
 * — a fit over every usable order at once, not a mean of the steps between
 * them. R² comes back with it, because a ratio quoted from a set of points
 * that are not on a line is not describing anything.
 */
function horton(value: (o: number) => number, sign: number): {
  ratio: number; r2: number; used: number[];
} {
  const used = orders.filter((o) => (nSeg.get(o) ?? 0) >= MIN_SEGMENTS);
  if (used.length < 2) return { ratio: 0, r2: 0, used };
  const ys = used.map((o) => Math.log(Math.max(value(o), 1e-9)));
  const xb = used.reduce((a, b) => a + b, 0) / used.length;
  const yb = ys.reduce((a, b) => a + b, 0) / ys.length;
  let num = 0;
  let den = 0;
  used.forEach((o, i) => {
    num += (o - xb) * (ys[i] - yb);
    den += (o - xb) ** 2;
  });
  const slope = num / den;
  let ssTot = 0;
  let ssRes = 0;
  used.forEach((o, i) => {
    ssTot += (ys[i] - yb) ** 2;
    ssRes += (ys[i] - (yb + slope * (o - xb))) ** 2;
  });
  return { ratio: Math.exp(sign * slope), r2: ssTot > 0 ? 1 - ssRes / ssTot : 1, used };
}

process.stdout.write('drainage network\n  order   segments   mean length     Rb step\n');
for (const o of orders) {
  const n = nSeg.get(o)!;
  const l = lenSum.get(o)! / n / 1000;
  const prev = nSeg.get(o - 1);
  const stepRb = prev && n > 0 ? `${(prev / n).toFixed(2).padStart(7)}` : '      -';
  const thin = n < MIN_SEGMENTS ? '  (too few to fit)' : '';
  process.stdout.write(
    `  ${String(o).padStart(5)}   ${String(n).padStart(8)}   ${l.toFixed(0).padStart(8)} km   ${stepRb}${thin}\n`,
  );
}

const rb = horton((o) => nSeg.get(o)!, -1);
const rl = horton((o) => lenSum.get(o)! / nSeg.get(o)!, 1);

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
  bifurcation ratio Rb       ${rng(rb.ratio, EARTH.bifurcation)}   R2 ${rb.r2.toFixed(3)}
  length ratio Rl            ${rng(rl.ratio, EARTH.lengthRatio)}   R2 ${rl.r2.toFixed(3)}
                                             fitted over orders ${rb.used.join(', ')} of
                                             ${orders.join(', ')} — see MIN_SEGMENTS

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
  if (SEED !== DEFAULT_BAKE.tectonics.seed) {
    process.stdout.write(
      `\n  This is a different planet from the one the checked-in baselines describe.\n` +
        `  Reissue them before trusting npm run realism:\n` +
        `    npx tsx tools/findSites.ts        (rewrites the site table in src/tour.ts)\n` +
        `    npm run realism -- --update\n` +
        `  and set DEFAULT_TECTONICS.seed in src/bake/plates.ts if you want to keep it.\n`,
    );
  }
}
