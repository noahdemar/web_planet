/**
 * The drainage network as *curves*, and the distance field baked from them.
 *
 * ── why this is not a raster of cells ──────────────────────────────────────
 *
 * The network the LEM solves is a graph of cell-to-cell links, and a link goes
 * to one of eight neighbours. Follow it and you get a polyline whose every
 * segment is a multiple of 45°: a staircase. That is visible from orbit, and no
 * amount of filtering removes it, because the staircase is in the data and not
 * in the reconstruction. D∞ routing does not fix this either — it makes the
 * *share* of flow continuous, which is what the erosion needs, but the links it
 * splits between are still the same eight neighbours.
 *
 * So the network is lifted off the lattice before anything is measured from it.
 * The cells give the topology, which is the part the solve got right and the
 * part no local function could recover; a corner-cutting subdivision then
 * replaces the polyline with a smooth curve through the same valleys, and the
 * distance field is measured to *that*. A spline through a D8 path is not a D8
 * path, so the angularity is gone by construction rather than by threshold.
 *
 * The curve also has no texel size. That matters more than it sounds: the
 * distance to a curve is exact wherever it is evaluated, so the 9 km cell stops
 * being the limit on where the channel *is* and becomes only the limit on how
 * often the distance is sampled — and a distance field survives bilinear
 * magnification, which is the whole reason it is stored rather than a mask
 * (LESSONS §3).
 *
 * ── strands, not reaches ───────────────────────────────────────────────────
 *
 * A channel network is a tree, and smoothing it edge by edge would put a corner
 * at every confluence — including on the trunk, which is exactly where a corner
 * is most visible. Instead the tree is decomposed into *strands*: each one
 * follows the main stem (the largest contributing donor) from a source all the
 * way down, and a tributary is its own strand ending on the cell where it joins.
 * The trunk is then one continuous curve from headwater to sea and gets smoothed
 * as one, while junctions keep the sharp angle they physically have.
 */

import { cellAt, type Grid } from './grid.js';
import { MinHeap } from './heap.js';
import { RADIUS } from '../planet.js';
import type { Flow } from './lem.js';

export interface ChannelParams {
  /** Drainage area at which a channel head starts, m². */
  supportArea: number;
  /** Cap on the stored distance, metres. Beyond this the runtime sees "far". */
  maxDistance: number;
  widthCoeff: number;
  minWidth: number;
  maxWidth: number;
  depthCoeff: number;
  maxDepth: number;
  /**
   * Corner-cutting passes over each strand.
   *
   * Chaikin's algorithm converges to a quadratic B-spline, and the curve stays
   * inside the convex hull of the polyline it came from — so however many
   * passes are run, the river cannot wander out of the valley the LEM carved
   * for it. Two passes already break the 45° segments; three puts the residual
   * bearing spread below what a 9 km cell can express anyway.
   */
  smoothPasses: number;
}

export const DEFAULT_CHANNELS: ChannelParams = {
  // 10^10.1 m². Chosen from the measured MFD distribution rather than guessed:
  // multiple-flow accumulation puts the median land cell at 10^8.45, so a
  // threshold anywhere near that calls half the planet a river. 10^9.6 put a
  // channel head on roughly every valley the LEM resolved, which was denser
  // than anything visible from orbit; this keeps the trunks.
  supportArea: 10 ** 10.1,
  maxDistance: 20_000,
  // W ~ k*sqrt(Q). A 10^11 m² basin lands near 250 m.
  widthCoeff: 0.0008,
  minWidth: 120,
  maxWidth: 2_600,
  depthCoeff: 0.35,
  maxDepth: 60,
  smoothPasses: 3,
};

export interface Strands {
  /** Vertex directions, unit, packed xyz. */
  points: Float32Array;
  /** Drainage area carried at each vertex, m². */
  area: Float32Array;
  /** Index of the first vertex of each strand, plus a terminating total. */
  start: Int32Array;
}

/**
 * Decompose the channel network into main-stem strands.
 *
 * `flow.receiver` is the dominant D∞ link, which is the main stem by
 * construction — so "keep following the largest donor" and "keep following the
 * receiver" agree, and a strand is just a downstream walk that stops when the
 * cell it is walking into is fed mainly by somebody else.
 */
export function extractStrands(
  grid: Grid,
  z: Float32Array,
  flow: Flow,
  area: Float64Array,
  p: ChannelParams,
  seaLevel = 0,
): Strands {
  const { count } = grid;
  const isChannel = new Uint8Array(count);
  for (let c = 0; c < count; c++) {
    isChannel[c] = z[c] > seaLevel && area[c] >= p.supportArea ? 1 : 0;
  }

  // The largest channel donor of each cell — the one the strand continues
  // through. Everything else joining that cell starts a strand of its own.
  const mainDonor = new Int32Array(count).fill(-1);
  const mainArea = new Float64Array(count);
  for (let c = 0; c < count; c++) {
    if (!isChannel[c]) continue;
    const r = flow.receiver[c];
    if (r === c || !isChannel[r]) continue;
    if (area[c] > mainArea[r]) {
      mainArea[r] = area[c];
      mainDonor[r] = c;
    }
  }

  const pts: number[] = [];
  const areas: number[] = [];
  const start: number[] = [];

  const push = (c: number): void => {
    pts.push(grid.dirs[c * 3], grid.dirs[c * 3 + 1], grid.dirs[c * 3 + 2]);
    areas.push(area[c]);
  };

  for (let head = 0; head < count; head++) {
    if (!isChannel[head]) continue;
    // A strand starts at a source: a channel cell with no channel donor at all.
    if (mainDonor[head] >= 0) continue;

    start.push(areas.length);
    push(head);
    let c = head;
    for (;;) {
      const r = flow.receiver[c];
      if (r === c) break; // reached base level
      push(r);
      // Off the end of the network, or this strand is a tributary and has just
      // touched the trunk it joins. Either way it ends here — having included
      // the junction cell, so the curves meet rather than stopping short.
      if (!isChannel[r] || mainDonor[r] !== c) break;
      c = r;
    }
    // A two-cell strand has no corner to cut and no length worth drawing.
    if (areas.length - start[start.length - 1] < 3) {
      const s = start.pop()!;
      pts.length = s * 3;
      areas.length = s;
    }
  }
  start.push(areas.length);

  return smoothAll(
    Float32Array.from(pts),
    Float32Array.from(areas),
    Int32Array.from(start),
    p.smoothPasses,
  );
}

/**
 * Chaikin corner cutting on every strand, endpoints pinned.
 *
 * Each pass replaces every segment with its middle half, so a corner becomes
 * two shallower ones and the polyline converges on a smooth curve. Endpoints
 * are kept exactly where they were: a tributary has to still land on its trunk
 * and a trunk has to still reach the sea, and those are the two vertices whose
 * position is topology rather than shape.
 *
 * The drainage area rides along as a per-vertex attribute under the same
 * weights, so width still grows downstream on the smoothed curve.
 */
function smoothAll(
  points: Float32Array,
  area: Float32Array,
  start: Int32Array,
  passes: number,
): Strands {
  let p = points;
  let a = area;
  let s = start;

  for (let pass = 0; pass < passes; pass++) {
    const outP: number[] = [];
    const outA: number[] = [];
    const outS: number[] = [];
    for (let k = 0; k + 1 < s.length; k++) {
      const i0 = s[k];
      const i1 = s[k + 1];
      const n = i1 - i0;
      outS.push(outA.length);
      if (n < 3) {
        for (let i = i0; i < i1; i++) {
          outP.push(p[i * 3], p[i * 3 + 1], p[i * 3 + 2]);
          outA.push(a[i]);
        }
        continue;
      }
      // First vertex, unchanged.
      outP.push(p[i0 * 3], p[i0 * 3 + 1], p[i0 * 3 + 2]);
      outA.push(a[i0]);
      for (let i = i0; i + 1 < i1; i++) {
        const j = i + 1;
        for (const t of [0.25, 0.75]) {
          const x = p[i * 3] * (1 - t) + p[j * 3] * t;
          const y = p[i * 3 + 1] * (1 - t) + p[j * 3 + 1] * t;
          const zz = p[i * 3 + 2] * (1 - t) + p[j * 3 + 2] * t;
          // Back onto the sphere: the cut points sit on chords, and a chord
          // between cells 9 km apart is 1.6 mm below the surface — irrelevant
          // for distance, but the direction has to be a unit vector for cellAt.
          const m = Math.hypot(x, y, zz) || 1;
          outP.push(x / m, y / m, zz / m);
          outA.push(a[i] * (1 - t) + a[j] * t);
        }
      }
      // Last vertex, unchanged.
      outP.push(p[(i1 - 1) * 3], p[(i1 - 1) * 3 + 1], p[(i1 - 1) * 3 + 2]);
      outA.push(a[i1 - 1]);
    }
    outS.push(outA.length);
    p = Float32Array.from(outP);
    a = Float32Array.from(outA);
    s = Int32Array.from(outS);
  }

  return { points: p, area: a, start: s };
}

/**
 * Distance to the nearest channel curve or coast, metres, and the drainage the
 * nearest channel carries.
 *
 * Two stages, because the accuracy is only needed where the field is read
 * closely. Every cell within a couple of rings of a curve gets the *exact*
 * point-to-segment distance, which is what the carve and the river tint sample
 * and what has to be sub-cell to place a 250 m channel inside a 9 km texel.
 * Beyond that a Dijkstra relaxation over the grid's own neighbour distances
 * fills out to the cap, where the only consumer is a broad falloff.
 *
 * The second output is new and worth its four bytes: the carve used to size
 * itself from the drainage of the cell being carved, so a cell two cells off
 * the axis widened the valley by *its own* trickle rather than by the river
 * that made it. Carrying the area from the nearest curve point fixes that.
 */
export function channelDistance(
  grid: Grid,
  z: Float32Array,
  strands: Strands,
  p: ChannelParams,
  seaLevel = 0,
): { dist: Float32Array; nearArea: Float32Array } {
  const { count } = grid;
  const dist = new Float32Array(count).fill(p.maxDistance);
  const nearArea = new Float32Array(count);

  // ── exact, near the curve ────────────────────────────────────────────────
  //
  // Walked segment by segment rather than cell by cell: a segment knows which
  // cells are near it (flood out from the one holding its midpoint) far more
  // cheaply than a cell knows which segments are near it.
  const RINGS = 2;
  // One monotonic id per segment across the whole network. Using the
  // per-strand vertex index here would collide between strands and silently
  // skip cells that a later strand should have measured.
  const stamp = new Int32Array(count).fill(-1);
  let visitId = 0;
  const frontier = new Int32Array(512);
  const next = new Int32Array(512);

  const pts = strands.points;
  const areas = strands.area;
  for (let k = 0; k + 1 < strands.start.length; k++) {
    const i0 = strands.start[k];
    const i1 = strands.start[k + 1];
    for (let i = i0; i + 1 < i1; i++) {
      const ax = pts[i * 3];
      const ay = pts[i * 3 + 1];
      const az = pts[i * 3 + 2];
      const bx = pts[(i + 1) * 3];
      const by = pts[(i + 1) * 3 + 1];
      const bz = pts[(i + 1) * 3 + 2];
      const segArea = Math.max(areas[i], areas[i + 1]);

      const mid = cellAt(grid, [(ax + bx) / 2, (ay + by) / 2, (az + bz) / 2]);
      const id = visitId++;
      let nf = 0;
      frontier[nf++] = mid;
      stamp[mid] = id;

      for (let ring = 0; ring <= RINGS; ring++) {
        let nn = 0;
        for (let f = 0; f < nf; f++) {
          const c = frontier[f];
          // Exact distance from this cell centre to the segment.
          const px = grid.dirs[c * 3];
          const py = grid.dirs[c * 3 + 1];
          const pz = grid.dirs[c * 3 + 2];
          const ux = bx - ax;
          const uy = by - ay;
          const uz = bz - az;
          const uu = ux * ux + uy * uy + uz * uz;
          let t = uu > 0 ? ((px - ax) * ux + (py - ay) * uy + (pz - az) * uz) / uu : 0;
          t = t < 0 ? 0 : t > 1 ? 1 : t;
          const dx = px - (ax + ux * t);
          const dy = py - (ay + uy * t);
          const dz = pz - (az + uz * t);
          // Chord for arc: 8 mm out at 20 km on this radius.
          const d = Math.sqrt(dx * dx + dy * dy + dz * dz) * RADIUS;
          if (d < dist[c]) {
            dist[c] = d;
            nearArea[c] = segArea;
          }
          if (ring === RINGS) continue;
          for (let q = 0; q < 8; q++) {
            const n = grid.nbr[c * 8 + q];
            if (stamp[n] === id) continue;
            stamp[n] = id;
            if (nn < next.length) next[nn++] = n;
          }
        }
        nf = nn;
        for (let f = 0; f < nf; f++) frontier[f] = next[f];
        if (nf === 0) break;
      }
    }
  }

  // ── the coast ────────────────────────────────────────────────────────────
  //
  // So a coastal cell measures its distance to the sea rather than inland to
  // the nearest river. The shoreline ring only: seeding every submerged cell
  // put 4.5 M sources on the heap and relaxed the entire sea floor, for a field
  // only ever read on land.
  const heap = new MinHeap(1 << 16);
  for (let c = 0; c < count; c++) {
    if (z[c] > seaLevel) {
      if (dist[c] < p.maxDistance) heap.push(dist[c], c);
      continue;
    }
    for (let q = 0; q < 8; q++) {
      if (z[grid.nbr[c * 8 + q]] > seaLevel) {
        dist[c] = 0;
        heap.push(0, c);
        break;
      }
    }
  }

  // ── approximate, further out ─────────────────────────────────────────────
  while (heap.size > 0) {
    const key = heap.peekKey();
    const c = heap.pop();
    if (key > dist[c]) continue; // stale entry
    for (let q = 0; q < 8; q++) {
      const n = grid.nbr[c * 8 + q];
      // Land only. Nothing reads the distance under water, and relaxing into
      // the sea is what made this the slowest pass in the bake.
      if (z[n] <= seaLevel) continue;
      const nd = key + grid.nbrDist[c * 8 + q];
      if (nd < dist[n]) {
        dist[n] = nd;
        if (nearArea[n] === 0) nearArea[n] = nearArea[c];
        if (nd < p.maxDistance) heap.push(nd, n);
      }
    }
  }

  return { dist, nearArea };
}

/**
 * Cut the channels into the surface.
 *
 * A bowl whose depth and width come from the drainage the *channel* carries,
 * tapering out over a couple of widths, so the banks have a shape rather than
 * being a step. Unchanged in form from the raster version — what changed is
 * that `dist` is now measured to a smooth curve, so the valley floor it cuts is
 * a smooth curve too.
 */
export function carveChannels(
  grid: Grid,
  z: Float32Array,
  dist: Float32Array,
  nearArea: Float32Array,
  p: ChannelParams,
  seaLevel = 0,
): void {
  // ── the carve has to be wider than the grid it is sampled on ─────────────
  //
  // A channel's own half-width is 2.6 km at most, and a cell here is 9 to 18
  // km. Carving that profile directly means sampling it below Nyquist, and the
  // raster version got away with it only by accident: its distance came from a
  // Dijkstra over cell links, so a channel cell read exactly 0 and its
  // neighbours read a whole cell, and the carve was really "lower the channel
  // cells". An exact distance to a curve does not quantise like that — a cell
  // centre sits anywhere from 0 to half a cell off the curve — so the carve
  // depth flickered *along* the channel, left pits the LEM never filled, and
  // the re-routed network collapsed: Strahler order 3 went from 167 segments
  // to 9 and orders 4 and 5 disappeared entirely.
  //
  // The fix is to carve what this grid can actually represent, which is the
  // valley, not the channel. The runtime cuts the channel itself from the same
  // distance field, where it has the resolution to do it (see channel_ and
  // CHANNEL_HALF_LO in planet.ts). So the floor here is a cell, and the
  // hydraulic width only matters where it exceeds one.
  const floor = grid.spacing * 1.15;

  for (let c = 0; c < grid.count; c++) {
    if (z[c] <= seaLevel) continue;
    const a = nearArea[c];
    if (a <= 1) continue;
    const w = Math.min(p.maxWidth, Math.max(p.minWidth, p.widthCoeff * Math.sqrt(a)));
    const half = Math.max(w * 2.0, floor);
    const t = dist[c] / half;
    if (t >= 1) continue;
    const depth = Math.min(p.maxDepth, p.depthCoeff * Math.pow(a, 0.2));
    z[c] -= depth * (1 - t * t);
  }
}

/**
 * How much the network *turns*, per unit of its length.
 *
 * The instrument for the whole exercise, and it took two attempts. The first
 * measured the distribution of segment *bearings* against the cube face's own
 * axes, on the theory that a lattice network can only run at multiples of 45°.
 * It cannot: the tangent warp makes cells non-square away from a face centre,
 * so a diagonal step is at 43° here and 52° there, and the check read 39% where
 * it should have read 100% on a network with no smoothing at all. It was
 * measuring the warp.
 *
 * Total curvature has no such problem. It is a property of the curve and not of
 * any frame: sum the turn at every interior vertex and divide by the length, and
 * a staircase and a smooth line through the same cells are unmistakable. A D8
 * path at this cell size turns 45° every cell or two, which is hundreds of
 * degrees per 100 km; a curve through the same cells turns by the amount the
 * valley actually bends.
 *
 * `sharpShare` is the second half of the question — *how* the turning is
 * delivered. A meandering river and a staircase can total the same curvature,
 * one as a continuous bend and the other as a few hard corners, and it is the
 * corners that read as artificial. So: what fraction of all turning arrives in
 * single steps of more than 20°.
 */
export function cornerStats(strands: Strands): {
  turnPer100km: number;
  sharpShare: number;
  medianSegment: number;
} {
  const pts = strands.points;
  let totalTurn = 0;
  let sharpTurn = 0;
  let totalLen = 0;
  const segLens: number[] = [];

  for (let k = 0; k + 1 < strands.start.length; k++) {
    const i0 = strands.start[k];
    const i1 = strands.start[k + 1];
    let px = 0, py = 0, pz = 0, plen = 0;
    for (let i = i0; i + 1 < i1; i++) {
      let sx = pts[(i + 1) * 3] - pts[i * 3];
      let sy = pts[(i + 1) * 3 + 1] - pts[i * 3 + 1];
      let sz = pts[(i + 1) * 3 + 2] - pts[i * 3 + 2];
      const len = Math.hypot(sx, sy, sz);
      if (len < 1e-12) continue;
      sx /= len; sy /= len; sz /= len;
      const metres = len * RADIUS;
      totalLen += metres;
      segLens.push(metres);
      if (plen > 0) {
        const dot = Math.max(-1, Math.min(1, sx * px + sy * py + sz * pz));
        const turn = (Math.acos(dot) * 180) / Math.PI;
        totalTurn += turn;
        if (turn > 20) sharpTurn += turn;
      }
      px = sx; py = sy; pz = sz; plen = len;
    }
  }
  segLens.sort((a, b) => a - b);
  return {
    turnPer100km: (totalTurn / Math.max(totalLen, 1)) * 100_000,
    sharpShare: sharpTurn / Math.max(totalTurn, 1e-9),
    medianSegment: segLens.length ? segLens[segLens.length >> 1] : 0,
  };
}
