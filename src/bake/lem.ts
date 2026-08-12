/**
 * B2 — landscape evolution. See SPEC.md §4.
 *
 * Stream-power incision with hillslope diffusion, solved on the global grid:
 *
 *     ∂z/∂t = U − K·A^m·S^n + D·∇²z
 *
 * The middle term is the one that matters. It says erosion rate scales with
 * upstream drainage area, which is why rivers cut valleys, why valleys have
 * tributaries at consistent junction angles, and why ridges sit exactly
 * halfway between adjacent channels. Every one of those is a *global*
 * property — a cell's fate depends on how much of the planet drains through
 * it — which is why this cannot be done per-tile at runtime and has to be
 * baked (SPEC.md §1).
 *
 * Three pieces of machinery make it tractable:
 *
 *   1. Priority-Flood+ε (Barnes 2014) to resolve depressions, so every land
 *      cell has a strictly descending path to the sea. Without this the flow
 *      graph is disconnected and drainage area is nonsense over most of the
 *      continents.
 *   2. Braun & Willett's O(n) ordering: build the receiver tree, walk it
 *      depth-first into a stack, and both accumulation and the incision solve
 *      become single linear passes.
 *   3. An *implicit* incision update, which is unconditionally stable in Δt.
 *      That is what allows 10⁵-year steps and a bake measured in seconds
 *      rather than the 10³-year steps an explicit scheme would need.
 */

import { type Grid } from './grid.js';
import { MinHeap } from './heap.js';

export interface LemParams {
  /** Erodibility, yr⁻¹ scaled by the area exponent. Sets total relief. */
  K: number;
  /** Area exponent. 0.4–0.6 in the field; 0.5 with n = 1 is the classic pair. */
  m: number;
  /** Slope exponent. */
  n: number;
  /** Hillslope diffusivity, m²/yr. At this cell size it stands in for all
   *  sub-grid mass wasting, so it is far above a true soil-creep value. */
  D: number;
  /**
   * Courant number for the diffusion sub-steps. The critical-slope flux
   * diverges near the threshold, so the explicit update needs a far smaller
   * step than the (unconditionally stable) implicit incision does, and the
   * number of sub-steps is derived from this rather than guessed — a fixed
   * count silently goes unstable the moment resolution or D changes, and an
   * unstable diffusion *adds* relief, which is indistinguishable from the
   * model simply producing tall mountains.
   */
  diffuseCourant: number;
  /** Timestep, years. */
  dt: number;
  /** Number of steps. */
  steps: number;
  /** Re-route flow every this many steps. Routing dominates the cost. */
  rerouteEvery: number;
  /**
   * Fraction of eroded rock mass returned as isostatic rebound. Physically
   * ρ_crust/ρ_mantle ≈ 0.83: strip a kilometre off a range and the root
   * floats it most of the way back up. Without this, erosion is a one-way
   * loss and every mountain belt ends the run as a low plain.
   */
  isostasy: number;
  /**
   * Flexural smoothing length for that rebound, metres. The lithosphere is a
   * plate, not a stack of independent columns — it responds over ~100 km. At
   * zero the rebound would just locally undo the erosion that caused it and
   * nothing would ever incise.
   */
  flexure: number;
  /** Elevation above which the absolute envelope starts to compress, metres. */
  ceilingSoft: number;
  /** Asymptotic maximum elevation, metres. Earth's is 8848. */
  ceilingMax: number;
}

export const DEFAULT_LEM: LemParams = {
  K: 2.8e-6,
  m: 0.5,
  n: 1,
  D: 45,
  dt: 2.0e5,
  diffuseCourant: 0.35,
  steps: 140,
  rerouteEvery: 6,
  isostasy: 0.83,
  flexure: 140_000,
  ceilingSoft: 4600,
  ceilingMax: 9100,
};

export interface Flow {
  /** Steepest-descent receiver. Equal to the cell itself at base level. */
  receiver: Int32Array;
  /** Distance to the receiver, metres. */
  length: Float32Array;
  /** Cells in downstream-to-upstream order. */
  stack: Int32Array;
  /** Upstream drainage area, m². */
  area: Float64Array;
}

/** Cardinal neighbour slots in the grid's D8 ordering: S, W, E, N. */
const CARDINAL = [1, 3, 4, 6];

/**
 * How much the depression filler must raise a cell before it counts as a lake,
 * metres. The filler's own ε is 1 mm per cell, so this only has to be well
 * clear of that; 4 m is also about the shallowest standing water worth drawing
 * at a 9 km cell.
 */
const LAKE_MIN_DEPTH = 4;

/** Per-cell surface area, from the local neighbour spacing. */
export function cellAreas(grid: Grid): Float64Array {
  const a = new Float64Array(grid.count);
  for (let c = 0; c < grid.count; c++) {
    const dy = 0.5 * (grid.nbrDist[c * 8 + 1] + grid.nbrDist[c * 8 + 6]);
    const dx = 0.5 * (grid.nbrDist[c * 8 + 3] + grid.nbrDist[c * 8 + 4]);
    a[c] = dx * dy;
  }
  return a;
}

/**
 * Priority-Flood+ε: raise every closed depression to its spill point plus a
 * vanishing increment, so flow routing always finds a strictly descending
 * path without visibly altering the surface.
 *
 * The plain FIFO alongside the heap is Barnes' optimisation and it is not
 * cosmetic: cells that are already at or above the current flood level cannot
 * lower it, so they never need to be ordered. On a real landscape that is the
 * large majority of cells, and it takes the routing pass from the dominant
 * cost of the bake to roughly a third of it.
 */
export function fillDepressions(grid: Grid, z: Float32Array, seaLevel = 0): void {
  const { count } = grid;
  const closed = new Uint8Array(count);
  const heap = new MinHeap(1 << 16);
  const fifo = new Int32Array(count);
  let head = 0;
  let tail = 0;

  // The ocean is the boundary condition: everything drains to it.
  for (let c = 0; c < count; c++) {
    if (z[c] <= seaLevel) {
      closed[c] = 1;
      heap.push(z[c], c);
    }
  }

  // Nothing below sea level anywhere — seed from the single lowest cell so the
  // flood still has somewhere to start.
  if (heap.size === 0) {
    let lo = 0;
    for (let c = 1; c < count; c++) if (z[c] < z[lo]) lo = c;
    closed[lo] = 1;
    heap.push(z[lo], lo);
  }

  const EPS = 1e-3; // metres per cell; 1 mm of gradient is invisible

  while (heap.size > 0 || head < tail) {
    let c: number;
    if (head < tail && (heap.size === 0 || z[fifo[head]] <= heap.peekKey())) {
      c = fifo[head++];
    } else {
      c = heap.pop();
    }

    const zc = z[c];
    for (let k = 0; k < 8; k++) {
      const n = grid.nbr[c * 8 + k];
      if (closed[n]) continue;
      closed[n] = 1;
      if (z[n] <= zc) {
        z[n] = zc + EPS;
        fifo[tail++] = n;
      } else {
        heap.push(z[n], n);
      }
    }
  }
}

/**
 * Receivers, topological stack and drainage area for the current surface.
 *
 * `z` must already be depression-free. The stack is a depth-first ordering of
 * the receiver forest, so a downstream cell always precedes its donors:
 * incision can then be solved in one forward pass and area in one backward
 * pass, both O(n).
 */
export function routeFlow(
  grid: Grid,
  z: Float32Array,
  area0: Float64Array,
  seaLevel = 0,
): Flow {
  const { count } = grid;
  const receiver = new Int32Array(count);
  const length = new Float32Array(count);

  for (let c = 0; c < count; c++) {
    if (z[c] <= seaLevel) {
      receiver[c] = c; // base level
      length[c] = 1;
      continue;
    }
    let best = c;
    let bestSlope = 0;
    let bestLen = 1;
    for (let k = 0; k < 8; k++) {
      const n = grid.nbr[c * 8 + k];
      const d = grid.nbrDist[c * 8 + k];
      const s = (z[c] - z[n]) / d;
      if (s > bestSlope) {
        bestSlope = s;
        best = n;
        bestLen = d;
      }
    }
    receiver[c] = best;
    length[c] = bestLen;
  }

  // Donor lists in CSR form: count, prefix-sum, scatter.
  const ndonor = new Int32Array(count + 1);
  for (let c = 0; c < count; c++) if (receiver[c] !== c) ndonor[receiver[c]]++;
  const offset = new Int32Array(count + 1);
  let acc = 0;
  for (let c = 0; c < count; c++) {
    offset[c] = acc;
    acc += ndonor[c];
  }
  offset[count] = acc;
  const cursor = offset.slice();
  const donor = new Int32Array(acc);
  for (let c = 0; c < count; c++) if (receiver[c] !== c) donor[cursor[receiver[c]]++] = c;

  // Depth-first from every base-level cell.
  const stack = new Int32Array(count);
  let sp = 0;
  const work = new Int32Array(count);
  let wp = 0;
  for (let c = 0; c < count; c++) if (receiver[c] === c) work[wp++] = c;
  while (wp > 0) {
    const c = work[--wp];
    stack[sp++] = c;
    for (let i = offset[c]; i < offset[c + 1]; i++) work[wp++] = donor[i];
  }

  // Every cell must appear exactly once. A short stack means the receiver
  // graph has a cycle, which means depression filling did not complete —
  // silently continuing would give plausible-looking but wrong drainage.
  if (sp !== count) {
    throw new Error(`routeFlow: stack has ${sp} of ${count} cells — flow graph has a cycle`);
  }

  const area = new Float64Array(count);
  for (let c = 0; c < count; c++) area[c] = area0[c];
  for (let i = count - 1; i >= 0; i--) {
    const c = stack[i];
    const r = receiver[c];
    if (r !== c) area[r] += area[c];
  }

  return { receiver, length, stack, area };
}

/**
 * Upstream drainage area by multiple flow direction (Freeman 1991).
 *
 * The erosion wants D8: stream power is a channel process, the Braun–Willett
 * O(n) solve needs a receiver *tree*, and the elevation it produces measures
 * out at only 6% grid-anisotropic, so single-flow is doing no visible harm
 * there. The rendered *wetness* is a different question. D8 gives every cell
 * exactly one of eight directions, so a trunk river drawn from it is a
 * polyline of 0°, 45° and 90° segments — a staircase, and at 573 km altitude
 * that is precisely what it looked like.
 *
 * MFD splits each cell's accumulation across every downslope neighbour in
 * proportion to slope^p. The direction quantisation disappears because flow is
 * no longer a choice between eight options; it is a weighted average of them,
 * and the average moves continuously as the surface tilts. It also spreads the
 * corridor across the valley floor rather than concentrating it in one cell,
 * which is both what a riparian zone actually looks like and what stops the
 * field from being a one-texel line that no amount of filtering can make look
 * like anything but a one-texel line.
 *
 * p = 1.1 is Freeman's fitted value: p → ∞ recovers D8, p → 0 spreads flow
 * evenly downhill and dissolves the network entirely.
 *
 * Ordering is by descending elevation over the whole grid, which is what makes
 * one pass sufficient — a cell's total is final before any of its receivers is
 * visited. The D8 stack cannot serve here: it orders the receiver *tree*, and
 * MFD sends flow along links that tree does not contain.
 */
export function accumulateMFD(
  grid: Grid,
  z: Float32Array,
  area0: Float64Array,
  seaLevel = 0,
): Float64Array {
  const { count } = grid;
  const acc = new Float64Array(count);
  for (let c = 0; c < count; c++) acc[c] = area0[c];

  // Descending elevation. A comparison sort of 6.3M indices is a few seconds;
  // a bucket sort on a 16-bit key is a few tens of milliseconds, and one metre
  // of key resolution is far finer than any slope this matters for.
  let lo = Infinity;
  let hi = -Infinity;
  for (let c = 0; c < count; c++) {
    if (z[c] < lo) lo = z[c];
    if (z[c] > hi) hi = z[c];
  }
  const BUCKETS = 1 << 16;
  const scale = (BUCKETS - 1) / Math.max(hi - lo, 1e-6);
  const hist = new Int32Array(BUCKETS + 1);
  const key = new Int32Array(count);
  for (let c = 0; c < count; c++) {
    // Descending: bucket 0 is the highest ground.
    const k = BUCKETS - 1 - Math.min(BUCKETS - 1, Math.max(0, Math.round((z[c] - lo) * scale)));
    key[c] = k;
    hist[k + 1]++;
  }
  for (let b = 0; b < BUCKETS; b++) hist[b + 1] += hist[b];
  const order = new Int32Array(count);
  const cursor = hist.slice();
  for (let c = 0; c < count; c++) order[cursor[key[c]]++] = c;

  const P = 1.1;
  const w = new Float64Array(8);
  for (let i = 0; i < count; i++) {
    const c = order[i];
    // The sea absorbs; it does not route. Without this, coastal cells push
    // their area along the shore and the ocean lights up as one huge basin.
    if (z[c] <= seaLevel) continue;
    const zc = z[c];
    let tot = 0;
    for (let k = 0; k < 8; k++) {
      const n = grid.nbr[c * 8 + k];
      const s = (zc - z[n]) / grid.nbrDist[c * 8 + k];
      // Cells within a bucket are visited in arbitrary order, so a neighbour at
      // the same key may not be settled yet. Requiring a strictly positive
      // slope keeps the pass acyclic regardless — z is depression-free, so
      // every land cell has at least one.
      const ww = s > 0 ? Math.pow(s, P) : 0;
      w[k] = ww;
      tot += ww;
    }
    if (tot <= 0) continue;
    const a = acc[c] / tot;
    for (let k = 0; k < 8; k++) if (w[k] > 0) acc[grid.nbr[c * 8 + k]] += a * w[k];
  }

  return acc;
}

/**
 * One implicit stream-power step over the whole grid.
 *
 * For n = 1 the update is closed-form. For n ≠ 1 it needs Newton iterations
 * per cell, which is why n = 1 is the default: it is within the observed
 * range and costs a fraction as much.
 */
function incise(
  z: Float32Array,
  flow: Flow,
  uplift: Float32Array,
  erodibility: Float32Array,
  p: LemParams,
  seaLevel: number,
): void {
  const { stack, receiver, length, area } = flow;
  const { K, m, n, dt } = p;

  for (let i = 0; i < stack.length; i++) {
    const c = stack[i];
    const r = receiver[c];
    if (r === c) continue; // base level is held fixed
    if (z[c] <= seaLevel) continue; // no fluvial incision below the sea

    const zr = z[r];
    const zi = z[c] + dt * uplift[c];
    const f = K * erodibility[c] * dt * Math.pow(area[c], m) / length[c];

    if (n === 1) {
      z[c] = (zi + f * zr) / (1 + f);
    } else {
      // Newton on  z + f·(z − zr)^n − zi = 0, from the explicit guess.
      let x = Math.max(zr, zi);
      for (let it = 0; it < 6; it++) {
        const d = Math.max(x - zr, 1e-9);
        const g = x + f * Math.pow(d, n) - zi;
        const gp = 1 + f * n * Math.pow(d, n - 1);
        const nx = x - g / gp;
        if (Math.abs(nx - x) < 1e-4) {
          x = nx;
          break;
        }
        x = nx;
      }
      z[c] = Math.max(zr, x);
    }
    // Rivers cut down, they do not cut below their own outlet.
    if (z[c] < zr) z[c] = zr;
  }
}

/**
 * Threshold-hillslope cap on relief across a single flow link.
 *
 * Stream power alone puts no bound on headwater relief: erosion scales with
 * drainage area, a channel head has almost none, so an uplifting ridge crest
 * grows without limit. Left alone this model reached a 15.4 km summit.
 *
 * Real landscapes are capped by mass wasting — bedrock landslides strip
 * anything past the threshold angle — so above some steepness relief stops
 * responding to uplift at all. At the cell sizes here a link spans tens of
 * kilometres, so the bound is not the 33° landslide angle but the observed
 * maximum *relief over a given length* in the steepest terrain on Earth,
 * which goes as roughly L^0.6. Writing it that way keeps the cap honest as
 * resolution changes, instead of baking in one grid spacing.
 */
function capRelief(z: Float32Array, flow: Flow, p: LemParams, seaLevel: number): void {
  const { stack, receiver, length } = flow;
  const h0 = p.ceilingSoft;
  const span = p.ceilingMax - h0;

  for (let i = 0; i < stack.length; i++) {
    const c = stack[i];
    const r = receiver[c];
    if (r === c) continue;
    if (z[c] <= seaLevel) continue;
    // 1500 m over 10 km, growing as L^0.6 — the Himalayan front, which is the
    // steepest sustained relief measured anywhere.
    const maxRise = 1500 * Math.pow(length[c] / 10_000, 0.6);
    const lim = z[r] + maxRise;
    if (z[c] > lim) z[c] = lim;

    // Absolute envelope. This one is empirical, not derived: the per-link cap
    // bounds each step of a chain but not the chain, so a long enough run of
    // maximum-steepness links still reaches 26 km. Crustal strength does put a
    // real ceiling on mountain height — nothing on Earth exceeds 8.85 km — and
    // this is that ceiling, applied as a smooth compression rather than a clip
    // so it cannot manufacture the flat-topped plateaus this whole stage
    // exists to get rid of.
    if (z[c] > h0) z[c] = h0 + span * Math.tanh((z[c] - h0) / span);
  }
}

/**
 * Critical-slope hillslope transport (Roering et al. 1999):
 *
 *     q = D·S / (1 − (S/Sc)²)
 *
 * Linear diffusion cannot limit mountain relief. Its flux is proportional to
 * slope, while stream power's is proportional to A^m·S — and at every cell
 * size used here the stream-power term is an order of magnitude larger, so
 * relief simply kept growing with resolution: 14.5 km peaks at 72 km cells,
 * 20.4 km at 18 km cells. A model whose mountains depend on grid spacing is
 * measuring the grid.
 *
 * The nonlinear form fixes that at the source rather than clipping the
 * result. Below Sc it behaves like ordinary creep; as slope approaches Sc the
 * flux diverges, so no amount of uplift can push a hillslope past the
 * threshold angle. That is what actually caps mountains on Earth: bedrock
 * landsliding, not fluvial incision.
 *
 * Sc is the maximum *mean* gradient over one cell, so it depends on cell
 * size. It comes from the observed relief-over-length envelope — about 1500 m
 * across 10 km in the steepest terrain anywhere, growing as L^0.6 — which
 * makes the cap resolution-aware instead of tuned to one grid.
 */
function diffuse(grid: Grid, z: Float32Array, tmp: Float32Array, p: LemParams, seaLevel: number): void {
  const { count } = grid;
  // Stability for the 4-point explicit Laplacian is 4·D·dt/d² ≤ 1/2. With the
  // critical-slope amplification capped at 1/(1−0.92) = 12.5, the worst-case
  // diffusivity is 12.5·D and the smallest cell sets d.
  const dMin = grid.minSpacing;
  const dtMax = (p.diffuseCourant * dMin * dMin) / (4 * p.D * 12.5);
  const sub = Math.max(1, Math.ceil(p.dt / dtMax));
  const dt = p.dt / sub;

  for (let s = 0; s < sub; s++) {
    tmp.set(z);
    for (let c = 0; c < count; c++) {
      if (tmp[c] <= seaLevel) continue;
      let flux = 0;
      for (let i = 0; i < 4; i++) {
        const k = CARDINAL[i];
        const nn = grid.nbr[c * 8 + k];
        const d = grid.nbrDist[c * 8 + k];
        const dz = tmp[nn] - tmp[c];
        const sc = 1500 * Math.pow(d / 10_000, 0.6) / d;
        const r = Math.min(0.92, (dz / d / sc) ** 2); // 0.92 keeps it finite
        flux += (p.D / (1 - r)) * dz / (d * d);
      }
      z[c] = Math.max(seaLevel, tmp[c] + dt * flux);
    }
  }
}

/**
 * Flexural isostatic response to the rock removed this step.
 *
 * The lithosphere unloads as a plate, over ~100 km, not column by column. So
 * a cell rises by the *regional average* of erosion and falls by its own:
 *
 *     Δz = isostasy · (⟨E⟩_flexural − E_local)
 *
 * This is the Molnar & England (1990) effect. Valleys erode faster than the
 * ridges between them, the regional load drops, and the whole block floats
 * up — so summits gain elevation *while* the range is being destroyed. It is
 * why glaciated ranges have summits above their pre-glacial height, and it is
 * the mechanism that turns a smooth uplifted dome into sharp peaks.
 *
 * Writing it as a deviation rather than as `isostasy · ⟨E⟩` matters. The
 * latter is not wrong physically, but then `uplift` means *tectonic* rock
 * uplift and steady-state erosion runs at U/(1 − isostasy) ≈ 5.9 U — which
 * silently multiplies every calibrated uplift rate by six. The deviation form
 * is mean-preserving, so `uplift` keeps meaning surface uplift and the
 * hypsometry stays where it was tuned.
 */
function rebound(
  grid: Grid,
  z: Float32Array,
  removed: Float32Array,
  scratch: Float32Array,
  p: LemParams,
  seaLevel: number,
): void {
  // Repeated cardinal box-blur approximates a Gaussian of the target width;
  // three passes is close enough and each is a single linear sweep.
  const passes = 3;
  const sigmaPerPass = p.flexure / Math.sqrt(passes);
  const w = Math.min(0.24, (sigmaPerPass * sigmaPerPass) / (grid.spacing * grid.spacing) / 4);

  const smooth = scratch;
  smooth.set(removed);
  for (let it = 0; it < passes; it++) {
    for (let c = 0; c < grid.count; c++) {
      let s = 0;
      for (let i = 0; i < 4; i++) s += smooth[grid.nbr[c * 8 + CARDINAL[i]]];
      smooth[c] = smooth[c] + w * (s - 4 * smooth[c]);
    }
  }

  for (let c = 0; c < grid.count; c++) {
    if (z[c] <= seaLevel) continue;
    z[c] += p.isostasy * (smooth[c] - removed[c]);
  }
}

export interface LemResult {
  /** Final elevation, metres relative to sea level. */
  z: Float32Array;
  /** Upstream drainage area at the final state, m². */
  area: Float64Array;
  /** Receivers at the final state, for channel extraction. */
  flow: Flow;
  /**
   * Standing-water surface, metres. Equal to `z` on dry ground, above it over
   * a lake. See the note on the final conditioning pass in runLEM.
   */
  water: Float32Array;
}


/**
 * Distance to the nearest channel, metres, and the channel carved in.
 *
 * ── Why this is baked and not reconstructed ──
 *
 * The runtime used to find the channel axis from the wetness field's own
 * derivatives: near its axis wetness is a smooth ridge, so the transverse
 * gradient vanishes there and its magnitude is proportional to the distance.
 * That is true, and it does not work. The estimator is built from a 9 km
 * stencil, so it is noisy at exactly the scale the channel lives at, and the
 * river came out as a chain of ponds — the mask flickered on and off along its
 * own length. Widening it until it was stable gave a river 10 km across.
 *
 * The flow network is *already contiguous*: it is a tree, every channel cell
 * has a receiver, and it was solved properly. Throwing that away at bake time
 * and trying to recover it from a gradient at runtime was the mistake. So this
 * measures the distance directly — a multi-source Dijkstra out from every
 * channel cell, over the grid's own neighbour distances — and hands the
 * runtime a field whose zero set *is* the network.
 *
 * A distance field is also the right thing to interpolate. Bilinear filtering
 * of a distance is still a distance to within the cell size, so the runtime
 * can place a 200 m river inside a 9 km texel and have it land in the right
 * place and stay connected. Bilinear filtering of a *mask* cannot do that, and
 * that is why every threshold-based attempt produced texel-shaped blobs.
 *
 * The carve is the same field applied to the elevation: a channel that the
 * water sits in rather than on.
 */
export function carveChannels(
  grid: Grid,
  z: Float32Array,
  area: Float64Array,
  p: ChannelParams,
  seaLevel = 0,
): Float32Array {
  const { count } = grid;
  const dist = new Float32Array(count).fill(p.maxDistance);
  const heap = new MinHeap(1 << 16);

  // Sources: land cells carrying enough drainage to be a channel — the support
  // area threshold, which is the standard definition of a channel head — plus
  // the *shoreline*, so a coastal cell measures its distance to the sea rather
  // than inland to the nearest river.
  //
  // The shoreline, not the whole ocean. Seeding every submerged cell is the
  // obvious way to write that and it made the bake unusable: 4.5 M sources on
  // the heap and essentially the entire sea floor relaxed, for a field only
  // ever read on land. One ring of cells does the same job.
  for (let c = 0; c < count; c++) {
    if (z[c] > seaLevel) {
      if (area[c] >= p.supportArea) {
        dist[c] = 0;
        heap.push(0, c);
      }
      continue;
    }
    for (let k = 0; k < 8; k++) {
      if (z[grid.nbr[c * 8 + k]] > seaLevel) {
        dist[c] = 0;
        heap.push(0, c);
        break;
      }
    }
  }

  while (heap.size > 0) {
    const key = heap.peekKey();
    const c = heap.pop();
    // Stale entry: this cell was already settled at a shorter distance.
    if (key > dist[c]) continue;
    for (let k = 0; k < 8; k++) {
      const n = grid.nbr[c * 8 + k];
      // Land only. Nothing reads the distance under water, and relaxing into
      // the sea is what made this the slowest pass in the bake.
      if (z[n] <= seaLevel) continue;
      const nd = key + grid.nbrDist[c * 8 + k];
      if (nd < dist[n]) {
        dist[n] = nd;
        if (nd < p.maxDistance) heap.push(nd, n);
      }
    }
  }

  // Carve. Width and depth grow with the drainage the *channel* carries, and
  // the profile is a parabola in the distance so the banks have a shape rather
  // than being a step. Only cells inside a couple of widths are touched, which
  // at this resolution is a handful either side of the axis.
  for (let c = 0; c < count; c++) {
    if (z[c] <= seaLevel) continue;
    const a = Math.max(area[c], 1);
    // Hydraulic geometry, widened to what a 9 km cell can express.
    const w = Math.min(p.maxWidth, Math.max(p.minWidth, p.widthCoeff * Math.sqrt(a)));
    const t = dist[c] / (w * 2.0);
    if (t >= 1) continue;
    const depth = Math.min(p.maxDepth, p.depthCoeff * Math.pow(a, 0.2));
    z[c] -= depth * (1 - t * t);
  }

  return dist;
}

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
}

export const DEFAULT_CHANNELS: ChannelParams = {
  // 10^9.6 m². Chosen from the measured MFD distribution rather than guessed:
  // multiple-flow accumulation puts the median land cell at 10^8.45, so a
  // threshold anywhere near that calls half the planet a river.
  // Raised from 10^9.6. That threshold put a channel head on roughly every
  // valley the LEM resolved; the network it produced was denser than anything
  // visible from orbit. 10^10.1 keeps the trunks and drops the tributaries
  // that were never going to be more than a dark smear at this resolution.
  supportArea: 10 ** 10.1,
  maxDistance: 20_000,
  // W ~ k*sqrt(Q). A 10^11 m² basin lands near 250 m.
  widthCoeff: 0.0008,
  minWidth: 120,
  maxWidth: 2_600,
  depthCoeff: 0.35,
  maxDepth: 60,
};

/**
 * Run the model to (approximate) topographic steady state.
 *
 * @param onProgress called with a 0–1 fraction; lets the caller drive a bar
 *        without this module knowing anything about the UI.
 */
export function runLEM(
  grid: Grid,
  base: Float32Array,
  uplift: Float32Array,
  erodibility: Float32Array,
  p = DEFAULT_LEM,
  seaLevel = 0,
  onProgress?: (t: number) => void,
): LemResult {
  const z = Float32Array.from(base);
  const areas = cellAreas(grid);
  const tmp = new Float32Array(grid.count);
  const removed = new Float32Array(grid.count);
  const before = new Float32Array(grid.count);

  let flow = routeFlow(grid, z, areas, seaLevel);

  for (let s = 0; s < p.steps; s++) {
    if (s > 0 && s % p.rerouteEvery === 0) {
      fillDepressions(grid, z, seaLevel);
      flow = routeFlow(grid, z, areas, seaLevel);
    }
    before.set(z);
    incise(z, flow, uplift, erodibility, p, seaLevel);
    diffuse(grid, z, tmp, p, seaLevel);
    // Erosion only, and *signed*: deposition is negative removal and loads
    // the crust rather than unloading it. Clamping at zero would count every
    // erosional cell and no depositional one, which is a net mass source.
    for (let c = 0; c < grid.count; c++) {
      removed[c] = before[c] + p.dt * uplift[c] - z[c];
    }
    rebound(grid, z, removed, tmp, p, seaLevel);
    capRelief(z, flow, p, seaLevel);
    onProgress?.((s + 1) / p.steps);
  }

  // Final conditioning pass so the drainage handed to the runtime matches the
  // surface handed to the runtime.
  //
  // Every previous fill was destructive, and rightly so: the model needs a
  // depression-free surface to route across. This last one is kept separately,
  // because the thing it destroys is the thing we want. A closed basin that the
  // filler raises to its spill point is not a modelling artefact to be smoothed
  // away — it is a lake, and raising the ground to the waterline is exactly how
  // to make a lake invisible. Worse, the +ε gradient the filler leaves behind is
  // in 8-neighbour BFS order from the spill point, and BFS parent chains on a
  // grid are straight rays along the eight directions: route flow down that and
  // the basin fills with perfectly straight rivers meeting at right angles.
  // From orbit those read as rectangles drawn on the continent, which is how
  // this was found.
  //
  // So: keep the basin in the terrain, and hand the filled level out as the
  // water surface. The straight-line drainage still exists inside the basin and
  // is now underwater, where it belongs and cannot be seen.
  const zPreFill = Float32Array.from(z);
  const water = Float32Array.from(z);
  fillDepressions(grid, water, seaLevel);
  z.set(water);
  flow = routeFlow(grid, z, areas, seaLevel);
  // Restore the basins to the surface now that routing is done. Only genuine
  // standing water survives the threshold: the filler's own ε is a millimetre
  // per cell and would otherwise mark half the planet as a lake.
  for (let c = 0; c < grid.count; c++) {
    if (water[c] - zPreFill[c] > LAKE_MIN_DEPTH) z[c] = zPreFill[c];
    else water[c] = z[c];
  }

  return { z, area: flow.area, flow, water };
}
