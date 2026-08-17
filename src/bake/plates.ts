/**
 * B0 — tectonics. See SPEC.md §4.
 *
 * The reason the previous height field produced plateaus rather than ranges is
 * that it had no notion of *why* ground is high. Ridged noise raised to a power
 * makes broad high regions because that is what ridged noise does. Earth's
 * relief is organised along plate boundaries: linear belts, with a coherent
 * strike, a steep and a shallow flank, and a foreland basin. That structure
 * cannot be recovered downstream — it has to be the input.
 *
 * Likewise the 38%-of-the-planet shallow shelf came from treating the ocean as
 * "land, minus a constant". Real bathymetry is set by half-space cooling:
 * ocean floor subsides as the square root of its age, from ~2.6 km at a ridge
 * to ~6 km at 180 Myr. That single term is what makes Earth's hypsometric
 * curve bimodal, with a *narrow* shelf between the two modes.
 *
 * So this module produces, per cell:
 *
 *   plate        which plate owns it
 *   continental  crust type (isostatic base elevation differs by ~4.5 km)
 *   base         isostatic elevation before any erosion
 *   uplift       rock uplift rate, the forcing term for the LEM
 *
 * Nothing here is a height field in the usual sense. It is the boundary
 * condition the landscape evolution model is then run against.
 */

import { noise3 } from '../heightCPU.js';
import { type V3, cross, dot, normalize, scale, sub } from '../math/vec3d.js';
import { type Grid } from './grid.js';
import { MinHeap } from './heap.js';

export interface TectonicsParams {
  seed: number;
  /** Major plates. Earth has 7 major + 8 minor; this counts both. */
  plates: number;
  /** Fraction of plates carrying continental crust. */
  continentalFraction: number;
  /**
   * Continental blocks, and the fraction of them that is land.
   *
   * Crust type is a property of the *crust*, not of the plate carrying it, and
   * conflating the two is the single least Earth-like thing about this stage.
   * With `continental` assigned per plate the plate edge is exactly the
   * coastline everywhere, so every coast is a plate boundary and therefore
   * every coast is orogenic. Earth is mostly the other way round: the Atlantic
   * margins sit deep inside the American, African and Eurasian plates and have
   * no mountains at all, because rifting left them there and the boundary
   * moved away. Only the Pacific rim is an active margin.
   *
   * So continental blocks get their own seeds and their own warp, uncorrelated
   * with the plates. What falls out is what Earth has: passive margins where a
   * coast lies in a plate interior, Andean arcs where a convergent boundary
   * happens to cross the continent–ocean line, and collisional belts where it
   * runs continent to continent. All three profiles were already written (see
   * CONVERGENT_CC / _OC / _OO below); until now the geometry could only ever
   * present one of them.
   */
  cratons: number;
  cratonFraction: number;
  /** Typical plate speed, m/yr. Earth: 0.01–0.10. */
  plateSpeed: number;
}

export const DEFAULT_TECTONICS: TectonicsParams = {
  seed: 20260808,
  plates: 26,
  continentalFraction: 0.445,
  cratons: 19,
  // 0.44 is what the shipped asset in public/planet was baked at. A res-256
  // sweep puts 0.52 at 27.7% land against Earth's 29.2% — closer, and the
  // right value — but changing it here without re-baking would leave the
  // source describing a planet nobody is loading. Change both together.
  cratonFraction: 0.44,
  plateSpeed: 0.09,
};

export interface Tectonics {
  /** Owning plate per cell. */
  plate: Int32Array;
  /** 1 where the crust is continental. */
  continental: Uint8Array;
  /** Isostatic elevation before erosion, metres. */
  base: Float32Array;
  /** Rock uplift rate, m/yr. Negative in subsiding basins. */
  uplift: Float32Array;
  /** Great-circle distance to the nearest plate boundary, metres. */
  boundaryDist: Float32Array;
  /** Seafloor age, Myr. Meaningless on continental crust. */
  age: Float32Array;
  /**
   * Erodibility multiplier on K, in [0, 1]. This is lithology, and it is not
   * a detail: with a uniform K the model erodes continental interiors to
   * within tens of metres of sea level, because nothing resists. Real cratons
   * are old, cold and hard — order of magnitude less erodible than the young,
   * fractured rock of an active margin — and that contrast is what keeps
   * Earth's interiors at 300–800 m instead of at the coast.
   */
  erodibility: Float32Array;
}

/** Boundary classes, in the order they are tested. */
const CONVERGENT_CC = 0; // continent–continent: collisional orogen (Himalaya)
const CONVERGENT_OC = 1; // ocean–continent: arc orogen + trench (Andes)
const CONVERGENT_OO = 2; // ocean–ocean: island arc + deep trench (Marianas)
const DIVERGENT = 3; // ridge or rift
const TRANSFORM = 4;

/** Deterministic scalar in [0,1) from an integer. */
function rand01(n: number): number {
  let x = (n ^ 0x9e3779b9) >>> 0;
  x = Math.imul(x ^ (x >>> 16), 0x85ebca6b) >>> 0;
  x = Math.imul(x ^ (x >>> 13), 0xc2b2ae35) >>> 0;
  return ((x ^ (x >>> 16)) >>> 0) / 4294967296;
}

/**
 * Plate seeds, spread by a Fibonacci sphere and then jittered.
 *
 * Evenly spaced seeds give suspiciously uniform plates; pure random seeds give
 * slivers. Fibonacci plus a jitter of about a third of the spacing lands where
 * Earth is: a few large plates, several small ones.
 */
function plateSeeds(n: number, seed: number): V3[] {
  const out: V3[] = [];
  const ga = Math.PI * (3 - Math.sqrt(5));
  for (let i = 0; i < n; i++) {
    const y = 1 - (2 * (i + 0.5)) / n;
    const r = Math.sqrt(Math.max(0, 1 - y * y));
    const th = ga * i;
    const p: V3 = [Math.cos(th) * r, y, Math.sin(th) * r];
    const j = 0.55 / Math.sqrt(n);
    const d: V3 = [
      rand01(seed + i * 3 + 1) - 0.5,
      rand01(seed + i * 3 + 2) - 0.5,
      rand01(seed + i * 3 + 3) - 0.5,
    ];
    out.push(normalize([p[0] + d[0] * j, p[1] + d[1] * j, p[2] + d[2] * j]));
  }
  return out;
}

/**
 * Multi-source Dijkstra over the grid, returning distance and the label of the
 * nearest source. Used for both distance-to-boundary and seafloor age.
 */
function distanceField(
  grid: Grid,
  sources: Int32Array,
  labels: Float32Array,
  limit: number,
): { dist: Float32Array; label: Float32Array } {
  const dist = new Float32Array(grid.count).fill(Infinity);
  const label = new Float32Array(grid.count);
  const heap = new MinHeap(Math.max(1024, sources.length * 4));

  for (let k = 0; k < sources.length; k++) {
    const c = sources[k];
    dist[c] = 0;
    label[c] = labels[k];
    heap.push(0, c);
  }

  while (heap.size > 0) {
    const d0 = heap.peekKey();
    const c = heap.pop();
    if (d0 > dist[c]) continue; // stale entry
    if (d0 > limit) continue;
    for (let k = 0; k < 8; k++) {
      const n = grid.nbr[c * 8 + k];
      const nd = d0 + grid.nbrDist[c * 8 + k];
      if (nd < dist[n]) {
        dist[n] = nd;
        label[n] = label[c];
        heap.push(nd, n);
      }
    }
  }
  return { dist, label };
}

export function buildTectonics(grid: Grid, p = DEFAULT_TECTONICS): Tectonics {
  const { count } = grid;
  const seeds = plateSeeds(p.plates, p.seed);

  // Euler pole and rate per plate. Rotation about a pole is the only motion a
  // rigid plate on a sphere can have, so this is not a simplification.
  const poles: V3[] = [];
  const rates: number[] = [];
  // Retained only to seed the craton land flags with plate-scale variety; the
  // crust a cell actually has comes from the craton field below.
  const continentalPlate: boolean[] = [];
  for (let i = 0; i < p.plates; i++) {
    const a = rand01(p.seed + 7000 + i * 5) * Math.PI * 2;
    const z = rand01(p.seed + 7001 + i * 5) * 2 - 1;
    const r = Math.sqrt(Math.max(0, 1 - z * z));
    poles.push([Math.cos(a) * r, z, Math.sin(a) * r]);
    // Speed spread of 0.4×–1.6× so some boundaries are far more active.
    rates.push((p.plateSpeed * (0.4 + 1.2 * rand01(p.seed + 7002 + i * 5))) / 6_371_000);
    continentalPlate.push(rand01(p.seed + 7003 + i * 5) < p.continentalFraction);
  }

  const plate = new Int32Array(count);
  const continental = new Uint8Array(count);
  const dir: V3 = [0, 0, 0];

  // Voronoi assignment, but on a *warped* direction. A plain Voronoi gives
  // straight-edged polygons that read as artificial the moment a coastline
  // follows one, and continental crust is assigned per plate — so the plate
  // edge *is* the coastline, and whatever structure the warp has is the only
  // structure the shoreline will ever have.
  //
  // Two octaves at 3000 km and 1100 km were not enough. They bent a continent
  // as a whole and left everything below ~1000 km straight: coastlines ran as
  // clean geodesics for hundreds of kilometres and then jumped to the 18 km
  // texel jitter with nothing in between. Real coastlines are self-similar
  // over five decades, and the missing middle is what made continents read as
  // polygons from orbit.
  //
  // Five octaves down to 56 km. Stopping there is deliberate: the solve grid
  // resolves 18 km cells, so a finer octave would only alias against it.
  //
  // The octave count was the smaller half of the problem. What actually
  // decides whether a boundary meanders is the ratio of displacement to
  // wavelength, and at the old amplitude that was 0.05 at *every* octave —
  // a 3000 km wave that moves the boundary 161 km is a gentle lean, not a
  // meander. Adding octaves at 0.05 changed total coastline length by 1.3%,
  // measured; it had to be the amplitude. At 0.13 the boundary genuinely
  // wanders, and the ratio is held roughly constant down the band by
  // WARP_GAIN ≈ 1/lacunarity so every scale contributes equally.
  //
  // The ceiling is folding: the warp is a deformation of the direction field,
  // and once the displacement gradient approaches 1 the map stops being
  // injective and plates tear into disconnected islands. 0.13 leaves useful
  // margin; much past 0.2 does not.
  const WARP_OCT = 5;
  const WARP_LAC = 2.71;
  const WARP_GAIN = 0.37;
  const WARP_AMP = 0.24;
  const warpAxis = (o0: number, o1: number, o2: number): number => {
    let a = WARP_AMP;
    let fq = 2.1;
    let sum = 0;
    for (let i = 0; i < WARP_OCT; i++) {
      sum += a * noise3(dir[0] * fq + o0, dir[1] * fq + o1, dir[2] * fq + o2);
      a *= WARP_GAIN;
      fq *= WARP_LAC;
    }
    return sum;
  };
  // Continental blocks: their own seeds, their own land flags, their own warp.
  const cratonSeeds = plateSeeds(p.cratons, p.seed + 811_000);
  const cratonIsLand: boolean[] = [];
  for (let i = 0; i < p.cratons; i++) {
    cratonIsLand.push(rand01(p.seed + 811_500 + i * 7) < p.cratonFraction);
  }
  // Same construction as the plate warp, different offsets, so the two
  // boundaries are uncorrelated. A shared warp would put the coastline back on
  // the plate edge by a different route.
  const cratonWarp = (o0: number, o1: number, o2: number): number => {
    let a = WARP_AMP;
    let fq = 2.1;
    let sum = 0;
    for (let i = 0; i < WARP_OCT; i++) {
      sum += a * noise3(dir[0] * fq + o0, dir[1] * fq + o1, dir[2] * fq + o2);
      a *= WARP_GAIN;
      fq *= WARP_LAC;
    }
    return sum;
  };

  for (let c = 0; c < count; c++) {
    dir[0] = grid.dirs[c * 3];
    dir[1] = grid.dirs[c * 3 + 1];
    dir[2] = grid.dirs[c * 3 + 2];
    const s = p.seed * 0.0001;
    const cs = p.seed * 0.0001 + 5.5;
    // Distinct offsets per axis, or all three components warp together and the
    // displacement collapses onto one direction instead of being a vector.
    const w = normalize([
      dir[0] + warpAxis(s, 0, 0),
      dir[1] + warpAxis(31.4, 0, s),
      dir[2] + warpAxis(0, 57.1, 0),
    ]);

    let best = 0;
    let bestDot = -2;
    for (let i = 0; i < p.plates; i++) {
      const d = w[0] * seeds[i][0] + w[1] * seeds[i][1] + w[2] * seeds[i][2];
      if (d > bestDot) {
        bestDot = d;
        best = i;
      }
    }
    plate[c] = best;

    // Crust type from its own Voronoi, warped by its own field. Independent of
    // the plate above, which is the whole point — see `cratons`.
    const wc = normalize([
      dir[0] + cratonWarp(cs, 0, 0),
      dir[1] + cratonWarp(12.7, 0, cs),
      dir[2] + cratonWarp(0, 88.9, 0),
    ]);
    let bestC = 0;
    let bestCDot = -2;
    for (let i = 0; i < p.cratons; i++) {
      const d = wc[0] * cratonSeeds[i][0] + wc[1] * cratonSeeds[i][1] + wc[2] * cratonSeeds[i][2];
      if (d > bestCDot) {
        bestCDot = d;
        bestC = i;
      }
    }
    continental[c] = cratonIsLand[bestC] ? 1 : 0;
  }

  // --- boundary classification -------------------------------------------
  // A boundary cell is one with a neighbour on another plate. Its class comes
  // from the relative velocity resolved onto the boundary normal.
  const bCells: number[] = [];
  const bClass: number[] = [];
  const bRate: number[] = [];
  const va: V3 = [0, 0, 0];

  for (let c = 0; c < count; c++) {
    const pc = plate[c];
    let found = -1;
    for (let k = 0; k < 8; k++) {
      const n = grid.nbr[c * 8 + k];
      if (plate[n] !== pc) {
        found = n;
        break;
      }
    }
    if (found < 0) continue;

    dir[0] = grid.dirs[c * 3];
    dir[1] = grid.dirs[c * 3 + 1];
    dir[2] = grid.dirs[c * 3 + 2];
    const nDir: V3 = [grid.dirs[found * 3], grid.dirs[found * 3 + 1], grid.dirs[found * 3 + 2]];

    // Velocities from the two Euler poles, evaluated at the same point so the
    // difference is the true relative motion across the boundary.
    const v1 = scale(cross(poles[pc], dir), rates[pc]);
    const v2 = scale(cross(poles[plate[found]], dir), rates[plate[found]]);
    va[0] = v1[0] - v2[0];
    va[1] = v1[1] - v2[1];
    va[2] = v1[2] - v2[2];

    // Boundary normal: the tangential direction from this cell to the other
    // plate. Positive projection means the plates are closing.
    let nrm = sub(nDir, dir);
    nrm = sub(nrm, scale(dir, dot(nrm, dir)));
    const nl = Math.hypot(nrm[0], nrm[1], nrm[2]);
    if (nl < 1e-12) continue;
    nrm = scale(nrm, 1 / nl);

    const closing = dot(va, nrm) * 6_371_000; // back to m/yr
    const speed = Math.hypot(va[0], va[1], va[2]) * 6_371_000;
    // Crust type at the two cells that actually meet, not the flag on the
    // plates carrying them. Now that a plate can hold both kinds, the plate's
    // own label says nothing about what is colliding here — reading it would
    // raise an Andean arc in mid-ocean and a mid-ocean ridge through a
    // continent, because the boundary would be classified from crust that may
    // be five thousand kilometres away.
    const cc = continental[c] === 1;
    const co = continental[found] === 1;

    let cls: number;
    if (closing > 0.15 * speed) {
      cls = cc && co ? CONVERGENT_CC : cc || co ? CONVERGENT_OC : CONVERGENT_OO;
    } else if (closing < -0.15 * speed) {
      cls = DIVERGENT;
    } else {
      cls = TRANSFORM;
    }

    bCells.push(c);
    bClass.push(cls);
    bRate.push(Math.abs(closing));
  }

  const srcAll = Int32Array.from(bCells);
  // Pack class and rate into one label: the distance field carries a single
  // float, and both are needed at the receiving cell. The rate is normalised
  // against twice the nominal plate speed so the fractional part is a
  // dimensionless 0–1 "how active is this boundary", not a velocity — the
  // uplift coefficients below are then real uplift rates in m/yr.
  const vNorm = 1 / (2 * p.plateSpeed);
  const labAll = Float32Array.from(
    bClass.map((cl, i) => cl + Math.min(0.99, bRate[i] * vNorm)),
  );
  const { dist: boundaryDist, label: nearLabel } = distanceField(grid, srcAll, labAll, 2.2e6);

  // Seafloor age: distance from the nearest *spreading* boundary divided by
  // the spreading rate. This is the term that makes ocean basins deep.
  const divIdx: number[] = [];
  for (let i = 0; i < bCells.length; i++) if (bClass[i] === DIVERGENT) divIdx.push(i);
  const ridgeSrc = Int32Array.from(divIdx.map((i) => bCells[i]));
  const ridgeLab = Float32Array.from(divIdx.map((i) => Math.max(0.006, bRate[i])));
  const ridge =
    ridgeSrc.length > 0
      ? distanceField(grid, ridgeSrc, ridgeLab, 3.2e7)
      : { dist: new Float32Array(count).fill(6e6), label: new Float32Array(count).fill(0.03) };

  // --- isostatic base and uplift -----------------------------------------
  const base = new Float32Array(count);
  const uplift = new Float32Array(count);
  const age = new Float32Array(count);
  const erodibility = new Float32Array(count);

  for (let c = 0; c < count; c++) {
    const bd = Math.min(boundaryDist[c], 4e6);
    const lab = nearLabel[c];
    const cls = Math.floor(lab);
    // Back to 0–2 × nominal: 1 is an average boundary, 2 a very fast one.
    // Capped at 1.4 — the uncapped tail put peak rock uplift at 16 mm/yr,
    // above anything measured on Earth, and produced an 18.9 km summit.
    const rate = Math.min(1.4, (lab - cls) * 2);

    if (continental[c]) {
      // Continental crust floats ~4.5 km higher than oceanic. Interiors are
      // slightly higher than margins, which is what thins crust into a shelf.
      base[c] = 190;
      age[c] = 0;
    } else {
      // Ocean depth against age. Half-space cooling (√t) is right near the
      // ridge but keeps deepening without limit; past ~20 Myr the plate model
      // is what matches soundings, flattening toward ~5.65 km. Using √t alone
      // is exactly the mistake that put the ocean median 850 m too deep.
      // Spreading is two-sided: crust moves away at half the relative rate.
      const half = Math.max(0.004, ridge.label[c]) * 0.5;
      const t = Math.min(180, ridge.dist[c] / half / 1e6);
      age[c] = t;
      base[c] =
        t < 20
          ? -(2600 + 365 * Math.sqrt(t))
          : -(5651 - 2473 * Math.exp(-0.0278 * t));
    }

    // Uplift. Each boundary class contributes a profile in distance, with the
    // asymmetry and width that class actually has on Earth.
    let u = 0;
    const kmToBoundary = bd / 1000;

    if (cls === CONVERGENT_CC) {
      // Collisional orogen.
      //
      // This was widened to 700 km with a flat-topped exponent to fill the
      // 2000-4000 m bands of the hypsographic curve, and it did — at the cost
      // of turning every collision into a featureless plateau stretching past
      // the horizon, which is the exact defect this whole stage exists to fix.
      // A curve fit is not the goal; it is a check that the goal was not
      // missed in some way the eye does not catch. Trading the thing being
      // measured for the measurement is backwards.
      //
      // Narrow and steeply flanked instead. Tibet is real, but it is one
      // plateau on Earth, not the signature of every convergent margin.
      u = 0.0046 * rate * Math.exp(-((kmToBoundary / 400) ** 1.5));
    } else if (cls === CONVERGENT_OC) {
      // Arc orogen: narrow and steep on the ocean side, plus a trench.
      u = 0.0064 * rate * Math.exp(-((kmToBoundary / 260) ** 1.5));
      if (!continental[c]) {
        base[c] -= 3400 * Math.exp(-((kmToBoundary / 90) ** 2));
        u = 0;
      }
    } else if (cls === CONVERGENT_OO) {
      // Island arc: a thin ridge of volcanoes over a very deep trench.
      u = 0.0030 * rate * Math.exp(-((kmToBoundary / 70) ** 2));
      base[c] -= 4200 * Math.exp(-(((kmToBoundary - 130) / 110) ** 2));
    } else if (cls === DIVERGENT) {
      if (continental[c]) {
        // Continental rift: flanking shoulders with a graben between them.
        u = 0.0018 * rate * Math.exp(-(((kmToBoundary - 110) / 90) ** 2));
        base[c] -= 900 * Math.exp(-((kmToBoundary / 70) ** 2));
      }
    } else if (cls === TRANSFORM) {
      u = 0.0008 * rate * Math.exp(-((kmToBoundary / 60) ** 2));
    }

    // Slow interior uplift keeps cratons from eroding flat over the run.
    uplift[c] = u + (continental[c] ? 4.6e-5 : 0);

    // Lithology. Young orogenic and rift rock is fractured and weak; shield
    // interiors are the hardest rock on the planet. Distance to the nearest
    // boundary is a decent proxy for both age and damage, and it costs
    // nothing extra because the distance field is already computed.
    const soft = Math.exp(-((kmToBoundary / 550) ** 1.3));
    // A modest floor: even a craton erodes, just slowly.
    erodibility[c] = 0.075 + 0.925 * soft;
  }

  // Passive margins. Getting these wrong is expensive: shelf, slope and rise
  // together are 11% of Earth's surface — more than every mountain above
  // 1000 m combined — so a margin modelled as a single exponential leaves a
  // seventh of the planet at the wrong depth.
  //
  // The real profile has three parts with very different gradients:
  //   shelf   0–130 km    almost flat, ~130 m
  //   slope   130–250 km  the steepest large landform on Earth, to ~2.8 km
  //   rise    250–850 km  a sediment apron, merging into the abyssal plain
  const marginSrc: number[] = [];
  for (let c = 0; c < count; c++) {
    if (continental[c]) continue;
    for (let k = 0; k < 8; k++) {
      if (continental[grid.nbr[c * 8 + k]]) {
        marginSrc.push(c);
        break;
      }
    }
  }
  if (marginSrc.length > 0) {
    const ms = Int32Array.from(marginSrc);
    const md = distanceField(grid, ms, new Float32Array(ms.length), 1.7e6);
    const ss = (a: number, b: number, x: number): number => {
      const t = Math.max(0, Math.min(1, (x - a) / (b - a)));
      return t * t * (3 - 2 * t);
    };
    for (let c = 0; c < count; c++) {
      const d = md.dist[c];
      if (!isFinite(d)) continue;
      if (continental[c]) {
        // Coastal plain. Wide, because most of the world's low ground is
        // exactly this and the curve was short of it by 3.6 points.
        base[c] -= 360 * Math.exp(-((d / 560e3) ** 1.5));
      } else {
        const shelfZ = -130 - 90 * ss(0, 130e3, d);
        const t = Math.max(0, Math.min(1, (d - 130e3) / 560e3));
        const slopeZ = shelfZ + (-3000 - shelfZ) * t;
        // Blend into whatever the cooling model says the abyss is here, so a
        // margin next to a young ridge stays shallow.
        base[c] = slopeZ + (base[c] - slopeZ) * ss(690e3, 1300e3, d);
      }
    }
  }

  return { plate, continental, base, uplift, boundaryDist, age, erodibility };
}
