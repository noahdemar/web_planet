/**
 * B1 — base relief. See SPEC.md §4.
 *
 * A multi-fractal roughness added to the tectonic surface before erosion runs.
 * This looks like decoration and is not: it is what lets a drainage network
 * exist at all.
 *
 * The tectonic surface from B0 is smooth — uplift is a distance function from
 * plate boundaries, so its slopes are locally parallel. Route flow across a
 * smooth cone and you get parallel flow: every cell has one donor, the network
 * never branches, and drainage area grows *linearly* with distance instead of
 * following Hack's law. Channel relief then goes as Σ i^-0.5 ≈ 2√n over a
 * chain of n cells, so total relief scales as the square root of resolution.
 * Measured: 15 km peaks at 72 km cells, 27 km at 18 km cells, from the same
 * physics and the same parameters. A model whose mountains depend on grid
 * spacing is measuring the grid.
 *
 * Real networks branch because real surfaces are rough at every scale, and
 * incision amplifies whichever hollows happen to be there. Seeding that
 * roughness is what makes the network self-organise into a proper tree, which
 * both fixes the scaling and produces the branching valleys that read as
 * "eroded" rather than "noisy".
 */

import { noise3 } from '../heightCPU.js';
import { type Grid } from './grid.js';

export interface ReliefParams {
  /** Coarsest octave, cycles per radian. */
  baseFrequency: number;
  /** Amplitude of that octave on land, metres. */
  amplitude: number;
  /** Amplitude multiplier per octave. */
  gain: number;
  /** Frequency multiplier per octave. */
  lacunarity: number;
  /**
   * Extra amplitude where uplift is high, as a multiple of `amplitude` per
   * mm/yr of uplift. Active belts are rougher than cratons at every scale.
   */
  upliftGain: number;
  /** Abyssal hill roughness, metres. Small — it must not disturb B0's curve. */
  oceanAmplitude: number;
}

export const DEFAULT_RELIEF: ReliefParams = {
  baseFrequency: 7,
  amplitude: 420,
  gain: 0.52,
  lacunarity: 2.03,
  upliftGain: 260,
  oceanAmplitude: 90,
};

/**
 * Add base relief in place.
 *
 * Octaves stop at the grid's Nyquist frequency. Going past it would add
 * variance the grid cannot represent, which aliases into the flow routing as
 * spurious pits — and the depression filler would then quietly erase it,
 * costing time and producing nothing (SPEC.md I2).
 */
export function addBaseRelief(
  grid: Grid,
  z: Float32Array,
  uplift: Float32Array,
  p = DEFAULT_RELIEF,
  seaLevel = 0,
): void {
  // A great circle is 2π radians and holds 4·res cells, so the highest
  // representable frequency is 4·res / (2π) / 2 cycles per radian.
  const nyquist = grid.res / Math.PI;
  let octaves = 1;
  while (p.baseFrequency * p.lacunarity ** octaves < nyquist) octaves++;

  // Each octave is *rotated*, not merely offset.
  //
  // The same fault the albedo meso rungs had, and the same fix. noise3 is
  // gradient noise on an integer lattice, so its features prefer the cell axes
  // and the diagonals between them, and it is exactly zero at every lattice
  // point. An offset slides that pattern around without turning it, so six
  // octaves sampled from one direction vector all inherit one orientation and
  // their artefacts land on top of each other. What comes out is a rectilinear
  // weave at the scale of the finest octaves — 166 and 337 km here — and on a
  // desert, where nothing else is carrying variation, it reads from orbit as a
  // diamond lattice drawn across the continent.
  //
  // This is baked, so it cannot be fixed at runtime: it is in the elevation
  // every consumer reads. Three dot products per octave, cycled through four
  // arbitrary rotations — adjacent octaves are what stack visibly, and by the
  // time an orientation repeats the two rungs are nine times apart in scale.
  // They need not be exactly orthonormal; a percent of scale error on a noise
  // lookup is not a quantity anything downstream can observe.
  const ROT = [
    [1, 0, 0, 0, 1, 0, 0, 0, 1],
    [0.8, 0.36, -0.48, -0.36, 0.93, 0.1, 0.48, 0.08, 0.87],
    [0.62, -0.61, 0.49, 0.71, 0.7, -0.02, -0.33, 0.36, 0.87],
    [0.51, 0.77, -0.38, -0.62, 0.64, 0.46, 0.6, -0.06, 0.8],
  ];

  for (let c = 0; c < grid.count; c++) {
    const x = grid.dirs[c * 3];
    const y = grid.dirs[c * 3 + 1];
    const zz = grid.dirs[c * 3 + 2];

    let f = p.baseFrequency;
    let a = 1;
    let sum = 0;
    let norm = 0;
    for (let o = 0; o < octaves; o++) {
      const R = ROT[o % ROT.length];
      const rx = x * R[0] + y * R[1] + zz * R[2];
      const ry = x * R[3] + y * R[4] + zz * R[5];
      const rz = x * R[6] + y * R[7] + zz * R[8];
      sum += a * noise3(rx * f + 13.7, ry * f + 41.2, rz * f + 7.9);
      norm += a;
      a *= p.gain;
      f *= p.lacunarity;
    }
    const n = sum / Math.max(norm, 1e-6);

    if (z[c] > seaLevel) {
      // mm/yr, so a 3 mm/yr orogen roughly doubles the roughness.
      const active = uplift[c] * 1000;
      z[c] += n * (p.amplitude + p.upliftGain * active);
    } else {
      z[c] += n * p.oceanAmplitude;
    }
  }
}
