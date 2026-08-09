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

  for (let c = 0; c < grid.count; c++) {
    const x = grid.dirs[c * 3];
    const y = grid.dirs[c * 3 + 1];
    const zz = grid.dirs[c * 3 + 2];

    let f = p.baseFrequency;
    let a = 1;
    let sum = 0;
    let norm = 0;
    for (let o = 0; o < octaves; o++) {
      sum += a * noise3(x * f + 13.7, y * f + 41.2, zz * f + 7.9);
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
