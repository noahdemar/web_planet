/**
 * How many storm systems does the planet have?
 *
 * "~5 large systems rather than a lot of small ones" is a requirement about a
 * number, and the number is not something you can read off the constant that
 * sets it. CLOUD_SYN_FREQ is a noise frequency; the count of *visible storms*
 * is the count of peaks in that field, and the two are related by more than
 * the 4*pi*f^2 feature-count arithmetic suggests. Setting f from that formula
 * gives one supersystem across a third of the planet. This measures instead.
 *
 * Two things it gets right that the obvious version does not:
 *
 *   peaks, not blobs   Thresholding coverage and counting connected regions
 *                      collapses under percolation: at the ~35% cover a real
 *                      sky has, adjacent systems touch and the count drops to
 *                      one 30%-of-the-sphere lump, which says nothing. A local
 *                      maximum is one storm whether or not its skirt overlaps.
 *   the coarse field   Only CLOUD_SYN_FREQ sets where systems are. Counting
 *                      peaks of the summed field counts CLOUD_SYN_FREQ2's arms
 *                      as separate storms, and it is 2.5x finer, so the answer
 *                      comes out several times too high.
 *
 * The neighbourhood a peak must dominate scales as 1/f, since that is how big
 * a system is. A fixed radius measures a different thing at every frequency.
 *
 * The field is fixed-seed, so this is not a sample of possible planets — it is
 * the planet. Measured at the shipped CLOUD_SYN_FREQ = 1.3: five centres, all
 * strong. At the previous 2.1: twenty centres, eight strong.
 *
 *   npm run storms
 */

import { shaderNoise } from '../src/shaderNoiseCPU.js';
import { CLOUD_SYN_FREQ } from '../src/planet.js';

const NLON = 480, NLAT = 240;

function count(f0: number, f1: number, label: string): void {
  const val = new Float64Array(NLON * NLAT);
  const coarse = new Float64Array(NLON * NLAT);
  for (let j = 0; j < NLAT; j++) {
    const th = ((j + 0.5) / NLAT) * Math.PI;
    for (let i = 0; i < NLON; i++) {
      const ph = ((i + 0.5) / NLON) * Math.PI * 2;
      const x = Math.sin(th) * Math.cos(ph), y = Math.cos(th), z = Math.sin(th) * Math.sin(ph);
      const s0 = shaderNoise(x * f0, y * f0, z * f0);
      const s1 = shaderNoise(x * f1 + 37.1, y * f1, z * f1 + 37.1);
      val[j * NLON + i] = s0 * 1.15 + s1 * 0.3;
      coarse[j * NLON + i] = s0;
    }
  }
  // Count system *centres*, not thresholded blobs. At the coverage a real sky
  // has, neighbouring systems touch and a connected-component count collapses
  // them into one 30%-of-the-sphere lump — which says nothing about how many
  // storms you can see. A local maximum is one storm whether or not its skirt
  // overlaps its neighbour's.
  let mx = -1e9, mn = 1e9;
  for (const v of coarse) { if (v > mx) mx = v; if (v < mn) mn = v; }
  const cut = mn + (mx - mn) * 0.55;
  const sizes: number[] = [];
  // A system is about 1/f0 radians across, so a peak must dominate half of
  // that to count as its own storm rather than a lump inside one.
  const R = Math.max(2, Math.round((0.5 / f0) / (Math.PI / NLAT)));
  for (let j2 = R; j2 < NLAT - R; j2++) {
    for (let i2 = 0; i2 < NLON; i2++) {
      const v = coarse[j2 * NLON + i2]!;
      if (v < cut) continue;
      let peak = true;
      for (let dj = -R; dj <= R && peak; dj++) {
        for (let di = -R; di <= R; di++) {
          if (!di && !dj) continue;
          if (coarse[(j2 + dj) * NLON + ((i2 + di + NLON) % NLON)]! > v) { peak = false; break; }
        }
      }
      if (peak) sizes.push((v - mn) / (mx - mn));
    }
  }
  sizes.sort((x, y) => y - x);
  const big = sizes.filter((v) => v >= 0.75);
  process.stdout.write(
    `  ${label}  f=${f0}/${f1}\n` +
      `    storm centres         ${sizes.length}\n` +
      `    strong (>0.75)        ${big.length}\n` +
      `    strengths             ${sizes.slice(0, 8).map((v) => v.toFixed(2)).join('  ')}\n\n`,
  );
}

// The shipped value, with the previous one for contrast.
for (const f of [2.1, CLOUD_SYN_FREQ]) count(f, f * 2.5, f === CLOUD_SYN_FREQ ? 'shipped' : 'previous');
