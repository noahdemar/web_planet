/**
 * The fixed set of places this planet is judged at.
 *
 * "Does it still look right" is not a question a screenshot of one valley can
 * answer, and it is not a question that survives being asked from memory. This
 * is the standing answer: every biome at both ends of the relief range, plus
 * the two features that break most often — a shoreline and a trunk valley
 * floor — at a ladder of altitudes from eye height to orbit.
 *
 * Two consumers, one table, which is the point:
 *
 *   npm run realism   walks it on the CPU and checks the measurable
 *                     invariants against a stored baseline. Under three
 *                     minutes, no GPU, no browser, exits non-zero on drift.
 *   sim.tour()        flies the camera round the same list in the running
 *                     app, for the half of realism a number cannot capture.
 *
 * The directions were found once, by tools/findSites (see LESSONS: a site
 * chosen by flying around is a site chosen for being memorable, which is the
 * opposite of a representative sample). They are fixed data now: if the bake
 * is regenerated the classification moves and this table has to be reissued,
 * which is deliberate — a baseline that silently follows the thing it is
 * measuring is not a baseline.
 */

import type { V3 } from './math/vec3d.js';

export interface Site {
  /** Stable id, used as the baseline key. */
  key: string;
  name: string;
  /** Unit direction on the sphere. */
  dir: V3;
  /** Ground elevation there, metres — for placing the camera, not a test. */
  ground: number;
  /** Baked relief, 0..1: which end of the amplitude range this site sits at. */
  relief: number;
}

export const SITES: readonly Site[] = [
  { key: 'desert-flat', name: "Desert & xeric shrubland — flat",
    dir: [0.464989, -0.381821, -0.798747], ground: 454, relief: 0.02 },
  { key: 'desert-steep', name: "Desert & xeric shrubland — steep",
    dir: [0.235150, 0.531778, -0.813583], ground: 2501, relief: 1.00 },
  { key: 'taiga-flat', name: "Taiga / boreal forest — flat",
    dir: [0.473123, 0.875778, 0.095753], ground: 1044, relief: 0.02 },
  { key: 'taiga-steep', name: "Taiga / boreal forest — steep",
    dir: [-0.559325, 0.697086, 0.448582], ground: 2486, relief: 1.00 },
  { key: 'savanna-flat', name: "Tropical savanna — flat",
    dir: [-0.903493, 0.059573, 0.424442], ground: 149, relief: 0.02 },
  { key: 'savanna-steep', name: "Tropical savanna — steep",
    dir: [0.086829, -0.070683, -0.993712], ground: 2207, relief: 1.00 },
  { key: 'steppe-flat', name: "Temperate grassland & steppe — flat",
    dir: [-0.270516, 0.824355, 0.497252], ground: 240, relief: 0.02 },
  { key: 'steppe-steep', name: "Temperate grassland & steppe — steep",
    dir: [-0.850642, -0.383441, 0.359698], ground: 2494, relief: 1.00 },
  { key: 'rainforest-flat', name: "Tropical rainforest — flat",
    dir: [0.096008, -0.051594, 0.994043], ground: 63, relief: 0.02 },
  { key: 'rainforest-steep', name: "Tropical rainforest — steep",
    dir: [0.279311, 0.027826, -0.959798], ground: 2460, relief: 1.00 },
  { key: 'tundra-flat', name: "Tundra — flat",
    dir: [-0.329394, 0.905991, 0.265857], ground: 1250, relief: 0.02 },
  { key: 'tundra-steep', name: "Tundra — steep",
    dir: [-0.543142, 0.827925, 0.139772], ground: 2842, relief: 0.95 },
  { key: 'broadleaf-flat', name: "Temperate broadleaf forest — flat",
    dir: [0.250350, 0.732336, -0.633252], ground: 1404, relief: 0.02 },
  { key: 'broadleaf-steep', name: "Temperate broadleaf forest — steep",
    dir: [-0.849719, -0.337769, 0.404833], ground: 2466, relief: 1.00 },
  { key: 'dryforest-flat', name: "Tropical dry & seasonal forest — flat",
    dir: [-0.936708, 0.242379, 0.252648], ground: 1267, relief: 0.02 },
  { key: 'dryforest-steep', name: "Tropical dry & seasonal forest — steep",
    dir: [0.211848, 0.160775, -0.963987], ground: 2281, relief: 1.00 },
  { key: 'chaparral-flat', name: "Mediterranean scrubland — flat",
    dir: [0.247591, 0.663222, -0.706282], ground: 1320, relief: 0.02 },
  { key: 'chaparral-steep', name: "Mediterranean scrubland — steep",
    dir: [-0.854594, -0.219336, 0.470703], ground: 2607, relief: 1.00 },
  { key: 'montane-flat', name: "Montane grassland & shrubland — flat",
    dir: [-0.886951, -0.277834, 0.368953], ground: 4304, relief: 0.02 },
  { key: 'montane-steep', name: "Montane grassland & shrubland — steep",
    dir: [0.109484, -0.007193, -0.993963], ground: 5978, relief: 1.00 },
  { key: 'ice-flat', name: "Permanent ice — flat",
    dir: [-0.058845, 0.997470, 0.039883], ground: 213, relief: 0.02 },
  { key: 'ice-steep', name: "Permanent ice — steep",
    dir: [0.146714, 0.009593, -0.989132], ground: 8674, relief: 1.00 },
  { key: 'coast', name: 'Shoreline',
    dir: [0.639968, -0.148034, 0.754007], ground: 0, relief: 0.00 },
  { key: 'valley', name: 'Trunk valley floor',
    dir: [-0.422249, 0.904613, 0.058142], ground: 31, relief: 0.00 },
];

/**
 * Altitudes to view a site from, metres above local ground.
 *
 * Chosen where the renderer changes character rather than by round numbers:
 * eye height is grass and stems, 900 m is the band where the mesh stops
 * carrying the finest octaves and the fragment ladder takes over, 12 km is
 * landform scale, and 120 km is where the frame stops being a place.
 */
export const ALTITUDES: readonly number[] = [1.7, 900, 12_000, 120_000];

/** Orbit, where the whole disc and the terminator are the subject. */
export const ORBIT_ALTITUDE = 3_000_000;

export const siteByKey = (k: string): Site | undefined => SITES.find((s) => s.key === k);
