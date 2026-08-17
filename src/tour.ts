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
    dir: [0.133074, 0.581032, -0.802928], ground: 391, relief: 0.02 },
  { key: 'desert-steep', name: "Desert & xeric shrubland — steep",
    dir: [-0.685551, 0.695184, 0.216193], ground: 2502, relief: 1.00 },
  { key: 'taiga-flat', name: "Taiga / boreal forest — flat",
    dir: [-0.246176, 0.954557, 0.167983], ground: 370, relief: 0.02 },
  { key: 'taiga-steep', name: "Taiga / boreal forest — steep",
    dir: [-0.663715, 0.685779, 0.298647], ground: 2501, relief: 1.00 },
  { key: 'savanna-flat', name: "Tropical savanna — flat",
    dir: [0.471057, -0.190701, 0.861243], ground: 188, relief: 0.02 },
  { key: 'savanna-steep', name: "Tropical savanna — steep",
    dir: [0.083004, -0.064918, -0.994433], ground: 2203, relief: 1.00 },
  { key: 'steppe-flat', name: "Temperate grassland & steppe — flat",
    dir: [-0.208040, 0.821703, -0.530587], ground: 328, relief: 0.02 },
  { key: 'steppe-steep', name: "Temperate grassland & steppe — steep",
    dir: [0.396801, -0.344333, -0.850872], ground: 2500, relief: 1.00 },
  { key: 'rainforest-flat', name: "Tropical rainforest — flat",
    dir: [-0.949822, -0.029655, 0.311383], ground: 337, relief: 0.02 },
  { key: 'rainforest-steep', name: "Tropical rainforest — steep",
    dir: [0.690452, -0.025805, -0.722918], ground: 2455, relief: 1.00 },
  { key: 'tundra-flat', name: "Tundra — flat",
    dir: [-0.196779, 0.957679, 0.210067], ground: 460, relief: 0.02 },
  { key: 'tundra-steep', name: "Tundra — steep",
    dir: [0.551625, 0.829476, -0.087627], ground: 2848, relief: 0.96 },
  { key: 'broadleaf-flat', name: "Temperate broadleaf forest — flat",
    dir: [0.530431, 0.813379, 0.238868], ground: 95, relief: 0.02 },
  { key: 'broadleaf-steep', name: "Temperate broadleaf forest — steep",
    dir: [-0.876900, -0.315568, 0.362577], ground: 2469, relief: 1.00 },
  { key: 'dryforest-flat', name: "Tropical dry & seasonal forest — flat",
    dir: [0.768602, 0.637484, 0.053538], ground: 88, relief: 0.02 },
  { key: 'dryforest-steep', name: "Tropical dry & seasonal forest — steep",
    dir: [0.225372, -0.165213, -0.960163], ground: 2287, relief: 1.00 },
  { key: 'chaparral-flat', name: "Mediterranean scrubland — flat",
    dir: [0.109457, 0.651907, -0.750358], ground: 295, relief: 0.02 },
  { key: 'chaparral-steep', name: "Mediterranean scrubland — steep",
    dir: [-0.830304, -0.227545, 0.508742], ground: 2617, relief: 1.00 },
  { key: 'montane-flat', name: "Montane grassland & shrubland — flat",
    dir: [-0.291835, 0.792123, -0.536072], ground: 2939, relief: 0.02 },
  { key: 'montane-steep', name: "Montane grassland & shrubland — steep",
    dir: [0.106982, -0.015979, -0.994133], ground: 5970, relief: 1.00 },
  { key: 'ice-flat', name: "Permanent ice — flat",
    dir: [-0.053614, 0.998435, 0.015927], ground: 1272, relief: 0.02 },
  { key: 'ice-steep', name: "Permanent ice — steep",
    dir: [0.138011, 0.104593, -0.984892], ground: 8946, relief: 1.00 },
  { key: 'coast', name: 'Shoreline',
    dir: [-0.364890, 0.873032, -0.323527], ground: 0, relief: 0.01 },
  { key: 'valley', name: 'Trunk valley floor',
    dir: [0.302440, 0.233749, -0.924063], ground: 678, relief: 0.01 },
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
