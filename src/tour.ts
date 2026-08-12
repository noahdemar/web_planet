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
    dir: [0.130454, 0.951412, 0.278922], ground: 174, relief: 0.02 },
  { key: 'desert-steep', name: "Desert & xeric shrubland — steep",
    dir: [-0.873259, -0.470452, 0.126862], ground: 2501, relief: 1.00 },
  { key: 'taiga-flat', name: "Taiga / boreal forest — flat",
    dir: [-0.642880, 0.762987, -0.067497], ground: 1348, relief: 0.02 },
  { key: 'taiga-steep', name: "Taiga / boreal forest — steep",
    dir: [-0.601032, 0.686335, -0.409517], ground: 2500, relief: 1.00 },
  { key: 'savanna-flat', name: "Tropical savanna — flat",
    dir: [0.100962, -0.253423, -0.962073], ground: 974, relief: 0.02 },
  { key: 'savanna-steep', name: "Tropical savanna — steep",
    dir: [0.131053, -0.211497, -0.968552], ground: 2027, relief: 0.99 },
  { key: 'steppe-flat', name: "Temperate grassland & steppe — flat",
    dir: [-0.465694, -0.878006, -0.110607], ground: 267, relief: 0.02 },
  { key: 'steppe-steep', name: "Temperate grassland & steppe — steep",
    dir: [-0.860299, -0.504149, 0.075627], ground: 2484, relief: 1.00 },
  { key: 'rainforest-flat', name: "Tropical rainforest — flat",
    dir: [-0.996967, -0.066003, 0.041227], ground: 154, relief: 0.02 },
  { key: 'rainforest-steep', name: "Tropical rainforest — steep",
    dir: [0.043844, 0.106948, -0.993297], ground: 2445, relief: 0.99 },
  { key: 'tundra-flat', name: "Tundra — flat",
    dir: [-0.411088, 0.911405, -0.018657], ground: 1321, relief: 0.02 },
  { key: 'tundra-steep', name: "Tundra — steep",
    dir: [-0.440188, -0.830039, -0.342448], ground: 2824, relief: 1.00 },
  { key: 'broadleaf-flat', name: "Temperate broadleaf forest — flat",
    dir: [-0.243927, 0.852409, -0.462492], ground: 273, relief: 0.02 },
  { key: 'broadleaf-steep', name: "Temperate broadleaf forest — steep",
    dir: [-0.116184, 0.569366, -0.813832], ground: 2238, relief: 1.00 },
  { key: 'dryforest-flat', name: "Tropical dry & seasonal forest — flat",
    dir: [-0.963998, -0.240483, -0.113473], ground: 174, relief: 0.02 },
  { key: 'dryforest-steep', name: "Tropical dry & seasonal forest — steep",
    dir: [-0.870838, -0.332718, 0.361858], ground: 2105, relief: 1.00 },
  { key: 'chaparral-flat', name: "Mediterranean scrubland — flat",
    dir: [-0.692783, -0.622941, -0.363312], ground: 1361, relief: 0.02 },
  { key: 'chaparral-steep', name: "Mediterranean scrubland — steep",
    dir: [-0.396375, -0.359332, -0.844847], ground: 2442, relief: 1.00 },
  { key: 'montane-flat', name: "Montane grassland & shrubland — flat",
    dir: [-0.643499, 0.760849, 0.083773], ground: 2937, relief: 0.03 },
  { key: 'montane-steep', name: "Montane grassland & shrubland — steep",
    dir: [-0.292980, -0.462031, -0.837073], ground: 4949, relief: 1.00 },
  { key: 'ice-flat', name: "Permanent ice — flat",
    dir: [-0.256230, -0.959465, -0.117363], ground: 1308, relief: 0.02 },
  { key: 'ice-steep', name: "Permanent ice — steep",
    dir: [-0.669790, -0.727026, -0.151042], ground: 8316, relief: 1.00 },
  { key: 'coast', name: 'Shoreline',
    dir: [-0.883456, 0.403339, -0.238377], ground: 0, relief: 0.00 },
  { key: 'valley', name: 'Trunk valley floor',
    dir: [-0.567242, 0.785745, -0.246663], ground: 297, relief: 0.01 },
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
