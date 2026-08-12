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
    dir: [-0.378495, 0.888988, -0.257762], ground: 111, relief: 0.02 },
  { key: 'desert-steep', name: "Desert & xeric shrubland — steep",
    dir: [-0.349029, -0.470844, -0.810237], ground: 2499, relief: 1.00 },
  { key: 'taiga-flat', name: "Taiga / boreal forest — flat",
    dir: [-0.367565, -0.929851, -0.016503], ground: 90, relief: 0.02 },
  { key: 'taiga-steep', name: "Taiga / boreal forest — steep",
    dir: [-0.693722, -0.692091, -0.199397], ground: 2501, relief: 1.00 },
  { key: 'savanna-flat', name: "Tropical savanna — flat",
    dir: [-0.960183, -0.164547, 0.225773], ground: 88, relief: 0.02 },
  { key: 'savanna-steep', name: "Tropical savanna — steep",
    dir: [0.151208, -0.207410, -0.966498], ground: 1971, relief: 1.00 },
  { key: 'steppe-flat', name: "Temperate grassland & steppe — flat",
    dir: [-0.551772, 0.753245, -0.358008], ground: 1591, relief: 0.02 },
  { key: 'steppe-steep', name: "Temperate grassland & steppe — steep",
    dir: [-0.575162, 0.635774, -0.514763], ground: 2498, relief: 1.00 },
  { key: 'rainforest-flat', name: "Tropical rainforest — flat",
    dir: [-0.063975, 0.714209, -0.697002], ground: 197, relief: 0.02 },
  { key: 'rainforest-steep', name: "Tropical rainforest — steep",
    dir: [-0.174366, 0.121046, -0.977213], ground: 2436, relief: 1.00 },
  { key: 'tundra-flat', name: "Tundra — flat",
    dir: [-0.074832, 0.909664, -0.408548], ground: 1405, relief: 0.02 },
  { key: 'tundra-steep', name: "Tundra — steep",
    dir: [-0.562906, 0.825427, 0.042512], ground: 2821, relief: 1.00 },
  { key: 'broadleaf-flat', name: "Temperate broadleaf forest — flat",
    dir: [-0.077286, 0.876522, 0.475117], ground: 125, relief: 0.02 },
  { key: 'broadleaf-steep', name: "Temperate broadleaf forest — steep",
    dir: [-0.815336, -0.578178, -0.030612], ground: 2166, relief: 1.00 },
  { key: 'dryforest-flat', name: "Tropical dry & seasonal forest — flat",
    dir: [-0.059408, 0.564704, -0.823152], ground: 129, relief: 0.02 },
  { key: 'dryforest-steep', name: "Tropical dry & seasonal forest — steep",
    dir: [-0.871709, -0.322801, 0.368677], ground: 1916, relief: 1.00 },
  { key: 'chaparral-flat', name: "Mediterranean scrubland — flat",
    dir: [-0.426012, -0.608446, -0.669558], ground: 1338, relief: 0.02 },
  { key: 'chaparral-steep', name: "Mediterranean scrubland — steep",
    dir: [-0.379729, -0.356727, -0.853552], ground: 2532, relief: 1.00 },
  { key: 'montane-flat', name: "Montane grassland & shrubland — flat",
    dir: [-0.012828, 0.183947, -0.982852], ground: 2704, relief: 0.02 },
  { key: 'montane-steep', name: "Montane grassland & shrubland — steep",
    dir: [-0.292980, -0.462031, -0.837073], ground: 4781, relief: 1.00 },
  { key: 'ice-flat', name: "Permanent ice — flat",
    dir: [-0.018324, 0.995946, 0.088067], ground: 494, relief: 0.02 },
  { key: 'ice-steep', name: "Permanent ice — steep",
    dir: [-0.669790, -0.727026, -0.151042], ground: 8282, relief: 1.00 },
  { key: 'coast', name: 'Shoreline',
    dir: [-0.205774, 0.792374, -0.574283], ground: 0, relief: 0.00 },
  { key: 'valley', name: 'Trunk valley floor',
    dir: [0.263137, 0.737902, -0.621498], ground: 48, relief: 0.00 },
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
