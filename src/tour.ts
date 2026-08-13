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
    dir: [-0.864940, 0.435259, -0.249858], ground: 120, relief: 0.02 },
  { key: 'desert-steep', name: "Desert & xeric shrubland — steep",
    dir: [-0.687833, 0.703371, 0.179318], ground: 2482, relief: 0.90 },
  { key: 'taiga-flat', name: "Taiga / boreal forest — flat",
    dir: [-0.159850, 0.945942, 0.282207], ground: 115, relief: 0.02 },
  { key: 'taiga-steep', name: "Taiga / boreal forest — steep",
    dir: [-0.682841, -0.699387, -0.211153], ground: 2500, relief: 1.00 },
  { key: 'savanna-flat', name: "Tropical savanna — flat",
    dir: [-0.984764, -0.170183, 0.035743], ground: 1129, relief: 0.02 },
  { key: 'savanna-steep', name: "Tropical savanna — steep",
    dir: [0.221622, -0.257618, -0.940487], ground: 1819, relief: 1.00 },
  { key: 'steppe-flat', name: "Temperate grassland & steppe — flat",
    dir: [0.365184, 0.896872, 0.249522], ground: 335, relief: 0.02 },
  { key: 'steppe-steep', name: "Temperate grassland & steppe — steep",
    dir: [-0.601331, 0.595338, -0.532892], ground: 2492, relief: 1.00 },
  { key: 'rainforest-flat', name: "Tropical rainforest — flat",
    dir: [0.325379, 0.709829, -0.624718], ground: 340, relief: 0.02 },
  { key: 'rainforest-steep', name: "Tropical rainforest — steep",
    dir: [0.045610, 0.101959, -0.993742], ground: 2115, relief: 1.00 },
  { key: 'tundra-flat', name: "Tundra — flat",
    dir: [0.254332, 0.951029, 0.175667], ground: 868, relief: 0.02 },
  { key: 'tundra-steep', name: "Tundra — steep",
    dir: [-0.404509, -0.831500, -0.380762], ground: 2833, relief: 1.00 },
  { key: 'broadleaf-flat', name: "Temperate broadleaf forest — flat",
    dir: [0.530165, 0.821962, 0.208093], ground: 193, relief: 0.02 },
  { key: 'broadleaf-steep', name: "Temperate broadleaf forest — steep",
    dir: [-0.838830, -0.539971, 0.069253], ground: 1843, relief: 1.00 },
  { key: 'dryforest-flat', name: "Tropical dry & seasonal forest — flat",
    dir: [-0.348662, 0.184957, -0.918818], ground: 195, relief: 0.02 },
  { key: 'dryforest-steep', name: "Tropical dry & seasonal forest — steep",
    dir: [0.165509, -0.165294, -0.972257], ground: 2151, relief: 1.00 },
  { key: 'chaparral-flat', name: "Mediterranean scrubland — flat",
    dir: [-0.758220, 0.613390, -0.221032], ground: 1295, relief: 0.02 },
  { key: 'chaparral-steep', name: "Mediterranean scrubland — steep",
    dir: [-0.381684, -0.361888, -0.850502], ground: 2447, relief: 1.00 },
  { key: 'montane-flat', name: "Montane grassland & shrubland — flat",
    dir: [0.158937, -0.169083, -0.972703], ground: 2623, relief: 0.00 },
  { key: 'montane-steep', name: "Montane grassland & shrubland — steep",
    dir: [-0.699402, -0.703046, -0.128697], ground: 3886, relief: 1.00 },
  { key: 'ice-flat', name: "Permanent ice — flat",
    dir: [-0.196802, -0.969607, -0.145363], ground: 1060, relief: 0.02 },
  { key: 'ice-steep', name: "Permanent ice — steep",
    dir: [-0.672812, -0.723588, -0.154092], ground: 7866, relief: 1.00 },
  { key: 'coast', name: 'Shoreline',
    dir: [-0.518409, -0.230585, -0.823457], ground: 0, relief: 0.00 },
  { key: 'valley', name: 'Trunk valley floor',
    dir: [-0.651131, -0.434833, -0.622052], ground: 14, relief: 0.00 },
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
