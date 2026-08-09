/**
 * Auto-exposure.
 *
 * Shading is physical — reflectance times irradiance, with the sun at 17 —
 * so scene luminance spans orders of magnitude between a forest at dusk and a
 * snowfield at noon. A fixed exposure cannot serve both, and at 0.55 it did
 * not: snow clipped to flat white with no readable relief, and anything not
 * dominated by mid-tone forest was either blown out or black. Photographs of
 * mountains are not blown out because a camera meters.
 *
 * Metered analytically rather than by reading back the framebuffer. A
 * downsample-and-read costs a GPU→CPU round trip and a frame or two of lag,
 * and everything the meter needs is already known on the CPU: where the sun
 * is relative to the local horizon, how high the camera is, and what the
 * ground under it is made of. That makes the exposure a pure function of
 * state — no readback, no lag, no oscillation, and identical from frame to
 * frame for the same viewpoint, which matters because a wandering exposure
 * would corrupt any A/B comparison of a rendering change.
 *
 * Adaptation is still smoothed over time. Instant correctness reads as wrong:
 * flying into shadow should darken and then recover, the way an eye does.
 */

import { RADIUS } from './planet.js';

/** Target post-tone-map key value. 0.18 is the classic mid grey. */
const TARGET = 0.17;

/** Seconds to close most of the gap to the metered value. */
const ADAPT_UP = 0.55; // toward brighter — pupils close fast
const ADAPT_DOWN = 1.6; // toward darker — dark adaptation is slower

/** Clamp, so a pathological meter reading cannot black out the screen. */
const MIN_EXPOSURE = 0.02;
const MAX_EXPOSURE = 4.0;

/** Sun irradiance used by the shaders. Must match `sunColour` in main.ts. */
export const SUN_IRRADIANCE = 17;

/** Disc-average reflectance, used once the planet no longer fills the view. */
const PLANET_ALBEDO = 0.21;

/**
 * Reflectance the shading uses, by surface type. Approximate on purpose: the
 * meter needs the order of magnitude, and the difference between snow and
 * forest is a factor of eight.
 */
function albedoAt(elevation: number): number {
  if (elevation < 0) return 0.06; // water
  if (elevation > 3000) return 0.62; // snow and bare high rock
  if (elevation > 1600) return 0.24; // alpine rock and scree
  return 0.13; // vegetated and soil-toned ground
}

export class AutoExposure {
  /** Current smoothed exposure, what the renderer should use. */
  value = 0.55;

  /**
   * Meter the scene and step the adaptation.
   *
   * @param sunDotUp   cosine of the sun's elevation above the local horizon
   * @param groundElev elevation under the camera, metres
   * @param altitude   metres above the local surface
   * @param dt         seconds since the last frame
   */
  update(sunDotUp: number, groundElev: number, altitude: number, dt: number): void {
    const cosSun = Math.max(0, sunDotUp);

    // Which albedo to meter against depends on how much of the frame one
    // surface fills. On the ground the point under the camera is
    // representative; from orbit the frame holds ocean, forest and icecap at
    // once and metering the single point below is meaningless — it would open
    // up over an ocean and blow out the continent in the same view.
    const orbital = Math.min(1, altitude / (RADIUS * 0.35));
    const albedo = albedoAt(groundElev) * (1 - orbital) + PLANET_ALBEDO * orbital;
    const direct = (SUN_IRRADIANCE * cosSun * albedo) / Math.PI;
    const sky = SUN_IRRADIANCE * 0.09 * albedo * (0.03 + 0.6 * cosSun);

    // From orbit much of the frame is black space, but metering that in would
    // push the exposure up and blow out the disc. Discount it instead.
    const fill = 1 - 0.35 * orbital * orbital;

    const key = Math.max(1e-4, (direct + sky) * fill);
    const target = Math.min(MAX_EXPOSURE, Math.max(MIN_EXPOSURE, TARGET / key));

    const tau = target > this.value ? ADAPT_UP : ADAPT_DOWN;
    // Exponential approach, frame-rate independent.
    const k = 1 - Math.exp(-Math.max(dt, 0) / tau);
    this.value += (target - this.value) * k;
  }

  /** Jump straight to the metered value — for teleports, where a fade is wrong. */
  snap(sunDotUp: number, groundElev: number, altitude: number): void {
    this.update(sunDotUp, groundElev, altitude, 1e6);
  }
}
