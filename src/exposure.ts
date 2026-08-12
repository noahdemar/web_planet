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

/**
 * The same ceiling from orbit, where opening up is the wrong instinct.
 *
 * On the ground, opening up in deep shade is right — an eye does it, and there
 * is always something in frame that the extra stops reveal. From orbit on the
 * night side there is nothing to open up *for*: the frame is black space and
 * an unlit hemisphere, the metered key falls to 0.006, and `TARGET / key` asks
 * for 27 stops. It got MAX_EXPOSURE, 4.0 — against 0.20 for the same view in
 * daylight, so the night hemisphere was amplified twentyfold and clipped to a
 * flat white sheet, taking the atmosphere limb and the sunlit crescent with
 * it. Measured directly: renderer.toneMappingExposure read 4.000 in orbit over
 * the anti-solar point.
 *
 * 0.6 is what the terminator meters at under the disc-average illumination
 * below, so the night side now settles at roughly the exposure that renders
 * the lit limb correctly and leaves the dark hemisphere dark. Which is what a
 * camera pointed at a night side does.
 */
const MAX_EXPOSURE_ORBIT = 0.6;

/**
 * Altitude by which that ceiling is fully in force, metres.
 *
 * Its own ramp rather than `orbital`, which is fitted to when the *albedo*
 * stops being one surface and does not reach 1 until 2230 km. The ceiling is
 * answering a different question — when does the frame contain the sunlit limb
 * and the atmosphere ring, which are orders of magnitude brighter than unlit
 * ground and are what actually clips. That is when the planet reads as a body
 * rather than a place, a few hundred kilometres up. At 400 km, a perfectly
 * ordinary place to look at the night side from, `orbital` is only 0.18 and
 * left the ceiling at 3.39 — still a five-fold over-exposure.
 *
 * Deliberately *not* applied at low altitude: opening up at night on the
 * ground is correct and intended, and this must not undo it.
 */
const ORBIT_CEILING_ALT = 300_000;

/**
 * Mean cosine of incidence over the lit half of a sphere, weighted by
 * projected area — 2/3 for a Lambertian disc at full phase.
 */
const DISC_LIT_COS = 0.66;

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

    // The illumination has to become a disc average wherever the albedo does,
    // and it did not. That mismatch is the whole bug: from orbit the albedo
    // was already the whole planet's average while the *lighting* was still
    // the single point under the camera, so crossing the terminator dropped
    // the metered irradiance to zero for a frame that still had a sunlit limb
    // in it.
    //
    // From orbit sunDotUp is the cosine of the phase angle, so (1+s)/2 is the
    // classic phase function: 1 with the sun behind the camera, 0.5 over the
    // terminator, 0 looking into the shadow cone. Times the disc's own mean
    // cosine, that is the mean irradiance of what is actually in frame.
    const phase = 0.5 * (1 + sunDotUp);
    const cosMeter = cosSun + (DISC_LIT_COS * phase - cosSun) * orbital;

    const direct = (SUN_IRRADIANCE * cosMeter * albedo) / Math.PI;
    const sky = SUN_IRRADIANCE * 0.09 * albedo * (0.03 + 0.6 * cosMeter);

    // From orbit much of the frame is black space, but metering that in would
    // push the exposure up and blow out the disc. Discount it instead.
    const fill = 1 - 0.35 * orbital * orbital;

    const key = Math.max(1e-4, (direct + sky) * fill);
    const aloft = Math.min(1, altitude / ORBIT_CEILING_ALT);
    const ceiling = MAX_EXPOSURE + (MAX_EXPOSURE_ORBIT - MAX_EXPOSURE) * aloft;
    const target = Math.min(ceiling, Math.max(MIN_EXPOSURE, TARGET / key));

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
