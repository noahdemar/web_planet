/**
 * Atmospheric scattering. See SPEC.md §7.
 *
 * Single-scattering Rayleigh + Mie with Earth coefficients. Two entry points,
 * both emitted as a shared block so terrain, vegetation and the sky dome apply
 * *identical* air — anything else shows up as a seam between them.
 *
 *   `skyRadiance_S`   ray with no surface hit: raymarched to the atmosphere top
 *   `aerial_S`        ray that hit a surface: extinction + in-scatter over the
 *                     path, i.e. aerial perspective
 *
 * Aerial perspective is not decoration here, it is the blending mechanism.
 * Distant terrain, the vegetation fade-out and the LOD transitions all sit
 * behind progressively more air, which is exactly why those transitions are
 * invisible in a photograph and obvious in a renderer without atmosphere.
 *
 * Positions are planet-centred and reach 6.4 × 10⁶, so f32 gives ~0.8 m here.
 * That is irrelevant: nothing in this file needs altitude to better than a
 * metre, and no geometry is derived from these values (SPEC.md I4).
 */

/** Rayleigh scale height, metres. */
const H_RAYLEIGH = 8000;
/** Mie scale height, metres. Aerosols hug the ground far more closely. */
const H_MIE = 1200;
/** Top of the modelled atmosphere above the reference sphere. */
const ATMOS_HEIGHT = 100_000;

export function atmosphere(s: string): string {
  return /* wgsl */ `

const PI_${s}: f32 = 3.14159265359;
// Earth sea-level scattering coefficients, per metre (Bruneton).
const BETA_R_${s} = vec3<f32>(5.802e-6, 13.558e-6, 33.1e-6);
const BETA_M_${s}: f32 = 3.996e-6;
// Mie absorption as well as scattering — this is what makes haze grey, not white.
const BETA_M_EXT_${s}: f32 = 4.4e-6;
const H_R_${s}: f32 = ${H_RAYLEIGH}.0;
const H_M_${s}: f32 = ${H_MIE}.0;
const ATMO_${s}: f32 = ${ATMOS_HEIGHT}.0;

fn phaseR_${s}(c: f32) -> f32 {
  return 3.0 / (16.0 * PI_${s}) * (1.0 + c * c);
}

fn phaseM_${s}(c: f32, g: f32) -> f32 {
  let g2 = g * g;
  let d = 1.0 + g2 - 2.0 * g * c;
  return 3.0 / (8.0 * PI_${s}) * ((1.0 - g2) * (1.0 + c * c))
       / ((2.0 + g2) * max(d, 1e-4) * sqrt(max(d, 1e-4)));
}

/** Far intersection of a ray with a sphere of radius rad about the origin. */
fn sphereFar_${s}(ro: vec3<f32>, rd: vec3<f32>, rad: f32) -> f32 {
  let b = dot(ro, rd);
  let c = dot(ro, ro) - rad * rad;
  let disc = b * b - c;
  if (disc < 0.0) { return -1.0; }
  return -b + sqrt(disc);
}

/**
 * Optical depth toward the sun from a point, as (Rayleigh, Mie).
 *
 * Analytic slab rather than a second raymarch: for an exponential atmosphere
 * the integral along a straight ray at cosine mu is H·exp(-h/H)/mu. It breaks
 * down near the horizon, hence the floor on mu, but the error there is hidden
 * by how little light arrives anyway.
 */
fn sunDepth_${s}(p: vec3<f32>, sd: vec3<f32>, Rg: f32) -> vec2<f32> {
  let h = max(length(p) - Rg, 0.0);
  let mu = max(dot(normalize(p), sd), 0.035);
  return vec2<f32>(H_R_${s} * exp(-h / H_R_${s}) / mu,
                   H_M_${s} * exp(-h / H_M_${s}) / mu);
}

/** Transmittance for a given (Rayleigh, Mie) optical depth pair. */
fn transmit_${s}(od: vec2<f32>) -> vec3<f32> {
  return exp(-(BETA_R_${s} * od.x + vec3<f32>(BETA_M_${s} + BETA_M_EXT_${s}) * od.y));
}

/**
 * Exact optical depth for a path whose altitude varies linearly from h0 to h1
 * over distance d. Valid for the surface rays aerial perspective deals with;
 * a long sky ray is not linear in altitude, which is why the sky raymarches.
 */
fn slabDepth_${s}(h0: f32, h1: f32, d: f32, H: f32) -> f32 {
  let dh = h1 - h0;
  if (abs(dh) < 1.0) {
    return d * exp(-h0 / H);
  }
  return d * H / dh * (exp(-h0 / H) - exp(-h1 / H));
}

/**
 * Aerial perspective: what the air between camera and surface does to a colour.
 *
 * camP and surfP are planet-centred. Returns the surface colour attenuated
 * by extinction plus the light scattered into the path, using the standard
 * energy-consistent single-scatter form scat/ext · (1 − e^−ext).
 */
fn aerial_${s}(colour: vec3<f32>, camP: vec3<f32>, surfP: vec3<f32>,
               sd: vec3<f32>, Rg: f32, sunCol: vec3<f32>) -> vec3<f32> {
  let seg = surfP - camP;
  let d = length(seg);
  if (d < 1.0) { return colour; }
  let rd = seg / d;

  let h0 = max(length(camP) - Rg, 0.0);
  let h1 = max(length(surfP) - Rg, 0.0);
  let odR = slabDepth_${s}(h0, h1, d, H_R_${s});
  let odM = slabDepth_${s}(h0, h1, d, H_M_${s});

  let ext = BETA_R_${s} * odR + vec3<f32>(BETA_M_${s} + BETA_M_EXT_${s}) * odM;
  let tr = exp(-ext);

  let c = dot(rd, sd);
  // Sun transmittance sampled at the path midpoint — one evaluation instead of
  // a march, which is plenty over the tens of kilometres this ever covers.
  let mid = camP + seg * 0.5;
  let sunTr = transmit_${s}(sunDepth_${s}(mid, sd, Rg));

  let scat = BETA_R_${s} * (odR * phaseR_${s}(c))
           + vec3<f32>(BETA_M_${s} * odM * phaseM_${s}(c, 0.76));
  let inscat = sunCol * sunTr * scat / max(ext, vec3<f32>(1e-9)) * (1.0 - tr);

  return colour * tr + inscat;
}

/** Sky colour along a ray that never hits the ground. */
fn skyRadiance_${s}(camP: vec3<f32>, rd: vec3<f32>, sd: vec3<f32>,
                    Rg: f32, sunCol: vec3<f32>) -> vec3<f32> {
  let tTop = sphereFar_${s}(camP, rd, Rg + ATMO_${s});
  if (tTop <= 0.0) { return vec3<f32>(0.0); }

  // Stop at the ground if the ray hits it, so the horizon meets the terrain
  // instead of showing sky through it.
  let b = dot(camP, rd);
  let cg = dot(camP, camP) - Rg * Rg;
  let dg = b * b - cg;
  var tMax = tTop;
  if (dg > 0.0) {
    let tg = -b - sqrt(dg);
    if (tg > 0.0) { tMax = min(tMax, tg); }
  }

  let STEPS = 12;
  let dt = tMax / f32(STEPS);
  var sumR = vec3<f32>(0.0);
  var sumM = vec3<f32>(0.0);
  var odR = 0.0;
  var odM = 0.0;

  for (var i = 0; i < STEPS; i = i + 1) {
    let p = camP + rd * (dt * (f32(i) + 0.5));
    let h = max(length(p) - Rg, 0.0);
    let dR = exp(-h / H_R_${s}) * dt;
    let dM = exp(-h / H_M_${s}) * dt;
    odR = odR + dR;
    odM = odM + dM;

    // View transmittance to here, times sun transmittance down to here.
    let tView = transmit_${s}(vec2<f32>(odR, odM));
    let tSun = transmit_${s}(sunDepth_${s}(p, sd, Rg));
    sumR = sumR + tView * tSun * dR;
    sumM = sumM + tView * tSun * dM;
  }

  let c = dot(rd, sd);
  return sunCol * (BETA_R_${s} * sumR * phaseR_${s}(c)
                 + vec3<f32>(BETA_M_${s}) * sumM * phaseM_${s}(c, 0.76));
}
`;
}
