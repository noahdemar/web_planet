/**
 * Terrain WGSL. See SPEC.md §3 (spatial index) and §6 (amplification).
 *
 * ── The precision problem, and why this file looks the way it does ──
 *
 * The naive vertex shader is:
 *
 *     dir      = normalize(warp(faceUV))      // unit
 *     worldPos = dir * (R + h)                // ~6.4e6
 *     rel      = worldPos - camPos            // ~10 m
 *
 * That last line is catastrophic cancellation: two f32 values near 6.4e6
 * (ULP 0.76 m) differenced down to metres. Everything below ~1 m is noise.
 *
 * The fix is to never form the full-magnitude vector on the GPU. A per-patch
 * anchor arrives with its subtraction already done on the CPU in f64, and the
 * within-patch offset is reconstructed analytically so it carries full f32
 * *relative* precision:
 *
 *   1. Tangent addition gives the warp delta exactly:
 *        w(c+t) - w(c) = w(t)(1 + w(c)²) / (1 - w(c)w(t))
 *      so the small quantity comes from small inputs, never from differencing
 *      two O(1) numbers.
 *
 *   2. A stable expansion gives the direction delta:
 *        normalize(Pc + dP) - normalize(Pc) = (dP - Pc·k) / (|Pc|(1 + k))
 *      with k = sqrt(1+s) - 1 evaluated as s/(1+sqrt(1+s)) to avoid
 *      cancellation as s → 0.
 *
 * At L19 the delta is ~3e-6 carrying ~1e-7 relative error, so the absolute
 * error after scaling by R is ~2 µm. Sub-millimetre ground detail from a
 * 6371 km origin, in pure f32.
 *
 * ── Work splitting ──
 *
 * `patchSurface` does the expensive part (noise + gradient) once. `position`
 * takes the resulting height as an *input* and only redoes the cheap geometric
 * offset, so the noise is evaluated exactly once per vertex. The helper blocks
 * are emitted with distinct suffixes because TSL inlines each wgslFn's source
 * into the same stage, where duplicate function names would collide.
 */

import { wgslFn } from 'three/tsl';
import { REFERENCE_SPECTRUM, SEASON_F0, biomeBlock } from '../biome.js';
import {
  AMP_BASE,
  AMP_F0,
  AMP_RELIEF,
  BIOME_F0,
  FLOODPLAIN_AMP,
  BIOME_GAIN,
  BAND_FADE_HI,
  BAND_FADE_LO,
  BIOME_LACUNARITY,
  CHANNEL_DEPTH,
  DEFAULT_OCTAVES,
  CHANNEL_HALF_HI,
  CHANNEL_HALF_LO,
  CHANNEL_WIDTH_K,
  COAST_WARP_AMP,
  COAST_WARP_F0,
  CLOUD_ALT,
  CLOUD_ZONAL,
  COAST_WARP_FADE,
  COAST_WARP_OCTAVES,
  CLUMP_MEDIAN,
  CLUMP_STRENGTH,
  HILLSLOPE_GAIN,
  HILLSLOPE_SOFT,
  LOG2_LACUNARITY,
  LAKE_ON_HI,
  LAKE_ON_LO,
  LAPSE,
  NIGHT_SKY,
  LAT_EXP,
  RELIEF_GAIN,
  RELIEF_LACUNARITY,
  RELIEF_SLOPE_HI,
  RELIEF_SLOPE_LO,
  LOCAL_PERIOD,
  MIN_NOISE_LAMBDA,
  RADIUS,
  RIDGE_MEAN,
  SHORE_FLAT_FLOOR,
  SHORE_FLAT_HI,
  VALLEY_WET_HI,
  VALLEY_WET_LO,
  VEG_MAX_SLOPE,
  VEG_MIN_ELEVATION,
  VEG_SLOPE_FULL,
} from '../planet.js';
import { atmosphere } from './atmosphere.js';

/**
 * WGSL float literal. `70` has to reach the shader as `70.0` or it is an i32
 * and the compile fails; anything already fractional passes through. The
 * previous code wrote `${AMP_BASE}.0` at each use site, which is correct right
 * up until a constant is retuned to a non-integer and every use site breaks.
 */
const f = (v: number): string => (Number.isInteger(v) ? `${v}.0` : `${v}`);

/** Geometry only: the precision-critical offset and the CDLOD morph. Cheap. */
export function geom(s: string): string {
  return /* wgsl */ `

// Direction delta from the patch centre — the precision-critical routine.
fn offset_${s}(g: vec2<f32>, A: f32, B: f32, hs: f32,
               Pc: vec3<f32>, lenPc: f32,
               BU: vec3<f32>, BV: vec3<f32>) -> vec3<f32> {
  // tangent addition: w(c+t) - w(c) = w(t)(1 + w(c)²) / (1 - w(c)w(t))
  let a = tan(hs * g.x * 0.78539816339744831);
  let b = tan(hs * g.y * 0.78539816339744831);
  let dwu = a * (1.0 + A * A) / (1.0 - A * a);
  let dwv = b * (1.0 + B * B) / (1.0 - B * b);
  let dP = BU * dwu + BV * dwv;

  // normalize(Pc + dP) - normalize(Pc), stable as dP -> 0
  let inv = 1.0 / lenPc;
  let sq = (2.0 * dot(Pc, dP) + dot(dP, dP)) * inv * inv;
  let k = sq / (1.0 + sqrt(1.0 + sq));   // sqrt(1+s) - 1, without cancellation
  return (dP - Pc * k) * (inv / (1.0 + k));
}

fn dirCOf_${s}(Pc: vec3<f32>, lenPc: f32) -> vec3<f32> { return Pc / lenPc; }

// The morphed grid coordinate. Distance is evaluated per *vertex*, which is
// what makes CDLOD crack-free: two patches sharing an edge see the same world
// position, so they agree exactly, and a finer patch lands precisely on its
// parent's grid before the parent takes over.
fn morphed_${s}(gpos: vec2<f32>, gpar: vec3<f32>, A: f32, B: f32, hs: f32,
                Pc: vec3<f32>, lenPc: f32, BU: vec3<f32>, BV: vec3<f32>,
                anchorRel: vec3<f32>, radius: f32, refR: f32,
                morph: vec2<f32>, stepG: f32) -> vec3<f32> {
  let d0 = offset_${s}(gpos, A, B, hs, Pc, lenPc, BU, BV);
  // Distance must be measured on the same sphere the CPU selected against:
  // the camera's own ground radius, not sea level. Standing on 2.5 km of
  // terrain, measuring to sea level inflates every distance by 2.5 km, which
  // saturates morphK to 1 at every deep level — each patch then renders as
  // its parent's grid and adjacent levels no longer line up.
  let dist = length(anchorRel + dirCOf_${s}(Pc, lenPc) * (refR - radius) + d0 * refR);
  let mk = clamp((dist - morph.x) / max(morph.y - morph.x, 1e-6), 0.0, 1.0);
  // The morph factor rides along: a fully morphed patch has its parent's
  // vertex spacing, and the band limit must follow that or it would disagree
  // with the coarser neighbour across a shared edge.
  return vec3<f32>(gpos - gpar.xy * stepG * mk, mk);
}
`;
}

/**
 * Gradient noise and its analytic derivative.
 *
 * Split out from `field` so the climate block can be emitted on its own. TSL
 * inlines each wgslFn's source into the stage that uses it, and two stages
 * that both want noise would otherwise have to pull in the whole elevation
 * field to get at it.
 */
export function noiseBlock(s: string): string {
  return /* wgsl */ `

// PCG3D integer hash. Deliberately not sin-based: at octave 12 the lattice
// coordinate reaches ~10^4, and an f32 sin() of the resulting large argument
// loses its significant bits. An integer hash is exact at every octave.
fn hash33_${s}(p0: vec3<f32>) -> vec3<f32> {
  var p = bitcast<vec3<u32>>(vec3<i32>(p0));
  p = p * 1664525u + vec3<u32>(1013904223u);
  p.x = p.x + p.y * p.z;
  p.y = p.y + p.z * p.x;
  p.z = p.z + p.x * p.y;
  p = p ^ (p >> vec3<u32>(16u));
  p.x = p.x + p.y * p.z;
  p.y = p.y + p.z * p.x;
  p.z = p.z + p.x * p.y;
  return -1.0 + 2.0 * vec3<f32>(p) * (1.0 / 4294967296.0);
}

// Gradient noise returning (value, d/dx, d/dy, d/dz). One evaluation yields
// both the height and the exact surface normal — no finite differencing and
// no extra noise taps per vertex.
fn noised_${s}(x: vec3<f32>) -> vec4<f32> {
  let p = floor(x);
  let w = x - p;
  let u  = w * w * w * (w * (w * 6.0 - 15.0) + 10.0);
  let du = 30.0 * w * w * (w * (w - 2.0) + 1.0);

  let ga = hash33_${s}(p + vec3<f32>(0.0, 0.0, 0.0));
  let gb = hash33_${s}(p + vec3<f32>(1.0, 0.0, 0.0));
  let gc = hash33_${s}(p + vec3<f32>(0.0, 1.0, 0.0));
  let gd = hash33_${s}(p + vec3<f32>(1.0, 1.0, 0.0));
  let ge = hash33_${s}(p + vec3<f32>(0.0, 0.0, 1.0));
  let gf = hash33_${s}(p + vec3<f32>(1.0, 0.0, 1.0));
  let gg = hash33_${s}(p + vec3<f32>(0.0, 1.0, 1.0));
  let gh = hash33_${s}(p + vec3<f32>(1.0, 1.0, 1.0));

  let va = dot(ga, w - vec3<f32>(0.0, 0.0, 0.0));
  let vb = dot(gb, w - vec3<f32>(1.0, 0.0, 0.0));
  let vc = dot(gc, w - vec3<f32>(0.0, 1.0, 0.0));
  let vd = dot(gd, w - vec3<f32>(1.0, 1.0, 0.0));
  let ve = dot(ge, w - vec3<f32>(0.0, 0.0, 1.0));
  let vf = dot(gf, w - vec3<f32>(1.0, 0.0, 1.0));
  let vg = dot(gg, w - vec3<f32>(0.0, 1.0, 1.0));
  let vh = dot(gh, w - vec3<f32>(1.0, 1.0, 1.0));

  let k0 = va;
  let k1 = vb - va;
  let k2 = vc - va;
  let k3 = ve - va;
  let k4 = va - vb - vc + vd;
  let k5 = va - vc - ve + vg;
  let k6 = va - vb - ve + vf;
  let k7 = -va + vb + vc - vd + ve - vf - vg + vh;

  let value = k0 + k1 * u.x + k2 * u.y + k3 * u.z
            + k4 * u.x * u.y + k5 * u.y * u.z + k6 * u.z * u.x
            + k7 * u.x * u.y * u.z;

  let deriv = ga
    + u.x * (gb - ga) + u.y * (gc - ga) + u.z * (ge - ga)
    + u.x * u.y * (ga - gb - gc + gd)
    + u.y * u.z * (ga - gc - ge + gg)
    + u.z * u.x * (ga - gb - ge + gf)
    + u.x * u.y * u.z * (-ga + gb + gc - gd + ge - gf - gg + gh)
    + du * vec3<f32>(
        k1 + k4 * u.y + k6 * u.z + k7 * u.y * u.z,
        k2 + k5 * u.z + k4 * u.x + k7 * u.z * u.x,
        k3 + k6 * u.x + k5 * u.y + k7 * u.x * u.y);

  return vec4<f32>(value, deriv);
}

// smoothstep with its derivative, so the land mask can be differentiated
// through the product rule rather than sampled numerically.
fn sstepd_${s}(e0: f32, e1: f32, x: f32) -> vec2<f32> {
  let inv = 1.0 / (e1 - e0);
  let t = clamp((x - e0) * inv, 0.0, 1.0);
  let v = t * t * (3.0 - 2.0 * t);
  let d = select(0.0, 6.0 * t * (1.0 - t) * inv, x > e0 && x < e1);
  return vec2<f32>(v, d);
}
`;
}

/**
 * The reconstructed channel. Needs nothing; pure arithmetic on the bake.
 *
 * Separate from `waterBlock` and `field` because both need it and TSL inlines
 * each block's source into the same stage — defining it in both would collide.
 */
export function channelBlock(s: string): string {
  return /* wgsl */ `
/**
 * Channel half-width and incision at this point, from the reconstructed
 * distance to the drainage axis. See CHANNEL_WIDTH_K.
 *
 * Returns (incision below the floodplain, channel mask 0..1). The incision is
 * a smooth bowl rather than a step so the bank has a cross-section: the
 * terrain dips into it and the water fills the bottom, which is what makes it
 * read as a river from a metre away and from 400 km.
 */
fn channel_${s}(wet: f32, distAxis: f32) -> vec2<f32> {
  let on = smoothstep(${f(VALLEY_WET_LO)}, ${f(VALLEY_WET_HI)}, wet);
  if (on <= 0.0) { return vec2<f32>(0.0, 0.0); }
  // Half-width from drainage area, and now it can be honest: the distance is
  // measured, not inferred, so a 150 m river can be placed inside a 9 km texel
  // instead of being widened until the reconstruction stopped flickering.
  let halfW = clamp(${f(CHANNEL_WIDTH_K)} * sqrt(pow(10.0, wet)),
                    ${f(CHANNEL_HALF_LO)}, ${f(CHANNEL_HALF_HI)});
  // The bowl is 2.5 half-widths across so the banks are inside it too.
  let m = 1.0 - smoothstep(0.0, halfW * 2.5, distAxis);
  return vec2<f32>(${f(CHANNEL_DEPTH)} * on * m * m, on * m);
}
`;
}

/**
 * Coastline warp: a small displacement of the *lookup* direction.
 *
 * See COAST_WARP_AMP in planet.ts for why this exists. Three octaves, fixed
 * frequencies, no band limit — the coastline must not depend on the camera.
 *
 * The displacement is built in the tangent plane so it does not change the
 * radius, and it uses two decorrelated noise fields rather than one vector
 * field so the warp has no preferred direction.
 */
export const coastWarp = wgslFn(/* wgsl */ `
fn coastWarp(dir: vec3<f32>, radius: f32, bakeH: f32) -> vec3<f32> {
  // Local to the coast. See COAST_WARP_FADE: a global warp slides the whole
  // planet sideways under the camera.
  let w = 1.0 - smoothstep(0.0, ${f(COAST_WARP_FADE)}, abs(bakeH));
  if (w <= 0.0) { return vec3<f32>(0.0); }
  var axis = vec3<f32>(0.0, 1.0, 0.0);
  if (abs(dir.y) > 0.95) { axis = vec3<f32>(1.0, 0.0, 0.0); }
  let e0 = normalize(cross(axis, dir));
  let e1 = cross(dir, e0);

  var amp = 1.0;
  var frq = ${f(COAST_WARP_F0)};
  var u = 0.0;
  var v = 0.0;
  var norm = 0.0;
  for (var i = 0; i < ${COAST_WARP_OCTAVES}; i = i + 1) {
    u = u + amp * noised_C2(dir * frq + vec3<f32>(f32(i) * 3.71 + 1.5)).x;
    v = v + amp * noised_C2(dir * frq + vec3<f32>(f32(i) * 3.71 + 61.3)).x;
    norm = norm + amp;
    amp = amp * 0.5;
    frq = frq * 2.17;
  }
  // Metres, converted to a direction delta.
  let k = ${f(COAST_WARP_AMP)} * w / (norm * radius);
  return (e0 * u + e1 * v) * k;
}
${noiseBlock('C2')}
`);

/**
 * Cloud coverage as a function of direction, shared by the deck and the ground.
 *
 * Split out of clouds.ts so the terrain can cast cloud shadows from exactly the
 * field the deck draws. Two layers, because one never looks like weather:
 *
 *   cumulus   billow noise — |n| summed, which piles up into rounded lumps
 *             with flat bases instead of the smooth blobs plain fBm gives. This
 *             is the layer with edges, and the one that shadows the ground.
 *   cirrus    high, thin, stretched along the flow. Contributes almost no
 *             opacity but a great deal of sky.
 *
 * Returns (cumulus 0..1, cirrus 0..1).
 */
export function cloudFieldBlock(s: string): string {
  return /* wgsl */ `
fn cloudField_${s}(dir: vec3<f32>, t: f32, cover: f32, px: f32) -> vec2<f32> {
  // Hadley structure: rising air at the equator and mid latitudes, subsiding
  // in the subtropics. Getting this band right is most of what separates a
  // planet from a marble with noise on it.
  let lat = abs(dir.y);
  let itcz    = exp(-((lat / 0.14) * (lat / 0.14)));
  let dryBelt = exp(-(((lat - 0.45) / 0.16) * ((lat - 0.45) / 0.16)));
  let storm   = exp(-(((lat - 0.74) / 0.17) * ((lat - 0.74) / 0.17)));
  let bias = 0.26 * itcz - 0.32 * dryBelt + 0.22 * storm;

  // ── advection ─────────────────────────────────────────────────────────
  //
  // Wind runs east in the mid latitudes and west in the tropics, as it does on
  // Earth. One drift direction for the whole planet made the deck rotate like
  // a painted ball — but the fix for that tore the deck in half.
  //
  // It was a *translation* of the noise domain, t * rate * select(1, -1, lat
  // < 0.35). Two things wrong and they compound: a select is a discontinuity,
  // and the offset grows with t, so the tear widens for as long as the sim
  // runs. Half an hour in, the tropics and the mid latitudes were six
  // planet-widths out of step and the boundary was a hard circle of latitude —
  // unmissable from over a pole, where it draws a ring with unrelated weather
  // inside and outside it.
  //
  // Rotating the sample direction about the axis instead is continuous on the
  // sphere by construction, so there is nothing to tear however long it runs,
  // and it is what a zonal wind physically is. The profile is smooth as well,
  // which turns the reversal between the trades and the westerlies into a shear
  // zone — where fronts actually form — rather than a cut. Rotation about Y
  // leaves dir.y alone, so every latitude term above still means what it says.
  let band = mix(-1.0, 1.0, smoothstep(0.26, 0.48, lat));
  let spin = t * band * ${f(CLOUD_ZONAL)};
  let cs = cos(spin);
  let sn = sin(spin);
  let sdir = vec3<f32>(dir.x * cs + dir.z * sn, dir.y, -dir.x * sn + dir.z * cs);

  // Systems also have to change shape, not merely slide. A *uniform* offset is
  // safe where a latitude-dependent one is not: every point moves the same way,
  // so the field translates rigidly and there is no seam to open.
  let drift0 = vec3<f32>(0.0, t * 0.0011, 0.0);

  // ── synoptic scale ────────────────────────────────────────────────────
  //
  // The single biggest thing separating this from a photograph of Earth was
  // that the cloud was *uniform*: the same density of the same cells
  // everywhere, modulated only by latitude. The Blue Marble is nothing like
  // that. It is a few enormous rotating systems, long frontal bands trailing
  // off them, and huge genuinely clear areas in between.
  //
  // Both come out of one low-frequency field. Its value gives the synoptic
  // coverage — where the systems *are*, and therefore where the clear air is.
  // Its gradient, rotated a quarter turn about the local vertical, gives a
  // flow that circulates rather than diverging: warping the domain along that
  // is what shears the cells into spiral arms and drawn-out fronts. This is
  // the same reason real cloud organises — it is advected by a rotational
  // field — arrived at by the cheapest route rather than by solving for it.
  let syn = noised_${s}(sdir * 2.1 + drift0 * 0.35);
  let sg = syn.yzw - sdir * dot(sdir, syn.yzw);
  let curl = cross(sdir, sg);
  // Second, finer swirl so the arms have arms.
  let syn2 = noised_${s}(sdir * 5.3 + drift0 * 0.6 + vec3<f32>(37.1));
  let sg2 = syn2.yzw - sdir * dot(sdir, syn2.yzw);
  // Enough to organise, not so much that the cells stretch into marbling. At
  // 0.16 the whole planet came out as swirled paint: the arms were longer than
  // the cells they were made of, so there was no cloud left, only filament.
  let flow = normalize(sdir + curl * 0.085 + cross(sdir, sg2) * 0.028 + sg * 0.015);
  // Strong: this is what makes big clear subtropics and big cloudy storm
  // tracks instead of an even sprinkle.
  // Strong, because the clear air matters as much as the cloud. Half the Blue
  // Marble is open ocean under nothing at all, and that contrast is what makes
  // the systems read as systems.
  let synoptic = syn.x * 1.15 + syn2.x * 0.3;

  // ── cumulus: billow ───────────────────────────────────────────────────
  //
  // |n|, not 1 − |n|, and the difference is the whole layer. Billow noise has
  // a crease at every zero crossing of the underlying field; |n| puts the
  // creases at the *bottom* and rounded lumps between them, which is a cumulus
  // field. The first version had the sign the other way round, so the ridges
  // ran *along* the zero set — and the zero set of a smooth field is a family
  // of curves, so the planet came out wrapped in a white reticulated web
  // rather than clouds.
  //
  // Plain fBm would not do either: it is symmetric, so a threshold on it gives
  // soft-edged puddles with no sense of separate cells.
  // Eight octaves, not five, and the last three are the whole difference
  // between cloud and blobs.
  //
  // Five reached 21.8 km. A cumulus *cell* is 1-10 km across and its margin is
  // ragged down to a few hundred metres, so at 21.8 km the deck could only
  // ever be smooth round lumps — which is exactly what it was. The extra
  // octaves reach 1.9 km, and because they ride the same billow they erode the
  // edges as well as the interior: a cauliflower margin is not a separate
  // effect, it is the fine octaves of the same field showing up at the
  // threshold.
  //
  // px is the ground size of a pixel at the deck, so each octave switches
  // off once it drops below eight of them — the same Nyquist rule the terrain
  // uses (LESSONS §4). Without it the fine octaves would alias into a crawling
  // stipple from orbit, which is the failure this project has already had once
  // on the ground. The normaliser stays unweighted so switching an octave off
  // low-passes the field instead of rescaling it, and the coverage threshold
  // below keeps meaning the same thing at every altitude.
  var amp = 1.0;
  var frq = 11.0;
  var sum = 0.0;
  var norm = 0.0;
  for (var i = 0; i < 8; i = i + 1) {
    let lam = ${f(RADIUS)} / frq;
    let w = 1.0 - smoothstep(lam * 0.125, lam * 0.4, px);
    if (w > 0.004) {
      let d = drift0 * (1.0 + 0.55 * f32(i));
      let n = noised_${s}(flow * frq + d + vec3<f32>(f32(i) * 13.7)).x;
      sum = sum + w * amp * abs(n);
    }
    norm = norm + amp;
    amp = amp * 0.5;
    frq = frq * 2.27;
  }
  // The window is placed on the field's own *measured* mean. An earlier version
  // guessed it and was out by 0.22, which put the entire distribution above the
  // upper edge: the planet went 100% overcast, and since the ground now takes a
  // cloud shadow from this same field, every landscape went dark with it.
  //
  //   mean 0.1545   p05 0.059   p33 0.117   p50 0.146   p66 0.176   p95 0.280
  let billow = sum / norm - 0.1545;

  // The window is narrow so the field spends its time near 0 or 1 and the
  // edges stay edges. A deck that is 40% opaque everywhere is haze, not
  // weather.
  // Window from the same measurement: [0.000, 0.056] puts roughly 40% of the
  // sky under cumulus at neutral coverage, about right for Earth once the
  // cirrus layer is added on top.
  let cu = smoothstep(0.020, 0.082,
                      billow + bias * 0.06 + synoptic * 0.085 + (cover - 0.5) * 0.09);

  // ── cirrus ────────────────────────────────────────────────────────────
  //
  // Stretched hard along the flow: ice cloud is sheared out by the wind into
  // streaks with an aspect ratio of tens to one, and that streaking is the
  // whole visual signature. Squashing the sample direction is the cheapest way
  // to get it.
  // Cirrus rides the same flow — on Earth it is drawn out ahead of the front
  // that produced it, which is why the streaks curve with the system.
  let sh = vec3<f32>(flow.x * 0.42, flow.y * 2.1, flow.z * 0.42);
  var camp = 1.0;
  var cfrq = 9.0;
  var csum = 0.0;
  var cnorm = 0.0;
  for (var i = 0; i < 3; i = i + 1) {
    csum = csum + camp * noised_${s}(sh * cfrq + drift0 * 2.4 + vec3<f32>(f32(i) * 7.1)).x;
    cnorm = cnorm + camp;
    camp = camp * 0.55;
    cfrq = cfrq * 2.6;
  }
  let ci = smoothstep(0.10, 0.40,
                      csum / cnorm + bias * 0.25 + synoptic * 0.09 + (cover - 0.5) * 0.3);

  return vec2<f32>(cu, ci);
}
`;
}

/**
 * Standing-water surface elevation, metres. Below the ground where there is
 * none, so `max(h, waterLevel)` is the drawn surface and the difference is the
 * depth.
 *
 * Three sources, combined by taking the highest:
 *
 *   sea      zero. Land that dips below it is sea; that is what a coast is.
 *   lakes    the level the depression filler found. The basin is kept in the
 *            terrain rather than raised to the waterline (see runLEM), so this
 *            is real bathymetry and not a flat disc.
 *   rivers   see RIVER_FREEBOARD. A 9 km bake cell cannot hold a channel, so
 *            the waterline goes just under the valley floor and the
 *            amplification decides where the water actually runs.
 *
 * Everything off water is pushed down by a kilometre rather than branched
 * around: `max` then cannot pick it, and there is no divergence.
 */
export function waterBlock(s: string): string {
  return /* wgsl */ `
fn waterLevel_${s}(bakeH: f32, lakeD: f32, wet: f32, distAxis: f32) -> f32 {
  var w = 0.0;

  let lakeOn = smoothstep(${f(LAKE_ON_LO)}, ${f(LAKE_ON_HI)}, lakeD);
  w = max(w, bakeH + lakeD - mix(20000.0, 0.0, lakeOn));

  // The channel, filled to CHANNEL_FILL of its own depth. Both the bed and the
  // waterline come from the same bowl, so the waterline can never leave the
  // channel however the terrain around it moves.
  // ── no river water, and this is deliberate ────────────────────────────
  //
  // Three approaches were built and measured, and all three failed the same
  // way, for the same reason:
  //
  //   1. threshold the wetness field        texel-shaped blobs; widening it
  //                                         until it was stable gave a river
  //                                         10 km across
  //   2. reconstruct the axis from the      the estimator is built from a 9 km
  //      transverse wetness gradient        stencil, so it is noisiest at the
  //                                         scale the channel lives at; the
  //                                         river broke into a chain of ponds
  //   3. bake a Dijkstra distance field     the path is right and contiguous,
  //      and read it directly               but a 250 m half-width against a
  //                                         9 km texel means the mask flips
  //                                         inside a fraction of one bilinear
  //                                         cell — blue rectangles
  //
  // The common cause is not the method. A continental trunk river is 1–3 km
  // wide and this bake stores 9 km per texel, so the *water* is one to two
  // orders of magnitude below what the data can place. No amount of filtering
  // recovers a feature that is not in the field, and every attempt to force it
  // produced something worse than nothing.
  //
  // So: no river water. What stays is everything that is genuinely resolved —
  // the sea, the lakes the depression filler found, the channel *valleys*
  // carveChannels cuts into the terrain, and the riparian moisture that greens
  // them. The drainage network is still visible, as topography and as
  // vegetation, which is how it reads from altitude anyway. Drawing water in
  // it needs per-tile data at metre scale, which is a later milestone, not a
  // shader constant.

  return w;
}
`;
}



/**
 * Gradient noise on a *periodic* lattice, evaluated in metres.
 *
 * The near-field escape hatch from f32. See LOCAL_PERIOD: the caller works in
 * metres about a snapped origin instead of in units of a planet radius, so the
 * coordinate stays in the thousands and the lattice fraction keeps ten bits
 * where the global form has none.
 *
 * The cell index is masked to a power of two, which is what makes the origin
 * snap seamless — the camera crossing a period boundary shifts the coordinate
 * by exactly `mask + 1` cells, and the mask makes that a no-op. Wrapping is
 * therefore not an approximation to hide a seam, it is what removes the seam.
 */
export function localNoiseBlock(sfx: string): string {
  return /* wgsl */ `
fn hashL_${sfx}(c0: vec3<i32>, mask: i32) -> vec3<f32> {
  var p = bitcast<vec3<u32>>(c0 & vec3<i32>(mask));
  p = p * 1664525u + vec3<u32>(1013904223u);
  p.x = p.x + p.y * p.z;
  p.y = p.y + p.z * p.x;
  p.z = p.z + p.x * p.y;
  p = p ^ (p >> vec3<u32>(16u));
  p.x = p.x + p.y * p.z;
  p.y = p.y + p.z * p.x;
  p.z = p.z + p.x * p.y;
  return -1.0 + 2.0 * vec3<f32>(p) * (1.0 / 4294967296.0);
}

fn noiseL_${sfx}(x: vec3<f32>, mask: i32) -> f32 {
  let i = floor(x);
  let w = x - i;
  let u = w * w * w * (w * (w * 6.0 - 15.0) + 10.0);
  let c = vec3<i32>(i);
  var acc = 0.0;
  for (var k = 0; k < 8; k = k + 1) {
    // "of" is a WGSL reserved keyword and the shader will not parse with it —
    // a name, not a technique, cost the whole terrain material one afternoon.
    let oi = vec3<i32>(k & 1, (k >> 1) & 1, (k >> 2) & 1);
    let og = vec3<f32>(oi);
    let tw = mix(vec3<f32>(1.0) - u, u, og);
    acc = acc + dot(hashL_${sfx}(c + oi, mask), w - og) * tw.x * tw.y * tw.z;
  }
  return acc;
}
`;
}

/**
 * Amplitude of the amplification at a point, metres.
 *
 * Its own function because two stages need it and neither should own it: the
 * vertex stage scales the detail it displaces by this, and the fragment stage
 * scales the octaves it *shades* by the same number — see the sub-mesh block in
 * shadeTerrain. Two copies of this would be the LESSONS §12 failure with the
 * roughness of every mountain riding on it.
 */
export function ampBlock(sfx: string): string {
  return /* wgsl */ `
fn ampAt_${sfx}(bakeH: f32, slope: f32, wet: f32) -> f32 {

  let relief = smoothstep(${f(RELIEF_SLOPE_LO)}, ${f(RELIEF_SLOPE_HI)}, slope);
  let landW = smoothstep(-350.0, 40.0, bakeH);
  // Floodplain. Tracks the river thresholds — the ground the amplification has
  // to leave flat is exactly the ground the water runs over.
  let valley = smoothstep(${f(VALLEY_WET_LO)}, ${f(VALLEY_WET_HI)}, wet);
  // Sea level is the one height where a small change alters the *topology* of
  // what is drawn rather than its shape, so it is the one place worth spending
  // detail to hold still. Collapsing the amplification within a few hundred
  // metres of it pins the shoreline to the bake, which does not depend on the
  // camera — so islands and lakes stop materialising on approach. Free where
  // it matters: relief above 400 m is untouched.
  let shore = mix(${f(SHORE_FLAT_FLOOR)}, 1.0,
                  smoothstep(0.0, ${f(SHORE_FLAT_HI)}, abs(bakeH)));
  let land = (${f(AMP_BASE)} + ${f(AMP_RELIEF)} * relief)
           * mix(0.22, 1.0, landW)
           * shore;
  // Toward an absolute floodplain amplitude, not a fraction of the hillslope
  // one — see FLOODPLAIN_AMP. This is what makes a river continuous.
  return mix(land, ${f(FLOODPLAIN_AMP)}, valley);
}
`;
}

/**
 * The elevation field. Needs `noiseBlock`. Expensive.
 *
 * Emits `ampBlock` itself rather than asking the caller to. It gained that
 * dependency when the amplitude was extracted for the fragment stage to share,
 * and three of the four call sites — grass, the vegetation scatter — did not
 * know: their WGSL referenced an undefined ampAt_ and the whole pipeline
 * silently failed to build, taking every blade and every tree with it.
 * A block that needs another one should carry it (LESSONS §8).
 */
export function field(s: string): string {
  return /* wgsl */ `
${ampBlock(s)}

/**
 * Elevation and its gradient: the baked surface plus amplification.
 *
 * The coarse structure no longer comes from noise. bakeH and bakeG are
 * sampled from the M3 global bake — tectonics, erosion and drainage solved on
 * a 1.57 M cell sphere (SPEC.md §4) — and all this adds is the detail finer
 * than the bake's 18 km cell, which is the amplification half of SPEC.md §6.
 *
 * That division is what the old placeholder could not express. Ridged fBm
 * raised to the 7th power can be tuned to Earth's *hypsometric curve* while
 * still having no rivers, no ranges and no reason for high ground to be where
 * it is, because there is no upstream anything in a local function.
 *
 * Two things condition the amplification, both from the bake:
 *
 *   slope    sub-grid roughness tracks resolved relief. A floodplain is flat
 *            at every scale; a range is rough at every scale. Using one
 *            amplitude everywhere is what made the old field read as noise.
 *   wetness  log₁₀ upstream drainage area. Amplitude falls in valley floors,
 *            so trunk rivers keep a smooth corridor instead of having ridges
 *            dropped across them — the amplification has to *respect* the
 *            drainage it is sitting on, or it destroys it.
 *
 * bandLimit is radius / vertex-spacing. Octaves finer than the mesh can
 * represent are faded out rather than evaluated: at LOD 10 the mesh samples
 * every 281 m while the finest octave has a 35 m wavelength, so it was 8x
 * undersampled and every vertex was landing on an essentially random phase of
 * it. Morphing then slid vertices through that field and the surface boiled.
 *
 * The weights are *not* renormalised. Dropping an octave must lower-pass the
 * surface, not rescale what is left, or the terrain would change shape as you
 * approached it (SPEC.md I2).
 *
 * The gain is not constant across octaves — it falls toward 1/lacunarity past
 * the hillslope crossover so slope stops growing with depth (planet.ts,
 * HILLSLOPE_WAVELENGTH). Because that is a function of the octave index alone
 * and not of position, the analytic gradient below stays exact and the band
 * limit above keeps meaning what it says.
 *
 * spec is (gain, crossover octave) for this point, from the biome table. It
 * is what makes a dune field, a badland and an alpine ridge different *shapes*
 * rather than the same shape at three sizes — see the gain field on Biome.
 * Pass REFERENCE_SPECTRUM for the old single-spectrum behaviour.
 */
fn height_${s}(dir: vec3<f32>, oct: i32, hscale: f32,
               bakeH: f32, bakeG: vec3<f32>, wet: f32, distAxis: f32,
               radius: f32, bandLimit: f32, spec: vec2<f32>) -> vec4<f32> {
  // dh/ds where s is arc length: the gradient is per unit direction, and a
  // unit of direction is one planet radius of surface.
  let slope = length(bakeG) / radius;
  let amp = ampAt_${s}(bakeH, slope, wet);

  var mAmp = 1.0; var mFrq = ${f(AMP_F0)}; var mSum = 0.0;
  var mG = vec3<f32>(0.0); var mSq = 0.0; var mBias = 0.0;
  // The reference ladder, walked alongside. Its two norms are what this
  // point's ladder is rescaled to — see the normaliser below.
  var rAmp = 1.0; var rSum = 0.0; var rSq = 0.0;
  // Position of the crossover on the ladder, in octaves, for this point and
  // for the reference. Both advance by a constant, so the whole per-point
  // crossover costs one add per octave and no logarithm anywhere.
  var mOct = spec.y; var rOct = ${f(REFERENCE_SPECTRUM[1])};
  for (var i = 0; i < oct; i = i + 1) {
    // Full weight while the wavelength spans BAND_FADE_HI samples, zero by
    // BAND_FADE_LO. The window is fitted, not chosen — see planet.ts.
    let w = smoothstep(${f(BAND_FADE_LO)}, ${f(BAND_FADE_HI)}, bandLimit / mFrq);
    if (w > 0.002) {
      let n = noised_${s}(dir * mFrq + vec3<f32>(f32(i) * 7.77));
      let sg = select(-1.0, 1.0, n.x >= 0.0);
      let r = 1.0 - abs(n.x);
      mSum  = mSum + w * mAmp * r * r;
      mG    = mG - w * mAmp * 2.0 * r * sg * mFrq * n.yzw;
      // r² has a positive mean, so the octaves that are switched on would
      // otherwise raise the ground by an amount that changes with distance —
      // moving the coastline as you fly toward it. Track the bias of exactly
      // the octaves in play and remove it.
      mBias = mBias + w * mAmp * ${f(RIDGE_MEAN)};
    }
    // Unweighted: the normalisers must not change with the band limit.
    mSq = mSq + mAmp * mAmp;
    rSum = rSum + rAmp;
    rSq = rSq + rAmp * rAmp;
    // Past the hillslope crossover the gain falls to 1/lacunarity, so relief
    // falls as λ¹ and every finer octave contributes the same slope instead of
    // 1.28x more. Without this the slope distribution diverges and the
    // steepest terrain renders as vertical facets — see HILLSLOPE_WAVELENGTH.
    // Soft rather than a select, because the crossover is now a per-point
    // quantity and a hard switch in it steps the ground along a climate
    // contour — see HILLSLOPE_SOFT. Still a constant per octave at any given
    // point, so the gradient above stays exact.
    mAmp = mAmp * mix(spec.x, ${f(HILLSLOPE_GAIN)},
                      smoothstep(${f(-HILLSLOPE_SOFT)}, ${f(HILLSLOPE_SOFT)}, mOct));
    rAmp = rAmp * mix(${f(RELIEF_GAIN)}, ${f(HILLSLOPE_GAIN)},
                      smoothstep(${f(-HILLSLOPE_SOFT)}, ${f(HILLSLOPE_SOFT)}, rOct));
    mOct = mOct + ${f(LOG2_LACUNARITY)};
    rOct = rOct + ${f(LOG2_LACUNARITY)};
    mFrq = mFrq * ${f(RELIEF_LACUNARITY)};
  }
  // Normalised so the *variance* of the detail is the same whatever spectrum
  // this point was given, not the sum of the ladder.
  //
  // The octaves are independent, so the detail's RMS goes as sqrt(Σa²)/Σa —
  // which moves with the gain. Dividing by Σa, as this did when there was one
  // spectrum for the whole planet, would have handed the smoothest biomes 15%
  // more total relief than the roughest and made the table an amplitude
  // control as well as a spectral one. Scaling sqrt(Σa²) back to the
  // reference ladder's own ratio holds the RMS to ±0.6% across the table and
  // leaves the reference spectrum bit-identical to Σa.
  let norm = sqrt(mSq) * rSum / sqrt(rSq);
  let detail = (mSum - mBias) / norm;
  let detailG = mG / norm;

  // The valley floor is cut down into a channel — a bowl CHANNEL_DEPTH deep at
  // the drainage axis, tapering out over a couple of channel widths. This is
  // the only part of the surface that is not band-limited to the mesh, and it
  // does not need to be: it is a smooth function of a smooth field, so it
  // refines continuously instead of arriving as new octaves do.
  let ch = channel_${s}(wet, distAxis);
  let h0 = (bakeH - ch.x + detail * amp) * hscale;
  // The amplitude field varies over the bake's cell size, three orders of
  // magnitude coarser than the detail it scales, so d(amp)/d(dir) is
  // negligible next to amp·d(detail)/d(dir) and is dropped.
  let g0 = (bakeG + detailG * amp) * hscale;

  let h = h0;
  let g = g0;

  return vec4<f32>(h, g);
}
`;
}

/**
 * Climate. Needs `noiseBlock`, and deliberately not the height field — the
 * fragment stage wants climate without paying for elevation it already has.
 */
export function climateBlock(s: string): string {
  return /* wgsl */ `

/**
 * Sea-level temperature and moisture, both in [0,1] — the two axes of a
 * Whittaker diagram, and between them the reason any patch of ground looks the
 * way it does at continental scale. See the climate block in planet.ts.
 *
 * Sea *level*, not the local surface: the lapse rate is applied separately by
 * tempAt_. Splitting it that way is what lets the vertex stage and the
 * fragment stage share one evaluation of the noise — they interpolate this
 * across the triangle and each applies the lapse rate to the elevation it
 * already has, so the two cannot disagree about where the treeline is.
 *
 * Cheap on purpose: two surviving noise octaves and arithmetic on values the
 * caller already has.
 */
fn climate_${s}(dir: vec3<f32>, wet: f32, bandLimit: f32) -> vec2<f32> {
  // |sin(latitude)|. The pole is +Y, the same convention the cube faces use.
  let lat = abs(dir.y);

  // Insolation by latitude. Height is not in here; see tempAt_.
  let temp = pow(max(1.0 - lat * lat, 0.0), ${f(LAT_EXP)});

  // Moisture provinces at 2000 km / 790 km / 300 km. The coarsest octave has
  // to survive the band limit at level 0 or the planet loses all large-scale
  // colour from orbit, which is the failure this whole field exists to fix:
  // at bandLimit 22 the first octave still carries full weight.
  var amp = 1.0;
  var frq = ${f(BIOME_F0)};
  var sum = 0.0;
  var norm = 0.0;
  for (var i = 0; i < 3; i = i + 1) {
    let w = smoothstep(1.0, 2.5, bandLimit / frq);
    if (w > 0.002) {
      sum = sum + w * amp * noised_${s}(dir * frq + vec3<f32>(f32(i) * 19.13 + 5.0)).x;
    }
    norm = norm + amp;
    amp = amp * ${f(BIOME_GAIN)};
    frq = frq * ${f(BIOME_LACUNARITY)};
  }
  let province = sum / norm;

  // The Hadley signature: wet equator, dry belt near |sin lat| = 0.45, wet
  // again at mid latitudes. Without it the provinces are just coloured noise;
  // with it, deserts sit in a band where deserts actually sit, and that band
  // is the single most recognisable thing about a planet seen from orbit.
  let q = (lat - 0.45) / 0.17;
  let subtropics = exp(-q * q);

  // Upstream drainage area. Valley floors are wetter than the ground above
  // them at any latitude, which is what puts trees along rivers in dry
  // country — and it is the only use the baked wetness channel had beyond
  // flattening the amplification in trunk valleys.
  // Broad valley-floor dampness, not the river itself. This carried the whole
  // riparian signal at 0.22 of the moisture axis — a biome window is about 0.2
  // wide, so it moved the biome outright across a corridor tens of kilometres
  // across, which is what made the drainage a smudge. The sharp part of the
  // signal now comes from the distance field per pixel (see the river corridor
  // block in shadeTerrain); what is left here is the genuinely broad effect —
  // a floodplain really is damper than the hills either side, over kilometres.
  let riparian = smoothstep(8.6, 10.2, wet);

  let moist = clamp(0.58 + province * 1.6 - subtropics * 0.55 + riparian * 0.07,
                    0.0, 1.0);
  return vec2<f32>(temp, moist);
}

`;
}

/**
 * The lapse rate alone.
 *
 * Its own block because the fragment stage needs exactly this and nothing else
 * from the climate — emitting `climateBlock` there would drag in `noised_`,
 * which the fragment shader has no other use for.
 */
export function lapseBlock(s: string): string {
  return /* wgsl */ `
/**
 * Temperature at elevation h, from the sea-level value.
 *
 * 6.5 K/km, expressed in the 0..1 units the rest of the climate uses. Every
 * consumer applies this to the same elevation, so the treeline, the snow line
 * and where the scatter puts instances are the same boundary by construction
 * rather than by three constants agreeing.
 */
fn tempAt_${s}(base: f32, h: f32) -> f32 {
  return clamp(base - max(h, 0.0) * ${f(LAPSE)}, 0.0, 1.0);
}
`;
}

/**
 * The spectrum `height_` should use here: (fBm gain, crossover octave).
 *
 * Needs `noiseBlock`, `lapseBlock` and `biomeBlock`. One noise octave — the
 * seasonality swing — plus the biome weights, which are arithmetic on values
 * the caller already has.
 *
 * ── why the elevation argument is the *baked* one ──────────────────────────
 *
 * Two of the biome's axes are elevation: the lapse rate, and montane
 * grassland's window. The obvious thing to pass is the elevation being
 * computed, which is circular — the spectrum sets the detail and the detail
 * sets the elevation. Passing the bake's own elevation breaks the loop and is
 * also the more defensible model: the amplification is by definition the
 * detail *below* the bake's cell, and what kind of country it is should be a
 * property of the landscape the bake resolved, not of the metres this function
 * is about to invent. The two differ by at most the amplification's own
 * amplitude, against a 450 m window softness.
 *
 * ── why the band limit does not have to be threaded in ─────────────────────
 *
 * `clim` may be evaluated at the caller's band limit rather than at full
 * resolution, and a spectrum that changed with distance would change the
 * *shape* of the terrain as you approached it (SPEC.md I2). It cannot here.
 * Climate's finest octave is at frequency 21 and saturates by a band limit of
 * 52; the amplification's *first* octave does not switch on until the band
 * limit passes 560. There is an order of magnitude between the two, so
 * wherever the spectrum can affect anything, climate is already fully
 * resolved.
 */
export function spectrumBlock(s: string): string {
  return /* wgsl */ `
fn spectrum_${s}(dir: vec3<f32>, clim: vec2<f32>, bakeH: f32) -> vec2<f32> {
  let swing = noised_${s}(dir * ${f(SEASON_F0)} + vec3<f32>(61.7)).x;
  return biomeSpectrum_${s}(tempAt_${s}(clim.x, bakeH), clim.x, clim.y,
                            season_${s}(abs(dir.y), swing), bakeH);
}
`;
}

/**
 * Canopy closure from climate and terrain. No noise, so the fragment stage can
 * have it without dragging in `noiseBlock`.
 */
export function closureBlock(s: string): string {
  return /* wgsl */ `
/**
 * Closure the climate supports, before local clumping. Cold and drought are
 * separate gates, not one: the treeline and the desert edge are different
 * boundaries and a single "growth" term cannot put both where they belong.
 */
fn canopyClosure_${s}(temp: f32, moist: f32, h: f32, slope: f32) -> f32 {
  // The cold gate has to close *before* the snow line opens, or the two
  // overlap and trees stand on permanent ice.
  //
  // It used to be [0.06, 0.26], and the snow line in shadeTerrain fades in
  // over [0.14, 0.015] going down in temperature — so there was a whole band,
  // temp in [0.06, 0.14], where the ground was painted as snow and the scatter
  // still placed stems on it. Measured over the sphere that was 2.09% of all
  // land: trees fully gone only at 77.6° while snow starts at 73.4°, plus the
  // same overlap in altitude everywhere via the lapse rate.
  //
  // The ground tint already multiplied its canopy by (1 - snowLine) and the
  // scatter had no equivalent, which is why the tint went white and the trees
  // stayed — the two disagreed about a boundary that is supposed to be one
  // function. Fixing it here fixes both, which is the point of the shared
  // closure.
  //
  // [0.18, 0.38] leaves a deliberate gap: full canopy to 63.2°, thinning out
  // by 71.5°, bare ground from there to the snow line at 73.4°. That gap is
  // alpine tundra, and it is what the sequence forest → bare → snow looks like
  // on a real mountainside and on a real Arctic coast.
  let warm = smoothstep(0.18, 0.38, temp);
  let wetEnough = smoothstep(0.26, 0.55, moist);
  let lo = smoothstep(${f(VEG_MIN_ELEVATION)}, ${f(VEG_MIN_ELEVATION)} + 30.0, h);
  let sl = 1.0 - smoothstep(${f(VEG_SLOPE_FULL)}, ${f(VEG_MAX_SLOPE)}, slope);
  return clamp(warm * wetEnough * lo * sl, 0.0, 1.0);
}
`;
}

/** The canopy clump. Needs `noiseBlock` and `closureBlock`. */
export function coverBlock(s: string): string {
  return /* wgsl */ `

/**
 * Local canopy texture, normalised so that losing it costs nothing.
 *
 * Three octaves at 7 km / 3 km / 1.3 km — glades, stand edges, dense pockets.
 * Without this the forest is a uniform mat, which reads as fake from the air
 * long before any individual tree does.
 *
 * The division by CLUMP_MEDIAN is the whole fix for the orbital view. This
 * returns a *modulation*, centred on 1, not a cover fraction: when every
 * octave has been band-limited away the sum is zero, the smoothstep returns
 * CLUMP_MEDIAN, and the quotient is exactly 1 — the detail disappears and
 * leaves whatever the climate said, which is what a forest genuinely looks
 * like once its glades are sub-pixel. Before, the same situation returned
 * 0.673 *as the cover itself*, so every continent below the treeline took the
 * same 56% canopy tint and the planet went one flat colour.
 */
fn forestClump_${s}(dir: vec3<f32>, bandLimit: f32) -> f32 {
  // Six octaves, 7.1 km down to 110 m. Three stopped at 1.3 km, which is
  // landscape-scale patchiness and nothing you could ever stand in: from
  // inside the wood the canopy was an unbroken mat to the horizon, which is
  // both wrong and what has been hiding the ground — and with it the grass.
  // A clearing is tens to a couple of hundred metres across, so the field has
  // to reach that far down.
  var amp = 1.0;
  var frq = 900.0;
  var sum = 0.0;
  var norm = 0.0;
  for (var i = 0; i < 6; i = i + 1) {
    let w = smoothstep(1.0, 2.5, bandLimit / frq);
    if (w > 0.002) {
      sum = sum + w * amp * noised_${s}(dir * frq + vec3<f32>(f32(i) * 11.37)).x;
    }
    norm = norm + amp;
    amp = amp * 0.55;
    frq = frq * 2.3;
  }
  let raw = sum / norm;

  // Median measured at 0.6737 and, usefully, unchanged by the octave count —
  // so CLUMP_MEDIAN still normalises this to 1 when every octave is banded
  // away, which is the property the orbital view depends on.
  let c = smoothstep(-0.42, 0.26, raw) / ${f(CLUMP_MEDIAN)};
  let thin = mix(1.0, c, ${f(CLUMP_STRENGTH)});

  // Clearings, as a separate term, because more octaves alone do not make
  // them: averaging octaves *narrows* the distribution, so the field gets
  // finer structure and a shorter tail at the same time — measured, the 10th
  // percentile rose from 0.333 to 0.376 going from three octaves to six. Real
  // forest is not uniformly thinned, it has holes in it: blowdowns, rock,
  // burns, boggy ground. This punches the lowest decile or so of the field
  // down to near nothing and leaves the rest alone.
  let gap = smoothstep(-0.50, -0.14, raw);
  return thin * mix(0.12, 1.0, gap);
}

/**
 * Fraction of ground under canopy, in [0,1].
 *
 * Shared verbatim by the scatter and the terrain shading. That is the whole
 * point: the ground tint and where plants actually stand are the same
 * function, so they cannot disagree, and instances can dissolve into the tint
 * without revealing an edge.
 */
fn forestCover_${s}(dir: vec3<f32>, h: f32, slope: f32, density: f32,
                    bandLimit: f32, clim: vec2<f32>) -> f32 {
  let closure = canopyClosure_${s}(clim.x, clim.y, h, slope);
  return clamp(closure * forestClump_${s}(dir, bandLimit) * density, 0.0, 1.0);
}
`;
}

const ARGS = `gpos: vec2<f32>, gpar: vec3<f32>,
   iCenter: vec4<f32>, iDirLen: vec4<f32>, iAnchor: vec4<f32>,
   iBU: vec3<f32>, iBV: vec3<f32>, iMorph: vec2<f32>,
   cfg: vec4<f32>, cfg2: vec4<f32>, cfg3: vec4<f32>`;

const UNPACK = `
  let A = iCenter.x;
  let B = iCenter.y;
  let hs = iCenter.z;
  let dirC = iDirLen.xyz;
  let lenPc = iDirLen.w;
  let anchorRel = iAnchor.xyz;
  let radius = cfg.x;
  let Pc = dirC * lenPc;`;

/**
 * (normal.xyz, elevation). Carries the whole cost of the noise field; the node
 * is reused by both the position and the fragment stage, so it runs once.
 */
/**
 * Unit direction and vertex spacing for this vertex.
 *
 * Exists purely so the bake can be sampled *before* `patchSurface` runs.
 * Texture reads cannot happen inside a `wgslFn` — TSL has no way to bind a
 * texture to one — so the sampling is done in the node graph and the results
 * are passed in. The morph is therefore evaluated once more per vertex, which
 * is about twenty ALU ops against the hundreds the noise field costs.
 */
/**
 * Direction to atlas UV: the third and last copy of the cube-face mapping.
 *
 * The other two are `cubeTexelDirection` in bake/cubemap.ts, which writes the
 * atlas, and `faceCoords` in planetData.ts, which reads it on the CPU. All
 * three must agree; tools/mirror.ts is what proves the CPU and the asset do,
 * and the terrain simply looking correct is what proves this one does.
 *
 * atlas is (faceSize, atlasWidth, atlasHeight, pad).
 */
export const cubeAtlasUV = wgslFn(/* wgsl */ `
fn cubeAtlasUV(dir: vec3<f32>, atlas: vec4<f32>) -> vec2<f32> {
  let a = abs(dir);
  var face: i32;
  var sc: f32;
  var tc: f32;
  var ma: f32;
  if (a.x >= a.y && a.x >= a.z) {
    ma = a.x;
    if (dir.x > 0.0) { face = 0; sc = -dir.z; } else { face = 1; sc = dir.z; }
    tc = -dir.y;
  } else if (a.y >= a.z) {
    ma = a.y;
    if (dir.y > 0.0) { face = 2; tc = dir.z; } else { face = 3; tc = -dir.z; }
    sc = dir.x;
  } else {
    ma = a.z;
    if (dir.z > 0.0) { face = 4; sc = dir.x; } else { face = 5; sc = -dir.x; }
    tc = -dir.y;
  }

  let n = atlas.x;
  let pad = atlas.w;
  let cell = n + 2.0 * pad;
  let col = f32(face % 3);
  let row = f32(face / 3);
  let s = (sc / ma + 1.0) * 0.5;
  let t = (tc / ma + 1.0) * 0.5;
  // Texel centres: the face occupies [pad, pad+n) of its cell, and the border
  // ring outside that carries the neighbouring face so filtering at s or t
  // exactly 0 or 1 blends across the seam instead of clamping.
  return vec2<f32>((col * cell + pad + s * n) / atlas.y,
                   (row * cell + pad + t * n) / atlas.z);
}
`);

export const patchDirection = wgslFn(/* wgsl */ `
fn patchDirection(${ARGS}) -> vec4<f32> {
  ${UNPACK}
  let m = morphed_D(gpos, gpar, A, B, hs, Pc, lenPc, iBU, iBV,
                    anchorRel, radius, cfg2.z, iMorph, cfg.w);
  let dd = offset_D(m.xy, A, B, hs, Pc, lenPc, iBU, iBV);
  return vec4<f32>(dirC + dd, cfg2.w * hs * (1.0 + m.z));
}
${geom('D')}
`);

/**
 * Sea-level climate for this vertex.
 *
 * Separate from `patchSurface` because both the vertex and the fragment stage
 * need it: this runs once, the result is interpolated, and each side applies
 * the lapse rate to the elevation it already has. `dirSp` is `patchDirection`'s
 * output — direction in xyz, vertex spacing in w — so the direction and the
 * band limit are the same values `patchSurface` derives, not a second guess at
 * them.
 */
/**
 * Carries the canopy *clump* as well, which is why it is a vec3.
 *
 * The clump is three noise octaves at 7 / 3 / 1.3 km and belongs per vertex —
 * it is the expensive half of the cover. Everything else the cover needs is
 * cheap arithmetic on values the fragment stage already has, so the closure is
 * evaluated per pixel instead. That is what frees `patchSurface` to spend all
 * four of its channels on the normal and the elevation.
 */
export const patchClimate = wgslFn(/* wgsl */ `
fn patchClimate(dirSp: vec4<f32>, wet: f32, radius: f32) -> vec3<f32> {
  let bandLimit = radius / max(dirSp.w, 0.01);
  let c = climate_C(dirSp.xyz, wet, bandLimit);
  return vec3<f32>(c, forestClump_C(dirSp.xyz, bandLimit));
}
${noiseBlock('C')}
${climateBlock('C')}
${closureBlock('C')}
${coverBlock('C')}
`);

/**
 * (normal.xyz, elevation).
 *
 * The normal used to be octahedral-packed into two channels so the fourth
 * could carry canopy cover. That is a sound way to *store* a unit vector and a
 * wrong way to interpolate one: octDecode folds on the sign of 1−|x|−|y|,
 * which is neither linear nor continuous, so interpolating in oct space and
 * decoding afterwards creases the normal along the octahedron's seams — and
 * those seams are axis-aligned and diagonal. Lit, that showed up as hard-edged
 * facets following the mesh grid, a maze of contours over the whole surface,
 * worst wherever the normal swung most across a single triangle. Raising the
 * amplification amplitude made it unmissable.
 *
 * Interpolating the raw vector and renormalising per pixel is exact to within
 * the usual linear-interpolation error, has no discontinuity anywhere, and
 * costs one channel — paid for by moving the clump into `patchClimate`.
 */
/**
 * (fBm gain, crossover octave) for this vertex — the spectrum of the ground
 * under it, from the biome table.
 *
 * Its own node because two consumers need the same value: `patchSurface` puts
 * it through `height_`, and the fragment stage continues that ladder below the
 * mesh. Computing it here and passing it to both is one evaluation; the
 * alternative is a second one in the fragment stage against a slightly
 * different elevation, which would leave the two halves of the ladder
 * disagreeing about what kind of country this is.
 *
 * Takes the climate `patchClimate` already computed, so the only noise it pays
 * for is the one seasonality octave.
 */
export const patchSpectrum = wgslFn(/* wgsl */ `
fn patchSpectrum(dir: vec3<f32>, clim: vec2<f32>, bakeH: f32) -> vec2<f32> {
  return spectrum_S(dir, clim, bakeH);
}
${noiseBlock('S')}
${lapseBlock('S')}
${biomeBlock('S')}
${spectrumBlock('S')}
`);

export const patchSurface = wgslFn(/* wgsl */ `
fn patchSurface(${ARGS}, baked: vec4<f32>, bake2: vec4<f32>, spec: vec2<f32>) -> vec4<f32> {
  ${UNPACK}
  let m = morphed_N(gpos, gpar, A, B, hs, Pc, lenPc, iBU, iBV,
                    anchorRel, radius, cfg2.z, iMorph, cfg.w);
  let g = m.xy;
  let dd = offset_N(g, A, B, hs, Pc, lenPc, iBU, iBV);
  let dir = dirC + dd;

  // Vertex spacing for this patch, widened by the morph: a fully morphed patch
  // is drawing its parent's grid. cfg2.w is faceEdge/segments, so cfg2.w * hs
  // is the unmorphed spacing.
  let spacing = cfg2.w * hs * (1.0 + m.z);
  let bandLimit = radius / max(spacing, 0.01);

  let hn = height_N(dir, i32(cfg.z), cfg.y, baked.x, baked.yzw, bake2.x, bake2.y,
                    radius, bandLimit, spec);

  // Surface normal from the tangential component of the height gradient.
  let gT = hn.yzw - dir * dot(dir, hn.yzw);
  let nrm = normalize(dir - gT / (radius + hn.x));

  return vec4<f32>(nrm, hn.x);
}
${geom('N')}
${noiseBlock('N')}
${channelBlock('N')}
${field('N')}
`);

/**
 * Standing-water surface for one vertex, from the three baked channels.
 *
 * Split out so the node graph can call it: the caller has the atlas sample and
 * `patchSurface`'s unflooded elevation, and needs both to get a depth. Doing it
 * inside `patchSurface` would mean returning five values through a vec4.
 */
/**
 * The amplification amplitude for this vertex, so the fragment stage can scale
 * the octaves the mesh could not carry by the same number the mesh used. Runs
 * the shared ampAt_ — no noise, a handful of smoothsteps.
 */
export const patchAmp = wgslFn(/* wgsl */ `
fn patchAmp(bakeH: f32, bakeG: vec3<f32>, wet: f32, radius: f32) -> f32 {
  return ampAt_A(bakeH, length(bakeG) / radius, wet);
}
${ampBlock('A')}
`);

export const patchWater = wgslFn(/* wgsl */ `
fn patchWater(bakeH: f32, lakeD: f32, wet: f32, distAxis: f32) -> f32 {
  return waterLevel_W(bakeH, lakeD, wet, distAxis);
}
${channelBlock('W')}
${waterBlock('W')}
`);

/**
 * Camera-relative vertex position. Takes the elevation from `patchSurface`
 * rather than recomputing it, so this side is pure geometry.
 */
export const patchPosition = wgslFn(/* wgsl */ `
fn patchPosition(${ARGS}, hgt: f32) -> vec3<f32> {
  ${UNPACK}
  let g = morphed_P(gpos, gpar, A, B, hs, Pc, lenPc, iBU, iBV,
                    anchorRel, radius, cfg2.z, iMorph, cfg.w).xy;
  let dd = offset_P(g, A, B, hs, Pc, lenPc, iBU, iBV);

  // Skirt depth scales with this patch's own vertex spacing, but the gap it
  // has to cover is bounded by terrain relief, not by spacing — at level 0,
  // 8x the spacing would be 2300 km and would collapse the planet. Cap it at
  // the largest plausible height discontinuity.
  // cfg2.w is faceEdge/segments; hs is the patch half-size in face-uv.
  let skirt = gpar.z * min(8.0 * cfg2.w * hs, 3000.0);

  //  dir·(R+h) - camPos  ==  (dirC·R - camPos) + dirC·h + ddir·(R+h)
  //  the first term arrived pre-subtracted in f64; the rest are small.
  return anchorRel + dirC * (hgt - skirt) + dd * (radius + hgt);
}
${geom('P')}
`);

/**
 * Terrain shading.
 *
 * `surf` is (normal.xyz, elevation), interpolated and renormalised here. See
 * `patchSurface` for why the normal is not packed.
 *
 * Roughly physical: albedo is lit by sunlight that has already been attenuated
 * by the air it travelled through, plus a hemispheric sky term, and the result
 * then goes through aerial perspective. That last step is what makes distance
 * read correctly and, incidentally, what hides every LOD and vegetation
 * transition beyond a few hundred metres.
 *
 * `clim` is (sea-level temperature, moisture, canopy clump) from
 * `patchClimate`, interpolated. The lapse rate and the growth gates are applied
 * here, against this pixel's own elevation and slope — so cover is a per-pixel
 * quantity now, and no longer inherits the mesh's faceting.
 *
 * `lvl` is (quadtree level, vertex spacing, fBm gain, crossover octave). The
 * last two are `patchSpectrum`'s output, carried rather than recomputed: the
 * detail-normal ladder below has to walk the *same* ladder the mesh did or its
 * residual is not the band the mesh is missing.
 *
 * mode: 0 = natural, 1 = LOD level, 2 = slope, 3 = normals, 4 = canopy cover,
 * 5 = albedo, 6 = climate. Debug modes skip the atmosphere so they stay
 * legible.
 */
export const shadeTerrain = wgslFn(/* wgsl */ `
fn shadeTerrain(surf: vec4<f32>, clim: vec4<f32>, camPos: vec3<f32>, rel: vec3<f32>,
                lvl: vec4<f32>, sunDir: vec3<f32>, sunCol: vec3<f32>,
                mode: f32, grid: f32, cfg3: vec4<f32>, skyView: vec4<f32>,
                snap: vec3<f32>,
                shadow: f32) -> vec3<f32> {
  let n = normalize(surf.xyz);
  let hgt = surf.w;
  let Rg = cfg3.y;
  let temp = tempAt_T(clim.x, hgt);
  let moist = clim.y;
  // ── the waterline ─────────────────────────────────────────────────────
  //
  // Signed height of the waterline above the ground: positive is the surface
  // of the sea, a lake or a river, negative is dry land that far below it.
  // See patchWater, and the signed-depth note in terrainMesh.ts.
  //
  // This used to be a hard "if (depth > 0.0)". A binary test on a smooth field
  // is a step function, and a step function sampled once per pixel aliases —
  // which is invisible where a coast is steep, because the transition happens
  // inside one pixel anyway, and severe on a flat coastal plain seen at a
  // grazing angle, where one pixel spans tens of metres of ground and the
  // waterline wanders across many pixels' worth of it. That is what turned
  // shallow coastal water into a scatter of disconnected chips: each pixel
  // independently voted "all water" or "all land" on a value that was, over
  // that pixel's actual footprint, somewhere in between.
  //
  // fwidth is the change in the signed depth over one pixel, i.e. the pixel's
  // own footprint expressed in the units of the thing being thresholded. So
  // depth/footprint is how many pixels away the waterline is, and clamping
  // it around the crossing gives exactly the coverage that pixel should have.
  // The band is one pixel wide wherever you stand: sub-pixel on a cliff, tens
  // of metres of ground on a plain, and never a hard edge.
  let sDepth = clim.w;
  let depth = max(sDepth, 0.0);
  let dFoot = max(fwidth(sDepth), 1e-4);
  let waterMix = clamp(sDepth / dFoot + 0.5, 0.0, 1.0);
  var waterCol = vec3<f32>(0.0);

  // Planet-centred surface point. Local up per pixel, not per patch — a
  // per-instance value steps visibly at every patch boundary.
  let wp = camPos + rel;
  let up = normalize(wp);
  let sd = normalize(sunDir);

  // ── the octaves the mesh could not carry ──────────────────────────────
  //
  // CDLOD holds roughly nine pixels per vertex, so the mesh always stops about
  // three octaves above what the screen can show. At 219 km that is a vertex
  // every 2.25 km against a 247 m pixel: the surface has content at 12 km and
  // 5.9 km and then *nothing* down to the pixel, and since slope, aspect and
  // curvature all come from that normal, every albedo term inherits the hole.
  // It is why the mid-altitude view read as a smudge — not missing texture,
  // missing topography for the texture to follow.
  //
  // So continue the amplification's own ladder here. Not a stipple: the same
  // ridged octaves at the same frequencies with the same gain schedule and the
  // same amplitude, evaluated as a gradient only. The weight of each octave is
  // what this pixel can show minus what the mesh already displaced, so the term
  // is exactly the missing band and nothing else — it self-cancels to zero at
  // ground level, where the mesh reaches 55 cm and carries everything.
  //
  // Four pixels per wavelength rather than the mesh's eight. Eight is the right
  // rule for a *displaced* octave on a morphing grid, where a vertex landing on
  // an arbitrary phase makes the surface boil (LESSONS §4). Nothing moves here,
  // the sampling is one per pixel by construction, and four is still twice
  // Nyquist — that extra octave is most of what this buys.
  let mPerPx = max(length(fwidth(rel)), 0.02);
  let pixLam = max(mPerPx * 4.0, ${f(MIN_NOISE_LAMBDA)});
  let meshBand = cfg3.y / max(lvl.y, 0.01);
  let pixBand = cfg3.y / pixLam;

  // lvl.zw is this pixel's spectrum, interpolated from the same value the
  // vertex stage put through height_ — so the two halves of the ladder are one
  // ladder, and the octaves added here are the ones the mesh dropped rather
  // than a differently-shaped set at the same frequencies.
  var dG = vec3<f32>(0.0);
  var dAmp = 1.0;
  var dFrq = ${f(AMP_F0)};
  var dSq = 0.0;
  var rAmp = 1.0; var rSum = 0.0; var rSq = 0.0;
  var dOct = lvl.w; var rOct = ${f(REFERENCE_SPECTRUM[1])};
  for (var i = 0; i < ${DEFAULT_OCTAVES}; i = i + 1) {
    let wMesh = smoothstep(${f(BAND_FADE_LO)}, ${f(BAND_FADE_HI)}, meshBand / dFrq);
    let wPix = smoothstep(${f(BAND_FADE_LO)}, ${f(BAND_FADE_HI)}, pixBand / dFrq);
    let w = clamp(wPix - wMesh, 0.0, 1.0);
    if (w > 0.004) {
      let nz = noised_T(up * dFrq + vec3<f32>(f32(i) * 7.77));
      let sg = select(-1.0, 1.0, nz.x >= 0.0);
      let rr = 1.0 - abs(nz.x);
      dG = dG - w * dAmp * 2.0 * rr * sg * dFrq * nz.yzw;
    }
    dSq = dSq + dAmp * dAmp;
    rSum = rSum + rAmp;
    rSq = rSq + rAmp * rAmp;
    dAmp = dAmp * mix(lvl.z, ${f(HILLSLOPE_GAIN)},
                      smoothstep(${f(-HILLSLOPE_SOFT)}, ${f(HILLSLOPE_SOFT)}, dOct));
    rAmp = rAmp * mix(${f(RELIEF_GAIN)}, ${f(HILLSLOPE_GAIN)},
                      smoothstep(${f(-HILLSLOPE_SOFT)}, ${f(HILLSLOPE_SOFT)}, rOct));
    dOct = dOct + ${f(LOG2_LACUNARITY)};
    rOct = rOct + ${f(LOG2_LACUNARITY)};
    dFrq = dFrq * ${f(RELIEF_LACUNARITY)};
  }
  // Same normaliser as height_, for the same reason and so the two agree.
  // skyView.w is the same amplitude the vertex stage scaled its detail by.
  // Under water the surface is flat and none of this applies.
  let dn = dG / (sqrt(dSq) * rSum / sqrt(rSq)) * skyView.w
         * (1.0 - smoothstep(0.0, 0.5, depth));
  // Same expression patchSurface uses to turn a height gradient into a normal,
  // applied to the residual gradient — so the shaded surface is continuous with
  // the displaced one instead of being a different surface laid over it.
  let dnT = dn - up * dot(up, dn);
  let n2 = normalize(n - dnT / (Rg + hgt));
  let slope = clamp(1.0 - dot(n2, up), 0.0, 1.0);

  // ── aspect ────────────────────────────────────────────────────────────
  //
  // Which way the slope faces, relative to the pole of its own hemisphere.
  // Pole-facing ground takes the sun at a shallower angle all year, so it is
  // cooler, holds its moisture, and carries forest hundreds of metres higher
  // than the equator-facing slope directly opposite. In real mountains that
  // asymmetry is one of the most legible things there is — one side of a
  // valley wooded, the other side grass and rock — and nothing here knew about
  // it, so every ridge was symmetric and read as a heightfield rather than as
  // a place.
  let poleAx = vec3<f32>(0.0, select(-1.0, 1.0, wp.y >= 0.0), 0.0);
  let poleT = poleAx - up * dot(up, poleAx);
  let nT = n2 - up * dot(up, n2);
  // Zero on flat ground and at the poles themselves, where there is no aspect
  // to speak of; the length product handles both without a branch.
  let aspect = dot(nT, poleT) / max(length(poleT), 1e-4);

  // ── cloud shadow ──────────────────────────────────────────────────────
  //
  // Where the sun ray from this point crosses the cloud deck, sample the same
  // field the deck draws and dim by what is up there. Cheap — one intersection
  // and one cloud field — and it is the single largest thing missing from the
  // ground: a landscape under a broken deck is mostly *in shadow*, moving, and
  // that motion is what makes a sky feel like weather rather than wallpaper.
  //
  // The intersection is with a sphere of radius Rg + CLOUD_ALT from a point
  // essentially on Rg, solved directly rather than by marching.
  let sunUpT = dot(up, sd);
  var cloudLit = 1.0;
  if (sunUpT > 0.02) {
    let rc = Rg + ${f(CLOUD_ALT)};
    let b = dot(wp, sd);
    let c0 = dot(wp, wp) - rc * rc;
    let disc = b * b - c0;
    if (disc > 0.0) {
      let hitDir = normalize(wp + sd * (-b + sqrt(disc)));
      // The shadow is cast by the same field that is drawn, at the same
      // detail this pixel can resolve — see the px note in cloudFieldBlock.
      // A shadow edge softer than the cloud edge is also physically right:
      // that is the penumbra.
      let cf = cloudField_T(hitDir, cfg3.w, cfg3.z, mPerPx);
      // Cumulus blocks most of the beam; cirrus only veils it. Never fully
      // dark: a cloud shadow on a real landscape is still lit by the rest of
      // the sky, and that fill is most of its colour.
      let block = clamp(cf.x * cf.x * 0.82 + cf.y * 0.18, 0.0, 1.0);
      // Faded out at grazing sun, where the intersection is kilometres away
      // laterally and the shadow does not belong under this pixel at all.
      cloudLit = 1.0 - block * smoothstep(0.02, 0.20, sunUpT);
    }
  }

  // Cover, per pixel. clim.z is the interpolated clump — the expensive part,
  // three noise octaves, still evaluated per vertex; the growth gates are a
  // handful of smoothsteps and are better applied here, against this pixel's
  // own elevation and slope, than baked into a varying.
  // Aspect shifts the effective moisture, which is the axis both the ground
  // colour and the canopy already key on — so one term moves the tree line,
  // the biome and the scatter density together and they cannot disagree.
  // ── river corridor ────────────────────────────────────────────────────
  //
  // Drawn from the baked distance-to-axis field, not by thresholding wetness.
  //
  // Wetness is log10 upstream drainage area: broad, smooth, and stored at 9 km,
  // so a threshold on it paints a corridor tens of kilometres wide with soft
  // edges — and it then arrives here as a vertex varying interpolated across
  // another 9 km. The network was topologically perfect and read as a smudge.
  // The distance field interpolates to a sub-texel zero set instead, which is
  // exactly why carveChannels bakes it (LESSONS §5).
  //
  // Half-width from the drainage area on the same rule the channel incision
  // uses, then widened: a gallery corridor is several times the channel it
  // follows, because the floodplain is what is green, not the water.
  let wetV = skyView.z;
  let ripHalf = clamp(${f(CHANNEL_WIDTH_K)} * 5.5 * sqrt(pow(10.0, wetV)),
                      320.0, 6500.0);
  // Prefiltered rather than thresholded. Below a pixel a corridor cannot be
  // drawn, only aliased, so it is widened to a pixel and dimmed by the same
  // factor — which is what area-averaging it would have done, and it is what
  // lets a tributary fade out with distance instead of flickering.
  let ripW = max(ripHalf, mPerPx * 1.5);
  // ── the meander ───────────────────────────────────────────────────────
  //
  // The baked axis is a Dijkstra distance from D8 channels, and D8 picks one
  // of eight directions per cell — so the network it descends from is
  // grid-locked, and the corridor inherits straight runs meeting at 45° and
  // 90° (LESSONS §5). The bake cannot be un-gridded without re-running the
  // LEM, but the *drawn* river does not have to inherit it.
  //
  // Displacing a distance field bends its zero set sideways without breaking
  // it: add a smooth field to the distance and the corridor wanders while
  // staying continuous, staying the right width, and staying attached to the
  // drainage it came from. Two octaves at 20 km and 5.5 km, which is the scale
  // real trunk rivers meander on, and the amplitude is in units of the
  // corridor's own width so a big river swings wider than a creek.
  //
  // Fixed frequencies, no band limit: the meander is a property of the river,
  // not of the camera, and it must not move as you approach.
  let mw = noised_T(up * 320.0 + vec3<f32>(11.7)).x * 0.64
         + noised_T(up * 1150.0 + vec3<f32>(53.1)).x * 0.36;
  let dAxis = max(skyView.y + mw * ripHalf * 1.6, 0.0);

  let corridor = (1.0 - smoothstep(0.30, 1.0, dAxis / ripW))
               * (ripHalf / ripW)
               * smoothstep(${f(VALLEY_WET_LO)} - 1.6, ${f(VALLEY_WET_LO)} + 0.4, wetV);

  let moistA = clamp(moist + aspect * 0.085 + corridor * 0.20, 0.0, 1.0);
  let cover = clamp(canopyClosure_T(temp, moistA, hgt, slope) * clim.z * cfg3.x,
                    0.0, 1.0);

  // Metric reference grid: the instrument for M1's jitter check.
  var overlay = 0.0;
  if (grid > 0.0) {
    let q = rel / grid;
    let w = fwidth(q);
    let f = abs(fract(q - vec3<f32>(0.5)) - vec3<f32>(0.5)) / max(w, vec3<f32>(1e-8));
    overlay = 1.0 - min(min(f.x, min(f.y, f.z)), 1.0);
  }
  let gridCol = vec3<f32>(1.0, 0.9, 0.3);

  if (mode > 6.5) {
    // Absolute elevation, encoded for a pixel readback rather than for the eye.
    //
    // This used to be 50 m contours, which can only ever show that the CPU and
    // the GPU agree *modulo 50 m* — and the divergence this exists to catch is
    // measured in hundreds. Two bytes over a ±12 km range resolve 0.37 m, which
    // is well under the camera's 1.7 m clearance, so a disagreement that
    // matters cannot hide in the quantisation.
    //
    //   hgt = ((R + G / 255) / 255) * 24000 - 12000
    //
    // Read it with sim.probeHeight(); tools/gpuHeight.ts drives that against
    // heightAt for a set of directions.
    let v = clamp((hgt + 12000.0) / 24000.0, 0.0, 1.0);
    return vec3<f32>(floor(v * 255.0) / 255.0, fract(v * 255.0), 0.0);
  }
  if (mode > 5.5) {
    // Climate: red is warm, blue is wet. The view that shows whether the
    // planet has provinces at all, which is the thing the orbital view lives
    // or dies on.
    return vec3<f32>(temp, 0.25 + 0.35 * moist, moist);
  }
  if (mode > 3.5 && mode < 4.5) {
    // Canopy cover, the field the scatter and the ground tint share.
    return vec3<f32>(cover, cover * 0.85, cover * 0.35);
  }
  // Every branch is an exclusive range. An open-ended mode > k ladder means
  // each new debug view is silently swallowed by an earlier one.
  if (mode > 2.5 && mode < 3.5) {
    return mix(n2 * 0.5 + 0.5, gridCol, overlay * 0.85);
  }
  if (mode > 1.5 && mode < 2.5) {
    return mix(mix(vec3<f32>(0.10, 0.30, 0.16), vec3<f32>(0.92, 0.35, 0.20), slope),
               gridCol, overlay * 0.85);
  }
  if (mode > 0.5 && mode < 1.5) {
    let t = lvl.x / 19.0;
    let c = vec3<f32>(
      0.5 + 0.5 * cos(6.2831853 * (t * 3.0 + 0.00)),
      0.5 + 0.5 * cos(6.2831853 * (t * 3.0 + 0.33)),
      0.5 + 0.5 * cos(6.2831853 * (t * 3.0 + 0.67)));
    return mix(c, gridCol, overlay * 0.85);
  }

  // ── albedo ────────────────────────────────────────────────────────────
  // Reflectances, not screen colours: everything below is multiplied by
  // incoming light, so these are the values a spectrophotometer would read.
  // A Whittaker set rather than a ramp. Nine classes is the point where the
  // planet stops reading as "green with brown bits" from orbit: the eye picks
  // out the Sahara, the Amazon, the taiga and the steppe as *places*, and it
  // cannot do that from three colours.
  // The biome colours moved to biome.ts, beside the windows that select them,
  // and arrive as bc_<key>_T. What is left here are the *overlays* — surfaces
  // that sit on top of whatever biome is underneath rather than being one.
  let ice       = bc_ice_T;
  let temperate = bc_broadleaf_T;
  let savanna   = bc_savanna_T;
  let desert    = bc_desert_T;
  let badland   = vec3<f32>(0.30, 0.205, 0.135);
  let beach     = vec3<f32>(0.42, 0.38, 0.29);
  let rock      = vec3<f32>(0.17, 0.155, 0.145);
  let scree     = vec3<f32>(0.22, 0.20, 0.19);

  // Variation must come from something smooth. A per-pixel hash of the surface
  // direction was chaotic in screen space and flickered whenever the camera
  // moved — an aliasing source with no filterable band.
  let hv = clamp(hgt / 2200.0, 0.0, 1.0) * 0.65 + slope * 0.35;

  // ── ground texture ────────────────────────────────────────────────────
  //
  // Until now the only variation in the albedo was at 46 and 155 km — the
  // biome dither — so from a metre away the ground was a single flat colour
  // over the whole field of view. That is the difference between standing on a
  // planet and standing on a painted sphere, and no amount of biome work fixes
  // it because biomes are the wrong scale entirely.
  //
  // Band-limited to the pixel, on exactly the discipline the detail normal
  // uses: the finest octave sits at 8 px, above Nyquist, so it reads as texture
  // rather than as the crawling grain that came of putting it at 2.5. It also
  // costs nothing at range, because past a few hundred metres it simply stops
  // resolving and the biome colour is all that is left — which is correct.
  // ── this band cannot come from a global function of a unit direction ──
  //
  // It used to. gLam was floored at 0.30 m, which put the frequency at 21.2 M
  // against a MAX_NOISE_FREQ of 336 k — sixty-three times past it, where one
  // f32 ULP spans two whole lattice cells. What reached the screen was
  // therefore the lattice and not a field: hard-edged quads carpeting the near
  // ground, several metres across, in the *albedo only*. That last part is why
  // it survived so long — the normal was clean, the slope was clean and
  // turning shadows off changed nothing, so every geometric explanation was
  // wrong and the debug modes all looked fine except mode 5.
  //
  // The sibling ladder further down had already been floored at
  // MIN_NOISE_LAMBDA for precisely this reason. This one was missed, and the
  // local-lattice escape hatch that makes metre-scale detail reachable at all
  // was sitting in the same function (see localNoiseBlock and LOCAL_PERIOD).
  // So it is used here too, on the same fixed power-of-two rungs faded by
  // pixel size — fixed rather than pixel-scaled because the wrap is only
  // seamless when the period is a whole number of cells, and because a rung
  // fading in beats a wavelength sliding around.
  let pLocal = rel + snap;
  var lg = 0.0;
  var lgAmp = 1.0;
  var lgNorm = 0.0;
  var lgCells = 16384;
  for (var i = 0; i < 3; i = i + 1) {
    let lam = ${f(LOCAL_PERIOD)} / f32(lgCells);
    let lw = 1.0 - smoothstep(lam * 0.5, lam * 2.0, mPerPx);
    if (lw > 0.004) {
      lg = lg + lw * lgAmp
         * noiseL_T(pLocal / lam + vec3<f32>(f32(i * 37)), lgCells - 1);
    }
    lgNorm = lgNorm + lgAmp;
    lgAmp = lgAmp * 0.55;
    lgCells = lgCells / 4;
  }
  // Soil, gravel and litter live between about 0.1 and 3 m, and the rungs
  // above already stop resolving past that, so the ladder carries its own
  // distance fade and does not need a second one.
  let grain = lg / lgNorm;

  // ── the middle scale ──────────────────────────────────────────────────
  //
  // With the grain faded at 7 m and the biome dither living at 46 km, the
  // albedo had nothing at all between them — and that gap is precisely the
  // range you see from a hillside. A kilometre of ground came out as one flat
  // colour, which is why the mid distance read as painted however good the
  // silhouette was.
  //
  // Two fixed wavelengths, 140 m and 430 m: the scale of a stand of trees, a
  // patch of scrub, a change of soil. Fixed rather than pixel-derived because
  // these are real features of the ground and should stay put as you approach;
  // faded only once a pixel is wider than the feature, which is the ordinary
  // anti-aliasing condition and does not bite until ~30 km.
  let mFade = 1.0 - smoothstep(24.0, 90.0, mPerPx);
  let mA = noised_T(up * (${f(RADIUS)} / 140.0) + vec3<f32>(71.3)).x;
  let mB = noised_T(up * (${f(RADIUS)} / 430.0) + vec3<f32>(17.9)).x;
  let meso = (mA * 0.45 + mB * 0.55) * mFade;

  // Loose material collects in hollows and is stripped off convexities, so the
  // same field that mottles the colour also says where soil should be. Sign is
  // the whole content: down is where things gather.
  let hollow = clamp(-grain * 1.6, 0.0, 1.0);
  let clast = clamp(grain * 1.6, 0.0, 1.0);

  // Slope, perturbed by the grain. Rock does not break out along a contour of
  // constant steepness — it appears in patches where the soil has failed — and
  // an unperturbed threshold traces the curvature of every bump in the
  // amplification, which is how you end up seeing the noise's topology instead
  // of a landscape.
  let slopeB = clamp(slope + grain * 0.06, 0.0, 1.0);

  if (waterMix > 0.0) {
    // Water is a mirror with a little colour underneath, not a blue surface.
    // Splitting reflected from transmitted by Fresnel is the whole thing: the
    // previous version multiplied sun irradiance by Fresnel directly, so at
    // grazing angles — i.e. most of any ocean view — it blew out to white.
    let v = normalize(-rel);
    let f0 = 0.02;
    let fres = f0 + (1.0 - f0) * pow(1.0 - clamp(dot(v, up), 0.0, 1.0), 5.0);

    // Actually sample the sky in the mirror direction. A constant blue is the
    // reason CG oceans read as plastic: real water is bright at the horizon
    // and dark overhead purely because of what it reflects.
    let refl = reflect(-v, up);
    var skyRefl = skyRadiance_T(wp, refl, sd, Rg, sunCol);

    // At range the mirror has to give way to the average sky.
    //
    // Water is only a mirror at scales where its waves resolve. A lake four
    // kilometres off subtends far less than the wave field that roughens it,
    // so a point sample of the sky in the reflected direction is the wrong
    // answer — and at grazing incidence that direction is the bright horizon,
    // returned at Fresnel ~1 straight down the barrel. The result was a
    // scattering of white chips on distant hillsides, texel-shaped because the
    // lakes are, and bright enough to read as a rendering fault rather than as
    // water.
    //
    // Converging to the overhead sky with distance is both the cheap fix and
    // the correct one: it is what an unresolvable rough mirror actually
    // returns.
    let dRough = smoothstep(400.0, 5000.0, length(rel));
    skyRefl = mix(skyRefl, skyRadiance_T(wp, up, sd, Rg, sunCol), dRough * 0.85);

    // Sun glint as a GGX lobe on a slightly rough surface, not a fixed power.
    let hv = normalize(v + sd);
    let rough = 0.062;
    let a2 = rough * rough * rough * rough;
    let nh = max(dot(up, hv), 0.0);
    let dd = nh * nh * (a2 - 1.0) + 1.0;
    let ggx = a2 / (3.14159265 * dd * dd);

    let sunTrW = sunLight_T(wp, sd, Rg);
    let sunUpW = max(dot(up, sd), 0.0);

    // Only the transmitted fraction reaches the water body. Shallow water over
    // a lit bed is green; deep water is nearly black.
    // Real depth now, not the seabed elevation standing in for it — so a
    // river a metre deep over a lit bed reads as a river, and 4 km of ocean
    // reads as ocean, from the same expression.
    let dt = clamp(depth / 45.0, 0.0, 1.0);
    let body = mix(vec3<f32>(0.075, 0.175, 0.165), vec3<f32>(0.004, 0.019, 0.042), dt);
    // Gated with the sky it comes from — see the dusk note in the ground
    // ambient below. Water is the term that mattered most here: its albedo is
    // low but it reflects a whole hemisphere of sky, so an ungated floor made
    // every ocean on the night side glow.
    let skyAmb = sunCol * vec3<f32>(0.055, 0.085, 0.155)
               * (${f(NIGHT_SKY)} + 0.045 * sunUp_T(wp, sd, Rg) + 0.6 * sunUpW);
    let sub = body * (sunCol * sunTrW * sunUpW * (1.0 / 3.14159265) * shadow * cloudLit + skyAmb);

    // Fresnel is capped with the same distance term. Grazing incidence sends it
    // to 1, which is right for a mirror and wrong for something whose mirror
    // character has just been averaged away.
    let fresD = mix(fres, min(fres, 0.34), dRough);
    var wc = mix(sub, skyRefl, fresD)
           + sunCol * sunTrW * ggx * fresD * sunUpW * shadow * cloudLit;

    // Surf line where the sea meets land. Crude — a depth band, no waves — but
    // a coastline with no tonal change at all reads as a paint boundary.
    let surf = (1.0 - smoothstep(0.0, 6.0, depth)) * smoothstep(0.0, 1.5, depth);
    wc = mix(wc, wc + sunCol * sunTrW * 0.035 * shadow * cloudLit, surf);

    // Aerial perspective is applied once, at the end, to the blended colour.
    // aerial_ is "colour * transmittance + inscatter" — affine in the
    // colour, and both sides pass it the same path — so mixing before it and
    // mixing after it are the same number, and this way costs one call.
    waterCol = wc;
    // Fully submerged: nothing on the land side can contribute, so skip it.
    // This is what keeps the two 12-step sky marches above off every land
    // pixel in the frame.
    if (waterMix >= 1.0) {
      let wOut = aerial_T(waterCol, camPos, wp, sd, Rg, sunCol);
      return mix(wOut, gridCol, overlay * 0.85);
    }
  }

  // Ground colour along the moisture axis, then the cold end overrides it —
  // which is the Whittaker diagram, and the reason this is two mixes and not
  // one ramp. Elevation appears only through temp, so the same colour
  // sequence happens with height in the tropics and with latitude at sea
  // level, exactly as it does on Earth. The old ramp keyed everything on raw
  // elevation, so the 700–1700 m band — which is where most land sits — swung
  // green to tan across every continent and produced the mottling that read
  // as noise from 40 km.
  // ── the moisture axis, walked from arid to wet ────────────────────────
  //
  // Every one of these transitions is a smoothstep on the moisture value, and
  // that value is smooth, so every biome boundary was a clean band — forest
  // meeting desert along a contour, which is the one thing no satellite image
  // of Earth shows. The Sahel is hundreds of kilometres of *mixture*: thinning
  // scrub, patches of green in the wadis, tongues of sand reaching south.
  //
  // Perturbing the moisture axis with a two-octave field breaks the contour
  // into that mixture for the cost of the noise. It is applied to the axis
  // rather than to the colour so the *sequence* is preserved — a perturbed
  // point takes the colour of ground slightly drier or wetter than it is,
  // which is exactly what patchiness is, and never a colour from somewhere
  // else in the diagram.
  // As an octave ladder rather than two fixed scales.
  //
  // The two were at 155 km and 46 km, so at 1000 km — where a pixel is 1.1 km
  // and the bake's own texel is 9 km — there was nothing whatever between the
  // texel and 46 km. A continent read as one flat wash with a few large
  // blotches on it, which is the "low resolution" of the orbital view: not
  // missing geometry, missing *variegation*. Ground at 1 km/px is dense with
  // structure at every scale down to the pixel.
  //
  // Each octave fades out once its wavelength drops below eight pixels, the
  // same Nyquist rule the detail normal and the grain already use (LESSONS §4),
  // so the ladder lengthens as you descend and no octave is ever drawn below
  // the resolution that can carry it. The normaliser is unweighted for the
  // reason the terrain's is: dropping an octave has to low-pass the field, not
  // rescale what is left, or the biome boundaries would crawl as you approach.
  var bmSum = 0.0;
  var bmAmp = 1.0;
  var bmNorm = 0.0;
  var bmLam = ${f(RADIUS / 41)};
  for (var i = 0; i < 6; i = i + 1) {
    let w = 1.0 - smoothstep(bmLam * 0.125, bmLam * 0.34, mPerPx);
    if (w > 0.003) {
      bmSum = bmSum + w * bmAmp
            * noised_T(up * (${f(RADIUS)} / bmLam) + vec3<f32>(f32(i) * 9.17 + 3.7)).x;
    }
    bmNorm = bmNorm + bmAmp;
    bmAmp = bmAmp * 0.62;
    bmLam = bmLam * 0.34;
  }
  let moistB = clamp(moistA + (bmSum / bmNorm) * 0.105, 0.0, 1.0);
  //
  // Each window overlaps its neighbour so no boundary is a line. The forest
  // windows deliberately match canopyClosure_'s moisture gate: where it is wet
  // enough for trees the ground between them is leaf litter and meadow, not
  // bare steppe, and any mismatch there becomes a two-tone mottle at exactly
  // the scale you see from 20–100 km.
  // ── seasonality, the third climate axis ───────────────────────────────
  //
  // How pronounced the dry season is. Rainforest, tropical dry forest and
  // savanna can share an annual temperature and a similar annual rainfall;
  // what separates them is whether the rain arrives all year or in one season,
  // and for Mediterranean scrub that distribution *is* the biome. See
  // biome.ts. One noise octave breaks the belts out of perfect zonal stripes.
  let swing = noised_T(up * ${f(SEASON_F0)} + vec3<f32>(61.7)).x;
  let season = season_T(abs(up.y), swing);

  // Ten biomes plus permanent ice, as a weighted blend rather than a choice —
  // every boundary on Earth is hundreds of kilometres of mixture, and a hard
  // classification of a smooth field aliases (LESSONS §13). clim.x is the
  // sea-level temperature, which montane grassland needs in order to tell a
  // subtropical plateau from Arctic coast; both lapse to nearly zero.
  //
  // Fitted to Earth's land-area shares by npm run biomes: 10.0 points of total
  // absolute error across eleven classes, none more than 1.3 out.
  let bio = biome_T(temp, clim.x, moistB, season, hgt);
  var alb = bio.rgb;
  // How much loose material this ground carries — see the grain field in
  // biome.ts. Desert pavement is gravel over sand and reads at full contrast
  // from a metre away; a rainforest floor is uniform litter under closed
  // canopy and has almost none. One scalar, blended by the same weights as
  // the colour, so the two cannot describe different ground.
  let grainK = bio.w;

  // Badlands: hot, dry and steep. Bare sedimentary rock stripped of soil, and
  // the one biome that is a *slope* class rather than a climate class.
  alb = mix(alb, badland,
            smoothstep(0.16, 0.40, slopeB) * (1.0 - smoothstep(0.10, 0.30, moist))
            * smoothstep(0.35, 0.60, temp));

  // A beach is tens of metres of shore, not the first ninety metres of
  // altitude. The old window painted a bright band tens of kilometres deep
  // wherever the coastal shelf was gentle.
  //
  // Faded out once a pixel is wider than a beach is. The band is a window on
  // *elevation*, so on a shelf sloping at 1:1000 those 22 m are 22 km of
  // ground — and from 1000 km up that painted a hard bright outline around
  // every continent, tracing the bake's own contour and making the coast read
  // as a cut-out. A real beach is tens of metres wide and is sub-pixel from
  // orbit, so the honest thing for it to do up there is disappear.
  let beachFade = 1.0 - smoothstep(60.0, 400.0, mPerPx);
  alb = mix(alb, mix(beach, alb, smoothstep(2.0, 22.0, hgt)), beachFade);

  // Steep ground sheds soil at any altitude — the strongest single cue that a
  // slope is a slope. Widened from [0.30, 0.62]: against the amplification's
  // ridged octaves that band caught the steepest ring of every bump, and in
  // dry country, where bare rock is 2.4x darker than the ground around it,
  // that outlined every ridge and made the noise's topology the most visible
  // thing in the frame. Bare rock also bleaches where nothing grows, so it
  // reads closer to the ground it sits in.
  let bareRock = mix(rock, mix(rock, desert, 0.45), 1.0 - smoothstep(0.22, 0.52, moist));
  alb = mix(alb, bareRock, smoothstep(0.34, 0.72, slopeB));
  alb = mix(alb, scree,
            smoothstep(0.22, 0.45, slopeB) * (1.0 - smoothstep(0.62, 0.8, slopeB)) * hv);

  // Snow lies where it is cold, flat, and blows off ridges. Driven by
  // temperature rather than elevation, so the snow line descends with latitude
  // and the poles carry ice at sea level — which is most of what makes a
  // planet look like a planet from orbit.
  // Broken up, and it has to be. Snow is a hard threshold on temperature, and
  // temperature is a smooth function of a field stored at 9 km — so an
  // unperturbed snow line paints whole texels white and leaves the rectangles
  // that were showing on every mid-distance ridge. Real snow at the line is
  // patchy: it survives in hollows and on shaded aspects and goes first off
  // the sunlit convexities, which is what the aspect and hollow terms below
  // already describe.
  //
  // Two fixed wavelengths at the scale snow patches actually are, ~700 m and
  // ~2.4 km, plus the aspect: a pole-facing slope keeps its snow hundreds of
  // metres lower than the one opposite.
  let snA = noised_T(up * (${f(RADIUS)} / 700.0) + vec3<f32>(43.9)).x;
  let snB = noised_T(up * (${f(RADIUS)} / 2400.0) + vec3<f32>(88.1)).x;
  let snowT = temp
            + (snA * 0.45 + snB * 0.55) * 0.020
            + aspect * 0.014
            - hollow * 0.010;
  let snowLine = (1.0 - smoothstep(0.015, 0.14, snowT))
               * (1.0 - smoothstep(0.30, 0.60, slopeB));
  alb = mix(alb, ice, snowLine);

  // ── apply the ground texture ──────────────────────────────────────────
  //
  // Three separate things, because ground is not one material:
  //
  //   tone     a gentle multiplicative mottle. Multiplicative so it survives
  //            whatever biome is underneath instead of tinting everything
  //            toward one colour.
  //   hollows  darker and slightly cooler: damp, organic, shaded by their own
  //            rim. This is the cheapest thing that reads as *depth* in a
  //            surface, and it is why bare ground stops looking like a decal.
  //   clasts   loose stone on the raised side, strongest where it is steep
  //            enough to have shed its soil and dry enough not to have grown
  //            anything over it.
  // ── metre-scale material ──────────────────────────────────────────────
  //
  // Below the finest octave the mesh carries, ground is not a smooth tint: it
  // is gravel, litter, crust and bare patches, and that variegation is most of
  // what separates a photograph from a shaded heightfield at arm's length. The
  // existing grain octave is one wavelength; this is a short ladder under it,
  // pinned to eight pixels at the fine end so it is texture rather than noise
  // (LESSONS §4), and scaled by the biome's own grain so a desert is coarse
  // and a rainforest floor is not.
  //
  // Multiplicative, so it survives whatever biome is underneath instead of
  // pulling everything toward one colour, and paired with a small hue shift —
  // loose stone is greyer and slightly cooler than the soil it sits on.
  var fg = 0.0;
  var fgAmp = 1.0;
  var fgNorm = 0.0;
  // Floored by f32, not by ambition. This read 0.35 m, which is 54x past
  // MAX_NOISE_FREQ — under one step of fractional position per lattice cell,
  // so the "detail" was the hash aliasing against itself rather than a field.
  // Genuine sub-metre material has to come from per-tile data with a local
  // origin (M5); a global function of a unit direction cannot reach it.
  var fgLam = max(mPerPx * 8.0, ${f(MIN_NOISE_LAMBDA)});
  for (var i = 0; i < 3; i = i + 1) {
    fg = fg + fgAmp * noised_T(up * (${f(RADIUS)} / fgLam) + vec3<f32>(f32(i) * 4.13 + 51.0)).x;
    fgNorm = fgNorm + fgAmp;
    fgAmp = fgAmp * 0.55;
    fgLam = fgLam * 3.1;
  }

  // ── and the metre scale, off the local lattice ────────────────────────
  //
  // The ladder above bottoms out at MIN_NOISE_LAMBDA, 76 m, because it is a
  // global function of a unit direction and f32 has nothing below that. These
  // three rungs are 4 m, 1 m and 25 cm, and they are reachable because they
  // are evaluated in *metres about a snapped origin* rather than in units of a
  // planet radius — see LOCAL_PERIOD and localNoiseBlock. Measured: the global
  // form gets 1 lattice step per cell at 25 cm, this one gets 1866.
  //
  // This is the material a slope is made of rather than the shape of it, and
  // its absence is why close ground read as tinted plastic. Each rung is the
  // period over a power of two, so the cell mask is exact and the origin snap
  // leaves no seam; each fades once it falls under about two pixels, so the
  // ladder shortens with distance instead of stippling the horizon.
  // The same local ladder the grain is built from, hoisted above — one
  // evaluation, two consumers, rather than two ladders over the same band.
  fg = fg + grain * 1.4;
  fgNorm = fgNorm + 1.0;
  // Faded where a pixel is wider than the coarsest rung, so it recedes with
  // distance instead of stippling the whole landscape.
  let fgFade = (1.0 - smoothstep(40.0, 260.0, mPerPx)) * grainK;
  let fine = (fg / fgNorm) * fgFade;
  alb = alb * (1.0 + fine * 0.30);
  alb = mix(alb, alb * vec3<f32>(0.82, 0.84, 0.88), clamp(fine, 0.0, 1.0) * 0.34);

  // No second multiplicative pass on the grain: it rides inside fine now.
  // The hollow tint stays, because a damp shaded hollow is a hue shift and not
  // the same effect as the tone mottle.
  alb = mix(alb, alb * vec3<f32>(0.74, 0.78, 0.80), hollow * 0.30);

  // The middle scale, applied as a shift along the *moisture* axis rather than
  // as a tint — the same argument as the biome dither. A patch of ground that
  // is greener is greener because it holds more water, so it should take the
  // colour of wetter ground and not merely a green cast. Gentle ground holds
  // the patchiness; steep ground has shed it along with its soil.
  let mesoWet = clamp(moistB + meso * 0.10 * (1.0 - smoothstep(0.20, 0.50, slopeB)),
                      0.0, 1.0);
  alb = mix(alb, mix(alb, savanna, 0.55), clamp(-meso, 0.0, 1.0) * 0.22
                                          * (1.0 - smoothstep(0.34, 0.58, mesoWet)));
  alb = mix(alb, mix(alb, temperate, 0.55), clamp(meso, 0.0, 1.0) * 0.26
                                            * smoothstep(0.20, 0.46, mesoWet));
  alb = alb * (1.0 + meso * 0.07);
  alb = mix(alb, scree, clast * smoothstep(0.10, 0.34, slopeB)
                        * (1.0 - smoothstep(0.30, 0.60, moist)) * 0.40);

  // Canopy. Same cover field the scatter uses, so ground colour and where
  // plants actually stand cannot disagree, and instances can dissolve into
  // this without revealing an edge.
  let canopy = mix(vec3<f32>(0.036, 0.055, 0.026), vec3<f32>(0.055, 0.080, 0.032), hv);
  alb = mix(alb, canopy, cover * (1.0 - snowLine) * 0.92);

  if (mode > 4.5 && mode < 5.5) {
    // Raw albedo, unlit — separates "the surface is the wrong colour" from
    // "the surface is lit wrongly", which look identical in the final image.
    return alb;
  }

  // ── lighting ──────────────────────────────────────────────────────────
  let ndl = max(dot(n2, sd), 0.0);
  let sunTr = sunLight_T(wp, sd, Rg);
  // Soft terminator: unresolved relief keeps scattering light just past the
  // geometric horizon, and a hard cut there reads as a CG edge.
  let soft = smoothstep(-0.10, 0.12, dot(n2, sd));
  // Shadow attenuates only the direct term. Sky light still reaches a
  // shadowed surface, which is why real shadows are blue rather than black.
  let direct = alb * (1.0 / 3.14159265) * sunCol * sunTr * ndl * soft * shadow * cloudLit;

  let sunUp = max(dot(up, sd), 0.0);
  // The 0.045 is skylight from air the sun has *set* on — the blue that keeps
  // a shaded wall visible after the sun leaves it — so it has to end when the
  // sun does. Held constant it lit the whole night hemisphere, and that was
  // the last of what made the dark side read as day once the exposure and the
  // cloud deck were fixed. sunUp_T is the same twilight geometry the
  // atmosphere uses, so the ground and the air it sits under go out together.
  let dusk = sunUp_T(wp, sd, Rg);
  let sky = sunCol * vec3<f32>(0.055, 0.085, 0.155)
          * (${f(NIGHT_SKY)} + 0.045 * dusk + 0.6 * sunUp);
  let ambient = alb * sky * (0.5 + 0.5 * dot(n2, up)) * skyView.x;

  // One crude bounce off the surrounding lit ground, deliberately *not*
  // Scaled by sky view — see the AO note in terrainMesh. A valley floor sees a
  // fraction of the hemisphere and a summit sees all of it, and applying that
  // to the *ambient* rather than to the albedo is what makes it read as depth
  // instead of as dirt.
  //
  // shadowed. Without it a cast shadow goes almost black, which is the single
  // clearest tell that a renderer has direct light and nothing else — real
  // shadows are filled by everything around them.
  //
  // Written as a fraction of the direct term, not as its own light. The
  // previous form omitted the 1/pi and so returned pi times more than a
  // Lambertian surface can reflect — for snow it was 2.5x the direct term,
  // which is why every snowfield clipped to flat white however the exposure
  // was set. Physically this is the multiple-scatter enhancement a/(1-a)
  // truncated to its first useful order: about +25% for snow, +5% for forest.
  let bounce = alb * alb * (1.0 / 3.14159265) * sunCol * sunTr * sunUp * 0.42;

  var col = direct + ambient + bounce;
  // The shoreline, resolved rather than decided. waterMix is this pixel's
  // water coverage, so a pixel the waterline runs through gets the two
  // shadings in the proportion it actually contains.
  col = mix(col, waterCol, waterMix);
  col = aerial_T(col, camPos, wp, sd, Rg, sunCol);
  return mix(col, mix(gridCol * 0.5, gridCol, waterMix), overlay * 0.85);
}
${noiseBlock('T')}
${atmosphere('T')}
${lapseBlock('T')}
${closureBlock('T')}
${localNoiseBlock('T')}
${biomeBlock('T')}
${cloudFieldBlock('T')}
`);
