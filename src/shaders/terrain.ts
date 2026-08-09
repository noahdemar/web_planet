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
import {
  RELIEF_GAIN,
  RELIEF_LACUNARITY,
  VEG_MAX_SLOPE,
  VEG_MIN_ELEVATION,
  VEG_TREELINE,
} from '../planet.js';
import { atmosphere } from './atmosphere.js';

/**
 * Octahedral normal packing. Buys the fourth channel of the surface varying
 * for canopy cover: a unit vector only needs two numbers.
 */
export function octPack(s: string): string {
  return /* wgsl */ `
fn octWrap_${s}(v: vec2<f32>) -> vec2<f32> {
  return (1.0 - abs(v.yx)) * select(vec2<f32>(-1.0), vec2<f32>(1.0), v >= vec2<f32>(0.0));
}
fn octEncode_${s}(n0: vec3<f32>) -> vec2<f32> {
  let n = n0 / (abs(n0.x) + abs(n0.y) + abs(n0.z));
  return select(octWrap_${s}(n.xy), n.xy, n.z >= 0.0);
}
fn octDecode_${s}(f: vec2<f32>) -> vec3<f32> {
  var n = vec3<f32>(f.x, f.y, 1.0 - abs(f.x) - abs(f.y));
  let t = max(-n.z, 0.0);
  n = vec3<f32>(n.xy + select(vec2<f32>(t), vec2<f32>(-t), n.xy >= vec2<f32>(0.0)), n.z);
  return normalize(n);
}
`;
}

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

/** Noise, its analytic derivative, and the elevation field. Expensive. */
/**
 * Frequency at which amplification starts, in cycles per unit direction —
 * wavelength is RADIUS/f. The bake resolves 18 km cells, so its Nyquist is
 * f = 177 and the band from there to 354 is already attenuated by the resample.
 * Starting a little below 354 puts detail back into that attenuated band
 * rather than double-counting it.
 */
const AMP_F0 = 260;

/**
 * Mean of (1 − |noise|)² for this gradient noise, measured by
 * tools/ridgeMean.ts. Subtracted per octave so switching octaves on and off
 * with distance cannot move the coastline.
 */
const RIDGE_MEAN = 0.7452;

/**
 * Amplification amplitude, metres: a floor everywhere plus a term scaled by
 * the relief the bake already resolves.
 *
 * These are large because the ridged sum they multiply is small. Summing 17
 * octaves at gain 0.62 and dividing by the unweighted normaliser leaves a
 * signal of roughly ±0.1, so an amplitude of 1450 produced about ±145 m of
 * sub-grid relief in the steepest terrain and 2 m as a planet-wide average.
 * Standing on it, that is a flat green plain to the horizon. Real sub-18 km
 * relief in a mountain belt is well over a kilometre.
 */
const AMP_BASE = 70;
const AMP_RELIEF = 7800;

export function field(s: string): string {
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
 * every 281 m while octave 17 has a 19 m wavelength, so it was 29x
 * undersampled and every vertex was landing on an essentially random phase of
 * it. Morphing then slid vertices through that field and the surface boiled.
 *
 * The weights are *not* renormalised. Dropping an octave must lower-pass the
 * surface, not rescale what is left, or the terrain would change shape as you
 * approached it (SPEC.md I2).
 */
fn height_${s}(dir: vec3<f32>, oct: i32, hscale: f32,
               bakeH: f32, bakeG: vec3<f32>, wet: f32,
               radius: f32, bandLimit: f32) -> vec4<f32> {
  // dh/ds where s is arc length: the gradient is per unit direction, and a
  // unit of direction is one planet radius of surface.
  let slope = length(bakeG) / radius;

  let relief = smoothstep(0.006, 0.085, slope);
  let landW = smoothstep(-350.0, 40.0, bakeH);
  // Trunk valleys: 10^10 m² is a continental river, 10^7.5 a headwater.
  let valley = smoothstep(7.5, 10.5, wet);
  let amp = (${AMP_BASE}.0 + ${AMP_RELIEF}.0 * relief)
          * mix(0.22, 1.0, landW)
          * mix(1.0, 0.28, valley);

  var mAmp = 1.0; var mFrq = ${AMP_F0}.0; var mSum = 0.0;
  var mG = vec3<f32>(0.0); var mNorm = 0.0; var mBias = 0.0;
  for (var i = 0; i < oct; i = i + 1) {
    // Full weight while the wavelength spans 2.5 samples, zero by 1.
    let w = smoothstep(1.0, 2.5, bandLimit / mFrq);
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
      mBias = mBias + w * mAmp * ${RIDGE_MEAN};
    }
    // Unweighted: the normaliser must not change with the band limit.
    mNorm = mNorm + mAmp;
    mAmp = mAmp * ${RELIEF_GAIN};
    mFrq = mFrq * ${RELIEF_LACUNARITY};
  }
  let detail = (mSum - mBias) / mNorm;
  let detailG = mG / mNorm;

  let h = (bakeH + detail * amp) * hscale;
  // The amplitude field varies over the bake's cell size, three orders of
  // magnitude coarser than the detail it scales, so d(amp)/d(dir) is
  // negligible next to amp·d(detail)/d(dir) and is dropped.
  let g = (bakeG + detailG * amp) * hscale;

  return vec4<f32>(h, g);
}

/**
 * Fraction of ground under canopy, in [0,1].
 *
 * Shared verbatim by the scatter and the terrain shading. That is the whole
 * point: the ground tint and where plants actually stand are the same
 * function, so they cannot disagree, and instances can dissolve into the tint
 * without revealing an edge.
 */
fn forestClump_${s}(dir: vec3<f32>, bandLimit: f32) -> f32 {
  // Three octaves at 7 km / 3 km / 1.3 km — glades, stand edges, dense pockets.
  // Without this the forest is a uniform mat, which reads as fake from the air
  // long before any individual tree does.
  var amp = 1.0;
  var frq = 900.0;
  var sum = 0.0;
  var norm = 0.0;
  for (var i = 0; i < 3; i = i + 1) {
    let w = smoothstep(1.0, 2.5, bandLimit / frq);
    if (w > 0.002) {
      sum = sum + w * amp * noised_${s}(dir * frq + vec3<f32>(f32(i) * 11.37)).x;
    }
    norm = norm + amp;
    amp = amp * 0.55;
    frq = frq * 2.3;
  }
  return smoothstep(-0.42, 0.26, sum / norm);
}

fn forestCover_${s}(dir: vec3<f32>, h: f32, slope: f32, density: f32,
                    bandLimit: f32) -> f32 {
  let clump = forestClump_${s}(dir, bandLimit);
  let lo = smoothstep(${VEG_MIN_ELEVATION}.0, ${VEG_MIN_ELEVATION}.0 + 30.0, h);
  let hi = 1.0 - smoothstep(${VEG_TREELINE}.0 - 350.0, ${VEG_TREELINE}.0, h);
  let sl = 1.0 - smoothstep(0.30, ${VEG_MAX_SLOPE}, slope);
  return clamp(clump * lo * hi * sl * density, 0.0, 1.0);
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

export const patchSurface = wgslFn(/* wgsl */ `
fn patchSurface(${ARGS}, baked: vec4<f32>, bake2: vec4<f32>) -> vec4<f32> {
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

  let hn = height_N(dir, i32(cfg.z), cfg.y, baked.x, baked.yzw, bake2.x,
                    radius, bandLimit);

  // Surface normal from the tangential component of the height gradient.
  let gT = hn.yzw - dir * dot(dir, hn.yzw);
  let nrm = normalize(dir - gT / (radius + hn.x));
  let slope = 1.0 - dot(nrm, dir);

  // Cover is evaluated per vertex, not per pixel: three extra noise octaves
  // over ~700k vertices instead of ~3.7M pixels, and the field is smooth
  // enough at 1.3 km that interpolation costs nothing visible.
  let cover = forestCover_N(dir, hn.x, slope, cfg3.x, bandLimit);
  return vec4<f32>(octEncode_N(nrm), hn.x, cover);
}
${octPack('N')}
${geom('N')}
${field('N')}
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
 * `surf` is (octNormal.xy, elevation, canopyCover) — the normal is octahedral
 * so the fourth channel can carry cover.
 *
 * Roughly physical: albedo is lit by sunlight that has already been attenuated
 * by the air it travelled through, plus a hemispheric sky term, and the result
 * then goes through aerial perspective. That last step is what makes distance
 * read correctly and, incidentally, what hides every LOD and vegetation
 * transition beyond a few hundred metres.
 *
 * mode: 0 = natural, 1 = LOD level, 2 = slope, 3 = normals, 4 = canopy cover.
 * Debug modes skip the atmosphere so they stay legible.
 */
export const shadeTerrain = wgslFn(/* wgsl */ `
fn shadeTerrain(surf: vec4<f32>, camPos: vec3<f32>, rel: vec3<f32>,
                lvl: f32, sunDir: vec3<f32>, sunCol: vec3<f32>,
                mode: f32, grid: f32, cfg3: vec4<f32>, shadow: f32) -> vec3<f32> {
  let n = octDecode_T(surf.xy);
  let hgt = surf.z;
  let cover = surf.w;
  let Rg = cfg3.y;

  // Planet-centred surface point. Local up per pixel, not per patch — a
  // per-instance value steps visibly at every patch boundary.
  let wp = camPos + rel;
  let up = normalize(wp);
  let slope = clamp(1.0 - dot(n, up), 0.0, 1.0);
  let sd = normalize(sunDir);

  // Metric reference grid: the instrument for M1's jitter check.
  var overlay = 0.0;
  if (grid > 0.0) {
    let q = rel / grid;
    let w = fwidth(q);
    let f = abs(fract(q - vec3<f32>(0.5)) - vec3<f32>(0.5)) / max(w, vec3<f32>(1e-8));
    overlay = 1.0 - min(min(f.x, min(f.y, f.z)), 1.0);
  }
  let gridCol = vec3<f32>(1.0, 0.9, 0.3);

  if (mode > 3.5 && mode < 4.5) {
    // Canopy cover, the field the scatter and the ground tint share.
    return vec3<f32>(cover, cover * 0.85, cover * 0.35);
  }
  // Every branch is an exclusive range. An open-ended mode > k ladder means
  // each new debug view is silently swallowed by an earlier one.
  if (mode > 2.5 && mode < 3.5) {
    return mix(n * 0.5 + 0.5, gridCol, overlay * 0.85);
  }
  if (mode > 1.5 && mode < 2.5) {
    return mix(mix(vec3<f32>(0.10, 0.30, 0.16), vec3<f32>(0.92, 0.35, 0.20), slope),
               gridCol, overlay * 0.85);
  }
  if (mode > 0.5 && mode < 1.5) {
    let t = lvl / 19.0;
    let c = vec3<f32>(
      0.5 + 0.5 * cos(6.2831853 * (t * 3.0 + 0.00)),
      0.5 + 0.5 * cos(6.2831853 * (t * 3.0 + 0.33)),
      0.5 + 0.5 * cos(6.2831853 * (t * 3.0 + 0.67)));
    return mix(c, gridCol, overlay * 0.85);
  }

  // ── albedo ────────────────────────────────────────────────────────────
  // Reflectances, not screen colours: everything below is multiplied by
  // incoming light, so these are the values a spectrophotometer would read.
  let sand  = vec3<f32>(0.38, 0.32, 0.23);
  let grass = vec3<f32>(0.11, 0.15, 0.07);
  let dry   = vec3<f32>(0.24, 0.22, 0.12);
  let rock  = vec3<f32>(0.17, 0.155, 0.145);
  let scree = vec3<f32>(0.22, 0.20, 0.19);
  let snow  = vec3<f32>(0.72, 0.74, 0.78);

  // Variation must come from something smooth. A per-pixel hash of the surface
  // direction was chaotic in screen space and flickered whenever the camera
  // moved — an aliasing source with no filterable band.
  let hv = clamp(hgt / 2200.0, 0.0, 1.0) * 0.65 + slope * 0.7;

  if (hgt < 0.0) {
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
    let skyRefl = skyRadiance_T(wp, refl, sd, Rg, sunCol);

    // Sun glint as a GGX lobe on a slightly rough surface, not a fixed power.
    let hv = normalize(v + sd);
    let rough = 0.062;
    let a2 = rough * rough * rough * rough;
    let nh = max(dot(up, hv), 0.0);
    let dd = nh * nh * (a2 - 1.0) + 1.0;
    let ggx = a2 / (3.14159265 * dd * dd);

    let sunTrW = transmit_T(sunDepth_T(wp, sd, Rg));
    let sunUpW = max(dot(up, sd), 0.0);

    // Only the transmitted fraction reaches the water body. Shallow water over
    // a lit bed is green; deep water is nearly black.
    let depth = clamp(-hgt / 45.0, 0.0, 1.0);
    let body = mix(vec3<f32>(0.075, 0.175, 0.165), vec3<f32>(0.004, 0.019, 0.042), depth);
    let skyAmb = sunCol * vec3<f32>(0.055, 0.085, 0.155) * (0.045 + 0.6 * sunUpW);
    let sub = body * (sunCol * sunTrW * sunUpW * (1.0 / 3.14159265) * shadow + skyAmb);

    var wc = mix(sub, skyRefl, fres) + sunCol * sunTrW * ggx * fres * sunUpW * shadow;

    // Surf line where the sea meets land. Crude — a depth band, no waves — but
    // a coastline with no tonal change at all reads as a paint boundary.
    let surf = (1.0 - smoothstep(0.0, 6.0, -hgt)) * smoothstep(-0.2, 1.5, -hgt);
    wc = mix(wc, wc + sunCol * sunTrW * 0.035 * shadow, surf);

    wc = aerial_T(wc, camPos, wp, sd, Rg, sunCol);
    return mix(wc, gridCol, overlay * 0.85);
  }

  var alb = mix(sand, grass, smoothstep(6.0, 90.0, hgt));
  alb = mix(alb, dry, smoothstep(700.0, 1700.0, hgt) * (0.35 + 0.5 * hv));
  alb = mix(alb, rock, smoothstep(1900.0, 3200.0, hgt));
  // Steep ground sheds soil at any altitude — the strongest single cue that a
  // slope is a slope.
  alb = mix(alb, rock, smoothstep(0.30, 0.62, slope));
  alb = mix(alb, scree, smoothstep(0.22, 0.45, slope) * (1.0 - smoothstep(0.62, 0.8, slope)) * hv);
  // Snow lies where it is high, flat and cold, and blows off ridges.
  let snowLine = smoothstep(2500.0, 3600.0, hgt) * (1.0 - smoothstep(0.30, 0.60, slope));
  alb = mix(alb, snow, snowLine);

  // Canopy. Same cover field the scatter uses, so ground colour and where
  // plants actually stand cannot disagree, and instances can dissolve into
  // this without revealing an edge.
  let canopy = mix(vec3<f32>(0.036, 0.055, 0.026), vec3<f32>(0.055, 0.080, 0.032), hv);
  alb = mix(alb, canopy, cover * (1.0 - snowLine) * 0.92);

  if (mode > 4.5) {
    // Raw albedo, unlit — separates "the surface is the wrong colour" from
    // "the surface is lit wrongly", which look identical in the final image.
    return alb;
  }

  // ── lighting ──────────────────────────────────────────────────────────
  let ndl = max(dot(n, sd), 0.0);
  let sunTr = transmit_T(sunDepth_T(wp, sd, Rg));
  // Soft terminator: unresolved relief keeps scattering light just past the
  // geometric horizon, and a hard cut there reads as a CG edge.
  let soft = smoothstep(-0.10, 0.12, dot(n, sd));
  // Shadow attenuates only the direct term. Sky light still reaches a
  // shadowed surface, which is why real shadows are blue rather than black.
  let direct = alb * (1.0 / 3.14159265) * sunCol * sunTr * ndl * soft * shadow;

  let sunUp = max(dot(up, sd), 0.0);
  let sky = sunCol * vec3<f32>(0.055, 0.085, 0.155) * (0.045 + 0.6 * sunUp);
  let ambient = alb * sky * (0.5 + 0.5 * dot(n, up));

  // One crude bounce off the surrounding lit ground, deliberately *not*
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
  col = aerial_T(col, camPos, wp, sd, Rg, sunCol);
  return mix(col, gridCol * 0.5, overlay * 0.85);
}
${octPack('T')}
${atmosphere('T')}
`);
