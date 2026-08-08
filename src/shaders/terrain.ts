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
  OCEAN_DEPTH,
  RELIEF_BASE,
  RELIEF_GAIN,
  RELIEF_LACUNARITY,
  RELIEF_PEAK,
} from '../planet.js';

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
                morph: vec2<f32>, stepG: f32) -> vec2<f32> {
  let d0 = offset_${s}(gpos, A, B, hs, Pc, lenPc, BU, BV);
  // Distance must be measured on the same sphere the CPU selected against:
  // the camera's own ground radius, not sea level. Standing on 2.5 km of
  // terrain, measuring to sea level inflates every distance by 2.5 km, which
  // saturates morphK to 1 at every deep level — each patch then renders as
  // its parent's grid and adjacent levels no longer line up.
  let dist = length(anchorRel + dirCOf_${s}(Pc, lenPc) * (refR - radius) + d0 * refR);
  let mk = clamp((dist - morph.x) / max(morph.y - morph.x, 1e-6), 0.0, 1.0);
  return gpos - gpar.xy * stepG * mk;
}
`;
}

/** Noise, its analytic derivative, and the elevation field. Expensive. */
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

// Elevation in metres, plus its gradient with respect to the unit direction.
// M1 placeholder: continents + ridged relief. Replaced at M5/M6 by tile
// sampling plus context-modulated amplification (SPEC.md §6).
fn height_${s}(dir: vec3<f32>, oct: i32, hscale: f32, sea: f32, band: f32) -> vec4<f32> {
  var cAmp = 1.0; var cFrq = 1.15; var cSum = 0.0;
  var cG = vec3<f32>(0.0); var cNorm = 0.0;
  for (var i = 0; i < 4; i = i + 1) {
    let n = noised_${s}(dir * cFrq + vec3<f32>(f32(i) * 19.37));
    cSum  = cSum + cAmp * n.x;
    cG    = cG + cAmp * cFrq * n.yzw;
    cNorm = cNorm + cAmp;
    cAmp = cAmp * 0.5;
    cFrq = cFrq * 2.03;
  }
  cSum = cSum / cNorm;
  cG = cG / cNorm;

  var mAmp = 1.0; var mFrq = 2.9; var mSum = 0.0;
  var mG = vec3<f32>(0.0); var mNorm = 0.0;
  for (var i = 0; i < oct; i = i + 1) {
    let n = noised_${s}(dir * mFrq + vec3<f32>(f32(i) * 7.77));
    let sg = select(-1.0, 1.0, n.x >= 0.0);
    let r = 1.0 - abs(n.x);
    mSum  = mSum + mAmp * r * r;
    mG    = mG - mAmp * 2.0 * r * sg * mFrq * n.yzw;
    mNorm = mNorm + mAmp;
    mAmp = mAmp * ${RELIEF_GAIN};
    mFrq = mFrq * ${RELIEF_LACUNARITY};
  }
  mSum = mSum / mNorm;
  mG = mG / mNorm;

  // Ridged fBm has a mean near 0.56, so used raw it lifts the whole planet
  // above the snow line. The 7th power skews relief toward lowland, matching
  // Earth's hypsometric curve — calibrated by tools/hypsometry.ts (SPEC.md §13).
  let m2 = mSum * mSum;
  let m3 = m2 * mSum;
  let m6 = m3 * m3;
  let mp = m6 * mSum;
  let mpG = mG * 7.0 * m6;

  let ls = sstepd_${s}(sea - band, sea + band, cSum);
  let land = ls.x;
  let landG = cG * ls.y;

  let h = (cSum * ${RELIEF_BASE}.0
        + mp * land * land * ${RELIEF_PEAK}.0
        + (1.0 - land) * (${OCEAN_DEPTH}.0)) * hscale;

  // product rule through  mp·land²  and  (1-land)·oceanDepth
  let g = (cG * ${RELIEF_BASE}.0
        + (mpG * land * land + mp * 2.0 * land * landG) * ${RELIEF_PEAK}.0
        - landG * (${OCEAN_DEPTH}.0)) * hscale;

  return vec4<f32>(h, g);
}
`;
}

const ARGS = `gpos: vec2<f32>, gpar: vec3<f32>,
   iCenter: vec4<f32>, iDirLen: vec4<f32>, iAnchor: vec4<f32>,
   iBU: vec3<f32>, iBV: vec3<f32>, iMorph: vec2<f32>,
   cfg: vec4<f32>, cfg2: vec4<f32>`;

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
export const patchSurface = wgslFn(/* wgsl */ `
fn patchSurface(${ARGS}) -> vec4<f32> {
  ${UNPACK}
  let g = morphed_N(gpos, gpar, A, B, hs, Pc, lenPc, iBU, iBV,
                    anchorRel, radius, cfg2.z, iMorph, cfg.w);
  let dd = offset_N(g, A, B, hs, Pc, lenPc, iBU, iBV);
  let dir = dirC + dd;
  let hn = height_N(dir, i32(cfg.z), cfg.y, cfg2.x, cfg2.y);

  // Surface normal from the tangential component of the height gradient.
  let gT = hn.yzw - dir * dot(dir, hn.yzw);
  let nrm = normalize(dir - gT / (radius + hn.x));
  return vec4<f32>(nrm, hn.x);
}
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
                    anchorRel, radius, cfg2.z, iMorph, cfg.w);
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
 * Shading. Self-contained — no shared helpers, so no suffixing needed.
 * mode: 0 = natural, 1 = LOD level, 2 = slope, 3 = normals.
 */
export const shadeTerrain = wgslFn(/* wgsl */ `
fn shadeTerrain(nrm: vec3<f32>, hgt: f32, camPos: vec3<f32>, rel: vec3<f32>,
                lvl: f32, sunDir: vec3<f32>, mode: f32, grid: f32) -> vec3<f32> {
  let n = normalize(nrm);
  // Local up, per pixel. Deriving it from the interpolated position rather
  // than a per-instance attribute matters: a per-patch value is constant
  // across the patch and steps visibly at every patch boundary.
  // f32 camPos is fine here — it feeds a normalize, and 0.8 m of error at
  // planetary radius is ~1e-7 rad of direction.
  let up = normalize(camPos + rel);
  let slope = clamp(1.0 - dot(n, up), 0.0, 1.0);

  let sd = normalize(sunDir);
  let ndl = max(dot(n, sd), 0.0);
  // Cheap hemispheric ambient so shadowed faces stay readable.
  let amb = 0.22 + 0.18 * clamp(dot(n, up) * 0.5 + 0.5, 0.0, 1.0);
  let lit = ndl * 0.95 + amb;

  // Metric reference grid. rel is camera-relative, so at ground level it is a
  // small number carrying full f32 precision — which makes this the instrument
  // for M1's jitter check: were the precision architecture wrong, these lines
  // would swim and crawl as the camera moves.
  var overlay = 0.0;
  if (grid > 0.0) {
    let q = rel / grid;
    let w = fwidth(q);
    let f = abs(fract(q - vec3<f32>(0.5)) - vec3<f32>(0.5)) / max(w, vec3<f32>(1e-8));
    overlay = 1.0 - min(min(f.x, min(f.y, f.z)), 1.0);
  }
  let gridCol = vec3<f32>(1.0, 0.9, 0.3);

  if (mode > 2.5) {
    return mix(n * 0.5 + 0.5, gridCol, overlay * 0.85);
  }
  if (mode > 1.5) {
    let sc = mix(vec3<f32>(0.10, 0.30, 0.16), vec3<f32>(0.92, 0.35, 0.20), slope) * lit;
    return mix(sc, gridCol, overlay * 0.85);
  }
  if (mode > 0.5) {
    // Level palette: cycles every ~6 levels so adjacent levels stay distinct.
    let t = lvl / 19.0;
    let c = vec3<f32>(
      0.5 + 0.5 * cos(6.2831853 * (t * 3.0 + 0.00)),
      0.5 + 0.5 * cos(6.2831853 * (t * 3.0 + 0.33)),
      0.5 + 0.5 * cos(6.2831853 * (t * 3.0 + 0.67)));
    return mix(c * lit, gridCol, overlay * 0.85);
  }

  // Natural: ocean, then an altitude ramp with slope-driven rock exposure.
  let deep  = vec3<f32>(0.016, 0.055, 0.105);
  let shelf = vec3<f32>(0.045, 0.145, 0.205);
  let sand  = vec3<f32>(0.62, 0.56, 0.40);
  let grass = vec3<f32>(0.16, 0.30, 0.14);
  let taiga = vec3<f32>(0.13, 0.22, 0.14);
  let rock  = vec3<f32>(0.34, 0.31, 0.29);
  let snow  = vec3<f32>(0.92, 0.94, 0.96);

  if (hgt < 0.0) {
    let t = clamp(-hgt / 3200.0, 0.0, 1.0);
    let w = mix(shelf, deep, t);
    // Water reads as a smooth dielectric: flat normal, tighter highlight.
    let wl = max(dot(up, sd), 0.0) * 0.9 + 0.12;
    return mix(w * wl, gridCol, overlay * 0.85);
  }

  var c = sand;
  c = mix(c, grass, smoothstep(20.0, 240.0, hgt));
  c = mix(c, taiga, smoothstep(900.0, 2100.0, hgt));
  c = mix(c, rock,  smoothstep(2000.0, 3400.0, hgt));
  // Snow accumulates with altitude but sheds off steep faces.
  let snowLine = smoothstep(3100.0, 4600.0, hgt) * (1.0 - smoothstep(0.35, 0.72, slope));
  c = mix(c, snow, snowLine);
  // Rock shows through wherever the surface is steep, at any altitude.
  c = mix(c, rock, smoothstep(0.42, 0.78, slope));

  return mix(c * lit, gridCol, overlay * 0.85);
}
`);
