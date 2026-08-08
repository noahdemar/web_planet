/**
 * Vegetation WGSL. See SPEC.md §8.
 *
 * Two pure functions, both reusing the terrain's precision reconstruction and
 * height field verbatim — a plant must sit on exactly the surface the terrain
 * shader draws, and the only way to guarantee that is to run the same code.
 *
 *   `vegSample`   scatter: cell → world position, or rejected
 *   `billboard`   draw: instance → camera-facing quad corner
 *
 * On invariant I3 (anchoring): plants must take their ground height from a
 * *fixed* reference, never the resident terrain LOD, or they float and sink as
 * you approach. The M1 field is analytic and has no LOD, so I3 holds for free
 * right now. When M5 replaces it with tile sampling, this must pin to L13
 * explicitly or the bug appears.
 */

import { wgslFn } from 'three/tsl';
import { geom, field } from './terrain.js';
import {
  VEG_MAX_SLOPE,
  VEG_MIN_ELEVATION,
  VEG_TREELINE,
} from '../planet.js';

/**
 * One scatter candidate.
 *
 * Deterministic and O(1): position and acceptance are a pure function of the
 * cell index and the seed, with no neighbour queries and no ordering. That is
 * what lets this run as one flat compute dispatch, be regenerated at any time,
 * and give identical results on every run (**I1**).
 *
 * Returns `(positionRelativeToCamera, scale)`; scale ≤ 0 means rejected.
 */
export const vegSample = wgslFn(/* wgsl */ `
fn vegSample(t0: vec4<f32>, t1: vec4<f32>, t2: vec4<f32>,
             t3: vec4<f32>, t4: vec4<f32>,
             cell: vec2<f32>, cfg: vec4<f32>, cfg2: vec4<f32>,
             vcfg: vec4<f32>) -> vec4<f32> {
  let A = t0.x;
  let B = t0.y;
  let hs = t0.z;
  let dirC = t1.xyz;
  let lenPc = t1.w;
  let anchorRel = t2.xyz;
  let BU = t3.xyz;
  let BV = t4.xyz;
  let Pc = dirC * lenPc;

  let cells = vcfg.x;
  let density = vcfg.y;

  // Globally unique cell coordinate on this cube face. Built from the tile's
  // own indices, never from its slot in the resident list — the pattern has to
  // be a function of the ground, not of what happens to be in view (I1).
  // It is also continuous across tile borders, so no seams in the layout.
  let gcell = vec2<f32>(t3.w, t4.w) * cells + cell;

  // One hash drives jitter, thinning, species and size.
  let r = hash33_V(vec3<f32>(gcell, t0.w * 4096.0 + vcfg.z));

  // Thin first: it is free, and it removes ~60% of candidates before the
  // expensive height evaluation.
  if (r.z * 0.5 + 0.5 > density) {
    return vec4<f32>(0.0, 0.0, 0.0, -1.0);
  }

  // Jittered grid rather than a regular one. A pure lattice reads as an
  // orchard from the air; full jitter clumps badly. 0.42 is close to the
  // best blue-noise-like spacing you get from one hash.
  let g = (cell + vec2<f32>(0.5) + r.xy * 0.42) / cells * 2.0 - vec2<f32>(1.0);

  let dd = offset_V(g, A, B, hs, Pc, lenPc, BU, BV);
  let dir = dirC + dd;
  let hn = height_V(dir, i32(cfg.z), cfg.y, cfg2.x, cfg2.y);
  let h = hn.x;

  if (h < ${VEG_MIN_ELEVATION}.0 || h > ${VEG_TREELINE}.0) {
    return vec4<f32>(0.0, 0.0, 0.0, -1.0);
  }

  // Surface normal from the tangential height gradient, as in the terrain
  // shader — trees do not grow on cliffs.
  let gT = hn.yzw - dir * dot(dir, hn.yzw);
  let nrm = normalize(dir - gT / (cfg.x + h));
  let slope = 1.0 - dot(nrm, dir);
  if (slope > ${VEG_MAX_SLOPE}) {
    return vec4<f32>(0.0, 0.0, 0.0, -1.0);
  }

  let pos = anchorRel + dirC * h + dd * (cfg.x + h);

  // Inverse-J size distribution: many small stems, few large. Uniform sizing
  // is the single clearest tell of a procedural forest (SPEC.md §8). r.x is
  // roughly uniform on [-1,1]; the fourth power skews hard toward small.
  let u = r.x * 0.5 + 0.5;
  let small = u * u * u * u;
  var scale = mix(2.5, 27.0, small);
  // Thin toward the treeline and toward the shore.
  scale = scale * (1.0 - smoothstep(${VEG_TREELINE}.0 - 350.0, ${VEG_TREELINE}.0, h));
  scale = scale * smoothstep(${VEG_MIN_ELEVATION}.0, ${VEG_MIN_ELEVATION}.0 + 25.0, h);
  // Crowding: steeper ground carries smaller stems.
  scale = scale * (1.0 - 0.55 * smoothstep(0.15, ${VEG_MAX_SLOPE}, slope));

  return vec4<f32>(pos, max(scale, 0.0));
}
${geom('V')}
${field('V')}
`);

/**
 * Camera-facing quad corner for one instance.
 *
 * Cylindrical, not spherical: the billboard rotates about the local vertical
 * only, so trees stay upright when the camera pitches. Cheap because the
 * camera sits at the origin — the view vector *is* the instance position.
 */
export const billboard = wgslFn(/* wgsl */ `
fn billboard(inst: vec4<f32>, corner: vec2<f32>, camPos: vec3<f32>) -> vec3<f32> {
  let base = inst.xyz;
  let scale = inst.w;
  let up = normalize(camPos + base);
  var right = cross(up, base);
  let l = length(right);
  // Degenerate only when looking straight down the instance's own axis, where
  // the quad is edge-on and invisible anyway.
  right = select(vec3<f32>(1.0, 0.0, 0.0), right / max(l, 1e-6), l > 1e-6);
  return base + right * (corner.x * scale * 0.55) + up * (corner.y * scale);
}
`);

/** Foliage shading. Deliberately flat and cheap — this is a placeholder canopy. */
export const shadeVegetation = wgslFn(/* wgsl */ `
fn shadeVegetation(inst: vec4<f32>, uv: vec2<f32>, camPos: vec3<f32>,
                   sunDir: vec3<f32>, band: f32, mode: f32) -> vec4<f32> {
  // Soft elliptical canopy mask, so the quad does not read as a rectangle.
  let d = vec2<f32>((uv.x - 0.5) * 2.0, (uv.y - 0.55) * 2.2);
  let m = 1.0 - smoothstep(0.55, 1.0, dot(d, d));
  // Trunk: a narrow column in the lower third.
  let trunk = (1.0 - smoothstep(0.035, 0.075, abs(uv.x - 0.5))) *
              (1.0 - smoothstep(0.30, 0.42, uv.y));
  let alpha = clamp(max(m, trunk), 0.0, 1.0);
  if (alpha < 0.35) { discard; }

  if (mode > 0.5) {
    // Band debug: near green, mid amber, far magenta.
    let c = select(select(vec3<f32>(0.95, 0.25, 0.85),
                          vec3<f32>(0.95, 0.65, 0.15), band < 1.5),
                   vec3<f32>(0.25, 0.85, 0.35), band < 0.5);
    return vec4<f32>(c, alpha);
  }

  let up = normalize(camPos + inst.xyz);
  let sd = normalize(sunDir);
  // Vertical gradient stands in for self-shadowing under the canopy.
  let ao = mix(0.35, 1.0, smoothstep(0.0, 0.75, uv.y));
  let lit = max(dot(up, sd), 0.0) * 0.55 + 0.45;

  // Hash the instance so neighbouring crowns differ in hue and value.
  let v = fract(inst.x * 0.013 + inst.z * 0.029 + inst.w * 0.17);
  let canopy = mix(vec3<f32>(0.11, 0.26, 0.10), vec3<f32>(0.22, 0.38, 0.15), v);
  let bark = vec3<f32>(0.16, 0.12, 0.09);
  let c = mix(canopy, bark, trunk * (1.0 - m));
  return vec4<f32>(c * ao * lit, alpha);
}
`);
