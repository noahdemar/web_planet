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
import { atmosphere } from './atmosphere.js';
import { VEG_MAX_SLOPE } from '../planet.js';

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
  let u = r.z * 0.5 + 0.5;

  // Jittered grid rather than a regular one. A pure lattice reads as an
  // orchard from the air; full jitter clumps badly.
  let g = (cell + vec2<f32>(0.5) + r.xy * 0.42) / cells * 2.0 - vec2<f32>(1.0);
  let dd = offset_V(g, A, B, hs, Pc, lenPc, BU, BV);
  let dir = dirC + dd;

  // Cheap conservative reject: cover can only be reduced by the growth gates
  // below, so anything failing against clump*density alone can never pass.
  // Three noise octaves here save seventeen there.
  // Full band: instances only survive within ~1 km, where the mesh is at its
  // finest levels anyway, so the surface they stand on is not band-limited.
  let fullBand = cfg.x / 0.4;
  let clump = forestClump_V(dir, fullBand);
  if (u > clump * density) {
    return vec4<f32>(0.0, 0.0, 0.0, -1.0);
  }

  let hn = height_V(dir, i32(cfg.z), cfg.y, cfg2.x, cfg2.y, fullBand);
  let h = hn.x;

  // Surface normal from the tangential height gradient, as in the terrain
  // shader — trees do not grow on cliffs.
  let gT = hn.yzw - dir * dot(dir, hn.yzw);
  let nrm = normalize(dir - gT / (cfg.x + h));
  let slope = 1.0 - dot(nrm, dir);

  // Accept against exactly the cover the terrain shader tints the ground with,
  // using the same hash sample. Instance density and ground colour are then
  // the same function, which is what lets one dissolve into the other.
  let cover = forestCover_V(dir, h, slope, density, fullBand);
  if (u > cover) {
    return vec4<f32>(0.0, 0.0, 0.0, -1.0);
  }

  let pos = anchorRel + dirC * h + dd * (cfg.x + h);

  // Inverse-J size distribution: many small stems, few large. Uniform sizing
  // is the single clearest tell of a procedural forest (SPEC.md §8).
  let sv = r.x * 0.5 + 0.5;
  let small = sv * sv * sv * sv;
  var scale = mix(3.0, 29.0, small);
  // Thinner cover grows smaller stems — edges and glades taper rather than
  // ending in full-height trees against bare ground.
  scale = scale * (0.45 + 0.55 * smoothstep(0.0, 0.55, cover));
  scale = scale * (1.0 - 0.45 * smoothstep(0.15, ${VEG_MAX_SLOPE}, slope));

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
fn billboard(inst: vec4<f32>, corner: vec2<f32>, camPos: vec3<f32>,
             fadeCfg: vec4<f32>, plane: f32) -> vec3<f32> {
  let base = inst.xyz;
  let d = length(base);

  // Sub-pixel fade. A quad thinner than a couple of pixels contributes nothing
  // but aliasing, and removing it costs less than filtering it.
  let px = inst.w * fadeCfg.z / max(d, 1.0);
  let sizeFade = smoothstep(fadeCfg.w * 0.35, fadeCfg.w, px);
  // Range fade. Instances must be gone before the scatter radius, where the
  // terrain's canopy tint takes over — otherwise the forest ends at a circle.
  let rangeFade = 1.0 - smoothstep(fadeCfg.x, fadeCfg.y, d);
  let scale = inst.w * min(sizeFade, rangeFade);

  let up = normalize(camPos + base);

  var right: vec3<f32>;
  if (plane < 0.0) {
    // Camera-facing: one quad, cylindrical so trees stay upright on pitch.
    var r = cross(up, base);
    let l = length(r);
    right = select(vec3<f32>(1.0, 0.0, 0.0), r / max(l, 1e-6), l > 1e-6);
  } else {
    // Fixed-orientation crossed quads. A single facing quad has no parallax,
    // so a close tree slides against its neighbours and reads as a cut-out;
    // three fixed planes give real depth for the cost of four more triangles.
    // Not named 'ref' — that is a reserved keyword in WGSL.
    var axis = vec3<f32>(0.0, 1.0, 0.0);
    if (abs(up.y) > 0.99) { axis = vec3<f32>(0.0, 0.0, 1.0); }
    let east = normalize(cross(axis, up));
    let north = cross(up, east);
    // Per-instance yaw, so stands do not line up into rows.
    let spin = fract(inst.x * 0.017 + inst.z * 0.031) * 6.2831853;
    let a = plane + spin;
    right = east * cos(a) + north * sin(a);
  }

  return base + right * (corner.x * scale * 0.62) + up * (corner.y * scale);
}
`);

/**
 * Foliage shading.
 *
 * Lit and hazed with exactly the same air as the terrain, so a tree and the
 * ground it stands on never disagree about distance — a tree shaded without
 * aerial perspective sits in front of the landscape like a sticker.
 */
export const shadeVegetation = wgslFn(/* wgsl */ `
fn shadeVegetation(inst: vec4<f32>, uv: vec2<f32>, camPos: vec3<f32>,
                   sunDir: vec3<f32>, sunCol: vec3<f32>,
                   band: f32, mode: f32, cfg: vec4<f32>, shadow: f32) -> vec4<f32> {
  // Two crown profiles rather than one ellipse for everything. A forest of
  // identical blobs reads as procedural at a glance; a conifer/broadleaf mix
  // with varied proportions is the cheapest way out of that.
  let v = fract(inst.x * 0.013 + inst.z * 0.029 + inst.w * 0.17);
  let conifer = step(0.45, fract(inst.z * 0.041 + inst.x * 0.007));
  let x = abs(uv.x - 0.5) * 2.0;

  // Conifer: a tapered spire, widest low, with a ragged edge.
  let taper = 1.0 - smoothstep(0.12, 1.0, uv.y);
  let ragged = 0.86 + 0.14 * sin(uv.y * 34.0 + v * 20.0);
  let spire = 1.0 - smoothstep(taper * ragged * 0.92, taper * ragged, x);

  // Broadleaf: a rounded mass sitting above a clear bole.
  let wob = 0.85 + 0.4 * v;
  let d = vec2<f32>(x, (uv.y - 0.62) * (2.3 * wob));
  let round = 1.0 - smoothstep(0.62, 1.0, dot(d, d));

  let crown = mix(round, spire, conifer);
  // Conifers carry foliage lower, so their bole is shorter.
  let boleTop = mix(0.34, 0.16, conifer);
  let trunk = (1.0 - smoothstep(0.028, 0.058, abs(uv.x - 0.5))) *
              (1.0 - smoothstep(boleTop - 0.1, boleTop, uv.y));
  // Analytic edge so the crown outline resolves against MSAA coverage instead
  // of stair-stepping. The width is *capped*: on a crown only a few pixels
  // across, fwidth spans the whole blob, every pixel lands mid-ramp, and
  // alpha-to-coverage turns the entire tree into dither noise. Clamping keeps
  // the edge a genuine edge and the interior solid.
  let raw = max(crown, trunk);
  let aa = clamp(fwidth(raw), 1e-4, 0.30);
  let alpha = clamp((raw - 0.42) / aa + 0.5, 0.0, 1.0);
  if (alpha < 0.02) { discard; }

  if (mode > 0.5) {
    let c = select(select(vec3<f32>(0.95, 0.25, 0.85),
                          vec3<f32>(0.95, 0.65, 0.15), band < 1.5),
                   vec3<f32>(0.25, 0.85, 0.35), band < 0.5);
    return vec4<f32>(c, alpha);
  }

  let Rg = cfg.x;
  let wp = camPos + inst.xyz;
  let up = normalize(wp);
  let sd = normalize(sunDir);

  // Needle and broadleaf albedos, both low — foliage is far darker than
  // intuition suggests, and getting this wrong is why CG forests glow.
  let needle = vec3<f32>(0.030, 0.052, 0.022);
  let broad  = vec3<f32>(0.052, 0.082, 0.030);
  let canopy = mix(needle, broad, v);
  let bark   = vec3<f32>(0.055, 0.042, 0.030);
  let alb = mix(canopy, bark, trunk * (1.0 - crown));

  // Vertical gradient stands in for self-shadowing within the crown, and the
  // horizontal term fakes a round form on a flat quad.
  let ao = mix(0.22, 1.0, smoothstep(0.0, 0.8, uv.y));
  let form = mix(0.55, 1.0, 1.0 - abs(uv.x - 0.5) * 1.6);

  let sunTr = transmit_A(sunDepth_A(wp, sd, Rg));
  let sunUp = max(dot(up, sd), 0.0);
  // Foliage is a thin scatterer: some light comes through the leaf as well as
  // off it, which is why canopies glow slightly against the sun.
  let through = pow(max(dot(normalize(-inst.xyz), sd), 0.0), 3.0) * 0.35;
  // The 1/pi is not optional: the terrain uses it, and omitting it here made
  // foliage pi times brighter than the ground it stands on, which is exactly
  // the tonal step that gives a vegetation boundary away.
  let direct = alb * (1.0 / 3.14159265) * sunCol * sunTr
             * (sunUp * 1.7 + through) * ao * form * shadow;
  let sky = sunCol * vec3<f32>(0.055, 0.085, 0.155) * (0.045 + 0.6 * sunUp);
  // Same unshadowed bounce as the terrain, so a shadowed crown sits in the
  // same tonal range as shadowed ground rather than reading as a hole.
  let bounce = alb * alb * sunCol * sunTr * sunUp * 0.55 * ao;
  var col = direct + alb * sky * ao + bounce;

  col = aerial_A(col, camPos, wp, sd, Rg, sunCol);
  return vec4<f32>(col, alpha);
}
${atmosphere('A')}
`);
