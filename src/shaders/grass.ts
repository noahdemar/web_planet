/**
 * Grass WGSL. See SPEC.md §8 and the ground-clutter block in planet.ts.
 *
 * The structure follows momentchan/false-earth: a compute pass writes packed
 * per-blade data, and the draw unpacks it into a Bezier blade. What is packed
 * is deliberately small — one vec4 per blade — because at 65 k blades the
 * instance buffer is read once per vertex and there are nine vertices each.
 *
 * ── Where a blade *is* ──
 *
 * The grid is camera-centred and world-snapped. Camera-centred because grass
 * only exists within GRASS_RANGE and following the camera is the only way to
 * spend the whole budget on ground that is actually visible; world-snapped
 * because a grid that slid with the camera would give every blade a new seed
 * every frame, and the field would boil. The snap is done on the CPU in f64
 * and arrives as an integer cell origin, so the hash argument is exact.
 *
 * ── Precision ──
 *
 * No catastrophic cancellation anywhere, and it costs nothing here. The naive
 * form is dir·(R+h) − camPos, two vectors near 6.4e6 differenced down to
 * metres. But the camera's own anchor is
 *
 *     camDir·R − camPos  =  camDir·R − camDir·(R + alt)  =  −camUp·alt
 *
 * which is exact and small, and the within-field direction delta is just the
 * tangent offset over R. So the whole position is
 *
 *     −camUp·alt  +  camDir·h  +  tangentOffset
 *
 * with every term already of metre scale. This is the same identity the
 * terrain's patch anchors use, in the one case where it collapses to a line.
 */

import { wgslFn } from 'three/tsl';
import {
  GRASS_GRID,
  GRASS_H_HI,
  GRASS_H_LO,
  GRASS_MAX_SLOPE,
  GRASS_FULL,
  GRASS_RANGE,
  GRASS_SLOPE_FULL,
  GRASS_SPACING,
  GRASS_WIDTH,
} from '../planet.js';
import { atmosphere } from './atmosphere.js';
import {
  channelBlock,
  climateBlock,
  closureBlock,
  coverBlock,
  field,
  lapseBlock,
  noiseBlock,
  spectrumBlock,
  waterBlock,
} from './terrain.js';
import { biomeBlock } from '../biome.js';

const f = (v: number): string => (Number.isInteger(v) ? `${v}.0` : `${v}`);

/**
 * Scatter one blade.
 *
 * cell     integer cell index within the grid, [0, GRID)²
 * bake     (elevation, gradient) at the camera, reconstructed linearly across
 *          the field — it varies over the bake's 9 km cell and the field is
 *          34 m, so first order is exact to millimetres
 * bake2    (wetness, lake depth, distance to drainage axis, —)
 * frame    camera direction and altitude
 * cfg      (radius, heightScale, octaves, bandLimit)
 * cfg2     (gridOriginX, gridOriginY, time, density)
 *
 * Returns (positionRelativeToCamera, packed); packed < 0 means rejected.
 */
export const grassSample = wgslFn(/* wgsl */ `
fn grassSample(cell: vec2<f32>, bake: vec4<f32>, bake2: vec4<f32>,
               camDir: vec3<f32>, camEast: vec3<f32>, camNorth: vec3<f32>,
               alt: f32, cfg: vec4<f32>, cfg2: vec4<f32>, view: vec4<f32>) -> vec4<f32> {
  let radius = cfg.x;
  let half = ${f(GRASS_GRID / 2)};

  // Global cell coordinate, for the seed. cfg2.xy is the integer origin the
  // CPU snapped this frame from the camera's own face-uv position, so this is
  // a *world* index: a blade keeps it as the grid slides underneath, which is
  // the whole reason the field does not boil when you walk.
  let gx = cfg2.x + cell.x;
  let gy = cfg2.y + cell.y;

  // One hash drives jitter, height, lean and colour.
  let r = hash33_R(vec3<f32>(gx, gy, 11.0));
  let j = r.xy * 0.5 + 0.5;

  // Offset from the camera, in metres. Built from the *fractional* part of the
  // camera's grid position (cfg2.zw) rather than by differencing two absolute
  // coordinates near 4.6e6 — the snap is what keeps this exact.
  let dxm = (cell.x - half + j.x - cfg2.z) * ${f(GRASS_SPACING)};
  let dym = (cell.y - half + j.y - cfg2.w) * ${f(GRASS_SPACING)};

  let d2 = dxm * dxm + dym * dym;
  if (d2 > ${f(GRASS_RANGE * GRASS_RANGE)}) {
    return vec4<f32>(0.0, 0.0, 0.0, -1.0);
  }

  // ── rejection, after siliconjungle/inkwell-webgpu-grass (MIT) ─────────
  //
  // That renderer sustains ~234k drawn blades out of 16.8M candidates — a 70:1
  // rejection — and almost all of the ratio is these two tests. This field had
  // neither: every candidate inside the radius was expanded and drawn, whether
  // it was behind the camera or not, which is why the range had to stay at 22 m
  // to stay affordable.
  //
  // Frustum first, as a horizontal cone rather than a clip-space test. Grass is
  // a ground-plane field, so the horizontal half-angle is the whole of what
  // matters, and a cone costs a dot product against a vec2 instead of a matrix
  // multiply per candidate. view.xy is the camera's forward in this frame's
  // (east, north) basis and view.z is cos of the half-angle plus margin.
  // Blades very close to the camera are kept regardless — the cone is
  // meaningless there and they are the ones under your feet.
  let dist = sqrt(d2);
  if (dist > 2.0) {
    if (dot(vec2<f32>(dxm, dym) / dist, view.xy) < view.z) {
      return vec4<f32>(0.0, 0.0, 0.0, -1.0);
    }
  }

  // Then a radial stochastic dissolve. Holding full density to the horizon is
  // what makes a long range expensive; thinning it with a hash means the field
  // fades out instead of ending, and no orbit ever exposes the boundary. The
  // hash is on the *global* cell index, so a blade's fate is a property of the
  // ground and does not change as you walk toward it (I1).
  let full = ${f(GRASS_FULL)};
  let density = 1.0 - smoothstep(full * full, ${f(GRASS_RANGE * GRASS_RANGE)}, d2);
  // The tail is thinner than it was, because it is now four times longer. The
  // product of the two is what costs, and past the full-density radius a blade
  // is a couple of pixels — the eye reads the tail as ground colour with a
  // texture on it, not as countable blades.
  if (r.z * 0.5 + 0.5 > density * density * 0.18 + 0.82 * step(d2, full * full)) {
    return vec4<f32>(0.0, 0.0, 0.0, -1.0);
  }

  // Tangent offset as a direction delta, then the surface there.
  let tangent = camEast * dxm + camNorth * dym;
  let dd = tangent / radius;
  let dir = normalize(camDir + dd);

  let bakeH = bake.x + dot(bake.yzw, dd);
  let distAxis = max(bake2.z, 0.0);
  // Climate first: the height field needs the spectrum the biome asks for, and
  // the growth test below needs the same climate. One evaluation serves both.
  let clim = climate_R(dir, bake2.x, 1.0e9);
  let hn = height_R(dir, i32(cfg.z), cfg.y, bakeH, bake.yzw, bake2.x, distAxis,
                    radius, 1.0e9, spectrum_R(dir, clim, bakeH), clim);
  let h = hn.x;

  // Nothing grows under water.
  if (h < waterLevel_R(bakeH, bake2.y, bake2.x, distAxis) + 0.05) {
    return vec4<f32>(0.0, 0.0, 0.0, -1.0);
  }

  // Slope, from the same gradient the terrain uses. Grass holds on far steeper
  // ground than a tree does, but not on a cliff.
  let gT = hn.yzw - dir * dot(dir, hn.yzw);
  let nrm = normalize(dir - gT / (radius + h));
  let slope = 1.0 - dot(nrm, dir);

  // Grass is the ground cover *between* the trees, so it wants the same
  // moisture the canopy does but tolerates the dry end far better — that
  // difference is what puts steppe where the forest stops. The climate itself
  // was evaluated above, because the height field needed the spectrum from it.
  let temp = tempAt_R(clim.x, h);
  //
  // Hard reject above the slope limit rather than only thinning towards it,
  // matching the canopy scatter. A fade alone leaves the tail of the
  // distribution standing on ground far past the point anything grows on it,
  // and those are exactly the blades that show, because steep ground is what
  // the camera is usually pointed at.
  if (slope > ${f(GRASS_MAX_SLOPE)}) {
    return vec4<f32>(0.0, 0.0, 0.0, -1.0);
  }
  let grow = smoothstep(0.05, 0.22, temp)
           * smoothstep(0.06, 0.28, clim.y)
           * (1.0 - smoothstep(${f(GRASS_SLOPE_FULL)}, ${f(GRASS_MAX_SLOPE)}, slope));
  if (j.x * 0.37 + j.y * 0.63 > grow * cfg.w) {
    return vec4<f32>(0.0, 0.0, 0.0, -1.0);
  }

  // Position: see the precision note at the top of this file. Every term is
  // already metre-scale, so there is nothing to cancel.
  let pos = camDir * (h - alt) + tangent;

  // Height, biased small: a stand of identical blades is the same tell as a
  // forest of identical trees.
  let sv = r.z * 0.5 + 0.5;
  let height = mix(${f(GRASS_H_LO)}, ${f(GRASS_H_HI)}, sv * sv) * (0.55 + 0.45 * grow);

  return vec4<f32>(pos, packBlade_R(height, j.x));
}
${noiseBlock('R')}
${channelBlock('R')}
${field('R')}
${waterBlock('R')}
${climateBlock('R')}
${lapseBlock('R')}
${biomeBlock('R')}
${spectrumBlock('R')}
${closureBlock('R')}
${coverBlock('R')}
${bladePack('R')}
`);

/** Height and identity share one channel, as the tree instances do. */
export function bladePack(s: string): string {
  return /* wgsl */ `
fn packBlade_${s}(height: f32, rnd: f32) -> f32 {
  return floor(max(height, 0.0) * 512.0) + clamp(rnd, 0.0, 0.9999);
}
fn bladeHeight_${s}(inst: vec4<f32>) -> f32 { return floor(inst.w) * (1.0 / 512.0); }
fn bladeRnd_${s}(inst: vec4<f32>) -> vec3<f32> {
  return hash33_${s}(vec3<f32>(fract(inst.w) * 65536.0, 5.0, 19.0)) * 0.5 + 0.5;
}
`;
}

/**
 * Blade vertex.
 *
 * A quadratic Bezier from the base to the tip, widening nowhere and tapering
 * to a point — real grass is widest just above the sheath and the difference
 * is invisible at this size, so the width is a single taper.
 *
 * seg.x is the parameter along the blade, seg.y the side (−1 or +1, and 0 at
 * the tip vertex). The control point leans with the per-blade random and with
 * the wind, so a gust runs across the field rather than every blade nodding in
 * place.
 */
export const grassVertex = wgslFn(/* wgsl */ `
fn grassVertex(inst: vec4<f32>, seg: vec4<f32>, camPos: vec3<f32>,
               cfg: vec4<f32>) -> vec3<f32> {
  let anchor = inst.xyz;
  let rnd = bladeRnd_G2(inst);
  let variation = fract(rnd.x + seg.z * 7.13 + seg.w * 11.71);
  let h = bladeHeight_G2(inst) * mix(0.72, 1.08, variation);
  let t = seg.x;

  let up = normalize(camPos + anchor);
  var axis = vec3<f32>(0.0, 1.0, 0.0);
  if (abs(up.y) > 0.99) { axis = vec3<f32>(0.0, 0.0, 1.0); }
  let east = normalize(cross(axis, up));
  let north = cross(up, east);
  let base = anchor + east * seg.z + north * seg.w;

  // Per-blade yaw, so the field has no grain.
  let a = variation * 6.2831853;
  let facing = east * cos(a) + north * sin(a);

  // Wind. One travelling wave over the field plus a per-blade phase: the wave
  // is what makes it read as wind rather than as noise, and the phase is what
  // stops the field moving as one sheet.
  let tsec = cfg.z;
  let phase = dot(base, east) * 0.35 + dot(base, north) * 0.22;
  let gust = sin(tsec * 1.7 + phase) * 0.5 + 0.5;
  // Every blade leans, and by different amounts. A field of vertical spikes
  // has no silhouette at all — the lean is most of what makes grass legible
  // from a metre away, and the per-blade spread is what stops a gust looking
  // like a single sheet tipping over.
  let sway = (0.30 + 0.45 * gust) * (0.45 + 1.1 * rnd.y);

  // Quadratic Bezier: straight up, then bent over.
  let p0 = vec3<f32>(0.0);
  let p1 = up * (h * 0.55);
  let p2 = up * (h * (0.92 - 0.35 * sway)) + facing * (h * sway);
  let om = 1.0 - t;
  let p = p0 * (om * om) + p1 * (2.0 * om * t) + p2 * (t * t);
  let tan = normalize((p1 - p0) * (2.0 * om) + (p2 - p1) * (2.0 * t));

  // Width across the blade, tapering to nothing at the tip.
  let side = normalize(cross(tan, facing));
  // Widest just above the base and tapering to a point, rather than a linear
  // wedge: a blade is a ribbon, and the shoulder is what catches the light.
  let taper = (1.0 - t) * (0.55 + 0.45 * (1.0 - t));
  let w = ${f(GRASS_WIDTH)} * h * taper * 0.5;

  return base + p + side * (seg.y * w);
}
${noiseBlock('G2')}
${bladePack('G2')}
`);

/**
 * Blade shading.
 *
 * Grass is thin and translucent, so the dominant term at low sun is light
 * coming *through* the blade, not off it. A one-sided N·L makes a field go
 * flat black when the sun is behind it, which is the opposite of what grass
 * does — so the diffuse term is wrapped and the back face is lit by
 * transmission rather than being discarded.
 */
export const shadeGrass = wgslFn(/* wgsl */ `
fn shadeGrass(inst: vec4<f32>, nrm: vec3<f32>, tt: f32, camPos: vec3<f32>,
              sunDir: vec3<f32>, sunCol: vec3<f32>, cfg: vec4<f32>,
              shadow: f32) -> vec4<f32> {
  let Rg = cfg.x;
  let wp = camPos + inst.xyz;
  let up = normalize(wp);
  let sd = normalize(sunDir);
  let v = normalize(-inst.xyz);
  let rnd = bladeRnd_S(inst);

  // Colour varies blade to blade and up the blade: the base sits in shade and
  // holds more moisture, the tip is bleached. A single flat green is the
  // clearest tell of procedural grass there is.
  let dry = rnd.z;
  let baseCol = mix(vec3<f32>(0.045, 0.085, 0.030), vec3<f32>(0.085, 0.115, 0.038), dry);
  let tipCol  = mix(vec3<f32>(0.105, 0.150, 0.055), vec3<f32>(0.180, 0.185, 0.085), dry);
  let alb = mix(baseCol, tipCol, tt * tt);

  var n = nrm;
  if (dot(n, v) < 0.0) { n = -n; }

  // Wrapped diffuse plus transmission through the blade.
  let ndl = dot(n, sd);
  let wrapped = clamp((ndl + 0.45) / 1.45, 0.0, 1.0);
  let trans = pow(clamp(-ndl, 0.0, 1.0), 1.6) * 0.55;

  let sunTr = sunLight_S(wp, sd, Rg);
  var col = alb * (1.0 / 3.14159265) * sunCol * sunTr * (wrapped * shadow + trans * shadow);

  // Sky, and self-shadowing down the blade — the base of a sward sees almost
  // no sky, and without this the field has no depth at all.
  let ao = 0.20 + 0.80 * tt;
  col = col + alb * sunCol * vec3<f32>(0.055, 0.085, 0.155) * 0.55 * ao;

  col = aerial_S(col, camPos, wp, sd, Rg, sunCol);
  return vec4<f32>(col, 1.0);
}
${atmosphere('S')}
${noiseBlock('S')}
${bladePack('S')}
`);
