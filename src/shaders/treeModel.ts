/**
 * Scanned-model vegetation WGSL. See treeAssets.ts for what feeds it.
 *
 * The procedural tree these replace built its own shape in the vertex shader
 * from a per-instance hash, which is why it needed nothing but a vec4 per
 * instance. A model has a fixed shape, so the vertex shader's job here is
 * smaller and entirely rigid: take a vertex of a unit-height tree standing at
 * the origin, and place it on the planet at the instance's stem.
 *
 * What has *not* changed is the precision architecture. The camera sits at the
 * origin and instance positions are relative to it, so the local frame comes
 * from `camPos + inst.xyz` and the vertex never leaves camera-relative space
 * (I3). Nor has the lighting: `shadeTreeModel` and `shadeImpostor` run exactly
 * the sun, sky, bounce and aerial terms the terrain and the old billboards ran,
 * because a tree lit by anything else sits in front of the landscape rather
 * than in it.
 */

import { wgslFn } from 'three/tsl';
import { atmosphere } from './atmosphere.js';
import { instBlock } from './vegetation.js';
import { noiseBlock } from './terrain.js';
import { IMPOSTOR_DROP, IMPOSTOR_HALF, IMPOSTOR_PAD, IMPOSTOR_YAWS } from '../treeAssets.js';
import { VEG_SPECIES } from '../planet.js';

/**
 * Local frame and yaw, shared by every function here.
 *
 * The yaw is the same `instRnd(inst).z` the old billboards used, and it has to
 * stay that way for two reasons: it is stable under camera motion (it is
 * hashed from the packed identity, not from the camera-relative position), and
 * the impostor's slice lookup has to undo exactly the rotation the geometry
 * bands apply, or a tree spins as it crosses the band boundary.
 */
function frameBlock(s: string): string {
  return /* wgsl */ `
fn treeYaw_${s}(inst: vec4<f32>) -> f32 {
  return instRnd_${s}(inst).z * 6.2831853;
}
/**
 * Which model an instance is: 0 conifer, 1 broadleaf, matching the order of
 * SPECIES in treeAssets.ts and of the species-split bands in planet.ts.
 *
 * Deliberately the same hash treeConifer_ used for the procedural tree, so
 * the scatter's biome-driven conifer fraction still decides the mix — a taiga
 * is still nearly all spruce, it is just a scanned spruce now.
 */
fn treeSpecies_${s}(inst: vec4<f32>) -> f32 {
  return 1.0 - step(0.45, instRnd_${s}(inst).y);
}
/** (east, north, up) at the instance. Up is the planet normal, so trees stand. */
fn modelFrame_${s}(inst: vec4<f32>, camPos: vec3<f32>) -> mat3x3<f32> {
  let up = normalize(camPos + inst.xyz);
  var axis = vec3<f32>(0.0, 1.0, 0.0);
  if (abs(up.y) > 0.99) { axis = vec3<f32>(0.0, 0.0, 1.0); }
  let east = normalize(cross(axis, up));
  return mat3x3<f32>(east, cross(up, east), up);
}
/** Model space (+Y up) → the instance's frame, yaw applied. */
fn modelToWorld_${s}(inst: vec4<f32>, F: mat3x3<f32>, v: vec3<f32>) -> vec3<f32> {
  let a = treeYaw_${s}(inst);
  let c = cos(a);
  let s2 = sin(a);
  let x = v.x * c - v.z * s2;
  let z = v.x * s2 + v.z * c;
  return F[0] * x + F[1] * z + F[2] * v.y;
}
`;
}

/**
 * sRGB → linear.
 *
 * The base-colour array is stored in sRGB (see treeAssets.ts) because eight
 * bits of linear crushes the darks, and foliage is almost entirely darks. The
 * exact curve rather than a 2.2 power: the toe is where leaf shadow lives, and
 * the approximation is visibly wrong there.
 */
function srgbBlock(s: string): string {
  return /* wgsl */ `
fn toLinear_${s}(c: vec3<f32>) -> vec3<f32> {
  let lo = c / 12.92;
  let hi = pow((c + vec3<f32>(0.055)) / 1.055, vec3<f32>(2.4));
  return select(lo, hi, c > vec3<f32>(0.04045));
}
`;
}

/**
 * Scanned albedo, retargeted to the range the rest of the renderer assumes.
 *
 * Scan and game-art texture sets are authored to look right under a generic
 * "sunlight" that already has an exposure baked into it, and come out around
 * 0.15–0.25 reflectance. Real foliage is 0.04–0.09, and the terrain, the
 * grass and the old billboards were all written to that — so dropping the
 * textures in unscaled makes the forest roughly three times brighter than the
 * ground it stands on, which is the exact tonal step that gives a vegetation
 * boundary away (the note in shadeVegetation about the missing 1/pi is the
 * same failure from the other direction).
 *
 * Scaling rather than replacing keeps every bit of variation the scan captured
 * — the leaf-to-leaf hue shifts and the bark's grain — and only moves the mean.
 */
function albedoBlock(s: string): string {
  return /* wgsl */ `
fn treeAlbedo_${s}(texel: vec3<f32>, leaf: f32) -> vec3<f32> {
  let lin = toLinear_${s}(texel);
  return lin * mix(0.72, 0.42, leaf);
}
${srgbBlock(s)}
`;
}

/* ── geometry bands ──────────────────────────────────────────────────────── */

/**
 * Place one model vertex.
 *
 * No fade here, unlike the billboards. The geometry bands stop at 110 m and
 * both the sub-pixel and the range fade start thousands of metres further out,
 * so there is nothing to apply — and scaling a solid model toward zero would
 * shrink it into the ground rather than fade it out anyway.
 */
export const modelVertex = wgslFn(/* wgsl */ `
fn modelVertex(inst: vec4<f32>, pos: vec3<f32>, camPos: vec3<f32>) -> vec3<f32> {
  let F = modelFrame_M(inst, camPos);
  return inst.xyz + modelToWorld_M(inst, F, pos * instScale_M(inst));
}
${frameBlock('M')}
${instBlock('M')}
${noiseBlock('M')}
`);

/** The model's own normal, carried into the instance's frame. */
export const modelNormal = wgslFn(/* wgsl */ `
fn modelNormal(inst: vec4<f32>, nrm: vec3<f32>, camPos: vec3<f32>) -> vec3<f32> {
  let F = modelFrame_MN(inst, camPos);
  return normalize(modelToWorld_MN(inst, F, nrm));
}
${frameBlock('MN')}
${instBlock('MN')}
${noiseBlock('MN')}
`);

/**
 * Geometry-band shading.
 *
 * Same air, same sun and the same bounce as the terrain. Two things the old
 * procedural tree could not do are worth the change: the albedo is the scan's
 * rather than one of two constants, and leaf cards are two-sided sheets, so
 * N·L has to be taken on whichever face is toward the light or half the canopy
 * goes black at every sun angle.
 */
export const shadeTreeModel = wgslFn(/* wgsl */ `
fn shadeTreeModel(inst: vec4<f32>, nrm: vec3<f32>, texel: vec4<f32>, leaf: f32,
                  hUp: f32, camPos: vec3<f32>, sunDir: vec3<f32>, sunCol: vec3<f32>,
                  mode: f32, cfg: vec4<f32>, shadow: f32) -> vec4<f32> {
  // Coverage first, because it is shared with the debug path and because
  // getting it wrong is the loudest artefact this shader can produce.
  //
  // A cut-out needs opposite treatment at the two ends of its scale, and the
  // whole trick here is deciding which end a fragment is at.
  //
  // Magnified, the texture's alpha is a smooth antialiased ramp spread over
  // many pixels. Handing that to alpha-to-coverage asks MSAA to dither half
  // the canopy across four samples, and the pattern reshuffles with every
  // sub-pixel of camera motion — the tree crawls. There the fix is to divide
  // by the screen-space derivative, collapsing the ramp to a one-pixel step
  // so only the true silhouette is partially covered.
  //
  // Minified, that same step is exactly wrong. A mip level averages alpha as
  // well as colour, so a crown whose needles cover a third of each texel
  // arrives with alpha near a third *everywhere* — and a third is the honest
  // answer. Thresholding it at a half deletes the canopy and keeps whatever
  // happens to sit above the line, which is the speckle this used to show on
  // a sparse conifer at forty metres. Coverage-preserving mips are the proper
  // cure and are not available: three's WebGPU backend ignores supplied
  // mipmaps for an array texture and generates its own.
  //
  // So: read how minified we are from the width of the ramp itself, and cross
  // over. A narrow ramp means magnified, and gets the step. A wide one means
  // the mip chain has already done the averaging, and the average is used as
  // coverage directly.
  let w = clamp(fwidth(texel.a), 1e-4, 0.5);
  let sharp = clamp((texel.a - 0.5) / w + 0.5, 0.0, 1.0);
  // The gain is the same one the impostor uses, and for the same reason: a
  // conifer is mostly edge, so its averaged alpha understates how much of the
  // crown a viewer actually sees.
  let soft = min(texel.a * 1.6, 1.0);
  let alpha = mix(sharp, soft, smoothstep(0.06, 0.30, w));
  if (alpha < 0.02) { discard; }

  if (mode > 0.5) {
    return vec4<f32>(mix(vec3<f32>(0.55, 0.35, 0.18), vec3<f32>(0.25, 0.85, 0.35), leaf),
                     alpha);
  }

  let Rg = cfg.x;
  let wp = camPos + inst.xyz;
  let up = normalize(wp);
  let sd = normalize(sunDir);
  let alb = treeAlbedo_T3(texel.rgb, leaf);

  // Leaf cards have no meaningful front: the same quad is lit from either
  // side. Flipping the normal toward the sun and letting transmission carry
  // the rest is what stops a canopy reading as half-dead.
  var n = normalize(nrm);
  if (leaf > 0.5 && dot(n, sd) < 0.0) { n = -n; }

  let sunTr = sunLight_T3(wp, sd, Rg);
  let sunUp = max(dot(up, sd), 0.0);
  let ndl = max(dot(n, sd), 0.0);
  // Foliage transmits as well as reflects, which is why a canopy glows when
  // the sun is behind it. Only the leaves — bark does not.
  let through = pow(max(-dot(normalize(nrm), sd), 0.0), 2.0) * 0.30 * leaf;

  // The scan carries no occlusion of its own, and a canopy without any reads
  // as a cloud of loose leaves rather than a solid mass. hUp is the vertex's
  // height up the model, 0 at the base and 1 at the tip, which is the same
  // approximation the procedural crown used — deep canopy is dark, the top is
  // open. Baking real AO per model would be better and is a build step this
  // does not have.
  let h = clamp(hUp, 0.0, 1.0);
  // Boles get the gentler ramp: they stand *under* the crown, not inside it.
  let ao = mix(mix(0.55, 0.85, h), mix(0.30, 1.0, smoothstep(0.0, 0.7, h)), leaf);

  let direct = alb * (1.0 / 3.14159265) * sunCol * sunTr
             * (ndl + through) * ao * shadow;
  let sky = sunCol * vec3<f32>(0.055, 0.085, 0.155) * (0.045 + 0.6 * sunUp);
  let bounce = alb * alb * sunCol * sunTr * sunUp * 0.55 * ao;
  var col = direct + alb * sky * ao * (0.5 + 0.5 * dot(n, up)) + bounce;

  col = aerial_T3(col, camPos, wp, sd, Rg, sunCol);
  return vec4<f32>(col, alpha);
}
${albedoBlock('T3')}
${atmosphere('T3')}
${instBlock('T3')}
${noiseBlock('T3')}
`);

/* ── impostor bands ──────────────────────────────────────────────────────── */

/**
 * Impostor quad corner.
 *
 * Cylindrical like the old billboard — rotation about the local vertical only,
 * so trees stay upright when the camera pitches — but the box it spans is not
 * chosen for looks any more. It is exactly the box the orthographic bake saw,
 * because the texture in it is a photograph of that box (IMPOSTOR_HALF and
 * IMPOSTOR_DROP in treeAssets.ts). Any other extent stretches the model.
 */
export const impostorVertex = wgslFn(/* wgsl */ `
fn impostorVertex(inst: vec4<f32>, corner: vec2<f32>, camPos: vec3<f32>,
                  fadeCfg: vec4<f32>) -> vec3<f32> {
  let base = inst.xyz;
  let d = length(base);

  // Sub-pixel fade. A quad thinner than a couple of pixels contributes nothing
  // but aliasing, and removing it costs less than filtering it.
  let h = instScale_I(inst);
  let px = h * fadeCfg.z / max(d, 1.0);
  let sizeFade = smoothstep(fadeCfg.w * 0.35, fadeCfg.w, px);
  // Range fade. Instances must be gone before the scatter radius, where the
  // terrain's canopy tint takes over — otherwise the forest ends at a circle.
  let rangeFade = 1.0 - smoothstep(fadeCfg.x, fadeCfg.y, d);
  let scale = h * min(sizeFade, rangeFade);

  let up = normalize(camPos + base);
  var r = cross(up, base);
  let l = length(r);
  let right = select(vec3<f32>(1.0, 0.0, 0.0), r / max(l, 1e-6), l > 1e-6);

  let span = ${(IMPOSTOR_HALF * 2).toFixed(4)};
  return base + right * (corner.x * span * scale)
              + up * ((corner.y * span - ${IMPOSTOR_DROP.toFixed(4)}) * scale);
}
${instBlock('I')}
${noiseBlock('I')}
`);

/**
 * Which baked yaw to read, and where in the atlas.
 *
 * Evaluated per vertex and interpolated, which is safe because the slice is a
 * function of the instance and the camera only — every corner of a quad picks
 * the same one, so nothing is interpolated across a tile boundary.
 *
 * The angle undoes the instance yaw before indexing. The geometry bands rotate
 * the model by that yaw; if the lookup did not remove it, a tree would appear
 * to snap to a different facing the moment it crossed into an impostor band.
 */
export const impostorUV = wgslFn(/* wgsl */ `
fn impostorUV(inst: vec4<f32>, corner: vec2<f32>, camPos: vec3<f32>) -> vec4<f32> {
  let up = normalize(camPos + inst.xyz);
  var axis = vec3<f32>(0.0, 1.0, 0.0);
  if (abs(up.y) > 0.99) { axis = vec3<f32>(0.0, 0.0, 1.0); }
  let east = normalize(cross(axis, up));
  let north = cross(up, east);

  // Camera is at the origin, so the direction from the tree to the viewer is
  // simply the negated instance position.
  let view = normalize(-inst.xyz);
  let phi = atan2(dot(view, north), dot(view, east)) - instRnd_U(inst).z * 6.2831853;
  // The bake put its camera at (sin a, ., cos a), so its azimuth measured the
  // same way is pi/2 - a. Inverting that is what maps a view direction back to
  // the column it was rendered from.
  let a = 1.5707963 - phi;
  let yaws = ${IMPOSTOR_YAWS.toFixed(1)};
  let yawAt = fract(a / 6.2831853) * yaws;
  let slice0 = floor(yawAt);
  let slice1 = floor(fract((slice0 + 1.0) / yaws) * yaws);
  let yawMix = smoothstep(0.18, 0.82, fract(yawAt));

  // Species s owns rows [s, s+1) of the atlas counted from v = 0, because
  // that is the order treeAssets.ts assembles them in.
  //
  // Inset into the tile by IMPOSTOR_PAD on every side: the bake rendered a
  // correspondingly wider box, so this lands the quad's corners exactly on
  // the box the camera saw, with the empty margin left over for the mip chain
  // to bleed into instead of the neighbouring tree.
  let sp = ${VEG_SPECIES.toFixed(1)};
  let species = treeSpecies_U(inst);
  let pad = ${IMPOSTOR_PAD.toFixed(4)};
  let span = 1.0 - 2.0 * pad;
  let tileX = pad + (corner.x + 0.5) * span;
  let tileY = (species + pad + corner.y * span) / sp;
  return vec4<f32>((slice0 + tileX) / yaws, (slice1 + tileX) / yaws,
                   tileY, yawMix);
}
${frameBlock('U')}
${instBlock('U')}
${noiseBlock('U')}
`);

/**
 * Impostor shading.
 *
 * The atlas holds albedo and coverage only, so everything below is the same
 * lighting the geometry bands run, minus the one term an impostor cannot have:
 * a surface normal. The vertical occlusion gradient and the horizontal form
 * term stand in for it, exactly as they did on the procedural billboard — at
 * these ranges the difference between that and a real normal is well under the
 * aerial perspective sitting on top of both.
 */
export const shadeImpostor = wgslFn(/* wgsl */ `
fn shadeImpostor(inst: vec4<f32>, texel: vec4<f32>, uv: vec2<f32>, band: f32,
                 camPos: vec3<f32>, sunDir: vec3<f32>, sunCol: vec3<f32>,
                 mode: f32, cfg: vec4<f32>, shadow: f32) -> vec4<f32> {
  // Coverage, taken from the atlas directly rather than thresholded against
  // it. That is a deliberate departure from how the procedural billboard cut
  // its silhouette, and it is forced by the atlas being mipped.
  //
  // A mip of a cut-out averages alpha. The bake writes coverage that is binary
  // at mip zero, but a conifer is mostly edge — needles and gaps — so by the
  // time an impostor is fifty pixels tall its average alpha across a texel is
  // well under a half over the entire crown. Threshold that at anything near a
  // half and the whole canopy disappears at range while the trunk, which is
  // solid, survives: a forest visible from head height and empty from forty
  // metres up.
  //
  // Treating it as coverage is also the more honest model. A canopy really is
  // partly transparent at a distance, and the gain lets a crown that has
  // averaged down to a third still read as foliage instead of vanishing.
  let alpha = min(texel.a * 1.5, 1.0);
  if (alpha < 0.02) { discard; }

  if (mode > 0.5) {
    let c = select(vec3<f32>(0.95, 0.25, 0.85), vec3<f32>(0.95, 0.65, 0.15), band < 4.5);
    return vec4<f32>(c, alpha);
  }

  let Rg = cfg.x;
  let wp = camPos + inst.xyz;
  let up = normalize(wp);
  let sd = normalize(sunDir);
  // Everything in the atlas is foliage as far as the retarget is concerned:
  // the bole is a handful of pixels at these ranges and is in shadow in all
  // of them.
  let alb = treeAlbedo_IM(texel.rgb, 1.0);

  // Vertical gradient stands in for self-shadowing within the crown, and the
  // horizontal term fakes a round form on a flat quad.
  let ao = mix(0.22, 1.0, smoothstep(0.0, 0.8, uv.y));
  let form = mix(0.55, 1.0, 1.0 - abs(uv.x - 0.5) * 1.6);

  let sunTr = sunLight_IM(wp, sd, Rg);
  let sunUp = max(dot(up, sd), 0.0);
  // Foliage is a thin scatterer: some light comes through the leaf as well as
  // off it, which is why canopies glow slightly against the sun.
  let through = pow(max(dot(normalize(-inst.xyz), sd), 0.0), 3.0) * 0.35;
  let direct = alb * (1.0 / 3.14159265) * sunCol * sunTr
             * (sunUp * 1.7 + through) * ao * form * shadow;
  let sky = sunCol * vec3<f32>(0.055, 0.085, 0.155) * (0.045 + 0.6 * sunUp);
  let bounce = alb * alb * sunCol * sunTr * sunUp * 0.55 * ao;
  var col = direct + alb * sky * ao + bounce;

  col = aerial_IM(col, camPos, wp, sd, Rg, sunCol);
  return vec4<f32>(col, alpha);
}
${albedoBlock('IM')}
${atmosphere('IM')}
${instBlock('IM')}
${noiseBlock('IM')}
`);

/**
 * The species bit, for the scatter.
 *
 * A band is one indirect draw and one geometry, so it can only hold one
 * species — which means binning has to happen at scatter time, on the packed
 * instance, rather than at draw time. This is the same expression
 * `treeSpecies_` evaluates in the draw shaders, exported separately because
 * the scatter has only the vec4 and none of the rest of the frame machinery.
 */
export const instSpecies = wgslFn(/* wgsl */ `
fn instSpecies(inst: vec4<f32>) -> f32 {
  return 1.0 - step(0.45, instRnd_S(inst).y);
}
${instBlock('S')}
${noiseBlock('S')}
`);
