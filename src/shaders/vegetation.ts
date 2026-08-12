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
 * On invariant I3 (anchoring): a plant must not move relative to the ground it
 * stands on. This used to be read as "take the height from a fixed reference,
 * never the resident LOD", which was right while the field was analytic and
 * had no LOD — but it has one now. `height_` fades out every octave finer than
 * the mesh's vertex spacing (BAND_FADE_LO/HI), so there is no single surface to
 * stand on: there is the surface at 55 cm spacing under your feet and the
 * surface at 140 m spacing six kilometres away, and they differ by metres.
 *
 * A fixed reference therefore produces exactly the bug it was meant to prevent.
 * The invariant is satisfied by *tracking* the drawn band limit, not by
 * ignoring it — see the placement block at the end of vegSample. Acceptance is
 * the opposite case and does still use a fixed reference, because which cells
 * grow a tree has to be a function of the ground rather than of the view (I1).
 */

import { wgslFn } from 'three/tsl';
import {
  climateBlock,
  closureBlock,
  coverBlock,
  field,
  geom,
  lapseBlock,
  channelBlock,
  noiseBlock,
  spectrumBlock,
  waterBlock,
} from './terrain.js';
import { atmosphere } from './atmosphere.js';
import { PATCH_SEGS, VEG_MAX_SLOPE, VEG_MIN_COVER, VEG_SLOPE_FULL } from '../planet.js';
import { SEASON_F0, biomeBlock } from '../biome.js';

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
             t3: vec4<f32>, t4: vec4<f32>, t5: vec4<f32>, t6: vec4<f32>,
             t7: vec4<f32>,
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

  // Cheap conservative reject: cover is closure * clump * density with
  // closure in [0,1], so nothing can pass that already fails against
  // clump*density. Three noise octaves here save ten there.
  // The band limit of the tile itself, not an unlimited one.
  //
  // This used to be radius/0.4 — every octave at full weight — on the argument
  // that instances only survive close in, where the mesh is at its finest
  // levels anyway. That argument does not survive the range going to 1800 m,
  // and it was never exactly right: the mesh fades octaves out by its own
  // vertex spacing, so a scatter that keeps them is standing on a rougher
  // surface than the one being drawn, and every stem is off by the amplitude
  // of the octaves the mesh dropped. cfg2.z is faceEdge/segments and t0.z is
  // the tile's half-size, so this is the same expression patchSurface uses.
  let fullBand = cfg.x / max(cfg2.z * hs, 0.01);
  let clump = forestClump_V(dir, fullBand);
  if (u > min(1.0, clump * density)) {
    return vec4<f32>(0.0, 0.0, 0.0, -1.0);
  }

  // The bake, reconstructed linearly from the tile-centre sample. t5 is
  // (elevation, gradient) and t6.x is wetness, all evaluated on the CPU at
  // dirC. The bake varies over 18 km and this tile is a few hundred metres, so
  // a first-order expansion about the tile centre is accurate to centimetres —
  // and it costs nothing, against five cube-map fetches per candidate.
  let bakeH = t5.x + dot(t5.yzw, dd);
  // Reconstructed across the tile, not held constant: it drives a 45 m channel
  // cut, so a tile-centre value leaves stems near a river hanging in the air.
  let distAxis = max(t7.x + dot(t7.yzw, dd), 0.0);
  // Climate before the height field: it sets the spectrum height_ walks, and
  // the cover test further down wants the same evaluation of it.
  let clim = climate_V(dir, t6.x, fullBand);
  let spec = spectrum_V(dir, clim, bakeH);
  let hn = height_V(dir, i32(cfg.z), cfg.y, bakeH, t5.yzw, t6.x, distAxis, cfg.x,
                    fullBand, spec);
  let h = hn.x;

  // Surface normal from the tangential height gradient, as in the terrain
  // shader — trees do not grow on cliffs.
  let gT = hn.yzw - dir * dot(dir, hn.yzw);
  let nrm = normalize(dir - gT / (cfg.x + h));
  let slope = 1.0 - dot(nrm, dir);

  // No trees on water. The terrain now draws the higher of the ground and the
  // waterline, so without this a stem placed on the riverbed stands in the
  // river up to its crown — and the drainage network is exactly where the
  // moisture term wants to put the densest forest.
  if (h < waterLevel_V(bakeH, t6.y, t6.x, distAxis) + 0.5) {
    return vec4<f32>(0.0, 0.0, 0.0, -1.0);
  }

  // Hard reject on slope, not just the density fade the cover applies. The
  // fade makes a stem on a cliff *unlikely*; with half a million candidates a
  // frame, unlikely is still hundreds of them, and every one is visible
  // because a tree sticking out of a rock face is exactly what the eye looks
  // for. See VEG_MAX_SLOPE.
  if (slope > ${VEG_MAX_SLOPE}) {
    return vec4<f32>(0.0, 0.0, 0.0, -1.0);
  }

  // Accept against exactly the cover the terrain shader tints the ground with,
  // using the same hash sample. Instance density and ground colour are then
  // the same function, which is what lets one dissolve into the other — which
  // now includes the climate, so the scatter stops at the treeline and the
  // desert edge for the same reason the tint does.
  // Same aspect term the terrain shades with — see the aspect block in
  // shadeTerrain. Without it the ground tint would carry the wooded/bare
  // asymmetry between opposing valley sides and the stems would not, which is
  // exactly the disagreement the shared cover function exists to prevent.
  let poleAxV = vec3<f32>(0.0, select(-1.0, 1.0, dir.y >= 0.0), 0.0);
  let poleTV = poleAxV - dir * dot(dir, poleAxV);
  let nTV = nrm - dir * dot(dir, nrm);
  let aspectV = dot(nTV, poleTV) / max(length(poleTV), 1e-4);

  let cover = forestCover_V(dir, h, slope, density, fullBand,
                            vec2<f32>(tempAt_V(clim.x, h),
                                      clamp(clim.y + aspectV * 0.085, 0.0, 1.0)));
  // A floor as well as the probability test. The tail of the cover function
  // places isolated stems on ground that should simply be bare — see
  // VEG_MIN_COVER.
  if (cover < ${VEG_MIN_COVER} || u > cover) {
    return vec4<f32>(0.0, 0.0, 0.0, -1.0);
  }

  // ── placement: stand on the surface that will actually be drawn ───────
  //
  // Everything above ran at the tile's own fixed band limit, and has to: which
  // cells grow a tree must be a function of the ground and not of where the
  // camera is (I1), or the forest reshuffles as you walk into it.
  //
  // Placement is the opposite requirement. CDLOD does not draw one surface —
  // it draws the field band-limited to whatever vertex spacing the distance
  // earns, 55 cm under your feet against this tile's fixed 17.6 m. Every octave
  // between those two is in the drawn ground and was missing from the ground
  // the scatter measured, which is p90 1.4 m of relief and up to 5.2 m, under
  // stems as short as three metres. That is the floating, and it was worst
  // exactly where it is most visible: close in, where the mesh is finest and
  // the tile band limit is furthest from it.
  //
  // The spacing is continuous rather than quantised to the level the selector
  // would pick. The mesh morphs between levels, so a stem that stepped at each
  // level boundary would pop against ground that does not. cfg2.w is the LOD
  // factor and cfg2.z is faceEdge/segments, so this is the selector's own rule
  // — subdivide while d <= lodFactor * faceEdge / 2^L — solved for spacing
  // instead of for L.
  //
  // Only accepted stems pay for the second evaluation; the rejects above are
  // the overwhelming majority of candidates.
  let dCam = max(length(anchorRel + dd * cfg.x), 1.0);
  let meshSpacing = dCam / max(cfg2.w * ${PATCH_SEGS}.0, 1.0);
  let meshBand = cfg.x / max(meshSpacing, 0.01);
  let hDraw = height_V(dir, i32(cfg.z), cfg.y, bakeH, t5.yzw, t6.x, distAxis,
                       cfg.x, meshBand, spec).x;

  let pos = anchorRel + dirC * hDraw + dd * (cfg.x + hDraw);

  // ── what grows here ───────────────────────────────────────────────────
  //
  // Stem height and species from the same biome blend the ground colour uses,
  // so a taiga is spruce at 21 m, a rainforest is broadleaf at 36 m, and a
  // savanna is a scatter of 10 m crowns over open grass. Before this every
  // biome on the planet grew the same 29 m tree in the same 50/50 mix of
  // conifer and broadleaf, which made the ground colour and the thing standing
  // on it describe different places.
  let swing = noised_V(dir * ${SEASON_F0} + vec3<f32>(61.7)).x;
  let bveg = biomeVeg_V(tempAt_V(clim.x, h), clim.x, clim.y,
                        season_V(abs(dir.y), swing), h);
  let tallest = max(bveg.x, 0.6);

  // Inverse-J size distribution: many small stems, few large. Uniform sizing
  // is the single clearest tell of a procedural forest (SPEC.md §8).
  let sv = r.x * 0.5 + 0.5;
  let small = sv * sv * sv * sv;
  var scale = mix(tallest * 0.14, tallest, small);
  // Thinner cover grows smaller stems — edges and glades taper rather than
  // ending in full-height trees against bare ground.
  scale = scale * (0.45 + 0.55 * smoothstep(0.0, 0.55, cover));
  scale = scale * (1.0 - 0.45 * smoothstep(${VEG_SLOPE_FULL}, ${VEG_MAX_SLOPE}, slope));

  // Scale and identity in one channel — see packInst_.
  // Species centred on the biome's conifer fraction, with enough spread that a
  // stand is mixed rather than uniform — a boreal forest is nearly all spruce
  // but not entirely, and that minority is what stops it reading as a texture.
  let kind = clamp(bveg.y + (r.y * 0.5 + 0.5 - 0.5) * 0.55, 0.0, 1.0);
  return vec4<f32>(pos, packInst_V(scale, kind));
}
${geom('V')}
${channelBlock('V')}
${waterBlock('V')}
${noiseBlock('V')}
${instBlock('V')}
${field('V')}
${climateBlock('V')}
${biomeBlock('V')}
${lapseBlock('V')}
${spectrumBlock('V')}
${closureBlock('V')}
${coverBlock('V')}
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
  let h = instScale_B(inst);
  let px = h * fadeCfg.z / max(d, 1.0);
  let sizeFade = smoothstep(fadeCfg.w * 0.35, fadeCfg.w, px);
  // Range fade. Instances must be gone before the scatter radius, where the
  // terrain's canopy tint takes over — otherwise the forest ends at a circle.
  let rangeFade = 1.0 - smoothstep(fadeCfg.x, fadeCfg.y, d);
  let scale = h * min(sizeFade, rangeFade);

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
    // Per-instance yaw, so stands do not line up into rows. From the packed
    // identity, not the position: see packInst_.
    let spin = instRnd_B(inst).z * 6.2831853;
    let a = plane + spin;
    right = east * cos(a) + north * sin(a);
  }

  return base + right * (corner.x * scale * 0.62) + up * (corner.y * scale);
}
${instBlock('B')}
${noiseBlock('B')}
`);

/**
 * Per-instance identity. Needs `noiseBlock` for the hash.
 *
 * See the note inside: this is the fix for trees that re-rolled their species
 * and their yaw every time the camera moved.
 */
export function instBlock(s: string): string {
  return /* wgsl */ `
/**
 * A tree's species, its crown wobble and its yaw have to be functions of the
 * *tree*, and every one of them used to be a hash of inst.xyz. That vector is
 * camera-relative — it is the whole point of the precision architecture — so
 * every one of them re-rolled as the camera moved. Trees rotated on the spot,
 * and a broadleaf became a conifer when you walked toward it.
 *
 * The scatter already has a stable hash, built from the tile's global cell
 * index. The only problem was carrying it to the draw shaders. Rather than a
 * second vec4 per instance — 19 MB more at capacity, and a second read per
 * vertex — it rides in the low bits of the scale, which needs nowhere near the
 * 24 bits of mantissa it was given. 12.5 cm of size resolution buys 16 bits of
 * identity, and a tree does not have a preferred height to that precision.
 */
fn packInst_${s}(scale: f32, rnd: f32) -> f32 {
  return floor(max(scale, 0.0) * 8.0) + clamp(rnd, 0.0, 0.9999);
}
fn instScale_${s}(inst: vec4<f32>) -> f32 { return floor(inst.w) * 0.125; }
/** Three decorrelated randoms from the packed one. */
fn instRnd_${s}(inst: vec4<f32>) -> vec3<f32> {
  return hash33_${s}(vec3<f32>(fract(inst.w) * 65536.0, 31.0, 7.0)) * 0.5 + 0.5;
}
`;
}

/**
 * The tree as a surface of revolution: (radius, height), both as fractions of
 * the instance's height.
 *
 * Everything about a given tree comes out of `conifer` and `v`, two hashes of
 * its position, so the near band needs no per-instance data beyond the vec4
 * the scatter already writes.
 *
 * `wob` is a smooth analytic term rather than a hash so the surface stays
 * differentiable — the normal below is taken by differencing this function,
 * and a hashed radius would return a normal that is pure noise.
 */
export function treeProfile(s: string): string {
  return /* wgsl */ `
fn boleTop_${s}(conifer: f32) -> f32 { return mix(0.38, 0.20, conifer); }

fn treeRH_${s}(u: f32, theta: f32, kind: f32, conifer: f32, v: f32) -> vec2<f32> {
  let bole = boleTop_${s}(conifer);
  if (kind > 0.5) {
    // Bole: tapers, and flares a little at the base where real trunks do.
    // Radius as a fraction of height — a 29 m tree wants roughly a 0.35 m
    // radius, so ~0.012, not the 0.03 that put 3 m thick columns in the
    // foreground.
    let flare = 1.0 + 0.55 * pow(max(1.0 - u * 3.2, 0.0), 2.0);
    let r = mix(0.014, 0.008, u) * flare * (0.8 + 0.4 * v);
    return vec2<f32>(r, u * bole * 1.02);
  }
  // Canopy, from the top of the bole to the tip.
  let h = mix(bole, 1.0, u);
  // Conifer: a spire that sheds outward at the bottom, with layered whorls.
  let whorl = 0.88 + 0.12 * sin(u * 21.0 + v * 17.0);
  let spire = 0.30 * pow(max(1.0 - u, 0.0), 0.85) * whorl;
  // Broadleaf: an ellipsoid mass, widest a little below the middle.
  let q = clamp(u * 1.06 - 0.03, 0.0, 1.0);
  let dome = 0.36 * sqrt(max(1.0 - (2.0 * q - 1.0) * (2.0 * q - 1.0), 0.0));
  let wob = 1.0 + 0.15 * sin(theta * 3.0 + u * 5.0 + v * 23.0)
                + 0.08 * sin(theta * 5.0 - u * 9.0 + v * 11.0);
  return vec2<f32>(mix(dome, spire, conifer) * wob * (0.85 + 0.3 * v), h);
}

/** Species and variation hashes. Must agree everywhere they are used. */
fn treeV_${s}(inst: vec4<f32>) -> f32 { return instRnd_${s}(inst).x; }
fn treeConifer_${s}(inst: vec4<f32>) -> f32 { return step(0.45, instRnd_${s}(inst).y); }

/** Local frame: up is the planet normal, so trees stand up rather than lean. */
fn treeFrame_${s}(inst: vec4<f32>, camPos: vec3<f32>) -> mat3x3<f32> {
  let up = normalize(camPos + inst.xyz);
  var axis = vec3<f32>(0.0, 1.0, 0.0);
  if (abs(up.y) > 0.99) { axis = vec3<f32>(0.0, 0.0, 1.0); }
  let east = normalize(cross(axis, up));
  return mat3x3<f32>(east, cross(up, east), up);
}

/** Offset from the instance base, in metres, for one (u, θ) on the sleeve. */
fn treeOffset_${s}(inst: vec4<f32>, F: mat3x3<f32>, u: f32, theta: f32,
                   kind: f32, height: f32) -> vec3<f32> {
  let conifer = treeConifer_${s}(inst);
  let v = treeV_${s}(inst);
  // Per-instance yaw, or every crown wobble in the stand points the same way.
  let a = theta + instRnd_${s}(inst).z * 6.2831853;
  let rh = treeRH_${s}(u, a, kind, conifer, v);
  return (F[0] * cos(a) + F[1] * sin(a)) * (rh.x * height) + F[2] * (rh.y * height);
}
`;
}

/**
 * Near-band tree vertex. Real geometry, not a billboard.
 *
 * Beyond about 45 m a crown is a few pixels across and a quad is honest; below
 * it, a camera-facing quad is the single most obvious thing in the frame — it
 * has no parallax against its neighbours, no silhouette that changes as you
 * walk around it, and it cannot self-shadow. This puts an actual bole and
 * crown there for about 150 triangles a tree, which at the ~350 instances the
 * near band holds is a rounding error against the terrain.
 */
export const treeVertex = wgslFn(/* wgsl */ `
fn treeVertex(inst: vec4<f32>, tsec: vec3<f32>, camPos: vec3<f32>,
              fadeCfg: vec4<f32>) -> vec3<f32> {
  let F = treeFrame_G(inst, camPos);
  return inst.xyz + treeOffset_G(inst, F, tsec.x, tsec.y, tsec.z, instScale_G(inst));
}
${treeProfile('G')}
${instBlock('G')}
${noiseBlock('G')}
`);

/**
 * The same surface, differenced for its normal.
 *
 * Taken by finite difference rather than analytically because the profile
 * carries a crown wobble and a whorl term; differentiating those by hand is
 * error-prone and this is two extra profile evaluations on a vertex shader
 * that runs a few tens of thousands of times.
 */
export const treeNormal = wgslFn(/* wgsl */ `
fn treeNormal(inst: vec4<f32>, tsec: vec3<f32>, camPos: vec3<f32>) -> vec3<f32> {
  let F = treeFrame_N2(inst, camPos);
  let e = 0.012;
  let p0 = treeOffset_N2(inst, F, tsec.x, tsec.y, tsec.z, 1.0);
  let pu = treeOffset_N2(inst, F, min(tsec.x + e, 1.0), tsec.y, tsec.z, 1.0);
  let pt = treeOffset_N2(inst, F, tsec.x, tsec.y + e, tsec.z, 1.0);
  let n = cross(pt - p0, pu - p0);
  let l = length(n);
  // The tip closes to a point, where both differences vanish; fall back to up.
  return select(F[2], n / l, l > 1e-9);
}
${treeProfile('N2')}
${instBlock('N2')}
${noiseBlock('N2')}
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
  let v = instRnd_A(inst).x;
  let conifer = step(0.45, instRnd_A(inst).y);
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
${instBlock('A')}
${noiseBlock('A')}
`);

/**
 * Near-band tree shading. Same air and the same albedos as the billboards, so
 * the band boundary is a change of silhouette and nothing else.
 *
 * The lighting differs in one way that matters: this has a real surface
 * normal, so N·L does the work the billboard had to fake with a vertical
 * gradient and a horizontal form term.
 */
export const shadeTree = wgslFn(/* wgsl */ `
fn shadeTree(inst: vec4<f32>, nrm: vec3<f32>, kind: f32, uu: f32,
             camPos: vec3<f32>, sunDir: vec3<f32>, sunCol: vec3<f32>,
             mode: f32, cfg: vec4<f32>, shadow: f32) -> vec4<f32> {
  if (mode > 0.5) { return vec4<f32>(0.25, 0.85, 0.35, 1.0); }

  let v = instRnd_T2(inst).x;
  let Rg = cfg.x;
  let wp = camPos + inst.xyz;
  let up = normalize(wp);
  let sd = normalize(sunDir);
  let n = normalize(nrm);

  let needle = vec3<f32>(0.030, 0.052, 0.022);
  let broad  = vec3<f32>(0.052, 0.082, 0.030);
  // Bark is not as dark as foliage. At the foliage albedo a shaded trunk went
  // to black, which reads as a hole in the tree rather than as wood.
  let bark   = vec3<f32>(0.105, 0.082, 0.058);
  let alb = mix(mix(needle, broad, v), bark, kind);

  // Crowns are not opaque shells — the inside is shaded by the outside. A
  // little occlusion toward the base of the canopy is most of what stops a
  // low-poly tree reading as a plastic cone. Boles get a gentler version: they
  // stand under the crown, not inside it.
  let canopyAO = mix(0.35, 1.0, smoothstep(0.0, 0.7, uu));
  let ao = mix(canopyAO, mix(0.55, 0.85, uu), kind);

  let sunTr = transmit_T2(sunDepth_T2(wp, sd, Rg));
  let sunUp = max(dot(up, sd), 0.0);
  let ndl = max(dot(n, sd), 0.0);
  // Foliage transmits as well as reflects, which is why a canopy glows when
  // the sun is behind it. Only the leaves — bark does not.
  let through = pow(max(-dot(n, sd), 0.0), 2.0) * 0.30 * (1.0 - kind);

  let direct = alb * (1.0 / 3.14159265) * sunCol * sunTr
             * (ndl + through) * ao * shadow;
  let sky = sunCol * vec3<f32>(0.055, 0.085, 0.155) * (0.045 + 0.6 * sunUp);
  let bounce = alb * alb * sunCol * sunTr * sunUp * 0.55 * ao;
  var col = direct + alb * sky * ao * (0.5 + 0.5 * dot(n, up)) + bounce;

  col = aerial_T2(col, camPos, wp, sd, Rg, sunCol);
  return vec4<f32>(col, 1.0);
}
${atmosphere('T2')}
${instBlock('T2')}
${noiseBlock('T2')}
`);
