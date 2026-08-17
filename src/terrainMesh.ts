/**
 * Patch geometry and material. See SPEC.md §6.
 *
 * One shared 33×33 grid, instanced across every visible patch. Nothing about
 * the terrain's shape lives in a vertex buffer — the grid carries only local
 * coordinates and CDLOD parity, and the sphere mapping happens in the shader
 * from per-instance data. That is what keeps patch turnover free: selecting
 * a different set of patches rewrites 20 floats each, not 1089 vertices.
 */

import {
  BufferAttribute,
  DynamicDrawUsage,
  InstancedBufferAttribute,
  InstancedBufferGeometry,
  Mesh,
  Vector3,
  Vector4,
} from 'three';
import { MeshBasicNodeMaterial } from 'three/webgpu';
import {
  attribute,
  cross,
  texture,
  dot as tslDot,
  float,
  mix as tslMix,
  normalize,
  smoothstep as tslSmoothstep,
  uniform,
  varying,
  vec2,
  vec3,
  vec4,
  int,
  select,
} from 'three/tsl';
import {
  DEFAULT_OCTAVES,
  MAX_PATCHES,
  PATCH_SEGS,
  PATCH_VERTS,
  RADIUS,
  FACE_EDGE,
  FLAT_WET_CUT,
  FLAT_WET_HI,
  FLAT_WET_LO,
  FOREST_DENSITY,
  LOCAL_PERIOD,
  SEA_BAND,
  SEA_LEVEL,
} from './planet.js';
import { SHADOW_DEPTH_OFFSET } from './shadows.js';
import type { PatchBuffers } from './quadtree.js';
import type { PlanetSurface } from './planetData.js';
import {
  LITTER_LAYER,
  ROCK_LAYER,
  SAND_LAYER,
  SNOW_LAYER,
  SOIL_LAYER,
  type TerrainMaterials,
} from './terrainMaterials.js';
import { ATLAS_PAD } from './bake/cubemap.js';
import {
  cubeAtlasUV,
  patchClimate,
  patchAmp,
  patchDirection,
  patchPosition,
  coastWarp,
  patchSpectrum,
  patchSurface,
  patchWater,
  shadeTerrain,
} from './shaders/terrain.js';

/** 0 natural · 1 LOD · 2 slope · 3 normals · 4 cover · 5 albedo · 6 climate
 *  · 7 elevation readback · 8 shadow factor. */
export type ShadeMode = 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8;

/** Returns a scalar node in [0,1]: the fraction of sun reaching a point. */
export type ShadowFactor = (rel: unknown) => ReturnType<typeof float>;

// wgslFn is declared as returning an untyped Node, and TSL only attaches
// swizzle/assignment types to nodes with a known type. These recover the
// types the WGSL signatures already guarantee.
type Vec2Node = ReturnType<typeof vec2>;
type Vec3Node = ReturnType<typeof vec3>;
type Vec4Node = ReturnType<typeof vec4>;
const asVec2 = (n: unknown): Vec2Node => n as Vec2Node;
const asVec3 = (n: unknown): Vec3Node => n as Vec3Node;
const asVec4 = (n: unknown): Vec4Node => n as Vec4Node;

/**
 * The TSL node surface the bake-sampling block relies on.
 *
 * TSL's own types are exact about which builder produced a node, so chaining
 * `.sub().mul().div()` across `attribute`, `cubeTexture` and `wgslFn` results
 * does not type-check even though every one of those operations is valid.
 * Same approach as shaders/shadowSample.ts: state what is used rather than
 * reaching for `any`, and keep every cast in the five wrappers below.
 */
interface N {
  add(x: unknown): N;
  sub(x: unknown): N;
  mul(x: unknown): N;
  div(x: unknown): N;
  level(x: unknown): N;
  readonly r: N;
  readonly g: N;
  sqrt(): N;
  negate(): N;
  min(x: unknown): N;
  mul(x: unknown): N;
  div(x: unknown): N;
  readonly b: N;
  readonly a: N;
  readonly w: N;
  clamp(a: unknown, b: unknown): N;
  negate(): N;
  readonly xyz: N;
  sub(x: unknown): N;
  add(x: unknown): N;
  max(x: unknown): N;
}
const n = (x: unknown): N => x as N;
/** wgslFn's declared parameter type does not admit a named-argument object. */
const nCall = (f: unknown, a: Record<string, unknown>): N =>
  n((f as (x: Record<string, unknown>) => unknown)(a));
const nNormalize = (x: N): N => n(normalize(x as never));
const nCross = (a: N, b: N): N => n(cross(a as never, b as never));
const nDot = (a: N, b: N): N => n(tslDot(a as never, b as never));
const nVec4 = (...xs: unknown[]): N => n((vec4 as (...a: unknown[]) => unknown)(...xs));
const nMix = (a: unknown, b: unknown, t: unknown): N =>
  n((tslMix as (...x: unknown[]) => unknown)(a, b, t));
const nStep = (e0: unknown, e1: unknown, x: unknown): N =>
  n((tslSmoothstep as (...a: unknown[]) => unknown)(e0, e1, x));
const nTex = (t: unknown, uv: N): N =>
  n((texture as (a: unknown, b: unknown) => unknown)(t, uv));

/**
 * Patch grid plus a one-cell skirt ring.
 *
 * CDLOD is only crack-free while neighbouring patches differ by at most one
 * level. Near the ground that assumption breaks: the LOD distance varies by
 * roughly one patch width per patch, so a two-level jump can appear at a
 * quadtree boundary, and a single-parity morph cannot bridge two levels.
 * Guaranteeing 2:1 balance would need cross-face neighbour queries; raising
 * lodFactor past ~4.25 would work but quadruples the patch count.
 *
 * The skirt is the standard cheap answer: a ring of vertices that duplicate
 * the patch border and are pushed radially inward, so any T-junction gap is
 * covered by geometry that is otherwise hidden below the surface. Costs 13%
 * more triangles and removes that failure mode entirely.
 *
 * It removes only that one. A skirt is depth, and the other way a seam opens
 * is width: two patches agree about their shared edge analytically and not
 * bitwise, and a rasteriser leaves a pixel of background between two edges
 * that differ in the last bits of f32. That is PATCH_BLEED's job, and the two
 * are not interchangeable — see planet.ts and LESSONS §23.
 *
 * It was, for a while, blamed for two things that were not its fault: holes
 * along coastlines from above, and a picket fence of bright slivers at patch
 * borders over flat ground at grazing angles. Turning the skirt off did make
 * both go away, which is exactly the misleading result — it removed the
 * geometry that was being mis-rejected, not the thing rejecting it. Both were
 * the logarithmic depth buffer; see the note in main.ts. The skirt is
 * unchanged and correct, and sim.audit() reads 0.000% with it in place.
 */
function buildGrid(): InstancedBufferGeometry {
  const n = PATCH_VERTS + 2; // interior grid plus the skirt ring
  const verts = n * n;
  const pos = new Float32Array(verts * 2); // local coords in [-1, 1]
  // (parityX, parityY, isSkirt). Packed into one attribute because WebGPU
  // caps a pipeline at 8 vertex buffers and this geometry already uses 8;
  // a ninth fails pipeline creation outright.
  const par = new Float32Array(verts * 3);

  const clamp = (i: number) => Math.min(PATCH_SEGS, Math.max(0, i));

  for (let jy = 0; jy < n; jy++) {
    for (let jx = 0; jx < n; jx++) {
      const ix = clamp(jx - 1);
      const iy = clamp(jy - 1);
      const k = (jy * n + jx) * 2;
      const k3 = (jy * n + jx) * 3;
      pos[k] = (ix / PATCH_SEGS) * 2 - 1;
      pos[k + 1] = (iy / PATCH_SEGS) * 2 - 1;
      // Odd-indexed vertices collapse onto their even neighbour when the
      // morph completes, turning the grid into its own parent (SPEC.md §6).
      // Ring vertices inherit their border vertex's parity so the skirt
      // follows the morph exactly.
      par[k3] = ix & 1;
      par[k3 + 1] = iy & 1;
      par[k3 + 2] = jx === 0 || jy === 0 || jx === n - 1 || jy === n - 1 ? 1 : 0;
    }
  }

  const segs = n - 1;
  const idx = new Uint32Array(segs * segs * 6);
  let t = 0;
  for (let jy = 0; jy < segs; jy++) {
    for (let jx = 0; jx < segs; jx++) {
      const a = jy * n + jx;
      const b = a + 1;
      const c = a + n + 1;
      const d = a + n;
      // CCW seen from +W, matching the right-handed face bases.
      idx[t++] = a; idx[t++] = b; idx[t++] = c;
      idx[t++] = a; idx[t++] = c; idx[t++] = d;
    }
  }

  const geo = new InstancedBufferGeometry();
  geo.setAttribute('gpos', new BufferAttribute(pos, 2));
  geo.setAttribute('gpar', new BufferAttribute(par, 3));
  geo.setIndex(new BufferAttribute(idx, 1));
  geo.instanceCount = 0;
  return geo;
}

export class TerrainMesh {
  readonly mesh: Mesh;
  /**
   * Same vertex position the display material uses, with a trivial fragment
   * shader. The shadow pass must reproduce the surface exactly — reusing the
   * node is the only way to guarantee that as the terrain evolves.
   */
  readonly depthMaterial: MeshBasicNodeMaterial;
  readonly geometry: InstancedBufferGeometry;
  readonly material: MeshBasicNodeMaterial;

  private attrs: InstancedBufferAttribute[] = [];

  /** cfg = (radius, heightScale, mountainOctaves, 2/segments) */
  private cfg = uniform(new Vector4(RADIUS, 1, DEFAULT_OCTAVES, 2 / PATCH_SEGS));
  /** cfg2 = (seaLevel, seaBand, referenceRadius, faceEdge/segments) */
  private cfg2 = uniform(new Vector4(SEA_LEVEL, SEA_BAND, RADIUS, FACE_EDGE / PATCH_SEGS));
  /** cfg3 = (forestDensity, planetRadius, cloudCover, timeSeconds) */
  private cfg3 = uniform(new Vector4(FOREST_DENSITY, RADIUS, 0.5, 0));
  private weather = uniform(new Vector4());
  private sun = uniform(new Vector3(0.55, 0.42, 0.72).normalize());
  /** Sun irradiance, not a screen colour — everything is multiplied by it. */
  private sunCol = uniform(new Vector3(1, 0.97, 0.92));
  /** Camera world position in f32 — shading only, never geometry (SPEC.md I4). */
  private camPos = uniform(new Vector3());
  /**
   * Camera position folded into one LOCAL_PERIOD cell, computed in f64.
   *
   * `rel + camSnap` is the world position modulo the period — small enough for
   * f32 to carry a quarter-metre lattice. Whole periods are what get dropped,
   * and the detail lattice wraps on exactly that boundary, so nothing moves
   * when the camera crosses one.
   */
  private camSnap = uniform(new Vector3());
  /** Light direction for the depth pass; kept in step with `sun`. */
  private shadowSun = uniform(new Vector3(0, 1, 0));
  private mode = uniform(0);
  private gridSpacing = uniform(0);

  /**
   * Angular size of one bake cell, radians. The finite-difference step for the
   * baked gradient; see the sampling block in the constructor.
   */
  private bakeStep = uniform(0.0028);

  constructor(
    buffers: PatchBuffers,
    surface: PlanetSurface,
    materials: TerrainMaterials,
    shadowFactor?: ShadowFactor,
  ) {
    // FACE_EDGE/size is the cell's arc length; divide by the radius for the
    // angle. Derived rather than hard-coded so changing the bake resolution
    // cannot silently leave the normals sampling the wrong stencil.
    this.bakeStep.value = FACE_EDGE / surface.size / RADIUS;
    this.geometry = buildGrid();

    const inst = (name: string, arr: Float32Array, size: number) => {
      const a = new InstancedBufferAttribute(arr, size);
      a.setUsage(DynamicDrawUsage);
      this.geometry.setAttribute(name, a);
      this.attrs.push(a);
      return a;
    };
    inst('iCenter', buffers.center, 4);
    inst('iDirLen', buffers.dirLen, 4);
    inst('iAnchor', buffers.anchor, 4);
    inst('iBU', buffers.basisU, 3);
    inst('iBV', buffers.basisV, 3);
    inst('iMorph', buffers.morph, 2);

    const iDirLen = asVec4(attribute('iDirLen', 'vec4'));
    const iAnchor = asVec4(attribute('iAnchor', 'vec4'));

    // Shared by both entry points; each wgslFn is called with exactly the
    // names its WGSL signature declares.
    const args = {
      gpos: attribute('gpos', 'vec2'),
      gpar: attribute('gpar', 'vec3'),
      iCenter: attribute('iCenter', 'vec4'),
      iDirLen,
      iAnchor,
      iBU: attribute('iBU', 'vec3'),
      iBV: attribute('iBV', 'vec3'),
      iMorph: attribute('iMorph', 'vec2'),
      cfg: this.cfg,
      cfg2: this.cfg2,
      cfg3: this.cfg3,
    };

    // ── Sampling the M3 bake ────────────────────────────────────────────
    //
    // A texture cannot be bound to a wgslFn — TSL has no way to pass one in —
    // so the bake is read here in the node graph and handed to `patchSurface`
    // as plain vectors. That needs the vertex's direction before the surface
    // is evaluated, which is what `patchDirection` exists for.
    const dirSp = n(patchDirection(args));
    const dir = dirSp.xyz;
    const iBUn = n(attribute('iBU', 'vec3'));

    // Tangent frame for the finite difference. The patch's own face basis is
    // used rather than a fixed world axis because it cannot be parallel to a
    // direction inside its own face — no pole case, no branch.
    const t1 = nNormalize(iBUn.sub(dir.mul(nDot(dir, iBUn))));
    const t2 = nCross(dir, t1);

    const atlas = uniform(
      new Vector4(surface.size, surface.width, surface.height, ATLAS_PAD),
    );
    const sampleBake = (d: N): N =>
      nTex(surface.texture, nCall(cubeAtlasUV, { dir: d, atlas })).level(0);

    // One bake cell. A shorter step would only read the slope of the bilinear
    // interpolant, which is piecewise constant, so the normals would facet
    // along cell boundaries.
    const e = n(this.bakeStep);
    const twoE = e.mul(2);
    const step1 = t1.mul(e);
    const step2 = t2.mul(e);
    // ── coastline warp ──────────────────────────────────────────────────
    //
    // The bake is sampled at a warped direction, not the true one. See
    // COAST_WARP_AMP: this is what turns a texel-scale chain of bilinear arcs
    // into a coastline. It is a fixed function of direction, so it costs one
    // noise evaluation and moves nothing as the camera approaches.
    // One unwarped sample to find out how close to the coast this is, then the
    // real one. The extra fetch is what keeps the warp from moving inland
    // terrain — see COAST_WARP_FADE.
    const probe = sampleBake(dir).r;
    const warp = asVec3(nCall(coastWarp, { dir, radius: n(this.cfg).r, bakeH: probe }));
    const wdir = nNormalize(dir.add(warp));

    const centre = sampleBake(wdir);
    const wpx = sampleBake(nNormalize(wdir.add(step1)));
    const wmx = sampleBake(nNormalize(wdir.sub(step1)));
    const wpy = sampleBake(nNormalize(wdir.add(step2)));
    const wmy = sampleBake(nNormalize(wdir.sub(step2)));
    const gx = wpx.r.sub(wmx.r).div(twoE);
    const gy = wpy.r.sub(wmy.r).div(twoE);
    const bakeGrad = t1.mul(gx).add(t2.mul(gy));

    // Distance to the nearest channel, straight out of the bake — see
    // carveChannels. This used to be reconstructed here from the transverse
    // wetness gradient, four extra taps and a calibration constant, and it was
    // never contiguous: the estimator's own noise broke the river into ponds.
    // A baked distance field interpolates to a smooth sub-texel zero set, so
    // the channel lands in the right place and stays joined up.
    const distAxis = centre.a;

    // Wetness with the flat-ground fan taken out, for the *climate* only —
    // see FLAT_WET_CUT. Everything that decides a surface height reads the raw
    // value and does its own withdrawal inside height_, because vegetation and
    // grass evaluate that function from their own tile data and never come
    // through here. Attenuating on this side alone made the ground the trees
    // stand on a different surface from the ground the terrain draws, and they
    // floated (I3).
    const bakeSlope = n(tslDot(bakeGrad as never, bakeGrad as never)).sqrt().div(n(this.cfg).r);
    const sloped = nStep(FLAT_WET_LO, FLAT_WET_HI, bakeSlope);
    const wet = n(centre.g).sub(n(float(FLAT_WET_CUT)).mul(n(float(1)).sub(sloped)));

    // ── large-scale ambient occlusion ───────────────────────────────────
    //
    // The trace of the Hessian of the baked elevation, from the four taps the
    // gradient already paid for. Negative on a ridge, positive in a valley —
    // which is, to first order, how much sky the point can see. A valley floor
    // is walled in and receives a fraction of the hemisphere; a summit gets all
    // of it. Nothing in the shading knew that: ambient was a constant, so every
    // hollow was lit exactly as brightly as the ridge above it and the terrain
    // read flat wherever the sun was not doing the work.
    //
    // Scaled by the cell size so it is a curvature and not a second difference,
    // and clamped hard — this is a lighting hint, not a measurement.
    const eM2 = n(e).mul(n(this.cfg).r);
    const lapE = wpx.r
      .add(wmx.r)
      .add(wpy.r)
      .add(wmy.r)
      .sub(centre.r.mul(4))
      .div(eM2.mul(eM2));
    const skyView = n(float(1)).sub(lapE.mul(2.6e7).clamp(-0.55, 0.62));

    const baked = nVec4(centre.r, bakeGrad);
    const bake2 = nVec4(centre.g, distAxis, float(0), float(0));

    // Sea-level climate plus the canopy clump, evaluated once per vertex. The
    // fragment stage lapses the temperature and applies the growth gates
    // against its own elevation and slope, so cover is per pixel; only the
    // three-octave clump has to be a varying.
    const clim = asVec3(nCall(patchClimate, { dirSp: dirSp, wet, radius: n(this.cfg).r }));

    // The spectrum of the ground here — (fBm gain, crossover octave) from the
    // biome table. Its own evaluation because both halves of the octave ladder
    // need the same value: the mesh displaces by it below, and the fragment
    // stage continues the ladder with it. Takes the climate above rather than
    // computing a second one, so it costs one noise octave.
    const spec = asVec2(nCall(patchSpectrum, {
      dir: dirSp.xyz, clim: asVec2(clim.xy), bakeH: centre.r,
    }));

    // One expensive evaluation, reused: the same node feeds both the vertex
    // position and the fragment stage, so the noise runs once per vertex.
    const surf = asVec4(nCall(patchSurface, { ...args, baked, bake2, spec }));

    // ── standing water ──────────────────────────────────────────────────
    //
    // The drawn surface is the higher of the ground and the waterline, so
    // lakes, rivers and the sea are all genuinely flat — the ocean in
    // particular used to be the *seabed*, shaded as water and using its own
    // depth as a stand-in for the surface, which is why it had no waterline
    // anywhere except at the coast.
    //
    // Flooding here rather than inside `patchSurface` is what makes the depth
    // available: `patchSurface` has all four channels spoken for, and the
    // depth is the difference between its output and the waterline, so it has
    // to be taken on this side. Costs one small function and no extra noise.
    const waterZ = nCall(patchWater, { bakeH: centre.r, lakeD: centre.b, wet: centre.g, distAxis });
    // Signed, and the sign is the point. The geometry wants the clamped depth
    // — the surface is the higher of ground and waterline — but the fragment
    // stage wants to know how far *below* the waterline the ground is on the
    // dry side too, because that is the only thing that lets it antialias the
    // shoreline. Clamped at zero, every dry pixel reads exactly 0 and there is
    // no gradient to measure the pixel footprint against.
    const signedDepth = waterZ.sub(surf.w);
    const depth = signedDepth.max(float(0));
    const hOut = n(surf.w).add(depth);
    // Under water the surface is flat, so its normal is the planet normal.
    // Blending on depth rather than switching keeps the shoreline continuous
    // across the vertex where the ground crosses the waterline.
    const wetFrac = nStep(0, 0.75, depth as never);
    const surfW = nVec4(nMix(surf.xyz, dir, wetFrac), hOut);
    const position = asVec3(nCall(patchPosition, { ...args, hgt: hOut }));

    const surfV = asVec4(varying(surfW as never, 'vSurf'));
    const climV = asVec4(varying(nVec4(clim, signedDepth) as never, 'vClim'));
    // Sky view was a lone float, which is three wasted channels of an
    // interpolator. It now carries what the fragment stage needs from the bake
    // and could not otherwise reach: the distance to the drainage axis, and the
    // wetness that sets how wide that drainage is. See the river corridor block
    // in shadeTerrain — the tint used to be built by thresholding wetness,
    // which is a broad field stored at 9 km, so the rivers came out as a
    // 20-40 km smudge. A distance field survives the same interpolation with a
    // sharp zero set, which is the whole reason it was baked.
    // .w is the amplification amplitude, the same number height_ scaled its
    // detail by — the fragment stage needs it to continue the octave ladder
    // below the mesh at the right size. Computed by the shared ampAt_, not a
    // second copy of the formula.
    const ampV = nCall(patchAmp, {
      bakeH: centre.r, bakeG: bakeGrad, wet: centre.g, radius: n(this.cfg).r,
    });
    const aoV = varying(nVec4(skyView, distAxis, centre.g, ampV) as never, 'vSky');
    // (quadtree level, vertex spacing, fBm gain, crossover octave). The spacing
    // is what tells the fragment stage where the mesh's own band limit is, so it
    // can add the octaves below it and none of the ones above; the spectrum is
    // what tells it the *shape* of those octaves, which has to be the shape the
    // mesh used or the residual is not the band the mesh is missing. Two
    // channels that were sitting at zero.
    const level = varying(nVec4(iAnchor.w, dirSp.w, spec.x, spec.y) as never, 'vLevel');
    const relPos = varying(position, 'vRel');
    const materialPos = asVec3(relPos.add(this.camSnap));
    const baseLayer = select(
      climV.x.lessThan(0.15),
      int(SNOW_LAYER),
      select(
        climV.y.lessThan(0.28),
        int(SAND_LAYER),
        select(climV.z.greaterThan(0.72), int(LITTER_LAYER), int(SOIL_LAYER)),
      ),
    );
    const triWeight = asVec3(surfV.xyz.abs().pow(4));
    const triNorm = triWeight.x.add(triWeight.y).add(triWeight.z).max(1e-5);
    const tri = (map: TerrainMaterials['albedo'], layer: unknown, pos: Vec3Node) => {
      const tx = asVec4(texture(map, pos.yz.mul(0.18)).depth(layer as never));
      const ty = asVec4(texture(map, pos.xz.mul(0.18)).depth(layer as never));
      const tz = asVec4(texture(map, pos.xy.mul(0.18)).depth(layer as never));
      return asVec4(tx.mul(triWeight.x).add(ty.mul(triWeight.y)).add(tz.mul(triWeight.z)).div(triNorm));
    };
    const baseA = tri(materials.albedo, baseLayer, materialPos);
    const baseB = tri(materials.albedo, baseLayer, asVec3(materialPos.yzx.mul(1.071).add(vec3(19.1, 7.3, 31.7))));
    const rockA = tri(materials.albedo, int(ROCK_LAYER), materialPos);
    const rockB = tri(materials.albedo, int(ROCK_LAYER), asVec3(materialPos.zxy.mul(1.113).add(vec3(43.7, 13.9, 5.1))));
    const baseAlbedo = asVec4(nMix(baseA, baseB, float(0.5)));
    const rockAlbedo = asVec4(nMix(rockA, rockB, float(0.5)));
    const baseNormal = tri(materials.normal, baseLayer, materialPos);
    const rockNormal = tri(materials.normal, int(ROCK_LAYER), materialPos);
    const baseRough = tri(materials.roughness, baseLayer, materialPos);
    const rockRough = tri(materials.roughness, int(ROCK_LAYER), materialPos);

    const material = new MeshBasicNodeMaterial();
    material.positionNode = position;
    material.colorNode = asVec3(
      nCall(shadeTerrain, {
        surf: surfV,
        clim: climV,
        camPos: this.camPos,
        rel: relPos,
        lvl: level,
        sunDir: this.sun,
        sunCol: this.sunCol,
        mode: this.mode,
        grid: this.gridSpacing,
        cfg3: this.cfg3,
        skyView: aoV,
        snap: this.camSnap,
        weather: this.weather,
        matBase: baseAlbedo,
        matRock: rockAlbedo,
        matBaseNR: nVec4(baseNormal.xyz, baseRough.x),
        matRockNR: nVec4(rockNormal.xyz, rockRough.x),
        shadow: shadowFactor ? shadowFactor(relPos) : float(1),
      }),
    );
    this.material = material;

    // Linear distance along the light, written to an R32F target.
    const depthMat = new MeshBasicNodeMaterial();
    depthMat.positionNode = position;
    // Offset so the payload is never negative — see SHADOW_DEPTH_OFFSET.
    depthMat.colorNode = asVec3(
      vec3(tslDot(relPos, this.shadowSun).add(float(SHADOW_DEPTH_OFFSET))),
    );
    this.depthMaterial = depthMat;

    this.mesh = new Mesh(this.geometry, material);
    // Culling is ours (SPEC.md §3); three must not second-guess it, and the
    // mesh has no meaningful bounding volume anyway.
    this.mesh.frustumCulled = false;
    this.mesh.matrixAutoUpdate = false;
  }

  /**
   * Radius of the terrain under the camera. The shader's CDLOD morph must
   * measure distance against the same sphere the CPU selected against.
   */
  setReferenceRadius(r: number): void {
    this.cfg2.value.z = r;
  }

  /** Camera position, for per-pixel local-up in the shader. */
  setCameraPosition(x: number, y: number, z: number): void {
    this.camPos.value.set(x, y, z);
    // The f64 subtraction the near-field lattice rests on. Done here, in JS
    // numbers, because f32 cannot hold 6.4e6 and a quarter metre at once.
    const P = LOCAL_PERIOD;
    this.camSnap.value.set(
      x - Math.round(x / P) * P,
      y - Math.round(y / P) * P,
      z - Math.round(z / P) * P,
    );
  }

  /** Push the current selection to the GPU. */
  update(count: number): void {
    const n = Math.min(count, MAX_PATCHES);
    for (const a of this.attrs) {
      a.needsUpdate = true;
      a.clearUpdateRanges();
      a.addUpdateRange(0, n * a.itemSize);
    }
    this.geometry.instanceCount = n;
  }

  setOctaves(n: number): void {
    this.cfg.value.z = n;
  }
  get octaves(): number {
    return this.cfg.value.z;
  }

  setHeightScale(s: number): void {
    this.cfg.value.y = s;
  }
  get heightScale(): number {
    return this.cfg.value.y;
  }

  /**
   * Cloud coverage and time, so the ground can be shadowed by the deck.
   * Must be the same pair the Clouds material gets or the shadow drifts away
   * from the cloud casting it.
   */
  setClouds(cover: number, timeSec: number): void {
    this.cfg3.value.z = cover;
    this.cfg3.value.w = timeSec;
  }

  setWeather(wetness: number, flash: number): void {
    this.weather.value.x = wetness;
    this.weather.value.y = flash;
  }

  setMode(m: ShadeMode): void {
    this.mode.value = m;
  }
  get shadeMode(): ShadeMode {
    return this.mode.value as ShadeMode;
  }

  setGrid(spacing: number): void {
    this.gridSpacing.value = spacing;
  }
  get grid(): number {
    return this.gridSpacing.value;
  }

  setSun(d: Vector3, colour?: Vector3): void {
    this.sun.value.copy(d).normalize();
    this.shadowSun.value.copy(this.sun.value);
    if (colour) this.sunCol.value.copy(colour);
  }

  setForestDensity(v: number): void {
    this.cfg3.value.x = v;
  }
}
