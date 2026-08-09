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
import { attribute, dot as tslDot, float, uniform, varying, vec3, vec4 } from 'three/tsl';
import {
  DEFAULT_OCTAVES,
  MAX_PATCHES,
  PATCH_SEGS,
  PATCH_VERTS,
  RADIUS,
  FACE_EDGE,
  FOREST_DENSITY,
  SEA_BAND,
  SEA_LEVEL,
} from './planet.js';
import type { PatchBuffers } from './quadtree.js';
import { patchPosition, patchSurface, shadeTerrain } from './shaders/terrain.js';

export type ShadeMode = 0 | 1 | 2 | 3 | 4 | 5;

/** Returns a scalar node in [0,1]: the fraction of sun reaching a point. */
export type ShadowFactor = (rel: unknown) => ReturnType<typeof float>;

// wgslFn is declared as returning an untyped Node, and TSL only attaches
// swizzle/assignment types to nodes with a known type. These recover the
// types the WGSL signatures already guarantee.
type Vec3Node = ReturnType<typeof vec3>;
type Vec4Node = ReturnType<typeof vec4>;
const asVec3 = (n: unknown): Vec3Node => n as Vec3Node;
const asVec4 = (n: unknown): Vec4Node => n as Vec4Node;

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
 * more triangles and removes the failure mode entirely.
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
  /** cfg3 = (forestDensity, planetRadius, —, —) */
  private cfg3 = uniform(new Vector4(FOREST_DENSITY, RADIUS, 0, 0));
  private sun = uniform(new Vector3(0.55, 0.42, 0.72).normalize());
  /** Sun irradiance, not a screen colour — everything is multiplied by it. */
  private sunCol = uniform(new Vector3(1, 0.97, 0.92));
  /** Camera world position in f32 — shading only, never geometry (SPEC.md I4). */
  private camPos = uniform(new Vector3());
  /** Light direction for the depth pass; kept in step with `sun`. */
  private shadowSun = uniform(new Vector3(0, 1, 0));
  private mode = uniform(0);
  private gridSpacing = uniform(0);

  constructor(buffers: PatchBuffers, shadowFactor?: ShadowFactor) {
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

    // One expensive evaluation, reused: the same node feeds both the vertex
    // position and the fragment stage, so the noise runs once per vertex.
    const surf = asVec4(patchSurface(args));
    const position = asVec3(patchPosition({ ...args, hgt: surf.z }));

    const surfV = varying(surf, 'vSurf');
    const level = varying(iAnchor.w, 'vLevel');
    const relPos = varying(position, 'vRel');

    const material = new MeshBasicNodeMaterial();
    material.positionNode = position;
    material.colorNode = asVec3(
      shadeTerrain({
        surf: surfV,
        camPos: this.camPos,
        rel: relPos,
        lvl: level,
        sunDir: this.sun,
        sunCol: this.sunCol,
        mode: this.mode,
        grid: this.gridSpacing,
        cfg3: this.cfg3,
        shadow: shadowFactor ? shadowFactor(relPos) : float(1),
      }),
    );
    this.material = material;

    // Linear distance along the light, written to an R32F target.
    const depthMat = new MeshBasicNodeMaterial();
    depthMat.positionNode = position;
    depthMat.colorNode = asVec3(vec3(tslDot(relPos, this.shadowSun)));
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
