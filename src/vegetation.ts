/**
 * GPU-driven vegetation. See SPEC.md §8.
 *
 * This is the substrate people actually mean when they say "Nanite": the CPU
 * never touches an instance. Per frame it uploads ~24 tiles × 20 floats and
 * dispatches; everything else — placement, gating, LOD binning, draw counts —
 * happens on the GPU and is consumed by indirect draws that read their instance
 * count out of a buffer the compute pass wrote.
 *
 *   reset    zero the three instance counters
 *   scatter  one thread per candidate cell → accepted instances, binned by band
 *   draw     one indirect draw per band, count supplied by the GPU
 *
 * The cost is O(cells), independent of how many instances survive, and there is
 * no readback anywhere — so frame time does not depend on forest density in the
 * way a CPU-culled system would.
 *
 * LOD here changes *representation*, not triangle count. Decimating aggregate
 * geometry deletes leaves and thins the canopy, which is exactly why Nanite
 * needed a separate voxel path for foliage (SPEC.md §8).
 *
 * What an instance *is* comes from treeAssets.ts now: scanned tree models near
 * the camera, and impostors baked from those same models further out. The
 * substrate is unchanged — the scatter still writes 16 bytes per instance and
 * the CPU still never touches one — but binning gained a second axis. A band
 * is one indirect draw, an indirect draw is one geometry, and a geometry is
 * one species, so the two geometry distances are each split conifer/broadleaf
 * and the scatter picks the band from distance *and* the instance's species.
 */

import {
  BufferAttribute,
  BufferGeometry,
  DoubleSide,
  Mesh,
  Vector3,
  Vector4,
} from 'three';
import type { ComputeNode, WebGPURenderer } from 'three/webgpu';
import { SHADOW_DEPTH_OFFSET } from './shadows.js';
import {
  IndirectStorageBufferAttribute,
  MeshBasicNodeMaterial,
  StorageBufferAttribute,
} from 'three/webgpu';
import {
  Fn,
  If,
  attribute,
  atomicAdd,
  atomicStore,
  compute,
  dot as tslDot,
  float,
  instanceIndex,
  int,
  select,
  storage,
  texture as tslTextureNode,
  uint,
  uniform,
  varying,
  vec2,
  vec3,
} from 'three/tsl';
import {
  DEFAULT_LOD_FACTOR,
  DEFAULT_OCTAVES,
  MAX_VEG_TILES,
  FACE_EDGE,
  PATCH_SEGS,
  VEG_TILE_VEC4,
  RADIUS,
  SEA_BAND,
  SEA_LEVEL,
  FOREST_DENSITY,
  VEG_BAND_BASE,
  VEG_BAND_SLOTS,
  VEG_BANDS,
  VEG_CAPACITY,
  VEG_DIST_BANDS,
  VEG_MODEL_BANDS,
  VEG_MODEL_LODS,
  VEG_SPECIES,
  VEG_CELLS,
  VEG_FADE_START,
  VEG_MIN_PIXELS,
  VEG_RANGE,
} from './planet.js';
import { vegSample } from './shaders/vegetation.js';
import {
  impostorUV,
  impostorVertex,
  instSpecies,
  modelNormal,
  modelVertex,
  shadeImpostor,
  shadeTreeModel,
} from './shaders/treeModel.js';
import type { TreeAssets } from './treeAssets.js';

type Vec3Node = ReturnType<typeof vec3>;
type Vec4Node = ReturnType<typeof vec3>['xyzz'];
const asVec3 = (n: unknown): Vec3Node => n as Vec3Node;
const asVec4 = (n: unknown): Vec4Node => n as Vec4Node;
type UintNode = ReturnType<typeof uint>;
const asUint = (n: unknown): UintNode => n as UintNode;
type FloatNode = ReturnType<typeof float>;
const asFloat = (n: unknown): FloatNode => n as FloatNode;

/** Returns a scalar node in [0,1]: the fraction of sun reaching a point. */
export type ShadowFactor = (rel: unknown) => ReturnType<typeof float>;

const BANDS = VEG_BANDS.length;
const DISTS = VEG_DIST_BANDS.length;
const CELLS_SQ = VEG_CELLS * VEG_CELLS;
/** Indirect draw args are 5 u32: indexCount, instanceCount, firstIndex, baseVertex, firstInstance. */
const ARGS = 5;

/**
 * The impostor quad.
 *
 * One camera-facing quad and nothing else. The crossed-plane variant this used
 * to carry — three fixed quads about the vertical, so a close tree had real
 * parallax — is gone with the band it served: the mid band draws the model
 * itself now, and crossing three copies of a *photograph* of a tree gives you
 * three visibly intersecting cut-outs rather than depth.
 *
 * `corner` is the position in the baked box, x in [-0.5, 0.5] and y in [0, 1];
 * `quv` is the same span as a texture coordinate, which the shading uses for
 * its occlusion and form terms. The atlas lookup is not this uv — it depends
 * on the view direction, and impostorUV builds it.
 */
function quadGeometry(): BufferGeometry {
  const g = new BufferGeometry();
  const corner = [-0.5, 0, 0.5, 0, 0.5, 1, -0.5, 1];
  const quv = [0, 0, 1, 0, 1, 1, 0, 1];
  const idx = [0, 1, 2, 0, 2, 3];

  g.setAttribute('corner', new BufferAttribute(new Float32Array(corner), 2));
  g.setAttribute('quv', new BufferAttribute(new Float32Array(quv), 2));
  g.setIndex(new BufferAttribute(new Uint32Array(idx), 1));
  return g;
}

export interface VegStats {
  tiles: number;
  perBand: number[];
  /** What the scatter *wanted* per band, before the capacity clamp. */
  rawPerBand: number[];
  total: number;
  candidates: number;
  overflow: boolean;
}

export class Vegetation {
  readonly meshes: Mesh[] = [];
  /** One per band; same vertex path as the display material. */
  readonly depthMaterials: MeshBasicNodeMaterial[] = [];
  readonly stats: VegStats = {
    tiles: 0,
    perBand: new Array(BANDS).fill(0),
    rawPerBand: new Array(BANDS).fill(0),
    total: 0,
    candidates: 0,
    overflow: false,
  };

  /** Tile parameters: five vec4 per tile, rewritten by the CPU each frame. */
  private tileAttr = new StorageBufferAttribute(
    new Float32Array(MAX_VEG_TILES * VEG_TILE_VEC4 * 4),
    4,
  );

  /**
   * Instances: one vec4 each — (posRel.xyz, scale). Nothing else is stored.
   * The band is implied by which region of the buffer an instance lives in,
   * and everything else the shader needs (hue, crown variation) is hashed from
   * the position, so the record stays 16 bytes.
   */
  private instAttr = new StorageBufferAttribute(
    new Float32Array(VEG_CAPACITY * 4),
    4,
  );

  /**
   * Draw arguments, one set per band. Also the atomic counters the scatter
   * increments: slot 1 of each set *is* `instanceCount`, so the compute pass
   * writes the draw call directly and nothing is ever read back.
   */
  private argsAttr = new IndirectStorageBufferAttribute(
    new Uint32Array(BANDS * ARGS),
    1,
  );

  private cfg = uniform(new Vector4(RADIUS, 1, DEFAULT_OCTAVES, 0));
  /** (seaLevel, seaBand, faceEdge/segments, lodFactor) */
  private cfg2 = uniform(new Vector4(SEA_LEVEL, SEA_BAND, FACE_EDGE / PATCH_SEGS, DEFAULT_LOD_FACTOR));
  /** (cells, density, seed, tileCount) */
  private vcfg = uniform(new Vector4(VEG_CELLS, FOREST_DENSITY, 17, 0));
  /** (fadeStart, fadeEnd, pixelsPerRadian, minPixels) */
  private fadeCfg = uniform(
    new Vector4(VEG_FADE_START, VEG_RANGE, 1000, VEG_MIN_PIXELS),
  );
  private sunCol = uniform(new Vector3(1, 0.97, 0.92));
  private camPos = uniform(new Vector3());
  private sun = uniform(new Vector3(0.62, 0.28, 0.73).normalize());
  private shadowSun = uniform(new Vector3(0, 1, 0));
  private mode = uniform(0);

  private resetPass: ComputeNode;
  /** Whether the GPU's indirect instance counts are non-zero. See `update`. */
  private liveCounts = false;
  private scatterPass: ComputeNode;
  private enabled = true;

  /**
   * @param assets Models and the baked impostor atlas. Required, and taken as
   *   a constructor argument rather than loaded here, because the geometry and
   *   the textures are compiled into the node materials below — there is no
   *   point at which this class could swap them afterwards.
   */
  constructor(assets: TreeAssets, shadowFactor?: ShadowFactor) {
    const tiles = storage(this.tileAttr, 'vec4', MAX_VEG_TILES * VEG_TILE_VEC4).toReadOnly();
    const insts = storage(this.instAttr, 'vec4', VEG_CAPACITY);
    const counters = storage(this.argsAttr, 'uint', BANDS * ARGS).toAtomic();

    this.resetPass = compute(
      Fn(() => {
        atomicStore(counters.element(instanceIndex.mul(ARGS).add(1)), uint(0));
      })() as never,
      BANDS,
    );

    this.scatterPass = compute(
      Fn(() => {
      // No bounds guard: the dispatch is exactly tileCount * CELLS_SQ, and
      // CELLS_SQ is a multiple of the workgroup size, so there is no rounding
      // tail to guard against. A guard here would also have been wrong — it
      // would have to read the live uniform, not a JS value captured when the
      // kernel was built.
      const tile = instanceIndex.div(uint(CELLS_SQ));
      const cellId = instanceIndex.mod(uint(CELLS_SQ));

      {
        const base = tile.mul(uint(VEG_TILE_VEC4));
        const cell = vec2(
          float(cellId.mod(uint(VEG_CELLS))),
          float(cellId.div(uint(VEG_CELLS))),
        );

        const s = asVec4(
          vegSample({
            t0: tiles.element(base),
            t1: tiles.element(base.add(1)),
            t2: tiles.element(base.add(2)),
            t3: tiles.element(base.add(3)),
            t4: tiles.element(base.add(4)),
            t5: tiles.element(base.add(5)),
            t6: tiles.element(base.add(6)),
            t7: tiles.element(base.add(7)),
            cell,
            cfg: this.cfg,
            cfg2: this.cfg2,
            vcfg: this.vcfg,
          }),
        );

        // Scale is packed with the per-instance identity (packInst_ in
        // shaders/vegetation.ts), so it has to be decoded before it means
        // metres. Rejected candidates come back negative and floor keeps them
        // negative.
        const sScale = s.w.floor().mul(0.125);
        If(sScale.greaterThan(0), () => {
          // Bin by distance, then by species. The camera is at the origin, so
          // the instance position is also the view vector — no subtraction.
          const d = s.xyz.length();
          // Built from VEG_DIST_BANDS rather than written out, so adding a
          // distance is a one-line change here instead of a nested select
          // nobody notices is still three deep.
          let distN: unknown = uint(DISTS - 1);
          for (let k = DISTS - 2; k >= 0; k--) {
            distN = select(d.greaterThan(float(VEG_DIST_BANDS[k])), distN as never, uint(k));
          }
          const dist = asUint(distN);

          // The species split, and why it lives here rather than in the draw.
          //
          // A band is one indirect draw, one draw is one geometry, and a
          // scanned conifer and a scanned broadleaf are not the same geometry.
          // The alternative — merging every species into one mesh and
          // collapsing the vertices of the wrong one to a point — costs the
          // whole vertex count of every species on every instance, which for
          // two models is a doubling of the near band for nothing.
          //
          // So the geometry distances each become two bands, and the impostor
          // distances stay one apiece: the species there picks a *row of the
          // atlas*, which costs nothing, not a draw.
          const sp = asUint(asFloat(instSpecies({ inst: s })).toUint());
          const band = asUint(
            select(
              dist.lessThan(uint(VEG_MODEL_LODS)) as never,
              dist.mul(uint(VEG_SPECIES)).add(sp) as never,
              dist.add(uint(VEG_MODEL_BANDS - VEG_MODEL_LODS)) as never,
            ),
          );

          const px = sScale.mul(this.fadeCfg.z).div(d.max(float(1)));
          If(
            d.lessThan(float(VEG_RANGE))
              .and(px.greaterThan(float(VEG_MIN_PIXELS * 0.35))),
            () => {
            const slot = atomicAdd(counters.element(band.mul(ARGS).add(1)), uint(1));
            // Bands have their own sizes now, so the base and the ceiling are
            // lookups rather than arithmetic. Built as select chains over the
            // compile-time table for the same reason the distance binning is:
            // six entries resolve to five selects, which is cheaper than a
            // buffer read and keeps the table the single source of truth.
            let baseN: unknown = uint(VEG_BAND_BASE[BANDS - 1]);
            let capN: unknown = uint(VEG_BAND_SLOTS[BANDS - 1]);
            for (let k = BANDS - 2; k >= 0; k--) {
              const hit = band.equal(uint(k)) as never;
              baseN = select(hit, uint(VEG_BAND_BASE[k]) as never, baseN as never);
              capN = select(hit, uint(VEG_BAND_SLOTS[k]) as never, capN as never);
            }
            If(slot.lessThan(asUint(capN)), () => {
              insts.element(asUint(baseN).add(slot)).assign(s);
            });
          });
        });
      }
      })() as never,
      MAX_VEG_TILES * CELLS_SQ,
    );

    // One mesh per band. Each reads the shared instance buffer at its own base
    // and draws with its own slice of the indirect args, so no band needs the
    // `indirect-first-instance` feature.
    for (let b = 0; b < BANDS; b++) {
      // What an instance *is* changes with the band, which is the whole point
      // of binning them (SPEC.md §8). The two nearest distances draw the
      // scanned model — its near LOD out to 45 m, its low LOD to 110 m — and
      // everything past that is an impostor quad carrying a baked photograph
      // of the same model from the nearest of eight yaws.
      //
      // The geometry distances are shorter than the 130 m the procedural tree
      // held, and that is the honest cost of real models: 512 triangles became
      // 2.2–4.8k. What buys it back is that the thing on the far side of the
      // handover is no longer a different tree. The old boundary swapped a
      // modelled crown for a procedural silhouette that agreed with it only in
      // gross proportion; this one swaps a model for a picture of itself.
      const spec = VEG_BANDS[b];
      const geo = spec.model
        ? assets.geometries[spec.lod][spec.species]
        : quadGeometry();
      geo.indirect = this.argsAttr;
      geo.indirectOffset = b * ARGS * 4; // bytes

      // indexCount, taken from the geometry rather than written by hand.
      //
      // It used to be the literal `b === 0 ? 18 : 6` — correct while every band
      // was quads, and silently wrong the moment the near band became a 462
      // index tree: the indirect draw kept asking for 18 indices, so six
      // triangles of the bole rasterised and the entire canopy sleeve never
      // reached the raster at all. The forest looked like a field of bare
      // stumps and nothing about the geometry, winding or shader was at fault.
      // Only instanceCount is written at runtime, by the compute pass.
      (this.argsAttr.array as Uint32Array)[b * ARGS] = geo.getIndex()!.count;

      const readInsts = storage(this.instAttr, 'vec4', VEG_CAPACITY).toReadOnly();
      const inst = asVec4(readInsts.element(uint(VEG_BAND_BASE[b]).add(instanceIndex)));
      /**
       * The instance record, hoisted into a varying for everything downstream
       * of the vertex shader.
       *
       * `inst` is a storage read indexed by `instanceIndex`, and that index
       * only exists while the vertex stage is running. Anything evaluated per
       * fragment has to go through this, not through `inst` — reading the raw
       * node there compiles and runs and returns a different record for every
       * pixel, which is how the impostor lookup came to smear a whole row of
       * the atlas across each quad instead of picking one tile out of it.
       */
      const vInst = asVec4(varying(inst, `vInst${b}`));
      const shadow = shadowFactor ? shadowFactor(inst.xyz) : float(1);

      const mat = new MeshBasicNodeMaterial();
      let shaded: Vec4Node;
      /** Base colour and coverage, whatever the band reads it from. */
      let texel: Vec4Node;
      if (spec.model) {
        const pos = asVec3(attribute('position', 'vec3'));
        const nrmA = asVec3(attribute('normal', 'vec3'));
        const leaf = attribute('leaf', 'float');
        // One sampler for bark and leaves both. The layer is a vertex
        // attribute rather than a uniform because a band draws every material
        // of its model in the same pass — see treeAssets.ts.
        texel = asVec4(
          tslTextureNode(assets.albedo, attribute('uv', 'vec2')).depth(
            int(attribute('layer', 'float') as never) as never,
          ),
        );
        mat.positionNode = asVec3(modelVertex({ inst, pos, camPos: this.camPos }));
        const nrm = asVec3(modelNormal({ inst, nrm: nrmA, camPos: this.camPos }));
        shaded = asVec4(
          shadeTreeModel({
            inst: vInst,
            nrm: varying(nrm, `vNrm${b}`),
            texel,
            leaf: varying(leaf, `vLeaf${b}`),
            // Model space is unit height with the base at zero, so the y of
            // the untransformed vertex *is* the fraction of the way up the
            // tree the occlusion term wants.
            hUp: varying(pos.y, `vUp${b}`),
            camPos: this.camPos,
            sunDir: this.sun,
            sunCol: this.sunCol,
            mode: this.mode,
            cfg: this.cfg,
            shadow,
          }),
        );
      } else {
        const corner = attribute('corner', 'vec2');
        const quv = attribute('quv', 'vec2');
        mat.positionNode = asVec3(
          impostorVertex({
            inst,
            corner,
            camPos: this.camPos,
            fadeCfg: this.fadeCfg,
          }),
        );
        // The atlas lookup has to be built in the vertex shader and carried
        // across: it selects a *tile*, and a tile boundary interpolated per
        // fragment would bleed the neighbouring yaw in along one edge.
        // Built from the varying record, so the tile is a property of the
        // instance rather than of the pixel.
        const auv = impostorUV({ inst: vInst, corner, camPos: this.camPos });
        texel = asVec4(tslTextureNode(assets.impostor, auv));
        shaded = asVec4(
          shadeImpostor({
            inst: vInst,
            texel,
            uv: varying(quv, `vUv${b}`),
            band: float(b),
            camPos: this.camPos,
            sunDir: this.sun,
            sunCol: this.sunCol,
            mode: this.mode,
            cfg: this.cfg,
            shadow,
          }),
        );
      }
      mat.colorNode = asVec3(shaded.xyz);
      mat.opacityNode = shaded.w;
      // Alpha-to-coverage rather than a hard cutout: MSAA then resolves the
      // outline, which is where most of the foliage crawl came from.
      //
      // Every band wants it now. It used to be off for the near band on the
      // grounds that a procedural tree is a closed surface with alpha 1 and
      // dithering it would only add noise — but a scanned tree is not a closed
      // surface. It is a bole plus a few hundred alpha-cut leaf cards, and
      // those cards are exactly the geometry the argument was protecting.
      mat.transparent = false;
      mat.alphaToCoverage = true;
      // No alphaTest on top of it. The shading already resolves the cutout —
      // it sharpens the scan's soft alpha to a one-pixel edge and discards
      // what falls outside — and a fixed test applied to *that* would clip the
      // antialiased edge back off, which is the whole thing coverage is here
      // to keep. The depth pass is the exception and does want a hard test;
      // see below.
      // Leaf cards are single-sided sheets and would vanish from behind, and
      // the models mark themselves double-sided for that reason. The boles pay
      // for it too; that is cheaper than a second draw to separate them.
      mat.side = DoubleSide;

      // Depth pass: same vertex path, distance along the light as the payload.
      const depthMat = new MeshBasicNodeMaterial();
      depthMat.positionNode = mat.positionNode;
      // Offset so the payload is never negative — see SHADOW_DEPTH_OFFSET.
      depthMat.colorNode = asVec3(
        vec3(tslDot(inst.xyz, this.shadowSun).add(float(SHADOW_DEPTH_OFFSET))),
      );
      // Leaf cards have to cut out in the depth pass too, or every tree casts
      // the shadow of a solid box and the ground under a canopy goes flat
      // black instead of dappled.
      depthMat.opacityNode = texel.w;
      depthMat.alphaTest = 0.4;
      depthMat.side = DoubleSide;
      this.depthMaterials.push(depthMat);

      const mesh = new Mesh(geo, mat);
      mesh.frustumCulled = false;
      mesh.matrixAutoUpdate = false;
      // Front to back, so early-Z kills the far band where near trees already
      // cover the pixel. Overdraw, not triangles, is what this costs.
      mesh.renderOrder = 1 + b;
      this.meshes.push(mesh);
    }
  }

  setEnabled(v: boolean): void {
    this.enabled = v;
    for (const m of this.meshes) m.visible = v;
  }
  get isEnabled(): boolean {
    return this.enabled;
  }

  setDensity(d: number): void {
    this.vcfg.value.y = Math.max(0.02, Math.min(1, d));
  }
  get density(): number {
    return this.vcfg.value.y;
  }

  setOctaves(n: number): void {
    this.cfg.value.z = n;
  }

  /**
   * The selector's LOD factor. The scatter needs it to work out the vertex
   * spacing the terrain will be drawn at under each stem, which is what it
   * band-limits the ground to — see the placement block in vegSample. Pushed
   * every frame rather than at construction because `[` and `]` change it.
   */
  setLodFactor(f: number): void {
    this.cfg2.value.w = f;
  }
  setMode(m: number): void {
    this.mode.value = m;
  }
  get debugBands(): boolean {
    return this.mode.value > 0.5;
  }

  setCamera(x: number, y: number, z: number): void {
    this.camPos.value.set(x, y, z);
  }

  setSun(d: Vector3, colour?: Vector3): void {
    this.sun.value.copy(d).normalize();
    this.shadowSun.value.copy(this.sun.value);
    if (colour) this.sunCol.value.copy(colour);
  }

  /**
   * Pixels an object one metre tall subtends at one metre. Drives the
   * sub-pixel fade, so it has to follow the viewport and field of view.
   */
  setProjectionScale(pxPerRadian: number): void {
    this.fadeCfg.value.z = pxPerRadian;
  }

  /**
   * Upload this frame's tiles and run the GPU passes.
   * Returns without dispatching when nothing is in range.
   */
  update(renderer: WebGPURenderer, tileData: Float32Array, tileCount: number): void {
    const n = Math.min(tileCount, MAX_VEG_TILES);
    this.stats.tiles = n;
    this.stats.candidates = n * CELLS_SQ;
    if (!this.enabled || n === 0) {
      this.stats.total = 0;
      this.stats.perBand.fill(0);
      // Zeroing the *stats* is not zeroing the draw. The instance counts the
      // bands draw with live in a GPU buffer that only `resetPass` clears, so
      // returning here left the last populated frame's counts standing and the
      // indirect draws kept issuing them — with no tiles, and nothing to
      // overwrite the instance buffer.
      //
      // It is invisible near the ground because the stale set is roughly the
      // live one. Climb past ~1 km, where no tile is in range, and it becomes
      // very visible: instance positions are camera-relative, so last frame's
      // offsets get reinterpreted about the new camera, and the billboards
      // face the viewer — which from directly overhead orients every one of
      // them radially and paints concentric rings around the nadir.
      //
      // Only on the transition. Dispatching three threads every frame from
      // orbit would be harmless but pointless.
      if (this.liveCounts) {
        renderer.compute(this.resetPass, BANDS);
        this.liveCounts = false;
      }
      return;
    }
    this.liveCounts = true;

    (this.tileAttr.array as Float32Array).set(tileData.subarray(0, n * VEG_TILE_VEC4 * 4));
    this.tileAttr.needsUpdate = true;
    this.vcfg.value.w = n;

    renderer.compute(this.resetPass, BANDS);
    renderer.compute(this.scatterPass, n * CELLS_SQ);
  }

  /**
   * Read the per-band instance counts back. Costs a GPU sync, so this is a
   * diagnostic — the render path never does it.
   */
  async readCounts(renderer: WebGPURenderer): Promise<number[]> {
    const buf = await renderer.getArrayBufferAsync(this.argsAttr);
    const u = new Uint32Array(buf);
    const out: number[] = [];
    const rawOut: number[] = [];
    let total = 0;
    let overflow = false;
    for (let b = 0; b < BANDS; b++) {
      const raw = u[b * ARGS + 1];
      const c = Math.min(raw, VEG_BAND_SLOTS[b]);
      if (raw > VEG_BAND_SLOTS[b]) overflow = true;
      out.push(c);
      rawOut.push(raw);
      total += c;
    }
    this.stats.rawPerBand = rawOut;
    this.stats.perBand = out;
    this.stats.total = total;
    this.stats.overflow = overflow;
    return out;
  }
}
