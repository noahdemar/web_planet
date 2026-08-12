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
 */

import {
  BufferAttribute,
  BufferGeometry,
  DoubleSide,
  FrontSide,
  Mesh,
  Vector3,
  Vector4,
} from 'three';
import type { ComputeNode, WebGPURenderer } from 'three/webgpu';
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
  select,
  storage,
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
  VEG_BAND_CAPACITY,
  VEG_BANDS,
  VEG_CAPACITY,
  VEG_CELLS,
  VEG_FADE_START,
  VEG_MIN_PIXELS,
  VEG_RANGE,
} from './planet.js';
import {
  billboard,
  shadeTree,
  shadeVegetation,
  treeNormal,
  treeVertex,
  vegSample,
} from './shaders/vegetation.js';

type Vec3Node = ReturnType<typeof vec3>;
type Vec4Node = ReturnType<typeof vec3>['xyzz'];
const asVec3 = (n: unknown): Vec3Node => n as Vec3Node;
const asVec4 = (n: unknown): Vec4Node => n as Vec4Node;

/** Returns a scalar node in [0,1]: the fraction of sun reaching a point. */
export type ShadowFactor = (rel: unknown) => ReturnType<typeof float>;

const BANDS = VEG_BANDS.length;
const CELLS_SQ = VEG_CELLS * VEG_CELLS;
/** Indirect draw args are 5 u32: indexCount, instanceCount, firstIndex, baseVertex, firstInstance. */
const ARGS = 5;

/**
 * Instance geometry, origin at the base so plants sit on the ground.
 *
 * `planes` = 0 gives one camera-facing quad. Any higher count gives that many
 * fixed-orientation quads crossed evenly about the vertical, which is what
 * makes a close tree hold up: a single facing quad has no parallax against its
 * neighbours and slides as the camera moves.
 */
function quadGeometry(planes = 0): BufferGeometry {
  const g = new BufferGeometry();
  const n = Math.max(1, planes);
  const corner: number[] = [];
  const quv: number[] = [];
  const idx: number[] = [];

  for (let p = 0; p < n; p++) {
    // Negative marks the camera-facing case; the shader branches on the sign.
    const angle = planes === 0 ? -1 : (p * Math.PI) / n;
    const base = p * 4;
    corner.push(-0.5, 0, 0.5, 0, 0.5, 1, -0.5, 1);
    quv.push(0, 0, angle, 1, 0, angle, 1, 1, angle, 0, 1, angle);
    idx.push(base, base + 1, base + 2, base, base + 2, base + 3);
  }

  g.setAttribute('corner', new BufferAttribute(new Float32Array(corner), 2));
  g.setAttribute('quv', new BufferAttribute(new Float32Array(quv), 3));
  g.setIndex(new BufferAttribute(new Uint32Array(idx), 1));
  return g;
}

/**
 * A tree, as a surface of revolution the vertex shader gives a shape to.
 *
 * The mesh carries no shape of its own — only (u, θ, isTrunk) per vertex. The
 * radius and height profile is evaluated in the shader from a per-instance
 * species hash, so one geometry covers conifers and broadleaves, every size,
 * and a different crown wobble per tree, with nothing per-instance uploaded
 * beyond the 16 bytes the scatter already writes.
 *
 * Two sleeves: a tapered bole and a canopy that closes to a point at the top.
 * 7 sides is deliberate — an even count lines the silhouette up with itself
 * when trees are seen in rows, and 7 is enough that a crown reads as round at
 * the ranges this band covers.
 */
function treeGeometry(segs: number, trunkRings: number, canopyRings: number): BufferGeometry {
  const g = new BufferGeometry();
  const tsec: number[] = [];
  const idx: number[] = [];
  let base = 0;

  const sleeve = (rings: number, kind: number): void => {
    for (let i = 0; i <= rings; i++) {
      for (let j = 0; j < segs; j++) tsec.push(i / rings, (j / segs) * Math.PI * 2, kind);
    }
    for (let i = 0; i < rings; i++) {
      for (let j = 0; j < segs; j++) {
        const a = base + i * segs + j;
        const b = base + i * segs + ((j + 1) % segs);
        // Wound so the face normal is cross(dθ, du) — outward, matching what
        // `treeNormal` returns. The other order culls every front face and
        // leaves you looking at the inside of the tree.
        idx.push(a, b + segs, a + segs, a, b, b + segs);
      }
    }
    base += (rings + 1) * segs;
  };
  sleeve(trunkRings, 1);
  sleeve(canopyRings, 0);

  g.setAttribute('tsec', new BufferAttribute(new Float32Array(tsec), 3));
  g.setIndex(new BufferAttribute(new Uint32Array(idx), 1));
  return g;
}

export interface VegStats {
  tiles: number;
  perBand: number[];
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

  constructor(shadowFactor?: ShadowFactor) {
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
          // Band by distance. The camera is at the origin, so the instance
          // position is also the view vector — no subtraction needed.
          const d = s.xyz.length();
          // Built from VEG_BANDS rather than written out, so adding a band is
          // a one-line change here instead of a nested select nobody notices
          // is still three deep.
          let bandN: unknown = uint(BANDS - 1);
          for (let k = BANDS - 2; k >= 0; k--) {
            bandN = select(d.greaterThan(float(VEG_BANDS[k].maxDist)), bandN as never, uint(k));
          }
          const band = bandN as ReturnType<typeof uint>;

          // Cull by projected size, not only by distance.
          //
          // The billboard already fades an instance out below
          // VEG_MIN_PIXELS * 0.35 — it is stored, binned, drawn and then
          // multiplied by zero. Rejecting it here instead is free visually and
          // is what pays for a longer range: sizes follow an inverse-J
          // distribution, so past a few hundred metres the overwhelming
          // majority of instances are saplings contributing nothing, while the
          // handful of large trees are exactly what you see across open water.
          const px = sScale.mul(this.fadeCfg.z).div(d.max(float(1)));
          If(
            d.lessThan(float(VEG_BANDS[BANDS - 1].maxDist))
              .and(px.greaterThan(float(VEG_MIN_PIXELS * 0.35))),
            () => {
            const slot = atomicAdd(counters.element(band.mul(ARGS).add(1)), uint(1));
            // Equal band capacities make the base plain arithmetic.
            If(slot.lessThan(uint(VEG_BAND_CAPACITY)), () => {
              insts.element(band.mul(uint(VEG_BAND_CAPACITY)).add(slot)).assign(s);
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
      // of binning them (SPEC.md §8). The near band is real geometry — a bole
      // and a crown, ~150 triangles; the mid band is crossed quads, which have
      // enough parallax to hold up at 45–220 m; the far band is one
      // camera-facing quad.
      const isTree = b === 0;
      const geo = isTree ? treeGeometry(7, 3, 8) : quadGeometry(b === 1 ? 3 : 0);
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
      const inst = asVec4(readInsts.element(uint(b * VEG_BAND_CAPACITY).add(instanceIndex)));

      const mat = new MeshBasicNodeMaterial();
      let shaded: Vec4Node;
      if (isTree) {
        const tsec = asVec3(attribute('tsec', 'vec3'));
        mat.positionNode = asVec3(
          treeVertex({ inst, tsec, camPos: this.camPos, fadeCfg: this.fadeCfg }),
        );
        const nrm = asVec3(treeNormal({ inst, tsec, camPos: this.camPos }));
        shaded = asVec4(
          shadeTree({
            inst: varying(inst, `vInst${b}`),
            nrm: varying(nrm, `vNrm${b}`),
            kind: varying(tsec.z, `vKind${b}`),
            uu: varying(tsec.x, `vU${b}`),
            camPos: this.camPos,
            sunDir: this.sun,
            sunCol: this.sunCol,
            mode: this.mode,
            cfg: this.cfg,
            shadow: shadowFactor ? shadowFactor(inst.xyz) : float(1),
          }),
        );
      } else {
        const corner = attribute('corner', 'vec2');
        const quv = attribute('quv', 'vec3');
        mat.positionNode = asVec3(
          billboard({
            inst,
            corner,
            camPos: this.camPos,
            fadeCfg: this.fadeCfg,
            plane: asVec3(quv).z,
          }),
        );
        shaded = asVec4(
          shadeVegetation({
            inst: varying(inst, `vInst${b}`),
            uv: asVec3(varying(quv, `vUv${b}`)).xy,
            camPos: this.camPos,
            sunDir: this.sun,
            sunCol: this.sunCol,
            band: float(b),
            mode: this.mode,
            cfg: this.cfg,
            shadow: shadowFactor ? shadowFactor(inst.xyz) : float(1),
          }),
        );
      }
      mat.colorNode = asVec3(shaded.xyz);
      mat.opacityNode = shaded.w;
      // Alpha-to-coverage rather than a hard cutout: MSAA then resolves the
      // crown outline, which is where most of the foliage crawl came from.
      // The tree band is solid geometry with alpha 1, so it does not need it —
      // and must not have it, or MSAA dithers a surface that has no edge.
      mat.transparent = false;
      mat.alphaToCoverage = !isTree;
      // Real trees are closed surfaces and want backface culling; the quads
      // are single-sided sheets and would vanish from behind.
      mat.side = isTree ? FrontSide : DoubleSide;

      // Depth pass: same vertex path, distance along the light as the payload.
      const depthMat = new MeshBasicNodeMaterial();
      depthMat.positionNode = mat.positionNode;
      depthMat.colorNode = asVec3(vec3(tslDot(inst.xyz, this.shadowSun)));
      depthMat.side = isTree ? FrontSide : DoubleSide;
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
    let total = 0;
    let overflow = false;
    for (let b = 0; b < BANDS; b++) {
      const raw = u[b * ARGS + 1];
      const c = Math.min(raw, VEG_BAND_CAPACITY);
      if (raw > VEG_BAND_CAPACITY) overflow = true;
      out.push(c);
      total += c;
    }
    this.stats.perBand = out;
    this.stats.total = total;
    this.stats.overflow = overflow;
    return out;
  }
}
