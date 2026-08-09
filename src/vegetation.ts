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

import { BufferAttribute, BufferGeometry, DoubleSide, Mesh, Vector3, Vector4 } from 'three';
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
  DEFAULT_OCTAVES,
  MAX_VEG_TILES,
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
import { billboard, shadeVegetation, vegSample } from './shaders/vegetation.js';

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

/** A unit quad with its origin at the base, so instances sit on the ground. */
function quadGeometry(): BufferGeometry {
  const g = new BufferGeometry();
  g.setAttribute(
    'corner',
    new BufferAttribute(new Float32Array([-0.5, 0, 0.5, 0, 0.5, 1, -0.5, 1]), 2),
  );
  g.setAttribute(
    'quv',
    new BufferAttribute(new Float32Array([0, 0, 1, 0, 1, 1, 0, 1]), 2),
  );
  g.setIndex(new BufferAttribute(new Uint32Array([0, 1, 2, 0, 2, 3]), 1));
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
    new Float32Array(MAX_VEG_TILES * 5 * 4),
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
  private cfg2 = uniform(new Vector4(SEA_LEVEL, SEA_BAND, 0, 0));
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
  private scatterPass: ComputeNode;
  private enabled = true;

  constructor(shadowFactor?: ShadowFactor) {
    // Index count and the two zero fields never change; only instanceCount is
    // touched at runtime, by the GPU.
    for (let b = 0; b < BANDS; b++) {
      (this.argsAttr.array as Uint32Array)[b * ARGS] = 6;
    }

    const tiles = storage(this.tileAttr, 'vec4', MAX_VEG_TILES * 5).toReadOnly();
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
        const base = tile.mul(uint(5));
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
            cell,
            cfg: this.cfg,
            cfg2: this.cfg2,
            vcfg: this.vcfg,
          }),
        );

        If(s.w.greaterThan(0), () => {
          // Band by distance. The camera is at the origin, so the instance
          // position is also the view vector — no subtraction needed.
          const d = s.xyz.length();
          const band = select(
            d.greaterThan(float(VEG_BANDS[0].maxDist)),
            select(d.greaterThan(float(VEG_BANDS[1].maxDist)), uint(2), uint(1)),
            uint(0),
          );

          If(d.lessThan(float(VEG_BANDS[BANDS - 1].maxDist)), () => {
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
      const geo = quadGeometry();
      geo.indirect = this.argsAttr;
      geo.indirectOffset = b * ARGS * 4; // bytes

      const readInsts = storage(this.instAttr, 'vec4', VEG_CAPACITY).toReadOnly();
      const inst = asVec4(readInsts.element(uint(b * VEG_BAND_CAPACITY).add(instanceIndex)));
      const corner = attribute('corner', 'vec2');
      const quv = attribute('quv', 'vec2');

      const mat = new MeshBasicNodeMaterial();
      mat.positionNode = asVec3(
        billboard({ inst, corner, camPos: this.camPos, fadeCfg: this.fadeCfg }),
      );
      const shaded = asVec4(
        shadeVegetation({
          inst: varying(inst, `vInst${b}`),
          uv: varying(quv, `vUv${b}`),
          camPos: this.camPos,
          sunDir: this.sun,
          sunCol: this.sunCol,
          band: float(b),
          mode: this.mode,
          cfg: this.cfg,
          shadow: shadowFactor ? shadowFactor(inst.xyz) : float(1),
        }),
      );
      mat.colorNode = asVec3(shaded.xyz);
      mat.opacityNode = shaded.w;
      // Alpha-to-coverage rather than a hard cutout: MSAA then resolves the
      // crown outline, which is where most of the foliage crawl came from.
      mat.transparent = false;
      mat.alphaToCoverage = true;
      mat.side = DoubleSide;

      // Depth pass: same billboard, distance along the light as the payload.
      const depthMat = new MeshBasicNodeMaterial();
      depthMat.positionNode = mat.positionNode;
      depthMat.colorNode = asVec3(vec3(tslDot(inst.xyz, this.shadowSun)));
      depthMat.side = DoubleSide;
      this.depthMaterials.push(depthMat);

      const mesh = new Mesh(geo, mat);
      mesh.frustumCulled = false;
      mesh.matrixAutoUpdate = false;
      mesh.renderOrder = 1;
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
      return;
    }

    (this.tileAttr.array as Float32Array).set(tileData.subarray(0, n * 20));
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
