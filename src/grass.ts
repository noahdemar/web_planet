/**
 * Ground clutter: grass. See SPEC.md §8 and shaders/grass.ts.
 *
 * Deliberately *not* a fourth vegetation band. The tree scatter walks resident
 * quadtree tiles and puts one candidate per tile cell, which at the tile sizes
 * that reach the camera is metres apart — three orders too coarse for a sward.
 * Grass needs its own grid, and once it has one it needs nothing else the tree
 * system provides: no bands, no billboards, no per-tile records. So it is its
 * own pass, and it cannot break the forest.
 *
 * One compute dispatch of GRASS_GRID² threads writes an append buffer; one
 * indirect draw renders it. The same shape as `Vegetation`, a third the code.
 */

import { BufferGeometry, BufferAttribute, DoubleSide, Mesh, Vector3, Vector4 } from 'three';
import {
  IndirectStorageBufferAttribute,
  MeshBasicNodeMaterial,
  StorageBufferAttribute,
  type WebGPURenderer,
} from 'three/webgpu';
import {
  Fn,
  atomicAdd,
  atomicStore,
  attribute,
  compute,
  float,
  instanceIndex,
  storage,
  uint,
  uniform,
  varying,
  vec3,
  vec4,
  If,
} from 'three/tsl';
import {
  FACE_EDGE,
  GRASS_CAPACITY,
  GRASS_GRID,
  GRASS_RANGE,
  GRASS_SEGMENTS,
  GRASS_SPACING,
  RADIUS,
} from './planet.js';
import { directionToFace } from './cubesphere.js';
import { sampleSurface, type PlanetSurface } from './planetData.js';
import { warpForCoast } from './heightCPU.js';
import { grassSample, grassVertex, shadeGrass } from './shaders/grass.js';

type N = ReturnType<typeof float>;
const asVec3 = (x: unknown): ReturnType<typeof vec3> => x as ReturnType<typeof vec3>;
const asVec4 = (x: unknown): ReturnType<typeof vec4> => x as ReturnType<typeof vec4>;
const call = (f: unknown, a: Record<string, unknown>): N =>
  (f as (x: Record<string, unknown>) => N)(a);

/** Indirect draw arguments: indexCount, instanceCount, firstIndex, baseVertex, firstInstance. */
const ARGS = 5;

export type ShadowFactor = (rel: unknown) => N;

/**
 * One blade: a strip of GRASS_SEGMENTS quads narrowing to a triangle tip.
 *
 * `bseg` is (t along the blade, side). The tip vertex carries side 0 so the
 * strip closes to a point rather than to a short edge — a blunt tip is
 * surprisingly visible in a field, because every blade has one.
 */
function bladeGeometry(segments: number): BufferGeometry {
  const seg: number[] = [];
  const idx: number[] = [];
  const offsets = [
    [-0.14, -0.11], [0.15, -0.07], [-0.08, 0.16], [0.13, 0.14],
  ];
  for (const [ox, oy] of offsets) {
    const base = seg.length / 4;
    for (let i = 0; i < segments; i++) {
      const t = i / segments;
      seg.push(t, -1, ox, oy, t, 1, ox, oy);
    }
    seg.push(1, 0, ox, oy);
    for (let i = 0; i < segments - 1; i++) {
      const a = base + i * 2;
      idx.push(a, a + 1, a + 3, a, a + 3, a + 2);
    }
    const last = base + (segments - 1) * 2;
    idx.push(last, last + 1, base + segments * 2);
  }

  const g = new BufferGeometry();
  g.setAttribute('bseg', new BufferAttribute(new Float32Array(seg), 4));
  g.setIndex(idx);
  return g;
}

export class Grass {
  readonly mesh: Mesh;

  private instAttr = new StorageBufferAttribute(new Float32Array(GRASS_CAPACITY * 4), 4);
  private argsAttr = new IndirectStorageBufferAttribute(new Uint32Array(ARGS), ARGS);

  /** (radius, heightScale, octaves, density) */
  private cfg = uniform(new Vector4(RADIUS, 1, 9, 1));
  /** (gridOriginX, gridOriginY, camFracX, camFracY) */
  private cfg2 = uniform(new Vector4());
  /**
   * (forwardEast, forwardNorth, cos of the culling half-angle, —).
   *
   * The camera's forward projected into this frame's tangent basis, so the
   * scatter can reject candidates outside the view with a dot product. See the
   * rejection block in shaders/grass.ts.
   */
  private view = uniform(new Vector4(0, 1, -1, 0));
  /** (radius, —, time, —) for the draw. */
  private dcfg = uniform(new Vector4(RADIUS, 0, 0, 0));
  /** Bake at the camera: (elevation, gradient). */
  private bake = uniform(new Vector4());
  /** (wetness, lakeDepth, distToAxis, —) */
  private bake2 = uniform(new Vector4());
  private camDir = uniform(new Vector3(0, 1, 0));
  private camEast = uniform(new Vector3(1, 0, 0));
  private camNorth = uniform(new Vector3(0, 0, 1));
  private alt = uniform(0);
  private camPos = uniform(new Vector3());
  private sun = uniform(new Vector3(0.62, 0.28, 0.73).normalize());
  private sunCol = uniform(new Vector3(1, 0.97, 0.92));

  private resetPass: ReturnType<typeof compute>;
  private scatterPass: ReturnType<typeof compute>;
  private enabled = true;
  private live = false;

  constructor(shadowFactor?: ShadowFactor) {
    const insts = storage(this.instAttr, 'vec4', GRASS_CAPACITY);
    const counter = storage(this.argsAttr, 'uint', ARGS).toAtomic();

    this.resetPass = compute(
      Fn(() => {
        atomicStore(counter.element(uint(1)), uint(0));
      })() as never,
      1,
    );

    this.scatterPass = compute(
      Fn(() => {
        const cx = instanceIndex.mod(uint(GRASS_GRID)).toFloat();
        const cy = instanceIndex.div(uint(GRASS_GRID)).toFloat();
        const s = asVec4(
          call(grassSample, {
            cell: asVec3(vec3(cx, cy, 0)).xy,
            bake: this.bake,
            bake2: this.bake2,
            camDir: this.camDir,
            camEast: this.camEast,
            camNorth: this.camNorth,
            alt: this.alt,
            cfg: this.cfg,
            cfg2: this.cfg2,
            view: this.view,
          }),
        );
        If((s.w as unknown as N).greaterThan(0) as never, () => {
          const slot = atomicAdd(counter.element(uint(1)), uint(1));
          If(slot.lessThan(uint(GRASS_CAPACITY)) as never, () => {
            insts.element(slot).assign(s);
          });
        });
      })() as never,
      GRASS_GRID * GRASS_GRID,
    );

    const geo = bladeGeometry(GRASS_SEGMENTS);
    geo.indirect = this.argsAttr;
    geo.indirectOffset = 0;
    // Index count from the geometry, never written by hand — see the note on
    // the same line in vegetation.ts, which cost a whole canopy to learn.
    (this.argsAttr.array as Uint32Array)[0] = geo.getIndex()!.count;

    const readInsts = storage(this.instAttr, 'vec4', GRASS_CAPACITY).toReadOnly();
    const inst = asVec4(readInsts.element(instanceIndex));
    const bseg = asVec4(attribute('bseg', 'vec4'));

    const mat = new MeshBasicNodeMaterial();
    mat.positionNode = asVec3(
      call(grassVertex, { inst, seg: bseg, camPos: this.camPos, cfg: this.dcfg }),
    );

    // Normal by finite difference across the blade, as the trees do: the
    // Bezier carries a wind term and differentiating that by hand is a bug
    // waiting to happen for two extra evaluations on a nine-vertex strip.
    const e = 0.02;
    const p0 = asVec3(call(grassVertex, { inst, seg: bseg, camPos: this.camPos, cfg: this.dcfg }));
    const pu = asVec3(
      call(grassVertex, {
        inst,
        seg: asVec4(vec4(bseg.x.add(e), bseg.y, bseg.z, bseg.w)),
        camPos: this.camPos,
        cfg: this.dcfg,
      }),
    );
    const pv = asVec3(
      call(grassVertex, {
        inst,
        seg: asVec4(vec4(bseg.x, bseg.y.add(e), bseg.z, bseg.w)),
        camPos: this.camPos,
        cfg: this.dcfg,
      }),
    );
    const nrm = asVec3(pu.sub(p0).cross(pv.sub(p0)).normalize());

    const shaded = asVec4(
      call(shadeGrass, {
        inst: varying(inst as never, 'gInst'),
        nrm: varying(nrm as never, 'gNrm'),
        tt: varying(bseg.x as never, 'gT'),
        camPos: this.camPos,
        sunDir: this.sun,
        sunCol: this.sunCol,
        cfg: this.dcfg,
        shadow: shadowFactor ? shadowFactor(inst.xyz) : float(1),
      }),
    );
    mat.colorNode = asVec3(shaded.xyz);
    // Both faces: a blade is a surface with no inside, and back-face culling a
    // field removes about half of it depending on which way the wind blew.
    mat.side = DoubleSide;
    mat.toneMapped = true;

    this.mesh = new Mesh(geo, mat);
    this.mesh.frustumCulled = false;
    this.mesh.matrixAutoUpdate = false;
    this.mesh.renderOrder = 6;
  }

  /**
   * Place the grid for this frame.
   *
   * The snap is the load-bearing part. The camera's position on its cube face,
   * in metres, is quantised to GRASS_SPACING in f64 here; the integer part
   * seeds the hash and the fraction positions the grid. Doing it the obvious
   * way — grid origin at the camera — would reseed every blade every frame and
   * the field would crawl.
   */
  update(
    renderer: WebGPURenderer,
    surface: PlanetSurface | null,
    camPos: readonly [number, number, number],
    octaves: number,
    heightScale: number,
    timeSec: number,
    /** Metres between the camera and the surface it is actually standing on. */
    clearance: number,
    /** Camera forward, world space — the axis of the scatter's culling cone. */
    fwd: readonly [number, number, number],
    fovDeg: number,
    aspect: number,
  ): void {
    const r = Math.hypot(camPos[0], camPos[1], camPos[2]);
    const dir: [number, number, number] = [camPos[0] / r, camPos[1] / r, camPos[2] / r];

    if (!this.enabled || !surface) {
      if (this.live) {
        renderer.compute(this.resetPass as never);
        this.live = false;
      }
      this.mesh.visible = false;
      return;
    }

    // Warped, like every other consumer of the bake — see warpForCoast.
    const dw = warpForCoast(dir, sampleSurface(surface, dir[0], dir[1], dir[2]).elevation);
    const b = sampleSurface(surface, dw[0], dw[1], dw[2]);
    const alt = r - RADIUS;
    // Only worth running when the camera is low enough for a blade to be more
    // than a pixel. Above that the terrain's own albedo is the grass.
    //
    // Against the *amplified* surface, not the bake: on a 900 m plateau the
    // two differ by the whole plateau, and gating on the bake switched the
    // grass off everywhere the ground was high — which is most of the land.
    if (clearance > GRASS_RANGE * 3) {
      if (this.live) {
        renderer.compute(this.resetPass as never);
        this.live = false;
      }
      this.mesh.visible = false;
      return;
    }
    this.mesh.visible = true;

    // Bake and its gradient at the camera, by the same one-cell central
    // difference the terrain uses.
    const e = FACE_EDGE / surface.size / RADIUS;
    const ax: [number, number, number] =
      Math.abs(dir[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
    const dp = ax[0] * dir[0] + ax[1] * dir[1] + ax[2] * dir[2];
    const t1 = norm([ax[0] - dir[0] * dp, ax[1] - dir[1] * dp, ax[2] - dir[2] * dp]);
    const t2 = norm(cross(dir, t1));
    const at = (t: readonly [number, number, number], sg: number) => {
      const d0 = norm([dir[0] + t[0] * e * sg, dir[1] + t[1] * e * sg, dir[2] + t[2] * e * sg]);
      const d = warpForCoast(d0, sampleSurface(surface, d0[0], d0[1], d0[2]).elevation);
      return sampleSurface(surface, d[0], d[1], d[2]);
    };
    const gu = (at(t1, 1).elevation - at(t1, -1).elevation) / (2 * e);
    const gv = (at(t2, 1).elevation - at(t2, -1).elevation) / (2 * e);
    this.bake.value.set(
      b.elevation,
      t1[0] * gu + t2[0] * gv,
      t1[1] * gu + t2[1] * gv,
      t1[2] * gu + t2[2] * gv,
    );
    this.bake2.value.set(b.wetness, b.lakeDepth, b.channelDist, 0);

    this.camDir.value.set(dir[0], dir[1], dir[2]);
    this.camEast.value.set(t1[0], t1[1], t1[2]);
    this.camNorth.value.set(t2[0], t2[1], t2[2]);

    // The view cone the scatter culls against. The forward is projected into
    // the same (east, north) basis the candidate offsets are built in, so the
    // test in grassSample is a dot product on two numbers it already has.
    //
    // The half-angle is the *horizontal* one — atan(tan(vfov/2) · aspect) — plus
    // a generous margin, because a blade leans, casts, and is a metre wide at
    // the near plane. Too tight shows as blades popping in at the screen edge
    // when you turn; the margin is cheap because area grows with the angle.
    const fe = fwd[0] * t1[0] + fwd[1] * t1[1] + fwd[2] * t1[2];
    const fn = fwd[0] * t2[0] + fwd[1] * t2[1] + fwd[2] * t2[2];
    const fl = Math.hypot(fe, fn) || 1;
    const halfH = Math.atan(Math.tan((fovDeg * Math.PI) / 360) * Math.max(aspect, 0.2));
    this.view.value.set(fe / fl, fn / fl, Math.cos(Math.min(halfH + 0.55, Math.PI * 0.5)), 0);
    this.alt.value = alt;
    this.camPos.value.set(camPos[0], camPos[1], camPos[2]);

    // The snap. Face-uv scaled to metres is a global 2D coordinate per face,
    // so quantising it gives an index a blade keeps for as long as it exists.
    const fc = directionToFace(dir);
    const su = (fc.u * FACE_EDGE) / 2 / GRASS_SPACING;
    const sv = (fc.v * FACE_EDGE) / 2 / GRASS_SPACING;
    const iu = Math.floor(su);
    const iv = Math.floor(sv);
    this.cfg2.value.set(iu - GRASS_GRID / 2, iv - GRASS_GRID / 2, su - iu, sv - iv);

    this.cfg.value.set(RADIUS, heightScale, octaves, 1);
    this.dcfg.value.set(RADIUS, 0, timeSec, 0);

    renderer.compute(this.resetPass as never);
    renderer.compute(this.scatterPass as never);
    this.live = true;
  }

  setSun(d: Vector3, colour: Vector3): void {
    this.sun.value.copy(d).normalize();
    this.sunCol.value.copy(colour);
  }

  setEnabled(on: boolean): void {
    this.enabled = on;
  }
  get isEnabled(): boolean {
    return this.enabled;
  }
}

function norm(v: readonly number[]): [number, number, number] {
  const l = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / l, v[1] / l, v[2] / l];
}
function cross(
  a: readonly number[],
  b: readonly number[],
): [number, number, number] {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}
