/**
 * CDLOD patch selection over the cube-sphere quadtree. See SPEC.md §3, §6.
 *
 * Per frame this walks six root quadtrees, culls against the frustum and the
 * planet's horizon, and writes camera-relative instance data for the survivors.
 *
 * Crack-freeness rests on two properties:
 *
 *   1. LOD ranges double per level, so neighbouring selected patches differ
 *      by at most one level.
 *   2. The morph factor is evaluated per *vertex* from its world distance
 *      (in the shader), not per patch. Two patches sharing an edge therefore
 *      compute identical morph at the shared vertices, and a finer patch is
 *      fully morphed onto its parent's grid before the parent takes over.
 *
 * No stitching geometry, no skirts, no neighbour lookups.
 */

import { Frustum, Matrix4, Vector3 } from 'three';
import {
  MAX_ELEVATION,
  MAX_LEVEL,
  MAX_PATCHES,
  MAX_VEG_TILES,
  VEG_TILE_VEC4,
  FACE_EDGE,
  MIN_ELEVATION,
  morphStartFor,
  RADIUS,
  MIN_SELECT_LEVEL,
  VEG_LEVEL,
  VEG_TILE_RANGE,
  edgeLengthAt,
} from './planet.js';
import { FACES, cubePoint, warp } from './cubesphere.js';
import { type V3, addScaled, cross, dot, len, normalize, sub } from './math/vec3d.js';
import { sampleSurface, type PlanetSurface } from './planetData.js';
import { warpForCoast } from './heightCPU.js';

export interface SelectStats {
  patches: number;
  visited: number;
  culledFrustum: number;
  culledHorizon: number;
  minLevel: number;
  maxLevel: number;
  overflow: boolean;
  /** Vegetation tiles collected this frame (SPEC.md §8). */
  vegTiles: number;
}

/** Per-instance GPU arrays, sized once at construction. */
export class PatchBuffers {
  readonly center = new Float32Array(MAX_PATCHES * 4); // A, B, halfSize, —
  readonly dirLen = new Float32Array(MAX_PATCHES * 4); // dirC.xyz, |Pc|
  readonly anchor = new Float32Array(MAX_PATCHES * 4); // anchorRel.xyz, level
  readonly basisU = new Float32Array(MAX_PATCHES * 3);
  readonly basisV = new Float32Array(MAX_PATCHES * 3);
  readonly morph = new Float32Array(MAX_PATCHES * 2); // start, end
}

const MID_ELEV = (MAX_ELEVATION + MIN_ELEVATION) / 2;

/** Lowest the surface can be — the conservative occluder for horizon culling. */
const R_OCCLUDER = RADIUS + MIN_ELEVATION;

/** How far a maximum-elevation peak can appear beyond that horizon. */
const PEEK_ANGLE = Math.acos(R_OCCLUDER / (RADIUS + MAX_ELEVATION));

export class PatchSelector {
  readonly buffers = new PatchBuffers();
  readonly stats: SelectStats = {
    patches: 0,
    visited: 0,
    culledFrustum: 0,
    culledHorizon: 0,
    minLevel: MAX_LEVEL,
    maxLevel: 0,
    overflow: false,
    vegTiles: 0,
  };

  /**
   * Scatter tiles for vegetation: five vec4 of parameters each, in the same
   * form the terrain patches use, so the scatter shader can reuse the terrain's
   * precision reconstruction unchanged.
   *
   * Collected during the same traversal rather than in a second walk — every
   * node at VEG_LEVEL within VEG_RANGE is necessarily visited, because
   * range[VEG_LEVEL-1] is more than twice VEG_RANGE.
   */
  readonly vegTileData = new Float32Array(MAX_VEG_TILES * VEG_TILE_VEC4 * 4);
  private vegCount = 0;

  /** range[L]: a node subdivides while the camera is nearer than this. */
  private ranges = new Float64Array(MAX_LEVEL + 1);

  private frustum = new Frustum();
  private tmpMat = new Matrix4();
  private tmpVec = new Vector3();

  private camPos: V3 = [0, 0, RADIUS];
  private camDir: V3 = [0, 0, 1];
  private camR = RADIUS;
  private refR = RADIUS;
  private horizonAngle = 0;
  private count = 0;
  private maxLevelCap = MAX_LEVEL;
  private distanceCap = Infinity;

  /** The M3 bake. Sampled per vegetation tile; see emitVegTile. */
  private surface: PlanetSurface | null = null;

  setPlanetSurface(s: PlanetSurface): void {
    this.surface = s;
  }

  constructor(lodFactor: number) {
    this.setLodFactor(lodFactor);
  }

  private morphStart = morphStartFor(1);

  setLodFactor(f: number): void {
    for (let l = 0; l <= MAX_LEVEL; l++) this.ranges[l] = f * edgeLengthAt(l);
    this.morphStart = morphStartFor(f);
  }

  setMaxLevel(l: number): void {
    this.maxLevelCap = Math.max(0, Math.min(MAX_LEVEL, l | 0));
  }

  /**
   * Discard patches beyond this distance entirely. Used for the shadow pass,
   * which needs terrain out to the largest cascade and nothing else — drawing
   * the full horizon into a 61 m cascade was 75% of all geometry in the frame.
   */
  setDistanceCap(d: number): void {
    this.distanceCap = d;
  }

  /**
   * Select visible patches for this camera pose.
   *
   * @param groundR  radius of the terrain surface beneath the camera. LOD
   *   distance is measured against a sphere of this radius rather than a
   *   bounding volume: patch bounding spheres are dominated by the ±6.6 km
   *   global elevation span, which collapses the distance metric to zero for
   *   everything within 6.6 km and forces maximum subdivision over a huge area.
   *
   * `viewMatrix` and `projMatrix` must already be camera-relative
   * (camera at the origin), which is how the renderer is driven.
   */
  select(
    camPos: V3,
    groundR: number,
    viewMatrix: Matrix4,
    projMatrix: Matrix4,
  ): SelectStats {
    this.camPos = camPos;
    this.camR = len(camPos);
    this.camDir = this.camR > 0 ? normalize(camPos) : [0, 0, 1];
    this.refR = groundR;

    // Horizon cap, measured from the camera's nadir. The occluder is the
    // *lowest* the terrain can be, so the test stays conservative; the second
    // term is how far a maximum-elevation peak can peek over that horizon.
    // Testing against the max-elevation shell instead would disable culling
    // entirely whenever the camera is below it — i.e. exactly at ground level,
    // where it matters most.
    if (this.camR <= R_OCCLUDER) {
      this.horizonAngle = Math.PI;
    } else {
      this.horizonAngle =
        Math.acos(Math.min(1, R_OCCLUDER / this.camR)) + PEEK_ANGLE;
    }

    this.tmpMat.multiplyMatrices(projMatrix, viewMatrix);
    this.frustum.setFromProjectionMatrix(this.tmpMat);

    this.count = 0;
    const s = this.stats;
    s.visited = 0;
    s.culledFrustum = 0;
    s.culledHorizon = 0;
    s.minLevel = MAX_LEVEL;
    s.maxLevel = 0;
    s.overflow = false;
    this.vegCount = 0;

    for (let face = 0; face < 6; face++) this.walk(face, 0, 0, 0);

    s.patches = this.count;
    s.vegTiles = this.vegCount;
    if (s.patches === 0) s.minLevel = 0;
    return s;
  }

  /**
   * @param face  cube face index
   * @param level quadtree depth
   * @param i,j   tile indices within the face at this level
   */
  private walk(face: number, level: number, i: number, j: number): void {
    this.stats.visited++;

    const n = 1 << level;
    const hs = 1 / n; // half-size in face-uv (face spans [-1,1])
    const cu = -1 + (2 * i + 1) * hs;
    const cv = -1 + (2 * j + 1) * hs;

    const A = warp(cu);
    const B = warp(cv);
    const Pc = cubePoint(face, A, B);
    const lenPc = len(Pc);
    const dirC: V3 = [Pc[0] / lenPc, Pc[1] / lenPc, Pc[2] / lenPc];

    // Angular radius: the corner that deviates most from the patch centre.
    let cosMin = 1;
    for (let k = 0; k < 4; k++) {
      const du = k & 1 ? hs : -hs;
      const dv = k & 2 ? hs : -hs;
      const c = cubePoint(face, warp(cu + du), warp(cv + dv));
      const cl = len(c);
      const cd = (c[0] * dirC[0] + c[1] * dirC[1] + c[2] * dirC[2]) / cl;
      if (cd < cosMin) cosMin = cd;
    }
    const angRadius = Math.acos(Math.max(-1, Math.min(1, cosMin)));

    // Horizon cull. Conservative: keep the patch if any part of its angular
    // extent falls inside the visible cap.
    const centreAngle = Math.acos(
      Math.max(-1, Math.min(1, dot(this.camDir, dirC))),
    );
    if (centreAngle - angRadius > this.horizonAngle) {
      this.stats.culledHorizon++;
      return;
    }

    // Bounding sphere, centred at mid-elevation on the patch axis. Used only
    // for frustum culling, where it must stay conservative.
    const bsC: V3 = [
      dirC[0] * (RADIUS + MID_ELEV),
      dirC[1] * (RADIUS + MID_ELEV),
      dirC[2] * (RADIUS + MID_ELEV),
    ];
    // Chord to a corner at max elevation, plus the elevation half-span.
    const chord = 2 * (RADIUS + MAX_ELEVATION) * Math.sin(angRadius / 2);
    const bsR = Math.hypot(chord, (MAX_ELEVATION - MIN_ELEVATION) / 2) + 1;

    // Frustum cull in camera-relative space (camera sits at the origin).
    const rel = sub(bsC, this.camPos);
    this.tmpVec.set(rel[0], rel[1], rel[2]);
    if (!this.frustum.intersectsSphere({ center: this.tmpVec, radius: bsR } as never)) {
      this.stats.culledFrustum++;
      return;
    }

    // LOD distance: camera to the nearest point of the patch, both taken on a
    // sphere of the camera's own ground radius. Ignoring the elevation span
    // here is deliberate — see the note on `select`.
    const ang = Math.max(0, centreAngle - angRadius);
    const d = Math.sqrt(
      Math.max(
        0,
        this.camR * this.camR +
          this.refR * this.refR -
          2 * this.camR * this.refR * Math.cos(ang),
      ),
    );

    if (level === VEG_LEVEL && d <= VEG_TILE_RANGE) {
      this.emitVegTile(face, i, j, A, B, hs, dirC, lenPc);
    }

    if (d > this.distanceCap) return;

    // A floor on subdivision, independent of distance.
    //
    // The LOD range is fitted to *geometric* error, and from orbit that is
    // satisfied by level 1–2 — patches thousands of kilometres across, with
    // 72 km between vertices. But the climate, the elevation and the normal all
    // reach the fragment stage as varyings, and the albedo thresholds them:
    // biome edges, the snow line, the shoreline. Thresholding a field that has
    // been linearly interpolated over 72 km turns every one of those boundaries
    // into a polygon, which is what made ice caps and deserts look faceted from
    // 9000 km.
    //
    // MIN_SELECT_LEVEL puts vertices exactly one bake cell apart, so nothing
    // downstream is interpolated further than the data it came from. It is
    // derived from BAKE_RES rather than written down — see planet.ts, where it
    // spent a while asserting 18 km was the cell size after the bake moved to
    // 9 km. It costs little where it applies: from orbit the horizon and
    // frustum culls leave a few hundred patches, and there is nothing else on
    // screen.
    const subdivide =
      level < this.maxLevelCap && (level < MIN_SELECT_LEVEL || d <= this.ranges[level]);
    if (subdivide) {
      const i2 = i * 2;
      const j2 = j * 2;
      this.walk(face, level + 1, i2, j2);
      this.walk(face, level + 1, i2 + 1, j2);
      this.walk(face, level + 1, i2, j2 + 1);
      this.walk(face, level + 1, i2 + 1, j2 + 1);
      return;
    }

    this.emit(face, level, A, B, hs, dirC, lenPc);
  }

  private emitVegTile(
    face: number,
    ti: number,
    tj: number,
    A: number,
    B: number,
    hs: number,
    dirC: V3,
    lenPc: number,
  ): void {
    if (this.vegCount >= MAX_VEG_TILES) return;
    const o = this.vegCount++ * VEG_TILE_VEC4 * 4;
    const t = this.vegTileData;
    const { U, V } = FACES[face];

    t[o] = A;
    t[o + 1] = B;
    t[o + 2] = hs;
    t[o + 3] = face;

    t[o + 4] = dirC[0];
    t[o + 5] = dirC[1];
    t[o + 6] = dirC[2];
    t[o + 7] = lenPc;

    // The one f64 subtraction, as for terrain patches (SPEC.md I4).
    t[o + 8] = dirC[0] * RADIUS - this.camPos[0];
    t[o + 9] = dirC[1] * RADIUS - this.camPos[1];
    t[o + 10] = dirC[2] * RADIUS - this.camPos[2];
    t[o + 11] = 0;

    t[o + 12] = U[0];
    t[o + 13] = U[1];
    t[o + 14] = U[2];
    // Tile indices, so the scatter can build a globally unique cell coordinate.
    // Salting by list position instead would break determinism (I1): the same
    // ground would rescatter differently as tiles entered and left view.
    t[o + 15] = ti;

    t[o + 16] = V[0];
    t[o + 17] = V[1];
    t[o + 18] = V[2];
    t[o + 19] = tj;

    // The M3 bake, sampled once for the whole tile.
    //
    // The scatter needs the baked elevation to stand trees on the ground, but
    // the bake resolves 18 km cells and this tile is a few hundred metres
    // across — the field is very nearly linear over it. So one value and one
    // gradient here replace five cube-map fetches per candidate cell, and
    // there are ~500 k candidates a frame.
    if (this.surface) {
      // Warped, like the terrain's own lookup — see warpForCoast. Sampling the
      // true direction here would put every tree on the unwarped surface.
      const dirW = warpForCoast(dirC, sampleSurface(this.surface, dirC[0], dirC[1], dirC[2]).elevation);
      const b = sampleSurface(this.surface, dirW[0], dirW[1], dirW[2]);
      const e = FACE_EDGE / this.surface.size / RADIUS;
      const at = (d: V3): { elevation: number; wetness: number; channelDist: number } =>
        sampleSurface(this.surface!, d[0], d[1], d[2]);
      const step = (base: V3, ax: V3, sg: number, k = 1): V3 =>
        normalize([
          base[0] + ax[0] * e * sg * k,
          base[1] + ax[1] * e * sg * k,
          base[2] + ax[2] * e * sg * k,
        ]);
      const warped = (d: V3): V3 =>
        warpForCoast(d, sampleSurface(this.surface!, d[0], d[1], d[2]).elevation);
      // U and V are tangent to the face, so they are never parallel to dirC
      // inside it — the same frame the terrain shader differences along.
      const tu = normalize(addScaled(U, dirC, -dot(U, dirC)));
      const tv = normalize(cross(dirC, tu));
      const eM = e * RADIUS;

      const up = at(warped(step(dirC, tu, 1)));
      const um = at(warped(step(dirC, tu, -1)));
      const vp = at(warped(step(dirC, tv, 1)));
      const vm = at(warped(step(dirC, tv, -1)));
      const gu = (up.elevation - um.elevation) / (2 * e);
      const gv = (vp.elevation - vm.elevation) / (2 * e);

      // Distance to the nearest channel, from the bake — see carveChannels.
      const distAxisOf = (d: V3): number => at(warped(d)).channelDist;
      const distAxisAt = b.channelDist;
      const gdu = (distAxisOf(step(dirC, tu, 1, 2)) - distAxisOf(step(dirC, tu, -1, 2))) / (4 * eM);
      const gdv = (distAxisOf(step(dirC, tv, 1, 2)) - distAxisOf(step(dirC, tv, -1, 2))) / (4 * eM);

      t[o + 20] = b.elevation;
      t[o + 21] = tu[0] * gu + tv[0] * gv;
      t[o + 22] = tu[1] * gu + tv[1] * gv;
      t[o + 23] = tu[2] * gu + tv[2] * gv;
      t[o + 24] = b.wetness;
      t[o + 25] = b.lakeDepth;
      // Distance to the drainage axis, from the same four taps read on
      // wetness — see CHANNEL_WIDTH_K. Constant over the tile, which is fine:
      // a tile is a few hundred metres and this varies over the bake's 9 km.
      // Distance to the drainage axis and its gradient, reconstructed across
      // the tile exactly as the elevation is — it drives a 45 m channel cut, so
      // a per-tile constant would leave stems hanging that far off the ground.
      t[o + 26] = distAxisAt;
      t[o + 27] = 0;
      t[o + 28] = distAxisAt;
      t[o + 29] = tu[0] * gdu + tv[0] * gdv;
      t[o + 30] = tu[1] * gdu + tv[1] * gdv;
      t[o + 31] = tu[2] * gdu + tv[2] * gdv;
    }
  }

  private emit(
    face: number,
    level: number,
    A: number,
    B: number,
    hs: number,
    dirC: V3,
    lenPc: number,
  ): void {
    if (this.count >= MAX_PATCHES) {
      this.stats.overflow = true;
      return;
    }
    const k = this.count++;
    const b = this.buffers;

    // The one subtraction that must happen in f64 (SPEC.md I4). Everything
    // the GPU sees downstream of this is a small, camera-relative quantity.
    const ax = dirC[0] * RADIUS - this.camPos[0];
    const ay = dirC[1] * RADIUS - this.camPos[1];
    const az = dirC[2] * RADIUS - this.camPos[2];

    const o4 = k * 4;
    b.center[o4] = A;
    b.center[o4 + 1] = B;
    b.center[o4 + 2] = hs;
    b.center[o4 + 3] = 0;

    b.dirLen[o4] = dirC[0];
    b.dirLen[o4 + 1] = dirC[1];
    b.dirLen[o4 + 2] = dirC[2];
    b.dirLen[o4 + 3] = lenPc;

    b.anchor[o4] = ax;
    b.anchor[o4 + 1] = ay;
    b.anchor[o4 + 2] = az;
    b.anchor[o4 + 3] = level;

    const o3 = k * 3;
    const { U, V } = FACES[face];
    b.basisU[o3] = U[0];
    b.basisU[o3 + 1] = U[1];
    b.basisU[o3 + 2] = U[2];
    b.basisV[o3] = V[0];
    b.basisV[o3 + 1] = V[1];
    b.basisV[o3 + 2] = V[2];

    // Morph fully completes before the parent takes over: the node is active
    // for d ∈ (range[L], 2·range[L]], and MORPH_START > 0.5 keeps the whole
    // transition inside that window.
    const end = 2 * this.ranges[level];
    const o2 = k * 2;
    b.morph[o2] = this.morphStart * end;
    b.morph[o2 + 1] = end;

    if (level < this.stats.minLevel) this.stats.minLevel = level;
    if (level > this.stats.maxLevel) this.stats.maxLevel = level;
  }
}
