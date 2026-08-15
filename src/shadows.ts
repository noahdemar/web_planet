/**
 * Cascaded shadow maps, camera-relative. See SPEC.md §7.
 *
 * Standard CSM assumes a bounded view distance. This world spans centimetres
 * to thousands of kilometres, so the cascade ranges are derived from the
 * camera's altitude and the whole system fades out above the height where
 * shadows stop contributing anything.
 *
 * Two things fall out of the camera-relative architecture and make this much
 * simpler than usual:
 *
 *   - every mesh already emits positions relative to the camera, so a light
 *     camera placed in that same space needs no world transform at all;
 *   - there is no origin to shift, so shadow matrices never suffer the
 *     precision loss that normally forces cascade re-centring hacks.
 *
 * Depth is written as linear distance along the light direction into an R32F
 * target rather than using a hardware depth texture with a comparison sampler.
 * The comparison is then an ordinary subtract, which avoids a pile of
 * depth-texture plumbing for no visual difference at this quality level.
 */

import {
  type Camera,
  Color,
  Matrix4,
  NearestFilter,
  OrthographicCamera,
  RedFormat,
  RenderTarget,
  Scene,
  Vector3,
  type Material,
  type Mesh,
  FloatType,
} from 'three';
import type { WebGPURenderer } from 'three/webgpu';
import { QUALITY } from './quality.js';

/**
 * Cascades. Three is enough given how fast the ranges grow, and two is what a
 * phone gets: the third covers 260–1300 m, which is where aerial perspective
 * has already taken over, and dropping it also shortens the shadow pass's
 * terrain re-selection — the vertex cost that dominates this pass.
 *
 * Fixed at import time. `shaders/shadowSample.ts` builds one node graph per
 * cascade and bakes MAP_SIZE into it as a literal, so neither can move once
 * the shaders exist. See the note on QUALITY about why reading it here is
 * safe.
 */
export const CASCADES = QUALITY.shadowCascades;

/** Shadow map resolution per cascade. */
export const MAP_SIZE = QUALITY.shadowMapSize;

/**
 * Cascade outer radii at ground level, metres. Deliberately tight near the
 * camera — that is where contact shadows read — and coarse beyond, where
 * aerial perspective is already dissolving the detail.
 *
 * With two cascades the far one is stretched to 420 m rather than left at 260:
 * losing the third cascade should cost range, but not that much of it.
 */
const BASE_RADII = CASCADES >= 3 ? [55, 260, 1300] : [55, 420];

/** Above this altitude shadows contribute nothing and are switched off. */
export const SHADOW_MAX_ALTITUDE = 14_000;

/**
 * Clear value for a shadow map texel no caster wrote to. See the note in
 * render(). Large and negative so the lit test always passes there.
 */
const NO_OCCLUDER = new Color();
NO_OCCLUDER.r = -1e9;
NO_OCCLUDER.g = -1e9;
NO_OCCLUDER.b = -1e9;

/** How far the light frustum extends behind the receivers, to catch casters. */
export const CASTER_DEPTH = 4500;

/**
 * Added to every stored and compared distance so the payload is never negative.
 *
 * The map stores signed distance along the light, and roughly half of any scene
 * is on the far side of the camera from the sun, so half the values are
 * negative. They were not surviving the write: measured across a row of cascade
 * 2, 1566 written texels came back with **min 0, not one negative, and 1206 of
 * them exactly 0**, while positive values passed through untouched (max 4590).
 * Something in three's WebGPU output path clamps at zero — it is not the clear
 * (the sentinel reads back as -1e9 exactly) and it is not tone mapping
 * (`toneMapped = false` changes nothing).
 *
 * The consequence is the same artefact LESSONS §8 records against the old
 * 0xffffff clear, and for the same reason: a caster stores 0 where it should
 * store a large negative, so `frag + bias >= stored` fails for every receiver
 * with `dot(rel, sunDir) < -bias`. That set is a half-space, its boundary is a
 * plane through the camera, and it draws a dark region that follows the camera
 * across the landscape and covers the half of the world facing away from the
 * sun. Fixing the clear removed one half of that bug in M3 and left this half.
 *
 * Rather than keep hunting the clamp, the payload is made unsigned. A float
 * target carrying signed values through an output chain nobody controls is a
 * fragile contract; a positive offset costs nothing and cannot be broken by it.
 *
 * The bound, so the offset is derived rather than picked: the largest cascade
 * is `BASE_RADII[2] · (1 + SHADOW_MAX_ALTITUDE/900)` = 21.5 km, its centre sits
 * at most `radius·0.65 + SHADOW_MAX_ALTITUDE` = 28 km from the camera, and the
 * frustum reaches `CASTER_DEPTH + radius` = 26 km either side of that. So no
 * point inside any cascade exceeds ~76 km, and 200 km leaves 2.6x of margin.
 * f32 resolves 1.6 cm at that magnitude, against a minimum bias of 0.35 m.
 */
export const SHADOW_DEPTH_OFFSET = 200_000;

export interface ShadowCaster {
  mesh: Mesh;
  /** Same positionNode as the display material, trivial fragment shader. */
  depthMaterial: Material;
  /**
   * Skip this caster for cascades whose radius exceeds this. Distant foliage
   * casting into the far cascade costs 200k instances per cascade and changes
   * essentially nothing on screen.
   */
  maxCascadeRadius?: number;
}

export class Shadows {
  readonly targets: RenderTarget[] = [];
  readonly matrices: Matrix4[] = [];
  /** Outer radius of each cascade this frame, for the shader's band select. */
  readonly radii: number[] = new Array(CASCADES).fill(1);
  /** 0 when shadows are off (too high, or sun below the horizon). */
  strength = 1;

  private cams: OrthographicCamera[] = [];
  private sun = new Vector3(0, 1, 0);
  private tmpCentre = new Vector3();
  private tmpUp = new Vector3();

  constructor() {
    for (let i = 0; i < CASCADES; i++) {
      const rt = new RenderTarget(MAP_SIZE, MAP_SIZE, {
        format: RedFormat,
        type: FloatType,
        // Nearest: the stored value is a distance, and interpolating distances
        // across a depth discontinuity invents surfaces that are not there.
        minFilter: NearestFilter,
        magFilter: NearestFilter,
        depthBuffer: true,
        generateMipmaps: false,
      });
      rt.texture.name = `shadow${i}`;
      this.targets.push(rt);
      this.matrices.push(new Matrix4());
      this.cams.push(new OrthographicCamera(-1, 1, 1, -1, 0, 1));
    }
  }

  /**
   * Position the cascades for this frame.
   *
   * @param forward  camera view direction, world space
   * @param sunDir   direction *toward* the sun
   * @param altitude metres above the local surface
   */
  update(forward: Vector3, sunDir: Vector3, altitude: number, localUp: Vector3): void {
    this.sun.copy(sunDir).normalize();

    // Fade out with altitude, and with the sun near or below the horizon where
    // shadows would be kilometres long and mostly wrong.
    const alt = 1 - smoothstep(SHADOW_MAX_ALTITUDE * 0.55, SHADOW_MAX_ALTITUDE, altitude);
    const elev = smoothstep(0.02, 0.15, this.sun.dot(localUp));
    this.strength = alt * elev;

    // Grow the cascades with altitude so they still cover what is on screen,
    // but sub-linearly — detail shadows stop mattering long before the ground
    // does.
    const grow = 1 + Math.min(altitude, SHADOW_MAX_ALTITUDE) / 900;

    for (let i = 0; i < CASCADES; i++) {
      const radius = BASE_RADII[i] * grow;
      this.radii[i] = radius;

      // Centre the cascade on the *ground*, led forward.
      //
      // This used to be `forward * radius * 0.65` alone, which centres the box
      // near the camera. That is right at eye height, where the ground is a
      // couple of metres away, and wrong the moment there is any altitude: at
      // 8 km the ground under the camera is 8 km below, so cascades 0 and 1 —
      // 540 m and 2.6 km at that height — do not reach it at all, and cascade 2
      // only just does. The ground under the camera then sat on cascade 2's
      // footprint edge, where `coverOf` is part-way through its shoulder, so it
      // slid in and out of shadow as the camera turned. That is the pop.
      //
      // Dropping the centre to the surface first fixes it at every altitude and
      // changes nothing at eye level, where the drop is 1.7 m. The lead uses
      // the *horizontal* part of forward so that looking straight down centres
      // the box under the camera instead of pushing it out to one side.
      this.tmpCentre
        .copy(forward)
        .addScaledVector(localUp, -forward.dot(localUp));
      const lead = this.tmpCentre.length();
      if (lead > 1e-4) this.tmpCentre.multiplyScalar((radius * 0.65) / lead);
      this.tmpCentre.addScaledVector(localUp, -altitude);

      const cam = this.cams[i];
      const texel = (2 * radius) / MAP_SIZE;

      // Snap the centre to whole texels along the light's own axes. Without
      // this the map re-samples slightly every frame and every shadow edge
      // crawls — the single most visible shadow artefact in motion.
      this.tmpUp.set(0, 1, 0);
      if (Math.abs(this.sun.dot(this.tmpUp)) > 0.99) this.tmpUp.set(1, 0, 0);
      const right = new Vector3().crossVectors(this.tmpUp, this.sun).normalize();
      const up = new Vector3().crossVectors(this.sun, right).normalize();
      const cx = Math.round(this.tmpCentre.dot(right) / texel) * texel;
      const cy = Math.round(this.tmpCentre.dot(up) / texel) * texel;
      const cz = this.tmpCentre.dot(this.sun);
      this.tmpCentre
        .copy(right)
        .multiplyScalar(cx)
        .addScaledVector(up, cy)
        .addScaledVector(this.sun, cz);

      cam.left = -radius;
      cam.right = radius;
      cam.top = radius;
      cam.bottom = -radius;
      cam.near = 0;
      cam.far = CASTER_DEPTH * 2 + radius * 2;
      cam.position.copy(this.tmpCentre).addScaledVector(this.sun, CASTER_DEPTH + radius);
      cam.up.copy(up);
      cam.lookAt(this.tmpCentre);
      cam.updateMatrixWorld(true);
      cam.updateProjectionMatrix();

      this.matrices[i].multiplyMatrices(cam.projectionMatrix, cam.matrixWorldInverse);
    }
  }

  /**
   * Render the casters into every cascade.
   *
   * Materials are swapped rather than using `scene.overrideMaterial`, because
   * each caster's vertex position comes from its own node graph — terrain
   * reconstructs the sphere, vegetation builds billboards — and a single
   * override material cannot express both.
   */
  render(
    renderer: WebGPURenderer,
    scene: Scene,
    casters: ShadowCaster[],
    hide: { visible: boolean }[],
  ): void {
    if (this.strength <= 0) return;

    const wasVisible = hide.map((o) => o.visible);
    hide.forEach((o) => (o.visible = false));
    const displayMaterials = casters.map((c) => c.mesh.material);
    const casterVisible = casters.map((c) => c.mesh.visible);
    casters.forEach((c) => (c.mesh.material = c.depthMaterial));

    for (let i = 0; i < CASCADES; i++) {
      // Per-cascade caster selection, not per-frame: what is worth recording
      // in a 61 m map is not what is worth recording in a 1.4 km one.
      casters.forEach((c, k) => {
        const limit = c.maxCascadeRadius ?? Infinity;
        c.mesh.visible = casterVisible[k] && this.radii[i] <= limit;
      });

      renderer.setRenderTarget(this.targets[i]);
      // "Nothing here" has to mean *nothing occludes*, and the stored value is
      // signed distance along the light in metres — so the sentinel belongs at
      // minus infinity, not at white.
      //
      // It was 0xffffff, i.e. 1.0, which reads as "there is an occluder one
      // metre along the light from the camera". The lit test is
      // frag + bias >= stored, and frag is dot(rel, sunDir), so every point
      // with dot(rel, sunDir) < 1 came back shadowed — and that set is a
      // half-space. Its boundary is a plane through the camera, which is why
      // the artefact was a hard *straight-edged* wedge sitting on the ground
      // beside the camera at every low altitude, covering roughly the half of
      // the world facing away from the sun. The map is R32F, so a real
      // sentinel costs nothing; Color carries raw floats without clamping.
      renderer.setClearColor(NO_OCCLUDER, 1);
      renderer.clear();
      renderer.render(scene, this.cams[i] as unknown as Camera);
    }

    renderer.setRenderTarget(null);
    casters.forEach((c, k) => {
      c.mesh.material = displayMaterials[k];
      c.mesh.visible = casterVisible[k];
    });
    hide.forEach((o, k) => (o.visible = wasVisible[k]));
  }
}

function smoothstep(a: number, b: number, x: number): number {
  const t = Math.max(0, Math.min(1, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
}
