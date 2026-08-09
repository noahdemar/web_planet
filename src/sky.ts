/**
 * Sky dome. See SPEC.md §7.
 *
 * A back-faced sphere at the camera, drawn first with no depth interaction, so
 * everything else paints over it. Its radius tracks the near plane because the
 * near plane spans centimetres to hundreds of kilometres across the altitude
 * range — a fixed radius would sit inside the near plane at orbit and be
 * clipped away entirely.
 *
 * It shares `atmosphere()` with the terrain and vegetation shaders, so the
 * horizon is continuous by construction: the air a ground pixel looks through
 * and the air a sky pixel looks through are the same integral.
 */

import { BackSide, Mesh, SphereGeometry, Vector3, Vector4 } from 'three';
import { MeshBasicNodeMaterial } from 'three/webgpu';
import { normalLocal, positionLocal, uniform, varying, vec3 } from 'three/tsl';
import { wgslFn } from 'three/tsl';
import { RADIUS } from './planet.js';
import { atmosphere } from './shaders/atmosphere.js';

type Vec3Node = ReturnType<typeof vec3>;
const asVec3 = (n: unknown): Vec3Node => n as Vec3Node;

const skyColour = wgslFn(/* wgsl */ `
fn skyColour(dirIn: vec3<f32>, camPos: vec3<f32>, sunDir: vec3<f32>,
             sunCol: vec3<f32>, cfg: vec4<f32>) -> vec3<f32> {
  let rd = normalize(dirIn);
  let sd = normalize(sunDir);
  let Rg = cfg.x;

  var col = skyRadiance_S(camPos, rd, sd, Rg, sunCol);

  // Solar disc, with a soft edge about the true angular radius (~0.27°). Only
  // drawn when the ray misses the planet, so the sun cannot shine through it.
  let b = dot(camPos, rd);
  let cg = dot(camPos, camPos) - Rg * Rg;
  let hitsGround = (b * b - cg) > 0.0 && b < 0.0;
  if (!hitsGround) {
    let c = dot(rd, sd);
    let disc = smoothstep(0.99996, 0.99999, c);
    // Attenuated by the air along the view ray, which is what reddens it low
    // in the sky without any special-casing.
    let tr = transmit_S(sunDepth_S(camPos + rd * 1000.0, rd, Rg));
    col = col + sunCol * tr * disc * 8.0;
  }
  return col;
}
${atmosphere('S')}
`);

export class Sky {
  readonly mesh: Mesh;

  private camPos = uniform(new Vector3());
  private sun = uniform(new Vector3(0.62, 0.28, 0.73).normalize());
  private sunCol = uniform(new Vector3(1, 0.97, 0.92));
  private cfg = uniform(new Vector4(RADIUS, 0, 0, 0));
  private radius = uniform(1000);

  constructor() {
    // Modest tessellation: the sky is smooth, and the ray direction is a
    // normalised interpolant so faceting does not show.
    const geo = new SphereGeometry(1, 32, 24);

    const mat = new MeshBasicNodeMaterial();
    mat.positionNode = asVec3(positionLocal.mul(this.radius));
    mat.colorNode = asVec3(
      skyColour({
        dirIn: varying(normalLocal, 'vSkyDir'),
        camPos: this.camPos,
        sunDir: this.sun,
        sunCol: this.sunCol,
        cfg: this.cfg,
      }),
    );
    mat.side = BackSide;
    mat.depthWrite = false;
    mat.depthTest = false;
    mat.toneMapped = true;

    this.mesh = new Mesh(geo, mat);
    this.mesh.frustumCulled = false;
    this.mesh.matrixAutoUpdate = false;
    this.mesh.renderOrder = -1000;
  }

  /** `near` keeps the dome outside the near plane at every altitude. */
  update(camX: number, camY: number, camZ: number, near: number): void {
    this.camPos.value.set(camX, camY, camZ);
    this.radius.value = Math.max(10, near * 20);
  }

  setSun(d: Vector3, colour: Vector3): void {
    this.sun.value.copy(d).normalize();
    this.sunCol.value.copy(colour);
  }
}
