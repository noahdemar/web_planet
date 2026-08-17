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
import { AURORA_ALT, AURORA_BAND, AURORA_GAIN, AURORA_NIGHT, AURORA_WIDTH, RADIUS } from './planet.js';
import { atmosphere } from './shaders/atmosphere.js';
import { noiseBlock } from './shaders/terrain.js';

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
  col = col + aurora_S(camPos, rd, sd, Rg, cfg.y);
  return col;
}

/**
 * Auroral oval on a shell at AURORA_ALT. See the aurora block in planet.ts.
 *
 * Emission is proportional to path length through the emitting layer, which is
 * why an aurora is faint overhead and a bright wall when seen edge-on at the
 * limb — the same reason the atmosphere itself brightens toward the horizon.
 * Taking it from the ray-shell geometry gets that for free and is most of what
 * makes this read as something at an altitude rather than a decal on the sky.
 */
fn aurora_S(camPos: vec3<f32>, rd: vec3<f32>, sd: vec3<f32>, Rg: f32,
            t: f32) -> vec3<f32> {
  let Ra = Rg + ${AURORA_ALT}.0;
  let b = dot(camPos, rd);
  let c = dot(camPos, camPos) - Ra * Ra;
  let disc = b * b - c;
  if (disc <= 0.0) { return vec3<f32>(0.0); }
  let sq = sqrt(disc);
  // Far root: the emitting layer on the far side of the sky from inside it,
  // and the near limb from outside.
  var tt = -b + sq;
  if (tt <= 0.0) { return vec3<f32>(0.0); }
  let p = camPos + rd * tt;
  let d = normalize(p);

  // Do not draw it through the planet.
  let bg = dot(camPos, rd);
  let cg = dot(camPos, camPos) - Rg * Rg;
  if ((bg * bg - cg) > 0.0 && bg < 0.0 && length(camPos) > Rg) {
    let tg = -bg - sqrt(bg * bg - cg);
    if (tg > 0.0 && tg < tt) { return vec3<f32>(0.0); }
  }

  // Night only, measured where the light is emitted.
  let night = 1.0 - smoothstep(${AURORA_NIGHT}, ${AURORA_NIGHT} + 0.22, dot(d, sd));
  if (night <= 0.001) { return vec3<f32>(0.0); }

  // The oval: a band of latitude in each hemisphere, not a cap.
  let lat = abs(d.y);
  let band = exp(-pow((lat - ${AURORA_BAND}) / ${AURORA_WIDTH}, 2.0));
  if (band <= 0.002) { return vec3<f32>(0.0); }

  // Curtains. Structure runs along the oval and drifts with time, and the
  // rays are much finer across the band than along it — which is the shape of
  // a curtain and the reason it reads as folded sheets rather than as a stain.
  let ang = atan2(d.z, d.x);
  let n1 = noised_S(vec3<f32>(ang * 5.0, lat * 46.0, t * 0.05)).x;
  let n2 = noised_S(vec3<f32>(ang * 17.0 + 11.3, lat * 150.0, t * 0.09)).x;
  var curtain = 0.55 + 0.45 * n1 + 0.35 * n2;
  curtain = max(curtain, 0.0);

  // Path length through the layer, normalised so overhead is 1 and the limb
  // is several. This is the limb brightening.
  let grazing = 1.0 / max(abs(dot(d, rd)), 0.09);

  // Green below, red-magenta above: the 557.7 nm line dominates the lower
  // layer and 630 nm the top, so the gradient runs with height through the
  // curtain rather than being a colour ramp chosen by eye.
  let hi = smoothstep(${AURORA_BAND} - ${AURORA_WIDTH}, ${AURORA_BAND} + ${AURORA_WIDTH} * 1.4, lat);
  let tint = mix(vec3<f32>(0.16, 1.00, 0.42), vec3<f32>(0.72, 0.28, 0.95), hi * 0.55);

  return tint * (band * curtain * grazing * night * ${AURORA_GAIN});
}
${atmosphere('S')}
${noiseBlock('S')}
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
  update(camX: number, camY: number, camZ: number, near: number, timeSec = 0): void {
    this.camPos.value.set(camX, camY, camZ);
    this.radius.value = Math.max(10, near * 20);
    // Drives the auroral curtains, and nothing else in here moves.
    this.cfg.value.y = timeSec;
  }

  setSun(d: Vector3, colour: Vector3): void {
    this.sun.value.copy(d).normalize();
    this.sunCol.value.copy(colour);
  }
}
