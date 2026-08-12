/**
 * Cloud deck. See SPEC.md §7.
 *
 * The README has listed "no clouds" as a known gap since M1, with the note
 * that the Blue Marble is roughly half cloud — which is the whole reason the
 * planet read as a terrain demo from orbit rather than as a world.
 *
 * This is deliberately the cheapest thing that gets the *signature* rather
 * than the physics: one alpha-blended shell at CLOUD_ALT, coverage from five
 * noise octaves, banded by latitude. No raymarching, no volume, no second
 * depth pass. What it buys, in order of how much it matters from orbit:
 *
 *   the terminator      cloud tops stay lit after the ground below has gone
 *                       dark, which is most of what makes a limb read as 3D
 *   the banding         a dry subtropical belt with wet equator and storm
 *                       tracks either side — the most recognisable thing
 *                       about Earth from space, and free here because the
 *                       moisture field already has the same Hadley structure
 *   scale reference     cloud is the only feature whose size the eye already
 *                       knows, so it calibrates everything under it
 *
 * The shape function is evaluated from the *unit sphere position*, which is
 * exact — no precision reconstruction is needed anywhere in this file, unlike
 * the terrain, because nothing here is differenced against the planet radius
 * except the draw position, and a cloud edge that moves by the f32 ULP at
 * 6.4e6 (0.76 m, at 6.5 km distance) is far inside a pixel.
 */

import { DoubleSide, Mesh, SphereGeometry, Vector3, Vector4 } from 'three';
import { MeshBasicNodeMaterial } from 'three/webgpu';
import { normalLocal, positionLocal, uniform, varying, vec3, vec4, wgslFn } from 'three/tsl';
import { CLOUD_ALT, CLOUD_THICK, RADIUS } from './planet.js';
import { atmosphere } from './shaders/atmosphere.js';
import { cloudFieldBlock, noiseBlock } from './shaders/terrain.js';

type Vec3Node = ReturnType<typeof vec3>;
type Vec4Node = ReturnType<typeof vec4>;
const asVec3 = (n: unknown): Vec3Node => n as Vec3Node;
const asVec4 = (n: unknown): Vec4Node => n as Vec4Node;

const cloudShade = wgslFn(/* wgsl */ `
fn cloudShade(dirIn: vec3<f32>, camPos: vec3<f32>, sunDir: vec3<f32>,
              sunCol: vec3<f32>, cfg: vec4<f32>) -> vec4<f32> {
  let dir = normalize(dirIn);
  let sd = normalize(sunDir);
  let Rg = cfg.x;
  let t = cfg.y;
  let cover = cfg.z;

  // Both layers from the shared field, so the shadow the ground receives comes
  // from exactly the cloud that is drawn here. See cloudFieldBlock.
  // Ground size of a pixel on the deck, so the field draws only the octaves
  // this view can resolve. fwidth on the shell position is the same measure
  // the terrain uses.
  let px = max(length(fwidth(dir)) * (Rg + ${CLOUD_ALT}.0), 1.0);
  let f = cloudField_C(dir, t, cover, px);
  var cu = f.x;
  let ci = f.y;

  // ── parallax: give the shell a thickness it does not have ─────────────
  //
  // One alpha shell has no vertical extent, so the deck reads as a texture on
  // glass however good the coverage field is. What tells the eye a cumulus is
  // an object is that its top is *offset* from its base whenever you are not
  // looking straight down it — and that offset is pure geometry, no volume
  // required.
  //
  // The tops sit CLOUD_THICK above the base, so seen along the view ray they
  // are displaced tangentially by thickness / |cos of the view angle from
  // vertical|. Sampling the same field there and taking the greater coverage
  // extends the silhouette in the direction the tower leans away from the
  // viewer, which is exactly what a real cumulus field does at the limb and at
  // any oblique angle. Straight down it collapses to nothing, correctly.
  //
  // The 0.30 floor on the vertical term stops the offset running away at the
  // horizon, where the geometry says infinity and the honest answer is that
  // the cloud is edge-on and something else is in the way.
  let vd = normalize(dirIn * (Rg + ${CLOUD_ALT}.0) - camPos);
  let vUp = dot(vd, dir);
  let vT = vd - dir * vUp;
  let lean = length(vT) / max(abs(vUp), 0.30) * ${CLOUD_THICK}.0;
  if (lean > 1.0) {
    let dirTop = normalize(dir - (vT / max(length(vT), 1e-6)) * (lean / (Rg + ${CLOUD_ALT}.0)));
    let cuTop = cloudField_C(dirTop, t, cover, px).x;
    // The tower is brighter than the base it stands on: this is the sunlit
    // flank coming into view, not extra coverage in the same plane.
    cu = max(cu, cuTop * 0.92);
  }

  // Opacity from *thickness*, not from coverage directly. A cumulus is a
  // volume: it is nearly transparent where it tapers to nothing and opaque
  // through the middle, and squaring the coverage is the cheapest stand-in for
  // the path length through it. The previous version used the coverage itself,
  // which gave every cloud the same flat opacity right up to a hard edge, and
  // that is most of why the deck read as a decal.
  let thick = cu * cu;
  let alpha = clamp(thick + ci * 0.30 * (1.0 - cu), 0.0, 1.0);
  if (alpha < 0.004) { discard; }

  // ── shading ─────────────────────────────────────────────────────────────
  let wp = dir * (Rg + ${CLOUD_ALT}.0);
  let sunEl = dot(dir, sd);
  let ndl = max(sunEl, 0.0);
  let sunTr = transmit_C(sunDepth_C(wp, sd, Rg));

  // Is there any sun on this piece of deck at all?
  //
  // Three terms below carry a deliberate floor so the deck does not go black
  // the instant the direct term does — a cloud is lit from the sides, from
  // below, and by the whole sky, and the terminator has to be a gradient. All
  // three were *constants*, which is right at the terminator and wrong sixty
  // degrees into the night: over the anti-solar point the deck came out as
  // bright as the day side. That is what the dark side blowing out to white
  // actually was — the white was the clouds, and the exposure clamp was only
  // amplifying them.
  //
  // The window is physical rather than tuned. The deck sits at CLOUD_ALT, so
  // it is still geometrically in sunlight about 2.7 deg past the ground's own
  // terminator, and the sky above it keeps scattering for a few degrees more.
  // Below that there is no light source left to floor anything to.
  let twilight = smoothstep(-0.18, 0.10, sunEl);

  // Cloud is bright and close to neutral. The floor is not ambient fudge: a
  // deck this thick is lit from the sides and from below by the ground, so it
  // never goes fully black on the day side, and the terminator is a gradient
  // rather than an edge.
  //
  // The bracket must not exceed 1. It did — (0.30 + 1.15·ndl) peaks at 1.45 —
  // and with a 0.86 albedo that made cloud eight times brighter than the
  // ground rather than the ~5x its albedo ratio earns. ACES has a warm
  // shoulder, so the excess did not clip to white, it turned the whole deck
  // tan, and from orbit that read as a dust planet.
  let alb = vec3<f32>(0.74, 0.75, 0.78);
  var col = alb * (1.0 / 3.14159265) * sunCol * sunTr * (0.10 * twilight + 0.90 * ndl);

  // Self-shadowing, and it is what gives a cumulus field its shape. Real cloud
  // is dark underneath and brilliant on top because sunlight is scattered out
  // long before it reaches the base; without a term for it every cloud is a
  // uniformly bright blob and the deck has no relief at all. Thickness is the
  // proxy for depth into the cloud, and the effect is strongest where the sun
  // is high because that is when the path to the base is longest.
  let sunUp = max(dot(dir, sd), 0.0);
  let base = 1.0 - 0.55 * thick * (0.35 + 0.65 * sunUp);
  col = col * base;

  // ── relief: which flank faces the sun ──────────────────────────────────
  //
  // The term above darkens by thickness, which is isotropic — it makes a cloud
  // darker in the middle and lighter at the rim whatever the sun is doing, so
  // the deck reads as a flat mask with soft edges rather than as a field of
  // lumps. What gives a cumulus field its shape from orbit is that every cell
  // has a brilliant sunward flank and a shadowed one, and its neighbour casts
  // across it.
  //
  // Both come from one extra sample of the same field, one cell toward the
  // sun along the deck. Where the field is *thicker* that way, something
  // taller stands between this point and the sun; where it is thinner, this
  // point is the sunward flank. The difference is a directional derivative, so
  // this is bump lighting on the coverage field — no volume, no march, and it
  // costs one field evaluation on a shell that covers a fraction of the frame.
  //
  // The step is tied to px so it is always a resolvable distance: a step finer
  // than the octaves being drawn returns the same value twice and the term
  // vanishes.
  let sunT = sd - dir * dot(dir, sd);
  let sunLen = length(sunT);
  if (sunLen > 1e-4) {
    let stepR = max(px * 6.0, 2500.0) / (Rg + ${CLOUD_ALT}.0);
    let dSun = normalize(dir + (sunT / sunLen) * stepR);
    let cuSun = cloudField_C(dSun, t, cover, px).x;
    // Positive where this point stands proud of what is between it and the
    // sun. Scaled hard because the coverage field is smooth and the useful
    // range of the difference is small.
    let relief = clamp((cu - cuSun) * 2.6, -1.0, 1.0);
    // Only where there is cloud to shade, and strongest at grazing sun, which
    // is when a flank actually catches the light edge-on.
    col = col * (1.0 + 0.42 * relief * cu * (1.0 - 0.45 * sunUp));
  }

  // Skylight, and it is not a small term. A cloud deck sees a whole hemisphere
  // of sky, and with the sun only 38° up — which is the default, and follows
  // the camera — most of the visible disc is near the terminator, where the
  // direct term has collapsed and sunTr has gone deep red from the grazing
  // path. Lit by the sun alone the deck came out dark brown over half the
  // planet while the ground beneath it looked fine, because the ground is
  // brown anyway and cloud is not.
  col = col + alb * sunCol * vec3<f32>(0.055, 0.085, 0.155) * 0.30 * twilight;

  // Thin edges scatter forward and go bright against the sun — the silver
  // lining. Strongest exactly where the cloud is thinnest, which is why it is
  // weighted by (1 − thick) and not by alpha.
  let fwd = pow(max(dot(normalize(camPos - wp), -sd), 0.0), 8.0);
  col = col + sunCol * sunTr * fwd * 0.22 * (1.0 - thick);

  // Cirrus is ice: brighter, bluer, and it barely self-shadows.
  let ciCol = vec3<f32>(0.86, 0.88, 0.95) * (1.0 / 3.14159265) * sunCol * sunTr
            * (0.25 * twilight + 0.75 * ndl);
  col = mix(col, ciCol, clamp(ci * (1.0 - cu) * 0.8, 0.0, 1.0));

  col = aerial_C(col, camPos, wp, sd, Rg, sunCol);
  return vec4<f32>(col, alpha);
}
${noiseBlock('C')}
${cloudFieldBlock('C')}
${atmosphere('C')}
`);

export class Clouds {
  readonly mesh: Mesh;

  private camPos = uniform(new Vector3());
  private sun = uniform(new Vector3(0.62, 0.28, 0.73).normalize());
  private sunCol = uniform(new Vector3(1, 0.97, 0.92));
  /** (planetRadius, time, coverage, —) */
  private cfg = uniform(new Vector4(RADIUS, 0, 0.5, 0));

  constructor() {
    // Finer than the sky dome: this shell is only 6.5 km away at ground level,
    // where the sky dome is effectively at infinity, so its silhouette against
    // the horizon would facet at the sky's tessellation.
    const geo = new SphereGeometry(RADIUS + CLOUD_ALT, 128, 64);

    const mat = new MeshBasicNodeMaterial();
    // The outward planet normal, from the geometry's own normal attribute —
    // for a sphere about the origin that is exactly the unit direction.
    //
    // It must NOT come from `positionLocal`: that resolves to whatever
    // `positionNode` sets, which here is the camera-relative position. Reading
    // it gave the camera-to-cloud direction instead of the surface normal, so
    // looking down at the deck N·L was dot(-up, sun) — negative everywhere,
    // clamped to zero — and every lit cloud rendered as unlit, reddened by a
    // grazing sun path that was never actually grazing. The sky dome has
    // always used `normalLocal` for the same reason.
    const dir = varying(normalLocal, 'vCloudDir');
    mat.positionNode = asVec3(positionLocal.sub(this.camPos));

    const shaded = asVec4(
      cloudShade({
        dirIn: dir,
        camPos: this.camPos,
        sunDir: this.sun,
        sunCol: this.sunCol,
        cfg: this.cfg,
      }),
    );
    mat.colorNode = asVec3(shaded.xyz);
    mat.opacityNode = shaded.w;
    mat.transparent = true;
    // Seen from below as often as from above, and the shell is thin enough
    // that writing depth would let it occlude mountains that are genuinely in
    // front of it.
    mat.side = DoubleSide;
    mat.depthWrite = false;
    mat.toneMapped = true;

    this.mesh = new Mesh(geo, mat);
    this.mesh.frustumCulled = false;
    this.mesh.matrixAutoUpdate = false;
    // After the terrain, so it composites over the ground it is above.
    this.mesh.renderOrder = 500;
  }

  update(camX: number, camY: number, camZ: number, timeSec: number): void {
    this.camPos.value.set(camX, camY, camZ);
    this.cfg.value.y = timeSec;
  }

  setSun(d: Vector3, colour: Vector3): void {
    this.sun.value.copy(d).normalize();
    this.sunCol.value.copy(colour);
  }

  /** 0 = clear, 1 = overcast. */
  setCoverage(v: number): void {
    this.cfg.value.z = Math.max(0, Math.min(1, v));
  }
  get coverage(): number {
    return this.cfg.value.z;
  }
}
