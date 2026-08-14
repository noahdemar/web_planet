/**
 * Shadow-map inspector.
 *
 * Every shadow bug this project has had was chased through the *consumer* —
 * the bias, the cascade selection, the caster frustum, the patch skirt — and
 * each time the answer was in the map itself, which nobody could see. The
 * distinction that matters is not visible from the final image:
 *
 *   nothing was written here      the texel holds the sentinel, so the depth
 *                                 test passes and the ground reads lit. A
 *                                 *lit* artefact means a missing caster.
 *   something was written here    the texel holds a real distance, so the
 *                                 ground reads shadowed. A *dark* artefact
 *                                 means a wrong caster or a wrong compare.
 *
 * Both look like a straight-edged quadrilateral on the ground. They have
 * opposite causes and opposite fixes, and telling them apart by reasoning from
 * the shaded image has repeatedly gone wrong. So: draw the map.
 *
 * The payload is linear distance along the light plus SHADOW_DEPTH_OFFSET, and
 * the offset exists precisely so that everything written is positive — which
 * makes the sentinel test exact rather than a threshold. Sentinel texels are
 * drawn magenta, written ones as a grey ramp about the cascade centre, so an
 * occluder is a shape and an empty map is a flat magenta field.
 *
 * Off by default and costs nothing when off: it is a second scene, drawn only
 * when a cascade is selected.
 */

import { Mesh, OrthographicCamera, PlaneGeometry, Scene, Vector2 } from 'three';
import { MeshBasicNodeMaterial } from 'three/webgpu';
import { Fn, float, mix, texture, uniform, uv, vec3, vec4 } from 'three/tsl';
import type { WebGPURenderer } from 'three/webgpu';
import { CASCADES, SHADOW_DEPTH_OFFSET, type Shadows } from './shadows.js';

/** Fraction of the viewport height the overlay occupies. */
const SIZE = 0.42;

/**
 * The same deliberately loose node alias `shadowSample.ts` uses, for the same
 * reason: TSL's published types do not model swizzles or comparisons on
 * sampled nodes, and spelling out the surface beats scattering `any`.
 */
interface N {
  add(x: unknown): N;
  sub(x: unknown): N;
  mul(x: unknown): N;
  div(x: unknown): N;
  clamp(a: unknown, b: unknown): N;
  greaterThan(x: unknown): N;
  sample(uv: unknown): N;
  readonly x: N;
  readonly r: N;
}
const n = (x: unknown): N => x as N;

export class ShadowInspector {
  /** -1 for off, otherwise the cascade to display. */
  cascade = -1;

  private scene = new Scene();
  private camera = new OrthographicCamera(-1, 1, 1, -1, 0, 1);
  private meshes: Mesh[] = [];
  /** (halfRange, —) — the distance either side of centre the ramp spans. */
  private cfg = uniform(new Vector2(6000, 0));

  constructor(shadows: Shadows) {
    for (let i = 0; i < CASCADES; i++) {
      const map = n(texture(shadows.targets[i].texture));
      const mat = new MeshBasicNodeMaterial();
      mat.colorNode = Fn(() => {
        const stored = n(map.sample(uv())).r;
        const d = n(stored.sub(float(SHADOW_DEPTH_OFFSET)));
        // Written values are positive by construction — see SHADOW_DEPTH_OFFSET
        // — so anything at or below zero is the clear value and nothing else.
        const written = n(stored.greaterThan(float(0)));
        const half = n(n(this.cfg).x);
        const t = n(n(n(d.div(n(half.mul(2)))).add(float(0.5))).clamp(0, 1));
        const grey = n(vec3(t as never, t as never, t as never));
        const magenta = n(vec3(1, 0, 0.8));
        return n(vec4(n(mix(magenta as never, grey as never, written as never)) as never, 1));
      })() as never;
      // toneMapped would push the ramp through ACES and make the readout lie.
      mat.toneMapped = false;
      mat.depthTest = false;
      mat.depthWrite = false;

      const mesh = new Mesh(new PlaneGeometry(SIZE * 2, SIZE * 2), mat);
      mesh.position.set(1 - SIZE, -1 + SIZE, 0);
      mesh.frustumCulled = false;
      mesh.visible = false;
      this.scene.add(mesh);
      this.meshes.push(mesh);
    }
  }

  /** Distance either side of the cascade centre the grey ramp spans, metres. */
  setRange(metres: number): void {
    (this.cfg.value as Vector2).x = Math.max(1, metres);
  }

  /** Cycle off -> 0 -> 1 -> 2 -> off. */
  cycle(): number {
    this.cascade = this.cascade >= CASCADES - 1 ? -1 : this.cascade + 1;
    return this.cascade;
  }

  /** Draw over the frame just rendered. Call after renderer.render. */
  render(renderer: WebGPURenderer): void {
    if (this.cascade < 0) return;
    // The map is square and the viewport is not, so the quad has to be
    // narrowed by the aspect or the readout is a stretched lie about where
    // things are in the map.
    const el = renderer.domElement;
    const aspect = el.height > 0 ? el.width / el.height : 1;
    this.meshes.forEach((m, i) => {
      m.visible = i === this.cascade;
      m.scale.set(1 / Math.max(aspect, 1e-3), 1, 1);
      m.position.set(1 - SIZE / Math.max(aspect, 1e-3), -1 + SIZE, 0);
    });
    const was = renderer.autoClear;
    renderer.autoClear = false;
    renderer.render(this.scene, this.camera);
    renderer.autoClear = was;
  }
}
