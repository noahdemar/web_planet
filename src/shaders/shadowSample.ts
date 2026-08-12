/**
 * Shadow lookup, in TSL rather than WGSL.
 *
 * Textures cannot be passed into a `wgslFn`, so the sampling lives here as a
 * node graph and hands a single scalar to the shading functions. The split is
 * a good one: the hand-written maths stays in WGSL, and the plumbing lives
 * where the plumbing tools are.
 *
 * TSL's published types do not model swizzles on uniform or sampled nodes, so
 * this file works through one deliberately loose alias rather than scattering
 * casts through every expression.
 */

import { Matrix4, Vector3, Vector4 } from 'three';
import { Fn, If, float, mix, smoothstep, step, texture, uniform, vec2, vec4 } from 'three/tsl';
import { CASCADES, MAP_SIZE, type Shadows } from '../shadows.js';

/**
 * The subset of the TSL node surface this file uses. Spelling it out is better
 * than reaching for `any`: it documents exactly what is being relied on, and
 * it still fails if one of these disappears.
 */
interface N {
  add(x: unknown): N;
  sub(x: unknown): N;
  abs(): N;
  max(x: unknown): N;
  greaterThan(x: unknown): N;
  greaterThanEqual(x: unknown): N;
  mul(x: unknown): N;
  dot(x: unknown): N;
  length(): N;
  lessThan(x: unknown): N;
  sample(uv: unknown): N;
  assign(x: unknown): void;
  readonly x: N;
  readonly y: N;
  readonly z: N;
  readonly w: N;
  readonly r: N;
  readonly xy: N;
}
const n = (x: unknown): N => x as N;

// Every cast into TSL's own typed builders is contained in these four.
const nStep = (edge: N, x: N): N => n(step(edge as never, x as never));
const nVec4 = (xyz: N, w: number): N => n(vec4(xyz as never, w));
const nVec2 = (a: number, b: number): N => n(vec2(a, b));
const nMix = (a: unknown, b: unknown, t: N): N => n(mix(a as never, b as never, t as never));
const nIf = (cond: N, body: () => void) => If(cond as never, body);
const nSmoothstep = (a: N, b: N, x: N): N =>
  n(smoothstep(a as never, b as never, x as never));

/** 3×3 PCF taps — enough to hide the texel grid without a second pass. */
/**
 * PCF kernel, in shadow-map texels.
 *
 * A 3x3 at one-texel spacing gives a penumbra three texels wide, which on the
 * near cascade is a couple of screen pixels — filtered in principle and hard
 * in practice. Real shadow edges are soft: even direct sun has a half-degree
 * angular diameter, so a metre-high step casts a centimetres-wide penumbra and
 * a hillside casts metres of one.
 *
 * These twelve taps span two and a half texels on a ring-plus-centre layout
 * rather than a grid. A grid kernel leaves the axis-aligned signature of its
 * own lattice in the penumbra — which, on top of a grid-aligned caster, is how
 * a soft edge still manages to look stepped.
 */
const TAPS: readonly (readonly [number, number])[] = [
  [0, 0],
  [-1, 0], [1, 0], [0, -1], [0, 1],
  [-0.7, -0.7], [0.7, -0.7], [-0.7, 0.7], [0.7, 0.7],
  [-2.1, -0.9], [2.1, 0.9], [0.9, -2.1],
];

export interface ShadowUniforms {
  matrices: ReturnType<typeof uniform>[];
  /** (radius0, radius1, radius2, strength) */
  cfg: ReturnType<typeof uniform>;
  sunDir: ReturnType<typeof uniform>;
  /** Call once per frame, after Shadows.update. */
  sync(shadows: Shadows, sun: Vector3): void;
}

export function createShadowUniforms(): ShadowUniforms {
  const matrices = Array.from({ length: CASCADES }, () => uniform(new Matrix4()));
  const cfg = uniform(new Vector4(1, 1, 1, 0));
  const sunDir = uniform(new Vector3(0, 1, 0));

  return {
    matrices,
    cfg,
    sunDir,
    sync(shadows, sun) {
      for (let i = 0; i < CASCADES; i++) {
        (matrices[i].value as Matrix4).copy(shadows.matrices[i]);
      }
      (cfg.value as Vector4).set(
        shadows.radii[0],
        shadows.radii[1],
        shadows.radii[2],
        shadows.strength,
      );
      (sunDir.value as Vector3).copy(sun).normalize();
    },
  };
}

/**
 * Fraction of the sun visible at a camera-relative position, in [0,1].
 *
 * The stored value is distance along the light, so the test is a plain
 * comparison. Bias scales with the cascade's texel footprint: a fixed bias is
 * either acne in the near cascade or peter-panning in the far one.
 */
export function makeShadowFactor(u: ShadowUniforms, shadows: Shadows) {
  const maps = shadows.targets.map((t) => texture(t.texture));

  /** Where this point lands in cascade i's map, in [0,1] if it is covered. */
  function uvOf(i: number, rel: N): N {
    const clip = n(n(u.matrices[i]).mul(nVec4(rel, 1)));
    return n(n(clip.xy).mul(0.5).add(0.5));
  }

  /**
   * How far inside cascade i's map this point is: 1 well inside, 0 at the edge.
   *
   * This is the whole cascade-selection fix. Selection used to be by radial
   * distance — a sphere — while a cascade actually covers a *square*, and one
   * centred 0.65 radii ahead of the camera rather than on it. So ground beside
   * and behind the camera selected the near cascade, projected outside its map,
   * sampled a clamped edge texel and compared against a depth belonging to
   * somewhere else entirely. It came back occluded, and the result was a hard
   * dark wedge with straight edges anchored to the map boundary, sitting on the
   * ground next to the camera at every low altitude.
   *
   * Testing the footprint directly makes the sphere/box mismatch impossible to
   * express, and the smooth shoulder means the handover between cascades and
   * the fall-off past the last one are both gradients rather than seams.
   */
  function coverOf(uv: N): N {
    const q = n(n(uv.sub(0.5)).abs());
    const m = n(n(q.x).max(n(q.y)));
    return n(n(float(1)).sub(nSmoothstep(n(float(0.42)), n(float(0.5)), m)));
  }

  function cascade(i: number, rel: N, radius: N, uv: N): N {
    const frag = n(rel.dot(n(u.sunDir)));

    const texel = n(radius.mul(2 / MAP_SIZE));
    // The shadow pass draws a coarser mesh than the one being shaded, so the
    // bias has to cover that divergence as well as texel quantisation.
    const bias = n(texel.mul(3.0).add(0.35));

    let sum = n(float(0));
    for (const [dx, dy] of TAPS) {
      const s = n(n(maps[i]).sample(n(uv.add(nVec2(dx / MAP_SIZE, dy / MAP_SIZE)))));
      // Lit where the nearest-to-sun surface recorded there is not in front.
      sum = n(sum.add(nStep(n(s.r), n(frag.add(bias)))));
    }
    return n(sum.mul(1 / TAPS.length));
  }

  return Fn(([rel]: [N]) => {
    const cfg = n(u.cfg);
    const lit = n(float(1).toVar());

    const uv0 = uvOf(0, rel);
    const uv1 = uvOf(1, rel);
    const uv2 = uvOf(2, rel);
    const c0 = coverOf(uv0);
    const c1 = coverOf(uv1);
    const c2 = coverOf(uv2);

    // Each cascade fades into the *next one out*, never into "unshadowed".
    //
    // This used to read `mix(1, cascade(0), c0)` and then stop — so anywhere in
    // cascade 0's fade ring, where c0 is between 0 and 1, the shadow was
    // blended toward fully lit and cascade 1 was never consulted, even though
    // it covers that ground perfectly well. The result was a bright
    // quadrilateral drawn on the terrain: the near cascade's own boundary,
    // lit, tracking the camera across the landscape.
    //
    // Cascades nest, so wherever cascade i has partial coverage, i+1 has full
    // coverage — which makes it the correct thing to blend into and turns the
    // seam into a genuine cross-fade. Only the outermost fades to 1, and there
    // there really is no shadow data.
    //
    // The full-coverage case is split out so the common path still evaluates a
    // single cascade. coverOf is flat until 0.42 of the half-extent, so ~84% of
    // each cascade's area pays for one lookup and only the ring pays for two.
    nIf(n(c0.greaterThanEqual(float(1))), () => {
      lit.assign(cascade(0, rel, n(cfg.x), uv0));
    }).Else(() => {
      nIf(n(c0.greaterThan(float(0))), () => {
        lit.assign(nMix(cascade(1, rel, n(cfg.y), uv1),
                        cascade(0, rel, n(cfg.x), uv0), c0));
      }).Else(() => {
        nIf(n(c1.greaterThanEqual(float(1))), () => {
          lit.assign(cascade(1, rel, n(cfg.y), uv1));
        }).Else(() => {
          nIf(n(c1.greaterThan(float(0))), () => {
            lit.assign(nMix(cascade(2, rel, n(cfg.z), uv2),
                            cascade(1, rel, n(cfg.y), uv1), c1));
          }).Else(() => {
            nIf(n(c2.greaterThan(float(0))), () => {
              lit.assign(nMix(float(1), cascade(2, rel, n(cfg.z), uv2), c2));
            });
          });
        });
      });
    });

    // Faded, not switched: shadows would otherwise pop off as you climb.
    return nMix(float(1), lit, n(cfg.w));
  });
}
