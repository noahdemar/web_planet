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
import { Fn, If, float, mix, step, texture, uniform, vec2, vec4 } from 'three/tsl';
import { CASCADES, MAP_SIZE, type Shadows } from '../shadows.js';

/**
 * The subset of the TSL node surface this file uses. Spelling it out is better
 * than reaching for `any`: it documents exactly what is being relied on, and
 * it still fails if one of these disappears.
 */
interface N {
  add(x: unknown): N;
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

/** 3×3 PCF taps — enough to hide the texel grid without a second pass. */
const TAPS: readonly (readonly [number, number])[] = [
  [-1, -1], [0, -1], [1, -1],
  [-1, 0], [0, 0], [1, 0],
  [-1, 1], [0, 1], [1, 1],
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

  function cascade(i: number, rel: N, radius: N): N {
    const clip = n(n(u.matrices[i]).mul(nVec4(rel, 1)));
    const uv = n(n(clip.xy).mul(0.5).add(0.5));
    const frag = n(rel.dot(n(u.sunDir)));

    const texel = n(radius.mul(2 / MAP_SIZE));
    const bias = n(texel.mul(2.5).add(0.05));

    let sum = n(float(0));
    for (const [dx, dy] of TAPS) {
      const s = n(n(maps[i]).sample(n(uv.add(nVec2(dx / MAP_SIZE, dy / MAP_SIZE)))));
      // Lit where the nearest-to-sun surface recorded there is not in front.
      sum = n(sum.add(nStep(n(s.r), n(frag.add(bias)))));
    }
    return n(sum.mul(1 / TAPS.length));
  }

  return Fn(([rel]: [N]) => {
    const d = n(rel.length());
    const cfg = n(u.cfg);
    const lit = n(float(1).toVar());

    // Nested rather than chained: cascade selection has to be a strict
    // first-match, and the ranges overlap by design.
    nIf(n(d.lessThan(n(cfg.x))), () => {
      lit.assign(cascade(0, rel, n(cfg.x)));
    }).Else(() => {
      nIf(n(d.lessThan(n(cfg.y))), () => {
        lit.assign(cascade(1, rel, n(cfg.y)));
      }).Else(() => {
        nIf(n(d.lessThan(n(cfg.z))), () => {
          lit.assign(cascade(2, rel, n(cfg.z)));
        });
      });
    });

    // Faded, not switched: shadows would otherwise pop off as you climb.
    return nMix(float(1), lit, n(cfg.w));
  });
}
