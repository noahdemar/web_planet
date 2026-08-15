/**
 * Exact CPU mirror of the gradient noise in src/shaders/terrain.ts.
 *
 * This is a duplicate of shader code, which is normally a smell, and it is
 * here on purpose. The CPU needs the surface for ground-following and for
 * offline tools, and "close enough" is not close enough: the camera stands on
 * whatever the CPU says the ground is, so a disagreement of a few metres is a
 * camera that visibly floats or sinks as you fly.
 *
 * heightCPU.ts has its *own* gradient noise with a different hash, used by the
 * bake for plate warping and base relief. That one does not have to match
 * anything on the GPU. This one does, down to the integer arithmetic — hence
 * the `>>> 0` after every step, which is what makes JavaScript reproduce
 * WGSL's u32 wraparound.
 */

const U32 = 4294967296;

const h = new Uint32Array(3);

/** PCG3D, matching hash33_ in the shader. */
function hash33(ix: number, iy: number, iz: number): void {
  let x = ix >>> 0;
  let y = iy >>> 0;
  let z = iz >>> 0;
  x = (Math.imul(x, 1664525) + 1013904223) >>> 0;
  y = (Math.imul(y, 1664525) + 1013904223) >>> 0;
  z = (Math.imul(z, 1664525) + 1013904223) >>> 0;
  x = (x + Math.imul(y, z)) >>> 0;
  y = (y + Math.imul(z, x)) >>> 0;
  z = (z + Math.imul(x, y)) >>> 0;
  x = (x ^ (x >>> 16)) >>> 0;
  y = (y ^ (y >>> 16)) >>> 0;
  z = (z ^ (z >>> 16)) >>> 0;
  // Sequential, not parallel. Each line feeds the next — `p.y = p.y + p.z *
  // p.x` in the shader reads the p.x written by the line above it, and `p.z`
  // reads both. Writing this round straight into the output array instead
  // evaluates all three from the pre-round values, which is a different
  // function: x still agrees (it is the first assignment either way) and y and
  // z disagree at every lattice point in the domain.
  //
  // That was the CPU/GPU height divergence. Two of the three gradient
  // components were an unrelated field, so `heightAt` and the shader were
  // describing different planets — measured 800 m apart in mountains, which is
  // what put the camera in the air in walk mode and left the trees hanging.
  // The first round has always been written this way; only the last three
  // lines were not, and they read identically to the correct version.
  x = (x + Math.imul(y, z)) >>> 0;
  y = (y + Math.imul(z, x)) >>> 0;
  z = (z + Math.imul(x, y)) >>> 0;
  h[0] = x;
  h[1] = y;
  h[2] = z;
}

/** Gradient noise value, matching noised_(x).x in the shader. */
export function shaderNoise(x: number, y: number, z: number): number {
  const px = Math.floor(x);
  const py = Math.floor(y);
  const pz = Math.floor(z);
  const wx = x - px;
  const wy = y - py;
  const wz = z - pz;
  const ux = wx * wx * wx * (wx * (wx * 6 - 15) + 10);
  const uy = wy * wy * wy * (wy * (wy * 6 - 15) + 10);
  const uz = wz * wz * wz * (wz * (wz * 6 - 15) + 10);

  const corner = (cx: number, cy: number, cz: number): number => {
    hash33(px + cx, py + cy, pz + cz);
    return (
      (-1 + (2 * h[0]) / U32) * (wx - cx) +
      (-1 + (2 * h[1]) / U32) * (wy - cy) +
      (-1 + (2 * h[2]) / U32) * (wz - cz)
    );
  };

  const va = corner(0, 0, 0);
  const vb = corner(1, 0, 0);
  const vc = corner(0, 1, 0);
  const vd = corner(1, 1, 0);
  const ve = corner(0, 0, 1);
  const vf = corner(1, 0, 1);
  const vg = corner(0, 1, 1);
  const vh = corner(1, 1, 1);

  const k0 = va;
  const k1 = vb - va;
  const k2 = vc - va;
  const k3 = ve - va;
  const k4 = va - vb - vc + vd;
  const k5 = va - vc - ve + vg;
  const k6 = va - vb - ve + vf;
  const k7 = -va + vb + vc - vd + ve - vf - vg + vh;

  return (
    k0 + k1 * ux + k2 * uy + k3 * uz +
    k4 * ux * uy + k5 * uy * uz + k6 * uz * ux + k7 * ux * uy * uz
  );
}

/*
 * The amplification constants used to be duplicated here, with a comment on
 * each saying it must match the one in src/shaders/terrain.ts. Both sides now
 * import them from src/planet.ts, so there is nothing left to keep in sync.
 */

/**
 * The same field, with its analytic gradient — the mirror of noised_ in
 * shaders/terrain.ts, and written in that function's 8-corner loop form rather
 * than the collapsed k0..k7 expansion above.
 *
 * The expansion is fine for a value and useless for a derivative: collapsing
 * the corners into k0..k7 throws away the per-corner gradient vectors, and the
 * gradient of gradient noise is not the gradient of its trilinear weights
 * alone — it is `g*W + dot(g,d)*dW` summed over the eight corners, and the
 * first of those two terms is exactly what the k-form discarded.
 *
 * Same hash, same corner order, same quintic as the shader, so the two agree
 * cell for cell. The `fround` discipline in heightCPU applies to the argument
 * before it arrives here, which is where the lattice cell is decided.
 *
 * Returns [value, d/dx, d/dy, d/dz].
 */
export function shaderNoiseD(x: number, y: number, z: number): [number, number, number, number] {
  const px = Math.floor(x);
  const py = Math.floor(y);
  const pz = Math.floor(z);
  const w = [x - px, y - py, z - pz];
  const u = w.map((t) => t * t * t * (t * (t * 6 - 15) + 10));
  // d/dt of the quintic: 30 t^2 (t - 1)^2.
  const du = w.map((t) => 30 * t * t * (t * (t - 2) + 1));

  let acc = 0;
  const grad = [0, 0, 0];
  for (let k = 0; k < 8; k++) {
    const o = [k & 1, (k >> 1) & 1, (k >> 2) & 1];
    hash33(px + o[0], py + o[1], pz + o[2]);
    const g = [-1 + (2 * h[0]) / U32, -1 + (2 * h[1]) / U32, -1 + (2 * h[2]) / U32];
    const d = [w[0] - o[0], w[1] - o[1], w[2] - o[2]];
    const tw = [
      o[0] ? u[0] : 1 - u[0],
      o[1] ? u[1] : 1 - u[1],
      o[2] ? u[2] : 1 - u[2],
    ];
    const dtw = [
      (o[0] * 2 - 1) * du[0],
      (o[1] * 2 - 1) * du[1],
      (o[2] * 2 - 1) * du[2],
    ];
    const wgt = tw[0] * tw[1] * tw[2];
    const dg = g[0] * d[0] + g[1] * d[1] + g[2] * d[2];
    acc += dg * wgt;
    grad[0] += g[0] * wgt + dg * (dtw[0] * tw[1] * tw[2]);
    grad[1] += g[1] * wgt + dg * (tw[0] * dtw[1] * tw[2]);
    grad[2] += g[2] * wgt + dg * (tw[0] * tw[1] * dtw[2]);
  }
  return [acc, grad[0], grad[1], grad[2]];
}
