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
  h[0] = (x + Math.imul(y, z)) >>> 0;
  h[1] = (y + Math.imul(z, x)) >>> 0;
  h[2] = (z + Math.imul(x, y)) >>> 0;
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

/** Must match AMP_F0 in src/shaders/terrain.ts. */
export const AMP_F0 = 260;
/** Must match RIDGE_MEAN in src/shaders/terrain.ts; see tools/ridgeMean.ts. */
export const RIDGE_MEAN = 0.7452;
/** Must match AMP_BASE / AMP_RELIEF in src/shaders/terrain.ts. */
export const AMP_BASE = 70;
export const AMP_RELIEF = 7800;
