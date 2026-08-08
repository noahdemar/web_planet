/**
 * Precision verification for the patch reconstruction. SPEC.md M0/M1.
 *
 *   npm run precision
 *
 * Reproduces the shader's arithmetic in emulated f32 (Math.fround after every
 * operation) and compares it against an f64 ground truth, alongside what the
 * naive `normalize(warp(uv)) * R - camPos` formulation would produce.
 *
 * This is the numerical form of M1's "no precision jitter" criterion: the
 * screenshot shows the grid does not swim, this shows why.
 */

import { FACES, warp } from '../src/cubesphere.ts';
import { RADIUS } from '../src/planet.ts';
import type { V3 } from '../src/math/vec3d.ts';

const f = Math.fround;
const QP = Math.PI / 4;

const dot3 = (a: V3, b: V3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

/** f32-emulated vector helpers. */
const fAdd = (a: V3, b: V3): V3 => [f(a[0] + b[0]), f(a[1] + b[1]), f(a[2] + b[2])];
const fSub = (a: V3, b: V3): V3 => [f(a[0] - b[0]), f(a[1] - b[1]), f(a[2] - b[2])];
const fMul = (a: V3, s: number): V3 => [f(a[0] * s), f(a[1] * s), f(a[2] * s)];
const fDot = (a: V3, b: V3) => f(f(f(a[0] * b[0]) + f(a[1] * b[1])) + f(a[2] * b[2]));
const fLen = (a: V3) => f(Math.sqrt(fDot(a, a)));
const f32v = (a: V3): V3 => [f(a[0]), f(a[1]), f(a[2])];

function cube(face: number, wu: number, wv: number): V3 {
  const { U, V, W } = FACES[face];
  return [
    U[0] * wu + V[0] * wv + W[0],
    U[1] * wu + V[1] * wv + W[1],
    U[2] * wu + V[2] * wv + W[2],
  ];
}

function norm(a: V3): V3 {
  const l = Math.hypot(a[0], a[1], a[2]);
  return [a[0] / l, a[1] / l, a[2] / l];
}

const FACE = 4;
const ALT = 1.7; // eye height
// An off-centre patch: A and B both non-zero exercises the tangent-addition
// path properly. Dead centre would hide errors behind A = B = 0.
const FRAC_U = 0.37;
const FRAC_V = -0.61;

interface Row {
  level: number;
  edge: number;
  ours: number;
  naive: number;
}

function measure(level: number): Row {
  const n = 1 << level;
  const hs = 1 / n;
  // Snap the patch centre onto the level's grid, near the chosen fractions.
  const i = Math.min(n - 1, Math.max(0, Math.round(((FRAC_U + 1) / 2) * n - 0.5)));
  const j = Math.min(n - 1, Math.max(0, Math.round(((FRAC_V + 1) / 2) * n - 0.5)));
  const cu = -1 + (2 * i + 1) * hs;
  const cv = -1 + (2 * j + 1) * hs;

  // ── f64 reference frame ───────────────────────────────────────────────
  const A = warp(cu);
  const B = warp(cv);
  const Pc64 = cube(FACE, A, B);
  const lenPc64 = Math.hypot(Pc64[0], Pc64[1], Pc64[2]);
  const dirC64 = norm(Pc64);
  const camPos: V3 = [
    dirC64[0] * (RADIUS + ALT),
    dirC64[1] * (RADIUS + ALT),
    dirC64[2] * (RADIUS + ALT),
  ];

  // The one f64 subtraction the CPU performs per patch, then rounded to f32.
  const anchorRel = f32v([
    dirC64[0] * RADIUS - camPos[0],
    dirC64[1] * RADIUS - camPos[1],
    dirC64[2] * RADIUS - camPos[2],
  ]);

  // f32 versions of the per-instance attributes, as uploaded.
  const A32 = f(A);
  const B32 = f(B);
  const Pc32 = f32v(cube(FACE, A32, B32));
  const lenPc32 = fLen(Pc32);
  const BU = f32v(FACES[FACE].U as V3);
  const BV = f32v(FACES[FACE].V as V3);
  const camPos32 = f32v(camPos);

  let worstOurs = 0;
  let worstNaive = 0;

  const S = 9;
  for (let a = 0; a < S; a++) {
    for (let b = 0; b < S; b++) {
      const gx = (a / (S - 1)) * 2 - 1;
      const gy = (b / (S - 1)) * 2 - 1;

      // ── ground truth, f64 ────────────────────────────────────────────
      const u = cu + hs * gx;
      const v = cv + hs * gy;
      const dirT = norm(cube(FACE, warp(u), warp(v)));
      const truth: V3 = [
        dirT[0] * RADIUS - camPos[0],
        dirT[1] * RADIUS - camPos[1],
        dirT[2] * RADIUS - camPos[2],
      ];

      // ── the shader's algorithm, emulated in f32 ──────────────────────
      const ta = f(Math.tan(f(f(hs * gx) * QP)));
      const tb = f(Math.tan(f(f(hs * gy) * QP)));
      const dwu = f(f(ta * f(1 + f(A32 * A32))) / f(1 - f(A32 * ta)));
      const dwv = f(tb * f(1 + f(B32 * B32)) / f(1 - f(B32 * tb)));
      const dP = fAdd(fMul(BU, dwu), fMul(BV, dwv));

      const inv = f(1 / lenPc32);
      const sq = f(f(f(2 * fDot(Pc32, dP)) + fDot(dP, dP)) * f(inv * inv));
      const k = f(sq / f(1 + f(Math.sqrt(f(1 + sq)))));
      const dd = fMul(fSub(dP, fMul(Pc32, k)), f(inv / f(1 + k)));
      const ours = fAdd(anchorRel, fMul(dd, RADIUS));

      // ── the naive formulation, emulated in f32 ───────────────────────
      const nu = f(cu + f(hs * gx));
      const nv = f(cv + f(hs * gy));
      const nP = f32v(cube(FACE, f(Math.tan(f(nu * QP))), f(Math.tan(f(nv * QP)))));
      const nl = fLen(nP);
      const nDir: V3 = [f(nP[0] / nl), f(nP[1] / nl), f(nP[2] / nl)];
      const naive = fSub(fMul(nDir, RADIUS), camPos32);

      worstOurs = Math.max(worstOurs, Math.hypot(...fSub(ours, truth as V3)));
      worstNaive = Math.max(worstNaive, Math.hypot(...fSub(naive, truth as V3)));
    }
  }

  return {
    level,
    edge: (Math.sqrt((4 * Math.PI * RADIUS * RADIUS) / 6) / n),
    ours: worstOurs,
    naive: worstNaive,
  };
}

const fmt = (m: number): string => {
  if (m < 1e-6) return `${(m * 1e9).toFixed(1)} nm`;
  if (m < 1e-3) return `${(m * 1e6).toFixed(1)} µm`;
  if (m < 1) return `${(m * 1e3).toFixed(2)} mm`;
  if (m < 1000) return `${m.toFixed(2)} m`;
  return `${(m / 1000).toFixed(1)} km`;
};

/**
 * The meaningful criterion is error relative to *that level's* own sample
 * spacing. Sub-millimetre error on a 9220 km patch is irrelevant; the same
 * error at L19, where samples sit 6.9 cm apart, would be ruinous.
 */
const TOLERANCE = 0.001; // 0.1% of the level's ground sample distance

console.log(`
  Vertex position error against an f64 ground truth.
  Camera at ${ALT} m on face +Z; worst of 81 samples per patch;
  terrain height held at zero to isolate the geometry.

                            this build              naive f32
  level     GSD          error   (xGSD)        error   (xGSD)
  ---------------------------------------------------------------------`);

let worstRatio = 0;
let naiveBreaksAt = -1;

for (const l of [0, 2, 4, 6, 8, 10, 12, 14, 16, 17, 18, 19]) {
  const r = measure(l);
  const gsd = r.edge / 256;
  const ro = r.ours / gsd;
  const rn = r.naive / gsd;
  worstRatio = Math.max(worstRatio, ro);
  if (naiveBreaksAt < 0 && rn > 1) naiveBreaksAt = l;

  console.log(
    `  ${String(r.level).padStart(4)}` +
      `${fmt(gsd).padStart(12)}` +
      `${fmt(r.ours).padStart(15)}` +
      `${ro.toExponential(1).padStart(10)}` +
      `${fmt(r.naive).padStart(14)}` +
      `${(rn < 1 ? rn.toExponential(1) : `${rn.toFixed(1)} BROKEN`).padStart(16)}`,
  );
}

console.log(`
  worst error, any level        ${worstRatio.toExponential(2)} x GSD    (tolerance ${TOLERANCE})
  naive f32 first exceeds its own GSD at level ${naiveBreaksAt >= 0 ? naiveBreaksAt : 'never'}; beyond that it
  cannot represent the surface it is drawing, at any tessellation.
`);

if (worstRatio > TOLERANCE) {
  console.error('  FAIL: error exceeds tolerance at some level\n');
  process.exitCode = 1;
} else {
  console.log('  PASS\n');
}
