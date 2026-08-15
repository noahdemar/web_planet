/**
 * The adaptive resolution scaler, against the cases that actually break it.
 *
 * This exists because the scaler shipped with a bug that no amount of looking
 * at it would have found, and that only showed up against a real machine's
 * numbers: it steered by wall-clock frame time, which under vsync is quantised
 * to the refresh interval. A healthy frame on a 60 Hz panel measures 16.7 ms
 * whether the GPU spent 4 ms or 15 ms on it, so "comfortably under a 15 ms
 * budget" was unreachable and the scaler could drop resolution but never
 * restore it. The recovery cases below are the regression.
 *
 *   npm run scaler
 */

import { AdaptiveResolution, type QualityTier } from '../src/quality.js';

/**
 * The low tier's numbers, stated here rather than imported: this checks the
 * scaler's behaviour, and it should not start failing because a preset was
 * retuned.
 */
const TIER: QualityTier = {
  tier: 'low', reason: 'test', handheld: true,
  maxPixelRatio: 1, antialias: false,
  octaves: 9, lodFactor: 1.5, maxLevel: 17,
  shadows: true, shadowMapSize: 1024, shadowCascades: 2,
  shadowLodFactor: 1.5, shadowMaxLevel: 15,
  vegetation: true, vegDensity: 0.45, vegBandCapacity: 150_000, grass: false,
  frameBudgetMs: 15, gpuBudgetMs: 11, minRenderScale: 0.5,
};

let failures = 0;

function check(label: string, ok: boolean, detail: string): void {
  console.log(`  ${ok ? 'ok  ' : 'FAIL'}  ${label.padEnd(52)} ${detail}`);
  if (!ok) failures++;
}

/** Runs `frames` frames at a fixed dt/gpu pair and returns the scaler. */
function soak(frames: number, dtMs: number, gpuMs: number, r?: AdaptiveResolution) {
  const s = r ?? new AdaptiveResolution(() => {}, TIER);
  for (let i = 0; i < frames; i++) s.update(dtMs, gpuMs);
  return s;
}

console.log('\n  adaptive resolution scaler\n');

// ── GPU time, the signal it prefers ────────────────────────────────────────
{
  const r = soak(900, 16.7, 4);
  check('idle GPU under budget holds full resolution', r.scale === 1, `scale ${r.scale}, src ${r.source}`);
  check('  and steers by the GPU clock', r.source === 'gpu', `src ${r.source}`);
}
{
  const r = soak(900, 16.7, 18);
  check('sustained GPU overrun drops resolution', r.scale < 1, `scale ${r.scale}`);
}

// ── the regression ─────────────────────────────────────────────────────────
{
  const r = soak(600, 16.7, 20);
  const dropped = r.scale;
  soak(3000, 16.7, 3, r);
  check(
    'recovers once the load clears (vsync regression)',
    dropped < 1 && r.scale > dropped,
    `${dropped.toFixed(2)} -> ${r.scale.toFixed(2)}`,
  );
}

// ── wall-clock fallback, for browsers without timestamp queries ────────────
{
  const r = soak(900, 16.7, 0);
  check('fallback: vsync-pinned dt is not read as overrun', r.scale === 1, `scale ${r.scale}, src ${r.source}`);
  check('  and falls back rather than probing forever', r.source === 'wall', `src ${r.source}`);
}
{
  const r = soak(900, 33.4, 0);
  const dropped = r.scale;
  soak(4000, 16.7, 0, r);
  check('fallback: drops on missed frames, then recovers', dropped < 1 && r.scale > dropped,
    `${dropped.toFixed(2)} -> ${r.scale.toFixed(2)}`);
}

// ── bounds ─────────────────────────────────────────────────────────────────
{
  const r = soak(6000, 16.7, 200);
  check('never scales below minRenderScale', r.scale >= TIER.minRenderScale, `scale ${r.scale}`);
}
{
  const r = soak(900, 16.7, 4);
  check('never scales above 1', r.scale <= 1, `scale ${r.scale}`);
}
{
  // A backgrounded tab returns multi-second gaps that describe nothing.
  const r = soak(900, 5000, 0);
  check('ignores backgrounded-tab gaps', r.scale === 1, `scale ${r.scale}`);
}

console.log(failures === 0 ? '\n  scaler ok\n' : `\n  ${failures} FAILED\n`);
process.exit(failures === 0 ? 0 : 1);
