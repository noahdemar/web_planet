/**
 * Where the planet comes from at run time.
 *
 * Three routes, tried in order:
 *
 *   1. IndexedDB, keyed by seed and resolution. SPEC's determinism invariant is
 *      "bake once, persist, never re-derive" — within one machine the planet
 *      must never silently change under the player — so persistence is not an
 *      optimisation here, it is the invariant.
 *   2. A shipped `public/planet/` asset, if one is present. Kept as a route
 *      because the offline bake still exists and is still the way to produce a
 *      *reference* planet the checked-in baselines describe.
 *   3. Bake it, in a worker, with a progress callback.
 *
 * The point of (3) is not to avoid a download. It is that the seed stops being
 * a build-time constant: a different planet becomes a text field and a wait,
 * rather than a rebuild and a redeploy.
 *
 * Cross-engine determinism is *not* claimed. `Math.pow`, `exp` and `atan2` are
 * not bit-identical across JS engines, so a planet baked in one browser can
 * differ microscopically from one baked in Node — which is why the checked-in
 * baselines describe the Node bake and are validated in CI, not here. SPEC
 * relaxes exactly this ("nobody can tell that two planets differ by a metre")
 * and keeps the within-machine guarantee, which the cache is what provides.
 */

import { surfaceFromBuffer, type Meta, type PlanetSurface } from './planetData.js';
import { BAKE_RES } from './planet.js';
import { DEFAULT_TECTONICS } from './bake/plates.js';
import type { BakeMessage } from './bake/worker.js';

export type BakeProgress = (stage: string, t: number, note?: string) => void;

/** Seed of the planet to build, from `?seed=` if given. */
export function seedFromLocation(): number {
  const raw = new URLSearchParams(location.search).get('seed');
  if (raw === null) return DEFAULT_TECTONICS.seed;
  // Any string is a valid seed — hash the ones that are not numbers, so
  // ?seed=hello is a planet rather than an error.
  const n = Number(raw);
  if (Number.isFinite(n)) return Math.floor(n);
  let h = 2166136261;
  for (let i = 0; i < raw.length; i++) {
    h = Math.imul(h ^ raw.charCodeAt(i), 16777619);
  }
  return h >>> 0;
}

/* ── the cache ───────────────────────────────────────────────────────────── */

const DB = 'world_sim';
const STORE = 'planets';

function idb(): Promise<IDBDatabase | null> {
  return new Promise((resolve) => {
    if (!('indexedDB' in globalThis)) return resolve(null);
    const req = indexedDB.open(DB, 1);
    req.onupgradeneeded = () => req.result.createObjectStore(STORE);
    req.onsuccess = () => resolve(req.result);
    // A private window, a disabled-storage setting or a quota refusal all land
    // here. None of them is fatal — it just means baking every visit.
    req.onerror = () => resolve(null);
  });
}

interface Cached {
  meta: Meta;
  data: Uint16Array;
}

const key = (seed: number): string => `${seed}@${BAKE_RES}`;

async function readCache(seed: number): Promise<Cached | null> {
  const db = await idb();
  if (!db) return null;
  return new Promise((resolve) => {
    const req = db.transaction(STORE, 'readonly').objectStore(STORE).get(key(seed));
    req.onsuccess = () => resolve((req.result as Cached | undefined) ?? null);
    req.onerror = () => resolve(null);
  });
}

async function writeCache(seed: number, value: Cached): Promise<void> {
  const db = await idb();
  if (!db) return;
  await new Promise<void>((resolve) => {
    const tx = db.transaction(STORE, 'readwrite');
    tx.objectStore(STORE).put(value, key(seed));
    // Storing is best-effort: a quota error must not stop a planet that is
    // already in memory from being drawn.
    tx.oncomplete = () => resolve();
    tx.onerror = () => resolve();
    tx.onabort = () => resolve();
  });
}

/* ── the shipped asset, if there is one ──────────────────────────────────── */

async function fromAsset(): Promise<Cached | null> {
  const root = (import.meta as { env?: { BASE_URL?: string } }).env?.BASE_URL ?? '/';
  try {
    const r = await fetch(`${root}planet/meta.json`);
    if (!r.ok) return null;
    const meta = (await r.json()) as Meta;
    // Only useful if it is the planet this build is set up for; surfaceFromBuffer
    // would refuse it anyway, but failing here keeps the fallback silent.
    if (meta.solveRes !== BAKE_RES) return null;
    const b = await fetch(`${root}planet/surface.f16`);
    if (!b.ok) return null;
    return { meta, data: new Uint16Array(await b.arrayBuffer()) };
  } catch {
    return null;
  }
}

/* ── the bake ────────────────────────────────────────────────────────────── */

function bakeInWorker(seed: number, onProgress: BakeProgress): Promise<Cached> {
  return new Promise((resolve, reject) => {
    // Bundled by Vite as a module worker — this URL form is what lets it follow
    // the import graph rather than needing a hand-written bundle.
    const worker = new Worker(new URL('./bake/worker.ts', import.meta.url), {
      type: 'module',
    });
    worker.onmessage = (e: MessageEvent<BakeMessage>) => {
      const m = e.data;
      if (m.kind === 'progress') {
        onProgress(m.stage, m.t);
      } else if (m.kind === 'done') {
        worker.terminate();
        resolve({ meta: m.meta as Meta, data: m.data });
      } else {
        worker.terminate();
        reject(new Error(m.message));
      }
    };
    worker.onerror = (e) => {
      worker.terminate();
      reject(new Error(e.message || 'bake worker failed'));
    };
    worker.postMessage({ seed });
  });
}

/* ── the whole thing ─────────────────────────────────────────────────────── */

export async function planetSurface(
  seed: number,
  onProgress: BakeProgress,
): Promise<PlanetSurface> {
  const cached = await readCache(seed);
  if (cached) {
    onProgress('cached', 1);
    return surfaceFromBuffer(cached.meta, bufferOf(cached.data));
  }

  if (seed === DEFAULT_TECTONICS.seed) {
    const asset = await fromAsset();
    if (asset) {
      onProgress('downloaded', 1);
      await writeCache(seed, asset);
      return surfaceFromBuffer(asset.meta, bufferOf(asset.data));
    }
  }

  const baked = await bakeInWorker(seed, onProgress);
  // Cache before the texture is built: surfaceFromBuffer hands the array to a
  // DataTexture, and the copy stored here has to be taken while it is still
  // plainly ours.
  await writeCache(seed, { meta: baked.meta, data: baked.data.slice() });
  return surfaceFromBuffer(baked.meta, bufferOf(baked.data));
}

/** The exact bytes of a view, without dragging in the rest of its buffer. */
function bufferOf(a: Uint16Array): ArrayBuffer {
  return a.byteOffset === 0 && a.byteLength === a.buffer.byteLength
    ? (a.buffer as ArrayBuffer)
    : (a.slice().buffer as ArrayBuffer);
}
