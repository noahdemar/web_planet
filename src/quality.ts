/**
 * Device tiers, and the render scale that tracks the frame clock.
 *
 * Every calibrated number in this project was chosen on an M-series laptop at
 * 1440p, and a phone is not a small one of those. Three differences decide
 * everything here:
 *
 *  - **Fill rate.** A 2024 phone reports `devicePixelRatio` 3. The old cap of
 *    2 meant a 393x852 CSS viewport rasterised 786x1704 — 1.34 Mpx, on a panel
 *    whose own pixels the viewer cannot resolve individually anyway. Terrain's
 *    fragment shader carries atmosphere, biome blending and a 3x3 PCF shadow
 *    tap, so those are not cheap pixels. Pixel ratio is the largest single
 *    lever and it is the first one this file pulls.
 *
 *  - **Vertex throughput.** The shadow pass re-selects and re-draws the whole
 *    terrain, and every shadow vertex pays the same octave stack as a display
 *    vertex — README measures it at 75% of all geometry in the frame. A mobile
 *    GPU has an order of magnitude less of this to spend, so shadow map size,
 *    cascade count and the octave stack are all tiered.
 *
 *  - **Bandwidth.** Tiled GPUs pay for MSAA at resolve rather than at raster,
 *    and three 2048^2 R32F cascades are 50 MB of attachment written every
 *    frame. Both are tiered down rather than tuned.
 *
 * **These numbers are reasoned from those cost ratios, not measured on a
 * handset** — unlike the rest of the constants in this codebase, which is a
 * difference worth stating plainly rather than burying. What the file does
 * provide is the means to settle them by measurement: `?quality=low|medium|high`
 * forces a tier on any device, `?scale=` pins the render scale, and the HUD
 * prints the tier, the reason it was chosen and the live scale. The adaptive
 * scaler below is the part that does not depend on the guesses being right —
 * it reads the frame clock and moves resolution until the frame fits.
 */

export type TierName = 'low' | 'medium' | 'high';

export interface QualityTier {
  readonly tier: TierName;
  /** Why this tier was chosen. Printed in the HUD. */
  readonly reason: string;
  /** True for phones and tablets: gates the touch UI, not the quality. */
  readonly handheld: boolean;

  /** Cap on devicePixelRatio, before the adaptive scale multiplies it. */
  readonly maxPixelRatio: number;
  /** MSAA. Off below `high` — the resolve bandwidth buys less than the pixels do. */
  readonly antialias: boolean;

  /** Terrain amplification octaves (`DEFAULT_OCTAVES` is the high-tier value). */
  readonly octaves: number;
  /** CDLOD factor. Patch count scales with roughly its square. */
  readonly lodFactor: number;
  /** Deepest LOD level the selector may reach. */
  readonly maxLevel: number;

  readonly shadows: boolean;
  readonly shadowMapSize: number;
  readonly shadowCascades: number;
  readonly shadowLodFactor: number;
  readonly shadowMaxLevel: number;

  readonly vegetation: boolean;
  readonly vegDensity: number;
  /** Instance slots per vegetation band — sizes the scatter buffer. */
  readonly vegBandCapacity: number;
  readonly grass: boolean;

  /**
   * Frame time the adaptive scaler aims to stay under, ms. 60 Hz is 16.7 ms;
   * the budget sits under it so the scaler reacts before frames are actually
   * dropped. 0 disables adaptation.
   */
  readonly frameBudgetMs: number;
  /**
   * GPU time the scaler aims to stay under, ms — the signal it actually
   * prefers. See the note on AdaptiveResolution for why wall-clock frame time
   * cannot answer this question on a vsync-locked display. 0 disables.
   */
  readonly gpuBudgetMs: number;
  /** Floor for the adaptive scale — past this the blur is worse than the stutter. */
  readonly minRenderScale: number;
}

const HIGH: QualityTier = {
  tier: 'high',
  reason: 'desktop',
  handheld: false,
  maxPixelRatio: 2,
  antialias: true,
  octaves: 14,
  lodFactor: 2.2,
  maxLevel: 19,
  shadows: true,
  shadowMapSize: 2048,
  shadowCascades: 3,
  shadowLodFactor: 2.0,
  shadowMaxLevel: 17,
  vegetation: true,
  vegDensity: 1,
  vegBandCapacity: 400_000,
  grass: true,
  frameBudgetMs: 0,
  gpuBudgetMs: 0,
  minRenderScale: 1,
};

/**
 * Tablets, and desktops that look underpowered.
 *
 * Keeps all three systems and both of the cascades that carry contact shadows;
 * spends the saving on resolution, which is what a retina tablet most wants
 * back.
 */
const MEDIUM: QualityTier = {
  ...HIGH,
  tier: 'medium',
  maxPixelRatio: 1.5,
  antialias: false,
  octaves: 12,
  lodFactor: 1.8,
  maxLevel: 18,
  shadowMapSize: 1024,
  shadowCascades: 2,
  shadowLodFactor: 1.7,
  shadowMaxLevel: 16,
  vegDensity: 0.7,
  vegBandCapacity: 200_000,
  frameBudgetMs: 14,
  gpuBudgetMs: 11,
  minRenderScale: 0.6,
};

/**
 * Phones.
 *
 * Grass goes first: it is a separate full pass of sub-pixel geometry over the
 * near ground, and at phone resolution the blades land under a pixel, so it
 * costs a pass in order to produce aliasing. Shadows stay, at a quarter of the
 * attachment area and two cascades — dropping the 1300 m cascade also shortens
 * the shadow pass's terrain re-selection, which is the vertex cost behind it
 * rather than the texels.
 *
 * Octaves at 9 is the first thing to raise if the ground reads too smooth. It
 * is a per-vertex cost and it is paid twice, in the display and shadow passes.
 */
const LOW: QualityTier = {
  ...HIGH,
  tier: 'low',
  handheld: true,
  maxPixelRatio: 1,
  antialias: false,
  octaves: 9,
  lodFactor: 1.5,
  maxLevel: 17,
  shadows: true,
  shadowMapSize: 1024,
  shadowCascades: 2,
  shadowLodFactor: 1.5,
  shadowMaxLevel: 15,
  vegetation: true,
  vegDensity: 0.45,
  // The buffer is allocated for the worst case, not the typical one, so it is
  // sized against the density that will actually be scattered into it. At 0.45
  // the far band — 96% of the scattered area — lands near 94k, so 150k keeps
  // the cap clear of the forest while returning 12 MB of GPU and 12 MB of JS
  // heap that the desktop figure spends without noticing.
  vegBandCapacity: 150_000,
  grass: false,
  frameBudgetMs: 15,
  gpuBudgetMs: 11,
  minRenderScale: 0.5,
};

const PRESETS: Record<TierName, QualityTier> = { low: LOW, medium: MEDIUM, high: HIGH };

function detect(): QualityTier {
  // `navigator`/`matchMedia` are absent under the Node tools that import the
  // constants downstream of this file (tools/precision.ts and friends). None of
  // them render, so the desktop preset is the right answer there.
  if (typeof navigator === 'undefined' || typeof matchMedia === 'undefined') {
    return { ...HIGH, reason: 'no DOM' };
  }

  const forced = new URLSearchParams(location.search).get('quality');
  if (forced === 'low' || forced === 'medium' || forced === 'high') {
    return { ...PRESETS[forced], reason: `?quality=${forced}` };
  }

  const ua = navigator.userAgent;
  const touchPoints = navigator.maxTouchPoints ?? 0;
  const coarse = matchMedia('(pointer: coarse)').matches;
  const phone = /Android.*Mobile|iPhone|iPod|IEMobile|Opera Mini/i.test(ua);
  // iPadOS 13+ reports a Macintosh UA. Touch points are what separate it from a
  // MacBook, which reports none however many trackpads are attached.
  const tablet = /iPad|Tablet|Silk/i.test(ua)
    || (/Android/i.test(ua) && !/Mobile/i.test(ua))
    || (/Macintosh/i.test(ua) && touchPoints > 1);
  const handheld = phone || tablet || (coarse && touchPoints > 0);

  const cores = navigator.hardwareConcurrency ?? 4;
  // Chromium only — absent on Safari and Firefox, so it may only ever lower a
  // decision, never raise one.
  const mem = (navigator as { deviceMemory?: number }).deviceMemory ?? 8;

  if (phone) return { ...LOW, reason: `phone, ${cores} cores` };
  if (handheld) {
    return cores <= 4 || mem <= 3
      ? { ...LOW, handheld: true, reason: `handheld, ${cores} cores / ${mem} GB` }
      : { ...MEDIUM, handheld: true, reason: `tablet, ${cores} cores` };
  }
  if (cores <= 4 || mem <= 4) {
    return { ...MEDIUM, reason: `low-spec desktop, ${cores} cores / ${mem} GB` };
  }
  return { ...HIGH, reason: `desktop, ${cores} cores` };
}

/**
 * The tier for this session, decided once at module load.
 *
 * Eager, and deliberately so: `shadows.ts` reads `shadowMapSize` and
 * `shadowCascades` into module constants, and `shaders/shadowSample.ts` bakes
 * the map size into the node graph as a literal. Both happen at import time,
 * which is strictly after this module is evaluated because they import it.
 * Nothing here touches the GPU or the DOM, so there is nothing to wait for.
 */
export const QUALITY: QualityTier = detect();

/** Discrete rungs for the adaptive scale. */
const STEPS = [1, 0.85, 0.72, 0.6, 0.5];

/**
 * Resolution that follows the frame clock.
 *
 * The tier above is a guess about a device; this is a measurement of one. It
 * moves the pixel ratio in discrete steps rather than continuously, because
 * every change reallocates the swap chain and every target sized from it — a
 * hitch worth paying a few times in a session and never once per frame.
 *
 * **It reads GPU time, not wall-clock frame time**, and the difference is not
 * a refinement. Wall-clock time under vsync is quantised to the refresh
 * interval: a healthy frame on a 60 Hz phone measures 16.7 ms whether the GPU
 * spent 4 ms or 15 ms on it. Against an absolute 15 ms budget that reads as a
 * permanent near-overrun, and the test for headroom — comfortably under
 * budget — becomes unreachable, because the interval can never fall below the
 * refresh however cheap the frame gets. A scaler on that signal drops
 * resolution at the first jitter and can never raise it again. GPU time is
 * the quantity resolution actually controls and it is not quantised, so it is
 * the one to steer by; the wall-clock path survives only as a fallback for
 * browsers without timestamp queries, and there the thresholds are multiples
 * of the measured refresh interval rather than absolute milliseconds.
 *
 * Hysteresis is asymmetric on purpose. Dropping resolution answers a sustained
 * overrun and should be quick, about half a second of bad frames. Raising it
 * risks oscillating against whatever just happened to get cheap, so it needs
 * four times the evidence and a comfortable margin under budget. After any
 * change both counters reset and a cooldown covers the reallocation itself, so
 * the hitch it causes is never read as the overrun that justifies another.
 */
export class AdaptiveResolution {
  private stepIdx = 0;
  private ema = 0;
  private over = 0;
  private under = 0;
  private cooldown = 60;
  private refreshMs = 0;
  /**
   * Which clock is driving the decision. `probing` until a GPU timestamp
   * arrives, because three resolves them a frame or two late and reports 0
   * until then; `wall` if none ever does.
   */
  private mode: 'probing' | 'gpu' | 'wall' = 'probing';
  private probed = 0;
  private readonly base: number;
  private readonly budget: number;
  private readonly gpuBudget: number;
  private readonly minScale: number;
  private readonly pinned: number;

  constructor(
    private readonly apply: (pixelRatio: number) => void,
    tier: QualityTier = QUALITY,
  ) {
    this.base = Math.min(
      typeof devicePixelRatio === 'number' ? devicePixelRatio : 1,
      tier.maxPixelRatio,
    );
    this.budget = tier.frameBudgetMs;
    this.gpuBudget = tier.gpuBudgetMs;
    this.minScale = tier.minRenderScale;

    const pin = typeof location !== 'undefined'
      ? Number(new URLSearchParams(location.search).get('scale'))
      : NaN;
    // A pinned scale exists to measure one specific resolution, so it bypasses
    // the ladder entirely rather than snapping to the nearest rung.
    this.pinned = Number.isFinite(pin) && pin > 0 ? Math.min(pin, 1) : 0;
    this.apply(this.pixelRatio);
  }

  /** Current multiplier on the tier's pixel ratio. */
  get scale(): number {
    return this.pinned || STEPS[this.stepIdx];
  }

  /** Pixel ratio actually in use. */
  get pixelRatio(): number {
    return this.base * this.scale;
  }

  /** Smoothed value of whichever clock is driving, ms. 0 until frames land. */
  get frameMs(): number {
    return this.ema;
  }

  /** Which clock the decision is currently being made on. */
  get source(): string {
    return this.mode;
  }

  /**
   * One frame. `dtMs` is the interval since the previous frame; `gpuMs` is the
   * GPU time that frame cost, 0 when not yet resolved.
   */
  update(dtMs: number, gpuMs = 0): void {
    if (this.pinned) return;
    if (this.budget <= 0 && this.gpuBudget <= 0) return;
    // A tab returning from the background, or the first frame after a stall,
    // arrives as a multi-second dt that says nothing about steady-state cost.
    if (dtMs > 500) return;

    // The refresh interval, taken as the fastest frame yet seen. Only the
    // wall-clock path uses it, and only as a scale for its thresholds.
    if (dtMs > 1 && (this.refreshMs === 0 || dtMs < this.refreshMs)) {
      this.refreshMs = dtMs;
    }

    if (this.mode === 'probing') {
      this.probed++;
      if (gpuMs > 0 && this.gpuBudget > 0) this.mode = 'gpu';
      else if (this.probed > 180 || this.gpuBudget <= 0) this.mode = 'wall';
      else return;
    }

    let sample: number;
    let lo: number;
    let hi: number;
    if (this.mode === 'gpu') {
      // Timestamps resolve late, so a frame without one carries no information
      // rather than a zero.
      if (gpuMs <= 0) return;
      sample = gpuMs;
      hi = this.gpuBudget * 1.15;
      lo = this.gpuBudget * 0.7;
    } else {
      sample = dtMs;
      // Wall-clock time under vsync is quantised to the refresh interval, so
      // the thresholds have to be expressed in multiples of it rather than in
      // absolute milliseconds. Half a frame of overrun means frames are being
      // missed; running within 8% of the interval means the display is the
      // limit and there is headroom underneath.
      // Clamped to 60 Hz, the slowest display worth assuming. The estimate is
      // the fastest frame yet seen, and on a machine that never once reaches
      // its refresh rate that is a *dropped* frame — which would then be taken
      // as the healthy baseline and nothing would ever scale down. Capping it
      // means the worst case is a fallback that tolerates 60 Hz on a 144 Hz
      // panel, rather than one that tolerates anything at all.
      const refresh = Math.min(this.refreshMs || 16.7, 16.7);
      hi = refresh * 1.5;
      lo = refresh * 1.08;
    }

    this.ema = this.ema === 0 ? sample : this.ema + (sample - this.ema) * 0.05;

    if (this.cooldown > 0) {
      this.cooldown--;
      return;
    }

    const belowFloor = STEPS.findIndex((st) => st < this.minScale);
    const maxIdx = (belowFloor === -1 ? STEPS.length : belowFloor) - 1;

    if (this.ema > hi && this.stepIdx < maxIdx) {
      this.under = 0;
      if (++this.over >= 30) this.step(this.stepIdx + 1);
    } else if (this.ema < lo && this.stepIdx > 0) {
      this.over = 0;
      if (++this.under >= 120) this.step(this.stepIdx - 1);
    } else {
      this.over = 0;
      this.under = 0;
    }
  }

  private step(to: number): void {
    this.stepIdx = to;
    this.over = 0;
    this.under = 0;
    this.cooldown = 60;
    // The reallocation hitch would otherwise dominate the average for the next
    // hundred frames and argue for a further drop.
    this.ema = 0;
    this.apply(this.pixelRatio);
  }
}
