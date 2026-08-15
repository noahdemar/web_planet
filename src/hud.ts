/**
 * Debug HUD. SPEC.md §11 asks for a live budget readout from M1 rather than a
 * reckoning at the end, so this exists before anything it measures does.
 *
 * The precision row is the point of the whole milestone: it shows the f32 ULP
 * at the current planetary radius next to the ground sample distance actually
 * being drawn. When ULP exceeds GSD, a naive renderer has already failed —
 * this one has not, and the metric grid makes that visible.
 */

import { MAX_LEVEL, MAX_VEG_TILES, RADIUS, gsdAt } from './planet.js';
import { QUALITY } from './quality.js';
import type { SelectStats } from './quadtree.js';
import type { VegStats } from './vegetation.js';

const FACE_NAMES = ['+X', '−X', '+Y', '−Y', '+Z', '−Z'];

function metres(m: number): string {
  const a = Math.abs(m);
  if (a < 1) return `${(m * 100).toFixed(1)} cm`;
  if (a < 1000) return `${m.toFixed(1)} m`;
  if (a < 1e6) return `${(m / 1000).toFixed(2)} km`;
  return `${(m / 1000).toFixed(0)} km`;
}

function speed(v: number): string {
  if (v < 1000) return `${v.toFixed(1)} m/s`;
  return `${(v / 1000).toFixed(1)} km/s`;
}

export interface HudState {
  fps: number;
  frameMs: number;
  worstMs: number;
  /** Live multiplier the adaptive scaler is applying to the tier pixel ratio. */
  renderScale: number;
  /** Pixel ratio in use, after that multiplier. */
  pixelRatio: number;
  /** Which clock the scaler is steering by: gpu, wall, or probing. */
  scaleSource: string;
  altitude: number;
  radius: number;
  speed: number;
  speedMul: number;
  face: number;
  octaves: number;
  lodFactor: number;
  maxLevel: number;
  gridSpacing: number;
  shadeMode: number;
  groundFollow: boolean;
  stats: SelectStats;
  veg: VegStats;
  gpu: { render: number; compute: number; drawCalls: number; triangles: number };
}

const MODES = [
  'natural', 'LOD level', 'slope', 'normals', 'canopy cover', 'albedo', 'climate',
  // Encoded for sim.probeHeight() rather than for the eye — see the mode 7
  // branch in shaders/terrain.ts.
  'elevation (readback)',
];

export class Hud {
  private el: HTMLElement;
  private last = 0;

  constructor(el: HTMLElement) {
    this.el = el;
  }

  toggle(): void {
    this.el.classList.toggle('hidden');
  }

  setVisible(visible: boolean): void {
    this.el.classList.toggle('hidden', !visible);
  }

  isVisible(): boolean {
    return !this.el.classList.contains('hidden');
  }

  render(s: HudState, now: number): void {
    // Hidden is hidden: no string building, no innerHTML write. The panel is
    // off by default now, so this is the common path rather than a corner of
    // it, and it was rebuilding forty rows of DOM every frame for nobody.
    if (!this.isVisible()) return;
    // 10 Hz: DOM writes are not free, and this must not perturb what it measures.
    if (now - this.last < 100) return;
    this.last = now;

    const st = s.stats;
    const gsd = gsdAt(st.maxLevel);
    const ulp = Math.pow(2, Math.ceil(Math.log2(s.radius)) - 24);
    const ratio = ulp / gsd;
    const cls = ratio > 1 ? 'warn' : 'good';

    const budget = s.frameMs <= 16.7 ? 'good' : 'warn';

    this.el.innerHTML = `
      <h1>world_sim · M1</h1>
      ${row('frame', `<span class="${budget}">${s.frameMs.toFixed(2)} ms · ${s.fps.toFixed(0)} fps</span>`)}
      ${row('worst (2s)', `${s.worstMs.toFixed(2)} ms`)}
      ${row('gpu render', `${s.gpu.render.toFixed(2)} ms`)}
      ${row('gpu compute', `${s.gpu.compute.toFixed(2)} ms`)}
      ${row('draws / tris', `${s.gpu.drawCalls} · ${fmtCount(s.gpu.triangles)}`)}
      ${row('quality', `${QUALITY.tier} <span style="opacity:.5">${QUALITY.reason}</span>`)}
      ${row('render scale', `${s.renderScale.toFixed(2)}× <span style="opacity:.5">dpr ${s.pixelRatio.toFixed(2)} · ${s.scaleSource}</span>`)}
      <div class="sep"></div>
      ${row('altitude', metres(s.altitude))}
      ${row('speed', `${speed(s.speed)} <span style="opacity:.5">×${s.speedMul.toFixed(2)}</span>`)}
      ${row('face', FACE_NAMES[s.face] ?? '—')}
      <div class="sep"></div>
      ${row('patches', `${st.patches}${st.overflow ? ' <span class="warn">OVERFLOW</span>' : ''}`)}
      ${row('nodes visited', `${st.visited}`)}
      ${row('culled', `${st.culledFrustum} frustum · ${st.culledHorizon} horizon`)}
      ${row('LOD levels', `${st.minLevel} – ${st.maxLevel} / ${s.maxLevel}`)}
      ${row('GSD @ finest', metres(gsd))}
      <div class="sep"></div>
      ${row('f32 ULP @ radius', `<span class="${cls}">${metres(ulp)}</span>`)}
      ${row('ULP / GSD', `<span class="${cls}">${ratio.toFixed(1)}×${ratio > 1 ? ' naive fails here' : ''}</span>`)}
      <div class="sep"></div>
      ${row('octaves', `${s.octaves}`)}
      ${row('lod factor', s.lodFactor.toFixed(2))}
      ${row('shading', MODES[s.shadeMode] ?? '—')}
      ${row('grid', s.gridSpacing > 0 ? metres(s.gridSpacing) : 'off')}
      ${row('ground follow', s.groundFollow ? 'on' : 'off')}
      <div class="sep"></div>
      ${row(
        'veg tiles',
        s.veg.tiles >= MAX_VEG_TILES
          ? `<span class="warn">${s.veg.tiles} AT CAP</span>`
          : `${s.veg.tiles}`,
      )}
      ${row('candidates', fmtCount(s.veg.candidates))}
      ${row(
        'instances',
        s.veg.total > 0
          ? `<span class="good">${fmtCount(s.veg.total)}</span>${s.veg.overflow ? ' <span class="warn">OVF</span>' : ''}`
          : '<span style="opacity:.5">press N</span>',
      )}
      ${row('near / mid / far', s.veg.perBand.map(fmtCount).join(' · '))}
      <div class="hint">
        <kbd>W</kbd><kbd>A</kbd><kbd>S</kbd><kbd>D</kbd> fly ·
        <kbd>Space</kbd>/<kbd>Shift</kbd> up·down · <kbd>Ctrl</kbd> boost<br />
        drag to look · wheel = speed · <kbd>F</kbd> walk / fly<br />
        <kbd>1</kbd>–<kbd>4</kbd> shading · <kbd>G</kbd> metric grid ·
        <kbd>[</kbd><kbd>]</kbd> LOD<br />
        <kbd>,</kbd><kbd>.</kbd> octaves · <kbd>-</kbd><kbd>=</kbd> max level<br />
        <kbd>L</kbd> rugged land · <kbd>T</kbd> surface · <kbd>R</kbd> reset<br />
        <kbd>K</kbd> day cycle / follow sun · <kbd>J</kbd> pause the day<br />
        <kbd>V</kbd> vegetation · <kbd>B</kbd> bands · <kbd>;</kbd><kbd>'</kbd> density ·
        <kbd>N</kbd> count
      </div>
    `;
  }
}

function fmtCount(n: number): string {
  if (n < 1000) return `${n}`;
  if (n < 1e6) return `${(n / 1000).toFixed(1)}k`;
  return `${(n / 1e6).toFixed(2)}M`;
}

function row(label: string, value: string): string {
  return `<div class="row"><span>${label}</span><span>${value}</span></div>`;
}

export const MAX_LEVEL_CAP = MAX_LEVEL;
export const PLANET_RADIUS = RADIUS;
