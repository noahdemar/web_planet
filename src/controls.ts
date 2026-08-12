/**
 * Orbit-to-ground camera. See SPEC.md §7.
 *
 * The camera's world position is f64 and lives here. The three.js camera
 * itself never leaves the origin — only its orientation is set. That is the
 * whole floating-origin story: there is no origin to shift, because the
 * renderer's origin *is* the camera (SPEC.md I4).
 *
 * Orientation is stored as yaw/pitch in the local horizon frame rather than a
 * free quaternion, so "up" stays up as you cross the planet and the horizon
 * never rolls.
 */

import { Matrix4, PerspectiveCamera, Vector3 } from 'three';
import { DEFAULT_OCTAVES, MAX_ELEVATION, RADIUS } from './planet.js';
import { heightAt } from './heightCPU.js';
import {
  type V3,
  addScaled,
  cross,
  dot,
  len,
  normalize,
} from './math/vec3d.js';

const MIN_CLEARANCE = 1.7; // eye height when ground-following
/** Walking pace, m/s. Ctrl runs. */
const WALK_SPEED = 1.5;
const RUN_MULTIPLIER = 3.4;

/** Keeps the view direction clear of the local vertical, where the basis dies. */
const PITCH_LIMIT = Math.PI / 2 - 0.05;

/**
 * Pick a spot worth looking at: land, with real relief nearby.
 *
 * 71% of the planet is ocean, so spawning at an arbitrary point lands you on
 * a featureless abyssal plain. Scores a coarse Fibonacci set by elevation plus
 * local variation, which is cheap and deterministic for a given seed.
 */
function findRuggedLand(octaves: number, hscale: number): V3 {
  const N = 4000;
  const probe = 4e-4; // ~2.5 km on the surface
  let best: V3 = [0, 0, 1];
  let bestScore = -Infinity;

  for (let i = 0; i < N; i++) {
    const z = 1 - (2 * i + 1) / N;
    const r = Math.sqrt(Math.max(0, 1 - z * z));
    const th = i * 2.399963229728653;
    const d: V3 = [r * Math.cos(th), r * Math.sin(th), z];

    const h = heightAt(d, octaves, hscale);
    if (h < 150 || h > 1600) continue; // forested montane band, not a summit

    const a = heightAt(normalize([d[0] + probe, d[1], d[2]]), octaves, hscale);
    const b = heightAt(normalize([d[0], d[1] + probe, d[2]]), octaves, hscale);
    const relief = Math.abs(a - h) + Math.abs(b - h);

    // Favour relief, but stay well below the treeline so the spot is forested.
    const score = relief * 3 - Math.abs(h - 700) * 0.25;
    if (score > bestScore) {
      bestScore = score;
      best = d;
    }
  }
  return best;
}

export class FlyControls {
  /** World position, f64. The single source of truth for where we are. */
  pos: V3 = [0, 0, RADIUS + 8_000_000];

  yaw = 0;
  pitch = -Math.PI / 2.2;
  speedMul = 1;
  groundFollow = false;

  /**
   * Walk mode. Ground-follow already pins the camera to MIN_CLEARANCE, which
   * is eye height, so most of this is taking things *away*: no vertical
   * thrust, and a fixed human speed instead of one that scales with altitude.
   * Without those two you are still flying, just at ground level.
   */
  walk = false;
  /** Vertical look inversion, toggled with I. Horizontal is never inverted. */
  invertY = false;

  /** Local frame, refreshed each update. */
  up: V3 = [0, 0, 1];
  forward: V3 = [0, 1, 0];
  right: V3 = [1, 0, 0];

  private keys = new Set<string>();
  private dragging = false;
  private locked = false;
  private octaves = DEFAULT_OCTAVES;
  private heightScale = 1;

  private tmpM = new Matrix4();
  private vx = new Vector3();
  private vy = new Vector3();
  private vz = new Vector3();

  constructor(private dom: HTMLElement) {
    this.attach();
    // Spawn immediately rather than on the first frame. Deferring it meant the
    // first update could overwrite a position set from elsewhere.
    this.reset();
    this.refreshFrame();
  }

  setTerrain(octaves: number, heightScale: number): void {
    this.octaves = octaves;
    this.heightScale = heightScale;
  }

  get altitude(): number {
    return len(this.pos) - RADIUS - this.groundHeight();
  }

  get radius(): number {
    return len(this.pos);
  }

  /** Radius of the terrain surface directly beneath the camera. */
  get groundRadius(): number {
    return RADIUS + this.groundHeight();
  }

  /** Metres per second at the current altitude, before the user multiplier. */
  get speed(): number {
    // A person walks at 1.4 m/s and runs at 5. Altitude-scaled speed happens
    // to give 1.4 at eye height, but it climbs the moment the ground does, so
    // walking uphill would accelerate you.
    if (this.walk) return WALK_SPEED * this.speedMul;
    const a = Math.max(1, this.altitude);
    return Math.min(3e6, Math.max(1.4, a * 0.45)) * this.speedMul;
  }

  private groundHeight(): number {
    const d = normalize(this.pos);
    return heightAt(d, this.octaves, this.heightScale);
  }

  private attach(): void {
    const d = this.dom;

    d.addEventListener('mousedown', (e) => {
      if (e.button === 0) {
        this.dragging = true;
        d.requestPointerLock?.();
      }
    });
    window.addEventListener('mouseup', () => {
      this.dragging = false;
    });
    document.addEventListener('pointerlockchange', () => {
      this.locked = document.pointerLockElement === d;
    });

    window.addEventListener('mousemove', (e) => {
      if (!this.locked && !this.dragging) return;
      const s = 0.0022;
      // Increasing yaw rotates forward toward east, and east is the right-hand
      // direction (right = cross(forward, up) = east at yaw 0). So moving the
      // mouse right must *increase* yaw; subtracting turned the wrong way.
      this.yaw += e.movementX * s;
      this.pitch += (this.invertY ? 1 : -1) * e.movementY * s;
      this.pitch = Math.max(-PITCH_LIMIT, Math.min(PITCH_LIMIT, this.pitch));
    });

    window.addEventListener(
      'wheel',
      (e) => {
        this.speedMul *= e.deltaY < 0 ? 1.25 : 0.8;
        this.speedMul = Math.max(1e-4, Math.min(1e4, this.speedMul));
        e.preventDefault();
      },
      { passive: false },
    );

    window.addEventListener('keydown', (e) => {
      this.keys.add(e.code);
      // Space would otherwise scroll the page even with the canvas focused.
      if (e.code === 'Space') e.preventDefault();
    });
    window.addEventListener('keyup', (e) => this.keys.delete(e.code));
    window.addEventListener('blur', () => this.keys.clear());
  }

  /** Rebuild the local horizon frame and the view direction from yaw/pitch. */
  private refreshFrame(): void {
    // Clamped here rather than only at the input, so no code path — reset,
    // scripted poses, save restore — can produce a look direction antiparallel
    // to `up`, which would collapse the camera basis to zero.
    this.pitch = Math.max(-PITCH_LIMIT, Math.min(PITCH_LIMIT, this.pitch));
    this.up = normalize(this.pos);

    // A stable tangent reference; swapped near the poles where +Y degenerates.
    const ref: V3 = Math.abs(dot([0, 1, 0], this.up)) > 0.999 ? [0, 0, 1] : [0, 1, 0];
    const east = normalize(cross(ref, this.up));
    const north = normalize(cross(this.up, east));

    const cp = Math.cos(this.pitch);
    const sp = Math.sin(this.pitch);
    const cy = Math.cos(this.yaw);
    const sy = Math.sin(this.yaw);

    this.forward = normalize([
      cp * (cy * north[0] + sy * east[0]) + sp * this.up[0],
      cp * (cy * north[1] + sy * east[1]) + sp * this.up[1],
      cp * (cy * north[2] + sy * east[2]) + sp * this.up[2],
    ]);
    this.right = normalize(cross(this.forward, this.up));
  }

  update(dt: number, camera: PerspectiveCamera): void {
    this.refreshFrame();

    const k = this.keys;
    const fwd = (k.has('KeyW') ? 1 : 0) - (k.has('KeyS') ? 1 : 0);
    const str = (k.has('KeyD') ? 1 : 0) - (k.has('KeyA') ? 1 : 0);
    const vert = this.walk
      ? 0
      : (k.has('Space') ? 1 : 0) - (k.has('ShiftLeft') || k.has('ShiftRight') ? 1 : 0);

    const running = k.has('ControlLeft') || k.has('ControlRight');
    const boost = running ? (this.walk ? RUN_MULTIPLIER : 8) : 1;
    const v = this.speed * boost * dt;

    if (fwd) this.pos = addScaled(this.pos, this.forward, fwd * v);
    if (str) this.pos = addScaled(this.pos, this.right, str * v);
    if (vert) this.pos = addScaled(this.pos, this.up, vert * v);

    // Keep the camera out of the rock. Ground-follow pins it to eye height;
    // otherwise it is a floor, so you can still fly freely above the surface.
    const ground = RADIUS + this.groundHeight();
    const r = len(this.pos);
    const floor = ground + MIN_CLEARANCE;
    const pinned = this.groundFollow || this.walk;
    if (pinned || r < floor) {
      const target = pinned ? floor : Math.max(r, floor);
      const s = target / r;
      this.pos = [this.pos[0] * s, this.pos[1] * s, this.pos[2] * s];
    }
    // Hard ceiling so the quadtree always has something to select.
    const ceiling = RADIUS * 12;
    if (r > ceiling) {
      const s = ceiling / r;
      this.pos = [this.pos[0] * s, this.pos[1] * s, this.pos[2] * s];
    }

    // Three's camera looks down -Z, and stays at the origin forever.
    this.vz.set(-this.forward[0], -this.forward[1], -this.forward[2]);
    this.vx.set(this.up[0], this.up[1], this.up[2]).cross(this.vz).normalize();
    this.vy.copy(this.vz).cross(this.vx);
    this.tmpM.makeBasis(this.vx, this.vy, this.vz);
    camera.quaternion.setFromRotationMatrix(this.tmpM);
    camera.position.set(0, 0, 0);
    camera.updateMatrixWorld(true);
  }

  /** Drop into a low orbit looking down at the most interesting land found. */
  reset(): void {
    const d = findRuggedLand(this.octaves, this.heightScale);
    this.pos = [
      d[0] * (RADIUS + 8_000_000),
      d[1] * (RADIUS + 8_000_000),
      d[2] * (RADIUS + 8_000_000),
    ];
    this.yaw = 0;
    this.pitch = -1.32; // steeply down, but clear of the pole
    this.speedMul = 1;
    this.groundFollow = false;
  }

  /**
   * Enter or leave walk mode. Entering drops the camera to the surface under
   * it rather than leaving it in the air, so the button is one click rather
   * than a click and then a fall.
   */
  setWalk(on: boolean): void {
    this.walk = on;
    this.groundFollow = on;
    this.speedMul = 1;
    if (on) {
      const d = normalize(this.pos);
      const g = RADIUS + this.groundHeight() + MIN_CLEARANCE;
      this.pos = [d[0] * g, d[1] * g, d[2] * g];
      // Looking at the horizon, not at your feet or at the sky.
      this.pitch = Math.max(-0.35, Math.min(0.25, this.pitch));
    }
  }

  /** Move to the most rugged land on the planet, at eye height. */
  gotoRuggedLand(): void {
    const d = findRuggedLand(this.octaves, this.heightScale);
    const g = RADIUS + heightAt(d, this.octaves, this.heightScale) + MIN_CLEARANCE;
    this.pos = [d[0] * g, d[1] * g, d[2] * g];
    this.pitch = -0.04;
    this.speedMul = 1;
    this.groundFollow = true;
  }

  /** Teleport to just above the surface, keeping the current ground track. */
  dropToSurface(): void {
    const d = normalize(this.pos);
    const g = RADIUS + Math.max(this.groundHeight(), 0) + MAX_ELEVATION * 0.004;
    this.pos = [d[0] * g, d[1] * g, d[2] * g];
    this.pitch = -0.05;
    this.speedMul = 1;
  }
}
