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

interface TouchState {
  startX: number;
  startY: number;
  lastX: number;
  lastY: number;
  side: 'left' | 'right';
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

  /** Mobile touch input overrides */
  verticalInput = 0;
  boostInput = false;
  onJoyChange?: (state: {
    active: boolean;
    startX: number;
    startY: number;
    thumbX: number;
    thumbY: number;
    maxR: number;
  }) => void;

  /**
   * Tangent north, carried from frame to frame rather than rebuilt.
   *
   * This is what removes the pole singularity — see refreshFrame.
   */
  private northRef: V3 = [0, 1, 0];

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

  private touches = new Map<number, TouchState>();
  private joy = { x: 0, y: 0 };
  private joyActive = false;
  private pinchActive = false;
  private pinchStart = 0;

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
    this.attachTouch();
  }

  /**
   * Touch handling.
   *
   *   one finger, anywhere    look
   *   two fingers, pinch      zoom — toward or away from the ground
   *   joystick pad            move, from the lower-left corner only
   *
   * The old split put the joystick across the *entire* left half and look
   * across the entire right, so which half of the glass you happened to touch
   * decided what a swipe meant, and a pinch had to be started on the right or
   * it did nothing. Pinch also drove `speedMul`, which is a flight control
   * rather than a zoom: it changed how fast a later movement would be instead
   * of moving anything, so the picture did not respond to the gesture at all.
   *
   * One finger looking everywhere is what a touch user expects, and the
   * joystick keeps translation without claiming half the screen — it now has
   * to be started inside the pad in the corner.
   */
  private attachTouch(): void {
    const d = this.dom;

    // The joystick pad: a corner, not a half. Anything outside it looks.
    const getSide = (clientX: number, clientY: number): 'left' | 'right' => {
      const padR = Math.min(160, d.clientWidth * 0.32);
      const inPad = clientX < padR && clientY > d.clientHeight - padR;
      return inPad ? 'left' : 'right';
    };

    const onStart = (e: TouchEvent) => {
      for (let i = 0; i < e.changedTouches.length; i++) {
        const t = e.changedTouches[i];
        this.touches.set(t.identifier, {
          startX: t.clientX,
          startY: t.clientY,
          lastX: t.clientX,
          lastY: t.clientY,
          side: getSide(t.clientX, t.clientY),
        });
      }
      this.updateTouchPinch();
      this.updateTouchJoy();
      e.preventDefault();
    };

    const onMove = (e: TouchEvent) => {
      const rightSide = [...this.touches.values()].filter((s) => s.side === 'right');
      for (let i = 0; i < e.changedTouches.length; i++) {
        const t = e.changedTouches[i];
        const s = this.touches.get(t.identifier);
        if (!s) continue;
        const dx = t.clientX - s.lastX;
        const dy = t.clientY - s.lastY;
        s.lastX = t.clientX;
        s.lastY = t.clientY;
        if (s.side === 'right' && rightSide.length === 1 && !this.pinchActive) {
          // Single finger anywhere outside the pad: look.
          const scale = 0.0024;
          this.yaw += dx * scale;
          this.pitch += (this.invertY ? 1 : -1) * dy * scale;
          this.pitch = Math.max(-PITCH_LIMIT, Math.min(PITCH_LIMIT, this.pitch));
        }
      }
      this.updateTouchPinch();
      this.updateTouchJoy();
      e.preventDefault();
    };

    const onEnd = (e: TouchEvent) => {
      for (let i = 0; i < e.changedTouches.length; i++) {
        this.touches.delete(e.changedTouches[i].identifier);
      }
      this.updateTouchPinch();
      this.updateTouchJoy();
    };

    d.addEventListener('touchstart', onStart, { passive: false });
    d.addEventListener('touchmove', onMove, { passive: false });
    d.addEventListener('touchend', onEnd, { passive: false });
    d.addEventListener('touchcancel', onEnd, { passive: false });
  }

  /**
   * Pinch to zoom.
   *
   * Zoom here is a *dolly*, not a field-of-view change: the camera moves along
   * its view direction, so parallax responds and the horizon behaves, which is
   * what makes the planet read as a place rather than a photograph being
   * scaled. Field of view would have been cheaper and would have looked like a
   * zoom lens.
   *
   * The step is proportional to altitude, because that is the only scale that
   * works across seven decades of range: a fixed step is a crawl from orbit
   * and a jump into the ground at eye height. Clamped so one gesture cannot
   * put the camera under the terrain.
   */
  private updateTouchPinch(): void {
    const right = [...this.touches.values()].filter((s) => s.side === 'right');
    if (right.length >= 2) {
      const a = right[0];
      const b = right[1];
      const dist = Math.hypot(a.lastX - b.lastX, a.lastY - b.lastY);
      if (!this.pinchActive) {
        this.pinchStart = dist;
        this.pinchActive = true;
      } else if (this.pinchStart > 0.5 && dist > 0.5) {
        // Spreading the fingers moves in; pinching moves out.
        const ratio = dist / this.pinchStart;
        this.pinchStart = dist;
        const alt = Math.max(2, this.altitude);
        // A doubling of finger separation covers about half the remaining
        // height to the ground, which feels like one decisive gesture without
        // being able to overshoot through the surface in a single frame.
        const step = alt * 0.5 * (ratio - 1);
        const move = Math.max(-alt * 0.6, Math.min(alt * 0.6, step));
        this.pos = [
          this.pos[0] + this.forward[0] * move,
          this.pos[1] + this.forward[1] * move,
          this.pos[2] + this.forward[2] * move,
        ];
      }
    } else {
      this.pinchActive = false;
    }
  }

  private readonly joyMaxR = 50;

  private updateTouchJoy(): void {
    const left = [...this.touches.values()].filter((s) => s.side === 'left')[0];
    let startX = 0;
    let startY = 0;
    let thumbX = 0;
    let thumbY = 0;

    if (left) {
      this.joyActive = true;
      startX = left.startX;
      startY = left.startY;
      const dx = left.lastX - left.startX;
      const dy = left.lastY - left.startY;
      const r = Math.hypot(dx, dy);
      const clampedR = Math.min(r, this.joyMaxR);
      const nx = r === 0 ? 0 : dx / r;
      const ny = r === 0 ? 0 : dy / r;
      thumbX = startX + nx * clampedR;
      thumbY = startY + ny * clampedR;

      const deadzone = 4;
      if (r < deadzone) {
        this.joy.x = 0;
        this.joy.y = 0;
      } else {
        const factor = (clampedR - deadzone) / (this.joyMaxR - deadzone);
        this.joy.x = nx * factor;
        this.joy.y = ny * factor;
      }
    } else {
      this.joyActive = false;
      this.joy.x = 0;
      this.joy.y = 0;
    }

    this.onJoyChange?.({
      active: this.joyActive,
      startX,
      startY,
      thumbX,
      thumbY,
      maxR: this.joyMaxR,
    });
  }

  /** Rebuild the local horizon frame and the view direction from yaw/pitch. */
  private refreshFrame(): void {
    // Clamped here rather than only at the input, so no code path — reset,
    // scripted poses, save restore — can produce a look direction antiparallel
    // to `up`, which would collapse the camera basis to zero.
    this.pitch = Math.max(-PITCH_LIMIT, Math.min(PITCH_LIMIT, this.pitch));
    this.up = normalize(this.pos);

    // The horizon frame is *transported*, not rebuilt from a world axis.
    //
    // It used to be built each frame from +Y, swapping to +Z within 0.999 of
    // the pole. Both halves of that fail, and they fail together exactly where
    // the user noticed:
    //
    //   cross([0,1,0], up) has magnitude sin(angle from the pole), so as the
    //   camera approaches, that vector shrinks toward zero and `normalize`
    //   turns whatever is left — mostly rounding error — into a full-length
    //   basis. The heading spins on its own.
    //
    //   Then the swap to +Z is a *hard* switch, so the moment it trips the
    //   basis rotates about ninety degrees in one frame while yaw stays the
    //   number it was. The view snaps.
    //
    // Neither is fixable by moving the threshold, because the underlying
    // quantity — the azimuth of a fixed world axis, seen from the local
    // horizon — is genuinely undefined at the pole. There is no reference
    // direction to pick.
    //
    // So none is picked. `northRef` is carried across frames and merely
    // re-orthogonalised against the new `up`, which is parallel transport: it
    // is continuous everywhere including across the pole itself, and the
    // horizon still stays level because it is a tangent vector by
    // construction. What it gives up is that "north" stops meaning geographic
    // north after a long traverse — the frame accumulates the holonomy of the
    // path, which is the honest behaviour on a sphere and is invisible unless
    // something draws a compass.
    let north = this.northRef;
    let tx = north[0] - this.up[0] * dot(north, this.up);
    let ty = north[1] - this.up[1] * dot(north, this.up);
    let tz = north[2] - this.up[2] * dot(north, this.up);
    let tl = Math.hypot(tx, ty, tz);
    if (tl < 1e-4) {
      // Only reachable on a teleport that lands a quarter turn away in one
      // step — the tour does exactly that. Any tangent will do; continuity was
      // already broken by the jump.
      const alt: V3 = Math.abs(this.up[1]) > 0.9 ? [0, 0, 1] : [0, 1, 0];
      tx = alt[0] - this.up[0] * dot(alt, this.up);
      ty = alt[1] - this.up[1] * dot(alt, this.up);
      tz = alt[2] - this.up[2] * dot(alt, this.up);
      tl = Math.hypot(tx, ty, tz);
    }
    north = [tx / tl, ty / tl, tz / tl];
    this.northRef = north;
    const east = normalize(cross(north, this.up));

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
    const joyX = this.joyActive ? this.joy.x : 0;
    const joyY = this.joyActive ? this.joy.y : 0;
    const fwd = (k.has('KeyW') ? 1 : 0) - (k.has('KeyS') ? 1 : 0) - joyY;
    const str = (k.has('KeyD') ? 1 : 0) - (k.has('KeyA') ? 1 : 0) + joyX;
    // Shift boosts, Ctrl descends. The other way round is the more common
    // flight-sim assignment, but Shift-to-sprint is what every game that has a
    // sprint uses, and the hand is already there.
    const vert = this.walk
      ? 0
      : (k.has('Space') ? 1 : 0) - (k.has('ControlLeft') || k.has('ControlRight') ? 1 : 0) + this.verticalInput;

    const running = k.has('ShiftLeft') || k.has('ShiftRight') || this.boostInput;
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
