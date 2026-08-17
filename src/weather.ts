import {
  AdditiveBlending,
  BufferAttribute,
  BufferGeometry,
  DynamicDrawUsage,
  LineBasicMaterial,
  LineSegments,
  Vector3,
} from 'three';
import { CLOUD_ALT, CLOUD_SYN_FREQ, CLOUD_SYN_FREQ2 } from './planet.js';
import { QUALITY } from './quality.js';
import { shaderNoise } from './shaderNoiseCPU.js';

export type WeatherMode = 'auto' | 'rain' | 'storm' | 'clear';

const fract = (x: number): number => x - Math.floor(x);
const hash = (x: number): number => fract(Math.sin(x * 127.1) * 43758.5453);
const smoothstep = (a: number, b: number, x: number): number => {
  const t = Math.max(0, Math.min(1, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
};

function automaticRain(x: number, y: number, z: number, time: number, cover: number): number {
  const r = Math.hypot(x, y, z) || 1;
  const dx = x / r;
  const dy = y / r;
  const dz = z / r;
  const lat = Math.abs(dy);
  const cs = Math.cos(time * 0.002);
  const sn = Math.sin(time * 0.002);
  const sx = dx * cs + dz * sn;
  const sz = -dx * sn + dz * cs;
  // The same synoptic field cloudField_ gates the deck with — see
  // CLOUD_SYN_FREQ. It has to be the identical frequency or rain falls where
  // there is no storm: this decides whether it is raining *at the camera*, and
  // the player can see the sky it is supposed to be falling out of.
  const f0 = CLOUD_SYN_FREQ;
  const f1 = CLOUD_SYN_FREQ2;
  const n0 = shaderNoise(sx * f0, dy * f0 + time * 0.00038, sz * f0);
  const n1 = shaderNoise(sx * f1 + 37.1, dy * f1 + time * 0.00066, sz * f1 + 37.1);
  const itcz = Math.exp(-((lat / 0.14) ** 2));
  const dry = Math.exp(-(((lat - 0.45) / 0.16) ** 2));
  const storm = Math.exp(-(((lat - 0.74) / 0.17) ** 2));
  return smoothstep(0.12, 0.43, n0 * 0.72 + n1 * 0.22 + itcz * 0.18 - dry * 0.24 + storm * 0.2 + (cover - 0.5) * 0.35);
}

export class Weather {
  readonly rain: LineSegments;
  readonly lightning: LineSegments;

  mode: WeatherMode = 'auto';
  intensity = 0;
  wetness = 0;
  flash = 0;

  private readonly count = QUALITY.tier === 'low' ? 1800 : QUALITY.tier === 'medium' ? 3600 : 6000;
  private readonly rainPositions: Float32Array;
  private readonly rainMaterial: LineBasicMaterial;
  private readonly boltPositions = new Float32Array(30 * 2 * 3);
  private readonly boltMaterial: LineBasicMaterial;
  private readonly flashLayer: HTMLDivElement;
  private target = 0;
  private nextStrike = 4;
  private strikeStart = -10;
  private strikeSeed = 1;

  constructor() {
    this.rainPositions = new Float32Array(this.count * 2 * 3);
    const rainGeometry = new BufferGeometry();
    rainGeometry.setAttribute('position', new BufferAttribute(this.rainPositions, 3).setUsage(DynamicDrawUsage));
    this.rainMaterial = new LineBasicMaterial({
      color: 0xaac7d9,
      transparent: true,
      opacity: 0,
      depthWrite: false,
    });
    this.rain = new LineSegments(rainGeometry, this.rainMaterial);
    this.rain.frustumCulled = false;
    this.rain.renderOrder = 80;

    const boltGeometry = new BufferGeometry();
    boltGeometry.setAttribute('position', new BufferAttribute(this.boltPositions, 3).setUsage(DynamicDrawUsage));
    this.boltMaterial = new LineBasicMaterial({
      color: 0xdcecff,
      transparent: true,
      opacity: 0,
      depthWrite: false,
      blending: AdditiveBlending,
    });
    this.lightning = new LineSegments(boltGeometry, this.boltMaterial);
    this.lightning.frustumCulled = false;
    this.lightning.renderOrder = 90;

    this.flashLayer = document.createElement('div');
    Object.assign(this.flashLayer.style, {
      position: 'fixed', inset: '0', pointerEvents: 'none', zIndex: '20',
      background: 'rgb(205 225 255)', opacity: '0', mixBlendMode: 'screen',
    });
    document.body.appendChild(this.flashLayer);
  }

  cycleMode(): WeatherMode {
    const modes: WeatherMode[] = ['auto', 'rain', 'storm', 'clear'];
    this.mode = modes[(modes.indexOf(this.mode) + 1) % modes.length];
    return this.mode;
  }

  update(
    time: number,
    dt: number,
    camera: readonly [number, number, number],
    upIn: readonly [number, number, number],
    forwardIn: readonly [number, number, number],
    altitude: number,
    cloudCover: number,
  ): void {
    const auto = automaticRain(camera[0], camera[1], camera[2], time, cloudCover);
    this.target = this.mode === 'clear' ? 0 : this.mode === 'rain' ? 0.72 : this.mode === 'storm' ? 1 : auto;
    if (altitude > CLOUD_ALT) this.target *= 1 - smoothstep(CLOUD_ALT, CLOUD_ALT + 2000, altitude);
    this.intensity += (this.target - this.intensity) * (1 - Math.exp(-dt * 0.8));
    this.wetness = Math.max(0, Math.min(1, this.wetness + dt * (this.intensity * 0.09 - (1 - this.intensity) * 0.006)));

    const up = new Vector3(upIn[0], upIn[1], upIn[2]).normalize();
    const forward = new Vector3(forwardIn[0], forwardIn[1], forwardIn[2]);
    forward.addScaledVector(up, -forward.dot(up)).normalize();
    const right = new Vector3().crossVectors(forward, up).normalize();
    this.updateRain(time, up, forward, right);

    if (this.target > 0.82 && time >= this.nextStrike) this.strike(time, altitude, up, forward, right);
    const age = time - this.strikeStart;
    this.flash = age >= 0 && age < 0.65
      ? Math.max(Math.exp(-age * 13), Math.exp(-Math.abs(age - 0.19) * 38) * 0.72)
      : 0;
    this.boltMaterial.opacity = this.flash > 0.08 ? Math.min(1, this.flash * 2.4) : 0;
    this.flashLayer.style.opacity = String(this.flash * 0.34);
  }

  private updateRain(time: number, up: Vector3, forward: Vector3, right: Vector3): void {
    const p = this.rainPositions;
    const fall = time * 34;
    const wind = new Vector3().copy(right).multiplyScalar(0.22).addScaledVector(forward, 0.08).addScaledVector(up, -1).normalize();
    for (let i = 0; i < this.count; i++) {
      const cycle = fract(hash(i + 1) + fall / (65 + hash(i + 8) * 35));
      const x = (hash(i * 3 + 2) * 2 - 1) * 52;
      const z = 4 + hash(i * 3 + 3) * 105;
      const y = 55 - cycle * 78;
      const base = i * 6;
      p[base] = right.x * x + forward.x * z + up.x * y;
      p[base + 1] = right.y * x + forward.y * z + up.y * y;
      p[base + 2] = right.z * x + forward.z * z + up.z * y;
      const len = 1.2 + hash(i * 3 + 4) * 3.6;
      p[base + 3] = p[base] - wind.x * len;
      p[base + 4] = p[base + 1] - wind.y * len;
      p[base + 5] = p[base + 2] - wind.z * len;
    }
    (this.rain.geometry.getAttribute('position') as BufferAttribute).needsUpdate = true;
    this.rainMaterial.opacity = smoothstep(0.04, 0.75, this.intensity) * 0.48;
    this.rain.visible = this.rainMaterial.opacity > 0.005;
  }

  private strike(time: number, altitude: number, up: Vector3, forward: Vector3, right: Vector3): void {
    this.strikeStart = time;
    this.strikeSeed += 17;
    this.nextStrike = time + 3.5 + hash(this.strikeSeed) * 9;
    const lateral = (hash(this.strikeSeed + 1) * 2 - 1) * 1800;
    const ahead = 900 + hash(this.strikeSeed + 2) * 3600;
    const top = Math.max(1200, CLOUD_ALT - Math.min(altitude, CLOUD_ALT - 300));
    const bottom = -Math.min(Math.max(altitude, 0), 200);
    let previous = new Vector3().copy(right).multiplyScalar(lateral).addScaledVector(forward, ahead).addScaledVector(up, top);
    for (let i = 0; i < 30; i++) {
      const t = (i + 1) / 30;
      const wander = (1 - t) * 240;
      const next = new Vector3().copy(right).multiplyScalar(lateral + (hash(this.strikeSeed + i * 2 + 3) * 2 - 1) * wander)
        .addScaledVector(forward, ahead + (hash(this.strikeSeed + i * 2 + 4) * 2 - 1) * wander)
        .addScaledVector(up, top + (bottom - top) * t);
      const o = i * 6;
      this.boltPositions[o] = previous.x;
      this.boltPositions[o + 1] = previous.y;
      this.boltPositions[o + 2] = previous.z;
      this.boltPositions[o + 3] = next.x;
      this.boltPositions[o + 4] = next.y;
      this.boltPositions[o + 5] = next.z;
      previous = next;
    }
    (this.lightning.geometry.getAttribute('position') as BufferAttribute).needsUpdate = true;
  }
}
