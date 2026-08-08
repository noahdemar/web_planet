/**
 * Crack audit — the automated form of M1's "no cracks" criterion.
 *
 * Renders the current view to an offscreen target cleared to a colour the
 * terrain shader can never produce, then counts background pixels that are
 * *enclosed* by terrain. Any such pixel is a hole between patches.
 *
 * Reading the swap chain back directly does not work — a WebGPU canvas
 * texture is gone after present — so this re-renders into a RenderTarget.
 *
 * From the console:  await sim.audit()
 */

import {
  Color,
  NoToneMapping,
  PerspectiveCamera,
  RenderTarget,
  Scene,
  UnsignedByteType,
} from 'three';
import type { WebGPURenderer } from 'three/webgpu';

/** Pure magenta: unreachable by the terrain palette, which is desaturated. */
const PROBE = new Color(1, 0, 1);

export interface AuditResult {
  width: number;
  height: number;
  terrainPixels: number;
  /** Background pixels with terrain both above and below them in a column. */
  crackPixels: number;
  crackFraction: string;
  pass: boolean;
}

export async function auditCracks(
  renderer: WebGPURenderer,
  scene: Scene,
  camera: PerspectiveCamera,
  width = 1280,
  height = 720,
): Promise<AuditResult> {
  const target = new RenderTarget(width, height, { type: UnsignedByteType });

  const prevClear = new Color();
  renderer.getClearColor(prevClear);
  const prevAlpha = renderer.getClearAlpha();
  const prevTone = renderer.toneMapping;
  const prevAspect = camera.aspect;

  renderer.setClearColor(PROBE, 1);
  // Tone mapping would shift the probe colour away from pure magenta.
  renderer.toneMapping = NoToneMapping;
  camera.aspect = width / height;
  camera.updateProjectionMatrix();

  renderer.setRenderTarget(target);
  await renderer.renderAsync(scene, camera);
  const buf = (await renderer.readRenderTargetPixelsAsync(
    target,
    0,
    0,
    width,
    height,
  )) as Uint8Array;
  renderer.setRenderTarget(null);

  renderer.setClearColor(prevClear, prevAlpha);
  renderer.toneMapping = prevTone;
  camera.aspect = prevAspect;
  camera.updateProjectionMatrix();
  target.dispose();

  const isProbe = (i: number) => buf[i] > 200 && buf[i + 1] < 80 && buf[i + 2] > 200;

  // Per column, find the topmost and bottommost terrain pixel; any probe pixel
  // strictly between them is enclosed, and therefore a hole rather than sky.
  let cracks = 0;
  let terrain = 0;
  for (let x = 0; x < width; x++) {
    let first = -1;
    let last = -1;
    for (let y = 0; y < height; y++) {
      if (!isProbe((y * width + x) * 4)) {
        if (first < 0) first = y;
        last = y;
        terrain++;
      }
    }
    if (first < 0) continue;
    for (let y = first; y <= last; y++) {
      if (isProbe((y * width + x) * 4)) cracks++;
    }
  }

  return {
    width,
    height,
    terrainPixels: terrain,
    crackPixels: cracks,
    crackFraction: `${((cracks / Math.max(1, terrain)) * 100).toFixed(5)}%`,
    pass: cracks === 0,
  };
}
