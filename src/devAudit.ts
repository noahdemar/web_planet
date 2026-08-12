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

/**
 * The elevation the GPU actually drew, at the centre of the view.
 *
 * The counterpart to `heightAt`: point the camera at a direction, call this,
 * and the two numbers are the same quantity computed on the two sides. There
 * was no way to get this before, which is why a CPU/GPU height divergence
 * could sit at the top of the bug list unexplained — every other instrument in
 * the project measures the CPU against the *asset*, and the asset was never
 * the thing in doubt.
 *
 * Shade mode 7 encodes elevation as two bytes over ±12 km; see the mode 7
 * branch in shaders/terrain.ts. Rendering at 1x1 would be ideal but a
 * degenerate viewport upsets the projection, so it renders small and reads the
 * middle pixel.
 *
 * `setMode` is passed in rather than importing TerrainMesh, which would make
 * this module depend on the thing it inspects.
 */
export async function probeDrawnHeight(
  renderer: WebGPURenderer,
  scene: Scene,
  camera: PerspectiveCamera,
  setMode: (m: number) => void,
  restoreMode: number,
): Promise<number> {
  const N = 33;
  const target = new RenderTarget(N, N, { type: UnsignedByteType });
  const prevTone = renderer.toneMapping;
  const prevAspect = camera.aspect;
  const prevFov = camera.fov;

  // A pinhole field of view, because the caller compares this against a CPU
  // evaluation at one *direction*. At the display's 55° a centre pixel is a
  // 24 m footprint on ground a kilometre away — several wavelengths of the
  // finest octave — so the amplification's own variance reads as CPU/GPU
  // disagreement. At 0.02° the pixel is under a centimetre at that range and
  // the two sides are genuinely being asked the same question.
  setMode(7);
  renderer.toneMapping = NoToneMapping;
  camera.aspect = 1;
  camera.fov = 0.02;
  camera.updateProjectionMatrix();

  renderer.setRenderTarget(target);
  await renderer.renderAsync(scene, camera);
  const buf = (await renderer.readRenderTargetPixelsAsync(
    target,
    (N - 1) / 2,
    (N - 1) / 2,
    1,
    1,
  )) as Uint8Array;
  renderer.setRenderTarget(null);

  setMode(restoreMode);
  renderer.toneMapping = prevTone;
  camera.aspect = prevAspect;
  camera.fov = prevFov;
  camera.updateProjectionMatrix();
  target.dispose();

  return ((buf[0] + buf[1] / 255) / 255) * 24000 - 12000;
}

export interface AuditResult {
  width: number;
  height: number;
  terrainPixels: number;
  /** Background pixels with terrain both above and below them in a column. */
  crackPixels: number;
  crackFraction: string;
  pass: boolean;
}

/**
 * `height` defaults to `width / camera.aspect` rather than to 720.
 *
 * The audit re-renders the scene but does *not* re-select patches — the
 * selection belongs to the frame that is already on screen. Forcing a 16:9
 * frustum onto a selection made for some other aspect widens the view past the
 * patches that were chosen for it, and every pixel out there is background
 * enclosed by terrain: the tool then reports several percent cracks on a mesh
 * with none. On a 16:9 window it happened to agree, which is why it went
 * unnoticed. Keeping the camera's own aspect is what makes the number mean
 * "holes in the mesh" instead of "holes in the frustum".
 */
export async function auditCracks(
  renderer: WebGPURenderer,
  scene: Scene,
  camera: PerspectiveCamera,
  width = 1280,
  height = Math.max(1, Math.round(width / Math.max(camera.aspect, 1e-3))),
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
  // Deliberately not touching camera.aspect: the default height already
  // matches it, and an explicit width/height pair is the caller saying it
  // knows what it wants.
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
