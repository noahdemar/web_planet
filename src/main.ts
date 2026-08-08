/**
 * world_sim — M1.
 *
 * Cube-sphere quadtree, CDLOD, camera-relative precision, analytic noise.
 * Verification targets (SPEC.md §12): orbit → ground with no cracks, no
 * precision jitter, stable frame time.
 */

import { PerspectiveCamera, Scene, Vector3 } from 'three';
import { WebGPURenderer } from 'three/webgpu';
import { DEFAULT_LOD_FACTOR, MAX_LEVEL, RADIUS } from './planet.js';
import { PatchSelector } from './quadtree.js';
import { TerrainMesh, type ShadeMode } from './terrainMesh.js';
import { FlyControls } from './controls.js';
import { Hud } from './hud.js';
import { directionToFace } from './cubesphere.js';
import { normalize } from './math/vec3d.js';
import { auditCracks } from './devAudit.js';

const GRID_STEPS = [0, 100, 10, 1, 0.1];

function fatal(err: unknown): void {
  const box = document.getElementById('err')!;
  const msg = err instanceof Error ? `${err.message}\n\n${err.stack ?? ''}` : String(err);
  box.querySelector('div')!.innerHTML =
    `<b>Could not start</b>${msg.replace(/[<>&]/g, (c) => ({ '<': '&lt;', '>': '&gt;', '&': '&amp;' })[c]!)}`;
  box.style.display = 'flex';
  console.error(err);
}

async function main(): Promise<void> {
  if (!('gpu' in navigator)) {
    throw new Error(
      'WebGPU is not available in this browser.\n\n' +
        'M1 needs compute-capable WebGPU — WebGL2 has no compute shaders, and ' +
        'the bake pipeline at M3 depends on them. Use Chrome 113+, Edge 113+, ' +
        'or Safari 18+.',
    );
  }

  const renderer = new WebGPURenderer({
    antialias: true,
    // The orbit-to-ground depth range is ~7 orders of magnitude; a linear
    // depth buffer cannot hold it (SPEC.md §7).
    logarithmicDepthBuffer: true,
  });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.setSize(innerWidth, innerHeight);
  renderer.setClearColor(0x05070a, 1);
  document.body.appendChild(renderer.domElement);
  await renderer.init();

  const scene = new Scene();
  const camera = new PerspectiveCamera(62, innerWidth / innerHeight, 1, 1e7);

  const selector = new PatchSelector(DEFAULT_LOD_FACTOR);
  const terrain = new TerrainMesh(selector.buffers);
  scene.add(terrain.mesh);

  const controls = new FlyControls(renderer.domElement);
  const hud = new Hud(document.getElementById('hud')!);

  let lodFactor = DEFAULT_LOD_FACTOR;
  let maxLevel = MAX_LEVEL;
  let gridIdx = 0;

  addEventListener('resize', () => {
    camera.aspect = innerWidth / innerHeight;
    renderer.setSize(innerWidth, innerHeight);
  });

  addEventListener('keydown', (e) => {
    switch (e.code) {
      case 'Digit1': case 'Digit2': case 'Digit3': case 'Digit4':
        terrain.setMode((+e.code.slice(5) - 1) as ShadeMode);
        break;
      case 'KeyG':
        gridIdx = (gridIdx + 1) % GRID_STEPS.length;
        terrain.setGrid(GRID_STEPS[gridIdx]);
        break;
      case 'BracketLeft':
        lodFactor = Math.max(0.5, lodFactor - 0.2);
        selector.setLodFactor(lodFactor);
        break;
      case 'BracketRight':
        lodFactor = Math.min(8, lodFactor + 0.2);
        selector.setLodFactor(lodFactor);
        break;
      case 'Comma':
        terrain.setOctaves(Math.max(1, terrain.octaves - 1));
        break;
      case 'Period':
        terrain.setOctaves(Math.min(20, terrain.octaves + 1));
        break;
      case 'Minus':
        maxLevel = Math.max(0, maxLevel - 1);
        selector.setMaxLevel(maxLevel);
        break;
      case 'Equal':
        maxLevel = Math.min(MAX_LEVEL, maxLevel + 1);
        selector.setMaxLevel(maxLevel);
        break;
      case 'KeyF':
        controls.groundFollow = !controls.groundFollow;
        break;
      case 'KeyT':
        controls.dropToSurface();
        break;
      case 'KeyL':
        controls.gotoRuggedLand();
        break;
      case 'KeyR':
        controls.reset();
        break;
    }
  });

  // Rolling worst-case frame time — a mean hides exactly the hitches that
  // matter (SPEC.md I5).
  const window2s: number[] = [];
  let prev = performance.now();
  let smoothed = 16.7;

  renderer.setAnimationLoop(() => {
    const now = performance.now();
    const dt = Math.min(0.1, (now - prev) / 1000);
    prev = now;

    controls.setTerrain(terrain.octaves, terrain.heightScale);
    controls.update(dt, camera);

    // Near/far track altitude: at eye height the near plane is centimetres,
    // from orbit it is kilometres. Far reaches past the geometric horizon so
    // distant peaks stay visible over the curve.
    const alt = Math.max(1, controls.altitude);
    const camR = controls.radius;
    camera.near = Math.max(0.05, Math.min(alt * 0.02, 2e5));
    const horizon = camR > RADIUS ? Math.sqrt(Math.max(0, camR * camR - RADIUS * RADIUS)) : 0;
    // Far reaches the tangent horizon plus enough slack for peaks beyond it.
    camera.far = Math.max(horizon + 4.0e5, camera.near * 50);
    camera.updateProjectionMatrix();

    terrain.setCameraPosition(controls.pos[0], controls.pos[1], controls.pos[2]);
    terrain.setReferenceRadius(controls.groundRadius);
    const stats = selector.select(
      controls.pos,
      controls.groundRadius,
      camera.matrixWorldInverse,
      camera.projectionMatrix,
    );
    terrain.update(stats.patches);

    renderer.render(scene, camera);

    const frameMs = performance.now() - now;
    smoothed += (frameMs - smoothed) * 0.1;
    window2s.push(frameMs);
    if (window2s.length > 120) window2s.shift();

    hud.render(
      {
        fps: 1000 / Math.max(smoothed, 0.01),
        frameMs: smoothed,
        worstMs: Math.max(...window2s),
        altitude: controls.altitude,
        radius: camR,
        speed: controls.speed,
        speedMul: controls.speedMul,
        face: directionToFace(normalize(controls.pos)).face,
        octaves: terrain.octaves,
        lodFactor,
        maxLevel,
        gridSpacing: terrain.grid,
        shadeMode: terrain.shadeMode,
        groundFollow: controls.groundFollow,
        stats,
      },
      now,
    );
  });

  // A low sun rakes the terrain and makes both relief and any cracks obvious.
  terrain.setSun(new Vector3(0.62, 0.28, 0.73));

  // Handle for driving the sim from the devtools console.
  Object.assign(window, {
    sim: {
      renderer,
      scene,
      camera,
      controls,
      terrain,
      selector,
      audit: () => auditCracks(renderer, scene, camera),
    },
  });
}

main().catch(fatal);
