/**
 * world_sim — M1.
 *
 * Cube-sphere quadtree, CDLOD, camera-relative precision, analytic noise.
 * Verification targets (SPEC.md §12): orbit → ground with no cracks, no
 * precision jitter, stable frame time.
 */

import { ACESFilmicToneMapping, PerspectiveCamera, Scene, Vector3 } from 'three';
import { WebGPURenderer } from 'three/webgpu';
import { DEFAULT_LOD_FACTOR, MAX_LEVEL, RADIUS } from './planet.js';
import { loadPlanetSurface } from './planetData.js';
import { AutoExposure } from './exposure.js';
import { setPlanetSurface } from './heightCPU.js';
import { PatchSelector } from './quadtree.js';
import { TerrainMesh, type ShadeMode } from './terrainMesh.js';
import { FlyControls } from './controls.js';
import { Hud } from './hud.js';
import { directionToFace } from './cubesphere.js';
import { normalize } from './math/vec3d.js';
import { auditCracks } from './devAudit.js';
import { Vegetation } from './vegetation.js';
import { Sky } from './sky.js';
import { CASCADES, Shadows } from './shadows.js';
import { createShadowUniforms, makeShadowFactor } from './shaders/shadowSample.js';

const GRID_STEPS = [0, 100, 10, 1, 0.1];

/**
 * LOD factor for the shadow pass. Much coarser than the display value: the
 * cost is per vertex, and a shadow map cannot resolve detail finer than its
 * own texel. Too low and the caster surface diverges from the receiver's,
 * which shows up as false self-shadowing.
 */
const SHADOW_LOD_FACTOR = 1.1;

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
    // Real GPU time per pass. CPU-side timing measures queue submission, not
    // work, and every performance decision here has been guesswork without it.
    trackTimestamp: true,
    // The orbit-to-ground depth range is ~7 orders of magnitude; a linear
    // depth buffer cannot hold it (SPEC.md §7).
    logarithmicDepthBuffer: true,
  });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.setSize(innerWidth, innerHeight);
  renderer.setClearColor(0x000000, 1);
  // The shaders emit radiance, not screen colour, so a tone curve is required
  // rather than decorative. ACES also keeps the sun disc and snow from
  // clipping to flat white, which is most of what makes a render look CG.
  renderer.toneMapping = ACESFilmicToneMapping;
  renderer.toneMappingExposure = 0.55;
  document.body.appendChild(renderer.domElement);
  await renderer.init();

  const scene = new Scene();
  const camera = new PerspectiveCamera(62, innerWidth / innerHeight, 1, 1e7);

  const shadows = new Shadows();
  const shadowU = createShadowUniforms();
  const shadowFactor = makeShadowFactor(shadowU, shadows);
  const shadowOf = (rel: unknown) => shadowFactor(rel as never) as never;

  // The M3 bake. Everything the terrain draws sits on top of this, so it is
  // loaded before anything is built rather than streamed in — a planet whose
  // continents appear a second late is worse than a slightly longer start.
  const surface = await loadPlanetSurface();

  const selector = new PatchSelector(DEFAULT_LOD_FACTOR);
  selector.setPlanetSurface(surface);
  setPlanetSurface(surface);
  const terrain = new TerrainMesh(selector.buffers, surface, shadowOf);
  scene.add(terrain.mesh);

  const sky = new Sky();
  scene.add(sky.mesh);

  const vegetation = new Vegetation(shadowOf);
  for (const m of vegetation.meshes) scene.add(m);

  const shadowCasters = [
    { mesh: terrain.mesh, depthMaterial: terrain.depthMaterial },
    // Only the near two bands cast. The far band is ~95% of all instances and
    // its shadows land where aerial perspective has already taken over.
    ...vegetation.meshes.slice(0, 2).map((m, i) => ({
      mesh: m,
      depthMaterial: vegetation.depthMaterials[i],
      maxCascadeRadius: i === 0 ? 400 : 2000,
    })),
  ];
  let shadowsOn = true;

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
      case 'Digit1': case 'Digit2': case 'Digit3': case 'Digit4': case 'Digit5': case 'Digit6':
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
      case 'KeyO':
        sunEl = Math.max(-8, Math.min(88, sunEl + (e.shiftKey ? -4 : 4)));
        aimSun(sunEl, sunAz);
        break;
      case 'KeyP':
        sunAz = (sunAz + (e.shiftKey ? -12 : 12)) % 360;
        aimSun(sunEl, sunAz);
        break;
      case 'KeyK':
        sunFollow = !sunFollow;
        if (sunFollow) aimSun(sunEl, sunAz);
        break;
      case 'KeyH':
        shadowsOn = !shadowsOn;
        break;
      case 'KeyI':
        controls.invertY = !controls.invertY;
        break;
      case 'KeyV':
        vegetation.setEnabled(!vegetation.isEnabled);
        break;
      case 'KeyB':
        vegetation.setMode(vegetation.debugBands ? 0 : 1);
        break;
      case 'Semicolon':
        vegetation.setDensity(vegetation.density - 0.06);
        break;
      case 'Quote':
        vegetation.setDensity(vegetation.density + 0.06);
        break;
      case 'KeyN':
        // Instance counts live only on the GPU; reading them costs a sync, so
        // it is on demand rather than every frame.
        void vegetation.readCounts(renderer);
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
    if (sunFollow) aimSun(sunEl, sunAz);

    // Meter before rendering, from this frame's camera.
    {
      const p = controls.pos;
      const inv = 1 / Math.hypot(p[0], p[1], p[2]);
      const dotUp = (sun.x * p[0] + sun.y * p[1] + sun.z * p[2]) * inv;
      exposure.update(dotUp, controls.groundRadius - RADIUS, controls.altitude, dt);
      renderer.toneMappingExposure = exposure.value;
    }

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
    sky.update(controls.pos[0], controls.pos[1], controls.pos[2], camera.near);
    // Pixels subtended by a one-metre object at one metre — drives the
    // sub-pixel vegetation fade, so it must track viewport and field of view.
    vegetation.setProjectionScale(
      renderer.domElement.height / (2 * Math.tan((camera.fov * Math.PI) / 360)),
    );
    const stats = selector.select(
      controls.pos,
      controls.groundRadius,
      camera.matrixWorldInverse,
      camera.projectionMatrix,
    );
    terrain.update(stats.patches);

    // GPU-driven vegetation: upload tiles, then scatter and bin entirely on
    // the GPU. The draw counts are written by the compute pass, so no instance
    // is ever touched by the CPU (SPEC.md §8).
    vegetation.setCamera(controls.pos[0], controls.pos[1], controls.pos[2]);
    vegetation.setOctaves(terrain.octaves);
    vegetation.update(renderer, selector.vegTileData, stats.vegTiles);

    // Shadow pass before the main render: cascades are placed from this
    // frame's camera, and the sky is excluded because it would fill every map.
    if (shadowsOn) {
      shadows.update(
        new Vector3(controls.forward[0], controls.forward[1], controls.forward[2]),
        sun,
        alt,
        new Vector3(controls.up[0], controls.up[1], controls.up[2]),
      );

      if (shadows.strength > 0) {
        // Re-select terrain for the shadow pass, capped at the largest cascade.
        // The display selection reaches the horizon — tens of kilometres — and
        // none of that belongs in a shadow map.
        // Every shadow vertex pays the full 17-octave terrain shader, so the
        // shadow pass is vertex-bound and patch count is the lever. A coarser
        // LOD is nearly free visually: cascade 2's texel is 1.4 m, so geometry
        // finer than that cannot be recorded anyway.
        selector.setDistanceCap(shadows.radii[CASCADES - 1] * 1.5);
        selector.setLodFactor(SHADOW_LOD_FACTOR);
        selector.setMaxLevel(17);
        const shadowStats = selector.select(
          controls.pos,
          controls.groundRadius,
          camera.matrixWorldInverse,
          camera.projectionMatrix,
        );
        terrain.update(shadowStats.patches);
        shadows.render(renderer, scene, shadowCasters, [sky.mesh]);
        renderer.setClearColor(0x000000, 1);

        // Restore the display selection.
        selector.setDistanceCap(Infinity);
        selector.setLodFactor(lodFactor);
        selector.setMaxLevel(maxLevel);
        selector.select(
          controls.pos,
          controls.groundRadius,
          camera.matrixWorldInverse,
          camera.projectionMatrix,
        );
        terrain.update(stats.patches);
      }
    } else {
      shadows.strength = 0;
    }
    shadowU.sync(shadows, sun);

    renderer.render(scene, camera);

    // Timestamps resolve a frame or two late; that is fine for a readout.
    void renderer.resolveTimestampsAsync().catch(() => {});

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
        veg: vegetation.stats,
        gpu: {
          render: renderer.info.render.timestamp,
          compute: renderer.info.compute.timestamp,
          drawCalls: renderer.info.render.drawCalls,
          triangles: renderer.info.render.triangles,
        },
      },
      now,
    );
  });

  // The sun is fixed in world space, as it must be, but a fixed vector lands
  // at an arbitrary local elevation — the spawn happened to get sunset. Aim it
  // once from the spawn's own horizon frame so the default view is lit, and
  // expose the control for scrubbing time of day.
  const sun = new Vector3();
  const sunColour = new Vector3(1.0, 0.965, 0.92).multiplyScalar(17);

  function aimSun(elevationDeg: number, azimuthDeg: number): void {
    const el = (elevationDeg * Math.PI) / 180;
    const az = (azimuthDeg * Math.PI) / 180;
    const up = normalize(controls.pos);
    const ref: [number, number, number] =
      Math.abs(up[1]) > 0.999 ? [0, 0, 1] : [0, 1, 0];
    const east = normalize([
      ref[1] * up[2] - ref[2] * up[1],
      ref[2] * up[0] - ref[0] * up[2],
      ref[0] * up[1] - ref[1] * up[0],
    ]);
    const north = normalize([
      up[1] * east[2] - up[2] * east[1],
      up[2] * east[0] - up[0] * east[2],
      up[0] * east[1] - up[1] * east[0],
    ]);
    const ce = Math.cos(el);
    sun.set(
      up[0] * Math.sin(el) + (north[0] * Math.cos(az) + east[0] * Math.sin(az)) * ce,
      up[1] * Math.sin(el) + (north[1] * Math.cos(az) + east[1] * Math.sin(az)) * ce,
      up[2] * Math.sin(el) + (north[2] * Math.cos(az) + east[2] * Math.sin(az)) * ce,
    ).normalize();
    terrain.setSun(sun, sunColour);
    vegetation.setSun(sun, sunColour);
    sky.setSun(sun, sunColour);
  }

  let sunEl = 38;
  let sunAz = 130;
  /**
   * Re-aim the sun from the camera's own local up every frame.
   *
   * With a world-fixed sun most of the planet is in darkness, and travelling
   * anywhere means arriving at night — every location visited during the audit
   * needed the sun aimed by hand before it could even be assessed. Following
   * the camera keeps `sunEl` meaning what it says: degrees above *your*
   * horizon. It still produces a terminator from orbit, because at altitude
   * the local up is the sub-camera point and the sun is 38° off it.
   *
   * Press K for a world-fixed sun, which is what you want to watch a real
   * terminator sweep, and wrong for everything else.
   */
  let sunFollow = true;
  aimSun(sunEl, sunAz);

  const exposure = new AutoExposure();

  // Handle for driving the sim from the devtools console.
  Object.assign(window, {
    sim: {
      renderer,
      scene,
      camera,
      controls,
      terrain,
      selector,
      vegetation,
      sky,
      shadows,
      aimSun,
      // Both vegetation and the sky are hidden for the audit. Crowns
      // silhouetted against the sky would read as enclosed background, and the
      // sky dome paints over the probe colour entirely — either one makes the
      // measurement meaningless rather than merely noisy.
      audit: async () => {
        const wasVeg = vegetation.isEnabled;
        const wasSky = sky.mesh.visible;
        vegetation.setEnabled(false);
        sky.mesh.visible = false;
        try {
          return await auditCracks(renderer, scene, camera);
        } finally {
          vegetation.setEnabled(wasVeg);
          sky.mesh.visible = wasSky;
        }
      },
      vegCounts: () => vegetation.readCounts(renderer),
    },
  });
}

main().catch(fatal);
