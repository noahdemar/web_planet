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
import { planetSurface, seedFromLocation } from './planetSource.js';
import { AutoExposure } from './exposure.js';
import { ALTITUDES, SITES } from './tour.js';
import { heightAt, setPlanetSurface } from './heightCPU.js';
import { PatchSelector } from './quadtree.js';
import { TerrainMesh, type ShadeMode } from './terrainMesh.js';
import { FlyControls } from './controls.js';
import { MobileControls } from './mobileControls.js';
import { Grass } from './grass.js';
import { Hud } from './hud.js';
import { directionToFace } from './cubesphere.js';
import { normalize } from './math/vec3d.js';
import { auditCracks, probeDrawnHeight } from './devAudit.js';
import { Vegetation } from './vegetation.js';
import { Sky } from './sky.js';
import { Clouds } from './clouds.js';
import { CASCADES, CASTER_DEPTH, Shadows } from './shadows.js';
import { ShadowInspector } from './shadowDebug.js';
import { createShadowUniforms, makeShadowFactor } from './shaders/shadowSample.js';

const GRID_STEPS = [0, 100, 10, 1, 0.1];

/**
 * LOD factor for the shadow pass.
 *
 * Was 1.1 against the display's 2.2 — half the linear tessellation, so the
 * caster mesh was about four times coarser in area than the surface receiving
 * the shadow. A shadow silhouette then follows the *caster's* triangle edges,
 * and near the ground that read as a comb of triangular teeth along every
 * light/dark boundary. The argument for coarsening it was that a cascade texel
 * is 1.4 m so finer geometry cannot be recorded — true for the far cascade and
 * false for the near one, which is exactly where the teeth were.
 *
 * 2.0 keeps a little of the saving and puts the caster within one level of the
 * receiver almost everywhere. The pass is vertex-bound, so this is the lever
 * to pull back if the shadow pass ever shows up in the frame budget.
 */
const SHADOW_LOD_FACTOR = 2.0;

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
    // ── no logarithmic depth ──────────────────────────────────────────────
    //
    // SPEC.md §7 asks for this because the orbit-to-ground depth range spans
    // ~7 orders of magnitude. What actually solves that here is the *adaptive*
    // near/far below: both are recomputed every frame from altitude, so a
    // single frame never spans more than it has to. Logarithmic depth was the
    // second belt, and it was tearing holes in the mesh.
    //
    // three's WebGPU path evaluates the log depth per *vertex* and lets the
    // rasteriser interpolate it. Log depth is not linear in screen space, so
    // the interpolated value is wrong across a triangle's interior, and it is
    // most wrong where the depth gradient across the triangle is steepest.
    // Fragments then lose the depth test against nothing and the background
    // shows through. That is exactly where the two long-standing artefacts
    // were: coastlines seen from above, where a triangle spans sea level and a
    // hillside, and flat ground at grazing angles, where a triangle spans
    // hundreds of metres of range. The first was "small dark speckles along
    // coastlines" (LESSONS §13) and the second read as a picket fence of
    // slivers at patch borders, which is why it looked like a skirt problem.
    //
    // Measured with sim.audit() over the same coastline, all patches at one
    // level, background pixels enclosed by terrain:
    //
    //   lodFactor 1.2 / 2.2 / 4.4    on: 0.116% / 0.158% / 0.292%
    //                               off: 0.000% / 0.000% / 0.000%
    //
    // The cost is real and worth stating: at eye height near is 5 cm and far
    // reaches past the horizon for distant peaks, so a 24-bit buffer is spread
    // thin at range. Nothing in the scene is coplanar out there — the terrain
    // is one surface, the skirt hangs below it, and vegetation is inside the
    // near few kilometres where precision is ample — so it does not show. If
    // z-fighting ever does appear on distant terrain, this is the line that
    // bought it, and the fix is reversed-Z rather than turning this back on.
    logarithmicDepthBuffer: false,
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
  const shadowInspector = new ShadowInspector(shadows);
  const shadowU = createShadowUniforms();
  const shadowFactor = makeShadowFactor(shadowU, shadows);
  const shadowOf = (rel: unknown) => shadowFactor(rel as never) as never;

  // The M3 bake. Everything the terrain draws sits on top of this, so it is
  // loaded before anything is built rather than streamed in — a planet whose
  // continents appear a second late is worse than a slightly longer start.
  // The planet, from the cache, the shipped asset, or a bake in a worker —
  // see planetSource.ts. The seed comes from ?seed=, so a different world is a
  // URL rather than a rebuild.
  const boot = document.getElementById('boot')!;
  const seed = seedFromLocation();
  const surface = await planetSurface(seed, () => {
    // The boot screen now shows a single static message.
  });

  const selector = new PatchSelector(DEFAULT_LOD_FACTOR);
  selector.setPlanetSurface(surface);
  setPlanetSurface(surface);
  const terrain = new TerrainMesh(selector.buffers, surface, shadowOf);
  scene.add(terrain.mesh);

  const sky = new Sky();
  scene.add(sky.mesh);

  const clouds = new Clouds();
  scene.add(clouds.mesh);

  const vegetation = new Vegetation(shadowOf);
  for (const m of vegetation.meshes) scene.add(m);

  // Ground clutter. Its own pass rather than a fourth vegetation band — see
  // the note at the top of grass.ts.
  const grass = new Grass(shadowOf);
  scene.add(grass.mesh);

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

  let mobileControls: MobileControls | undefined;

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
      case 'Digit5': case 'Digit6': case 'Digit7':
        terrain.setMode((+e.code.slice(5) - 1) as ShadeMode);
        break;
      case 'KeyG':
        gridIdx = (gridIdx + 1) % GRID_STEPS.length;
        terrain.setGrid(GRID_STEPS[gridIdx]);
        break;
      case 'KeyM':
        shadowInspector.cycle();
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
        // Walk supersedes plain ground-follow: it is the same pinning plus the
        // two constraints that stop it being flight at ankle height.
        controls.setWalk(!controls.walk);
        mobileControls?.syncState();
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
        // Off the clock and onto the camera-following sun, which is a
        // debugging aid: it guarantees the ground in front of you is lit.
        sunFollow = !sunFollow;
        if (sunFollow) aimSun(sunEl, sunAz);
        else applySun(sunFromClock());
        break;
      case 'KeyJ':
        dayRunning = !dayRunning;
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

  // The boot screen goes when there is something behind it, not when the data
  // arrives: the first frame at level 0 still costs a moment of shader compile,
  // and fading out onto a black canvas reads as a failure.
  let firstFrame = true;

  renderer.setAnimationLoop(() => {
    const now = performance.now();
    const dt = Math.min(0.1, (now - prev) / 1000);
    prev = now;

    controls.setTerrain(terrain.octaves, terrain.heightScale);
    controls.update(dt, camera);
    if (sunFollow) {
      aimSun(sunEl, sunAz);
    } else {
      if (dayRunning) timeOfDay = (timeOfDay + dt / dayLength) % 1;
      applySun(sunFromClock());
    }

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
    clouds.update(controls.pos[0], controls.pos[1], controls.pos[2], now * 0.001);
    // The ground samples the same cloud field to shadow itself — see the cloud
    // shadow block in shadeTerrain. Same coverage, same clock, or the shadow
    // separates from the cloud casting it.
    terrain.setClouds(clouds.coverage, now * 0.001);
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
    vegetation.setLodFactor(lodFactor);
    vegetation.update(renderer, selector.vegTileData, stats.vegTiles);
    grass.update(
      renderer,
      surface,
      controls.pos as readonly [number, number, number],
      terrain.octaves,
      1,
      now * 0.001,
      controls.altitude, // already the clearance above the surface
      controls.forward,
      camera.fov,
      camera.aspect,
    );

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
        // The cap is a *sphere* and a cascade is a *box*, so it has to reach
        // the box's far corner or the corners hold no casters. The centre is
        // led 0.65 radii ahead and dropped to the surface, and the corner is
        // another radius x sqrt(2) out, so the far corner sits 2.07 radii from
        // the camera's ground point and `alt` above it. At 1.5 the corners were
        // simply not selected — the missing wedge the inspector draws magenta.
        const far2 = shadows.radii[CASCADES - 1] * (0.65 + Math.SQRT2);
        selector.setDistanceCap(Math.hypot(far2, alt) * 1.08);
        selector.setLodFactor(SHADOW_LOD_FACTOR);
        selector.setMaxLevel(17);
        selector.setFrustumCull(false);
        const shadowStats = selector.select(
          controls.pos,
          controls.groundRadius,
          camera.matrixWorldInverse,
          camera.projectionMatrix,
        );
        terrain.update(shadowStats.patches);
        // Hidden for the shadow pass: everything in the scene that is *not*
        // a registered caster and therefore keeps its display material.
        //
        // Those materials sample the shadow maps, and during this pass the maps
        // are render attachments — binding a texture for reading while it is
        // attached for writing fails WebGPU validation and invalidates the
        // whole command buffer, so the shadow pass silently produced nothing.
        // Casters are safe because their material is swapped for a depth one
        // that samples no shadows; the sky and cloud domes were already here
        // for a different reason (they would fill every map).
        shadows.render(renderer, scene, shadowCasters, [
          sky.mesh,
          clouds.mesh,
          grass.mesh,
          ...vegetation.meshes.slice(2),
        ]);
        renderer.setClearColor(0x000000, 1);

        // Restore the display selection.
        selector.setFrustumCull(true);
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
    // The ramp has to span the cascade's *depth* extent, not its radius: the
    // light frustum reaches CASTER_DEPTH either side of the centre, so a ramp
    // of one radius clips most of the map to flat black and flat white and
    // hides exactly the structure this is for.
    shadowInspector.setRange(shadows.radii[Math.max(0, shadowInspector.cascade)] + CASTER_DEPTH);
    shadowInspector.render(renderer);
    if (firstFrame) {
      firstFrame = false;
      boot.classList.add('gone');
      setTimeout(() => { boot.style.display = 'none'; }, 600);
    }

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
    grass.setSun(sun, sunColour);
    sky.setSun(sun, sunColour);
    clouds.setSun(sun, sunColour);
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
  let sunFollow = false;

  // ── true day and night ────────────────────────────────────────────────
  //
  // `sunFollow` re-aims the sun from the *camera's* local up every frame, so
  // wherever you stand it is mid-morning: there is no night on this planet and
  // never has been, only a terminator drawn from orbit by the fact that the
  // sub-camera point is not the sub-solar one.
  //
  // A real cycle is one world-fixed direction that rotates about the planet's
  // spin axis. The planet does not turn — nothing here is in a rotating frame —
  // so turning the sun the other way is the same thing and costs nothing.
  //
  // The declination is the season: the axis is tilted, so the sub-solar point
  // moves off the equator and the poles get their midnight sun and their polar
  // night out of the same two lines. Held at a fixed value rather than run on
  // a second, year-long clock, because a season that changes while you watch
  // is a worse lie than one that does not change at all.
  const AXIAL_TILT = (23.44 * Math.PI) / 180;
  /** Seasonal phase, 0 = equinox, ±1 = solstice. */
  let season = 0.38;
  /** Fraction of a day, [0,1). */
  let timeOfDay = 0.31;
  /** Seconds of wall clock per planetary day. Scrub with , and . */
  let dayLength = 240;
  /**
   * Paused at start, resumed with J.
   *
   * A moving sun is the wrong default for the thing people do first, which is
   * look at something and judge it. It also quietly invalidates any A/B: two
   * screenshots of the same viewpoint seconds apart have different lighting,
   * and every comparison this project makes — shadows, cloud seams, terrain
   * relief — is a comparison of shading. The day cycle is a feature you turn
   * on, not a clock you have to outrun.
   */
  let dayRunning = false;

  function sunFromClock(): Vector3 {
    const a = timeOfDay * Math.PI * 2;
    const decl = Math.sin(season * Math.PI * 0.5) * AXIAL_TILT;
    // +Y is the spin axis, matching the latitude the climate uses (|dir.y|).
    return new Vector3(
      Math.cos(a) * Math.cos(decl),
      Math.sin(decl),
      Math.sin(a) * Math.cos(decl),
    ).normalize();
  }

  function applySun(dir: Vector3): void {
    sun.copy(dir).normalize();
    terrain.setSun(sun, sunColour);
    vegetation.setSun(sun, sunColour);
    grass.setSun(sun, sunColour);
    sky.setSun(sun, sunColour);
    clouds.setSun(sun, sunColour);
  }
  aimSun(sunEl, sunAz);

  let currentShadeMode = 0;
  const cycleShadeMode = () => {
    currentShadeMode = (currentShadeMode + 1) % 7;
    terrain.setMode(currentShadeMode as ShadeMode);
  };
  const toggleSun = () => {
    sunFollow = !sunFollow;
    if (sunFollow) aimSun(sunEl, sunAz);
    else applySun(sunFromClock());
  };

  mobileControls = new MobileControls({
    controls,
    hud,
    terrain,
    onToggleSun: toggleSun,
    onCycleShadeMode: cycleShadeMode,
  });

  const exposure = new AutoExposure();
  /** Where sim.tour.next() is up to. */
  let tourAt = 0;

  /**
   * The visual half of the realism suite — see src/tour.ts.
   *
   * `npm run realism` checks everything about these sites that is a number.
   * This is for everything that is not: whether the ground reads as a place,
   * whether the biome's colour and its shape describe the same country,
   * whether a horizon looks like a horizon. Same fixed sites, so the two
   * halves are talking about the same planet.
   *
   *   sim.tour()                  list the sites
   *   sim.tour('desert-flat')     or sim.tour(3)
   *   sim.tour(3, 12000)          the same site from 12 km
   *   sim.tour.next()             step through, wrapping
   *
   * The sun is aimed in the site's *own* horizon frame every time, so the
   * lighting is the same relative to the camera at every stop. Without that
   * one azimuth means a different thing at each site and half the tour comes
   * out backlit, which is not a property of the terrain.
   */
  const tourGo = (which?: number | string, altitude?: number, elevationDeg = 34, azimuthDeg = 130): string | string[] => {
    if (which === undefined) {
      return SITES.map((s, i) => `${String(i).padStart(2)}  ${s.key.padEnd(18)} ${s.name}`);
    }
    const i = typeof which === 'number'
      ? ((which % SITES.length) + SITES.length) % SITES.length
      : SITES.findIndex((s) => s.key === which);
    const site = SITES[i];
    if (!site) return `no such site: ${which}`;
    tourAt = i;
    const alt = altitude ?? ALTITUDES[1];
    const g = RADIUS + heightAt(site.dir, terrain.octaves, terrain.heightScale);
    controls.pos = [site.dir[0] * (g + alt), site.dir[1] * (g + alt), site.dir[2] * (g + alt)];
    controls.groundFollow = alt <= 3;
    // Look at the ground a little ahead rather than at the horizon or
    // straight down: from 1.7 m that is the horizon anyway, and from
    // 120 km it is the limb.
    controls.pitch = -Math.min(1.1, 0.12 + 0.5 * Math.log10(1 + alt / 300));
    controls.speedMul = 1;
    // Through the module's own sun state, not by calling aimSun directly: the
    // frame loop re-aims from sunEl/sunAz every frame while sunFollow is on, so
    // a direct call is overwritten before it is ever drawn.
    sunEl = elevationDeg;
    sunAz = azimuthDeg;
    sunFollow = true;
    aimSun(sunEl, sunAz);
    exposure.snap(Math.sin((elevationDeg * Math.PI) / 180), site.ground, alt);
    return `${i}  ${site.key} — ${site.name}, ${alt} m up, ground ${site.ground} m`;
  };

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
      clouds,
      shadows,
      shadowInspector,
      aimSun,
      // Both vegetation and the sky are hidden for the audit. Crowns
      // silhouetted against the sky would read as enclosed background, and the
      // sky dome paints over the probe colour entirely — either one makes the
      // measurement meaningless rather than merely noisy.
      audit: async () => {
        const wasVeg = vegetation.isEnabled;
        const wasSky = sky.mesh.visible;
        const wasCloud = clouds.mesh.visible;
        vegetation.setEnabled(false);
        sky.mesh.visible = false;
        clouds.mesh.visible = false;
        try {
          return await auditCracks(renderer, scene, camera);
        } finally {
          vegetation.setEnabled(wasVeg);
          sky.mesh.visible = wasSky;
          clouds.mesh.visible = wasCloud;
        }
      },
      vegCounts: () => vegetation.readCounts(renderer),
      tour: Object.assign(tourGo, {
        next: (altitude?: number) => tourGo(tourAt + 1, altitude),
        prev: (altitude?: number) => tourGo(tourAt - 1, altitude),
        altitudes: ALTITUDES,
      }),
      /**
       * The elevation the GPU drew at the centre of the view, against the
       * elevation heightCPU says is there.
       *
       * The comparison has to be at the *same direction* on both sides, and the
       * centre pixel is not the nadir — the pitch limit keeps the camera 2.9°
       * off it, and if the camera is a kilometre up in the air that is fifty
       * metres of ground. Fifty metres is several wavelengths of the finest
       * octave, so comparing against `heightAt` at the nadir reads the
       * amplification's own variance as divergence. So: take the drawn
       * elevation, intersect the centre ray with the sphere of that radius,
       * and evaluate the CPU at the direction that lands on.
       *
       * Vegetation and the sky are hidden for the same reason the crack audit
       * hides them: either one can be the thing under the crosshair, and then
       * the reading is of a leaf.
       */
      probeHeight: async () => {
        const wasVeg = vegetation.isEnabled;
        const wasSky = sky.mesh.visible;
        const wasCloud = clouds.mesh.visible;
        vegetation.setEnabled(false);
        sky.mesh.visible = false;
        clouds.mesh.visible = false;
        try {
          const gpu = await probeDrawnHeight(
            renderer, scene, camera, (m) => terrain.setMode(m as ShadeMode), terrain.shadeMode,
          );
          const p = controls.pos;
          const f = controls.forward;
          const rho = RADIUS + gpu;
          const b = p[0] * f[0] + p[1] * f[1] + p[2] * f[2];
          const c = p[0] * p[0] + p[1] * p[1] + p[2] * p[2] - rho * rho;
          const disc = b * b - c;
          // No intersection means the centre ray missed the shell the drawn
          // pixel sits on, which only happens if the readback is garbage.
          const t = disc >= 0 ? -b - Math.sqrt(disc) : 0;
          const dir = normalize([p[0] + f[0] * t, p[1] + f[1] * t, p[2] + f[2] * t]);
          const cpu = heightAt(dir, terrain.octaves, terrain.heightScale);
          return { gpu, cpu, delta: cpu - gpu, range: t, dir };
        } finally {
          vegetation.setEnabled(wasVeg);
          sky.mesh.visible = wasSky;
          clouds.mesh.visible = wasCloud;
        }
      },
    },
  });
}

main().catch(fatal);
