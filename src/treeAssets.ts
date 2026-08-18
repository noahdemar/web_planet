/**
 * Scanned tree models, turned into something the GPU-driven scatter can draw.
 *
 * vegetation.ts never touches an instance — placement, gating and binning all
 * happen in a compute pass, and each band is a single indirect draw. That puts
 * three hard constraints on an imported model, and this file exists to satisfy
 * all three before `Vegetation` ever sees the geometry:
 *
 *   one draw is one geometry   every primitive of a species and LOD is merged
 *                              into a single indexed BufferGeometry
 *   one draw is one material   the models' base-colour maps are stacked into a
 *                              DataArrayTexture and the layer rides along as a
 *                              vertex attribute, so bark and leaves shade in
 *                              the same pass
 *   instances carry 16 bytes   the geometry is normalised to unit height with
 *                              its base at the origin, so the only per-instance
 *                              data the vertex shader needs is the vec4 the
 *                              scatter already writes
 *
 * The far bands are impostors baked here rather than authored: eight yaws of
 * each species rendered orthographically into one atlas at boot. That is what
 * makes the handover at 110 m survivable — the quad past that distance is a
 * photograph of the same model, from roughly the right angle, instead of a
 * procedural silhouette that agrees with it only in gross proportion.
 */

import {
  BufferAttribute,
  BufferGeometry,
  ClampToEdgeWrapping,
  DataArrayTexture,
  DataTexture,
  DoubleSide,
  LinearFilter,
  LinearMipmapLinearFilter,
  Matrix4,
  Mesh,
  OrthographicCamera,
  RenderTarget,
  RepeatWrapping,
  RGBAFormat,
  Scene,
  Vector3,
  type Texture,
  type Object3D,
  type Material,
} from 'three';
import { MeshBasicNodeMaterial, type WebGPURenderer } from 'three/webgpu';
import { attribute, int, step, texture as tslTexture } from 'three/tsl';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { VEG_SPECIES } from './planet.js';

/** Yaws baked into the impostor atlas. Also its tile columns. */
export const IMPOSTOR_YAWS = 8;
/**
 * Impostor tile edge, pixels.
 *
 * Eight of these across and one per species down, so 512 puts the atlas at
 * 4096 x 1024 — sixteen megabytes, and worth it: this is the representation
 * almost every tree in the frame is drawn with, and the band it serves starts
 * at 110 m, where a 30 m tree is still a couple of hundred pixels tall.
 */
const IMPOSTOR_TILE = 512;
/**
 * Half-width of the baked view, as a fraction of tree height, and how far the
 * bottom of the tile sits below the base.
 *
 * These two numbers are the contract between the bake and the billboard: the
 * quad drawn at range must span exactly the box the orthographic camera saw,
 * or the impostor is a stretched or cropped copy of the model. Square, so the
 * tile is square and the sampler has no aspect to correct.
 */
export const IMPOSTOR_HALF = 0.62;
export const IMPOSTOR_DROP = 0.12;

/**
 * Transparent margin around each tile, as a fraction of the tile.
 *
 * Atlases and mipmaps do not get along: every level averages a wider
 * neighbourhood, and a few levels down the footprint spills across the tile
 * border and pulls in the tree next door. Because the draw side reads the
 * result as coverage, that spill arrives as a faint alpha wash over the entire
 * quad — a pale rectangle standing in the field where a tree should be.
 *
 * Padding the tile is the standard answer: the bake sees a slightly wider box
 * than the quad spans, so the border texels are empty and the first several
 * mip levels have nothing but transparency to bleed inward. 6% of 512 is 31
 * pixels, which covers the levels anything at these ranges actually samples.
 */
export const IMPOSTOR_PAD = 0.06;

/** Edge of one layer of the base-colour array, pixels. */
const ATLAS_SIZE = 1024;

/**
 * Anisotropic taps on the foliage textures.
 *
 * Foliage is the worst case trilinear filtering has. A leaf card is a flat
 * sheet, and from inside a canopy almost every one of them is seen close to
 * edge-on — so the texture footprint of a pixel is a long thin sliver, and an
 * isotropic mip has to pick one level for both axes. It picks for the long
 * axis, blurs along the short one, and the error it makes changes with every
 * sub-pixel of camera motion. That is the crawl.
 *
 * Four, and that number is measured rather than chosen. Inside a canopy at eye
 * height — the worst overdraw in the renderer — the frame cost is flat from 1
 * to 4 taps and then falls off a cliff:
 *
 *   1x  24.2 ms      4x  23.8 ms
 *   2x  23.1 ms      8x  78.9 ms
 *
 * That is not a curve with an awkward knee, it is a fast path ending. Whatever
 * the driver does for foliage at four taps it stops doing at eight, and the
 * frame triples for a difference you cannot see. Four is the last free step,
 * so four is what this takes.
 */
const ANISOTROPY = 4;

interface LodSpec {
  /** Node names to merge. Null takes every mesh in the file. */
  readonly nodes: readonly string[] | null;
}

interface SpeciesSpec {
  readonly name: string;
  readonly url: string;
  /** One entry per geometry LOD, nearest first. */
  readonly lods: readonly LodSpec[];
}

/**
 * The two species, in the order the scatter's species bit indexes them:
 * 0 conifer, 1 broadleaf. `treeConifer_` in shaders/vegetation.ts is what
 * decides which an instance is, and it has to agree with this order.
 *
 * The broadleaf's near LOD is deliberately not the file's own "high": that
 * mesh spends 6.4k triangles on a *trunk*, which is more than the whole near
 * band can afford and is geometry you cannot see through the canopy anyway.
 * Pairing the high-detail leaves with the low-detail trunk is 4.8k triangles
 * and looks identical — the two LODs are modelled at the same scale, so they
 * interchange cleanly.
 */
const SPECIES: readonly SpeciesSpec[] = [
  {
    name: 'conifer',
    url: 'trees/pine_tree.glb',
    // Only one authored LOD, and at 2.2k triangles it is already cheap
    // enough to serve both geometry bands.
    lods: [{ nodes: null }, { nodes: null }],
  },
];

export interface TreeAssets {
  /** [lod][species] → merged geometry, unit height, base at the origin. */
  readonly geometries: BufferGeometry[][];
  /** Base-colour maps of every material, stacked. */
  readonly albedo: DataArrayTexture;
  /** Baked yaws: IMPOSTOR_YAWS columns by VEG_SPECIES rows. */
  readonly impostor: Texture;
  /** Triangles in each merged geometry, for the HUD. */
  readonly triangles: number[][];
}

/* ── merging ─────────────────────────────────────────────────────────────── */

/** Every Mesh under `root`, with world matrices already resolved. */
function meshesUnder(root: Object3D, names: readonly string[] | null): Mesh[] {
  root.updateWorldMatrix(false, true);
  const out: Mesh[] = [];
  root.traverse((o) => {
    const m = o as Mesh;
    if (!m.isMesh) return;
    if (names && !names.includes(m.name)) return;
    out.push(m);
  });
  return out;
}

/**
 * Model space → the space the vertex shader wants: +Y up, base at the origin,
 * one unit tall.
 *
 * The up axis is taken from the bounding box rather than assumed, because the
 * two files disagree about it — one is Y-up and the other is a converted FBX
 * that is Z-up — and a tree is by a wide margin longest along its trunk, so
 * the longest box axis is a reliable read of which way is up.
 */
function normalising(meshes: Mesh[]): Matrix4 {
  const lo = new Vector3(Infinity, Infinity, Infinity);
  const hi = new Vector3(-Infinity, -Infinity, -Infinity);
  const v = new Vector3();
  for (const m of meshes) {
    const pos = m.geometry.getAttribute('position');
    for (let i = 0; i < pos.count; i++) {
      v.fromBufferAttribute(pos, i).applyMatrix4(m.matrixWorld);
      lo.min(v);
      hi.max(v);
    }
  }

  const ext = [hi.x - lo.x, hi.y - lo.y, hi.z - lo.z];
  let up = 0;
  for (let i = 1; i < 3; i++) if (ext[i] > ext[up]) up = i;

  // Rotate the up axis onto +Y. Only ever a 90° turn, so it is written out
  // rather than derived — and written so the frame stays right-handed, or
  // every face winds backwards and the model culls inside out.
  const r = new Matrix4();
  if (up === 0) r.set(0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1);
  else if (up === 2) r.set(1, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 1);

  const loR = lo.clone().applyMatrix4(r);
  const hiR = hi.clone().applyMatrix4(r);
  const height = Math.max(hiR.y - loR.y, 1e-6);
  const s = 1 / height;

  // Centre on the trunk in plan and put the base on y = 0. Centring matters:
  // the instance position is where the stem meets the ground, and a model
  // whose origin is off to one side plants the whole forest at an offset.
  return new Matrix4()
    .makeTranslation(
      (-(loR.x + hiR.x) / 2) * s,
      -loR.y * s,
      (-(loR.z + hiR.z) / 2) * s,
    )
    .multiply(new Matrix4().makeScale(s, s, s))
    .multiply(r);
}

/** Layer assignment, shared across every species so one array serves all. */
class AlbedoAtlas {
  readonly maps: Texture[] = [];
  private byUuid = new Map<string, number>();

  layer(map: Texture): number {
    const seen = this.byUuid.get(map.uuid);
    if (seen !== undefined) return seen;
    const i = this.maps.length;
    this.byUuid.set(map.uuid, i);
    this.maps.push(map);
    return i;
  }

  /**
   * Two textures over one buffer: the mipped array the forest draws with, and
   * an unmipped copy the impostor bake reads.
   *
   * They cannot be the same object. Mips of a cut-out average alpha as well as
   * colour, and the bake minifies 1024-pixel leaf cards into a 512-pixel tile
   * — so sampling the mipped array there hands the alpha test a value that is
   * a blend of leaf and empty margin, and no cutoff is right for it. A high
   * one keeps only the bole and bakes a forest of bare sticks; a low one keeps
   * the margin too and bakes fat dark blobs. Reading mip zero removes the
   * question: the bake sees exactly the silhouette the artist cut, and the
   * only averaging left is the render target's own.
   */
  async build(maxAniso: number): Promise<{ display: DataArrayTexture; bake: DataArrayTexture }> {
    const n = Math.max(1, this.maps.length);
    const layerPx = ATLAS_SIZE * ATLAS_SIZE * 4;
    const data = new Uint8Array(layerPx * n);
    const canvas = new OffscreenCanvas(ATLAS_SIZE, ATLAS_SIZE);
    // The atlas is read back once per layer, which is exactly the access
    // pattern the hint exists for.
    const ctx = canvas.getContext('2d', { willReadFrequently: true })!;
    for (let i = 0; i < this.maps.length; i++) {
      const src = this.maps[i].image as CanvasImageSource;
      ctx.clearRect(0, 0, ATLAS_SIZE, ATLAS_SIZE);
      ctx.drawImage(src, 0, 0, ATLAS_SIZE, ATLAS_SIZE);
      const px = ctx.getImageData(0, 0, ATLAS_SIZE, ATLAS_SIZE).data;
      dilate(px, ATLAS_SIZE);
      data.set(px, i * layerPx);
    }

    const make = (mips: boolean): DataArrayTexture => {
      const tex = new DataArrayTexture(data, ATLAS_SIZE, ATLAS_SIZE, n);
      // Repeat, because the bark maps are tiled along the trunk and clamping
      // them smears the last row of pixels the length of the bole.
      tex.wrapS = RepeatWrapping;
      tex.wrapT = RepeatWrapping;
      tex.magFilter = LinearFilter;
      // Mips are not optional for the display copy. A leaf card is a few
      // pixels across for most of the geometry bands' range, and unfiltered
      // foliage at that size is the loudest source of crawl in the frame.
      tex.minFilter = mips ? LinearMipmapLinearFilter : LinearFilter;
      tex.generateMipmaps = mips;
      // Only the display copy: the bake reads mip zero head-on, where there is
      // no anisotropy to correct.
      if (mips) tex.anisotropy = Math.min(ANISOTROPY, maxAniso);
      // Left in sRGB on purpose — the shader linearises. Storing linear in
      // eight bits crushes exactly the range foliage lives in.
      tex.needsUpdate = true;
      return tex;
    };
    return { display: make(true), bake: make(false) };
  }
}

/**
 * Push colour outward into the transparent margin, in place.
 *
 * A cut-out leaf texture carries a colour under alpha = 0 that was never meant
 * to be seen, and in both of these files that colour is white paper. Nothing
 * samples it directly — but *everything* samples near it. Bilinear filtering
 * mixes the texel just outside the leaf into the one just inside, and each mip
 * level averages a wider ring of it, so by the time a canopy is thirty metres
 * away half of what reaches the screen is background. That is what made the
 * broadleaves read as pale and dusty next to the conifers rather than green.
 *
 * The fix is the standard one: give the transparent texels the colour of their
 * nearest opaque neighbour, so the filter has nothing foreign to mix in. Alpha
 * is untouched — the shape is still exactly the shape the artist cut.
 *
 * Eight passes of neighbour-fill, and then the rest of the sheet flooded with
 * the mean.
 *
 * Eight was originally justified as "roughly the radius the coarse mips
 * reach", and that was simply wrong arithmetic: one texel of mip L averages
 * 2^L base texels, so mip 3 already spans the entire dilated margin and every
 * level past it reaches into untouched background. On these files that
 * background is white paper, so a canopy minified past 8x — which is any tree
 * beyond about thirty metres — had white averaged into it a little more at
 * each level. It arrived as bright speckle crawling over the crown.
 *
 * Running the neighbour-fill until the sheet is full would fix it and costs
 * hundreds of passes over a megapixel. Filling what the local passes did not
 * reach with the mean opaque colour costs one more sweep and is
 * indistinguishable: those texels only ever contribute through coarse mips,
 * where what matters is that they carry foliage colour rather than paper.
 */
function dilate(px: Uint8ClampedArray, size: number, passes = 8): void {
  const solid = new Uint8Array(size * size);
  for (let i = 0; i < solid.length; i++) solid[i] = px[i * 4 + 3] > 0 ? 1 : 0;

  const next = new Uint8Array(solid);
  for (let p = 0; p < passes; p++) {
    let spread = false;
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        const i = y * size + x;
        if (solid[i]) continue;
        let r = 0;
        let g = 0;
        let b = 0;
        let k = 0;
        for (let dy = -1; dy <= 1; dy++) {
          const yy = y + dy;
          if (yy < 0 || yy >= size) continue;
          for (let dx = -1; dx <= 1; dx++) {
            const xx = x + dx;
            if (xx < 0 || xx >= size) continue;
            const j = yy * size + xx;
            if (!solid[j]) continue;
            r += px[j * 4];
            g += px[j * 4 + 1];
            b += px[j * 4 + 2];
            k++;
          }
        }
        if (k === 0) continue;
        px[i * 4] = r / k;
        px[i * 4 + 1] = g / k;
        px[i * 4 + 2] = b / k;
        next[i] = 1;
        spread = true;
      }
    }
    if (!spread) break;
    solid.set(next);
  }

  // Everything the passes above could not reach, flooded with the mean of what
  // was opaque to begin with.
  let r = 0;
  let g = 0;
  let b = 0;
  let k = 0;
  for (let i = 0; i < solid.length; i++) {
    if (px[i * 4 + 3] === 0) continue;
    r += px[i * 4];
    g += px[i * 4 + 1];
    b += px[i * 4 + 2];
    k++;
  }
  if (k === 0) return;
  r /= k;
  g /= k;
  b /= k;
  for (let i = 0; i < solid.length; i++) {
    if (solid[i]) continue;
    px[i * 4] = r;
    px[i * 4 + 1] = g;
    px[i * 4 + 2] = b;
  }
}

/* ── impostor bake ───────────────────────────────────────────────────────── */

/**
 * Render each species from IMPOSTOR_YAWS directions into one atlas.
 *
 * Unlit: the impostor stores albedo and coverage only, and every light-dependent
 * term — sun, sky, bounce, aerial perspective — is applied at draw time by the
 * same code the geometry bands use. Baking the lighting in would freeze the
 * forest at whatever time of day the page happened to load.
 */
async function bakeImpostors(
  renderer: WebGPURenderer,
  geometries: BufferGeometry[][],
  albedo: DataArrayTexture,
  maxAniso: number,
): Promise<Texture> {
  const w = IMPOSTOR_TILE * IMPOSTOR_YAWS;
  const h = IMPOSTOR_TILE * VEG_SPECIES;

  // One tile-sized target, rendered sixteen times and assembled on the CPU,
  // rather than one atlas-sized target with a viewport per tile.
  //
  // The viewport version is the obvious one and it does not work here: the
  // WebGPU backend derives the render pass's viewport and scissor from the
  // render target, so setViewport before each render was quietly ignored and
  // every yaw drew over the whole atlas. What came out was two overlapping
  // trees on an otherwise empty sheet, which the draw side then sampled tile
  // by tile — hence impostors that were a smear of somebody else's canopy.
  //
  // Reading each tile back costs a megabyte and a sync, sixteen times, once,
  // at boot. That is a fair price for placement that is simply arithmetic.
  const rt = new RenderTarget(IMPOSTOR_TILE, IMPOSTOR_TILE, { depthBuffer: true });
  const atlas = new Uint8Array(w * h * 4);

  const mat = new MeshBasicNodeMaterial();
  const texel = tslTexture(albedo, attribute('uv', 'vec2')).depth(
    int(attribute('layer', 'float') as never) as never,
  );
  mat.colorNode = texel.rgb as never;
  // Binary coverage, not the texture's own alpha. Whatever survives the cutout
  // is written fully opaque, so the atlas starts its mip chain with the most
  // coverage the silhouette can honestly claim — the draw side has to average
  // that down over several levels and needs the headroom.
  mat.opacityNode = step(0.5, texel.a) as never;
  mat.transparent = false;
  // The cutout itself. Legitimate at a plain half because the source is mip
  // zero: the alpha reaching this test is the texture's own, not a blend of
  // leaf and empty margin, so the usual threshold means what it usually means.
  mat.alphaTest = 0.5;
  // Leaf cards are single-sided sheets; from behind they would vanish.
  mat.side = DoubleSide;

  const scene = new Scene();
  // Wider than the quad spans, by exactly the margin the lookup insets by.
  const bakeHalf = IMPOSTOR_HALF / (1 - 2 * IMPOSTOR_PAD);
  const cam = new OrthographicCamera(-bakeHalf, bakeHalf, bakeHalf, -bakeHalf, 0.1, 8);

  const prevTarget = renderer.getRenderTarget();
  const prevClear = renderer.getClearAlpha();
  renderer.setRenderTarget(rt);
  renderer.setClearColor(0x000000, 0);

  // The centre of the baked box, in tree heights. Must match what the
  // billboard reconstructs — see IMPOSTOR_HALF/IMPOSTOR_DROP.
  const midY = IMPOSTOR_HALF - IMPOSTOR_DROP;
  const row = IMPOSTOR_TILE * 4;

  for (let s = 0; s < VEG_SPECIES; s++) {
    const mesh = new Mesh(geometries[0][s], mat);
    scene.clear();
    scene.add(mesh);
    for (let y = 0; y < IMPOSTOR_YAWS; y++) {
      const a = (y / IMPOSTOR_YAWS) * Math.PI * 2;
      cam.position.set(Math.sin(a) * 3, midY, Math.cos(a) * 3);
      cam.up.set(0, 1, 0);
      cam.lookAt(0, midY, 0);
      cam.updateProjectionMatrix();
      await renderer.renderAsync(scene, cam);

      // Flipped on the way in. The readback hands back rows top-down — row 0
      // is the top of the image, so the crown — while a DataTexture treats
      // row 0 as v = 0, and the quad's v = 0 is the base of the tree. Copying
      // straight through stands the whole forest on its head.
      const px = await renderer.readRenderTargetPixelsAsync(
        rt, 0, 0, IMPOSTOR_TILE, IMPOSTOR_TILE,
      ) as Uint8Array;
      for (let r = 0; r < IMPOSTOR_TILE; r++) {
        const src = (IMPOSTOR_TILE - 1 - r) * row;
        const dst = ((s * IMPOSTOR_TILE + r) * w + y * IMPOSTOR_TILE) * 4;
        atlas.set(px.subarray(src, src + row), dst);
      }
    }
  }

  renderer.setRenderTarget(prevTarget);
  renderer.setClearColor(0x000000, prevClear);
  rt.dispose();

  const tex = new DataTexture(atlas, w, h, RGBAFormat);
  // Clamped: a tile lookup that ran a texel past its edge would otherwise
  // wrap to the far side of the atlas and pull in the neighbouring yaw.
  tex.wrapS = ClampToEdgeWrapping;
  tex.wrapT = ClampToEdgeWrapping;
  tex.magFilter = LinearFilter;
  tex.minFilter = LinearMipmapLinearFilter;
  tex.generateMipmaps = true;
  // Impostors are upright quads seen from a camera that is usually above them,
  // so their footprint is squashed vertically for the same reason the leaf
  // cards' is squashed horizontally.
  tex.anisotropy = Math.min(ANISOTROPY, maxAniso);
  tex.needsUpdate = true;
  return tex;
}

/* ── entry point ─────────────────────────────────────────────────────────── */

/**
 * Load the models and produce everything `Vegetation` needs.
 *
 * Called once, before the vegetation is constructed, because the geometry and
 * the array texture are both baked into the node materials at construction.
 */
export async function loadTreeAssets(
  renderer: WebGPURenderer,
  onProgress?: (t: number) => void,
): Promise<TreeAssets> {
  const root = (import.meta as { env?: { BASE_URL?: string } }).env?.BASE_URL ?? '/';
  const loader = new GLTFLoader();
  const atlas = new AlbedoAtlas();

  const scenes = await Promise.all(
    SPECIES.map(async (sp, i) => {
      const g = await loader.loadAsync(`${root}${sp.url}`);
      onProgress?.((i + 1) / (SPECIES.length + 1));
      return g.scene;
    }),
  );

  const lodCount = SPECIES[0].lods.length;
  const geometries: BufferGeometry[][] = [];
  const triangles: number[][] = [];

  for (let lod = 0; lod < lodCount; lod++) {
    const row: BufferGeometry[] = [];
    const tris: number[] = [];
    for (let s = 0; s < SPECIES.length; s++) {
      const meshes = meshesUnder(scenes[s], SPECIES[s].lods[lod].nodes);
      if (meshes.length === 0) {
        throw new Error(`tree assets: ${SPECIES[s].name} lod ${lod} matched no mesh`);
      }
      // Normalise from the *near* LOD in both cases, so the two LODs of a
      // species end up at exactly the same scale. Deriving each from its own
      // bounding box would let them differ by whatever the decimator trimmed
      // off the silhouette, and the tree would jump at the band boundary.
      const ref = meshesUnder(scenes[s], SPECIES[s].lods[0].nodes);
      const geo = mergeSpecies(meshes, normalising(ref), atlas);
      row.push(geo);
      tris.push(geo.getIndex()!.count / 3);
    }
    geometries.push(row);
    triangles.push(tris);
  }

  const maxAniso = renderer.getMaxAnisotropy();
  const { display: albedo, bake } = await atlas.build(maxAniso);
  onProgress?.(1);
  const impostor = await bakeImpostors(renderer, geometries, bake, maxAniso);
  // The unmipped copy has done its one job and is another sixteen megabytes.
  bake.dispose();
  return { geometries, albedo, impostor, triangles };
}

/**
 * Merge a species' primitives into one geometry.
 *
 * Carries position, normal and uv from the model, plus two attributes the
 * merge invents: `layer`, the slice of the albedo array this vertex reads, and
 * `leaf`, which separates foliage from bark for the shading. Both are constant
 * across a primitive — they are per-vertex only because that is the only
 * channel available once the primitives share a draw.
 */
function mergeSpecies(meshes: Mesh[], norm: Matrix4, atlas: AlbedoAtlas): BufferGeometry {
  const pos: number[] = [];
  const nrm: number[] = [];
  const uv: number[] = [];
  const layer: number[] = [];
  const leaf: number[] = [];
  const idx: number[] = [];

  const p = new Vector3();
  const n = new Vector3();
  // Normals transform by the inverse transpose; the normalising matrix is a
  // rotation and a uniform scale, so the rotation alone is the whole of it.
  const nm = new Matrix4().extractRotation(norm);

  for (const mesh of meshes) {
    const g = mesh.geometry;
    const m = new Matrix4().multiplyMatrices(norm, mesh.matrixWorld);
    const mn = new Matrix4().extractRotation(mesh.matrixWorld).premultiply(nm);

    const aP = g.getAttribute('position');
    const aN = g.getAttribute('normal');
    const aU = g.getAttribute('uv');
    const material = (Array.isArray(mesh.material) ? mesh.material[0] : mesh.material) as
      Material & { map?: Texture | null; transparent?: boolean };
    const map = material.map;
    if (!map) throw new Error(`tree assets: material ${material.name} has no base colour map`);
    const L = atlas.layer(map);
    // Foliage is what the glTF marked as blended. Bark is opaque in both
    // files, and nothing else is in there.
    const isLeaf = material.transparent ? 1 : 0;

    const base = pos.length / 3;
    for (let i = 0; i < aP.count; i++) {
      p.fromBufferAttribute(aP, i).applyMatrix4(m);
      pos.push(p.x, p.y, p.z);
      if (aN) {
        n.fromBufferAttribute(aN, i).applyMatrix4(mn).normalize();
        nrm.push(n.x, n.y, n.z);
      } else {
        nrm.push(0, 1, 0);
      }
      if (aU) uv.push(aU.getX(i), aU.getY(i));
      else uv.push(0, 0);
      layer.push(L);
      leaf.push(isLeaf);
    }

    const gi = g.getIndex();
    if (gi) for (let i = 0; i < gi.count; i++) idx.push(base + gi.getX(i));
    else for (let i = 0; i < aP.count; i++) idx.push(base + i);
  }

  const out = new BufferGeometry();
  out.setAttribute('position', new BufferAttribute(new Float32Array(pos), 3));
  out.setAttribute('normal', new BufferAttribute(new Float32Array(nrm), 3));
  out.setAttribute('uv', new BufferAttribute(new Float32Array(uv), 2));
  out.setAttribute('layer', new BufferAttribute(new Float32Array(layer), 1));
  out.setAttribute('leaf', new BufferAttribute(new Float32Array(leaf), 1));
  out.setIndex(new BufferAttribute(new Uint32Array(idx), 1));
  return out;
}
