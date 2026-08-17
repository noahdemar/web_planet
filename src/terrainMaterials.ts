import {
  DataArrayTexture,
  LinearFilter,
  LinearMipmapLinearFilter,
  NoColorSpace,
  RepeatWrapping,
  SRGBColorSpace,
} from 'three';

export const TERRAIN_MATERIAL_SIZE = 1024;
export const TERRAIN_MATERIAL_NAMES = ['rock', 'soil', 'litter', 'snow', 'sand'] as const;
export const ROCK_LAYER = 0;
export const SOIL_LAYER = 1;
export const LITTER_LAYER = 2;
export const SNOW_LAYER = 3;
export const SAND_LAYER = 4;

export interface TerrainMaterials {
  readonly albedo: DataArrayTexture;
  readonly normal: DataArrayTexture;
  readonly roughness: DataArrayTexture;
}

async function imageData(url: string): Promise<Uint8ClampedArray> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`terrain material missing: ${url}`);
  const image = await createImageBitmap(await response.blob());
  const canvas = new OffscreenCanvas(TERRAIN_MATERIAL_SIZE, TERRAIN_MATERIAL_SIZE);
  const context = canvas.getContext('2d', { willReadFrequently: true });
  if (!context) throw new Error('terrain material canvas unavailable');
  context.drawImage(image, 0, 0, TERRAIN_MATERIAL_SIZE, TERRAIN_MATERIAL_SIZE);
  image.close();
  return context.getImageData(0, 0, TERRAIN_MATERIAL_SIZE, TERRAIN_MATERIAL_SIZE).data;
}

function makeArray(data: Uint8Array, colour: boolean, anisotropy: number): DataArrayTexture {
  const texture = new DataArrayTexture(
    data,
    TERRAIN_MATERIAL_SIZE,
    TERRAIN_MATERIAL_SIZE,
    TERRAIN_MATERIAL_NAMES.length,
  );
  texture.wrapS = RepeatWrapping;
  texture.wrapT = RepeatWrapping;
  texture.magFilter = LinearFilter;
  texture.minFilter = LinearMipmapLinearFilter;
  texture.generateMipmaps = true;
  texture.anisotropy = Math.min(8, anisotropy);
  texture.colorSpace = colour ? SRGBColorSpace : NoColorSpace;
  texture.needsUpdate = true;
  return texture;
}

export async function loadTerrainMaterials(maxAnisotropy: number): Promise<TerrainMaterials> {
  const root = (import.meta as { env?: { BASE_URL?: string } }).env?.BASE_URL ?? '/';
  const kinds = ['albedo', 'normal', 'roughness'] as const;
  const pixels = TERRAIN_MATERIAL_SIZE * TERRAIN_MATERIAL_SIZE * 4;
  const arrays = await Promise.all(kinds.map(async (kind) => {
    const data = new Uint8Array(pixels * TERRAIN_MATERIAL_NAMES.length);
    const layers = await Promise.all(TERRAIN_MATERIAL_NAMES.map((name) =>
      imageData(`${root}materials/${name}_${kind}.jpg`),
    ));
    layers.forEach((layer, index) => data.set(layer, index * pixels));
    return makeArray(data, kind === 'albedo', maxAnisotropy);
  }));
  return { albedo: arrays[0], normal: arrays[1], roughness: arrays[2] };
}
