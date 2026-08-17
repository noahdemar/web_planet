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
  readonly normalRoughness: DataArrayTexture;
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
  const pixels = TERRAIN_MATERIAL_SIZE * TERRAIN_MATERIAL_SIZE * 4;
  const albedoData = new Uint8Array(pixels * TERRAIN_MATERIAL_NAMES.length);
  const normalRoughnessData = new Uint8Array(pixels * TERRAIN_MATERIAL_NAMES.length);
  await Promise.all(TERRAIN_MATERIAL_NAMES.map(async (name, index) => {
    const [albedo, normal, roughness] = await Promise.all([
      imageData(`${root}materials/${name}_albedo.jpg`),
      imageData(`${root}materials/${name}_normal.jpg`),
      imageData(`${root}materials/${name}_roughness.jpg`),
    ]);
    for (let p = 0; p < pixels; p += 4) normal[p + 3] = roughness[p];
    albedoData.set(albedo, index * pixels);
    normalRoughnessData.set(normal, index * pixels);
  }));
  return {
    albedo: makeArray(albedoData, true, maxAnisotropy),
    normalRoughness: makeArray(normalRoughnessData, false, maxAnisotropy),
  };
}
