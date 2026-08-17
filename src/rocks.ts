import {
  BufferGeometry,
  InstancedMesh,
  Matrix4,
  Mesh,
  MeshStandardMaterial,
  Quaternion,
  Vector3,
} from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { SimplifyModifier } from 'three/examples/jsm/modifiers/SimplifyModifier.js';
import { heightAt } from './heightCPU.js';
import { RADIUS } from './planet.js';

const CAPACITY = 360;
const RANGE = 520;
const Y = new Vector3(0, 1, 0);

const fract = (x: number): number => x - Math.floor(x);
const hash = (x: number): number => fract(Math.sin(x * 127.1) * 43758.5453);

interface Placement {
  world: Vector3;
  up: Vector3;
  yaw: number;
  scale: number;
}

export class Rocks {
  readonly meshes: InstancedMesh[];

  private placements: Placement[] = [];
  private centre = new Vector3(Infinity, Infinity, Infinity);
  private matrix = new Matrix4();
  private rotation = new Quaternion();
  private yawRotation = new Quaternion();
  private scale = new Vector3();
  private relative = new Vector3();

  private constructor(geometries: BufferGeometry[], material: MeshStandardMaterial) {
    this.meshes = geometries.map((geometry) => {
      const mesh = new InstancedMesh(geometry, material, CAPACITY);
      mesh.count = 0;
      mesh.frustumCulled = false;
      mesh.matrixAutoUpdate = false;
      return mesh;
    });
  }

  static async load(maxAnisotropy: number): Promise<Rocks> {
    const root = (import.meta as { env?: { BASE_URL?: string } }).env?.BASE_URL ?? '/';
    const gltf = await new GLTFLoader().loadAsync(`${root}rocks/rock_07.gltf`);
    let source: Mesh | undefined;
    gltf.scene.traverse((object) => {
      if (!source && object instanceof Mesh) source = object;
    });
    if (!source) throw new Error('rock model contains no mesh');
    const mesh = source as Mesh;
    const geometry = mesh.geometry.clone();
    geometry.computeVertexNormals();
    const modifier = new SimplifyModifier();
    const vertices = geometry.getAttribute('position').count;
    const simplify = (ratio: number): BufferGeometry => {
      try {
        const result = modifier.modify(geometry.clone(), Math.max(12, Math.floor(vertices * (1 - ratio))));
        result.computeVertexNormals();
        return result;
      } catch {
        return geometry.clone();
      }
    };
    const material = (Array.isArray(mesh.material) ? mesh.material[0] : mesh.material).clone() as MeshStandardMaterial;
    material.roughness = 0.78;
    material.metalness = 0;
    for (const texture of [material.map, material.normalMap, material.roughnessMap, material.aoMap]) {
      if (texture) texture.anisotropy = Math.min(8, maxAnisotropy);
    }
    return new Rocks([geometry, simplify(0.58), simplify(0.86)], material);
  }

  update(cameraIn: readonly [number, number, number], octaves: number, heightScale: number): void {
    const camera = new Vector3(cameraIn[0], cameraIn[1], cameraIn[2]);
    if (camera.distanceToSquared(this.centre) > 180 * 180) this.regenerate(camera, octaves, heightScale);
    const counts = [0, 0, 0];
    for (const placement of this.placements) {
      const distance = placement.world.distanceTo(camera);
      const lod = distance < 55 ? 0 : distance < 190 ? 1 : 2;
      const slot = counts[lod]++;
      if (slot >= CAPACITY) continue;
      this.relative.copy(placement.world).sub(camera);
      this.rotation.setFromUnitVectors(Y, placement.up);
      this.yawRotation.setFromAxisAngle(placement.up, placement.yaw);
      this.rotation.premultiply(this.yawRotation);
      this.scale.setScalar(placement.scale);
      this.matrix.compose(this.relative, this.rotation, this.scale);
      this.meshes[lod].setMatrixAt(slot, this.matrix);
    }
    this.meshes.forEach((mesh, lod) => {
      mesh.count = Math.min(counts[lod], CAPACITY);
      mesh.instanceMatrix.needsUpdate = true;
      mesh.visible = mesh.count > 0;
    });
  }

  private regenerate(camera: Vector3, octaves: number, heightScale: number): void {
    this.centre.copy(camera);
    this.placements = [];
    const up = camera.clone().normalize();
    let east = new Vector3().crossVectors(new Vector3(0, 1, 0), up);
    if (east.lengthSq() < 1e-6) east = new Vector3(1, 0, 0);
    east.normalize();
    const north = new Vector3().crossVectors(up, east).normalize();
    const seed = Math.floor(camera.x / 173) * 73856093 ^ Math.floor(camera.y / 173) * 19349663 ^ Math.floor(camera.z / 173) * 83492791;
    for (let i = 0; i < 780; i++) {
      const rr = Math.sqrt(hash(seed + i * 7 + 1)) * RANGE;
      const angle = hash(seed + i * 7 + 2) * Math.PI * 2;
      const x = Math.cos(angle) * rr;
      const z = Math.sin(angle) * rr;
      const dir = camera.clone().addScaledVector(east, x).addScaledVector(north, z).normalize();
      const h = heightAt([dir.x, dir.y, dir.z], octaves, heightScale);
      if (h <= 2) continue;
      const eps = 3;
      const dirE = dir.clone().addScaledVector(east, eps / camera.length()).normalize();
      const dirN = dir.clone().addScaledVector(north, eps / camera.length()).normalize();
      const slope = Math.hypot(
        heightAt([dirE.x, dirE.y, dirE.z], octaves, heightScale) - h,
        heightAt([dirN.x, dirN.y, dirN.z], octaves, heightScale) - h,
      ) / eps;
      const chance = 0.08 + Math.min(0.48, slope * 0.9);
      if (hash(seed + i * 7 + 3) > chance) continue;
      const world = dir.multiplyScalar(RADIUS + h);
      const scale = (2.2 + hash(seed + i * 7 + 4) * 8.5) * (0.55 + Math.min(slope, 0.8));
      this.placements.push({ world, up: world.clone().normalize(), yaw: hash(seed + i * 7 + 5) * Math.PI * 2, scale });
      if (this.placements.length >= CAPACITY) break;
    }
  }
}
