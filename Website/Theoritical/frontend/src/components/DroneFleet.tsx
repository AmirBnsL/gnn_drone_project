import { Html, useGLTF } from "@react-three/drei";
import { useFrame } from "@react-three/fiber";
import {
  Component,
  ReactNode,
  Suspense,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import * as THREE from "three";
import {
  DRONE_FALLBACK_SIZE_M,
  DRONE_GLB_SIZE_M,
  PROPELLER_ARM_FRAC,
  PROPELLER_SPIN_RAD_PER_S,
} from "../constants";
import type { DroneState } from "../api/client";

/** PyBullet Z-up → Three.js Y-up */
export function simToThree(pos: [number, number, number]): [number, number, number] {
  return [pos[0], pos[2], pos[1]];
}

const PROP_MESH_NAME_RE = /propeller|prop|rotor|blade|fan/i;
const PROP_MESH_EXCLUDE_RE =
  /body|chassis|frame|arm|hub|base|scene|root|crazyflie|collision|shell|mount|cap|link/i;

const PROC_ARM_POSITIONS: [number, number][] = [
  [1, 1],
  [-1, 1],
  [1, -1],
  [-1, -1],
];

function findPropellerMeshes(root: THREE.Object3D, droneSize: number): THREE.Mesh[] {
  const meshes: THREE.Mesh[] = [];
  const maxPropDim = droneSize * 0.4;

  root.traverse((child) => {
    if (!(child instanceof THREE.Mesh) || !child.name) return;
    const name = child.name;
    if (PROP_MESH_EXCLUDE_RE.test(name)) return;
    if (!PROP_MESH_NAME_RE.test(name)) return;

    const box = new THREE.Box3().setFromObject(child);
    const sz = new THREE.Vector3();
    box.getSize(sz);
    if (Math.max(sz.x, sz.y, sz.z) > maxPropDim) return;

    meshes.push(child);
  });

  return meshes;
}

/** Hub axis = shortest dimension of prop geometry in mesh-local space. */
function getMeshSpinAxis(mesh: THREE.Mesh): THREE.Vector3 {
  const geo = mesh.geometry;
  geo.computeBoundingBox();
  const box = geo.boundingBox;
  if (!box) return new THREE.Vector3(0, 1, 0);

  const sz = new THREE.Vector3();
  box.getSize(sz);
  if (sz.x <= sz.y && sz.x <= sz.z) return new THREE.Vector3(1, 0, 0);
  if (sz.y <= sz.x && sz.y <= sz.z) return new THREE.Vector3(0, 1, 0);
  return new THREE.Vector3(0, 0, 1);
}

type PropSpinEntry = { pivot: THREE.Group; mesh: THREE.Mesh; spinAxis: THREE.Vector3 };

function setupSpinPivots(meshes: THREE.Mesh[]): PropSpinEntry[] {
  const entries: PropSpinEntry[] = [];
  for (const mesh of meshes) {
    const parent = mesh.parent;
    if (!parent) continue;

    const pivot = new THREE.Group();
    pivot.position.copy(mesh.position);

    parent.add(pivot);
    parent.remove(mesh);
    mesh.position.set(0, 0, 0);
    pivot.add(mesh);

    entries.push({
      pivot,
      mesh,
      spinAxis: getMeshSpinAxis(mesh),
    });
  }
  return entries;
}

function teardownSpinPivots(entries: PropSpinEntry[]) {
  for (const { pivot, mesh } of entries) {
    const parent = pivot.parent;
    if (!parent) continue;

    parent.add(mesh);
    mesh.position.copy(pivot.position);
    parent.remove(pivot);
  }
}

function defaultArmPositions(size: number): THREE.Vector3[] {
  const arm = size * PROPELLER_ARM_FRAC;
  const lift = size * 0.06;
  return PROC_ARM_POSITIONS.map(
    ([sx, sz]) => new THREE.Vector3(sx * arm, lift, sz * arm)
  );
}

function SpinningDisc({
  position,
  discR,
  height,
  isPlaying,
  playbackSpeed,
}: {
  position: THREE.Vector3;
  discR: number;
  height: number;
  isPlaying: boolean;
  playbackSpeed: number;
}) {
  const discRef = useRef<THREE.Mesh>(null);

  /** Cylinder is rotated π/2 on X so the hub axis is local Z. */
  const spinAxis = useMemo(() => new THREE.Vector3(0, 0, 1), []);

  useFrame((_, delta) => {
    if (!isPlaying || !discRef.current) return;
    const w = PROPELLER_SPIN_RAD_PER_S * playbackSpeed * delta;
    discRef.current.rotateOnAxis(spinAxis, w);
  });

  return (
    <group position={position}>
      <mesh ref={discRef} rotation={[Math.PI / 2, 0, 0]}>
        <cylinderGeometry args={[discR, discR, height, 16]} />
        <meshStandardMaterial
          color="#f2f2f2"
          metalness={0.2}
          roughness={0.5}
        />
      </mesh>
    </group>
  );
}

function PropellerRotors({
  root,
  size,
  isPlaying,
  playbackSpeed,
}: {
  root?: THREE.Object3D | null;
  size: number;
  isPlaying: boolean;
  playbackSpeed: number;
}) {
  const propMeshes = useMemo(
    () => (root ? findPropellerMeshes(root, size) : []),
    [root, size]
  );
  const useProcedural = propMeshes.length < 4;
  const spinEntriesRef = useRef<PropSpinEntry[]>([]);

  useEffect(() => {
    if (!root || useProcedural) return;
    const entries = setupSpinPivots(propMeshes);
    spinEntriesRef.current = entries;
    return () => {
      teardownSpinPivots(entries);
      spinEntriesRef.current = [];
    };
  }, [root, propMeshes, useProcedural]);

  useFrame((_, delta) => {
    if (!isPlaying || spinEntriesRef.current.length === 0) return;
    const w = PROPELLER_SPIN_RAD_PER_S * playbackSpeed * delta;
    for (const { mesh, spinAxis } of spinEntriesRef.current) {
      mesh.rotateOnAxis(spinAxis, w);
    }
  });

  if (!useProcedural) return null;

  const discR = size * 0.12;
  const height = size * 0.015;
  const sites = defaultArmPositions(size);

  return (
    <group>
      {sites.map((pos, i) => (
        <SpinningDisc
          key={i}
          position={pos}
          discR={discR}
          height={height}
          isPlaying={isPlaying}
          playbackSpeed={playbackSpeed}
        />
      ))}
    </group>
  );
}

function FallbackBody({ size, dimmed }: { size: number; dimmed?: boolean }) {
  return (
    <group>
      <mesh>
        <boxGeometry args={[size, size * 0.12, size]} />
        <meshStandardMaterial
          color="#6ab0e8"
          metalness={0.35}
          roughness={0.55}
          emissive="#224466"
          emissiveIntensity={dimmed ? 0.08 : 0.25}
          transparent={dimmed}
          opacity={dimmed ? 0.15 : 1}
        />
      </mesh>
      {!dimmed && (
        <mesh position={[0, size * 0.08, 0]}>
          <sphereGeometry args={[size * 0.18, 12, 12]} />
          <meshBasicMaterial color="#8ec8ff" />
        </mesh>
      )}
    </group>
  );
}

function AlertBadge({ size }: { size: number }) {
  return (
    <Html position={[0, size * 0.9, 0]} center distanceFactor={14}>
      <div
        style={{
          color: "#ff4444",
          fontSize: 22,
          fontWeight: 800,
          textShadow: "0 0 8px #000",
          pointerEvents: "none",
        }}
      >
        !
      </div>
    </Html>
  );
}

export function DroneMarker({
  drone,
  showFallback = true,
  dimFallback = false,
  isPlaying = false,
  playbackSpeed = 1,
}: {
  drone: DroneState;
  showFallback?: boolean;
  dimFallback?: boolean;
  isPlaying?: boolean;
  playbackSpeed?: number;
}) {
  const [x, y, z] = simToThree(drone.pos);

  return (
    <group position={[x, y, z]} rotation={[0, drone.yaw, 0]}>
      {showFallback && (
        <group>
          <FallbackBody size={DRONE_FALLBACK_SIZE_M} dimmed={dimFallback} />
          {!dimFallback && (
            <PropellerRotors
              size={DRONE_FALLBACK_SIZE_M}
              isPlaying={isPlaying}
              playbackSpeed={playbackSpeed}
            />
          )}
        </group>
      )}
      {drone.alert && <AlertBadge size={DRONE_FALLBACK_SIZE_M} />}
    </group>
  );
}

function GlbDroneInstance({
  drone,
  template,
  isPlaying,
  playbackSpeed,
}: {
  drone: DroneState;
  template: THREE.Object3D;
  isPlaying: boolean;
  playbackSpeed: number;
}) {
  const mesh = useMemo(() => template.clone(true), [template, drone.id]);
  const [x, y, z] = simToThree(drone.pos);

  return (
    <group position={[x, y, z]} rotation={[0, drone.yaw, 0]}>
      <primitive object={mesh}>
        <PropellerRotors
          root={mesh}
          size={DRONE_GLB_SIZE_M}
          isPlaying={isPlaying}
          playbackSpeed={playbackSpeed}
        />
      </primitive>
      {drone.alert && <AlertBadge size={DRONE_GLB_SIZE_M} />}
    </group>
  );
}

function FleetGlbLayer({
  drones,
  onReady,
  isPlaying,
  playbackSpeed,
}: {
  drones: DroneState[];
  onReady: () => void;
  isPlaying: boolean;
  playbackSpeed: number;
}) {
  const gltf = useGLTF("/models/crazyflie.glb");

  const template = useMemo(() => {
    const c = gltf.scene.clone(true);
    const box = new THREE.Box3().setFromObject(c);
    const size = new THREE.Vector3();
    box.getSize(size);
    const maxDim = Math.max(size.x, size.y, size.z, 1e-6);
    c.scale.setScalar(DRONE_GLB_SIZE_M / maxDim);
    return c;
  }, [gltf.scene]);

  useEffect(() => {
    onReady();
  }, [onReady]);

  return (
    <group>
      {drones.map((drone) => (
        <GlbDroneInstance
          key={drone.id}
          drone={drone}
          template={template}
          isPlaying={isPlaying}
          playbackSpeed={playbackSpeed}
        />
      ))}
    </group>
  );
}

class GlbErrorBoundary extends Component<
  { children: ReactNode },
  { failed: boolean }
> {
  state = { failed: false };

  static getDerivedStateFromError() {
    return { failed: true };
  }

  componentDidCatch(error: Error) {
    if (import.meta.env.DEV) {
      console.warn("[DroneFleet] GLB load failed, using fallback meshes:", error);
    }
  }

  render() {
    if (this.state.failed) return null;
    return this.props.children;
  }
}

export function FleetFallback({
  drones,
  isPlaying = false,
  playbackSpeed = 1,
}: {
  drones: DroneState[];
  isPlaying?: boolean;
  playbackSpeed?: number;
}) {
  return (
    <group>
      {drones.map((d) => (
        <DroneMarker
          key={d.id}
          drone={d}
          showFallback
          isPlaying={isPlaying}
          playbackSpeed={playbackSpeed}
        />
      ))}
    </group>
  );
}

export function DroneFleet({
  drones,
  isPlaying = false,
  playbackSpeed = 1,
}: {
  drones: DroneState[];
  isPlaying?: boolean;
  playbackSpeed?: number;
}) {
  const [glbReady, setGlbReady] = useState(false);

  if (drones.length === 0) {
    if (import.meta.env.DEV) {
      console.warn("[DroneFleet] No drones in frame — check /api/state");
    }
    return null;
  }

  return (
    <group>
      {drones.map((d) => (
        <DroneMarker
          key={d.id}
          drone={d}
          showFallback={!glbReady}
          dimFallback={glbReady}
          isPlaying={isPlaying}
          playbackSpeed={playbackSpeed}
        />
      ))}
      <GlbErrorBoundary>
        <Suspense fallback={null}>
          <FleetGlbLayer
            drones={drones}
            onReady={() => setGlbReady(true)}
            isPlaying={isPlaying}
            playbackSpeed={playbackSpeed}
          />
        </Suspense>
      </GlbErrorBoundary>
    </group>
  );
}

useGLTF.preload("/models/crazyflie.glb");
