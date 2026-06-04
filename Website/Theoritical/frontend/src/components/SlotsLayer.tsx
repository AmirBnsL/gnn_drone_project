import { Line } from "@react-three/drei";
import { simToThree } from "./DroneFleet";
import type { DroneState } from "../api/client";

type Props = {
  drones: DroneState[];
  slots: [number, number, number][];
  assignment: number[];
  center: [number, number];
  showLines: boolean;
  slotsShifted?: boolean;
};

export function SlotsLayer({
  drones,
  slots,
  assignment,
  center,
  showLines,
  slotsShifted = false,
}: Props) {
  const [cx, , cz] = simToThree([center[0], center[1], 0]);

  const linePoints: [number, number, number][][] = [];
  if (showLines && assignment.length === drones.length) {
    for (let i = 0; i < drones.length; i++) {
      const si = assignment[i];
      if (si < 0 || si >= slots.length) continue;
      const a = simToThree(drones[i].pos);
      const b = simToThree(slots[si]);
      linePoints.push([a, b]);
    }
  }

  return (
    <group>
      <mesh position={[cx, 0.02, cz]}>
        <ringGeometry args={[0.35, 0.5, 32]} />
        <meshBasicMaterial color="#4aa8e8" transparent opacity={0.35} />
      </mesh>
      <mesh position={[cx, 0.01, cz]} rotation={[-Math.PI / 2, 0, 0]}>
        <planeGeometry args={[0.8, 0.8]} />
        <meshBasicMaterial color="#4aa8e8" wireframe transparent opacity={0.25} />
      </mesh>

      {slots.map((s, j) => {
        const [x, y, z] = simToThree(s);
        return (
          <group key={j} position={[x, y, z]}>
            <mesh rotation={[-Math.PI / 2, 0, 0]}>
              <ringGeometry args={[0.22, 0.32, 24]} />
              <meshBasicMaterial
                color={slotsShifted ? "#FFD700" : "#2ee8d0"}
                transparent
                opacity={slotsShifted ? 0.95 : 0.85}
              />
            </mesh>
          </group>
        );
      })}

      {linePoints.map((pts, idx) => (
        <Line
          key={idx}
          points={pts}
          color="#88c8ff"
          lineWidth={1}
          dashed
          dashSize={0.4}
          gapSize={0.25}
        />
      ))}
    </group>
  );
}
