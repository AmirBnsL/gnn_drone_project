import { Line } from "@react-three/drei";
import { useMemo } from "react";
import { COMM_RADIUS_M } from "../constants";
import type { DroneState } from "../api/client";
import { simToThree } from "./DroneFleet";

type Props = {
  drones: DroneState[];
  radiusM?: number;
};

function buildCommSegments(
  drones: DroneState[],
  radiusM: number
): [number, number, number][][] {
  const segments: [number, number, number][][] = [];
  const r2 = radiusM * radiusM;
  for (let i = 0; i < drones.length; i++) {
    const pi = drones[i].pos;
    for (let j = i + 1; j < drones.length; j++) {
      const pj = drones[j].pos;
      const dx = pi[0] - pj[0];
      const dy = pi[1] - pj[1];
      const dz = pi[2] - pj[2];
      if (dx * dx + dy * dy + dz * dz <= r2) {
        segments.push([simToThree(pi), simToThree(pj)]);
      }
    }
  }
  return segments;
}

export function CommLinksLayer({ drones, radiusM = COMM_RADIUS_M }: Props) {
  const segments = useMemo(
    () => (drones.length >= 2 ? buildCommSegments(drones, radiusM) : []),
    [drones, radiusM]
  );

  if (segments.length === 0) return null;

  return (
    <group>
      {segments.map((pts, idx) => (
        <Line
          key={idx}
          points={pts}
          color="#9b7bff"
          lineWidth={1.5}
          transparent
          opacity={0.55}
        />
      ))}
    </group>
  );
}
