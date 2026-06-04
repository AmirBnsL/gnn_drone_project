import { OBSTACLE_MESH_VIS_SCALE } from "../constants";
import { simToThree } from "./DroneFleet";

/** Each obstacle is [x, y, z, radius] in PyBullet coordinates. */
type Obstacle = [number, number, number, number];

type Props = {
  obstacles: Obstacle[] | [number, number, number][];
};

export function ObstaclesLayer({ obstacles }: Props) {
  if (!obstacles.length) return null;

  return (
    <group>
      {obstacles.map((obs, i) => {
        const o = obs.length >= 4 ? obs : [obs[0], obs[1], 0, obs[2]];
        const [x, y, z] = simToThree([o[0], o[1], o[2]]);
        const rVis = o[3] * OBSTACLE_MESH_VIS_SCALE;
        return (
          <mesh key={i} position={[x, y, z]}>
            <sphereGeometry args={[rVis, 24, 24]} />
            <meshStandardMaterial
              color="#e84040"
              transparent
              opacity={0.55}
              roughness={0.4}
            />
          </mesh>
        );
      })}
    </group>
  );
}
