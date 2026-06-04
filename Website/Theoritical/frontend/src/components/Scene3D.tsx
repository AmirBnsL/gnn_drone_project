import { Grid, OrbitControls } from "@react-three/drei";
import { Canvas } from "@react-three/fiber";
import { Suspense, useEffect, useRef } from "react";
import { ARENA_HALF } from "../constants";
import type { DroneState } from "../api/client";
import { CommLinksLayer } from "./CommLinksLayer";
import { DroneFleet, FleetFallback } from "./DroneFleet";
import { ObstaclesLayer } from "./ObstaclesLayer";
import { SlotsLayer } from "./SlotsLayer";

type Props = {
  drones: DroneState[];
  slots: [number, number, number][];
  assignment: number[];
  obstacles: [number, number, number, number][];
  center: [number, number];
  digit: number | null;
  isPlaying: boolean;
  playbackSpeed?: number;
  slotsShifted?: boolean;
};

export function Scene3D({
  drones,
  slots,
  assignment,
  obstacles,
  center,
  digit,
  isPlaying,
  playbackSpeed = 1,
  slotsShifted = false,
}: Props) {
  const warnedEmpty = useRef(false);
  useEffect(() => {
    if (drones.length === 0 && !warnedEmpty.current) {
      warnedEmpty.current = true;
      if (import.meta.env.DEV) {
        console.warn("[Scene3D] drones array is empty");
      }
    }
    if (drones.length > 0) warnedEmpty.current = false;
  }, [drones.length]);

  return (
    <Canvas
      camera={{ position: [18, 14, 18], fov: 50, near: 0.1, far: 200 }}
      gl={{ antialias: true }}
    >
      <color attach="background" args={["#0a0e14"]} />
      <fog attach="fog" args={["#0a0e14", 40, 90]} />
      <ambientLight intensity={0.45} />
      <directionalLight position={[12, 20, 8]} intensity={1.1} castShadow />
      <directionalLight position={[-8, 10, -12]} intensity={0.35} />

      <Grid
        args={[ARENA_HALF * 2, ARENA_HALF * 2]}
        cellSize={1}
        cellThickness={0.4}
        sectionSize={5}
        sectionThickness={0.8}
        fadeDistance={45}
        position={[0, 0, 0]}
        rotation={[-Math.PI / 2, 0, 0]}
        infiniteGrid={false}
      />

      <Suspense
        fallback={
          <FleetFallback
            drones={drones}
            isPlaying={isPlaying}
            playbackSpeed={playbackSpeed}
          />
        }
      >
        <DroneFleet
          drones={drones}
          isPlaying={isPlaying}
          playbackSpeed={playbackSpeed}
        />
      </Suspense>
      {drones.length >= 2 && <CommLinksLayer drones={drones} />}
      {digit !== null && slots.length > 0 && (
        <SlotsLayer
          drones={drones}
          slots={slots}
          assignment={assignment}
          center={center}
          showLines={!isPlaying}
          slotsShifted={slotsShifted}
        />
      )}
      <ObstaclesLayer obstacles={obstacles} />

      <OrbitControls makeDefault maxPolarAngle={Math.PI / 2.05} />
    </Canvas>
  );
}
