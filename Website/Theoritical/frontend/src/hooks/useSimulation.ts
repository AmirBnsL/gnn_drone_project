import { useCallback, useEffect, useRef, useState } from "react";
import {
  api,
  DroneState,
  Scenario,
  SimMode,
  SimState,
  Trajectory,
  TrajectoryFrame,
} from "../api/client";

function lerp(a: number, b: number, t: number) {
  return a + (b - a) * t;
}

function lerpDrone(a: DroneState, b: DroneState, t: number): DroneState {
  return {
    id: a.id,
    pos: [
      lerp(a.pos[0], b.pos[0], t),
      lerp(a.pos[1], b.pos[1], t),
      lerp(a.pos[2], b.pos[2], t),
    ],
    yaw: lerp(a.yaw, b.yaw, t),
    alert: t < 0.5 ? a.alert : b.alert,
  };
}

function lerpSlot(
  a: [number, number, number],
  b: [number, number, number],
  t: number
): [number, number, number] {
  return [lerp(a[0], b[0], t), lerp(a[1], b[1], t), lerp(a[2], b[2], t)];
}

function interpolateFrame(
  f0: TrajectoryFrame,
  f1: TrajectoryFrame,
  t: number
): TrajectoryFrame {
  const drones = f0.drones.map((d, i) => lerpDrone(d, f1.drones[i] ?? d, t));
  const slots = f0.slots.map((s, i) =>
    lerpSlot(s, f1.slots[i] ?? s, t)
  );
  return {
    center: [
      lerp(f0.center[0], f1.center[0], t),
      lerp(f0.center[1], f1.center[1], t),
    ],
    drones,
    slots,
    assignment: f0.assignment,
    obstacles: t < 0.5 ? f0.obstacles : f1.obstacles,
    slots_shifted: Boolean(f0.slots_shifted || f1.slots_shifted),
    mode: f0.mode,
    meta: f0.meta,
  };
}

export function useSimulation() {
  const [state, setState] = useState<SimState | null>(null);
  const [activeMode, setActiveMode] = useState<SimMode>("central");
  const [trajectoryCentral, setTrajectoryCentral] = useState<Trajectory | null>(
    null
  );
  const [trajectoryDecentral, setTrajectoryDecentral] =
    useState<Trajectory | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [scrubIndex, setScrubIndex] = useState(0);
  const [playhead, setPlayhead] = useState(0);
  const [savedNames, setSavedNames] = useState<string[]>([]);
  const rafRef = useRef<number>(0);
  const lastTsRef = useRef<number>(0);

  const refreshSaves = useCallback(async () => {
    try {
      const { names } = await api.getSaves();
      setSavedNames(names);
    } catch {
      setSavedNames([]);
    }
  }, []);

  const trajectory =
    activeMode === "decentral" ? trajectoryDecentral : trajectoryCentral;

  const refresh = useCallback(async () => {
    const s = await api.getState();
    setState(s);
    if (s.active_mode) setActiveMode(s.active_mode);
    if (s.has_trajectory_central) {
      setTrajectoryCentral(await api.getTrajectory("central"));
    }
    if (s.has_trajectory_decentral) {
      setTrajectoryDecentral(await api.getTrajectory("decentral"));
    }
  }, []);

  useEffect(() => {
    refresh().catch((e) => setError(String(e)));
    refreshSaves().catch(() => undefined);
  }, [refresh, refreshSaves]);

  const switchMode = async (mode: SimMode) => {
    setActiveMode(mode);
    setScrubIndex(0);
    setPlayhead(0);
    setPlaying(false);
    try {
      const s = await api.postMode(mode);
      setState(s);
    } catch (e) {
      setError(String(e));
    }
  };

  const saveConfig = async (num_drones: number, scenario: Scenario) => {
    setError(null);
    setLoading(true);
    try {
      const s = await api.postConfig(num_drones, scenario);
      setState(s);
      setTrajectoryCentral(null);
      setTrajectoryDecentral(null);
      setPlaying(false);
      setScrubIndex(0);
      setPlayhead(0);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  };

  const selectFormation = async (digit: number) => {
    setError(null);
    setLoading(true);
    try {
      const s = await api.postFormation(digit);
      setState(s);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  };

  const runSimulation = async () => {
    setError(null);
    setLoading(true);
    setPlaying(false);
    try {
      const res = await api.postSimulate();
      setState(res);
      setTrajectoryCentral(res.trajectory_central ?? null);
      setTrajectoryDecentral(res.trajectory_decentral ?? null);
      setActiveMode(res.active_mode ?? "central");
      setScrubIndex(0);
      setPlayhead(0);
      setPlaying(true);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  };

  const reset = async () => {
    setError(null);
    setLoading(true);
    setPlaying(false);
    try {
      const s = await api.postReset();
      setState(s);
      setTrajectoryCentral(null);
      setTrajectoryDecentral(null);
      setScrubIndex(0);
      setPlayhead(0);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (!playing || !trajectory?.frames.length) return;
    const dt = trajectory.dt;
    const maxIdx = trajectory.frames.length - 1;

    const tick = (ts: number) => {
      if (!lastTsRef.current) lastTsRef.current = ts;
      const elapsed = ((ts - lastTsRef.current) / 1000) * speed;
      lastTsRef.current = ts;
      setPlayhead((p) => {
        const next = p + elapsed;
        const frameFloat = next / dt;
        if (frameFloat >= maxIdx) {
          setPlaying(false);
          return maxIdx * dt;
        }
        setScrubIndex(Math.floor(frameFloat));
        return next;
      });
      rafRef.current = requestAnimationFrame(tick);
    };
    lastTsRef.current = 0;
    rafRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafRef.current);
  }, [playing, speed, trajectory]);

  const displayFrame = (): TrajectoryFrame | null => {
    if (!state) return null;
    if (!trajectory?.frames.length) {
      return {
        center: state.center,
        drones: state.drones,
        slots: state.slots,
        assignment: state.assignment,
        obstacles: state.obstacles,
        mode: activeMode,
      };
    }
    const frames = trajectory.frames;
    const frameFloat = playhead / trajectory.dt;
    const idx = Math.min(Math.floor(frameFloat), frames.length - 1);
    const frac = frameFloat - idx;
    const nextIdx = Math.min(idx + 1, frames.length - 1);
    if (idx === nextIdx || frac <= 0) return frames[idx];
    return interpolateFrame(frames[idx], frames[nextIdx], frac);
  };

  const frameCount = trajectory?.frames.length ?? 0;

  const seek = (index: number) => {
    if (!trajectory) return;
    const i = Math.max(0, Math.min(index, trajectory.frames.length - 1));
    setScrubIndex(i);
    setPlayhead(i * trajectory.dt);
    setPlaying(false);
  };

  const saveRun = async (name: string) => {
    setError(null);
    setLoading(true);
    try {
      await api.postSave(name.trim());
      await refreshSaves();
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  };

  const loadRun = async (name: string) => {
    if (!name) return;
    setError(null);
    setLoading(true);
    setPlaying(false);
    try {
      const res = await api.postLoadSave(name);
      setState(res);
      setTrajectoryCentral(res.trajectory_central ?? null);
      setTrajectoryDecentral(res.trajectory_decentral ?? null);
      setActiveMode(res.active_mode ?? "central");
      setScrubIndex(0);
      setPlayhead(0);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  };

  return {
    state,
    activeMode,
    switchMode,
    trajectory,
    trajectoryCentral,
    trajectoryDecentral,
    loading,
    error,
    playing,
    setPlaying,
    speed,
    setSpeed,
    scrubIndex,
    playhead,
    frameCount,
    displayFrame,
    seek,
    saveConfig,
    selectFormation,
    runSimulation,
    reset,
    savedNames,
    saveRun,
    loadRun,
  };
};
