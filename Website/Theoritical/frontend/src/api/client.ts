export type Scenario = "clean" | "path" | "slot" | "both";
export type SimMode = "central" | "decentral";

export interface DroneState {
  id: number;
  pos: [number, number, number];
  yaw: number;
  alert: boolean;
}

export interface FrameMeta {
  messages_sent?: number;
  shift_prob?: number;
  shift_proposals?: number;
  shift_fired?: boolean;
  did_shift_total?: boolean;
  shift_blocked_reason?: string;
  shift_block_reasons?: Record<string, number>;
  inference_turns?: number[];
  slot_error_mean?: number;
  max_slot_error?: number;
}

export interface SimState {
  num_drones: number;
  scenario: Scenario;
  digit: number | null;
  center: [number, number];
  drones: DroneState[];
  slots: [number, number, number][];
  assignment: number[];
  obstacles: [number, number, number, number][];
  has_trajectory: boolean;
  has_trajectory_central?: boolean;
  has_trajectory_decentral?: boolean;
  active_mode?: SimMode;
  frame_count: number;
  dt: number;
  is_running: boolean;
  last_error: string | null;
  dual_metadata?: Record<string, unknown>;
  trajectory?: Trajectory;
  trajectory_central?: Trajectory;
  trajectory_decentral?: Trajectory;
}

export interface TrajectoryFrame {
  center: [number, number];
  drones: DroneState[];
  slots: [number, number, number][];
  assignment: number[];
  obstacles: [number, number, number, number][];
  slots_shifted?: boolean;
  mode?: SimMode;
  meta?: FrameMeta;
}

export interface Trajectory {
  frames: TrajectoryFrame[];
  dt: number;
  converged?: boolean;
  stopped_reason?: "converged" | "max_steps";
  max_steps_cap?: number;
  steps?: number;
  mode?: SimMode;
  did_slot_shift?: boolean;
  messages_sent?: number;
  final_max_slot_error?: number;
}

export interface SimulateResponse extends SimState {
  trajectory: Trajectory;
  trajectory_central: Trajectory;
  trajectory_decentral: Trajectory;
}

const BASE = "/api";
const TIMEOUT_MS = 600_000;

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), TIMEOUT_MS);
  try {
    const res = await fetch(`${BASE}${path}`, {
      ...init,
      signal: ctrl.signal,
      headers: {
        "Content-Type": "application/json",
        ...(init?.headers ?? {}),
      },
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail ?? res.statusText);
    }
    return res.json() as Promise<T>;
  } finally {
    clearTimeout(timer);
  }
}

export const api = {
  getState: () => request<SimState>("/state"),
  getTrajectory: (mode?: SimMode) =>
    request<Trajectory>(mode ? `/trajectory?mode=${mode}` : "/trajectory"),
  postMode: (mode: SimMode) =>
    request<SimState>("/mode", {
      method: "POST",
      body: JSON.stringify({ mode }),
    }),
  postConfig: (num_drones: number, scenario: Scenario) =>
    request<SimState>("/config", {
      method: "POST",
      body: JSON.stringify({ num_drones, scenario }),
    }),
  postFormation: (digit: number) =>
    request<SimState>("/formation", {
      method: "POST",
      body: JSON.stringify({ digit }),
    }),
  postReset: () => request<SimState>("/reset", { method: "POST" }),
  postSimulate: () =>
    request<SimulateResponse>("/simulate", {
      method: "POST",
    }),
  getSaves: () => request<{ names: string[] }>("/saves"),
  postSave: (name: string) =>
    request<{ name: string; path: string }>("/saves", {
      method: "POST",
      body: JSON.stringify({ name }),
    }),
  postLoadSave: (name: string) =>
    request<SimulateResponse & SimState>("/saves/load", {
      method: "POST",
      body: JSON.stringify({ name }),
    }),
};
