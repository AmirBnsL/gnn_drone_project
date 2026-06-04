import { DEFAULT_DRONES } from "./constants";
import { FormationBar } from "./components/FormationBar";
import { Scene3D } from "./components/Scene3D";
import { TopBar } from "./components/TopBar";
import { useSimulation } from "./hooks/useSimulation";
import "./styles.css";

function stopLabel(
  traj: {
    converged?: boolean;
    stopped_reason?: string;
    final_max_slot_error?: number;
  } | null
): string {
  if (!traj) return "";
  const err =
    traj.final_max_slot_error != null
      ? ` err ${traj.final_max_slot_error.toFixed(2)}m`
      : "";
  if (traj.stopped_reason === "converged" || traj.converged) {
    return `converged${err}`;
  }
  if (traj.stopped_reason === "max_steps") return `max_steps${err}`;
  return traj.converged ? `converged${err}` : "running";
}

function formatRunMeta(sim: ReturnType<typeof useSimulation>): string {
  const parts: string[] = [];
  const c = sim.trajectoryCentral;
  const d = sim.trajectoryDecentral;
  if (c && d) {
    parts.push(`central: ${stopLabel(c)}`);
    parts.push(`decentral: ${stopLabel(d)}`);
  } else {
    const traj =
      sim.activeMode === "decentral" ? sim.trajectoryDecentral : sim.trajectoryCentral;
    const label = stopLabel(traj);
    if (label) parts.push(label);
  }
  const active =
    sim.activeMode === "decentral" ? sim.trajectoryDecentral : sim.trajectoryCentral;
  if (active?.did_slot_shift) parts.push("shift");
  if (active?.messages_sent != null) parts.push(`assign msgs ${active.messages_sent}`);
  const meta = active?.frames[active.frames.length - 1]?.meta;
  if (meta?.shift_prob != null) {
    parts.push(`shift p=${meta.shift_prob.toFixed(2)}`);
  }
  return parts.join(" · ");
}

export default function App() {
  const sim = useSimulation();
  const frame = sim.displayFrame();
  const state = sim.state;
  const activeTrajectory =
    sim.activeMode === "decentral" ? sim.trajectoryDecentral : sim.trajectoryCentral;
  const maxIdx = Math.max(0, Math.min(sim.scrubIndex, (activeTrajectory?.frames.length ?? 1) - 1));
  const slotsShiftedEver = Boolean(
    activeTrajectory?.frames
      ?.slice(0, maxIdx + 1)
      .some((f) => Boolean(f.slots_shifted) || Boolean(f.meta?.did_shift_total))
  );

  return (
    <div className="app">
      <TopBar
        numDrones={state?.num_drones ?? DEFAULT_DRONES}
        scenario={state?.scenario ?? "clean"}
        activeMode={sim.activeMode}
        hasCentral={Boolean(state?.has_trajectory_central)}
        hasDecentral={Boolean(state?.has_trajectory_decentral)}
        loading={sim.loading}
        playing={sim.playing}
        frameCount={sim.frameCount}
        scrubIndex={sim.scrubIndex}
        speed={sim.speed}
        runMeta={formatRunMeta(sim)}
        savedNames={sim.savedNames}
        canSaveRun={Boolean(
          state?.has_trajectory_central || state?.has_trajectory_decentral
        )}
        onSaveRun={sim.saveRun}
        onLoadRun={sim.loadRun}
        onSaveConfig={sim.saveConfig}
        onRun={sim.runSimulation}
        onReset={sim.reset}
        onModeChange={sim.switchMode}
        onPlayPause={() => sim.setPlaying((p) => !p)}
        onSeek={sim.seek}
        onSpeed={sim.setSpeed}
      />

      {sim.error && <div className="error-banner">{sim.error}</div>}

      <div className="canvas-wrap">
        {frame && (
          <Scene3D
            drones={frame.drones}
            slots={frame.slots}
            assignment={frame.assignment}
            obstacles={frame.obstacles}
            center={frame.center}
            digit={state?.digit ?? null}
            isPlaying={sim.playing}
            playbackSpeed={sim.speed}
            slotsShifted={slotsShiftedEver}
          />
        )}
        {sim.loading && (
          <div className="overlay">
            <div className="overlay-box">
              Running dual PyFlyt simulation…
              <br />
              <span style={{ fontSize: 12, color: "#9ab0c8" }}>
                Central + decentral GNN (may take 1–3 min)
              </span>
            </div>
          </div>
        )}
      </div>

      <FormationBar
        activeDigit={state?.digit ?? null}
        disabled={sim.loading}
        onSelect={sim.selectFormation}
      />
    </div>
  );
}
