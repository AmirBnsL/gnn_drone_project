import { useState } from "react";
import { MAX_DRONES, MIN_DRONES } from "../constants";
import type { Scenario } from "../api/client";

const SCENARIO_HINTS: Record<Scenario, string> = {
  clean: "No obstacles during simulation.",
  path: "One sphere on a random drone→slot path (mid-segment). Training: place_path_obstacles.",
  slot: "One sphere centered on a random assigned slot. Training: place_slot_obstacles.",
  both: "One path obstacle + one slot obstacle (training default for hard episodes).",
};

type Props = {
  onClose: () => void;
  onSave: (n: number, scenario: Scenario) => void;
  initialDrones: number;
  initialScenario: Scenario;
};

export function ConfigPanel({
  onClose,
  onSave,
  initialDrones,
  initialScenario,
}: Props) {
  const [drones, setDrones] = useState(initialDrones);
  const [scenario, setScenario] = useState<Scenario>(initialScenario);

  return (
    <div className="dropdown-panel">
      <label>Number of drones ({MIN_DRONES}–{MAX_DRONES})</label>
      <input
        type="range"
        min={MIN_DRONES}
        max={MAX_DRONES}
        value={drones}
        onChange={(e) => setDrones(Number(e.target.value))}
      />
      <div style={{ fontSize: 13, marginBottom: 8 }}>{drones} drones</div>

      <label>Obstacles</label>
      <select
        value={scenario}
        onChange={(e) => setScenario(e.target.value as Scenario)}
      >
        <option value="clean">None</option>
        <option value="path">In path</option>
        <option value="slot">At slot</option>
        <option value="both">Path + slot</option>
      </select>
      <p className="hint">{SCENARIO_HINTS[scenario]}</p>
      <p className="hint">
        Physics and rendering both use 1.0 m obstacle radius.
      </p>
      <p className="hint">
        Obstacles appear when you Run Simulation (not on Save). Slot sliding
        runs during sim when drones stall near blocked goals.
      </p>

      <div style={{ display: "flex", gap: 8, marginTop: 14 }}>
        <button
          type="button"
          className="btn btn-primary"
          onClick={() => {
            onSave(drones, scenario);
            onClose();
          }}
        >
          Save
        </button>
        <button type="button" className="btn" onClick={onClose}>
          Cancel
        </button>
      </div>
    </div>
  );
}
