import { useState } from "react";
import type { Scenario, SimMode } from "../api/client";
import { ConfigPanel } from "./ConfigPanel";

type Props = {
  numDrones: number;
  scenario: Scenario;
  activeMode: SimMode;
  hasCentral: boolean;
  hasDecentral: boolean;
  loading: boolean;
  playing: boolean;
  frameCount: number;
  scrubIndex: number;
  speed: number;
  runMeta?: string;
  savedNames: string[];
  canSaveRun: boolean;
  onSaveRun: (name: string) => void;
  onLoadRun: (name: string) => void;
  onSaveConfig: (n: number, s: Scenario) => void;
  onRun: () => void;
  onReset: () => void;
  onModeChange: (m: SimMode) => void;
  onPlayPause: () => void;
  onSeek: (i: number) => void;
  onSpeed: (v: number) => void;
};

export function TopBar({
  numDrones,
  scenario,
  activeMode,
  hasCentral,
  hasDecentral,
  loading,
  playing,
  frameCount,
  scrubIndex,
  speed,
  runMeta,
  savedNames,
  canSaveRun,
  onSaveRun,
  onLoadRun,
  onSaveConfig,
  onRun,
  onReset,
  onModeChange,
  onPlayPause,
  onSeek,
  onSpeed,
}: Props) {
  const [panelOpen, setPanelOpen] = useState(false);
  const [saveName, setSaveName] = useState("");
  const [loadName, setLoadName] = useState("");

  return (
    <header className="top-bar">
      <div className="dropdown-wrap">
        <button
          type="button"
          className="btn btn-primary"
          onClick={() => setPanelOpen((o) => !o)}
          disabled={loading}
        >
          Setup ▾
        </button>
        {panelOpen && (
          <div className="dropdown-menu">
            <ConfigPanel
              key={`${numDrones}-${scenario}`}
              onClose={() => setPanelOpen(false)}
              onSave={onSaveConfig}
              initialDrones={numDrones}
              initialScenario={scenario}
            />
          </div>
        )}
      </div>

      <button
        type="button"
        className="btn btn-primary"
        onClick={onRun}
        disabled={loading}
      >
        {loading ? "Running…" : "Run Dual Sim"}
      </button>

      <button
        type="button"
        className="btn btn-danger"
        onClick={onReset}
        disabled={loading}
      >
        Reset
      </button>

      <div className="save-load-row">
        <input
          type="text"
          className="save-name-input"
          placeholder="Save name"
          value={saveName}
          disabled={loading}
          onChange={(e) => setSaveName(e.target.value)}
        />
        <button
          type="button"
          className="btn"
          disabled={loading || !canSaveRun || !saveName.trim()}
          onClick={() => {
            onSaveRun(saveName.trim());
            setSaveName("");
          }}
        >
          Save run
        </button>
        <select
          className="save-select"
          value={loadName}
          disabled={loading || savedNames.length === 0}
          onChange={(e) => setLoadName(e.target.value)}
        >
          <option value="">Load saved…</option>
          {savedNames.map((n) => (
            <option key={n} value={n}>
              {n}
            </option>
          ))}
        </select>
        <button
          type="button"
          className="btn"
          disabled={loading || !loadName}
          onClick={() => {
            onLoadRun(loadName);
          }}
        >
          Load
        </button>
      </div>

      <div className="mode-tabs">
        <button
          type="button"
          className={`btn mode-tab ${activeMode === "central" ? "mode-tab-active" : ""}`}
          disabled={!hasCentral && !loading}
          onClick={() => onModeChange("central")}
        >
          Central
        </button>
        <button
          type="button"
          className={`btn mode-tab ${activeMode === "decentral" ? "mode-tab-active" : ""}`}
          disabled={!hasDecentral && !loading}
          onClick={() => onModeChange("decentral")}
        >
          Decentral (GNN)
        </button>
      </div>

      {runMeta && (
        <span className="run-meta" title={runMeta}>
          {runMeta}
        </span>
      )}

      <div className="spacer" />

      <div className="timeline">
        <button
          type="button"
          className="btn"
          onClick={onPlayPause}
          disabled={frameCount < 2 || loading}
        >
          {playing ? "Pause" : "Play"}
        </button>
        <input
          type="range"
          min={0}
          max={Math.max(0, frameCount - 1)}
          value={scrubIndex}
          disabled={frameCount < 1}
          onChange={(e) => onSeek(Number(e.target.value))}
        />
        <span style={{ fontSize: 12, minWidth: 72 }}>
          {frameCount ? `${scrubIndex + 1}/${frameCount}` : "—"}
        </span>
        <label style={{ fontSize: 11, color: "#9ab0c8" }}>Speed</label>
        <input
          type="range"
          min={0.25}
          max={4}
          step={0.25}
          value={speed}
          onChange={(e) => onSpeed(Number(e.target.value))}
        />
        <span style={{ fontSize: 12 }}>{speed.toFixed(2)}×</span>
      </div>
    </header>
  );
}
