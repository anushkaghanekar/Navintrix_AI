import { StrictMode, useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

const DIRECTIONS = ["North", "East", "South", "West"];
const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

function formatLabel(value) {
  return value.replace(/_/g, " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function UploadCard({ direction, file, onFile, emergency, onEmergency }) {
  return (
    <article className="upload-card">
      <div className="card-heading">
        <div><span className="direction-index">0{DIRECTIONS.indexOf(direction) + 1}</span><h3>{direction}</h3></div>
        <span className="approach-label">approach</span>
      </div>
      <label className={`drop-zone ${file ? "has-file" : ""}`}>
        <input type="file" accept="image/jpeg,image/png" onChange={(event) => onFile(event.target.files?.[0] || null)} />
        <span className="upload-mark">{file ? "✓" : "+"}</span>
        <strong>{file ? file.name : "Choose image"}</strong>
        <small>{file ? `${Math.round(file.size / 1024)} KB ready` : "JPG or PNG · camera frame"}</small>
      </label>
      <div className="override-row">
        <label htmlFor={`emergency-${direction}`}>Emergency priority</label>
        <input id={`emergency-${direction}`} type="number" min="0" max="10" value={emergency} onChange={(event) => onEmergency(Number(event.target.value))} />
      </div>
    </article>
  );
}

function SignalLight({ color, active }) {
  return <span className={`lamp lamp-${color} ${active ? "active" : ""}`} />;
}

function TrafficSimulation({ data }) {
  const [running, setRunning] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const schedule = useMemo(() => data.phaseOrder.flatMap((direction) => [
    { direction, color: "green", seconds: data.results.find((item) => item.direction === direction).greenSeconds },
    { direction, color: "yellow", seconds: 4 },
    { direction: null, color: "all-red", seconds: 2 },
  ]), [data]);
  const cycle = schedule.reduce((total, item) => total + item.seconds, 0);

  useEffect(() => {
    if (!running) return undefined;
    const interval = window.setInterval(() => setElapsed((value) => (value + 1) % cycle), 1000);
    return () => window.clearInterval(interval);
  }, [running, cycle]);

  let cursor = 0;
  let active = schedule[0];
  let activeStart = 0;
  for (const item of schedule) {
    if (elapsed < cursor + item.seconds) {
      active = item;
      activeStart = cursor;
      break;
    }
    cursor += item.seconds;
  }
  const remaining = activeStart + active.seconds - elapsed;
  const progress = ((elapsed + 1) / cycle) * 100;

  return (
    <section className="simulation-panel">
      <div className="section-heading compact-heading">
        <div><span className="section-kicker">04 / controller</span><h2>Live signal simulation</h2></div>
        <button className={`live-toggle ${running ? "on" : ""}`} onClick={() => setRunning((value) => !value)}><span /> {running ? "Pause cycle" : "Run cycle"}</button>
      </div>
      <div className="simulation-status"><div><span className="status-label">CURRENT PHASE</span><strong>{active.direction ? `${active.direction} ${active.color}` : "All approaches stopped"}</strong></div><span>{remaining}s remaining</span></div>
      <div className="cycle-track"><span style={{ width: `${progress}%` }} /></div>
      <div className="lights-grid">
        {DIRECTIONS.map((direction) => {
          const current = active.direction === direction ? active.color : "red";
          return <div className="light-card" key={direction}><div className="light-card-title"><strong>{direction}</strong><span>{current}</span></div><div className="lamp-stack"><div><SignalLight color="red" active={current === "red"} />Red</div><div><SignalLight color="yellow" active={current === "yellow"} />Yellow</div><div><SignalLight color="green" active={current === "green"} />Green</div></div></div>;
        })}
      </div>
    </section>
  );
}

function Results({ data }) {
  return <>
    <section className="section-block results-section" id="plan">
      <div className="section-heading"><div><span className="section-kicker">02 / decision layer</span><h2>Junction overview</h2></div><span className="quiet-badge">Analysis complete</span></div>
      <div className="metric-grid"><Metric label="Vehicles detected" value={data.totalVehicles} /><Metric label="Demand points" value={data.totalPoints} /><Metric label="Emergency priority" value={data.totalEmergency} /><Metric label="First phase" value={data.phaseOrder[0]} /></div>
    </section>
    <section className="section-block">
      <div className="section-heading"><div><span className="section-kicker">03 / adaptive plan</span><h2>Recommended cycle</h2></div><span className="quiet-badge">120 sec cycle</span></div>
      <div className="plan-layout"><div className="priority-list">{data.phaseOrder.map((direction, index) => { const result = data.results.find((item) => item.direction === direction); return <div className="priority-row" key={direction}><span className="rank">0{index + 1}</span><div className="priority-name"><strong>{direction}</strong><span>{result.demandPoints} points · {result.queueMeters}m queue</span></div><strong className="green-time">{result.greenSeconds}s</strong></div>; })}</div><div className="score-legend"><strong>How timing is decided</strong><p>Vehicle class points + automatic queue points + emergency priority are shared across the cycle. Higher demand gets more green time.</p><div className="legend-line"><span className="legend-dot blue" /> vehicle load</div><div className="legend-line"><span className="legend-dot amber" /> queue pressure</div><div className="legend-line"><span className="legend-dot red" /> emergency priority</div></div></div>
    </section>
    <TrafficSimulation data={data} />
    <section className="section-block evidence-section" id="evidence"><div className="section-heading"><div><span className="section-kicker">05 / computer vision</span><h2>Detection evidence</h2></div><span className="muted-copy">Annotated frames from YOLO26m</span></div><div className="evidence-grid">{data.results.map((result) => <article className="evidence-card" key={result.direction}><div className="evidence-title"><strong>{result.direction}</strong><span>{result.demandPoints} pts</span></div><img src={result.annotatedImage} alt={`${result.direction} annotated detections`} /><p>{Object.entries(result.counts).map(([label, count]) => `${formatLabel(label)} ${count}`).join(" · ") || "No detections"}</p></article>)}</div></section>
  </>;
}

function Metric({ label, value }) { return <div className="metric-card"><span>{label}</span><strong>{value}</strong></div>; }

function App() {
  const [files, setFiles] = useState({ North: null, East: null, South: null, West: null });
  const [emergency, setEmergency] = useState({ North: 0, East: 0, South: 0, West: 0 });
  const [confidence, setConfidence] = useState(0.35);
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const ready = DIRECTIONS.every((direction) => files[direction]);

  async function analyze() {
    setLoading(true); setError("");
    const form = new FormData();
    DIRECTIONS.forEach((direction) => form.append(direction.toLowerCase(), files[direction]));
    form.append("confidence", confidence);
    DIRECTIONS.forEach((direction) => form.append(`emergency_${direction.toLowerCase()}`, emergency[direction]));
    try { const response = await fetch(`${API_URL}/api/analyze`, { method: "POST", body: form }); if (!response.ok) throw new Error("Analysis failed. Check that the Python API is running."); setData(await response.json()); } catch (requestError) { setError(requestError.message); } finally { setLoading(false); }
  }

  return <div className="app-shell"><aside className="sidebar"><div className="brand-lockup"><span className="brand-mark">SS</span><div><strong>Signal Sense</strong><span>Intersection intelligence</span></div></div><div className="side-divider" /><nav><a className="active" href="#upload">Analysis workspace</a><a href="#plan">Signal plan</a><a href="#evidence">Evidence</a></nav><div className="side-note"><span className="status-dot" /> Local model online<strong>YOLO26m</strong><small>Inference stays on this machine.</small></div></aside><main className="main-content"><header className="topbar"><span>Traffic control / Demo workspace</span><span className="date-label">4 approaches · still-image mode</span></header><section className="hero"><div><span className="eyebrow">Adaptive intersection control</span><h1>Make every second at the junction count.</h1><p>Upload one frame from each approach. Signal Sense turns vehicle demand, queue pressure, and emergency priority into a clear signal plan.</p></div><div className="hero-orbit"><div className="orbit-center">4<span>views</span></div><i className="orbit-dot north" /><i className="orbit-dot east" /><i className="orbit-dot south" /><i className="orbit-dot west" /></div></section><section className="section-block upload-section" id="upload"><div className="section-heading"><div><span className="section-kicker">01 / input frames</span><h2>Load the four approaches</h2></div><label className="confidence-control">Confidence <input type="range" min="0.1" max="0.9" step="0.05" value={confidence} onChange={(event) => setConfidence(Number(event.target.value))} /><b>{confidence.toFixed(2)}</b></label></div><div className="upload-grid">{DIRECTIONS.map((direction) => <UploadCard key={direction} direction={direction} file={files[direction]} onFile={(file) => setFiles((value) => ({ ...value, [direction]: file }))} emergency={emergency[direction]} onEmergency={(value) => setEmergency((current) => ({ ...current, [direction]: value }))} />)}</div><div className="action-row"><div>{error && <p className="error-message">{error}</p>}{!ready && !error && <span className="helper-text">Add one image for each approach to unlock analysis.</span>}</div><button className="analyze-button" disabled={!ready || loading} onClick={analyze}>{loading ? "Analyzing with YOLO26m…" : "Analyze junction →"}</button></div></section>{data && <Results data={data} />}</main></div>;
}

const rootElement = document.getElementById("root");
const root = window.__signalSenseRoot || createRoot(rootElement);
window.__signalSenseRoot = root;
root.render(<StrictMode><App /></StrictMode>);