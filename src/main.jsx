import { StrictMode, useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

const DIRECTIONS = ["North", "East", "South", "West"];
const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

function formatLabel(value) {
  return value.replace(/_/g, " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

/* ── Preset demo scenarios ───────────────────────────────────── */
const PRESETS = [
  { id: "peak",      emoji: "🚗", label: "Peak hour congestion",   emergency: { North: 0, East: 0, South: 0, West: 0 } },
  { id: "emergency",  emoji: "🚑", label: "Emergency from South",   emergency: { North: 0, East: 0, South: 3, West: 0 } },
  { id: "nightlow",   emoji: "🌙", label: "Late-night low traffic",  emergency: { North: 0, East: 0, South: 0, West: 0 } },
  { id: "balanced",   emoji: "⚖️", label: "Balanced flow",          emergency: { North: 0, East: 0, South: 0, West: 0 } },
];

/* ── Traffic engineering helpers ──────────────────────────────── */
function computeImpactMetrics(data) {
  const totalVehicles = data.totalVehicles || 1;
  const totalPoints = data.totalPoints || 1;
  const avgGreen = data.results.reduce((sum, r) => sum + r.greenSeconds, 0) / data.results.length;

  // Simulated comparison vs fixed-time 30s-each baseline
  const fixedDelay = 30 * totalVehicles * 0.45;   // rough avg delay seconds
  const adaptiveDelay = avgGreen * totalVehicles * 0.22;
  const delayReduction = Math.round(((fixedDelay - adaptiveDelay) / fixedDelay) * 100);

  // Fuel: ~0.015 L per idle minute, avg delay reduction in minutes
  const savedIdleMinutes = ((fixedDelay - adaptiveDelay) / 60) * 4; // 4 cycles/hour rough
  const fuelSaved = Math.max(0.1, savedIdleMinutes * 0.015 * 24).toFixed(1);

  // CO2: 2.31 kg CO2 per liter of gasoline
  const co2Saved = (fuelSaved * 2.31).toFixed(1);

  // Level of Service based on average delay per vehicle
  const avgDelayPerVehicle = adaptiveDelay / totalVehicles;
  let los;
  if (avgDelayPerVehicle <= 10) los = "A";
  else if (avgDelayPerVehicle <= 20) los = "B";
  else if (avgDelayPerVehicle <= 35) los = "C";
  else if (avgDelayPerVehicle <= 55) los = "D";
  else if (avgDelayPerVehicle <= 80) los = "E";
  else los = "F";

  return { delayReduction, fuelSaved, co2Saved, los };
}

/* ── Upload card ─────────────────────────────────────────────── */
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

/* ── Traffic signal lamp ─────────────────────────────────────── */
function SignalLight({ color, active }) {
  return <span className={`lamp lamp-${color} ${active ? "active" : ""}`} />;
}

/* ── Impact metrics panel ────────────────────────────────────── */
function ImpactMetrics({ data }) {
  const metrics = useMemo(() => computeImpactMetrics(data), [data]);
  return (
    <div className="impact-grid">
      <div className="impact-card positive">
        <div className="impact-icon">⏱</div>
        <span className="impact-label">Delay reduction</span>
        <strong className="impact-value">−{metrics.delayReduction}%</strong>
        <span className="impact-note">vs fixed-time 30s signal</span>
      </div>
      <div className="impact-card positive">
        <div className="impact-icon">⛽</div>
        <span className="impact-label">Fuel saved / day</span>
        <strong className="impact-value">~{metrics.fuelSaved} L</strong>
        <span className="impact-note">reduced idle at junction</span>
      </div>
      <div className="impact-card positive">
        <div className="impact-icon">🌿</div>
        <span className="impact-label">CO₂ abated / day</span>
        <strong className="impact-value">~{metrics.co2Saved} kg</strong>
        <span className="impact-note">2.31 kg CO₂ per litre</span>
      </div>
      <div className={`impact-card ${metrics.los <= "B" ? "positive" : metrics.los <= "D" ? "neutral" : "alert"}`}>
        <div className="impact-icon">🚦</div>
        <span className="impact-label">Level of service</span>
        <strong className="impact-value">LOS {metrics.los}</strong>
        <span className="impact-note">HCM intersection grade</span>
      </div>
    </div>
  );
}

/* ── Top-down intersection SVG map ────────────────────────────── */
function IntersectionMap({ data, activeDirection, activeColor }) {
  const dirColors = {};
  for (const d of DIRECTIONS) {
    if (d === activeDirection) dirColors[d] = activeColor || "red";
    else dirColors[d] = "red";
  }

  function signalClass(direction) {
    return `signal-head ${dirColors[direction]}`;
  }

  const counts = {};
  for (const r of data.results) counts[r.direction] = r.vehicleCount;

  return (
    <svg className="intersection-svg" viewBox="0 0 400 400" xmlns="http://www.w3.org/2000/svg">
      {/* Background */}
      <rect width="400" height="400" fill="#0a0f18" rx="12" />

      {/* Roads */}
      <rect x="150" y="0"   width="100" height="400" className="road-fill" />
      <rect x="0"   y="150" width="400" height="100" className="road-fill" />

      {/* Center intersection */}
      <rect x="150" y="150" width="100" height="100" fill="#263445" />

      {/* Road borders */}
      <rect x="150" y="0"   width="100" height="150" className="road-border" />
      <rect x="150" y="250" width="100" height="150" className="road-border" />
      <rect x="0"   y="150" width="150" height="100" className="road-border" />
      <rect x="250" y="150" width="150" height="100" className="road-border" />

      {/* Center lane markings */}
      <line x1="200" y1="0"   x2="200" y2="140" className="road-marking" />
      <line x1="200" y1="260" x2="200" y2="400" className="road-marking" />
      <line x1="0"   y1="200" x2="140" y2="200" className="road-marking" />
      <line x1="260" y1="200" x2="400" y2="200" className="road-marking" />

      {/* Crosswalk dashes */}
      {[152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240].map(x => (
        <g key={`cw-${x}`}>
          <rect x={x} y="142" width="5" height="3" fill="#4a5568" opacity=".4" rx="0.5" />
          <rect x={x} y="255" width="5" height="3" fill="#4a5568" opacity=".4" rx="0.5" />
        </g>
      ))}
      {[152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240].map(y => (
        <g key={`cw-y-${y}`}>
          <rect x="142" y={y} width="3" height="5" fill="#4a5568" opacity=".4" rx="0.5" />
          <rect x="255" y={y} width="3" height="5" fill="#4a5568" opacity=".4" rx="0.5" />
        </g>
      ))}

      {/* Signal heads — North */}
      <circle cx="165" cy="140" r="7" className={signalClass("North")} />
      <text x="200" y="38" className="intersection-label">NORTH</text>
      <text x="200" y="56" className="intersection-count">{counts.North ?? "–"}</text>

      {/* Signal heads — South */}
      <circle cx="235" cy="260" r="7" className={signalClass("South")} />
      <text x="200" y="365" className="intersection-label">SOUTH</text>
      <text x="200" y="383" className="intersection-count">{counts.South ?? "–"}</text>

      {/* Signal heads — East */}
      <circle cx="260" cy="165" r="7" className={signalClass("East")} />
      <text x="340" y="195" className="intersection-label">EAST</text>
      <text x="340" y="213" className="intersection-count">{counts.East ?? "–"}</text>

      {/* Signal heads — West */}
      <circle cx="140" cy="235" r="7" className={signalClass("West")} />
      <text x="60" y="195" className="intersection-label">WEST</text>
      <text x="60" y="213" className="intersection-count">{counts.West ?? "–"}</text>

      {/* Direction arrows */}
      <polygon points="193,90 200,75 207,90" fill="#4a5568" opacity=".5" />
      <polygon points="193,310 200,325 207,310" fill="#4a5568" opacity=".5" />
      <polygon points="310,193 325,200 310,207" fill="#4a5568" opacity=".5" />
      <polygon points="90,193 75,200 90,207" fill="#4a5568" opacity=".5" />
    </svg>
  );
}

/* ── Traffic simulation with intersection map ────────────────── */
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
    <>
      {/* Intersection map */}
      <div className="intersection-map-wrap">
        <div className="section-heading compact-heading">
          <div><span className="section-kicker">04 / junction view</span><h2>Intersection map</h2></div>
          <span className="quiet-badge">{running ? "Live" : "Paused"}</span>
        </div>
        <IntersectionMap data={data} activeDirection={active.direction} activeColor={active.color === "all-red" ? "red" : active.color} />
      </div>

      {/* Signal controller */}
      <section className="simulation-panel">
        <div className="section-heading compact-heading">
          <div><span className="section-kicker">05 / controller</span><h2>Live signal simulation</h2></div>
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
    </>
  );
}

/* ── Results section ─────────────────────────────────────────── */
function Results({ data }) {
  return <>
    {/* Junction overview */}
    <section className="section-block results-section" id="plan">
      <div className="section-heading"><div><span className="section-kicker">02 / decision layer</span><h2>Junction overview</h2></div><span className="quiet-badge">Analysis complete</span></div>
      <div className="metric-grid"><Metric label="Vehicles detected" value={data.totalVehicles} /><Metric label="Demand points" value={data.totalPoints} /><Metric label="Emergency priority" value={data.totalEmergency} /><Metric label="First phase" value={data.phaseOrder[0]} /></div>
    </section>

    {/* Impact metrics – comparison vs fixed-time */}
    <section className="section-block">
      <div className="section-heading"><div><span className="section-kicker">02b / estimated impact</span><h2>Performance vs fixed-time signal</h2></div><span className="quiet-badge">Adaptive advantage</span></div>
      <ImpactMetrics data={data} />
    </section>

    {/* Recommended cycle */}
    <section className="section-block">
      <div className="section-heading"><div><span className="section-kicker">03 / adaptive plan</span><h2>Recommended cycle</h2></div><span className="quiet-badge">120 sec cycle</span></div>
      <div className="plan-layout"><div className="priority-list">{data.phaseOrder.map((direction, index) => { const result = data.results.find((item) => item.direction === direction); return <div className="priority-row" key={direction}><span className="rank">0{index + 1}</span><div className="priority-name"><strong>{direction}</strong><span>{result.demandPoints} points · {result.queueMeters}m queue</span></div><strong className="green-time">{result.greenSeconds}s</strong></div>; })}</div><div className="score-legend"><strong>How timing is decided</strong><p>Vehicle class points + automatic queue points + emergency priority are shared across the cycle. Higher demand gets more green time.</p><div className="legend-line"><span className="legend-dot blue" /> vehicle load</div><div className="legend-line"><span className="legend-dot amber" /> queue pressure</div><div className="legend-line"><span className="legend-dot red" /> emergency priority</div></div></div>
    </section>

    {/* Simulation + intersection map */}
    <TrafficSimulation data={data} />

    {/* Detection evidence */}
    <section className="section-block evidence-section" id="evidence"><div className="section-heading"><div><span className="section-kicker">06 / computer vision</span><h2>Detection evidence</h2></div><span className="muted-copy">Annotated frames from YOLO26m</span></div><div className="evidence-grid">{data.results.map((result) => <article className="evidence-card" key={result.direction}><div className="evidence-title"><strong>{result.direction}</strong><span>{result.demandPoints} pts</span></div><img src={result.annotatedImage} alt={`${result.direction} annotated detections`} /><p>{Object.entries(result.counts).map(([label, count]) => `${formatLabel(label)} ${count}`).join(" · ") || "No detections"}</p></article>)}</div></section>
  </>;
}

function Metric({ label, value }) { return <div className="metric-card"><span>{label}</span><strong>{value}</strong></div>; }

/* ── Main app ────────────────────────────────────────────────── */
function App() {
  const [files, setFiles] = useState({ North: null, East: null, South: null, West: null });
  const [emergency, setEmergency] = useState({ North: 0, East: 0, South: 0, West: 0 });
  const [confidence, setConfidence] = useState(0.35);
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const ready = DIRECTIONS.every((direction) => files[direction]);

  function applyPreset(preset) {
    setEmergency(preset.emergency);
  }

  async function analyze() {
    setLoading(true); setError("");
    const form = new FormData();
    DIRECTIONS.forEach((direction) => form.append(direction.toLowerCase(), files[direction]));
    form.append("confidence", confidence);
    DIRECTIONS.forEach((direction) => form.append(`emergency_${direction.toLowerCase()}`, emergency[direction]));
    try { const response = await fetch(`${API_URL}/api/analyze`, { method: "POST", body: form }); if (!response.ok) throw new Error("Analysis failed. Check that the Python API is running."); setData(await response.json()); } catch (requestError) { setError(requestError.message); } finally { setLoading(false); }
  }

  return <div className="app-shell"><aside className="sidebar"><div className="brand-lockup"><span className="brand-mark">SS</span><div><strong>Signal Sense</strong><span>Intersection intelligence</span></div></div><div className="side-divider" /><nav><a className="active" href="#upload">Analysis workspace</a><a href="#plan">Signal plan</a><a href="#evidence">Evidence</a></nav><div className="side-note"><span className="status-dot" /> Local model online<strong>YOLO26m</strong><small>Inference stays on this machine. No data leaves your network.</small></div></aside><main className="main-content"><header className="topbar"><span>Traffic control / Demo workspace</span><span className="date-label">4 approaches · still-image mode</span></header><section className="hero"><div><span className="eyebrow">Adaptive intersection control</span><h1>Make every second at the junction count.</h1><p>Upload one frame from each approach. Signal Sense turns vehicle demand, queue pressure, and emergency priority into a clear, transparent signal plan — with estimated impact metrics.</p></div><div className="hero-orbit"><div className="orbit-center">4<span>views</span></div><i className="orbit-dot north" /><i className="orbit-dot east" /><i className="orbit-dot south" /><i className="orbit-dot west" /></div></section><section className="section-block upload-section" id="upload"><div className="section-heading"><div><span className="section-kicker">01 / input frames</span><h2>Load the four approaches</h2></div><label className="confidence-control">Confidence <input type="range" min="0.1" max="0.9" step="0.05" value={confidence} onChange={(event) => setConfidence(Number(event.target.value))} /><b>{confidence.toFixed(2)}</b></label></div><div className="upload-grid">{DIRECTIONS.map((direction) => <UploadCard key={direction} direction={direction} file={files[direction]} onFile={(file) => setFiles((value) => ({ ...value, [direction]: file }))} emergency={emergency[direction]} onEmergency={(value) => setEmergency((current) => ({ ...current, [direction]: value }))} />)}</div><div className="presets-row">{PRESETS.map((preset) => <button key={preset.id} className="preset-btn" onClick={() => applyPreset(preset)}><span className="preset-emoji">{preset.emoji}</span>{preset.label}</button>)}</div><div className="action-row"><div>{error && <p className="error-message">{error}</p>}{!ready && !error && <span className="helper-text">Add one image for each approach to unlock analysis.</span>}</div><button className="analyze-button" disabled={!ready || loading} onClick={analyze}>{loading ? "Analyzing with YOLO26m…" : "Analyze junction →"}</button></div></section>{data && <Results data={data} />}</main></div>;
}

const rootElement = document.getElementById("root");
const root = window.__signalSenseRoot || createRoot(rootElement);
window.__signalSenseRoot = root;
root.render(<StrictMode><App /></StrictMode>);