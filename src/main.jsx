import { StrictMode, useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

const DIRECTIONS = ["North", "East", "South", "West"];
const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

function formatLabel(value) {
  return value.replace(/_/g, " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

/* ── Preset scenarios ────────────────────────────────────────── */
const PRESETS = [
  { id: "peak",      label: "Peak hour",              emergency: { North: 0, East: 0, South: 0, West: 0 } },
  { id: "emergency", label: "Ambulance on South",     emergency: { North: 0, East: 0, South: 3, West: 0 } },
  { id: "nightlow",  label: "Late night",             emergency: { North: 0, East: 0, South: 0, West: 0 } },
  { id: "balanced",  label: "Even traffic",           emergency: { North: 0, East: 0, South: 0, West: 0 } },
];

/* ── Impact estimation (vs naive fixed-time 30s each) ────────── */
function computeImpactMetrics(data) {
  const totalVehicles = data.totalVehicles || 1;
  const avgGreen = data.results.reduce((sum, r) => sum + r.greenSeconds, 0) / data.results.length;

  const fixedDelay = 30 * totalVehicles * 0.45;
  const adaptiveDelay = avgGreen * totalVehicles * 0.22;
  const delayReduction = Math.round(((fixedDelay - adaptiveDelay) / fixedDelay) * 100);

  const savedIdleMinutes = ((fixedDelay - adaptiveDelay) / 60) * 4;
  const fuelSaved = Math.max(0.1, savedIdleMinutes * 0.015 * 24).toFixed(1);
  const co2Saved = (fuelSaved * 2.31).toFixed(1);

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

/* ── Components ──────────────────────────────────────────────── */

function UploadCard({ direction, file, onFile, emergency, onEmergency }) {
  return (
    <article className="upload-card">
      <div className="card-heading">
        <div><span className="direction-index">{DIRECTIONS.indexOf(direction) + 1}</span><h3>{direction}</h3></div>
        <span className="approach-label">approach</span>
      </div>
      <label className={`drop-zone ${file ? "has-file" : ""}`}>
        <input type="file" accept="image/jpeg,image/png" onChange={(e) => onFile(e.target.files?.[0] || null)} />
        <span className="upload-mark">{file ? "✓" : "+"}</span>
        <strong>{file ? file.name : "Choose image"}</strong>
        <small>{file ? `${Math.round(file.size / 1024)} KB` : "JPG or PNG"}</small>
      </label>
      <div className="override-row">
        <label htmlFor={`em-${direction}`}>Emergency vehicles</label>
        <input id={`em-${direction}`} type="number" min="0" max="10" value={emergency} onChange={(e) => onEmergency(Number(e.target.value))} />
      </div>
    </article>
  );
}

function SignalLight({ color, active }) {
  return <span className={`lamp lamp-${color} ${active ? "active" : ""}`} />;
}

function ImpactMetrics({ data }) {
  const m = useMemo(() => computeImpactMetrics(data), [data]);
  return (
    <div className="impact-grid">
      <div className="impact-card positive">
        <span className="impact-label">Delay reduction</span>
        <strong className="impact-value">−{m.delayReduction}%</strong>
        <span className="impact-note">vs fixed-time 30 s signal</span>
      </div>
      <div className="impact-card positive">
        <span className="impact-label">Fuel saved per day</span>
        <strong className="impact-value">~{m.fuelSaved} L</strong>
        <span className="impact-note">less idle at this junction</span>
      </div>
      <div className="impact-card positive">
        <span className="impact-label">CO₂ avoided per day</span>
        <strong className="impact-value">~{m.co2Saved} kg</strong>
        <span className="impact-note">at 2.31 kg per litre</span>
      </div>
      <div className={`impact-card ${m.los <= "B" ? "positive" : m.los <= "D" ? "neutral" : "alert"}`}>
        <span className="impact-label">Level of service</span>
        <strong className="impact-value">LOS {m.los}</strong>
        <span className="impact-note">HCM intersection grade</span>
      </div>
    </div>
  );
}

function IntersectionMap({ data, activeDirection, activeColor }) {
  const dirColors = {};
  for (const d of DIRECTIONS) {
    dirColors[d] = d === activeDirection ? (activeColor || "red") : "red";
  }
  const counts = {};
  for (const r of data.results) counts[r.direction] = r.vehicleCount;

  return (
    <svg className="intersection-svg" viewBox="0 0 400 400" xmlns="http://www.w3.org/2000/svg">
      <rect width="400" height="400" fill="#111110" rx="10" />
      {/* Roads */}
      <rect x="150" y="0" width="100" height="400" className="road-fill" />
      <rect x="0" y="150" width="400" height="100" className="road-fill" />
      <rect x="150" y="150" width="100" height="100" fill="#282826" />
      {/* Borders */}
      <rect x="150" y="0" width="100" height="150" className="road-border" />
      <rect x="150" y="250" width="100" height="150" className="road-border" />
      <rect x="0" y="150" width="150" height="100" className="road-border" />
      <rect x="250" y="150" width="150" height="100" className="road-border" />
      {/* Lane markings */}
      <line x1="200" y1="0" x2="200" y2="140" className="road-marking" />
      <line x1="200" y1="260" x2="200" y2="400" className="road-marking" />
      <line x1="0" y1="200" x2="140" y2="200" className="road-marking" />
      <line x1="260" y1="200" x2="400" y2="200" className="road-marking" />
      {/* Signals */}
      <circle cx="165" cy="140" r="6" className={`signal-head ${dirColors.North}`} />
      <circle cx="235" cy="260" r="6" className={`signal-head ${dirColors.South}`} />
      <circle cx="260" cy="165" r="6" className={`signal-head ${dirColors.East}`} />
      <circle cx="140" cy="235" r="6" className={`signal-head ${dirColors.West}`} />
      {/* Labels + counts */}
      <text x="200" y="40" className="intersection-label">NORTH</text>
      <text x="200" y="57" className="intersection-count">{counts.North ?? "–"}</text>
      <text x="200" y="367" className="intersection-label">SOUTH</text>
      <text x="200" y="384" className="intersection-count">{counts.South ?? "–"}</text>
      <text x="340" y="196" className="intersection-label">EAST</text>
      <text x="340" y="213" className="intersection-count">{counts.East ?? "–"}</text>
      <text x="60" y="196" className="intersection-label">WEST</text>
      <text x="60" y="213" className="intersection-count">{counts.West ?? "–"}</text>
      {/* Arrows */}
      <polygon points="194,92 200,78 206,92" fill="#5c5752" opacity=".5" />
      <polygon points="194,308 200,322 206,308" fill="#5c5752" opacity=".5" />
      <polygon points="308,194 322,200 308,206" fill="#5c5752" opacity=".5" />
      <polygon points="92,194 78,200 92,206" fill="#5c5752" opacity=".5" />
    </svg>
  );
}

function TrafficSimulation({ data }) {
  const [running, setRunning] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const schedule = useMemo(() => data.phaseOrder.flatMap((dir) => [
    { direction: dir, color: "green", seconds: data.results.find((r) => r.direction === dir).greenSeconds },
    { direction: dir, color: "yellow", seconds: 4 },
    { direction: null, color: "all-red", seconds: 2 },
  ]), [data]);
  const cycle = schedule.reduce((t, s) => t + s.seconds, 0);

  useEffect(() => {
    if (!running) return;
    const id = setInterval(() => setElapsed((v) => (v + 1) % cycle), 1000);
    return () => clearInterval(id);
  }, [running, cycle]);

  let cursor = 0, active = schedule[0], activeStart = 0;
  for (const s of schedule) {
    if (elapsed < cursor + s.seconds) { active = s; activeStart = cursor; break; }
    cursor += s.seconds;
  }
  const remaining = activeStart + active.seconds - elapsed;
  const progress = ((elapsed + 1) / cycle) * 100;

  return (
    <>
      <div className="intersection-map-wrap">
        <div className="section-heading compact-heading">
          <div><span className="section-kicker">Junction view</span><h2>Intersection map</h2></div>
          <span className="quiet-badge">{running ? "Live" : "Paused"}</span>
        </div>
        <IntersectionMap data={data} activeDirection={active.direction} activeColor={active.color === "all-red" ? "red" : active.color} />
      </div>

      <section className="simulation-panel">
        <div className="section-heading compact-heading">
          <div><span className="section-kicker">Controller</span><h2>Signal cycle</h2></div>
          <button className={`live-toggle ${running ? "on" : ""}`} onClick={() => setRunning((v) => !v)}>
            <span /> {running ? "Pause" : "Run cycle"}
          </button>
        </div>
        <div className="simulation-status">
          <div>
            <span className="status-label">Current phase</span>
            <strong>{active.direction ? `${active.direction} — ${active.color}` : "All red"}</strong>
          </div>
          <span>{remaining}s left</span>
        </div>
        <div className="cycle-track"><span style={{ width: `${progress}%` }} /></div>
        <div className="lights-grid">
          {DIRECTIONS.map((dir) => {
            const current = active.direction === dir ? active.color : "red";
            return (
              <div className="light-card" key={dir}>
                <div className="light-card-title"><strong>{dir}</strong><span>{current}</span></div>
                <div className="lamp-stack">
                  <div><SignalLight color="red" active={current === "red"} />Red</div>
                  <div><SignalLight color="yellow" active={current === "yellow"} />Yellow</div>
                  <div><SignalLight color="green" active={current === "green"} />Green</div>
                </div>
              </div>
            );
          })}
        </div>
      </section>
    </>
  );
}

function Results({ data }) {
  return (
    <>
      <section className="section-block results-section" id="plan">
        <div className="section-heading">
          <div><span className="section-kicker">Results</span><h2>Junction overview</h2></div>
          <span className="quiet-badge">Done</span>
        </div>
        <div className="metric-grid">
          <Metric label="Vehicles" value={data.totalVehicles} />
          <Metric label="Demand points" value={data.totalPoints} />
          <Metric label="Emergency" value={data.totalEmergency} />
          <Metric label="First green" value={data.phaseOrder[0]} />
        </div>
      </section>

      <section className="section-block">
        <div className="section-heading">
          <div><span className="section-kicker">Estimated impact</span><h2>Adaptive vs fixed-time</h2></div>
        </div>
        <ImpactMetrics data={data} />
      </section>

      <section className="section-block">
        <div className="section-heading">
          <div><span className="section-kicker">Signal plan</span><h2>Recommended cycle</h2></div>
          <span className="quiet-badge">120 s</span>
        </div>
        <div className="plan-layout">
          <div className="priority-list">
            {data.phaseOrder.map((dir, i) => {
              const r = data.results.find((x) => x.direction === dir);
              return (
                <div className="priority-row" key={dir}>
                  <span className="rank">{i + 1}</span>
                  <div className="priority-name">
                    <strong>{dir}</strong>
                    <span>{r.demandPoints} pts · {r.queueMeters} m queue</span>
                  </div>
                  <strong className="green-time">{r.greenSeconds}s</strong>
                </div>
              );
            })}
          </div>
          <div className="score-legend">
            <strong>How timing works</strong>
            <p>Green time is split proportionally across vehicle load, queue length, and emergency priority. More demand means more green.</p>
            <div className="legend-line"><span className="legend-dot blue" /> Vehicle load</div>
            <div className="legend-line"><span className="legend-dot amber" /> Queue pressure</div>
            <div className="legend-line"><span className="legend-dot red" /> Emergency</div>
          </div>
        </div>
      </section>

      <TrafficSimulation data={data} />

      <section className="section-block evidence-section" id="evidence">
        <div className="section-heading">
          <div><span className="section-kicker">Computer vision</span><h2>Detection evidence</h2></div>
          <span className="muted-copy">YOLO26m annotated frames</span>
        </div>
        <div className="evidence-grid">
          {data.results.map((r) => (
            <article className="evidence-card" key={r.direction}>
              <div className="evidence-title">
                <strong>{r.direction}</strong>
                <span>{r.demandPoints} pts</span>
              </div>
              <img src={r.annotatedImage} alt={`${r.direction} detections`} />
              <p>{Object.entries(r.counts).map(([l, c]) => `${formatLabel(l)} ${c}`).join(" · ") || "No detections"}</p>
            </article>
          ))}
        </div>
      </section>
    </>
  );
}

function Metric({ label, value }) {
  return <div className="metric-card"><span>{label}</span><strong>{value}</strong></div>;
}

/* ── App ─────────────────────────────────────────────────────── */
function App() {
  const [files, setFiles] = useState({ North: null, East: null, South: null, West: null });
  const [emergency, setEmergency] = useState({ North: 0, East: 0, South: 0, West: 0 });
  const [confidence, setConfidence] = useState(0.35);
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [backendStatus, setBackendStatus] = useState("checking");
  const ready = DIRECTIONS.every((d) => files[d]);

  useEffect(() => {
    let active = true;
    const checkHealth = () => {
      fetch(`${API_URL}/api/health`)
        .then((res) => {
          if (!active) return;
          setBackendStatus(res.ok ? "online" : "offline");
        })
        .catch(() => {
          if (!active) return;
          setBackendStatus("offline");
        });
    };
    checkHealth();
    const interval = setInterval(checkHealth, 8000);
    return () => {
      active = false;
      clearInterval(interval);
    };
  }, []);

  async function analyze() {
    setLoading(true);
    setError("");
    const form = new FormData();
    DIRECTIONS.forEach((d) => form.append(d.toLowerCase(), files[d]));
    form.append("confidence", confidence);
    DIRECTIONS.forEach((d) => form.append(`emergency_${d.toLowerCase()}`, emergency[d]));
    try {
      const res = await fetch(`${API_URL}/api/analyze`, { method: "POST", body: form });
      if (!res.ok) {
        const errorText = await res.text();
        throw new Error(errorText || "Analysis failed — check the Python API logs.");
      }
      setData(await res.json());
      setBackendStatus("online");
    } catch (err) {
      if (err.name === "TypeError" && err.message.toLowerCase().includes("fetch")) {
        setError(
          "Cannot connect to Python API (http://localhost:8000). Make sure the backend is running with: uvicorn api:app --reload --port 8000"
        );
        setBackendStatus("offline");
      } else {
        setError(err.message || "An error occurred during junction analysis.");
      }
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand-lockup">
          <span className="brand-mark">Nx</span>
          <div>
            <strong>Navintrix AI</strong>
            <span>Traffic intelligence</span>
          </div>
        </div>
        <div className="side-divider" />
        <nav>
          <a className="active" href="#upload">Workspace</a>
          <a href="#plan">Signal plan</a>
          <a href="#evidence">Evidence</a>
        </nav>
        <div className="side-note">
          <span className={`status-dot ${backendStatus === "offline" ? "offline" : ""}`} />{" "}
          {backendStatus === "online" ? "Model ready" : backendStatus === "offline" ? "Backend offline" : "Connecting..."}
          <strong>YOLO26m</strong>
          <small>All inference runs locally on your machine. Nothing leaves your network.</small>
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <span>Navintrix AI / workspace</span>
          <span>4 approaches · still-image mode</span>
        </header>

        <section className="hero">
          <h1>Smarter signals for <em>every junction.</em></h1>
          <p>
            Drop a camera frame from each of the four approaches and Navintrix will figure out
            how long each direction should get — accounting for vehicle types, queue build-up,
            and any emergency overrides you set.
          </p>
        </section>

        <section className="section-block upload-section" id="upload">
          <div className="section-heading">
            <div><span className="section-kicker">Input</span><h2>Upload the four approaches</h2></div>
            <label className="confidence-control">
              Confidence <input type="range" min="0.1" max="0.9" step="0.05" value={confidence} onChange={(e) => setConfidence(Number(e.target.value))} />
              <b>{confidence.toFixed(2)}</b>
            </label>
          </div>
          <div className="upload-grid">
            {DIRECTIONS.map((d) => (
              <UploadCard
                key={d}
                direction={d}
                file={files[d]}
                onFile={(f) => setFiles((prev) => ({ ...prev, [d]: f }))}
                emergency={emergency[d]}
                onEmergency={(v) => setEmergency((prev) => ({ ...prev, [d]: v }))}
              />
            ))}
          </div>
          <div className="presets-row">
            {PRESETS.map((p) => (
              <button key={p.id} className="preset-btn" onClick={() => setEmergency(p.emergency)}>
                {p.label}
              </button>
            ))}
          </div>
          <div className="action-row">
            <div>
              {error && <p className="error-message">{error}</p>}
              {!ready && !error && <span className="helper-text">Pick one image for each direction to get started.</span>}
            </div>
            <button className="analyze-button" disabled={!ready || loading} onClick={analyze}>
              {loading ? "Running YOLO26m…" : "Analyze junction →"}
            </button>
          </div>
        </section>

        {data && <Results data={data} />}
      </main>
    </div>
  );
}

const rootEl = document.getElementById("root");
const root = window.__navintrixRoot || createRoot(rootEl);
window.__navintrixRoot = root;
root.render(<StrictMode><App /></StrictMode>);