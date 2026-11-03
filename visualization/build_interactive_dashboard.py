"""
Build script for AQSF-NAV v2 Interactive HTML5 Replay Dashboard.
Embeds the actual recorded simulation trajectories from results/v2_all_benchmarks.json
into a rich, standalone HTML5 Canvas visualization.
"""

import json
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
OUTPUT_HTML = Path(__file__).resolve().parent / "aqsf_v2_interactive_dashboard.html"


def generate_html():
    json_path = RESULTS_DIR / "v2_all_benchmarks.json"
    if not json_path.exists():
        print("Error: results/v2_all_benchmarks.json not found.")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        benchmarks = json.load(f)

    # Convert to compact scenario dictionary for fast client-side switching
    data_by_scenario = {}
    for item in benchmarks:
        sc = item["scenario_name"]
        bl = item["baseline"]
        if sc not in data_by_scenario:
            data_by_scenario[sc] = {}
        data_by_scenario[sc][bl] = {
            "success": item["success"],
            "steps": item["total_steps"],
            "min_robot_dist": item["min_inter_robot_distance"],
            "min_obs_dist": item["min_obstacle_distance"],
            "collisions": item["total_collisions"],
            "cbf_interventions": item["total_cbf_interventions"],
            "recon_error": item["mean_reconstruction_error"],
            "records": item["step_records"]
        }

    embedded_json = json.dumps(data_by_scenario)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>AQSF-NAV v2: Predictive Electrostatic Safety Architecture Dashboard</title>
<style>
  :root {{
    --bg-dark: #0f172a;
    --card-bg: #1e293b;
    --text-main: #f8fafc;
    --text-muted: #94a3b8;
    --accent-blue: #38bdf8;
    --accent-green: #4ade80;
    --accent-red: #f87171;
    --accent-amber: #fbbf24;
    --accent-purple: #c084fc;
    --border: #334155;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace; }}
  body {{ background: var(--bg-dark); color: var(--text-main); display: flex; flex-direction: column; height: 100vh; overflow: hidden; }}
  header {{ background: var(--card-bg); padding: 12px 24px; border-bottom: 1px solid var(--border); display: flex; justify-content: space-between; align-items: center; }}
  h1 {{ font-size: 1.15rem; font-weight: 700; color: var(--accent-blue); display: flex; align-items: center; gap: 8px; }}
  .tag {{ background: rgba(56, 189, 248, 0.15); color: var(--accent-blue); border: 1px solid var(--accent-blue); padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; }}
  .main-container {{ display: flex; flex: 1; overflow: hidden; }}
  .sidebar {{ width: 340px; background: var(--card-bg); border-right: 1px solid var(--border); padding: 16px; display: flex; flex-direction: column; gap: 14px; overflow-y: auto; }}
  .viewport-container {{ flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center; position: relative; padding: 16px; background: #090d16; }}
  canvas {{ background: #0b1120; border: 1px solid var(--border); border-radius: 8px; box-shadow: 0 8px 24px rgba(0,0,0,0.5); }}
  .controls-bar {{ width: 100%; max-width: 650px; margin-top: 12px; display: flex; align-items: center; gap: 12px; background: var(--card-bg); padding: 10px 16px; border-radius: 8px; border: 1px solid var(--border); }}
  button {{ background: #2563eb; color: #fff; border: none; padding: 6px 14px; border-radius: 6px; cursor: pointer; font-weight: 600; font-size: 0.85rem; transition: background 0.2s; }}
  button:hover {{ background: #1d4ed8; }}
  button.secondary {{ background: #475569; }}
  button.secondary:hover {{ background: #334155; }}
  select {{ width: 100%; background: #0f172a; color: var(--text-main); border: 1px solid var(--border); padding: 8px; border-radius: 6px; font-size: 0.85rem; }}
  .slider-container {{ flex: 1; display: flex; align-items: center; gap: 8px; }}
  input[type=range] {{ flex: 1; accent-color: var(--accent-blue); cursor: pointer; }}
  .panel {{ background: #0f172a; padding: 12px; border-radius: 6px; border: 1px solid var(--border); }}
  .panel-title {{ font-size: 0.75rem; text-transform: uppercase; color: var(--text-muted); margin-bottom: 8px; font-weight: 700; letter-spacing: 0.5px; }}
  .telemetry-row {{ display: flex; justify-content: space-between; margin-bottom: 6px; font-size: 0.82rem; }}
  .val {{ font-weight: 600; }}
  .status-badge {{ padding: 3px 8px; border-radius: 4px; font-weight: 700; font-size: 0.75rem; text-align: center; }}
  .badge-pass {{ background: rgba(74, 222, 128, 0.2); color: var(--accent-green); border: 1px solid var(--accent-green); }}
  .badge-fail {{ background: rgba(248, 113, 113, 0.2); color: var(--accent-red); border: 1px solid var(--accent-red); }}
  .legend-item {{ display: flex; align-items: center; gap: 8px; font-size: 0.78rem; margin-bottom: 4px; }}
  .dot {{ width: 10px; height: 10px; border-radius: 50%; display: inline-block; }}
</style>
</head>
<body>

<header>
  <h1>AQSF-NAV v2 <span class="tag">Predictive Electrostatic Safety Architecture</span></h1>
  <div><span id="headerStatus" class="status-badge badge-pass">VALIDATED RUN</span></div>
</header>

<div class="main-container">
  <!-- Sidebar Controls & Telemetry -->
  <div class="sidebar">
    <div>
      <label class="panel-title">1. Scenario</label>
      <select id="scenarioSelect" onchange="onScenarioChange()"></select>
    </div>

    <div>
      <label class="panel-title">2. Comparison Baseline</label>
      <select id="baselineSelect" onchange="onBaselineChange()"></select>
    </div>

    <div class="panel">
      <div class="panel-title">Mission & Safety Status</div>
      <div class="telemetry-row"><span>Status:</span> <span id="telSuccess" class="val">-</span></div>
      <div class="telemetry-row"><span>Total Steps:</span> <span id="telTotalSteps" class="val">-</span></div>
      <div class="telemetry-row"><span>Min Separation:</span> <span id="telMinDist" class="val">-</span></div>
      <div class="telemetry-row"><span>Physical Collisions:</span> <span id="telCollisions" class="val">-</span></div>
      <div class="telemetry-row"><span>CBF Interventions:</span> <span id="telCBFInterv" class="val">-</span></div>
    </div>

    <div class="panel">
      <div class="panel-title">Real-time Step Telemetry</div>
      <div class="telemetry-row"><span>Step:</span> <span id="telStep" class="val">0 / 0</span></div>
      <div class="telemetry-row"><span>Sim Time:</span> <span id="telSimTime" class="val">0.00 s</span></div>
      <div class="telemetry-row"><span>Robot Separation:</span> <span id="telCurrentSep" class="val">-</span></div>
      <div class="telemetry-row"><span>Pred. Min Dist (d*):</span> <span id="telPredDist" class="val">-</span></div>
      <div class="telemetry-row"><span>Time to Collision (t*):</span> <span id="telPredTime" class="val">-</span></div>
      <div class="telemetry-row"><span>CBF Active:</span> <span id="telCurrentCBF" class="val">-</span></div>
    </div>

    <div class="panel">
      <div class="panel-title">Quantum Spectral Metrics</div>
      <div class="telemetry-row"><span>Complexity C(t):</span> <span id="telComplexity" class="val">-</span></div>
      <div class="telemetry-row"><span>Bandwidth B(t):</span> <span id="telBandwidth" class="val">-</span></div>
      <div class="telemetry-row"><span>Retained Power:</span> <span id="telPower" class="val">-</span></div>
      <div class="telemetry-row"><span>Reconstruction L2:</span> <span id="telReconError" class="val">-</span></div>
      <div class="telemetry-row"><span>Gradient Cosine:</span> <span id="telCosine" class="val">-</span></div>
    </div>

    <div class="panel">
      <div class="panel-title">Legend</div>
      <div class="legend-item"><span class="dot" style="background:#38bdf8;"></span> Robot 0 (Blue)</div>
      <div class="legend-item"><span class="dot" style="background:#4ade80;"></span> Robot 1 (Green)</div>
      <div class="legend-item"><span class="dot" style="background:#f87171;"></span> Moving Obstacles (Red)</div>
      <div class="legend-item"><span class="dot" style="background:rgba(56,189,248,0.3); border:1px solid #38bdf8;"></span> Safety Region (D_ij = 0.70)</div>
      <div class="legend-item"><span class="dot" style="background:#fbbf24;"></span> Target Goal (*)</div>
    </div>
  </div>

  <!-- Main Viewport -->
  <div class="viewport-container">
    <canvas id="navCanvas" width="650" height="650"></canvas>
    
    <div class="controls-bar">
      <button id="playBtn" onclick="togglePlay()">Play</button>
      <button class="secondary" onclick="stepBackward()">&#9664;</button>
      <button class="secondary" onclick="stepForward()">&#9654;</button>
      <button class="secondary" onclick="resetPlayback()">Reset</button>
      <div class="slider-container">
        <input type="range" id="scrubber" min="0" max="100" value="0" oninput="onScrub(this.value)">
      </div>
      <span id="stepCounter" style="font-size:0.8rem; font-weight:600; min-width:45px;">0/0</span>
    </div>
  </div>
</div>

<script>
const BENCHMARK_DATA = {embedded_json};

const canvas = document.getElementById("navCanvas");
const ctx = canvas.getContext("2d");
const WORLD_SIZE = 16.0;
const SCALE = canvas.width / WORLD_SIZE;

let currentScenario = Object.keys(BENCHMARK_DATA)[0];
let currentBaseline = Object.keys(BENCHMARK_DATA[currentScenario])[0];
let currentRecords = [];
let currentStepIdx = 0;
let isPlaying = false;
let playInterval = null;

function init() {{
  const scSelect = document.getElementById("scenarioSelect");
  scSelect.innerHTML = "";
  for (const sc in BENCHMARK_DATA) {{
    const opt = document.createElement("option");
    opt.value = sc;
    opt.textContent = sc.replace(/_/g, " ");
    scSelect.appendChild(opt);
  }}
  onScenarioChange();
}}

function onScenarioChange() {{
  currentScenario = document.getElementById("scenarioSelect").value;
  const blSelect = document.getElementById("baselineSelect");
  blSelect.innerHTML = "";
  for (const bl in BENCHMARK_DATA[currentScenario]) {{
    const opt = document.createElement("option");
    opt.value = bl;
    opt.textContent = bl.replace(/_/g, " ");
    blSelect.appendChild(opt);
  }}
  onBaselineChange();
}}

function onBaselineChange() {{
  currentBaseline = document.getElementById("baselineSelect").value;
  const runData = BENCHMARK_DATA[currentScenario][currentBaseline];
  currentRecords = runData.records || [];
  currentStepIdx = 0;

  // Update summary telemetry
  const statusBadge = document.getElementById("telSuccess");
  if (runData.success) {{
    statusBadge.innerHTML = '<span class="status-badge badge-pass">PASS</span>';
  }} else {{
    statusBadge.innerHTML = '<span class="status-badge badge-fail">FAIL</span>';
  }}
  document.getElementById("telTotalSteps").textContent = runData.steps;
  document.getElementById("telMinDist").textContent = runData.min_robot_dist < 1000 ? runData.min_robot_dist.toFixed(3) + " u" : "N/A";
  document.getElementById("telCollisions").textContent = runData.collisions;
  document.getElementById("telCBFInterv").textContent = runData.cbf_interventions;

  const scrubber = document.getElementById("scrubber");
  scrubber.max = Math.max(0, currentRecords.length - 1);
  scrubber.value = 0;

  render();
}}

function toCanvasX(wx) {{ return wx * SCALE; }}
function toCanvasY(wy) {{ return canvas.height - (wy * SCALE); }}

function render() {{
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Draw Grid Lines
  ctx.strokeStyle = "rgba(51, 65, 85, 0.4)";
  ctx.lineWidth = 1;
  for (let i = 0; i <= WORLD_SIZE; i++) {{
    ctx.beginPath();
    ctx.moveTo(toCanvasX(i), 0);
    ctx.lineTo(toCanvasX(i), canvas.height);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(0, toCanvasY(i));
    ctx.lineTo(canvas.width, toCanvasY(i));
    ctx.stroke();
  }}

  if (!currentRecords || currentRecords.length === 0) return;
  const rec = currentRecords[currentStepIdx];

  // Draw Trajectory Trails up to current step
  const colors = ["#38bdf8", "#4ade80", "#c084fc"];
  for (let r = 0; r < rec.robot_positions.length; r++) {{
    ctx.strokeStyle = colors[r % colors.length];
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let s = 0; s <= currentStepIdx; s++) {{
      const pos = currentRecords[s].robot_positions[r];
      const cx = toCanvasX(pos[0]);
      const cy = toCanvasY(pos[1]);
      if (s === 0) ctx.moveTo(cx, cy);
      else ctx.lineTo(cx, cy);
    }}
    ctx.stroke();
  }}

  // Draw Goals
  for (let r = 0; r < rec.goal_positions.length; r++) {{
    const gx = toCanvasX(rec.goal_positions[r][0]);
    const gy = toCanvasY(rec.goal_positions[r][1]);
    ctx.fillStyle = "#fbbf24";
    ctx.beginPath();
    ctx.arc(gx, gy, 8, 0, 2 * Math.PI);
    ctx.fill();
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }}

  // Draw Moving Obstacles
  for (let o = 0; o < rec.moving_obstacle_positions.length; o++) {{
    const ox = toCanvasX(rec.moving_obstacle_positions[o][0]);
    const oy = toCanvasY(rec.moving_obstacle_positions[o][1]);
    ctx.fillStyle = "#f87171";
    ctx.beginPath();
    ctx.arc(ox, oy, 0.35 * SCALE, 0, 2 * Math.PI);
    ctx.fill();
    // Safety bubble
    ctx.strokeStyle = "rgba(248, 113, 113, 0.4)";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(ox, oy, 0.70 * SCALE, 0, 2 * Math.PI);
    ctx.stroke();
  }}

  // Draw Robots & Safety Margins
  for (let r = 0; r < rec.robot_positions.length; r++) {{
    const rx = toCanvasX(rec.robot_positions[r][0]);
    const ry = toCanvasY(rec.robot_positions[r][1]);
    const col = colors[r % colors.length];

    // Safety circle (D_safe / 2 or D_ij)
    ctx.fillStyle = r === 0 ? "rgba(56, 189, 248, 0.15)" : "rgba(74, 222, 128, 0.15)";
    ctx.beginPath();
    ctx.arc(rx, ry, 0.35 * SCALE, 0, 2 * Math.PI);
    ctx.fill();
    ctx.strokeStyle = col;
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // Robot body
    ctx.fillStyle = col;
    ctx.beginPath();
    ctx.arc(rx, ry, 0.25 * SCALE, 0, 2 * Math.PI);
    ctx.fill();

    // Heading Velocity vector
    const vx = rec.safe_velocities[r][0];
    const vy = rec.safe_velocities[r][1];
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.moveTo(rx, ry);
    ctx.lineTo(rx + vx * 25, ry - vy * 25);
    ctx.stroke();
  }}

  // Update Real-Time Telemetry panel
  document.getElementById("telStep").textContent = (rec.step) + " / " + currentRecords.length;
  document.getElementById("telSimTime").textContent = rec.simulation_time.toFixed(2) + " s";
  document.getElementById("telCurrentSep").textContent = rec.min_inter_robot_distance < 1000 ? rec.min_inter_robot_distance.toFixed(3) + " u" : "N/A";
  document.getElementById("telPredDist").textContent = rec.predicted_min_distance < 1000 ? rec.predicted_min_distance.toFixed(3) + " u" : "N/A";
  document.getElementById("telPredTime").textContent = rec.predicted_collision_time < 1000 ? rec.predicted_collision_time.toFixed(2) + " s" : "N/A";
  document.getElementById("telCurrentCBF").textContent = rec.cbf_interventions_count > 0 ? "ACTIVE (Shielding)" : "Nominal";
  document.getElementById("telComplexity").textContent = rec.spectral_complexity.toFixed(5);
  document.getElementById("telBandwidth").textContent = rec.adaptive_bandwidth.toFixed(3);
  document.getElementById("telPower").textContent = (rec.retained_spectral_power * 100).toFixed(2) + "%";
  document.getElementById("telReconError").textContent = (rec.field_reconstruction_error * 100).toFixed(2) + "%";
  document.getElementById("telCosine").textContent = rec.mean_gradient_cosine.toFixed(4);

  document.getElementById("scrubber").value = currentStepIdx;
  document.getElementById("stepCounter").textContent = (currentStepIdx + 1) + "/" + currentRecords.length;
}}

function togglePlay() {{
  isPlaying = !isPlaying;
  document.getElementById("playBtn").textContent = isPlaying ? "Pause" : "Play";
  if (isPlaying) {{
    playInterval = setInterval(() => {{
      if (currentStepIdx < currentRecords.length - 1) {{
        currentStepIdx++;
        render();
      }} else {{
        togglePlay();
      }}
    }}, 60);
  }} else {{
    clearInterval(playInterval);
  }}
}}

function stepForward() {{
  if (currentStepIdx < currentRecords.length - 1) {{
    currentStepIdx++;
    render();
  }}
}}

function stepBackward() {{
  if (currentStepIdx > 0) {{
    currentStepIdx--;
    render();
  }}
}}

function resetPlayback() {{
  if (isPlaying) togglePlay();
  currentStepIdx = 0;
  render();
}}

function onScrub(val) {{
  currentStepIdx = parseInt(val);
  render();
}}

window.onload = init;
</script>
</body>
</html>
"""
    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"Generated standalone interactive dashboard at: {OUTPUT_HTML}")


if __name__ == "__main__":
    generate_html()

