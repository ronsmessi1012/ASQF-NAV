"""
Visualization Script for AQSF-NAV v2.

Generates high-resolution comparative figures using actual recorded experiment data
from results/v2_all_benchmarks.json.
"""

import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
OUTPUT_DIR = Path(__file__).resolve().parent


def plot_v2_experiment_dashboard():
    benchmarks_path = RESULTS_DIR / "v2_all_benchmarks.json"
    if not benchmarks_path.exists():
        print(f"Error: {benchmarks_path} not found. Run run_v2_experiments.py first.")
        return

    with open(benchmarks_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Filter Exp1, Exp2, Exp6
    selected_scenarios = ["Exp1_HeadOn", "Exp2_Crossing", "Exp6_DenseDynamicStress"]
    filtered_data = {sc: [] for sc in selected_scenarios}
    for item in data:
        if item["scenario_name"] in filtered_data:
            filtered_data[item["scenario_name"]].append(item)

    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig.suptitle("AQSF-NAV v2: Predictive Electrostatic Safety Architecture — Benchmark Analysis", fontsize=16, fontweight="bold")

    colors = {
        "A_classical_apf": "#d62728",      # Red
        "B_apf_electrostatic": "#ff7f0e",  # Orange
        "C_predictive_dynamic": "#9467bd",# Purple
        "D_predictive_cbf": "#1f77b4",     # Blue
        "E_aqsf_v2_quantum": "#2ca02c"     # Green
    }

    labels = {
        "A_classical_apf": "A: Classical APF",
        "B_apf_electrostatic": "B: APF + Electrostatic",
        "C_predictive_dynamic": "C: Predictive Dynamic",
        "D_predictive_cbf": "D: Predictive + CBF",
        "E_aqsf_v2_quantum": "E: AQSF-NAV v2 (Quantum + CBF)"
    }

    # Row 0: Trajectories for Exp 1, Exp 2, Exp 6 under Baseline E
    for col_idx, sc_name in enumerate(selected_scenarios):
        ax = axes[0, col_idx]
        sc_runs = filtered_data[sc_name]
        v2_run = next((r for r in sc_runs if r["baseline"] == "E_aqsf_v2_quantum"), None)
        if v2_run and v2_run["step_records"]:
            recs = v2_run["step_records"]
            num_robots = len(recs[0]["robot_positions"])
            for r_i in range(num_robots):
                rx = [step["robot_positions"][r_i][0] for step in recs]
                ry = [step["robot_positions"][r_i][1] for step in recs]
                ax.plot(rx, ry, label=f"Robot {r_i}", linewidth=2.5)
                ax.scatter(rx[0], ry[0], marker="o", s=70, label=f"R{r_i} Start" if col_idx == 0 else None)
                ax.scatter(rx[-1], ry[-1], marker="*", s=120, label=f"R{r_i} Goal" if col_idx == 0 else None)

            # Draw moving obstacles if any
            if recs[0]["moving_obstacle_positions"]:
                num_obs = len(recs[0]["moving_obstacle_positions"])
                for o_i in range(num_obs):
                    ox = [step["moving_obstacle_positions"][o_i][0] for step in recs]
                    oy = [step["moving_obstacle_positions"][o_i][1] for step in recs]
                    ax.plot(ox, oy, ":", color="gray", linewidth=1.5, label=f"Obstacle {o_i}" if col_idx == 2 else None)

        ax.set_title(f"{sc_name} — AQSF-NAV v2 Trajectories", fontsize=11, fontweight="bold")
        ax.set_xlim([0, 16])
        ax.set_ylim([0, 16])
        ax.set_xlabel("X (world units)")
        ax.set_ylabel("Y (world units)")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=8, loc="upper right")

    # Row 1: Inter-robot clearance over time across baselines for Exp 1, Exp 2, Exp 6
    for col_idx, sc_name in enumerate(selected_scenarios):
        ax = axes[1, col_idx]
        sc_runs = filtered_data[sc_name]
        for run in sc_runs:
            b = run["baseline"]
            recs = run["step_records"]
            if recs and len(recs[0]["robot_positions"]) > 1:
                times = [s["simulation_time"] for s in recs]
                dists = [s["min_inter_robot_distance"] for s in recs]
                ax.plot(times, dists, color=colors.get(b, "black"), label=labels.get(b, b), linewidth=1.8)

        # Draw safety distance line D_safe = 0.70
        ax.axhline(0.70, color="red", linestyle="--", linewidth=1.5, label="Safety Distance D_ij (0.70)")
        ax.axhline(0.50, color="darkred", linestyle=":", linewidth=1.5, label="Physical Collision (0.50)")
        ax.set_title(f"{sc_name} — Inter-Robot Clearance", fontsize=11, fontweight="bold")
        ax.set_xlabel("Simulation Time (s)")
        ax.set_ylabel("Separation Distance (units)")
        ax.set_ylim([0, 6.0])
        ax.grid(True, linestyle="--", alpha=0.5)
        if col_idx == 0:
            ax.legend(fontsize=8, loc="upper right")

    # Row 2: Quantum Telemetry (Exp 6 Dense Stress Test)
    exp6_v2 = next((r for r in filtered_data["Exp6_DenseDynamicStress"] if r["baseline"] == "E_aqsf_v2_quantum"), None)
    if exp6_v2 and exp6_v2["step_records"]:
        recs = exp6_v2["step_records"]
        times = [s["simulation_time"] for s in recs]
        comp = [s["spectral_complexity"] for s in recs]
        band = [s["adaptive_bandwidth"] for s in recs]
        pwr = [s["retained_spectral_power"] for s in recs]
        recon = [s["field_reconstruction_error"] for s in recs]
        cbf_interv = [s["cbf_interventions_count"] for s in recs]

        # Subplot 2,0: Spectral Complexity & Bandwidth
        ax20 = axes[2, 0]
        ax20_twin = ax20.twinx()
        l1 = ax20.plot(times, comp, color="teal", label="Complexity C(t)", linewidth=2)
        l2 = ax20_twin.plot(times, band, color="navy", linestyle="--", label="Bandwidth B(t)", linewidth=2)
        ax20.set_title("Exp 6: Quantum Spectral Complexity & Bandwidth", fontsize=11, fontweight="bold")
        ax20.set_xlabel("Simulation Time (s)")
        ax20.set_ylabel("Spectral Complexity C", color="teal")
        ax20_twin.set_ylabel("Adaptive Bandwidth B", color="navy")
        ax20.grid(True, linestyle="--", alpha=0.5)
        lines = l1 + l2
        ax20.legend(lines, [l.get_label() for l in lines], fontsize=8, loc="center right")

        # Subplot 2,1: Retained Spectral Power & Field Reconstruction Error
        ax21 = axes[2, 1]
        ax21_twin = ax21.twinx()
        l3 = ax21.plot(times, pwr, color="green", label="Retained Power", linewidth=2)
        l4 = ax21_twin.plot(times, recon, color="crimson", linestyle="--", label="Recon Error", linewidth=2)
        ax21.set_title("Exp 6: Spectral Power & Reconstruction Fidelity", fontsize=11, fontweight="bold")
        ax21.set_xlabel("Simulation Time (s)")
        ax21.set_ylabel("Retained Power Fraction", color="green")
        ax21_twin.set_ylabel("Field L2 Error", color="crimson")
        ax21.grid(True, linestyle="--", alpha=0.5)
        lines2 = l3 + l4
        ax21.legend(lines2, [l.get_label() for l in lines2], fontsize=8, loc="center right")

        # Subplot 2,2: CBF Interventions over time
        ax22 = axes[2, 2]
        ax22.step(times, np.cumsum(cbf_interv), color="purple", linewidth=2, label="Cumulative Interventions")
        ax22.set_title("Exp 6: Cumulative CBF Safety Interventions", fontsize=11, fontweight="bold")
        ax22.set_xlabel("Simulation Time (s)")
        ax22.set_ylabel("Total Interventions")
        ax22.grid(True, linestyle="--", alpha=0.5)
        ax22.legend(fontsize=8, loc="lower right")

    plt.tight_layout()
    out_img = OUTPUT_DIR / "v2_experiment_results.png"
    plt.savefig(out_img, dpi=200)
    print(f"Saved dashboard figure to: {out_img}")
    plt.close()


if __name__ == "__main__":
    plot_v2_experiment_dashboard()

