"""
Master Experiment Runner for AQSF-NAV v2.

Executes all 7 required scientific experiments across the 5 comparison baselines:
- Experiment 1: Two Robots Head-On Approach
- Experiment 2: Two Robots Perpendicular Crossing Trajectories
- Experiment 3: Moving Obstacle Crossing Single Robot Path
- Experiment 4: Multiple Moving Obstacles Dynamic Environment
- Experiment 5: Multi-Agent Warehouse Navigation (Robots + Static Shelves + Moving Obstacles)
- Experiment 6: Dense Dynamic Environment Stress Test (Multiple Robots + Multiple Obstacles)
- Experiment 7: QFT Filtering Sensitivity Ablation (Classical vs Unfiltered vs Filtered vs CBF Shield)

Saves step-by-step telemetry to CSV and full structured summaries to JSON in results/.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import List
import numpy as np

from navigation_v2.config import AQSFConfigV2
from navigation_v2.dynamic_entities import MovingObstacle
from navigation_v2.simulation import run_v2_simulation, ScenarioSummary
from navigation_v2.metrics_logger import (
    export_step_telemetry_to_csv,
    export_summary_to_json,
    format_comparison_markdown_table,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def get_experiment_scenarios(config: AQSFConfigV2):
    """
    Define the standardized, reproducible scenarios for Experiments 1 through 7.
    """
    scenarios = {}

    # -------------------------------------------------------------------------
    # Exp 1: Two Robots Head-On Approach
    # -------------------------------------------------------------------------
    scenarios["Exp1_HeadOn"] = {
        "robot_starts": np.array([[2.0, 7.5], [13.0, 7.5]]),
        "robot_goals": np.array([[13.0, 7.5], [2.0, 7.5]]),
        "static_obstacles": [],
        "moving_obstacles": [],
        "max_steps": 120
    }

    # -------------------------------------------------------------------------
    # Exp 2: Crossing Trajectories (Perpendicular Intersection at (7.5, 7.5))
    # -------------------------------------------------------------------------
    scenarios["Exp2_Crossing"] = {
        "robot_starts": np.array([[2.0, 7.5], [7.5, 2.0]]),
        "robot_goals": np.array([[13.0, 7.5], [7.5, 13.0]]),
        "static_obstacles": [],
        "moving_obstacles": [],
        "max_steps": 120
    }

    # -------------------------------------------------------------------------
    # Exp 3: Moving Obstacle Crossing Single Robot Path
    # -------------------------------------------------------------------------
    obs_crossing = MovingObstacle(
        id="cross_obs_1",
        position=np.array([7.5, 3.0]),
        velocity=np.array([0.0, 0.45]),
        radius=0.35,
        movement_model="constant_velocity",
        bounds=(0.0, 15.0, 0.0, 15.0)
    )
    scenarios["Exp3_MovingObsCrossing"] = {
        "robot_starts": np.array([[2.0, 7.5]]),
        "robot_goals": np.array([[13.0, 7.5]]),
        "static_obstacles": [],
        "moving_obstacles": [obs_crossing],
        "max_steps": 120
    }

    # -------------------------------------------------------------------------
    # Exp 4: Multiple Moving Obstacles Dynamic Environment
    # -------------------------------------------------------------------------
    obs_list_exp4 = [
        MovingObstacle(
            id="dyn_patrol_1",
            position=np.array([5.0, 3.0]),
            velocity=np.array([0.0, 0.40]),
            radius=0.30,
            movement_model="constant_velocity",
            bounds=(0.0, 15.0, 2.0, 13.0)
        ),
        MovingObstacle(
            id="dyn_patrol_2",
            position=np.array([9.0, 12.0]),
            velocity=np.array([0.0, -0.40]),
            radius=0.30,
            movement_model="constant_velocity",
            bounds=(0.0, 15.0, 2.0, 13.0)
        ),
        MovingObstacle(
            id="dyn_patrol_3",
            position=np.array([3.0, 7.5]),
            velocity=np.array([0.35, 0.0]),
            radius=0.25,
            movement_model="constant_velocity",
            bounds=(2.0, 13.0, 0.0, 15.0)
        )
    ]
    scenarios["Exp4_MultiMovingObstacles"] = {
        "robot_starts": np.array([[2.0, 2.0]]),
        "robot_goals": np.array([[13.0, 13.0]]),
        "static_obstacles": [(6.0, 6.0), (9.0, 9.0)],
        "moving_obstacles": obs_list_exp4,
        "max_steps": 150
    }

    # -------------------------------------------------------------------------
    # Exp 5: Multi-Agent Warehouse Navigation
    # Two robots, static shelves/columns, and dynamic warehouse carts
    # -------------------------------------------------------------------------
    warehouse_shelves = [
        (4.0, 5.0), (4.0, 6.0), (4.0, 7.0),
        (11.0, 8.0), (11.0, 9.0), (11.0, 10.0),
        (7.5, 4.0), (7.5, 11.0)
    ]
    warehouse_cart = MovingObstacle(
        id="cart_1",
        position=np.array([7.5, 2.0]),
        velocity=np.array([0.0, 0.35]),
        radius=0.35,
        movement_model="constant_velocity",
        bounds=(0.0, 15.0, 1.0, 14.0)
    )
    scenarios["Exp5_WarehouseNavigation"] = {
        "robot_starts": np.array([[2.0, 2.0], [13.0, 13.0]]),
        "robot_goals": np.array([[13.0, 13.0], [2.0, 2.0]]),
        "static_obstacles": warehouse_shelves,
        "moving_obstacles": [warehouse_cart],
        "max_steps": 160
    }

    # -------------------------------------------------------------------------
    # Exp 6: Dense Dynamic Environment Stress Test
    # 3 Robots + 3 Moving Obstacles + Static Obstacles
    # -------------------------------------------------------------------------
    scenarios["Exp6_DenseDynamicStress"] = {
        "robot_starts": np.array([
            [2.0, 2.0],
            [13.0, 2.0],
            [7.5, 13.0]
        ]),
        "robot_goals": np.array([
            [13.0, 13.0],
            [2.0, 13.0],
            [7.5, 2.0]
        ]),
        "static_obstacles": [(5.0, 8.0), (10.0, 8.0)],
        "moving_obstacles": [
            MovingObstacle(
                id="cart_dense_1",
                position=np.array([4.0, 7.5]),
                velocity=np.array([0.3, 0.0]),
                radius=0.30,
                bounds=(2.0, 13.0, 0.0, 15.0)
            ),
            MovingObstacle(
                id="cart_dense_2",
                position=np.array([11.0, 7.5]),
                velocity=np.array([-0.3, 0.0]),
                radius=0.30,
                bounds=(2.0, 13.0, 0.0, 15.0)
            )
        ],
        "max_steps": 160
    }

    return scenarios


def run_all_experiments():
    print("=" * 80)
    print("     AQSF-NAV v2: COMPREHENSIVE BENCHMARK AND EXPERIMENT SUITE")
    print("=" * 80)

    config = AQSFConfigV2()
    scenarios = get_experiment_scenarios(config)

    baselines = [
        "A_classical_apf",
        "B_apf_electrostatic",
        "C_predictive_dynamic",
        "D_predictive_cbf",
        "E_aqsf_v2_quantum"
    ]

    all_summaries: List[ScenarioSummary] = []

    # -------------------------------------------------------------------------
    # Run Experiments 1 through 6 across all baselines
    # -------------------------------------------------------------------------
    for sc_name, sc_data in scenarios.items():
        print(f"\n>>> Running Scenario: {sc_name} <<<")
        for b in baselines:
            print(f"  [Baseline: {b:22s}] ... ", end="", flush=True)
            t0 = time.perf_counter()
            summary = run_v2_simulation(
                scenario_name=sc_name,
                baseline=b,
                robot_starts=sc_data["robot_starts"],
                robot_goals=sc_data["robot_goals"],
                static_obstacles=sc_data["static_obstacles"],
                moving_obstacles=sc_data["moving_obstacles"],
                config=config,
                max_steps=sc_data["max_steps"],
                record_step_data=True
            )
            elapsed = time.perf_counter() - t0
            status_str = "PASS" if summary.success else "FAIL"
            print(f"{status_str:4s} | Steps: {summary.total_steps:3d} | MinRobotDist: {summary.min_inter_robot_distance:6.3f} | MinObsDist: {summary.min_obstacle_distance:6.3f} | Collisions: {summary.total_collisions} | Time: {elapsed:5.2f}s")
            all_summaries.append(summary)

            # Export step telemetry for this run
            csv_path = RESULTS_DIR / f"telemetry_{sc_name}_{b}.csv"
            export_step_telemetry_to_csv(summary, csv_path)

    # -------------------------------------------------------------------------
    # Experiment 7: QFT Filtering Sensitivity Ablation
    # -------------------------------------------------------------------------
    print(f"\n>>> Running Scenario: Exp7_QFT_Filtering_Sensitivity <<<")
    # Base scenario: Multi-agent warehouse crossing
    sc_data_exp7 = scenarios["Exp5_WarehouseNavigation"]
    exp7_variants = [
        ("Classical_APF_CBF", "D_predictive_cbf", False),
        ("Quantum_Unfiltered_CBF", "E_aqsf_v2_quantum", True),
        ("Quantum_Filtered_NoCBF", "C_predictive_dynamic", False),  # QFT filtered without CBF
        ("AQSF_NAV_v2_Full", "E_aqsf_v2_quantum", False),           # QFT filtered WITH CBF
    ]

    for variant_name, baseline_mode, unfiltered_flag in exp7_variants:
        print(f"  [Exp7 Variant: {variant_name:25s}] ... ", end="", flush=True)
        t0 = time.perf_counter()
        summary = run_v2_simulation(
            scenario_name="Exp7_QFT_Sensitivity",
            baseline=baseline_mode,
            robot_starts=sc_data_exp7["robot_starts"],
            robot_goals=sc_data_exp7["robot_goals"],
            static_obstacles=sc_data_exp7["static_obstacles"],
            moving_obstacles=sc_data_exp7["moving_obstacles"],
            config=config,
            max_steps=sc_data_exp7["max_steps"],
            qft_unfiltered=unfiltered_flag,
            record_step_data=True
        )
        summary.scenario_name = f"Exp7_{variant_name}"
        elapsed = time.perf_counter() - t0
        status_str = "PASS" if summary.success else "FAIL"
        print(f"{status_str:4s} | Steps: {summary.total_steps:3d} | MinRobotDist: {summary.min_inter_robot_distance:6.3f} | CBF Interventions: {summary.total_cbf_interventions:2d} | ReconError: {summary.mean_reconstruction_error:6.4f} | Time: {elapsed:5.2f}s")
        all_summaries.append(summary)
        csv_path = RESULTS_DIR / f"telemetry_Exp7_{variant_name}.csv"
        export_step_telemetry_to_csv(summary, csv_path)

    # -------------------------------------------------------------------------
    # Export full benchmark JSON
    # -------------------------------------------------------------------------
    json_path = RESULTS_DIR / "v2_all_benchmarks.json"
    export_summary_to_json(all_summaries, json_path)
    print(f"\nSaved complete benchmark JSON to: {json_path}")

    # -------------------------------------------------------------------------
    # Print Markdown Comparison Table
    # -------------------------------------------------------------------------
    table_md = format_comparison_markdown_table(all_summaries)
    print("\n" + "=" * 80)
    print("                    EXPERIMENTAL COMPARISON TABLE")
    print("=" * 80)
    print(table_md)
    print("=" * 80)

    # Also save table to results/summary_table.md
    with open(RESULTS_DIR / "summary_table.md", "w") as f:
        f.write(table_md)

    return all_summaries


if __name__ == "__main__":
    run_all_experiments()

