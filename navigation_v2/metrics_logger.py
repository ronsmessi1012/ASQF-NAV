"""
Structured Telemetry and Benchmark Data Exporter for AQSF-NAV v2.

Exports:
1. Complete simulation trajectories and step-by-step telemetry to CSV.
2. Comprehensive benchmark summaries and ablated metrics to JSON.
3. Formatted comparison tables for scientific analysis and documentation.
"""

import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np

from navigation_v2.simulation import ScenarioSummary, SimulationStepRecord


def export_step_telemetry_to_csv(
    summary: ScenarioSummary,
    output_path: Union[str, Path]
) -> Path:
    """
    Export all timestep telemetry to a structured CSV file.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "step",
        "simulation_time",
        "robot_0_x", "robot_0_y",
        "robot_0_vx", "robot_0_vy",
        "robot_1_x", "robot_1_y",
        "robot_1_vx", "robot_1_vy",
        "min_inter_robot_distance",
        "min_obstacle_distance",
        "predicted_min_distance",
        "predicted_collision_time",
        "collision_warnings_count",
        "cbf_interventions_count",
        "safety_violations_count",
        "spectral_complexity",
        "normalized_complexity",
        "adaptive_bandwidth",
        "retained_spectral_power",
        "field_reconstruction_error",
        "mean_gradient_cosine"
    ]

    with open(path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)

        for rec in summary.step_records:
            r0_p = rec.robot_positions[0] if len(rec.robot_positions) > 0 else [0.0, 0.0]
            r0_v = rec.robot_velocities[0] if len(rec.robot_velocities) > 0 else [0.0, 0.0]
            r1_p = rec.robot_positions[1] if len(rec.robot_positions) > 1 else [0.0, 0.0]
            r1_v = rec.robot_velocities[1] if len(rec.robot_velocities) > 1 else [0.0, 0.0]

            row = [
                rec.step,
                f"{rec.simulation_time:.3f}",
                f"{r0_p[0]:.4f}", f"{r0_p[1]:.4f}",
                f"{r0_v[0]:.4f}", f"{r0_v[1]:.4f}",
                f"{r1_p[0]:.4f}", f"{r1_p[1]:.4f}",
                f"{r1_v[0]:.4f}", f"{r1_v[1]:.4f}",
                f"{rec.min_inter_robot_distance:.4f}",
                f"{rec.min_obstacle_distance:.4f}",
                f"{rec.predicted_min_distance:.4f}",
                f"{rec.predicted_collision_time:.4f}",
                rec.collision_warnings_count,
                rec.cbf_interventions_count,
                rec.safety_violations_count,
                f"{rec.spectral_complexity:.6f}",
                f"{rec.normalized_complexity:.6f}",
                f"{rec.adaptive_bandwidth:.4f}",
                f"{rec.retained_spectral_power:.6f}",
                f"{rec.field_reconstruction_error:.6f}",
                f"{rec.mean_gradient_cosine:.6f}",
            ]
            writer.writerow(row)

    return path


def export_summary_to_json(
    summaries: Union[ScenarioSummary, List[ScenarioSummary]],
    output_path: Union[str, Path]
) -> Path:
    """
    Export benchmark summary dictionary to a formatted JSON file.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(summaries, list):
        data = [s.to_dict() for s in summaries]
    else:
        data = summaries.to_dict()

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return path


def format_comparison_markdown_table(summaries: List[ScenarioSummary]) -> str:
    """
    Generate a markdown comparison table comparing all tested baselines.
    """
    headers = [
        "Scenario",
        "Baseline",
        "Success",
        "Steps",
        "Min Robot Dist",
        "Min Obs Dist",
        "Collisions",
        "CBF Interv.",
        "Recon Error",
        "Time (s)"
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join([":---"] + [":---:"] * (len(headers) - 1)) + " |"
    ]

    for s in summaries:
        min_r = f"{s.min_inter_robot_distance:.3f}" if s.min_inter_robot_distance < 1e5 else "N/A"
        min_o = f"{s.min_obstacle_distance:.3f}" if s.min_obstacle_distance < 1e5 else "N/A"
        recon = f"{s.mean_reconstruction_error:.4f}" if s.mean_reconstruction_error > 0 else "0.0000"
        time_s = f"{s.computation_time_total:.3f}"

        row = [
            s.scenario_name,
            s.baseline,
            "PASS" if s.success else "FAIL",
            str(s.total_steps),
            min_r,
            min_o,
            str(s.total_collisions),
            str(s.total_cbf_interventions),
            recon,
            time_s
        ]
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)

