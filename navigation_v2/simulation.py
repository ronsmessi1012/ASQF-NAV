"""
Simulation Engine and Benchmark Orchestration for AQSF-NAV v2.

Supports 5 formal baseline configurations:
- Baseline A: "A_classical_apf" (v1 APF + agent potential, no electrostatic, no CBF)
- Baseline B: "B_apf_electrostatic" (APF + Electrostatic dynamic field, no predictive, no CBF)
- Baseline C: "C_predictive_dynamic" (APF + Electrostatic + Predictive steering bias, no CBF)
- Baseline D: "D_predictive_cbf" (Classical full stack: APF + Electrostatic + Predictive + CBF Shield)
- Baseline E: "E_aqsf_v2_quantum" (Full AQSF-NAV v2: Combined Field + 8-Qubit 2D QFT + Adaptive Filter + IQFT + CBF Shield)

Collects complete navigation, safety, quantum, and computational metrics.
Provides structured step-by-step and summary data exports to JSON and CSV.
"""

import copy
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from environment.field import generate_field
from navigation.controller import (
    bilinear_interpolate,
    sample_field_gradient,
    reached_goal,
)
from navigation_v2.config import AQSFConfigV2
from navigation_v2.dynamic_entities import MovingObstacle
from navigation_v2.electrostatic_potential import compute_dynamic_field_grid
from navigation_v2.predictive_collision import (
    CollisionPrediction,
    analyze_all_predictive_collisions,
)
from navigation_v2.combined_field import generate_combined_field
from navigation_v2.cbf_safety_shield import (
    CBFSafetyShield,
    CBFSolverResult,
)
from navigation_v2.quantum_pipeline import process_field_through_qft
from navigation_v2.controller_v2 import compute_desired_velocity


@dataclass
class SimulationStepRecord:
    step: int
    simulation_time: float
    robot_positions: List[List[float]]
    robot_velocities: List[List[float]]
    desired_velocities: List[List[float]]
    safe_velocities: List[List[float]]
    goal_positions: List[List[float]]
    moving_obstacle_positions: List[List[float]]
    moving_obstacle_velocities: List[List[float]]
    min_inter_robot_distance: float
    min_obstacle_distance: float
    predicted_min_distance: float
    predicted_collision_time: float
    collision_warnings_count: int
    cbf_interventions_count: int
    cbf_infeasible_count: int
    safety_violations_count: int
    spectral_complexity: float = 0.0
    normalized_complexity: float = 0.0
    adaptive_bandwidth: float = 1.0
    retained_spectral_power: float = 1.0
    field_reconstruction_error: float = 0.0
    mean_gradient_cosine: float = 1.0

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "simulation_time": self.simulation_time,
            "robot_positions": self.robot_positions,
            "robot_velocities": self.robot_velocities,
            "desired_velocities": self.desired_velocities,
            "safe_velocities": self.safe_velocities,
            "goal_positions": self.goal_positions,
            "moving_obstacle_positions": self.moving_obstacle_positions,
            "moving_obstacle_velocities": self.moving_obstacle_velocities,
            "min_inter_robot_distance": self.min_inter_robot_distance,
            "min_obstacle_distance": self.min_obstacle_distance,
            "predicted_min_distance": self.predicted_min_distance,
            "predicted_collision_time": self.predicted_collision_time,
            "collision_warnings_count": self.collision_warnings_count,
            "cbf_interventions_count": self.cbf_interventions_count,
            "cbf_infeasible_count": self.cbf_infeasible_count,
            "safety_violations_count": self.safety_violations_count,
            "spectral_complexity": self.spectral_complexity,
            "normalized_complexity": self.normalized_complexity,
            "adaptive_bandwidth": self.adaptive_bandwidth,
            "retained_spectral_power": self.retained_spectral_power,
            "field_reconstruction_error": self.field_reconstruction_error,
            "mean_gradient_cosine": self.mean_gradient_cosine,
        }


@dataclass
class ScenarioSummary:
    scenario_name: str
    baseline: str
    success: bool
    all_goals_reached: bool
    total_steps: int
    total_simulation_time: float
    total_path_lengths: List[float]
    average_path_length: float
    final_goal_errors: List[float]
    min_inter_robot_distance: float
    min_obstacle_distance: float
    total_collisions: int
    total_safety_violations: int
    total_predictive_warnings: int
    total_cbf_interventions: int
    total_emergency_stops: int
    total_infeasible_constraints: int
    mean_spectral_complexity: float
    mean_adaptive_bandwidth: float
    mean_retained_spectral_power: float
    mean_reconstruction_error: float
    mean_gradient_cosine_similarity: float
    computation_time_total: float
    computation_time_per_step: float
    step_records: List[SimulationStepRecord] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = copy.copy(self.__dict__)
        d["step_records"] = [sr.to_dict() for sr in self.step_records]
        return d


def run_v2_simulation(
    scenario_name: str,
    baseline: str,
    robot_starts: np.ndarray,
    robot_goals: np.ndarray,
    static_obstacles: List[Tuple[float, float]],
    moving_obstacles: Optional[List[MovingObstacle]] = None,
    config: Optional[AQSFConfigV2] = None,
    max_steps: Optional[int] = None,
    qft_unfiltered: bool = False,
    record_step_data: bool = True
) -> ScenarioSummary:
    """
    Execute a full multi-agent simulation run for a given scenario and baseline.
    """
    if config is None:
        config = AQSFConfigV2()
    if max_steps is None:
        max_steps = config.max_steps

    t_wall_start = time.perf_counter()

    num_robots = len(robot_starts)
    positions = np.array(robot_starts, dtype=float).copy()
    goals = np.array(robot_goals, dtype=float).copy()
    velocities = np.zeros((num_robots, 2), dtype=float)

    # Initialize moving obstacles
    active_obstacles = [copy.deepcopy(obs) for obs in (moving_obstacles or [])]

    shield = CBFSafetyShield(config)
    d_safe_robot = config.compute_robot_safety_distance()

    # Telemetry accumulators
    step_records: List[SimulationStepRecord] = []
    path_lengths = [0.0] * num_robots
    prev_positions = positions.copy()

    total_collisions = 0
    total_safety_violations = 0
    total_predictive_warnings = 0
    total_cbf_interventions = 0
    total_emergency_stops = 0
    total_infeasible_constraints = 0

    min_robot_dist_observed = float("inf")
    min_obstacle_dist_observed = float("inf")

    spectral_complexities = []
    adaptive_bandwidths = []
    retained_powers = []
    reconstruction_errors = []
    gradient_cosines = []

    dt = config.step_size

    # Baseline configuration flags
    use_v1_apf = (baseline == "A_classical_apf")
    use_electrostatic = (baseline in ["B_apf_electrostatic", "C_predictive_dynamic", "D_predictive_cbf", "E_aqsf_v2_quantum"])
    enable_predictive = (baseline in ["C_predictive_dynamic", "D_predictive_cbf", "E_aqsf_v2_quantum"])
    enable_cbf = (baseline in ["D_predictive_cbf", "E_aqsf_v2_quantum"])
    use_quantum = (baseline == "E_aqsf_v2_quantum")

    completed_step = 0

    for step in range(max_steps):
        completed_step = step + 1
        current_sim_time = step * dt

        # 1. Advance moving obstacles
        for obs in active_obstacles:
            obs.step(dt, current_sim_time)

        # 2. Check current separations and safety violations
        step_min_robot_dist = float("inf")
        if num_robots > 1:
            for i in range(num_robots):
                for j in range(i + 1, num_robots):
                    d_ij = float(np.linalg.norm(positions[i] - positions[j]))
                    if d_ij < step_min_robot_dist:
                        step_min_robot_dist = d_ij
                    if d_ij < d_safe_robot:
                        total_safety_violations += 1
                        # Physical collision occurs if distance < R_i + R_j (0.50)
                        if d_ij < (config.robot_radius * 2.0):
                            total_collisions += 1

            if step_min_robot_dist < min_robot_dist_observed:
                min_robot_dist_observed = step_min_robot_dist

        step_min_obs_dist = float("inf")
        if active_obstacles:
            for i in range(num_robots):
                for obs in active_obstacles:
                    d_io = float(np.linalg.norm(positions[i] - obs.position))
                    safe_d_io = config.compute_safety_distance(obs.radius)
                    if d_io < step_min_obs_dist:
                        step_min_obs_dist = d_io
                    if d_io < safe_d_io:
                        total_safety_violations += 1
                        if d_io < (config.robot_radius + obs.radius):
                            total_collisions += 1

            if step_min_obs_dist < min_obstacle_dist_observed:
                min_obstacle_dist_observed = step_min_obs_dist

        # Arrays for next step state
        new_positions = positions.copy()
        new_velocities = velocities.copy()
        step_desired_vels = []
        step_safe_vels = []

        step_cbf_interventions = 0
        step_infeasible = 0
        step_warnings = 0
        step_pred_min_dist = float("inf")
        step_pred_t = float("inf")

        step_complexity = 0.0
        step_norm_complexity = 0.0
        step_bandwidth = 1.0
        step_retained_power = 1.0
        step_recon_error = 0.0
        step_mean_cosine = 1.0

        for i in range(num_robots):
            p_i = positions[i]
            g_i = goals[i]

            # -------------------------------------------------------------
            # Generate Navigation Field
            # -------------------------------------------------------------
            if use_v1_apf:
                # Original v1 APF field generator
                X, Y, field = generate_field(
                    agents=positions,
                    goals=goals,
                    obstacles=static_obstacles,
                    current_agent=i
                )
            else:
                # v2 combined field
                fdata = generate_combined_field(
                    goal=g_i,
                    static_obstacles=static_obstacles,
                    robot_positions=positions,
                    moving_obstacles=active_obstacles,
                    current_robot_idx=i,
                    config=config,
                    include_dynamic=use_electrostatic
                )
                field = fdata["F_total"]

            # -------------------------------------------------------------
            # Quantum Spectral Processing Pipeline (if active)
            # -------------------------------------------------------------
            if use_quantum:
                q_field, q_telemetry = process_field_through_qft(
                    field=field,
                    config=config,
                    apply_filter=(not qft_unfiltered),
                    use_qiskit_circuit=True
                )
                nav_field = q_field
                step_complexity = q_telemetry["spectral_complexity"]
                step_norm_complexity = q_telemetry["normalized_complexity"]
                step_bandwidth = q_telemetry["adaptive_bandwidth"]
                step_retained_power = q_telemetry["retained_power_fraction"]
                step_recon_error = q_telemetry["field_reconstruction_error"]
                step_mean_cosine = q_telemetry["mean_gradient_cosine_similarity"]

                spectral_complexities.append(step_complexity)
                adaptive_bandwidths.append(step_bandwidth)
                retained_powers.append(step_retained_power)
                reconstruction_errors.append(step_recon_error)
                gradient_cosines.append(step_mean_cosine)
            else:
                nav_field = field

            # -------------------------------------------------------------
            # Predictive Collision Detection
            # -------------------------------------------------------------
            predictions = analyze_all_predictive_collisions(
                robot_idx=i,
                robot_positions=positions,
                robot_velocities=velocities,
                moving_obstacles=active_obstacles,
                config=config
            )

            pred_steering_bias = np.zeros(2, dtype=float)
            goal_vec = g_i - p_i
            g_dist = np.linalg.norm(goal_vec)
            heading = (goal_vec / g_dist) if g_dist > 1e-6 else np.array([1.0, 0.0])
            right_lateral = np.array([heading[1], -heading[0]])

            for pred in predictions:
                if pred.time_to_closest_approach < step_pred_t:
                    step_pred_t = pred.time_to_closest_approach
                if pred.predicted_minimum_distance < step_pred_min_dist:
                    step_pred_min_dist = pred.predicted_minimum_distance

                if pred.warning_status in ["warning", "critical"]:
                    step_warnings += 1
                    total_predictive_warnings += 1

                if enable_predictive and pred.collision_risk > config.warning_risk_threshold:
                    pred_steering_bias += pred.collision_risk * 1.2 * right_lateral

            # -------------------------------------------------------------
            # Compute Desired Velocity
            # -------------------------------------------------------------
            v_des, _, _ = compute_desired_velocity(
                field=nav_field,
                position=p_i,
                goal=g_i,
                max_speed=config.max_speed,
                predictive_steering=pred_steering_bias if enable_predictive else None
            )
            step_desired_vels.append(v_des.tolist())

            # -------------------------------------------------------------
            # CBF Safety Shield
            # -------------------------------------------------------------
            if enable_cbf:
                cbf_res = shield.filter_velocity(
                    robot_idx=i,
                    desired_velocity=v_des,
                    robot_positions=positions,
                    robot_velocities=velocities,
                    moving_obstacles=active_obstacles
                )
                v_safe = cbf_res.safe_velocity
                if cbf_res.intervened:
                    step_cbf_interventions += 1
                    total_cbf_interventions += 1
                if cbf_res.status == "infeasible_fallback":
                    step_infeasible += 1
                    total_infeasible_constraints += 1
                elif cbf_res.status == "emergency_stop":
                    total_emergency_stops += 1
            else:
                v_safe = v_des

            step_safe_vels.append(v_safe.tolist())

            # Position integration: p_{t+1} = p_t + dt * v_safe
            new_p = p_i + dt * v_safe
            new_positions[i] = new_p
            new_velocities[i] = v_safe

            # Accumulate path length
            path_lengths[i] += float(np.linalg.norm(new_p - p_i))

        # Update world state
        positions = new_positions
        velocities = new_velocities

        # Record step telemetry if requested
        if record_step_data:
            step_rec = SimulationStepRecord(
                step=step + 1,
                simulation_time=current_sim_time,
                robot_positions=positions.tolist(),
                robot_velocities=velocities.tolist(),
                desired_velocities=step_desired_vels,
                safe_velocities=step_safe_vels,
                goal_positions=goals.tolist(),
                moving_obstacle_positions=[obs.position.tolist() for obs in active_obstacles],
                moving_obstacle_velocities=[obs.velocity.tolist() for obs in active_obstacles],
                min_inter_robot_distance=step_min_robot_dist,
                min_obstacle_distance=step_min_obs_dist,
                predicted_min_distance=step_pred_min_dist,
                predicted_collision_time=step_pred_t,
                collision_warnings_count=step_warnings,
                cbf_interventions_count=step_cbf_interventions,
                cbf_infeasible_count=step_infeasible,
                safety_violations_count=total_safety_violations,
                spectral_complexity=step_complexity,
                normalized_complexity=step_norm_complexity,
                adaptive_bandwidth=step_bandwidth,
                retained_spectral_power=step_retained_power,
                field_reconstruction_error=step_recon_error,
                mean_gradient_cosine=step_mean_cosine,
            )
            step_records.append(step_rec)

        # Check goal termination
        all_goals_done = all(
            reached_goal(positions[i], goals[i]) for i in range(num_robots)
        )
        if all_goals_done:
            break

    t_wall_end = time.perf_counter()
    total_computation_time = t_wall_end - t_wall_start

    goal_errors = [
        float(np.linalg.norm(positions[i] - goals[i])) for i in range(num_robots)
    ]
    all_goals_reached = all(err <= config.goal_tolerance for err in goal_errors)

    # Mission success: reached goals AND zero physical collisions
    mission_success = all_goals_reached and (total_collisions == 0)

    summary = ScenarioSummary(
        scenario_name=scenario_name,
        baseline=baseline,
        success=mission_success,
        all_goals_reached=all_goals_reached,
        total_steps=completed_step,
        total_simulation_time=completed_step * dt,
        total_path_lengths=path_lengths,
        average_path_length=float(np.mean(path_lengths)),
        final_goal_errors=goal_errors,
        min_inter_robot_distance=min_robot_dist_observed,
        min_obstacle_distance=min_obstacle_dist_observed,
        total_collisions=total_collisions,
        total_safety_violations=total_safety_violations,
        total_predictive_warnings=total_predictive_warnings,
        total_cbf_interventions=total_cbf_interventions,
        total_emergency_stops=total_emergency_stops,
        total_infeasible_constraints=total_infeasible_constraints,
        mean_spectral_complexity=float(np.mean(spectral_complexities)) if spectral_complexities else 0.0,
        mean_adaptive_bandwidth=float(np.mean(adaptive_bandwidths)) if adaptive_bandwidths else 1.0,
        mean_retained_spectral_power=float(np.mean(retained_powers)) if retained_powers else 1.0,
        mean_reconstruction_error=float(np.mean(reconstruction_errors)) if reconstruction_errors else 0.0,
        mean_gradient_cosine_similarity=float(np.mean(gradient_cosines)) if gradient_cosines else 1.0,
        computation_time_total=total_computation_time,
        computation_time_per_step=total_computation_time / max(completed_step, 1),
        step_records=step_records
    )

    return summary

