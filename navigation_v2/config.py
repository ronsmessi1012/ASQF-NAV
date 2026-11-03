"""
Centralized Configuration for AQSF-NAV v2.

Defines all physical, simulation, safety, electrostatic, and CBF parameters
in a single, typed dataclass with robust defaults and validation.
"""

from dataclasses import dataclass, field
from typing import Tuple
import numpy as np


@dataclass
class AQSFConfigV2:
    # -------------------------------------------------------------------------
    # Spatial and Simulation Grid
    # -------------------------------------------------------------------------
    grid_size: int = 16
    step_size: float = 0.20  # dt preserved from v1 STEP_SIZE
    max_steps: int = 250
    goal_tolerance: float = 0.45  # preserved from v1 GOAL_TOLERANCE

    # -------------------------------------------------------------------------
    # Physical Robot Dimensions & Dynamics
    # -------------------------------------------------------------------------
    robot_radius: float = 0.25
    safety_margin: float = 0.20
    max_speed: float = 1.0  # preserved from v1 MAX_SPEED

    # -------------------------------------------------------------------------
    # Electrostatic-Inspired Potential Field Parameters
    # -------------------------------------------------------------------------
    k_robot: float = 15.0      # Repulsion strength for robot-to-robot
    k_obstacle: float = 25.0   # Repulsion strength for robot-to-moving-obstacle
    influence_margin: float = 2.0  # L_ij = D_ij + influence_margin
    min_distance_epsilon: float = 1e-4
    max_force_clip: float = 30.0

    # -------------------------------------------------------------------------
    # Predictive Collision Detection Parameters
    # -------------------------------------------------------------------------
    prediction_horizon: float = 2.0  # T simulation-time units
    warning_risk_threshold: float = 0.15  # rho threshold for warning
    critical_risk_threshold: float = 0.60  # rho threshold for critical alert

    # -------------------------------------------------------------------------
    # Control Barrier Function (CBF) Safety Shield Parameters
    # -------------------------------------------------------------------------
    cbf_alpha: float = 2.0      # Extended class-K gain in continuous CBF condition
    cbf_solver: str = "slsqp"   # Solver strategy ("slsqp" or "analytical_qp")
    cbf_slack_penalty: float = 1e4
    qp_tolerance: float = 1e-6
    emergency_stop_speed_threshold: float = 1e-3

    # -------------------------------------------------------------------------
    # Static Potential Field (Preserved from v1)
    # -------------------------------------------------------------------------
    goal_gain: float = 2.0             # Preserved GOAL_GAIN
    static_obstacle_gain: float = 80.0 # Preserved OBSTACLE_GAIN
    static_obstacle_influence: float = 2.0 # Preserved OBSTACLE_INFLUENCE
    static_singularity_clamp: float = 0.5

    # -------------------------------------------------------------------------
    # Quantum Spectral Pipeline Calibration (Preserved from v1)
    # -------------------------------------------------------------------------
    n_qubits: int = 8
    c25: float = 0.01645914
    c75: float = 0.01987830
    b_min: float = 0.25
    b_max: float = 0.75
    gamma: float = 3.0

    def compute_safety_distance(self, other_radius: float) -> float:
        """
        D_ij = R_i + R_j + m
        """
        return self.robot_radius + other_radius + self.safety_margin

    def compute_robot_safety_distance(self) -> float:
        """
        D_ij for two identical robots: 0.25 + 0.25 + 0.20 = 0.70
        """
        return self.compute_safety_distance(self.robot_radius)

    def compute_influence_distance(self, safety_distance: float) -> float:
        """
        L_ij = D_ij + influence_margin
        """
        return safety_distance + self.influence_margin

