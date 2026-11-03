"""
Navigation Controller for AQSF-NAV v2.

Integrates:
1. Continuous bilinear gradient sampling on the navigation field (classical or quantum reconstructed).
2. Blending of field gradient and goal direction (0.65 * (-grad/||grad||) + 0.35 * d_goal).
3. Predictive collision urgency analysis.
4. Independent Control Barrier Function (CBF) safety enforcement to guarantee collision avoidance.
"""

from typing import List, Optional, Tuple, Union
import numpy as np

from navigation.controller import (
    bilinear_interpolate,
    sample_field_gradient,
    reached_goal,
)
from navigation_v2.config import AQSFConfigV2
from navigation_v2.dynamic_entities import MovingObstacle
from navigation_v2.predictive_collision import (
    CollisionPrediction,
    analyze_all_predictive_collisions,
)
from navigation_v2.cbf_safety_shield import (
    CBFSafetyShield,
    CBFSolverResult,
)


def compute_desired_velocity(
    field: np.ndarray,
    position: np.ndarray,
    goal: np.ndarray,
    max_speed: float = 1.0,
    predictive_steering: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute candidate navigation velocity from the field gradient and goal direction:
        v_base = 0.65 * (-grad / ||grad||) + 0.35 * d_goal
    Optionally applies predictive steering bias if active.
    Returns:
        v_desired, gradient, goal_direction
    """
    px, py = position[0], position[1]

    # Sample gradient via bilinear interpolation
    gx, gy = sample_field_gradient(field, (px, py))
    gradient = np.array([gx, gy], dtype=float)

    # Gradient descent direction
    grad_norm = np.linalg.norm(gradient)
    if grad_norm > 1e-10:
        grad_dir = -gradient / grad_norm
    else:
        grad_dir = np.zeros(2, dtype=float)

    # Goal direction
    goal_vec = np.asarray(goal, dtype=float) - np.asarray(position, dtype=float)
    goal_dist = np.linalg.norm(goal_vec)
    if goal_dist > 1e-10:
        goal_dir = goal_vec / goal_dist
    else:
        goal_dir = np.zeros(2, dtype=float)

    # Base velocity combination (preserved 0.65 / 0.35 blend)
    v_base = 0.65 * grad_dir + 0.35 * goal_dir

    if predictive_steering is not None:
        v_base += predictive_steering

    # Normalize to max_speed
    base_mag = np.linalg.norm(v_base)
    if base_mag > 1e-10:
        v_desired = (v_base / base_mag) * max_speed
    else:
        v_desired = np.zeros(2, dtype=float)

    return v_desired, gradient, goal_dir


def navigation_step_v2(
    robot_idx: int,
    position: np.ndarray,
    goal: np.ndarray,
    field: np.ndarray,
    all_robot_positions: np.ndarray,
    all_robot_velocities: np.ndarray,
    moving_obstacles: Optional[List[MovingObstacle]] = None,
    shield: Optional[CBFSafetyShield] = None,
    config: Optional[AQSFConfigV2] = None,
    enable_cbf: bool = True,
    enable_predictive: bool = True
) -> Tuple[np.ndarray, np.ndarray, CBFSolverResult, List[CollisionPrediction]]:
    """
    Execute one complete v2 navigation step for robot i:
    1. Predictive collision analysis (Component B).
    2. Candidate desired velocity calculation from field (Component C).
    3. Independent CBF safety shield (Component E).
    4. State integration: p_{t+1} = p_t + dt * v_safe.
    """
    if config is None:
        config = AQSFConfigV2()
    if shield is None:
        shield = CBFSafetyShield(config)

    p_i = np.asarray(position, dtype=float)
    v_i = np.asarray(all_robot_velocities[robot_idx], dtype=float)

    # 1. Predictive collision detection
    predictions = analyze_all_predictive_collisions(
        robot_idx=robot_idx,
        robot_positions=all_robot_positions,
        robot_velocities=all_robot_velocities,
        moving_obstacles=moving_obstacles or [],
        config=config
    )

    # Calculate predictive steering bias if enabled
    pred_bias = np.zeros(2, dtype=float)
    if enable_predictive:
        # Determine current heading / goal direction for lateral reference
        goal_vec = np.asarray(goal, dtype=float) - p_i
        goal_dist = np.linalg.norm(goal_vec)
        if goal_dist > 1e-6:
            heading = goal_vec / goal_dist
        else:
            heading = np.array([1.0, 0.0], dtype=float)

        # Right-hand lateral unit vector: [heading_y, -heading_x]
        right_lateral = np.array([heading[1], -heading[0]], dtype=float)

        for pred in predictions:
            if pred.collision_risk > config.warning_risk_threshold:
                # Steer laterally using maritime right-hand bypass rule, scaled by risk
                pred_bias += pred.collision_risk * 1.2 * right_lateral

    # 2. Desired velocity from potential field
    v_des, gradient, goal_dir = compute_desired_velocity(
        field=field,
        position=p_i,
        goal=goal,
        max_speed=config.max_speed,
        predictive_steering=pred_bias if enable_predictive else None
    )

    # 3. CBF Safety Shield
    if enable_cbf:
        cbf_res = shield.filter_velocity(
            robot_idx=robot_idx,
            desired_velocity=v_des,
            robot_positions=all_robot_positions,
            robot_velocities=all_robot_velocities,
            moving_obstacles=moving_obstacles
        )
        v_final = cbf_res.safe_velocity
    else:
        cbf_res = CBFSolverResult(
            safe_velocity=v_des,
            desired_velocity=v_des,
            intervened=False,
            status="cbf_disabled",
            min_barrier_value=0.0,
            active_constraints_count=0,
            solver_time=0.0,
            violations_detected=0,
            message="CBF disabled."
        )
        v_final = v_des

    # 4. Integrate position: p_{new} = p + dt * v
    dt = config.step_size
    new_position = p_i + dt * v_final

    return new_position, v_final, cbf_res, predictions
