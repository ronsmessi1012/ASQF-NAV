"""
Component B: Predictive Collision Detection.

Implements predictive collision analysis using relative position and velocity:
    r_ij = p_i - q_j
    v_rel = v_i - u_j
    t_star = clip(-dot(r_ij, v_rel) / ||v_rel||^2, 0, T)
    d_star = ||r_ij + t_star * v_rel||
    rho_ij = max(0, (D_ij - d_star) / D_ij)

Handles zero/near-zero relative velocity safely.
Returns structured telemetry and configurable warning indicators.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import numpy as np

from navigation_v2.config import AQSFConfigV2
from navigation_v2.dynamic_entities import MovingObstacle


@dataclass
class CollisionPrediction:
    entity_id: str
    entity_type: str  # "robot" or "moving_obstacle"
    current_distance: float
    relative_velocity: np.ndarray
    time_to_closest_approach: float  # t*
    predicted_minimum_distance: float  # d*
    safety_distance: float  # D_ij
    collision_risk: float  # rho_ij in [0, 1]
    warning_status: str  # "none", "warning", "critical"

    def to_dict(self) -> dict:
        return {
            "entity_id": self.entity_id,
            "entity_type": self.entity_type,
            "current_distance": float(self.current_distance),
            "relative_velocity": self.relative_velocity.tolist(),
            "time_to_closest_approach": float(self.time_to_closest_approach),
            "predicted_minimum_distance": float(self.predicted_minimum_distance),
            "safety_distance": float(self.safety_distance),
            "collision_risk": float(self.collision_risk),
            "warning_status": self.warning_status,
        }


def predict_closest_approach(
    p_i: np.ndarray,
    v_i: np.ndarray,
    q_j: np.ndarray,
    u_j: np.ndarray,
    safety_distance: float,
    horizon_T: float = 2.0,
    warning_threshold: float = 0.15,
    critical_threshold: float = 0.60,
    entity_id: str = "entity",
    entity_type: str = "unknown"
) -> CollisionPrediction:
    """
    Compute predicted closest approach and collision risk score between robot i and entity j.
    """
    p = np.asarray(p_i, dtype=float)
    v = np.asarray(v_i, dtype=float)
    q = np.asarray(q_j, dtype=float)
    u = np.asarray(u_j, dtype=float)

    r_ij = p - q
    current_dist = float(np.linalg.norm(r_ij))

    v_rel = v - u
    v_rel_sq = float(np.dot(v_rel, v_rel))

    # Safe handling of near-zero relative velocity
    if v_rel_sq < 1e-10:
        t_star = 0.0
        d_star = current_dist
    else:
        # t_star = clip(-dot(r, v_rel) / ||v_rel||^2, 0, T)
        t_unclipped = -np.dot(r_ij, v_rel) / v_rel_sq
        t_star = float(np.clip(t_unclipped, 0.0, horizon_T))
        d_star = float(np.linalg.norm(r_ij + t_star * v_rel))

    # rho_ij = max(0, (D_ij - d_star) / D_ij)
    if safety_distance > 1e-8:
        risk = max(0.0, (safety_distance - d_star) / safety_distance)
    else:
        risk = 0.0

    # Warning status classification
    if risk >= critical_threshold:
        status = "critical"
    elif risk >= warning_threshold:
        status = "warning"
    else:
        status = "none"

    return CollisionPrediction(
        entity_id=entity_id,
        entity_type=entity_type,
        current_distance=current_dist,
        relative_velocity=v_rel.copy(),
        time_to_closest_approach=t_star,
        predicted_minimum_distance=d_star,
        safety_distance=safety_distance,
        collision_risk=risk,
        warning_status=status
    )


def analyze_all_predictive_collisions(
    robot_idx: int,
    robot_positions: np.ndarray,
    robot_velocities: np.ndarray,
    moving_obstacles: List[MovingObstacle],
    config: Optional[AQSFConfigV2] = None
) -> List[CollisionPrediction]:
    """
    Evaluate predictive collision metrics for robot i against all other robots
    and all dynamic moving obstacles.
    """
    if config is None:
        config = AQSFConfigV2()

    predictions: List[CollisionPrediction] = []
    p_i = robot_positions[robot_idx]
    v_i = robot_velocities[robot_idx]

    # 1. Other robots
    robot_safety_dist = config.compute_robot_safety_distance()
    for j in range(len(robot_positions)):
        if j == robot_idx:
            continue
        pred = predict_closest_approach(
            p_i=p_i,
            v_i=v_i,
            q_j=robot_positions[j],
            u_j=robot_velocities[j],
            safety_distance=robot_safety_dist,
            horizon_T=config.prediction_horizon,
            warning_threshold=config.warning_risk_threshold,
            critical_threshold=config.critical_risk_threshold,
            entity_id=f"robot_{j}",
            entity_type="robot"
        )
        predictions.append(pred)

    # 2. Moving obstacles
    if moving_obstacles:
        for obs in moving_obstacles:
            obs_safety_dist = config.compute_safety_distance(obs.radius)
            pred = predict_closest_approach(
                p_i=p_i,
                v_i=v_i,
                q_j=obs.position,
                u_j=obs.velocity,
                safety_distance=obs_safety_dist,
                horizon_T=config.prediction_horizon,
                warning_threshold=config.warning_risk_threshold,
                critical_threshold=config.critical_risk_threshold,
                entity_id=obs.id,
                entity_type="moving_obstacle"
            )
            predictions.append(pred)

    return predictions

