"""
Component C: Combined Navigation Field.

Preserves the existing static APF while integrating the electrostatic dynamic field:
    F_static = F_goal + F_static_obstacle
    F_dynamic = sum U_robot + sum U_moving_obstacle
    F_total = F_static + F_dynamic

Provides separate measurement, access, and visualization for static and dynamic fields.
Ensures 100% backward compatibility with existing APF parameterization.
"""

from typing import List, Optional, Tuple, Union
import numpy as np

from environment.field import (
    distance_field,
    goal_potential,
    obstacle_potential,
    GRID_SIZE,
)
from navigation_v2.config import AQSFConfigV2
from navigation_v2.dynamic_entities import MovingObstacle
from navigation_v2.electrostatic_potential import compute_dynamic_field_grid


def generate_static_field_grid(
    X: np.ndarray,
    Y: np.ndarray,
    goal: np.ndarray,
    static_obstacles: List[Tuple[float, float]],
    config: Optional[AQSFConfigV2] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute static potential field using the existing v1 equations:
        F_static = F_goal + F_obstacle
    """
    f_goal = goal_potential(X, Y, goal)
    f_obstacle = obstacle_potential(X, Y, static_obstacles)
    f_static = f_goal + f_obstacle
    return f_goal, f_obstacle, f_static


def generate_combined_field(
    goal: np.ndarray,
    static_obstacles: List[Tuple[float, float]],
    robot_positions: np.ndarray,
    moving_obstacles: Optional[List[MovingObstacle]] = None,
    current_robot_idx: Optional[int] = None,
    config: Optional[AQSFConfigV2] = None,
    include_dynamic: bool = True
) -> dict:
    """
    Generate static, dynamic, and combined fields on a 16x16 grid.

    Returns dict containing:
        - X, Y: Coordinate meshgrids (16x16)
        - F_goal: Goal attraction grid
        - F_static_obs: Static obstacle repulsion grid
        - F_static: Total static field
        - F_dynamic: Total dynamic electrostatic field
        - F_total: Superimposed field (F_static + F_dynamic)
    """
    if config is None:
        config = AQSFConfigV2()

    x = np.arange(config.grid_size)
    y = np.arange(config.grid_size)
    X, Y = np.meshgrid(x, y, indexing="xy")

    f_goal, f_static_obs, f_static = generate_static_field_grid(
        X, Y, goal, static_obstacles, config
    )

    if include_dynamic:
        f_dynamic = compute_dynamic_field_grid(
            X, Y,
            robot_positions=robot_positions,
            moving_obstacles=moving_obstacles or [],
            current_robot_idx=current_robot_idx,
            config=config
        )
    else:
        f_dynamic = np.zeros_like(X, dtype=float)

    f_total = f_static + f_dynamic

    return {
        "X": X,
        "Y": Y,
        "F_goal": f_goal,
        "F_static_obs": f_static_obs,
        "F_static": f_static,
        "F_dynamic": f_dynamic,
        "F_total": f_total
    }

