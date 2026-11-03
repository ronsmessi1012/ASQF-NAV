"""
Dynamic Entity Representation for Moving Obstacles.

Supports:
1. Constant-velocity movement with boundary bounce / wraparound / open bounds.
2. Waypoint-based movement with smooth velocity tracking.
3. Deterministic scripted movement (custom callable or parameterized functions)
   for strictly reproducible experiments.
"""

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple, Union
import numpy as np


@dataclass
class MovingObstacle:
    """
    Representation of a dynamic moving obstacle in the navigation field.
    """
    id: str
    position: np.ndarray
    velocity: np.ndarray
    radius: float
    movement_model: str = "constant_velocity"  # "constant_velocity", "waypoint", "scripted"
    target_position: Optional[np.ndarray] = None
    waypoints: Optional[List[np.ndarray]] = None
    current_waypoint_idx: int = 0
    waypoint_tolerance: float = 0.20
    script_fn: Optional[Callable[[float, "MovingObstacle"], Tuple[np.ndarray, np.ndarray]]] = None
    bounds: Optional[Tuple[float, float, float, float]] = None  # (min_x, max_x, min_y, max_y)
    speed: float = 0.5

    def __post_init__(self):
        self.position = np.asarray(self.position, dtype=float).copy()
        self.velocity = np.asarray(self.velocity, dtype=float).copy()
        if self.target_position is not None:
            self.target_position = np.asarray(self.target_position, dtype=float).copy()
        if self.waypoints is not None:
            self.waypoints = [np.asarray(wp, dtype=float).copy() for wp in self.waypoints]

    def step(self, dt: float, current_time: float = 0.0) -> None:
        """
        Advance the obstacle's state by time step dt according to its movement model.
        """
        if self.movement_model == "constant_velocity":
            self._step_constant_velocity(dt)
        elif self.movement_model == "waypoint":
            self._step_waypoint(dt)
        elif self.movement_model == "scripted":
            self._step_scripted(dt, current_time)
        else:
            raise ValueError(f"Unknown movement model: {self.movement_model}")

        # Check boundary collision/reflection if bounds specified
        if self.bounds is not None:
            min_x, max_x, min_y, max_y = self.bounds
            for dim, (b_min, b_max) in enumerate([(min_x, max_x), (min_y, max_y)]):
                if self.position[dim] < b_min:
                    self.position[dim] = b_min
                    self.velocity[dim] = abs(self.velocity[dim])
                elif self.position[dim] > b_max:
                    self.position[dim] = b_max
                    self.velocity[dim] = -abs(self.velocity[dim])

    def _step_constant_velocity(self, dt: float) -> None:
        self.position += self.velocity * dt

    def _step_waypoint(self, dt: float) -> None:
        if not self.waypoints or len(self.waypoints) == 0:
            self._step_constant_velocity(dt)
            return

        target = self.waypoints[self.current_waypoint_idx]
        diff = target - self.position
        dist = np.linalg.norm(diff)

        if dist < self.waypoint_tolerance:
            # Advance to next waypoint (cycling)
            self.current_waypoint_idx = (self.current_waypoint_idx + 1) % len(self.waypoints)
            target = self.waypoints[self.current_waypoint_idx]
            diff = target - self.position
            dist = np.linalg.norm(diff)

        if dist > 1e-8:
            direction = diff / dist
            self.velocity = direction * self.speed
        else:
            self.velocity = np.zeros(2)

        self.position += self.velocity * dt

    def _step_scripted(self, dt: float, current_time: float) -> None:
        if self.script_fn is None:
            raise ValueError(f"MovingObstacle '{self.id}' configured as scripted, but script_fn is None.")
        new_pos, new_vel = self.script_fn(current_time + dt, self)
        self.position = np.asarray(new_pos, dtype=float).copy()
        self.velocity = np.asarray(new_vel, dtype=float).copy()

    def get_state(self) -> dict:
        return {
            "id": self.id,
            "position": self.position.copy(),
            "velocity": self.velocity.copy(),
            "radius": float(self.radius),
            "movement_model": self.movement_model,
        }

