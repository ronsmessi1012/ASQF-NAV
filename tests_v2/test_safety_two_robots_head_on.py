"""
Checkpoint 3: Safety Testing — Two Robots Approaching Each Other Head-On
Validates the classical dynamic safety mechanism (Electrostatic APF + Predictive + CBF)
BEFORE integrating the QFT spectral pipeline.
"""

import unittest
import numpy as np

from navigation.controller import reached_goal
from navigation_v2.config import AQSFConfigV2
from navigation_v2.combined_field import generate_combined_field
from navigation_v2.controller_v2 import navigation_step_v2
from navigation_v2.cbf_safety_shield import CBFSafetyShield


class TestTwoRobotsHeadOn(unittest.TestCase):
    def test_two_robots_head_on_collision_avoidance(self):
        """
        Two robots approach each other on a direct head-on collision course along y = 7.5.
        Robot 0: Start (2.0, 7.5), Goal (13.0, 7.5)
        Robot 1: Start (13.0, 7.5), Goal (2.0, 7.5)
        Verify:
        1. Minimum inter-robot distance remains strictly >= D_safe (0.70).
        2. CBF interventions occur to enforce safety.
        3. Both robots successfully navigate around each other and reach their goals.
        """
        config = AQSFConfigV2()
        shield = CBFSafetyShield(config)
        d_safe = config.compute_robot_safety_distance()  # 0.70

        positions = np.array([
            [2.0, 7.5],
            [13.0, 7.5]
        ], dtype=float)

        goals = np.array([
            [13.0, 7.5],
            [2.0, 7.5]
        ], dtype=float)

        velocities = np.array([
            [1.0, 0.0],
            [-1.0, 0.0]
        ], dtype=float)

        static_obstacles = []  # Open space head-on encounter
        min_dist = float("inf")
        cbf_intervention_count = 0
        collision_detected = False

        max_steps = 150
        for step in range(max_steps):
            # Compute inter-robot distance
            dist = float(np.linalg.norm(positions[0] - positions[1]))
            if dist < min_dist:
                min_dist = dist
            if dist < d_safe:
                collision_detected = True

            # Step both robots
            new_positions = positions.copy()
            new_velocities = velocities.copy()

            for i in range(2):
                field_data = generate_combined_field(
                    goal=goals[i],
                    static_obstacles=static_obstacles,
                    robot_positions=positions,
                    moving_obstacles=[],
                    current_robot_idx=i,
                    config=config
                )
                field = field_data["F_total"]

                new_pos, new_vel, cbf_res, preds = navigation_step_v2(
                    robot_idx=i,
                    position=positions[i],
                    goal=goals[i],
                    field=field,
                    all_robot_positions=positions,
                    all_robot_velocities=velocities,
                    moving_obstacles=[],
                    shield=shield,
                    config=config,
                    enable_cbf=True,
                    enable_predictive=True
                )
                new_positions[i] = new_pos
                new_velocities[i] = new_vel
                if cbf_res.intervened:
                    cbf_intervention_count += 1

            positions = new_positions
            velocities = new_velocities

            # Check goal convergence
            r0_done = reached_goal(positions[0], goals[0])
            r1_done = reached_goal(positions[1], goals[1])
            if r0_done and r1_done:
                break

        print(f"\n--- Checkpoint 3: Two Robots Head-On Result ---")
        print(f"Steps to completion: {step + 1}")
        print(f"Minimum observed distance: {min_dist:.4f} (Required safe distance: {d_safe:.4f})")
        print(f"CBF interventions: {cbf_intervention_count}")
        print(f"Robot 0 final: {positions[0]}, Goal: {goals[0]}")
        print(f"Robot 1 final: {positions[1]}, Goal: {goals[1]}")

        self.assertFalse(collision_detected, f"Collision detected! Min dist: {min_dist} < {d_safe}")
        self.assertGreaterEqual(min_dist, d_safe)
        self.assertGreater(cbf_intervention_count, 0)
        self.assertTrue(reached_goal(positions[0], goals[0]))
        self.assertTrue(reached_goal(positions[1], goals[1]))


if __name__ == "__main__":
    unittest.main()

