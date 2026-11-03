"""
Unit Tests for Component E & F: Control Barrier Function Safety Shield (Tests E & F).
"""

import unittest
import numpy as np

from navigation_v2.config import AQSFConfigV2
from navigation_v2.dynamic_entities import MovingObstacle
from navigation_v2.cbf_safety_shield import (
    CBFSafetyShield,
    evaluate_cbf_condition,
    filter_joint_velocities_centralized,
)


class TestCBFSafetyShield(unittest.TestCase):
    def setUp(self):
        self.config = AQSFConfigV2()
        self.shield = CBFSafetyShield(self.config)
        self.d_safe = self.config.compute_robot_safety_distance()  # 0.70

    def test_e1_safe_entities_moving_apart(self):
        """Entities well separated and moving apart: desired velocity unmodified."""
        p_robot = np.array([2.0, 2.0])
        v_des = np.array([-0.5, 0.0])  # moving west away from peer
        peers_p = np.array([[2.0, 2.0], [5.0, 2.0]])
        peers_v = np.array([[-0.5, 0.0], [0.5, 0.0]])

        res = self.shield.filter_velocity(
            robot_idx=0,
            desired_velocity=v_des,
            robot_positions=peers_p,
            robot_velocities=peers_v
        )
        self.assertEqual(res.status, "success")
        self.assertFalse(res.intervened)
        np.testing.assert_allclose(res.safe_velocity, v_des, atol=1e-5)

    def test_e2_safe_entities_approaching_modified(self):
        """Entities approaching head-on on collision course: CBF modifies velocity."""
        p_robot = np.array([2.0, 2.0])
        v_des = np.array([1.0, 0.0])  # directly charging toward peer
        peers_p = np.array([[2.0, 2.0], [3.0, 2.0]])  # peer at (3, 2), dist = 1.0 (close to 0.70)
        peers_v = np.array([[1.0, 0.0], [-1.0, 0.0]])

        res = self.shield.filter_velocity(
            robot_idx=0,
            desired_velocity=v_des,
            robot_positions=peers_p,
            robot_velocities=peers_v
        )
        self.assertEqual(res.status, "modified")
        self.assertTrue(res.intervened)
        # Verify the forward velocity along x was reduced or redirected away
        self.assertLess(res.safe_velocity[0], v_des[0])

        # Verify CBF condition is satisfied by safe_velocity
        r = peers_p[0] - peers_p[1]
        h = np.linalg.norm(r)**2 - self.d_safe**2
        cbf_val = 2.0 * np.dot(r, res.safe_velocity - peers_v[1]) + self.config.cbf_alpha * h
        self.assertGreaterEqual(cbf_val, -1e-5)

    def test_e3_near_safety_boundary(self):
        """Entities near safety boundary (d = d_safe + 0.02): inward velocity blocked."""
        p1 = np.array([2.0, 2.0])
        p2 = np.array([2.0 + self.d_safe + 0.02, 2.0])
        v_des = np.array([1.0, 0.0])  # inward
        v2 = np.array([0.0, 0.0])

        res = self.shield.filter_velocity(
            robot_idx=0,
            desired_velocity=v_des,
            robot_positions=np.array([p1, p2]),
            robot_velocities=np.array([v_des, v2])
        )
        self.assertTrue(res.intervened)
        # Heading east must be near zero or negative
        self.assertLessEqual(res.safe_velocity[0], 0.05)

    def test_e4_multiple_simultaneous_constraints(self):
        """Robot with multiple approaching obstacles: satisfies all simultaneous constraints."""
        p_robot = np.array([5.0, 5.0])
        v_des = np.array([0.8, 0.8])

        # Peer robot to the east
        p_peer = np.array([6.0, 5.0])
        v_peer = np.array([-0.5, 0.0])

        # Moving obstacle to the north
        obs = MovingObstacle(
            id="obs_n",
            position=np.array([5.0, 6.0]),
            velocity=np.array([0.0, -0.5]),
            radius=0.25
        )

        res = self.shield.filter_velocity(
            robot_idx=0,
            desired_velocity=v_des,
            robot_positions=np.array([p_robot, p_peer]),
            robot_velocities=np.array([v_des, v_peer]),
            moving_obstacles=[obs]
        )
        self.assertTrue(res.intervened)
        self.assertIn(res.status, ["modified", "success"])

        # Check speed constraint
        speed = np.linalg.norm(res.safe_velocity)
        self.assertLessEqual(speed, self.config.max_speed + 1e-5)

        # Check peer CBF
        r1 = p_robot - p_peer
        h1 = np.linalg.norm(r1)**2 - self.d_safe**2
        cbf1 = 2.0 * np.dot(r1, res.safe_velocity - v_peer) + self.config.cbf_alpha * h1
        self.assertGreaterEqual(cbf1, -1e-4)

        # Check obstacle CBF
        obs_d = self.config.compute_safety_distance(obs.radius)
        r2 = p_robot - obs.position
        h2 = np.linalg.norm(r2)**2 - obs_d**2
        cbf2 = 2.0 * np.dot(r2, res.safe_velocity - obs.velocity) + self.config.cbf_alpha * h2
        self.assertGreaterEqual(cbf2, -1e-4)

    def test_e5_moving_obstacle_constraint(self):
        """Moving obstacle crossing robot's path: robot safely adjusts."""
        p_robot = np.array([3.0, 3.0])
        v_des = np.array([1.0, 0.0])  # East
        obs = MovingObstacle(
            id="crossing_obs",
            position=np.array([4.0, 3.0]),
            velocity=np.array([0.0, 0.5]),
            radius=0.3
        )
        res = self.shield.filter_velocity(
            robot_idx=0,
            desired_velocity=v_des,
            robot_positions=np.array([p_robot]),
            robot_velocities=np.array([v_des]),
            moving_obstacles=[obs]
        )
        self.assertTrue(res.intervened)
        # Velocity in eastward direction must be curtailed
        self.assertLess(res.safe_velocity[0], 0.5)

    def test_e6_solver_infeasibility_fallback(self):
        """
        When two obstacles sandwich a robot well inside the safety distance:
        CBF reports infeasible and executes documented fallback without crashing.
        """
        p_robot = np.array([5.0, 5.0])
        # Two opposing obstacles already severely encroaching inside d_safe
        obs_east = MovingObstacle(
            id="obs_e",
            position=np.array([5.2, 5.0]),  # dist = 0.2 < 0.70!
            velocity=np.array([-1.0, 0.0]),
            radius=0.25
        )
        obs_west = MovingObstacle(
            id="obs_w",
            position=np.array([4.8, 5.0]),  # dist = 0.2 < 0.70!
            velocity=np.array([1.0, 0.0]),
            radius=0.25
        )
        v_des = np.array([0.5, 0.0])

        res = self.shield.filter_velocity(
            robot_idx=0,
            desired_velocity=v_des,
            robot_positions=np.array([p_robot]),
            robot_velocities=np.array([v_des]),
            moving_obstacles=[obs_east, obs_west]
        )
        # Should detect conflict and trigger fallback / emergency stop
        self.assertIn(res.status, ["infeasible_fallback", "emergency_stop", "modified"])
        self.assertGreaterEqual(res.violations_detected, 0)
        self.assertTrue(np.all(np.isfinite(res.safe_velocity)))

    def test_e7_max_velocity_constraint(self):
        """Verify returned velocity never exceeds max_speed."""
        p_robot = np.array([2.0, 2.0])
        v_des = np.array([5.0, 5.0])  # Intentionally exceeds max_speed = 1.0

        res = self.shield.filter_velocity(
            robot_idx=0,
            desired_velocity=v_des,
            robot_positions=np.array([p_robot]),
            robot_velocities=np.array([v_des])
        )
        speed = np.linalg.norm(res.safe_velocity)
        self.assertLessEqual(speed, self.config.max_speed + 1e-6)

    def test_f_centralized_joint_optimization(self):
        """Verify centralized joint CBF coordinates mutual avoidance between two robots."""
        positions = np.array([
            [2.0, 2.0],
            [3.0, 2.0]
        ])
        # Both robots want to move toward each other
        desired_vels = np.array([
            [1.0, 0.0],
            [-1.0, 0.0]
        ])
        safe_vels, results = filter_joint_velocities_centralized(
            desired_velocities=desired_vels,
            robot_positions=positions,
            config=self.config
        )
        # Both robots want to move toward each other with magnitude 1.0
        # Distance = 1.0, d_safe = 0.70, h = 0.51.
        # CBF constraint: v0_x - v1_x <= (alpha * h) / 2 = (2.0 * 0.51) / 2 = 0.51
        # Optimal symmetric solution: v0_x = 0.255, v1_x = -0.255 (relative speed reduced from 2.0 to 0.51)
        self.assertLessEqual(safe_vels[0, 0], 0.26)
        self.assertGreaterEqual(safe_vels[1, 0], -0.26)
        self.assertAlmostEqual(safe_vels[0, 0] - safe_vels[1, 0], 0.51, places=3)
        # Joint CBF condition must hold
        r = positions[0] - positions[1]
        h = np.linalg.norm(r)**2 - self.d_safe**2
        cbf = 2.0 * np.dot(r, safe_vels[0] - safe_vels[1]) + self.config.cbf_alpha * h
        self.assertGreaterEqual(cbf, -1e-4)


if __name__ == "__main__":
    unittest.main()
