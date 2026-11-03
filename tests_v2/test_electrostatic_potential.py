"""
Unit Tests for Component A: Electrostatic-Inspired Potential and Forces (Tests A, B, C).
"""

import unittest
import numpy as np

from navigation_v2.config import AQSFConfigV2
from navigation_v2.dynamic_entities import MovingObstacle
from navigation_v2.electrostatic_potential import (
    electrostatic_potential_scalar,
    electrostatic_force_vector,
    compute_dynamic_field_grid,
    compute_dynamic_repulsive_force,
)


class TestElectrostaticPotential(unittest.TestCase):
    def setUp(self):
        self.config = AQSFConfigV2()
        self.d_safe = self.config.compute_robot_safety_distance()  # 0.70
        self.influence = self.config.compute_influence_distance(self.d_safe)  # 2.70
        self.k = self.config.k_robot

    def test_a_potential_zero_outside_influence(self):
        """Verify potential is exactly 0 for distance >= L_ij."""
        d_out = self.influence + 0.5
        u = electrostatic_potential_scalar(d_out, self.influence, self.k)
        self.assertEqual(u, 0.0)

        # Edge of influence
        u_edge = electrostatic_potential_scalar(self.influence, self.influence, self.k)
        self.assertEqual(u_edge, 0.0)

    def test_a_potential_monotonic_increase(self):
        """Verify potential strictly increases as distance decreases."""
        distances = np.linspace(0.1, self.influence - 0.05, 50)
        potentials = [
            electrostatic_potential_scalar(d, self.influence, self.k)
            for d in distances
        ]
        # Check monotonicity: as distance increases, potential decreases
        for i in range(len(potentials) - 1):
            self.assertGreater(potentials[i], potentials[i + 1])

    def test_a_repulsive_force_direction(self):
        """Verify repulsive force points away from the entity (along r = p - q)."""
        q = np.array([5.0, 5.0])
        # Position p is to the east of q
        p = np.array([6.0, 5.0])
        f = electrostatic_force_vector(p, q, self.influence, self.k)
        self.assertGreater(f[0], 0.0)  # Pushes east
        self.assertAlmostEqual(f[1], 0.0, places=6)

        # Position p is north-west of q
        p_nw = np.array([4.0, 6.0])
        r_nw = p_nw - q
        f_nw = electrostatic_force_vector(p_nw, q, self.influence, self.k)
        # Cosine similarity with r_nw must be 1.0
        cos_sim = np.dot(f_nw, r_nw) / (np.linalg.norm(f_nw) * np.linalg.norm(r_nw))
        self.assertAlmostEqual(cos_sim, 1.0, places=5)

    def test_a_finite_force_and_no_nan_at_boundary_or_coincident(self):
        """Verify force is finite and no NaN/inf at d -> 0 or d -> L."""
        q = np.array([2.0, 2.0])

        # Exactly coincident
        p_coincident = np.array([2.0, 2.0])
        f_coin = electrostatic_force_vector(p_coincident, q, self.influence, self.k)
        self.assertTrue(np.all(np.isfinite(f_coin)))
        self.assertLessEqual(np.linalg.norm(f_coin), self.config.max_force_clip + 1e-6)

        # Tiny separation
        p_tiny = np.array([2.0 + 1e-8, 2.0])
        f_tiny = electrostatic_force_vector(p_tiny, q, self.influence, self.k)
        self.assertTrue(np.all(np.isfinite(f_tiny)))
        self.assertLessEqual(np.linalg.norm(f_tiny), self.config.max_force_clip + 1e-6)

        # Boundary d == L
        p_bound = q + np.array([self.influence, 0.0])
        f_bound = electrostatic_force_vector(p_bound, q, self.influence, self.k)
        self.assertTrue(np.all(np.isfinite(f_bound)))
        self.assertAlmostEqual(np.linalg.norm(f_bound), 0.0, places=6)

    def test_b_robot_to_robot_symmetry(self):
        """Verify Newton's third law behavior F_12 = -F_21 for identical robots."""
        p1 = np.array([3.0, 4.0])
        p2 = np.array([4.0, 5.0])
        f_12 = electrostatic_force_vector(p1, p2, self.influence, self.k)
        f_21 = electrostatic_force_vector(p2, p1, self.influence, self.k)

        np.testing.assert_allclose(f_12, -f_21, atol=1e-10)

    def test_b_robot_to_robot_increasing_repulsion(self):
        """Verify repulsion between robots increases as distance shrinks."""
        p1 = np.array([5.0, 5.0])
        p2_far = np.array([6.5, 5.0])   # dist = 1.5
        p2_near = np.array([5.8, 5.0])  # dist = 0.8

        f_far = electrostatic_force_vector(p1, p2_far, self.influence, self.k)
        f_near = electrostatic_force_vector(p1, p2_near, self.influence, self.k)

        self.assertGreater(np.linalg.norm(f_near), np.linalg.norm(f_far))

    def test_c_moving_obstacle_repulsion(self):
        """Verify dynamic moving obstacle produces correct spatial repulsion."""
        obs = MovingObstacle(
            id="obs_1",
            position=np.array([10.0, 10.0]),
            velocity=np.array([-0.5, 0.0]),
            radius=0.4
        )
        p_robot = np.array([10.0, 8.5])  # south of obstacle
        obs_safe = self.config.compute_safety_distance(obs.radius)  # 0.25 + 0.40 + 0.20 = 0.85
        obs_infl = self.config.compute_influence_distance(obs_safe)

        f_obs = electrostatic_force_vector(
            p_robot,
            obs.position,
            influence_radius=obs_infl,
            k=self.config.k_obstacle
        )
        # Should push south (negative y)
        self.assertLess(f_obs[1], 0.0)
        self.assertAlmostEqual(f_obs[0], 0.0, places=6)


if __name__ == "__main__":
    unittest.main()

