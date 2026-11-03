"""
Unit Tests for Component B: Closest-Approach Prediction and Collision Risk (Test D).
"""

import unittest
import numpy as np

from navigation_v2.config import AQSFConfigV2
from navigation_v2.predictive_collision import (
    predict_closest_approach,
    analyze_all_predictive_collisions,
)


class TestPredictiveCollision(unittest.TestCase):
    def setUp(self):
        self.config = AQSFConfigV2()
        self.d_safe = 0.70  # Standard two-robot safety distance
        self.T = 2.0

    def test_d1_head_on_approach(self):
        """
        Head-on approach:
        Robot 1 at (0, 0) moving east at 1.0 -> v1 = (1, 0)
        Robot 2 at (4, 0) moving west at 1.0 -> v2 = (-1, 0)
        r = (-4, 0), v_rel = (2, 0).
        Time of collision: t = 4 / 2 = 2.0 s.
        Analytical: t* = 2.0, d* = 0.0, rho = 1.0 (imminent direct hit).
        """
        p1 = np.array([0.0, 0.0])
        v1 = np.array([1.0, 0.0])
        p2 = np.array([4.0, 0.0])
        v2 = np.array([-1.0, 0.0])

        pred = predict_closest_approach(p1, v1, p2, v2, self.d_safe, horizon_T=self.T)
        self.assertAlmostEqual(pred.time_to_closest_approach, 2.0, places=5)
        self.assertAlmostEqual(pred.predicted_minimum_distance, 0.0, places=5)
        self.assertAlmostEqual(pred.collision_risk, 1.0, places=5)
        self.assertEqual(pred.warning_status, "critical")

    def test_d2_perpendicular_crossing(self):
        """
        Perpendicular crossing:
        Robot 1 at (0, 2) moving east at 1.0 -> v1 = (1, 0)
        Robot 2 at (2, 0) moving north at 1.0 -> v2 = (0, 1)
        r = (-2, 2), v_rel = (1, -1).
        ||v_rel||^2 = 2.
        -dot(r, v_rel) = -(-2*1 + 2*(-1)) = -(-4) = 4.
        t_unclipped = 4 / 2 = 2.0 s.
        Both arrive at (2, 2) at t = 2.0 s!
        Analytical: t* = 2.0, d* = 0.0, rho = 1.0.
        """
        p1 = np.array([0.0, 2.0])
        v1 = np.array([1.0, 0.0])
        p2 = np.array([2.0, 0.0])
        v2 = np.array([0.0, 1.0])

        pred = predict_closest_approach(p1, v1, p2, v2, self.d_safe, horizon_T=self.T)
        self.assertAlmostEqual(pred.time_to_closest_approach, 2.0, places=5)
        self.assertAlmostEqual(pred.predicted_minimum_distance, 0.0, places=5)
        self.assertAlmostEqual(pred.collision_risk, 1.0, places=5)

    def test_d2_perpendicular_miss(self):
        """
        Perpendicular crossing that misses safely:
        Robot 1 at (0, 3) moving east at 1.0 -> v1 = (1, 0)
        Robot 2 at (1, 0) moving north at 1.0 -> v2 = (0, 1)
        r = (-1, 3), v_rel = (1, -1).
        ||v_rel||^2 = 2.
        -dot(r, v_rel) = -(-1*1 + 3*(-1)) = -(-4) = 4.
        t_unclipped = 4 / 2 = 2.0 s.
        At t = 2.0:
        p1(2) = (2, 3), p2(2) = (1, 2).
        d* = ||(2, 3) - (1, 2)|| = ||(1, 1)|| = sqrt(2) ~= 1.414.
        Since d* = 1.414 > d_safe = 0.70, rho = 0.0.
        """
        p1 = np.array([0.0, 3.0])
        v1 = np.array([1.0, 0.0])
        p2 = np.array([1.0, 0.0])
        v2 = np.array([0.0, 1.0])

        pred = predict_closest_approach(p1, v1, p2, v2, self.d_safe, horizon_T=self.T)
        self.assertAlmostEqual(pred.time_to_closest_approach, 2.0, places=5)
        self.assertAlmostEqual(pred.predicted_minimum_distance, np.sqrt(2.0), places=5)
        self.assertAlmostEqual(pred.collision_risk, 0.0, places=5)
        self.assertEqual(pred.warning_status, "none")

    def test_d3_same_direction_movement(self):
        """
        Same-direction movement:
        Robot 1 at (0, 0) moving east at 1.0.
        Robot 2 ahead at (1.0, 0) moving east at 0.5.
        v_rel = (0.5, 0).
        r = (-1.0, 0).
        -dot(r, v_rel) = -(-1.0 * 0.5) = 0.5.
        ||v_rel||^2 = 0.25.
        t_unclipped = 0.5 / 0.25 = 2.0 s.
        At t = 2.0, Robot 1 reaches 2.0, Robot 2 reaches 1.0 + 0.5*2 = 2.0 -> collision!
        """
        p1 = np.array([0.0, 0.0])
        v1 = np.array([1.0, 0.0])
        p2 = np.array([1.0, 0.0])
        v2 = np.array([0.5, 0.0])

        pred = predict_closest_approach(p1, v1, p2, v2, self.d_safe, horizon_T=self.T)
        self.assertAlmostEqual(pred.time_to_closest_approach, 2.0, places=5)
        self.assertAlmostEqual(pred.predicted_minimum_distance, 0.0, places=5)
        self.assertAlmostEqual(pred.collision_risk, 1.0, places=5)

    def test_d4_receding_entities(self):
        """
        Receding entities:
        Robot 1 at (0, 0) moving west at 1.0 -> (-1, 0)
        Robot 2 at (2, 0) moving east at 1.0 -> (1, 0)
        r = (-2, 0), v_rel = (-2, 0).
        -dot(r, v_rel) = -(-2 * -2) = -4.
        t_unclipped = -4 / 4 = -1.0 s.
        Since t < 0, closest approach was in the past; t* should be clipped to 0.0.
        d* = current distance = 2.0.
        rho = 0.0.
        """
        p1 = np.array([0.0, 0.0])
        v1 = np.array([-1.0, 0.0])
        p2 = np.array([2.0, 0.0])
        v2 = np.array([1.0, 0.0])

        pred = predict_closest_approach(p1, v1, p2, v2, self.d_safe, horizon_T=self.T)
        self.assertAlmostEqual(pred.time_to_closest_approach, 0.0, places=5)
        self.assertAlmostEqual(pred.predicted_minimum_distance, 2.0, places=5)
        self.assertAlmostEqual(pred.collision_risk, 0.0, places=5)

    def test_d5_stationary_entities(self):
        """
        Stationary other entity:
        Robot 1 at (0, 0) moving east at 1.0 -> (1, 0)
        Entity at (1.5, 0) stationary -> (0, 0)
        r = (-1.5, 0), v_rel = (1, 0).
        t* = 1.5 s, d* = 0.0.
        """
        p1 = np.array([0.0, 0.0])
        v1 = np.array([1.0, 0.0])
        p2 = np.array([1.5, 0.0])
        v2 = np.array([0.0, 0.0])

        pred = predict_closest_approach(p1, v1, p2, v2, self.d_safe, horizon_T=self.T)
        self.assertAlmostEqual(pred.time_to_closest_approach, 1.5, places=5)
        self.assertAlmostEqual(pred.predicted_minimum_distance, 0.0, places=5)
        self.assertAlmostEqual(pred.collision_risk, 1.0, places=5)

    def test_d6_near_zero_relative_velocity(self):
        """
        Both entities moving at same velocity:
        Robot 1 at (0, 0) with v = (1, 0)
        Robot 2 at (2, 0) with v = (1, 0)
        v_rel = (0, 0).
        Safely returns t* = 0.0, d* = 2.0, rho = 0.0 without division by zero.
        """
        p1 = np.array([0.0, 0.0])
        v1 = np.array([1.0, 0.0])
        p2 = np.array([2.0, 0.0])
        v2 = np.array([1.0, 0.0])

        pred = predict_closest_approach(p1, v1, p2, v2, self.d_safe, horizon_T=self.T)
        self.assertAlmostEqual(pred.time_to_closest_approach, 0.0, places=5)
        self.assertAlmostEqual(pred.predicted_minimum_distance, 2.0, places=5)
        self.assertAlmostEqual(pred.collision_risk, 0.0, places=5)


if __name__ == "__main__":
    unittest.main()

