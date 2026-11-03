"""
Unit Tests for Component D: Quantum Spectral Processing Pipeline on Dynamic Combined Fields.
"""

import unittest
import numpy as np

from navigation_v2.config import AQSFConfigV2
from navigation_v2.dynamic_entities import MovingObstacle
from navigation_v2.combined_field import generate_combined_field
from navigation_v2.quantum_pipeline import process_field_through_qft


class TestQuantumPipelineV2(unittest.TestCase):
    def setUp(self):
        self.config = AQSFConfigV2()
        self.goal = np.array([13.0, 13.0])
        self.static_obs = [(8.0, 9.0), (5.0, 11.0), (6.0, 10.0)]
        self.robots = np.array([[2.0, 2.0], [3.0, 13.0]])
        self.moving_obs = [
            MovingObstacle(
                id="dyn_1",
                position=np.array([7.0, 7.0]),
                velocity=np.array([0.2, -0.2]),
                radius=0.3
            )
        ]
        self.field_data = generate_combined_field(
            goal=self.goal,
            static_obstacles=self.static_obs,
            robot_positions=self.robots,
            moving_obstacles=self.moving_obs,
            current_robot_idx=0,
            config=self.config
        )
        self.field = self.field_data["F_total"]

    def test_unfiltered_qft_machine_precision(self):
        """Verify unfiltered 2D QFT + IQFT achieves machine precision on dynamic field."""
        rec_field, telemetry = process_field_through_qft(
            self.field,
            self.config,
            apply_filter=False,
            use_qiskit_circuit=True
        )
        self.assertEqual(telemetry["n_qubits"], 8)
        self.assertLess(telemetry["field_reconstruction_error"], 1e-14)
        self.assertLess(telemetry["max_imaginary_component"], 1e-15)
        self.assertAlmostEqual(telemetry["retained_power_fraction"], 1.0, places=6)

    def test_filtered_qft_pipeline(self):
        """Verify adaptive filtered QFT pipeline maintains power and computes valid bandwidth."""
        rec_field, telemetry = process_field_through_qft(
            self.field,
            self.config,
            apply_filter=True,
            use_qiskit_circuit=True
        )
        self.assertEqual(telemetry["n_qubits"], 8)
        self.assertGreaterEqual(telemetry["retained_power_fraction"], 0.95)
        self.assertLess(telemetry["max_imaginary_component"], 1e-14)
        self.assertGreater(telemetry["adaptive_bandwidth"], 0.25)
        self.assertLessEqual(telemetry["adaptive_bandwidth"], 0.75)
        self.assertGreater(telemetry["mean_gradient_cosine_similarity"], 0.95)


if __name__ == "__main__":
    unittest.main()

