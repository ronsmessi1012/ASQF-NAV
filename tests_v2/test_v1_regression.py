"""
Test G: Full v1 Regression Suite.
Runs all core existing v1 tests to guarantee zero regression and 100% reproducibility.
"""

import subprocess
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestV1Regression(unittest.TestCase):
    def run_v1_script(self, script_name: str) -> subprocess.CompletedProcess:
        script_path = PROJECT_ROOT / script_name
        self.assertTrue(script_path.exists(), f"Script {script_name} not found at {script_path}")
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True
        )
        return result

    def test_v1_spectral_module(self):
        """Regression test for 2-D spectral module."""
        res = self.run_v1_script("test_spectral_module.py")
        self.assertEqual(res.returncode, 0, f"Error: {res.stderr}")
        self.assertIn("OVERALL RESULT          : PASS", res.stdout)

    def test_v1_qiskit_2d_qft(self):
        """Regression test for Qiskit separable 2-D QFT circuit."""
        res = self.run_v1_script("test_qiskit_2d_qft.py")
        self.assertEqual(res.returncode, 0, f"Error: {res.stderr}")
        self.assertIn("OVERALL RESULT         : PASS", res.stdout)
        self.assertIn("QFT -> IQFT reconstruction error: 1.218551185428237e-15", res.stdout)

    def test_v1_qiskit_complex_iqft(self):
        """Regression test for complex filtered state IQFT inversion."""
        res = self.run_v1_script("test_qiskit_complex_iqft.py")
        self.assertEqual(res.returncode, 0, f"Error: {res.stderr}")
        self.assertIn("OVERALL RESULT         : PASS", res.stdout)

    def test_v1_quantum_navigation_step(self):
        """Regression test for Step 9W: single-step velocity & gradient consistency."""
        res = self.run_v1_script("test_quantum_navigation_step.py")
        self.assertEqual(res.returncode, 0, f"Error: {res.stderr}")
        self.assertIn("OVERALL RESULT         : PASS", res.stdout)

    def test_v1_single_agent_trajectory(self):
        """Regression test for Step 9X: single-agent closed-loop rollout."""
        res = self.run_v1_script("test_single_agent_trajectory.py")
        self.assertEqual(res.returncode, 0, f"Error: {res.stderr}")
        self.assertIn("OVERALL RESULT         : PASS", res.stdout)
        self.assertIn("Classical reached goal : PASS", res.stdout)
        self.assertIn("Quantum reached goal   : PASS", res.stdout)

    def test_v1_multi_agent_ablation(self):
        """Regression test for Step 9Y: 3-way multi-agent filter ablation benchmark."""
        res = self.run_v1_script("test_quantum_multi_agent.py")
        self.assertEqual(res.returncode, 0, f"Error: {res.stderr}")
        self.assertIn("Minimum inter-agent distance: 1.0582714713123524", res.stdout)
        self.assertIn("Minimum inter-agent distance: 0.9673522715935752", res.stdout)
        self.assertIn("The adaptive spectral filter is responsible for the observed collision-margin degradation.", res.stdout)


if __name__ == "__main__":
    unittest.main()

