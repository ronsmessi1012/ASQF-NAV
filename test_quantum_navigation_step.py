import numpy as np

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from environment.field import generate_field

from quantum.amplitude_encoding import encode_field

from quantum.qft_2d_qiskit import (
    apply_qft_2d,
    apply_inverse_qft_2d
)

from quantum.spectral_analysis_2d import (
    radial_frequency_grid,
    spectral_second_moment,
    normalize_complexity,
    iqft_2d
)

from quantum.adaptive_filter_2d import (
    adaptive_2d_filter
)

from navigation.controller import navigation_step


# ============================================================
# CONVERT QISKIT STATEVECTOR → 16 × 16 ARRAY
# ============================================================

def statevector_to_field(statevector):

    amplitudes = np.asarray(
        statevector.data,
        dtype=complex
    )

    if len(amplitudes) != 256:
        raise ValueError(
            f"Expected 256 amplitudes, got {len(amplitudes)}"
        )

    return amplitudes.reshape(16, 16)


# ============================================================
# QUANTUM FIELD RECONSTRUCTION
# ============================================================

def quantum_reconstruct_field(field):

    # --------------------------------------------------------
    # Normalize input field
    # --------------------------------------------------------

    field_norm = np.linalg.norm(field)

    if field_norm < 1e-12:
        raise ValueError(
            "Field norm is too small."
        )

    normalized_field = (
        field / field_norm
    )

    # --------------------------------------------------------
    # Amplitude encoding
    # --------------------------------------------------------

    qft_circuit, _ = encode_field(
        normalized_field
    )

    # --------------------------------------------------------
    # Actual 2-D QFT
    # --------------------------------------------------------

    qft_circuit = apply_qft_2d(
        qft_circuit
    )

    qft_state = Statevector.from_instruction(
        qft_circuit
    )

    # --------------------------------------------------------
    # Convert QFT state → 16 × 16 spectrum
    # --------------------------------------------------------

    spectrum = statevector_to_field(
        qft_state
    )

    # --------------------------------------------------------
    # Spectral analysis
    # --------------------------------------------------------

    radial_frequency = radial_frequency_grid(
        field.shape
    )

    complexity = spectral_second_moment(
        spectrum
    )

    normalized_complexity = normalize_complexity(
        complexity
    )

    # --------------------------------------------------------
    # Adaptive spectral filtering
    # --------------------------------------------------------

    filter_result = adaptive_2d_filter(
        spectrum,
        radial_frequency,
        normalized_complexity
    )

    filtered_spectrum = (
        filter_result["filtered_spectrum"]
    )

    # --------------------------------------------------------
    # Normalize filtered complex state
    # --------------------------------------------------------

    filtered_state = (
        filtered_spectrum.flatten()
    )

    filtered_norm = np.linalg.norm(
        filtered_state
    )

    if filtered_norm < 1e-12:
        raise ValueError(
            "Filtered spectrum norm is too small."
        )

    filtered_state = (
        filtered_state / filtered_norm
    )

    # --------------------------------------------------------
    # Load filtered complex spectrum
    # into Qiskit
    # --------------------------------------------------------

    filtered_circuit = QuantumCircuit(8)

    filtered_circuit.initialize(
        filtered_state,
        range(8)
    )

    # --------------------------------------------------------
    # Actual inverse 2-D QFT
    # --------------------------------------------------------

    filtered_circuit = apply_inverse_qft_2d(
        filtered_circuit
    )

    reconstructed_state = (
        Statevector.from_instruction(
            filtered_circuit
        )
    )

    # --------------------------------------------------------
    # Convert reconstructed state
    # back to spatial field
    # --------------------------------------------------------

    reconstructed_field = (
        statevector_to_field(
            reconstructed_state
        )
    )

    # Remove numerical imaginary noise
    reconstructed_field = np.real(
        reconstructed_field
    )

    # --------------------------------------------------------
    # Restore original field scale
    # --------------------------------------------------------

    reconstructed_norm = np.linalg.norm(
        reconstructed_field
    )

    if reconstructed_norm < 1e-12:
        raise ValueError(
            "Reconstructed field norm is too small."
        )

    reconstructed_field = (
        reconstructed_field /
        reconstructed_norm
    )

    reconstructed_field = (
        reconstructed_field *
        field_norm
    )

    return (
        reconstructed_field,
        complexity,
        normalized_complexity,
        filter_result
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 100)
    print(
        "STEP 9W — ONE NAVIGATION STEP COMPARISON"
    )
    print("=" * 100)

    # --------------------------------------------------------
    # Environment
    # --------------------------------------------------------

    agents = [
        (2.0, 2.0),
        (3.0, 13.0)
    ]

    goals = [
        (13.0, 13.0),
        (13.0, 2.0)
    ]

    obstacles = [
        (8.0, 9.0),
        (5.0, 11.0)
    ]

    # --------------------------------------------------------
    # Generate classical field
    # --------------------------------------------------------

    X, Y, classical_field = generate_field(
        agents,
        goals,
        obstacles,
        current_agent=0
    )

    # --------------------------------------------------------
    # Quantum reconstruction
    # --------------------------------------------------------

    (
        quantum_field,
        complexity,
        normalized_complexity,
        filter_result
    ) = quantum_reconstruct_field(
        classical_field
    )

    # --------------------------------------------------------
    # Adaptive filtering information
    # --------------------------------------------------------

    bandwidth = (
        filter_result["bandwidth"]
    )

    retained_power = (
        filter_result[
            "retained_power_fraction"
        ]
    )

    # --------------------------------------------------------
    # Field reconstruction error
    # --------------------------------------------------------

    field_error = (
        np.linalg.norm(
            classical_field -
            quantum_field
        )
        /
        np.linalg.norm(
            classical_field
        )
    )

    # --------------------------------------------------------
    # Navigation state
    # --------------------------------------------------------

    start_position = np.array(
        [2.0, 2.0]
    )

    goal_position = np.array(
        [13.0, 13.0]
    )

    # --------------------------------------------------------
    # Classical navigation step
    # --------------------------------------------------------

    classical_position, classical_velocity = (
        navigation_step(
            start_position,
            goal_position,
            classical_field,
            other_agents=None
        )
    )

    # --------------------------------------------------------
    # Quantum reconstructed-field step
    # --------------------------------------------------------

    quantum_position, quantum_velocity = (
        navigation_step(
            start_position,
            goal_position,
            quantum_field,
            other_agents=None
        )
    )

    # --------------------------------------------------------
    # Compare velocities
    # --------------------------------------------------------

    velocity_difference = np.linalg.norm(
        classical_velocity -
        quantum_velocity
    )

    # --------------------------------------------------------
    # Direction cosine
    # --------------------------------------------------------

    classical_velocity_norm = np.linalg.norm(
        classical_velocity
    )

    quantum_velocity_norm = np.linalg.norm(
        quantum_velocity
    )

    direction_cosine = (
        np.dot(
            classical_velocity,
            quantum_velocity
        )
        /
        (
            classical_velocity_norm *
            quantum_velocity_norm
        )
    )

    # --------------------------------------------------------
    # Compare positions
    # --------------------------------------------------------

    position_difference = np.linalg.norm(
        classical_position -
        quantum_position
    )

    classical_step_distance = np.linalg.norm(
        classical_position -
        start_position
    )

    quantum_step_distance = np.linalg.norm(
        quantum_position -
        start_position
    )

    # ========================================================
    # RESULTS
    # ========================================================

    print()
    print(
        "Initial position       :",
        start_position
    )

    print(
        "Goal position          :",
        goal_position
    )

    print()
    print(
        "Spectral complexity    :",
        complexity
    )

    print(
        "Normalized complexity  :",
        normalized_complexity
    )

    print(
        "Adaptive bandwidth     :",
        bandwidth
    )

    print(
        "Retained spectral power:",
        retained_power
    )

    print()
    print(
        "Field reconstruction error:",
        field_error
    )

    print()
    print(
        "Classical velocity     :",
        classical_velocity
    )

    print(
        "Quantum velocity       :",
        quantum_velocity
    )

    print()
    print(
        "Classical new position:",
        classical_position
    )

    print(
        "Quantum new position  :",
        quantum_position
    )

    print()
    print(
        "Velocity difference   :",
        velocity_difference
    )

    print(
        "Position difference   :",
        position_difference
    )

    print()
    print(
        "Classical step length :",
        classical_step_distance
    )

    print(
        "Quantum step length   :",
        quantum_step_distance
    )

    print()
    print(
        "Direction cosine      :",
        direction_cosine
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    field_check = (
        field_error < 0.30
    )

    direction_check = (
        direction_cosine > 0.99
    )

    velocity_check = (
        velocity_difference < 0.05
    )

    position_check = (
        position_difference < 0.02
    )

    print()
    print(
        "Field reconstruction   :",
        "PASS" if field_check else "FAIL"
    )

    print(
        "Navigation direction   :",
        "PASS" if direction_check else "FAIL"
    )

    print(
        "Velocity consistency   :",
        "PASS" if velocity_check else "FAIL"
    )

    print(
        "Position consistency   :",
        "PASS" if position_check else "FAIL"
    )

    overall = (
        field_check
        and direction_check
        and velocity_check
        and position_check
    )

    print()
    print(
        "OVERALL RESULT         :",
        "PASS" if overall else "FAIL"
    )

    print("=" * 100)