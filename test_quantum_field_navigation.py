import numpy as np

from environment.field import generate_field

from navigation.controller import (
    compute_velocity
)

from quantum.qft_2d_qiskit import (
    encode_field_into_circuit,
    apply_qft_2d,
    apply_inverse_qft_2d,
    get_statevector,
    statevector_to_field
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


# ============================================================
# COMPLEX STATE ENCODING
# ============================================================

def encode_complex_state(amplitudes):

    from qiskit import QuantumCircuit

    amplitudes = np.asarray(
        amplitudes,
        dtype=complex
    )

    if amplitudes.size != 256:
        raise ValueError(
            f"Expected 256 amplitudes, got {amplitudes.size}"
        )

    norm = np.linalg.norm(amplitudes)

    if norm < 1e-12:
        raise ValueError(
            "State norm is too small."
        )

    amplitudes = amplitudes / norm

    circuit = QuantumCircuit(8)

    circuit.initialize(
        amplitudes,
        range(8)
    )

    return circuit


# ============================================================
# QUANTUM SPECTRAL PIPELINE
# ============================================================

def quantum_reconstruct_field(field):

    # --------------------------------------------------------
    # Normalize spatial field
    # --------------------------------------------------------

    field_norm = np.linalg.norm(field)

    normalized_field = (
        field / field_norm
    )

    # --------------------------------------------------------
    # QISKIT AMPLITUDE ENCODING
    # --------------------------------------------------------

    qft_circuit = encode_field_into_circuit(
        normalized_field
    )

    # --------------------------------------------------------
    # 2-D QFT
    # --------------------------------------------------------

    apply_qft_2d(
        qft_circuit
    )

    qft_state = get_statevector(
        qft_circuit
    )

    spectrum = statevector_to_field(
        qft_state
    )

    # --------------------------------------------------------
    # SPECTRAL ANALYSIS
    # --------------------------------------------------------

    radial_frequency = (
        radial_frequency_grid(
            field.shape
        )
    )

    complexity = (
        spectral_second_moment(
            spectrum
        )
    )

    normalized_complexity = (
        normalize_complexity(
            complexity
        )
    )

    # --------------------------------------------------------
    # ADAPTIVE FILTER
    # --------------------------------------------------------

    filter_result = adaptive_2d_filter(
        spectrum,
        radial_frequency,
        normalized_complexity
    )

    filtered_spectrum = (
        filter_result[
            "filtered_spectrum"
        ]
    )

    # --------------------------------------------------------
    # NORMALIZE FILTERED COMPLEX STATE
    # --------------------------------------------------------

    filtered_amplitudes = (
        filtered_spectrum.flatten()
    )

    filtered_norm = np.linalg.norm(
        filtered_amplitudes
    )

    filtered_amplitudes = (
        filtered_amplitudes /
        filtered_norm
    )

    # --------------------------------------------------------
    # RELOAD FILTERED COMPLEX STATE
    # --------------------------------------------------------

    iqft_circuit = encode_complex_state(
        filtered_amplitudes
    )

    # --------------------------------------------------------
    # 2-D IQFT
    # --------------------------------------------------------

    apply_inverse_qft_2d(
        iqft_circuit
    )

    reconstructed_state = (
        get_statevector(
            iqft_circuit
        )
    )

    reconstructed_field = (
        statevector_to_field(
            reconstructed_state
        )
    )

    # --------------------------------------------------------
    # Remove tiny numerical imaginary component
    # --------------------------------------------------------

    reconstructed_field = np.real(
        reconstructed_field
    )

    # --------------------------------------------------------
    # Restore relative field scale
    #
    # Navigation only cares about the spatial structure and
    # gradient direction, not the arbitrary quantum-state norm.
    # --------------------------------------------------------

    reconstructed_field = (
        reconstructed_field /
        np.linalg.norm(reconstructed_field)
    )

    reconstructed_field = (
        reconstructed_field
        * field_norm
    )

    return (
        reconstructed_field,
        complexity,
        filter_result["bandwidth"],
        filter_result["retained_power_fraction"]
    )


# ============================================================
# MAIN TEST
# ============================================================

def main():

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

    current_agent = 0

    # --------------------------------------------------------
    # CLASSICAL FIELD
    # --------------------------------------------------------

    X, Y, classical_field = (
        generate_field(
            agents,
            goals,
            obstacles,
            current_agent
        )
    )

    # --------------------------------------------------------
    # QUANTUM RECONSTRUCTED FIELD
    # --------------------------------------------------------

    (
        quantum_field,
        complexity,
        bandwidth,
        retained_power
    ) = quantum_reconstruct_field(
        classical_field
    )

    # --------------------------------------------------------
    # FIELD COMPARISON
    # --------------------------------------------------------

    field_error = (
        np.linalg.norm(
            classical_field
            -
            quantum_field
        )
        /
        np.linalg.norm(
            classical_field
        )
    )

    # --------------------------------------------------------
    # NAVIGATION VELOCITIES
    # --------------------------------------------------------

    position = np.array(
        agents[current_agent],
        dtype=float
    )

    goal = np.array(
        goals[current_agent],
        dtype=float
    )

    other_agents = [
        np.array(
            agents[i],
            dtype=float
        )
        for i in range(len(agents))
        if i != current_agent
    ]

    classical_velocity = compute_velocity(
        classical_field,
        position,
        goal,
        other_agents
    )

    quantum_velocity = compute_velocity(
        quantum_field,
        position,
        goal,
        other_agents
    )

    # --------------------------------------------------------
    # VELOCITY COMPARISON
    # --------------------------------------------------------

    velocity_error = (
        np.linalg.norm(
            classical_velocity
            -
            quantum_velocity
        )
    )

    classical_norm = np.linalg.norm(
        classical_velocity
    )

    quantum_norm = np.linalg.norm(
        quantum_velocity
    )

    if (
        classical_norm > 1e-12
        and quantum_norm > 1e-12
    ):
        direction_similarity = (
            np.dot(
                classical_velocity,
                quantum_velocity
            )
            /
            (
                classical_norm
                *
                quantum_norm
            )
        )
    else:
        direction_similarity = 0.0

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    print()
    print("=" * 100)
    print(
        "       QUANTUM RECONSTRUCTED FIELD → NAVIGATION TEST"
    )
    print("=" * 100)

    print()
    print(
        "Spectral complexity       :",
        complexity
    )

    print(
        "Adaptive bandwidth        :",
        bandwidth
    )

    print(
        "Retained spectral power   :",
        retained_power
    )

    print()
    print(
        "Field reconstruction error:",
        field_error
    )

    print()
    print(
        "Classical velocity        :",
        classical_velocity
    )

    print(
        "Quantum velocity          :",
        quantum_velocity
    )

    print()
    print(
        "Velocity difference       :",
        velocity_error
    )

    print(
        "Direction cosine          :",
        direction_similarity
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    field_check = (
        field_error < 0.30
    )

    direction_check = (
        direction_similarity > 0.95
    )

    print()
    print("-" * 100)

    print(
        "Field reconstruction      :",
        "PASS" if field_check else "FAIL"
    )

    print(
        "Navigation direction      :",
        "PASS" if direction_check else "FAIL"
    )

    overall = (
        field_check
        and direction_check
    )

    print()
    print(
        "OVERALL RESULT            :",
        "PASS" if overall else "FAIL"
    )

    print("=" * 100)


if __name__ == "__main__":
    main()