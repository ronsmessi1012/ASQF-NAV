import numpy as np

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from environment.field import generate_field

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


def encode_complex_state(amplitudes):
    """
    Load an arbitrary complex 256-element normalized state
    into an 8-qubit Qiskit circuit.
    """

    amplitudes = np.asarray(
        amplitudes,
        dtype=complex
    )

    if amplitudes.size != 256:
        raise ValueError(
            f"Expected 256 amplitudes, got {amplitudes.size}"
        )

    norm = np.linalg.norm(
        amplitudes
    )

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


def main():

    # ========================================================
    # ENVIRONMENT
    # ========================================================

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

    # ========================================================
    # GENERATE NORMALIZED FIELD
    # ========================================================

    X, Y, field = generate_field(
        agents,
        goals,
        obstacles,
        current_agent=0
    )

    field = (
        field /
        np.linalg.norm(field)
    )

    print()
    print("=" * 100)
    print("           COMPLEX FILTERED STATE → QISKIT IQFT TEST")
    print("=" * 100)

    # ========================================================
    # ACTUAL QISKIT QFT
    # ========================================================

    qft_circuit = encode_field_into_circuit(
        field
    )

    apply_qft_2d(
        qft_circuit
    )

    qft_state = get_statevector(
        qft_circuit
    )

    spectrum = statevector_to_field(
        qft_state
    )

    # ========================================================
    # SPECTRAL ANALYSIS
    # ========================================================

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

    # ========================================================
    # ADAPTIVE FILTER
    # ========================================================

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

    print()
    print(
        "Complexity:",
        complexity
    )

    print(
        "Normalized complexity:",
        normalized_complexity
    )

    print(
        "Bandwidth:",
        filter_result["bandwidth"]
    )

    print(
        "Retained power:",
        filter_result[
            "retained_power_fraction"
        ]
    )

    # ========================================================
    # NORMALIZE COMPLEX FILTERED STATE
    # ========================================================

    filtered_amplitudes = (
        filtered_spectrum.flatten()
    )

    filtered_norm = np.linalg.norm(
        filtered_amplitudes
    )

    if filtered_norm < 1e-12:
        raise ValueError(
            "Filtered state norm is too small."
        )

    filtered_amplitudes = (
        filtered_amplitudes /
        filtered_norm
    )

    print()
    print(
        "Filtered state norm:",
        np.linalg.norm(
            filtered_amplitudes
        )
    )

    # ========================================================
    # LOAD COMPLEX STATE INTO QISKIT
    # ========================================================

    iqft_circuit = encode_complex_state(
        filtered_amplitudes
    )

    # ========================================================
    # ACTUAL QISKIT 2-D IQFT
    # ========================================================

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

    # ========================================================
    # MATHEMATICAL REFERENCE
    # ========================================================

    reference_field = (
        iqft_2d(
            filtered_spectrum
        )
    )

    reference_norm = np.linalg.norm(
        reference_field
    )

    reference_field = (
        reference_field /
        reference_norm
    )

    # ========================================================
    # COMPARE QISKIT IQFT WITH MATHEMATICAL IQFT
    # ========================================================

    complex_error = np.linalg.norm(
        reference_field
        -
        reconstructed_field
    )

    magnitude_error = np.linalg.norm(
        np.abs(reference_field)
        -
        np.abs(reconstructed_field)
    )

    power_error = np.linalg.norm(
        np.abs(reference_field) ** 2
        -
        np.abs(reconstructed_field) ** 2
    )

    # ========================================================
    # OUTPUT
    # ========================================================

    print()
    print("-" * 100)

    print(
        "Complex reconstruction error :",
        complex_error
    )

    print(
        "Magnitude reconstruction error:",
        magnitude_error
    )

    print(
        "Power reconstruction error    :",
        power_error
    )

    print(
        "Qiskit reconstructed norm     :",
        np.linalg.norm(
            reconstructed_field
        )
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    complex_check = (
        complex_error < 1e-10
    )

    magnitude_check = (
        magnitude_error < 1e-10
    )

    power_check = (
        power_error < 1e-10
    )

    print()
    print("=" * 100)
    print("                         VALIDATION")
    print("=" * 100)

    print(
        "Complex IQFT agreement :",
        "PASS" if complex_check else "FAIL"
    )

    print(
        "Magnitude agreement    :",
        "PASS" if magnitude_check else "FAIL"
    )

    print(
        "Power agreement        :",
        "PASS" if power_check else "FAIL"
    )

    overall = (
        complex_check
        and magnitude_check
        and power_check
    )

    print()
    print(
        "OVERALL RESULT         :",
        "PASS" if overall else "FAIL"
    )

    print("=" * 100)


if __name__ == "__main__":
    main()