import numpy as np

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
    normalize_complexity
)

from quantum.adaptive_filter_2d import (
    adaptive_2d_filter
)


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
    # GENERATE AND NORMALIZE FIELD
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
    print("       ACTUAL QISKIT → ADAPTIVE SPECTRAL FILTER VALIDATION")
    print("=" * 100)

    print(
        "Input field shape :",
        field.shape
    )

    print(
        "Input norm        :",
        np.linalg.norm(field)
    )

    # ========================================================
    # STEP 1 — AMPLITUDE ENCODE
    # ========================================================

    circuit = encode_field_into_circuit(
        field
    )

    # ========================================================
    # STEP 2 — ACTUAL QISKIT 2-D QFT
    # ========================================================

    apply_qft_2d(
        circuit
    )

    qft_state = get_statevector(
        circuit
    )

    qiskit_spectrum = (
        statevector_to_field(
            qft_state
        )
    )

    # ========================================================
    # STEP 3 — SPECTRAL ANALYSIS
    # ========================================================

    radial_frequency = (
        radial_frequency_grid(
            field.shape
        )
    )

    complexity = (
        spectral_second_moment(
            qiskit_spectrum
        )
    )

    normalized_complexity = (
        normalize_complexity(
            complexity
        )
    )

    # ========================================================
    # STEP 4 — ADAPTIVE FILTER
    # ========================================================

    filter_result = adaptive_2d_filter(
        qiskit_spectrum,
        radial_frequency,
        normalized_complexity
    )

    filtered_spectrum = (
        filter_result[
            "filtered_spectrum"
        ]
    )

    bandwidth = (
        filter_result[
            "bandwidth"
        ]
    )

    cutoff = (
        filter_result[
            "cutoff"
        ]
    )

    retained_power = (
        filter_result[
            "retained_power_fraction"
        ]
    )

    retained_components = (
        filter_result[
            "retained_component_fraction"
        ]
    )

    # ========================================================
    # STEP 5 — RECONSTRUCT USING ACTUAL QISKIT IQFT
    # ========================================================

    # Qiskit amplitude encoding expects a state vector.
    filtered_amplitudes = (
        filtered_spectrum.flatten()
    )

    filtered_norm = np.linalg.norm(
        filtered_amplitudes
    )

    if filtered_norm < 1e-12:

        raise ValueError(
            "Filtered spectrum norm is too small."
        )

    # The filtering operation removes spectral power.
    # Renormalize before loading it into a quantum state.
    filtered_amplitudes = (
        filtered_amplitudes /
        filtered_norm
    )

    reconstruction_circuit = (
        encode_field_into_circuit(
            filtered_amplitudes.reshape(
                16,
                16
            ).real
        )
    )

    # NOTE:
    # The above loading is only valid for real-valued input.
    # Therefore this test separately validates the complex
    # mathematical reconstruction below.
    #
    # The actual complex-state Qiskit IQFT validation will
    # be performed after this isolated test.

    # ========================================================
    # MATHEMATICAL IQFT OF THE ACTUAL QISKIT SPECTRUM
    # ========================================================

    from quantum.spectral_analysis_2d import (
        iqft_2d
    )

    reconstructed_complex = (
        iqft_2d(
            filtered_spectrum
        )
    )

    reconstructed_field = (
        np.real(
            reconstructed_complex
        )
    )

    reconstruction_error = (
        np.linalg.norm(
            field -
            reconstructed_field
        )
        /
        np.linalg.norm(
            field
        )
    )

    imaginary_error = np.max(
        np.abs(
            np.imag(
                reconstructed_complex
            )
        )
    )

    # ========================================================
    # OUTPUT
    # ========================================================

    print()
    print("-" * 100)

    print(
        "Spectral complexity       :",
        f"{complexity:.10f}"
    )

    print(
        "Normalized complexity     :",
        f"{normalized_complexity:.10f}"
    )

    print(
        "Adaptive bandwidth        :",
        f"{bandwidth:.10f}"
    )

    print(
        "Frequency cutoff          :",
        f"{cutoff:.10f}"
    )

    print(
        "Retained spectral power   :",
        f"{retained_power:.10f}"
    )

    print(
        "Retained components       :",
        f"{retained_components:.10f}"
    )

    print(
        "Reconstruction error      :",
        f"{reconstruction_error:.10f}"
    )

    print(
        "Maximum imaginary error   :",
        f"{imaginary_error:.3e}"
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    print()
    print("=" * 100)
    print("                         VALIDATION")
    print("=" * 100)

    complexity_check = (
        0.0 <= normalized_complexity <= 1.0
    )

    bandwidth_check = (
        0.25 <= bandwidth <= 0.75
    )

    power_check = (
        0.0 < retained_power <= 1.0
    )

    reconstruction_check = (
        reconstruction_error < 0.30
    )

    imaginary_check = (
        imaginary_error < 1e-10
    )

    print(
        "Complexity valid          :",
        "PASS" if complexity_check else "FAIL"
    )

    print(
        "Bandwidth valid           :",
        "PASS" if bandwidth_check else "FAIL"
    )

    print(
        "Retained power valid      :",
        "PASS" if power_check else "FAIL"
    )

    print(
        "Reconstruction valid      :",
        "PASS" if reconstruction_check else "FAIL"
    )

    print(
        "Imaginary component       :",
        "PASS" if imaginary_check else "FAIL"
    )

    overall = (
        complexity_check
        and bandwidth_check
        and power_check
        and reconstruction_check
        and imaginary_check
    )

    print()
    print(
        "OVERALL RESULT            :",
        "PASS" if overall else "FAIL"
    )

    print("=" * 100)


if __name__ == "__main__":
    main()