import numpy as np

from environment.field import generate_field

from quantum.spectral_analysis_2d import (
    qft_2d,
    iqft_2d
)

from quantum.qft_2d_qiskit import (
    encode_field_into_circuit,
    apply_qft_2d,
    apply_inverse_qft_2d,
    get_statevector,
    statevector_to_field
)


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

    # --------------------------------------------------------
    # Generate field
    # --------------------------------------------------------

    X, Y, field = generate_field(
        agents,
        goals,
        obstacles,
        current_agent=0
    )

    field = field / np.linalg.norm(field)

    print()
    print("=" * 80)
    print("              QISKIT 2-D QFT VALIDATION")
    print("=" * 80)

    print(
        "Input field shape:",
        field.shape
    )

    print(
        "Input norm:",
        np.linalg.norm(field)
    )

    # --------------------------------------------------------
    # Mathematical 2-D QFT
    # --------------------------------------------------------

    reference_spectrum = qft_2d(
        field
    )

    # --------------------------------------------------------
    # Qiskit circuit
    # --------------------------------------------------------

    circuit = encode_field_into_circuit(
        field
    )

    apply_qft_2d(
        circuit
    )

    qft_state = get_statevector(
        circuit
    )

    qiskit_spectrum = statevector_to_field(
        qft_state
    )

    # --------------------------------------------------------
    # Compare spectrum magnitudes
    # --------------------------------------------------------

    magnitude_error = np.linalg.norm(
        np.abs(reference_spectrum)
        -
        np.abs(qiskit_spectrum)
    )

    power_error = np.linalg.norm(
        np.abs(reference_spectrum) ** 2
        -
        np.abs(qiskit_spectrum) ** 2
    )

    print()
    print(
        "Magnitude error:",
        magnitude_error
    )

    print(
        "Power-spectrum error:",
        power_error
    )

    # --------------------------------------------------------
    # Qiskit QFT -> inverse QFT
    # --------------------------------------------------------

    reconstruction_circuit = (
        encode_field_into_circuit(
            field
        )
    )

    apply_qft_2d(
        reconstruction_circuit
    )

    apply_inverse_qft_2d(
        reconstruction_circuit
    )

    reconstructed_state = get_statevector(
        reconstruction_circuit
    )

    reconstructed_field = (
        statevector_to_field(
            reconstructed_state
        )
    )

    reconstruction_error = np.linalg.norm(
        field
        -
        np.real(reconstructed_field)
    )

    imaginary_error = np.max(
        np.abs(
            np.imag(
                reconstructed_field
            )
        )
    )

    print()
    print(
        "QFT -> IQFT reconstruction error:",
        reconstruction_error
    )

    print(
        "Maximum imaginary component:",
        imaginary_error
    )

    # --------------------------------------------------------
    # Checks
    # --------------------------------------------------------

    magnitude_check = (
        magnitude_error < 1e-10
    )

    power_check = (
        power_error < 1e-10
    )

    reconstruction_check = (
        reconstruction_error < 1e-10
    )

    imaginary_check = (
        imaginary_error < 1e-10
    )

    print()
    print("=" * 80)
    print("VALIDATION")
    print("=" * 80)

    print(
        "2-D QFT magnitude      :",
        "PASS" if magnitude_check else "FAIL"
    )

    print(
        "2-D spectral power     :",
        "PASS" if power_check else "FAIL"
    )

    print(
        "QFT -> IQFT            :",
        "PASS" if reconstruction_check else "FAIL"
    )

    print(
        "Imaginary component    :",
        "PASS" if imaginary_check else "FAIL"
    )

    overall = (
        magnitude_check
        and power_check
        and reconstruction_check
        and imaginary_check
    )

    print()
    print(
        "OVERALL RESULT         :",
        "PASS" if overall else "FAIL"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()