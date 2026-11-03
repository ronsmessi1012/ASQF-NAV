import numpy as np

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from quantum.qft_2d_qiskit import (
    apply_qft_1d,
    apply_inverse_qft_1d
)


def mathematical_qft(values):
    values = np.asarray(
        values,
        dtype=complex
    )

    n = len(values)

    x = np.arange(n)
    k = np.arange(n)

    matrix = (
        np.exp(
            2j * np.pi
            * np.outer(k, x)
            / n
        )
        / np.sqrt(n)
    )

    return matrix @ values


def main():

    print()
    print("=" * 80)
    print("              QISKIT 1-D QFT ISOLATION TEST")
    print("=" * 80)

    # --------------------------------------------------------
    # Deliberately asymmetric 16-element input
    # --------------------------------------------------------

    values = np.zeros(
        16,
        dtype=complex
    )

    values[1] = 0.7
    values[4] = 0.2
    values[7] = 0.5
    values[13] = 0.9

    values /= np.linalg.norm(values)

    # --------------------------------------------------------
    # Mathematical reference
    # --------------------------------------------------------

    reference = mathematical_qft(
        values
    )

    # --------------------------------------------------------
    # Qiskit QFT
    # --------------------------------------------------------

    circuit = QuantumCircuit(4)

    circuit.initialize(
        values,
        range(4)
    )

    apply_qft_1d(
    circuit,
    [3, 2, 1, 0]
)

    state = Statevector.from_instruction(
        circuit
    )

    qiskit_result = np.asarray(
        state.data
    )

    # --------------------------------------------------------
    # Compare complex spectrum
    # --------------------------------------------------------

    complex_error = np.linalg.norm(
        reference
        -
        qiskit_result
    )

    magnitude_error = np.linalg.norm(
        np.abs(reference)
        -
        np.abs(qiskit_result)
    )

    power_error = np.linalg.norm(
        np.abs(reference) ** 2
        -
        np.abs(qiskit_result) ** 2
    )

    print()
    print(
        "Complex-spectrum error :",
        complex_error
    )

    print(
        "Magnitude error         :",
        magnitude_error
    )

    print(
        "Power-spectrum error    :",
        power_error
    )

    # --------------------------------------------------------
    # Inverse QFT
    # --------------------------------------------------------

    reconstruction = QuantumCircuit(4)

    reconstruction.initialize(
        values,
        range(4)
    )

    apply_qft_1d(
    reconstruction,
    [3, 2, 1, 0]
)

    apply_inverse_qft_1d(
    reconstruction,
    [3, 2, 1, 0]
)

    reconstructed = (
        Statevector.from_instruction(
            reconstruction
        ).data
    )

    reconstruction_error = np.linalg.norm(
        values
        -
        reconstructed
    )

    print()
    print(
        "QFT -> IQFT error      :",
        reconstruction_error
    )

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    print()
    print("=" * 80)

    if (
        magnitude_error < 1e-10
        and power_error < 1e-10
        and reconstruction_error < 1e-10
    ):

        print(
            "RESULT: PASS"
        )

    else:

        print(
            "RESULT: FAIL"
        )

    print("=" * 80)


if __name__ == "__main__":
    main()