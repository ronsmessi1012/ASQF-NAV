import numpy as np

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from quantum.qft_2d_qiskit import apply_qft_2d


def main():

    print()
    print("=" * 80)
    print("             QISKIT 2-D QFT MAPPING TEST")
    print("=" * 80)

    # --------------------------------------------------------
    # Create a single computational-basis state.
    #
    # We deliberately choose a non-symmetric state so that
    # permutations cannot hide the ordering problem.
    # --------------------------------------------------------

    amplitudes = np.zeros(256, dtype=complex)

    test_index = 37

    amplitudes[test_index] = 1.0

    circuit = QuantumCircuit(8)

    circuit.initialize(
        amplitudes,
        range(8)
    )

    apply_qft_2d(circuit)

    state = Statevector.from_instruction(
        circuit
    )

    qiskit_state = np.asarray(
        state.data
    )

    # --------------------------------------------------------
    # Mathematical reference
    # --------------------------------------------------------

    field = amplitudes.reshape(16, 16)

    x = np.arange(16)
    y = np.arange(16)

    X, Y = np.meshgrid(
        x,
        y,
        indexing="xy"
    )

    reference = np.zeros(
        (16, 16),
        dtype=complex
    )

    for ky in range(16):
        for kx in range(16):

            phase = np.exp(
                2j
                * np.pi
                * (
                    kx * X
                    +
                    ky * Y
                )
                / 16
            )

            reference[ky, kx] = (
                np.sum(
                    field * phase
                )
                / 16
            )

    # --------------------------------------------------------
    # Print dominant Qiskit amplitudes
    # --------------------------------------------------------

    print()
    print("Input computational basis index:", test_index)

    print()
    print("Largest Qiskit amplitudes:")

    qiskit_indices = np.argsort(
        np.abs(qiskit_state)
    )[-10:][::-1]

    for index in qiskit_indices:

        print(
            f"index={index:3d}  "
            f"magnitude={abs(qiskit_state[index]):.6f}  "
            f"phase={np.angle(qiskit_state[index]): .6f}"
        )

    # --------------------------------------------------------
    # Compare magnitude arrays under several possible
    # coordinate interpretations.
    # --------------------------------------------------------

    qiskit_matrix = (
        qiskit_state.reshape(16, 16)
    )

    candidates = {
        "direct":
            qiskit_matrix,

        "transpose":
            qiskit_matrix.T,

        "flip_x":
            qiskit_matrix[:, ::-1],

        "flip_y":
            qiskit_matrix[::-1, :],

        "flip_xy":
            qiskit_matrix[::-1, ::-1],

        "transpose_flip_x":
            qiskit_matrix.T[:, ::-1],

        "transpose_flip_y":
            qiskit_matrix.T[::-1, :],

        "transpose_flip_xy":
            qiskit_matrix.T[::-1, ::-1],
    }

    print()
    print("Magnitude errors:")
    print("-" * 50)

    reference_magnitude = np.abs(
        reference
    )

    best_name = None
    best_error = float("inf")

    for name, candidate in candidates.items():

        error = np.linalg.norm(
            reference_magnitude
            -
            np.abs(candidate)
        )

        print(
            f"{name:<25} {error:.12f}"
        )

        if error < best_error:

            best_error = error
            best_name = name

    print()
    print(
        "Best mapping:",
        best_name
    )

    print(
        "Best magnitude error:",
        best_error
    )

    print("=" * 80)


if __name__ == "__main__":
    main()