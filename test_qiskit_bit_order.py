import numpy as np

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from quantum.qft_2d_qiskit import apply_qft_2d
from quantum.spectral_analysis_2d import qft_2d


def reverse_4_bits(index):
    """
    Reverse the four low-order bits.

    Example:
        0011 -> 1100
    """

    result = 0

    for bit in range(4):

        if index & (1 << bit):

            result |= (
                1 << (3 - bit)
            )

    return result


def reverse_x_bits(field):
    """
    Reverse the 4-bit X coordinate.
    """

    result = np.zeros_like(field)

    for y in range(16):

        for x in range(16):

            reversed_x = reverse_4_bits(x)

            result[y, reversed_x] = field[y, x]

    return result


def reverse_y_bits(field):
    """
    Reverse the 4-bit Y coordinate.
    """

    result = np.zeros_like(field)

    for y in range(16):

        reversed_y = reverse_4_bits(y)

        for x in range(16):

            result[reversed_y, x] = field[y, x]

    return result


def reverse_xy_bits(field):
    """
    Reverse the 4-bit ordering of both dimensions.
    """

    return reverse_y_bits(
        reverse_x_bits(field)
    )


def main():

    # --------------------------------------------------------
    # Create a deliberately asymmetric real field.
    # --------------------------------------------------------

    field = np.zeros(
        (16, 16),
        dtype=float
    )

    field[1, 3] = 0.7
    field[4, 11] = 0.2
    field[9, 6] = 0.5
    field[13, 14] = 0.9

    field /= np.linalg.norm(field)

    print()
    print("=" * 80)
    print("             QISKIT 2-D QFT BIT-ORDER TEST")
    print("=" * 80)

    # --------------------------------------------------------
    # Mathematical reference
    # --------------------------------------------------------

    reference = qft_2d(
        field
    )

    # --------------------------------------------------------
    # Qiskit circuit
    # --------------------------------------------------------

    amplitudes = field.flatten()

    circuit = QuantumCircuit(8)

    circuit.initialize(
        amplitudes,
        range(8)
    )

    apply_qft_2d(
        circuit
    )

    state = Statevector.from_instruction(
        circuit
    )

    qiskit = state.data.reshape(
        16,
        16
    )

    # --------------------------------------------------------
    # Test possible bit-order corrections.
    # --------------------------------------------------------

    candidates = {

        "direct":
            qiskit,

        "reverse_x":
            reverse_x_bits(qiskit),

        "reverse_y":
            reverse_y_bits(qiskit),

        "reverse_x_y":
            reverse_xy_bits(qiskit),

        "transpose":
            qiskit.T,

        "transpose_reverse_x":
            reverse_x_bits(qiskit.T),

        "transpose_reverse_y":
            reverse_y_bits(qiskit.T),

        "transpose_reverse_x_y":
            reverse_xy_bits(qiskit.T),
    }

    print()
    print("Complex-spectrum errors:")
    print("-" * 60)

    best_name = None
    best_error = float("inf")

    for name, candidate in candidates.items():

        error = np.linalg.norm(
            reference
            -
            candidate
        )

        print(
            f"{name:<30} {error:.12f}"
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
        "Best complex-spectrum error:",
        best_error
    )

    print("=" * 80)


if __name__ == "__main__":
    main()