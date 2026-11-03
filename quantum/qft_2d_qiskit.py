import numpy as np

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


def apply_qft_1d(circuit, qubits):
    """
    Apply a QFT using Qiskit's little-endian convention.

    The supplied qubits represent the logical register from
    least-significant to most-significant bit.

    The circuit implements the standard QFT convention used
    by Qiskit's computational-basis indexing.
    """

    n = len(qubits)

    # QFT circuit in little-endian ordering.
    for j in range(n):

        # Hadamard on the current qubit.
        circuit.h(qubits[j])

        # Controlled phase rotations.
        for k in range(j + 1, n):

            angle = np.pi / (
                2 ** (k - j)
            )

            circuit.cp(
                angle,
                qubits[k],
                qubits[j]
            )

    # Reverse the logical qubit order.
    for j in range(n // 2):

        circuit.swap(
            qubits[j],
            qubits[n - j - 1]
        )

    return circuit


def apply_inverse_qft_1d(circuit, qubits):
    """
    Apply the inverse of apply_qft_1d().
    """

    n = len(qubits)

    # Undo final swaps.
    for j in range(n // 2):

        circuit.swap(
            qubits[j],
            qubits[n - j - 1]
        )

    # Reverse the phase rotations and Hadamards.
    for j in reversed(range(n)):

        for k in reversed(range(j + 1, n)):

            angle = -np.pi / (
                2 ** (k - j)
            )

            circuit.cp(
                angle,
                qubits[k],
                qubits[j]
            )

        circuit.h(
            qubits[j]
        )

    return circuit


def apply_qft_2d(circuit):
    """
    Apply separable 2-D QFT.

    Qubits 0-3 -> X dimension.
    Qubits 4-7 -> Y dimension.

    Qiskit uses little-endian statevector ordering, so the
    logical qubit order is supplied in reverse.
    """

    x_qubits = [3, 2, 1, 0]
    y_qubits = [7, 6, 5, 4]

    apply_qft_1d(
        circuit,
        x_qubits
    )

    apply_qft_1d(
        circuit,
        y_qubits
    )

    return circuit


def apply_inverse_qft_2d(circuit):
    """
    Apply inverse separable 2-D QFT using the same logical
    qubit ordering as the forward transform.
    """

    x_qubits = [3, 2, 1, 0]
    y_qubits = [7, 6, 5, 4]

    apply_inverse_qft_1d(
        circuit,
        y_qubits
    )

    apply_inverse_qft_1d(
        circuit,
        x_qubits
    )

    return circuit


def encode_field_into_circuit(field):
    """
    Amplitude-encode a 16x16 field into 8 qubits.
    """

    field = np.asarray(
        field,
        dtype=float
    )

    if field.shape != (16, 16):

        raise ValueError(
            f"Expected 16x16 field, got {field.shape}"
        )

    amplitudes = field.flatten()

    norm = np.linalg.norm(
        amplitudes
    )

    if norm < 1e-12:

        raise ValueError(
            "Field norm is too small."
        )

    amplitudes = (
        amplitudes / norm
    )

    circuit = QuantumCircuit(8)

    circuit.initialize(
        amplitudes,
        range(8)
    )

    return circuit


def get_statevector(circuit):

    return Statevector.from_instruction(
        circuit
    )


def statevector_to_field(statevector):

    amplitudes = np.asarray(
        statevector.data,
        dtype=complex
    )

    if len(amplitudes) != 256:

        raise ValueError(
            f"Expected 256 amplitudes, got {len(amplitudes)}"
        )

    return amplitudes.reshape(
        16,
        16
    )