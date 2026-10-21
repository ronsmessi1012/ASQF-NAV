import numpy as np

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


def apply_qft(circuit, n_qubits):
    """
    Apply the Quantum Fourier Transform to all qubits.
    """

    for j in range(n_qubits):

        # Hadamard
        circuit.h(j)

        # Controlled phase rotations
        for k in range(j + 1, n_qubits):

            angle = np.pi / (2 ** (k - j))

            circuit.cp(
                angle,
                k,
                j
            )

    # Reverse qubit order
    for j in range(n_qubits // 2):
        circuit.swap(
            j,
            n_qubits - j - 1
        )

    return circuit


def apply_inverse_qft(circuit, n_qubits):
    """
    Apply the inverse Quantum Fourier Transform.
    """

    # Reverse qubit order first
    for j in range(n_qubits // 2):
        circuit.swap(
            j,
            n_qubits - j - 1
        )

    # Inverse controlled rotations
    for j in reversed(range(n_qubits)):

        for k in reversed(range(j + 1, n_qubits)):

            angle = -np.pi / (2 ** (k - j))

            circuit.cp(
                angle,
                k,
                j
            )

        # Inverse Hadamard
        circuit.h(j)

    return circuit