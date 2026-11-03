import numpy as np

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


def normalize_field(field):
    """
    Convert a 2D field into a normalized
    1D amplitude vector.
    """

    vector = field.flatten().astype(float)

    norm = np.linalg.norm(vector)

    if norm < 1e-12:
        raise ValueError("Field norm is too small.")

    normalized_vector = vector / norm

    return normalized_vector


def encode_field(field):
    """
    Encode a 16x16 field into an 8-qubit
    amplitude-encoded quantum state.
    """

    # --------------------------------------------------
    # Flatten field
    # --------------------------------------------------

    amplitudes = normalize_field(field)

    # --------------------------------------------------
    # Verify dimension
    # --------------------------------------------------

    if len(amplitudes) != 256:
        raise ValueError(
            f"Expected 256 amplitudes, "
            f"got {len(amplitudes)}"
        )

    # --------------------------------------------------
    # Create 8-qubit circuit
    # --------------------------------------------------

    circuit = QuantumCircuit(8)

    # Initialize quantum state
    circuit.initialize(
        amplitudes,
        range(8)
    )

    return circuit, amplitudes


def verify_encoding(field):

    circuit, amplitudes = encode_field(field)

    # Obtain exact simulated statevector
    statevector = Statevector.from_instruction(
        circuit
    )

    quantum_amplitudes = np.asarray(
        statevector.data
    )

    # --------------------------------------------------
    # Verification
    # --------------------------------------------------

    amplitude_error = np.linalg.norm(
        amplitudes - quantum_amplitudes
    )

    normalization_error = abs(
        np.linalg.norm(quantum_amplitudes) - 1.0
    )

    print("\n========== AMPLITUDE ENCODING ==========")

    print(
        "Classical field shape:",
        field.shape
    )

    print(
        "Number of amplitudes:",
        len(amplitudes)
    )

    print(
        "Number of qubits:",
        circuit.num_qubits
    )

    print(
        "Quantum state dimension:",
        len(quantum_amplitudes)
    )

    print(
        "Classical vector norm:",
        np.linalg.norm(amplitudes)
    )

    print(
        "Quantum state norm:",
        np.linalg.norm(quantum_amplitudes)
    )

    print(
        "Amplitude reconstruction error:",
        amplitude_error
    )

    print(
        "Normalization error:",
        normalization_error
    )

    return circuit, statevector