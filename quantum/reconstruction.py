import numpy as np

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from quantum.qft import apply_inverse_qft


def inverse_qft_state(spectrum):
    """
    Apply inverse QFT to a spectral statevector.

    The input is the filtered spectral state.
    """

    n_amplitudes = len(spectrum)
    n_qubits = int(np.log2(n_amplitudes))

    if 2 ** n_qubits != n_amplitudes:
        raise ValueError(
            "Spectrum length must be a power of 2."
        )

    circuit = QuantumCircuit(n_qubits)

    circuit.initialize(
        spectrum,
        range(n_qubits)
    )

    apply_inverse_qft(
        circuit,
        n_qubits
    )

    reconstructed_state = Statevector.from_instruction(
        circuit
    )

    return np.asarray(reconstructed_state.data)


def reconstruct_field(
    filtered_spectrum,
    original_field_shape
):
    """
    Reconstruct the spatial field from the
    filtered spectral representation.
    """

    reconstructed_state = inverse_qft_state(
        filtered_spectrum
    )

    imaginary_magnitude = np.max(
        np.abs(np.imag(reconstructed_state))
    )

    print(
        "Maximum imaginary component:",
        imaginary_magnitude
    )

    reconstructed_vector = np.real(
        reconstructed_state
    )

    reconstructed_field = reconstructed_vector.reshape(
        original_field_shape
    )

    return reconstructed_field, reconstructed_state


def reconstruction_error(
    original_field,
    reconstructed_field
):
    """
    Compute relative L2 reconstruction error.
    """

    numerator = np.linalg.norm(
        original_field - reconstructed_field
    )

    denominator = np.linalg.norm(
        original_field
    )

    if denominator < 1e-12:
        raise ValueError(
            "Original field norm is too small."
        )

    return numerator / denominator