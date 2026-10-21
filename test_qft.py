import numpy as np

from environment.field import generate_field
from quantum.amplitude_encoding import encode_field
from quantum.qft import apply_qft, apply_inverse_qft
from qiskit.quantum_info import Statevector


# --------------------------------------------------
# Environment
# --------------------------------------------------

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


# --------------------------------------------------
# Generate field
# --------------------------------------------------

X, Y, field = generate_field(
    agents=agents,
    goals=goals,
    obstacles=obstacles,
    current_agent=0
)


# --------------------------------------------------
# Amplitude encode
# --------------------------------------------------

circuit, original_amplitudes = encode_field(field)

n_qubits = circuit.num_qubits


# --------------------------------------------------
# Obtain original state
# --------------------------------------------------

original_state = Statevector.from_instruction(
    circuit
)


# --------------------------------------------------
# Apply QFT
# --------------------------------------------------

qft_circuit = circuit.copy()

apply_qft(
    qft_circuit,
    n_qubits
)

qft_state = Statevector.from_instruction(
    qft_circuit
)


# --------------------------------------------------
# Apply inverse QFT
# --------------------------------------------------

iqft_circuit = qft_circuit.copy()

apply_inverse_qft(
    iqft_circuit,
    n_qubits
)

reconstructed_state = Statevector.from_instruction(
    iqft_circuit
)


# --------------------------------------------------
# Compare original and reconstructed states
# --------------------------------------------------

original = np.asarray(
    original_state.data
)

reconstructed = np.asarray(
    reconstructed_state.data
)

reconstruction_error = np.linalg.norm(
    original - reconstructed
)

state_norm = np.linalg.norm(
    reconstructed
)


# --------------------------------------------------
# QFT spectrum information
# --------------------------------------------------

spectrum = np.asarray(
    qft_state.data
)

spectral_power = np.abs(spectrum) ** 2


# --------------------------------------------------
# Results
# --------------------------------------------------

print("\n========== STEP 4: QFT VALIDATION ==========")

print(
    "Number of qubits:",
    n_qubits
)

print(
    "State dimension:",
    len(original)
)

print(
    "Original state norm:",
    np.linalg.norm(original)
)

print(
    "QFT state norm:",
    np.linalg.norm(spectrum)
)

print(
    "Reconstructed state norm:",
    state_norm
)

print(
    "QFT -> IQFT reconstruction error:",
    reconstruction_error
)

print(
    "Total spectral power:",
    np.sum(spectral_power)
)

print(
    "Maximum spectral power:",
    np.max(spectral_power)
)

print(
    "Minimum spectral power:",
    np.min(spectral_power)
)