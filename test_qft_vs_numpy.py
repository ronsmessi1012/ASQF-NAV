import numpy as np

from environment.field import generate_field
from quantum.amplitude_encoding import encode_field
from quantum.qft import apply_qft

from qiskit.quantum_info import Statevector


# ============================================================
# SAME ENVIRONMENT AS OUR EXISTING EXPERIMENTS
# ============================================================

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


# ============================================================
# GENERATE FIELD
# ============================================================

X, Y, field = generate_field(
    agents,
    goals,
    obstacles,
    current_agent=0
)

print("\n========== FIELD ==========")
print("Field shape:", field.shape)
print("Field norm:", np.linalg.norm(field))


# ============================================================
# NORMALIZED FIELD
# ============================================================

vector = field.flatten().astype(float)
normalized_field = vector / np.linalg.norm(vector)

print("\n========== NORMALIZED FIELD ==========")
print("Number of amplitudes:", len(normalized_field))
print("Norm:", np.linalg.norm(normalized_field))


# ============================================================
# NUMPY FFT
# ============================================================

numpy_fft = np.fft.fft(normalized_field)

numpy_power = np.abs(numpy_fft) ** 2

print("\n========== NUMPY FFT ==========")
print("Spectrum norm:", np.linalg.norm(numpy_fft))
print("Total spectral power:", np.sum(numpy_power))
print("Maximum spectral power:", np.max(numpy_power))


# ============================================================
# QISKIT QFT
# ============================================================

circuit, amplitudes = encode_field(field)

apply_qft(
    circuit,
    circuit.num_qubits
)

qft_state = Statevector.from_instruction(circuit)

qiskit_fft = np.asarray(qft_state.data)

qiskit_power = np.abs(qiskit_fft) ** 2

print("\n========== QISKIT QFT ==========")
print("Spectrum norm:", np.linalg.norm(qiskit_fft))
print("Total spectral power:", np.sum(qiskit_power))
print("Maximum spectral power:", np.max(qiskit_power))


# ============================================================
# DIRECT COMPARISON
# ============================================================

print("\n========== COMPARISON ==========")

print(
    "NumPy spectrum norm:",
    np.linalg.norm(numpy_fft)
)

print(
    "Qiskit spectrum norm:",
    np.linalg.norm(qiskit_fft)
)

print(
    "NumPy total power:",
    np.sum(numpy_power)
)

print(
    "Qiskit total power:",
    np.sum(qiskit_power)
)


# ============================================================
# MAGNITUDE COMPARISON
# ============================================================

numpy_magnitude = np.abs(numpy_fft)
qiskit_magnitude = np.abs(qiskit_fft)

sorted_numpy = np.sort(numpy_magnitude)[::-1]
sorted_qiskit = np.sort(qiskit_magnitude)[::-1]

print("\n========== TOP 10 MAGNITUDES ==========")

print("\nNumPy:")
print(sorted_numpy[:10])

print("\nQiskit:")
print(sorted_qiskit[:10])


# ============================================================
# RAW COMPONENT COMPARISON
# ============================================================

print("\n========== FIRST 10 COMPONENTS ==========")

for i in range(10):

    print(
        f"{i:3d} | "
        f"NumPy = {numpy_fft[i]: .6f} | "
        f"Qiskit = {qiskit_fft[i]: .6f}"
    )


# ============================================================
# CHECK IF MAGNITUDES CAN BE MATCHED BY REORDERING
# ============================================================

magnitude_difference = np.linalg.norm(
    np.sort(numpy_magnitude)
    -
    np.sort(qiskit_magnitude)
)

print("\n========== ORDER-INDEPENDENT COMPARISON ==========")

print(
    "Difference between sorted magnitudes:",
    magnitude_difference
)


# ============================================================
# FIND BEST CIRCULAR SHIFT
# ============================================================

best_shift = None
best_error = float("inf")

for shift in range(len(numpy_fft)):

    shifted = np.roll(
        qiskit_magnitude,
        shift
    )

    error = np.linalg.norm(
        numpy_magnitude - shifted
    )

    if error < best_error:
        best_error = error
        best_shift = shift

print("\n========== BEST CIRCULAR SHIFT ==========")

print("Best shift:", best_shift)
print("Magnitude error:", best_error)


# ============================================================
# QFT -> IQFT RECONSTRUCTION CHECK
# ============================================================

from quantum.reconstruction import inverse_qft_state

reconstructed = inverse_qft_state(
    qiskit_fft
)

reconstruction_error = np.linalg.norm(
    normalized_field - reconstructed
)

print("\n========== QFT → IQFT ==========")

print(
    "Reconstruction error:",
    reconstruction_error
)

print(
    "Maximum imaginary component:",
    np.max(np.abs(np.imag(reconstructed)))
)


print("\n========================================")
print("        DIAGNOSTIC COMPLETE")
print("========================================")