import numpy as np

from environment.field import generate_field
from quantum.amplitude_encoding import normalize_field

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


# ============================================================
# 1. SAME ENVIRONMENT
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
# 2. GENERATE FIELD
# ============================================================

X, Y, field = generate_field(
    agents,
    goals,
    obstacles,
    current_agent=0
)

print("\n========== ORIGINAL FIELD ==========")
print("Shape:", field.shape)
print("Norm:", np.linalg.norm(field))


# ============================================================
# 3. NORMALIZE
# ============================================================

normalized_field = normalize_field(field)

field_2d = normalized_field.reshape(16, 16)

print("\n========== NORMALIZED FIELD ==========")
print("Shape:", field_2d.shape)
print("Norm:", np.linalg.norm(field_2d))


# ============================================================
# 4. CLASSICAL 2-D FFT
# ============================================================

fft_2d = np.fft.fft2(field_2d)

# Quantum QFT uses 1/sqrt(N) normalization.
# For a 16x16 field, N = 256.
quantum_reference = fft_2d / 16.0

print("\n========== NUMPY 2-D FFT ==========")
print("FFT norm:", np.linalg.norm(fft_2d))
print("Normalized FFT norm:", np.linalg.norm(quantum_reference))

print(
    "Total normalized spectral power:",
    np.sum(np.abs(quantum_reference) ** 2)
)


# ============================================================
# 5. 2-D QFT USING SEPARABLE TRANSFORMS
#
# Apply a 16-point QFT along rows,
# then a 16-point QFT along columns.
#
# We use NumPy here ONLY as the mathematical prototype.
# This lets us verify the 2-D formulation before building
# the corresponding 8-qubit circuit.
# ============================================================

def qft_1d_normalized(vector):
    """
    Normalized discrete Fourier transform:
        1/sqrt(N) * sum_x f(x) exp(2*pi*i*x*k/N)
    """

    vector = np.asarray(vector, dtype=complex)

    n = len(vector)

    result = np.zeros(n, dtype=complex)

    for k in range(n):

        for x in range(n):

            result[k] += (
                vector[x]
                * np.exp(
                    2j * np.pi * x * k / n
                )
            )

    result /= np.sqrt(n)

    return result


# QFT along x dimension
after_x = np.zeros_like(field_2d, dtype=complex)

for y in range(16):

    after_x[y, :] = qft_1d_normalized(
        field_2d[y, :]
    )


# QFT along y dimension
qft_2d = np.zeros_like(field_2d, dtype=complex)

for x in range(16):

    qft_2d[:, x] = qft_1d_normalized(
        after_x[:, x]
    )


# ============================================================
# 6. COMPARE WITH CLASSICAL 2-D FFT
# ============================================================

print("\n========== 2-D QFT vs 2-D FFT ==========")

print(
    "QFT norm:",
    np.linalg.norm(qft_2d)
)

print(
    "FFT reference norm:",
    np.linalg.norm(quantum_reference)
)

magnitude_error = np.linalg.norm(
    np.abs(qft_2d)
    -
    np.abs(quantum_reference)
)

complex_error = np.linalg.norm(
    qft_2d
    -
    quantum_reference
)

print(
    "Magnitude error:",
    magnitude_error
)

print(
    "Complex spectrum error:",
    complex_error
)


# ============================================================
# 7. 2-D INVERSE QFT
# ============================================================

def iqft_1d_normalized(vector):

    vector = np.asarray(vector, dtype=complex)

    n = len(vector)

    result = np.zeros(n, dtype=complex)

    for x in range(n):

        for k in range(n):

            result[x] += (
                vector[k]
                * np.exp(
                    -2j * np.pi * x * k / n
                )
            )

    result /= np.sqrt(n)

    return result


# Inverse along y
after_inverse_y = np.zeros_like(qft_2d, dtype=complex)

for x in range(16):

    after_inverse_y[:, x] = iqft_1d_normalized(
        qft_2d[:, x]
    )


# Inverse along x
reconstructed = np.zeros_like(qft_2d, dtype=complex)

for y in range(16):

    reconstructed[y, :] = iqft_1d_normalized(
        after_inverse_y[y, :]
    )


# ============================================================
# 8. RECONSTRUCTION
# ============================================================

reconstruction_error = np.linalg.norm(
    field_2d - reconstructed
)

max_imaginary = np.max(
    np.abs(np.imag(reconstructed))
)

print("\n========== 2-D QFT → IQFT ==========")

print(
    "Reconstruction error:",
    reconstruction_error
)

print(
    "Maximum imaginary component:",
    max_imaginary
)

print(
    "Reconstructed norm:",
    np.linalg.norm(reconstructed)
)


# ============================================================
# 9. SPECTRAL POWER
# ============================================================

spectral_power = np.abs(qft_2d) ** 2

print("\n========== 2-D SPECTRAL POWER ==========")

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


# ============================================================
# 10. FINAL
# ============================================================

print("\n========================================")
print("       2-D QFT PROTOTYPE COMPLETE")
print("========================================")