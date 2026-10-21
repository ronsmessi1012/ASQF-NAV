import numpy as np

from environment.field import generate_field
from quantum.amplitude_encoding import normalize_field


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

normalized_field = normalize_field(field)

field_2d = normalized_field.reshape(16, 16)


# ============================================================
# 3. 2-D NORMALIZED QFT
# ============================================================

def qft_1d(vector):

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


def iqft_1d(vector):

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


def qft_2d(field):

    # Transform x direction
    after_x = np.zeros_like(
        field,
        dtype=complex
    )

    for y in range(field.shape[0]):

        after_x[y, :] = qft_1d(
            field[y, :]
        )

    # Transform y direction
    result = np.zeros_like(
        field,
        dtype=complex
    )

    for x in range(field.shape[1]):

        result[:, x] = qft_1d(
            after_x[:, x]
        )

    return result


def iqft_2d(spectrum):

    # Inverse y direction
    after_y = np.zeros_like(
        spectrum,
        dtype=complex
    )

    for x in range(spectrum.shape[1]):

        after_y[:, x] = iqft_1d(
            spectrum[:, x]
        )

    # Inverse x direction
    result = np.zeros_like(
        spectrum,
        dtype=complex
    )

    for y in range(spectrum.shape[0]):

        result[y, :] = iqft_1d(
            after_y[y, :]
        )

    return result


# ============================================================
# 4. COMPUTE 2-D SPECTRUM
# ============================================================

spectrum = qft_2d(field_2d)

power = np.abs(spectrum) ** 2

print("\n========== 2-D SPECTRUM ==========")

print(
    "Spectrum norm:",
    np.linalg.norm(spectrum)
)

print(
    "Total spectral power:",
    np.sum(power)
)


# ============================================================
# 5. BUILD RADIAL FREQUENCY GRID
# ============================================================

n = 16

frequencies = np.fft.fftfreq(n) * n

kx, ky = np.meshgrid(
    frequencies,
    frequencies,
    indexing="xy"
)

radial_frequency = np.sqrt(
    kx ** 2 + ky ** 2
)

print("\n========== FREQUENCY GRID ==========")

print(
    "Minimum radial frequency:",
    np.min(radial_frequency)
)

print(
    "Maximum radial frequency:",
    np.max(radial_frequency)
)


# ============================================================
# 6. SPECTRAL COMPLEXITY
# ============================================================

# Frequencies above this radius are considered
# high-frequency components.

HIGH_FREQUENCY_THRESHOLD = 4.0

high_frequency_mask = (
    radial_frequency
    >
    HIGH_FREQUENCY_THRESHOLD
)

high_frequency_power = np.sum(
    power[high_frequency_mask]
)

total_power = np.sum(power)

complexity = (
    high_frequency_power
    /
    total_power
)

print("\n========== SPECTRAL COMPLEXITY ==========")

print(
    "High-frequency threshold:",
    HIGH_FREQUENCY_THRESHOLD
)

print(
    "High-frequency power:",
    high_frequency_power
)

print(
    "Total power:",
    total_power
)

print(
    "Spectral complexity:",
    complexity
)

print(
    "Spectral complexity (%):",
    complexity * 100
)


# ============================================================
# 7. ADAPTIVE BANDWIDTH
# ============================================================

MIN_BANDWIDTH = 0.25
MAX_BANDWIDTH = 0.75

complexity_clipped = np.clip(
    complexity,
    0.0,
    1.0
)

bandwidth = (
    MIN_BANDWIDTH
    +
    (
        MAX_BANDWIDTH
        -
        MIN_BANDWIDTH
    )
    *
    (
        1.0
        -
        complexity_clipped
    )
)

# Maximum radial frequency in a 16x16 FFT grid
MAX_RADIAL_FREQUENCY = np.sqrt(
    8 ** 2 + 8 ** 2
)

cutoff_radius = (
    bandwidth
    *
    MAX_RADIAL_FREQUENCY
)

print("\n========== ADAPTIVE BANDWIDTH ==========")

print(
    "Minimum bandwidth:",
    MIN_BANDWIDTH
)

print(
    "Maximum bandwidth:",
    MAX_BANDWIDTH
)

print(
    "Adaptive bandwidth:",
    bandwidth
)

print(
    "Maximum radial frequency:",
    MAX_RADIAL_FREQUENCY
)

print(
    "Cutoff radius:",
    cutoff_radius
)


# ============================================================
# 8. 2-D LOW-PASS FILTER
# ============================================================

filter_mask = (
    radial_frequency
    <=
    cutoff_radius
)

filtered_spectrum = (
    spectrum.copy()
)

filtered_spectrum[
    ~filter_mask
] = 0.0


# ============================================================
# 9. RETAINED POWER
# ============================================================

retained_power = np.sum(
    np.abs(filtered_spectrum) ** 2
)

retained_power_fraction = (
    retained_power
    /
    total_power
)

retained_components = np.sum(
    filter_mask
)

total_components = (
    n * n
)

print("\n========== FILTER RESULT ==========")

print(
    "Retained spectral power:",
    retained_power
)

print(
    "Retained power fraction:",
    retained_power_fraction
)

print(
    "Retained components:",
    retained_components,
    "/",
    total_components
)


# ============================================================
# 10. INVERSE 2-D QFT
# ============================================================

reconstructed = iqft_2d(
    filtered_spectrum
)


# ============================================================
# 11. RECONSTRUCTION ERROR
# ============================================================

reconstruction_error = (
    np.linalg.norm(
        field_2d
        -
        reconstructed
    )
    /
    np.linalg.norm(field_2d)
)

maximum_imaginary = np.max(
    np.abs(
        np.imag(reconstructed)
    )
)

print("\n========== RECONSTRUCTION ==========")

print(
    "Relative reconstruction error:",
    reconstruction_error
)

print(
    "Reconstruction error (%):",
    reconstruction_error * 100
)

print(
    "Maximum imaginary component:",
    maximum_imaginary
)

print(
    "Reconstructed norm:",
    np.linalg.norm(reconstructed)
)


# ============================================================
# 12. SANITY CHECK
# ============================================================

if maximum_imaginary < 1e-10:

    print(
        "\nImaginary component is "
        "numerically negligible."
    )

else:

    print(
        "\nWARNING: Significant imaginary "
        "component detected."
    )


print("\n========================================")
print("      2-D ADAPTIVE FILTER COMPLETE")
print("========================================")