import numpy as np

from environment.field import generate_field
from quantum.amplitude_encoding import normalize_field


# ============================================================
# 2-D QFT UTILITIES
# ============================================================

def qft_1d(vector):
    vector = np.asarray(vector, dtype=complex)
    n = len(vector)

    result = np.zeros(n, dtype=complex)

    for k in range(n):
        for x in range(n):
            result[k] += (
                vector[x]
                * np.exp(2j * np.pi * x * k / n)
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
                * np.exp(-2j * np.pi * x * k / n)
            )

    result /= np.sqrt(n)

    return result


def qft_2d(field):
    after_x = np.zeros_like(field, dtype=complex)

    for y in range(field.shape[0]):
        after_x[y, :] = qft_1d(field[y, :])

    result = np.zeros_like(field, dtype=complex)

    for x in range(field.shape[1]):
        result[:, x] = qft_1d(after_x[:, x])

    return result


def iqft_2d(spectrum):
    after_y = np.zeros_like(spectrum, dtype=complex)

    for x in range(spectrum.shape[1]):
        after_y[:, x] = iqft_1d(spectrum[:, x])

    result = np.zeros_like(spectrum, dtype=complex)

    for y in range(spectrum.shape[0]):
        result[y, :] = iqft_1d(after_y[y, :])

    return result


# ============================================================
# ADAPTIVE SPECTRAL ANALYSIS
# ============================================================

def analyze_environment(
    name,
    agents,
    goals,
    obstacles
):

    print("\n")
    print("=" * 65)
    print(f"                 {name}")
    print("=" * 65)

    # --------------------------------------------------------
    # Generate field
    # --------------------------------------------------------

    _, _, field = generate_field(
        agents,
        goals,
        obstacles,
        current_agent=0
    )

    normalized = normalize_field(field)
    field_2d = normalized.reshape(16, 16)

    # --------------------------------------------------------
    # 2-D QFT
    # --------------------------------------------------------

    spectrum = qft_2d(field_2d)

    power = np.abs(spectrum) ** 2

    total_power = np.sum(power)

    # --------------------------------------------------------
    # Frequency grid
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Spectral complexity
    # --------------------------------------------------------

    HIGH_FREQUENCY_THRESHOLD = 4.0

    high_frequency_mask = (
        radial_frequency
        >
        HIGH_FREQUENCY_THRESHOLD
    )

    high_frequency_power = np.sum(
        power[high_frequency_mask]
    )

    complexity = (
        high_frequency_power
        /
        total_power
    )

    # --------------------------------------------------------
    # Adaptive bandwidth
    # --------------------------------------------------------

    MIN_BANDWIDTH = 0.25
    MAX_BANDWIDTH = 0.75

    bandwidth = (
        MIN_BANDWIDTH
        +
        (
            MAX_BANDWIDTH
            -
            MIN_BANDWIDTH
        )
        *
        (1.0 - complexity)
    )

    max_radial_frequency = np.sqrt(
        8 ** 2 + 8 ** 2
    )

    cutoff_radius = (
        bandwidth
        *
        max_radial_frequency
    )

    # --------------------------------------------------------
    # Filter
    # --------------------------------------------------------

    filter_mask = (
        radial_frequency
        <=
        cutoff_radius
    )

    filtered_spectrum = spectrum.copy()

    filtered_spectrum[
        ~filter_mask
    ] = 0.0

    retained_power = np.sum(
        np.abs(filtered_spectrum) ** 2
    )

    retained_power_fraction = (
        retained_power
        /
        total_power
    )

    # --------------------------------------------------------
    # Reconstruction
    # --------------------------------------------------------

    reconstructed = iqft_2d(
        filtered_spectrum
    )

    reconstruction_error = (
        np.linalg.norm(
            field_2d
            -
            reconstructed
        )
        /
        np.linalg.norm(field_2d)
    )

    max_imaginary = np.max(
        np.abs(
            np.imag(reconstructed)
        )
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\nEnvironment:")
    print("  Agents:", len(agents))
    print("  Obstacles:", len(obstacles))

    print("\nSpectral quantities:")

    print(
        "  Spectral complexity:",
        complexity
    )

    print(
        "  Complexity (%):",
        complexity * 100
    )

    print(
        "  Adaptive bandwidth:",
        bandwidth
    )

    print(
        "  Cutoff radius:",
        cutoff_radius
    )

    print(
        "  Retained power fraction:",
        retained_power_fraction
    )

    print(
        "  Retained components:",
        np.sum(filter_mask),
        "/",
        256
    )

    print(
        "  Reconstruction error:",
        reconstruction_error
    )

    print(
        "  Reconstruction error (%):",
        reconstruction_error * 100
    )

    print(
        "  Maximum imaginary component:",
        max_imaginary
    )

    return {
        "name": name,
        "complexity": complexity,
        "bandwidth": bandwidth,
        "cutoff": cutoff_radius,
        "retained_power": retained_power_fraction,
        "reconstruction_error": reconstruction_error,
        "imaginary": max_imaginary
    }


# ============================================================
# CASE A — SIMPLE ENVIRONMENT
# ============================================================

case_a = analyze_environment(

    name="CASE A — SIMPLE",

    agents=[
        (2.0, 2.0)
    ],

    goals=[
        (13.0, 13.0)
    ],

    obstacles=[]
)


# ============================================================
# CASE B — MODERATE ENVIRONMENT
# ============================================================

case_b = analyze_environment(

    name="CASE B — MODERATE",

    agents=[
        (2.0, 2.0),
        (3.0, 13.0)
    ],

    goals=[
        (13.0, 13.0),
        (13.0, 2.0)
    ],

    obstacles=[
        (8.0, 9.0),
        (5.0, 11.0)
    ]
)


# ============================================================
# CASE C — COMPLEX ENVIRONMENT
# ============================================================

case_c = analyze_environment(

    name="CASE C — COMPLEX",

    agents=[
        (2.0, 2.0),
        (3.0, 13.0),
        (13.0, 3.0),
        (12.0, 12.0)
    ],

    goals=[
        (13.0, 13.0),
        (13.0, 2.0),
        (2.0, 13.0),
        (2.0, 2.0)
    ],

    obstacles=[
        (4.0, 4.0),
        (5.0, 8.0),
        (8.0, 5.0),
        (8.0, 9.0),
        (11.0, 6.0),
        (6.0, 12.0),
        (10.0, 11.0),
        (12.0, 8.0)
    ]
)


# ============================================================
# SUMMARY
# ============================================================

results = [
    case_a,
    case_b,
    case_c
]

print("\n\n")
print("=" * 80)
print("                       FINAL SUMMARY")
print("=" * 80)

print(
    f"{'Case':<25}"
    f"{'Complexity':>15}"
    f"{'Bandwidth':>15}"
    f"{'Retained':>15}"
    f"{'Error':>15}"
)

print("-" * 80)

for result in results:

    print(
        f"{result['name']:<25}"
        f"{result['complexity']:>15.6f}"
        f"{result['bandwidth']:>15.6f}"
        f"{result['retained_power']:>15.6f}"
        f"{result['reconstruction_error']:>15.6f}"
    )

print("=" * 80)