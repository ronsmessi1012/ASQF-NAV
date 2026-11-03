import numpy as np

from environment.field import generate_field
from quantum.amplitude_encoding import normalize_field


# ============================================================
# 2-D QFT
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

    return result / np.sqrt(n)


def qft_2d(field):
    after_x = np.zeros_like(field, dtype=complex)

    for y in range(field.shape[0]):
        after_x[y, :] = qft_1d(field[y, :])

    result = np.zeros_like(field, dtype=complex)

    for x in range(field.shape[1]):
        result[:, x] = qft_1d(after_x[:, x])

    return result


# ============================================================
# SPECTRAL METRICS
# ============================================================

def spectral_metrics(spectrum):

    power = np.abs(spectrum) ** 2

    total_power = np.sum(power)

    n = spectrum.shape[0]

    frequencies = np.fft.fftfreq(n) * n

    kx, ky = np.meshgrid(
        frequencies,
        frequencies,
        indexing="xy"
    )

    radius = np.sqrt(
        kx ** 2 + ky ** 2
    )

    max_radius = np.max(radius)

    # --------------------------------------------------------
    # 1. OLD HIGH-FREQUENCY RATIO
    # --------------------------------------------------------

    high_frequency_threshold = 4.0

    high_mask = (
        radius > high_frequency_threshold
    )

    old_complexity = (
        np.sum(power[high_mask])
        /
        total_power
    )

    # --------------------------------------------------------
    # 2. SPECTRAL CENTROID
    # --------------------------------------------------------

    spectral_centroid = (
        np.sum(radius * power)
        /
        total_power
    )

    normalized_centroid = (
        spectral_centroid
        /
        max_radius
    )

    # --------------------------------------------------------
    # 3. SPECTRAL SECOND MOMENT
    # --------------------------------------------------------

    second_moment = (
        np.sum(
            (radius ** 2) * power
        )
        /
        total_power
    )

    normalized_second_moment = (
        second_moment
        /
        (max_radius ** 2)
    )

    # --------------------------------------------------------
    # 4. SPECTRAL RMS FREQUENCY
    # --------------------------------------------------------

    rms_frequency = np.sqrt(
        second_moment
    )

    normalized_rms = (
        rms_frequency
        /
        max_radius
    )

    # --------------------------------------------------------
    # 5. SPECTRAL ENTROPY
    # --------------------------------------------------------

    probabilities = (
        power
        /
        total_power
    )

    nonzero = (
        probabilities > 1e-15
    )

    entropy = -np.sum(
        probabilities[nonzero]
        *
        np.log(
            probabilities[nonzero]
        )
    )

    max_entropy = np.log(
        len(probabilities)
    )

    normalized_entropy = (
        entropy
        /
        max_entropy
    )

    return {
        "old_complexity": old_complexity,
        "centroid": normalized_centroid,
        "second_moment": normalized_second_moment,
        "rms": normalized_rms,
        "entropy": normalized_entropy
    }


# ============================================================
# ENVIRONMENT ANALYSIS
# ============================================================

def analyze_environment(
    name,
    agents,
    goals,
    obstacles
):

    _, _, field = generate_field(
        agents,
        goals,
        obstacles,
        current_agent=0
    )

    normalized = normalize_field(field)

    field_2d = normalized.reshape(
        16,
        16
    )

    spectrum = qft_2d(
        field_2d
    )

    metrics = spectral_metrics(
        spectrum
    )

    return metrics


# ============================================================
# CASE A — SIMPLE
# ============================================================

case_a = analyze_environment(

    "SIMPLE",

    agents=[
        (2.0, 2.0)
    ],

    goals=[
        (13.0, 13.0)
    ],

    obstacles=[]
)


# ============================================================
# CASE B — MODERATE
# ============================================================

case_b = analyze_environment(

    "MODERATE",

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
# CASE C — COMPLEX
# ============================================================

case_c = analyze_environment(

    "COMPLEX",

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
# PRINT RESULTS
# ============================================================

results = [
    ("SIMPLE", case_a),
    ("MODERATE", case_b),
    ("COMPLEX", case_c)
]

print("\n")
print("=" * 105)
print("                    SPECTRAL COMPLEXITY COMPARISON")
print("=" * 105)

print(
    f"{'Environment':<15}"
    f"{'Old C':>14}"
    f"{'Centroid':>14}"
    f"{'2nd Moment':>14}"
    f"{'RMS':>14}"
    f"{'Entropy':>14}"
)

print("-" * 105)

for name, metrics in results:

    print(
        f"{name:<15}"
        f"{metrics['old_complexity']:>14.6f}"
        f"{metrics['centroid']:>14.6f}"
        f"{metrics['second_moment']:>14.6f}"
        f"{metrics['rms']:>14.6f}"
        f"{metrics['entropy']:>14.6f}"
    )

print("=" * 105)


# ============================================================
# MONOTONICITY CHECK
# ============================================================

print("\n========== MONOTONICITY CHECK ==========")

metric_names = [
    "old_complexity",
    "centroid",
    "second_moment",
    "rms",
    "entropy"
]

for metric in metric_names:

    values = [
        case_a[metric],
        case_b[metric],
        case_c[metric]
    ]

    increasing = (
        values[0]
        <= values[1]
        <= values[2]
    )

    print(
        f"{metric:<20}: "
        f"{'MONOTONIC ↑' if increasing else 'NOT MONOTONIC'}"
    )

    print(
        "    ",
        values
    )


print("\n========================================")
print("       SPECTRAL METRIC TEST COMPLETE")
print("========================================")