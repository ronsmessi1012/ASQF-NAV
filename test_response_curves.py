import numpy as np

from environment.field import generate_field


# ============================================================
# CALIBRATION
# ============================================================

C25 = 0.01645914
C75 = 0.01987830

MIN_BANDWIDTH = 0.25
MAX_BANDWIDTH = 0.75


# ============================================================
# 1-D NORMALIZED QFT
# ============================================================

def qft_1d(values):

    values = np.asarray(values, dtype=complex)

    n = len(values)

    x = np.arange(n)
    k = np.arange(n)

    matrix = (
        np.exp(
            2j * np.pi * np.outer(k, x) / n
        )
        / np.sqrt(n)
    )

    return matrix @ values


# ============================================================
# 2-D QFT
# ============================================================

def qft_2d(field):

    transformed = np.zeros_like(
        field,
        dtype=complex
    )

    for y in range(field.shape[0]):

        transformed[y, :] = qft_1d(
            field[y, :]
        )

    result = np.zeros_like(
        transformed,
        dtype=complex
    )

    for x in range(field.shape[1]):

        result[:, x] = qft_1d(
            transformed[:, x]
        )

    return result


# ============================================================
# 1-D INVERSE QFT
# ============================================================

def iqft_1d(values):

    values = np.asarray(values, dtype=complex)

    n = len(values)

    x = np.arange(n)
    k = np.arange(n)

    matrix = (
        np.exp(
            -2j * np.pi * np.outer(k, x) / n
        )
        / np.sqrt(n)
    )

    return matrix @ values


# ============================================================
# 2-D INVERSE QFT
# ============================================================

def iqft_2d(spectrum):

    transformed = np.zeros_like(
        spectrum,
        dtype=complex
    )

    for y in range(spectrum.shape[0]):

        transformed[y, :] = iqft_1d(
            spectrum[y, :]
        )

    result = np.zeros_like(
        transformed,
        dtype=complex
    )

    for x in range(spectrum.shape[1]):

        result[:, x] = iqft_1d(
            transformed[:, x]
        )

    return result


# ============================================================
# SPECTRAL COMPLEXITY
# ============================================================

def spectral_second_moment(spectrum):

    power = np.abs(spectrum) ** 2

    rows, cols = spectrum.shape

    fx = np.fft.fftfreq(cols) * cols
    fy = np.fft.fftfreq(rows) * rows

    KX, KY = np.meshgrid(
        fx,
        fy,
        indexing="xy"
    )

    radial_frequency = np.sqrt(
        KX ** 2 + KY ** 2
    )

    r_max = np.max(
        radial_frequency
    )

    total_power = np.sum(power)

    complexity = np.sum(
        radial_frequency ** 2 * power
    ) / (
        r_max ** 2 * total_power
    )

    return complexity, radial_frequency


# ============================================================
# RESPONSE CURVE
# ============================================================

def compute_bandwidth(
    complexity,
    gamma
):

    base_complexity = np.clip(
        (
            complexity - C25
        )
        /
        (
            C75 - C25
        ),
        0.0,
        1.0
    )

    normalized_complexity = (
        base_complexity ** gamma
    )

    bandwidth = (
        MAX_BANDWIDTH
        -
        (
            MAX_BANDWIDTH
            - MIN_BANDWIDTH
        )
        *
        normalized_complexity
    )

    return (
        normalized_complexity,
        bandwidth
    )


# ============================================================
# RADIAL FILTER
# ============================================================

def apply_filter(
    spectrum,
    radial_frequency,
    bandwidth
):

    r_max = np.max(
        radial_frequency
    )

    cutoff = bandwidth * r_max

    mask = (
        radial_frequency <= cutoff
    )

    filtered = spectrum.copy()

    filtered[~mask] = 0.0

    original_power = np.sum(
        np.abs(spectrum) ** 2
    )

    retained_power = np.sum(
        np.abs(filtered) ** 2
    )

    retained_fraction = (
        retained_power
        / original_power
    )

    return (
        filtered,
        cutoff,
        retained_fraction
    )


# ============================================================
# ENVIRONMENTS
# ============================================================

def get_environments():

    simple = {
        "agents": [
            (2.0, 2.0)
        ],
        "goals": [
            (13.0, 13.0)
        ],
        "obstacles": []
    }

    moderate = {
        "agents": [
            (2.0, 2.0),
            (3.0, 13.0)
        ],
        "goals": [
            (13.0, 13.0),
            (13.0, 2.0)
        ],
        "obstacles": [
            (8.0, 9.0),
            (5.0, 11.0)
        ]
    }

    complex = {
        "agents": [
            (2.0, 2.0),
            (3.0, 13.0),
            (13.0, 3.0),
            (12.0, 12.0)
        ],
        "goals": [
            (13.0, 13.0),
            (13.0, 2.0),
            (2.0, 13.0),
            (2.0, 2.0)
        ],
        "obstacles": [
            (4.0, 5.0),
            (6.0, 8.0),
            (8.0, 9.0),
            (10.0, 11.0),
            (5.0, 11.0),
            (11.0, 5.0),
            (8.0, 4.0),
            (12.0, 8.0)
        ]
    }

    return {
        "SIMPLE": simple,
        "MODERATE": moderate,
        "COMPLEX": complex
    }


# ============================================================
# MAIN
# ============================================================

def main():

    environments = get_environments()

    gammas = [
        1.0,
        1.5,
        2.0,
        3.0
    ]

    print()
    print("=" * 120)
    print(
        "                 RESPONSE CURVE EXPERIMENT"
    )
    print("=" * 120)

    for gamma in gammas:

        print()
        print(
            f"==================== GAMMA = {gamma} ===================="
        )

        print(
            f"{'Environment':<15}"
            f"{'Complexity':>15}"
            f"{'Norm C':>12}"
            f"{'Bandwidth':>14}"
            f"{'Retained':>14}"
            f"{'Recon Error':>16}"
        )

        print("-" * 90)

        for name, env in environments.items():

            # Generate field
            X, Y, field = generate_field(
                env["agents"],
                env["goals"],
                env["obstacles"],
                current_agent=0
            )

            normalized_field = (
                field
                /
                np.linalg.norm(field)
            )

            # QFT
            spectrum = qft_2d(
                normalized_field
            )

            # Complexity
            (
                complexity,
                radial_frequency
            ) = spectral_second_moment(
                spectrum
            )

            # Adaptive bandwidth
            (
                normalized_complexity,
                bandwidth
            ) = compute_bandwidth(
                complexity,
                gamma
            )

            # Filter
            (
                filtered,
                cutoff,
                retained_fraction
            ) = apply_filter(
                spectrum,
                radial_frequency,
                bandwidth
            )

            # IQFT
            reconstructed = iqft_2d(
                filtered
            )

            reconstructed_real = np.real(
                reconstructed
            )

            # Reconstruction error
            error = (
                np.linalg.norm(
                    normalized_field
                    - reconstructed_real
                )
                /
                np.linalg.norm(
                    normalized_field
                )
            )

            print(
                f"{name:<15}"
                f"{complexity:>15.8f}"
                f"{normalized_complexity:>12.6f}"
                f"{bandwidth:>14.6f}"
                f"{retained_fraction:>14.6f}"
                f"{error:>16.6f}"
            )

    print()
    print("=" * 120)
    print(
        "                 RESPONSE CURVE TEST COMPLETE"
    )
    print("=" * 120)


if __name__ == "__main__":
    main()