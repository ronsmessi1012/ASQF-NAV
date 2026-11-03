import numpy as np

from environment.field import generate_field


# ============================================================
# CONFIGURATION
# ============================================================

GRID_SIZE = 16

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

    matrix = np.exp(
        2j * np.pi * np.outer(k, x) / n
    ) / np.sqrt(n)

    return matrix @ values


# ============================================================
# 2-D NORMALIZED QFT
# ============================================================

def qft_2d(field):

    transformed = np.zeros_like(
        field,
        dtype=complex
    )

    # Transform rows
    for y in range(field.shape[0]):
        transformed[y, :] = qft_1d(
            field[y, :]
        )

    # Transform columns
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
# 2-D NORMALIZED INVERSE QFT
# ============================================================

def iqft_1d(values):

    values = np.asarray(values, dtype=complex)
    n = len(values)

    x = np.arange(n)
    k = np.arange(n)

    matrix = np.exp(
        -2j * np.pi * np.outer(k, x) / n
    ) / np.sqrt(n)

    return matrix @ values


def iqft_2d(spectrum):

    transformed = np.zeros_like(
        spectrum,
        dtype=complex
    )

    # Inverse transform rows
    for y in range(spectrum.shape[0]):
        transformed[y, :] = iqft_1d(
            spectrum[y, :]
        )

    # Inverse transform columns
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

    frequencies_x = np.fft.fftfreq(cols) * cols
    frequencies_y = np.fft.fftfreq(rows) * rows

    KX, KY = np.meshgrid(
        frequencies_x,
        frequencies_y,
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
# CALIBRATED BANDWIDTH
# ============================================================

def compute_adaptive_bandwidth(
    complexity
):

    normalized_complexity = np.clip(
        (
            complexity - C25
        ) / (
            C75 - C25
        ),
        0.0,
        1.0
    )

    bandwidth = (
        MAX_BANDWIDTH
        - (
            MAX_BANDWIDTH
            - MIN_BANDWIDTH
        )
        * normalized_complexity
    )

    return (
        normalized_complexity,
        bandwidth
    )


# ============================================================
# ADAPTIVE RADIAL FILTER
# ============================================================

def adaptive_2d_filter(
    spectrum,
    complexity,
    radial_frequency
):

    _, bandwidth = (
        compute_adaptive_bandwidth(
            complexity
        )
    )

    r_max = np.max(
        radial_frequency
    )

    cutoff = (
        bandwidth * r_max
    )

    mask = (
        radial_frequency <= cutoff
    )

    filtered_spectrum = (
        spectrum.copy()
    )

    filtered_spectrum[
        ~mask
    ] = 0.0

    original_power = np.sum(
        np.abs(spectrum) ** 2
    )

    retained_power = np.sum(
        np.abs(filtered_spectrum) ** 2
    )

    retained_power_fraction = (
        retained_power
        / original_power
    )

    return (
        filtered_spectrum,
        bandwidth,
        cutoff,
        mask,
        retained_power_fraction
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
# MAIN TEST
# ============================================================

def main():

    environments = get_environments()

    print()
    print("=" * 110)
    print(
        "             CALIBRATED 2-D ADAPTIVE SPECTRAL FILTER"
    )
    print("=" * 110)

    print(
        f"{'Environment':<15}"
        f"{'Complexity':>15}"
        f"{'Norm C':>12}"
        f"{'Bandwidth':>14}"
        f"{'Cutoff':>12}"
        f"{'Retained Power':>18}"
        f"{'Recon Error':>16}"
    )

    print("-" * 110)

    for name, env in environments.items():

        # ----------------------------------------------------
        # Generate field
        # ----------------------------------------------------

        X, Y, field = generate_field(
            env["agents"],
            env["goals"],
            env["obstacles"],
            current_agent=0
        )

        # Normalize
        field_norm = np.linalg.norm(
            field
        )

        normalized_field = (
            field / field_norm
        )

        # ----------------------------------------------------
        # 2-D QFT
        # ----------------------------------------------------

        spectrum = qft_2d(
            normalized_field
        )

        # ----------------------------------------------------
        # Complexity
        # ----------------------------------------------------

        (
            complexity,
            radial_frequency
        ) = spectral_second_moment(
            spectrum
        )

        (
            normalized_complexity,
            bandwidth
        ) = compute_adaptive_bandwidth(
            complexity
        )

        # ----------------------------------------------------
        # Adaptive filtering
        # ----------------------------------------------------

        (
            filtered_spectrum,
            bandwidth,
            cutoff,
            mask,
            retained_power
        ) = adaptive_2d_filter(
            spectrum,
            complexity,
            radial_frequency
        )

        # ----------------------------------------------------
        # Inverse 2-D QFT
        # ----------------------------------------------------

        reconstructed = iqft_2d(
            filtered_spectrum
        )

        reconstructed_real = np.real(
            reconstructed
        )

        # ----------------------------------------------------
        # Reconstruction error
        # ----------------------------------------------------

        reconstruction_error = (
            np.linalg.norm(
                normalized_field
                - reconstructed_real
            )
            / np.linalg.norm(
                normalized_field
            )
        )

        imaginary_component = np.max(
            np.abs(
                np.imag(reconstructed)
            )
        )

        # ----------------------------------------------------
        # Output
        # ----------------------------------------------------

        print(
            f"{name:<15}"
            f"{complexity:>15.8f}"
            f"{normalized_complexity:>12.6f}"
            f"{bandwidth:>14.6f}"
            f"{cutoff:>12.6f}"
            f"{retained_power:>18.8f}"
            f"{reconstruction_error:>16.8f}"
        )

        print(
            f"{'':<15}"
            f"{'Imaginary max:':>15}"
            f"{imaginary_component:.3e}"
        )

    print("=" * 110)

    print()
    print(
        "Expected behavior:"
    )
    print(
        "  Complexity       : SIMPLE < MODERATE < COMPLEX"
    )
    print(
        "  Bandwidth        : SIMPLE > MODERATE > COMPLEX"
    )
    print(
        "  Cutoff           : SIMPLE > MODERATE > COMPLEX"
    )

    print()
    print("=" * 110)
    print(
        "             FILTER TEST COMPLETE"
    )
    print("=" * 110)


if __name__ == "__main__":
    main()