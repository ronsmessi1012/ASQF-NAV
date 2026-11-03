import numpy as np


# ============================================================
# CALIBRATION CONSTANTS
# ============================================================

# Obtained from the 50-environment calibration experiment.
C25 = 0.01645914
C75 = 0.01987830


# ============================================================
# 2-D NORMALIZED QFT
# ============================================================

def qft_1d(values):
    """
    Normalized 1-D Quantum Fourier Transform.

    Convention:
        F[k] = 1/sqrt(N) *
               sum_x f[x] * exp(+2*pi*i*x*k/N)
    """

    values = np.asarray(
        values,
        dtype=complex
    )

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


def qft_2d(field):
    """
    Separable normalized 2-D QFT.

    The transform is applied along both spatial axes.
    """

    field = np.asarray(
        field,
        dtype=complex
    )

    # Transform rows
    transformed = np.zeros_like(
        field,
        dtype=complex
    )

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
    """
    Normalized inverse 1-D QFT.
    """

    values = np.asarray(
        values,
        dtype=complex
    )

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


def iqft_2d(spectrum):
    """
    Separable normalized 2-D inverse QFT.
    """

    spectrum = np.asarray(
        spectrum,
        dtype=complex
    )

    # Inverse transform rows
    transformed = np.zeros_like(
        spectrum,
        dtype=complex
    )

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
# SPECTRAL POWER
# ============================================================

def spectral_power(spectrum):
    """
    Spectral power at every 2-D frequency coordinate.
    """

    return np.abs(spectrum) ** 2


# ============================================================
# RADIAL FREQUENCY GRID
# ============================================================

def radial_frequency_grid(shape):
    """
    Construct the 2-D radial spatial-frequency grid.

    Parameters
    ----------
    shape : tuple
        (rows, columns)

    Returns
    -------
    radial_frequency : ndarray
        Radial frequency at each spectral coordinate.
    """

    rows, cols = shape

    frequencies_x = (
        np.fft.fftfreq(cols)
        * cols
    )

    frequencies_y = (
        np.fft.fftfreq(rows)
        * rows
    )

    KX, KY = np.meshgrid(
        frequencies_x,
        frequencies_y,
        indexing="xy"
    )

    radial_frequency = np.sqrt(
        KX ** 2 + KY ** 2
    )

    return radial_frequency


# ============================================================
# SPECTRAL SECOND MOMENT
# ============================================================

def spectral_second_moment(spectrum):
    """
    Normalized spectral second moment.

    C =
        sum(r^2 * P)
        ----------------
        r_max^2 * sum(P)

    Higher C indicates that more spectral energy is
    distributed toward higher spatial frequencies.
    """

    power = spectral_power(
        spectrum
    )

    radial_frequency = (
        radial_frequency_grid(
            spectrum.shape
        )
    )

    r_max = np.max(
        radial_frequency
    )

    total_power = np.sum(
        power
    )

    if total_power < 1e-12:
        raise ValueError(
            "Total spectral power is too small."
        )

    complexity = (
        np.sum(
            radial_frequency ** 2
            * power
        )
        /
        (
            r_max ** 2
            * total_power
        )
    )

    return complexity


# ============================================================
# NORMALIZED COMPLEXITY
# ============================================================

def normalize_complexity(
    complexity
):
    """
    Normalize spectral complexity using the
    calibrated 25th and 75th percentiles.
    """

    normalized = (
        complexity - C25
    ) / (
        C75 - C25
    )

    return np.clip(
        normalized,
        0.0,
        1.0
    )


# ============================================================
# COMPLETE SPECTRAL ANALYSIS
# ============================================================

def analyze_spectrum(field):
    """
    Complete 2-D spectral analysis pipeline.

    Returns:
        spectrum
        power
        radial_frequency
        complexity
        normalized_complexity
    """

    field = np.asarray(
        field,
        dtype=float
    )

    norm = np.linalg.norm(
        field
    )

    if norm < 1e-12:
        raise ValueError(
            "Field norm is too small."
        )

    normalized_field = (
        field / norm
    )

    spectrum = qft_2d(
        normalized_field
    )

    power = spectral_power(
        spectrum
    )

    radial_frequency = (
        radial_frequency_grid(
            field.shape
        )
    )

    complexity = (
        spectral_second_moment(
            spectrum
        )
    )

    normalized_complexity = (
        normalize_complexity(
            complexity
        )
    )

    return {
        "normalized_field":
            normalized_field,

        "spectrum":
            spectrum,

        "power":
            power,

        "radial_frequency":
            radial_frequency,

        "complexity":
            complexity,

        "normalized_complexity":
            normalized_complexity
    }