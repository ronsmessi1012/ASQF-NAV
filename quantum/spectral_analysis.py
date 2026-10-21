import numpy as np


def spectral_power(spectrum):
    """
    Compute spectral power from QFT amplitudes.
    """

    return np.abs(spectrum) ** 2


def frequency_indices(n):
    """
    Construct centered discrete frequency indices.

    Example for N = 8:

        [0, 1, 2, 3, -4, -3, -2, -1]
    """

    return np.fft.fftfreq(n) * n


def spectral_complexity(
    spectrum,
    high_frequency_threshold_ratio=0.25
):
    """
    Compute the ratio of high-frequency spectral
    power to total spectral power.
    """

    n = len(spectrum)

    power = spectral_power(spectrum)

    frequencies = frequency_indices(n)

    threshold = (
        n * high_frequency_threshold_ratio
    )

    high_frequency_mask = (
        np.abs(frequencies) > threshold
    )

    high_frequency_power = np.sum(
        power[high_frequency_mask]
    )

    total_power = np.sum(power)

    if total_power < 1e-12:
        raise ValueError(
            "Total spectral power is too small."
        )

    complexity = (
        high_frequency_power
        / total_power
    )

    return complexity


def analyze_spectrum(spectrum):

    power = spectral_power(spectrum)

    complexity = spectral_complexity(
        spectrum
    )

    frequencies = frequency_indices(
        len(spectrum)
    )

    print(
        "\n========== SPECTRAL ANALYSIS =========="
    )

    print(
        "Number of spectral components:",
        len(spectrum)
    )

    print(
        "Total spectral power:",
        np.sum(power)
    )

    print(
        "High-frequency spectral power:",
        complexity * np.sum(power)
    )

    print(
        "Spectral complexity:",
        complexity
    )

    print(
        "Spectral complexity (%):",
        complexity * 100
    )

    return {
        "power": power,
        "frequencies": frequencies,
        "complexity": complexity
    }