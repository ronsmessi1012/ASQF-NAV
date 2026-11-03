import numpy as np


def compute_adaptive_bandwidth(
    complexity,
    min_bandwidth=0.25,
    max_bandwidth=0.75
):
    """
    Convert spectral complexity into an adaptive
    bandwidth fraction.

    Low complexity  -> wider bandwidth
    High complexity -> narrower bandwidth
    """

    complexity = np.clip(complexity, 0.0, 1.0)

    bandwidth = (
        min_bandwidth
        + (max_bandwidth - min_bandwidth)
        * (1.0 - complexity)
    )

    return bandwidth


def adaptive_spectral_filter(
    spectrum,
    complexity,
    min_bandwidth=0.25,
    max_bandwidth=0.75
):
    """
    Apply an adaptive low-pass spectral filter.

    The filter retains frequencies within the
    adaptive bandwidth and suppresses frequencies
    outside it.
    """

    n = len(spectrum)

    frequencies = np.fft.fftfreq(n) * n

    bandwidth = compute_adaptive_bandwidth(
        complexity,
        min_bandwidth,
        max_bandwidth
    )

    cutoff = (n / 2) * bandwidth
    mask = np.abs(frequencies) <= cutoff

    filtered_spectrum = spectrum.copy()
    filtered_spectrum[~mask] = 0.0

    # Renormalize so that the filtered quantum
    # state remains normalized.
    norm = np.linalg.norm(filtered_spectrum)

    if norm < 1e-12:
        raise ValueError(
            "Filtering removed all spectral components."
        )

    filtered_spectrum /= norm

    retained_fraction = np.sum(mask) / n

    return (
        filtered_spectrum,
        bandwidth,
        cutoff,
        retained_fraction,
        mask
    )