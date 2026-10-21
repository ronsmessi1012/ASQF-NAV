import numpy as np


# ============================================================
# ADAPTIVE BANDWIDTH PARAMETERS
# ============================================================

MIN_BANDWIDTH = 0.25
MAX_BANDWIDTH = 0.75

GAMMA = 3.0


# ============================================================
# ADAPTIVE BANDWIDTH
# ============================================================

def compute_adaptive_bandwidth(
    normalized_complexity
):
    """
    Convert normalized spectral complexity into
    an adaptive bandwidth.

    Cubic response:
        B = Bmax - (Bmax - Bmin) * C^gamma
    """

    normalized_complexity = np.clip(
        normalized_complexity,
        0.0,
        1.0
    )

    bandwidth = (
        MAX_BANDWIDTH
        -
        (
            MAX_BANDWIDTH
            - MIN_BANDWIDTH
        )
        *
        normalized_complexity ** GAMMA
    )

    return bandwidth


# ============================================================
# 2-D RADIAL ADAPTIVE FILTER
# ============================================================

def adaptive_2d_filter(
    spectrum,
    radial_frequency,
    normalized_complexity
):
    """
    Apply a radial low-pass filter whose cutoff
    adapts to spectral complexity.
    """

    bandwidth = (
        compute_adaptive_bandwidth(
            normalized_complexity
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

    if original_power < 1e-12:
        raise ValueError(
            "Original spectral power is too small."
        )

    retained_power_fraction = (
        retained_power
        / original_power
    )

    retained_components = (
        np.sum(mask)
    )

    total_components = (
        mask.size
    )

    retained_component_fraction = (
        retained_components
        / total_components
    )

    return {
        "filtered_spectrum":
            filtered_spectrum,

        "bandwidth":
            bandwidth,

        "cutoff":
            cutoff,

        "mask":
            mask,

        "retained_power_fraction":
            retained_power_fraction,

        "retained_component_fraction":
            retained_component_fraction
    }