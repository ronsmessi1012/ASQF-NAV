import numpy as np


# ============================================================
# CALIBRATION STATISTICS
# ============================================================

C25 = 0.01645914
C75 = 0.01987830


# ============================================================
# ADAPTIVE BANDWIDTH PARAMETERS
# ============================================================

MIN_BANDWIDTH = 0.25
MAX_BANDWIDTH = 0.75


# ============================================================
# NORMALIZED COMPLEXITY
# ============================================================

def normalize_complexity(complexity):
    """
    Convert raw spectral complexity into [0, 1]
    using the calibration 25th and 75th percentiles.
    """

    normalized = (
        (complexity - C25)
        / (C75 - C25)
    )

    return np.clip(
        normalized,
        0.0,
        1.0
    )


# ============================================================
# ADAPTIVE BANDWIDTH
# ============================================================

def compute_adaptive_bandwidth(complexity):

    normalized_complexity = normalize_complexity(
        complexity
    )

    bandwidth = (
        MAX_BANDWIDTH
        - (
            MAX_BANDWIDTH
            - MIN_BANDWIDTH
        )
        * normalized_complexity
    )

    return normalized_complexity, bandwidth


# ============================================================
# TEST CASES
# ============================================================

def main():

    test_cases = {
        "Very Low Complexity": 0.00849377,
        "25th Percentile": C25,
        "Simple": 0.01629580,
        "Moderate": 0.01702802,
        "Median": 0.01797952,
        "Complex": 0.01937761,
        "75th Percentile": C75,
        "Very High Complexity": 0.02321662,
    }

    print()
    print("=" * 90)
    print("              CALIBRATED ADAPTIVE BANDWIDTH")
    print("=" * 90)

    print(
        f"{'Environment':<25}"
        f"{'Complexity':>15}"
        f"{'Normalized C':>18}"
        f"{'Bandwidth':>15}"
    )

    print("-" * 90)

    for name, complexity in test_cases.items():

        normalized, bandwidth = (
            compute_adaptive_bandwidth(
                complexity
            )
        )

        print(
            f"{name:<25}"
            f"{complexity:>15.8f}"
            f"{normalized:>18.6f}"
            f"{bandwidth:>15.6f}"
        )

    print("=" * 90)

    # --------------------------------------------------------
    # Monotonicity check
    # --------------------------------------------------------

    complexity_values = np.array([
        0.00849377,
        C25,
        0.01797952,
        C75,
        0.02321662
    ])

    bandwidth_values = np.array([
        compute_adaptive_bandwidth(c)[1]
        for c in complexity_values
    ])

    print()
    print("========== MONOTONICITY CHECK ==========")

    print(
        "Complexity values:"
    )

    print(
        complexity_values
    )

    print()

    print(
        "Bandwidth values:"
    )

    print(
        bandwidth_values
    )

    is_monotonic = np.all(
        np.diff(bandwidth_values) <= 0
    )

    print()

    if is_monotonic:
        print(
            "PASS: Complexity ↑ → Bandwidth ↓"
        )
    else:
        print(
            "FAIL: Bandwidth is not monotonically decreasing."
        )

    print()
    print("=" * 90)
    print(
        "           ADAPTIVE BANDWIDTH TEST COMPLETE"
    )
    print("=" * 90)


if __name__ == "__main__":
    main()