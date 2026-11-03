import numpy as np

from environment.field import generate_field

from quantum.spectral_analysis_2d import (
    analyze_spectrum,
    iqft_2d
)

from quantum.adaptive_filter_2d import (
    adaptive_2d_filter
)


# ============================================================
# TEST ENVIRONMENTS
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

    complex_env = {
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
        "COMPLEX": complex_env
    }


# ============================================================
# MAIN
# ============================================================

def main():

    environments = get_environments()

    results = {}

    print()
    print("=" * 120)
    print(
        "             FINAL 2-D SPECTRAL MODULE VALIDATION"
    )
    print("=" * 120)

    print(
        f"{'Environment':<15}"
        f"{'Complexity':>15}"
        f"{'Norm C':>12}"
        f"{'Bandwidth':>14}"
        f"{'Retained Power':>18}"
        f"{'Recon Error':>16}"
        f"{'Imaginary':>15}"
    )

    print("-" * 120)

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

        # ----------------------------------------------------
        # Spectral analysis
        # ----------------------------------------------------

        analysis = analyze_spectrum(
            field
        )

        normalized_field = (
            analysis["normalized_field"]
        )

        spectrum = (
            analysis["spectrum"]
        )

        radial_frequency = (
            analysis["radial_frequency"]
        )

        complexity = (
            analysis["complexity"]
        )

        normalized_complexity = (
            analysis["normalized_complexity"]
        )

        # ----------------------------------------------------
        # Adaptive filtering
        # ----------------------------------------------------

        filter_result = adaptive_2d_filter(
            spectrum,
            radial_frequency,
            normalized_complexity
        )

        filtered_spectrum = (
            filter_result["filtered_spectrum"]
        )

        bandwidth = (
            filter_result["bandwidth"]
        )

        retained_power = (
            filter_result[
                "retained_power_fraction"
            ]
        )

        # ----------------------------------------------------
        # Inverse QFT
        # ----------------------------------------------------

        reconstructed_complex = (
            iqft_2d(
                filtered_spectrum
            )
        )

        reconstructed_field = (
            np.real(
                reconstructed_complex
            )
        )

        # ----------------------------------------------------
        # Reconstruction error
        # ----------------------------------------------------

        reconstruction_error = (
            np.linalg.norm(
                normalized_field
                - reconstructed_field
            )
            /
            np.linalg.norm(
                normalized_field
            )
        )

        imaginary_error = np.max(
            np.abs(
                np.imag(
                    reconstructed_complex
                )
            )
        )

        results[name] = {
            "complexity": complexity,
            "normalized_complexity":
                normalized_complexity,
            "bandwidth": bandwidth,
            "retained_power":
                retained_power,
            "reconstruction_error":
                reconstruction_error,
            "imaginary_error":
                imaginary_error
        }

        print(
            f"{name:<15}"
            f"{complexity:>15.8f}"
            f"{normalized_complexity:>12.6f}"
            f"{bandwidth:>14.6f}"
            f"{retained_power:>18.8f}"
            f"{reconstruction_error:>16.8f}"
            f"{imaginary_error:>15.3e}"
        )

    # ========================================================
    # VALIDATION CHECKS
    # ========================================================

    print()
    print("=" * 120)
    print("                    VALIDATION CHECKS")
    print("=" * 120)

    simple = results["SIMPLE"]
    moderate = results["MODERATE"]
    complex_result = results["COMPLEX"]

    complexity_check = (
        simple["complexity"]
        <
        moderate["complexity"]
        <
        complex_result["complexity"]
    )

    bandwidth_check = (
        simple["bandwidth"]
        >
        moderate["bandwidth"]
        >
        complex_result["bandwidth"]
    )

    reconstruction_check = all(
        result["reconstruction_error"] < 0.30
        for result in results.values()
    )

    imaginary_check = all(
        result["imaginary_error"] < 1e-10
        for result in results.values()
    )

    print(
        "Complexity monotonicity :",
        "PASS" if complexity_check else "FAIL"
    )

    print(
        "Bandwidth monotonicity  :",
        "PASS" if bandwidth_check else "FAIL"
    )

    print(
        "Reconstruction error    :",
        "PASS" if reconstruction_check else "FAIL"
    )

    print(
        "Imaginary component     :",
        "PASS" if imaginary_check else "FAIL"
    )

    overall = (
        complexity_check
        and bandwidth_check
        and reconstruction_check
        and imaginary_check
    )

    print()
    print(
        "OVERALL RESULT          :",
        "PASS" if overall else "FAIL"
    )

    print("=" * 120)

    if overall:

        print(
            "2-D adaptive spectral module "
            "is ready for integration."
        )

    else:

        print(
            "Module requires correction "
            "before integration."
        )

    print("=" * 120)


if __name__ == "__main__":
    main()