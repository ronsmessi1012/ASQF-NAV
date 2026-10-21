import numpy as np

from environment.field import generate_field


# ============================================================
# CONFIGURATION
# ============================================================

GRID_SIZE = 16
NUM_ENVIRONMENTS = 50

RNG = np.random.default_rng(42)


# ============================================================
# 2-D NORMALIZED QFT PROTOTYPE
# ============================================================

def qft_1d(values):
    """
    Normalized 1-D QFT:
        F[k] = 1/sqrt(N) * sum_x f[x] exp(+2*pi*i*x*k/N)
    """
    values = np.asarray(values, dtype=complex)
    n = len(values)

    x = np.arange(n)
    k = np.arange(n)

    matrix = np.exp(
        2j * np.pi * np.outer(k, x) / n
    ) / np.sqrt(n)

    return matrix @ values


def qft_2d(field):
    """
    Separable normalized 2-D QFT.
    """

    transformed = np.zeros_like(
        field,
        dtype=complex
    )

    # Transform along x-axis
    for y in range(field.shape[0]):
        transformed[y, :] = qft_1d(field[y, :])

    # Transform along y-axis
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
# SPECTRAL SECOND MOMENT
# ============================================================

def spectral_second_moment(spectrum):

    power = np.abs(spectrum) ** 2

    n_rows, n_cols = spectrum.shape

    frequencies_x = np.fft.fftfreq(n_cols)
    frequencies_y = np.fft.fftfreq(n_rows)

    # Convert normalized frequencies into
    # integer spatial-frequency coordinates.
    kx = frequencies_x * n_cols
    ky = frequencies_y * n_rows

    KX, KY = np.meshgrid(
        kx,
        ky,
        indexing="xy"
    )

    radial_frequency = np.sqrt(
        KX ** 2 + KY ** 2
    )

    r_max = np.max(radial_frequency)

    total_power = np.sum(power)

    if total_power < 1e-12:
        raise ValueError(
            "Total spectral power is too small."
        )

    second_moment = np.sum(
        (radial_frequency ** 2) * power
    ) / (
        r_max ** 2 * total_power
    )

    return second_moment


# ============================================================
# RANDOM ENVIRONMENT GENERATION
# ============================================================

def generate_random_environment():

    num_agents = RNG.integers(
        1,
        5
    )

    num_obstacles = RNG.integers(
        0,
        9
    )

    agents = []

    for _ in range(num_agents):

        position = (
            RNG.uniform(1.0, 14.0),
            RNG.uniform(1.0, 14.0)
        )

        agents.append(position)

    goals = []

    for _ in range(num_agents):

        goal = (
            RNG.uniform(1.0, 14.0),
            RNG.uniform(1.0, 14.0)
        )

        goals.append(goal)

    obstacles = []

    for _ in range(num_obstacles):

        obstacle = (
            RNG.uniform(2.0, 13.0),
            RNG.uniform(2.0, 13.0)
        )

        obstacles.append(obstacle)

    return agents, goals, obstacles


# ============================================================
# MAIN CALIBRATION EXPERIMENT
# ============================================================

def main():

    complexities = []

    print("\n")
    print("=" * 90)
    print("             SPECTRAL COMPLEXITY CALIBRATION")
    print("=" * 90)

    print(
        f"Generating {NUM_ENVIRONMENTS} random environments..."
    )

    print("\n")

    for i in range(NUM_ENVIRONMENTS):

        agents, goals, obstacles = (
            generate_random_environment()
        )

        # Analyze the field of agent 0.
        X, Y, field = generate_field(
            agents,
            goals,
            obstacles,
            current_agent=0
        )

        # Normalize field.
        norm = np.linalg.norm(field)

        if norm < 1e-12:
            continue

        normalized_field = field / norm

        # Proper 2-D spectral transform.
        spectrum = qft_2d(
            normalized_field
        )

        # Spectral second moment.
        complexity = spectral_second_moment(
            spectrum
        )

        complexities.append(
            complexity
        )

    complexities = np.asarray(
        complexities
    )

    print("=" * 90)

    print(
        f"Valid environments: {len(complexities)}"
    )

    print()

    print(
        f"Minimum complexity : "
        f"{np.min(complexities):.8f}"
    )

    print(
        f"25th percentile    : "
        f"{np.percentile(complexities, 25):.8f}"
    )

    print(
        f"Median complexity  : "
        f"{np.median(complexities):.8f}"
    )

    print(
        f"75th percentile    : "
        f"{np.percentile(complexities, 75):.8f}"
    )

    print(
        f"Maximum complexity : "
        f"{np.max(complexities):.8f}"
    )

    print(
        f"Mean complexity    : "
        f"{np.mean(complexities):.8f}"
    )

    print(
        f"Std deviation      : "
        f"{np.std(complexities):.8f}"
    )

    print()
    print("=" * 90)

    print(
        "SORTED COMPLEXITY VALUES"
    )

    print("=" * 90)

    sorted_values = np.sort(
        complexities
    )

    for i, value in enumerate(
        sorted_values,
        start=1
    ):

        print(
            f"{i:02d}: {value:.8f}"
        )

    print("=" * 90)

    print(
        "              CALIBRATION COMPLETE"
    )

    print("=" * 90)


if __name__ == "__main__":
    main()