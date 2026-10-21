import numpy as np

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from environment.field import generate_field

from quantum.amplitude_encoding import encode_field

from quantum.qft_2d_qiskit import (
    apply_qft_2d,
    apply_inverse_qft_2d
)

from quantum.spectral_analysis_2d import (
    radial_frequency_grid,
    spectral_second_moment,
    normalize_complexity
)

from quantum.adaptive_filter_2d import (
    adaptive_2d_filter
)

from navigation.controller import (
    navigation_step,
    reached_goal
)


# ============================================================
# QISKIT STATEVECTOR → 16 × 16 FIELD
# ============================================================

def statevector_to_field(statevector):

    amplitudes = np.asarray(
        statevector.data,
        dtype=complex
    )

    if len(amplitudes) != 256:
        raise ValueError(
            f"Expected 256 amplitudes, got {len(amplitudes)}"
        )

    return amplitudes.reshape(16, 16)


# ============================================================
# QUANTUM RECONSTRUCTION — NO FILTER
# ============================================================

def quantum_reconstruct_unfiltered(field):

    field_norm = np.linalg.norm(field)

    if field_norm < 1e-12:
        raise ValueError(
            "Field norm is too small."
        )

    normalized_field = field / field_norm

    # --------------------------------------------------------
    # Amplitude encoding
    # --------------------------------------------------------

    qft_circuit, _ = encode_field(
        normalized_field
    )

    # --------------------------------------------------------
    # Actual 2-D QFT
    # --------------------------------------------------------

    qft_circuit = apply_qft_2d(
        qft_circuit
    )

    qft_state = Statevector.from_instruction(
        qft_circuit
    )

    # --------------------------------------------------------
    # Convert quantum state → spectrum
    # --------------------------------------------------------

    spectrum = statevector_to_field(
        qft_state
    )

    # --------------------------------------------------------
    # Spectral complexity
    # --------------------------------------------------------

    complexity = spectral_second_moment(
        spectrum
    )

    normalized_complexity = normalize_complexity(
        complexity
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # NO FILTERING HERE
    #
    # We directly perform the inverse QFT.
    # --------------------------------------------------------

    spectrum_state = (
        spectrum.flatten()
    )

    spectrum_norm = np.linalg.norm(
        spectrum_state
    )

    if spectrum_norm < 1e-12:
        raise ValueError(
            "Spectrum norm is too small."
        )

    spectrum_state = (
        spectrum_state / spectrum_norm
    )

    # --------------------------------------------------------
    # Load spectrum into Qiskit
    # --------------------------------------------------------

    inverse_circuit = QuantumCircuit(8)

    inverse_circuit.initialize(
        spectrum_state,
        range(8)
    )

    # --------------------------------------------------------
    # Actual inverse 2-D QFT
    # --------------------------------------------------------

    inverse_circuit = apply_inverse_qft_2d(
        inverse_circuit
    )

    reconstructed_state = (
        Statevector.from_instruction(
            inverse_circuit
        )
    )

    # --------------------------------------------------------
    # Convert back to spatial field
    # --------------------------------------------------------

    reconstructed_field = (
        statevector_to_field(
            reconstructed_state
        )
    )

    reconstructed_field = np.real(
        reconstructed_field
    )

    # --------------------------------------------------------
    # Restore original scale
    # --------------------------------------------------------

    reconstructed_norm = np.linalg.norm(
        reconstructed_field
    )

    if reconstructed_norm < 1e-12:
        raise ValueError(
            "Reconstructed field norm is too small."
        )

    reconstructed_field = (
        reconstructed_field /
        reconstructed_norm
    )

    reconstructed_field = (
        reconstructed_field *
        field_norm
    )

    return (
        reconstructed_field,
        complexity,
        normalized_complexity
    )


# ============================================================
# QUANTUM RECONSTRUCTION — ADAPTIVE FILTER
# ============================================================

def quantum_reconstruct_filtered(field):

    field_norm = np.linalg.norm(field)

    if field_norm < 1e-12:
        raise ValueError(
            "Field norm is too small."
        )

    normalized_field = field / field_norm

    # --------------------------------------------------------
    # Amplitude encoding
    # --------------------------------------------------------

    qft_circuit, _ = encode_field(
        normalized_field
    )

    # --------------------------------------------------------
    # Actual 2-D QFT
    # --------------------------------------------------------

    qft_circuit = apply_qft_2d(
        qft_circuit
    )

    qft_state = Statevector.from_instruction(
        qft_circuit
    )

    # --------------------------------------------------------
    # Spectrum
    # --------------------------------------------------------

    spectrum = statevector_to_field(
        qft_state
    )

    # --------------------------------------------------------
    # Spectral analysis
    # --------------------------------------------------------

    radial_frequency = radial_frequency_grid(
        field.shape
    )

    complexity = spectral_second_moment(
        spectrum
    )

    normalized_complexity = normalize_complexity(
        complexity
    )

    # --------------------------------------------------------
    # Adaptive filtering
    # --------------------------------------------------------

    filter_result = adaptive_2d_filter(
        spectrum,
        radial_frequency,
        normalized_complexity
    )

    filtered_spectrum = (
        filter_result["filtered_spectrum"]
    )

    # --------------------------------------------------------
    # Normalize filtered spectrum
    # --------------------------------------------------------

    filtered_state = (
        filtered_spectrum.flatten()
    )

    filtered_norm = np.linalg.norm(
        filtered_state
    )

    if filtered_norm < 1e-12:
        raise ValueError(
            "Filtered spectrum norm is too small."
        )

    filtered_state = (
        filtered_state / filtered_norm
    )

    # --------------------------------------------------------
    # Load filtered spectrum
    # --------------------------------------------------------

    inverse_circuit = QuantumCircuit(8)

    inverse_circuit.initialize(
        filtered_state,
        range(8)
    )

    # --------------------------------------------------------
    # Actual inverse 2-D QFT
    # --------------------------------------------------------

    inverse_circuit = apply_inverse_qft_2d(
        inverse_circuit
    )

    reconstructed_state = (
        Statevector.from_instruction(
            inverse_circuit
        )
    )

    # --------------------------------------------------------
    # Spatial field
    # --------------------------------------------------------

    reconstructed_field = (
        statevector_to_field(
            reconstructed_state
        )
    )

    reconstructed_field = np.real(
        reconstructed_field
    )

    # --------------------------------------------------------
    # Restore original scale
    # --------------------------------------------------------

    reconstructed_norm = np.linalg.norm(
        reconstructed_field
    )

    if reconstructed_norm < 1e-12:
        raise ValueError(
            "Reconstructed field norm is too small."
        )

    reconstructed_field = (
        reconstructed_field /
        reconstructed_norm
    )

    reconstructed_field = (
        reconstructed_field *
        field_norm
    )

    return (
        reconstructed_field,
        complexity,
        normalized_complexity,
        filter_result
    )


# ============================================================
# ENVIRONMENT
# ============================================================

INITIAL_AGENTS = [
    (2.0, 2.0),
    (3.0, 13.0)
]

GOALS = [
    (13.0, 13.0),
    (13.0, 2.0)
]

OBSTACLES = [
    (8.0, 9.0),
    (5.0, 11.0)
]


# ============================================================
# SETTINGS
# ============================================================

MAX_STEPS = 200


# ============================================================
# PATH LENGTH
# ============================================================

def compute_path_length(path):

    path = np.asarray(path)

    if len(path) < 2:
        return 0.0

    differences = (
        path[1:] -
        path[:-1]
    )

    distances = np.linalg.norm(
        differences,
        axis=1
    )

    return np.sum(distances)


# ============================================================
# MINIMUM INTER-AGENT DISTANCE
# ============================================================

def compute_min_agent_distance(
    trajectories
):

    minimum_distance = np.inf

    num_steps = min(
        len(trajectories[0]),
        len(trajectories[1])
    )

    for step in range(num_steps):

        position_0 = trajectories[0][step]
        position_1 = trajectories[1][step]

        distance = np.linalg.norm(
            position_0 -
            position_1
        )

        minimum_distance = min(
            minimum_distance,
            distance
        )

    return minimum_distance


# ============================================================
# RUN SIMULATION
# ============================================================

def run_simulation(mode):

    positions = [
        np.array(position, dtype=float)
        for position in INITIAL_AGENTS
    ]

    trajectories = [
        [position.copy()]
        for position in positions
    ]

    complexities = []
    normalized_complexities = []
    bandwidths = []
    retained_powers = []
    reconstruction_errors = []

    for step in range(MAX_STEPS):

        new_positions = []

        for agent_index in range(2):

            # ------------------------------------------------
            # Current environment
            # ------------------------------------------------

            current_agents = [
                tuple(position)
                for position in positions
            ]

            _, _, classical_field = generate_field(
                current_agents,
                GOALS,
                OBSTACLES,
                current_agent=agent_index
            )

            # ------------------------------------------------
            # CLASSICAL
            # ------------------------------------------------

            if mode == "classical":

                navigation_field = classical_field

            # ------------------------------------------------
            # QUANTUM — NO FILTER
            # ------------------------------------------------

            elif mode == "quantum_unfiltered":

                (
                    navigation_field,
                    complexity,
                    normalized_complexity
                ) = quantum_reconstruct_unfiltered(
                    classical_field
                )

                complexities.append(
                    complexity
                )

                normalized_complexities.append(
                    normalized_complexity
                )

                reconstruction_error = (
                    np.linalg.norm(
                        classical_field -
                        navigation_field
                    )
                    /
                    np.linalg.norm(
                        classical_field
                    )
                )

                reconstruction_errors.append(
                    reconstruction_error
                )

            # ------------------------------------------------
            # QUANTUM — ADAPTIVE FILTER
            # ------------------------------------------------

            elif mode == "quantum_filtered":

                (
                    navigation_field,
                    complexity,
                    normalized_complexity,
                    filter_result
                ) = quantum_reconstruct_filtered(
                    classical_field
                )

                complexities.append(
                    complexity
                )

                normalized_complexities.append(
                    normalized_complexity
                )

                bandwidths.append(
                    filter_result["bandwidth"]
                )

                retained_powers.append(
                    filter_result[
                        "retained_power_fraction"
                    ]
                )

                reconstruction_error = (
                    np.linalg.norm(
                        classical_field -
                        navigation_field
                    )
                    /
                    np.linalg.norm(
                        classical_field
                    )
                )

                reconstruction_errors.append(
                    reconstruction_error
                )

            else:

                raise ValueError(
                    f"Unknown mode: {mode}"
                )

            # ------------------------------------------------
            # Other agents
            #
            # UNCHANGED
            # ------------------------------------------------

            other_agents = [
                positions[index]
                for index in range(2)
                if index != agent_index
            ]

            # ------------------------------------------------
            # Navigation
            # ------------------------------------------------

            new_position, velocity = (
                navigation_step(
                    positions[agent_index],
                    np.array(
                        GOALS[agent_index]
                    ),
                    navigation_field,
                    other_agents=other_agents
                )
            )

            new_positions.append(
                new_position
            )

        # ----------------------------------------------------
        # Simultaneous update
        # ----------------------------------------------------

        positions = new_positions

        for agent_index in range(2):

            trajectories[agent_index].append(
                positions[agent_index].copy()
            )

        # ----------------------------------------------------
        # Goal condition
        # ----------------------------------------------------

        if all(
            reached_goal(
                positions[index],
                GOALS[index]
            )
            for index in range(2)
        ):
            break

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    steps = max(
        len(trajectory)
        for trajectory in trajectories
    ) - 1

    path_lengths = [
        compute_path_length(
            trajectories[index]
        )
        for index in range(2)
    ]

    goal_errors = [
        np.linalg.norm(
            positions[index] -
            np.array(GOALS[index])
        )
        for index in range(2)
    ]

    minimum_distance = (
        compute_min_agent_distance(
            trajectories
        )
    )

    return {
        "positions": positions,
        "trajectories": trajectories,
        "steps": steps,
        "path_lengths": path_lengths,
        "goal_errors": goal_errors,
        "minimum_distance": minimum_distance,
        "complexities": complexities,
        "normalized_complexities": normalized_complexities,
        "bandwidths": bandwidths,
        "retained_powers": retained_powers,
        "reconstruction_errors": reconstruction_errors
    }


# ============================================================
# RUN EXPERIMENTS
# ============================================================

print()
print("=" * 100)
print("STEP 9Y — FILTER ABLATION DIAGNOSTIC")
print("=" * 100)

print()
print("Running CLASSICAL...")
classical = run_simulation(
    "classical"
)

print()
print("Running QUANTUM — NO FILTER...")
quantum_unfiltered = run_simulation(
    "quantum_unfiltered"
)

print()
print("Running QUANTUM — ADAPTIVE FILTER...")
quantum_filtered = run_simulation(
    "quantum_filtered"
)


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(
    name,
    result
):

    print()
    print(
        f"----------- {name} -----------"
    )

    print(
        "Steps:",
        result["steps"]
    )

    for agent_index in range(2):

        print()
        print(
            f"Robot {agent_index}"
        )

        print(
            "  Final position   :",
            result["positions"][agent_index]
        )

        print(
            "  Goal             :",
            GOALS[agent_index]
        )

        print(
            "  Goal error       :",
            result["goal_errors"][agent_index]
        )

        print(
            "  Path length      :",
            result["path_lengths"][agent_index]
        )

    print()

    print(
        "Total path length :",
        sum(result["path_lengths"])
    )

    print(
        "Minimum inter-agent distance:",
        result["minimum_distance"]
    )

    if result["reconstruction_errors"]:

        print(
            "Average reconstruction error:",
            np.mean(
                result["reconstruction_errors"]
            )
        )

        print(
            "Maximum reconstruction error:",
            np.max(
                result["reconstruction_errors"]
            )
        )


# ============================================================
# RESULTS
# ============================================================

print_results(
    "CLASSICAL",
    classical
)

print_results(
    "QUANTUM — NO FILTER",
    quantum_unfiltered
)

print_results(
    "QUANTUM — ADAPTIVE FILTER",
    quantum_filtered
)


# ============================================================
# QUANTUM FILTER STATISTICS
# ============================================================

print()
print("----------- QUANTUM FILTER STATISTICS -----------")

print(
    "Average spectral complexity:",
    np.mean(
        quantum_filtered["complexities"]
    )
)

print(
    "Average normalized complexity:",
    np.mean(
        quantum_filtered[
            "normalized_complexities"
        ]
    )
)

print(
    "Average adaptive bandwidth:",
    np.mean(
        quantum_filtered["bandwidths"]
    )
)

print(
    "Average retained spectral power:",
    np.mean(
        quantum_filtered["retained_powers"]
    )
)


# ============================================================
# VALIDATION
# ============================================================

classical_reached = all(
    error <= 0.45
    for error in classical["goal_errors"]
)

unfiltered_reached = all(
    error <= 0.45
    for error in quantum_unfiltered["goal_errors"]
)

filtered_reached = all(
    error <= 0.45
    for error in quantum_filtered["goal_errors"]
)

classical_safe = (
    classical["minimum_distance"] >= 1.05
)

unfiltered_safe = (
    quantum_unfiltered["minimum_distance"] >= 1.05
)

filtered_safe = (
    quantum_filtered["minimum_distance"] >= 1.05
)


# ============================================================
# VALIDATION OUTPUT
# ============================================================

print()
print("----------- VALIDATION -----------")

print(
    "Classical reached goals       :",
    "PASS"
    if classical_reached
    else "FAIL"
)

print(
    "Quantum unfiltered goals      :",
    "PASS"
    if unfiltered_reached
    else "FAIL"
)

print(
    "Quantum filtered goals        :",
    "PASS"
    if filtered_reached
    else "FAIL"
)

print()

print(
    "Classical collision safety    :",
    "PASS"
    if classical_safe
    else "FAIL"
)

print(
    "Quantum unfiltered safety    :",
    "PASS"
    if unfiltered_safe
    else "FAIL"
)

print(
    "Quantum filtered safety      :",
    "PASS"
    if filtered_safe
    else "FAIL"
)


# ============================================================
# DIAGNOSIS
# ============================================================

print()
print("----------- DIAGNOSIS -----------")

if (
    unfiltered_safe
    and not filtered_safe
):

    print(
        "FILTER ISOLATION: PASS"
    )

    print(
        "The adaptive spectral filter is responsible "
        "for the observed collision-margin degradation."
    )

elif (
    not unfiltered_safe
    and not filtered_safe
):

    print(
        "FILTER ISOLATION: INCONCLUSIVE"
    )

    print(
        "The unfiltered quantum reconstruction also "
        "reduces the collision margin."
    )

    print(
        "Therefore the problem is not caused exclusively "
        "by adaptive spectral truncation."
    )

elif (
    unfiltered_safe
    and filtered_safe
):

    print(
        "Both quantum configurations satisfy the "
        "collision constraint."
    )

else:

    print(
        "Unexpected trajectory behavior detected."
    )


print()
print("=" * 100)