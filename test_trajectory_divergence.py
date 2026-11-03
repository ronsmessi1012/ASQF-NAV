import numpy as np

from environment.field import generate_field

from navigation.controller import (
    navigation_step,
    reached_goal
)

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

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


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

MAX_STEPS = 100


# ============================================================
# STATEVECTOR → FIELD
# ============================================================

def statevector_to_field(statevector):

    amplitudes = np.asarray(
        statevector.data,
        dtype=complex
    )

    return amplitudes.reshape(16, 16)


# ============================================================
# QUANTUM FILTERED FIELD
# ============================================================

def quantum_filtered_field(field):

    field_norm = np.linalg.norm(field)

    normalized_field = (
        field / field_norm
    )

    # --------------------------------------------------------
    # Amplitude encoding
    # --------------------------------------------------------

    circuit, _ = encode_field(
        normalized_field
    )

    # --------------------------------------------------------
    # QFT
    # --------------------------------------------------------

    circuit = apply_qft_2d(circuit)

    state = Statevector.from_instruction(
        circuit
    )

    spectrum = statevector_to_field(
        state
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

    normalized_complexity = (
        normalize_complexity(
            complexity
        )
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
        filter_result[
            "filtered_spectrum"
        ]
    )

    # --------------------------------------------------------
    # Normalize filtered spectrum
    # --------------------------------------------------------

    filtered_state = (
        filtered_spectrum.flatten()
    )

    filtered_state /= np.linalg.norm(
        filtered_state
    )

    # --------------------------------------------------------
    # Reload filtered state
    # --------------------------------------------------------

    inverse_circuit = QuantumCircuit(8)

    inverse_circuit.initialize(
        filtered_state,
        range(8)
    )

    # --------------------------------------------------------
    # IQFT
    # --------------------------------------------------------

    inverse_circuit = (
        apply_inverse_qft_2d(
            inverse_circuit
        )
    )

    reconstructed = (
        Statevector.from_instruction(
            inverse_circuit
        )
    )

    reconstructed_field = (
        statevector_to_field(
            reconstructed
        )
    )

    reconstructed_field = np.real(
        reconstructed_field
    )

    # --------------------------------------------------------
    # Restore scale
    # --------------------------------------------------------

    reconstructed_field /= np.linalg.norm(
        reconstructed_field
    )

    reconstructed_field *= field_norm

    return (
        reconstructed_field,
        complexity,
        filter_result
    )


# ============================================================
# RUN CLASSICAL / QUANTUM TRAJECTORIES
# ============================================================

def run_trajectory(mode):

    positions = [
        np.array(position, dtype=float)
        for position in INITIAL_AGENTS
    ]

    trajectories = [
        [position.copy()]
        for position in positions
    ]

    distances = []

    for step in range(MAX_STEPS):

        new_positions = []

        for agent_index in range(2):

            current_agents = [
                tuple(position)
                for position in positions
            ]

            _, _, classical_field = (
                generate_field(
                    current_agents,
                    GOALS,
                    OBSTACLES,
                    current_agent=agent_index
                )
            )

            # ------------------------------------------------
            # Select navigation field
            # ------------------------------------------------

            if mode == "classical":

                navigation_field = (
                    classical_field
                )

            elif mode == "quantum":

                (
                    navigation_field,
                    _,
                    _
                ) = quantum_filtered_field(
                    classical_field
                )

            else:

                raise ValueError(
                    f"Unknown mode: {mode}"
                )

            # ------------------------------------------------
            # Other agent
            # ------------------------------------------------

            other_agents = [
                positions[index]
                for index in range(2)
                if index != agent_index
            ]

            # ------------------------------------------------
            # Navigation
            # ------------------------------------------------

            new_position, _ = (
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
        # Inter-agent distance
        # ----------------------------------------------------

        distance = np.linalg.norm(
            positions[0] -
            positions[1]
        )

        distances.append(distance)

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

    return trajectories, distances


# ============================================================
# RUN
# ============================================================

print()
print("=" * 100)
print("STEP 9AA — TRAJECTORY DIVERGENCE ANALYSIS")
print("=" * 100)

print()
print("Running classical trajectory...")

classical_trajectories, classical_distances = (
    run_trajectory("classical")
)

print("Running filtered quantum trajectory...")

quantum_trajectories, quantum_distances = (
    run_trajectory("quantum")
)


# ============================================================
# TRAJECTORY DIVERGENCE
# ============================================================

num_steps = min(
    len(classical_trajectories[0]),
    len(quantum_trajectories[0])
)

divergences = []

for step in range(num_steps):

    total_divergence = 0.0

    for agent_index in range(2):

        classical_position = (
            classical_trajectories[
                agent_index
            ][step]
        )

        quantum_position = (
            quantum_trajectories[
                agent_index
            ][step]
        )

        difference = np.linalg.norm(
            classical_position -
            quantum_position
        )

        total_divergence += difference

    divergences.append(
        total_divergence
    )


divergences = np.asarray(
    divergences
)


# ============================================================
# FIRST MEANINGFUL DIVERGENCE
# ============================================================

thresholds = [
    0.001,
    0.005,
    0.010,
    0.050,
    0.100
]

print()
print("TRAJECTORY DIVERGENCE")
print("-" * 60)

for threshold in thresholds:

    indices = np.where(
        divergences >= threshold
    )[0]

    if len(indices) > 0:

        print(
            f"First divergence >= "
            f"{threshold:.3f}: "
            f"step {indices[0]}"
        )

    else:

        print(
            f"First divergence >= "
            f"{threshold:.3f}: "
            f"never"
        )


# ============================================================
# MAXIMUM DIVERGENCE
# ============================================================

maximum_index = np.argmax(
    divergences
)

print()
print("MAXIMUM DIVERGENCE")
print("-" * 60)

print(
    "Maximum position divergence:",
    divergences[maximum_index]
)

print(
    "Occurred at step:",
    maximum_index
)


# ============================================================
# DIVERGENCE AT SAFETY EVENT
# ============================================================

quantum_min_distance = min(
    quantum_distances
)

quantum_safety_index = (
    np.argmin(
        quantum_distances
    )
)

print()
print("QUANTUM SAFETY EVENT")
print("-" * 60)

print(
    "Minimum quantum distance:",
    quantum_min_distance
)

print(
    "Safety event step:",
    quantum_safety_index
)

if quantum_safety_index < len(
    divergences
):

    print(
        "Trajectory divergence at "
        "safety event:",
        divergences[
            quantum_safety_index
        ]
    )


# ============================================================
# SELECTED TIMESTEPS
# ============================================================

print()
print("SELECTED TRAJECTORY DIVERGENCES")
print("-" * 60)

selected_steps = [
    0,
    10,
    20,
    30,
    40,
    50,
    60,
    65,
    70,
    80,
    90
]

for step in selected_steps:

    if step < len(divergences):

        print(
            f"Step {step:3d} : "
            f"D = {divergences[step]:.6f}"
        )


# ============================================================
# INTERPRETATION
# ============================================================

print()
print("INTERPRETATION")
print("-" * 60)

if divergences[-1] < 0.01:

    print(
        "Very low trajectory divergence."
    )

elif divergences[-1] < 0.05:

    print(
        "Small but accumulated trajectory divergence."
    )

else:

    print(
        "Substantial accumulated trajectory divergence."
    )

print()
print(
    "This experiment identifies whether the "
    "safety degradation is caused by gradual "
    "trajectory drift or a localized divergence."
)

print()
print("=" * 100)