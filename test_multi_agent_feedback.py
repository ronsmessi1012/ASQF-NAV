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
    (5.0, 11.0),
    (6.0, 10.0)
]

MAX_STEPS = 100


# ============================================================
# STATEVECTOR -> FIELD
# ============================================================

def statevector_to_field(statevector):

    amplitudes = np.asarray(
        statevector.data,
        dtype=complex
    )

    if amplitudes.size != 256:
        raise ValueError(
            f"Expected 256 amplitudes, "
            f"got {amplitudes.size}"
        )

    return amplitudes.reshape(16, 16)


# ============================================================
# QUANTUM ADAPTIVE RECONSTRUCTION
# ============================================================

def quantum_filtered_field(field):

    field_norm = np.linalg.norm(field)

    if field_norm < 1e-12:
        raise ValueError(
            "Field norm is too small."
        )

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
    # 2-D QFT
    # --------------------------------------------------------

    circuit = apply_qft_2d(circuit)

    qft_state = Statevector.from_instruction(
        circuit
    )

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

    normalized_complexity = (
        normalize_complexity(
            complexity
        )
    )

    # --------------------------------------------------------
    # Adaptive filter
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

    filtered_norm = np.linalg.norm(
        filtered_state
    )

    if filtered_norm < 1e-12:
        raise ValueError(
            "Filtered spectrum norm is too small."
        )

    filtered_state /= filtered_norm

    # --------------------------------------------------------
    # Reload filtered spectrum
    # --------------------------------------------------------

    inverse_circuit = QuantumCircuit(8)

    inverse_circuit.initialize(
        filtered_state,
        range(8)
    )

    # --------------------------------------------------------
    # Inverse 2-D QFT
    # --------------------------------------------------------

    inverse_circuit = (
        apply_inverse_qft_2d(
            inverse_circuit
        )
    )

    reconstructed_state = (
        Statevector.from_instruction(
            inverse_circuit
        )
    )

    reconstructed_field = (
        statevector_to_field(
            reconstructed_state
        )
    )

    reconstructed_field = np.real(
        reconstructed_field
    )

    # --------------------------------------------------------
    # Restore original field scale
    # --------------------------------------------------------

    reconstructed_norm = np.linalg.norm(
        reconstructed_field
    )

    reconstructed_field /= reconstructed_norm

    reconstructed_field *= field_norm

    return (
        reconstructed_field,
        complexity,
        filter_result
    )


# ============================================================
# HELPER: TRAJECTORY DISTANCE
# ============================================================

def trajectory_distance(
    trajectory_a,
    trajectory_b
):

    num_steps = min(
        len(trajectory_a),
        len(trajectory_b)
    )

    distances = []

    for step in range(num_steps):

        distance = np.linalg.norm(
            trajectory_a[step]
            -
            trajectory_b[step]
        )

        distances.append(distance)

    return np.asarray(distances)


# ============================================================
# RUN CLASSICAL
# ============================================================

def run_classical():

    positions = [
        np.array(
            position,
            dtype=float
        )
        for position in INITIAL_AGENTS
    ]

    trajectories = [
        [position.copy()]
        for position in positions
    ]

    inter_agent_distances = []

    for step in range(MAX_STEPS):

        new_positions = []

        for agent_index in range(2):

            current_agents = [
                tuple(position)
                for position in positions
            ]

            _, _, field = generate_field(
                current_agents,
                GOALS,
                OBSTACLES,
                current_agent=agent_index
            )

            other_agents = [
                positions[index]
                for index in range(2)
                if index != agent_index
            ]

            new_position, _ = (
                navigation_step(
                    positions[agent_index],
                    np.array(
                        GOALS[agent_index]
                    ),
                    field,
                    other_agents=other_agents
                )
            )

            new_positions.append(
                new_position
            )

        positions = new_positions

        for agent_index in range(2):

            trajectories[agent_index].append(
                positions[agent_index].copy()
            )

        inter_agent_distances.append(
            np.linalg.norm(
                positions[0]
                -
                positions[1]
            )
        )

        if all(
            reached_goal(
                positions[index],
                GOALS[index]
            )
            for index in range(2)
        ):
            break

    return (
        trajectories,
        np.asarray(
            inter_agent_distances
        )
    )


# ============================================================
# RUN QUANTUM NORMAL
# ============================================================

def run_quantum_normal():

    positions = [
        np.array(
            position,
            dtype=float
        )
        for position in INITIAL_AGENTS
    ]

    trajectories = [
        [position.copy()]
        for position in positions
    ]

    inter_agent_distances = []

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

            quantum_field, _, _ = (
                quantum_filtered_field(
                    classical_field
                )
            )

            other_agents = [
                positions[index]
                for index in range(2)
                if index != agent_index
            ]

            new_position, _ = (
                navigation_step(
                    positions[agent_index],
                    np.array(
                        GOALS[agent_index]
                    ),
                    quantum_field,
                    other_agents=other_agents
                )
            )

            new_positions.append(
                new_position
            )

        positions = new_positions

        for agent_index in range(2):

            trajectories[agent_index].append(
                positions[agent_index].copy()
            )

        inter_agent_distances.append(
            np.linalg.norm(
                positions[0]
                -
                positions[1]
            )
        )

        if all(
            reached_goal(
                positions[index],
                GOALS[index]
            )
            for index in range(2)
        ):
            break

    return (
        trajectories,
        np.asarray(
            inter_agent_distances
        )
    )


# ============================================================
# RUN QUANTUM WITH CLASSICAL AGENT-FIELD COUNTERFACTUAL
# ============================================================
#
# Important:
#
# The robots are allowed to move according to their own
# quantum-filtered navigation field.
#
# BUT the agent-interaction contribution is evaluated from
# the CLASSICAL trajectory at the same timestep.
#
# This isolates the feedback loop.
# ============================================================

def run_quantum_classical_agent_feedback(
    classical_trajectories
):

    positions = [
        np.array(
            position,
            dtype=float
        )
        for position in INITIAL_AGENTS
    ]

    trajectories = [
        [position.copy()]
        for position in positions
    ]

    inter_agent_distances = []

    for step in range(MAX_STEPS):

        new_positions = []

        for agent_index in range(2):

            # ------------------------------------------------
            # Current quantum positions
            # ------------------------------------------------

            current_quantum_agents = [
                tuple(position)
                for position in positions
            ]

            # ------------------------------------------------
            # Classical reference positions at this step
            # ------------------------------------------------

            classical_reference_agents = [
                tuple(
                    classical_trajectories[index][
                        min(
                            step,
                            len(
                                classical_trajectories[
                                    index
                                ]
                            ) - 1
                        )
                    ]
                )
                for index in range(2)
            ]

            # ------------------------------------------------
            # Generate quantum agent's normal field
            #
            # This gives us the classical decomposition
            # used by generate_field().
            # ------------------------------------------------

            _, _, quantum_full_classical_field = (
                generate_field(
                    current_quantum_agents,
                    GOALS,
                    OBSTACLES,
                    current_agent=agent_index
                )
            )

            # ------------------------------------------------
            # Generate static goal + obstacle field
            #
            # We obtain it by temporarily putting the
            # other robot far outside the grid.
            # ------------------------------------------------

            isolated_agents = [
                current_quantum_agents[agent_index],
                (1000.0, 1000.0)
            ]

            _, _, static_field = generate_field(
                isolated_agents,
                GOALS,
                OBSTACLES,
                current_agent=agent_index
            )

            # ------------------------------------------------
            # Generate classical-reference full field
            # ------------------------------------------------

            _, _, classical_reference_field = (
                generate_field(
                    classical_reference_agents,
                    GOALS,
                    OBSTACLES,
                    current_agent=agent_index
                )
            )

            # ------------------------------------------------
            # Extract agent-interaction contribution from
            # classical reference.
            #
            # The static field is kept from the quantum
            # robot's current position.
            # ------------------------------------------------

            classical_agent_field = (
                classical_reference_field
                -
                generate_field(
                    [
                        classical_reference_agents[
                            agent_index
                        ],
                        (1000.0, 1000.0)
                    ],
                    GOALS,
                    OBSTACLES,
                    current_agent=agent_index
                )[2]
            )

            # ------------------------------------------------
            # Construct counterfactual field
            # ------------------------------------------------

            counterfactual_field = (
                static_field
                +
                classical_agent_field
            )

            # ------------------------------------------------
            # Quantum reconstruction
            # ------------------------------------------------

            quantum_field, _, _ = (
                quantum_filtered_field(
                    counterfactual_field
                )
            )

            # ------------------------------------------------
            # Navigation
            #
            # IMPORTANT:
            # We still pass the current quantum positions
            # to navigation_step, so the controller's explicit
            # safety mechanism remains active.
            #
            # The field itself is what uses the classical
            # reference interaction term.
            # ------------------------------------------------

            other_agents = [
                positions[index]
                for index in range(2)
                if index != agent_index
            ]

            new_position, _ = (
                navigation_step(
                    positions[agent_index],
                    np.array(
                        GOALS[agent_index]
                    ),
                    quantum_field,
                    other_agents=other_agents
                )
            )

            new_positions.append(
                new_position
            )

        positions = new_positions

        for agent_index in range(2):

            trajectories[agent_index].append(
                positions[agent_index].copy()
            )

        inter_agent_distances.append(
            np.linalg.norm(
                positions[0]
                -
                positions[1]
            )
        )

        if all(
            reached_goal(
                positions[index],
                GOALS[index]
            )
            for index in range(2)
        ):
            break

    return (
        trajectories,
        np.asarray(
            inter_agent_distances
        )
    )


# ============================================================
# RUN EXPERIMENTS
# ============================================================

print()
print("=" * 100)
print(
    "STEP 9AB — MULTI-AGENT FEEDBACK "
    "AMPLIFICATION TEST"
)
print("=" * 100)

print()
print("Running classical...")

classical_trajectories, classical_distances = (
    run_classical()
)

print("Running normal quantum...")

quantum_trajectories, quantum_distances = (
    run_quantum_normal()
)

print(
    "Running quantum + classical-reference "
    "agent interaction..."
)

counterfactual_trajectories, counterfactual_distances = (
    run_quantum_classical_agent_feedback(
        classical_trajectories
    )
)


# ============================================================
# TRAJECTORY DIVERGENCES
# ============================================================

quantum_divergence_agent_0 = trajectory_distance(
    classical_trajectories[0],
    quantum_trajectories[0]
)

quantum_divergence_agent_1 = trajectory_distance(
    classical_trajectories[1],
    quantum_trajectories[1]
)

counter_divergence_agent_0 = trajectory_distance(
    classical_trajectories[0],
    counterfactual_trajectories[0]
)

counter_divergence_agent_1 = trajectory_distance(
    classical_trajectories[1],
    counterfactual_trajectories[1]
)


quantum_total_divergence = (
    quantum_divergence_agent_0
    +
    quantum_divergence_agent_1
)

counter_total_divergence = (
    counter_divergence_agent_0
    +
    counter_divergence_agent_1
)


# ============================================================
# RESULTS
# ============================================================

print()
print("TRAJECTORY DIVERGENCE")
print("-" * 60)

print(
    "Normal quantum maximum divergence:",
    np.max(
        quantum_total_divergence
    )
)

print(
    "Counterfactual maximum divergence:",
    np.max(
        counter_total_divergence
    )
)

print(
    "Normal quantum divergence at step 64:",
    quantum_total_divergence[
        min(
            64,
            len(
                quantum_total_divergence
            ) - 1
        )
    ]
)

print(
    "Counterfactual divergence at step 64:",
    counter_total_divergence[
        min(
            64,
            len(
                counter_total_divergence
            ) - 1
        )
    ]
)


# ============================================================
# SAFETY
# ============================================================

print()
print("SAFETY")
print("-" * 60)

print(
    "Classical minimum distance:",
    np.min(
        classical_distances
    )
)

print(
    "Normal quantum minimum distance:",
    np.min(
        quantum_distances
    )
)

print(
    "Counterfactual minimum distance:",
    np.min(
        counterfactual_distances
    )
)


# ============================================================
# SELECTED TIMESTEPS
# ============================================================

print()
print("SELECTED TIMESTEPS")
print("-" * 60)

selected_steps = [
    10,
    20,
    30,
    40,
    50,
    60,
    64,
    65,
    67,
    70,
    80,
    90
]

for step in selected_steps:

    if (
        step < len(
            quantum_total_divergence
        )
        and
        step < len(
            counter_total_divergence
        )
    ):

        print(
            f"Step {step:3d} | "
            f"Normal Q = "
            f"{quantum_total_divergence[step]:.6f} | "
            f"Counterfactual = "
            f"{counter_total_divergence[step]:.6f}"
        )


# ============================================================
# INTERPRETATION
# ============================================================

print()
print("DIAGNOSIS")
print("-" * 60)

normal_max = np.max(
    quantum_total_divergence
)

counter_max = np.max(
    counter_total_divergence
)

normal_min_distance = np.min(
    quantum_distances
)

counter_min_distance = np.min(
    counterfactual_distances
)

if (
    counter_max < 0.5 * normal_max
):

    print(
        "STRONG EVIDENCE FOR MULTI-AGENT "
        "FEEDBACK AMPLIFICATION."
    )

    print(
        "Freezing the agent-interaction reference "
        "substantially reduces trajectory divergence."
    )

elif (
    counter_max < normal_max
):

    print(
        "PARTIAL EVIDENCE FOR MULTI-AGENT "
        "FEEDBACK AMPLIFICATION."
    )

    print(
        "The counterfactual reduces divergence, "
        "but does not eliminate it."
    )

else:

    print(
        "NO CLEAR EVIDENCE THAT AGENT FEEDBACK "
        "IS THE PRIMARY AMPLIFIER."
    )

print()
print("=" * 100)

# ============================================================
# STEP 9AC — FEEDBACK AMPLIFICATION ONSET
# ============================================================

print()
print("=" * 100)
print(
    "STEP 9AC — FEEDBACK AMPLIFICATION ONSET"
)
print("=" * 100)

# Avoid division by zero at step 0
epsilon = 1e-12

ratio = (
    quantum_total_divergence
    /
    np.maximum(
        counter_total_divergence,
        epsilon
    )
)

# Ignore step 0 for onset analysis
ratio[0] = 0.0


def first_crossing(values, threshold):

    indices = np.where(
        values >= threshold
    )[0]

    if len(indices) == 0:
        return None

    return int(indices[0])


print()
print("AMPLIFICATION THRESHOLDS")
print("-" * 60)

for threshold in [2, 5, 10]:

    step = first_crossing(
        ratio,
        threshold
    )

    if step is None:

        print(
            f"First ratio >= {threshold:2d}x : "
            "not reached"
        )

    else:

        print(
            f"First ratio >= {threshold:2d}x : "
            f"step {step}"
        )

        print(
            f"    Normal divergence      : "
            f"{quantum_total_divergence[step]:.6f}"
        )

        print(
            f"    Counterfactual         : "
            f"{counter_total_divergence[step]:.6f}"
        )

        if step - 1 >= 0:

            print(
                f"    Previous-step ratio    : "
                f"{ratio[step - 1]:.3f}x"
            )

        print(
            f"    Current ratio          : "
            f"{ratio[step]:.3f}x"
        )


# ============================================================
# MAXIMUM AMPLIFICATION RATIO
# ============================================================

max_ratio_step = int(
    np.argmax(ratio)
)

max_ratio = ratio[
    max_ratio_step
]

print()
print("MAXIMUM AMPLIFICATION")
print("-" * 60)

print(
    f"Maximum divergence ratio : "
    f"{max_ratio:.3f}x"
)

print(
    f"Occurred at step         : "
    f"{max_ratio_step}"
)

print(
    f"Normal divergence        : "
    f"{quantum_total_divergence[max_ratio_step]:.6f}"
)

print(
    f"Counterfactual divergence: "
    f"{counter_total_divergence[max_ratio_step]:.6f}"
)


# ============================================================
# SAFETY DISTANCES AT AMPLIFICATION ONSET
# ============================================================

print()
print("SAFETY DISTANCE AT AMPLIFICATION")
print("-" * 60)

for threshold in [2, 5, 10]:

    step = first_crossing(
        ratio,
        threshold
    )

    if step is not None:

        # Distances arrays are indexed from step 1,
        # while divergence arrays include step 0.
        distance_index = min(
            max(step - 1, 0),
            len(quantum_distances) - 1
        )

        print()
        print(
            f"{threshold}x amplification "
            f"at step {step}"
        )

        print(
            f"    Quantum distance      : "
            f"{quantum_distances[distance_index]:.6f}"
        )

        print(
            f"    Counterfactual distance: "
            f"{counterfactual_distances[distance_index]:.6f}"
        )


# ============================================================
# SELECTED RATIO VALUES
# ============================================================

print()
print("AMPLIFICATION EVOLUTION")
print("-" * 60)

for step in [
    10,
    20,
    30,
    40,
    45,
    50,
    55,
    60,
    64,
    65,
    67,
    70,
    80,
    90
]:

    if step < len(ratio):

        print(
            f"Step {step:3d} | "
            f"Q divergence = "
            f"{quantum_total_divergence[step]:.6f} | "
            f"CF divergence = "
            f"{counter_total_divergence[step]:.6f} | "
            f"Amplification = "
            f"{ratio[step]:.3f}x"
        )


print()
print("=" * 100)