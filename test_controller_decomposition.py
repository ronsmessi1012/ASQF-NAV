import numpy as np

from environment.field import generate_field

from navigation.controller import (
    compute_velocity,
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
# QUANTUM FIELD RECONSTRUCTION
# ============================================================

def statevector_to_field(statevector):

    amplitudes = np.asarray(
        statevector.data,
        dtype=complex
    )

    return amplitudes.reshape(16, 16)


def quantum_filtered_field(field):

    field_norm = np.linalg.norm(field)

    normalized_field = (
        field / field_norm
    )

    circuit, _ = encode_field(
        normalized_field
    )

    circuit = apply_qft_2d(circuit)

    qft_state = Statevector.from_instruction(
        circuit
    )

    spectrum = statevector_to_field(
        qft_state
    )

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

    filter_result = adaptive_2d_filter(
        spectrum,
        radial_frequency,
        normalized_complexity
    )

    filtered_spectrum = (
        filter_result["filtered_spectrum"]
    )

    filtered_state = (
        filtered_spectrum.flatten()
    )

    filtered_state /= np.linalg.norm(
        filtered_state
    )

    inverse_circuit = QuantumCircuit(8)

    inverse_circuit.initialize(
        filtered_state,
        range(8)
    )

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
# 9AE — EXACT CONTROLLER DECOMPOSITION
# ============================================================

def bilinear_interpolate(grid, x, y):

    height, width = grid.shape

    x = np.clip(
        x,
        0,
        width - 1
    )

    y = np.clip(
        y,
        0,
        height - 1
    )

    x0 = int(np.floor(x))
    x1 = min(
        x0 + 1,
        width - 1
    )

    y0 = int(np.floor(y))
    y1 = min(
        y0 + 1,
        height - 1
    )

    dx = x - x0
    dy = y - y0

    return (
        grid[y0, x0] * (1 - dx) * (1 - dy)
        + grid[y0, x1] * dx * (1 - dy)
        + grid[y1, x0] * (1 - dx) * dy
        + grid[y1, x1] * dx * dy
    )


def decompose_velocity(
    field,
    position,
    goal,
    other_agents=None
):

    # ========================================================
    # 1. RAW APF GRADIENT
    # ========================================================

    grad_y, grad_x = np.gradient(field)

    px, py = position

    gx = bilinear_interpolate(
        grad_x,
        px,
        py
    )

    gy = bilinear_interpolate(
        grad_y,
        px,
        py
    )

    gradient = np.array(
        [gx, gy],
        dtype=float
    )

    # Controller uses negative gradient
    raw_apf = -gradient

    gradient_magnitude = np.linalg.norm(
        raw_apf
    )


    # ========================================================
    # 2. NORMALIZED APF
    # ========================================================

    if gradient_magnitude > 1e-10:

        normalized_apf = (
            raw_apf
            / gradient_magnitude
        )

    else:

        normalized_apf = np.zeros(2)


    # ========================================================
    # 3. GOAL DIRECTION
    # ========================================================

    goal_direction = (
        np.array(goal, dtype=float)
        -
        np.array(position, dtype=float)
    )

    goal_distance = np.linalg.norm(
        goal_direction
    )

    if goal_distance > 1e-10:

        goal_direction = (
            goal_direction
            / goal_distance
        )

    else:

        goal_direction = np.zeros(2)


    # ========================================================
    # 4. APF + GOAL
    # ========================================================

    base_velocity = (
        0.65 * normalized_apf
        +
        0.35 * goal_direction
    )


    # ========================================================
    # 5. AGENT AVOIDANCE
    # ========================================================

    SAFE_DISTANCE = 1.05
    AVOIDANCE_DISTANCE = 1.5

    avoidance_vector = np.zeros(2)

    avoidance_mode = "none"

    agent_distances = []

    radial_projection = None

    for other in (
        other_agents
        if other_agents is not None
        else []
    ):

        relative_position = (
            np.array(position, dtype=float)
            -
            np.array(other, dtype=float)
        )

        distance = np.linalg.norm(
            relative_position
        )

        agent_distances.append(
            distance
        )

        if (
            distance > 1e-10
            and
            distance < AVOIDANCE_DISTANCE
        ):

            away_direction = (
                relative_position
                / distance
            )

            # ------------------------------------------------
            # Early avoidance
            # ------------------------------------------------

            if distance > SAFE_DISTANCE:

                avoidance_mode = (
                    "early_avoidance"
                )

                avoidance_strength = (
                    (
                        AVOIDANCE_DISTANCE
                        -
                        distance
                    )
                    /
                    (
                        AVOIDANCE_DISTANCE
                        -
                        SAFE_DISTANCE
                    )
                )

                avoidance_strength = np.clip(
                    avoidance_strength,
                    0.0,
                    1.0
                )

                avoidance_vector += (
                    avoidance_strength
                    * away_direction
                )

            # ------------------------------------------------
            # Safety projection
            # ------------------------------------------------

            else:

                avoidance_mode = (
                    "safety_projection"
                )

                radial_velocity = np.dot(
                    base_velocity
                    +
                    avoidance_vector,
                    away_direction
                )

                radial_projection = (
                    radial_velocity
                )

                if radial_velocity < 0:

                    avoidance_vector -= (
                        radial_velocity
                        *
                        away_direction
                    )

                avoidance_vector += (
                    (
                        SAFE_DISTANCE
                        -
                        distance
                    )
                    *
                    away_direction
                )


    # ========================================================
    # 6. PRE-NORMALIZATION VELOCITY
    # ========================================================

    pre_normalization_velocity = (
        base_velocity
        +
        avoidance_vector
    )


    # ========================================================
    # 7. FINAL NORMALIZATION
    # ========================================================

    magnitude = np.linalg.norm(
        pre_normalization_velocity
    )

    if magnitude > 1e-10:

        final_velocity = (
            pre_normalization_velocity
            /
            magnitude
        )

    else:

        final_velocity = np.zeros(2)


    # ========================================================
    # RETURN ALL TERMS
    # ========================================================

    return {

        "position":
            np.array(position),

        "raw_apf":
            raw_apf,

        "normalized_apf":
            normalized_apf,

        "goal_direction":
            goal_direction,

        "base_velocity":
            base_velocity,

        "avoidance_vector":
            avoidance_vector,

        "pre_normalization_velocity":
            pre_normalization_velocity,

        "final_velocity":
            final_velocity,

        "gradient_magnitude":
            gradient_magnitude,

        "goal_distance":
            goal_distance,

        "agent_distances":
            agent_distances,

        "avoidance_mode":
            avoidance_mode,

        "radial_projection":
            radial_projection
    }


# ============================================================
# RUN TRAJECTORIES
# ============================================================

def run_diagnostic():

    classical_positions = [
        np.array(p, dtype=float)
        for p in INITIAL_AGENTS
    ]

    quantum_positions = [
        np.array(p, dtype=float)
        for p in INITIAL_AGENTS
    ]

    records = []


    for step in range(MAX_STEPS):

        classical_new = []
        quantum_new = []

        # ----------------------------------------------------
        # Store timestep information
        # ----------------------------------------------------

        step_records = []


        for agent_index in range(2):

            # =================================================
            # CURRENT AGENTS
            # =================================================

            classical_agents = [
                tuple(p)
                for p in classical_positions
            ]

            quantum_agents = [
                tuple(p)
                for p in quantum_positions
            ]

            goal = np.array(
                GOALS[agent_index],
                dtype=float
            )


            # =================================================
            # CLASSICAL FIELD
            # =================================================

            _, _, classical_field = (
                generate_field(
                    classical_agents,
                    GOALS,
                    OBSTACLES,
                    current_agent=agent_index
                )
            )


            # =================================================
            # QUANTUM FIELD
            # =================================================

            _, _, quantum_input_field = (
                generate_field(
                    quantum_agents,
                    GOALS,
                    OBSTACLES,
                    current_agent=agent_index
                )
            )

            quantum_field, complexity, filter_result = (
                quantum_filtered_field(
                    quantum_input_field
                )
            )


            # =================================================
            # OTHER AGENTS
            # =================================================

            classical_other = [
                classical_positions[index]
                for index in range(2)
                if index != agent_index
            ]

            quantum_other = [
                quantum_positions[index]
                for index in range(2)
                if index != agent_index
            ]


            # =================================================
            # VELOCITIES
            # =================================================

            classical_velocity = compute_velocity(
                classical_field,
                classical_positions[agent_index],
                goal,
                classical_other
            )

            quantum_velocity = compute_velocity(
                quantum_field,
                quantum_positions[agent_index],
                goal,
                quantum_other
            )


            # =================================================
            # VELOCITY DIFFERENCE
            # =================================================

            velocity_difference = np.linalg.norm(
                quantum_velocity
                -
                classical_velocity
            )


            # =================================================
            # INTER-AGENT DISTANCES
            # =================================================

            classical_inter_agent_distance = (
                np.linalg.norm(
                    classical_positions[0]
                    -
                    classical_positions[1]
                )
            )

            quantum_inter_agent_distance = (
                np.linalg.norm(
                    quantum_positions[0]
                    -
                    quantum_positions[1]
                )
            )


            # =================================================
            # GOAL DISTANCE
            # =================================================

            classical_goal_distance = np.linalg.norm(
                classical_positions[agent_index]
                -
                goal
            )

            quantum_goal_distance = np.linalg.norm(
                quantum_positions[agent_index]
                -
                goal
            )


            # =================================================
            # POSITION DIFFERENCE
            # =================================================

            position_difference = np.linalg.norm(
                quantum_positions[agent_index]
                -
                classical_positions[agent_index]
            )


            # =================================================
            # RECORD
            # =================================================

            step_records.append({

                "agent":
                    agent_index,

                "classical_position":
                    classical_positions[
                        agent_index
                    ].copy(),

                "quantum_position":
                    quantum_positions[
                        agent_index
                    ].copy(),

                "classical_velocity":
                    classical_velocity.copy(),

                "quantum_velocity":
                    quantum_velocity.copy(),

                "velocity_difference":
                    velocity_difference,

                "classical_inter_agent_distance":
                    classical_inter_agent_distance,

                "quantum_inter_agent_distance":
                    quantum_inter_agent_distance,

                "classical_goal_distance":
                    classical_goal_distance,

                "quantum_goal_distance":
                    quantum_goal_distance,

                "position_difference":
                    position_difference,

                "complexity":
                    complexity,

                "bandwidth":
                    filter_result[
                        "bandwidth"
                    ],

                "retained_power":
                    filter_result[
                        "retained_power_fraction"
                    ],

                "classical_field":
                    classical_field.copy(),

                "quantum_field":
                    quantum_field.copy(),

                "classical_other":
                    [
                        p.copy()
                        for p in classical_other
                    ],

                "quantum_other":
                    [
                        p.copy()
                        for p in quantum_other
                    ]
            })


            # =================================================
            # NEXT POSITIONS
            # =================================================

            classical_new.append(
                classical_positions[
                    agent_index
                ]
                +
                0.20 * classical_velocity
            )

            quantum_new.append(
                quantum_positions[
                    agent_index
                ]
                +
                0.20 * quantum_velocity
            )


        # ----------------------------------------------------
        # Update positions
        # ----------------------------------------------------

        classical_positions = classical_new
        quantum_positions = quantum_new

        records.append(
            {
                "step": step,
                "agents": step_records
            }
        )


        # ----------------------------------------------------
        # Stop after enough trajectory
        # ----------------------------------------------------

        if (
            all(
                reached_goal(
                    classical_positions[i],
                    GOALS[i]
                )
                for i in range(2)
            )
            and
            all(
                reached_goal(
                    quantum_positions[i],
                    GOALS[i]
                )
                for i in range(2)
            )
        ):

            break


    return records


# ============================================================
# 9AE — CONTROLLER DECOMPOSITION
# ============================================================

def print_decomposition(
    step,
    agent,
    classical_data,
    quantum_data
):

    print()
    print("=" * 100)
    print(
        f"STEP {step} — AGENT {agent} — "
        f"CONTROLLER DECOMPOSITION"
    )
    print("=" * 100)


    # ========================================================
    # CLASSICAL
    # ========================================================

    classical_diag = decompose_velocity(
        classical_data["classical_field"],
        classical_data["classical_position"],
        GOALS[agent],
        classical_data["classical_other"]
    )


    print()
    print("CLASSICAL")
    print("-" * 100)

    print(
        "Position:",
        classical_diag["position"]
    )

    print(
        "Raw APF:",
        classical_diag["raw_apf"]
    )

    print(
        "Normalized APF:",
        classical_diag["normalized_apf"]
    )

    print(
        "Goal direction:",
        classical_diag["goal_direction"]
    )

    print(
        "Base velocity:",
        classical_diag["base_velocity"]
    )

    print(
        "Avoidance vector:",
        classical_diag["avoidance_vector"]
    )

    print(
        "Pre-normalization velocity:",
        classical_diag[
            "pre_normalization_velocity"
        ]
    )

    print(
        "Final velocity:",
        classical_diag["final_velocity"]
    )

    print(
        "Agent distances:",
        classical_diag["agent_distances"]
    )

    print(
        "Avoidance mode:",
        classical_diag["avoidance_mode"]
    )


    # ========================================================
    # QUANTUM
    # ========================================================

    quantum_diag = decompose_velocity(
        quantum_data["quantum_field"],
        quantum_data["quantum_position"],
        GOALS[agent],
        quantum_data["quantum_other"]
    )


    print()
    print("QUANTUM FILTERED")
    print("-" * 100)

    print(
        "Position:",
        quantum_diag["position"]
    )

    print(
        "Raw APF:",
        quantum_diag["raw_apf"]
    )

    print(
        "Normalized APF:",
        quantum_diag["normalized_apf"]
    )

    print(
        "Goal direction:",
        quantum_diag["goal_direction"]
    )

    print(
        "Base velocity:",
        quantum_diag["base_velocity"]
    )

    print(
        "Avoidance vector:",
        quantum_diag["avoidance_vector"]
    )

    print(
        "Pre-normalization velocity:",
        quantum_diag[
            "pre_normalization_velocity"
        ]
    )

    print(
        "Final velocity:",
        quantum_diag["final_velocity"]
    )

    print(
        "Agent distances:",
        quantum_diag["agent_distances"]
    )

    print(
        "Avoidance mode:",
        quantum_diag["avoidance_mode"]
    )


    # ========================================================
    # DIRECT COMPARISON
    # ========================================================

    print()
    print("DIRECT COMPARISON")
    print("-" * 100)

    print(
        "Raw APF difference:",
        np.linalg.norm(
            quantum_diag["raw_apf"]
            -
            classical_diag["raw_apf"]
        )
    )

    print(
        "Normalized APF difference:",
        np.linalg.norm(
            quantum_diag["normalized_apf"]
            -
            classical_diag["normalized_apf"]
        )
    )

    print(
        "Goal direction difference:",
        np.linalg.norm(
            quantum_diag["goal_direction"]
            -
            classical_diag["goal_direction"]
        )
    )

    print(
        "Base velocity difference:",
        np.linalg.norm(
            quantum_diag["base_velocity"]
            -
            classical_diag["base_velocity"]
        )
    )

    print(
        "Avoidance difference:",
        np.linalg.norm(
            quantum_diag["avoidance_vector"]
            -
            classical_diag["avoidance_vector"]
        )
    )

    print(
        "Pre-normalization difference:",
        np.linalg.norm(
            quantum_diag[
                "pre_normalization_velocity"
            ]
            -
            classical_diag[
                "pre_normalization_velocity"
            ]
        )
    )

    print(
        "Final velocity difference:",
        np.linalg.norm(
            quantum_diag["final_velocity"]
            -
            classical_diag["final_velocity"]
        )
    )

    print()
    print(
        "Reported controller velocity difference:",
        np.linalg.norm(
            quantum_data["quantum_velocity"]
            -
            classical_data["classical_velocity"]
        )
    )


# ============================================================
# RUN
# ============================================================

print()
print("=" * 100)
print(
    "STEP 9AE — EXACT CONTROLLER DECOMPOSITION"
)
print("=" * 100)

print()
print("Running diagnostic...")

records = run_diagnostic()


# ============================================================
# CRITICAL STEPS
# ============================================================

print()
print("=" * 100)
print(
    "CRITICAL CONTROLLER DECOMPOSITION — "
    "STEPS 57–61"
)
print("=" * 100)


for record in records:

    step = record["step"]

    if step in [57, 58, 59, 60, 61]:

        for data in record["agents"]:

            print_decomposition(
                step,
                data["agent"],
                data,
                data
            )


# ============================================================
# SPECIAL FOCUS — AGENT 1, STEP 59
# ============================================================

print()
print("=" * 100)
print(
    "PRIMARY EVENT — STEP 59, AGENT 1"
)
print("=" * 100)


target_record = None

for record in records:

    if record["step"] == 59:

        target_record = record
        break


if target_record is not None:

    target_data = None

    for data in target_record["agents"]:

        if data["agent"] == 1:

            target_data = data
            break


    if target_data is not None:

        print()
        print(
            "Classical position:",
            target_data[
                "classical_position"
            ]
        )

        print(
            "Quantum position:",
            target_data[
                "quantum_position"
            ]
        )

        print()
        print(
            "Classical velocity:",
            target_data[
                "classical_velocity"
            ]
        )

        print(
            "Quantum velocity:",
            target_data[
                "quantum_velocity"
            ]
        )

        print()
        print(
            "Classical inter-agent distance:",
            target_data[
                "classical_inter_agent_distance"
            ]
        )

        print(
            "Quantum inter-agent distance:",
            target_data[
                "quantum_inter_agent_distance"
            ]
        )

        print()
        print(
            "Classical field → controller:"
        )

        classical_diag = decompose_velocity(
            target_data["classical_field"],
            target_data["classical_position"],
            GOALS[1],
            target_data["classical_other"]
        )

        print(
            "  Raw APF:",
            classical_diag["raw_apf"]
        )

        print(
            "  Normalized APF:",
            classical_diag["normalized_apf"]
        )

        print(
            "  Goal direction:",
            classical_diag["goal_direction"]
        )

        print(
            "  Base velocity:",
            classical_diag["base_velocity"]
        )

        print(
            "  Avoidance:",
            classical_diag["avoidance_vector"]
        )

        print(
            "  Final:",
            classical_diag["final_velocity"]
        )

        print()
        print(
            "Quantum field → controller:"
        )

        quantum_diag = decompose_velocity(
            target_data["quantum_field"],
            target_data["quantum_position"],
            GOALS[1],
            target_data["quantum_other"]
        )

        print(
            "  Raw APF:",
            quantum_diag["raw_apf"]
        )

        print(
            "  Normalized APF:",
            quantum_diag["normalized_apf"]
        )

        print(
            "  Goal direction:",
            quantum_diag["goal_direction"]
        )

        print(
            "  Base velocity:",
            quantum_diag["base_velocity"]
        )

        print(
            "  Avoidance:",
            quantum_diag["avoidance_vector"]
        )

        print(
            "  Final:",
            quantum_diag["final_velocity"]
        )


# ============================================================
# FINISH
# ============================================================

print()
print("=" * 100)
print("9AE COMPLETE")
print("=" * 100)