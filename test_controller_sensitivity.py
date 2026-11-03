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
        # Store timestep information for BOTH agents
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
                "agent": agent_index,

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
# RUN
# ============================================================

print()
print("=" * 100)
print(
    "STEP 9AD — CONTROLLER REGIME "
    "TRANSITION ANALYSIS"
)
print("=" * 100)

print()
print("Running diagnostic...")

records = run_diagnostic()


# ============================================================
# FOCUS ON TRANSITION REGION
# ============================================================

print()
print("TRANSITION REGION")
print("-" * 100)

print(
    "Step | Agent | "
    "d_class | d_quant | "
    "|dv| | "
    "classical_goal | quantum_goal | "
    "pos_diff | "
    "bandwidth | power"
)

print("-" * 100)

for record in records:

    step = record["step"]

    if 45 <= step <= 67:

        for data in record["agents"]:

            print(
                f"{step:4d} | "
                f"{data['agent']:5d} | "
                f"{data['classical_inter_agent_distance']:.4f} | "
                f"{data['quantum_inter_agent_distance']:.4f} | "
                f"{data['velocity_difference']:.4f} | "
                f"{data['classical_goal_distance']:.4f} | "
                f"{data['quantum_goal_distance']:.4f} | "
                f"{data['position_difference']:.4f} | "
                f"{data['bandwidth']:.6f} | "
                f"{data['retained_power']:.6f}"
            )


# ============================================================
# FIND MAXIMUM VELOCITY DIFFERENCE
# ============================================================

all_velocity_differences = []

for record in records:

    for data in record["agents"]:

        all_velocity_differences.append(
            (
                data["velocity_difference"],
                record["step"],
                data["agent"]
            )
        )

all_velocity_differences.sort(
    reverse=True
)

print()
print("LARGEST VELOCITY DIFFERENCES")
print("-" * 100)

for value, step, agent in (
    all_velocity_differences[:10]
):

    print(
        f"Step {step:3d} | "
        f"Agent {agent} | "
        f"|vQ-vC| = {value:.6f}"
    )


# ============================================================
# FOCUS ON STEPS 58–61
# ============================================================

print()
print("CRITICAL TRANSITION: STEPS 58–61")
print("-" * 100)

for record in records:

    if record["step"] in [58, 59, 60, 61]:

        print(
            f"\nSTEP {record['step']}"
        )

        for data in record["agents"]:

            print(
                f"Agent {data['agent']}:"
            )

            print(
                f"  Classical velocity : "
                f"{data['classical_velocity']}"
            )

            print(
                f"  Quantum velocity   : "
                f"{data['quantum_velocity']}"
            )

            print(
                f"  Velocity difference: "
                f"{data['velocity_difference']:.6f}"
            )

            print(
                f"  Classical distance : "
                f"{data['classical_inter_agent_distance']:.6f}"
            )

            print(
                f"  Quantum distance   : "
                f"{data['quantum_inter_agent_distance']:.6f}"
            )

            print(
                f"  Position difference: "
                f"{data['position_difference']:.6f}"
            )


# ============================================================
# OVERALL
# ============================================================

max_velocity_difference = max(
    all_velocity_differences,
    key=lambda x: x[0]
)

print()
print("=" * 100)
print("9AD COMPLETE")
print("=" * 100)

print(
    f"Maximum |vQ-vC| : "
    f"{max_velocity_difference[0]:.6f}"
)

print(
    f"Occurred at step: "
    f"{max_velocity_difference[1]}"
)

print(
    f"Agent            : "
    f"{max_velocity_difference[2]}"
)

print("=" * 100)