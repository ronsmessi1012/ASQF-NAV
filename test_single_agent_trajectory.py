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
# STATEVECTOR → 16 × 16 FIELD
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
# QUANTUM FIELD RECONSTRUCTION
# ============================================================

def quantum_reconstruct_field(field):

    # --------------------------------------------------------
    # Normalize spatial field
    # --------------------------------------------------------

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
    # Convert quantum state to spectrum
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
    # Adaptive spectral filtering
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
    # Normalize filtered complex state
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
    # Load filtered spectrum into Qiskit
    # --------------------------------------------------------

    filtered_circuit = QuantumCircuit(8)

    filtered_circuit.initialize(
        filtered_state,
        range(8)
    )

    # --------------------------------------------------------
    # Actual inverse 2-D QFT
    # --------------------------------------------------------

    filtered_circuit = apply_inverse_qft_2d(
        filtered_circuit
    )

    reconstructed_state = (
        Statevector.from_instruction(
            filtered_circuit
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
    # Restore original field scale
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
        filter_result
    )


# ============================================================
# ENVIRONMENT
# ============================================================

agents = [
    (2.0, 2.0)
]

goals = [
    (13.0, 13.0)
]

obstacles = [
    (8.0, 9.0),
    (5.0, 11.0)
]


# ============================================================
# TRAJECTORY SETTINGS
# ============================================================

START = np.array(
    [2.0, 2.0],
    dtype=float
)

GOAL = np.array(
    [13.0, 13.0],
    dtype=float
)

MAX_STEPS = 200


# ============================================================
# CLASSICAL TRAJECTORY
# ============================================================

classical_position = START.copy()

classical_path = [
    classical_position.copy()
]

classical_complexities = []


for step in range(MAX_STEPS):

    # Generate field at current position
    agents[0] = tuple(
        classical_position
    )

    _, _, classical_field = generate_field(
        agents,
        goals,
        obstacles,
        current_agent=0
    )

    # Navigation step
    new_position, velocity = (
        navigation_step(
            classical_position,
            GOAL,
            classical_field,
            other_agents=None
        )
    )

    # Track movement
    classical_position = new_position

    classical_path.append(
        classical_position.copy()
    )

    # Stop if goal reached
    if reached_goal(
        classical_position,
        GOAL
    ):
        break


classical_steps = len(
    classical_path
) - 1


# ============================================================
# QUANTUM TRAJECTORY
# ============================================================

quantum_position = START.copy()

quantum_path = [
    quantum_position.copy()
]

quantum_complexities = []

quantum_bandwidths = []

quantum_reconstruction_errors = []

quantum_retained_powers = []


for step in range(MAX_STEPS):

    # Update agent position
    agents[0] = tuple(
        quantum_position
    )

    # Generate classical interaction field
    _, _, classical_field = generate_field(
        agents,
        goals,
        obstacles,
        current_agent=0
    )

    # Quantum spectral processing
    (
        quantum_field,
        complexity,
        filter_result
    ) = quantum_reconstruct_field(
        classical_field
    )

    # Track quantum statistics
    quantum_complexities.append(
        complexity
    )

    quantum_bandwidths.append(
        filter_result["bandwidth"]
    )

    quantum_retained_powers.append(
        filter_result[
            "retained_power_fraction"
        ]
    )

    reconstruction_error = (
        np.linalg.norm(
            classical_field -
            quantum_field
        )
        /
        np.linalg.norm(
            classical_field
        )
    )

    quantum_reconstruction_errors.append(
        reconstruction_error
    )

    # Navigation using reconstructed field
    new_position, velocity = (
        navigation_step(
            quantum_position,
            GOAL,
            quantum_field,
            other_agents=None
        )
    )

    quantum_position = new_position

    quantum_path.append(
        quantum_position.copy()
    )

    # Stop if goal reached
    if reached_goal(
        quantum_position,
        GOAL
    ):
        break


quantum_steps = len(
    quantum_path
) - 1


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


classical_path_length = (
    compute_path_length(
        classical_path
    )
)

quantum_path_length = (
    compute_path_length(
        quantum_path
    )
)


# ============================================================
# FINAL GOAL ERRORS
# ============================================================

classical_goal_error = np.linalg.norm(
    classical_position -
    GOAL
)

quantum_goal_error = np.linalg.norm(
    quantum_position -
    GOAL
)


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 100)
print(
    "STEP 9X — SINGLE-AGENT FULL TRAJECTORY"
)
print("=" * 100)

print()
print("Start position:")
print(START)

print()
print("Goal position:")
print(GOAL)

print()
print("----------- CLASSICAL -----------")

print(
    "Steps              :",
    classical_steps
)

print(
    "Path length        :",
    classical_path_length
)

print(
    "Final position     :",
    classical_position
)

print(
    "Final goal error   :",
    classical_goal_error
)

print()
print("------------ QUANTUM ------------")

print(
    "Steps              :",
    quantum_steps
)

print(
    "Path length        :",
    quantum_path_length
)

print(
    "Final position     :",
    quantum_position
)

print(
    "Final goal error   :",
    quantum_goal_error
)

print()
print("------ QUANTUM STATISTICS ------")

print(
    "Average complexity       :",
    np.mean(
        quantum_complexities
    )
)

print(
    "Average bandwidth        :",
    np.mean(
        quantum_bandwidths
    )
)

print(
    "Average retained power   :",
    np.mean(
        quantum_retained_powers
    )
)

print(
    "Average reconstruction error:",
    np.mean(
        quantum_reconstruction_errors
    )
)

print(
    "Maximum reconstruction error:",
    np.max(
        quantum_reconstruction_errors
    )
)

# ============================================================
# VALIDATION
# ============================================================

classical_reached = (
    classical_goal_error <= 0.45
)

quantum_reached = (
    quantum_goal_error <= 0.45
)

quantum_reconstruction_stable = (
    np.max(
        quantum_reconstruction_errors
    ) < 0.30
)

print()
print("----------- VALIDATION -----------")

print(
    "Classical reached goal :",
    "PASS" if classical_reached else "FAIL"
)

print(
    "Quantum reached goal   :",
    "PASS" if quantum_reached else "FAIL"
)

print(
    "Quantum reconstruction :",
    "PASS"
    if quantum_reconstruction_stable
    else "FAIL"
)

overall = (
    classical_reached
    and quantum_reached
    and quantum_reconstruction_stable
)

print()
print(
    "OVERALL RESULT         :",
    "PASS" if overall else "FAIL"
)

print("=" * 100)