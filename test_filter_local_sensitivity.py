import numpy as np

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
    # Encode
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

    spectrum = statevector_to_field(state)

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
    # Normalize filtered state
    # --------------------------------------------------------

    filtered_state = (
        filtered_spectrum.flatten()
    )

    filtered_state /= np.linalg.norm(
        filtered_state
    )

    # --------------------------------------------------------
    # Reload into quantum state
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
# BILINEAR FIELD SAMPLING
# ============================================================

def sample_field(field, position):

    x, y = position

    x = np.clip(
        x,
        0,
        field.shape[1] - 1
    )

    y = np.clip(
        y,
        0,
        field.shape[0] - 1
    )

    x0 = int(np.floor(x))
    x1 = min(
        x0 + 1,
        field.shape[1] - 1
    )

    y0 = int(np.floor(y))
    y1 = min(
        y0 + 1,
        field.shape[0] - 1
    )

    dx = x - x0
    dy = y - y0

    value = (
        field[y0, x0] * (1 - dx) * (1 - dy)
        +
        field[y0, x1] * dx * (1 - dy)
        +
        field[y1, x0] * (1 - dx) * dy
        +
        field[y1, x1] * dx * dy
    )

    return value


# ============================================================
# BILINEAR GRADIENT
# ============================================================

def compute_gradient(field, position):

    x, y = position

    x = np.clip(
        x,
        0,
        field.shape[1] - 1
    )

    y = np.clip(
        y,
        0,
        field.shape[0] - 1
    )

    gx = np.gradient(
        field,
        axis=1
    )

    gy = np.gradient(
        field,
        axis=0
    )

    gradient_x = sample_field(
        gx,
        (x, y)
    )

    gradient_y = sample_field(
        gy,
        (x, y)
    )

    return np.array([
        gradient_x,
        gradient_y
    ])


# ============================================================
# SIMULATION
# ============================================================

positions = [
    np.array(position, dtype=float)
    for position in INITIAL_AGENTS
]

records = []

for step in range(MAX_STEPS):

    for agent_index in range(2):

        current_agents = [
            tuple(position)
            for position in positions
        ]

        # ----------------------------------------------------
        # Classical field
        # ----------------------------------------------------

        _, _, classical_field = (
            generate_field(
                current_agents,
                GOALS,
                OBSTACLES,
                current_agent=agent_index
            )
        )

        # ----------------------------------------------------
        # Quantum filtered field
        # ----------------------------------------------------

        (
            quantum_field,
            complexity,
            filter_result
        ) = quantum_filtered_field(
            classical_field
        )

        # ----------------------------------------------------
        # Local position
        # ----------------------------------------------------

        position = positions[agent_index]

        # ----------------------------------------------------
        # Gradients
        # ----------------------------------------------------

        classical_gradient = (
            compute_gradient(
                classical_field,
                position
            )
        )

        quantum_gradient = (
            compute_gradient(
                quantum_field,
                position
            )
        )

        classical_magnitude = (
            np.linalg.norm(
                classical_gradient
            )
        )

        quantum_magnitude = (
            np.linalg.norm(
                quantum_gradient
            )
        )

        # ----------------------------------------------------
        # Gradient direction agreement
        # ----------------------------------------------------

        if (
            classical_magnitude > 1e-12
            and
            quantum_magnitude > 1e-12
        ):

            direction_cosine = (
                np.dot(
                    classical_gradient,
                    quantum_gradient
                )
                /
                (
                    classical_magnitude
                    *
                    quantum_magnitude
                )
            )

        else:

            direction_cosine = np.nan

        # ----------------------------------------------------
        # Local field error
        # ----------------------------------------------------

        classical_value = sample_field(
            classical_field,
            position
        )

        quantum_value = sample_field(
            quantum_field,
            position
        )

        local_field_error = (
            abs(
                classical_value -
                quantum_value
            )
            /
            max(
                abs(classical_value),
                1e-12
            )
        )

        # ----------------------------------------------------
        # Other robot distance
        # ----------------------------------------------------

        other_index = 1 - agent_index

        inter_agent_distance = np.linalg.norm(
            positions[agent_index]
            -
            positions[other_index]
        )

        # ----------------------------------------------------
        # Record
        # ----------------------------------------------------

        records.append({
            "step": step,
            "agent": agent_index,
            "distance": inter_agent_distance,
            "complexity": complexity,
            "bandwidth": filter_result[
                "bandwidth"
            ],
            "retained_power": filter_result[
                "retained_power_fraction"
            ],
            "classical_gradient": classical_magnitude,
            "quantum_gradient": quantum_magnitude,
            "gradient_ratio": (
                quantum_magnitude /
                classical_magnitude
                if classical_magnitude > 1e-12
                else np.nan
            ),
            "direction_cosine": direction_cosine,
            "local_field_error": local_field_error
        })

    # --------------------------------------------------------
    # We only need the field diagnostics here.
    #
    # Reproduce the original trajectory using the
    # existing controller.
    # --------------------------------------------------------

    new_positions = []

    from navigation.controller import navigation_step

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

        new_position, _ = navigation_step(
            positions[agent_index],
            np.array(GOALS[agent_index]),
            quantum_field,
            other_agents=other_agents
        )

        new_positions.append(
            new_position
        )

    positions = new_positions


# ============================================================
# ANALYSIS
# ============================================================

print()
print("=" * 100)
print(
    "STEP 9Z — LOCAL FILTER SENSITIVITY"
)
print("=" * 100)

distances = np.array([
    record["distance"]
    for record in records
])

cosines = np.array([
    record["direction_cosine"]
    for record in records
])

gradient_ratios = np.array([
    record["gradient_ratio"]
    for record in records
])

local_errors = np.array([
    record["local_field_error"]
    for record in records
])


# ============================================================
# GLOBAL
# ============================================================

print()
print("GLOBAL STATISTICS")
print("-" * 60)

print(
    "Minimum inter-agent distance:",
    np.min(distances)
)

print(
    "Mean gradient direction cosine:",
    np.nanmean(cosines)
)

print(
    "Minimum gradient direction cosine:",
    np.nanmin(cosines)
)

print(
    "Mean gradient magnitude ratio:",
    np.nanmean(gradient_ratios)
)

print(
    "Mean local field error:",
    np.nanmean(local_errors)
)


# ============================================================
# CRITICAL REGION
# ============================================================

critical_mask = distances <= 1.20

print()
print("CRITICAL REGION (distance <= 1.20)")
print("-" * 60)

if np.any(critical_mask):

    print(
        "Samples:",
        np.sum(critical_mask)
    )

    print(
        "Mean direction cosine:",
        np.nanmean(
            cosines[critical_mask]
        )
    )

    print(
        "Minimum direction cosine:",
        np.nanmin(
            cosines[critical_mask]
        )
    )

    print(
        "Mean gradient ratio:",
        np.nanmean(
            gradient_ratios[critical_mask]
        )
    )

    print(
        "Mean local field error:",
        np.nanmean(
            local_errors[critical_mask]
        )
    )

else:

    print(
        "No samples entered the critical region."
    )


# ============================================================
# WORST LOCAL DISTANCE
# ============================================================

worst_index = np.argmin(
    distances
)

worst_record = records[
    worst_index
]

print()
print("WORST SAFETY EVENT")
print("-" * 60)

print(
    "Step:",
    worst_record["step"]
)

print(
    "Agent:",
    worst_record["agent"]
)

print(
    "Inter-agent distance:",
    worst_record["distance"]
)

print(
    "Spectral complexity:",
    worst_record["complexity"]
)

print(
    "Bandwidth:",
    worst_record["bandwidth"]
)

print(
    "Retained spectral power:",
    worst_record["retained_power"]
)

print(
    "Gradient direction cosine:",
    worst_record["direction_cosine"]
)

print(
    "Gradient magnitude ratio:",
    worst_record["gradient_ratio"]
)

print(
    "Local field error:",
    worst_record["local_field_error"]
)


# ============================================================
# INTERPRETATION
# ============================================================

print()
print("INTERPRETATION")
print("-" * 60)

if (
    worst_record["direction_cosine"] < 0.98
):

    print(
        "SIGNIFICANT LOCAL DIRECTION DISTORTION"
    )

    print(
        "The adaptive filter changes the local "
        "navigation gradient direction."
    )

elif (
    worst_record["gradient_ratio"] < 0.95
):

    print(
        "LOCAL GRADIENT ATTENUATION"
    )

    print(
        "The adaptive filter weakens the local "
        "navigation gradient."
    )

else:

    print(
        "NO STRONG LOCAL GRADIENT DISTORTION"
    )

    print(
        "The safety degradation may depend on "
        "a subtler interaction between the filtered "
        "field and the multi-agent controller."
    )

print()
print("=" * 100)