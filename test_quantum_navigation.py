import numpy as np

from environment.field import generate_field
from navigation.quantum_controller import (
    quantum_reconstruct_field,
    compute_quantum_velocity
)


# -------------------------------------------------
# Same environment used in Step 2
# -------------------------------------------------

agents = [
    (2.0, 2.0),
    (3.0, 13.0)
]

goals = [
    (13.0, 13.0),
    (13.0, 2.0)
]

obstacles = [
    (8.0, 9.0),
    (5.0, 11.0)
]

current_agent = 0


# -------------------------------------------------
# Generate dynamic field for Robot 0
# -------------------------------------------------

X, Y, field = generate_field(
    agents,
    goals,
    obstacles,
    current_agent
)


print("\n========== STEP 8: QUANTUM NAVIGATION ==========")

print("Original field shape:", field.shape)
print("Original field norm:", np.linalg.norm(field))


# -------------------------------------------------
# Quantum spectral pipeline
# -------------------------------------------------

result = quantum_reconstruct_field(field)

reconstructed_field = result["reconstructed_field"]


print("\n---------- Quantum Processing ----------")

print(
    "Spectral complexity:",
    result["complexity"]
)

print(
    "Adaptive bandwidth:",
    result["bandwidth"]
)

print(
    "Retained fraction:",
    result["retained_fraction"]
)

print(
    "Original normalized norm:",
    np.linalg.norm(result["amplitudes"])
)

print(
    "Reconstructed state norm:",
    np.linalg.norm(result["reconstructed_state"])
)

print(
    "Reconstructed field shape:",
    reconstructed_field.shape
)

print(
    "Reconstructed field minimum:",
    np.min(reconstructed_field)
)

print(
    "Reconstructed field maximum:",
    np.max(reconstructed_field)
)


# -------------------------------------------------
# Compute navigation velocity
# -------------------------------------------------

agent_position = np.asarray(
    agents[current_agent],
    dtype=float
)

goal = np.asarray(
    goals[current_agent],
    dtype=float
)


velocity = compute_quantum_velocity(
    reconstructed_field,
    agent_position,
    goal
)


print("\n---------- Navigation ----------")

print(
    "Robot position:",
    agent_position
)

print(
    "Goal:",
    goal
)

print(
    "Quantum velocity:",
    velocity
)

print(
    "Velocity magnitude:",
    np.linalg.norm(velocity)
)