import numpy as np

from environment.field import generate_field
from navigation.controller import navigation_step, reached_goal
from navigation.quantum_controller import quantum_reconstruct_field


# ============================================================
# COMMON ENVIRONMENT
# ============================================================

INITIAL_AGENTS = [
    np.array([2.0, 2.0]),
    np.array([3.0, 13.0])
]

GOALS = [
    np.array([13.0, 13.0]),
    np.array([13.0, 2.0])
]

OBSTACLES = [
    (8.0, 9.0),
    (5.0, 11.0)
]

MAX_STEPS = 120


# ============================================================
# CLASSICAL NAVIGATION
# ============================================================

def run_classical():

    agents = [
        INITIAL_AGENTS[0].copy(),
        INITIAL_AGENTS[1].copy()
    ]

    trajectories = [
        [agents[0].copy()],
        [agents[1].copy()]
    ]

    minimum_distance = float("inf")
    path_lengths = [0.0, 0.0]

    for step in range(MAX_STEPS):

        new_positions = []

        for i in range(2):

            X, Y, field = generate_field(
                agents,
                GOALS,
                OBSTACLES,
                i
            )

            new_position, velocity = navigation_step(
                agents[i],
                GOALS[i],
                field,
                other_agents=[
                    agents[j]
                    for j in range(2)
                    if j != i
                ]
            )

            new_positions.append(new_position)

        for i in range(2):

            previous_position = agents[i].copy()

            agents[i] = new_positions[i]

            agents[i][0] = np.clip(
                agents[i][0],
                0.0,
                15.0
            )

            agents[i][1] = np.clip(
                agents[i][1],
                0.0,
                15.0
            )

            path_lengths[i] += np.linalg.norm(
                agents[i] - previous_position
            )

            trajectories[i].append(
                agents[i].copy()
            )

        distance = np.linalg.norm(
            agents[0] - agents[1]
        )

        minimum_distance = min(
            minimum_distance,
            distance
        )

        if (
            reached_goal(agents[0], GOALS[0])
            and
            reached_goal(agents[1], GOALS[1])
        ):
            break

    final_errors = [
        np.linalg.norm(
            agents[i] - GOALS[i]
        )
        for i in range(2)
    ]

    success = all(
        error <= 0.45
        for error in final_errors
    )

    return {
        "success": success,
        "steps": step + 1,
        "agents": agents,
        "trajectories": trajectories,
        "final_errors": final_errors,
        "minimum_distance": minimum_distance,
        "path_lengths": path_lengths
    }


# ============================================================
# QUANTUM-ADAPTIVE NAVIGATION
# ============================================================

def run_quantum():

    agents = [
        INITIAL_AGENTS[0].copy(),
        INITIAL_AGENTS[1].copy()
    ]

    trajectories = [
        [agents[0].copy()],
        [agents[1].copy()]
    ]

    minimum_distance = float("inf")
    path_lengths = [0.0, 0.0]

    complexities = []
    bandwidths = []
    reconstruction_errors = []

    for step in range(MAX_STEPS):

        new_positions = []

        for i in range(2):

            X, Y, field = generate_field(
                agents,
                GOALS,
                OBSTACLES,
                i
            )

            quantum_result = quantum_reconstruct_field(
                field
            )

            reconstructed_field = (
                quantum_result["reconstructed_field"]
            )

            complexities.append(
                quantum_result["complexity"]
            )

            bandwidths.append(
                quantum_result["bandwidth"]
            )

            normalized_field = (
                field /
                np.linalg.norm(field)
            )

            error = np.linalg.norm(
                normalized_field -
                reconstructed_field
            )

            reconstruction_errors.append(
                error
            )

            new_position, velocity = navigation_step(
                agents[i],
                GOALS[i],
                reconstructed_field,
                other_agents=[
                    agents[j]
                    for j in range(2)
                    if j != i
                ]
            )

            new_positions.append(
                new_position
            )

        for i in range(2):

            previous_position = agents[i].copy()

            agents[i] = new_positions[i]

            agents[i][0] = np.clip(
                agents[i][0],
                0.0,
                15.0
            )

            agents[i][1] = np.clip(
                agents[i][1],
                0.0,
                15.0
            )

            path_lengths[i] += np.linalg.norm(
                agents[i] - previous_position
            )

            trajectories[i].append(
                agents[i].copy()
            )

        distance = np.linalg.norm(
            agents[0] - agents[1]
        )

        minimum_distance = min(
            minimum_distance,
            distance
        )

        if (
            reached_goal(agents[0], GOALS[0])
            and
            reached_goal(agents[1], GOALS[1])
        ):
            break

    final_errors = [
        np.linalg.norm(
            agents[i] - GOALS[i]
        )
        for i in range(2)
    ]

    success = all(
        error <= 0.45
        for error in final_errors
    )

    return {
        "success": success,
        "steps": step + 1,
        "agents": agents,
        "trajectories": trajectories,
        "final_errors": final_errors,
        "minimum_distance": minimum_distance,
        "path_lengths": path_lengths,
        "average_complexity": np.mean(
            complexities
        ),
        "average_bandwidth": np.mean(
            bandwidths
        ),
        "average_reconstruction_error": np.mean(
            reconstruction_errors
        )
    }


# ============================================================
# RUN EXPERIMENT
# ============================================================

print("\n========================================")
print(" CLASSICAL VS QUANTUM NAVIGATION")
print("========================================")

print("\nRunning classical controller...")

classical = run_classical()

print("Classical simulation complete.")

print("\nRunning quantum-adaptive controller...")

quantum = run_quantum()

print("Quantum simulation complete.")


# ============================================================
# CLASSICAL RESULTS
# ============================================================

print("\n========================================")
print(" CLASSICAL RESULTS")
print("========================================")

print(
    "Success:",
    classical["success"]
)

print(
    "Steps:",
    classical["steps"]
)

print(
    "Robot 0 final error:",
    f"{classical['final_errors'][0]:.4f}"
)

print(
    "Robot 1 final error:",
    f"{classical['final_errors'][1]:.4f}"
)

print(
    "Minimum inter-agent distance:",
    f"{classical['minimum_distance']:.4f}"
)

print(
    "Robot 0 path length:",
    f"{classical['path_lengths'][0]:.4f}"
)

print(
    "Robot 1 path length:",
    f"{classical['path_lengths'][1]:.4f}"
)


# ============================================================
# QUANTUM RESULTS
# ============================================================

print("\n========================================")
print(" QUANTUM-ADAPTIVE RESULTS")
print("========================================")

print(
    "Success:",
    quantum["success"]
)

print(
    "Steps:",
    quantum["steps"]
)

print(
    "Robot 0 final error:",
    f"{quantum['final_errors'][0]:.4f}"
)

print(
    "Robot 1 final error:",
    f"{quantum['final_errors'][1]:.4f}"
)

print(
    "Minimum inter-agent distance:",
    f"{quantum['minimum_distance']:.4f}"
)

print(
    "Robot 0 path length:",
    f"{quantum['path_lengths'][0]:.4f}"
)

print(
    "Robot 1 path length:",
    f"{quantum['path_lengths'][1]:.4f}"
)

print(
    "Average spectral complexity:",
    f"{quantum['average_complexity']:.6f}"
)

print(
    "Average adaptive bandwidth:",
    f"{quantum['average_bandwidth']:.6f}"
)

print(
    "Average reconstruction error:",
    f"{quantum['average_reconstruction_error']:.6f}"
)


# ============================================================
# COMPARISON
# ============================================================

print("\n========================================")
print(" COMPARISON")
print("========================================")

print(
    "Step difference:",
    quantum["steps"] -
    classical["steps"]
)

print(
    "Minimum-distance difference:",
    f"{quantum['minimum_distance'] - classical['minimum_distance']:.4f}"
)

print(
    "Total path length - Classical:",
    f"{sum(classical['path_lengths']):.4f}"
)

print(
    "Total path length - Quantum:",
    f"{sum(quantum['path_lengths']):.4f}"
)

print("\nExperiment complete.")