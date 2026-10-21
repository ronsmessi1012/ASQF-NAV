import numpy as np

from environment.field import generate_field
from navigation.controller import navigation_step, reached_goal
from navigation.quantum_controller import quantum_reconstruct_field


# ============================================================
# ENVIRONMENT
# ============================================================

agents = [
    np.array([2.0, 2.0]),
    np.array([3.0, 13.0])
]

goals = [
    np.array([13.0, 13.0]),
    np.array([13.0, 2.0])
]

obstacles = [
    (8.0, 9.0),
    (5.0, 11.0)
]


# ============================================================
# SIMULATION PARAMETERS
# ============================================================

MAX_STEPS = 120

trajectories = [
    [agents[0].copy()],
    [agents[1].copy()]
]

minimum_inter_agent_distance = float("inf")

complexities = []
bandwidths = []
reconstruction_errors = []


# ============================================================
# QUANTUM-ASSISTED MULTI-AGENT NAVIGATION
# ============================================================

for step in range(MAX_STEPS):

    velocities = []
    new_positions = []

    # --------------------------------------------------------
    # Generate quantum-reconstructed field for each robot
    # --------------------------------------------------------

    for i in range(2):

        X, Y, field = generate_field(
            agents,
            goals,
            obstacles,
            i
        )

        # ----------------------------------------------------
        # Quantum spectral processing
        # ----------------------------------------------------

        quantum_result = quantum_reconstruct_field(field)

        reconstructed_field = quantum_result[
            "reconstructed_field"
        ]

        # ----------------------------------------------------
        # Store quantum metrics
        # ----------------------------------------------------

        complexities.append(
            quantum_result["complexity"]
        )

        bandwidths.append(
            quantum_result["bandwidth"]
        )

        # ----------------------------------------------------
        # Reconstruction error
        #
        # The quantum reconstructed field is normalized,
        # therefore compare it against the normalized
        # classical field.
        # ----------------------------------------------------

        normalized_field = (
            field / np.linalg.norm(field)
        )

        error = np.linalg.norm(
            normalized_field -
            reconstructed_field
        )

        reconstruction_errors.append(error)

        # ----------------------------------------------------
        # EXISTING CLASSICAL SAFETY CONTROLLER
        #
        # IMPORTANT:
        # navigation_step() expects:
        #
        # navigation_step(
        #     position,
        #     goal,
        #     field,
        #     other_agents
        # )
        #
        # We only replace the field with the
        # quantum-reconstructed field.
        # ----------------------------------------------------

        new_position, velocity = navigation_step(
            agents[i],
            goals[i],
            reconstructed_field,
            other_agents=[
                agents[j]
                for j in range(2)
                if j != i
            ]
        )

        new_positions.append(new_position)
        velocities.append(velocity)

    # --------------------------------------------------------
    # Move both robots simultaneously
    # --------------------------------------------------------

    for i in range(2):

        agents[i] = new_positions[i]

        # Keep robots inside the 16 x 16 environment
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

        trajectories[i].append(
            agents[i].copy()
        )

    # --------------------------------------------------------
    # Inter-agent distance
    # --------------------------------------------------------

    distance = np.linalg.norm(
        agents[0] - agents[1]
    )

    minimum_inter_agent_distance = min(
        minimum_inter_agent_distance,
        distance
    )

    # --------------------------------------------------------
    # Progress output
    # --------------------------------------------------------

    if step % 10 == 0:

        print(
            f"Step {step:3d} | "
            f"Robot 0 = {agents[0]} | "
            f"Robot 1 = {agents[1]} | "
            f"Distance = {distance:.3f}"
        )

    # --------------------------------------------------------
    # Stop when both robots reach their goals
    # --------------------------------------------------------

    if (
        reached_goal(agents[0], goals[0])
        and
        reached_goal(agents[1], goals[1])
    ):

        print(
            f"\nBoth robots reached their goals "
            f"in {step + 1} steps."
        )

        break


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n========== STEP 8B RESULTS ==========")

for i in range(2):

    final_distance = np.linalg.norm(
        agents[i] - goals[i]
    )

    print(
        f"Robot {i}: "
        f"Final = {agents[i]}, "
        f"Goal = {goals[i]}, "
        f"Distance = {final_distance:.3f}"
    )

print(
    "\nMinimum inter-agent distance:",
    f"{minimum_inter_agent_distance:.3f}"
)

print(
    "Average spectral complexity:",
    f"{np.mean(complexities):.6f}"
)

print(
    "Average adaptive bandwidth:",
    f"{np.mean(bandwidths):.6f}"
)

print(
    "Average reconstruction error:",
    f"{np.mean(reconstruction_errors):.6f}"
)