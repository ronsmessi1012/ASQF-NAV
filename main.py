import numpy as np
import matplotlib.pyplot as plt

from environment.field import generate_field
from navigation.controller import navigation_step, reached_goal


# --------------------------------------------------
# Environment
# --------------------------------------------------

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


# --------------------------------------------------
# Simulation parameters
# --------------------------------------------------

positions = np.array(agents, dtype=float)
goals = np.array(goals, dtype=float)

MAX_STEPS = 250
SAFE_DISTANCE = 1.0

trajectories = [
    [positions[0].copy()],
    [positions[1].copy()]
]

# Track minimum distance between robots
min_inter_agent_distance = float("inf")
min_distance_step = None


# --------------------------------------------------
# Multi-agent simulation
# --------------------------------------------------

for step in range(MAX_STEPS):

    new_positions = positions.copy()

    for i in range(len(positions)):

        # Generate dynamic field for this robot
        # Other robots are treated as dynamic repulsive agents
        X, Y, field = generate_field(
            agents=positions,
            goals=goals,
            obstacles=obstacles,
            current_agent=i
        )

        # Calculate robot motion
        other_agents = [
    positions[j]
    for j in range(len(positions))
    if j != i
]

        new_positions[i], velocity = navigation_step(
            positions[i],
            goals[i],
            field,
            other_agents
        )

    # Update both robots simultaneously
    positions = new_positions

    # Store trajectories
    trajectories[0].append(positions[0].copy())
    trajectories[1].append(positions[1].copy())

    # --------------------------------------------------
    # Inter-agent distance
    # --------------------------------------------------

    inter_agent_distance = np.linalg.norm(
        positions[0] - positions[1]
    )

    # Update minimum distance
    if inter_agent_distance < min_inter_agent_distance:
        min_inter_agent_distance = inter_agent_distance
        min_distance_step = step + 1

    # Collision / safety warning
    if inter_agent_distance < SAFE_DISTANCE:
        print(
            f"WARNING: agents too close at step {step + 1} "
            f"(distance = {inter_agent_distance:.3f})"
        )

    # --------------------------------------------------
    # Goal check
    # --------------------------------------------------

    robot_0_done = reached_goal(
        positions[0],
        goals[0]
    )

    robot_1_done = reached_goal(
        positions[1],
        goals[1]
    )

    if robot_0_done and robot_1_done:
        print(
            f"\nBoth robots reached their goals "
            f"in {step + 1} steps."
        )
        break


# --------------------------------------------------
# Convert trajectories to arrays
# --------------------------------------------------

trajectories = [
    np.array(trajectories[0]),
    np.array(trajectories[1])
]


# --------------------------------------------------
# Final Results
# --------------------------------------------------

print("\nFinal Results")

for i in range(2):

    final_distance = np.linalg.norm(
        positions[i] - goals[i]
    )

    print(
        f"Robot {i}: "
        f"Start={agents[i]}, "
        f"Final=({positions[i][0]:.3f}, "
        f"{positions[i][1]:.3f}), "
        f"Goal={goals[i]}, "
        f"Distance={final_distance:.3f}"
    )

print(
    f"\nMinimum inter-agent distance: "
    f"{min_inter_agent_distance:.3f}"
)

print(
    f"Minimum distance occurred at step: "
    f"{min_distance_step}"
)


# --------------------------------------------------
# Plot trajectories
# --------------------------------------------------

plt.figure(figsize=(8, 8))

plt.imshow(
    field,
    origin="lower",
    cmap="viridis"
)


# Robot 0 trajectory
plt.plot(
    trajectories[0][:, 0],
    trajectories[0][:, 1],
    linewidth=2,
    label="Robot 0"
)


# Robot 1 trajectory
plt.plot(
    trajectories[1][:, 0],
    trajectories[1][:, 1],
    linewidth=2,
    label="Robot 1"
)


# Starting positions
plt.scatter(
    agents[0][0],
    agents[0][1],
    marker="o",
    s=100,
    label="Robot 0 Start"
)

plt.scatter(
    agents[1][0],
    agents[1][1],
    marker="o",
    s=100,
    label="Robot 1 Start"
)


# Goals
plt.scatter(
    goals[0][0],
    goals[0][1],
    marker="*",
    s=180,
    label="Robot 0 Goal"
)

plt.scatter(
    goals[1][0],
    goals[1][1],
    marker="*",
    s=180,
    label="Robot 1 Goal"
)


# Obstacles
for ox, oy in obstacles:

    plt.scatter(
        ox,
        oy,
        marker="s",
        s=180,
        color="red"
    )


# Plot formatting
plt.title(
    "Step 2B — Multi-Agent Dynamic Navigation"
)

plt.xlabel("X")
plt.ylabel("Y")

plt.legend()
plt.grid(alpha=0.3)

plt.show()