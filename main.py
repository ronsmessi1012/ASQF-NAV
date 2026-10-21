import numpy as np
import matplotlib.pyplot as plt

from environment.field import generate_field
from navigation.controller import (
    navigation_step,
    reached_goal
)


# ============================================================
# ENVIRONMENT
# ============================================================

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


# ============================================================
# ROBOT 0
# ============================================================

position = np.array(
    agents[0],
    dtype=float
)

goal = np.array(
    goals[0],
    dtype=float
)


trajectory = [
    position.copy()
]


# ============================================================
# NAVIGATION LOOP
# ============================================================

MAX_STEPS = 200

for step in range(MAX_STEPS):

    # Generate current interaction field
    X, Y, field = generate_field(
        agents=[
            position,
            agents[1]
        ],
        goals=goals,
        obstacles=obstacles,
        current_agent=0
    )

    # Move robot
    position, velocity = navigation_step(
        position,
        goal,
        field
    )

    trajectory.append(
        position.copy()
    )

    # Check goal
    if reached_goal(
        position,
        goal
    ):

        print(
            f"Goal reached in {step + 1} steps."
        )

        break


trajectory = np.array(
    trajectory
)

print("\nTrajectory:")
for i, point in enumerate(trajectory[::10]):
    print(
        f"Step {i * 10:3d}: "
        f"({point[0]:.3f}, {point[1]:.3f})"
    )


# ============================================================
# RESULTS
# ============================================================

print(
    "Starting position:",
    agents[0]
)

print(
    "Final position:",
    position
)

print(
    "Goal:",
    goal
)

print(
    "Final distance:",
    np.linalg.norm(
        position - goal
    )
)


# ============================================================
# VISUALIZATION
# ============================================================

plt.figure(figsize=(8, 7))

plt.imshow(
    field,
    origin="lower",
    extent=[0, 15, 0, 15],
    cmap="viridis"
)


# ------------------------------------------------------------
# Obstacles
# ------------------------------------------------------------

for obstacle in obstacles:

    plt.scatter(
        obstacle[0],
        obstacle[1],
        marker="s",
        s=220,
        c="red"
    )


# ------------------------------------------------------------
# Robot trajectory
# ------------------------------------------------------------

plt.plot(
    trajectory[:, 0],
    trajectory[:, 1],
    linewidth=2,
    c="white",
    label="Robot trajectory"
)


# Starting point
plt.scatter(
    agents[0][0],
    agents[0][1],
    s=140,
    marker="o",
    c="cyan",
    label="Start"
)


# Goal
plt.scatter(
    goal[0],
    goal[1],
    s=250,
    marker="*",
    c="yellow",
    label="Goal"
)


plt.colorbar(
    label="Potential Field F(x,y)"
)

plt.xlabel("X")
plt.ylabel("Y")

plt.title(
    "AQSF-NAV — Gradient-Based Navigation"
)

plt.legend()

plt.tight_layout()

plt.show()