import numpy as np
import matplotlib.pyplot as plt

from environment.field import generate_field


# ============================================================
# ENVIRONMENT
# ============================================================

agents = [
    (2, 2),
    (13, 13)
]

goals = [
    (13, 13),
    (2, 2)
]

obstacles = [
    (8, 8),
    (5, 11)
]


# ============================================================
# GENERATE FIELD FOR AGENT 0
# ============================================================

X, Y, field = generate_field(
    agents=agents,
    goals=goals,
    obstacles=obstacles,
    current_agent=0
)


# ============================================================
# PRINT BASIC INFORMATION
# ============================================================

print("Grid shape:", field.shape)

print("Minimum field value:", np.min(field))

print("Maximum field value:", np.max(field))

print("Mean field value:", np.mean(field))


# ============================================================
# VISUALIZE
# ============================================================

plt.figure(figsize=(8, 7))

plt.imshow(
    field,
    origin="lower",
    extent=[0, 15, 0, 15],
    cmap="viridis"
)

# Obstacles
for obstacle in obstacles:

    plt.scatter(
        obstacle[0],
        obstacle[1],
        marker="s",
        s=180,
        c="red"
    )


# Agents
for i, agent in enumerate(agents):

    plt.scatter(
        agent[0],
        agent[1],
        marker="o",
        s=120,
        edgecolors="white",
        linewidths=2
    )

    plt.text(
        agent[0] + 0.3,
        agent[1] + 0.3,
        f"A{i}",
        color="white",
        fontsize=10
    )


# Goals
for i, goal in enumerate(goals):

    plt.scatter(
        goal[0],
        goal[1],
        marker="*",
        s=220,
        c="yellow"
    )

    plt.text(
        goal[0] + 0.3,
        goal[1] + 0.3,
        f"G{i}",
        color="white",
        fontsize=10
    )


plt.colorbar(label="Potential Field F(x,y)")

plt.title(
    "AQSF-NAV — Classical Interaction Potential Field"
)

plt.xlabel("X")
plt.ylabel("Y")

plt.tight_layout()

plt.show()