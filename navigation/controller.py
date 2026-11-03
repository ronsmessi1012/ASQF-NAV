import numpy as np


STEP_SIZE = 0.20
GOAL_TOLERANCE = 0.45
MAX_SPEED = 1.0

# Safety parameters
SAFE_DISTANCE = 1.05
AVOIDANCE_DISTANCE = 1.5


def bilinear_interpolate(grid, x, y):

    height, width = grid.shape

    x = np.clip(x, 0, width - 1)
    y = np.clip(y, 0, height - 1)

    x0 = int(np.floor(x))
    x1 = min(x0 + 1, width - 1)

    y0 = int(np.floor(y))
    y1 = min(y0 + 1, height - 1)

    dx = x - x0
    dy = y - y0

    return (
        grid[y0, x0] * (1 - dx) * (1 - dy)
        + grid[y0, x1] * dx * (1 - dy)
        + grid[y1, x0] * (1 - dx) * dy
        + grid[y1, x1] * dx * dy
    )


def sample_field_gradient(field, position):

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

    return gx, gy


def compute_velocity(
    field,
    position,
    goal,
    other_agents=None
):

    # --------------------------------------------------
    # 1. APF gradient
    # --------------------------------------------------

    grad_x, grad_y = sample_field_gradient(
        field,
        position
    )

    gradient = np.array(
        [grad_x, grad_y],
        dtype=float
    )

    velocity = -gradient

    gradient_magnitude = np.linalg.norm(
        velocity
    )

    # --------------------------------------------------
    # 2. Goal direction
    # --------------------------------------------------

    goal_direction = (
        np.array(goal, dtype=float)
        - np.array(position, dtype=float)
    )

    goal_distance = np.linalg.norm(
        goal_direction
    )

    if goal_distance > 1e-10:
        goal_direction /= goal_distance

    # --------------------------------------------------
    # 3. Combine APF + goal direction
    # --------------------------------------------------

    if gradient_magnitude > 1e-10:
        velocity /= gradient_magnitude

    velocity = (
        0.65 * velocity
        + 0.35 * goal_direction
    )

    # --------------------------------------------------
    # 4. Early collision avoidance
    # --------------------------------------------------

    if other_agents is not None:

        for other in other_agents:

            relative_position = (
                np.array(position, dtype=float)
                - np.array(other, dtype=float)
            )

            distance = np.linalg.norm(
                relative_position
            )

            if (
                distance > 1e-10
                and distance < AVOIDANCE_DISTANCE
            ):

                # Direction pointing away
                # from the other robot
                away_direction = (
                    relative_position / distance
                )

                # --------------------------------------------------
                # Early avoidance steering
                # --------------------------------------------------

                if distance > SAFE_DISTANCE:

                    # Smoothly increase avoidance strength
                    # as robots approach SAFE_DISTANCE.
                    avoidance_strength = (
                        (AVOIDANCE_DISTANCE - distance)
                        / (AVOIDANCE_DISTANCE - SAFE_DISTANCE)
                    )

                    avoidance_strength = np.clip(
                        avoidance_strength,
                        0.0,
                        1.0
                    )

                    velocity += (
                        avoidance_strength
                        * away_direction
                    )

                # --------------------------------------------------
                # Hard safety projection
                # --------------------------------------------------

                else:

                    radial_velocity = np.dot(
                        velocity,
                        away_direction
                    )

                    # Remove velocity component that
                    # moves toward the other robot.
                    if radial_velocity < 0:

                        velocity -= (
                            radial_velocity
                            * away_direction
                        )

                    # Add a small outward component
                    # while inside the safety radius.
                    velocity += (
                        (SAFE_DISTANCE - distance)
                        * away_direction
                    )

    # --------------------------------------------------
    # 5. Normalize velocity
    # --------------------------------------------------

    magnitude = np.linalg.norm(
        velocity
    )

    if magnitude > 1e-10:

        velocity = (
            velocity / magnitude
        ) * MAX_SPEED

    return velocity


def navigation_step(
    position,
    goal,
    field,
    other_agents=None
):

    velocity = compute_velocity(
        field,
        position,
        goal,
        other_agents
    )

    new_position = (
        np.array(position, dtype=float)
        + STEP_SIZE * velocity
    )

    return new_position, velocity


def reached_goal(position, goal):

    distance = np.linalg.norm(
        np.array(position)
        - np.array(goal)
    )

    return distance <= GOAL_TOLERANCE