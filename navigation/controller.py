import numpy as np

# ============================================================
# NAVIGATION PARAMETERS
# ============================================================

STEP_SIZE = 0.20
GOAL_TOLERANCE = 0.45
MAX_SPEED = 1.0


# ============================================================
# BILINEAR INTERPOLATION
# ============================================================

def bilinear_interpolate(grid, x, y):
    """
    Bilinearly interpolate a 2D grid at continuous
    coordinates (x, y).
    """

    height, width = grid.shape

    x = np.clip(x, 0, width - 1)
    y = np.clip(y, 0, height - 1)

    x0 = int(np.floor(x))
    x1 = min(x0 + 1, width - 1)

    y0 = int(np.floor(y))
    y1 = min(y0 + 1, height - 1)

    dx = x - x0
    dy = y - y0

    value = (
        grid[y0, x0] * (1 - dx) * (1 - dy)
        + grid[y0, x1] * dx * (1 - dy)
        + grid[y1, x0] * (1 - dx) * dy
        + grid[y1, x1] * dx * dy
    )

    return value


# ============================================================
# CONTINUOUS FIELD GRADIENT
# ============================================================

def sample_field_gradient(
    field,
    position
):
    """
    Calculate the numerical gradient of the field
    and continuously interpolate it at the robot position.
    """

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


# ============================================================
# ROBOT VELOCITY
# ============================================================

def compute_velocity(
    field,
    position,
    goal
):
    """
    Compute velocity using the negative potential gradient.

    A tangential component is introduced when the robot
    encounters a local minimum away from the goal.
    """

    grad_x, grad_y = sample_field_gradient(
        field,
        position
    )

    gradient = np.array(
        [grad_x, grad_y],
        dtype=float
    )

    # --------------------------------------------------------
    # Negative gradient
    # --------------------------------------------------------

    velocity = -gradient

    gradient_magnitude = np.linalg.norm(
        gradient
    )

    # --------------------------------------------------------
    # Goal direction
    # --------------------------------------------------------

    goal_direction = (
        np.array(goal, dtype=float)
        - np.array(position, dtype=float)
    )

    goal_distance = np.linalg.norm(
        goal_direction
    )

    if goal_distance > 1e-10:
        goal_direction /= goal_distance

    # --------------------------------------------------------
    # Local-minimum escape
    # --------------------------------------------------------

    if (
        gradient_magnitude < 0.15
        and goal_distance > GOAL_TOLERANCE
    ):

        # Tangential direction to the local field
        tangent = np.array(
            [-grad_y, grad_x],
            dtype=float
        )

        tangent_magnitude = np.linalg.norm(
            tangent
        )

        if tangent_magnitude > 1e-10:

            tangent /= tangent_magnitude

            # Choose tangent orientation that points
            # generally toward the goal
            if np.dot(
                tangent,
                goal_direction
            ) < 0:

                tangent = -tangent

            velocity = (
                0.45 * goal_direction
                + 0.55 * tangent
            )

        else:

            velocity = goal_direction

    else:

        # Normal APF navigation
        if gradient_magnitude > 1e-10:

            velocity /= gradient_magnitude

        velocity = (
            0.65 * velocity
            + 0.35 * goal_direction
        )

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    magnitude = np.linalg.norm(
        velocity
    )

    if magnitude > 1e-10:

        velocity = (
            velocity / magnitude
        ) * MAX_SPEED

    return velocity


# ============================================================
# NAVIGATION STEP
# ============================================================

def navigation_step(
    position,
    goal,
    field
):
    """
    Perform one continuous navigation step.
    """

    velocity = compute_velocity(
        field,
        position,
        goal
    )

    new_position = (
        np.array(position, dtype=float)
        + STEP_SIZE * velocity
    )

    # Keep robot inside 16x16 environment
    new_position = np.clip(
        new_position,
        0,
        15
    )

    return new_position, velocity


# ============================================================
# GOAL CHECK
# ============================================================

def reached_goal(
    position,
    goal
):
    """
    Check whether the robot has reached its goal.
    """

    distance = np.linalg.norm(
        np.array(position)
        - np.array(goal)
    )

    return distance <= GOAL_TOLERANCE