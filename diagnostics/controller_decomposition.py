import numpy as np

from navigation.controller import (
    bilinear_interpolate,
    STEP_SIZE,
    MAX_SPEED,
    SAFE_DISTANCE,
    AVOIDANCE_DISTANCE,
)


def decompose_velocity(
    field,
    position,
    goal,
    other_agents=None
):
    # --------------------------------------------------
    # 1. Raw APF gradient
    # --------------------------------------------------
    grad_y, grad_x = np.gradient(field)

    px, py = position

    gx = bilinear_interpolate(grad_x, px, py)
    gy = bilinear_interpolate(grad_y, px, py)

    gradient = np.array([gx, gy], dtype=float)

    # Controller uses negative gradient
    raw_apf = -gradient

    gradient_magnitude = np.linalg.norm(raw_apf)

    # --------------------------------------------------
    # 2. Normalize APF direction
    # --------------------------------------------------
    if gradient_magnitude > 1e-10:
        normalized_apf = (
            raw_apf / gradient_magnitude
        )
    else:
        normalized_apf = np.zeros(2)

    # --------------------------------------------------
    # 3. Goal direction
    # --------------------------------------------------
    goal_direction = (
        np.array(goal, dtype=float)
        - np.array(position, dtype=float)
    )

    goal_distance = np.linalg.norm(goal_direction)

    if goal_distance > 1e-10:
        goal_direction = (
            goal_direction / goal_distance
        )
    else:
        goal_direction = np.zeros(2)

    # --------------------------------------------------
    # 4. APF + goal combination
    # --------------------------------------------------
    base_velocity = (
        0.65 * normalized_apf
        + 0.35 * goal_direction
    )

    # --------------------------------------------------
    # 5. Agent avoidance
    # --------------------------------------------------
    avoidance_vector = np.zeros(2)

    # This records exactly which avoidance regime
    # was active.
    avoidance_mode = "none"

    agent_distances = []

    if other_agents is not None:

        for other in other_agents:

            relative_position = (
                np.array(position, dtype=float)
                - np.array(other, dtype=float)
            )

            distance = np.linalg.norm(
                relative_position
            )

            agent_distances.append(distance)

            if (
                distance > 1e-10
                and distance < AVOIDANCE_DISTANCE
            ):

                away_direction = (
                    relative_position / distance
                )

                if distance > SAFE_DISTANCE:

                    avoidance_mode = (
                        "early_avoidance"
                    )

                    avoidance_strength = (
                        (AVOIDANCE_DISTANCE - distance)
                        /
                        (
                            AVOIDANCE_DISTANCE
                            - SAFE_DISTANCE
                        )
                    )

                    avoidance_strength = np.clip(
                        avoidance_strength,
                        0.0,
                        1.0
                    )

                    avoidance_vector += (
                        avoidance_strength
                        * away_direction
                    )

                else:

                    avoidance_mode = (
                        "safety_projection"
                    )

                    radial_velocity = np.dot(
                        base_velocity
                        + avoidance_vector,
                        away_direction
                    )

                    if radial_velocity < 0:

                        avoidance_vector -= (
                            radial_velocity
                            * away_direction
                        )

                    avoidance_vector += (
                        (SAFE_DISTANCE - distance)
                        * away_direction
                    )

    # --------------------------------------------------
    # 6. Velocity before final normalization
    # --------------------------------------------------
    pre_normalization_velocity = (
        base_velocity
        + avoidance_vector
    )

    # --------------------------------------------------
    # 7. Final controller normalization
    # --------------------------------------------------
    magnitude = np.linalg.norm(
        pre_normalization_velocity
    )

    if magnitude > 1e-10:

        final_velocity = (
            pre_normalization_velocity
            / magnitude
        ) * MAX_SPEED

    else:

        final_velocity = np.zeros(2)

    return {
        "position": np.array(position),
        "raw_apf": raw_apf,
        "normalized_apf": normalized_apf,
        "goal_direction": goal_direction,
        "base_velocity": base_velocity,
        "avoidance_vector": avoidance_vector,
        "pre_normalization_velocity":
            pre_normalization_velocity,
        "final_velocity": final_velocity,
        "gradient_magnitude":
            gradient_magnitude,
        "goal_distance":
            goal_distance,
        "agent_distances":
            agent_distances,
        "avoidance_mode":
            avoidance_mode,
    }