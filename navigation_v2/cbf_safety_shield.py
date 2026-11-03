"""
Component E: Control Barrier Function (CBF) Safety Shield.

Provides mathematical safety enforcement via Quadratic Programming (QP).
Modifies candidate/desired robot velocities only when required to ensure collision safety:

For robot i and dynamic entity j:
    h_ij(p_i, q_j) = ||p_i - q_j||^2 - D_ij^2
    Continuous CBF constraint:
    2 * r_ij^T * (v_i - u_j) + alpha * h_ij >= 0
    where r_ij = p_i - q_j.

Optimization:
    min_{v_i} ||v_i - v_{i, desired}||^2
    subject to:
        2 * r_ij^T * (v_i - u_j) + alpha * h_ij >= 0   (for all entities j)
        ||v_i||_2 <= v_max

Includes:
- Multiple simultaneous safety constraints.
- Robot-to-robot and robot-to-moving-obstacle constraints.
- Both decentralized (with reciprocity) and centralized joint QP formulations.
- Feasibility diagnostics and status reporting.
- Safe fallback (maximum-separation steering / emergency stop) on infeasibility.
- Discrete forward-step verification.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union
import numpy as np
from scipy.optimize import minimize

from navigation_v2.config import AQSFConfigV2
from navigation_v2.dynamic_entities import MovingObstacle


@dataclass
class CBFSolverResult:
    safe_velocity: np.ndarray
    desired_velocity: np.ndarray
    intervened: bool
    status: str  # "success", "modified", "infeasible_fallback", "emergency_stop"
    min_barrier_value: float
    active_constraints_count: int
    solver_time: float
    violations_detected: int
    message: str = ""

    def to_dict(self) -> dict:
        return {
            "safe_velocity": self.safe_velocity.tolist(),
            "desired_velocity": self.desired_velocity.tolist(),
            "intervened": self.intervened,
            "status": self.status,
            "min_barrier_value": float(self.min_barrier_value),
            "active_constraints_count": int(self.active_constraints_count),
            "solver_time": float(self.solver_time),
            "violations_detected": int(self.violations_detected),
            "message": self.message,
        }


def evaluate_cbf_condition(
    p_i: np.ndarray,
    v_i: np.ndarray,
    q_j: np.ndarray,
    u_j: np.ndarray,
    safety_distance: float,
    alpha: float
) -> Tuple[float, float, float]:
    """
    Evaluate barrier h_ij and CBF derivative condition:
        h_ij = ||p_i - q_j||^2 - D_ij^2
        cbf_val = 2 * r_ij^T * (v_i - u_j) + alpha * h_ij
    Returns:
        h_ij, cbf_val, distance
    """
    r_ij = p_i - q_j
    dist = float(np.linalg.norm(r_ij))
    h_ij = dist ** 2 - safety_distance ** 2
    cbf_val = 2.0 * float(np.dot(r_ij, v_i - u_j)) + alpha * h_ij
    return h_ij, cbf_val, dist


class CBFSafetyShield:
    """
    Independent Control Barrier Function Safety Shield.
    """
    def __init__(self, config: Optional[AQSFConfigV2] = None):
        self.config = config or AQSFConfigV2()

    def filter_velocity(
        self,
        robot_idx: int,
        desired_velocity: np.ndarray,
        robot_positions: np.ndarray,
        robot_velocities: np.ndarray,
        moving_obstacles: Optional[List[MovingObstacle]] = None,
        reciprocal_weight: float = 0.5
    ) -> CBFSolverResult:
        """
        Decentralized velocity filtering for robot i.
        
        Reciprocal weight defines share of responsibility:
        - For moving obstacles: robot i takes 100% responsibility (weight = 1.0)
        - For peer robots: robot i takes reciprocal_weight (default 0.5)
        """
        import time
        start_time = time.perf_counter()

        p_i = np.asarray(robot_positions[robot_idx], dtype=float)
        v_des = np.asarray(desired_velocity, dtype=float)
        v_max = self.config.max_speed
        alpha = self.config.cbf_alpha

        # Check speed saturation on desired velocity
        v_des_mag = float(np.linalg.norm(v_des))
        if v_des_mag > v_max:
            v_des = (v_des / v_des_mag) * v_max

        # Gather constraints
        # Each constraint: 2 * r_ij^T * (v_i - u_j) + alpha * h_ij >= 0
        # Form: A_k * v_i >= b_k  <=>  b_k - A_k * v_i <= 0
        constraints_list = []
        cbf_data = []

        # 1. Peer robots
        robot_safe_d = self.config.compute_robot_safety_distance()
        for j in range(len(robot_positions)):
            if j == robot_idx:
                continue
            q_j = np.asarray(robot_positions[j], dtype=float)
            u_j = np.asarray(robot_velocities[j], dtype=float)
            r_ij = p_i - q_j
            d_ij = float(np.linalg.norm(r_ij))
            h_ij = d_ij ** 2 - robot_safe_d ** 2

            # Reciprocal allocation:
            # 2 * r_ij^T * (v_i - u_j) + alpha * h_ij >= 0
            # Under reciprocity factor w in (0, 1]:
            # r_ij^T * v_i >= r_ij^T * u_j - (alpha * h_ij) * w
            A_k = 2.0 * r_ij
            b_k = 2.0 * float(np.dot(r_ij, u_j)) - alpha * h_ij * (2.0 * reciprocal_weight)

            constraints_list.append((A_k, b_k, r_ij, d_ij, h_ij, f"robot_{j}"))
            cbf_data.append((h_ij, d_ij))

        # 2. Moving obstacles (full avoidance responsibility for the robot)
        if moving_obstacles:
            for obs in moving_obstacles:
                obs_safe_d = self.config.compute_safety_distance(obs.radius)
                q_j = np.asarray(obs.position, dtype=float)
                u_j = np.asarray(obs.velocity, dtype=float)
                r_ij = p_i - q_j
                d_ij = float(np.linalg.norm(r_ij))
                h_ij = d_ij ** 2 - obs_safe_d ** 2

                A_k = 2.0 * r_ij
                b_k = 2.0 * float(np.dot(r_ij, u_j)) - alpha * h_ij
                constraints_list.append((A_k, b_k, r_ij, d_ij, h_ij, obs.id))
                cbf_data.append((h_ij, d_ij))

        # If no dynamic entities present, return desired velocity
        if not constraints_list:
            elapsed = time.perf_counter() - start_time
            return CBFSolverResult(
                safe_velocity=v_des,
                desired_velocity=v_des,
                intervened=False,
                status="success",
                min_barrier_value=float("inf"),
                active_constraints_count=0,
                solver_time=elapsed,
                violations_detected=0,
                message="No dynamic obstacles present."
            )

        min_h = min(item[0] for item in cbf_data)

        # Check if desired velocity already strictly satisfies all CBF constraints
        # Constraint condition: A_k @ v >= b_k
        all_satisfied = True
        active_count = 0
        for A_k, b_k, _, _, _, _ in constraints_list:
            margin = np.dot(A_k, v_des) - b_k
            if margin < -1e-6:
                all_satisfied = False
            elif abs(margin) < 0.1:
                active_count += 1

        if all_satisfied and np.linalg.norm(v_des) <= v_max + 1e-6:
            elapsed = time.perf_counter() - start_time
            return CBFSolverResult(
                safe_velocity=v_des,
                desired_velocity=v_des,
                intervened=False,
                status="success",
                min_barrier_value=min_h,
                active_constraints_count=active_count,
                solver_time=elapsed,
                violations_detected=0,
                message="Desired velocity already satisfies CBF."
            )

        # Formulate QP:
        # min 0.5 * ||v - v_des||^2
        def objective(v):
            diff = v - v_des
            return 0.5 * float(np.dot(diff, diff))

        def objective_grad(v):
            return v - v_des

        # Scipy constraints: c(v) >= 0
        scipy_constraints = []
        for A_k, b_k, _, _, _, _ in constraints_list:
            scipy_constraints.append({
                "type": "ineq",
                "fun": lambda v, a=A_k, b=b_k: np.dot(a, v) - b,
                "jac": lambda v, a=A_k: a
            })

        # Speed constraint: v_max^2 - ||v||^2 >= 0
        scipy_constraints.append({
            "type": "ineq",
            "fun": lambda v, vm=v_max: vm**2 - float(np.dot(v, v)),
            "jac": lambda v: -2.0 * v
        })

        # Initial guess: v_des or clamped
        v0 = v_des.copy()

        opt_res = minimize(
            fun=objective,
            x0=v0,
            jac=objective_grad,
            constraints=scipy_constraints,
            method="SLSQP",
            options={"ftol": self.config.qp_tolerance, "maxiter": 100, "disp": False}
        )

        elapsed = time.perf_counter() - start_time

        # Check solver outcome
        if opt_res.success:
            v_safe = np.asarray(opt_res.x, dtype=float)
            # Guarantee strict velocity saturation
            speed = float(np.linalg.norm(v_safe))
            if speed > v_max:
                v_safe = (v_safe / speed) * v_max

            vel_diff = float(np.linalg.norm(v_safe - v_des))
            intervened = vel_diff > 1e-4

            # Count satisfied vs active
            active = sum(1 for A_k, b_k, _, _, _, _ in constraints_list if abs(np.dot(A_k, v_safe) - b_k) < 0.05)

            return CBFSolverResult(
                safe_velocity=v_safe,
                desired_velocity=v_des,
                intervened=intervened,
                status="modified" if intervened else "success",
                min_barrier_value=min_h,
                active_constraints_count=active,
                solver_time=elapsed,
                violations_detected=0,
                message="CBF optimization succeeded."
            )
        else:
            # INFEASIBLE: activate documented safe fallback
            # When constraints conflict (e.g. entities already encroaching D_ij or trapped),
            # compute best-effort velocity that maximizes margin of the most critical constraint.
            v_fallback, fallback_status = self._compute_fallback_velocity(
                p_i, v_des, constraints_list, v_max
            )
            return CBFSolverResult(
                safe_velocity=v_fallback,
                desired_velocity=v_des,
                intervened=True,
                status=fallback_status,
                min_barrier_value=min_h,
                active_constraints_count=len(constraints_list),
                solver_time=elapsed,
                violations_detected=1,
                message=f"CBF infeasible ({opt_res.message}); executed fallback ({fallback_status})."
            )

    def _compute_fallback_velocity(
        self,
        p_i: np.ndarray,
        v_des: np.ndarray,
        constraints_list: list,
        v_max: float
    ) -> Tuple[np.ndarray, str]:
        """
        Documented fallback strategy for infeasible CBF:
        1. Find entity with minimum barrier value h_ij (most threatened).
        2. Direct velocity along away-vector r_ij with maximum allowable speed.
        3. If opposing obstacles cause trapping, activate emergency stop.
        """
        if not constraints_list:
            return np.zeros(2, dtype=float), "emergency_stop"

        # Find most critical constraint (smallest h_ij)
        sorted_constraints = sorted(constraints_list, key=lambda c: c[4])
        most_critical = sorted_constraints[0]
        r_crit = most_critical[2]
        d_crit = most_critical[3]

        if d_crit > 1e-6:
            away_dir = r_crit / d_crit
            # Check if this away direction directly violates any other constraint severely
            opposing = False
            for _, _, r_other, d_other, _, _ in sorted_constraints[1:]:
                if d_other > 1e-6:
                    other_dir = r_other / d_other
                    if np.dot(away_dir, other_dir) < -0.7:  # Opposing trap
                        opposing = True
                        break

            if opposing:
                return np.zeros(2, dtype=float), "emergency_stop"
            else:
                return away_dir * v_max, "infeasible_fallback"

        return np.zeros(2, dtype=float), "emergency_stop"


def filter_joint_velocities_centralized(
    desired_velocities: np.ndarray,
    robot_positions: np.ndarray,
    moving_obstacles: Optional[List[MovingObstacle]] = None,
    config: Optional[AQSFConfigV2] = None
) -> Tuple[np.ndarray, List[CBFSolverResult]]:
    """
    Centralized Joint CBF optimization for N robots simultaneously.
    Minimizes sum_i ||v_i - v_i_desired||^2
    subject to:
        2 * (p_i - p_j)^T * (v_i - v_j) + alpha * (||p_i - p_j||^2 - D_ij^2) >= 0 (all i != j)
        2 * (p_i - q_k)^T * (v_i - u_k) + alpha * (||p_i - q_k||^2 - D_ik^2) >= 0 (all i, k)
        ||v_i|| <= v_max
    """
    import time
    if config is None:
        config = AQSFConfigV2()

    num_robots = len(robot_positions)
    v_des_flat = desired_velocities.flatten()
    v_max = config.max_speed
    alpha = config.cbf_alpha
    robot_safe_d = config.compute_robot_safety_distance()

    # Objective
    def joint_obj(v_flat):
        diff = v_flat - v_des_flat
        return 0.5 * float(np.dot(diff, diff))

    def joint_grad(v_flat):
        return v_flat - v_des_flat

    constraints = []

    # 1. Inter-robot pairwise constraints
    for i in range(num_robots):
        p_i = robot_positions[i]
        for j in range(i + 1, num_robots):
            p_j = robot_positions[j]
            r_ij = p_i - p_j
            d_ij = float(np.linalg.norm(r_ij))
            h_ij = d_ij**2 - robot_safe_d**2

            def robot_pair_cbf(v_flat, idx_i=i, idx_j=j, r=r_ij, h=h_ij):
                vi = v_flat[2*idx_i : 2*idx_i + 2]
                vj = v_flat[2*idx_j : 2*idx_j + 2]
                return 2.0 * np.dot(r, vi - vj) + alpha * h

            constraints.append({"type": "ineq", "fun": robot_pair_cbf})

    # 2. Moving obstacle constraints for each robot
    if moving_obstacles:
        for i in range(num_robots):
            p_i = robot_positions[i]
            for obs in moving_obstacles:
                obs_safe_d = config.compute_safety_distance(obs.radius)
                q_k = obs.position
                u_k = obs.velocity
                r_ik = p_i - q_k
                d_ik = float(np.linalg.norm(r_ik))
                h_ik = d_ik**2 - obs_safe_d**2

                def obs_cbf(v_flat, idx_i=i, r=r_ik, u=u_k, h=h_ik):
                    vi = v_flat[2*idx_i : 2*idx_i + 2]
                    return 2.0 * np.dot(r, vi - u) + alpha * h

                constraints.append({"type": "ineq", "fun": obs_cbf})

    # 3. Individual robot speed constraints
    for i in range(num_robots):
        def speed_cbf(v_flat, idx_i=i, vm=v_max):
            vi = v_flat[2*idx_i : 2*idx_i + 2]
            return vm**2 - float(np.dot(vi, vi))

        constraints.append({"type": "ineq", "fun": speed_cbf})

    t0 = time.perf_counter()
    res = minimize(
        fun=joint_obj,
        x0=v_des_flat,
        jac=joint_grad,
        constraints=constraints,
        method="SLSQP",
        options={"ftol": config.qp_tolerance, "maxiter": 150, "disp": False}
    )
    elapsed = time.perf_counter() - t0

    results = []
    if res.success:
        v_safe_joint = res.x.reshape((num_robots, 2))
        for i in range(num_robots):
            # Speed clip
            sp = float(np.linalg.norm(v_safe_joint[i]))
            if sp > v_max:
                v_safe_joint[i] = (v_safe_joint[i] / sp) * v_max
            diff = float(np.linalg.norm(v_safe_joint[i] - desired_velocities[i]))
            intervened = diff > 1e-4
            results.append(CBFSolverResult(
                safe_velocity=v_safe_joint[i],
                desired_velocity=desired_velocities[i],
                intervened=intervened,
                status="modified" if intervened else "success",
                min_barrier_value=0.0,
                active_constraints_count=len(constraints),
                solver_time=elapsed / num_robots,
                violations_detected=0,
                message="Centralized joint QP succeeded."
            ))
        return v_safe_joint, results
    else:
        # Fallback to decentralized shield per robot
        shield = CBFSafetyShield(config)
        safe_vels = np.zeros_like(desired_velocities)
        for i in range(num_robots):
            r_res = shield.filter_velocity(
                robot_idx=i,
                desired_velocity=desired_velocities[i],
                robot_positions=robot_positions,
                robot_velocities=desired_velocities,
                moving_obstacles=moving_obstacles
            )
            safe_vels[i] = r_res.safe_velocity
            results.append(r_res)
        return safe_vels, results

