import cvxpy as cp
import numpy as np

class CBFQPSolver:
    """
    Control Barrier Function Quadratic Program (CBF-QP) Solver
    for relative longitudinal safety maintenance.
    """
    def __init__(self, 
                 gamma: float = 1.5, 
                 d_min: float = 6.0, 
                 tau: float = 0.5,
                 a_min: float = -8.0, 
                 a_max: float = 3.0):
        """
        :param gamma: CBF class-K gain (higher = more aggressive braking near boundary)
        :param d_min: Minimum hard distance buffer (meters)
        :param tau: Time headway buffer factor (seconds)
        :param a_min: Minimum acceleration / max braking (m/s^2)
        :param a_max: Maximum acceleration (m/s^2)
        """
        self.gamma = gamma
        self.d_min = d_min
        self.tau = tau
        self.a_min = a_min
        self.a_max = a_max

    def compute_barrier(self, x_ego: float, v_ego: float, x_target: float) -> float:
        """Evaluates h(x) = relative_distance - (d_min + v_ego * tau)"""
        rel_dist = x_target - x_ego
        return rel_dist - (self.d_min + v_ego * self.tau)

    def solve(self, x_ego: float, v_ego: float, x_target: float, v_target: float, v_des: float, dt: float) -> float:
        """
        Solves the QP optimization to compute safe control input u (acceleration).
        Returns optimal acceleration in m/s^2.
        """
        # Define decision variable: control input u (acceleration)
        u = cp.Variable(1)

        # Predict next states as a function of control input u
        v_ego_next = v_ego + u * dt
        x_ego_next = x_ego + v_ego * dt
        x_target_next = x_target + v_target * dt

        # Current and next barrier evaluation
        h_current = self.compute_barrier(x_ego, v_ego, x_target)
        h_next = (x_target_next - x_ego_next) - (self.d_min + v_ego_next * self.tau)

        # Objective: minimize deviation from desired target speed v_des
        objective = cp.Minimize(0.5 * cp.square(v_ego_next - v_des))

        # Constraints
        cbf_constraint = (h_next - h_current) / dt >= -self.gamma * h_current
        accel_max_constraint = u <= self.a_max
        accel_min_constraint = u >= self.a_min

        constraints = [
            cbf_constraint,
            accel_max_constraint,
            accel_min_constraint
        ]

        # Solve Quadratic Program
        problem = cp.Problem(objective, constraints)
        
        try:
            problem.solve(solver=cp.OSQP, verbose=False)
            if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                return float(u.value[0])
            else:
                # If problem is infeasible, fall back to emergency full braking
                print(f"⚠️ QP Infeasible (Status: {problem.status}). Applying emergency maximum braking.")
                return self.a_min
        except Exception as e:
            print(f"❌ CVXPY Solver Error: {e}. Defaulting to full braking.")
            return self.a_min
