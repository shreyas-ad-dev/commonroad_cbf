from typing import Optional
import numpy as np

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False


class CBFQPSolver:
    """
    Control Barrier Function Quadratic Program (CBF-QP) Solver
    for relative longitudinal safety maintenance along any lane orientation.
    """
    def __init__(self, 
                 gamma: float = 1.2, 
                 d_min: float = 6.0, 
                 tau: float = 0.5, 
                 a_min: float = -8.0, 
                 a_max: float = 2.0,
                 use_cvxpy: bool = False):
        """
        :param gamma: CBF class-K gain (higher = more aggressive braking near boundary)
        :param d_min: Minimum hard distance buffer (meters)
        :param tau: Time headway buffer factor (seconds)
        :param a_min: Minimum acceleration / max braking (m/s^2)
        :param a_max: Maximum acceleration (m/s^2)
        :param use_cvxpy: Toggle CVXPY optimization solver vs analytical closed-form solution
        """
        self.gamma = gamma
        self.d_min = d_min
        self.tau = tau
        self.a_min = a_min
        self.a_max = a_max
        self.use_cvxpy = use_cvxpy and CVXPY_AVAILABLE

    def compute_barrier(self, longitudinal_dist: float, v_ego: float) -> float:
        """
        Computes CBF barrier value h(x) based on relative longitudinal distance.
        h(x) >= 0 implies safe state.
        """
        d_safe = self.d_min + (v_ego * self.tau)
        return longitudinal_dist - d_safe

    def solve(self, 
              longitudinal_dist: float, 
              v_ego: float, 
              v_target: float, 
              v_des: float, 
              dt: float) -> float:
        """
        Solves CBF-QP acceleration command (u) to enforce h(x) >= 0.
        
        :param longitudinal_dist: Relative distance in Ego's local longitudinal frame (x_local)
        :param v_ego: Current Ego speed (m/s)
        :param v_target: Target vehicle speed (m/s)
        :param v_des: Desired cruising speed (m/s)
        :param dt: Time step delta (seconds)
        :return: Optimal safe acceleration command (m/s^2)
        """
        h_val = self.compute_barrier(longitudinal_dist, v_ego)
        u_nom = 0.5 * (v_des - v_ego)

        if self.use_cvxpy:
            return self._solve_cvxpy(longitudinal_dist, v_ego, v_target, v_des, dt, h_val)
        else:
            return self._solve_analytical(h_val, v_ego, v_target, u_nom)

    def _solve_analytical(self, h_val: float, v_ego: float, v_target: float, u_nom: float) -> float:
        """
        Analytical solution to scalar CBF-QP:
        h_dot = v_rel - u * tau >= -gamma * h
        => u * tau <= v_rel + gamma * h
        """
        v_rel = v_target - v_ego
        
        if self.tau > 0.0:
            max_allowed_accel = (v_rel + self.gamma * h_val) / self.tau
        else:
            max_allowed_accel = self.a_max

        u_cbf = min(u_nom, max_allowed_accel)
        return float(np.clip(u_cbf, self.a_min, self.a_max))

    def _solve_cvxpy(self, longitudinal_dist: float, v_ego: float, v_target: float, v_des: float, dt: float, h_current: float) -> float:
        """Explicit CVXPY formulation using discrete-time barrier evolution."""
        u = cp.Variable(1)

        v_ego_next = v_ego + u * dt
        x_local_next = longitudinal_dist + (v_target - v_ego) * dt
        h_next = x_local_next - (self.d_min + v_ego_next * self.tau)

        objective = cp.Minimize(0.5 * cp.square(v_ego_next - v_des))
        cbf_constraint = (h_next - h_current) / dt >= -self.gamma * h_current

        constraints = [
            cbf_constraint,
            u <= self.a_max,
            u >= self.a_min
        ]

        problem = cp.Problem(objective, constraints)
        try:
            problem.solve(solver=cp.OSQP, verbose=False)
            if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                return float(u.value[0])
            else:
                return self.a_min
        except Exception:
            return self.a_min
