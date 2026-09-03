import numpy as np

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False

from src.ego_state import EgoState
from src.tracker import Track


class CBFQPSolver:
    """
    Control Barrier Function Quadratic Program (CBF-QP) Solver.
    
    Maintains relative longitudinal safety along any lane orientation by solving 
    for safe acceleration commands that satisfy barrier function constraints.
    """
    
    def __init__(self, 
                 gamma: float = 1.2, 
                 d_min: float = 6.0, 
                 tau: float = 0.5, 
                 a_min: float = -8.0, 
                 a_max: float = 2.0,
                 use_cvxpy: bool = False):
        """
        Initializes the CBFQPSolver instance.

        Args:
            gamma (float): CBF class-K gain (higher values yield more aggressive braking near boundaries).
            d_min (float): Minimum hard distance buffer in meters.
            tau (float): Time headway buffer factor in seconds.
            a_min (float): Minimum acceleration / maximum braking limit in m/s^2.
            a_max (float): Maximum acceleration limit in m/s^2.
            use_cvxpy (bool): Toggles CVXPY optimization solver vs analytical closed-form solution.
        """
        
        self.gamma = gamma
        self.d_min = d_min
        self.tau = tau
        self.a_min = a_min
        self.a_max = a_max
        self.use_cvxpy = use_cvxpy and CVXPY_AVAILABLE

    def compute_barrier(self, longitudinal_dist: float, v_ego: float) -> float:
        """
        Computes the CBF barrier value h(x) based on relative longitudinal distance.

        Args:
            longitudinal_dist (float): Relative distance to target vehicle in Ego's local longitudinal frame.
            v_ego (float): Current speed of the Ego vehicle in m/s.

        Returns:
            float: Value of the safety barrier h(x). Values >= 0 indicate a safe state.
        """
        
        d_safe = self.d_min + (v_ego * self.tau)
        return longitudinal_dist - d_safe

    def solve_from_track(self,
                         ego: EgoState,
                         lead_track: Track,
                         v_des: float,
                         dt: float) -> float:
        """
        Convenience wrapper to solve CBF-QP directly from a filtered Track object.
        """
        if lead_track is None:
            # No lead vehicle tracked, apply nominal acceleration towards target speed[cite: 12]
            u_nom = 0.5 * (v_des - ego.velocity)
            return float(np.clip(u_nom, self.a_min, self.a_max))

        u_road, _ = ego.road_frame_vectors
        d_vec = lead_track.position - ego.position
        longitudinal_dist = float(np.dot(d_vec, u_road))

        # Filtered target velocity magnitude from track state vector[cite: 10]
        v_target = float(np.hypot(lead_track.velocity[0], lead_track.velocity[1]))

        return self.solve(
            longitudinal_dist=longitudinal_dist,
            v_ego=ego.velocity,
            v_target=v_target,
            v_des=v_des,
            dt=dt
        )

    def solve(self, 
              longitudinal_dist: float, 
              v_ego: float, 
              v_target: float, 
              v_des: float, 
              dt: float) -> float:
        """
        Solves for the optimal safe acceleration command (u) enforcing h(x) >= 0.

        Args:
            longitudinal_dist (float): Relative distance in Ego's local longitudinal frame (x_local).
            v_ego (float): Current speed of the Ego vehicle in m/s.
            v_target (float): Current speed of the target vehicle in m/s.
            v_des (float): Desired cruising speed of the Ego vehicle in m/s.
            dt (float): Simulation time step delta in seconds.

        Returns:
            float: Optimal safe acceleration command in m/s^2.
        """

        h_val = self.compute_barrier(longitudinal_dist, v_ego)
        u_nom = 0.5 * (v_des - v_ego)

        if self.use_cvxpy:
            return self._solve_cvxpy(longitudinal_dist, v_ego, v_target, v_des, dt, h_val)
        else:
            return self._solve_analytical(h_val, v_ego, v_target, u_nom)

    def _solve_analytical(self,
                          h_val: float,
                          v_ego: float,
                          v_target: float,
                          u_nom: float) -> float:
        """
        Computes the closed-form analytical solution to the scalar CBF-QP.

        Enforces h_dot = v_rel - u * tau >= -gamma * h, which simplifies to:
        u * tau <= v_rel + gamma * h.

        Args:
            h_val (float): Current evaluation of the barrier function h(x).
            v_ego (float): Current speed of the Ego vehicle in m/s.
            v_target (float): Speed of the target lead vehicle in m/s.
            u_nom (float): Nominal unconstrained acceleration input.

        Returns:
            float: Safe acceleration control bounded by physical limits [a_min, a_max].
        """

        v_rel = v_target - v_ego
        
        if self.tau > 0.0:
            max_allowed_accel = (v_rel + self.gamma * h_val) / self.tau
        else:
            max_allowed_accel = self.a_max

        u_cbf = min(u_nom, max_allowed_accel)
        return float(np.clip(u_cbf, self.a_min, self.a_max))

    def _solve_cvxpy(self,
                     longitudinal_dist: float,
                     v_ego: float,
                     v_target: float,
                     v_des: float,
                     dt: float,
                     h_current: float) -> float:
        """
        Solves the discrete-time CBF-QP using explicit CVXPY optimization.

        Args:
            longitudinal_dist (float): Relative distance in Ego's local longitudinal frame (x_local).
            v_ego (float): Current speed of the Ego vehicle in m/s.
            v_target (float): Speed of the target lead vehicle in m/s.
            v_des (float): Desired cruising speed in m/s.
            dt (float): Simulation time step delta in seconds.
            h_current (float): Current value of the safety barrier function.

        Returns:
            float: Optimal control input (acceleration) in m/s^2. Returns minimum acceleration (a_min) if infeasible.
        """
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
        except (ValueError, RuntimeError):
            return self.a_min
