#src/lateral_controller.py

import numpy as np

from src.ego_state import EgoState


def generate_lane_change_path(
    ego: EgoState,
    target_lane_offset: float = 3.5,
    total_length: float = 150.0,
    num_points: int = 200) -> np.ndarray:
    """
    Generates reference waypoints for a smooth lane change maneuver along any road heading.

    Uses a quintic polynomial S-curve profile for smooth lateral acceleration and minimum jerk.

    Args:
        ego (EgoState): Current state of the Ego vehicle providing position and road frame vectors.
        target_lane_offset (float, optional): Lateral offset to target lane (+ for left, - for right). Defaults to 3.5.
        total_length (float, optional): Total longitudinal path distance in meters. Defaults to 150.0.
        num_points (int, optional): Number of discrete waypoint samples to generate. Defaults to 200.

    Returns:
        np.ndarray: An Nx2 array of global [x, y] coordinates forming the lane change reference path.
    """   
    
    # 1. Distance along track axis
    s = np.linspace(0, total_length, num_points)

    # 2. Smooth S-curve transition profile (Quintic polynomial)
    start_lc, lc_length = 10.0, 35.0
    t = np.clip((s - start_lc) / lc_length, 0.0, 1.0)
    d_offset = target_lane_offset * (6 * t**5 - 15 * t**4 + 10 * t**3)

    # 3. Direction vectors: Forward (u_hat) and Perpendicular Normal (n_hat)
    u_hat, n_hat = ego.road_frame_vectors

    # 4. Map back to global coordinates (Vectorized outer product)
    path_points = ego.position + np.outer(s, u_hat) + np.outer(d_offset, n_hat)

    return path_points

class StanleyController:
    """
    Nonlinear steering controller using the Stanley steering law.

    Targeting the vehicle's front axle to the reference path, balancing cross-track error 
    and heading error to compute saturated front-wheel steering commands.
    """
    
    def __init__(self,
                 k: float = 0.5,
                 k_soft: float = 1.0,
                 max_steer_deg: float = 25.0,
                 wheelbase: float = 2.8):
        """
        Initializes the StanleyController instance.

        Args:
            k (float, optional): Gain parameter for cross-track error response. Defaults to 0.5.
            k_soft (float, optional): Softening gain to prevent numerical instability at low speeds. Defaults to 1.0.
            max_steer_deg (float, optional): Maximum steering angle limit in degrees. Defaults to 25.0.
            wheelbase (float, optional): Vehicle wheelbase distance in meters. Defaults to 2.8.
        """

        self.k = k
        self.k_soft = k_soft
        self.max_steer = np.radians(max_steer_deg)
        self.wheelbase = wheelbase
        
        self.steering = 0


    def compute_steering(self,
                         ego: EgoState,
                         reference_path: np.ndarray) -> float:
        """
        Computes the front-axle Stanley steering command to track the target reference path.

        Args:
            ego (EgoState): Current state and kinematics of the Ego vehicle.
            reference_path (np.ndarray): Nx2 array of target waypoint coordinates [[x, y], ...].

        Returns:
            float: Saturated front-wheel steering angle command in radians.
        """

        # 1. Front axle position directly from EgoState
        front_axle = ego.front_axle_position

        # 2. Find nearest point on reference path from front axle
        d_vecs = reference_path - front_axle
        distances = np.hypot(d_vecs[:, 0], d_vecs[:, 1])
        min_idx = int(np.argmin(distances))

        # 3. Compute Path Tangent Angle (Path Yaw)
        if min_idx < len(reference_path) - 1:
            tangent = reference_path[min_idx + 1] - reference_path[min_idx]
        else:
            tangent = reference_path[min_idx] - reference_path[min_idx - 1]

        path_yaw = np.arctan2(tangent[1], tangent[0])

        # Heading error
        heading_error = ego.orientation - path_yaw
        heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi  # Normalize to [-pi, pi]

        # 4. Cross-track error (distance from front axle to path segment)
        vec_path_to_front = front_axle - reference_path[min_idx]
        perp_vec = np.array([-np.sin(path_yaw), np.cos(path_yaw)])
        crosstrack_error = np.dot(vec_path_to_front, perp_vec)

        # 5. Stanley Steering Law
        self.steering = -heading_error + np.arctan2(-self.k * crosstrack_error, ego.velocity + self.k_soft)
        return float(np.clip(self.steering, -self.max_steer, self.max_steer))

    def get_log_data(self) -> dict:
        return {
                "steering": self.steering
                }

