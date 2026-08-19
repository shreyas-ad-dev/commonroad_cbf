import numpy as np
from scipy.interpolate import interp1d

from src.ego_state import EgoState


def get_road_heading_at_position(scenario, position) -> float:
    """
    Computes the road heading angle (in radians) from the centerline 
    of the nearest lanelet at a given (x, y) position.
    """
    # Find lanelet(s) containing or closest to the position
    lanelet_ids = scenario.lanelet_network.find_lanelet_by_position([position])[0]
    
    if not lanelet_ids:
        # Fallback to absolute nearest lanelet if strictly outside boundaries
        lanelet = scenario.lanelet_network.find_lanelet_by_id(
            scenario.lanelet_network.find_lanelet_by_position([position])[0]
        )
    else:
        lanelet = scenario.lanelet_network.find_lanelet_by_id(lanelet_ids[0])

    # Extract centerline points
    centerline = lanelet.center_vertices
    
    # Find the closest segment on the centerline
    distances = np.linalg.norm(centerline - position, axis=1)
    idx = np.argmin(distances)

    # Compute heading vector along the centerline segment
    if idx < len(centerline) - 1:
        pt1, pt2 = centerline[idx], centerline[idx + 1]
    else:
        pt1, pt2 = centerline[idx - 1], centerline[idx]

    dx = pt2[0] - pt1[0]
    dy = pt2[1] - pt1[1]

    return np.arctan2(dy, dx)

def extract_target_lanelet_path(
        scenario,
        ego: EgoState,
        horizon_meters: float = 200.0) -> np.ndarray:
    """
    Extracts and chains lanelet centerlines along the successor path from the Ego vehicle's position.

    Traverses successor lanelets in a CommonRoad scenario until the cumulative length reaches 
    or exceeds the target horizon, then densely resamples the path points every 0.5 meters.

    Args:
        scenario: The CommonRoad scenario instance containing the lanelet network.
        ego (EgoState): Current state of the Ego vehicle.
        horizon_meters (float, optional): Total longitudinal distance forward to chain. Defaults to 200.0.

    Returns:
        np.ndarray: An Nx2 numpy array of resampled path coordinates [[x1, y1], [x2, y2], ...].

    Raises:
        ValueError: If no lanelet is found near the Ego vehicle's current position.
    """

    # 1. Find initial lanelet
    lanelet_ids = scenario.lanelet_network.find_lanelet_by_position([ego.position])[0]
    if not lanelet_ids:
        raise ValueError(f"No lanelet found near position {ego.position}")

    current_lanelet = scenario.lanelet_network.find_lanelet_by_id(lanelet_ids[0])
    all_center_verts = [np.array(current_lanelet.center_vertices)]
    
    total_length = 0.0
    for verts in all_center_verts:
        total_length += np.sum(np.hypot(np.diff(verts[:, 0]), np.diff(verts[:, 1])))

    # 2. Chain successor lanelets until horizon_meters is reached
    while total_length < horizon_meters and current_lanelet.successor:
        next_id = current_lanelet.successor[0]  # Follow primary successor
        current_lanelet = scenario.lanelet_network.find_lanelet_by_id(next_id)
        
        next_verts = np.array(current_lanelet.center_vertices)
        all_center_verts.append(next_verts[1:])  # Skip duplicate start vertex
        
        total_length += np.sum(np.hypot(np.diff(next_verts[:, 0]), np.diff(next_verts[:, 1])))

    # 3. Combine chained points
    center_verts = np.vstack(all_center_verts)

    # 4. Resample densely every 0.5m
    distances = np.zeros(len(center_verts))
    distances[1:] = np.cumsum(np.hypot(np.diff(center_verts[:, 0]), np.diff(center_verts[:, 1])))

    s_dense = np.arange(0, distances[-1], 0.5)
    interp_x = interp1d(distances, center_verts[:, 0], kind='linear')
    interp_y = interp1d(distances, center_verts[:, 1], kind='linear')

    return np.column_stack((interp_x(s_dense), interp_y(s_dense)))


def get_current_lane_width(
        scenario,
        ego: EgoState,
        default_width: float = 3.5) -> float:
    """
    Computes the local lane width surrounding the Ego vehicle's current position.

    Identifies the active lanelet and measures the average Euclidean distance 
    between corresponding vertices of its left and right boundaries.

    Args:
        scenario: The CommonRoad scenario instance containing the lanelet network.
        ego (EgoState): Current state of the Ego vehicle.
        default_width (float, optional): Fallback lane width in meters if no lanelet is found. Defaults to 3.5.

    Returns:
        float: Mean local lane width in meters.
    """
    
    # 1. Find lanelets containing the vehicle position
    lanelet_ids = scenario.lanelet_network.find_lanelet_by_position([ego.position])[0]

    if not lanelet_ids:
        return default_width

    # 2. Extract the active lanelet object
    lanelet = scenario.lanelet_network.find_lanelet_by_id(lanelet_ids[0])

    # 3. Calculate distance between left and right boundary vertices
    left_verts = np.array(lanelet.left_vertices)
    right_verts = np.array(lanelet.right_vertices)

    # Compute widths along all vertices and return the mean/median width
    widths = np.hypot(left_verts[:, 0] - right_verts[:, 0], left_verts[:, 1] - right_verts[:, 1])

    return float(np.mean(widths))

def generate_lane_change_path(
    start_pos: np.ndarray,
    road_heading: float,
    target_lane_offset: float = 3.5,
    total_length: float = 150.0,
    num_points: int = 200) -> np.ndarray:
    """
    Generates reference waypoints for a smooth lane change maneuver along any road heading.

    Uses a quintic polynomial S-curve profile for smooth lateral acceleration and minimum jerk.

    Args:
        start_pos (np.ndarray): Starting 2D position vector [x, y] of the maneuver in world frame.
        road_heading (float): Current road/lane orientation angle in radians.
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
    u_hat = np.array([np.cos(road_heading), np.sin(road_heading)])
    n_hat = np.array([-np.sin(road_heading), np.cos(road_heading)])

    # 4. Map back to global coordinates (Vectorized outer product)
    path_points = start_pos + np.outer(s, u_hat) + np.outer(d_offset, n_hat)

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
        steering = -heading_error + np.arctan2(-self.k * crosstrack_error, ego.velocity + self.k_soft)
        return float(np.clip(steering, -self.max_steer, self.max_steer))

