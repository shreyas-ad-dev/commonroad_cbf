from scipy.interpolate import interp1d
import numpy as np

def extract_target_lanelet_path(scenario, ego_x: float, ego_y: float, horizon_meters: float = 200.0) -> np.ndarray:
    """
    Universally extracts and chains lanelet centerlines along the successor path 
    from the Ego vehicle's start position across ANY CommonRoad scenario.
    """
    # 1. Find initial lanelet
    lanelet_ids = scenario.lanelet_network.find_lanelet_by_position([np.array([ego_x, ego_y])])[0]
    if not lanelet_ids:
        raise ValueError(f"No lanelet found near position ({ego_x}, {ego_y})")

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


def get_current_lane_width(scenario, ego_x: float, ego_y: float, default_width: float = 3.5) -> float:
    """
    Finds the lanelet surrounding (ego_x, ego_y) and computes its local width
    by measuring the distance between its left and right boundaries.
    """
    # 1. Find lanelets containing the vehicle position
    lanelet_ids = scenario.lanelet_network.find_lanelet_by_position([np.array([ego_x, ego_y])])[0]

    if not lanelet_ids:
        return default_width

    # 2. Extract the active lanelet object
    lanelet = scenario.lanelet_network.find_lanelet_by_id(lanelet_ids[0])

    # 3. Calculate distance between left and right boundary vertices
    left_verts = lanelet.left_vertices
    right_verts = lanelet.right_vertices

    # Compute widths along all vertices and return the mean/median width
    widths = np.hypot(left_verts[:, 0] - right_verts[:, 0], left_verts[:, 1] - right_verts[:, 1])

    return float(np.mean(widths))


def generate_lane_change_path(start_x, start_y, road_heading, target_lane_offset=3.5, total_length=150.0, num_points=200):
    """
    Generates waypoints for a lane change along an angled road track.

    road_heading: Initial road orientation in radians (e.g., -0.62 rad for US-101)
    target_lane_offset: Distance to adjacent lane in meters (+3.5m left, -3.5m right)
    """
    # 1. Distance along track axis
    s = np.linspace(0, total_length, num_points)

    # 2. Smooth S-curve transition profile (Quintic polynomial)
    start_lc, lc_length = 10.0, 35.0
    d_offset = np.zeros_like(s)

    for i, si in enumerate(s):
        if si < start_lc:
            d_offset[i] = 0.0
        elif si > start_lc + lc_length:
            d_offset[i] = target_lane_offset
        else:
            t = (si - start_lc) / lc_length
            d_offset[i] = target_lane_offset * (6 * t**5 - 15 * t**4 + 10 * t**3)

    # 3. Direction vectors: Forward (u_hat) and Perpendicular Normal (n_hat)
    u_hat = np.array([np.cos(road_heading), np.sin(road_heading)])
    n_hat = np.array([-np.sin(road_heading), np.cos(road_heading)])

    # 4. Map back to global coordinates
    path_points = []
    for i in range(num_points):
        pt = np.array([start_x, start_y]) + s[i] * u_hat + d_offset[i] * n_hat
        path_points.append(pt)

    return np.array(path_points)


class StanleyController:
    def __init__(self, k: float = 0.5, k_soft: float = 1.0, max_steer_deg: float = 25.0, wheelbase: float = 2.8):
        self.k = k
        self.k_soft = k_soft
        self.max_steer = np.radians(max_steer_deg)
        self.wheelbase = wheelbase

    def compute_steering(self, ego_x: float, ego_y: float, ego_yaw: float, ego_v: float, reference_path: np.ndarray) -> float:
        """
        ego_x, ego_y, ego_yaw, ego_v: Current state of ego vehicle
        reference_path: Nx2 numpy array of target lane waypoints [[x1, y1], [x2, y2], ...]
        """
        # 1. Estimate front axle position
        fx = ego_x + self.wheelbase * np.cos(ego_yaw)
        fy = ego_y + self.wheelbase * np.sin(ego_yaw)

        # 2. Find nearest point on reference path
        dx = reference_path[:, 0] - fx
        dy = reference_path[:, 1] - fy
        distances = np.hypot(dx, dy)
        min_idx = np.argmin(distances)

        # 3. Compute Path Tangent Angle (Path Yaw)
        if min_idx < len(reference_path) - 1:
            path_yaw = np.arctan2(
                reference_path[min_idx + 1, 1] - reference_path[min_idx, 1],
                reference_path[min_idx + 1, 0] - reference_path[min_idx, 0]
            )
        else:
            path_yaw = np.arctan2(
                reference_path[min_idx, 1] - reference_path[min_idx - 1, 1],
                reference_path[min_idx, 0] - reference_path[min_idx - 1, 0]
            )

        # Heading error
        heading_error = ego_yaw - path_yaw
        heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi  # Normalize to [-pi, pi]

        # 4. Cross-track error (distance to path)
        vec_path_to_front = np.array([fx - reference_path[min_idx, 0], fy - reference_path[min_idx, 1]])
        perp_vec = np.array([-np.sin(path_yaw), np.cos(path_yaw)])
        crosstrack_error = np.dot(vec_path_to_front, perp_vec)

        # 5. Stanley Law
        steering = -heading_error + np.arctan2(-self.k * crosstrack_error, ego_v + self.k_soft)
        return float(np.clip(steering, -self.max_steer, self.max_steer))
