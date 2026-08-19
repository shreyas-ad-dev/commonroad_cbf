import numpy as np

from src.ego_state import EgoState, get_car_polygon


class BaseSensor:
    """
    Base class for vehicle perception sensors. Handles local frame transformations 
    and geometric point sampling for bounding box corners.
    """

    def __init__(self,
                 range_max: float,
                 fov_deg: float):
        self.range_max = range_max
        self.fov_deg = fov_deg
        self.half_fov_rad = np.radians(fov_deg / 2.0)

    @staticmethod
    def to_local_frame(
            ego: EgoState,
            obs_x: float,
            obs_y: float) -> np.ndarray:
        """
        Transforms world coordinates into Ego's body-fixed local frame.

        Args:
            ego (EgoState): Current state of the Ego vehicle.
            obs_x (float): Target X coordinate in global world frame.
            obs_y (float): Target Y coordinate in global world frame.

        Returns:
            np.ndarray: A 2D vector [x_local, y_local] in Ego's body-fixed frame.
        """

        d_vec = np.array([obs_x, obs_y]) - ego.position
        x_local = np.dot(d_vec, ego.heading_vector)
        y_local = np.dot(d_vec, ego.normal_vector)
        return np.array([x_local, y_local])

    def get_obstacle_center_and_corners_in_local(self,
                                       ego: EgoState,
                                       obstacle: object,
                                       step: int) -> tuple[np.ndarray, list[np.ndarray]] | None:
        """
        Extracts the center and 4 bounding box corner points of an obstacle, 
        transformed into Ego's local coordinate frame.

        Returns:
            Tuple[local_obs_pos_center, local_obs_corner_points] or None if step is invalid.
        """
        st = obstacle.state_at_time(step)
        if st is None:
            return None

        # Center position in local frame
        local_obs_pos_center = self.to_local_frame(ego, st.position[0], st.position[1])

        # Extract dimensions & calculate world-frame corners
        length = getattr(obstacle.obstacle_shape, "length", 4.5)
        width = getattr(obstacle.obstacle_shape, "width", 2.0)
        o_orient = getattr(st, "orientation", 0.0)
        _, obs_corners = get_car_polygon(st.position[0], st.position[1], o_orient, length=length, width=width)

        # Convert center + 4 corners to local frame
        world_obs_corner_points = [st.position] + list(obs_corners)
        local_obs_corner_points = [self.to_local_frame(ego, pt[0], pt[1]) for pt in world_obs_corner_points]

        return local_obs_pos_center, local_obs_corner_points

    def get_min_distance(self,
                         ego: EgoState,
                         obstacle: object,
                         step: int) -> float:
        """Calculates minimum Euclidean distance from Ego to any obstacle corner/center."""
        res = self.get_obstacle_eval_points_local(ego, obstacle, step)
        if res is None:
            return float('inf')

        _, local_points = res
        return min(float(np.linalg.norm(pt)) for pt in local_points)
