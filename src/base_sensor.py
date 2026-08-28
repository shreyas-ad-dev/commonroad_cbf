import numpy as np

from src.ego_state import EgoState, get_car_polygon


class BaseSensor:
    """
    Base class for vehicle perception sensors.

    Provides common geometric utilities, including coordinate frame 
    transformations (world to ego-local) and obstacle bounding box 
    point extraction.

    Attributes:
        range_max (float): Maximum detection range of the sensor in meters.
        fov_deg (float): Total field-of-view angle in degrees.
        half_fov_rad (float): Half of the field-of-view angle in radians.
    """

    def __init__(self,
                 range_max: float,
                 fov_deg: float):
        """
        Initializes the base sensor parameters.

        Args:
            range_max (float): Maximum detection range in meters.
            fov_deg (float): Total field of view in degrees.
        """

        self.range_max = range_max
        self.fov_deg = fov_deg
        self.half_fov_rad = np.radians(fov_deg / 2.0)

    @staticmethod
    def to_local_frame(
            ego: EgoState,
            obs_x: float,
            obs_y: float) -> np.ndarray:
        """
        Transforms 2D global world coordinates into Ego's body-fixed local frame.

        Args:
            ego (EgoState): Current state of the Ego vehicle.
            obs_x (float): Target X position in the global frame.
            obs_y (float): Target Y position in the global frame.

        Returns:
            np.ndarray: A 2-element array [x_local, y_local] representing 
                the local coordinates relative to the Ego vehicle.
        """

        d_vec = np.array([obs_x, obs_y]) - ego.position
        x_local = np.dot(d_vec, ego.heading_vector)
        y_local = np.dot(d_vec, ego.normal_vector)
        return np.array([x_local, y_local])

    def get_obstacle_center_and_corners_in_local(self,
                                       ego: EgoState,
                                       obstacle: object,
                                       step: int) -> tuple[np.ndarray, list[np.ndarray]] | None:
        """Extracts the local center and corner points of an obstacle.

        Computes the center position and four bounding box corners of an 
        obstacle at a given time step, transformed into Ego's local frame.

        Args:
            ego (EgoState): Current state of the Ego vehicle.
            obstacle (object): Target obstacle object supporting `state_at_time(step)`.
            step (int): Discrete simulation time step.

        Returns:
            tuple[np.ndarray, list[np.ndarray]] | None: A tuple containing:
                - local_obs_pos_center (np.ndarray): Local center position [x, y].
                - local_obs_corner_points (list[np.ndarray]): List of 5 local points 
                  (center followed by the 4 bounding box corners).
                Returns None if the obstacle state is unavailable at the given step.
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
#
#    def get_min_distance(self,
#                         ego: EgoState,
#                         obstacle: object,
#                         step: int) -> float:
#        """
#        Calculates the minimum distance from Ego to an obstacle's key points.
#
#        Evaluates distance across all key local points (center + 4 bounding corners) 
#        and returns the shortest Euclidean distance.
#
#        Args:
#            ego (EgoState): Current state of the Ego vehicle.
#            obstacle (object): Target obstacle object.
#            step (int): Discrete simulation time step.
#
#        Returns:
#            float: Minimum Euclidean distance to any obstacle point in meters. 
#                Returns float('inf') if the obstacle state is invalid.
#        """
#
#        res = self.get_obstacle_eval_points_local(ego, obstacle, step)
#        if res is None:
#            return float('inf')
#
#        _, local_points = res
#        return min(float(np.linalg.norm(pt)) for pt in local_points)
