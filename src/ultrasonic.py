# src/ultrasonic.py
import numpy as np
from typing import Tuple, Optional, Set
from src.ego_state import EgoState, get_car_polygon

class SideUltrasonicSensor:
    """
    Simulates short-range side-facing ultrasonic sensors (USS) for blind-spot monitoring.

    Mount positions can be 'left' (+90 deg) or 'right' (-90 deg) relative to the Ego body frame.
    """

    def __init__(self,
                 range_max: float = 8.0,
                 fov_deg: float = 100.0,
                 side: str = "left"):
        """
        Initializes the SideUltrasonicSensor instance.

        Args:
            range_max (float, optional): Maximum detection range in meters. Defaults to 8.0.
            fov_deg (float, optional): Total field of view in degrees. Defaults to 100.0.
            side (str, optional): Sensor side mounting ('left' or 'right'). Defaults to "left".

        Raises:
            ValueError: If side is not 'left' or 'right'.
        """

        if side not in ["left", "right"]:
            raise ValueError("side must be either 'left' or 'right'")
        self.range_max = range_max
        self.fov_deg = fov_deg
        self.side = side
        self.half_fov_rad = np.radians(fov_deg / 2.0)

    def is_in_fov(self,
                  ego: EgoState,
                  obstacle: object,
                  step: int) -> tuple[bool, float]:
        """
        Evaluates whether an obstacle's center or bounding box corners fall within the sensor FOV cone.

        Args:
            ego (EgoState): Current state of the Ego vehicle.
            obstacle (object): Dynamic obstacle instance to test against.
            step (int): Current simulation time step index.

        Returns:
            tuple[bool, float]: A tuple containing:
                - any_corner_in_fov (bool): True if any corner or center of obstacle is in FOV.
                - min_dist (float): Minimum Euclidean distance from Ego position to obstacle points.
        """

        st = obstacle.state_at_time(step)
        if st is None:
            return False, float('inf')

#        if target_offset != 0.0:
#            d_center = st.position - ego.position
#            y_center_local = np.dot(d_center, ego.normal_vector)
#            if abs(y_center_local - target_offset) > lane_tolerance:
#                return False, float('inf')

        # Extract length and width of the obstacle
        l = getattr(obstacle.obstacle_shape, "length", 4.5)
        w = getattr(obstacle.obstacle_shape, "width", 2.0)
        ox, oy = st.position[0], st.position[1]
        o_orient = getattr(st, "orientation", 0.0)

        # Get the 4 corner points of the obstacle vehicle
        _, obs_corners = get_car_polygon(ox, oy, o_orient, length=l, width=w)

        min_dist = float('inf')
        any_corner_in_fov = False

        # Check center + all 4 corners
        points_to_check = [st.position] + list(obs_corners)

        for pt in points_to_check:
            d_vec = pt - ego.position
            x_local = np.dot(d_vec, ego.heading_vector)
            y_local = np.dot(d_vec, ego.normal_vector)
            dist = float(np.hypot(x_local, y_local))

            if dist < min_dist:
                min_dist = dist

            if dist <= self.range_max:
                # Side check
                if (self.side == "left" and y_local > 0.0) or (self.side == "right" and y_local < 0.0):
                    sensor_y = y_local if self.side == "left" else -y_local
                    angle = np.arctan2(x_local, sensor_y)
                    if abs(angle) <= self.half_fov_rad:
                        #if target_offset == 0.0 or abs(y_local - target_offset) <= (lane_tolerance + w /2.0):
                        any_corner_in_fov = True

        return any_corner_in_fov, min_dist

    def get_detected_obstacle_ids(self,
                                  ego: EgoState,
                                  obstacles: list,
                                  step: int) -> Set:
        """
        Queries obstacle IDs currently residing inside this sensor's blind-spot FOV.

        Args:
            ego (EgoState): Current state of the Ego vehicle.
            obstacles (list): List of dynamic obstacle objects.
            step (int): Current simulation time step index.

        Returns:
            Set: Set of obstacle IDs detected within the FOV cone.
        """

        detected_ids = set()
        for obs in obstacles:
            st = obs.state_at_time(step)
            if st is None:
                continue
            ox, oy = st.position[0], st.position[1]
            in_fov, _ = self.is_in_fov(ego, obs, step)
            if in_fov:
                detected_ids.add(obs.obstacle_id)
        return detected_ids

    def is_adjacent_lane_clear(self,
                               ego: EgoState,
                               obstacles: list,
                               step: int,
                               target_offset: float) -> bool:
        """
        Evaluates whether the side blind spot is clear of obstacles in the target direction.

        Args:
            ego (EgoState): Current state of the Ego vehicle.
            obstacles (list): List of dynamic obstacles in the scene.
            step (int): Current simulation time step index.
            target_offset (float): Target lateral offset (> 0 for left change, < 0 for right change).

        Returns:
            bool: True if the targeted side blind spot has no detected obstacles.
        """

        # Ignore check if target offset direction does not match sensor mounting side

        if (target_offset > 0 and self.side != "left") or (target_offset < 0 and self.side != "right"):
            return True

        # Query tracked obstacle IDs in USS field of view
        detected_ids = self.get_detected_obstacle_ids(ego, obstacles, step)
        
        # Lane is clear only if zero obstacles occupy the side blind spot
        return len(detected_ids) == 0
