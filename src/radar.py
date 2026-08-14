# src/radar.py
import numpy as np
from src.ego_state import EgoState

class RadarSensor:
    """
    Simulates a body-frame aligned Radar sensor with finite range and Field of View (FOV).

    Supports 'front' (+x_local) and 'rear' (-x_local) mounting orientations on the vehicle.
    """
    def __init__(self,
                 range_max: float = 70.0,
                 fov_deg: float = 60.0,
                 mount_position: str = "front"):
        """
        Initializes the RadarSensor instance.

        Args:
            range_max (float, optional): Maximum detection range in meters. Defaults to 70.0.
            fov_deg (float, optional): Total azimuth field of view in degrees. Defaults to 60.0.
            mount_position (str, optional): Orientation relative to vehicle ('front' or 'rear'). Defaults to "front".

        Raises:
            ValueError: If mount_position is not 'front' or 'rear'.
        """
       
        if mount_position not in ["front", "rear"]:
            raise ValueError("mount_position must be either 'front' or 'rear'")

        self.range_max = range_max
        self.fov_deg = fov_deg
        self.mount_position = mount_position
        self.half_fov_rad = np.radians(fov_deg / 2.0)

    def to_local_frame(self,
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
       
        # [dx, dy]
        d_vec = np.array([obs_x, obs_y]) - ego.position

        # dx * cos_a + dy * sin_a
        x_local = np.dot(d_vec, ego.heading_vector)
        # -dx * sin_a + dy * cos_a
        y_local = np.dot(d_vec, ego.normal_vector) 

        return np.array([x_local, y_local])

    def is_in_fov(self,
                  ego: EgoState,
                  obs_x: float,
                  obs_y: float) -> tuple[bool, float, float, float]:
        """
        Checks if a target coordinate falls within the Radar's field of view cone.

        Args:
            ego (EgoState): Current state of the Ego vehicle.
            obs_x (float): Target X coordinate in global world frame.
            obs_y (float): Target Y coordinate in global world frame.

        Returns:
            tuple[bool, float, float, float]: A tuple containing:
                - in_fov (bool): True if target is within detection cone.
                - dist (float): Euclidean distance to target in meters.
                - x_local (float): Target longitudinal offset in Ego local frame.
                - y_local (float): Target lateral offset in Ego local frame.
        """

        pos_local = self.to_local_frame(ego, obs_x, obs_y)
        dist = float(np.linalg.norm(pos_local))

        if dist > self.range_max:
            return False, dist, pos_local[0], pos_local[1]

        # Longitudinal direction constraint based on mounting point
        if self.mount_position == "front" and pos_local[0] <= 0.0:
            return False, dist, pos_local[0], pos_local[1]
        elif self.mount_position == "rear" and pos_local[0] >= 0.0:
            return False, dist, pos_local[0], pos_local[1]

        # Measure relative azimuth angle from sensor bore-axis
        sensor_x = pos_local[0] if self.mount_position == "front" else -pos_local[0]
        angle = np.arctan2(pos_local[1], sensor_x)

        in_fov = abs(angle) <= self.half_fov_rad
        return in_fov, dist, pos_local[0], pos_local[1]

    def track_lead_vehicle(self, 
                           ego: EgoState,
                           obstacles: list, 
                           step: int, 
                           lane_corridor_width: float = 2.5,
                           target_offset: float = 0.0,
                           road_heading: float | None = None,
                           is_changing_lane: bool = False) -> tuple[float, float, float, object, float] | None:
        """
        Scans vehicles in Ego's FOV cone and tracks the closest lead target.

        Uses road_heading (if provided) for lane corridor alignment to prevent vehicle 
        orientation during diagonal maneuvers from misclassifying target lane traffic.

        Args:
            ego (EgoState): Current state of the Ego vehicle.
            obstacles (list): List of surrounding dynamic obstacle objects.
            step (int): Current simulation time step index.
            lane_corridor_width (float, optional): Total width of lane corridor to monitor. Defaults to 2.5.
            target_offset (float, optional): Lateral offset to target lane. Defaults to 0.0.
            road_heading (float | None, optional): Reference road angle in radians. Defaults to None.
            is_changing_lane (bool, optional): Whether Ego is currently executing a lane change. Defaults to False.

        Returns:
            tuple[float, float, float, object, float] | None: Tuple of (x, y, velocity, obstacle_id, x_local)
            for the lead vehicle, or None if no vehicle is tracked.
        """
        if self.mount_position != "front":
            return None

        closest_dist = self.range_max
        lead_target = None
        half_corridor = lane_corridor_width / 2.0

        # Reference orientation for lane projection (use road angle if turning)
        if road_heading is not None:
            u_road = np.array([np.cos(road_heading), np.sin(road_heading)])
            n_road = np.array([-np.sin(road_heading), np.cos(road_heading)])
        else:
            u_road = ego.heading_vector
            n_road = ego.normal_vector


        for obs in obstacles:
            st = obs.state_at_time(step)
            if st is None:
                continue

            ox, oy = st.position[0], st.position[1]
            
            # 1. Sensor FOV Check
            in_fov, dist, x_local, _ = self.is_in_fov(ego, ox, oy)
            if not in_fov:
                continue

            # 2. Road-Aligned Corridor Projection
            # dx, dy
            d_vec = st.position - ego.position
            # dx * cos_a + dy * sin_a
            long_road = np.dot(d_vec, u_road)
            # -dx * sin_a + dy * cos_a
            lat_road = np.dot(d_vec, n_road)
       
            if long_road > 0.0:  # Vehicle must be ahead along the road
                in_current_lane = abs(lat_road) <= half_corridor
                # Consider target lane vehicles if ego is changing lanes OR if a target offset is defined
                in_target_lane = is_changing_lane and (abs(lat_road - target_offset) <= half_corridor)

                if in_current_lane or in_target_lane:
                    if dist < closest_dist:
                        closest_dist = dist
                        target_v = float(getattr(st, 'velocity', 15.0))
                        lead_target = (ox, oy, target_v, obs.obstacle_id, x_local)

        return lead_target

    def is_adjacent_lane_clear(self,
                               ego: EgoState,
                               surrounding_obstacles: list,
                               step: int,
                               target_lane_offset: float,
                               safety_gap_front: float = 12.0,
                               safety_gap_rear: float = 10.0,
                               road_heading: float | None = None,
                               rear_radar: "RadarSensor | None" = None,
                               lane_tolerance: float = 1.8) -> bool:
        """
        Evaluates whether an adjacent lane target gap is clear using front and rear radars.

        Args:
            ego (EgoState): Current state of the Ego vehicle.
            surrounding_obstacles (list): List of dynamic obstacles in the scene.
            step (int): Current simulation time step index.
            target_lane_offset (float): Lateral offset to target lane (+ for left, - for right).
            safety_gap_front (float, optional): Longitudinal safety buffer ahead in meters. Defaults to 12.0.
            safety_gap_rear (float, optional): Longitudinal safety buffer behind in meters. Defaults to 10.0.
            road_heading (float | None, optional): Reference road angle in radians. Defaults to None.
            rear_radar (RadarSensor | None, optional): Rear-facing radar sensor instance. Defaults to None.
            lane_tolerance (float, optional): Half-width tolerance for lane boundary matching. Defaults to 1.8.

        Returns:
            bool: True if target adjacent lane safety corridor is completely clear.
        """

        if road_heading is not None:
            u_hat = np.array([np.cos(road_heading), np.sin(road_heading)])
            n_hat = np.array([-np.sin(road_heading), np.cos(road_heading)])
        else:
            u_hat = ego.heading_vector
            n_hat = ego.normal_vector

        for obs in surrounding_obstacles:
            st = obs.state_at_time(step)
            if st is None:
                continue

            ox, oy = st.position[0], st.position[1]

            # Obstacle must be detected by either front OR rear radar
            in_front_fov, _, _, _ = self.is_in_fov(ego, ox, oy)
            
            in_rear_fov = rear_radar.is_in_fov(ego, ox, oy)[0] if rear_radar is not None else False

            if not (in_front_fov or in_rear_fov):
                continue

            # Vector from Ego to Obstacle projected into Ego frame
            d_vec = st.position - ego.position
            longitudinal_dist = np.dot(d_vec, u_hat)
            lateral_dist = np.dot(d_vec, n_hat)

            # 1. Check if obstacle is inside or close to target lane
            is_in_target_lane = abs(lateral_dist - target_lane_offset) <= lane_tolerance
            # 2. Check if obstacle falls within longitudinal safety window
            is_in_safety_window = -safety_gap_rear <= longitudinal_dist <= safety_gap_front
            if is_in_target_lane and is_in_safety_window:
                    return False

        return True

    def get_detected_obstacle_ids(self,
                                  ego: EgoState,
                                  obstacles: list,
                                  step: int) -> set:
        """
        Gets the set of obstacle IDs that fall within this radar's FOV at the given step.

        Args:
            ego (EgoState): Current state of the Ego vehicle.
            obstacles (list): List of dynamic obstacle objects.
            step (int): Current simulation time step index.

        Returns:
            set: Set of detected obstacle IDs.
        """

        detected_ids = set()
        for obs in obstacles:
            st = obs.state_at_time(step)
            if st is None:
                continue
            ox, oy = st.position[0], st.position[1]
            in_fov, _, _, _ = self.is_in_fov(ego, ox, oy)
            if in_fov:
                detected_ids.add(obs.obstacle_id)
        return detected_ids
