# src/radar.py
#from typing import List, Tuple, Optional
import numpy as np
from src.ego_state import EgoState


class RadarSensor:
    """
    Simulates a body-frame aligned Radar sensor with finite range and Field of View (FOV).
    Supports 'front' (+x_local) and 'rear' (-x_local) mounting orientations.
    """
    def __init__(
        self,
        range_max: float = 70.0,
        fov_deg: float = 60.0,
        mount_position: str = "front"
        ):
        """
        :param range_max: Maximum detection range in meters
        :param fov_deg: Total azimuth field of view in degrees (+/- fov_deg/2)
        :param mount_position: Orientation relative to vehicle heading ('front' or 'rear')
        """

        if mount_position not in ["front", "rear"]:
            raise ValueError("mount_position must be either 'front' or 'rear'")

        self.range_max = range_max
        self.fov_deg = fov_deg
        self.mount_position = mount_position
        self.half_fov_rad = np.radians(fov_deg / 2.0)

    def to_local_frame(
        self,
        ego: EgoState,
        obs_x: float,
        obs_y: float
        ) -> np.ndarray:
        """
        Transforms world coordinates into Ego's body-fixed local frame:
        x_local: Longitudinal distance along Ego's heading direction.
        y_local: Lateral distance perpendicular to Ego's heading direction.
        """

        # [dx, dy]
        d_vec = np.array([obs_x, obs_y]) - ego.position

        # dx * cos_a + dy * sin_a
        x_local = np.dot(d_vec, ego.heading_vector)
        # -dx * sin_a + dy * cos_a
        y_local = np.dot(d_vec, ego.normal_vector) 

        return np.array([x_local, y_local])

    def is_in_fov(
            self,
            ego: EgoState,
            obs_x: float,
            obs_y: float) -> tuple[bool, float, float, float]:
        """
        Checks if a coordinate is within the Radar's field of view cone.
        Returns: (in_fov, distance, x_local, y_local)
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

    def track_lead_vehicle(
            self, 
            ego: EgoState,
            obstacles: list, 
            step: int, 
            lane_corridor_width: float = 2.5,
            target_offset: float = 0.0,
            road_heading: float | None = None,
            is_changing_lane: bool = False
            ) -> tuple[float, float, float, object, float] | None:
        """
        Scans vehicles in Ego's rotated FOV cone and tracks the closest lead vehicle.
        Uses road_heading (if provided) for lane corridor matching so diagonal vehicle 
        orientation during lane changes doesn't misclassify target lane vehicles.
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
            #long_road = dx * u_road[0] + dy * u_road[1]
            #lat_road = dx * n_road[0] + dy * n_road[1]

            if long_road > 0.0:  # Vehicle must be ahead along the road
                in_current_lane = abs(lat_road) <= half_corridor
                #in_target_lane = is_changing_lane and abs(lat_road - target_offset) <= half_corridor
                # Consider target lane vehicles if ego is changing lanes OR if a target offset is defined
                in_target_lane = (is_changing_lane or abs(target_offset) > 0.0) and (abs(lat_road - target_offset) <= half_corridor)

                if in_current_lane or in_target_lane:
                    if dist < closest_dist:
                        closest_dist = dist
                        target_v = float(getattr(st, 'velocity', 15.0))
                        lead_target = (ox, oy, target_v, obs.obstacle_id, x_local)

        return lead_target

    def is_adjacent_lane_clear(
            self,
            ego: EgoState,
            surrounding_obstacles: list,
            step: int,
            target_lane_offset: float,
            safety_gap_front: float = 12.0,
            safety_gap_rear: float = 10.0,
            road_heading: float | None = None,
            rear_radar: "RadarSensor | None" = None,
            lane_tolerance: float = 1.8,
            ) -> bool:
        """
        Checks if the target adjacent lane is free of obstacles within a 
        longitudinal safety corridor around the Ego vehicle using front and rear radars.
        """
        if road_heading is not None:
            u_hat = np.array([np.cos(road_heading), np.sin(road_heading)])
            n_hat = np.array([-np.sin(road_heading), np.cos(road_heading)])
        else:
            u_hat = ego.heading_vector
            n_hat = ego.normal_vector
        #ref_heading = road_heading if road_heading is not None else ego.orientation
        #u_hat = np.array([np.cos(ref_heading), np.sin(ref_heading)])   # Forward
        #n_hat = np.array([-np.sin(ref_heading), np.cos(ref_heading)])  # Perpendicular (Left)

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

    def get_detected_obstacle_ids(
            self,
            ego: EgoState,
            obstacles: list,
            step: int
            ) -> set:
        """Returns a set of obstacle IDs that fall within this radar's FOV at the current step."""
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
