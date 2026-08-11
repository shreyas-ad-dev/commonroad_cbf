from typing import List, Tuple, Optional
import numpy as np

class RadarSensor:
    """
    Simulates a body-frame aligned Radar sensor with finite range and Field of View (FOV).
    """
    def __init__(self, range_max: float = 70.0, fov_deg: float = 60.0):
        """
        :param range_max: Maximum detection range in meters
        :param fov_deg: Total azimuth field of view in degrees (+/- fov_deg/2 from centerline)
        """
        self.range_max = range_max
        self.fov_deg = fov_deg
        self.half_fov_rad = np.radians(fov_deg / 2.0)

    def to_local_frame(self, ego_x: float, ego_y: float, ego_orient: float, obs_x: float, obs_y: float) -> Tuple[float, float]:
        """
        Transforms world coordinates into Ego's body-fixed local frame:
        x_local: Longitudinal distance along Ego's heading direction.
        y_local: Lateral distance perpendicular to Ego's heading direction.
        """
        dx = obs_x - ego_x
        dy = obs_y - ego_y
        cos_a, sin_a = np.cos(ego_orient), np.sin(ego_orient)

        x_local = dx * cos_a + dy * sin_a
        y_local = -dx * sin_a + dy * cos_a
        return x_local, y_local

    def is_in_fov(self, ego_x: float, ego_y: float, ego_orient: float, obs_x: float, obs_y: float) -> Tuple[bool, float, float, float]:
        """
        Checks if a coordinate is within the Radar's field of view cone.
        Returns: (in_fov, distance, x_local, y_local)
        """
        x_local, y_local = self.to_local_frame(ego_x, ego_y, ego_orient, obs_x, obs_y)
        dist = np.hypot(x_local, y_local)

        if dist > self.range_max or x_local <= 0.0:
            return False, dist, x_local, y_local

        angle = np.arctan2(y_local, x_local)
        in_fov = abs(angle) <= self.half_fov_rad
        return in_fov, dist, x_local, y_local

    def track_lead_vehicle(
        self, 
        ego_x: float, 
        ego_y: float, 
        ego_orient: float, 
        obstacles: list, 
        step: int, 
        lane_corridor_width: float = 2.5,
        target_offset: float = 0.0,
        road_heading: Optional[float] = None
    ) -> Optional[Tuple[float, float, float, object, float]]:
        """
        Scans vehicles in Ego's rotated FOV cone and tracks the closest lead vehicle.
        Uses road_heading (if provided) for lane corridor matching so diagonal vehicle 
        orientation during lane changes doesn't misclassify target lane vehicles.
        """
        closest_dist = self.range_max
        lead_target = None
        half_corridor = lane_corridor_width / 2.0

        # Reference orientation for lane projection (use road angle if turning)
        ref_heading = road_heading if road_heading is not None else ego_orient
        u_road = np.array([np.cos(ref_heading), np.sin(ref_heading)])
        n_road = np.array([-np.sin(ref_heading), np.cos(ref_heading)])

        for obs in obstacles:
            st = obs.state_at_time(step)
            if st is None:
                continue

            ox, oy = st.position[0], st.position[1]
            
            # 1. Sensor FOV Check (Must be visible to physical radar cone)
            in_fov, dist, x_local, y_local = self.is_in_fov(ego_x, ego_y, ego_orient, ox, oy)
            if not in_fov:
                continue

            # 2. Road-Aligned Corridor Projection
            dx, dy = ox - ego_x, oy - ego_y
            long_road = dx * u_road[0] + dy * u_road[1]
            lat_road = dx * n_road[0] + dy * n_road[1]

            if long_road > 0.0:  # Vehicle must be ahead along the road
                in_current_lane = abs(lat_road) <= half_corridor
                in_target_lane = abs(lat_road - target_offset) <= half_corridor

                if in_current_lane or in_target_lane:
                    if dist < closest_dist:
                        closest_dist = dist
                        target_v = float(getattr(st, 'velocity', 15.0))
                        lead_target = (ox, oy, target_v, obs.obstacle_id, x_local)

        return lead_target


    def is_adjacent_lane_clear(
        self,
        ego_x: float,
        ego_y: float,
        ego_yaw: float,
        surrounding_obstacles: list,
        step: int,
        target_lane_offset: float,
        safety_gap_front: float = 12.0,
        safety_gap_rear: float = 10.0,
    ) -> bool:
        """
        Checks if the target adjacent lane is free of obstacles within a 
        longitudinal safety corridor around the Ego vehicle.
        
        target_lane_offset: Lateral distance to target lane center (+ left, - right)
        safety_gap_front: Minimum safe distance ahead in target lane (meters)
        safety_gap_rear: Minimum safe distance behind in target lane (meters)
        """
        # Direction vectors in Ego frame
        u_hat = np.array([np.cos(ego_yaw), np.sin(ego_yaw)])   # Forward
        n_hat = np.array([-np.sin(ego_yaw), np.cos(ego_yaw)])  # Perpendicular (Left)

        for obs in surrounding_obstacles:
            st = obs.state_at_time(step)
            if st is None:
                continue

            # Vector from Ego to Obstacle
            dx = st.position[0] - ego_x
            dy = st.position[1] - ego_y

            # Project into Ego local frame
            longitudinal_dist = dx * u_hat[0] + dy * u_hat[1]
            lateral_dist = dx * n_hat[0] + dy * n_hat[1]

            # 1. Check if obstacle is inside or close to the target lane corridor
            lane_tolerance = 1.8  # ~half a lane width
            if abs(lateral_dist - target_lane_offset) <= lane_tolerance:
                # 2. Check if obstacle falls within our longitudinal safety window
                if -safety_gap_rear <= longitudinal_dist <= safety_gap_front:
                    return False  # Target lane is blocked!

        return True  # Safe to initiate lane change
