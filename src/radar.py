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

    def track_lead_vehicle(self, ego_x: float, ego_y: float, ego_orient: float, 
                           obstacles: list, step: int, lane_corridor_width: float = 2.5) -> Optional[Tuple[float, float, float, object, float]]:
        """
        Scans vehicles in Ego's rotated FOV cone and tracks the closest in-path vehicle.
        
        Returns: (target_x, target_y, target_velocity, target_obstacle_id, x_local) or None
        """
        closest_dist = self.range_max
        lead_target = None

        for obs in obstacles:
            st = obs.state_at_time(step)
            if st is None:
                continue

            ox, oy = st.position[0], st.position[1]
            in_fov, dist, x_local, y_local = self.is_in_fov(ego_x, ego_y, ego_orient, ox, oy)

            # Filter for vehicles inside FOV cone AND within the current lane corridor
            if in_fov and abs(y_local) < (lane_corridor_width / 2.0):
                if dist < closest_dist:
                    closest_dist = dist
                    target_v = float(getattr(st, 'velocity', 15.0))
                    lead_target = (ox, oy, target_v, obs.obstacle_id, x_local)

        return lead_target
