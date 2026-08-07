import numpy as np
from typing import List, Tuple, Optional

class RadarSensor:
    """
    Simulates a forward-facing Radar sensor with finite range and Field of View (FOV).
    """
    def __init__(self, range_max: float = 80.0, fov_deg: float = 60.0):
        """
        :param range_max: Maximum detection range in meters
        :param fov_deg: Total azimuth field of view in degrees (+/- fov_deg/2 from centerline)
        """
        self.range_max = range_max
        self.fov_rad = np.radians(fov_deg / 2.0)

    def is_in_fov(self, ego_x: float, ego_y: float, ego_orient: float, obs_x: float, obs_y: float) -> Tuple[bool, float, float]:
        """
        Checks if a target coordinate is within the Radar's cone.
        Returns: (in_fov, relative_range, relative_bearing_rad)
        """
        dx = obs_x - ego_x
        dy = obs_y - ego_y
        range_dist = np.hypot(dx, dy)

        if range_dist > self.range_max or range_dist == 0:
            return False, range_dist, 0.0

        # Bearing relative to Ego's heading angle
        world_angle = np.arctan2(dy, dx)
        bearing = world_angle - ego_orient
        
        # Normalize bearing to [-pi, pi]
        bearing = (bearing + np.pi) % (2 * np.pi) - np.pi

        in_fov = abs(bearing) <= self.fov_rad
        return in_fov, range_dist, bearing

    def track_lead_vehicle(self, ego_x: float, ego_y: float, ego_orient: float, 
                           surrounding_obstacles: list, step_idx: int) -> Optional[Tuple[float, float, float, float]]:
        """
        Scans all surrounding obstacles, filters those inside the Radar FOV, 
        and selects the primary lead vehicle trajectory threat.
        
        Returns: (target_x, target_y, target_vx, target_obstacle_id) or None
        """
        lead_target = None
        min_range = float('inf')

        for obs in surrounding_obstacles:
            st = obs.state_at_time(step_idx)
            if st is None:
                continue

            ox, oy = st.position[0], st.position[1]
            ov = getattr(st, 'velocity', 0.0)

            in_fov, r_dist, bearing = self.is_in_fov(ego_x, ego_y, ego_orient, ox, oy)

            # Filter for forward-facing targets in sensor cone
            if in_fov and ox > ego_x:
                if r_dist < min_range:
                    min_range = r_dist
                    lead_target = (ox, oy, ov, obs.obstacle_id)

        return lead_target
