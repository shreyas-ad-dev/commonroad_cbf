# src/ultrasonic.py
import numpy as np
from typing import Tuple, Optional, Set

class SideUltrasonicSensor:
    """
    Simulates short-range side-facing ultrasonic sensors (USS) for blind-spot monitoring.
    Mount positions: 'left' (+90 deg) or 'right' (-90 deg relative to ego body frame).
    """
    def __init__(self, range_max: float = 8.0, fov_deg: float = 100.0, side: str = "left"):
        if side not in ["left", "right"]:
            raise ValueError("side must be either 'left' or 'right'")
        self.range_max = range_max
        self.fov_deg = fov_deg
        self.side = side
        self.half_fov_rad = np.radians(fov_deg / 2.0)

    def is_in_fov(self, ego_x: float, ego_y: float, ego_orient: float, obs_x: float, obs_y: float) -> Tuple[bool, float]:
        dx = obs_x - ego_x
        dy = obs_y - ego_y
        cos_a, sin_a = np.cos(ego_orient), np.sin(ego_orient)

        x_local = dx * cos_a + dy * sin_a
        y_local = -dx * sin_a + dy * cos_a
        dist = np.hypot(x_local, y_local)

        if dist > self.range_max:
            return False, dist

        # Left sensor expects y_local > 0; Right sensor expects y_local < 0
        if self.side == "left" and y_local <= 0.0:
            return False, dist
        if self.side == "right" and y_local >= 0.0:
            return False, dist

        # Measure relative angle from side bore-axis
        sensor_y = y_local if self.side == "left" else -y_local
        angle = np.arctan2(x_local, sensor_y)

        return abs(angle) <= self.half_fov_rad, dist

    def get_detected_obstacle_ids(self, ego_x: float, ego_y: float, ego_orient: float, obstacles: list, step: int) -> Set:
        """Returns a set of obstacle IDs currently inside this sensor's blind-spot FOV."""
        detected_ids = set()
        for obs in obstacles:
            st = obs.state_at_time(step)
            if st is None:
                continue
            ox, oy = st.position[0], st.position[1]
            in_fov, _ = self.is_in_fov(ego_x, ego_y, ego_orient, ox, oy)
            if in_fov:
                detected_ids.add(obs.obstacle_id)
        return detected_ids

    def is_adjacent_lane_clear(self, ego_x: float, ego_y: float, ego_orient: float, obstacles: list, step: int, target_offset: float) -> bool:
        """
            Checks if the adjacent lane in the target direction is free of blind-spot obstacles.
        
            target_offset > 0: Left Lane Change
            target_offset < 0: Right Lane Change
        """
        # Ignore check if target offset direction does not match sensor mounting side

        if (target_offset > 0 and self.side != "left") or (target_offset < 0 and self.side != "right"):
            return True

        # Query tracked obstacle IDs in USS field of view
        detected_ids = self.get_detected_obstacle_ids(ego_x, ego_y, ego_orient, obstacles, step)
        
        # Lane is clear only if zero obstacles occupy the side blind spot
        return len(detected_ids) == 0
