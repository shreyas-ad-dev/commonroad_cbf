from typing import Any
import numpy as np
from src.base_sensor import BaseSensor
from src.ego_state import EgoState


class RadarSensor(BaseSensor):
    """
    Simulates a body-frame aligned Radar sensor with finite range and Field of View (FOV).

    Supports 'front' (+x_local) and 'rear' (-x_local) mounting orientations on the vehicle.
    Inherits coordinate transformations from BaseSensor and uses step-caching to avoid
    redundant FOV checks across controller, behavior planner, and visualizer calls.
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

        super().__init__(range_max=range_max, fov_deg=fov_deg)
        self.mount_position = mount_position

        # Stateful internal cache
        self._last_step: int | None = None
        self._scan_cache: dict[str, Any] = {
            "detected_ids": set(),
            "fov_data": {},  # Maps obs_id -> (in_fov, dist, x_local, y_local)
            "lead_target": None
        }

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
        x_local, y_local = pos_local[0], pos_local[1]
        dist = float(np.hypot(x_local, y_local))

        if dist > self.range_max:
            return False, dist, x_local, y_local

        # Longitudinal direction constraint based on mounting point
        if self.mount_position == "front" and x_local <= 0.0:
            return False, dist, x_local, y_local
        elif self.mount_position == "rear" and x_local >= 0.0:
            return False, dist, x_local, y_local

        # Measure relative azimuth angle from sensor bore-axis
        sensor_x = x_local if self.mount_position == "front" else -x_local
        angle = np.arctan2(y_local, sensor_x)

        in_fov = abs(angle) <= self.half_fov_rad
        return in_fov, dist, x_local, y_local

    def scan(self,
             ego: EgoState,
             obstacles: list,
             step: int) -> dict[str, Any]:
        """
        Executes single-pass FOV perception evaluations and updates step cache.

        Args:
            ego (EgoState): Current state of the Ego vehicle.
            obstacles (list): List of dynamic obstacle objects.
            step (int): Current simulation time step index.

        Returns:
            dict[str, Any]: Reference to internal scan cache containing:
                - 'detected_ids': set[int] of detected obstacle IDs.
                - 'fov_data': dict[int, tuple] mapping ID to (in_fov, dist, x_local, y_local).
        """
        if self._last_step != step:
            self._last_step = step
            detected_ids: set[int] = set()
            fov_data: dict[int, tuple[bool, float, float, float]] = {}

            for obs in obstacles:
                st = obs.state_at_time(step)
                if st is None:
                    continue

                ox, oy = st.position[0], st.position[1]
                in_fov, dist, x_local, y_local = self.is_in_fov(ego, ox, oy)

                fov_data[obs.obstacle_id] = (in_fov, dist, x_local, y_local)
                if in_fov:
                    detected_ids.add(obs.obstacle_id)

            self._scan_cache = {
                "detected_ids": detected_ids,
                "fov_data": fov_data,
                "lead_target": None
            }

        return self._scan_cache

    def get_detected_obstacle_ids(self,
                                  ego: EgoState,
                                  obstacles: list,
                                  step: int) -> set[int]:
        """Gets cached set of obstacle IDs that fall within this radar's FOV at the given step."""
        return self.scan(ego, obstacles, step)["detected_ids"]

    def track_lead_vehicle(self,
                           ego: EgoState,
                           obstacles: list,
                           step: int,
                           lane_corridor_width: float = 2.5,
                           target_offset: float = 0.0,
                           road_heading: float | None = None,
                           is_changing_lane: bool = False) -> tuple[float, float, float, int, float] | None:
        """
        Scans vehicles in Ego's FOV cone and tracks the closest lead target.

        Uses cached FOV evaluation results to eliminate redundant coordinate transformations.
        """
        if self.mount_position != "front":
            return None

        scan_res = self.scan(ego, obstacles, step)
        closest_dist = self.range_max
        lead_target = None
        half_corridor = lane_corridor_width / 2.0

        if road_heading is not None:
            u_road = np.array([np.cos(road_heading), np.sin(road_heading)])
            n_road = np.array([-np.sin(road_heading), np.cos(road_heading)])
        else:
            u_road = ego.heading_vector
            n_road = ego.normal_vector

        for obs in obstacles:
            st = obs.state_at_time(step)
            if st is None or obs.obstacle_id not in scan_res["fov_data"]:
                continue

            in_fov, dist, x_local, _ = scan_res["fov_data"][obs.obstacle_id]
            if not in_fov:
                continue

            # Road-aligned corridor projection
            d_vec = st.position - ego.position
            long_road = np.dot(d_vec, u_road)
            lat_road = np.dot(d_vec, n_road)

            if long_road > 0.0:  # Vehicle must be ahead along the road
                in_current_lane = abs(lat_road) <= half_corridor
                in_target_lane = is_changing_lane and (abs(lat_road - target_offset) <= half_corridor)

                if in_current_lane or in_target_lane:
                    if dist < closest_dist:
                        closest_dist = dist
                        target_v = float(getattr(st, 'velocity', 15.0))
                        lead_target = (st.position[0], st.position[1], target_v, obs.obstacle_id, x_local)

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
        """Evaluates whether an adjacent lane target gap is clear using front and rear radars."""
        if road_heading is not None:
            u_hat = np.array([np.cos(road_heading), np.sin(road_heading)])
            n_hat = np.array([-np.sin(road_heading), np.cos(road_heading)])
        else:
            u_hat = ego.heading_vector
            n_hat = ego.normal_vector

        front_scan = self.scan(ego, surrounding_obstacles, step)
        rear_scan = rear_radar.scan(ego, surrounding_obstacles, step) if rear_radar is not None else None

        for obs in surrounding_obstacles:
            st = obs.state_at_time(step)
            if st is None:
                continue

            obs_id = obs.obstacle_id
            in_front_fov = front_scan["fov_data"].get(obs_id, (False,))[0]
            in_rear_fov = rear_scan["fov_data"].get(obs_id, (False,))[0] if rear_scan else False

            if not (in_front_fov or in_rear_fov):
                continue

            d_vec = st.position - ego.position
            longitudinal_dist = np.dot(d_vec, u_hat)
            lateral_dist = np.dot(d_vec, n_hat)

            is_in_target_lane = abs(lateral_dist - target_lane_offset) <= lane_tolerance
            is_in_safety_window = -safety_gap_rear <= longitudinal_dist <= safety_gap_front
            
            if is_in_target_lane and is_in_safety_window:
                return False

        return True
