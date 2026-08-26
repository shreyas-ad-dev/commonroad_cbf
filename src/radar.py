from typing import Any

import numpy as np

from src.base_sensor import BaseSensor
from src.ego_state import EgoState


class RadarSensor(BaseSensor):
    """
    Simulates a body-frame aligned Radar sensor with finite range and Field of View (FOV).

    Supports 'front' (+x_local) and 'rear' (-x_local) mounting orientations on the vehicle.
    Evaluates FOV containment across all bounding box corners and center of dynamic obstacles.
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
            "fov_data": {},  # Maps obs_id -> (in_fov, min_dist, center_x_local, center_y_local)
            "lead_target": None
        }

    def is_in_fov(self,
                  ego: EgoState,
                  obstacle: object,
                  step: int) -> tuple[bool, float, float, float]:
        """
        Checks if an obstacle's center or any of its bounding box corners fall within the Radar's FOV.

        Args:
            ego (EgoState): Current state of the Ego vehicle.
            obstacle (object): Dynamic obstacle instance to evaluate.
            step (int): Current simulation time step index.

        Returns:
            tuple[bool, float, float, float]: A tuple containing:
                - any_corner_in_fov (bool): True if any point falls inside detection cone.
                - min_dist (float): Minimum Euclidean distance across all points.
                - center_x_local (float): Obstacle center longitudinal offset in Ego frame.
                - center_y_local (float): Obstacle center lateral offset in Ego frame.
        """
        eval_data = self.get_obstacle_center_and_corners_in_local(ego, obstacle, step)
        if eval_data is None:
            return False, float('inf'), 0.0, 0.0

        center_local, local_points = eval_data
        center_x_local, center_y_local = center_local[0], center_local[1]

        min_dist = float('inf')
        any_corner_in_fov = False

        for pt in local_points:
            x_local, y_local = pt[0], pt[1]
            dist = float(np.hypot(x_local, y_local))

            min_dist = min(min_dist, dist)

            if dist <= self.range_max:
                # Direction constraint based on mounting orientation
                is_valid_direction = (
                    (self.mount_position == "front" and x_local > 0.0) or
                    (self.mount_position == "rear" and x_local < 0.0)
                )

                if is_valid_direction:
                    sensor_x = x_local if self.mount_position == "front" else -x_local
                    angle = np.arctan2(y_local, sensor_x)
                    if abs(angle) <= self.half_fov_rad:
                        any_corner_in_fov = True

        return any_corner_in_fov, min_dist, center_x_local, center_y_local

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
                - 'fov_data': dict[int, tuple] mapping ID to (in_fov, min_dist, x_local, y_local).
        """
        if self._last_step != step:
            self._last_step = step
            detected_ids: set[int] = set()
            fov_data: dict[int, tuple[bool, float, float, float]] = {}

            for obs in obstacles:
                in_fov, min_dist, center_x_local, center_y_local = self.is_in_fov(ego, obs, step)

                fov_data[obs.obstacle_id] = (in_fov, min_dist, center_x_local, center_y_local)
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
                           ) -> tuple[float, float, float, int, float] | None:
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
        u_road, n_road = ego.road_frame_vectors

        for obs in obstacles:
            st = obs.state_at_time(step)
            if st is None or obs.obstacle_id not in scan_res["fov_data"]:
                continue

            in_fov, min_dist, center_x_local, _ = scan_res["fov_data"][obs.obstacle_id]
            if not in_fov:
                continue

            # Road-aligned corridor projection using center position
            d_vec = st.position - ego.position
            long_road = np.dot(d_vec, u_road)
            lat_road = np.dot(d_vec, n_road)

            if long_road > 0.0:  # Vehicle must be ahead along the road
                in_corridor = abs(lat_road - target_offset) <= half_corridor

                if in_corridor and (min_dist < closest_dist):
                    closest_dist = min_dist
                    target_v = float(getattr(st, 'velocity', 15.0))
                    # Calculate bumper-to-bumper longitudinal distance offset
                    obs_length = getattr(obs.obstacle_shape, 'length', 4.5)
                    ego_length = ego.length
                    bumper_x_local = max(0.1, center_x_local - (obs_length / 2.0) - (ego_length / 2.0))

                    lead_target = (st.position[0], st.position[1], target_v, obs.obstacle_id, bumper_x_local)
        return lead_target

    def is_adjacent_lane_clear(self,
                               ego: EgoState,
                               surrounding_obstacles: list,
                               step: int,
                               target_lane_offset: float,
                               safety_gap_front: float = 12.0,
                               safety_gap_rear: float = 10.0,
                               rear_radar: "RadarSensor | None" = None,
                               lane_tolerance: float = 1.8) -> bool:
        """Evaluates whether an adjacent lane target gap is clear using front and rear radars."""

        u_hat, n_hat = ego.road_frame_vectors

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
