from typing import Any

import numpy as np

from src.base_sensor import BaseSensor
from src.ego_state import EgoState


class SideUltrasonicSensor(BaseSensor):
    """
    Simulates short-range side-facing ultrasonic sensors (USS) for blind-spot monitoring.

    Mount positions can be 'left' (+90 deg) or 'right' (-90 deg) relative to the Ego body frame.
    Uses stateful step-caching to guarantee single-pass FOV evaluations per frame.
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

        super().__init__(range_max=range_max, fov_deg=fov_deg)
        self.side = side

        # Stateful internal cache
        self._last_step: int | None = None
        self._scan_cache: dict[str, Any] = {
            "detected_ids": set(),
            "min_distances": {},
            "target_clearance": {}
        }

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
        eval_data = self.get_obstacle_center_and_corners_in_local(ego, obstacle, step)
        if eval_data is None:
            return False, float('inf')

        _, local_points = eval_data
        min_dist = float('inf')
        any_corner_in_fov = False

        for pt in local_points:
            x_local, y_local = pt[0], pt[1]
            dist = float(np.hypot(x_local, y_local))

            min_dist = min(min_dist, dist)

            is_side_aligned= (self.side == "left" and y_local > 0.0) or (self.side == "right" and y_local < 0.0)
            if dist <= self.range_max and is_side_aligned:
                sensor_y = y_local if self.side == "left" else -y_local
                angle = np.arctan2(x_local, sensor_y)
                if abs(angle) <= self.half_fov_rad:
                    any_corner_in_fov = True

        return any_corner_in_fov, min_dist

    def scan(self,
             ego: EgoState,
             obstacles: list,
             step: int,
             target_offset: float = 0.0) -> dict[str, Any]:
        """
        Executes perception checks and updates instance-level cache if step has changed.

        Args:
            ego (EgoState): Current state of the Ego vehicle.
            obstacles (list): List of dynamic obstacle objects.
            step (int): Current simulation time step index.
            target_offset (float, optional): Lateral lane offset to assess clearance. Defaults to 0.0.

        Returns:
            dict[str, Any]: Reference to the internal sensor scan state cache.
        """
        # If new step, perform primary scan and wipe stale cache
        if self._last_step != step:
            self._last_step = step
            detected_ids: set[int] = set()
            min_distances: dict[int, float] = {}

            for obs in obstacles:
                in_fov, min_dist = self.is_in_fov(ego, obs, step)
                if in_fov:
                    detected_ids.add(obs.obstacle_id)
                    min_distances[obs.obstacle_id] = min_dist

            self._scan_cache = {
                "detected_ids": detected_ids,
                "min_distances": min_distances,
                "target_clearance": {}
            }

        # Sub-check: Memoize clearance evaluation for target_offset in current step
        if target_offset not in self._scan_cache["target_clearance"]:
            if (target_offset > 0 and self.side != "left") or (target_offset < 0 and self.side != "right"):
                is_clear = True
            else:
                is_clear = len(self._scan_cache["detected_ids"]) == 0

            self._scan_cache["target_clearance"][target_offset] = is_clear

        return self._scan_cache

    def get_detected_obstacle_ids(self,
                                  ego: EgoState,
                                  obstacles: list,
                                  step: int) -> set[int]:
        """Returns cached set of detected obstacle IDs for current step."""
        return self.scan(ego, obstacles, step)["detected_ids"]

    def is_adjacent_lane_clear(self,
                               ego: EgoState,
                               obstacles: list,
                               step: int,
                               target_offset: float) -> bool:
        """Returns cached lane clearance bool for given target offset and current step."""
        cache = self.scan(ego, obstacles, step, target_offset=target_offset)
        return cache["target_clearance"][target_offset]
