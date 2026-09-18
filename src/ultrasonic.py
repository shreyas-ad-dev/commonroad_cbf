from dataclasses import dataclass
from typing import Any

import numpy as np
from shapely.geometry import LineString
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.ops import unary_union

from src.base_sensor import BaseSensor, SensorOcclusionData
from src.ego_state import EgoState, get_car_polygon
from src.tracker import Detection


class SideUltrasonicSensor(BaseSensor):
    """
    Simulates short-range side-facing ultrasonic sensors (USS) for blind-spot monitoring.

    Mount positions can be 'left' (+90 deg) or 'right' (-90 deg) relative to the Ego body frame.
    Uses stateful step-caching to guarantee single-pass FOV evaluations per frame.

    Attributes:
        side (str): Sensor mounting side ('left' or 'right').
        range_max (float): Maximum detection range in meters.
        fov_deg (float): Total field of view in degrees.
    """
    
    def __init__(self,
                 range_max: float = 8.0,
                 fov_deg: float = 100.0,
                 side: str = "left",
                 noise_std: float = 0.1,
                 ray_count: int = 30):
        """
        Initializes the SideUltrasonicSensor instance.

        Args:
            range_max (float, optional): Maximum detection range in meters. Defaults to 8.0.
            fov_deg (float, optional): Total field of view in degrees. Defaults to 100.0.
            side (str, optional): Sensor side mounting ('left' or 'right'). Defaults to "left".
            noise_std (float, optional): Gaussian noise standard deviation. Defaults to 0.1.

        Raises:
            ValueError: If side is not 'left' or 'right'.
        """
        if side not in ["left", "right"]:
            raise ValueError("side must be either 'left' or 'right'")

        super().__init__(range_max=range_max, fov_deg=fov_deg)
        self.side = side
        self.sensor_id = f"uss_{side}"
        self.noise_std = noise_std
        self.ray_count = ray_count
        self.R = np.eye(2) * (noise_std ** 2)

        # Stateful internal cache
        self._last_step: int | None = None
        self._scan_cache: dict[str, Any] = {
            "detected_ids": set(),
            "min_distances": {},
            "detections": [],
            "target_clearance": {}
        }


    def _get_sensor_transform(self, ego: EgoState) -> tuple[np.ndarray, float]:
        """Calculates world frame position and heading for the side ultrasonic sensor."""
        sensor_pos = ego.position
        heading_deg = np.degrees(ego.orientation)
        sensor_heading_deg = heading_deg + 90.0 if self.side == "left" else heading_deg - 90.0
        return sensor_pos, sensor_heading_deg

    def _build_fov_wedge(self, sensor_pos: np.ndarray, sensor_heading_deg: float) -> ShapelyPolygon:
        """Constructs Shapely polygon FOV wedge for the ultrasonic sensor."""
        t1 = sensor_heading_deg - (self.fov_deg / 2.0)
        t2 = sensor_heading_deg + (self.fov_deg / 2.0)
        angles = np.radians(np.linspace(t1, t2, self.ray_count))

        arc_pts = [
            (sensor_pos[0] + self.range_max * np.cos(a), sensor_pos[1] + self.range_max * np.sin(a))
            for a in angles
        ]
        return ShapelyPolygon([tuple(sensor_pos)] + arc_pts + [tuple(sensor_pos)])

    def _compute_occlusion_shadow(self, sensor_pos: np.ndarray, obs_poly: ShapelyPolygon) -> ShapelyPolygon | None:
        """Computes projection shadow polygon cast behind an obstacle."""
        pts = np.array(obs_poly.exterior.coords)[:-1]
        if len(pts) == 0:
            return None

        angles = [np.arctan2(p[1] - sensor_pos[1], p[0] - sensor_pos[0]) for p in pts]
        min_idx, max_idx = np.argmin(angles), np.argmax(angles)

        if angles[max_idx] - angles[min_idx] > np.pi:
            pos_angles = [a if a >= 0 else a + 2 * np.pi for a in angles]
            min_idx, max_idx = np.argmin(pos_angles), np.argmax(pos_angles)

        p1, p2 = pts[min_idx], pts[max_idx]

        proj_factor = self.range_max * 2.0
        v1 = (p1 - sensor_pos) / np.linalg.norm(p1 - sensor_pos)
        v2 = (p2 - sensor_pos) / np.linalg.norm(p2 - sensor_pos)

        p1_proj = p1 + v1 * proj_factor
        p2_proj = p2 + v2 * proj_factor

        return ShapelyPolygon([p1, p2, p2_proj, p1_proj])
    
  #  def is_in_fov(self,
  #                ego: EgoState,
  #                obstacle: object,
  #                step: int) -> tuple[bool, float, float, float]:
  #      """
  #      Evaluates whether an obstacle's center or bounding box corners fall within the sensor FOV cone.

  #      Returns:
  #          tuple[bool, float, float, float]: (in_fov, min_dist, center_x_local, center_y_local)
  #      """
  #      eval_data = self.get_obstacle_center_and_corners_in_local(ego, obstacle, step)
  #      if eval_data is None:
  #          return False, float('inf'), 0.0, 0.0

  #      center_local, local_points = eval_data
  #      center_x_local, center_y_local = center_local[0], center_local[1]

  #      min_dist = float('inf')
  #      any_corner_in_fov = False

  #      for pt in local_points:
  #          x_local, y_local = pt[0], pt[1]
  #          dist = float(np.hypot(x_local, y_local))

  #          min_dist = min(min_dist, dist)

  #          is_side_aligned = (self.side == "left" and y_local > 0.0) or (self.side == "right" and y_local < 0.0)
  #          if dist <= self.range_max and is_side_aligned:
  #              sensor_y = y_local if self.side == "left" else -y_local
  #              angle = np.arctan2(x_local, sensor_y)
  #              if abs(angle) <= self.half_fov_rad:
  #                  any_corner_in_fov = True

  #      return any_corner_in_fov, min_dist, center_x_local, center_y_local

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
            detections: list[Detection] = []
            fov_data: dict[int, SensorOcclusionData] = {}

            timestamp = step * 0.1

            sensor_pos, sensor_heading = self._get_sensor_transform(ego)
            fov_wedge = self._build_fov_wedge(sensor_pos, sensor_heading)

            valid_obstacles = []
            for obs in obstacles:
                #in_fov, min_dist, center_x_local, center_y_local= self.is_in_fov(ego, obs, step)
                #if in_fov:
                    #detected_ids.add(obs.obstacle_id)
                    #min_distances[obs.obstacle_id] = min_dist
                eval_data = self.get_obstacle_center_and_corners_in_local(ego, obs, step)
                st = obs.state_at_time(step)
                if eval_data is None or st is None:
                    continue

                center_local, local_points = eval_data
                obs_length = getattr(obs.obstacle_shape, 'length', getattr(st, 'length', 4.5))
                obs_width = getattr(obs.obstacle_shape, 'width', getattr(st, 'width', 2.0))
                obs_yaw = getattr(st, 'orientation', getattr(st, 'yaw', 0.0))

                obs_poly, _ = get_car_polygon(
                    x=st.position[0],
                    y=st.position[1],
                    orientation=obs_yaw,
                    length=obs_length,
                    width=obs_width
                )

                if obs_poly.intersects(fov_wedge):
                    dist_to_sensor = np.linalg.norm(st.position - sensor_pos)
                    valid_obstacles.append({
                        "obs": obs,
                        "st": st,
                        "poly": obs_poly,
                        "dist": dist_to_sensor,
                        "center_local": center_local,
                        "local_points": local_points
                    })

            valid_obstacles.sort(key=lambda item: item["dist"])
            accumulated_shadows = []

            for item in valid_obstacles:
                obs = item["obs"]
                obs_poly = item["poly"]
                center_local = item["center_local"]
                local_points = item["local_points"]
                center_x_local, center_y_local = center_local[0], center_local[1]

                if accumulated_shadows:
                    combined_shadow = unary_union(accumulated_shadows)
                    effective_fov = fov_wedge.difference(combined_shadow)
                else:
                    effective_fov = fov_wedge

                if effective_fov.is_empty or not obs_poly.intersects(effective_fov):
                    continue

                min_dist = min(float(np.hypot(pt[0], pt[1])) for pt in local_points)
                detected_ids.add(obs.obstacle_id)

                visible_segments = []
                pts = np.array(obs_poly.exterior.coords)[:-1]
                num_pts = len(pts)

                for i in range(num_pts):
                    p1, p2 = pts[i], pts[(i + 1) % num_pts]
                    edge = p2 - p1
                    normal = np.array([edge[1], -edge[0]])

                    if np.dot(normal, sensor_pos - p1) > 0:
                        seg = LineString([p1, p2])
                        if seg.intersects(effective_fov):
                            intersection = seg.intersection(effective_fov)
                            if not intersection.is_empty:
                                geoms = intersection.geoms if hasattr(intersection, 'geoms') else [intersection]
                                for g in geoms:
                                    if g.geom_type in ['LineString', 'LinearRing']:
                                        visible_segments.append(np.array(g.coords))

                fov_data[obs.obstacle_id] = SensorOcclusionData(
                    obstacle_id=obs.obstacle_id,
                    in_fov=True,
                    min_dist=min_dist,
                    center_x_local=center_x_local,
                    center_y_local=center_y_local,
                    visible_segments=visible_segments
                )

                shadow_poly = self._compute_occlusion_shadow(sensor_pos, obs_poly)
                if shadow_poly and shadow_poly.is_valid:
                    accumulated_shadows.append(shadow_poly)

                # Apply measurement noise in local frame
                noisy_x_local = center_x_local + np.random.normal(0.0, self.noise_std)
                noisy_y_local = center_y_local + np.random.normal(0.0, self.noise_std)

                # Transform noisy measurement back to global coordinate frame
                cos_yaw = np.cos(ego.orientation)
                sin_yaw = np.sin(ego.orientation)
                global_x = ego.x + (noisy_x_local * cos_yaw - noisy_y_local * sin_yaw)
                global_y = ego.y + (noisy_x_local * sin_yaw + noisy_y_local * cos_yaw)

                detections.append(
                    Detection(
                        sensor_id=self.sensor_id,
                        timestamp=timestamp,
                        z=np.array([global_x, global_y], dtype=np.float64),
                        obstacle_id=obs.obstacle_id,
                        R=self.R
                    )
                )

            visible_fov_geometry = fov_wedge
            if accumulated_shadows:
                combined_shadows = unary_union(accumulated_shadows)
                visible_fov_geometry = fov_wedge.difference(combined_shadows)
            self._scan_cache = {
                "detected_ids": detected_ids,
                "min_distances": min_distances,
                "detections": detections,
                "fov_data": fov_data,
                "target_clearance": {},
                "visible_fov": visible_fov_geometry
            }

        # Sub-check: Memoize clearance evaluation for target_offset in current step
        if target_offset not in self._scan_cache["target_clearance"]:
            if (target_offset > 0 and self.side != "left") or (target_offset < 0 and self.side != "right"):
                is_clear = True
            else:
                is_clear = len(self._scan_cache["detected_ids"]) == 0

            self._scan_cache["target_clearance"][target_offset] = is_clear

        return self._scan_cache

    def get_detections(self,
                       ego: EgoState,
                       obstacles: list,
                       step: int) -> list[Detection]:
        """Gets cached list of Detection objects for MOT tracking pipeline."""
        return self.scan(ego, obstacles, step)["detections"]
    
    def get_detected_obstacle_ids(self,
                                  ego: EgoState,
                                  obstacles: list,
                                  step: int) -> set[int]:
        """
        Returns cached set of detected obstacle IDs for current step.

        Args:
            ego (EgoState): Current state of the Ego vehicle.
            obstacles (list): List of dynamic obstacle objects.
            step (int): Current simulation time step index.

        Returns:
            set[int]: Set of unique obstacle IDs currently in FOV.
        """
        
        return self.scan(ego, obstacles, step)["detected_ids"]

    def is_adjacent_lane_clear(self,
                               ego: EgoState,
                               obstacles: list,
                               step: int,
                               target_offset: float) -> bool:
        """
        Returns cached lane clearance boolean for given target offset and current step.

        Args:
            ego (EgoState): Current state of the Ego vehicle.
            obstacles (list): List of dynamic obstacle objects.
            step (int): Current simulation time step index.
            target_offset (float): Target lateral lane offset (+ for left, - for right).

        Returns:
            bool: True if blind spot is clear for target lane change, False otherwise.
        """
        
        cache = self.scan(ego, obstacles, step, target_offset=target_offset)
        return cache["target_clearance"][target_offset]
