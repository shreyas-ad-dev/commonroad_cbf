#src/radar.py

from typing import Any

import numpy as np
from shapely.geometry import LineString
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.ops import unary_union

from src.base_sensor import BaseSensor, SensorOcclusionData
from src.config import RADAR, TEST, VEHICLE
from src.ego_state import EgoState, get_car_polygon
from src.tracker import Detection


class RadarSensor(BaseSensor):
    """
    Simulates a body-frame aligned Radar sensor with finite range and Field of View (FOV).

    Supports 'front' (+x_local) and 'rear' (-x_local) mounting orientations on the vehicle.
    Evaluates FOV containment across all bounding box corners and center of dynamic obstacles.

    Attributes:
        mount_position (str): Orientation relative to vehicle ('front' or 'rear').
        range_max (float): Maximum detection range in meters.
        fov_deg (float): Total azimuth field of view in degrees.
        """

    def __init__(
            self,
            range_max: float = RADAR.DEFAULT_RADAR_MAX_RANGE,
            fov_deg: float = RADAR.DEFAULT_RADAR_FOV_DEG,
            mount_position: str = "front",
            noise_std: float = RADAR.DEFAULT_RADAR_RANGE_NOISE_STD,
            ray_count: int = 40
        ):
        """
        Initializes the RadarSensor instance.

        Args:
            range_max (float, optional): Maximum detection range in meters. Defaults to 70.0.
            fov_deg (float, optional): Total azimuth field of view in degrees. Defaults to 60.0.
            mount_position (str, optional): Orientation relative to vehicle ('front' or 'rear'). Defaults to "front".
            noise_std (float, optional): Standard deviation of Gaussian measurement noise. Defaults to 0.5.

        Raises:
            ValueError: If mount_position is not 'front' or 'rear'.
        """

        if mount_position not in ["front", "rear"]:
            raise ValueError("mount_position must be either 'front' or 'rear'")

        super().__init__(range_max=range_max, fov_deg=fov_deg)
        self.mount_position = mount_position
        self.sensor_id  = f"radar_{mount_position}"
        self.noise_std = noise_std
        self.R = np.eye(2) * (noise_std **2)
        self.ray_count = ray_count

        # Stateful internal cache
        self._last_step: int | None = None
        self._scan_cache: dict[str, Any] = {
            "detected_ids": set(),
            "fov_data": {},  # Maps obs_id -> (in_fov, min_dist, center_x_local, center_y_local)
            "detections": [],
            "lead_target": None,
            "visible_fov": None
        }

    def _get_sensor_transform(
            self,
            ego: EgoState
        ) -> (np.ndarray, float):

        if self.mount_position == "front":
            offset = (ego.length / 2.0)
            sensor_heading_deg = np.degrees(ego.orientation)
        else:
            offset = -(ego.length / 2.0)
            sensor_heading_deg = np.degrees(ego.orientation) + 180.0

        sensor_pos = ego.position + offset*ego.heading_vector
        
        return sensor_pos, sensor_heading_deg
    
    def _build_fov_wedge(
            self,
            sensor_pos: np.ndarray,
            sensor_heading_deg: float
        ) -> ShapelyPolygon:

        t1 = sensor_heading_deg - (self.fov_deg / 2.0)
        t2 = sensor_heading_deg + (self.fov_deg / 2.0)
        angles = np.radians(np.linspace(t1, t2, self.ray_count))

        arc_pts = [
                (sensor_pos[0] + self.range_max * np.cos(a), sensor_pos[1] + self.range_max * np.sin(a)) for a in angles
                ]
        return ShapelyPolygon([tuple(sensor_pos)] + arc_pts + [tuple(sensor_pos)])

    def _compute_occlusion_shadow(
            self,
            sensor_pos: np.ndarray,
            obs_poly: ShapelyPolygon
        ) -> ShapelyPolygon | None:

        pts = np.array(obs_poly.exterior.coords)[:-1]
        if len(pts) == 0:
            return None

        angles = [np.arctan2(p[1] - sensor_pos[1], p[0] - sensor_pos[0]) for p in pts]
        min_idx, max_idx = np.argmin(angles), np.argmax(angles)

        if angles[max_idx] - angles[min_idx] > np.pi:
            pos_angles = [a if a >= 0 else a + 2* np.pi for a in angles]
            min_idx, max_idx = np.argmin(pos_angles), np.argmax(pos_angles)

        p1, p2 = pts[min_idx], pts[max_idx]

        proj_factor = self.range_max * 2.0
        v1 = (p1 - sensor_pos) / np.linalg.norm(p1 - sensor_pos)
        v2 = (p2 - sensor_pos) / np.linalg.norm(p2 - sensor_pos)

        p1_proj = p1 + v1 * proj_factor
        p2_proj = p2 + v2 * proj_factor

        return ShapelyPolygon([p1, p2, p2_proj, p1_proj])


    def scan(
            self,
            ego: EgoState,
            obstacles: list,
            step: int
        ) -> dict[str, Any]:
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
            #fov_data: dict[int, tuple[bool, float, float, float]] = {}
            fov_data: {int, SensorOcclusionData} = {}
            detections: list[Detection] = []

            sensor_pos, sensor_heading = self._get_sensor_transform(ego)
            fov_wedge = self._build_fov_wedge(sensor_pos, sensor_heading)
            timestamp = step * TEST.DEFAULT_SAMPLING_TIME_SEC # 10 Hz step delta time

            valid_obstacles = []
            for obs in obstacles:
                eval_data = self.get_obstacle_center_and_corners_in_local(ego, obs, step)
                st = obs.state_at_time(step)
                if eval_data is None or st is None:
                    continue

                center_local, local_points = eval_data
                obs_length = getattr(obs.obstacle_shape, 'length', getattr(st, 'length', VEHICLE.DEFAULT_LENGTH))
                obs_width = getattr(obs.obstacle_shape, 'width', getattr(st, 'width', VEHICLE.DEFAULT_WIDTH))
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
                    p1, p2 = pts[i], pts[(i+1) % num_pts]
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

                # Shadow generation for occlusions
                shadow_poly = self._compute_occlusion_shadow(sensor_pos, obs_poly)
                if shadow_poly and shadow_poly.is_valid:
                    accumulated_shadows.append(shadow_poly)


                # Apply measurement noise to local center coordinates
                noisy_x_local = center_x_local + np.random.normal(0.0, self.noise_std)
                noisy_y_local = center_y_local + np.random.normal(0.0, self.noise_std)

                # Transform noisy measurement back to global coordinates
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
                "fov_data": fov_data,
                "detections": detections,
                "lead_target": None,
                "visible_fov": visible_fov_geometry
            }

        return self._scan_cache

    def get_detections(
            self,
            ego: EgoState,
            obstacles: list,
            step: int
        ) -> list[Detection]:
        """Gets cached list of Detection objects for MOT tracking pipeline."""
        return self.scan(ego, obstacles, step)["detections"]

    def serialize(self):
        return {
                "sensor_id": self.sensor_id,
                "position": self.mount_position,
                "range": self.range_max,
                "fov_deg": self.fov_deg,
                "noise_std": self.noise_std,
                "R": self.R
                }

