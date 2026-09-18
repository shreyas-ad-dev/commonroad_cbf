from typing import Any

from dataclasses import dataclass
import numpy as np
from shapely.geometry import LineString
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.ops import unary_union

from src.base_sensor import BaseSensor, SensorOcclusionData
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

    def __init__(self,
                 range_max: float = 70.0,
                 fov_deg: float = 60.0,
                 mount_position: str = "front",
                 noise_std: float = 0.5,
                 ray_count: int = 40):
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

    def _get_sensor_transform(self, ego: EgoState) -> (np.ndarray, float):
        if self.mount_position == "front":
            offset = (ego.length / 2.0)
            sensor_heading_deg = np.degrees(ego.orientation)
        else:
            offset = -(ego.length / 2.0)
            sensor_heading_deg = np.degrees(ego.orientation) + 180.0

        sensor_pos = ego.position + offset*ego.heading_vector
        
        return sensor_pos, sensor_heading_deg
    
    def _build_fov_wedge(self, sensor_pos: np.ndarray, sensor_heading_deg: float) -> ShapelyPolygon:
        t1 = sensor_heading_deg - (self.fov_deg / 2.0)
        t2 = sensor_heading_deg + (self.fov_deg / 2.0)
        angles = np.radians(np.linspace(t1, t2, self.ray_count))

        arc_pts = [
                (sensor_pos[0] + self.range_max * np.cos(a), sensor_pos[1] + self.range_max * np.sin(a)) for a in angles
                ]
        return ShapelyPolygon([tuple(sensor_pos)] + arc_pts + [tuple(sensor_pos)])

    def _compute_occlusion_shadow(self, sensor_pos: np.ndarray, obs_poly: ShapelyPolygon) -> ShapelyPolygon | None:
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


#    def is_in_fov(self,
#                  ego: EgoState,
#                  obstacle: object,
#                  step: int) -> tuple[bool, float, float, float]:
#        """
#        Checks if an obstacle's center or any of its bounding box corners fall within the Radar's FOV.
#
#        Args:
#            ego (EgoState): Current state of the Ego vehicle.
#            obstacle (object): Dynamic obstacle instance to evaluate.
#            step (int): Current simulation time step index.
#
#        Returns:
#            tuple[bool, float, float, float]: A tuple containing:
#                - any_corner_in_fov (bool): True if any point falls inside detection cone.
#                - min_dist (float): Minimum Euclidean distance across all points.
#                - center_x_local (float): Obstacle center longitudinal offset in Ego frame.
#                - center_y_local (float): Obstacle center lateral offset in Ego frame.
#        """
#        eval_data = self.get_obstacle_center_and_corners_in_local(ego, obstacle, step)
#        if eval_data is None:
#            return False, float('inf'), 0.0, 0.0
#
#        center_local, local_points = eval_data
#        center_x_local, center_y_local = center_local[0], center_local[1]
#
#        min_dist = float('inf')
#        any_corner_in_fov = False
#
#        for pt in local_points:
#            x_local, y_local = pt[0], pt[1]
#            dist = float(np.hypot(x_local, y_local))
#
#            min_dist = min(min_dist, dist)
#
#            if dist <= self.range_max:
#                # Direction constraint based on mounting orientation
#                is_valid_direction = (
#                    (self.mount_position == "front" and x_local > 0.0) or
#                    (self.mount_position == "rear" and x_local < 0.0)
#                )
#
#                if is_valid_direction:
#                    sensor_x = x_local if self.mount_position == "front" else -x_local
#                    angle = np.arctan2(y_local, sensor_x)
#                    if abs(angle) <= self.half_fov_rad:
#                        any_corner_in_fov = True
#
#        return any_corner_in_fov, min_dist, center_x_local, center_y_local

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
            #fov_data: dict[int, tuple[bool, float, float, float]] = {}
            fov_data: {int, SensorOcclusionData} = {}
            detections: list[Detection] = []

            sensor_pos, sensor_heading = self._get_sensor_transform(ego)
            fov_wedge = self._build_fov_wedge(sensor_pos, sensor_heading)
            timestamp = step * 0.1 # 10 Hz step delta time

            valid_obstacles = []
            for obs in obstacles:
                #in_fov, min_dist, center_x_local, center_y_local = self.is_in_fov(ego, obs, step)
                eval_data = self.get_obstacle_center_and_corners_in_local(ego, obs, step)
                st = obs.state_at_time(step)
                if eval_data is None or st is None:
                    continue

                center_local, local_points = eval_data
                obs_length = getattr(obs.obstacle_shape, 'length', getattr(st, 'length', 4.5))
                obs_width = getattr(obs.obstacle_shape, 'width', getattr(st, 'width', 4.5))
                obs_yaw = getattr(st, 'orientation', getattr(st, 'yaw', 0.0))

                obs_poly, _ = get_car_polygon(
                        x=st.position[0],
                        y=st.position[1],
                        orientation=obs_yaw,
                        length=obs_length,
                        width=obs_width
                        )

                #fov_data[obs.obstacle_id] = (in_fov, min_dist, center_x_local, center_y_local)
                #in_fov = obs_poly.intersects(fov_wedge)
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
        Gets cached set of obstacle IDs that fall within this radar's FOV at the given step.

        Args:
            ego (EgoState): Current state of the Ego vehicle.
            obstacles (list): List of dynamic obstacle objects.
            step (int): Current simulation time step index.

        Returns:
            set[int]: Set of detected obstacle unique identifiers.
        """

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

        Uses cached FOV evaluation results to eliminate redundant coordinate 
        transformations and projects obstacles onto the road corridor.

        Args:
            ego (EgoState): Current state of the Ego vehicle.
            obstacles (list): List of dynamic obstacle objects.
            step (int): Current simulation time step index.
            lane_corridor_width (float, optional): Corridor width for lead vehicle filtering in meters. Defaults to 2.5.
            target_offset (float, optional): Lateral lane offset in meters (+ for left, - for right). Defaults to 0.0.

        Returns:
            tuple[float, float, float, int, float] | None: A tuple containing 
                (x_world, y_world, velocity, obstacle_id, bumper_to_bumper_x_local), 
                or None if no lead vehicle is detected.
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

            #in_fov, min_dist, center_x_local, _ = scan_res["fov_data"][obs.obstacle_id]
            occ_data = scan_res["fov_data"][obs.obstacle_id]
            if not occ_data.in_fov:
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
        """
        Evaluates whether an adjacent lane target gap is clear using front and rear radars.

        Args:
            ego (EgoState): Current state of the Ego vehicle.
            surrounding_obstacles (list): List of dynamic obstacles in the scene.
            step (int): Current simulation time step index.
            target_lane_offset (float): Target lane offset in meters (+ for left, - for right).
            safety_gap_front (float, optional): Required clearance ahead in target lane. Defaults to 12.0.
            safety_gap_rear (float, optional): Required clearance behind in target lane. Defaults to 10.0.
            rear_radar (RadarSensor | None, optional): Optional rear radar instance for rear scanning. Defaults to None.
            lane_tolerance (float, optional): Half-width tolerance for target lane check. Defaults to 1.8.

        Returns:
            bool: True if target lane gap is completely clear of obstacles, False otherwise.
        """
        u_hat, n_hat = ego.road_frame_vectors

        front_scan = self.scan(ego, surrounding_obstacles, step)
        rear_scan = rear_radar.scan(ego, surrounding_obstacles, step) if rear_radar is not None else None

        for obs in surrounding_obstacles:
            st = obs.state_at_time(step)
            if st is None:
                continue

            obs_id = obs.obstacle_id
            #in_front_fov = front_scan["fov_data"].get(obs_id, (False,))[0]
            #in_rear_fov = rear_scan["fov_data"].get(obs_id, (False,))[0] if rear_scan else False
            front_occ = front_scan["fov_data"].get(obs_id)
            rear_occ = rear_scan["fov_data"].get(obs_id) if rear_scan else None

            in_front_fov = front_occ.in_fov if front_occ else False
            in_read_fov = rear_occ.in_fov if rear_occ else False

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
