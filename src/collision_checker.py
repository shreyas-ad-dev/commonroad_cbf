# src/collision_checker.py

from dataclasses import dataclass

import numpy as np

from src.config import BP, CBF, MAP, VEHICLE
from src.ego_state import EgoState, get_car_polygon
from src.tracker import Track


@dataclass
class TrajectoryConflict:
    """Represents a predicted spatiotemporal polygon intersection between Ego and a Track."""

    track: Track
    time_to_conflict: float
    ego_state_at_conflict: dict
    obs_state_at_conflict: dict


class CollisionChecker:
    """
    Standalone 2D polygon collision checker for both:
      1. Ground-truth simulation collision tracking & render state preparation.
      2. Predictive spatiotemporal conflict detection between Ego forecasts and Track predictions.
    """

    def __init__(
        self,
        default_obs_length: float = VEHICLE.DEFAULT_LENGTH,
        default_obs_width: float = VEHICLE.DEFAULT_WIDTH,
        longitudinal_buffer: float = CBF.SAFETY_DISTANCE_BUFFER_METERS,
        lateral_buffer: float = 0.5,
        in_lane_lateral_threshold: float = 0.5 * MAP.DEFAULT_LANE_WIDTH_METERS,
    ):
        self.default_obs_length = default_obs_length
        self.default_obs_width = default_obs_width
        self.longitudinal_buffer = longitudinal_buffer
        self.lateral_buffer = lateral_buffer
        self.in_lane_lateral_threshold = in_lane_lateral_threshold

        # Ground-truth simulation collision state
        self.has_collided: bool = False
        self.collision_step: int | None = None
        self.collided_obstacle_id: int | None = None
        self.frozen_obs_states: dict[int, tuple[float, float, float]] = {}

    def reset_sim_state(self) -> None:
        """Resets ground-truth collision tracking flags."""
        self.has_collided = False
        self.collision_step = None
        self.collided_obstacle_id = None
        self.frozen_obs_states.clear()

    # -------------------------------------------------------------------------
    # 1. Shared Low-Level Geometric Primitive
    # -------------------------------------------------------------------------
    @staticmethod
    def check_pose_collision(
        ego_pose: tuple[float, float, float, float, float],
        obs_pose: tuple[float, float, float, float, float],
    ) -> tuple[bool, np.ndarray]:
        """
        Checks 2D oriented bounding polygon intersection between Ego and an obstacle.

        Args:
            ego_pose: (x, y, orientation, length, width)
            obs_pose: (x, y, orientation, length, width)

        Returns:
            (intersects, obs_corners)
        """
        ego_poly, _ = get_car_polygon(*ego_pose)
        obs_poly, obs_corners = get_car_polygon(*obs_pose)
        return bool(ego_poly.intersects(obs_poly)), obs_corners

    # -------------------------------------------------------------------------
    # 2. Ground-Truth Simulation Step Evaluation (Replaces Run-Script Loop)
    # -------------------------------------------------------------------------
    def evaluate_ground_truth_step(
        self,
        ego: EgoState,
        surrounding_obstacles: list,
        step: int,
    ) -> list[tuple]:
        """
        Evaluates exact ground-truth polygon collisions at `step`, freezes the 
        collided obstacle pose upon impact, and builds `surrounding_render_states`.
        """
        surrounding_render_states: list[tuple] = []
        ego_pose = (ego.x, ego.y, ego.orientation, ego.length, ego.width)

        for obs in surrounding_obstacles:
            if self.has_collided and obs.obstacle_id == self.collided_obstacle_id:
                ox, oy, o_orient = self.frozen_obs_states[obs.obstacle_id]
            else:
                st = obs.state_at_time(step)
                if st is None:
                    continue
                ox, oy = float(st.position[0]), float(st.position[1])
                o_orient = float(getattr(st, "orientation", 0.0))

            l = float(getattr(obs.obstacle_shape, "length", self.default_obs_length))
            w = float(getattr(obs.obstacle_shape, "width", self.default_obs_width))

            intersects, obs_corners = self.check_pose_collision(
                ego_pose=ego_pose,
                obs_pose=(ox, oy, o_orient, l, w),
            )

            if not self.has_collided and intersects:
                self.has_collided = True
                self.collision_step = step
                self.collided_obstacle_id = obs.obstacle_id
                self.frozen_obs_states[obs.obstacle_id] = (ox, oy, o_orient)

            is_hit = self.has_collided and (obs.obstacle_id == self.collided_obstacle_id)
            surrounding_render_states.append((obs, obs_corners, is_hit))

        return surrounding_render_states

    # -------------------------------------------------------------------------
    # 3. Predictive Trajectory Conflict & Merge Hazard Checking
    # -------------------------------------------------------------------------
    def _is_in_ego_lane_currently(self, ego: EgoState, track: Track) -> bool:
        """Checks whether a track is currently in Ego's lane (handled longitudinally by CBF)."""
        _, n_road = ego.road_frame_vectors
        lat_offset = abs(float(np.dot(track.position - ego.position, n_road)))
        return lat_offset < self.in_lane_lateral_threshold

    def check_track_collision(
        self,
        ego: EgoState,
        ego_forecast: list[dict],
        track: Track,
        horizon_seconds: float,
    ) -> TrajectoryConflict | None:
        """
        Reuses `check_pose_collision` across predicted timestamps to detect future 
        polygon intersections between Ego's forecast and a Track's prediction.
        """
        if not ego_forecast:
            return None

        fallback_heading = ego.road_heading if ego.road_heading is not None else ego.orientation
        obs_predictions = track.predict_trajectory(
            horizon_seconds=horizon_seconds,
            fallback_heading=fallback_heading,
        )
        if not obs_predictions:
            return None

        num_steps = min(len(ego_forecast), len(obs_predictions))
        inflated_ego_len = ego.length + self.longitudinal_buffer
        inflated_ego_wid = ego.width + self.lateral_buffer

        for k in range(num_steps):
            ego_step = ego_forecast[k]
            obs_step = obs_predictions[k]

            intersects, _ = self.check_pose_collision(
                ego_pose=(
                    ego_step["x"],
                    ego_step["y"],
                    ego_step["orientation"],
                    inflated_ego_len,
                    inflated_ego_wid,
                ),
                obs_pose=(
                    obs_step["x"],
                    obs_step["y"],
                    obs_step["orientation"],
                    self.default_obs_length,
                    self.default_obs_width,
                ),
            )

            if intersects:
                return TrajectoryConflict(
                    track=track,
                    time_to_conflict=float(ego_step["time_offset"]),
                    ego_state_at_conflict=ego_step,
                    obs_state_at_conflict=obs_step,
                )

        return None

    def find_adjacent_conflicts(
        self,
        ego: EgoState,
        tracks: list[Track],
        horizon_seconds: float = BP.MERGE_TIME_MAX,
        reference_path: np.ndarray | None = None,
        dt: float | None = None,
        ignore_ego_lane: bool = True,
    ) -> list[TrajectoryConflict]:
        """
        Forecasts Ego's trajectory and checks adjacent-lane tracks for future 
        polygon intersections, sorted by earliest `time_to_conflict`.
        """
        if not tracks:
            return []

        step_dt = dt if dt is not None else tracks[0].kf.dt
        ego_forecast = ego.forecast(
            horizon_seconds=horizon_seconds,
            dt=step_dt,
            reference_path=reference_path,
        )

        conflicts: list[TrajectoryConflict] = []
        for track in tracks:
            if ignore_ego_lane and self._is_in_ego_lane_currently(ego, track):
                continue

            conflict = self.check_track_collision(
                ego=ego,
                ego_forecast=ego_forecast,
                track=track,
                horizon_seconds=horizon_seconds,
            )
            if conflict is not None:
                conflicts.append(conflict)

        conflicts.sort(key=lambda c: c.time_to_conflict)
        return conflicts

    def get_merge_hazard(
        self,
        ego: EgoState,
        tracks: list[Track],
        horizon_seconds: float = BP.MERGE_TIME_MAX,
        reference_path: np.ndarray | None = None,
        dt: float | None = None,
    ) -> TrajectoryConflict | None:
        """
        Returns the earliest-conflicting adjacent-lane track whose predicted trajectory 
        intersects Ego's forecasted path ahead of or alongside Ego.
        """
        conflicts = self.find_adjacent_conflicts(
            ego=ego,
            tracks=tracks,
            horizon_seconds=horizon_seconds,
            reference_path=reference_path,
            dt=dt,
            ignore_ego_lane=True,
        )

        for conflict in conflicts:
            ego_pos = np.array([
                conflict.ego_state_at_conflict["x"],
                conflict.ego_state_at_conflict["y"],
            ])
            obs_pos = np.array([
                conflict.obs_state_at_conflict["x"],
                conflict.obs_state_at_conflict["y"],
            ])
            ego_yaw = conflict.ego_state_at_conflict["orientation"]
            u_ego = np.array([np.cos(ego_yaw), np.sin(ego_yaw)])

            rel_long_at_conflict = float(np.dot(obs_pos - ego_pos, u_ego))
            if rel_long_at_conflict >= -0.5 * ego.length:
                return conflict

        return None
