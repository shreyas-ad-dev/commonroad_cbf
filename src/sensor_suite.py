# src/sensor_suite.py
from dataclasses import dataclass
import numpy as np
from src.ego_state import EgoState
from src.radar import RadarSensor
from src.ultrasonic import SideUltrasonicSensor


@dataclass
class PerceptionState:
    """Snapshot of perception results for the current time step."""
    lead_target: tuple | None = None
    radar_scans: dict = None
    uss_scans: dict = None
    filtered_obstacles: list = None

@dataclass
class LaneClearanceResult:
    is_safe: bool
    radar_clear: bool
    uss_clear: bool

    def __bool__(self) -> bool:
        """Allows direct boolean evaluation: if clearance: ..."""
        return self.is_safe

class SensorSuite:
    def __init__(
        self,
        front_radar: RadarSensor,
        rear_radar: RadarSensor | None = None,
        uss_left: SideUltrasonicSensor | None = None,
        uss_right: SideUltrasonicSensor | None = None,
        max_perception_radius: float = 85.0
    ):
        self.front_radar = front_radar
        self.rear_radar = rear_radar
        self.uss_left = uss_left
        self.uss_right = uss_right
        self.max_perception_radius = max_perception_radius
        self.latest_perception = PerceptionState()

    @property
    def lead_target(self) -> tuple | None:
        return self.latest_perception.lead_target

    @property
    def front_tracked_ids(self) -> set[int]:
        if not self.latest_perception.radar_scans:
            return set()
        return self.latest_perception.radar_scans.get("front", {}).get("detected_ids", set())

    @property
    def rear_tracked_ids(self) -> set[int]:
        if not self.latest_perception.radar_scans:
            return set()
        return self.latest_perception.radar_scans.get("rear", {}).get("detected_ids", set())

    @property
    def left_tracked_ids(self) -> set[int]:
        if not self.latest_perception.uss_scans:
            return set()
        return self.latest_perception.uss_scans.get("left", {}).get("detected_ids", set())

    @property
    def right_tracked_ids(self) -> set[int]:
        if not self.latest_perception.uss_scans:
            return set()
        return self.latest_perception.uss_scans.get("right", {}).get("detected_ids", set())

    def update(
            self,
            ego: EgoState,
            all_obstacles: list,
            step: int,
            target_offset : float = 0
            ) -> PerceptionState:
        """Runs spatial pre-filtering and updates all mounted sensors."""
        # 1. Fast Spatial Pre-filter (Single pass)
        r_sq = self.max_perception_radius ** 2
        ego_pos = ego.position
        nearby_obstacles = []
        
        for obs in all_obstacles:
            st = obs.state_at_time(step)
            if st is not None:
                d_vec = st.position - ego_pos
                if (d_vec[0]**2 + d_vec[1]**2) <= r_sq:
                    nearby_obstacles.append(obs)

        # 2. Run Individual Scans
        radar_results = {
            "front": self.front_radar.scan(ego, nearby_obstacles, step),
        }
        if self.rear_radar:
            radar_results["rear"] = self.rear_radar.scan(ego, nearby_obstacles, step)

        uss_results = {}
        if self.uss_left:
            uss_results["left"] = self.uss_left.scan(ego, nearby_obstacles, step)
        if self.uss_right:
            uss_results["right"] = self.uss_right.scan(ego, nearby_obstacles, step)

        # 3. Cache Lead Target
        lead_target = self.front_radar.track_lead_vehicle(ego=ego, obstacles=nearby_obstacles, step=step, target_offset=target_offset)

        self.latest_perception = PerceptionState(
            lead_target=lead_target,
            radar_scans=radar_results,
            uss_scans=uss_results,
            filtered_obstacles=nearby_obstacles
        )
        return self.latest_perception

    def track_lead(self, ego: EgoState, step: int, target_offset: float = 0.0):
        """Allows re-evaluating or updating lead target after planner executes."""
        obstacles = self.latest_perception.filtered_obstacles or []
        self.latest_perception.lead_target = self.front_radar.track_lead_vehicle(
            ego=ego,
            obstacles=obstacles,
            step=step,
            target_offset=target_offset
        )
        return self.latest_perception.lead_target

    def is_lane_change_safe(
        self,
        ego: EgoState,
        target_offset: float,
        step: int,
        safety_gap_front: float = 10.0,
        safety_gap_rear: float = 8.0
    ) -> LaneClearanceResult:
        """High-level semantic query fusing Radar and USS clearance checks."""
        obstacles = self.latest_perception.filtered_obstacles or []

        # 1. Check Radars
        radar_clear = self.front_radar.is_adjacent_lane_clear(
            ego=ego,
            surrounding_obstacles=obstacles,
            step=step,
            target_lane_offset=target_offset,
            safety_gap_front=safety_gap_front,
            safety_gap_rear=safety_gap_rear,
            rear_radar=self.rear_radar
        )

        # 2. Check Side USS based on target lane direction (+ for left, - for right)
        active_uss = self.uss_left if target_offset > 0 else self.uss_right
        uss_clear = (
            active_uss.is_adjacent_lane_clear(ego, obstacles, step, target_offset)
            if active_uss is not None else True
        )

        return LaneClearanceResult(
                is_safe=radar_clear and uss_clear,
                radar_clear=radar_clear,
                uss_clear=uss_clear
                )
