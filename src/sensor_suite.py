from dataclasses import dataclass, field

from src.data_association import associate_detections_to_tracks
from src.ego_state import EgoState
from src.radar import RadarSensor
from src.tracker import Track, TrackState
from src.ultrasonic import SideUltrasonicSensor


class MultiObjectTracker:
    """
    Manages track creation, prediction, association, updates, and deletion across time steps.
    """

    def __init__(self, dt: float = 0.1, confirm_hits: int = 3, max_age: int = 5, max_distance: float = 5.0):
        self.dt = dt
        self.confirm_hits = confirm_hits
        self.max_age = max_age
        self.max_distance = max_distance
        
        self.tracks: list[Track] = []
        self._next_track_id = 1

    @property
    def active_tracks(self) -> list[Track]:
        """Returns all confirmed and tentative active tracks."""
        return [t for t in self.tracks if t.state != TrackState.DELETED]

    @property
    def confirmed_tracks(self) -> list[Track]:
        """Returns only confirmed active tracks."""
        return [t for t in self.tracks if t.state == TrackState.CONFIRMED]

    def process_step(self, detections: list) -> list[Track]:
        """Advances active tracks, associates new detections, and updates Kalman filters."""
        # 1. Predict state for all existing tracks
        for track in self.tracks:
            track.predict()

        # 2. Perform global data association
        matches, unmatched_tracks, unmatched_dets = associate_detections_to_tracks(
            tracks=self.tracks,
            detections=detections,
            max_distance=self.max_distance
        )

        # 3. Update matched tracks
        for track_idx, det_idx in matches:
            self.tracks[track_idx].update(detections[det_idx])

        # 4. Handle missed tracks
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()

        # 5. Spawn new tentative tracks for unmatched detections
        for det_idx in unmatched_dets:
            new_track = Track(
                track_id=self._next_track_id,
                detection=detections[det_idx],
                dt=self.dt,
                confirm_hits=self.confirm_hits,
                max_age=self.max_age
            )
            self.tracks.append(new_track)
            self._next_track_id += 1

        # 6. Purge deleted tracks
        self.tracks = [t for t in self.tracks if t.state != TrackState.DELETED]

        return self.confirmed_tracks


@dataclass
class PerceptionState:
    """
    Snapshot of perception and tracking results for the current time step.
    """

    lead_target: tuple | None = None
    radar_scans: dict | None = None
    uss_scans: dict | None = None
    filtered_obstacles: list | None = None
    tracks: list[Track] = field(default_factory=list)


@dataclass
class LaneClearanceResult:
    """Encapsulates fused safety and clearance evaluations for lane change maneuvers."""

    is_safe: bool
    radar_clear: bool
    uss_clear: bool

    def __bool__(self) -> bool:
        return self.is_safe


class SensorSuite:
    """
    Fuses multi-modal perception sensors (Radar and USS) and manages multi-object tracking.
    """

    def __init__(
        self,
        front_radar: RadarSensor,
        rear_radar: RadarSensor | None = None,
        uss_left: SideUltrasonicSensor | None = None,
        uss_right: SideUltrasonicSensor | None = None,
        max_perception_radius: float = 85.0,
        dt: float = 0.1
    ):
        self.front_radar = front_radar
        self.rear_radar = rear_radar
        self.uss_left = uss_left
        self.uss_right = uss_right
        self.max_perception_radius = max_perception_radius

        self.tracker = MultiObjectTracker(dt=dt)
        self.latest_perception = PerceptionState()

    @property
    def lead_target(self) -> tuple | None:
        return self.latest_perception.lead_target

    @property
    def tracked_objects(self) -> list[Track]:
        """List[Track]: Confirmed active tracked objects from Kalman pipeline."""
        return self.latest_perception.tracks

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

    def clear_tracking(self):
        self.latest_perception = PerceptionState()
        self.tracker.tracks.clear()

    def update(
            self,
            ego: EgoState,
            all_obstacles: list,
            step: int,
            target_offset: float = 0.0
            ) -> PerceptionState:
        """
        Runs spatial pre-filtering, sensor scanning, data association, and multi-object tracking.
        """

        # 1. Fast Spatial Pre-filter
        r_sq = self.max_perception_radius ** 2
        ego_pos = ego.position
        nearby_obstacles = []

        for obs in all_obstacles:
            st = obs.state_at_time(step)
            if st is not None:
                d_vec = st.position - ego_pos
                if (d_vec[0]**2 + d_vec[1]**2) <= r_sq:
                    nearby_obstacles.append(obs)

        # 2. Execute Individual Sensor Scans
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

        # 3. Collect Raw Detections Across All Sensors
        raw_detections = []
        raw_detections.extend(self.front_radar.get_detections(ego, nearby_obstacles, step))
        if self.rear_radar:
            raw_detections.extend(self.rear_radar.get_detections(ego, nearby_obstacles, step))
        if self.uss_left:
            raw_detections.extend(self.uss_left.get_detections(ego, nearby_obstacles, step))
        if self.uss_right:
            raw_detections.extend(self.uss_right.get_detections(ego, nearby_obstacles, step))

        # 4. Ingest Detections into Multi-Object Tracker
        confirmed_tracks = self.tracker.process_step(raw_detections)

        # 5. Cache Lead Target
        lead_target = self.front_radar.track_lead_vehicle(
            ego=ego,
            obstacles=nearby_obstacles,
            step=step,
            target_offset=target_offset
        )

        self.latest_perception = PerceptionState(
            lead_target=lead_target,
            radar_scans=radar_results,
            uss_scans=uss_results,
            filtered_obstacles=nearby_obstacles,
            tracks=confirmed_tracks
        )
        return self.latest_perception

    def track_lead(self, ego: EgoState, step: int, target_offset: float = 0.0):
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
        obstacles = self.latest_perception.filtered_obstacles or []

        radar_clear = self.front_radar.is_adjacent_lane_clear(
            ego=ego,
            surrounding_obstacles=obstacles,
            step=step,
            target_lane_offset=target_offset,
            safety_gap_front=safety_gap_front,
            safety_gap_rear=safety_gap_rear,
            rear_radar=self.rear_radar
        )

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
