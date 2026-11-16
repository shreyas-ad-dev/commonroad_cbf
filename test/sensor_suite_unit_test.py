"""
Level-1 unit tests for src.sensor_suite.

Run from the repository root:
    pytest test/sensor_suite_unit_test.py -v

Covers MultiObjectTracker, PerceptionState, LaneClearanceResult and SensorSuite.
Sensor and tracker collaborators are replaced with fakes where needed so these
tests focus on this module's own contracts rather than re-testing other modules.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import src.sensor_suite as module
from src.sensor_suite import (
    LaneClearanceResult, MultiObjectTracker, PerceptionState, SensorSuite,
)
from src.tracker import TrackState


class FakeSensor:
    def __init__(self, name, detections=None):
        self.name = name
        self.scan_result = {"detected_ids": {name}}
        self.detections = list(detections or [])
        self.scan_calls = []
        self.detection_calls = []
        self.serialize_calls = 0

    def scan(self, ego, obstacles, step):
        self.scan_calls.append((ego, list(obstacles), step))
        return self.scan_result

    def get_detections(self, ego, obstacles, step):
        self.detection_calls.append((ego, list(obstacles), step))
        return list(self.detections)

    def serialize(self):
        self.serialize_calls += 1
        return {"sensor": self.name}


class FakeObstacle:
    def __init__(self, obstacle_id, x, y, available_step=0):
        self.obstacle_id = obstacle_id
        self.position = np.array([x, y], dtype=float)
        self.available_step = available_step
        self.state_calls = []

    def state_at_time(self, step):
        self.state_calls.append(step)
        if step != self.available_step:
            return None
        return SimpleNamespace(position=self.position.copy())


class FakeTrack:
    def __init__(self, track_id=1, x=0.0, y=0.0, state=TrackState.CONFIRMED):
        self.track_id = track_id
        self.position = np.array([x, y], dtype=float)
        self.state = state
        self.predict_calls = 0
        self.update_calls = []
        self.missed_calls = 0

    def predict(self):
        self.predict_calls += 1

    def update(self, detection):
        self.update_calls.append(detection)

    def mark_missed(self):
        self.missed_calls += 1

    def serialize(self):
        return {"track_id": self.track_id}


class FakeTracker:
    def __init__(self, result=None):
        self.result = list(result or [])
        self.received = []
        self.tracks = []
        self.process_calls = 0

    @property
    def confirmed_tracks(self):
        return self.result

    @property
    def active_tracks(self):
        return self.tracks

    def process_step(self, detections):
        self.process_calls += 1
        self.received.append(list(detections))
        return list(self.result)


class TrackFactory:
    def __init__(self):
        self.created = []

    def __call__(self, **kwargs):
        track = FakeTrack(kwargs["track_id"], state=TrackState.TENTATIVE)
        track.constructor_kwargs = kwargs
        self.created.append(track)
        return track


def ego(x=0.0, y=0.0, u=(1.0, 0.0), n=(0.0, 1.0)):
    return SimpleNamespace(
        x=x, y=y, position=np.array([x, y], dtype=float),
        road_frame_vectors=(np.array(u, dtype=float), np.array(n, dtype=float)),
    )


def suite(front=None, rear=None, left=None, right=None, radius=50.0, dt=0.1):
    return SensorSuite(
        front_radar=front or FakeSensor("front"),
        rear_radar=rear, uss_left=left, uss_right=right,
        max_perception_radius=radius, dt=dt,
    )


# Perception snapshot and clearance result

def test_perception_state_defaults_and_independent_track_lists():
    a, b = PerceptionState(), PerceptionState()
    assert a.lead_target is None
    assert a.radar_scans is None and a.uss_scans is None
    assert a.filtered_obstacles is None and a.merge_hazard is None
    assert a.tracks == [] and a.tracks is not b.tracks


def test_perception_state_accepts_snapshot_values():
    tracks, hazard = [FakeTrack()], FakeTrack(2)
    state = PerceptionState((4, 1), {"front": 1}, {"left": 2}, ["obs"], tracks, hazard)
    assert state.lead_target == (4, 1)
    assert state.radar_scans == {"front": 1}
    assert state.uss_scans == {"left": 2}
    assert state.filtered_obstacles == ["obs"]
    assert state.tracks is tracks and state.merge_hazard is hazard


@pytest.mark.parametrize("safe", [True, False])
def test_lane_clearance_result_bool_matches_is_safe(safe):
    result = LaneClearanceResult(safe, not safe, safe)
    assert bool(result) is safe


def test_lane_clearance_result_keeps_modality_flags_distinct():
    result = LaneClearanceResult(False, True, False)
    assert result.is_safe is False
    assert result.radar_clear is True
    assert result.uss_clear is False


# MultiObjectTracker

def test_tracker_constructor_stores_parameters_and_initializes_state():
    tracker = MultiObjectTracker(dt=.2, confirm_hits=3, max_age=5, max_distance=9)
    assert (tracker.dt, tracker.confirm_hits, tracker.max_age, tracker.max_distance) == (.2, 3, 5, 9)
    assert tracker.tracks == [] and tracker._next_track_id == 1


def test_tracker_defaults_come_from_configuration():
    from src.config import DA, TEST, TRACK
    tracker = MultiObjectTracker()
    assert tracker.dt == TEST.DEFAULT_SAMPLING_TIME_SEC
    assert tracker.confirm_hits == TRACK.MIN_HITS_TO_CONFIRM
    assert tracker.max_age == TRACK.MAX_MISSED_DETECTIONS
    assert tracker.max_distance == DA.MAX_ASSIGNMENT_DISTANCE


def test_active_tracks_excludes_deleted():
    tracker = MultiObjectTracker()
    a = FakeTrack(1, state=TrackState.TENTATIVE)
    b = FakeTrack(2, state=TrackState.CONFIRMED)
    c = FakeTrack(3, state=TrackState.DELETED)
    tracker.tracks = [a, b, c]
    assert tracker.active_tracks == [a, b]


def test_confirmed_tracks_includes_only_confirmed():
    tracker = MultiObjectTracker()
    a = FakeTrack(1, state=TrackState.TENTATIVE)
    b = FakeTrack(2, state=TrackState.CONFIRMED)
    c = FakeTrack(3, state=TrackState.DELETED)
    tracker.tracks = [a, b, c]
    assert tracker.confirmed_tracks == [b]


def test_process_step_predicts_all_existing_tracks_and_passes_association_inputs(monkeypatch):
    tracker = MultiObjectTracker(max_distance=7)
    track, detection = FakeTrack(), object()
    tracker.tracks = [track]
    captured = {}

    def associate(**kwargs):
        captured.update(kwargs)
        return [], [0], []

    monkeypatch.setattr(module, "associate_detections_to_tracks", associate)
    tracker.process_step([detection])
    assert track.predict_calls == 1
    assert captured == {"tracks": [track], "detections": [detection], "max_distance": 7}


def test_process_step_updates_matched_tracks(monkeypatch):
    tracker, track, detection = MultiObjectTracker(), FakeTrack(), object()
    tracker.tracks = [track]
    monkeypatch.setattr(module, "associate_detections_to_tracks",
                        lambda **kw: ([(0, 0)], [], []))
    tracker.process_step([detection])
    assert track.update_calls == [detection]


def test_process_step_marks_unmatched_tracks_missed(monkeypatch):
    tracker, track = MultiObjectTracker(), FakeTrack()
    tracker.tracks = [track]
    monkeypatch.setattr(module, "associate_detections_to_tracks",
                        lambda **kw: ([], [0], []))
    tracker.process_step([])
    assert track.missed_calls == 1


def test_process_step_spawns_tracks_for_unmatched_detections_and_increments_ids(monkeypatch):
    tracker = MultiObjectTracker(dt=.25, confirm_hits=2, max_age=4)
    detections = [object(), object()]
    factory = TrackFactory()
    monkeypatch.setattr(module, "Track", factory)
    monkeypatch.setattr(module, "associate_detections_to_tracks",
                        lambda **kw: ([], [], [0, 1]))
    tracker.process_step(detections)
    assert [t.track_id for t in factory.created] == [1, 2]
    assert factory.created[0].constructor_kwargs == {
        "track_id": 1, "detection": detections[0], "dt": .25,
        "confirm_hits": 2, "max_age": 4,
    }
    assert tracker._next_track_id == 3


def test_process_step_returns_only_confirmed_tracks(monkeypatch):
    tracker, confirmed = MultiObjectTracker(), FakeTrack(1)
    tentative = FakeTrack(2, state=TrackState.TENTATIVE)
    tracker.tracks = [confirmed, tentative]
    monkeypatch.setattr(module, "associate_detections_to_tracks",
                        lambda **kw: ([], [], []))
    assert tracker.process_step([]) == [confirmed]


def test_process_step_purges_deleted_tracks(monkeypatch):
    tracker, keep = MultiObjectTracker(), FakeTrack(1)
    deleted = FakeTrack(2, state=TrackState.DELETED)
    tracker.tracks = [keep, deleted]
    monkeypatch.setattr(module, "associate_detections_to_tracks",
                        lambda **kw: ([], [], []))
    tracker.process_step([])
    assert tracker.tracks == [keep]


def test_process_step_empty_input_and_empty_tracker(monkeypatch):
    tracker = MultiObjectTracker()
    monkeypatch.setattr(module, "associate_detections_to_tracks",
                        lambda **kw: ([], [], []))
    assert tracker.process_step([]) == []
    assert tracker.tracks == []


# SensorSuite construction and properties

def test_suite_stores_sensor_references_configuration_and_tracker_dt():
    f, r, l, rr = FakeSensor("f"), FakeSensor("r"), FakeSensor("l"), FakeSensor("rr")
    s = suite(f, r, l, rr, radius=80, dt=.05)
    assert (s.front_radar, s.rear_radar, s.uss_left, s.uss_right) == (f, r, l, rr)
    assert s.max_perception_radius == 80 and s.tracker.dt == .05
    assert isinstance(s.latest_perception, PerceptionState)


def test_suite_optional_sensors_default_to_none():
    s = suite()
    assert s.rear_radar is None and s.uss_left is None and s.uss_right is None


def test_latest_perception_properties_expose_lead_hazard_and_tracks():
    s = suite()
    hazard, tracks = FakeTrack(2), [FakeTrack(1)]
    s.latest_perception.lead_target = (8, .5)
    s.latest_perception.merge_hazard = hazard
    s.latest_perception.tracks = tracks
    assert s.lead_target == (8, .5)
    assert s.merge_hazard is hazard
    assert s.tracked_objects is tracks


@pytest.mark.parametrize("prop", ["front_tracked_ids", "rear_tracked_ids",
                                  "left_tracked_ids", "right_tracked_ids"])
def test_tracked_id_properties_return_empty_when_scans_absent(prop):
    assert getattr(suite(), prop) == set()


@pytest.mark.parametrize("prop,attr,key", [
    ("front_tracked_ids", "radar_scans", "front"),
    ("rear_tracked_ids", "radar_scans", "rear"),
    ("left_tracked_ids", "uss_scans", "left"),
    ("right_tracked_ids", "uss_scans", "right"),
])
def test_tracked_id_properties_read_corresponding_scan(prop, attr, key):
    s = suite()
    setattr(s.latest_perception, attr, {key: {"detected_ids": {10, 11}}})
    assert getattr(s, prop) == {10, 11}


def test_tracked_id_properties_return_empty_when_expected_sensor_key_missing():
    s = suite()
    s.latest_perception.radar_scans = {"front": {"detected_ids": {1}}}
    s.latest_perception.uss_scans = {"left": {"detected_ids": {2}}}
    assert s.rear_tracked_ids == set() and s.right_tracked_ids == set()


# SensorSuite.update spatial filtering

def test_update_includes_obstacles_inside_radius():
    f = FakeSensor("front")
    s = suite(f, radius=10)
    s.tracker = FakeTracker()
    obs = FakeObstacle(1, 6, 0)
    result = s.update(ego(), [obs], 0)
    assert result.filtered_obstacles == [obs]
    assert f.scan_calls[0][1] == [obs]


def test_update_includes_obstacle_exactly_on_radius_boundary():
    s = suite(radius=10)
    s.tracker = FakeTracker()
    obs = FakeObstacle(1, 6, 8)
    assert s.update(ego(), [obs], 0).filtered_obstacles == [obs]


def test_update_excludes_obstacle_outside_radius():
    s = suite(radius=10)
    s.tracker = FakeTracker()
    assert s.update(ego(), [FakeObstacle(1, 10.01, 0)], 0).filtered_obstacles == []


def test_update_uses_euclidean_not_axis_aligned_filter():
    s = suite(radius=5)
    s.tracker = FakeTracker()
    assert s.update(ego(), [FakeObstacle(1, 4, 4)], 0).filtered_obstacles == []


def test_update_excludes_obstacle_without_state_at_step():
    s = suite()
    s.tracker = FakeTracker()
    obs = FakeObstacle(1, 1, 1, available_step=5)
    assert s.update(ego(), [obs], 4).filtered_obstacles == []
    assert obs.state_calls == [4]


def test_update_measures_radius_from_ego_position():
    s = suite(radius=5)
    s.tracker = FakeTracker()
    obs = FakeObstacle(1, 13, 4)
    assert s.update(ego(10, 4), [obs], 0).filtered_obstacles == [obs]


def test_update_preserves_input_order_of_filtered_obstacles():
    s = suite()
    s.tracker = FakeTracker()
    obs = [FakeObstacle(i, i, 0) for i in (1, 2, 3)]
    assert s.update(ego(), obs, 0).filtered_obstacles == obs


def test_unavailable_obstacles_are_not_passed_to_sensor():
    f = FakeSensor("front")
    s = suite(f)
    s.tracker = FakeTracker()
    s.update(ego(), [FakeObstacle(1, 1, 1, available_step=3)], 0)
    assert f.scan_calls[0][1] == [] and f.detection_calls[0][1] == []


# Sensor dispatch, detections and snapshot creation

def test_update_always_scans_front_radar_and_records_result():
    f = FakeSensor("front")
    s = suite(f)
    s.tracker = FakeTracker()
    result = s.update(ego(), [], 4)
    assert f.scan_calls[0][2] == 4
    assert result.radar_scans["front"] is f.scan_result


def test_update_scans_rear_radar_only_when_configured():
    r = FakeSensor("rear")
    s = suite(rear=r)
    s.tracker = FakeTracker()
    result = s.update(ego(), [], 2)
    assert result.radar_scans["rear"] is r.scan_result
    assert r.scan_calls[0][2] == 2


@pytest.mark.parametrize("side", ["left", "right"])
def test_update_scans_configured_uss(side):
    sensor = FakeSensor(side)
    s = suite(left=sensor) if side == "left" else suite(right=sensor)
    s.tracker = FakeTracker()
    result = s.update(ego(), [], 6)
    assert result.uss_scans[side] is sensor.scan_result
    assert sensor.scan_calls[0][2] == 6


def test_update_omits_absent_optional_sensor_scan_entries():
    s = suite()
    s.tracker = FakeTracker()
    result = s.update(ego(), [], 0)
    assert "rear" not in result.radar_scans
    assert result.uss_scans == {}


def test_update_combines_detections_in_front_rear_left_right_order():
    ds = [object() for _ in range(4)]
    sensors = [FakeSensor(name, [d]) for name, d in zip(
        ("front", "rear", "left", "right"), ds
    )]
    s = suite(*sensors)
    tracker = FakeTracker()
    s.tracker = tracker
    result = s.update(ego(), [], 3)
    assert tracker.received == [ds]
    assert result.tracks == tracker.result


def test_update_calls_scans_before_detection_collection():
    events = []

    class Ordered(FakeSensor):
        def scan(self, *args):
            events.append(self.name + "_scan")
            return super().scan(*args)

        def get_detections(self, *args):
            events.append(self.name + "_detections")
            return super().get_detections(*args)

    s = suite(Ordered("front"), rear=Ordered("rear"))
    s.tracker = FakeTracker()
    s.update(ego(), [], 0)
    assert events == ["front_scan", "rear_scan", "front_detections", "rear_detections"]


def test_update_invokes_tracker_once_with_combined_detections():
    ds = [object(), object()]
    s = suite(FakeSensor("front", ds))
    tracker = FakeTracker()
    s.tracker = tracker
    result = s.update(ego(), [], 0)
    assert tracker.process_calls == 1 and tracker.received == [ds]
    assert result.tracks == tracker.result


def test_update_stores_and_returns_same_perception_snapshot():
    s = suite()
    s.tracker = FakeTracker()
    result = s.update(ego(), [], 7)
    assert result is s.latest_perception
    assert result.filtered_obstacles == []
    assert result.uss_scans == {}


def test_all_configured_sensors_receive_same_filtered_obstacles():
    sensors = [FakeSensor(n) for n in ("front", "rear", "left", "right")]
    s = suite(*sensors)
    s.tracker = FakeTracker()
    obs = FakeObstacle(1, 1, 1)
    s.update(ego(), [obs], 0)
    for sensor in sensors:
        assert sensor.scan_calls[0][1] == [obs]
        assert sensor.detection_calls[0][1] == [obs]


# Lane-change safety

def test_lane_change_safe_when_no_confirmed_tracks():
    s = suite()
    result = s.is_lane_change_safe_from_tracks(ego(), 3.5)
    assert isinstance(result, LaneClearanceResult)
    assert result.is_safe is True and bool(result) is True


def test_lane_change_unsafe_for_confirmed_track_in_target_lane_window():
    s = suite()
    s.tracker.tracks = [FakeTrack(1, 10, 3.5)]
    result = s.is_lane_change_safe_from_tracks(
        ego(), 3.5, safety_gap_front=20, safety_gap_rear=10
    )
    assert result.is_safe is False


@pytest.mark.parametrize("state", [TrackState.TENTATIVE, TrackState.DELETED])
def test_lane_change_ignores_nonconfirmed_tracks(state):
    s = suite()
    s.tracker.tracks = [FakeTrack(1, 5, 3.5, state)]
    assert s.is_lane_change_safe_from_tracks(ego(), 3.5).is_safe is True


def test_lane_change_uses_ego_road_frame_vectors():
    s = suite()
    # Ego forward is world +Y; ego lateral is world -X.
    e = ego(10, 10, u=(0, 1), n=(-1, 0))
    s.tracker.tracks = [FakeTrack(1, 6.5, 15)]
    result = s.is_lane_change_safe_from_tracks(
        e, 3.5, safety_gap_front=10, safety_gap_rear=5
    )
    assert result.is_safe is False


@pytest.mark.parametrize("x,y,expected", [
    (20, 3.5, False), (-10, 3.5, False),
    (20.01, 3.5, True), (-10.01, 3.5, True),
    (5, 4.5, False), (5, 4.51, True),
])
def test_lane_change_longitudinal_and_lateral_boundaries(x, y, expected):
    s = suite()
    s.tracker.tracks = [FakeTrack(1, x, y)]
    result = s.is_lane_change_safe_from_tracks(
        ego(), 3.5, safety_gap_front=20, safety_gap_rear=10, lane_tolerance=1
    )
    assert result.is_safe is expected


def test_lane_change_is_unsafe_if_any_confirmed_track_is_in_window():
    s = suite()
    s.tracker.tracks = [
        FakeTrack(1, 100, 3.5), FakeTrack(2, 5, 3.5), FakeTrack(3, 2, -3.5)
    ]
    assert not s.is_lane_change_safe_from_tracks(
        ego(), 3.5, safety_gap_front=20, safety_gap_rear=10
    ).is_safe


def test_current_lane_clearance_flags_mirror_track_based_result():
    s = suite()
    s.tracker.tracks = [FakeTrack(1, 5, 3.5)]
    result = s.is_lane_change_safe_from_tracks(ego(), 3.5)
    # Current implementation derives both flags from the same lane_clear value.
    assert result.radar_clear is result.is_safe
    assert result.uss_clear is result.is_safe


# Reset and serialization/logging

def test_clear_tracking_resets_perception_snapshot_and_track_list():
    s = suite()
    s.latest_perception = PerceptionState(lead_target=(1, 2), tracks=[FakeTrack()])
    s.tracker.tracks = [FakeTrack(1), FakeTrack(2)]
    s.clear_tracking()
    assert s.latest_perception.lead_target is None
    assert s.latest_perception.tracks == []
    assert s.tracker.tracks == []


def test_clear_tracking_preserves_tracker_object_and_next_id():
    s = suite()
    tracker = s.tracker
    tracker._next_track_id = 17
    s.clear_tracking()
    assert s.tracker is tracker and s.tracker._next_track_id == 17


def test_get_log_data_serializes_all_configured_sensors_and_tracks():
    sensors = [FakeSensor(n) for n in ("front", "rear", "left", "right")]
    s = suite(*sensors)
    active, confirmed = FakeTrack(1, state=TrackState.TENTATIVE), FakeTrack(2)
    s.tracker.tracks = [active, confirmed]
    data = s.get_log_data()
    assert data["front_radar"] == {"sensor": "front"}
    # The expected key spelling is part of the log-data contract.
    assert data["rear_radar"] == {"sensor": "rear"}
    assert data["left_uss"] == {"sensor": "left"}
    assert data["right_uss"] == {"sensor": "right"}
    assert data["active_tracks"] == [{"track_id": 1}, {"track_id": 2}]
    assert data["confirmed_tracks"] == [{"track_id": 2}]


def test_get_log_data_invokes_each_sensor_serializer_once():
    sensors = [FakeSensor(n) for n in ("front", "rear", "left", "right")]
    suite(*sensors).get_log_data()
    assert [x.serialize_calls for x in sensors] == [1, 1, 1, 1]


def test_get_log_data_handles_missing_optional_sensors():
    # rear_radar, USS sensors are optional in SensorSuite.__init__; logging
    # should therefore represent absent devices rather than dereference None.
    data = suite(FakeSensor("front")).get_log_data()
    assert data["front_radar"] == {"sensor": "front"}
    assert data["rear_radar"] is None
    assert data["left_uss"] is None
    assert data["right_uss"] is None
