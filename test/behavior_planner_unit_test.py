"""Comprehensive unit tests for the uploaded src/behavior_planner.py.

Run from repository root: pytest -q tests/test_behavior_planner_unit.py
Dependencies are replaced with fakes so these tests isolate BehaviorPlanner.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pytest

import src.behavior_planner as behavior_planner_module
from src.behavior_planner import BehaviorPlanner
from src.config import BP, MAP


class FakeMapModule:
    def __init__(self, merge_distance=float("inf"), path=None):
        self.merge_distance = merge_distance
        self.path = np.asarray(path if path is not None else [[100., 100.], [101., 100.]], dtype=float)
        self.merge_calls, self.path_calls = [], []
    def get_distance_to_next_merge(self, ego):
        self.merge_calls.append(ego)
        return self.merge_distance
    def extract_target_lanelet_path(self, ego):
        self.path_calls.append(ego)
        return self.path


class FakeEgo:
    def __init__(self, x=0., y=0., velocity=10., orientation=0.):
        self.x, self.y, self.velocity = float(x), float(y), float(velocity)
        self.position = np.array([self.x, self.y], dtype=float)
        c, s = np.cos(orientation), np.sin(orientation)
        self._u, self._n = np.array([c, s]), np.array([-s, c])
    @property
    def road_frame_vectors(self):
        return self._u.copy(), self._n.copy()


class FakeTrack:
    def __init__(self, track_id, position):
        self.track_id, self.position = track_id, np.asarray(position, dtype=float)
    def serialize(self):
        return {"track_id": self.track_id, "position": self.position.tolist()}


class FakeSensorSuite:
    def __init__(self, tracks=None, is_safe=True):
        self.tracked_objects = list(tracks or [])
        self.is_safe = is_safe
        self.latest_perception = SimpleNamespace(lead_target=None, merge_hazard=None)
        self.clearance_calls = []
    def is_lane_change_safe_from_tracks(self, ego, target_offset, safety_gap_front,
                                        safety_gap_rear, lane_tolerance):
        self.clearance_calls.append({"ego": ego, "target_offset": target_offset,
            "safety_gap_front": safety_gap_front, "safety_gap_rear": safety_gap_rear,
            "lane_tolerance": lane_tolerance})
        return SimpleNamespace(is_safe=self.is_safe)


def track_at(ego, longitudinal, lateral, track_id=1):
    u, n = ego.road_frame_vectors
    return FakeTrack(track_id, ego.position + longitudinal*u + lateral*n)


def make_planner(*, mode="GAP_SEARCH", target_offset=3.5, start_distance=0.,
                 merge_distance=float("inf"), map_path=None):
    m = FakeMapModule(merge_distance, map_path)
    return BehaviorPlanner(m, mode, target_offset, start_distance), m


def assert_clearance_call(call, ego, offset):
    assert call["ego"] is ego
    assert call["target_offset"] == pytest.approx(offset)
    assert call["safety_gap_front"] == BP.DEFAULT_SAFETY_GAP_FRONT
    assert call["safety_gap_rear"] == BP.DEFAULT_SAFETY_GAP_REAR
    assert call["lane_tolerance"] == pytest.approx(MAP.DEFAULT_LANE_WIDTH_METERS / 2)


# Construction

def test_constructor_defaults_and_initial_state():
    m = FakeMapModule(); p = BehaviorPlanner(m)
    assert p.map_module is m
    assert (p.mode, p.target_offset, p.start_distance, p.state) == ("GAP_SEARCH", 0., 0., "LANE_KEEP")
    assert p.start_x is p.start_y is p.lane_change_start_pos is None
    assert p._cached_lead_track is p._cached_merge_hazard is p._cached_selected_lead is None

@pytest.mark.parametrize("mode", ["GAP_SEARCH", "MAP_FOLLOW"])
def test_constructor_accepts_supported_modes(mode):
    p, _ = make_planner(mode=mode)
    assert p.mode == mode

@pytest.mark.parametrize("mode", ["", "map_follow", "LANE_KEEP", "UNKNOWN", None])
def test_constructor_rejects_unsupported_modes(mode):
    with pytest.raises(ValueError, match="mode must be either 'GAP_SEARCH' or 'MAP_FOLLOW'"):
        BehaviorPlanner(FakeMapModule(), mode=mode)

def test_constructor_preserves_supplied_configuration():
    p, _ = make_planner(target_offset=-3.8, start_distance=25.)
    assert p.target_offset == -3.8 and p.start_distance == 25.


# Lead-track selection

def test_get_lead_track_returns_none_for_empty_track_list():
    p, _ = make_planner()
    assert p.get_lead_track(FakeEgo(), FakeSensorSuite()) is None

def test_get_lead_track_selects_nearest_valid_track_ahead():
    p, _ = make_planner(); ego = FakeEgo()
    far, near = track_at(ego, 30, 0, 1), track_at(ego, 12, 0, 2)
    assert p.get_lead_track(ego, FakeSensorSuite([far, near])) is near
    assert p._cached_lead_track is near

def test_get_lead_track_ignores_tracks_behind_ego():
    p, _ = make_planner(); ego = FakeEgo()
    assert p.get_lead_track(ego, FakeSensorSuite([track_at(ego, -5, 0), track_at(ego, -.01, 0, 2)])) is None

def test_get_lead_track_ignores_tracks_outside_lateral_corridor():
    p, _ = make_planner(); ego = FakeEgo()
    d = BP.MERGE_HAZARD_LAT_RANGE_METERS + .01
    assert p.get_lead_track(ego, FakeSensorSuite([track_at(ego, 10, d), track_at(ego, 20, -d, 2)])) is None

def test_get_lead_track_includes_exact_lateral_boundary():
    p, _ = make_planner(); ego = FakeEgo()
    t = track_at(ego, 10, BP.MERGE_HAZARD_LAT_RANGE_METERS)
    assert p.get_lead_track(ego, FakeSensorSuite([t])) is t

def test_get_lead_track_uses_ego_road_frame_for_rotated_vehicle():
    p, _ = make_planner(); ego = FakeEgo(orientation=np.pi/2)
    ahead, beside = track_at(ego, 10, 0), track_at(ego, 0, 10, 2)
    assert p.get_lead_track(ego, FakeSensorSuite([beside, ahead])) is ahead

def test_get_lead_track_chooses_nearest_of_multiple_valid_tracks():
    p, _ = make_planner(); ego = FakeEgo()
    tracks = [track_at(ego, 25, .5, 1), track_at(ego, 8, -.5, 2), track_at(ego, 14, 0, 3)]
    assert p.get_lead_track(ego, FakeSensorSuite(tracks)) is tracks[1]


# Merge-hazard selection

def test_get_merge_hazard_returns_none_for_empty_track_list():
    p, _ = make_planner()
    assert p.get_merge_hazard_track(FakeEgo(), FakeSensorSuite()) is None

def test_get_merge_hazard_selects_nearest_candidate_in_zero_offset_corridor():
    p, _ = make_planner(); ego = FakeEgo(velocity=20)
    far = track_at(ego, 25, MAP.DEFAULT_LANE_WIDTH_METERS*.8, 1)
    near = track_at(ego, 12, -MAP.DEFAULT_LANE_WIDTH_METERS*.8, 2)
    s = FakeSensorSuite([far, near])
    assert p.get_merge_hazard_track(ego, s, 0.) is near
    assert p._cached_merge_hazard is near and s.latest_perception.merge_hazard is near

def test_get_merge_hazard_accepts_track_in_target_offset_corridor():
    p, _ = make_planner(); ego = FakeEgo(velocity=20); t = track_at(ego, 15, 3.5)
    assert p.get_merge_hazard_track(ego, FakeSensorSuite([t]), 3.5) is t

def test_get_merge_hazard_rejects_track_outside_target_offset_corridor():
    p, _ = make_planner(); ego = FakeEgo(velocity=20)
    d = 3.5 + MAP.DEFAULT_LANE_WIDTH_METERS/2 + .01
    assert p.get_merge_hazard_track(ego, FakeSensorSuite([track_at(ego, 15, d)]), 3.5) is None

def test_get_merge_hazard_zero_offset_uses_full_lane_width():
    p, _ = make_planner(); ego = FakeEgo(velocity=20)
    t = track_at(ego, 15, MAP.DEFAULT_LANE_WIDTH_METERS*.9)
    assert p.get_merge_hazard_track(ego, FakeSensorSuite([t]), 0.) is t

def test_get_merge_hazard_nonzero_offset_uses_half_lane_width():
    p, _ = make_planner(); ego = FakeEgo(velocity=20)
    d = 3.5 + MAP.DEFAULT_LANE_WIDTH_METERS/2 - .01
    t = track_at(ego, 15, d)
    assert p.get_merge_hazard_track(ego, FakeSensorSuite([t]), 3.5) is t

def test_get_merge_hazard_excludes_track_at_rear_range_boundary():
    p, _ = make_planner(); ego = FakeEgo(velocity=20)
    t = track_at(ego, BP.MERGE_HAZARD_REAR_RANGE_METERS, 0)
    assert p.get_merge_hazard_track(ego, FakeSensorSuite([t]), 0.) is None

def test_get_merge_hazard_excludes_track_at_ttc_range_boundary():
    p, _ = make_planner(); ego = FakeEgo(velocity=10)
    t = track_at(ego, ego.velocity*BP.TTC_MAX, 0)
    assert p.get_merge_hazard_track(ego, FakeSensorSuite([t]), 0.) is None

def test_get_merge_hazard_includes_track_behind_ego_within_rear_range():
    p, _ = make_planner()
    ego = FakeEgo(velocity=20)
    t = track_at(ego, -1, 0)

    assert p.get_merge_hazard_track(
        ego, FakeSensorSuite([t]), 0.
    ) is t

def test_get_merge_hazard_uses_ego_road_frame_when_rotated():
    p, _ = make_planner(); ego = FakeEgo(velocity=20, orientation=np.pi/2)
    t = track_at(ego, 15, 0)
    assert p.get_merge_hazard_track(ego, FakeSensorSuite([t]), 0.) is t


# Selection between ordinary lead and merge hazard

def test_select_lead_returns_primary_when_merge_is_far():
    p, m = make_planner(merge_distance=float("inf")); ego = FakeEgo()
    lead = track_at(ego, 10, 0); s = FakeSensorSuite([lead])
    assert p.select_lead_track(ego, s) is lead
    assert m.merge_calls == [ego]
    assert s.latest_perception.lead_target is lead and p._cached_selected_lead is lead

def test_select_lead_checks_merge_at_exact_threshold():
    p, _ = make_planner(merge_distance=BP.LANE_MERGE_LOOKAHED_DISTANCE_METERS)
    ego = FakeEgo(velocity=20); hazard = track_at(ego, 15, 0); s = FakeSensorSuite([hazard])
    assert p.select_lead_track(ego, s) is hazard

def test_select_lead_prefers_merge_hazard_even_if_primary_is_closer():
    p, _ = make_planner(merge_distance=1., target_offset=0.); ego = FakeEgo(velocity=20)
    primary, hazard = track_at(ego, 5, 0, 1), track_at(ego, 15, .5, 2)
    s = FakeSensorSuite([primary, hazard])
    assert p.select_lead_track(ego, s) is hazard
    assert s.latest_perception.lead_target is hazard and p._cached_selected_lead is hazard

def test_select_lead_falls_back_to_primary_when_no_merge_hazard(monkeypatch):
    p, _ = make_planner(merge_distance=1., target_offset=0.); ego = FakeEgo(velocity=20)
    lead = track_at(ego, 8, 0); s = FakeSensorSuite([lead])
    monkeypatch.setattr(p, "get_merge_hazard_track", lambda *args, **kwargs: None)
    assert p.select_lead_track(ego, s) is lead
    assert s.latest_perception.lead_target is lead

def test_select_lead_sets_selected_lead_none_when_no_candidates():
    p, _ = make_planner(); s = FakeSensorSuite()
    assert p.select_lead_track(FakeEgo(), s) is None
    assert s.latest_perception.lead_target is None and p._cached_selected_lead is None


# update_plan: MAP_FOLLOW and merge monitoring

def test_update_plan_map_follow_returns_map_path():
    p, m = make_planner(mode="MAP_FOLLOW"); ego = FakeEgo(); s = FakeSensorSuite()
    current = np.array([[0., 0.], [1., 0.]])
    state, path = p.update_plan(ego, 1, s, current)
    assert state == "LANE_KEEP"; np.testing.assert_array_equal(path, m.path)
    assert m.path_calls == [ego] and p.start_x is None and p.start_y is None

def test_map_follow_skips_clearance_when_not_near_merge():
    p, _ = make_planner(mode="MAP_FOLLOW"); s = FakeSensorSuite()
    p.update_plan(FakeEgo(), 1, s, np.zeros((2, 2)))
    assert s.clearance_calls == [] and p.is_checking_merge is False

def test_update_plan_checks_merge_clearance_at_time_threshold():
    ego = FakeEgo(velocity=10); d = ego.velocity*BP.MERGE_TIME_MAX
    p, _ = make_planner(mode="MAP_FOLLOW", target_offset=-3.5, merge_distance=d)
    s = FakeSensorSuite(is_safe=False)
    p.update_plan(ego, 7, s, np.zeros((2, 2)))
    assert p.is_checking_merge is True and len(s.clearance_calls) == 1
    assert_clearance_call(s.clearance_calls[0], ego, -3.5)

def test_update_plan_skips_merge_clearance_just_outside_time_threshold():
    ego = FakeEgo(velocity=10); d = ego.velocity*BP.MERGE_TIME_MAX + .01
    p, _ = make_planner(mode="MAP_FOLLOW", merge_distance=d); s = FakeSensorSuite()
    p.update_plan(ego, 1, s, np.zeros((2, 2)))
    assert p.is_checking_merge is False and s.clearance_calls == []


# update_plan: GAP_SEARCH distance gate and initiation

def test_gap_search_initializes_start_position_and_returns_current_path():
    p, _ = make_planner(start_distance=100); ego = FakeEgo(12, -4); s = FakeSensorSuite()
    current = np.array([[0., 0.], [1., 0.]])
    state, path = p.update_plan(ego, 1, s, current)
    assert state == "LANE_KEEP"; np.testing.assert_array_equal(path, current)
    assert p.start_x == 12 and p.start_y == -4 and s.clearance_calls == []

def test_distance_gate_blocks_lane_change_before_start_distance():
    p, _ = make_planner(start_distance=10); s = FakeSensorSuite(is_safe=True); current = np.zeros((2, 2))
    p.update_plan(FakeEgo(x=0), 1, s, current)
    state, path = p.update_plan(FakeEgo(x=9.99), 2, s, current)
    assert state == "LANE_KEEP"; np.testing.assert_array_equal(path, current)
    assert s.clearance_calls == []

def test_distance_gate_allows_clearance_check_at_exact_start_distance():
    p, _ = make_planner(start_distance=10); s = FakeSensorSuite(is_safe=False); current = np.zeros((2, 2))
    p.update_plan(FakeEgo(x=0), 1, s, current)
    p.update_plan(FakeEgo(x=10), 2, s, current)
    assert len(s.clearance_calls) == 1

def test_unsafe_gap_does_not_initiate_lane_change():
    p, _ = make_planner(); ego = FakeEgo(); s = FakeSensorSuite(is_safe=False); current = np.zeros((2, 2))
    state, path = p.update_plan(ego, 1, s, current)
    assert state == "LANE_KEEP"; np.testing.assert_array_equal(path, current)
    assert p.lane_change_start_pos is None and len(s.clearance_calls) == 1
    assert_clearance_call(s.clearance_calls[0], ego, p.target_offset)

def test_safe_gap_initiates_lane_change_and_generates_path(monkeypatch):
    p, _ = make_planner(target_offset=3.25); ego = FakeEgo(4, 2); s = FakeSensorSuite(is_safe=True)
    generated = np.array([[4., 2.], [5., 3.], [6., 4.]])
    calls = {}
    def fake_generator(**kwargs):
        calls.update(kwargs); return generated
    monkeypatch.setattr(behavior_planner_module, "generate_lane_change_path", fake_generator)
    state, path = p.update_plan(ego, 12, s, np.zeros((2, 2)))
    assert state == "LANE_CHANGE"; np.testing.assert_array_equal(path, generated)
    np.testing.assert_array_equal(p.lane_change_start_pos, ego.position)
    assert calls["ego"] is ego and calls["target_lane_offset"] == 3.25
    assert calls["total_length"] == BP.MANEUVER_PATH_LENGTH_METERS

def test_lane_change_generator_is_not_called_for_unsafe_gap(monkeypatch):
    p, _ = make_planner(); s = FakeSensorSuite(is_safe=False)
    def unexpected(**kwargs): pytest.fail("generator called for unsafe gap")
    monkeypatch.setattr(behavior_planner_module, "generate_lane_change_path", unexpected)
    assert p.update_plan(FakeEgo(), 1, s, np.zeros((2, 2)))[0] == "LANE_KEEP"

def test_clearance_call_receives_configured_gaps_and_lane_tolerance():
    p, _ = make_planner(target_offset=-3.6); ego = FakeEgo(); s = FakeSensorSuite(is_safe=False)
    p.update_plan(ego, 1, s, np.zeros((2, 2)))
    assert_clearance_call(s.clearance_calls[0], ego, -3.6)

def test_start_distance_gate_uses_euclidean_displacement():
    p, _ = make_planner(start_distance=5); s = FakeSensorSuite(is_safe=False); current = np.zeros((2, 2))
    p.update_plan(FakeEgo(0, 0), 1, s, current)
    p.update_plan(FakeEgo(3, 4), 2, s, current)
    assert len(s.clearance_calls) == 1


# update_plan: lane-change execution and completion

def planner_already_changing(offset=3.5, start=(0., 0.)):
    p, m = make_planner(target_offset=offset)
    p.state = "LANE_CHANGE"; p.lane_change_start_pos = np.asarray(start, dtype=float)
    return p, m

def test_lane_change_in_progress_keeps_current_path():
    p, m = planner_already_changing(); ego = FakeEgo(5, 1); current = np.array([[5., 0.], [6., 1.]])
    state, path = p.update_plan(ego, 3, FakeSensorSuite(), current)
    assert state == "LANE_CHANGE"; np.testing.assert_array_equal(path, current)
    assert m.path_calls == []

def test_lane_change_completes_at_positive_offset_threshold():
    offset = 3.5; p, m = planner_already_changing(offset)
    ego = FakeEgo(10, BP.LANE_CHANGE_COMPLETION_RATIO*offset)
    state, path = p.update_plan(ego, 20, FakeSensorSuite(), np.zeros((2, 2)))
    assert state == "LANE_KEEP" and p.mode == "MAP_FOLLOW" and p.target_offset == 0
    np.testing.assert_array_equal(path, m.path); assert m.path_calls == [ego]

def test_lane_change_completes_for_negative_target_offset():
    offset = -3.5; p, m = planner_already_changing(offset)
    ego = FakeEgo(10, BP.LANE_CHANGE_COMPLETION_RATIO*abs(offset))
    state, path = p.update_plan(ego, 21, FakeSensorSuite(), np.zeros((2, 2)))
    assert state == "LANE_KEEP" and p.mode == "MAP_FOLLOW" and p.target_offset == 0
    np.testing.assert_array_equal(path, m.path)

def test_lane_change_does_not_complete_just_below_threshold():
    offset = 3.5; p, m = planner_already_changing(offset)
    ego = FakeEgo(10, BP.LANE_CHANGE_COMPLETION_RATIO*offset - .01); current = np.zeros((2, 2))
    state, path = p.update_plan(ego, 22, FakeSensorSuite(), current)
    assert state == "LANE_CHANGE" and p.mode == "GAP_SEARCH" and p.target_offset == offset
    np.testing.assert_array_equal(path, current); assert m.path_calls == []

def test_lane_change_completion_uses_absolute_lateral_progress():
    offset = 3.5; p, m = planner_already_changing(offset)
    ego = FakeEgo(10, -BP.LANE_CHANGE_COMPLETION_RATIO*offset)
    state, path = p.update_plan(ego, 23, FakeSensorSuite(), np.zeros((2, 2)))
    assert state == "LANE_KEEP" and p.mode == "MAP_FOLLOW"
    np.testing.assert_array_equal(path, m.path)

def test_lane_change_completion_projects_progress_onto_current_road_normal():
    offset = 3.5; p, m = planner_already_changing(offset)
    progress = BP.LANE_CHANGE_COMPLETION_RATIO*offset
    ego = FakeEgo(x=-progress, y=0, orientation=np.pi/2)
    state, path = p.update_plan(ego, 24, FakeSensorSuite(), np.zeros((2, 2)))
    assert state == "LANE_KEEP" and p.mode == "MAP_FOLLOW"
    np.testing.assert_array_equal(path, m.path)

def test_lane_change_completion_does_not_run_gap_clearance():
    p, _ = planner_already_changing(); ego = FakeEgo(y=BP.LANE_CHANGE_COMPLETION_RATIO*3.5); s = FakeSensorSuite()
    p.update_plan(ego, 25, s, np.zeros((2, 2)))
    assert s.clearance_calls == []


# Logging / observability

def test_get_log_data_contains_state_and_configuration():
    p, _ = make_planner(target_offset=-3.5, start_distance=20)
    p.start_x, p.start_y = 1., 2.; p.lane_change_start_pos = np.array([3., 4.])
    d = p.get_log_data()
    assert d["mode"] == "GAP_SEARCH" and d["state"] == "LANE_KEEP"
    assert d["target_offset"] == -3.5 and d["start_distance"] == 20
    assert d["start_x"] == 1 and d["start_y"] == 2
    np.testing.assert_array_equal(d["lane_change_start_pos"], [3., 4.])

def test_get_log_data_returns_none_for_uncached_tracks():
    d = make_planner()[0].get_log_data()
    assert d["lead_track"] is d["merge_hazard"] is d["selected_lead"] is None

def test_get_log_data_serializes_cached_tracks():
    p, _ = make_planner(); lead = FakeTrack(11, [10, 1]); hazard = FakeTrack(12, [15, 3.5]); selected = FakeTrack(13, [8, 0])
    p._cached_lead_track, p._cached_merge_hazard, p._cached_selected_lead = lead, hazard, selected
    d = p.get_log_data()
    assert d["lead_track"] == {"track_id": 11, "position": [10., 1.]}
    assert d["merge_hazard"] == {"track_id": 12, "position": [15., 3.5]}
    assert d["selected_lead"] == {"track_id": 13, "position": [8., 0.]}
