"""Comprehensive unit tests for src.map.MapModule.

Run from the repository root:
    pytest -q tests/map_unit_test.py

Tests use small CommonRoad-like fakes to isolate map algorithms.
"""
import sys
from types import SimpleNamespace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import numpy as np
import pytest

import src.map as map_module
from src.map import (
    MapModule,
    extract_target_lanelet_path,
    get_current_lane_width,
    get_road_heading_at_position,
)


class FakeLanelet:
    def __init__(self, lanelet_id, center_vertices, *, left_vertices=None,
                 right_vertices=None, predecessor=None, successor=None):
        self.lanelet_id = lanelet_id
        self.center_vertices = np.asarray(center_vertices, dtype=float)
        self.left_vertices = (np.asarray(left_vertices, dtype=float)
                              if left_vertices is not None
                              else self.center_vertices + [0.0, 1.75])
        self.right_vertices = (np.asarray(right_vertices, dtype=float)
                               if right_vertices is not None
                               else self.center_vertices - [0.0, 1.75])
        self.predecessor = list(predecessor or [])
        self.successor = list(successor or [])


class FakeLaneletNetwork:
    def __init__(self, lanelets, position_results=None):
        self.lanelets = list(lanelets)
        self._by_id = {lane.lanelet_id: lane for lane in self.lanelets}
        self.position_results = position_results

    def find_lanelet_by_position(self, positions):
        if self.position_results is not None:
            return self.position_results
        return [[self.lanelets[0].lanelet_id]] if self.lanelets else [[]]

    def find_lanelet_by_id(self, lanelet_id):
        return self._by_id.get(lanelet_id)


class FakeScenario:
    def __init__(self, lanelet_network):
        self.lanelet_network = lanelet_network


def ego_at(x, y):
    return SimpleNamespace(position=np.asarray([x, y], dtype=float))


def make_module(lanelets, position_results=None, planning_problem_set=None):
    network = FakeLaneletNetwork(lanelets, position_results)
    return MapModule(FakeScenario(network), planning_problem_set), network


def make_goal_set(*states):
    return SimpleNamespace(planning_problem_dict={
        1: SimpleNamespace(goal=SimpleNamespace(state_list=list(states)))
    })


def goal_by_lanelet_ids(*ids):
    return SimpleNamespace(position=SimpleNamespace(lanelet_id=list(ids)))


def goal_by_center(center):
    return SimpleNamespace(position=SimpleNamespace(center=np.asarray(center, dtype=float)))


def straight_lane(lanelet_id, x0, x1, y=0.0, successor=None, predecessor=None):
    return FakeLanelet(lanelet_id, [[x0, y], [x1, y]],
                       successor=successor, predecessor=predecessor)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_constructor_retains_scenario_network_and_planning_problem_set():
    scenario = FakeScenario(FakeLaneletNetwork([straight_lane(1, 0, 10)]))
    pps = object()
    module = MapModule(scenario, pps)
    assert module.scenario is scenario
    assert module.lanelet_network is scenario.lanelet_network
    assert module.planning_problem_set is pps


def test_constructor_allows_missing_planning_problem_set():
    module, _ = make_module([straight_lane(1, 0, 10)])
    assert module.planning_problem_set is None


# ---------------------------------------------------------------------------
# Distance to next merge
# ---------------------------------------------------------------------------

def test_merge_distance_is_infinite_when_ego_is_not_on_a_lanelet():
    module, _ = make_module([straight_lane(1, 0, 10)], position_results=[[]])
    assert module.get_distance_to_next_merge(ego_at(2, 0)) == float("inf")


def test_merge_distance_is_infinite_when_start_lanelet_cannot_be_resolved():
    module, _ = make_module([straight_lane(1, 0, 10)], position_results=[[999]])
    assert module.get_distance_to_next_merge(ego_at(2, 0)) == float("inf")


def test_merge_distance_accepts_nested_lanelet_id_result():
    lane = FakeLanelet(1, [[0, 0], [10, 0]], predecessor=[2, 3])
    module, _ = make_module([lane], position_results=[[(1,)]])
    assert module.get_distance_to_next_merge(ego_at(2, 0)) == pytest.approx(10)


def test_merge_detected_on_current_lanelet_with_multiple_predecessors():
    lane = FakeLanelet(1, [[0, 0], [5, 0], [10, 0]], predecessor=[2, 3])
    module, _ = make_module([lane])
    assert module.get_distance_to_next_merge(ego_at(2, 0)) == pytest.approx(10)


def test_merge_distance_accumulates_from_nearest_vertex_not_projection():
    lane = FakeLanelet(1, [[0, 0], [5, 0], [10, 0]], predecessor=[2, 3])
    module, _ = make_module([lane])
    assert module.get_distance_to_next_merge(ego_at(4.8, 0.2)) == pytest.approx(5)


def test_merge_detected_when_successor_has_multiple_predecessors():
    start = FakeLanelet(1, [[0, 0], [5, 0]], successor=[2])
    merge = FakeLanelet(2, [[5, 0], [10, 0]], predecessor=[1, 3])
    other = FakeLanelet(3, [[5, 5], [10, 0]], successor=[2])
    module, _ = make_module([start, merge, other])
    assert module.get_distance_to_next_merge(ego_at(0, 0)) == pytest.approx(5)


def test_merge_checks_all_immediate_successors_for_convergence():
    first = FakeLanelet(1, [[0, 0], [5, 0]], successor=[2, 3])
    second = FakeLanelet(2, [[5, 0], [10, 0]], predecessor=[1])
    third = FakeLanelet(3, [[5, 2], [10, 2]], predecessor=[1, 4])
    module, _ = make_module([first, second, third])
    assert module.get_distance_to_next_merge(ego_at(0, 0)) == pytest.approx(5)


def test_merge_distance_is_infinite_when_no_merge_exists():
    first = FakeLanelet(1, [[0, 0], [5, 0]], successor=[2])
    second = FakeLanelet(2, [[5, 0], [10, 0]], predecessor=[1])
    module, _ = make_module([first, second])
    assert module.get_distance_to_next_merge(ego_at(0, 0)) == float("inf")


def test_merge_outside_lookahead_horizon_is_not_reported():
    first = FakeLanelet(1, [[0, 0], [5, 0]], successor=[2])
    merge = FakeLanelet(2, [[5, 0], [10, 0]], predecessor=[1, 3])
    module, _ = make_module([first, merge])
    assert module.get_distance_to_next_merge(ego_at(0, 0), 4.9) == float("inf")


def test_merge_exactly_at_horizon_boundary_is_not_examined():
    first = FakeLanelet(1, [[0, 0], [5, 0]], successor=[2])
    merge = FakeLanelet(2, [[5, 0], [10, 0]], predecessor=[1, 3])
    module, _ = make_module([first, merge])
    # The traversal condition is dist_accum < horizon_meters.
    assert module.get_distance_to_next_merge(ego_at(0, 0), 5.0) == float("inf")


def test_merge_just_inside_horizon_is_reported():
    first = FakeLanelet(1, [[0, 0], [5, 0]], successor=[2])
    merge = FakeLanelet(2, [[5, 0], [10, 0]], predecessor=[1, 3])
    module, _ = make_module([first, merge])
    assert module.get_distance_to_next_merge(ego_at(0, 0), 5.01) == pytest.approx(5)


def test_merge_traversal_stops_on_successor_cycle():
    first = FakeLanelet(1, [[0, 0], [5, 0]], successor=[2])
    second = FakeLanelet(2, [[5, 0], [10, 0]], successor=[1], predecessor=[1])
    module, _ = make_module([first, second])
    assert module.get_distance_to_next_merge(ego_at(0, 0)) == float("inf")


def test_merge_distance_is_zero_when_ego_is_at_final_vertex_of_merge_lane():
    lane = FakeLanelet(1, [[0, 0], [10, 0]], predecessor=[2, 3])
    module, _ = make_module([lane])
    assert module.get_distance_to_next_merge(ego_at(10, 0)) == pytest.approx(0)


# ---------------------------------------------------------------------------
# Road heading
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("vertices,position,expected", [
    ([[0, 0], [10, 0]], [2, 1], 0),
    ([[0, 0], [0, 10]], [0, 2], np.pi / 2),
    ([[0, 0], [10, 10]], [2, 2], np.pi / 4),
    ([[10, 0], [0, 0]], [5, 0], np.pi),
    ([[0, 10], [0, 0]], [0, 5], -np.pi / 2),
])
def test_heading_matches_nearest_centerline_segment(vertices, position, expected):
    module, _ = make_module([FakeLanelet(1, vertices)])
    assert module.get_road_heading_at_position(position) == pytest.approx(expected)


def test_heading_returns_none_when_no_lanelet_is_found():
    module, _ = make_module([straight_lane(1, 0, 10)], position_results=[[]])
    assert module.get_road_heading_at_position([2, 0]) is None


def test_heading_accepts_nested_lanelet_id_result():
    module, _ = make_module([FakeLanelet(7, [[0, 0], [0, 10]])],
                            position_results=[[(7,)]])
    assert module.get_road_heading_at_position([0, 3]) == pytest.approx(np.pi / 2)


def test_heading_uses_previous_segment_at_final_vertex():
    module, _ = make_module([FakeLanelet(1, [[0, 0], [5, 0], [5, 5]])])
    assert module.get_road_heading_at_position([5, 5]) == pytest.approx(np.pi / 2)


def test_heading_uses_nearest_vertex_segment_selection():
    module, _ = make_module([FakeLanelet(1, [[0, 0], [5, 0], [5, 5]])])
    assert module.get_road_heading_at_position([4.9, 0.1]) == pytest.approx(np.pi / 2)


# ---------------------------------------------------------------------------
# Lane width
# ---------------------------------------------------------------------------

def test_lane_width_is_mean_corresponding_boundary_distance():
    lane = FakeLanelet(1, [[0, 0], [5, 0], [10, 0]],
                       left_vertices=[[0, 2], [5, 3], [10, 4]],
                       right_vertices=[[0, 0], [5, 0], [10, 0]])
    module, _ = make_module([lane])
    assert module.get_current_lane_width(ego_at(2, 0)) == pytest.approx(3)


def test_lane_width_returns_default_when_no_lanelet_is_found():
    module, _ = make_module([straight_lane(1, 0, 10)], position_results=[[]])
    assert module.get_current_lane_width(ego_at(2, 0)) == pytest.approx(3.5)
    assert module.get_current_lane_width(ego_at(2, 0), 4.2) == pytest.approx(4.2)


def test_lane_width_averages_nonuniform_widths():
    lane = FakeLanelet(1, [[0, 0], [1, 0], [2, 0]],
                       left_vertices=[[0, 2], [1, 4], [2, 6]],
                       right_vertices=[[0, 0], [1, 0], [2, 0]])
    module, _ = make_module([lane])
    assert module.get_current_lane_width(ego_at(1, 0)) == pytest.approx(4)


# ---------------------------------------------------------------------------
# Target path extraction and route selection
# ---------------------------------------------------------------------------

def test_path_follows_successor_chain_without_planning_problem():
    lanes = [
        straight_lane(1, 0, 5, successor=[2]),
        straight_lane(2, 5, 10, successor=[3], predecessor=[1]),
        straight_lane(3, 10, 15, predecessor=[2]),
    ]
    module, _ = make_module(lanes)
    path = module.extract_target_lanelet_path(ego_at(0, 0), 12, 2)
    np.testing.assert_allclose(path[:, 0], [0, 2, 4, 6, 8, 10, 12, 14])
    np.testing.assert_allclose(path[:, 1], np.zeros(len(path)))


def test_path_stops_after_appending_lanelet_that_reaches_horizon():
    lanes = [
        straight_lane(1, 0, 5, successor=[2]),
        straight_lane(2, 5, 10, successor=[3], predecessor=[1]),
        straight_lane(3, 10, 20, predecessor=[2]),
    ]
    module, _ = make_module(lanes)
    path = module.extract_target_lanelet_path(ego_at(0, 0), 5, 2)
    # Lanelet 1 reaches the requested horizon, so lanelet 2 is not appended.
    np.testing.assert_allclose(path[:, 0], [0, 2, 4])


def test_path_omits_duplicate_join_vertex():
    lanes = [
        FakeLanelet(1, [[0, 0], [5, 0]], successor=[2]),
        FakeLanelet(2, [[5, 0], [5, 5]], predecessor=[1]),
    ]
    module, _ = make_module(lanes)
    path = module.extract_target_lanelet_path(ego_at(0, 0), 10, 1)
    assert len(path) == 10
    assert np.all(np.isfinite(path))


def test_resampling_step_controls_waypoint_spacing():
    module, _ = make_module([straight_lane(1, 0, 10)])
    path = module.extract_target_lanelet_path(ego_at(0, 0), 10, 0.5)
    np.testing.assert_allclose(np.diff(path[:, 0]), np.full(len(path) - 1, 0.5))
    assert path[0, 0] == pytest.approx(0)
    assert path[-1, 0] == pytest.approx(9.5)


def test_zero_horizon_produces_empty_two_column_path():
    module, _ = make_module([straight_lane(1, 0, 10)])
    path = module.extract_target_lanelet_path(ego_at(0, 0), 0, 1)
    assert path.shape == (0, 2)


def test_bfs_chooses_first_shortest_hop_route_to_goal():
    lanes = [
        straight_lane(1, 0, 1, successor=[2, 3]),
        straight_lane(2, 1, 2, y=0, successor=[4], predecessor=[1]),
        straight_lane(3, 1, 2, y=2, successor=[4], predecessor=[1]),
        straight_lane(4, 2, 3, y=0, predecessor=[2, 3]),
    ]
    module, _ = make_module(lanes, planning_problem_set=make_goal_set(goal_by_lanelet_ids(4)))
    path = module.extract_target_lanelet_path(ego_at(0, 0), 10, 0.5)
    assert np.max(np.abs(path[:, 1])) < 1e-9


def test_goal_lanelet_ids_are_collected_across_goal_states():
    lanes = [
        straight_lane(1, 0, 1, successor=[2, 3]),
        straight_lane(2, 1, 2, y=0, predecessor=[1]),
        straight_lane(3, 1, 2, y=3, predecessor=[1]),
    ]
    pps = make_goal_set(goal_by_lanelet_ids(999), goal_by_lanelet_ids(3))
    module, _ = make_module(lanes, planning_problem_set=pps)
    path = module.extract_target_lanelet_path(ego_at(0, 0), 10, 0.5)
    assert np.mean(path[:, 1]) > 1


def test_goal_center_is_resolved_to_lanelet_ids():
    lanes = [
        straight_lane(1, 0, 5, successor=[2]),
        straight_lane(2, 5, 10, predecessor=[1]),
    ]

    class GoalAwareNetwork(FakeLaneletNetwork):
        def find_lanelet_by_position(self, positions):
            return [[1]] if np.asarray(positions[0])[0] < 4 else [[2]]

    network = GoalAwareNetwork(lanes)
    module = MapModule(FakeScenario(network), make_goal_set(goal_by_center([9, 0])))
    path = module.extract_target_lanelet_path(ego_at(0, 0), 10, 1)
    assert path[-1, 0] == pytest.approx(9)


def test_path_falls_back_to_nearest_lanelet_when_position_lookup_is_empty():
    lanes = [
        straight_lane(1, 0, 5, y=0, successor=[2]),
        straight_lane(2, 5, 10, y=0, predecessor=[1]),
        straight_lane(3, 0, 5, y=10),
    ]
    module, _ = make_module(lanes, position_results=[[]])
    path = module.extract_target_lanelet_path(ego_at(1, 0.2), 8, 1)
    assert path[0, 0] == pytest.approx(0)
    assert np.max(np.abs(path[:, 1])) < 1e-9


def test_nearest_lanelet_fallback_selects_nearest_centerline_vertex():
    lanes = [straight_lane(1, 0, 5, y=10), straight_lane(2, 0, 5, y=0)]
    module, _ = make_module(lanes, position_results=[[]])
    path = module.extract_target_lanelet_path(ego_at(2, 0.1), 5, 1)
    assert np.max(np.abs(path[:, 1])) < 1e-9


def test_unreachable_goal_uses_default_first_successor_chain():
    lanes = [
        straight_lane(1, 0, 5, successor=[2, 3]),
        straight_lane(2, 5, 10, y=0, predecessor=[1]),
        straight_lane(3, 5, 10, y=3, predecessor=[1]),
        straight_lane(9, 0, 5, y=10),
    ]
    module, _ = make_module(lanes, planning_problem_set=make_goal_set(goal_by_lanelet_ids(9)))
    path = module.extract_target_lanelet_path(ego_at(0, 0), 10, 1)
    assert np.max(np.abs(path[:, 1])) < 1e-9


@pytest.mark.parametrize("bad_pps", [
    object(),
    SimpleNamespace(planning_problem_dict={}),
])
def test_malformed_goal_metadata_falls_back_to_successor_route(bad_pps):
    module, _ = make_module([straight_lane(1, 0, 10)], planning_problem_set=bad_pps)
    path = module.extract_target_lanelet_path(ego_at(0, 0), 5, 1)
    assert path.shape == (10, 2)


def test_near_duplicate_centerline_vertices_are_deduplicated(monkeypatch):
    monkeypatch.setattr(map_module, "MAP", SimpleNamespace(WAYPOINT_DEDUP_TOLERANCE=1e-6))
    lane = FakeLanelet(1, [[0, 0], [1, 0], [1 + 1e-9, 0], [2, 0]])
    module, _ = make_module([lane])
    path = module.extract_target_lanelet_path(ego_at(0, 0), 3, 0.5)
    assert np.all(np.isfinite(path))
    np.testing.assert_allclose(np.diff(path[:, 0]), 0.5)


def test_curved_centerline_is_interpolated_as_two_dimensional_path():
    module, _ = make_module([FakeLanelet(1, [[0, 0], [1, 1], [2, 0]])])
    path = module.extract_target_lanelet_path(ego_at(0, 0), 4, 0.25)
    assert path.shape[1] == 2
    assert np.all(np.isfinite(path))
    assert np.max(path[:, 1]) > 0.9


def test_path_begins_at_lanelet_centerline_start_not_ego_projection():
    module, _ = make_module([straight_lane(1, 0, 10)])
    path = module.extract_target_lanelet_path(ego_at(4, 0), 10, 2)
    assert path[0, 0] == pytest.approx(0)


def test_path_output_is_float_array_with_two_columns():
    module, _ = make_module([straight_lane(1, 0, 10)])
    path = module.extract_target_lanelet_path(ego_at(0, 0), 5, 1)
    assert isinstance(path, np.ndarray)
    assert path.ndim == 2 and path.shape[1] == 2
    assert np.issubdtype(path.dtype, np.floating)


# ---------------------------------------------------------------------------
# Backward-compatible wrappers
# ---------------------------------------------------------------------------

def test_heading_functional_wrapper():
    scenario = FakeScenario(FakeLaneletNetwork([straight_lane(1, 0, 10)]))
    assert get_road_heading_at_position(scenario, [2, 0]) == pytest.approx(0)


def test_lane_width_functional_wrapper_passes_default_width():
    lane = FakeLanelet(1, [[0, 0], [10, 0]])
    scenario = FakeScenario(FakeLaneletNetwork([lane], position_results=[[]]))
    assert get_current_lane_width(scenario, ego_at(1, 0), 4.25) == pytest.approx(4.25)


def test_path_functional_wrapper_uses_configured_default_resampling_step():
    scenario = FakeScenario(FakeLaneletNetwork([straight_lane(1, 0, 10)]))
    path = extract_target_lanelet_path(scenario, ego_at(0, 0), horizon_meters=5)
    assert path.ndim == 2 and path.shape[1] == 2
    assert path[0, 0] == pytest.approx(0)
    assert path[-1, 0] == 9.5
