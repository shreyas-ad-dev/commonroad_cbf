"""Unit tests for src.data_association.

These tests exercise data association in isolation. They deliberately use a
small Track-like stub so that association tests do not depend on Kalman-filter
or Track lifecycle behaviour; those belong in tracker unit tests.
"""
import sys
import numpy as np
import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.config import DA
from src.tracker import Detection
from src.data_association import (
    compute_cost_matrix,
    associate_detections_to_tracks,
)


class DummyTrack:
    """Minimal object implementing the Track.position interface."""

    def __init__(self, x, y):
        self.position = np.asarray([x, y], dtype=float)


def make_detection(x, y, obstacle_id=1, sensor_id="radar_front",
                   timestamp=0.0, covariance=None):
    if covariance is None:
        covariance = np.eye(2, dtype=float)
    return Detection(
        sensor_id=sensor_id,
        timestamp=timestamp,
        z=np.asarray([x, y], dtype=float),
        obstacle_id=obstacle_id,
        R=np.asarray(covariance, dtype=float),
    )


# ---------------------------------------------------------------------------
# compute_cost_matrix: empty inputs and dimensions
# ---------------------------------------------------------------------------

def test_cost_matrix_with_no_tracks_has_shape_zero_by_detection_count():
    detections = [make_detection(1, 2), make_detection(3, 4)]

    costs = compute_cost_matrix([], detections)

    assert costs.shape == (0, 2)
    assert costs.size == 0


def test_cost_matrix_with_no_detections_has_shape_track_count_by_zero():
    tracks = [DummyTrack(1, 2), DummyTrack(3, 4)]

    costs = compute_cost_matrix(tracks, [])

    assert costs.shape == (2, 0)
    assert costs.size == 0


def test_cost_matrix_with_both_inputs_empty_has_zero_by_zero_shape():
    costs = compute_cost_matrix([], [])

    assert costs.shape == (0, 0)
    assert costs.size == 0


# ---------------------------------------------------------------------------
# compute_cost_matrix: Euclidean distance and gating
# ---------------------------------------------------------------------------

def test_cost_matrix_contains_euclidean_distances():
    tracks = [DummyTrack(0, 0), DummyTrack(3, 4)]
    detections = [make_detection(3, 4), make_detection(0, 0)]

    costs = compute_cost_matrix(tracks, detections, max_distance=10.0)

    expected = np.array([
        [5.0, 0.0],
        [0.0, 5.0],
    ])
    np.testing.assert_allclose(costs, expected)


def test_cost_matrix_uses_first_two_detection_coordinates_only():
    tracks = [DummyTrack(1, 2)]
    detection = Detection(
        sensor_id="radar_front",
        timestamp=0.0,
        z=np.array([4.0, 6.0, 999.0]),
        obstacle_id=3,
        R=np.eye(2),
    )

    costs = compute_cost_matrix(tracks, [detection], max_distance=10.0)

    assert costs[0, 0] == pytest.approx(5.0)


def test_cost_matrix_accepts_pair_inside_gating_threshold():
    tracks = [DummyTrack(0, 0)]
    detections = [make_detection(3, 4)]

    costs = compute_cost_matrix(tracks, detections, max_distance=5.0)

    assert costs[0, 0] == pytest.approx(5.0)


def test_cost_matrix_assigns_penalty_to_pair_outside_gating_threshold():
    tracks = [DummyTrack(0, 0)]
    detections = [make_detection(3.0001, 4)]

    costs = compute_cost_matrix(tracks, detections, max_distance=5.0)

    assert costs[0, 0] == pytest.approx(DA.GATING_PENALTY_COST)


def test_cost_matrix_uses_custom_gating_threshold():
    tracks = [DummyTrack(0, 0)]
    detections = [make_detection(6, 8)]

    costs = compute_cost_matrix(tracks, detections, max_distance=11.0)

    assert costs[0, 0] == pytest.approx(10.0)


def test_cost_matrix_uses_configured_default_gating_threshold():
    tracks = [DummyTrack(0, 0)]
    distance = DA.GATING_THRESHOLD
    detections = [make_detection(distance, 0)]

    costs = compute_cost_matrix(tracks, detections)

    assert costs[0, 0] == pytest.approx(distance)


def test_cost_matrix_is_float64():
    costs = compute_cost_matrix(
        [DummyTrack(0, 0)],
        [make_detection(1, 1)],
        max_distance=10.0,
    )

    assert costs.dtype == np.float64


def test_cost_matrix_gates_each_pair_independently():
    tracks = [DummyTrack(0, 0), DummyTrack(20, 0)]
    detections = [make_detection(1, 0), make_detection(25, 0)]

    costs = compute_cost_matrix(tracks, detections, max_distance=5.0)

    expected = np.array([
        [1.0, DA.GATING_PENALTY_COST],
        [DA.GATING_PENALTY_COST, 5.0],
    ])
    np.testing.assert_allclose(costs, expected)


# ---------------------------------------------------------------------------
# associate_detections_to_tracks: empty input branches
# ---------------------------------------------------------------------------

def test_association_with_no_tracks_returns_all_detections_unmatched():
    detections = [
        make_detection(1, 0),
        make_detection(2, 0),
        make_detection(3, 0),
    ]

    matches, unmatched_tracks, unmatched_detections = (
        associate_detections_to_tracks([], detections)
    )

    assert matches == []
    assert unmatched_tracks == []
    assert unmatched_detections == [0, 1, 2]


def test_association_with_no_detections_returns_all_tracks_unmatched():
    tracks = [DummyTrack(0, 0), DummyTrack(1, 0)]

    matches, unmatched_tracks, unmatched_detections = (
        associate_detections_to_tracks(tracks, [])
    )

    assert matches == []
    assert unmatched_tracks == [0, 1]
    assert unmatched_detections == []


def test_association_with_both_inputs_empty_returns_three_empty_lists():
    result = associate_detections_to_tracks([], [])

    assert result == ([], [], [])


# ---------------------------------------------------------------------------
# associate_detections_to_tracks: matching behaviour
# ---------------------------------------------------------------------------

def test_association_matches_single_track_to_single_detection():
    tracks = [DummyTrack(0, 0)]
    detections = [make_detection(3, 4)]

    matches, unmatched_tracks, unmatched_detections = (
        associate_detections_to_tracks(
            tracks, detections, max_distance=10.0
        )
    )

    assert matches == [(0, 0)]
    assert unmatched_tracks == []
    assert unmatched_detections == []


def test_association_finds_nearest_detection_for_single_track():
    tracks = [DummyTrack(0, 0)]
    detections = [
        make_detection(8, 0),
        make_detection(2, 0),
        make_detection(5, 0),
    ]

    matches, unmatched_tracks, unmatched_detections = (
        associate_detections_to_tracks(
            tracks, detections, max_distance=10.0
        )
    )

    assert matches == [(0, 1)]
    assert unmatched_tracks == []
    assert unmatched_detections == [0, 2]


def test_association_finds_global_optimum_not_greedy_nearest_pair():
    """The assignment should minimise total cost across the whole matrix."""
    tracks = [DummyTrack(0, 0), DummyTrack(2, 0)]
    detections = [make_detection(1, 0), make_detection(-1, 0)]

    matches, unmatched_tracks, unmatched_detections = (
        associate_detections_to_tracks(
            tracks, detections, max_distance=10.0
        )
    )

    # Cross assignment costs 1 + 1 = 2; diagonal assignment costs 1 + 3 = 4.
    # The Hungarian algorithm must choose the globally cheaper assignment.
    assert len(matches) == 2
    assert set(matches) == {(0, 1), (1, 0)}
    assert unmatched_tracks == []
    assert unmatched_detections == []


def test_association_handles_more_tracks_than_detections():
    tracks = [
        DummyTrack(0, 0),
        DummyTrack(10, 0),
        DummyTrack(20, 0),
    ]
    detections = [make_detection(0.5, 0), make_detection(10.5, 0)]

    matches, unmatched_tracks, unmatched_detections = (
        associate_detections_to_tracks(
            tracks, detections, max_distance=5.0
        )
    )

    assert set(matches) == {(0, 0), (1, 1)}
    assert unmatched_tracks == [2]
    assert unmatched_detections == []


def test_association_handles_more_detections_than_tracks():
    tracks = [DummyTrack(0, 0), DummyTrack(20, 0)]
    detections = [
        make_detection(0.5, 0),
        make_detection(10, 0),
        make_detection(20.5, 0),
    ]

    matches, unmatched_tracks, unmatched_detections = (
        associate_detections_to_tracks(
            tracks, detections, max_distance=5.0
        )
    )

    assert set(matches) == {(0, 0), (1, 2)}
    assert unmatched_tracks == []
    assert unmatched_detections == [1]


def test_association_returns_unmatched_indices_in_original_order():
    tracks = [DummyTrack(0, 0), DummyTrack(100, 0), DummyTrack(200, 0)]
    detections = [make_detection(0.5, 0), make_detection(200.5, 0)]

    _, unmatched_tracks, unmatched_detections = (
        associate_detections_to_tracks(
            tracks, detections, max_distance=2.0
        )
    )

    assert unmatched_tracks == [1]
    assert unmatched_detections == []


# ---------------------------------------------------------------------------
# Association gating and edge conditions
# ---------------------------------------------------------------------------

def test_association_rejects_pair_outside_max_assignment_distance():
    tracks = [DummyTrack(0, 0)]
    detections = [make_detection(10.01, 0)]

    matches, unmatched_tracks, unmatched_detections = (
        associate_detections_to_tracks(
            tracks, detections, max_distance=10.0
        )
    )

    assert matches == []
    assert unmatched_tracks == [0]
    assert unmatched_detections == [0]


def test_association_rejects_pair_exactly_at_max_distance():
    """Association uses a strict '< max_distance' acceptance condition."""
    tracks = [DummyTrack(0, 0)]
    detections = [make_detection(5, 0)]

    matches, unmatched_tracks, unmatched_detections = (
        associate_detections_to_tracks(
            tracks, detections, max_distance=5.0
        )
    )

    assert matches == []
    assert unmatched_tracks == [0]
    assert unmatched_detections == [0]


def test_association_accepts_pair_just_inside_max_distance():
    tracks = [DummyTrack(0, 0)]
    detections = [make_detection(4.999, 0)]

    matches, unmatched_tracks, unmatched_detections = (
        associate_detections_to_tracks(
            tracks, detections, max_distance=5.0
        )
    )

    assert matches == [(0, 0)]
    assert unmatched_tracks == []
    assert unmatched_detections == []


def test_association_with_zero_distance_threshold_rejects_even_coincident_pair():
    """Documents current strict-boundary semantics for max_distance=0."""
    tracks = [DummyTrack(1, 2)]
    detections = [make_detection(1, 2)]

    matches, unmatched_tracks, unmatched_detections = (
        associate_detections_to_tracks(
            tracks, detections, max_distance=0.0
        )
    )

    assert matches == []
    assert unmatched_tracks == [0]
    assert unmatched_detections == [0]


def test_association_preserves_valid_match_when_other_pair_is_outside_gate():
    tracks = [DummyTrack(0, 0), DummyTrack(100, 0)]
    detections = [make_detection(1, 0), make_detection(500, 0)]

    matches, unmatched_tracks, unmatched_detections = (
        associate_detections_to_tracks(
            tracks, detections, max_distance=5.0
        )
    )

    assert matches == [(0, 0)]
    assert unmatched_tracks == [1]
    assert unmatched_detections == [1]


def test_association_with_all_pairs_outside_gate_returns_no_matches():
    tracks = [DummyTrack(0, 0), DummyTrack(100, 0)]
    detections = [make_detection(50, 0), make_detection(200, 0)]

    matches, unmatched_tracks, unmatched_detections = (
        associate_detections_to_tracks(
            tracks, detections, max_distance=1.0
        )
    )

    assert matches == []
    assert unmatched_tracks == [0, 1]
    assert unmatched_detections == [0, 1]


def test_association_uses_configured_default_assignment_distance():
    tracks = [DummyTrack(0, 0)]
    distance = DA.MAX_ASSIGNMENT_DISTANCE / 2.0
    detections = [make_detection(distance, 0)]

    matches, unmatched_tracks, unmatched_detections = (
        associate_detections_to_tracks(tracks, detections)
    )

    assert matches == [(0, 0)]
    assert unmatched_tracks == []
    assert unmatched_detections == []


def test_association_match_indices_are_unique_and_in_range():
    tracks = [DummyTrack(0, 0), DummyTrack(10, 0), DummyTrack(20, 0)]
    detections = [
        make_detection(0.1, 0),
        make_detection(10.1, 0),
        make_detection(20.1, 0),
    ]

    matches, unmatched_tracks, unmatched_detections = (
        associate_detections_to_tracks(
            tracks, detections, max_distance=2.0
        )
    )

    track_indices = [track_idx for track_idx, _ in matches]
    detection_indices = [det_idx for _, det_idx in matches]

    assert len(track_indices) == len(set(track_indices))
    assert len(detection_indices) == len(set(detection_indices))
    assert all(0 <= idx < len(tracks) for idx in track_indices)
    assert all(0 <= idx < len(detections) for idx in detection_indices)
    assert set(track_indices).isdisjoint(unmatched_tracks)
    assert set(detection_indices).isdisjoint(unmatched_detections)


# ---------------------------------------------------------------------------
# Regression tests for configuration assumptions / documented limitations
# ---------------------------------------------------------------------------

def test_gating_penalty_is_greater_than_assignment_threshold():
    """A penalty must not look cheaper than a feasible assignment."""
    assert DA.GATING_PENALTY_COST > DA.MAX_ASSIGNMENT_DISTANCE
