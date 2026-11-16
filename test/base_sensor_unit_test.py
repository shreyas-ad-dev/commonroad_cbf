"""Level-1 unit tests for src.base_sensor.

Run from repository root:
    pytest test/base_sensor_unit_test.py -v

These tests cover BaseSensor utilities only, not concrete sensor detection,
FOV clipping, occlusion algorithms, noise, or tracking.
"""
import sys
from types import SimpleNamespace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import numpy as np
import pytest
from shapely.geometry import Polygon

from src.base_sensor import BaseSensor, SensorOcclusionData
from src.config import VEHICLE
from src.ego_state import EgoState


def make_ego(x=0.0, y=0.0, yaw=0.0):
    return EgoState(x, y, yaw, 0.0, 4.5, 1.8, 2.7)


def make_obstacle(x=10.0, y=0.0, yaw=0.0, length=4.0, width=2.0,
                  available_step=0, missing_orientation=False):
    state_args = {"position": np.array([x, y], dtype=float)}
    if not missing_orientation:
        state_args["orientation"] = yaw
    state = SimpleNamespace(**state_args)

    class Obstacle:
        obstacle_shape = SimpleNamespace(length=length, width=width)

        def state_at_time(self, step):
            return state if step == available_step else None

    return Obstacle(), state


def make_obstacle_without_dimensions(x=10.0, y=0.0):
    state = SimpleNamespace(position=np.array([x, y], dtype=float),
                            orientation=0.0)

    class Obstacle:
        obstacle_shape = SimpleNamespace()

        def state_at_time(self, step):
            return state

    return Obstacle(), state


# Construction and parameters

def test_constructor_stores_range_and_fov():
    sensor = BaseSensor(100.0, 120.0)
    assert sensor.range_max == 100.0
    assert sensor.fov_deg == 120.0


@pytest.mark.parametrize("fov,half", [
    (0.0, 0.0), (30.0, np.pi / 12), (90.0, np.pi / 4),
    (120.0, np.pi / 3), (180.0, np.pi / 2), (360.0, np.pi),
])
def test_constructor_converts_total_fov_to_half_angle(fov, half):
    assert BaseSensor(50.0, fov).half_fov_rad == pytest.approx(half)


def test_constructor_accepts_zero_range():
    sensor = BaseSensor(0.0, 90.0)
    assert sensor.range_max == 0.0


def test_sensor_occlusion_data_stores_all_fields():
    segments = [np.array([[1.0, 2.0], [3.0, 4.0]])]
    data = SensorOcclusionData(7, True, 12.5, 10.0, -2.0, segments)
    assert data.obstacle_id == 7
    assert data.in_fov is True
    assert data.min_dist == 12.5
    assert data.center_x_local == 10.0
    assert data.center_y_local == -2.0
    assert data.visible_segments is segments


def test_sensor_occlusion_data_accepts_empty_segments():
    data = SensorOcclusionData(1, False, float("inf"), 0.0, 0.0, [])
    assert data.visible_segments == []
    assert not data.in_fov


# World-to-ego transformation

def test_local_frame_ahead_at_zero_yaw():
    ego = make_ego(2.0, 3.0, 0.0)
    np.testing.assert_allclose(BaseSensor.to_local_frame(ego, 7.0, 3.0),
                               [5.0, 0.0])


def test_local_frame_behind_at_zero_yaw():
    ego = make_ego(2.0, 3.0, 0.0)
    np.testing.assert_allclose(BaseSensor.to_local_frame(ego, -3.0, 3.0),
                               [-5.0, 0.0])


def test_local_frame_left_at_zero_yaw():
    ego = make_ego(2.0, 3.0, 0.0)
    np.testing.assert_allclose(BaseSensor.to_local_frame(ego, 2.0, 8.0),
                               [0.0, 5.0])


def test_local_frame_right_at_zero_yaw():
    ego = make_ego(2.0, 3.0, 0.0)
    np.testing.assert_allclose(BaseSensor.to_local_frame(ego, 2.0, -2.0),
                               [0.0, -5.0])


def test_local_frame_ego_position_maps_to_origin():
    ego = make_ego(13.0, -8.0, 1.17)
    np.testing.assert_allclose(BaseSensor.to_local_frame(ego, 13.0, -8.0),
                               [0.0, 0.0], atol=1e-12)


def test_local_frame_rotates_world_y_to_forward_for_ninety_degree_ego():
    ego = make_ego(yaw=np.pi / 2)
    np.testing.assert_allclose(BaseSensor.to_local_frame(ego, 0.0, 10.0),
                               [10.0, 0.0], atol=1e-12)


def test_local_frame_world_x_is_right_of_ninety_degree_ego():
    ego = make_ego(yaw=np.pi / 2)
    np.testing.assert_allclose(BaseSensor.to_local_frame(ego, 10.0, 0.0),
                               [0.0, -10.0], atol=1e-12)


@pytest.mark.parametrize("ex,ey,yaw,ox,oy,expected", [
    (5.0, -2.0, 0.0, 8.0, 4.0, [3.0, 6.0]),
    (5.0, -2.0, np.pi, 2.0, -2.0, [3.0, 0.0]),
    (1.0, 2.0, -np.pi / 2, 1.0, 7.0, [-5.0, 0.0]),
    (-3.0, 4.0, np.pi / 4, -3.0, 4.0 + 2*np.sqrt(2), [2.0, 2.0]),
])
def test_local_frame_general_transforms(ex, ey, yaw, ox, oy, expected):
    np.testing.assert_allclose(
        BaseSensor.to_local_frame(make_ego(ex, ey, yaw), ox, oy),
        expected, atol=1e-12)


def test_local_frame_preserves_euclidean_distance():
    ego = make_ego(4.0, -7.0, 0.83)
    target = np.array([15.0, 2.0])
    local = BaseSensor.to_local_frame(ego, *target)
    assert np.linalg.norm(local) == pytest.approx(
        np.linalg.norm(target - ego.position))


def test_local_frame_does_not_mutate_ego():
    ego = make_ego(2.0, 3.0, 0.5)
    before = (ego.x, ego.y, ego.orientation, ego.velocity)
    BaseSensor.to_local_frame(ego, 20.0, -5.0)
    assert (ego.x, ego.y, ego.orientation, ego.velocity) == before


def test_local_frame_returns_two_element_numpy_array():
    result = BaseSensor.to_local_frame(make_ego(), 3.0, 4.0)
    assert isinstance(result, np.ndarray)
    assert result.shape == (2,)


# Obstacle state extraction

def test_obstacle_state_unavailable_returns_none():
    sensor = BaseSensor(100.0, 90.0)
    obstacle, _ = make_obstacle(available_step=5)
    assert sensor.get_obstacle_center_and_corners_in_local(
        make_ego(), obstacle, 4) is None


def test_obstacle_extraction_queries_requested_step():
    sensor = BaseSensor(100.0, 90.0)
    steps = []

    class RecordingObstacle:
        obstacle_shape = SimpleNamespace(length=4.0, width=2.0)

        def state_at_time(self, step):
            steps.append(step)
            return None

    sensor.get_obstacle_center_and_corners_in_local(
        make_ego(), RecordingObstacle(), 23)
    assert steps == [23]


def test_obstacle_extraction_returns_center_and_four_corners():
    sensor = BaseSensor(100.0, 90.0)
    obstacle, _ = make_obstacle(10.0, 3.0)
    result = sensor.get_obstacle_center_and_corners_in_local(
        make_ego(), obstacle, 0)
    assert result is not None
    center, points = result
    np.testing.assert_allclose(center, [10.0, 3.0])
    assert len(points) == 5
    assert all(isinstance(p, np.ndarray) and p.shape == (2,) for p in points)


def test_first_returned_point_is_obstacle_center():
    sensor = BaseSensor(100.0, 90.0)
    obstacle, _ = make_obstacle(12.0, 4.0)
    center, points = sensor.get_obstacle_center_and_corners_in_local(
        make_ego(2.0, 1.0), obstacle, 0)
    np.testing.assert_allclose(points[0], center)
    np.testing.assert_allclose(center, [10.0, 3.0])


def test_axis_aligned_obstacle_corner_coordinates():
    sensor = BaseSensor(100.0, 90.0)
    obstacle, _ = make_obstacle(10.0, 0.0, length=4.0, width=2.0)
    center, points = sensor.get_obstacle_center_and_corners_in_local(
        make_ego(), obstacle, 0)
    expected = np.array([
        [10.0, 0.0], [8.0, -1.0], [12.0, -1.0],
        [12.0, 1.0], [8.0, 1.0],
    ])
    np.testing.assert_allclose(center, expected[0])
    np.testing.assert_allclose(np.array(points), expected)


def test_obstacle_rotation_changes_corner_coordinates():
    sensor = BaseSensor(100.0, 90.0)
    obstacle, _ = make_obstacle(10.0, 0.0, yaw=np.pi/2,
                                length=4.0, width=2.0)
    _, points = sensor.get_obstacle_center_and_corners_in_local(
        make_ego(), obstacle, 0)
    expected = np.array([
        [10.0, 0.0], [11.0, -2.0], [11.0, 2.0],
        [9.0, 2.0], [9.0, -2.0],
    ])
    np.testing.assert_allclose(np.array(points), expected, atol=1e-12)


def test_obstacle_geometry_is_transformed_into_rotated_ego_frame():
    sensor = BaseSensor(100.0, 180.0)
    obstacle, _ = make_obstacle(0.0, 10.0, length=4.0, width=2.0)
    center, points = sensor.get_obstacle_center_and_corners_in_local(
        make_ego(yaw=np.pi/2), obstacle, 0)
    np.testing.assert_allclose(center, [10.0, 0.0], atol=1e-12)
    expected = np.array([
        [10.0, 0.0], [9.0, 2.0], [9.0, -2.0],
        [11.0, -2.0], [11.0, 2.0],
    ])
    np.testing.assert_allclose(np.array(points), expected, atol=1e-12)


def test_obstacle_dimensions_determine_bounding_box():
    sensor = BaseSensor(100.0, 90.0)
    obstacle, _ = make_obstacle(10.0, 0.0, length=8.0, width=3.0)
    _, points = sensor.get_obstacle_center_and_corners_in_local(
        make_ego(), obstacle, 0)
    corners = np.array(points[1:])
    assert np.ptp(corners[:, 0]) == pytest.approx(8.0)
    assert np.ptp(corners[:, 1]) == pytest.approx(3.0)


def test_missing_shape_dimensions_use_vehicle_defaults():
    sensor = BaseSensor(100.0, 90.0)
    obstacle, _ = make_obstacle_without_dimensions()
    _, points = sensor.get_obstacle_center_and_corners_in_local(
        make_ego(), obstacle, 0)
    corners = np.array(points[1:])
    assert np.ptp(corners[:, 0]) == pytest.approx(VEHICLE.DEFAULT_LENGTH)
    assert np.ptp(corners[:, 1]) == pytest.approx(VEHICLE.DEFAULT_WIDTH)


def test_missing_obstacle_orientation_defaults_to_zero():
    sensor = BaseSensor(100.0, 90.0)
    obstacle, _ = make_obstacle(missing_orientation=True)
    _, points = sensor.get_obstacle_center_and_corners_in_local(
        make_ego(), obstacle, 0)
    np.testing.assert_allclose(np.array(points[1:]), [
        [8.0, -1.0], [12.0, -1.0], [12.0, 1.0], [8.0, 1.0],
    ])


def test_extraction_uses_state_position_for_center():
    sensor = BaseSensor(100.0, 90.0)
    obstacle, state = make_obstacle(10.0, 5.0)
    state.position = np.array([20.0, 7.0])
    center, _ = sensor.get_obstacle_center_and_corners_in_local(
        make_ego(1.0, 2.0), obstacle, 0)
    np.testing.assert_allclose(center, [19.0, 5.0])


def test_extraction_does_not_mutate_ego_or_obstacle_state():
    sensor = BaseSensor(100.0, 90.0)
    ego = make_ego(2.0, 3.0, 0.2)
    obstacle, state = make_obstacle(10.0, 5.0, yaw=0.4)
    ego_before = (ego.x, ego.y, ego.orientation)
    position_before = state.position.copy()
    orientation_before = state.orientation
    sensor.get_obstacle_center_and_corners_in_local(ego, obstacle, 0)
    assert (ego.x, ego.y, ego.orientation) == ego_before
    np.testing.assert_array_equal(state.position, position_before)
    assert state.orientation == orientation_before


def test_extraction_recomputes_geometry_on_subsequent_calls():
    sensor = BaseSensor(100.0, 90.0)
    obstacle, _ = make_obstacle()
    _, first = sensor.get_obstacle_center_and_corners_in_local(
        make_ego(), obstacle, 0)
    first[1][0] = -999.0
    _, second = sensor.get_obstacle_center_and_corners_in_local(
        make_ego(), obstacle, 0)
    assert second[1][0] != -999.0


def test_extracted_center_matches_to_local_frame_helper():
    sensor = BaseSensor(100.0, 90.0)
    ego = make_ego(-3.0, 6.0, -0.6)
    obstacle, state = make_obstacle(12.0, -4.0)
    center, points = sensor.get_obstacle_center_and_corners_in_local(
        ego, obstacle, 0)
    expected = BaseSensor.to_local_frame(
        ego, state.position[0], state.position[1])
    np.testing.assert_allclose(center, expected)
    np.testing.assert_allclose(points[0], expected)
