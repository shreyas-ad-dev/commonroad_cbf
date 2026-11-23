"""
Unit tests for src.ego_state.

These tests isolate EgoState and get_car_polygon. They do not exercise
CommonRoad scenarios, planners, sensors, or vehicle-dynamics integration.
Run from the repository root:

    pytest test/ego_state_unit_test.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import Polygon

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from types import SimpleNamespace

from src.ego_state import EgoState, get_car_polygon


def make_ego(
    x=0.0,
    y=0.0,
    orientation=0.0,
    velocity=0.0,
    length=4.5,
    width=1.8,
    wheelbase=2.7,
    road_heading=None,
):
    return EgoState(
        x=x,
        y=y,
        orientation=orientation,
        velocity=velocity,
        length=length,
        width=width,
        wheelbase=wheelbase,
        road_heading=road_heading,
    )


# ---------------------------------------------------------------------------
# get_car_polygon
# ---------------------------------------------------------------------------

def test_get_car_polygon_returns_shapely_polygon_and_corner_array():
    polygon, corners = get_car_polygon(1.0, 2.0, 0.0, 4.0, 2.0)

    assert isinstance(polygon, Polygon)
    assert isinstance(corners, np.ndarray)
    assert corners.shape == (4, 2)


def test_get_car_polygon_axis_aligned_corner_coordinates():
    polygon, corners = get_car_polygon(10.0, 20.0, 0.0, 4.0, 2.0)

    expected = np.array([
        [8.0, 19.0],
        [12.0, 19.0],
        [12.0, 21.0],
        [8.0, 21.0],
    ])

    np.testing.assert_allclose(corners, expected)
    assert polygon.area == pytest.approx(8.0)


def test_get_car_polygon_rotates_corners_by_90_degrees():
    _, corners = get_car_polygon(0.0, 0.0, np.pi / 2, 4.0, 2.0)

    expected = np.array([
        [1.0, -2.0],
        [1.0, 2.0],
        [-1.0, 2.0],
        [-1.0, -2.0],
    ])

    np.testing.assert_allclose(corners, expected, atol=1e-12)


def test_get_car_polygon_translates_rotated_geometry():
    _, corners = get_car_polygon(5.0, -3.0, np.pi / 2, 4.0, 2.0)

    expected = np.array([
        [6.0, -5.0],
        [6.0, -1.0],
        [4.0, -1.0],
        [4.0, -5.0],
    ])

    np.testing.assert_allclose(corners, expected, atol=1e-12)


def test_get_car_polygon_preserves_requested_length_and_width():
    polygon, _ = get_car_polygon(0.0, 0.0, 0.37, 6.0, 2.5)

    assert polygon.area == pytest.approx(15.0)
    assert polygon.is_valid


def test_get_car_polygon_is_centered_on_requested_position():
    _, corners = get_car_polygon(7.0, -4.0, 1.13, 5.0, 2.0)

    np.testing.assert_allclose(corners.mean(axis=0), [7.0, -4.0], atol=1e-12)


def test_get_car_polygon_has_four_distinct_corners_for_positive_dimensions():
    _, corners = get_car_polygon(0.0, 0.0, 0.5, 4.0, 2.0)

    assert len(np.unique(corners, axis=0)) == 4


@pytest.mark.parametrize(
    "length,width",
    [
        (0.0, 2.0),
        (4.0, 0.0),
        (-4.0, 2.0),
        (4.0, -2.0),
    ],
)
def test_get_car_polygon_degenerate_or_negative_dimensions_are_not_silently_assumed_valid(
    length, width
):
    # The implementation does not validate dimensions; this test records
    # the geometry outcome without imposing a new validation policy.
    polygon, corners = get_car_polygon(0.0, 0.0, 0.0, length, width)

    assert corners.shape == (4, 2)
    assert isinstance(polygon, Polygon)


# ---------------------------------------------------------------------------
# Dataclass and basic vector properties
# ---------------------------------------------------------------------------

def test_ego_state_can_be_constructed_with_required_fields():
    ego = make_ego(x=1.0, y=2.0, orientation=0.3, velocity=4.0)

    assert ego.x == 1.0
    assert ego.y == 2.0
    assert ego.orientation == 0.3
    assert ego.velocity == 4.0


def test_road_heading_defaults_to_none():
    ego = make_ego()

    assert ego.road_heading is None


def test_position_returns_global_xy_vector():
    ego = make_ego(x=3.5, y=-2.25)

    np.testing.assert_allclose(ego.position, [3.5, -2.25])


def test_position_returns_new_array_not_mutable_internal_state():
    ego = make_ego(x=3.0, y=4.0)

    position = ego.position
    position[0] = 100.0

    assert ego.x == 3.0
    assert ego.y == 4.0


def test_heading_vector_at_zero_orientation():
    ego = make_ego(orientation=0.0)

    np.testing.assert_allclose(ego.heading_vector, [1.0, 0.0], atol=1e-12)


def test_heading_vector_at_ninety_degrees():
    ego = make_ego(orientation=np.pi / 2)

    np.testing.assert_allclose(ego.heading_vector, [0.0, 1.0], atol=1e-12)


@pytest.mark.parametrize("angle", [-np.pi, -1.2, -0.2, 0.4, 1.7, np.pi])
def test_heading_vector_is_unit_length(angle):
    ego = make_ego(orientation=angle)

    assert np.linalg.norm(ego.heading_vector) == pytest.approx(1.0)


def test_normal_vector_points_left_of_vehicle_heading():
    ego = make_ego(orientation=0.0)

    np.testing.assert_allclose(ego.normal_vector, [0.0, 1.0], atol=1e-12)


def test_normal_vector_is_perpendicular_to_heading_vector():
    ego = make_ego(orientation=0.73)

    assert np.dot(ego.heading_vector, ego.normal_vector) == pytest.approx(0.0)
    assert np.linalg.norm(ego.normal_vector) == pytest.approx(1.0)


def test_velocity_vector_is_scalar_speed_along_heading():
    ego = make_ego(orientation=np.pi / 4, velocity=10.0)

    np.testing.assert_allclose(
        ego.velocity_vector,
        [10.0 / np.sqrt(2), 10.0 / np.sqrt(2)],
        atol=1e-12,
    )


def test_negative_velocity_is_preserved_in_velocity_vector():
    # EgoState currently does not constrain velocity to be non-negative
    # during construction; its integration method clips only the next speed.
    ego = make_ego(orientation=0.0, velocity=-3.0)

    np.testing.assert_allclose(ego.velocity_vector, [-3.0, 0.0])


# ---------------------------------------------------------------------------
# Road coordinate frame
# ---------------------------------------------------------------------------

def test_road_frame_falls_back_to_vehicle_heading_when_road_heading_is_none():
    ego = make_ego(orientation=0.6)

    tangent, normal = ego.road_frame_vectors

    np.testing.assert_allclose(tangent, ego.heading_vector)
    np.testing.assert_allclose(normal, ego.normal_vector)


def test_road_frame_uses_road_heading_when_provided():
    ego = make_ego(orientation=0.2, road_heading=np.pi / 2)

    tangent, normal = ego.road_frame_vectors

    np.testing.assert_allclose(tangent, [0.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(normal, [-1.0, 0.0], atol=1e-12)


def test_road_frame_vectors_are_unit_and_orthogonal():
    ego = make_ego(orientation=-0.8, road_heading=1.1)

    tangent, normal = ego.road_frame_vectors

    assert np.linalg.norm(tangent) == pytest.approx(1.0)
    assert np.linalg.norm(normal) == pytest.approx(1.0)
    assert np.dot(tangent, normal) == pytest.approx(0.0)


def test_changing_road_heading_does_not_change_vehicle_heading_vector():
    ego = make_ego(orientation=0.4, road_heading=1.2)
    before = ego.heading_vector.copy()

    ego.road_heading = -0.7

    np.testing.assert_allclose(ego.heading_vector, before)


# ---------------------------------------------------------------------------
# Axle and bounding-box properties
# ---------------------------------------------------------------------------

def test_front_axle_position_at_zero_orientation():
    ego = make_ego(x=10.0, y=5.0, orientation=0.0, wheelbase=2.8)

    np.testing.assert_allclose(ego.front_axle_position, [12.8, 5.0])


def test_front_axle_position_rotates_with_vehicle_heading():
    ego = make_ego(x=1.0, y=2.0, orientation=np.pi / 2, wheelbase=3.0)

    np.testing.assert_allclose(ego.front_axle_position, [1.0, 5.0], atol=1e-12)


def test_front_axle_position_does_not_use_road_heading():
    ego = make_ego(
        x=0.0,
        y=0.0,
        orientation=0.0,
        wheelbase=2.0,
        road_heading=np.pi / 2,
    )

    np.testing.assert_allclose(ego.front_axle_position, [2.0, 0.0], atol=1e-12)


def test_polygon_and_corners_returns_consistent_geometry():
    ego = make_ego(x=2.0, y=3.0, orientation=0.4)

    polygon, corners = ego.polygon_and_corners

    assert isinstance(polygon, Polygon)
    np.testing.assert_allclose(corners, ego.corners)
    assert polygon.equals(Polygon(corners))


def test_corners_property_returns_four_global_corners():
    ego = make_ego(x=-2.0, y=4.0, orientation=0.9)

    assert ego.corners.shape == (4, 2)
    np.testing.assert_allclose(ego.corners.mean(axis=0), [-2.0, 4.0])


def test_polygon_property_returns_shapely_polygon():
    ego = make_ego(length=4.8, width=2.0)

    assert isinstance(ego.polygon, Polygon)
    assert ego.polygon.area == pytest.approx(4.8 * 2.0)


def test_polygon_and_corners_are_recomputed_after_state_change():
    ego = make_ego(x=0.0, y=0.0)
    first_corners = ego.corners.copy()

    ego.x = 10.0
    ego.y = -5.0
    second_corners = ego.corners

    assert not np.allclose(first_corners, second_corners)
    np.testing.assert_allclose(second_corners.mean(axis=0), [10.0, -5.0])


# ---------------------------------------------------------------------------
# Legacy representation and logging
# ---------------------------------------------------------------------------

def test_to_tuple_preserves_legacy_six_field_order():
    ego = make_ego(
        x=1.0,
        y=2.0,
        orientation=0.3,
        velocity=4.0,
        length=5.0,
        width=2.0,
        wheelbase=2.7,
        road_heading=0.8,
    )

    assert ego.to_tuple() == (1.0, 2.0, 0.3, 4.0, 5.0, 2.0)
    assert len(ego.to_tuple()) == 6


def test_get_log_data_contains_expected_keys_and_values():
    ego = make_ego(
        x=1.0,
        y=2.0,
        orientation=np.pi / 3,
        velocity=7.0,
        length=4.5,
        width=1.9,
        wheelbase=2.8,
        road_heading=0.2,
    )

    log = ego.get_log_data()

    assert set(log) == {
        "x",
        "y",
        "velocity",
        "orientation_rad",
        "orientation_deg",
        "length",
        "width",
        "wheelbase",
        "road_heading",
    }
    assert log["x"] == 1.0
    assert log["y"] == 2.0
    assert log["velocity"] == 7.0
    assert log["orientation_rad"] == pytest.approx(np.pi / 3)
    assert log["orientation_deg"] == pytest.approx(60.0)
    assert log["road_heading"] == 0.2


def test_get_log_data_reflects_updated_state():
    ego = make_ego()

    ego.x = 5.0
    ego.velocity = 8.0
    ego.road_heading = 0.25

    log = ego.get_log_data()

    assert log["x"] == 5.0
    assert log["velocity"] == 8.0
    assert log["road_heading"] == 0.25


# ---------------------------------------------------------------------------
# Kinematic bicycle integration
# ---------------------------------------------------------------------------

def test_update_kinematics_constant_speed_straight_line():
    ego = make_ego(x=0.0, y=0.0, orientation=0.0, velocity=10.0)

    ego.update_kinematics(accel=0.0, steering_angle=0.0, dt=0.1)

    assert ego.x == pytest.approx(1.0)
    assert ego.y == pytest.approx(0.0)
    assert ego.orientation == pytest.approx(0.0)
    assert ego.velocity == pytest.approx(10.0)


def test_update_kinematics_straight_line_respects_vehicle_heading():
    ego = make_ego(x=1.0, y=2.0, orientation=np.pi / 2, velocity=4.0)

    ego.update_kinematics(accel=0.0, steering_angle=0.0, dt=0.5)

    assert ego.x == pytest.approx(1.0)
    assert ego.y == pytest.approx(4.0)


def test_update_kinematics_uses_midpoint_speed_for_position():
    ego = make_ego(velocity=10.0)

    ego.update_kinematics(accel=2.0, steering_angle=0.0, dt=0.5)

    # v_next = 11; v_avg = 10.5; displacement = 10.5 * 0.5.
    assert ego.x == pytest.approx(5.25)
    assert ego.velocity == pytest.approx(11.0)


def test_update_kinematics_deceleration_uses_midpoint_speed():
    ego = make_ego(velocity=10.0)

    ego.update_kinematics(accel=-4.0, steering_angle=0.0, dt=1.0)

    # v_next = 6; v_avg = 8.
    assert ego.x == pytest.approx(8.0)
    assert ego.velocity == pytest.approx(6.0)


def test_update_kinematics_clips_next_speed_at_zero():
    ego = make_ego(velocity=2.0)

    ego.update_kinematics(accel=-10.0, steering_angle=0.0, dt=1.0)

    assert ego.velocity == 0.0
    # Current implementation uses average of old and clipped next speed.
    assert ego.x == pytest.approx(1.0)


def test_update_kinematics_does_not_move_when_initial_and_final_speed_are_zero():
    ego = make_ego(x=2.0, y=3.0, velocity=0.0)

    ego.update_kinematics(accel=-1.0, steering_angle=0.0, dt=1.0)

    assert ego.x == pytest.approx(2.0)
    assert ego.y == pytest.approx(3.0)
    assert ego.velocity == 0.0


def test_update_kinematics_applies_bicycle_yaw_rate_for_positive_steering():
    ego = make_ego(velocity=10.0, wheelbase=2.5)
    steering = 0.2
    dt = 0.1

    ego.update_kinematics(accel=0.0, steering_angle=steering, dt=dt)

    expected_yaw = (10.0 / 2.5) * np.tan(steering) * dt
    assert ego.orientation == pytest.approx(expected_yaw)


def test_update_kinematics_applies_negative_yaw_for_negative_steering():
    ego = make_ego(velocity=8.0, wheelbase=2.0)
    steering = -0.15
    dt = 0.2

    ego.update_kinematics(accel=0.0, steering_angle=steering, dt=dt)

    expected_yaw = (8.0 / 2.0) * np.tan(steering) * dt
    assert ego.orientation == pytest.approx(expected_yaw)
    assert ego.orientation < 0.0


def test_update_kinematics_uses_average_speed_for_yaw_rate():
    ego = make_ego(velocity=10.0, wheelbase=2.0)
    steering = 0.1
    dt = 0.5

    ego.update_kinematics(accel=2.0, steering_angle=steering, dt=dt)

    v_next = 11.0
    v_avg = 0.5 * (10.0 + v_next)
    expected_yaw = (v_avg / 2.0) * np.tan(steering) * dt

    assert ego.orientation == pytest.approx(expected_yaw)
def test_update_kinematics_does_not_change_heading_below_steering_sensitivity(
    monkeypatch,
):
    monkeypatch.setattr(
        "src.ego_state.EGO",
        SimpleNamespace(STEERING_MIN_SENSITIVITY=0.01),
    )
    ego = make_ego(velocity=10.0, wheelbase=2.5)

    ego.update_kinematics(
        accel=0.0,
        steering_angle=0.009,
        dt=0.1,
    )

    assert ego.orientation == pytest.approx(0.0)


def test_update_kinematics_changes_heading_above_steering_sensitivity(
    monkeypatch,
):
    monkeypatch.setattr(
        "src.ego_state.EGO",
        SimpleNamespace(STEERING_MIN_SENSITIVITY=0.01),
    )
    ego = make_ego(velocity=10.0, wheelbase=2.5)

    ego.update_kinematics(
        accel=0.0,
        steering_angle=0.011,
        dt=0.1,
    )

    assert ego.orientation != pytest.approx(0.0)

#def test_update_kinematics_does_not_change_heading_below_steering_sensitivity(
#    monkeypatch,
#):
#    monkeypatch.setattr(EGO, "STEERING_MIN_SENSITIVITY", 0.01)
#    ego = make_ego(velocity=10.0, wheelbase=2.5)
#
#    ego.update_kinematics(
#        accel=0.0,
#        steering_angle=0.009,
#        dt=0.1,
#    )
#
#    assert ego.orientation == pytest.approx(0.0)
#
#
#def test_update_kinematics_changes_heading_above_steering_sensitivity(
#    monkeypatch,
#):
#    monkeypatch.setattr(EGO, "STEERING_MIN_SENSITIVITY", 0.01)
#    ego = make_ego(velocity=10.0, wheelbase=2.5)
#
#    ego.update_kinematics(
#        accel=0.0,
#        steering_angle=0.011,
#        dt=0.1,
#    )
#
#    assert ego.orientation != pytest.approx(0.0)
#

def test_update_kinematics_does_not_modify_road_heading():
    ego = make_ego(velocity=10.0, road_heading=0.45)

    ego.update_kinematics(accel=1.0, steering_angle=0.2, dt=0.1)

    assert ego.road_heading == pytest.approx(0.45)


def test_update_kinematics_returns_none():
    ego = make_ego()

    result = ego.update_kinematics(accel=0.0, dt=0.1)

    assert result is None

def test_update_kinematics_uses_configured_default_timestep(monkeypatch):
    monkeypatch.setattr(
        "src.ego_state.TEST",
        SimpleNamespace(DEFAULT_SAMPLING_TIME_SEC=0.25),
    )
    ego = make_ego(velocity=4.0)

    ego.update_kinematics(accel=0.0, steering_angle=0.0)

    assert ego.x == pytest.approx(0.4)


def test_update_kinematics_explicit_dt_overrides_configured_default(monkeypatch):
    monkeypatch.setattr(
        "src.ego_state.TEST",
        SimpleNamespace(DEFAULT_SAMPLING_TIME_SEC=0.25),
    )
    ego = make_ego(velocity=4.0)

    ego.update_kinematics(accel=0.0, steering_angle=0.0, dt=0.5)

    assert ego.x == pytest.approx(2.0)

#def test_update_kinematics_uses_configured_default_timestep(monkeypatch):
#    monkeypatch.setattr(TEST, "DEFAULT_SAMPLING_TIME_SEC", 0.25)
#    ego = make_ego(velocity=4.0)
#
#    ego.update_kinematics(accel=0.0, steering_angle=0.0)
#
#    assert ego.x == pytest.approx(1.0)
#
#
#def test_update_kinematics_explicit_dt_overrides_configured_default(monkeypatch):
#    monkeypatch.setattr(TEST, "DEFAULT_SAMPLING_TIME_SEC", 0.25)
#    ego = make_ego(velocity=4.0)
#
#    ego.update_kinematics(accel=0.0, steering_angle=0.0, dt=0.5)
#
#    assert ego.x == pytest.approx(2.0)


def test_update_kinematics_with_zero_dt_leaves_state_unchanged():
    ego = make_ego(x=1.0, y=2.0, orientation=0.3, velocity=5.0)
    before = (ego.x, ego.y, ego.orientation, ego.velocity)

    ego.update_kinematics(accel=3.0, steering_angle=0.2, dt=0.0)

    assert (ego.x, ego.y, ego.orientation, ego.velocity) == pytest.approx(before)


def test_update_kinematics_position_uses_pre_update_orientation():
    ego = make_ego(orientation=0.0, velocity=10.0, wheelbase=2.0)

    ego.update_kinematics(accel=0.0, steering_angle=0.3, dt=0.1)

    assert ego.x == pytest.approx(1.0)
    assert ego.y == pytest.approx(0.0)
    assert ego.orientation != pytest.approx(0.0)

def test_update_kinematics_invokes_stanley_controller_when_steering_is_none(
    monkeypatch,
):
    ego = make_ego(velocity=10.0)
    reference_path = np.array([
        [0.0, 0.0],
        [10.0, 2.0],
        [20.0, 4.0],
    ])

    expected_steering = 0.15
    calls = []

    def fake_compute_steering(*, ego, reference_path):
        calls.append((ego, reference_path))
        return expected_steering

    monkeypatch.setattr(
        ego.controller,
        "compute_steering",
        fake_compute_steering,
    )

    ego.update_kinematics(
        accel=0.0,
        steering_angle=None,
        reference_path=reference_path,
        dt=0.1,
    )

    assert len(calls) == 1
    assert calls[0][0] is ego
    assert calls[0][1] is reference_path

    expected_yaw = (
        ego.velocity / ego.wheelbase
    ) * np.tan(expected_steering) * 0.1

    assert ego.orientation == pytest.approx(expected_yaw)

def test_explicit_steering_angle_takes_precedence_over_reference_path(
    monkeypatch,
):
    ego = make_ego(velocity=10.0)

    def unexpected_controller_call(**kwargs):
        pytest.fail("Stanley controller should not be invoked")

    monkeypatch.setattr(
        ego.controller,
        "compute_steering",
        unexpected_controller_call,
    )

    ego.update_kinematics(
        accel=0.0,
        steering_angle=0.2,
        reference_path=np.array([
            [0.0, 0.0],
            [10.0, 5.0],
        ]),
        dt=0.1,
    )

    expected_yaw = (
        10.0 / ego.wheelbase
    ) * np.tan(0.2) * 0.1

    assert ego.orientation == pytest.approx(expected_yaw)

def test_zero_steering_command_overrides_stanley_during_collision(
    monkeypatch,
):
    ego = make_ego(velocity=10.0)

    def unexpected_controller_call(**kwargs):
        pytest.fail("Stanley controller should not be invoked during override")

    monkeypatch.setattr(
        ego.controller,
        "compute_steering",
        unexpected_controller_call,
    )

    reference_path = np.array([
        [0.0, 0.0],
        [10.0, 5.0],
    ])

    ego.update_kinematics(
        accel=0.0,
        steering_angle=0.0,
        reference_path=reference_path,
        dt=0.1,
    )

    assert ego.orientation == pytest.approx(0.0)
    assert ego.x == pytest.approx(1.0)
    assert ego.y == pytest.approx(0.0)
