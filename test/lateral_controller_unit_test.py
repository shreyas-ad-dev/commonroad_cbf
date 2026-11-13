"""Unit tests for src.lateral_controller.

These tests isolate lane-change path generation and Stanley steering control.
A lightweight ego stub is used because these functions only consume a small
set of EgoState attributes; EgoState itself should have its own unit tests.
"""

import sys
import numpy as np
import pytest
from types import SimpleNamespace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.config import LATERAL, MAP, STANLEY, VEHICLE
from src.lateral_controller import generate_lane_change_path, StanleyController


def make_path_ego(
    x=0.0,
    y=0.0,
    orientation=0.0,
    velocity=10.0,
    road_heading=None,
):
    """Stub exposing the properties consumed by the lateral controller."""
    if road_heading is None:
        road_heading = orientation
    u_hat = np.array([np.cos(road_heading), np.sin(road_heading)], dtype=float)
    n_hat = np.array([-np.sin(road_heading), np.cos(road_heading)], dtype=float)
    return SimpleNamespace(
        position=np.array([x, y], dtype=float),
        road_frame_vectors=(u_hat, n_hat),
        front_axle_position=np.array([x, y], dtype=float),
        orientation=float(orientation),
        velocity=float(velocity),
    )


def make_controller_ego(front_x=0.0, front_y=0.0, orientation=0.0, velocity=10.0):
    return SimpleNamespace(
        front_axle_position=np.array([front_x, front_y], dtype=float),
        orientation=float(orientation),
        velocity=float(velocity),
    )


def straight_path(y=0.0, x_start=-10.0, x_end=50.0, count=20):
    return np.column_stack((
        np.linspace(x_start, x_end, count),
        np.full(count, y, dtype=float),
    ))


# ===========================================================================
# generate_lane_change_path: output shape, defaults and coordinate frame
# ===========================================================================

def test_lane_change_path_has_requested_shape_and_finite_coordinates():
    ego = make_path_ego(x=4.0, y=-3.0, orientation=0.4)

    path = generate_lane_change_path(
        ego,
        target_lane_offset=3.5,
        total_length=100.0,
        num_points=51,
    )

    assert path.shape == (51, 2)
    assert np.all(np.isfinite(path))


def test_lane_change_path_starts_at_ego_position():
    ego = make_path_ego(x=12.0, y=-8.0, orientation=0.7)

    path = generate_lane_change_path(
        ego,
        target_lane_offset=3.5,
        total_length=100.0,
        num_points=20,
    )

    np.testing.assert_allclose(path[0], ego.position, atol=1e-12)


def test_lane_change_path_ends_at_requested_longitudinal_distance_and_offset():
    ego = make_path_ego(x=2.0, y=5.0, road_heading=0.0)
    length = max(
        80.0,
        LATERAL.LANE_CHANGE_START_X + LATERAL.MANEUVER_S_LENGHT + 10.0
    )
    offset = 3.7

    path = generate_lane_change_path(
        ego,
        target_lane_offset=offset,
        total_length=length,
        num_points=101,
    )

    assert path[-1, 0] == pytest.approx(ego.position[0] + length)
    assert path[-1, 1] == pytest.approx(ego.position[1] + offset)


def test_lane_change_path_negative_offset_changes_lane_in_opposite_direction():
    ego = make_path_ego(road_heading=0.0)

    total_length = max(
        100.0,
        LATERAL.LANE_CHANGE_START_X + LATERAL.MANEUVER_S_LENGHT + 10.0
    )
    left = generate_lane_change_path(
        ego, target_lane_offset=3.5, total_length=total_length, num_points=101
    )
    right = generate_lane_change_path(
        ego, target_lane_offset=-3.5, total_length=total_length, num_points=101
    )

    assert left[-1, 1] > ego.position[1]
    assert right[-1, 1] < ego.position[1]
    np.testing.assert_allclose(left[:, 0], right[:, 0])
    np.testing.assert_allclose(left[:, 1] - ego.position[1],
                               -(right[:, 1] - ego.position[1]))


def test_lane_change_path_zero_offset_is_straight():
    ego = make_path_ego(x=3.0, y=4.0, road_heading=0.0)

    path = generate_lane_change_path(
        ego, target_lane_offset=0.0, total_length=50.0, num_points=11
    )

    np.testing.assert_allclose(path[:, 1], 4.0)
    np.testing.assert_allclose(
        path[:, 0], np.linspace(3.0, 53.0, 11)
    )


def test_lane_change_path_uses_default_arguments():
    ego = make_path_ego(x=1.0, y=2.0)

    path = generate_lane_change_path(ego)

    assert path.shape == (LATERAL.DEFAULT_NUM_WAYPOINTS, 2)
    np.testing.assert_allclose(path[0], ego.position)
    assert path[-1, 0] == pytest.approx(
        ego.position[0] + LATERAL.DEFAULT_PATH_LOOKAHEAD_DISTANCE
    )
    assert path[-1, 1] == pytest.approx(
        ego.position[1] + MAP.DEFAULT_LANE_WIDTH_METERS
    )


@pytest.mark.parametrize(
    "heading",
    [0.0, np.pi / 4.0, np.pi / 2.0, np.pi, -np.pi / 3.0],
)
def test_lane_change_path_follows_arbitrary_road_heading(heading):
    ego = make_path_ego(x=10.0, y=-4.0, road_heading=heading)
    length = max(
        60.0,
        LATERAL.LANE_CHANGE_START_X + LATERAL.MANEUVER_S_LENGHT + 10.0
    )
    offset = 3.0

    path = generate_lane_change_path(
        ego, target_lane_offset=offset, total_length=length, num_points=101
    )

    u_hat, n_hat = ego.road_frame_vectors
    expected_end = ego.position + length * u_hat + offset * n_hat

    np.testing.assert_allclose(path[-1], expected_end, atol=1e-10)


def test_lane_change_path_progresses_monotonically_along_road_axis():
    heading = 0.3
    ego = make_path_ego(road_heading=heading)

    path = generate_lane_change_path(
        ego, target_lane_offset=3.5, total_length=100.0, num_points=101
    )

    u_hat, _ = ego.road_frame_vectors
    progress = (path - ego.position) @ u_hat

    assert np.all(np.diff(progress) > 0.0)


def test_lane_change_path_has_no_lateral_offset_before_maneuver_start():
    ego = make_path_ego(road_heading=0.0)
    # Ensure sampled s values include points before LANE_CHANGE_START_X.
    total_length = max(100.0, LATERAL.LANE_CHANGE_START_X + 10.0)

    path = generate_lane_change_path(
        ego,
        target_lane_offset=3.5,
        total_length=total_length,
        num_points=201,
    )

    s = np.linspace(0.0, total_length, 201)
    before_start = s <= LATERAL.LANE_CHANGE_START_X
    np.testing.assert_allclose(path[before_start, 1], 0.0, atol=1e-12)


def test_lane_change_path_lateral_offset_is_clamped_at_target_after_transition():
    ego = make_path_ego(road_heading=0.0)
    total_length = LATERAL.LANE_CHANGE_START_X + LATERAL.MANEUVER_S_LENGHT + 20.0

    path = generate_lane_change_path(
        ego,
        target_lane_offset=3.25,
        total_length=total_length,
        num_points=301,
    )

    s = np.linspace(0.0, total_length, 301)
    after_transition = s >= (
        LATERAL.LANE_CHANGE_START_X + LATERAL.MANEUVER_S_LENGHT
    )
    np.testing.assert_allclose(path[after_transition, 1], 3.25, atol=1e-10)


def test_lane_change_path_does_not_mutate_ego_position_or_frame_vectors():
    ego = make_path_ego(x=2.0, y=3.0, road_heading=0.6)
    original_position = ego.position.copy()
    original_u, original_n = (v.copy() for v in ego.road_frame_vectors)

    generate_lane_change_path(ego, num_points=25)

    np.testing.assert_array_equal(ego.position, original_position)
    np.testing.assert_array_equal(ego.road_frame_vectors[0], original_u)
    np.testing.assert_array_equal(ego.road_frame_vectors[1], original_n)


def test_lane_change_path_with_one_waypoint_returns_ego_position():
    ego = make_path_ego(x=5.0, y=6.0, road_heading=0.2)

    path = generate_lane_change_path(
        ego, target_lane_offset=3.5, total_length=100.0, num_points=1
    )

    assert path.shape == (1, 2)
    np.testing.assert_allclose(path[0], ego.position)


# ===========================================================================
# StanleyController: initialization and nominal behaviour
# ===========================================================================

def test_stanley_controller_initialization_uses_configured_defaults():
    controller = StanleyController()

    assert controller.k == pytest.approx(STANLEY.DEFAULT_STANLEY_GAIN)
    assert controller.k_soft == pytest.approx(STANLEY.DEFAULT_SOFTENING_GAIN)
    assert controller.max_steer == pytest.approx(
        np.radians(STANLEY.DEFAULT_MAX_STEER_DEG)
    )
    assert controller.wheelbase == pytest.approx(VEHICLE.DEFAULT_WHEELBASE)
    assert controller.steering == 0


def test_stanley_controller_initialization_accepts_custom_parameters():
    controller = StanleyController(
        k=1.2,
        k_soft=0.7,
        max_steer_deg=30.0,
        wheelbase=3.1,
    )

    assert controller.k == pytest.approx(1.2)
    assert controller.k_soft == pytest.approx(0.7)
    assert controller.max_steer == pytest.approx(np.radians(30.0))
    assert controller.wheelbase == pytest.approx(3.1)


def test_stanley_returns_zero_for_aligned_vehicle_on_straight_path():
    controller = StanleyController()
    ego = make_controller_ego(front_x=0.0, front_y=0.0,
                               orientation=0.0, velocity=10.0)

    steering = controller.compute_steering(ego, straight_path())

    assert steering == pytest.approx(0.0, abs=1e-12)


def test_stanley_returns_python_float():
    controller = StanleyController()
    ego = make_controller_ego()

    steering = controller.compute_steering(ego, straight_path())

    assert isinstance(steering, float)


def test_stanley_steering_is_zero_when_cross_track_and_heading_errors_are_zero():
    controller = StanleyController(k=0.8, k_soft=1.0)
    ego = make_controller_ego(front_x=5.0, front_y=0.0,
                               orientation=0.0, velocity=5.0)
    path = straight_path(y=0.0)

    assert controller.compute_steering(ego, path) == pytest.approx(0.0)


# ===========================================================================
# StanleyController: cross-track error response and sign
# ===========================================================================

def test_stanley_corrects_positive_cross_track_error_with_negative_steering():
    controller = StanleyController(k=1.0, k_soft=1.0, max_steer_deg=80.0)
    ego = make_controller_ego(front_x=0.0, front_y=2.0,
                               orientation=0.0, velocity=5.0)

    steering = controller.compute_steering(ego, straight_path())

    assert steering < 0.0


def test_stanley_corrects_negative_cross_track_error_with_positive_steering():
    controller = StanleyController(k=1.0, k_soft=1.0, max_steer_deg=80.0)
    ego = make_controller_ego(front_x=0.0, front_y=-2.0,
                               orientation=0.0, velocity=5.0)

    steering = controller.compute_steering(ego, straight_path())

    assert steering > 0.0


def test_stanley_cross_track_correction_magnitude_increases_with_gain():
    ego = make_controller_ego(front_x=0.0, front_y=2.0,
                               orientation=0.0, velocity=5.0)
    path = straight_path()

    low_gain = StanleyController(k=0.2, k_soft=1.0, max_steer_deg=80.0)
    high_gain = StanleyController(k=1.0, k_soft=1.0, max_steer_deg=80.0)

    low = low_gain.compute_steering(ego, path)
    high = high_gain.compute_steering(ego, path)

    assert abs(high) > abs(low)


def test_stanley_cross_track_correction_decreases_as_speed_increases():
    path = straight_path()
    slow_ego = make_controller_ego(front_x=0.0, front_y=2.0,
                                   orientation=0.0, velocity=2.0)
    fast_ego = make_controller_ego(front_x=0.0, front_y=2.0,
                                   orientation=0.0, velocity=20.0)
    controller_slow = StanleyController(k=1.0, k_soft=1.0, max_steer_deg=80.0)
    controller_fast = StanleyController(k=1.0, k_soft=1.0, max_steer_deg=80.0)

    slow = controller_slow.compute_steering(slow_ego, path)
    fast = controller_fast.compute_steering(fast_ego, path)

    assert abs(slow) > abs(fast)


def test_stanley_softening_gain_reduces_low_speed_cross_track_response():
    ego = make_controller_ego(front_x=0.0, front_y=2.0,
                               orientation=0.0, velocity=0.1)
    path = straight_path()

    low_softening = StanleyController(
        k=1.0, k_soft=0.1, max_steer_deg=80.0
    )
    high_softening = StanleyController(
        k=1.0, k_soft=5.0, max_steer_deg=80.0
    )

    low = low_softening.compute_steering(ego, path)
    high = high_softening.compute_steering(ego, path)

    assert abs(low) > abs(high)


# ===========================================================================
# StanleyController: heading error and angle wrapping
# ===========================================================================

def test_stanley_heading_error_produces_opposite_steering_correction():
    controller = StanleyController(k=0.0, k_soft=1.0, max_steer_deg=80.0)
    ego = make_controller_ego(front_x=0.0, front_y=0.0,
                               orientation=0.2, velocity=5.0)

    steering = controller.compute_steering(ego, straight_path())

    assert steering == pytest.approx(-0.2)


@pytest.mark.parametrize(
    "orientation, expected",
    [
        (np.pi - 0.01, -(np.pi - 0.01)),
        (-np.pi + 0.01, np.pi - 0.01),
        (2.0 * np.pi + 0.2, -0.2),
        (-2.0 * np.pi - 0.3, 0.3),
    ],
)
def test_stanley_normalizes_heading_error(orientation, expected):
    controller = StanleyController(k=0.0, k_soft=1.0, max_steer_deg=179.0)
    ego = make_controller_ego(
        front_x=0.0, front_y=0.0,
        orientation=orientation, velocity=5.0
    )

    steering = controller.compute_steering(ego, straight_path())

    assert steering == pytest.approx(expected, abs=1e-2)


def test_stanley_uses_path_tangent_at_nearest_waypoint():
    # A vertical path has yaw pi/2. With k=0, steering is -heading_error.
    path = np.column_stack((
        np.zeros(10),
        np.linspace(-5.0, 5.0, 10),
    ))
    ego = make_controller_ego(
        front_x=0.0, front_y=0.0,
        orientation=np.pi / 2.0, velocity=5.0
    )
    controller = StanleyController(k=0.0, max_steer_deg=80.0)

    assert controller.compute_steering(ego, path) == pytest.approx(0.0, abs=1e-10)


# ===========================================================================
# StanleyController: nearest waypoint and endpoint tangent branches
# ===========================================================================

def test_stanley_uses_nearest_reference_waypoint():
    path = np.array([
        [0.0, 10.0],
        [5.0, 0.0],
        [10.0, 10.0],
    ])
    ego = make_controller_ego(
        front_x=5.0, front_y=0.0, orientation=0.0, velocity=5.0
    )
    controller = StanleyController(k=0.0, max_steer_deg=80.0)

    # Nearest waypoint is the middle point; its forward tangent is diagonal.
    expected_yaw = np.arctan2(10.0, 5.0)
    steering = controller.compute_steering(ego, path)

    assert steering == pytest.approx(expected_yaw)


def test_stanley_last_waypoint_uses_backward_tangent_branch():
    path = np.array([
        [0.0, 0.0],
        [5.0, 0.0],
        [10.0, 5.0],
    ])
    ego = make_controller_ego(
        front_x=10.0, front_y=5.0,
        orientation=0.0, velocity=5.0
    )
    controller = StanleyController(k=0.0, max_steer_deg=80.0)

    expected_yaw = np.arctan2(5.0, 5.0)
    steering = controller.compute_steering(ego, path)

    assert steering == pytest.approx(expected_yaw)


def test_stanley_first_waypoint_uses_forward_tangent_branch():
    path = np.array([
        [0.0, 0.0],
        [5.0, 5.0],
        [10.0, 5.0],
    ])
    ego = make_controller_ego(
        front_x=0.0, front_y=0.0,
        orientation=0.0, velocity=5.0
    )
    controller = StanleyController(k=0.0, max_steer_deg=80.0)

    expected_yaw = np.arctan2(5.0, 5.0)
    steering = controller.compute_steering(ego, path)

    assert steering == pytest.approx(expected_yaw)


# ===========================================================================
# StanleyController: steering saturation and logging
# ===========================================================================

def test_stanley_saturates_positive_steering_at_maximum():
    max_deg = 20.0
    controller = StanleyController(
        k=0.0, max_steer_deg=max_deg
    )
    ego = make_controller_ego(
        front_x=0.0, front_y=0.0,
        orientation=-1.0, velocity=5.0
    )

    steering = controller.compute_steering(ego, straight_path())

    assert steering == pytest.approx(np.radians(max_deg))


def test_stanley_saturates_negative_steering_at_minimum():
    max_deg = 20.0
    controller = StanleyController(
        k=0.0, max_steer_deg=max_deg
    )
    ego = make_controller_ego(
        front_x=0.0, front_y=0.0,
        orientation=1.0, velocity=5.0
    )

    steering = controller.compute_steering(ego, straight_path())

    assert steering == pytest.approx(-np.radians(max_deg))


def test_stanley_unsaturated_command_is_within_limits():
    max_angle = np.radians(25.0)
    controller = StanleyController(k=0.5, max_steer_deg=25.0)
    ego = make_controller_ego(
        front_x=0.0, front_y=0.5,
        orientation=0.05, velocity=10.0
    )

    steering = controller.compute_steering(ego, straight_path())

    assert -max_angle <= steering <= max_angle


def test_stanley_log_data_exposes_internal_steering_value():
    controller = StanleyController(k=0.0, max_steer_deg=20.0)
    ego = make_controller_ego(
        front_x=0.0, front_y=0.0,
        orientation=-1.0, velocity=5.0
    )

    returned_steering = controller.compute_steering(ego, straight_path())
    log_data = controller.get_log_data()

    assert set(log_data) == {"steering"}
    # Current implementation stores the unsaturated command in self.steering
    # and returns the saturated command. This assertion documents that detail.
    assert log_data["steering"] == pytest.approx(1.0)
    assert returned_steering == pytest.approx(np.radians(20.0))


def test_stanley_log_data_updates_after_each_compute_call():
    controller = StanleyController(k=0.0, max_steer_deg=80.0)
    first_ego = make_controller_ego(orientation=0.1)
    second_ego = make_controller_ego(orientation=0.3)

    controller.compute_steering(first_ego, straight_path())
    assert controller.get_log_data()["steering"] == pytest.approx(-0.1)

    controller.compute_steering(second_ego, straight_path())
    assert controller.get_log_data()["steering"] == pytest.approx(-0.3)


# ===========================================================================
# Current input-contract / edge behaviour
# ===========================================================================

def test_stanley_raises_for_empty_reference_path():
    controller = StanleyController()
    ego = make_controller_ego()

    with pytest.raises(ValueError):
        controller.compute_steering(ego, np.empty((0, 2)))


def test_stanley_single_waypoint_returns_finite_steering():
    """Current implementation falls back to a zero tangent for one point."""
    controller = StanleyController(max_steer_deg=80.0)
    ego = make_controller_ego(
        front_x=0.0, front_y=0.0,
        orientation=0.1, velocity=5.0
    )

    steering = controller.compute_steering(
        ego, np.array([[0.0, 0.0]])
    )

    assert np.isfinite(steering)
    assert steering == pytest.approx(-0.1)


def test_stanley_zero_velocity_remains_finite_with_positive_softening_gain():
    controller = StanleyController(k=1.0, k_soft=1.0, max_steer_deg=80.0)
    ego = make_controller_ego(
        front_x=0.0, front_y=1.0,
        orientation=0.0, velocity=0.0
    )

    steering = controller.compute_steering(ego, straight_path())

    assert np.isfinite(steering)


def test_stanley_uses_front_axle_position_not_ego_reference_position():
    """The controller's cross-track and nearest-point calculations use front axle."""
    path = straight_path(y=0.0)
    ego = SimpleNamespace(
        position=np.array([0.0, 100.0]),
        front_axle_position=np.array([0.0, 2.0]),
        orientation=0.0,
        velocity=5.0,
    )
    controller = StanleyController(k=1.0, max_steer_deg=80.0)

    steering = controller.compute_steering(ego, path)

    assert steering < 0.0
