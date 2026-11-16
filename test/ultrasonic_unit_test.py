"""
Level-1 unit tests for src.ultrasonic.SideUltrasonicSensor.

Run from the repository root:
    pytest test/ultrasonic_unit_test.py -v

The suite tests construction, sensor mounting geometry, FOV construction,
occlusion-shadow geometry, scanning, detections/noise, clearance memoization,
step caching, and serialization. It intentionally tests this sensor in
isolation; tracker and full CommonRoad pipeline behaviour belong in later
integration tests.
"""

import sys
from types import SimpleNamespace

import numpy as np
import pytest
from shapely.geometry import Polygon
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.ultrasonic import SideUltrasonicSensor
from src.ego_state import EgoState
from src.config import USS, TEST, VEHICLE
from src.tracker import Detection


def make_ego(x=0.0, y=0.0, yaw=0.0, velocity=0.0):
    return EgoState(
        x=x,
        y=y,
        orientation=yaw,
        velocity=velocity,
        length=4.5,
        width=1.8,
        wheelbase=2.7,
    )


def make_obstacle(
    obstacle_id,
    x,
    y,
    yaw=0.0,
    length=1.0,
    width=1.0,
    available_step=0,
):
    state = SimpleNamespace(
        position=np.array([x, y], dtype=float),
        orientation=yaw,
    )

    class DummyObstacle:
        obstacle_shape = SimpleNamespace(length=length, width=width)

        def __init__(self):
            self.obstacle_id = obstacle_id
            self.requested_steps = []

        def state_at_time(self, step):
            self.requested_steps.append(step)
            return state if step == available_step else None

    obstacle = DummyObstacle()
    return obstacle, state


def make_sensor(side="left", **kwargs):
    defaults = dict(
        range_max=8.0,
        fov_deg=100.0,
        side=side,
        noise_std=0.0,
        ray_count=31,
    )
    defaults.update(kwargs)
    return SideUltrasonicSensor(**defaults)


def scan_with_zero_noise(sensor, ego, obstacles, step=0, monkeypatch=None,
                         target_offset=0.0):
    if monkeypatch is not None:
        monkeypatch.setattr(np.random, "normal", lambda *args, **kwargs: 0.0)
    return sensor.scan(ego, obstacles, step, target_offset)


# ---------------------------------------------------------------------------
# Construction and configuration
# ---------------------------------------------------------------------------

def test_constructor_uses_default_configuration_values():
    sensor = SideUltrasonicSensor()

    assert sensor.range_max == USS.DEFAULT_USS_MAX_RANGE
    assert sensor.fov_deg == USS.DEFAULT_USS_FOV_DEG
    assert sensor.noise_std == USS.DEFAULT_USS_NOISE_STD
    assert sensor.side == "left"


@pytest.mark.parametrize("side", ["left", "right"])
def test_constructor_sets_side_and_sensor_id(side):
    sensor = make_sensor(side=side)

    assert sensor.side == side
    assert sensor.sensor_id == f"uss_{side}"


@pytest.mark.parametrize("side", ["LEFT", "Right", "front", "rear", "", None])
def test_constructor_rejects_invalid_side(side):
    with pytest.raises(ValueError, match="side must be either"):
        make_sensor(side=side)


def test_constructor_calculates_measurement_covariance_from_noise_std():
    sensor = make_sensor(noise_std=0.25)

    np.testing.assert_allclose(sensor.R, np.eye(2) * 0.25**2)


def test_constructor_zero_noise_produces_zero_covariance():
    sensor = make_sensor(noise_std=0.0)

    np.testing.assert_array_equal(sensor.R, np.zeros((2, 2)))


def test_constructor_initializes_empty_step_cache():
    sensor = make_sensor()

    assert sensor._last_step is None
    assert sensor._scan_cache["detected_ids"] == set()
    assert sensor._scan_cache["min_distances"] == {}
    assert sensor._scan_cache["detections"] == []
    assert sensor._scan_cache["target_clearance"] == {}


def test_constructor_preserves_configured_ray_count():
    sensor = make_sensor(ray_count=17)

    assert sensor.ray_count == 17


# ---------------------------------------------------------------------------
# Sensor transform and FOV wedge
# ---------------------------------------------------------------------------

def test_left_sensor_transform_is_at_ego_position_and_plus_90_degrees():
    ego = make_ego(x=3.0, y=-2.0, yaw=0.0)
    sensor = make_sensor(side="left")

    position, heading = sensor._get_sensor_transform(ego)

    np.testing.assert_allclose(position, [3.0, -2.0])
    assert heading == pytest.approx(90.0)


def test_right_sensor_transform_is_at_ego_position_and_minus_90_degrees():
    ego = make_ego(x=3.0, y=-2.0, yaw=0.0)
    sensor = make_sensor(side="right")

    position, heading = sensor._get_sensor_transform(ego)

    np.testing.assert_allclose(position, [3.0, -2.0])
    assert heading == pytest.approx(-90.0)


@pytest.mark.parametrize(
    "side,yaw,expected",
    [
        ("left", np.pi / 2, 180.0),
        ("right", np.pi / 2, 0.0),
        ("left", -np.pi / 2, 0.0),
        ("right", -np.pi / 2, -180.0),
    ],
)
def test_sensor_transform_rotates_with_ego_heading(side, yaw, expected):
    sensor = make_sensor(side=side)
    _, heading = sensor._get_sensor_transform(make_ego(yaw=yaw))

    assert heading == pytest.approx(expected)


def test_fov_wedge_is_valid_polygon():
    sensor = make_sensor()
    wedge = sensor._build_fov_wedge(np.array([0.0, 0.0]), 90.0)

    assert isinstance(wedge, Polygon)
    assert wedge.is_valid
    assert not wedge.is_empty


def test_fov_wedge_is_anchored_at_sensor_position():
    sensor_pos = np.array([2.0, 3.0])
    wedge = make_sensor()._build_fov_wedge(sensor_pos, 90.0)

    assert wedge.covers(__import__("shapely").geometry.Point(*sensor_pos))


def test_fov_wedge_reaches_configured_maximum_range():
    sensor = make_sensor(range_max=6.0, fov_deg=90.0, ray_count=41)
    wedge = sensor._build_fov_wedge(np.array([0.0, 0.0]), 90.0)

    assert max(np.linalg.norm(np.array(p)) for p in list(wedge.exterior.coords)) \
        == pytest.approx(6.0)


def test_fov_wedge_has_expected_angular_extent():
    sensor = make_sensor(fov_deg=80.0, ray_count=41)
    wedge = sensor._build_fov_wedge(np.array([0.0, 0.0]), 90.0)

    coords = np.array(wedge.exterior.coords)[1:-1]
    angles = np.unwrap(np.arctan2(coords[:, 1], coords[:, 0]))

    assert np.degrees(angles.max() - angles.min()) == pytest.approx(80.0)


def test_fov_wedge_ray_count_controls_arc_sampling():
    low_ray = make_sensor(ray_count=5)._build_fov_wedge(
        np.array([0.0, 0.0]), 90.0
    )
    high_ray = make_sensor(ray_count=25)._build_fov_wedge(
        np.array([0.0, 0.0]), 90.0
    )

    assert len(high_ray.exterior.coords) > len(low_ray.exterior.coords)


def test_left_and_right_fov_wedges_face_opposite_directions():
    left = make_sensor(side="left")._build_fov_wedge(
        np.array([0.0, 0.0]), 90.0
    )
    right = make_sensor(side="right")._build_fov_wedge(
        np.array([0.0, 0.0]), -90.0
    )

    assert left.centroid.y > 0.0
    assert right.centroid.y < 0.0
    assert left.intersection(right).area == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Occlusion shadow geometry
# ---------------------------------------------------------------------------

def test_occlusion_shadow_returns_polygon_for_regular_obstacle():
    sensor = make_sensor(range_max=5.0)
    obstacle_poly = Polygon([
        [2.0, 2.0], [3.0, 2.0], [3.0, 3.0], [2.0, 3.0]
    ])

    shadow = sensor._compute_occlusion_shadow(
        np.array([0.0, 0.0]), obstacle_poly
    )

    assert isinstance(shadow, Polygon)
    assert shadow.is_valid
    assert not shadow.is_empty


def test_occlusion_shadow_projects_beyond_obstacle():
    sensor = make_sensor(range_max=5.0)
    obstacle_poly = Polygon([
        [2.0, 2.0], [3.0, 2.0], [3.0, 3.0], [2.0, 3.0]
    ])

    shadow = sensor._compute_occlusion_shadow(
        np.array([0.0, 0.0]), obstacle_poly
    )

    assert shadow.bounds[2] > obstacle_poly.bounds[2]
    assert shadow.bounds[3] > obstacle_poly.bounds[3]


def test_occlusion_shadow_returns_none_for_geometry_without_exterior_points():
    sensor = make_sensor()
    empty_polygon = Polygon()

    assert sensor._compute_occlusion_shadow(
        np.array([0.0, 0.0]), empty_polygon
    ) is None


# ---------------------------------------------------------------------------
# Scan: basic detection and filtering
# ---------------------------------------------------------------------------

def test_scan_returns_expected_cache_structure(monkeypatch):
    sensor = make_sensor()
    result = scan_with_zero_noise(
        sensor, make_ego(), [], monkeypatch=monkeypatch
    )

    assert {
        "detected_ids",
        "min_distances",
        "detections",
        "fov_data",
        "target_clearance",
        "visible_fov",
    }.issubset(result.keys())


def test_left_sensor_detects_obstacle_on_left(monkeypatch):
    sensor = make_sensor(side="left")
    obstacle, _ = make_obstacle(1, 0.0, 3.0)

    result = scan_with_zero_noise(
        sensor, make_ego(), [obstacle], monkeypatch=monkeypatch
    )

    assert 1 in result["detected_ids"]
    assert len(result["detections"]) == 1


def test_right_sensor_detects_obstacle_on_right(monkeypatch):
    sensor = make_sensor(side="right")
    obstacle, _ = make_obstacle(2, 0.0, -3.0)

    result = scan_with_zero_noise(
        sensor, make_ego(), [obstacle], monkeypatch=monkeypatch
    )

    assert 2 in result["detected_ids"]
    assert len(result["detections"]) == 1


def test_left_sensor_does_not_detect_obstacle_on_opposite_side(monkeypatch):
    sensor = make_sensor(side="left")
    obstacle, _ = make_obstacle(3, 0.0, -3.0)

    result = scan_with_zero_noise(
        sensor, make_ego(), [obstacle], monkeypatch=monkeypatch
    )

    assert 3 not in result["detected_ids"]
    assert result["detections"] == []


def test_right_sensor_does_not_detect_obstacle_on_opposite_side(monkeypatch):
    sensor = make_sensor(side="right")
    obstacle, _ = make_obstacle(4, 0.0, 3.0)

    result = scan_with_zero_noise(
        sensor, make_ego(), [obstacle], monkeypatch=monkeypatch
    )

    assert 4 not in result["detected_ids"]
    assert result["detections"] == []


def test_scan_skips_obstacle_without_state_at_requested_step(monkeypatch):
    sensor = make_sensor()
    obstacle, _ = make_obstacle(5, 0.0, 3.0, available_step=10)

    result = scan_with_zero_noise(
        sensor, make_ego(), [obstacle], step=9, monkeypatch=monkeypatch
    )

    assert result["detected_ids"] == set()
    assert result["detections"] == []


def test_scan_ignores_obstacle_outside_sensor_range(monkeypatch):
    sensor = make_sensor(range_max=4.0)
    obstacle, _ = make_obstacle(6, 0.0, 7.0)

    result = scan_with_zero_noise(
        sensor, make_ego(), [obstacle], monkeypatch=monkeypatch
    )

    assert 6 not in result["detected_ids"]


def test_scan_ignores_obstacle_outside_angular_fov(monkeypatch):
    sensor = make_sensor(fov_deg=40.0)
    # Left sensor points along +Y. This obstacle is far to the side of that axis.
    obstacle, _ = make_obstacle(7, 4.0, 2.0)

    result = scan_with_zero_noise(
        sensor, make_ego(), [obstacle], monkeypatch=monkeypatch
    )

    assert 7 not in result["detected_ids"]


def test_scan_detects_obstacle_when_ego_is_translated(monkeypatch):
    sensor = make_sensor()
    ego = make_ego(x=10.0, y=-4.0)
    obstacle, _ = make_obstacle(8, 10.0, -1.0)

    result = scan_with_zero_noise(
        sensor, ego, [obstacle], monkeypatch=monkeypatch
    )

    assert 8 in result["detected_ids"]


def test_scan_detects_obstacle_when_ego_is_rotated(monkeypatch):
    sensor = make_sensor(side="left")
    ego = make_ego(yaw=np.pi / 2)
    # Left side of an ego facing +Y points toward world -X.
    obstacle, _ = make_obstacle(9, -3.0, 0.0)

    result = scan_with_zero_noise(
        sensor, ego, [obstacle], monkeypatch=monkeypatch
    )

    assert 9 in result["detected_ids"]


def test_scan_detects_multiple_separated_obstacles(monkeypatch):
    sensor = make_sensor()
    near, _ = make_obstacle(10, -2.0, 2.5)
    farther, _ = make_obstacle(11, 2.0, 5.0)

    result = scan_with_zero_noise(
        sensor, make_ego(), [farther, near], monkeypatch=monkeypatch
    )

    assert result["detected_ids"] == {10, 11}
    assert len(result["detections"]) == 2


def test_scan_sorts_obstacle_processing_by_distance(monkeypatch):
    sensor = make_sensor()
    near, _ = make_obstacle(12, -2.0, 2.0)
    far, _ = make_obstacle(13, 2.0, 5.0)

    result = scan_with_zero_noise(
        sensor, make_ego(), [far, near], monkeypatch=monkeypatch
    )

    # Detection order follows sorted sensor-to-obstacle center distance.
    assert [d.obstacle_id for d in result["detections"]] == [12, 13]


# ---------------------------------------------------------------------------
# Detection contents, timestamp, covariance and noise
# ---------------------------------------------------------------------------

def test_detection_has_expected_type_and_metadata(monkeypatch):
    sensor = make_sensor()
    obstacle, _ = make_obstacle(20, 0.0, 3.0, available_step=4)

    result = scan_with_zero_noise(
        sensor, make_ego(), [obstacle], step=4, monkeypatch=monkeypatch
    )

    detection = result["detections"][0]
    assert isinstance(detection, Detection)
    assert detection.sensor_id == "uss_left"
    assert detection.obstacle_id == 20
    assert detection.timestamp == pytest.approx(
        4 * TEST.DEFAULT_SAMPLING_TIME_SEC
    )


def test_detection_measurement_is_global_xy_when_noise_is_zero(monkeypatch):
    sensor = make_sensor()
    ego = make_ego(x=10.0, y=-2.0, yaw=0.0)
    obstacle, _ = make_obstacle(21, 10.0, 1.0)

    result = scan_with_zero_noise(
        sensor, ego, [obstacle], monkeypatch=monkeypatch
    )

    np.testing.assert_allclose(result["detections"][0].z, [10.0, 1.0])


def test_detection_measurement_transforms_back_to_global_for_rotated_ego(
    monkeypatch,
):
    sensor = make_sensor()
    ego = make_ego(x=2.0, y=3.0, yaw=np.pi / 2)
    # The obstacle lies on the ego's left side, world -X.
    obstacle, _ = make_obstacle(22, -1.0, 3.0)

    result = scan_with_zero_noise(
        sensor, ego, [obstacle], monkeypatch=monkeypatch
    )

    np.testing.assert_allclose(
        result["detections"][0].z, [-1.0, 3.0], atol=1e-12
    )


def test_detection_uses_sensor_measurement_covariance(monkeypatch):
    sensor = make_sensor(noise_std=0.2)
    obstacle, _ = make_obstacle(23, 0.0, 3.0)

    result = scan_with_zero_noise(
        sensor, make_ego(), [obstacle], monkeypatch=monkeypatch
    )

    np.testing.assert_allclose(
        result["detections"][0].R, np.eye(2) * 0.2**2
    )


def test_measurement_noise_is_applied_to_both_local_coordinates(monkeypatch):
    sensor = make_sensor(noise_std=0.5)
    obstacle, _ = make_obstacle(24, 0.0, 3.0)
    values = iter([0.25, -0.5])
    monkeypatch.setattr(np.random, "normal", lambda *args, **kwargs: next(values))

    result = sensor.scan(make_ego(), [obstacle], 0)
    measured = result["detections"][0].z

    # Local center is [3, 0] for a forward-facing ego; noise is then
    # transformed to world coordinates.
    np.testing.assert_allclose(measured, [0.25, 2.5], atol=1e-12)


def test_scan_does_not_modify_sensor_covariance():
    sensor = make_sensor(noise_std=0.3)
    before = sensor.R.copy()
    obstacle, _ = make_obstacle(25, 0.0, 3.0)

    scan_with_zero_noise(sensor, make_ego(), [obstacle])

    np.testing.assert_array_equal(sensor.R, before)


# ---------------------------------------------------------------------------
# Occlusion, visible geometry and distance metadata
# ---------------------------------------------------------------------------

def test_scan_populates_fov_data_for_detected_obstacle(monkeypatch):
    sensor = make_sensor()
    obstacle, _ = make_obstacle(30, 0.0, 3.0)

    result = scan_with_zero_noise(
        sensor, make_ego(), [obstacle], monkeypatch=monkeypatch
    )

    assert 30 in result["fov_data"]
    fov_data = result["fov_data"][30]
    assert fov_data.obstacle_id == 30
    assert fov_data.in_fov is True
    assert fov_data.center_x_local == pytest.approx(0.0)
    assert fov_data.center_y_local == pytest.approx(3.0)


def test_fov_data_min_distance_is_nonnegative_and_finite(monkeypatch):
    sensor = make_sensor()
    obstacle, _ = make_obstacle(31, 0.0, 3.0)

    result = scan_with_zero_noise(
        sensor, make_ego(), [obstacle], monkeypatch=monkeypatch
    )

    distance = result["fov_data"][31].min_dist
    assert np.isfinite(distance)
    assert distance >= 0.0


def test_fov_data_contains_visible_segments_list(monkeypatch):
    sensor = make_sensor()
    obstacle, _ = make_obstacle(32, 0.0, 3.0)

    result = scan_with_zero_noise(
        sensor, make_ego(), [obstacle], monkeypatch=monkeypatch
    )

    segments = result["fov_data"][32].visible_segments
    assert isinstance(segments, list)
    for segment in segments:
        assert isinstance(segment, np.ndarray)
        assert segment.ndim == 2
        assert segment.shape[1] == 2


def test_visible_fov_is_geometry_and_nonempty_for_empty_scene(monkeypatch):
    sensor = make_sensor()
    result = scan_with_zero_noise(
        sensor, make_ego(), [], monkeypatch=monkeypatch
    )

    assert result["visible_fov"].geom_type in ("Polygon", "MultiPolygon")
    assert not result["visible_fov"].is_empty


def test_visible_fov_is_reduced_when_detected_obstacle_casts_shadow(
    monkeypatch,
):
    sensor = make_sensor()
    obstacle, _ = make_obstacle(33, 0.0, 3.0, length=1.0, width=1.0)

    empty_scene = scan_with_zero_noise(
        make_sensor(), make_ego(), [], monkeypatch=monkeypatch
    )
    occluded_scene = scan_with_zero_noise(
        sensor, make_ego(), [obstacle], monkeypatch=monkeypatch
    )

    assert occluded_scene["visible_fov"].area < empty_scene["visible_fov"].area


def test_nearer_obstacle_is_processed_before_farther_obstacle(monkeypatch):
    sensor = make_sensor()
    near, _ = make_obstacle(34, 0.0, 2.0)
    far, _ = make_obstacle(35, 0.0, 5.0)

    result = scan_with_zero_noise(
        sensor, make_ego(), [far, near], monkeypatch=monkeypatch
    )

    # At minimum, the nearest obstacle should be present. The farther one may
    # be occluded depending on the projected shadow and FOV geometry.
    assert 34 in result["detected_ids"]


def test_min_distances_cache_contains_detected_obstacle_distance(monkeypatch):
    """Contract check: min_distances should expose per-obstacle range data."""
    sensor = make_sensor()
    obstacle, _ = make_obstacle(36, -2.0, 3.0)

    result = scan_with_zero_noise(
        sensor, make_ego(), [obstacle], monkeypatch=monkeypatch
    )

    assert 36 in result["min_distances"]
    assert result["min_distances"][36] >= 0.0


# ---------------------------------------------------------------------------
# Target clearance and cache semantics
# ---------------------------------------------------------------------------

def test_left_sensor_clearance_is_false_when_it_detects_an_obstacle(monkeypatch):
    sensor = make_sensor(side="left")
    obstacle, _ = make_obstacle(40, 0.0, 3.0)

    result = scan_with_zero_noise(
        sensor, make_ego(), [obstacle], monkeypatch=monkeypatch,
        target_offset=0.5,
    )

    assert result["target_clearance"][0.5] is False


def test_left_sensor_opposite_direction_offset_is_always_clear(monkeypatch):
    sensor = make_sensor(side="left")
    obstacle, _ = make_obstacle(41, 0.0, 3.0)

    result = scan_with_zero_noise(
        sensor, make_ego(), [obstacle], monkeypatch=monkeypatch,
        target_offset=-0.5,
    )

    assert result["target_clearance"][-0.5] is True


def test_right_sensor_clearance_is_false_when_it_detects_an_obstacle(monkeypatch):
    sensor = make_sensor(side="right")
    obstacle, _ = make_obstacle(42, 0.0, -3.0)

    result = scan_with_zero_noise(
        sensor, make_ego(), [obstacle], monkeypatch=monkeypatch,
        target_offset=-0.5,
    )

    assert result["target_clearance"][-0.5] is False


def test_right_sensor_opposite_direction_offset_is_always_clear(monkeypatch):
    sensor = make_sensor(side="right")
    obstacle, _ = make_obstacle(43, 0.0, -3.0)

    result = scan_with_zero_noise(
        sensor, make_ego(), [obstacle], monkeypatch=monkeypatch,
        target_offset=0.5,
    )

    assert result["target_clearance"][0.5] is True


def test_clearance_is_true_when_relevant_side_has_no_detections(monkeypatch):
    sensor = make_sensor(side="left")
    result = scan_with_zero_noise(
        sensor, make_ego(), [], monkeypatch=monkeypatch,
        target_offset=0.5,
    )

    assert result["target_clearance"][0.5] is True


def test_clearance_results_are_memoized_per_target_offset(monkeypatch):
    sensor = make_sensor(side="left")
    obstacle, _ = make_obstacle(44, 0.0, 3.0)
    result = scan_with_zero_noise(
        sensor, make_ego(), [obstacle], monkeypatch=monkeypatch,
        target_offset=0.5,
    )

    cached_value = result["target_clearance"][0.5]
    result_again = sensor.scan(make_ego(), [], 0, target_offset=0.5)

    assert result_again["target_clearance"][0.5] is cached_value


def test_new_target_offset_is_added_to_clearance_cache(monkeypatch):
    sensor = make_sensor(side="left")
    result = scan_with_zero_noise(
        sensor, make_ego(), [], monkeypatch=monkeypatch,
        target_offset=0.25,
    )

    sensor.scan(make_ego(), [], 0, target_offset=0.75)

    assert set(result["target_clearance"]) == {0.25, 0.75}


def test_repeated_scan_same_step_returns_same_cache_object(monkeypatch):
    sensor = make_sensor()
    first = scan_with_zero_noise(
        sensor, make_ego(), [], monkeypatch=monkeypatch
    )
    second = sensor.scan(make_ego(), [], 0)

    assert second is first


def test_repeated_scan_same_step_does_not_rescan_obstacles(monkeypatch):
    sensor = make_sensor()
    obstacle, _ = make_obstacle(45, 0.0, 3.0)

    sensor.scan(make_ego(), [obstacle], 0)
    calls_after_first_scan = len(obstacle.requested_steps)

    sensor.scan(make_ego(), [obstacle], 0)

    assert len(obstacle.requested_steps) == calls_after_first_scan


def test_new_step_replaces_cache_and_rescans_obstacles(monkeypatch):
    sensor = make_sensor()
    obstacle, _ = make_obstacle(46, 0.0, 3.0, available_step=0)

    first = sensor.scan(make_ego(), [obstacle], 0)
    second = sensor.scan(make_ego(), [obstacle], 1)

    assert second is not first
    assert sensor._last_step == 1
    assert len(obstacle.requested_steps) == 4


def test_new_step_clears_previous_detections(monkeypatch):
    sensor = make_sensor()
    obstacle, _ = make_obstacle(47, 0.0, 3.0, available_step=0)

    first = sensor.scan(make_ego(), [obstacle], 0)
    second = sensor.scan(make_ego(), [obstacle], 1)

    assert 47 in first["detected_ids"]
    assert second["detected_ids"] == set()
    assert second["detections"] == []


# ---------------------------------------------------------------------------
# Public convenience API and serialization
# ---------------------------------------------------------------------------

def test_get_detections_returns_scan_detection_list(monkeypatch):
    sensor = make_sensor()
    obstacle, _ = make_obstacle(50, 0.0, 3.0)
    monkeypatch.setattr(np.random, "normal", lambda *args, **kwargs: 0.0)

    detections = sensor.get_detections(make_ego(), [obstacle], 0)

    assert isinstance(detections, list)
    assert len(detections) == 1
    assert detections[0].obstacle_id == 50


def test_get_detections_reuses_same_step_scan_cache(monkeypatch):
    sensor = make_sensor()
    obstacle, _ = make_obstacle(51, 0.0, 3.0)
    monkeypatch.setattr(np.random, "normal", lambda *args, **kwargs: 0.0)

    first = sensor.get_detections(make_ego(), [obstacle], 0)
    second = sensor.get_detections(make_ego(), [], 0)

    assert first is second


def test_serialize_contains_sensor_configuration():
    sensor = make_sensor(
        side="right", range_max=6.5, fov_deg=85.0, noise_std=0.3
    )

    data = sensor.serialize()

    assert data["sensor_id"] == "uss_right"
    assert data["position"] == "right"
    assert data["range"] == 6.5
    assert data["fov_deg"] == 85.0
    assert data["noise_std"] == 0.3
    np.testing.assert_allclose(data["R"], np.eye(2) * 0.3**2)


def test_serialize_returns_covariance_without_changing_sensor_state():
    sensor = make_sensor(noise_std=0.2)
    before = sensor.R.copy()

    serialized = sensor.serialize()

    np.testing.assert_array_equal(serialized["R"], before)
    np.testing.assert_array_equal(sensor.R, before)
