import sys
import numpy as np
import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.config import RADAR,VEHICLE
from src.ego_state import EgoState
from src.radar import RadarSensor


class DummyObstacle:
    """Mock obstacle class to simulate dynamic obstacles for the Radar sensor."""
    def __init__(self, obstacle_id: int, x: float, y: float, length: float = 4.0, width: float = 1.8, yaw: float = 0.0):
        self.obstacle_id = obstacle_id
        self.x = x
        self.y = y
        self.length = length
        self.width = width
        self.yaw = yaw
        # Simple shape attribute mock
        class Shape:
            pass
        self.obstacle_shape = Shape()
        self.obstacle_shape.length = length
        self.obstacle_shape.width = width

    def state_at_time(self, step: int):
        class State:
            def __init__(self, x, y, yaw, length, width):
                self.position = np.array([x, y], dtype=float)
                self.orientation = yaw
                self.yaw = yaw
                self.length = length
                self.width = width
        return State(self.x, self.y, self.yaw, self.length, self.width)

#--------------------------------------------------------
# Initialization tests
#--------------------------------------------------------
def test_radar_initialization_front():
    """Test valid initializations and configuration bounds."""
    radar_front = RadarSensor(mount_position="front",
                              range_max=50.0,
                              fov_deg=90.0
                              )

    assert radar_front.mount_position == "front"
    assert radar_front.sensor_id == "radar_front"
    assert radar_front.range_max == 50.0
    assert radar_front.fov_deg == 90.0


def test_radar_initialization_rear():
    """Test valid initializations and configuration bounds."""
    radar_rear= RadarSensor(mount_position="rear",
                              range_max=50.0,
                              fov_deg=90.0
                              )

    assert radar_rear.mount_position == "rear"
    assert radar_rear.sensor_id == "radar_rear"
    assert radar_rear.range_max == 50.0
    assert radar_rear.fov_deg == 90.0


def test_radar_initialization_invalid_position():
    """Test that an invalid mount position raises a ValueError."""
    with pytest.raises(ValueError, match="mount_position must be either 'front' or 'rear'"):
        RadarSensor(mount_position="left", range_max=50.0, fov_deg=90.0)


def test_radar_default_values():
    """Test that default range and FOV values are correctly assigned when omitted."""
    radar = RadarSensor(mount_position="front")
    # Assuming RADAR.DEFAULT_RADAR_MAX_RANGE and RADAR.DEFAULT_RADAR_FOV_DEG are your defaults
    assert radar.range_max == RADAR.DEFAULT_RADAR_MAX_RANGE
    assert radar.fov_deg == RADAR.DEFAULT_RADAR_FOV_DEG
    assert radar.noise_std == RADAR.DEFAULT_RADAR_RANGE_NOISE_STD 
    assert radar.ray_count == 40

#--------------------------------------------------------
# Internal supporting functions
#--------------------------------------------------------

def test_radar_build_fov_wedge():
    """Test that _build_fov_wedge constructs a valid Shapely polygon with correct geometry."""
    radar = RadarSensor(mount_position="front", range_max=60.0, fov_deg=60.0, ray_count=41)
    sensor_pos = np.array([0.0, 0.0], dtype=float)
    
    wedge = radar._build_fov_wedge(sensor_pos, sensor_heading_deg=0.0)
    
    assert not wedge.is_empty
    assert wedge.geom_type == "Polygon"
    # Check that the origin is part of the polygon coordinates
    coords = list(wedge.exterior.coords)
    assert coords[0] == (0.0, 0.0)
    center_ray_pt = coords[21]
    assert center_ray_pt[0] == pytest.approx(60.0, abs=5*1e-2)
    assert center_ray_pt[1] == pytest.approx(0.0, abs=5*1e-2)
    assert coords[-1] == (0.0, 0.0)


def test_sensor_transform_mounting_0deg():
    """Test that front-mounted sensor position and heading correctly account for vehicle length and orientation."""
    # Create an ego vehicle at (10.0, 20.0) with length 4.0 and orientation = 0.0 (facing East)
    ego = EgoState(x=0.0,
                   y=0.0,
                   orientation=0.0,
                   velocity=10.0,
                   length=VEHICLE.DEFAULT_LENGTH,
                   width=VEHICLE.DEFAULT_WIDTH,
                   wheelbase=VEHICLE.DEFAULT_WHEELBASE
                   )

    front_radar = RadarSensor(mount_position="front")
    front_sensor_pos, front_sensor_heading = front_radar._get_sensor_transform(ego)

    rear_radar = RadarSensor(mount_position="rear")
    rear_sensor_pos, rear_sensor_heading = rear_radar._get_sensor_transform(ego)

    np.testing.assert_allclose(front_sensor_pos, [VEHICLE.DEFAULT_LENGTH/2, 0.0], atol=1e-5)
    np.testing.assert_allclose(rear_sensor_pos, [-VEHICLE.DEFAULT_LENGTH/2, 0.0], atol=1e-5)

    assert front_sensor_heading == pytest.approx(0.0, abs=1e-5)
    assert rear_sensor_heading == pytest.approx(180.0, abs=1e-5)



def test_sensor_transform_mounting_90_deg():
    """Test that rear-mounted sensor position and heading apply negative offset and 180-degree rotation."""
    # Ego vehicle at origin, pointing along pi/2 radians (90 degrees, facing North)
    orientation = np.pi / 2.0
    ego = EgoState(x=0.0,
                   y=0.0,
                   orientation=orientation,
                   velocity=10.0,
                   length=VEHICLE.DEFAULT_LENGTH,
                   width=VEHICLE.DEFAULT_WIDTH,
                   wheelbase=VEHICLE.DEFAULT_WHEELBASE
                   )

    front_radar = RadarSensor(mount_position="front")
    front_sensor_pos, front_sensor_heading = front_radar._get_sensor_transform(ego)

    rear_radar = RadarSensor(mount_position="rear")
    rear_sensor_pos, rear_sensor_heading = rear_radar._get_sensor_transform(ego)

    np.testing.assert_allclose(front_sensor_pos, [ 0.0, VEHICLE.DEFAULT_LENGTH/2], atol=1e-5)
    np.testing.assert_allclose(rear_sensor_pos, [ 0.0, -VEHICLE.DEFAULT_LENGTH/2], atol=1e-5)

    assert front_sensor_heading == pytest.approx(90.0, abs=1e-5)
    assert rear_sensor_heading == pytest.approx(270.0, abs=1e-5)



#--------------------------------------------------------
#
#--------------------------------------------------------
def test_radar_scan_detection_in_fov():
    """Test that an obstacle directly in front of the ego vehicle within range is detected."""
    ego = EgoState(x=0.0,
                   y=0.0,
                   orientation=0.0,
                   velocity=10.0,
                   length=VEHICLE.DEFAULT_LENGTH,
                   width=VEHICLE.DEFAULT_WIDTH,
                   wheelbase=VEHICLE.DEFAULT_WHEELBASE
                   )
    # Obstacle directly ahead at x=20.0, y=0.0
    obstacle = DummyObstacle(obstacle_id=1, x=20.0, y=0.0)

    radar = RadarSensor(mount_position="front",
                        range_max=50.0,
                        fov_deg=60.0,
                        noise_std=0.0
                        )
    scan_result = radar.scan(ego, [obstacle], step=0)

    assert 1 in scan_result["detected_ids"]
    assert len(scan_result["detections"]) == 1
    
    detection = scan_result["detections"][0]
    assert detection.obstacle_id == 1
    assert detection.sensor_id == "radar_front"
    # With zero noise, measured global coordinates should match obstacle position
    np.testing.assert_allclose(detection.z, [20.0, 0.0], atol=1e-5)


def test_radar_scan_out_of_fov():
    """Test that obstacles outside the field of view or range limits are ignored."""
    ego = EgoState(x=0.0,
                   y=0.0,
                   orientation=0.0,
                   velocity=10.0,
                   length=VEHICLE.DEFAULT_LENGTH,
                   width=VEHICLE.DEFAULT_WIDTH,
                   wheelbase=VEHICLE.DEFAULT_WHEELBASE
                   )
    
    # Obstacle behind the front-mounted radar (negative x)
    obs_behind = DummyObstacle(obstacle_id=1, x=-10.0, y=0.0)
    # Obstacle out of maximum range (range_max = 50.0)
    obs_out_of_range = DummyObstacle(obstacle_id=2, x=60.0, y=0.0)
    # Obstacle out of angular FOV (60 deg total -> +/- 30 deg; y=20 at x=10 is outside ~63 deg)
    obs_out_of_angle = DummyObstacle(obstacle_id=3, x=10.0, y=20.0)

    radar = RadarSensor(
            mount_position="front",
            range_max=50.0,
            fov_deg=60.0
            )
    scan_result = radar.scan(ego, [obs_behind, obs_out_of_range, obs_out_of_angle], step=0)

    assert len(scan_result["detected_ids"]) == 0
    assert len(scan_result["detections"]) == 0


def test_radar_occlusion():
    """Test that a closer obstacle occludes a farther obstacle along the same line-of-sight."""
    
    ego = EgoState(x=0.0,
                   y=0.0,
                   orientation=0.0,
                   velocity=10.0,
                   length=VEHICLE.DEFAULT_LENGTH,
                   width=VEHICLE.DEFAULT_WIDTH,
                   wheelbase=VEHICLE.DEFAULT_WHEELBASE
                   )

    
    # Closer obstacle
    obs_close = DummyObstacle(obstacle_id=1, x=15.0, y=0.0, length=2.0, width=1.8)
    # Farther obstacle directly behind the first one
    obs_far = DummyObstacle(obstacle_id=2, x=30.0, y=0.0, length=2.0, width=1.8)

    radar = RadarSensor(mount_position="front",
                        range_max=50.0,
                        fov_deg=60.0
                        )
    scan_result = radar.scan(ego, [obs_close, obs_far], step=0)

    # Only the closer obstacle should be detected due to shadow/occlusion calculation
    assert 1 in scan_result["detected_ids"]
    assert 2 not in scan_result["detected_ids"]
    assert len(scan_result["detections"]) == 1
    assert scan_result["detections"][0].obstacle_id == 1
import sys
import numpy as np
import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.config import RADAR,VEHICLE
from src.ego_state import EgoState
from src.radar import RadarSensor


class DummyObstacle:
    """Mock obstacle class to simulate dynamic obstacles for the Radar sensor."""
    def __init__(self, obstacle_id: int, x: float, y: float, length: float = 4.0, width: float = 1.8, yaw: float = 0.0):
        self.obstacle_id = obstacle_id
        self.x = x
        self.y = y
        self.length = length
        self.width = width
        self.yaw = yaw
        # Simple shape attribute mock
        class Shape:
            pass
        self.obstacle_shape = Shape()
        self.obstacle_shape.length = length
        self.obstacle_shape.width = width

    def state_at_time(self, step: int):
        class State:
            def __init__(self, x, y, yaw, length, width):
                self.position = np.array([x, y], dtype=float)
                self.orientation = yaw
                self.yaw = yaw
                self.length = length
                self.width = width
        return State(self.x, self.y, self.yaw, self.length, self.width)

#--------------------------------------------------------
# Initialization tests
#--------------------------------------------------------
def test_radar_initialization_front():
    """Test valid initializations and configuration bounds."""
    radar_front = RadarSensor(mount_position="front",
                              range_max=50.0,
                              fov_deg=90.0
                              )

    assert radar_front.mount_position == "front"
    assert radar_front.sensor_id == "radar_front"
    assert radar_front.range_max == 50.0
    assert radar_front.fov_deg == 90.0


def test_radar_initialization_rear():
    """Test valid initializations and configuration bounds."""
    radar_rear= RadarSensor(mount_position="rear",
                              range_max=50.0,
                              fov_deg=90.0
                              )

    assert radar_rear.mount_position == "rear"
    assert radar_rear.sensor_id == "radar_rear"
    assert radar_rear.range_max == 50.0
    assert radar_rear.fov_deg == 90.0


def test_radar_initialization_invalid_position():
    """Test that an invalid mount position raises a ValueError."""
    with pytest.raises(ValueError, match="mount_position must be either 'front' or 'rear'"):
        RadarSensor(mount_position="left", range_max=50.0, fov_deg=90.0)


def test_radar_default_values():
    """Test that default range and FOV values are correctly assigned when omitted."""
    radar = RadarSensor(mount_position="front")
    # Assuming RADAR.DEFAULT_RADAR_MAX_RANGE and RADAR.DEFAULT_RADAR_FOV_DEG are your defaults
    assert radar.range_max == RADAR.DEFAULT_RADAR_MAX_RANGE
    assert radar.fov_deg == RADAR.DEFAULT_RADAR_FOV_DEG
    assert radar.noise_std == RADAR.DEFAULT_RADAR_RANGE_NOISE_STD 
    assert radar.ray_count == 40

#--------------------------------------------------------
# Internal supporting functions
#--------------------------------------------------------

def test_radar_build_fov_wedge():
    """Test that _build_fov_wedge constructs a valid Shapely polygon with correct geometry."""
    radar = RadarSensor(mount_position="front", range_max=60.0, fov_deg=60.0, ray_count=41)
    sensor_pos = np.array([0.0, 0.0], dtype=float)
    
    wedge = radar._build_fov_wedge(sensor_pos, sensor_heading_deg=0.0)
    
    assert not wedge.is_empty
    assert wedge.geom_type == "Polygon"
    # Check that the origin is part of the polygon coordinates
    coords = list(wedge.exterior.coords)
    assert coords[0] == (0.0, 0.0)
    center_ray_pt = coords[21]
    assert center_ray_pt[0] == pytest.approx(60.0, abs=5*1e-2)
    assert center_ray_pt[1] == pytest.approx(0.0, abs=5*1e-2)
    assert coords[-1] == (0.0, 0.0)


def test_sensor_transform_mounting_0deg():
    """Test that front-mounted sensor position and heading correctly account for vehicle length and orientation."""
    # Create an ego vehicle at (10.0, 20.0) with length 4.0 and orientation = 0.0 (facing East)
    ego = EgoState(x=0.0,
                   y=0.0,
                   orientation=0.0,
                   velocity=10.0,
                   length=VEHICLE.DEFAULT_LENGTH,
                   width=VEHICLE.DEFAULT_WIDTH,
                   wheelbase=VEHICLE.DEFAULT_WHEELBASE
                   )

    front_radar = RadarSensor(mount_position="front")
    front_sensor_pos, front_sensor_heading = front_radar._get_sensor_transform(ego)

    rear_radar = RadarSensor(mount_position="rear")
    rear_sensor_pos, rear_sensor_heading = rear_radar._get_sensor_transform(ego)

    np.testing.assert_allclose(front_sensor_pos, [VEHICLE.DEFAULT_LENGTH/2, 0.0], atol=1e-5)
    np.testing.assert_allclose(rear_sensor_pos, [-VEHICLE.DEFAULT_LENGTH/2, 0.0], atol=1e-5)

    assert front_sensor_heading == pytest.approx(0.0, abs=1e-5)
    assert rear_sensor_heading == pytest.approx(180.0, abs=1e-5)



def test_sensor_transform_mounting_90_deg():
    """Test that rear-mounted sensor position and heading apply negative offset and 180-degree rotation."""
    # Ego vehicle at origin, pointing along pi/2 radians (90 degrees, facing North)
    orientation = np.pi / 2.0
    ego = EgoState(x=0.0,
                   y=0.0,
                   orientation=orientation,
                   velocity=10.0,
                   length=VEHICLE.DEFAULT_LENGTH,
                   width=VEHICLE.DEFAULT_WIDTH,
                   wheelbase=VEHICLE.DEFAULT_WHEELBASE
                   )

    front_radar = RadarSensor(mount_position="front")
    front_sensor_pos, front_sensor_heading = front_radar._get_sensor_transform(ego)

    rear_radar = RadarSensor(mount_position="rear")
    rear_sensor_pos, rear_sensor_heading = rear_radar._get_sensor_transform(ego)

    np.testing.assert_allclose(front_sensor_pos, [ 0.0, VEHICLE.DEFAULT_LENGTH/2], atol=1e-5)
    np.testing.assert_allclose(rear_sensor_pos, [ 0.0, -VEHICLE.DEFAULT_LENGTH/2], atol=1e-5)

    assert front_sensor_heading == pytest.approx(90.0, abs=1e-5)
    assert rear_sensor_heading == pytest.approx(270.0, abs=1e-5)



#--------------------------------------------------------
#
#--------------------------------------------------------
def test_radar_scan_detection_in_fov():
    """Test that an obstacle directly in front of the ego vehicle within range is detected."""
    ego = EgoState(x=0.0,
                   y=0.0,
                   orientation=0.0,
                   velocity=10.0,
                   length=VEHICLE.DEFAULT_LENGTH,
                   width=VEHICLE.DEFAULT_WIDTH,
                   wheelbase=VEHICLE.DEFAULT_WHEELBASE
                   )
    # Obstacle directly ahead at x=20.0, y=0.0
    obstacle = DummyObstacle(obstacle_id=1, x=20.0, y=0.0)

    radar = RadarSensor(mount_position="front",
                        range_max=50.0,
                        fov_deg=60.0,
                        noise_std=0.0
                        )
    scan_result = radar.scan(ego, [obstacle], step=0)

    assert 1 in scan_result["detected_ids"]
    assert len(scan_result["detections"]) == 1
    
    detection = scan_result["detections"][0]
    assert detection.obstacle_id == 1
    assert detection.sensor_id == "radar_front"
    # With zero noise, measured global coordinates should match obstacle position
    np.testing.assert_allclose(detection.z, [20.0, 0.0], atol=1e-5)


def test_radar_scan_out_of_fov():
    """Test that obstacles outside the field of view or range limits are ignored."""
    ego = EgoState(x=0.0,
                   y=0.0,
                   orientation=0.0,
                   velocity=10.0,
                   length=VEHICLE.DEFAULT_LENGTH,
                   width=VEHICLE.DEFAULT_WIDTH,
                   wheelbase=VEHICLE.DEFAULT_WHEELBASE
                   )
    
    # Obstacle behind the front-mounted radar (negative x)
    obs_behind = DummyObstacle(obstacle_id=1, x=-10.0, y=0.0)
    # Obstacle out of maximum range (range_max = 50.0)
    obs_out_of_range = DummyObstacle(obstacle_id=2, x=60.0, y=0.0)
    # Obstacle out of angular FOV (60 deg total -> +/- 30 deg; y=20 at x=10 is outside ~63 deg)
    obs_out_of_angle = DummyObstacle(obstacle_id=3, x=10.0, y=20.0)

    radar = RadarSensor(
            mount_position="front",
            range_max=50.0,
            fov_deg=60.0
            )
    scan_result = radar.scan(ego, [obs_behind, obs_out_of_range, obs_out_of_angle], step=0)

    assert len(scan_result["detected_ids"]) == 0
    assert len(scan_result["detections"]) == 0


def test_radar_occlusion():
    """Test that a closer obstacle occludes a farther obstacle along the same line-of-sight."""
    
    ego = EgoState(x=0.0,
                   y=0.0,
                   orientation=0.0,
                   velocity=10.0,
                   length=VEHICLE.DEFAULT_LENGTH,
                   width=VEHICLE.DEFAULT_WIDTH,
                   wheelbase=VEHICLE.DEFAULT_WHEELBASE
                   )

    
    # Closer obstacle
    obs_close = DummyObstacle(obstacle_id=1, x=15.0, y=0.0, length=2.0, width=1.8)
    # Farther obstacle directly behind the first one
    obs_far = DummyObstacle(obstacle_id=2, x=30.0, y=0.0, length=2.0, width=1.8)

    radar = RadarSensor(mount_position="front",
                        range_max=50.0,
                        fov_deg=60.0
                        )
    scan_result = radar.scan(ego, [obs_close, obs_far], step=0)

    # Only the closer obstacle should be detected due to shadow/occlusion calculation
    assert 1 in scan_result["detected_ids"]
    assert 2 not in scan_result["detected_ids"]
    assert len(scan_result["detections"]) == 1
    assert scan_result["detections"][0].obstacle_id == 1


#--------------------------------------------------------
# Additional sensor mounting and coordinate tests
#--------------------------------------------------------

@pytest.mark.parametrize(
    "mount_position, ego_orientation, expected_heading",
    [
        ("front", np.pi, 180.0),
        ("rear", np.pi, 360.0),
        ("front", np.pi / 4.0, 45.0),
        ("rear", np.pi / 4.0, 225.0),
    ],
)
def test_sensor_transform_additional_orientations(
    mount_position,
    ego_orientation,
    expected_heading
):
    ego = EgoState(
        x=12.0,
        y=-7.0,
        orientation=ego_orientation,
        velocity=10.0,
        length=VEHICLE.DEFAULT_LENGTH,
        width=VEHICLE.DEFAULT_WIDTH,
        wheelbase=VEHICLE.DEFAULT_WHEELBASE
    )

    radar = RadarSensor(mount_position=mount_position)
    sensor_pos, sensor_heading = radar._get_sensor_transform(ego)

    direction = np.array([np.cos(ego_orientation), np.sin(ego_orientation)])
    offset_sign = 1.0 if mount_position == "front" else -1.0
    expected_pos = ego.position + (
        offset_sign * ego.length / 2.0
    ) * direction

    np.testing.assert_allclose(sensor_pos, expected_pos, atol=1e-5)
    assert sensor_heading == pytest.approx(expected_heading, abs=1e-5)


def test_radar_scan_global_coordinates_with_translated_and_rotated_ego():
    """A noiseless detection should be returned in global coordinates."""
    ego_orientation = np.pi / 3.0
    ego = EgoState(
        x=12.0,
        y=-7.0,
        orientation=ego_orientation,
        velocity=10.0,
        length=VEHICLE.DEFAULT_LENGTH,
        width=VEHICLE.DEFAULT_WIDTH,
        wheelbase=VEHICLE.DEFAULT_WHEELBASE
    )

    local_offset = np.array([20.0, 0.0])
    rotation = np.array([
        [np.cos(ego_orientation), -np.sin(ego_orientation)],
        [np.sin(ego_orientation), np.cos(ego_orientation)]
    ])
    obstacle_position = ego.position + rotation @ local_offset

    obstacle = DummyObstacle(
        obstacle_id=40,
        x=float(obstacle_position[0]),
        y=float(obstacle_position[1])
    )
    radar = RadarSensor(
        mount_position="front",
        range_max=50.0,
        fov_deg=60.0,
        noise_std=0.0
    )

    result = radar.scan(ego, [obstacle], step=4)

    assert 40 in result["detected_ids"]
    assert len(result["detections"]) == 1
    np.testing.assert_allclose(
        result["detections"][0].z,
        obstacle_position,
        atol=1e-5
    )


#--------------------------------------------------------
# Additional FOV and range tests
#--------------------------------------------------------

def test_radar_detects_obstacle_partially_intersecting_fov():
    """Detection policy is polygon intersection, not centre-point containment."""
    ego = EgoState(
        x=0.0,
        y=0.0,
        orientation=0.0,
        velocity=10.0,
        length=VEHICLE.DEFAULT_LENGTH,
        width=VEHICLE.DEFAULT_WIDTH,
        wheelbase=VEHICLE.DEFAULT_WHEELBASE
    )

    # The centre lies just outside a 60-degree FOV, but the obstacle's
    # bounding box extends across the FOV boundary.
    obstacle = DummyObstacle(
        obstacle_id=41,
        x=20.0,
        y=12.2,
        length=4.0,
        width=1.8
    )
    radar = RadarSensor(
        mount_position="front",
        range_max=50.0,
        fov_deg=60.0,
        noise_std=0.0
    )

    result = radar.scan(ego, [obstacle], step=0)

    assert 41 in result["detected_ids"]


def test_radar_ignores_obstacle_completely_beyond_maximum_range():
    ego = EgoState(
        x=0.0,
        y=0.0,
        orientation=0.0,
        velocity=10.0,
        length=VEHICLE.DEFAULT_LENGTH,
        width=VEHICLE.DEFAULT_WIDTH,
        wheelbase=VEHICLE.DEFAULT_WHEELBASE
    )
    obstacle = DummyObstacle(
        obstacle_id=42,
        x=60.0,
        y=0.0,
        length=4.0,
        width=1.8
    )
    radar = RadarSensor(
        mount_position="front",
        range_max=50.0,
        fov_deg=60.0,
        noise_std=0.0
    )

    result = radar.scan(ego, [obstacle], step=0)

    assert 42 not in result["detected_ids"]
    assert result["detections"] == []


def test_radar_detects_obstacle_partially_intersecting_range_boundary():
    """An obstacle whose polygon reaches into the FOV wedge is detectable."""
    ego = EgoState(
        x=0.0,
        y=0.0,
        orientation=0.0,
        velocity=10.0,
        length=VEHICLE.DEFAULT_LENGTH,
        width=VEHICLE.DEFAULT_WIDTH,
        wheelbase=VEHICLE.DEFAULT_WHEELBASE
    )
    obstacle = DummyObstacle(
        obstacle_id=43,
        x=51.0,
        y=0.0,
        length=4.0,
        width=1.8
    )
    radar = RadarSensor(
        mount_position="front",
        range_max=50.0,
        fov_deg=60.0,
        noise_std=0.0
    )

    result = radar.scan(ego, [obstacle], step=0)

    assert 43 in result["detected_ids"]


#--------------------------------------------------------
# Rear radar perception tests
#--------------------------------------------------------

def test_rear_radar_detects_obstacle_directly_behind():
    ego = EgoState(
        x=0.0,
        y=0.0,
        orientation=0.0,
        velocity=10.0,
        length=VEHICLE.DEFAULT_LENGTH,
        width=VEHICLE.DEFAULT_WIDTH,
        wheelbase=VEHICLE.DEFAULT_WHEELBASE
    )
    obstacle = DummyObstacle(obstacle_id=50, x=-20.0, y=0.0)
    radar = RadarSensor(
        mount_position="rear",
        range_max=50.0,
        fov_deg=60.0,
        noise_std=0.0
    )

    result = radar.scan(ego, [obstacle], step=0)

    assert 50 in result["detected_ids"]
    assert len(result["detections"]) == 1
    assert result["detections"][0].sensor_id == "radar_rear"
    np.testing.assert_allclose(result["detections"][0].z, [-20.0, 0.0], atol=1e-5)


def test_rear_radar_ignores_obstacle_directly_in_front():
    ego = EgoState(
        x=0.0,
        y=0.0,
        orientation=0.0,
        velocity=10.0,
        length=VEHICLE.DEFAULT_LENGTH,
        width=VEHICLE.DEFAULT_WIDTH,
        wheelbase=VEHICLE.DEFAULT_WHEELBASE
    )
    obstacle = DummyObstacle(obstacle_id=51, x=20.0, y=0.0)
    radar = RadarSensor(
        mount_position="rear",
        range_max=50.0,
        fov_deg=60.0,
        noise_std=0.0
    )

    result = radar.scan(ego, [obstacle], step=0)

    assert 51 not in result["detected_ids"]
    assert result["detections"] == []


def test_rear_radar_ignores_obstacle_outside_rear_angular_fov():
    ego = EgoState(
        x=0.0,
        y=0.0,
        orientation=0.0,
        velocity=10.0,
        length=VEHICLE.DEFAULT_LENGTH,
        width=VEHICLE.DEFAULT_WIDTH,
        wheelbase=VEHICLE.DEFAULT_WHEELBASE
    )
    obstacle = DummyObstacle(obstacle_id=52, x=-10.0, y=20.0)
    radar = RadarSensor(
        mount_position="rear",
        range_max=50.0,
        fov_deg=60.0,
        noise_std=0.0
    )

    result = radar.scan(ego, [obstacle], step=0)

    assert 52 not in result["detected_ids"]


#--------------------------------------------------------
# Measurement noise and detection metadata tests
#--------------------------------------------------------

def test_radar_measurement_covariance_matches_noise_std_squared():
    noise_std = 0.35
    radar = RadarSensor(mount_position="front", noise_std=noise_std)

    np.testing.assert_allclose(
        radar.R,
        np.eye(2) * noise_std**2
    )


def test_radar_detection_contains_expected_metadata():
    ego = EgoState(
        x=0.0,
        y=0.0,
        orientation=0.0,
        velocity=10.0,
        length=VEHICLE.DEFAULT_LENGTH,
        width=VEHICLE.DEFAULT_WIDTH,
        wheelbase=VEHICLE.DEFAULT_WHEELBASE
    )
    obstacle = DummyObstacle(obstacle_id=60, x=15.0, y=0.0)
    radar = RadarSensor(
        mount_position="front",
        range_max=50.0,
        fov_deg=60.0,
        noise_std=0.0
    )

    step = 7
    result = radar.scan(ego, [obstacle], step=step)

    assert len(result["detections"]) == 1
    detection = result["detections"][0]

    assert detection.sensor_id == "radar_front"
    assert detection.obstacle_id == 60
    assert detection.timestamp == pytest.approx(
        step * __import__("src.config", fromlist=["TEST"]).TEST.DEFAULT_SAMPLING_TIME_SEC
    )
    np.testing.assert_allclose(detection.R, radar.R)


def test_radar_nonzero_noise_produces_measurement_variation():
    """Repeated scans at distinct steps should produce varying noisy measurements."""
    ego = EgoState(
        x=0.0,
        y=0.0,
        orientation=0.0,
        velocity=10.0,
        length=VEHICLE.DEFAULT_LENGTH,
        width=VEHICLE.DEFAULT_WIDTH,
        wheelbase=VEHICLE.DEFAULT_WHEELBASE
    )
    obstacle = DummyObstacle(obstacle_id=61, x=20.0, y=0.0)
    radar = RadarSensor(
        mount_position="front",
        range_max=50.0,
        fov_deg=60.0,
        noise_std=0.5
    )

    np.random.seed(1234)
    measurements = []
    for step in range(100):
        result = radar.scan(ego, [obstacle], step=step)
        assert len(result["detections"]) == 1
        measurements.append(result["detections"][0].z)

    measurements = np.asarray(measurements)

    assert np.std(measurements[:, 0]) > 0.1
    assert np.std(measurements[:, 1]) > 0.1


#--------------------------------------------------------
# Occlusion geometry tests
#--------------------------------------------------------

def test_compute_occlusion_shadow_returns_valid_nonempty_polygon():
    from shapely.geometry import Polygon

    radar = RadarSensor(
        mount_position="front",
        range_max=50.0,
        fov_deg=60.0
    )
    sensor_pos = np.array([0.0, 0.0])
    obstacle_polygon = Polygon([
        (10.0, -1.0),
        (12.0, -1.0),
        (12.0, 1.0),
        (10.0, 1.0)
    ])

    shadow = radar._compute_occlusion_shadow(sensor_pos, obstacle_polygon)

    assert shadow is not None
    assert not shadow.is_empty
    assert shadow.is_valid
    assert shadow.area > 0.0
    assert shadow.bounds[2] > obstacle_polygon.bounds[2]


def test_occlusion_does_not_hide_laterally_separated_obstacle():
    ego = EgoState(
        x=0.0,
        y=0.0,
        orientation=0.0,
        velocity=10.0,
        length=VEHICLE.DEFAULT_LENGTH,
        width=VEHICLE.DEFAULT_WIDTH,
        wheelbase=VEHICLE.DEFAULT_WHEELBASE
    )
    close_obstacle = DummyObstacle(
        obstacle_id=70,
        x=15.0,
        y=0.0,
        length=2.0,
        width=1.8
    )
    separated_obstacle = DummyObstacle(
        obstacle_id=71,
        x=30.0,
        y=8.0,
        length=2.0,
        width=1.8
    )
    radar = RadarSensor(
        mount_position="front",
        range_max=50.0,
        fov_deg=90.0,
        noise_std=0.0
    )

    result = radar.scan(ego, [close_obstacle, separated_obstacle], step=0)

    assert 70 in result["detected_ids"]
    assert 71 in result["detected_ids"]


def test_three_aligned_obstacles_are_occluded_in_distance_order():
    ego = EgoState(
        x=0.0,
        y=0.0,
        orientation=0.0,
        velocity=10.0,
        length=VEHICLE.DEFAULT_LENGTH,
        width=VEHICLE.DEFAULT_WIDTH,
        wheelbase=VEHICLE.DEFAULT_WHEELBASE
    )
    obstacles = [
        DummyObstacle(80, 15.0, 0.0, length=2.0, width=1.8),
        DummyObstacle(81, 25.0, 0.0, length=2.0, width=1.8),
        DummyObstacle(82, 35.0, 0.0, length=2.0, width=1.8),
    ]
    radar = RadarSensor(
        mount_position="front",
        range_max=50.0,
        fov_deg=60.0,
        noise_std=0.0
    )

    result = radar.scan(ego, obstacles, step=0)

    assert 80 in result["detected_ids"]
    assert 81 not in result["detected_ids"]
    assert 82 not in result["detected_ids"]
    assert len(result["detections"]) == 1


#--------------------------------------------------------
# Scan cache and output geometry tests
#--------------------------------------------------------

def test_radar_scan_same_step_reuses_cached_result():
    ego = EgoState(
        x=0.0,
        y=0.0,
        orientation=0.0,
        velocity=10.0,
        length=VEHICLE.DEFAULT_LENGTH,
        width=VEHICLE.DEFAULT_WIDTH,
        wheelbase=VEHICLE.DEFAULT_WHEELBASE
    )
    first_obstacle = DummyObstacle(obstacle_id=90, x=15.0, y=0.0)
    second_obstacle = DummyObstacle(obstacle_id=91, x=20.0, y=0.0)
    radar = RadarSensor(
        mount_position="front",
        range_max=50.0,
        fov_deg=60.0,
        noise_std=0.0
    )

    first_result = radar.scan(ego, [first_obstacle], step=3)
    second_result = radar.scan(ego, [second_obstacle], step=3)

    assert second_result is first_result
    assert second_result["detected_ids"] == {90}


def test_radar_scan_new_step_recomputes_result():
    ego = EgoState(
        x=0.0,
        y=0.0,
        orientation=0.0,
        velocity=10.0,
        length=VEHICLE.DEFAULT_LENGTH,
        width=VEHICLE.DEFAULT_WIDTH,
        wheelbase=VEHICLE.DEFAULT_WHEELBASE
    )
    first_obstacle = DummyObstacle(obstacle_id=92, x=15.0, y=0.0)
    second_obstacle = DummyObstacle(obstacle_id=93, x=20.0, y=0.0)
    radar = RadarSensor(
        mount_position="front",
        range_max=50.0,
        fov_deg=60.0,
        noise_std=0.0
    )

    first_result = radar.scan(ego, [first_obstacle], step=3)
    second_result = radar.scan(ego, [second_obstacle], step=4)

    assert second_result is not first_result
    assert second_result["detected_ids"] == {93}
    assert len(second_result["detections"]) == 1


def test_get_detections_reuses_scan_cache():
    ego = EgoState(
        x=0.0,
        y=0.0,
        orientation=0.0,
        velocity=10.0,
        length=VEHICLE.DEFAULT_LENGTH,
        width=VEHICLE.DEFAULT_WIDTH,
        wheelbase=VEHICLE.DEFAULT_WHEELBASE
    )
    obstacle = DummyObstacle(obstacle_id=94, x=15.0, y=0.0)
    radar = RadarSensor(
        mount_position="front",
        range_max=50.0,
        fov_deg=60.0,
        noise_std=0.0
    )

    scan_result = radar.scan(ego, [obstacle], step=5)
    detections = radar.get_detections(ego, [], step=5)

    assert detections is scan_result["detections"]
    assert len(detections) == 1
    assert detections[0].obstacle_id == 94


def test_radar_scan_returns_visible_fov_geometry():
    ego = EgoState(
        x=0.0,
        y=0.0,
        orientation=0.0,
        velocity=10.0,
        length=VEHICLE.DEFAULT_LENGTH,
        width=VEHICLE.DEFAULT_WIDTH,
        wheelbase=VEHICLE.DEFAULT_WHEELBASE
    )
    obstacle = DummyObstacle(obstacle_id=95, x=15.0, y=0.0)
    radar = RadarSensor(
        mount_position="front",
        range_max=50.0,
        fov_deg=60.0,
        noise_std=0.0
    )

    result = radar.scan(ego, [obstacle], step=0)

    assert result["visible_fov"] is not None
    assert not result["visible_fov"].is_empty
    assert result["visible_fov"].is_valid


def test_fov_data_contains_obstacle_visibility_metadata():
    ego = EgoState(
        x=0.0,
        y=0.0,
        orientation=0.0,
        velocity=10.0,
        length=VEHICLE.DEFAULT_LENGTH,
        width=VEHICLE.DEFAULT_WIDTH,
        wheelbase=VEHICLE.DEFAULT_WHEELBASE
    )
    obstacle = DummyObstacle(obstacle_id=96, x=15.0, y=0.0)
    radar = RadarSensor(
        mount_position="front",
        range_max=50.0,
        fov_deg=60.0,
        noise_std=0.0
    )

    result = radar.scan(ego, [obstacle], step=0)

    assert 96 in result["fov_data"]
    visibility = result["fov_data"][96]
    assert visibility.obstacle_id == 96
    assert visibility.in_fov is True
    assert visibility.min_dist >= 0.0
    assert np.isfinite(visibility.center_x_local)
    assert np.isfinite(visibility.center_y_local)
    assert isinstance(visibility.visible_segments, list)


#--------------------------------------------------------
# Serialization tests
#--------------------------------------------------------

def test_radar_serialization_contains_configuration():
    noise_std = 0.25
    radar = RadarSensor(
        mount_position="rear",
        range_max=45.0,
        fov_deg=100.0,
        noise_std=noise_std,
        ray_count=60
    )

    serialized = radar.serialize()

    assert serialized["sensor_id"] == "radar_rear"
    assert serialized["position"] == "rear"
    assert serialized["range"] == pytest.approx(45.0)
    assert serialized["fov_deg"] == pytest.approx(100.0)
    assert serialized["noise_std"] == pytest.approx(noise_std)
    np.testing.assert_allclose(serialized["R"], np.eye(2) * noise_std**2)
