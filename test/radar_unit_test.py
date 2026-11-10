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
