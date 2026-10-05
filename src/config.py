# src/config.py

from dataclasses import dataclass

@dataclass(frozen=True)
class DataAssociationConfig:
    GATING_PENALTY_COST = 1e5 
    GATING_THRESHOLD = 10.0
    MAX_ASSIGNMENT_DISTANCE = 5.0

@dataclass(frozen=True)
class KalmanConfig:
    INITIAL_POSITION_VAR = 1.0
    INITIAL_VELOCITY_VAR = 10.0
    DEFAULT_PROCESS_NOISE_VAR = 0.5

@dataclass(frozen=True)
class LateralConfig:
    DEFAULT_PATH_LOOKAHEAD_DISTANCE = 150.0
    DEFAULT_NUM_WAYPOINTS = 200
    LANE_CHANGE_START_X = 10.0
    MANEUVER_S_LENGHT = 35.0
    QPC_5 = 6
    QPC_4 = 15
    QPC_3 = 10
    
@dataclass(frozen=True)
class MapConfig:
    DEFAULT_LANE_WIDTH_METERS = 3.5
    DEFAULT_LOOKAHEAD_HORIZON_METERS = 100.0
    DEFAULT_EXTRACTION_HORIZON_METERS = 200.0
    DEFAULT_RESAMPLING_STEP_METERS = 0.5
    WAYPOINT_DEDUP_TOLERANCE = 1e-5

@dataclass(frozen=True)
class ScenarioConfig:
    OBSTACLE_MATCHING_ATOL = 1.5

@dataclass(frozen=True)
class StanleyConfig:
    DEFAULT_STANLEY_GAIN = 0.5
    DEFAULT_SOFTENING_GAIN = 1.0
    DEFAULT_MAX_STEER_DEG = 25.0

@dataclass(frozen=True)
class TestConfig:
    DEFAULT_SAMPLING_TIME_SEC = 0.1

@dataclass(frozen=True)
class TrackConfig:
    MIN_HITS_TO_CONFIRM = 3
    MAX_MISSED_DETECTIONS = 5

@dataclass(frozen=True)
class UltrasonicConfig:
    DEFAULT_USS_MAX_RANGE = 8.0
    DEFAULT_USS_FOV_DEG = 100.0
    DEFAULT_USS_NOISE_STD = 0.1

@dataclass(frozen=True)
class VehicleConfig:
    DEFAULT_WHEELBASE = 2.8
    DEFAULT_LENGTH = 4.5
    DEFAULT_VELOCITY = 15.0


DA = DataAssociationConfig()
KALMAN = KalmanConfig()
LATERAL = LateralConfig()
MAP = MapConfig()
SCENARIO = ScenarioConfig()
STANLEY = StanleyConfig()
TEST = TestConfig()
TRACK = TrackConfig()
USS = UltrasonicConfig()
VEHICLE = VehicleConfig()
