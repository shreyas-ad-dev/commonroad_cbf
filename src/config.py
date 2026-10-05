# src/config.py

from dataclasses import dataclass

@dataclass(frozen=True)
class TestConfig:
    DEFAULT_SAMPLING_TIME_SEC = 0.1

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
class TrackConfig:
    MIN_HITS_TO_CONFIRM = 3
    MAX_MISSED_DETECTIONS = 5


TEST = TestConfig()
DA = DataAssociationConfig()
KALMAN = KalmanConfig()
TRACK = TrackConfig()
