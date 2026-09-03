#src/tracker.py

from dataclasses import dataclass
from enum import Enum, auto

import numpy as np


class TrackState(Enum):
    TENTATIVE = auto()
    CONFIRMED = auto()
    DELETED = auto()


@dataclass
class Detection:
    """Raw sensor measurement with associated covariance."""
    sensor_id: str
    timestamp: float
    # Measurement vector: [x_local, y_local] 
    z: np.ndarray
    # Measurement noise covariance matrix (2x2)
    R: np.ndarray


class KalmanFilter2D:
    """2D Constant Velocity (CV) Kalman Filter.
    
    State vector x: [px, py, vx, vy]^T
    Measurement vector z: [px, py]^T
    """
    def __init__(self, init_pos: np.ndarray, dt: float = 0.1):
        self.dt = dt

        # Initial state estimate [px, py, vx, vy]
        self.x = np.array([init_pos[0], init_pos[1], 0.0, 0.0], dtype=np.float64)

        # Initial state covariance
        self.P = np.diag([1.0, 1.0, 10.0, 10.0])

        # State transition matrix (Constant Velocity Model)
        self.F = np.array([
            [1.0, 0.0,  dt, 0.0],
            [0.0, 1.0, 0.0,  dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float64)

        # Measurement matrix (Observing position x and y)
        self.H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ], dtype=np.float64)

        # Process noise covariance matrix
        q_var = 0.5  # Process noise magnitude
        G = np.array([
            [0.5 * dt**2, 0.0],
            [0.0, 0.5 * dt**2],
            [dt, 0.0],
            [0.0, dt]
        ])
        self.Q = G @ G.T * q_var

    def predict(self) -> np.ndarray:
        """Predict state and covariance to the next time step."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x

    def update(self, z: np.ndarray, R: np.ndarray) -> np.ndarray:
        """Update state estimate with incoming measurement."""
        y = z - (self.H @ self.x)  # Measurement residual
        S = self.H @ self.P @ self.H.T + R  # Residual covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain

        self.x = self.x + K @ y
        I = np.eye(len(self.x))
        self.P = (I - K @ self.H) @ self.P
        return self.x


class Track:
    """Maintains individual object track lifecycle and state filter."""

    def __init__(
        self,
        track_id: int,
        detection: Detection,
        dt: float = 0.1,
        confirm_hits: int = 3,
        max_age: int = 5
    ):
        self.track_id = track_id
        self.state = TrackState.TENTATIVE
        self.kf = KalmanFilter2D(init_pos=detection.z[:2], dt=dt)

        self.hits = 1
        self.age = 1
        self.time_since_update = 0

        self.confirm_hits = confirm_hits
        self.max_age = max_age

    @property
    def position(self) -> np.ndarray:
        return self.kf.x[:2]

    @property
    def velocity(self) -> np.ndarray:
        return self.kf.x[2:]

    def predict(self) -> np.ndarray:
        """Advance state estimate forward in time."""
        self.age += 1
        self.time_since_update += 1
        return self.kf.predict()

    def update(self, detection: Detection):
        """Incorporate a new measurement into the track."""
        self.hits += 1
        self.time_since_update = 0
        self.kf.update(detection.z[:2], detection.R)

        # Confirm track if hit threshold is satisfied
        if self.state == TrackState.TENTATIVE and self.hits >= self.confirm_hits:
            self.state = TrackState.CONFIRMED

    def mark_missed(self):
        """Handle steps where no detection was associated with this track."""
        if self.time_since_update > self.max_age:
            self.state = TrackState.DELETED
