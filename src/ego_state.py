# src/ego_state.py
from dataclasses import dataclass
import numpy as np


@dataclass
class EgoState:
    x: float
    y: float
    orientation: float  # Yaw angle in radians
    velocity: float     # Scalar speed in m/s
    length: float       # Vehicle length in meters
    width: float        # Vehicle width in meters

    # -------------------------------------------------------------------------
    # Properties & Vector Helpers
    # -------------------------------------------------------------------------
    @property
    def position(self) -> np.ndarray:
        """2D position vector [x, y] in world coordinates."""
        return np.array([self.x, self.y])

    @property
    def heading_vector(self) -> np.ndarray:
        """Unit vector pointing along the vehicle's longitudinal direction."""
        return np.array([np.cos(self.orientation), np.sin(self.orientation)])

    @property
    def normal_vector(self) -> np.ndarray:
        """Unit vector perpendicular to vehicle heading (pointing left)."""
        return np.array([-np.sin(self.orientation), np.cos(self.orientation)])

    @property
    def velocity_vector(self) -> np.ndarray:
        """2D velocity vector [vx, vy] in world coordinates."""
        return self.velocity * self.heading_vector

    def to_tuple(self) -> tuple:
        """Legacy helper for functions expecting (x, y, orientation, velocity, length, width)."""
        return (self.x, self.y, self.orientation, self.velocity, self.length, self.width)

    # -------------------------------------------------------------------------
    # Kinematic Integration
    # -------------------------------------------------------------------------
    def update_kinematics(
        self, 
        accel: float, 
        steering_angle: float = 0.0, 
        wheelbase: float = 2.8, 
        dt: float = 0.1
    ) -> None:
        """
        Integrates vehicle state forward by dt using a Kinematic Bicycle Model
        with mid-point velocity integration.
        
        :param accel: Acceleration command in m/s^2 (u_control)
        :param steering_angle: Front wheel angle in radians (default 0.0 for straight motion)
        :param wheelbase: Wheelbase distance in meters
        :param dt: Integration time step in seconds
        """
        # 1. Next speed clipped to non-negative
        v_next = max(0.0, self.velocity + accel * dt)
        v_avg = 0.5 * (self.velocity + v_next)

        # 2. Update position using current heading and average speed
        self.x += v_avg * np.cos(self.orientation) * dt
        self.y += v_avg * np.sin(self.orientation) * dt

        # 3. Update heading angle (yaw rate = v_avg / L * tan(delta))
        if abs(steering_angle) > 1e-6:
            self.orientation += (v_avg / wheelbase) * np.tan(steering_angle) * dt

        # 4. Commit updated speed
        self.velocity = v_next
