# src/ego_state.py
from dataclasses import dataclass
import numpy as np
from shapely.geometry import Polygon


def get_car_polygon(
    x: float, 
    y: float, 
    orientation: float, 
    length: float, 
    width: float
) -> tuple[Polygon, np.ndarray]:
    """
    Computes global 2D corner coordinates and a Shapely Polygon 
    for any vehicle (Ego or obstacle).
    
    :return: (Shapely Polygon, 4x2 numpy array of corners)
    """
    l2, w2 = length / 2.0, width / 2.0
    local_corners = np.array([
        [-l2, -w2],
        [ l2, -w2],
        [ l2,  w2],
        [-l2,  w2]
    ])
    
    cos_a, sin_a = np.cos(orientation), np.sin(orientation)
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    
    global_corners = local_corners @ rot.T + np.array([x, y])
    return Polygon(global_corners), global_corners


@dataclass
class EgoState:
    x: float
    y: float
    orientation: float  # Yaw angle in radians
    velocity: float     # Scalar speed in m/s
    length: float       # Vehicle length in meters
    width: float        # Vehicle width in meters
    wheelbase: float

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

    @property
    def front_axle_position(self) -> np.ndarray:
        """Computes the 2D global position of the front axle."""
        # Assuming wheelbase or front overhang offset is ~2.8m (or self.wheelbase)
        return self.position + self.wheelbase * self.heading_vector
# -------------------------------------------------------------------------
    # Geometric Bounding Box & Polygon
    # -------------------------------------------------------------------------
    @property
    def polygon_and_corners(self) -> tuple[Polygon, np.ndarray]:
        """Returns (Polygon, corners) using the global helper."""
        return get_car_polygon(self.x, self.y, self.orientation, self.length, self.width)

    @property
    def corners(self) -> np.ndarray:
        """Computes global 2D corner coordinates [4x2 numpy array]."""
        return self.polygon_and_corners[1]

    @property
    def polygon(self) -> Polygon:
        """Returns a Shapely Polygon representation for exact collision testing."""
        return self.polygon_and_corners[0]
#    @property
#    def corners(self) -> np.ndarray:
#        """Computes global 2D corner coordinates [4x2 numpy array]."""
#        l2, w2 = self.length / 2.0, self.width / 2.0
#        local_corners = np.array([
#            [-l2, -w2],
#            [ l2, -w2],
#            [ l2,  w2],
#            [-l2,  w2]
#        ])
#        return self.position + np.outer(local_corners[:, 0], self.heading_vector) + np.outer(local_corners[:, 1], self.normal_vector)
#
#    @property
#    def polygon(self) -> Polygon:
#        """Returns a Shapely Polygon representation for exact collision testing."""
#        return Polygon(self.corners)
#
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
            self.orientation += (v_avg / self.wheelbase) * np.tan(steering_angle) * dt

        # 4. Commit updated speed
        self.velocity = v_next
