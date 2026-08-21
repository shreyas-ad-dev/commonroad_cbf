# src/ego_state.py
from dataclasses import dataclass

import numpy as np
from shapely.geometry import Polygon


def get_car_polygon(
        x: float, 
        y: float, 
        orientation: float, 
        length: float, 
        width: float) -> tuple[Polygon, np.ndarray]:
    """
    Computes global 2D corner coordinates and a Shapely Polygon for any vehicle.

    Args:
        x (float): Global X position of the vehicle center in meters.
        y (float): Global Y position of the vehicle center in meters.
        orientation (float): Yaw angle in radians.
        length (float): Vehicle length in meters.
        width (float): Vehicle width in meters.

    Returns:
        tuple[Polygon, np.ndarray]: A tuple containing:
            - Polygon: Shapely Polygon object for geometry/collision operations.
            - np.ndarray: A 4x2 array containing global [x, y] corner coordinates.
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
    """
    Data class representing the state, dimensions, and kinematic dynamics of the Ego vehicle.

    Attributes:
        x (float): Global X coordinate of the vehicle center in meters.
        y (float): Global Y coordinate of the vehicle center in meters.
        orientation (float): Yaw orientation angle in radians.
        velocity (float): Scalar longitudinal speed in m/s.
        length (float): Vehicle length in meters.
        width (float): Vehicle width in meters.
        wheelbase (float): Distance between front and rear axles in meters.
    """
    
    x: float
    y: float
    orientation: float  # Yaw angle in radians
    velocity: float     # Scalar speed in m/s
    length: float       # Vehicle length in meters
    width: float        # Vehicle width in meters
    wheelbase: float
    road_heading: float | None = None

    # -------------------------------------------------------------------------
    # Properties & Vector Helpers
    # -------------------------------------------------------------------------
    @property
    def position(self) -> np.ndarray:
        """
        Gets the 2D global position vector of the vehicle.

        Returns:
            np.ndarray: 2D array [x, y] representing world position.
        """
        return np.array([self.x, self.y])

    @property
    def heading_vector(self) -> np.ndarray:
        """
        Gets the unit vector pointing along the vehicle's longitudinal direction.

        Returns:
            np.ndarray: 2D unit direction vector [cos(orientation), sin(orientation)].
        """
        return np.array([np.cos(self.orientation), np.sin(self.orientation)])

    @property
    def normal_vector(self) -> np.ndarray:
        """
        Gets the unit vector perpendicular to vehicle heading (pointing left).

        Returns:
            np.ndarray: 2D normal unit vector [-sin(orientation), cos(orientation)].
        """
        return np.array([-np.sin(self.orientation), np.cos(self.orientation)])

    @property
    def velocity_vector(self) -> np.ndarray:
        """
        Gets the 2D global velocity vector of the vehicle.

        Returns:
            np.ndarray: 2D velocity vector [vx, vy] in world coordinates.
        """
        return self.velocity * self.heading_vector

    @property
    def road_frame_vectors(self) -> tuple[np.ndarray, np.ndarray]:
        if self.road_heading is not None:
            return (
                    np.array([np.cos(self.road_heading), np.sin(self.road_heading)]),
                    np.array([-np.sin(self.road_heading), np.cos(self.road_heading)])
                    )
        return self.heading_vector, self.normal_vector


    @property
    def front_axle_position(self) -> np.ndarray:
        """
        Computes the 2D global position of the front axle.

        Returns:
            np.ndarray: 2D position vector [x_front, y_front] in world coordinates.
        """
        return self.position + self.wheelbase * self.heading_vector
# -------------------------------------------------------------------------
    # Geometric Bounding Box & Polygon
    # -------------------------------------------------------------------------
    @property
    def polygon_and_corners(self) -> tuple[Polygon, np.ndarray]:
        """
        Gets both the Shapely Polygon and bounding corner coordinates.

        Returns:
            tuple[Polygon, np.ndarray]: Tuple containing (Shapely Polygon, 4x2 corner array).
        """
        return get_car_polygon(self.x, self.y, self.orientation, self.length, self.width)

    @property
    def corners(self) -> np.ndarray:
        """
        Computes global 2D corner coordinates for the vehicle bounding box.

        Returns:
            np.ndarray: A 4x2 array containing [x, y] coordinates for all four corners.
        """
        return self.polygon_and_corners[1]

    @property
    def polygon(self) -> Polygon:
        """
        Gets the Shapely Polygon representation for exact collision testing.

        Returns:
            Polygon: Shapely Polygon object matching current position and orientation.
        """
        return self.polygon_and_corners[0]

    def to_tuple(self) -> tuple:
        """
        Provides a legacy tuple representation of the vehicle state.

        Returns:
            tuple: Tuple containing (x, y, orientation, velocity, length, width).
        """
        return (self.x, self.y, self.orientation, self.velocity, self.length, self.width)

    # -------------------------------------------------------------------------
    # Kinematic Integration
    # -------------------------------------------------------------------------
    def update_kinematics(self, 
                          accel: float, 
                          steering_angle: float = 0.0, 
                          dt: float = 0.1
                          ) -> None:
        """
        Integrates vehicle state forward in time using a Kinematic Bicycle Model.

        Uses mid-point velocity integration to update global position, orientation, 
        and speed based on commanded acceleration and steering angle.

        Args:
            accel (float): Longitudinal acceleration command in m/s^2.
            steering_angle (float, optional): Front wheel steering angle in radians. Defaults to 0.0.
            dt (float, optional): Integration time step in seconds. Defaults to 0.1.
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
