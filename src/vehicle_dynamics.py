import numpy as np
from shapely.geometry import Polygon

def get_car_polygon(x: float, y: float, angle_rad: float, length: float = 4.5, width: float = 2.0):
    """
    Computes global 2D corner coordinates and returns a Shapely Polygon
    for exact geometric intersection testing.
    """
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    l2, w2 = length / 2.0, width / 2.0
    
    local_corners = np.array([
        [-l2, -w2],
        [ l2, -w2],
        [ l2,  w2],
        [-l2,  w2]
    ])
    
    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    world_corners = local_corners @ rotation_matrix.T + np.array([x, y])
    
    return Polygon(world_corners), world_corners
