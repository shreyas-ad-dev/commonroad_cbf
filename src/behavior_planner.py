# src/behavior_planner.py
import numpy as np

from src.ego_state import EgoState
from src.lateral_controller import (
    extract_target_lanelet_path,
    generate_lane_change_path,
)
from src.visualizer import AdjacentGapConfig

class BehaviorPlanner:
    """
    High-level state machine responsible for managing autonomous driving behaviors.
    
    Evaluates sensor clearance (radar/ultrasonic) and determines when to maintain 
    lane keeping or initiate dynamic lane changes based on surrounding traffic gaps.
    """

    def __init__(self,
                 mode: str = "GAP_SEARCH",
                 target_offset: float = 0.0,
                 start_distance: float = 0.0):
        """
        Initializes the BehaviorPlanner instance.

        Args:
            mode (str): Planning mode. Must be either 'GAP_SEARCH' or 'MAP_FOLLOW'.
            target_offset (float): Lateral offset to target lane in meters (+ for left, - for right).
            start_distance (float): Minimum distance Ego must travel before triggering a lane change.

        Raises:
            ValueError: If mode is not 'GAP_SEARCH' or 'MAP_FOLLOW'.
        """
        
        if mode not in ["GAP_SEARCH", "MAP_FOLLOW"]:
            raise ValueError("mode must be either 'GAP_SEARCH' or 'MAP_FOLLOW'")
            
        self.mode = mode
        self.target_offset = target_offset
        self.start_distance = start_distance
        self.state = "LANE_KEEP"
        
        self.start_x = None
        self.start_y = None

    def update_plan(self,
                    scenario,
                    ego:EgoState,
                    surrounding_obstacles,
                    step: int,
                    radar,
                    current_path,
                    rear_radar=None,
                    uss_left=None,
                    uss_right=None,
                    road_heading: float | None = None
                    ):
        """
        Updates the high-level behavioral state and evaluates reference trajectory paths.

        Evaluates clearance from front/rear radar and side ultrasonic sensors. If all
        active sensors report an open gap in the target lane and the minimum travel distance
        threshold is met, transitions state from 'LANE_KEEP' to 'LANE_CHANGE'.

        Args:
            scenario: The CommonRoad scenario object containing map and timing info.
            ego (EgoState): Current state and kinematics of the Ego vehicle.
            surrounding_obstacles (list): List of dynamic obstacles in the scene.
            step (int): Current simulation time step index.
            radar (RadarSensor): Front radar sensor instance.
            current_path (np.ndarray): Current reference trajectory path coordinates.
            rear_radar (RadarSensor, optional): Rear radar sensor instance. Defaults to None.
            uss_left (SideUltrasonicSensor, optional): Left ultrasonic sensor instance. Defaults to None.
            uss_right (SideUltrasonicSensor, optional): Right ultrasonic sensor instance. Defaults to None.

        Returns:
            Tuple[str, np.ndarray]: Updated planner state ('LANE_KEEP' or 'LANE_CHANGE') 
            and the active reference path trajectory.
        """

        if self.start_x is None or self.start_y is None:
            self.start_x = ego.x
            self.start_y = ego.y

        heading = road_heading if road_heading is not None else ego.orientation
        gap_cfg = AdjacentGapConfig(
            target_lane_offset=self.target_offset,
            safety_gap_front=10.0,
            safety_gap_rear=5.0,
            road_heading=heading
        )

        if self.mode == "MAP_FOLLOW":
            updated_path = extract_target_lanelet_path(scenario, ego )
            return self.state, updated_path, gap_cfg

        if self.state == "LANE_CHANGE":
            return self.state, current_path, gap_cfg

        distance_traveled = np.hypot(ego.x - self.start_x, ego.y - self.start_y)
        if distance_traveled < self.start_distance:
            return self.state, current_path, gap_cfg

        # Dual radar clearance check
        radar_clear = radar.is_adjacent_lane_clear(
                ego,
                surrounding_obstacles,
                step,
                self.target_offset,
                safety_gap_front=10.0,
                safety_gap_rear=12.0,
                road_heading=ego.orientation,
                rear_radar=rear_radar
        )
        gap_cfg.is_clear = radar_clear

        active_uss = uss_left if self.target_offset > 0 else uss_right
        uss_clear = active_uss.is_adjacent_lane_clear(
                ego,
                surrounding_obstacles,
                step,
                self.target_offset
        ) if active_uss is not None else True

        is_clear = radar_clear and uss_clear

        if is_clear:
            self.state = "LANE_CHANGE"
            print(f" [Step {step} | Dist: {distance_traveled:.1f}m] Gap clear! Initiating dynamic lane change.")
            new_path = generate_lane_change_path(
                start_pos=ego.position,
                road_heading=ego.orientation,
                target_lane_offset=self.target_offset,
                total_length=120.0
            )
            return self.state, new_path, gap_cfg

        return self.state, current_path, gap_cfg

