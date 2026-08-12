# src/behavior_planner.py
import numpy as np
from src.lateral_controller import (generate_lane_change_path, extract_target_lanelet_path)

class BehaviorPlanner:
    def __init__(self, mode: str = "GAP_SEARCH", target_offset: float = 0.0, start_distance: float = 0.0):
        if mode not in ["GAP_SEARCH", "MAP_FOLLOW"]:
            raise ValueError("mode must be either 'GAP_SEARCH' or 'MAP_FOLLOW'")
            
        self.mode = mode
        self.target_offset = target_offset
        self.start_distance = start_distance
        self.state = "LANE_KEEP"
        
        self.start_x = None
        self.start_y = None

    def update_plan(self, scenario, ego_x, ego_y, ego_orient, surrounding_obstacles, step, radar, current_path, rear_radar=None):
        if self.start_x is None or self.start_y is None:
            self.start_x = ego_x
            self.start_y = ego_y

        if self.mode == "MAP_FOLLOW":
            updated_path = extract_target_lanelet_path(scenario, ego_x, ego_y)
            return "MAP_FOLLOW", updated_path

        if self.state == "LANE_CHANGE":
            return self.state, current_path

        distance_traveled = np.hypot(ego_x - self.start_x, ego_y - self.start_y)
        if distance_traveled < self.start_distance:
            return self.state, current_path

        # Dual radar clearance check
        is_clear = radar.is_adjacent_lane_clear(
            ego_x, ego_y, ego_orient, surrounding_obstacles, step, 
            self.target_offset, safety_gap_front=15.0, safety_gap_rear=18.0, 
            road_heading=ego_orient, rear_radar=rear_radar
        )

        if is_clear:
            self.state = "LANE_CHANGE"
            print(f"🚀 [Step {step} | Dist: {distance_traveled:.1f}m] Gap clear! Initiating dynamic lane change.")
            new_path = generate_lane_change_path(
                start_x=ego_x,
                start_y=ego_y,
                road_heading=ego_orient,
                target_lane_offset=self.target_offset,
                total_length=120.0
            )
            return self.state, new_path

        return self.state, current_path
