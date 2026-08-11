# src/behavior_planner.py
import numpy as np
from src.lateral_controller import generate_lane_change_path

class BehaviorPlanner:
    def __init__(self, scenario_name: str, target_offset: float):
        self.scenario_name = scenario_name
        self.target_offset = target_offset
        self.state = "LANE_KEEP"  # Initial state
        
    def update_plan(self, scenario, ego_x, ego_y, ego_orient, surrounding_obstacles, step, radar, current_path):
        """
        Evaluates perception data and returns the updated (state, target_path).
        """
        if self.scenario_name != "USA" or self.state == "LANE_CHANGE":
            return self.state, current_path

        # Check if adjacent lane is safe in both front and rear blind spots
        is_clear = radar.is_adjacent_lane_clear(
            ego_x, ego_y, ego_orient, surrounding_obstacles, step, self.target_offset
        )

        if is_clear:
            self.state = "LANE_CHANGE"
            print(f"🚀 [Step {step}] Adjacent lane clear! Triggering lane change maneuver.")
            
            # Dynamically plan new S-curve path from current coordinates
            new_path = generate_lane_change_path(
                start_x=ego_x,
                start_y=ego_y,
                road_heading=ego_orient,
                target_lane_offset=self.target_offset,
                total_length=120.0
            )
            return self.state, new_path

        return self.state, current_path
