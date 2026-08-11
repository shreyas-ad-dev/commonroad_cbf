# src/behavior_planner.py
import numpy as np
from src.lateral_controller import (generate_lane_change_path, extract_target_lanelet_path)

class BehaviorPlanner:
    def __init__(self, target_offset: float):
        """
        :param target_offset: Positive for USA (left shift), Negative for ZAM (right shift)
        """
        self.target_offset = target_offset
        self.state = "LANE_KEEP"

    def update_plan(self, scenario, ego_x, ego_y, ego_orient, surrounding_obstacles, step, radar, current_path):
        # For ZAM zip-merge, Ego simply follows the map's lanelet path (which includes the natural merge curve)
        if self.target_offset < 0:  # ZAM scenario indicator
            # Re-extracting path keeps Ego aligned with CommonRoad's successor lanelet geometry
            updated_path = extract_target_lanelet_path(scenario, ego_x, ego_y)
            return "ZIP_MERGE", updated_path

        # For USA highway, perform dynamic gap checking for proactive S-curve lane change
        if self.state == "LANE_CHANGE":
            return self.state, current_path

        is_clear = radar.is_adjacent_lane_clear(
            ego_x, ego_y, ego_orient, surrounding_obstacles, step, self.target_offset
        )

        if is_clear:
            self.state = "LANE_CHANGE"
            print(f"🚀 [Step {step}] Target lane clear! Initiating dynamic lane change.")
            new_path = generate_lane_change_path(
                start_x=ego_x,
                start_y=ego_y,
                road_heading=ego_orient,
                target_lane_offset=self.target_offset,
                total_length=120.0
            )
            return self.state, new_path

        return self.state, current_path
