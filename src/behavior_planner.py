# src/behavior_planner.py
import numpy as np

from src.ego_state import EgoState
from src.lateral_controller import (
    #    extract_target_lanelet_path,
    generate_lane_change_path,
)
from src.map import MapModule
from src.sensor_suite import SensorSuite
from src.tracker import Track


class BehaviorPlanner:
    """High-level state machine responsible for managing autonomous driving behaviors.

    Evaluates sensor clearance (via `SensorSuite`) and determines when to maintain 
    lane keeping or initiate dynamic lane changes based on surrounding traffic gaps.

    Attributes:
        mode (str): Planning mode ('GAP_SEARCH' or 'MAP_FOLLOW').
        target_offset (float): Lateral offset to target lane in meters.
        start_distance (float): Minimum distance Ego must travel before triggering a lane change.
        state (str): Current behavioral state ('LANE_KEEP' or 'LANE_CHANGE').
        start_x (float | None): Initial X position of the Ego vehicle.
        start_y (float | None): Initial Y position of the Ego vehicle.
        lane_change_start_pos (np.ndarray | None): Ego position when lane change was initiated.
    """

    def __init__(self,
                 map_module = MapModule,
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

        self.map_module = map_module

        if mode not in ["GAP_SEARCH", "MAP_FOLLOW"]:
            raise ValueError("mode must be either 'GAP_SEARCH' or 'MAP_FOLLOW'")
            
        self.mode = mode
        self.target_offset = target_offset
        self.start_distance = start_distance
        self.state = "LANE_KEEP"
        
        self.start_x = None
        self.start_y = None
        self.lane_change_start_pos = None

    def get_lead_track(self, ego: EgoState, sensor_suite: SensorSuite, lateral_margin: float = 3.2) -> Track | None:
        """Identifies the closest tracked lead vehicle in Ego's current corridor using tracked states."""
        tracks = sensor_suite.tracked_objects
        if not tracks:
            return None

        u_road, n_road = ego.road_frame_vectors
        closest_dist = float('inf')
        lead_track = None

        for track in tracks:
            # Vector from ego to track position estimate[cite: 10]
            d_vec = track.position - ego.position
            long_road = np.dot(d_vec, u_road)
            lat_road = np.dot(d_vec, n_road)

            # Check if track is ahead in lane corridor
            if long_road > 0.0:
                v_lat = float(np.dot(track.velocity, n_road))
                effective_lat_threshold = lateral_margin if abs(lat_road) < lateral_margin and np.sign(lat_road) != np.sign(v_lat) else 1.8
                if abs(lat_road) <= effective_lat_threshold:
                    if long_road < closest_dist:
                        closest_dist = long_road
                        lead_track = track

        return lead_track

    def get_merge_hazard_track(self, ego: EgoState, sensor_suite: SensorSuite, target_offset: float = 0.0) -> Track | None:
        """
        Scans tracked objects in both Ego's lane and target merging lane corridor.
        """
        tracks = sensor_suite.tracked_objects
        if not tracks:
            return None

        u_road, n_road = ego.road_frame_vectors
        closest_dist = float('inf')
        hazard_track = None

        for track in tracks:
            d_vec = track.position - ego.position
            long_road = np.dot(d_vec, u_road)
            lat_road = np.dot(d_vec, n_road)

            # Scan corridor including current lane and target merging lane offset
            in_merge_corridor = (abs(lat_road) <= 1.8) or (abs(lat_road - target_offset) <= 1.8)

            if long_road > 0.0 and in_merge_corridor:
                if long_road < closest_dist:
                    closest_dist = long_road
                    hazard_track = track

        return hazard_track

    def update_plan(self,
                    ego:EgoState,
                    step: int,
                    sensor_suite:SensorSuite,
                    current_path,
                    ):
        """
        Updates high-level behavioral state and generates target reference trajectories.

        Evaluates clearance around the Ego vehicle using the sensor suite. If 
        lane change clearance flags confirm a safe gap in the target lane and 
        the travel distance threshold is met, transitions state from 'LANE_KEEP' 
        to 'LANE_CHANGE'.

        Args:
            scenario: CommonRoad scenario object containing map and timing info.
            ego (EgoState): Current state and kinematics of the Ego vehicle.
            surrounding_obstacles (list): List of dynamic obstacles in the scene.
            step (int): Current simulation time step index.
            sensor_suite (SensorSuite): Suite containing active perception sensors.
            current_path (np.ndarray): Current reference trajectory path coordinates.

        Returns:
            tuple[str, np.ndarray]: A tuple containing:
                - state (str): Updated planner state ('LANE_KEEP' or 'LANE_CHANGE').
                - active_path (np.ndarray): The active reference path trajectory coordinates.
        """

        # Calculate distance to upcoming lane merge point
        dist_to_merge = self.map_module.get_distance_to_next_merge(ego=ego)

        # Trigger early crash checks if merge is within 10 meters
#        t_lookahead = 1.5
#        d_min = 5.0
#        merge_threshold = max(ego.velocity * t_lookahead + d_min, 15.0)
#        self.is_checking_merge = (dist_to_merge <= merge_threshold)

        self.is_checking_merge = (dist_to_merge <= 25.0)
        if self.is_checking_merge:
            # Preemptively check adjacent lane clearance via SensorSuite
            clearance = sensor_suite.is_lane_change_safe(
                    ego=ego, 
                    target_offset=self.target_offset, 
                    step=step, 
                    safety_gap_front=10.0, 
                    safety_gap_rear=8.0
                    )
            if not clearance.is_safe:
                print(f" [Step {step}] Merge hazard detected within {dist_to_merge:.1f}m! Preemptively adjusting velocity.")
            
        if self.mode == "MAP_FOLLOW":
            updated_path = self.map_module.extract_target_lanelet_path(ego)
            return self.state, updated_path

        if self.start_x is None or self.start_y is None:
            self.start_x = ego.x
            self.start_y = ego.y


        if self.mode == "MAP_FOLLOW":
            updated_path = self.map_module.extract_target_lanelet_path( ego )
            return self.state, updated_path

        if self.state == "LANE_CHANGE":
            _, n_road = ego.road_frame_vectors
            disp_vec = ego.position - self.lane_change_start_pos
            lat_progress = np.dot(disp_vec, n_road)
            

            if abs(lat_progress) >= 0.85 * abs(self.target_offset):
                print(f" [Step {step} Lane change complete. Transitioning back to LANE_KEEP/ MAP_FOLLOW.")
                self.state = "LANE_KEEP"
                self.target_offset = 0.0
                self.mode = "MAP_FOLLOW"
                updated_path = self.map_module.extract_target_lanelet_path( ego)
                return self.state, updated_path

            return self.state, current_path

        distance_traveled = np.hypot(ego.x - self.start_x, ego.y - self.start_y)
        if distance_traveled < self.start_distance:
            return self.state, current_path

        lane_change_clearance_flags  = sensor_suite.is_lane_change_safe(ego=ego, target_offset=self.target_offset, step=step, safety_gap_front=10.0, safety_gap_rear=8.0)

        if lane_change_clearance_flags.is_safe:
            self.state = "LANE_CHANGE"
            self.lane_change_start_pos = ego.position.copy()
            print(f" [Step {step} | Dist: {distance_traveled:.1f}m] Gap clear! Initiating dynamic lane change.")
            new_path = generate_lane_change_path(
                ego=ego,
                target_lane_offset=self.target_offset,
                total_length=120.0
            )
            return self.state, new_path

        return self.state, current_path

