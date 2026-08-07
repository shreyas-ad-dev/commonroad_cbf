from pathlib import Path
import numpy as np
from commonroad.common.file_reader import CommonRoadFileReader

def load_scenario_and_ego(xml_path: Path):
    """
    Loads CommonRoad scenario and automatically identifies Ego vehicle
    by matching initial state positions from planning_problem_set.
    """
    if not xml_path.exists():
        raise FileNotFoundError(f"Scenario file not found at: {xml_path}")

    scenario, planning_problem_set = CommonRoadFileReader(str(xml_path)).open()
    
    # Extract intended Ego initial state from planning problem
    planning_problem = list(planning_problem_set.planning_problem_dict.values())[0]
    ego_init_pos = planning_problem.initial_state.position

    ego_obstacle = None
    surrounding_obstacles = []

    # Automatically identify Ego obstacle based on position proximity
    for obs in scenario.obstacles:
        obs_pos = obs.initial_state.position
        if np.allclose(obs_pos, ego_init_pos, atol=1.0):
            ego_obstacle = obs
        else:
            surrounding_obstacles.append(obs)

    # Fallback safety if no exact match is found
    if ego_obstacle is None:
        print("⚠️ Warning: Could not match planning problem position to obstacle. Defaulting to first vehicle.")
        all_obs = list(scenario.obstacles)
        all_obs.sort(key=lambda o: o.initial_state.position[0])
        ego_obstacle = all_obs[0]
        surrounding_obstacles = all_obs[1:]

    return scenario, planning_problem_set, ego_obstacle, surrounding_obstacles
