from pathlib import Path

import numpy as np
from commonroad.common.file_reader import CommonRoadFileReader


def load_scenario_and_ego(xml_path: Path):
    """
    Loads a CommonRoad scenario XML and parses Ego vehicle and obstacle data.

    Supports both existing obstacle matching (ZAM style) where Ego is embedded 
    in scenario obstacles, and synthetic Ego spawns (USA style) where Ego is defined 
    purely by the scenario's initial planning problem.

    Args:
        xml_path (Path): Path to the CommonRoad scenario XML file.

    Returns:
        dict: A dictionary containing parsed scenario objects and Ego parameters:
            - 'scenario': The parsed CommonRoad Scenario instance.
            - 'planning_problem_set': The scenario's PlanningProblemSet instance.
            - 'ego_problem': Primary PlanningProblem object for the Ego vehicle.
            - 'ego_obstacle': Matched scenario obstacle object (if ZAM style), else None.
            - 'ego_params': Dict containing Ego initial state ('id', 'x', 'y', 
              'orientation', 'velocity', 'length', 'width').
            - 'surrounding_obstacles': List of remaining dynamic/static obstacles.

    Raises:
        FileNotFoundError: If xml_path does not exist on disk.
        ValueError: If no planning problems are present in the XML file.
    """
    
    if not xml_path.exists():
        raise FileNotFoundError(f"Scenario XML file not found at: {xml_path}")

    scenario, planning_problem_set = CommonRoadFileReader(str(xml_path)).open()
    planning_problems = list(planning_problem_set.planning_problem_dict.values())

    if not planning_problems:
        raise ValueError("No planning problems found in scenario XML.")

    ego_problem = planning_problems[0]
    initial_state = ego_problem.initial_state
    target_pos = initial_state.position

    # Extract orientation with robust fallbacks
    if hasattr(initial_state, 'orientation'):
        orient = float(initial_state.orientation)
    else:
        orient = 0.0

    # Extract initial velocity with robust fallbacks
    if hasattr(initial_state, 'velocity'):
        v_init = float(initial_state.velocity)
    else:
        v_init = 15.0

    ego_obstacle = None
    surrounding_obstacles = []

    # 1. Attempt to match Ego position with an existing scenario obstacle (ZAM style)
    for obs in scenario.obstacles:
        obs_pos = obs.initial_state.position
        if np.allclose(obs_pos, target_pos, atol=1.5):
            ego_obstacle = obs
        else:
            surrounding_obstacles.append(obs)

    # 2. Extract vehicle dimensions
    if ego_obstacle is not None:
        ego_l = float(getattr(ego_obstacle.obstacle_shape, 'length', 4.8))
        ego_w = float(getattr(ego_obstacle.obstacle_shape, 'width', 2.0))
        ego_id = ego_obstacle.obstacle_id
        # Override velocity/orientation from matched obstacle if available
        if hasattr(ego_obstacle.initial_state, 'velocity'):
            v_init = float(ego_obstacle.initial_state.velocity)
        if hasattr(ego_obstacle.initial_state, 'orientation'):
            orient = float(ego_obstacle.initial_state.orientation)
    else:
        # Synthetic Ego parameters (USA style)
        ego_l = 4.8
        ego_w = 2.0
        ego_id = "Ego_Synthetic"

    ego_params = {
        'id': ego_id,
        'x': float(target_pos[0]),
        'y': float(target_pos[1]),
        'orientation': orient,
        'velocity': v_init,
        'length': ego_l,
        'width': ego_w
    }

    return {
        'scenario': scenario,
        'planning_problem_set': planning_problem_set,
        'ego_problem': ego_problem,
        'ego_obstacle': ego_obstacle,
        'ego_params': ego_params,
        'surrounding_obstacles': surrounding_obstacles
    }
