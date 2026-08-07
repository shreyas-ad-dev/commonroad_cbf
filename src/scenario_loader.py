from pathlib import Path
import numpy as np
from commonroad.common.file_reader import CommonRoadFileReader

def load_scenario_and_ego(xml_path: Path):
    """
    Loads a CommonRoad scenario XML and returns a unified data structure
    supporting both existing obstacle matching (ZAM style) and synthetic 
    Ego spawns (USA style).
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
