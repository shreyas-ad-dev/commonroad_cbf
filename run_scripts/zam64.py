# zam64.py
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import numpy as np

from src.behavior_planner import BehaviorPlanner
from src.cbf_solver import CBFQPSolver
from src.collision_checker import CollisionChecker
from src.config import VEHICLE
from src.datalogger import DataLogger
from src.ego_state import EgoState
from src.map import MapModule
from src.radar import RadarSensor
from src.scenario_loader import load_scenario_and_ego
from src.sensor_suite import SensorSuite
from src.ultrasonic import SideUltrasonicSensor
from src.utils import build_gif_and_cleanup, setup_frames_directory
from src.visualizer import render_frame

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
SHOW_TRAJECTORIES = False
XML_FILE = PROJECT_ROOT / "scenarios" / "ZAM_Zip-1_64_T-1.xml"
GIF_NAME = "zam_zip64_merge.gif"
NUM_STEPS = 100
DESIRED_SPEED = 15.0  # m/s

FRAMES_DIR = PROJECT_ROOT / "frames_zam"
JSON_PATH = PROJECT_ROOT / "log_zam64.jsonl"
setup_frames_directory(FRAMES_DIR)

# -----------------------------------------------------------------------------
# 1. Load Scenario
# -----------------------------------------------------------------------------
scen_data = load_scenario_and_ego(XML_FILE)

scenario = scen_data["scenario"]
planning_set = scen_data["planning_problem_set"]
ego_params = scen_data["ego_params"]
surrounding_obstacles = scen_data["surrounding_obstacles"]

ego = EgoState(
    x=ego_params["x"],
    y=ego_params["y"],
    orientation=ego_params["orientation"],
    velocity=DESIRED_SPEED,
    length=ego_params["length"],
    width=ego_params["width"],
    wheelbase=ego_params.get("wheelbase", VEHICLE.DEFAULT_WHEELBASE),
    stanley_k=0.7,
    stanley_k_soft=1.0
)

collision_checker = CollisionChecker()

# -----------------------------------------------------------------------------
# 2. State Initialization & Module Setup
# -----------------------------------------------------------------------------
map_module = MapModule(scenario=scenario, planning_problem_set=planning_set)

lane_width = map_module.get_current_lane_width(ego=ego)
ego.y = ego.y - lane_width

print(f" Loaded Scenario: {scenario.scenario_id}")
print(f" Ego Initial Position: ({ego_params['x']:.2f}, {ego_params['y']:.2f})")

front_radar = RadarSensor(range_max=70.0, fov_deg=60.0, mount_position="front")
rear_radar = RadarSensor(range_max=50.0, fov_deg=80.0, mount_position="rear")
uss_left = SideUltrasonicSensor(range_max=8.0, fov_deg=100.0, side="left")
uss_right = SideUltrasonicSensor(range_max=8.0, fov_deg=100.0, side="right")

sensor_suite = SensorSuite(
        front_radar=front_radar,
        rear_radar=rear_radar,
        uss_left=uss_left,
        uss_right=uss_right,
        dt=scenario.dt
        )

cbf_solver = CBFQPSolver(gamma=1.2, d_min=6.0, tau=0.5, a_min=-8.0, a_max=2.0, use_cvxpy=True)

lane_width = map_module.get_current_lane_width(ego=ego)
#ego.y = ego.y - lane_width

planner = BehaviorPlanner(map_module=map_module, mode="MAP_FOLLOW", collision_checker=collision_checker)

logger = DataLogger(output_path=JSON_PATH)
target_path = map_module.extract_target_lanelet_path(ego=ego)

frame_files = []

# -----------------------------------------------------------------------------
# 3. Main Simulation Loop
# -----------------------------------------------------------------------------
for step in range(NUM_STEPS):
    # State Machine Update

    sensor_suite.update(ego=ego, all_obstacles=surrounding_obstacles, step=step)

    target_path = planner.update_plan(
        ego=ego,
        step=step,
        sensor_suite=sensor_suite,
        current_path=target_path
    )

    road_heading = map_module.get_road_heading_at_position( ego.position)

    # Radar Lead Tracking (using road heading for corridor accuracy)
    if planner.state == "LANE_CHANGE" and planner.lane_change_start_pos is not None:
        n_road = np.array([-np.sin(road_heading), np.cos(road_heading)])
        lat_progress = np.dot(ego.position - planner.lane_change_start_pos, n_road)
        remaining_offset = planner.target_offset - lat_progress
        eval_offset = remaining_offset if abs(lat_progress) < 0.5 * abs(planner.target_offset) else 0.0
    else:
        # In MAP_FOLLOW / LANE_KEEP: track vehicles directly in current lane
        eval_offset = 0.0


    if collision_checker.has_collided:
        u_control = 0.0
        steering_angle = 0.0
        sensor_suite.clear_tracking()
        h_val = None
    else:
        #lead_target = sensor_suite.track_lead(ego=ego, step=step, target_offset=eval_offset)
        steering_angle = None
        lead_track = planner.select_lead_track(ego=ego, sensor_suite=sensor_suite)
        d_safe = cbf_solver.d_min + (ego.velocity * cbf_solver.tau)

        if lead_track is not None:
            u_road, _ = ego.road_frame_vectors
            d_vec = lead_track.position - ego.position
            long_dist = float(np.dot(d_vec, u_road))

            h_val = cbf_solver.compute_barrier(long_dist, ego.velocity)
            u_control = cbf_solver.solve_from_track(
                    ego=ego,
                    lead_track=lead_track,
                    v_des=DESIRED_SPEED,
                    dt=scenario.dt,
                    )
        else:
            h_val = None
            u_control = 0.5 * (DESIRED_SPEED - ego.velocity)

        ego.update_kinematics(accel=u_control, steering_angle=steering_angle, reference_path=target_path, dt=scenario.dt)

    surrounding_render_states = collision_checker.evaluate_ground_truth_step(
            ego=ego,
            surrounding_obstacles=surrounding_obstacles,
            step=step,
            )

    frame_path = FRAMES_DIR / f"frame_{step:02d}.png"
    render_frame(
        scenario=scenario,
        planning_problem_set=planning_set,
        ego=ego,
        sensor_suite=sensor_suite,
        d_safe=d_safe,
        h_val=h_val,
        surrounding_states=surrounding_render_states,
        has_collided=collision_checker.has_collided,
        step=step,
        num_steps=NUM_STEPS,
        frame_path=frame_path,
        show_trajectories=SHOW_TRAJECTORIES,
    )
    frame_files.append(frame_path)

    logger.log_step(
            step=step,
            timestamp=step * scenario.dt,
            payload={
                "ego":ego.get_log_data(),
                "cbf": cbf_solver.get_log_data(),
                "planner": planner.get_log_data(),
                "sensors": sensor_suite.get_log_data()
                }
            )

logger.close()


# -----------------------------------------------------------------------------
# 4. GIF Generation & Cleanup
# -----------------------------------------------------------------------------
gif_path = PROJECT_ROOT / GIF_NAME
build_gif_and_cleanup(frame_files, gif_path, scenario.dt)

print(f"\n ZAM Simulation Complete! Output saved to: '{gif_path.name}'")
if collision_checker.has_collided:
    print(f" Collision detected at Step {collision_checker.collision_step}.")
else:
    print(" SAFE! CBF Safety Control maintained nominal distance.")
