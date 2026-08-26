# run_usa.py
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import numpy as np

from src.behavior_planner import BehaviorPlanner
from src.cbf_solver import CBFQPSolver
from src.ego_state import EgoState, get_car_polygon
from src.lateral_controller import (
    StanleyController,
    extract_target_lanelet_path,
    get_current_lane_width,
    get_road_heading_at_position,
)
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
XML_FILE = PROJECT_ROOT / "scenarios" / "USA_US101-9_1_T-1.xml"
GIF_NAME = "usa_us101_radar_cbf.gif"
NUM_STEPS = 100
DESIRED_SPEED = 16.5  # m/s

FRAMES_DIR = PROJECT_ROOT / "frames_usa"
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
    wheelbase=ego_params.get("wheelbase", 2.8)
)
print(f" Loaded Scenario: {scenario.scenario_id}")
print(f" Ego Initial Position: ({ego.x:.2f}, {ego.y:.2f})")

# -----------------------------------------------------------------------------
# 2. State Initialization & Module Setup
# -----------------------------------------------------------------------------
front_radar = RadarSensor(range_max=70.0, fov_deg=60.0, mount_position="front")
rear_radar = RadarSensor(range_max=50.0, fov_deg=80.0, mount_position="rear")
uss_left = SideUltrasonicSensor(range_max=5.0, fov_deg=100.0, side="left")
uss_right = SideUltrasonicSensor(range_max=5.0, fov_deg=100.0, side="right")
sensor_suite = SensorSuite( front_radar=front_radar, rear_radar=rear_radar, uss_left=uss_left, uss_right=uss_right)
cbf_solver = CBFQPSolver(gamma=1.2, d_min=6.0, tau=0.5, a_min=-8.0, a_max=2.0)
stanley_ctrl = StanleyController(k=0.7, k_soft=1.0, wheelbase=ego.wheelbase)

lane_width = get_current_lane_width(scenario, ego=ego)

# USA changes lane to the left (+lane_width) with no positional delay (trigger_x=None)
planner = BehaviorPlanner(mode="GAP_SEARCH", target_offset=lane_width, start_distance=30.0)
target_path = extract_target_lanelet_path(scenario, ego)

has_collided = False
collision_step = None
collided_obstacle_id = None
frozen_obs_states = {}
frame_files = []

# -----------------------------------------------------------------------------
# 3. Main Simulation Loop
# -----------------------------------------------------------------------------
for step in range(NUM_STEPS):
    # State Machine Update

    ego.road_heading = get_road_heading_at_position(scenario, ego.position)

    sensor_suite.update(ego=ego, all_obstacles=surrounding_obstacles, step=step)
    
    state, target_path = planner.update_plan(
        scenario=scenario,
        ego=ego,
        surrounding_obstacles=surrounding_obstacles,
        step=step,
        sensor_suite=sensor_suite,
        current_path=target_path,
    )

    # Radar Lead Tracking (using road heading for corridor accuracy)
    if planner.state == "LANE_CHANGE" and planner.lane_change_start_pos is not None:
        _, n_road = ego.road_frame_vectors
        lat_progress = np.dot(ego.position - planner.lane_change_start_pos, n_road)
        remaining_offset = planner.target_offset - lat_progress
        eval_offset = remaining_offset if abs(lat_progress) < 0.5 * abs(planner.target_offset) else 0.0
    else:
        eval_offset = 0.0

    lead_target = sensor_suite.track_lead(ego=ego, step=step, target_offset=eval_offset)

    d_safe = cbf_solver.d_min + (ego.velocity * cbf_solver.tau)

    if lead_target is not None and not has_collided:
        target_x, target_y, target_v, target_id, x_local = lead_target
        h_val = cbf_solver.compute_barrier(x_local, ego.velocity)

        u_control = cbf_solver.solve(
            longitudinal_dist=x_local,
            v_ego=ego.velocity,
            v_target=target_v,
            v_des=DESIRED_SPEED,
            dt=scenario.dt,
        )
    else:
        h_val = None
        u_control = 0.5 * (DESIRED_SPEED - ego.velocity)

    # Dynamics Propagation
    if not has_collided:
        steering_angle = stanley_ctrl.compute_steering(
            ego=ego, reference_path=target_path)
        ego.update_kinematics(accel=u_control, steering_angle=steering_angle, dt=scenario.dt)

    surrounding_render_states = []

    for obs in surrounding_obstacles:
        if has_collided and obs.obstacle_id == collided_obstacle_id:
            ox, oy, o_orient = frozen_obs_states[obs.obstacle_id]
        else:
            st = obs.state_at_time(step)
            if st is None:
                continue
            ox, oy = st.position[0], st.position[1]
            o_orient = getattr(st, "orientation", 0.0)

        l = getattr(obs.obstacle_shape, "length", 4.5)
        w = getattr(obs.obstacle_shape, "width", 2.0)
        obs_poly, obs_corners = get_car_polygon(ox, oy, o_orient, length=l, width=w)

        if not has_collided and ego.polygon.intersects(obs_poly):
            has_collided = True
            collision_step = step
            collided_obstacle_id = obs.obstacle_id
            frozen_obs_states[obs.obstacle_id] = (ox, oy, o_orient)

        is_hit = has_collided and (obs.obstacle_id == collided_obstacle_id)
        surrounding_render_states.append((obs, obs_corners, is_hit))

    frame_path = FRAMES_DIR / f"frame_{step:02d}.png"
    render_frame(
        scenario=scenario,
        planning_problem_set=planning_set,
        ego=ego,
        sensor_suite=sensor_suite,
        d_safe=d_safe,
        h_val=h_val,
        surrounding_states=surrounding_render_states,
        has_collided=has_collided,
        step=step,
        num_steps=NUM_STEPS,
        frame_path=frame_path,
        show_trajectories=SHOW_TRAJECTORIES,
    )
    frame_files.append(frame_path)

# -----------------------------------------------------------------------------
# 4. GIF Generation & Cleanup
# -----------------------------------------------------------------------------
gif_path = PROJECT_ROOT / GIF_NAME
build_gif_and_cleanup(frame_files, gif_path, scenario.dt)

print(f"\n USA Simulation Complete! Output saved to: '{gif_path.name}'")
if has_collided:
    print(f" Collision detected at Step {collision_step}.")
else:
    print(" SAFE! CBF Safety Control maintained nominal distance.")
