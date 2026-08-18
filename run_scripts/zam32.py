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
    get_current_lane_width,
    get_road_heading_at_position,
    extract_target_lanelet_path
)
from src.radar import RadarSensor
from src.scenario_loader import load_scenario_and_ego
from src.ultrasonic import SideUltrasonicSensor
from src.utils import setup_frames_directory, build_gif_and_cleanup
from src.visualizer import render_frame

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
SHOW_TRAJECTORIES = False
XML_FILE = PROJECT_ROOT / "scenarios" / "ZAM_Zip-1_32_T-1.xml"
GIF_NAME = "zam_zip32_merge.gif"
NUM_STEPS = 100
DESIRED_SPEED = 14.5  # High target speed to force late merge conflicts

FRAMES_DIR = PROJECT_ROOT / "frames_zam32"
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
uss_left = SideUltrasonicSensor(range_max=8.0, fov_deg=100.0, side="left")
uss_right = SideUltrasonicSensor(range_max=8.0, fov_deg=100.0, side="right")

cbf_solver = CBFQPSolver(gamma=1.2, d_min=6.0, tau=0.5, a_min=-8.0, a_max=2.0)
stanley_ctrl = StanleyController(k=0.7, k_soft=1.0, wheelbase=ego.wheelbase)

lane_width = get_current_lane_width(scenario, ego=ego)

# Behavior Planner configured for target lane merge (+lane_width)
planner = BehaviorPlanner(mode="MAP_FOLLOW")
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
    state, target_path = planner.update_plan(
        scenario=scenario,
        ego=ego,
        surrounding_obstacles=surrounding_obstacles,
        step=step,
        radar=front_radar,
        rear_radar=rear_radar,
        uss_left=uss_left,
        uss_right=uss_right,
        current_path=target_path
    )

    target_clear = front_radar.is_adjacent_lane_clear(
        ego, surrounding_obstacles, step, target_lane_offset=lane_width, rear_radar=rear_radar
    )

    # Lead Tracking via Radar
    road_heading = get_road_heading_at_position(scenario, ego.position)
    lead_target = front_radar.track_lead_vehicle(
        ego, surrounding_obstacles, step,
        target_offset=lane_width,
        road_heading=road_heading,
        is_changing_lane=True
    )

    d_safe = cbf_solver.d_min + (ego.velocity * cbf_solver.tau)

    if lead_target is not None and not has_collided:
        target_x, target_y, target_v, target_id, x_local = lead_target
        h_val = cbf_solver.compute_barrier(x_local, ego.velocity)

        u_control = cbf_solver.solve(
            longitudinal_dist=x_local,
            v_ego=ego.velocity,
            v_target=target_v,
            v_des=DESIRED_SPEED if target_clear else min(DESIRED_SPEED, target_v - 2.0),
            dt=scenario.dt,
        )
    else:
        h_val = None
        u_control = 0.5 * (DESIRED_SPEED - ego.velocity)

    # Vehicle Dynamics Integration
    if not has_collided:
        steering_angle = stanley_ctrl.compute_steering(ego=ego, reference_path=target_path)
        ego.update_kinematics(accel=u_control, steering_angle=steering_angle, dt=scenario.dt)

    # Polygon Collision Checking & Perception Rendering States
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

    front_tracked = front_radar.get_detected_obstacle_ids(ego, surrounding_obstacles, step)
    rear_tracked = rear_radar.get_detected_obstacle_ids(ego, surrounding_obstacles, step)
    left_tracked = uss_left.get_detected_obstacle_ids(ego, surrounding_obstacles, step)
    right_tracked = uss_right.get_detected_obstacle_ids(ego, surrounding_obstacles, step)

    frame_path = FRAMES_DIR / f"frame_{step:02d}.png"
    render_frame(
        scenario=scenario,
        planning_problem_set=planning_set,
        ego=ego,
        d_safe=d_safe,
        h_val=h_val,
        radar_range=front_radar.range_max,
        radar_fov_deg=front_radar.fov_deg,
        rear_radar_range=rear_radar.range_max,
        rear_radar_fov_deg=rear_radar.fov_deg,
        front_tracked_ids=front_tracked,
        rear_tracked_ids=rear_tracked,
        uss_range=uss_left.range_max,
        uss_fov_deg=uss_left.fov_deg,
        left_tracked_ids=left_tracked,
        right_tracked_ids=right_tracked,
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

print(f"\n ZAM32 Simulation Complete! Output saved to: '{gif_path.name}'")
if has_collided:
    print(f" Collision detected at Step {collision_step}.")
else:
    print(" SAFE! CBF Safety Control maintained nominal distance.")
