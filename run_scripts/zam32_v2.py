import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import numpy as np

from src.behavior_planner import BehaviorPlanner
from src.cbf_solver import CBFQPSolver
from src.ego_state import EgoState, get_car_polygon
from src.lateral_controller import StanleyController
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
XML_FILE = PROJECT_ROOT / "scenarios" / "ZAM_Zip-1_32_T-1.xml"
GIF_NAME = "zam_zip32_v2_merge.gif"
NUM_STEPS = 100
DESIRED_SPEED = 16  # High target speed to force late merge conflicts

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
map_module = MapModule(scenario=scenario, planning_problem_set=planning_set)

# Equipped sensors with measurement noise (noise_std) for tracking evaluation
front_radar = RadarSensor(range_max=70.0, fov_deg=60.0, mount_position="front", noise_std=0.3)
rear_radar = RadarSensor(range_max=50.0, fov_deg=80.0, mount_position="rear", noise_std=0.3)
uss_left = SideUltrasonicSensor(range_max=8.0, fov_deg=100.0, side="left", noise_std=0.1)
uss_right = SideUltrasonicSensor(range_max=8.0, fov_deg=100.0, side="right", noise_std=0.1)

# SensorSuite initializes internal MultiObjectTracker (Kalman Filter + Hungarian Matching)
sensor_suite = SensorSuite(
    front_radar=front_radar,
    rear_radar=rear_radar,
    uss_left=uss_left,
    uss_right=uss_right,
    dt=scenario.dt
)

cbf_solver = CBFQPSolver(gamma=1.2, d_min=6.0, tau=0.5, a_min=-8.0, a_max=2.0)
stanley_ctrl = StanleyController(k=0.7, k_soft=1.0, wheelbase=ego.wheelbase)

lane_width = map_module.get_current_lane_width(ego=ego)

# Behavior Planner configured for map target follow mode
planner = BehaviorPlanner(map_module=map_module, mode="MAP_FOLLOW")
target_path = map_module.extract_target_lanelet_path(ego)

has_collided = False
collision_step = None
collided_obstacle_id = None
frozen_obs_states = {}
frame_files = []

# -----------------------------------------------------------------------------
# 3. Main Simulation Loop
# -----------------------------------------------------------------------------
for step in range(NUM_STEPS):
    # 1. Update Perception Pipeline (Sensors -> Associations -> Kalman MultiObjectTracker)
    sensor_suite.update(ego=ego, all_obstacles=surrounding_obstacles, step=step)

    # 2. High-Level Behavior Planning Update
    state, target_path = planner.update_plan(
        ego=ego,
        step=step,
        sensor_suite=sensor_suite,
        current_path=target_path
    )

    # Check Lane Clearance using Fused Track/Sensor Checks
    clearance = sensor_suite.is_lane_change_safe_from_tracks(
        ego=ego,
        target_offset=lane_width,
        safety_gap_front=10.0,
        safety_gap_rear=8.0
    )
    target_clear = clearance.is_safe

    # 3. Lead Track Selection & CBF Safety Control
    if has_collided:
        u_control = 0.0
        steering_angle = 0.0
        sensor_suite.clear_tracking()
        h_val = None
        d_safe = cbf_solver.d_min + (ego.velocity * cbf_solver.tau)
    else:
        # Extract closest confirmed Kalman track directly from BehaviorPlanner
       # lead_track = planner.get_lead_track(ego=ego, sensor_suite=sensor_suite)
       # merge_hazard = planner.get_merge_hazard_track(ego=ego, sensor_suite=sensor_suite)
       # lead_track = lead_track if merge_hazard is None else merge_hazard
        lead_track = planner.select_lead_track(ego=ego, sensor_suite=sensor_suite)
        d_safe = cbf_solver.d_min + (ego.velocity * cbf_solver.tau)

        if lead_track is not None:
            u_road, _ = ego.road_frame_vectors
            d_vec = lead_track.position - ego.position
            long_dist = float(np.dot(d_vec, u_road))
            h_val = cbf_solver.compute_barrier(long_dist, ego.velocity)

            v_target_des = DESIRED_SPEED if target_clear else min(DESIRED_SPEED, float(np.hypot(lead_track.velocity[0], lead_track.velocity[1])) - 2.0)

            # Solve safe control acceleration using filtered track state estimate
            u_control = cbf_solver.solve_from_track(
                ego=ego,
                lead_track=lead_track,
                v_des=v_target_des,
                dt=scenario.dt
            )
        else:
            h_val = None
            u_control = 0.5 * (DESIRED_SPEED - ego.velocity)

        # Steering execution & Kinematics propagation
        steering_angle = stanley_ctrl.compute_steering(ego=ego, reference_path=target_path)
        ego.update_kinematics(accel=u_control, steering_angle=steering_angle, dt=scenario.dt)

    # 4. Collision Checking & Render Preparation
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

    # Frame output
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

print(f"\n ZAM32 Simulation Complete! Output saved to: '{gif_path.name}'")
if has_collided:
    print(f" Collision detected at Step {collision_step}.")
else:
    print(" SAFE! CBF Safety Control maintained nominal distance with Kalman perception.")
