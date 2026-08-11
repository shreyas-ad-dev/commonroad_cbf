# run_usa.py
from pathlib import Path
import numpy as np

from src.behavior_planner import BehaviorPlanner
from src.cbf_solver import CBFQPSolver
from src.lateral_controller import (
    StanleyController,
    get_current_lane_width,
    extract_target_lanelet_path
)
from src.radar import RadarSensor
from src.scenario_loader import load_scenario_and_ego
from src.vehicle_dynamics import get_car_polygon
from src.visualizer import create_gif_from_frames, render_frame

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
SHOW_TRAJECTORIES = False
PROJECT_ROOT = Path(__file__).resolve().parent
XML_FILE = PROJECT_ROOT / "scenarios" / "USA_US101-9_1_T-1.xml"
GIF_NAME = "usa_us101_radar_cbf.gif"
NUM_STEPS = 80
DESIRED_SPEED = 30.0  # m/s

FRAMES_DIR = PROJECT_ROOT / "frames_usa"

if FRAMES_DIR.exists():
    for f in FRAMES_DIR.glob("*.png"):
        f.unlink()
FRAMES_DIR.mkdir(exist_ok=True)

# -----------------------------------------------------------------------------
# 1. Load Scenario
# -----------------------------------------------------------------------------
scen_data = load_scenario_and_ego(XML_FILE)

scenario = scen_data["scenario"]
planning_set = scen_data["planning_problem_set"]
ego_params = scen_data["ego_params"]
surrounding_obstacles = scen_data["surrounding_obstacles"]

print(f" Loaded Scenario: {scenario.scenario_id}")
print(f" Ego Initial Position: ({ego_params['x']:.2f}, {ego_params['y']:.2f})")

# -----------------------------------------------------------------------------
# 2. State Initialization & Module Setup
# -----------------------------------------------------------------------------
ego_x = ego_params["x"]
ego_y = ego_params["y"]
ego_orient = ego_params["orientation"]
ego_v = ego_params["velocity"]
ego_l = ego_params["length"]
ego_w = ego_params["width"]

radar = RadarSensor(range_max=70.0, fov_deg=60.0)
cbf_solver = CBFQPSolver(gamma=1.2, d_min=6.0, tau=0.5, a_min=-8.0, a_max=2.0)
stanley_ctrl = StanleyController(k=0.7, k_soft=1.0)

lane_width = get_current_lane_width(scenario, ego_x=ego_params["x"], ego_y=ego_params["y"])

# USA changes lane to the left (+lane_width) with no positional delay (trigger_x=None)
planner = BehaviorPlanner(mode="GAP_SEARCH", target_offset=lane_width, start_distance=75.0)
#planner = BehaviorPlanner(target_offset=lane_width)
target_path = extract_target_lanelet_path(scenario, ego_params["x"], ego_params["y"])

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
        ego_x=ego_x,
        ego_y=ego_y,
        ego_orient=ego_orient,
        surrounding_obstacles=surrounding_obstacles,
        step=step,
        radar=radar,
        current_path=target_path
    )

    # Radar Lead Tracking (using road heading for corridor accuracy)
    lead_target = radar.track_lead_vehicle(
        ego_x, ego_y, ego_orient, surrounding_obstacles, step,
        target_offset=planner.target_offset,
        road_heading=ego_params["orientation"]
    )

    d_safe = cbf_solver.d_min + (ego_v * cbf_solver.tau)

    if lead_target is not None and not has_collided:
        target_x, target_y, target_v, target_id, x_local = lead_target
        h_val = cbf_solver.compute_barrier(x_local, ego_v)

        u_control = cbf_solver.solve(
            longitudinal_dist=x_local,
            v_ego=ego_v,
            v_target=target_v,
            v_des=DESIRED_SPEED,
            dt=scenario.dt,
        )
    else:
        h_val = None
        u_control = 0.5 * (DESIRED_SPEED - ego_v)

    # Dynamics Propagation
    if not has_collided:
        steering_angle = stanley_ctrl.compute_steering(
            ego_x, ego_y, ego_orient, ego_v, target_path
        )

        L = stanley_ctrl.wheelbase
        ego_v = max(0.0, ego_v + u_control * scenario.dt)
        ego_orient += (ego_v / L) * np.tan(steering_angle) * scenario.dt
        ego_x += ego_v * np.cos(ego_orient) * scenario.dt
        ego_y += ego_v * np.sin(ego_orient) * scenario.dt

    ego_poly, ego_corners = get_car_polygon(ego_x, ego_y, ego_orient, length=ego_l, width=ego_w)
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

        if not has_collided and ego_poly.intersects(obs_poly):
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
        ego_state=(ego_x, ego_y, ego_orient, ego_v, ego_l, ego_w),
        d_safe=d_safe,
        h_val=h_val,
        radar_range=radar.range_max,
        radar_fov_deg=radar.fov_deg,
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
create_gif_from_frames(frame_files, gif_path, scenario.dt)

for f in frame_files:
    f.unlink()
FRAMES_DIR.rmdir()

print(f"\n USA Simulation Complete! Output saved to: '{gif_path.name}'")
if has_collided:
    print(f"💥 Collision detected at Step {collision_step}.")
else:
    print("🛡️ SAFE! CBF Safety Control maintained nominal distance.")
