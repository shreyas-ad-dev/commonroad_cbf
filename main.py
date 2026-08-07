from pathlib import Path
import numpy as np

from src.scenario_loader import load_scenario_and_ego
from src.vehicle_dynamics import get_car_polygon
from src.visualizer import render_frame, create_gif_from_frames
from src.cbf_solver import CBFQPSolver

# Path setup
PROJECT_ROOT = Path(__file__).resolve().parent
XML_FILE = PROJECT_ROOT / "scenarios" / "ZAM_Zip-1_64_T-1.xml"
FRAMES_DIR = PROJECT_ROOT / "frames"

# Clean frames directory
if FRAMES_DIR.exists():
    for f in FRAMES_DIR.glob("*.png"):
        f.unlink()
FRAMES_DIR.mkdir(exist_ok=True)

# 1. Load Scenario and Auto-Select Ego
scenario, planning_set, ego_obstacle, surrounding_obstacles = load_scenario_and_ego(XML_FILE)

print(f"🚗 Automatically Selected Ego Vehicle ID: {ego_obstacle.obstacle_id}")
print(f"   Initial Position: ({ego_obstacle.initial_state.position[0]:.2f}, {ego_obstacle.initial_state.position[1]:.2f})")

# 2. Simulation Setup
DESIRED_SPEED = 20.0  # m/s setpoint
num_steps = 40
frame_files = []

# Instantiate state variables for dynamic Ego propagation
ego_x = float(ego_obstacle.initial_state.position[0])
ego_y = float(ego_obstacle.initial_state.position[1])
ego_orient = float(getattr(ego_obstacle.initial_state, "orientation", 0.0))
ego_v = DESIRED_SPEED  # Start at high speed

# Instantiate CBF Safety Filter
cbf_solver = CBFQPSolver(gamma=1.2, d_min=6.0, tau=0.5, a_min=-8.0, a_max=2.0)

# Freeze state tracking
has_collided = False
collision_step = None
collided_obstacle_id = None
frozen_obs_states = {}

# Identify lead vehicle in ego's lane to track for safety filter
def get_lead_vehicle_state(step_idx):
    """Finds the vehicle directly ahead of Ego in the same lane."""
    lead_obs = None
    min_ahead_dist = float('inf')
    
    for obs in surrounding_obstacles:
        st = obs.state_at_time(step_idx)
        if st is None:
            continue
        ox, oy = st.position[0], st.position[1]
        
        # Check if ahead of ego longitudinally and in similar lateral lane (y within 3m)
        if ox > ego_x and abs(oy - ego_y) < 3.0:
            dist = ox - ego_x
            if dist < min_ahead_dist:
                min_ahead_dist = dist
                ov = getattr(st, 'velocity', 0.0)
                lead_obs = (ox, oy, ov)
                
    return lead_obs

# 3. Main Loop
for step in range(num_steps):
    # Determine target vehicle ahead
    lead_target = get_lead_vehicle_state(step)
    
    if lead_target is not None and not has_collided:
        target_x, target_y, target_v = lead_target
        # Solve CBF-QP to compute safe acceleration
        u_control = cbf_solver.solve(
            x_ego=ego_x, 
            v_ego=ego_v, 
            x_target=target_x, 
            v_target=target_v, 
            v_des=DESIRED_SPEED, 
            dt=scenario.dt
        )
    else:
        # Open road: no constraint, drive at desired acceleration (0 if at speed)
        u_control = 0.0

    # Propagate Ego Dynamics using calculated u_control
    if not has_collided:
        ego_v = max(0.0, ego_v + u_control * scenario.dt)  # Speed cannot be negative
        ego_x += ego_v * scenario.dt * np.cos(ego_orient)
        ego_y += ego_v * scenario.dt * np.sin(ego_orient)

    ego_l = getattr(ego_obstacle.obstacle_shape, 'length', 4.5)
    ego_w = getattr(ego_obstacle.obstacle_shape, 'width', 2.0)
    ego_poly, ego_corners = get_car_polygon(ego_x, ego_y, ego_orient, length=ego_l, width=ego_w)

    surrounding_render_states = []

    # Process surrounding traffic
    for obs in surrounding_obstacles:
        if has_collided and obs.obstacle_id == collided_obstacle_id:
            ox, oy, o_orient = frozen_obs_states[obs.obstacle_id]
        else:
            st = obs.state_at_time(step)
            if st is None:
                continue
            ox, oy = st.position[0], st.position[1]
            o_orient = getattr(st, 'orientation', 0.0)

        l = getattr(obs.obstacle_shape, 'length', 4.5)
        w = getattr(obs.obstacle_shape, 'width', 2.0)
        obs_poly, obs_corners = get_car_polygon(ox, oy, o_orient, length=l, width=w)

        # Detect impact
        if not has_collided and ego_poly.intersects(obs_poly):
            has_collided = True
            collision_step = step
            collided_obstacle_id = obs.obstacle_id
            frozen_obs_states[obs.obstacle_id] = (ox, oy, o_orient)

        is_hit_target = has_collided and (obs.obstacle_id == collided_obstacle_id)
        surrounding_render_states.append((obs, (obs_corners, is_hit_target)))

    # Render frame
    frame_path = FRAMES_DIR / f"frame_{step:02d}.png"
    render_frame(
        scenario, planning_set, ego_obstacle, ego_corners,
        surrounding_render_states, has_collided, step, num_steps,
        ego_v, frame_path
    )
    frame_files.append(frame_path)

# 4. Generate Output GIF & Clean Up
gif_path = PROJECT_ROOT / "simulation_cbf_qp.gif"
create_gif_from_frames(frame_files, gif_path, scenario.dt)

for f in frame_files:
    f.unlink()
FRAMES_DIR.rmdir()

print(f"\n✅ CBF-QP Simulation Complete! Output saved to: '{gif_path.name}'")
if has_collided:
    print(f"💥 Collision occurred at Step {collision_step}.")
else:
    print("🛡️ SAFE! Control Barrier Function successfully prevented the collision!")
