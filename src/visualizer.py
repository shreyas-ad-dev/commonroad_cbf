from pathlib import Path
from src.vehicle_dynamics import get_car_polygon
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from commonroad.visualization.mp_renderer import MPRenderer

def draw_obstacle_trajectories(ax, obstacles, zorder=50):
    """Draws predicted dotted trajectory paths for surrounding traffic."""
    for obs in obstacles:
        positions = []
        if hasattr(obs, 'prediction') and obs.prediction is not None:
            if hasattr(obs.prediction, 'trajectory') and obs.prediction.trajectory is not None:
                positions = [s.position for s in obs.prediction.trajectory.state_list]
        if not positions:
            for t in range(100):
                st = obs.state_at_time(t)
                if st is not None:
                    positions.append(st.position)
                else:
                    break
        if len(positions) > 1:
            path = np.array(positions)
            ax.plot(path[:, 0], path[:, 1], color="black", linestyle=(0, (1, 2)), linewidth=1.2, zorder=zorder)

def render_frame(scenario, planning_problem_set, ego_state, d_safe, h_val, radar_range, radar_fov_deg, 
                 surrounding_states, has_collided, step, num_steps, frame_path, show_trajectories: bool = False, rear_radar_range: float = None, rear_radar_fov_deg: float = None):
    """
    Renders simulation frame with Radar FOV cone, dynamic CBF safety buffer, 
    optional trajectory prediction paths, and camera tracking centered on Ego.
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    renderer = MPRenderer(ax=ax)

    scenario.lanelet_network.draw(renderer)
    planning_problem_set.draw(renderer)
    renderer.render()

    # Dim yellow goal region patch intensity so it doesn't obscure vehicles
    for patch in ax.patches:
        patch.set_zorder(1)
        if patch.get_facecolor() in [(1.0, 0.8, 0.0, 1.0), (1.0, 0.7568627450980392, 0.027450980392156862, 1.0)]:
            patch.set_alpha(0.35)

    # 1. Optionally draw trajectory predictions for obstacles
    if show_trajectories:
        draw_obstacle_trajectories(ax, [obs for obs, _, _ in surrounding_states], zorder=50)

    ego_x, ego_y, ego_orient, ego_v, ego_l, ego_w = ego_state
    cos_a, sin_a = np.cos(ego_orient), np.sin(ego_orient)
    front_x = ego_x + (ego_l / 2.0) * cos_a
    front_y = ego_y + (ego_l / 2.0) * sin_a
    rear_x = ego_x - (ego_l / 2.0) * cos_a
    rear_y = ego_y - (ego_l / 2.0) * sin_a


    # 2. Render Radar FOV Cone
    front_fov_wedge = patches.Wedge(
        center=(front_x, front_y),
        r=radar_range,
        theta1=np.degrees(ego_orient) - (radar_fov_deg / 2.0),
        theta2=np.degrees(ego_orient) + (radar_fov_deg / 2.0),
        facecolor="#00E5FF",
        alpha=0.22,
        edgecolor="#0099CC",
        linestyle="-",
        linewidth=1.0,
        zorder=70
    )
    ax.add_patch(front_fov_wedge)

    if rear_radar_range is not None and rear_radar_fov_deg is not None:
        rear_fov_wedge = patches.Wedge(
                center=(rear_x, rear_y),
                r=rear_radar_range,
                theta1=np.degrees(ego_orient) + 180.0 - (rear_radar_fov_deg/ 2.0),
                theta2=np.degrees(ego_orient) + 180.0 + (rear_radar_fov_deg/ 2.0),
                facecolor="#AB47BC",
                alpha=0.22,
                edgecolor="#7B1FA2",
                linestyle="-",
                linewidth=1.0,
                zorder=70
        )
        ax.add_patch(rear_fov_wedge)

    # 3. Render CBF Safety Buffer Zone
    w2 = ego_w / 2.0
    local_buffer = np.array([
        [0.0, -w2],
        [d_safe, -w2],
        [d_safe, w2],
        [0.0, w2]
    ])
    rot_mat = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    world_buffer = local_buffer @ rot_mat.T + np.array([front_x, front_y])

    if h_val is None or h_val > 5.0:
        zone_color, zone_alpha = "#2ECC71", 0.25
    elif h_val > 0.0:
        zone_color, zone_alpha = "#FF6D00", 0.50
    else:
        zone_color, zone_alpha = "#E74C3C", 0.55

    ax.add_patch(patches.Polygon(
        world_buffer, closed=True, 
        facecolor=zone_color, edgecolor="black", 
        alpha=zone_alpha, linestyle="--", linewidth=1.5, zorder=80
    ))

    # 4. Render Surrounding Vehicles
    for obs, corners, is_hit in surrounding_states:
        obs_color = "#E67E22" if is_hit else "#1F77B4"
        ax.add_patch(patches.Polygon(
            corners, closed=True, 
            facecolor=obs_color, edgecolor="black", 
            linewidth=1.0, zorder=100
        ))

    # 5. Render Ego Vehicle
    _, ego_corners = get_car_polygon(ego_x, ego_y, ego_orient, length=ego_l, width=ego_w)
    
    ego_color = "#FF0000" if has_collided else "#00FF00"
    ax.add_patch(patches.Polygon(
        ego_corners, closed=True,
        facecolor=ego_color,
        edgecolor="yellow" if has_collided else "black",
        linewidth=2.0 if has_collided else 1.2,
        zorder=110
    ))

    # 6. Legend & Information

   # ax.plot([], [], color="#00FF00", marker="s", ls="", markersize=8, label="Ego Vehicle")
   # ax.plot([], [], color="#0099CC", linestyle="-", label=f"Radar FOV ({radar_range:.0f}m, {radar_fov_deg:.0f}°)")
   # ax.plot([], [], color=zone_color, linestyle="--", linewidth=2, label=f"CBF Buffer ({d_safe:.1f}m)")
   # 
    ax.plot([], [], color="#00FF00", marker="s", ls="", markersize=8, label="Ego Vehicle")
    ax.plot([], [], color="#0099CC", linestyle="-", label=f"Front Radar ({radar_range:.0f}m, {radar_fov_deg:.0f}°)")
    
    if rear_radar_range is not None and rear_radar_fov_deg is not None:
        ax.plot([], [], color="#7B1FA2", linestyle="-", label=f"Rear Radar ({rear_radar_range:.0f}m, {rear_radar_fov_deg:.0f}°)")

    ax.plot([], [], color=zone_color, linestyle="--", linewidth=2, label=f"CBF Buffer ({d_safe:.1f}m)")
    if show_trajectories:
        ax.plot([], [], color="black", linestyle=":", label="Obstacle Trajectory")

    if has_collided:
        ax.plot([], [], color="#FF0000", marker="X", ls="", markersize=10, label="COLLISION FROZEN")

    leg = ax.legend(loc="upper right", framealpha=0.85)
    leg.set_zorder(200)

    # 7. Camera Frame Tracking
    view_margin = 45.0
    ax.set_xlim(ego_x - view_margin, ego_x + view_margin)
    ax.set_ylim(ego_y - view_margin, ego_y + view_margin)
    ax.set_aspect('equal')

    h_str = f" | h(x)={h_val:.2f}" if h_val is not None else " | h(x)=N/A"
    title_text = f"{scenario.scenario_id} | Step {step}/{num_steps} (t={step * scenario.dt:.1f}s) | v={ego_v:.1f} m/s{h_str}"
    if has_collided:
        title_text += " ⚠️ COLLISION FROZEN!"
    
    ax.set_title(title_text, color="red" if has_collided else "black", fontweight="bold" if has_collided else "normal")
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.savefig(frame_path, dpi=100, bbox_inches='tight')
    plt.close('all')

def create_gif_from_frames(frame_files, output_gif_path, dt):
    """Stitches saved frame PNGs into an animated GIF."""
    images = [Image.open(f).copy() for f in frame_files]
    images[0].save(
        output_gif_path,
        save_all=True,
        append_images=images[1:],
        duration=int(dt * 1000),
        loop=0
    )
