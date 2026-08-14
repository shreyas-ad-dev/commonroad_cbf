from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from commonroad.visualization.mp_renderer import MPRenderer
from src.ego_state import EgoState

def draw_obstacle_trajectories(
        ax,
        obstacles,
        zorder=50):
    """
    Draws predicted dotted trajectory paths for surrounding traffic obstacles.

    Args:
        ax (plt.Axes): Matplotlib axes instance to draw trajectories on.
        obstacles (list): List of obstacle objects to project trajectories for.
        zorder (int, optional): Drawing layer z-order index for matplotlib. Defaults to 50.
    """

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

def render_frame(
        scenario,
        planning_problem_set,
        ego: EgoState,
        d_safe,
        h_val,
        radar_range,
        radar_fov_deg, 
        surrounding_states,
        has_collided,
        step,
        num_steps,
        frame_path,
        show_trajectories: bool = False,
        rear_radar_range: float = None,
        rear_radar_fov_deg: float = None,
        front_tracked_ids: set = None,
        rear_tracked_ids: set = None,
        uss_range: float = None,
        uss_fov_deg: float = None,
        left_tracked_ids: set = None,
        right_tracked_ids: set = None
        ):
    """
    Renders simulation frame with Radar/USS FOV wedges, CBF safety buffer, and ego tracking camera.

    Args:
        scenario: CommonRoad scenario instance containing lanelet network and metadata.
        planning_problem_set: CommonRoad planning problem set for goal region display.
        ego (EgoState): Current Ego vehicle state object.
        d_safe (float): Calculated safe longitudinal distance buffer in meters.
        h_val (float | None): Control Barrier Function barrier value h(x), if available.
        radar_range (float): Maximum range of front radar in meters.
        radar_fov_deg (float): Field of view of front radar in degrees.
        surrounding_states (list): List of tuples (obstacle, corners, is_hit) for surrounding traffic.
        has_collided (bool): Whether a collision has occurred at this step.
        step (int): Current simulation time step index.
        num_steps (int): Total number of simulation steps.
        frame_path (Path): File path where the rendered PNG frame image will be saved.
        show_trajectories (bool, optional): Whether to draw predicted paths for traffic. Defaults to False.
        rear_radar_range (float | None, optional): Range of rear radar in meters. Defaults to None.
        rear_radar_fov_deg (float | None, optional): FOV of rear radar in degrees. Defaults to None.
        front_tracked_ids (set | None, optional): Set of obstacle IDs detected by front radar. Defaults to None.
        rear_tracked_ids (set | None, optional): Set of obstacle IDs detected by rear radar. Defaults to None.
        uss_range (float | None, optional): Range of ultrasonic side sensors in meters. Defaults to None.
        uss_fov_deg (float | None, optional): FOV of ultrasonic side sensors in degrees. Defaults to None.
        left_tracked_ids (set | None, optional): Set of obstacle IDs detected by left USS. Defaults to None.
        right_tracked_ids (set | None, optional): Set of obstacle IDs detected by right USS. Defaults to None.
    """

    front_tracked_ids = front_tracked_ids or set()
    rear_tracked_ids = rear_tracked_ids or set()
    left_tracked_ids = left_tracked_ids or set()
    right_tracked_ids = right_tracked_ids or set()

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

    front_pos = ego.position + (ego.length / 2.0) * ego.heading_vector
    rear_pos = ego.position - (ego.length / 2.0) * ego. heading_vector
    heading_deg = np.degrees(ego.orientation)


    # 2. Render Radar and USS FOV Cone
    front_fov_wedge = patches.Wedge(
        center=front_pos,
        r=radar_range,
        theta1=heading_deg - (radar_fov_deg / 2.0),
        theta2=heading_deg + (radar_fov_deg / 2.0),
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
                center=rear_pos,
                r=rear_radar_range,
                theta1=heading_deg + 180.0 - (rear_radar_fov_deg/ 2.0),
                theta2=heading_deg + 180.0 + (rear_radar_fov_deg/ 2.0),
                facecolor="#AB47BC",
                alpha=0.22,
                edgecolor="#7B1FA2",
                linestyle="-",
                linewidth=1.0,
                zorder=70
        )
        ax.add_patch(rear_fov_wedge)

    if uss_range is not None and uss_fov_deg is not None:
        # Left USS (+90 deg)
        left_wedge = patches.Wedge(
                center=ego.position,
                r=uss_range,
                theta1=heading_deg + 90.0 - (uss_fov_deg / 2.0),
                theta2=heading_deg + 90.0 + (uss_fov_deg / 2.0),
                facecolor="#FFE082",
                alpha=0.60,
                edgecolor="#FFA000",
                linestyle="--",
                zorder=70
        )
        ax.add_patch(left_wedge)
            
        # Right USS (-90 deg)
        right_wedge = patches.Wedge(
                center=ego.position,
                r=uss_range,
                theta1=heading_deg - 90.0 - (uss_fov_deg / 2.0),
                theta2=heading_deg - 90.0 + (uss_fov_deg / 2.0),
                facecolor="#FFE082",
                alpha=0.60,
                edgecolor="#FFA000",
                linestyle="--",
                zorder=70
        )
        ax.add_patch(right_wedge)

    # 3. Render CBF Safety Buffer Zone
    w2 = ego.width / 2.0
    local_buffer = np.array([
        [0.0, -w2],
        [d_safe, -w2],
        [d_safe, w2],
        [0.0, w2]
    ])
    world_buffer = front_pos + np.outer(local_buffer[:,0], ego.heading_vector) + np.outer(local_buffer[:,1], ego.normal_vector)

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
        obs_id = obs.obstacle_id
        in_front = obs_id in front_tracked_ids
        in_rear = obs_id in rear_tracked_ids
        in_side = (obs_id in left_tracked_ids) or (obs_id in right_tracked_ids)

        obs_color = "#E67E22" if is_hit else "#1F77B4"

        if in_front:
            edge_color = "#00E5FF" # Cyan for Front Radar
            lw = 2.5
        elif in_rear:
            edge_color = "#E040FB" # Magenta for Rear Radar
            lw = 2.5
        elif in_side:
            edge_color = "#FFB300" # Amber/Gold for SIde USS
            lw = 2.5
        else:
            edge_color = "black"
            lw = 1.0

        ax.add_patch(patches.Polygon(
            corners, closed=True, 
            facecolor=obs_color, edgecolor=edge_color, 
            linewidth=lw, zorder=100
        ))

    # 5. Render Ego Vehicle
    
    ego_color = "#FF0000" if has_collided else "#00FF00"
    ax.add_patch(patches.Polygon(
        ego.corners, closed=True,
        facecolor=ego_color,
        edgecolor="yellow" if has_collided else "black",
        linewidth=2.0 if has_collided else 1.2,
        zorder=110
    ))

    # 6. Legend & Information

    ax.plot([], [], color="#00FF00", marker="s", ls="", markersize=8, label="Ego Vehicle")

    ax.plot([], [], color=zone_color, linestyle="--", linewidth=1, label=f"CBF Buffer ({d_safe:.1f}m)")

    ax.plot([], [], color="#0099CC", linestyle="-", label=f"Front Radar ({radar_range:.0f}m, {radar_fov_deg:.0f}°)")
    
    if rear_radar_range is not None and rear_radar_fov_deg is not None:
        ax.plot([], [], color="#7B1FA2", linestyle="-", label=f"Rear Radar ({rear_radar_range:.0f}m, {rear_radar_fov_deg:.0f}°)")
    if uss_range is not None and uss_fov_deg is not None:
        ax.plot([], [], color="#FFA000", linestyle="-", label=f"Ultrasonic Sensor ({uss_range:.0f}m, {uss_fov_deg:.0f}°)")

    if front_tracked_ids:
        ax.plot([], [], color="#00E5FF", linestyle="-", linewidth=2.5, label="Front Tracked Vehicle")

    if rear_tracked_ids:
        ax.plot([], [], color="#E040FB", linestyle="-", linewidth=2.5, label="Rear Tracked Vehicle")

    if left_tracked_ids or right_tracked_ids:
        ax.plot([], [], color="#FFB300", linestyle="-", linewidth=2.5, label="Side USS Tracked")
       
    if show_trajectories:
        ax.plot([], [], color="black", linestyle=":", label="Obstacle Trajectory")

    if has_collided:
        ax.plot([], [], color="#FF0000", marker="X", ls="", markersize=10, label="COLLISION FROZEN")

    leg = ax.legend(loc="upper right", framealpha=0.85)
    leg.set_zorder(200)

    # 7. Camera Frame Tracking
    view_margin = 45.0
    ax.set_xlim(ego.x - view_margin, ego.x + view_margin)
    ax.set_ylim(ego.y - view_margin, ego.y + view_margin)
    ax.set_aspect('equal')

    h_str = f" | h(x)={h_val:.2f}" if h_val is not None else " | h(x)=N/A"
    title_text = f"{scenario.scenario_id} | Step {step}/{num_steps} (t={step * scenario.dt:.1f}s) | v={ego.velocity:.1f} m/s{h_str}"
    if has_collided:
        title_text += " ⚠️ COLLISION FROZEN!"
    
    ax.set_title(title_text, color="red" if has_collided else "black", fontweight="bold" if has_collided else "normal")
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.savefig(frame_path, dpi=100, bbox_inches='tight')
    plt.close('all')

