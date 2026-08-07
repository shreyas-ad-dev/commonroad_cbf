from pathlib import Path
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
            ax.plot(path[:, 0], path[:, 1], color="black", linestyle=(0, (1, 2)), linewidth=1.5, zorder=zorder)

def render_frame(scenario, planning_problem_set, ego_obstacle, ego_corners, 
                 surrounding_states, has_collided, step, num_steps, speed, frame_path):
    """Renders a single simulation frame and saves to disk."""
    fig, ax = plt.subplots(figsize=(14, 5))
    renderer = MPRenderer(ax=ax)

    scenario.lanelet_network.draw(renderer)
    planning_problem_set.draw(renderer)
    renderer.render()

    # Soften lanelet fill colors
    for patch in ax.patches:
        patch.set_zorder(1)
        if patch.get_facecolor() in [(1.0, 0.8, 0.0, 1.0), (1.0, 0.7568627450980392, 0.027450980392156862, 1.0)]:
            patch.set_alpha(0.35)

    draw_obstacle_trajectories(ax, [obs for obs, _ in surrounding_states], zorder=50)

    # Draw surrounding obstacles
    for obs, (corners, is_hit) in surrounding_states:
        obs_color = "#E67E22" if is_hit else "#1F77B4"
        ax.add_patch(patches.Polygon(corners, closed=True, facecolor=obs_color, edgecolor="black", linewidth=1.2, zorder=100))

    # Draw Ego Vehicle: GREEN (#00FF00) normally, RED (#FF0000) on collision
    ego_color = "#FF0000" if has_collided else "#00FF00"
    ax.add_patch(patches.Polygon(
        ego_corners, closed=True,
        facecolor=ego_color,
        edgecolor="yellow" if has_collided else "black",
        linewidth=2.0 if has_collided else 1.2,
        zorder=110
    ))

    # Custom Legend
    ax.plot([], [], color="#00FF00", marker="s", ls="", markersize=10, label=f"Ego Vehicle (Obs #{ego_obstacle.obstacle_id})")
    ax.plot([], [], color="#1F77B4", marker="s", ls="", markersize=10, label="Surrounding Vehicles")
    ax.plot([], [], color="black", linestyle=":", label="Prediction Trajectory")

    if has_collided:
        ax.plot([], [], color="#FF0000", marker="X", ls="", markersize=12, label="COLLISION FROZEN")

    leg = ax.legend(loc="upper right", framealpha=0.9)
    leg.set_zorder(200)

    ax.set_xlim(-130, -30)
    ax.set_ylim(-5, 20)

    title_text = f"ZAM_Zip-1_64 | Step {step}/{num_steps} (t={step * scenario.dt:.1f}s) | v={speed:.1f} m/s"
    if has_collided:
        title_text += " ⚠️ COLLISION FROZEN!"
    ax.set_title(title_text, color="red" if has_collided else "black", fontweight="bold" if has_collided else "normal")
    ax.grid(True)

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
