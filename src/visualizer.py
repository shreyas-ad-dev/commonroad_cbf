# src/visualizer.py

import matplotlib.pyplot as plt
import numpy as np
from commonroad.visualization.mp_renderer import MPRenderer
from matplotlib import patches
from shapely.geometry import LineString
from shapely.geometry import Polygon as ShapelyPolygon

from src.ego_state import EgoState
from src.sensor_suite import SensorSuite


def create_wedge_polygon(center, r, theta1_deg, theta2_deg, num_points=30):
    """
    Creates a Shapely Polygon representing a sensor wedge/cone.

    Args:
        center (np.ndarray | tuple[float, float]): (x, y) coordinates of the sensor origin.
        r (float): Detection range radius in meters.
        theta1_deg (float): Starting angle of the arc in degrees.
        theta2_deg (float): Ending angle of the arc in degrees.
        num_points (int, optional): Number of linear arc interpolation points. Defaults to 30.

    Returns:
        ShapelyPolygon: Polygon geometry representing the sensor field of view cone.
    """
    
    angles = np.radians(np.linspace(theta1_deg, theta2_deg, num_points))
    arc_points = [
        (center[0] + r * np.cos(a), center[1] + r * np.sin(a)) 
        for a in angles
    ]
    # Wedge is center point connected to arc points
    wedge_coords = [tuple(center)] + arc_points + [tuple(center)]
    return ShapelyPolygon(wedge_coords)

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
        if (
            hasattr(obs, 'prediction') 
            and obs.prediction is not None
            and hasattr(obs.prediction, 'trajectory')
            and obs.prediction.trajectory is not None
            ):
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
        sensor_suite: SensorSuite,
        d_safe,
        h_val,
        surrounding_states,
        has_collided,
        step,
        num_steps,
        frame_path,
        show_trajectories: bool = False,
        ):
    """
    Renders simulation frame with Radar/USS FOV wedges, CBF safety buffer, and ego tracking camera.

    Args:
        scenario (object): CommonRoad scenario instance containing lanelet network and metadata.
        planning_problem_set (object): CommonRoad planning problem set for goal region display.
        ego (EgoState): Current Ego vehicle state object.
        sensor_suite (SensorSuite): Sensor suite containing active radar and ultrasonic sensors.
        d_safe (float): Calculated safe longitudinal distance buffer in meters.
        h_val (float | None): Control Barrier Function barrier value h(x), if available.
        surrounding_states (list[tuple[object, np.ndarray, bool]]): List of tuples (obstacle, corners, is_hit) for surrounding traffic.
        has_collided (bool): Whether a collision has occurred at this step.
        step (int): Current simulation time step index.
        num_steps (int): Total number of simulation steps.
        frame_path (Path): File path where the rendered PNG frame image will be saved.
        show_trajectories (bool, optional): Whether to draw predicted paths for traffic. Defaults to False.
    """

    front_radar = sensor_suite.front_radar
    rear_radar = sensor_suite.rear_radar
    uss_left = sensor_suite.uss_left
    uss_right = sensor_suite.uss_right

    front_tracked_ids = sensor_suite.front_tracked_ids
    rear_tracked_ids = sensor_suite.rear_tracked_ids
    left_tracked_ids = sensor_suite.left_tracked_ids
    right_tracked_ids = sensor_suite.right_tracked_ids

    lead_target = sensor_suite.lead_target
    lead_target_id = lead_target[3] if lead_target is not None else None

    _, ax = plt.subplots(figsize=(12, 7))
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
    sensor_polygons = []


    # 2. Render Radar and USS FOV Cone
    front_t1, front_t2 = heading_deg - (front_radar.fov_deg / 2.0), heading_deg + (front_radar.fov_deg / 2.0)
    front_fov_wedge = patches.Wedge(
        center=front_pos,
        r=front_radar.range_max,
        theta1=front_t1,
        theta2=front_t2,
        facecolor="#00E5FF",
        alpha=0.22,
        edgecolor="#0099CC",
        linestyle="-",
        linewidth=1.0,
        zorder=70
    )
    ax.add_patch(front_fov_wedge)
    sensor_polygons.append((
        create_wedge_polygon(front_pos, front_radar.range_max, front_t1, front_t2),
        "#00E5FF", # Highlight color for front radar
        front_tracked_ids,
        front_pos
    ))

    if rear_radar is not None:
        rear_t1 = heading_deg + 180.0 - (rear_radar.fov_deg / 2.0)
        rear_t2 = heading_deg + 180.0 + (rear_radar.fov_deg / 2.0)
        rear_fov_wedge = patches.Wedge(
                center=rear_pos,
                r=rear_radar.range_max,
                theta1=rear_t1,
                theta2=rear_t2,
                facecolor="#AB47BC",
                alpha=0.22,
                edgecolor="#7B1FA2",
                linestyle="-",
                linewidth=1.0,
                zorder=70
        )
        ax.add_patch(rear_fov_wedge)
        sensor_polygons.append((
            create_wedge_polygon(rear_pos, rear_radar.range_max, rear_t1, rear_t2), 
            "#E040FB",  # Highlight color for rear radar
            rear_tracked_ids,
            rear_pos
        ))


    if uss_left is not None:
        # Left USS (+90 deg)
        left_t1, left_t2 = heading_deg + 90.0 - (uss_left.fov_deg / 2.0), heading_deg + 90.0 + (uss_left.fov_deg / 2.0)
        left_wedge = patches.Wedge(
                center=ego.position,
                r=uss_left.range_max,
                theta1=left_t1,
                theta2=left_t2,
                facecolor="#FFE082",
                alpha=0.60,
                edgecolor="#FFA000",
                linestyle="--",
                zorder=70
        )
        ax.add_patch(left_wedge)
        sensor_polygons.append((
            create_wedge_polygon(ego.position, uss_left.range_max, left_t1, left_t2), 
            "#FFB300",  # Highlight color for side USS
            left_tracked_ids,
            ego.position
        ))

            
        if uss_right is not None:
            # Right USS (-90 deg)
            right_t1, right_t2 = heading_deg - 90.0 - (uss_right.fov_deg / 2.0), heading_deg - 90.0 + (uss_right.fov_deg / 2.0)
            right_wedge = patches.Wedge(
                    center=ego.position,
                    r=uss_right.range_max,
                    theta1=right_t1,
                    theta2=right_t2,
                    facecolor="#FFE082",
                    alpha=0.60,
                    edgecolor="#FFA000",
                    linestyle="--",
                    zorder=70
                )
            ax.add_patch(right_wedge)
            sensor_polygons.append((
                create_wedge_polygon(ego.position, uss_right.range_max, right_t1, right_t2), 
                "#FFB300", # Highlight color for side USS
                right_tracked_ids,
                ego.position
            ))


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
        obs_color = "#E67E22" if is_hit else "#1F77B4"

        if lead_target_id is not None and obs.obstacle_id == lead_target_id:
            obs_state = obs.state_at_time(step)
            if obs_state is not None:
                pos = obs_state.position
                ax.plot(pos[0], pos[1], marker="X", markersize=10, color="red", zorder=105)

        obs_poly_shapely = ShapelyPolygon(corners)
        ax.add_patch(patches.Polygon(
            corners, closed=True, 
            facecolor=obs_color, edgecolor="black", 
            linewidth=1.0, zorder=100
        ))

        # Find matching tracked object near the ground-truth obstacle center
        obs_center = np.mean(corners, axis=0)
        for track in sensor_suite.tracked_objects:
            if np.linalg.norm(track.position - obs_center) < 3.0:
                ax.text(
                    obs_center[0], obs_center[1] + 2.0,
                    f"TRK #{track.track_id}",
                    color="white",
                    fontsize=8,
                    fontweight="bold",
                    ha="center",
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="#111111", alpha=0.8),
                    zorder=120
                )
                break

        for wedge_poly, highlight_color, tracked_ids, sensor_origin in sensor_polygons:
            if obs_id in tracked_ids and obs_poly_shapely.intersects(wedge_poly):
                
                # Extract counter-clockwise outer boundary points
                pts = np.array(obs_poly_shapely.exterior.coords)[:-1]
                num_pts = len(pts)
                visible_segments = []

                # Find segments whose outward normal points towards the sensor origin
                for i in range(num_pts):
                    p1 = pts[i]
                    p2 = pts[(i + 1) % num_pts]
                    edge = p2 - p1
                    
                    # Outward normal vector for CCW polygon: (dy, -dx)
                    normal = np.array([edge[1], -edge[0]])
                    vec_to_sensor = sensor_origin - p1

                    # Edge faces sensor if dot product is positive
                    if np.dot(normal, vec_to_sensor) > 0:
                        visible_segments.append(LineString([p1, p2]))

                # Intersect visible segments with sensor wedge
                for seg in visible_segments:
                    if seg.intersects(wedge_poly):
                        intersected = seg.intersection(wedge_poly)
                        if intersected.is_empty:
                            continue

                        lines = (
                            intersected.geoms 
                            if hasattr(intersected, 'geoms') 
                            else [intersected]
                        )

                        for line in lines:
                            if line.geom_type in ['LineString', 'LinearRing']:
                                line_coords = np.array(line.coords)
                                ax.plot(
                                    line_coords[:, 0], line_coords[:, 1],
                                    color=highlight_color,
                                    linewidth=3.0,
                                    solid_capstyle='round',
                                    zorder=105
                                )


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

    ax.plot([], [], color="#0099CC", linestyle="-", label=f"Front Radar ({front_radar.range_max:.0f}m, {front_radar.fov_deg:.0f}°)")
    
    if rear_radar is not None:
        ax.plot([], [], color="#7B1FA2", linestyle="-", label=f"Rear Radar ({rear_radar.range_max:.0f}m, {rear_radar.fov_deg:.0f}°)")

    if uss_left is not None:
        ax.plot([], [], color="#FFA000", linestyle="-", label=f"Ultrasonic Sensor ({uss_left.range_max:.0f}m, {uss_left.fov_deg:.0f}°)")

    if uss_right is not None:
        ax.plot([], [], color="#FFA000", linestyle="-", label=f"Ultrasonic Sensor ({uss_right.range_max:.0f}m, {uss_right.fov_deg:.0f}°)")


    if lead_target_id is not None:
        ax.plot([], [], color="red", marker="X", ls="", markersize=9, label="Lead Target Vehicle")

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

