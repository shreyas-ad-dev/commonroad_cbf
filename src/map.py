import logging
from collections import deque
from typing import Any

import numpy as np
from scipy.interpolate import interp1d

from src.ego_state import EgoState

logger = logging.getLogger(__name__)


class MapModule:
    """
    Manages high-level spatial topology, map querying, and path generation
    for CommonRoad scenarios.
    """

    def __init__(self, scenario: Any, planning_problem_set: Any | None = None):
        """
        Initializes the MapModule with scenario and optional goal planning information.

        Args:
            scenario: The CommonRoad scenario object.
            planning_problem_set: Optional PlanningProblemSet object containing goal definitions.
        """
        self.scenario = scenario
        self.planning_problem_set = planning_problem_set
        self.lanelet_network = scenario.lanelet_network

    def get_distance_to_next_merge(self, ego: EgoState, horizon_meters: float = 100.0) -> float:
        """
        Computes distance to the next incoming lane merge or merging lanelet boundary.
        """
        start_ids = self.lanelet_network.find_lanelet_by_position([ego.position])[0]
        if not start_ids:
            return float('inf')

        curr_lnet = self.lanelet_network.find_lanelet_by_id(start_ids[0])
        dist_accum = 0.0

        while curr_lnet and dist_accum < horizon_meters:
            # Check if current lanelet receives adjacent merge or merges into another
            if hasattr(curr_lnet, 'predecessor') and len(curr_lnet.predecessor) > 1:
                return max(0.0, dist_accum)

            # Estimate length of current lanelet segment
            verts = np.array(curr_lnet.center_vertices)
            seg_len = float(np.sum(np.hypot(np.diff(verts[:, 0]), np.diff(verts[:, 1]))))
            dist_accum += seg_len

            if curr_lnet.successor:
                curr_lnet = self.lanelet_network.find_lanelet_by_id(curr_lnet.successor[0])
            else:
                break

        return float('inf')

    def get_road_heading_at_position(self, position: np.ndarray | list[float]) -> float | None:
        """
        Computes the road heading angle from the centerline of the nearest lanelet.

        Args:
            position (np.ndarray | list[float]): 2D world position [x, y] in meters.

        Returns:
            float | None: Heading angle in radians within [-pi, pi], or None if no lanelet is found.
        """
        position = np.array(position)
        lanelet_ids = self.lanelet_network.find_lanelet_by_position([position])[0]

        if not lanelet_ids:
            return None

        primary_id = (
            lanelet_ids[0][0]
            if isinstance(lanelet_ids[0], (list, tuple))
            else lanelet_ids[0]
        )
        lanelet = self.lanelet_network.find_lanelet_by_id(primary_id)
        centerline = lanelet.center_vertices

        distances = np.linalg.norm(centerline - position, axis=1)
        idx = int(np.argmin(distances))

        if idx < len(centerline) - 1:
            pt1, pt2 = centerline[idx], centerline[idx + 1]
        else:
            pt1, pt2 = centerline[idx - 1], centerline[idx]

        dx = pt2[0] - pt1[0]
        dy = pt2[1] - pt1[1]

        return float(np.arctan2(dy, dx))

    def get_current_lane_width(self, ego: EgoState, default_width: float = 3.5) -> float:
        """
        Computes the local lane width surrounding the Ego vehicle's current position.

        Args:
            ego (EgoState): Current state of the Ego vehicle.
            default_width (float, optional): Fallback lane width in meters. Defaults to 3.5.

        Returns:
            float: Mean local lane width in meters.
        """
        lanelet_ids = self.lanelet_network.find_lanelet_by_position([ego.position])[0]

        if not lanelet_ids:
            return default_width

        lanelet = self.lanelet_network.find_lanelet_by_id(lanelet_ids[0])
        left_verts = np.array(lanelet.left_vertices)
        right_verts = np.array(lanelet.right_vertices)

        widths = np.hypot(
            left_verts[:, 0] - right_verts[:, 0],
            left_verts[:, 1] - right_verts[:, 1],
        )

        return float(np.mean(widths))

    def extract_target_lanelet_path(
        self,
        ego: EgoState,
        horizon_meters: float = 200.0,
        resampling_step: float = 0.5,
    ) -> np.ndarray:
        """
        Extracts a sequence of lanelets connecting Ego to the goal using BFS,
        chaining centerlines and densely resampling waypoints.

        Args:
            ego (EgoState): Current state of the Ego vehicle.
            horizon_meters (float, optional): Total longitudinal distance forward. Defaults to 200.0.
            resampling_step (float, optional): Distance delta between resampled points in meters. Defaults to 0.5.

        Returns:
            np.ndarray: An Nx2 numpy array of densely resampled [x, y] coordinates.
        """
        # 1. Locate starting lanelet
        start_ids = self.lanelet_network.find_lanelet_by_position([ego.position])[0]
        if not start_ids:
            min_d = float("inf")
            best_id = None
            for lnet in self.lanelet_network.lanelets:
                d = np.min(np.linalg.norm(lnet.center_vertices - ego.position, axis=1))
                if d < min_d:
                    min_d = d
                    best_id = lnet.lanelet_id
            start_ids = [best_id]

        start_id = start_ids[0]

        # 2. Extract Goal Lanelet IDs from Planning Problem Set
        goal_ids = set()
        if self.planning_problem_set is not None:
            try:
                p_prob_dict = self.planning_problem_set.planning_problem_dict
                p_prob = next(iter(p_prob_dict.values()))
                for st in p_prob.goal.state_list:
                    if hasattr(st, "position"):
                        pos_attr = st.position
                        if hasattr(pos_attr, "lanelet_id"):
                            goal_ids.update(pos_attr.lanelet_id)
                        elif hasattr(pos_attr, "center"):
                            g_lanelets = self.lanelet_network.find_lanelet_by_position([pos_attr.center])[0]
                            if g_lanelets:
                                goal_ids.update(g_lanelets)
            except (AttributeError, KeyError, IndexError, StopIteration) as e:
                logger.warning("Parsing goal failed: %s", e)

        # 3. BFS search for route to goal
        route_ids: list[int] = []
        if goal_ids:
            queue = deque([[start_id]])
            visited = {start_id}
            found_route = None

            while queue:
                path = queue.popleft()
                curr_id = path[-1]

                if curr_id in goal_ids:
                    found_route = path
                    break

                curr_lnet = self.lanelet_network.find_lanelet_by_id(curr_id)
                if curr_lnet and curr_lnet.successor:
                    for succ in curr_lnet.successor:
                        if succ not in visited:
                            visited.add(succ)
                            queue.append(path + [succ])

            if found_route:
                route_ids = found_route

        # Fallback: Follow default successor chain if no goal route is found
        if not route_ids:
            curr = self.lanelet_network.find_lanelet_by_id(start_id)
            route_ids = [start_id]
            while curr and curr.successor:
                next_id = curr.successor[0]
                route_ids.append(next_id)
                curr = self.lanelet_network.find_lanelet_by_id(next_id)

        # 4. Chain centerline vertices along the route
        all_verts = []
        tot_len = 0.0

        for idx, lid in enumerate(route_ids):
            lnet = self.lanelet_network.find_lanelet_by_id(lid)
            verts = np.array(lnet.center_vertices)
            if idx > 0:
                verts = verts[1:]  # Avoid duplicate boundary point
            all_verts.append(verts)

            tot_len += float(np.sum(np.hypot(np.diff(verts[:, 0]), np.diff(verts[:, 1]))))
            if tot_len >= horizon_meters:
                break

        center_verts = np.vstack(all_verts)

        # 5. Resample densely
        distances = np.zeros(len(center_verts))
        distances[1:] = np.cumsum(np.hypot(np.diff(center_verts[:, 0]), np.diff(center_verts[:, 1])))

        unique_indices = np.where(np.diff(distances, prepend=-1.0) > 1e-5)[0]
        distances = distances[unique_indices]
        center_verts = center_verts[unique_indices]

        s_dense = np.arange(0, distances[-1], resampling_step)
        interp_x = interp1d(distances, center_verts[:, 0], kind="linear")
        interp_y = interp1d(distances, center_verts[:, 1], kind="linear")

        return np.column_stack((interp_x(s_dense), interp_y(s_dense)))


# -----------------------------------------------------------------------------
# Backward-Compatible Functional Wrappers
# -----------------------------------------------------------------------------
def get_road_heading_at_position(
    scenario: Any, position: np.ndarray | list[float]
) -> float | None:
    map_mod = MapModule(scenario=scenario)
    return map_mod.get_road_heading_at_position(position)


def get_current_lane_width(
    scenario: Any, ego: EgoState, default_width: float = 3.5
) -> float:
    map_mod = MapModule(scenario=scenario)
    return map_mod.get_current_lane_width(ego, default_width=default_width)


def extract_target_lanelet_path(
    scenario: Any,
    ego: EgoState,
    planning_problem_set: Any | None = None,
    horizon_meters: float = 200.0,
) -> np.ndarray:
    map_mod = MapModule(scenario=scenario, planning_problem_set=planning_problem_set)
    return map_mod.extract_target_lanelet_path(ego=ego, horizon_meters=horizon_meters)
