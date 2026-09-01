#src/data_association.py

from typing import List, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment

from src.tracker import Track, Detection


def compute_cost_matrix(tracks: List[Track], detections: List[Detection], max_distance: float = 10.0) -> np.ndarray:
    """Computes the cost (distance) matrix between active tracks and new detections.
    
    Unfeasible pairs exceeding max_distance are assigned a high gating cost.
    """
    num_tracks = len(tracks)
    num_detections = len(detections)

    if num_tracks == 0 or num_detections == 0:
        return np.empty((num_tracks, num_detections))

    cost_matrix = np.full((num_tracks, num_detections), fill_value=1e5, dtype=np.float64)

    for i, track in enumerate(tracks):
        track_pos = track.position  # [px, py]
        for j, det in enumerate(detections):
            det_pos = det.z[:2]  # [x, y]
            dist = np.linalg.norm(track_pos - det_pos)

            # Gating check
            if dist <= max_distance:
                cost_matrix[i, j] = dist

    return cost_matrix


def associate_detections_to_tracks(
    tracks: List[Track],
    detections: List[Detection],
    max_distance: float = 5.0
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Performs optimal bipartite matching between predicted tracks and incoming detections.
    
    Returns:
        matches: List of tuples (track_index, detection_index)
        unmatched_tracks: List of track indices without matching detections
        unmatched_detections: List of detection indices without matching tracks
    """
    if len(tracks) == 0:
        return [], [], list(range(len(detections)))
    if len(detections) == 0:
        return [], list(range(len(tracks))), []

    cost_matrix = compute_cost_matrix(tracks, detections, max_distance=max_distance)

    # Solve Hungarian Algorithm
    track_indices, det_indices = linear_sum_assignment(cost_matrix)

    matches = []
    unmatched_tracks = list(range(len(tracks)))
    unmatched_detections = list(range(len(detections)))

    for track_idx, det_idx in zip(track_indices, det_indices):
        # Reject assignments that violate the spatial gating threshold
        if cost_matrix[track_idx, det_idx] >= max_distance:
            continue

        matches.append((track_idx, det_idx))
        if track_idx in unmatched_tracks:
            unmatched_tracks.remove(track_idx)
        if det_idx in unmatched_detections:
            unmatched_detections.remove(det_idx)

    return matches, unmatched_tracks, unmatched_detections
