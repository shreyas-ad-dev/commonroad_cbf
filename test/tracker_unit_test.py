import sys
import numpy as np
import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.tracker import Detection, KalmanFilter2D, Track, TrackState


#--------------------------------------------------------
# Kalman tests
#--------------------------------------------------------
def test_kalman_filter_initialization():
    init_pos = np.array([10.0, 20.0])
    kf = KalmanFilter2D(init_pos=init_pos, dt=0.1)
    
    assert np.allclose(kf.x[:2], init_pos)
    assert np.allclose(kf.x[2:], [0.0, 0.0])
    assert kf.P.shape == (4, 4)
    assert kf.F.shape == (4, 4)
    assert kf.H.shape == (2, 4)
    assert kf.Q.shape == (4, 4)


def test_kalman_filter_predict_and_update():
    init_pos = np.array([0.0, 0.0])
    kf = KalmanFilter2D(init_pos=init_pos, dt=1.0)
    
    # Set explicit velocity to test constant velocity prediction
    kf.x = np.array([0.0, 0.0, 2.0, 3.0])
    
    x_pred = kf.predict()
    assert np.allclose(x_pred[:2], [2.0, 3.0])
    assert np.allclose(x_pred[2:], [2.0, 3.0])
    
    # Test update with a measurement
    z = np.array([2.1, 3.1])
    R = np.eye(2) * 0.1
    x_upd = kf.update(z, R)
    
    # Position should be adjusted towards the measurement
    assert 2.0 < x_upd[0] < 2.2
    assert 3.0 < x_upd[1] < 3.2


#--------------------------------------------------------
# Track tests
#--------------------------------------------------------
def test_track_lifecycle_confirmation():
    det = Detection(
        sensor_id="sensor_1",
        timestamp=0.0,
        z=np.array([5.0, 10.0]),
        obstacle_id=101,
        R=np.eye(2) * 0.05
    )
    
    track = Track(track_id=1, detection=det, dt=0.1, confirm_hits=2, max_age=3)
    
    assert track.track_id == 1
    assert track.state == TrackState.TENTATIVE
    assert track.hits == 1
    assert track.age == 1
    assert np.allclose(track.position, [5.0, 10.0])
    
    # Second hit to confirm track
    det2 = Detection(
        sensor_id="sensor_1",
        timestamp=0.1,
        z=np.array([5.2, 10.2]),
        obstacle_id=101,
        R=np.eye(2) * 0.05
    )
    track.predict()
    track.update(det2)
    
    assert track.track_id == 1
    assert track.state == TrackState.CONFIRMED
    assert track.hits == 2
    assert track.age == 2
    assert 5.0 < track.position[0] < 5.3
    assert 10.0 < track.position[1] < 10.3


def test_track_missed_and_deletion():
    det = Detection(
        sensor_id="sensor_1",
        timestamp=0.0,
        z=np.array([0.0, 0.0]),
        obstacle_id=102,
        R=np.eye(2) * 0.1
    )
    # confirm_hits=1 for immediate confirmation in this test
    track = Track(track_id=2, detection=det, dt=0.1, confirm_hits=1, max_age=2)
    track.update(det)
    assert track.state == TrackState.CONFIRMED
    
    # Miss 1
    track.predict()
    track.mark_missed()
    assert track.state == TrackState.CONFIRMED
    assert track.time_since_update == 1
    
    # Miss 2
    track.predict()
    track.mark_missed()
    assert track.state == TrackState.CONFIRMED
    
    # Miss 3 (exceeds max_age=2)
    track.predict()
    track.mark_missed()
    assert track.state == TrackState.DELETED


def test_predict_trajectory_isolates_live_state():
    det = Detection(
        sensor_id="sensor_1",
        timestamp=0.0,
        z=np.array([10.0, 20.0]),
        obstacle_id=103,
        R=np.eye(2) * 0.1
    )
    track = Track(track_id=3, detection=det, dt=0.5)
    track.kf.x = np.array([10.0, 20.0, 4.0, -2.0])  # Set explicit velocity
    
    horizon = 1.5  # 3 steps at dt=0.5
    traj = track.predict_trajectory(horizon)
    
    assert len(traj) == 3
    assert np.isclose(traj[0]["time_offset"], 0.5)
    assert np.isclose(traj[0]["x"], 12.0)
    assert np.isclose(traj[0]["y"], 19.0)
    
    assert np.isclose(traj[1]["time_offset"], 1.0)
    assert np.isclose(traj[1]["x"], 14.0)
    assert np.isclose(traj[1]["y"], 18.0)
    
    assert np.isclose(traj[2]["time_offset"], 1.5)
    assert np.isclose(traj[2]["x"], 16.0)
    assert np.isclose(traj[2]["y"], 17.0)
    
    # Verify live filter state remains strictly unaffected by trajectory rollouts
    assert np.allclose(track.kf.x[:2], [10.0, 20.0])
    assert np.allclose(track.kf.x[2:], [4.0, -2.0])
