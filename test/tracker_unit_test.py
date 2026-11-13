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


#--------------------------------------------------------
# Additional Kalman filter tests
#--------------------------------------------------------

def test_kalman_covariance_prediction():
    """Prediction covariance must follow F P F^T + Q."""
    kf = KalmanFilter2D(
        init_pos=np.array([0.0, 0.0]),
        dt=0.2
    )

    initial_P = kf.P.copy()
    expected_P = kf.F @ initial_P @ kf.F.T + kf.Q

    kf.predict()

    assert np.allclose(kf.P, expected_P)
    assert np.allclose(kf.P, kf.P.T)


def test_kalman_covariance_update_is_symmetric_and_reduces_uncertainty():
    kf = KalmanFilter2D(
        init_pos=np.array([0.0, 0.0]),
        dt=0.1
    )

    prior_position_uncertainty = np.trace(kf.P[:2, :2])
    kf.update(
        z=np.array([0.2, -0.1]),
        R=np.eye(2) * 0.05
    )

    assert np.all(np.isfinite(kf.P))
    assert np.allclose(kf.P, kf.P.T, atol=1e-10)
    assert np.trace(kf.P[:2, :2]) < prior_position_uncertainty


def test_kalman_higher_measurement_noise_produces_smaller_correction():
    measurement = np.array([10.0, 10.0])

    low_noise_filter = KalmanFilter2D(
        init_pos=np.array([0.0, 0.0]),
        dt=0.1
    )
    high_noise_filter = KalmanFilter2D(
        init_pos=np.array([0.0, 0.0]),
        dt=0.1
    )

    low_noise_filter.update(measurement, np.eye(2) * 0.01)
    high_noise_filter.update(measurement, np.eye(2) * 10.0)

    low_noise_correction = np.linalg.norm(low_noise_filter.x[:2])
    high_noise_correction = np.linalg.norm(high_noise_filter.x[:2])

    assert low_noise_correction > high_noise_correction


def test_kalman_estimates_velocity_from_successive_position_measurements():
    """Velocity should be inferred from a sequence of position-only observations."""
    dt = 1.0
    true_velocity = np.array([2.0, -1.0])
    kf = KalmanFilter2D(
        init_pos=np.array([0.0, 0.0]),
        dt=dt
    )

    measurement_noise = np.eye(2) * 0.01

    # Feed a constant-velocity trajectory; velocity is never assigned directly.
    for step in range(1, 31):
        kf.predict()
        measurement = true_velocity * step * dt
        kf.update(measurement, measurement_noise)

    assert np.allclose(kf.x[2:], true_velocity, atol=0.15)


@pytest.mark.parametrize("dt", [0.05, 0.1, 0.5, 1.0])
def test_kalman_prediction_respects_sampling_interval(dt):
    kf = KalmanFilter2D(
        init_pos=np.array([1.0, 2.0]),
        dt=dt
    )
    kf.x = np.array([1.0, 2.0, 4.0, -3.0])

    predicted = kf.predict()

    assert np.allclose(predicted[:2], [1.0 + 4.0 * dt, 2.0 - 3.0 * dt])
    assert np.allclose(predicted[2:], [4.0, -3.0])


def test_kalman_stationary_object_remains_stationary_without_measurements():
    kf = KalmanFilter2D(
        init_pos=np.array([3.0, -2.0]),
        dt=0.1
    )

    for _ in range(10):
        kf.predict()

    assert np.allclose(kf.x[:2], [3.0, -2.0])
    assert np.allclose(kf.x[2:], [0.0, 0.0])


#--------------------------------------------------------
# Additional track lifecycle tests
#--------------------------------------------------------

def make_detection(
    z=(0.0, 0.0),
    timestamp=0.0,
    obstacle_id=1,
    sensor_id="sensor_1",
    measurement_variance=0.1
):
    """Create a detection for tracker unit tests."""
    return Detection(
        sensor_id=sensor_id,
        timestamp=timestamp,
        z=np.asarray(z, dtype=float),
        obstacle_id=obstacle_id,
        R=np.eye(2) * measurement_variance
    )


def test_track_remains_tentative_below_confirmation_threshold():
    track = Track(
        track_id=10,
        detection=make_detection(),
        dt=0.1,
        confirm_hits=3
    )

    track.predict()
    track.update(make_detection(z=(0.1, 0.0), timestamp=0.1))

    assert track.hits == 2
    assert track.state == TrackState.TENTATIVE


def test_track_confirms_when_hit_threshold_is_reached():
    track = Track(
        track_id=11,
        detection=make_detection(),
        dt=0.1,
        confirm_hits=3
    )

    for hit in range(1, 3):
        track.predict()
        track.update(
            make_detection(z=(0.1 * hit, 0.0), timestamp=0.1 * hit)
        )

    assert track.hits == 3
    assert track.state == TrackState.CONFIRMED


def test_track_with_one_required_hit_confirms_on_first_update():
    """Current implementation checks confirmation inside update(), not __init__()."""
    track = Track(
        track_id=12,
        detection=make_detection(),
        dt=0.1,
        confirm_hits=1
    )

    # This captures the current lifecycle contract: construction alone leaves
    # the track tentative; the first update performs the confirmation check.
    assert track.state == TrackState.TENTATIVE

    track.update(make_detection(z=(0.0, 0.0), timestamp=0.1))

    assert track.state == TrackState.CONFIRMED
    assert track.hits == 2


def test_track_recovers_after_missed_detection():
    track = Track(
        track_id=13,
        detection=make_detection(),
        dt=0.1,
        confirm_hits=1,
        max_age=3
    )
    track.update(make_detection(z=(0.0, 0.0), timestamp=0.1))
    assert track.state == TrackState.CONFIRMED

    track.predict()
    track.mark_missed()
    assert track.time_since_update == 1

    track.predict()
    track.update(make_detection(z=(0.2, 0.0), timestamp=0.3))

    assert track.state == TrackState.CONFIRMED
    assert track.time_since_update == 0
    assert track.hits == 3


def test_track_age_increments_on_each_prediction():
    track = Track(
        track_id=14,
        detection=make_detection(),
        dt=0.1
    )

    assert track.age == 1

    for expected_age in range(2, 7):
        track.predict()
        assert track.age == expected_age


def test_tentative_track_is_deleted_after_max_age_is_exceeded():
    track = Track(
        track_id=15,
        detection=make_detection(),
        dt=0.1,
        confirm_hits=3,
        max_age=1
    )

    track.predict()
    track.mark_missed()
    assert track.state == TrackState.TENTATIVE

    track.predict()
    track.mark_missed()
    assert track.state == TrackState.DELETED


def test_deleted_track_is_not_revived_by_update():
    track = Track(
        track_id=16,
        detection=make_detection(),
        dt=0.1,
        confirm_hits=1,
        max_age=0
    )
    track.update(make_detection(z=(0.0, 0.0), timestamp=0.1))

    track.predict()
    track.mark_missed()
    assert track.state == TrackState.DELETED

    track.update(make_detection(z=(1.0, 1.0), timestamp=0.2))
    assert track.state == TrackState.DELETED


#--------------------------------------------------------
# Additional trajectory prediction tests
#--------------------------------------------------------

def test_predict_trajectory_preserves_velocity():
    track = Track(
        track_id=20,
        detection=make_detection(z=(0.0, 0.0)),
        dt=0.5
    )
    track.kf.x = np.array([0.0, 0.0, 3.0, -2.0])

    trajectory = track.predict_trajectory(2.0)

    assert len(trajectory) == 4
    for point in trajectory:
        assert np.isclose(point["vx"], 3.0)
        assert np.isclose(point["vy"], -2.0)


def test_predict_trajectory_propagates_position_covariance():
    track = Track(
        track_id=21,
        detection=make_detection(),
        dt=0.2
    )

    trajectory = track.predict_trajectory(0.6)

    assert len(trajectory) == 3
    covariances = [np.asarray(point["pos_cov"]) for point in trajectory]

    for covariance in covariances:
        assert covariance.shape == (2, 2)
        assert np.all(np.isfinite(covariance))
        assert np.allclose(covariance, covariance.T, atol=1e-10)

    # Under this constant-velocity model, uncertainty should not decrease
    # during an uncorrected prediction rollout.
    traces = [np.trace(covariance) for covariance in covariances]
    assert traces[1] >= traces[0]
    assert traces[2] >= traces[1]


def test_predict_trajectory_zero_horizon_returns_empty_list():
    track = Track(
        track_id=22,
        detection=make_detection(),
        dt=0.1
    )

    assert track.predict_trajectory(0.0) == []


def test_predict_trajectory_horizon_smaller_than_dt_rounds_up_to_one_step():
    track = Track(
        track_id=23,
        detection=make_detection(),
        dt=0.5
    )
    track.kf.x = np.array([0.0, 0.0, 2.0, 0.0])

    trajectory = track.predict_trajectory(0.1)

    assert len(trajectory) == 1
    assert np.isclose(trajectory[0]["time_offset"], 0.5)
    assert np.isclose(trajectory[0]["x"], 1.0)


def test_predict_trajectory_non_integer_horizon_rounds_up():
    track = Track(
        track_id=24,
        detection=make_detection(),
        dt=0.5
    )

    trajectory = track.predict_trajectory(1.2)

    assert len(trajectory) == 3
    assert np.isclose(trajectory[-1]["time_offset"], 1.5)


def test_repeated_trajectory_predictions_are_independent():
    track = Track(
        track_id=25,
        detection=make_detection(),
        dt=0.5
    )
    track.kf.x = np.array([1.0, 2.0, 2.0, 1.0])

    first_prediction = track.predict_trajectory(1.0)
    second_prediction = track.predict_trajectory(1.0)

    assert first_prediction == second_prediction
    assert np.allclose(track.kf.x, [1.0, 2.0, 2.0, 1.0])


def test_latest_trajectory_prediction_replaces_cached_prediction():
    track = Track(
        track_id=26,
        detection=make_detection(),
        dt=0.5
    )

    first_prediction = track.predict_trajectory(0.5)
    assert track._cached_obstacle_trajectory_prediction == first_prediction

    second_prediction = track.predict_trajectory(1.0)
    assert track._cached_obstacle_trajectory_prediction == second_prediction
    assert len(track._cached_obstacle_trajectory_prediction) == 2


#--------------------------------------------------------
# Serialization tests
#--------------------------------------------------------

def test_track_serialization_contains_expected_fields():
    detection = make_detection(
        z=(5.0, 7.0),
        obstacle_id=42,
        sensor_id="front_radar"
    )
    track = Track(
        track_id=30,
        detection=detection,
        dt=0.1
    )

    serialized = track.serialize()

    expected_keys = {
        "track_id",
        "cr_obs_id",
        "sensor_id",
        "state",
        "age",
        "hits",
        "confirm_hits",
        "time_since_update",
        "x",
        "y",
        "vx",
        "vy",
        "prediction"
    }

    assert set(serialized.keys()) == expected_keys
    assert serialized["track_id"] == 30
    assert serialized["cr_obs_id"] == 42
    assert serialized["sensor_id"] == "front_radar"
    assert serialized["state"] == TrackState.TENTATIVE.name
    assert serialized["age"] == 1
    assert serialized["hits"] == 1
    assert serialized["time_since_update"] == 0
    assert np.isclose(serialized["x"], 5.0)
    assert np.isclose(serialized["y"], 7.0)
    assert np.isclose(serialized["vx"], 0.0)
    assert np.isclose(serialized["vy"], 0.0)
    assert serialized["prediction"] == {}


def test_track_serialization_includes_latest_trajectory():
    track = Track(
        track_id=31,
        detection=make_detection(z=(1.0, 2.0)),
        dt=0.5
    )
    track.kf.x = np.array([1.0, 2.0, 2.0, 0.0])

    prediction = track.predict_trajectory(1.0)
    serialized = track.serialize()

    assert serialized["prediction"] == prediction
    assert len(serialized["prediction"]) == 2
    assert np.isclose(serialized["prediction"][-1]["x"], 3.0)
