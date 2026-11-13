import sys
import numpy as np
import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.config import VEHICLE
from src.cbf_solver import CBFQPSolver, CVXPY_AVAILABLE
from src.ego_state import EgoState
from src.tracker import Detection, Track


def test_compute_barrier():
    solver = CBFQPSolver(d_min=2.0, tau=1.5)
    # h = longitudinal_dist - (d_min + v_ego * tau)
    # v_ego = 10.0, d_min = 2.0, tau = 1.5 -> d_safe = 2.0 + 15.0 = 20.0
    h = solver.compute_barrier(longitudinal_dist=25.0, v_ego=10.0)
    assert np.isclose(h, 8.0)
    assert np.isclose(solver._cached_h_val, 8.0)


def test_solve_analytical_nominal_for_u_nom():
    solver = CBFQPSolver(a_min=-4.0, a_max=2.0)
    # When far away or safe, u_nom = 0.5 * (v_des - v_ego)
    u = solver.solve(
        longitudinal_dist=50.0,
        v_ego=10.0,
        v_target=10.0,
        v_des=15.0,
        dt=0.1
    )
    # max_allowed_accel calculated from v_rel is around 40
    # u_nom = 0.5 * (15 - 10) = 2.5, clipped to a_max = 2.0
    assert solver.tau == 0.5
    assert np.isclose(u, 2.0)

def test_solve_analytical_nominal_for_a_max():
    solver = CBFQPSolver(a_min=-4.0, a_max=4.0)
    # When far away or safe, u_nom = 0.5 * (v_des - v_ego)
    u = solver.solve(
        longitudinal_dist=50.0,
        v_ego=10.0,
        v_target=10.0,
        v_des=15.0,
        dt=0.1
    )
    # max_allowed_accel calculated from v_rel is around 40
    # u_nom = 0.5 * (15 - 10) = 2.5, which gets passed since it is less than a_max 
    assert solver.tau == 0.5
    assert np.isclose(u, 2.5)

def test_solve_analytical_with_tau_zero_for_u_nom():
    solver = CBFQPSolver(tau=0,a_min=-4.0, a_max=2.0)
    print("\n tau zero u nom")
    # When far away or safe, u_nom = 0.5 * (v_des - v_ego)
    u = solver.solve(
        longitudinal_dist=50.0,
        v_ego=10.0,
        v_target=10.0,
        v_des=15.0,
        dt=0.1
    )
    # u_nom = 0.5 * (15 - 10) = 2.5, clipped to a_max = 2.0
    assert np.isclose(u, 2.0)

def test_solve_analytical_with_tau_zereo_for_a_max():
    solver = CBFQPSolver(tau=0, a_min=-4.0, a_max=4.0)
    # When far away or safe, u_nom = 0.5 * (v_des - v_ego)
    u = solver.solve(
        longitudinal_dist=50.0,
        v_ego=10.0,
        v_target=10.0,
        v_des=15.0,
        dt=0.1
    )
    # u_nom = 0.5 * (15 - 10) = 2.5, which gets passed since it is less than a_max 
    assert np.isclose(u, 2.5)

def test_solve_analytical_braking_safety():
    solver = CBFQPSolver(gamma=1.0, d_min=2.0, tau=1.0, a_min=-2.0, a_max=2.0)
    # Force an unsafe/tight barrier condition where max_allowed_accel is low or negative
    # h_val = 1.0, v_rel = -5.0 (ego is faster than lead)
    u = solver.solve(
        longitudinal_dist=4.0,
        v_ego=15.0,
        v_target=10.0,
        v_des=15.0,
        dt=0.1
    )
    assert -2.0 <= u < 0.0  # Should trigger braking


@pytest.mark.skipif(not CVXPY_AVAILABLE, reason="cvxpy is not installed")
def test_solve_cvxpy_mode():
    solver = CBFQPSolver(use_cvxpy=True, a_min=-5.0, a_max=2.0)
    u = solver.solve(
        longitudinal_dist=30.0,
        v_ego=10.0,
        v_target=10.0,
        v_des=15.0,
        dt=0.1
    )
    assert isinstance(u, float)
    assert solver.tau == 0.5
    assert solver.use_cvxpy is True
    assert -5.0 <= u <= 2.0


def test_solve_from_track_none_lead():
    solver = CBFQPSolver(a_min=-3.0, a_max=3.0)
    ego = EgoState(x=0.0,
                   y=0.0,
                   orientation=0.0,
                   velocity=5.0,
                   length=VEHICLE.DEFAULT_LENGTH,
                   width=VEHICLE.DEFAULT_WIDTH,
                   wheelbase=VEHICLE.DEFAULT_WHEELBASE
                   )
    
    # When lead_track is None, falls back to nominal acceleration towards v_des
    u = solver.solve_from_track(ego=ego, lead_track=None, v_des=10.0, dt=0.1)
    assert np.isclose(u, 2.5)  # 0.5 * (10.0 - 5.0) = 2.5


def test_solve_from_track_with_valid_track():
    solver = CBFQPSolver(a_min=-4.0, a_max=2.0)
    ego = EgoState(x=0.0,
                   y=0.0,
                   orientation=0.0,
                   velocity=10.0,
                   length=VEHICLE.DEFAULT_LENGTH,
                   width=VEHICLE.DEFAULT_WIDTH,
                   wheelbase=VEHICLE.DEFAULT_WHEELBASE
                   )
    
    det = Detection(
        sensor_id="cam",
        timestamp=0.0,
        z=np.array([25.0, 0.0]),
        obstacle_id=1,
        R=np.eye(2)
    )
    track = Track(track_id=1, detection=det)
    track.kf.x = np.array([25.0, 0.0, 8.0, 0.0])  # lead position & velocity
    
    u = solver.solve_from_track(ego=ego, lead_track=track, v_des=15.0, dt=0.1)
    assert isinstance(u, float)
    assert -4.0 <= u <= 2.0

@pytest.mark.skipif(not CVXPY_AVAILABLE, reason="cvxpy is not installed")
def test_solve_cvxpy_infeasible_fallback():
    """Test that CVXPY solver gracefully falls back to a_min when constraints are infeasible."""
    solver = CBFQPSolver(use_cvxpy=True, a_min=-3.0, a_max=2.0, gamma=10.0, d_min=10.0)
    
    # Create an impossible/severely unsafe scenario where maintaining feasibility 
    # under harsh class-K gains forces an infeasible optimization status.
    u = solver.solve(
        longitudinal_dist=0.5,  # Extremely close (violates safety barrier heavily)
        v_ego=20.0,             # High ego speed
        v_target=0.0,           # Stationary target
        v_des=20.0,
        dt=0.1
    )
    # Should safely return a_min or a valid bounded float without throwing exceptions
    assert isinstance(u, float)
    assert u == solver.a_min


@pytest.mark.skipif(not CVXPY_AVAILABLE, reason="cvxpy is not installed")
def test_solve_cvxpy_suboptimal_or_inaccurate():
    """Test CVXPY behavior under tight constraints allowing suboptimal/inaccurate solutions."""
    solver = CBFQPSolver(use_cvxpy=True, a_min=-5.0, a_max=3.0)
    u = solver.solve(
        longitudinal_dist=15.0,
        v_ego=12.0,
        v_target=10.0,
        v_des=18.0,
        dt=0.1
    )
    assert isinstance(u, float)
    assert -5.0 <= u <= 3.0


def test_analytical_acceleration_clipping_bounds():
    """Test that analytical solver correctly clips extreme upper and lower acceleration bounds."""
    # Test upper bound clipping (a_max)
    solver_high = CBFQPSolver(a_min=-2.0, a_max=1.5, tau=0.5)
    u_high = solver_high.solve(
        longitudinal_dist=100.0,  # Very safe, large h_val
        v_ego=5.0,
        v_target=10.0,
        v_des=25.0,              # High desired speed -> high u_nom
        dt=0.1
    )
    assert np.isclose(u_high, solver_high.a_max)

    # Test lower bound clipping (a_min / hard braking)
    solver_low = CBFQPSolver(a_min=-3.5, a_max=2.0, gamma=5.0, tau=1.0)
    u_low = solver_low.solve(
        longitudinal_dist=2.0,   # Unsafe
        v_ego=25.0,              # Fast ego
        v_target=5.0,            # Slow target
        v_des=25.0,
        dt=0.1
    )
    assert np.isclose(u_low, solver_low.a_min)

"""Additional tests for src/cbf_solver.py.

Place this file beside the existing cbf_solver_unit_test.py in the tests/
directory. These tests supplement rather than replace the original suite.

The tests distinguish:
- behaviours the current implementation should satisfy;
- known design/implementation gaps, marked xfail so they are visible without
  making the baseline suite fail unexpectedly.

Run from the repository root:
    pytest -q tests/test_cbf_solver_comprehensive.py
"""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import VEHICLE
from src.cbf_solver import CBFQPSolver, CVXPY_AVAILABLE
from src.ego_state import EgoState
from src.tracker import Detection, Track


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_ego(x=0.0, y=0.0, orientation=0.0, velocity=10.0):
    return EgoState(
        x=x,
        y=y,
        orientation=orientation,
        velocity=velocity,
        length=VEHICLE.DEFAULT_LENGTH,
        width=VEHICLE.DEFAULT_WIDTH,
        wheelbase=VEHICLE.DEFAULT_WHEELBASE,
    )


def make_track(x, y=0.0, vx=0.0, vy=0.0, track_id=1):
    detection = Detection(
        sensor_id="unit-test",
        timestamp=0.0,
        z=np.array([x, y], dtype=float),
        obstacle_id=track_id,
        R=np.eye(2),
    )
    track = Track(track_id=track_id, detection=detection)
    track.kf.x = np.array([x, y, vx, vy], dtype=float)
    return track


# ---------------------------------------------------------------------------
# Barrier function: boundaries, signs, and parameter effects
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "distance, speed, d_min, tau, expected",
    [
        (20.0, 10.0, 2.0, 1.5, 3.0),
        (17.0, 10.0, 2.0, 1.5, 0.0),   # exactly on boundary
        (16.0, 10.0, 2.0, 1.5, -1.0),  # inside boundary
        (5.0, 0.0, 2.0, 1.5, 3.0),     # stationary ego
        (5.0, 10.0, 2.0, 0.0, 3.0),    # zero time headway
    ],
)
def test_compute_barrier_boundaries_and_sign(
    distance, speed, d_min, tau, expected
):
    solver = CBFQPSolver(d_min=d_min, tau=tau)
    assert solver.compute_barrier(distance, speed) == pytest.approx(expected)
    assert solver._cached_h_val == pytest.approx(expected)


def test_compute_barrier_updates_cache_on_every_call():
    solver = CBFQPSolver(d_min=2.0, tau=1.0)
    first = solver.compute_barrier(20.0, 5.0)
    second = solver.compute_barrier(8.0, 5.0)
    assert first == pytest.approx(13.0)
    assert second == pytest.approx(1.0)
    assert solver._cached_h_val == pytest.approx(second)


# ---------------------------------------------------------------------------
# Analytical solver: exact constraint, saturation, and relative-speed cases
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "h, v_ego, v_target, u_nom, gamma, tau, a_min, a_max",
    [
        # CBF constraint is inactive; nominal command is retained.
        (20.0, 10.0, 10.0, 1.0, 1.0, 1.0, -4.0, 3.0),
        # Lead vehicle is slower; constraint limits positive acceleration.
        (1.0, 12.0, 10.0, 2.0, 1.0, 1.0, -5.0, 3.0),
        # Faster lead vehicle allows nominal acceleration.
        (2.0, 10.0, 15.0, 1.5, 1.0, 1.0, -4.0, 3.0),
        # Equal speeds and zero barrier.
        (0.0, 10.0, 10.0, 1.0, 2.0, 1.0, -4.0, 3.0),
    ],
)
def test_analytical_solution_satisfies_cbf_inequality_when_feasible(
    h, v_ego, v_target, u_nom, gamma, tau, a_min, a_max
):
    solver = CBFQPSolver(
        gamma=gamma, tau=tau, a_min=a_min, a_max=a_max
    )
    u = solver._solve_analytical(h, v_ego, v_target, u_nom)

    assert a_min <= u <= a_max
    # The scalar continuous-time CBF condition implemented by this solver:
    # u*tau <= (v_target - v_ego) + gamma*h.
    rhs = (v_target - v_ego) + gamma * h
    if rhs / tau >= a_min:
        assert u * tau <= rhs + 1e-9


@pytest.mark.parametrize(
    "u_nom, expected",
    [
        (10.0, 2.0),   # upper saturation
        (-10.0, -3.0), # lower saturation
        (0.75, 0.75),  # no saturation
    ],
)
def test_analytical_solver_respects_acceleration_bounds(u_nom, expected):
    solver = CBFQPSolver(a_min=-3.0, a_max=2.0, tau=0.0)
    result = solver._solve_analytical(
        h_val=100.0, v_ego=10.0, v_target=10.0, u_nom=u_nom
    )
    assert result == pytest.approx(expected)


def test_analytical_solver_brakes_when_cbf_upper_bound_is_negative():
    solver = CBFQPSolver(
        gamma=1.0, tau=1.0, a_min=-5.0, a_max=3.0
    )
    # v_rel + gamma*h = -4 + 1 = -3, so u must be <= -3.
    result = solver._solve_analytical(
        h_val=1.0, v_ego=14.0, v_target=10.0, u_nom=2.0
    )
    assert result == pytest.approx(-3.0)


def test_analytical_solver_uses_a_min_when_constraint_is_more_restrictive_than_actuator():
    solver = CBFQPSolver(
        gamma=1.0, tau=1.0, a_min=-2.0, a_max=2.0
    )
    # The CBF upper bound is -10, below the strongest available braking.
    result = solver._solve_analytical(
        h_val=0.0, v_ego=20.0, v_target=10.0, u_nom=0.0
    )
    assert result == pytest.approx(-2.0)
    # This is an actuator-limited unsafe case, not a satisfied CBF constraint.
    assert result > -10.0


def test_analytical_solver_tau_zero_bypasses_cbf_acceleration_bound():
    solver = CBFQPSolver(tau=0.0, a_min=-4.0, a_max=2.0)
    result = solver._solve_analytical(
        h_val=-100.0, v_ego=20.0, v_target=0.0, u_nom=1.0
    )
    assert result == pytest.approx(1.0)


def test_solve_updates_barrier_cache():
    solver = CBFQPSolver(tau=1.0)
    solver.solve(
        longitudinal_dist=30.0,
        v_ego=10.0,
        v_target=10.0,
        v_des=12.0,
        dt=0.1,
    )
    assert solver._cached_h_val == pytest.approx(14.0)


# ---------------------------------------------------------------------------
# Track wrapper: projection, orientation, and cache behaviour
# ---------------------------------------------------------------------------

def test_solve_from_track_projects_distance_along_ego_heading():
    ego = make_ego(x=0.0, y=0.0, orientation=0.0, velocity=10.0)
    track = make_track(x=20.0, y=7.0, vx=10.0, vy=0.0)
    solver = CBFQPSolver(tau=0.0)

    solver.solve_from_track(ego, track, v_des=10.0, dt=0.1)

    # At heading zero, lateral displacement must not contribute to distance.
    assert solver._cached_h_val == pytest.approx(
        20.0 - solver.d_min
    )


def test_solve_from_track_projects_distance_for_rotated_ego():
    heading = np.pi / 2.0
    ego = make_ego(orientation=heading, velocity=10.0)
    # 20 m ahead along the ego's 90-degree heading, plus 5 m lateral.
    track = make_track(x=-5.0, y=20.0, vx=0.0, vy=10.0)
    solver = CBFQPSolver(tau=0.0)

    solver.solve_from_track(ego, track, v_des=10.0, dt=0.1)

    assert solver._cached_h_val == pytest.approx(20.0 - solver.d_min)


def test_solve_from_track_returns_float_and_updates_requested_acceleration():
    solver = CBFQPSolver()
    ego = make_ego(velocity=8.0)
    track = make_track(x=30.0, vx=8.0)

    result = solver.solve_from_track(ego, track, v_des=10.0, dt=0.1)

    assert isinstance(result, float)
    assert solver._cached_a_req == pytest.approx(result)


def test_solve_from_track_none_lead_returns_bounded_nominal_acceleration():
    solver = CBFQPSolver(a_min=-2.0, a_max=1.0)
    ego = make_ego(velocity=5.0)

    result = solver.solve_from_track(
        ego=ego, lead_track=None, v_des=20.0, dt=0.1
    )

    assert result == pytest.approx(1.0)


@pytest.mark.xfail(
    reason=(
        "Known cache gap: solve_from_track(None) returns nominal acceleration "
        "without updating _cached_a_req."
    ),
    strict=False,
)
def test_none_lead_should_update_requested_acceleration_cache():
    solver = CBFQPSolver()
    ego = make_ego(velocity=5.0)

    result = solver.solve_from_track(
        ego=ego, lead_track=None, v_des=10.0, dt=0.1
    )

    assert solver._cached_a_req == pytest.approx(result)


# ---------------------------------------------------------------------------
# Logging state
# ---------------------------------------------------------------------------

def test_get_log_data_has_expected_keys_and_initial_values():
    solver = CBFQPSolver(
        gamma=2.0, d_min=3.0, tau=1.5,
        a_min=-4.0, a_max=2.0, use_cvxpy=False,
    )
    data = solver.get_log_data()

    assert set(data) == {
        "h_val", "a_req", "gamma", "d_min", "tau",
        "a_min", "a_max", "use_cvxpy",
    }
    assert data == {
        "h_val": 0.0,
        "a_req": 0.0,
        "gamma": 2.0,
        "d_min": 3.0,
        "tau": 1.5,
        "a_min": -4.0,
        "a_max": 2.0,
        "use_cvxpy": False,
    }


def test_get_log_data_reflects_state_after_track_solve():
    solver = CBFQPSolver(tau=0.0)
    ego = make_ego(velocity=5.0)
    track = make_track(x=20.0, vx=5.0)

    result = solver.solve_from_track(ego, track, v_des=8.0, dt=0.1)
    data = solver.get_log_data()

    assert data["h_val"] == pytest.approx(20.0 - solver.d_min)
    assert data["a_req"] == pytest.approx(result)


# ---------------------------------------------------------------------------
# CVXPY path: mathematical feasibility and fallback
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not CVXPY_AVAILABLE, reason="cvxpy is not installed")
@pytest.mark.parametrize(
    "distance, v_ego, v_target, v_des, dt",
    [
        (50.0, 10.0, 10.0, 12.0, 0.1),
        (30.0, 10.0, 12.0, 15.0, 0.1),
        (20.0, 12.0, 10.0, 10.0, 0.05),
    ],
)
def test_cvxpy_feasible_solution_respects_bounds_and_discrete_cbf(
    distance, v_ego, v_target, v_des, dt
):
    solver = CBFQPSolver(
        use_cvxpy=True, gamma=1.0, d_min=2.0, tau=0.5,
        a_min=-5.0, a_max=2.0,
    )
    h_current = solver.compute_barrier(distance, v_ego)
    result = solver.solve(distance, v_ego, v_target, v_des, dt)

    assert isinstance(result, float)
    assert solver.a_min - 1e-6 <= result <= solver.a_max + 1e-6

    v_ego_next = v_ego + result * dt
    x_local_next = distance + (v_target - v_ego) * dt
    h_next = x_local_next - (
        solver.d_min + v_ego_next * solver.tau
    )
    assert (h_next - h_current) / dt >= (
        -solver.gamma * h_current - 1e-4
    )


@pytest.mark.skipif(not CVXPY_AVAILABLE, reason="cvxpy is not installed")
def test_cvxpy_mode_is_enabled_only_when_dependency_is_available():
    solver = CBFQPSolver(use_cvxpy=True)
    assert solver.use_cvxpy is True


def test_cvxpy_requested_but_unavailable_falls_back_to_analytical(monkeypatch):
    import src.cbf_solver as cbf_module

    monkeypatch.setattr(cbf_module, "CVXPY_AVAILABLE", False)
    solver = CBFQPSolver(use_cvxpy=True)
    assert solver.use_cvxpy is False

    result = solver.solve(
        longitudinal_dist=40.0,
        v_ego=10.0,
        v_target=10.0,
        v_des=12.0,
        dt=0.1,
    )
    assert result == pytest.approx(1.0)


@pytest.mark.skipif(not CVXPY_AVAILABLE, reason="cvxpy is not installed")
def test_cvxpy_exception_falls_back_to_minimum_acceleration(monkeypatch):
    import src.cbf_solver as cbf_module

    def raise_runtime_error(*args, **kwargs):
        raise RuntimeError("simulated solver failure")

    monkeypatch.setattr(cbf_module.cp.Problem, "solve", raise_runtime_error)
    solver = CBFQPSolver(use_cvxpy=True, a_min=-4.0)

    result = solver.solve(
        longitudinal_dist=30.0,
        v_ego=10.0,
        v_target=10.0,
        v_des=15.0,
        dt=0.1,
    )
    assert result == pytest.approx(solver.a_min)


@pytest.mark.xfail(
    reason=(
        "Known objective mismatch: analytical nominal acceleration uses "
        "0.5*(v_des-v_ego), while CVXPY's unconstrained optimum is "
        "(v_des-v_ego)/dt in acceleration units."
    ),
    strict=False,
)
@pytest.mark.skipif(not CVXPY_AVAILABLE, reason="cvxpy is not installed")
def test_analytical_and_cvxpy_match_in_a_nonbinding_safety_case():
    common = dict(
        longitudinal_dist=1000.0,
        v_ego=10.0,
        v_target=10.0,
        v_des=12.0,
        dt=0.1,
    )
    analytical = CBFQPSolver(
        use_cvxpy=False, tau=0.5, a_min=-5.0, a_max=5.0
    ).solve(**common)
    cvxpy_result = CBFQPSolver(
        use_cvxpy=True, tau=0.5, a_min=-5.0, a_max=5.0
    ).solve(**common)

    assert cvxpy_result == pytest.approx(analytical, abs=1e-3)


# ---------------------------------------------------------------------------
# Parameter/configuration behaviour
# ---------------------------------------------------------------------------

def test_solver_preserves_configured_parameters():
    solver = CBFQPSolver(
        gamma=0.7, d_min=4.0, tau=1.2,
        a_min=-6.0, a_max=1.8,
    )
    assert solver.gamma == pytest.approx(0.7)
    assert solver.d_min == pytest.approx(4.0)
    assert solver.tau == pytest.approx(1.2)
    assert solver.a_min == pytest.approx(-6.0)
    assert solver.a_max == pytest.approx(1.8)


@pytest.mark.parametrize(
    "v_des, expected",
    [
        (20.0, 2.0),  # upper acceleration saturation
        (0.0, -2.0),  # lower acceleration saturation
        (12.0, 1.0),  # nominal command is 1 m/s^2
    ],
)
def test_no_lead_nominal_acceleration_is_clipped(v_des, expected):
    solver = CBFQPSolver(a_min=-2.0, a_max=2.0)
    ego = make_ego(velocity=10.0)
    result = solver.solve_from_track(
        ego=ego, lead_track=None, v_des=v_des, dt=0.1
    )
    assert result == pytest.approx(expected)
