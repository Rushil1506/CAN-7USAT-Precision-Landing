import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from flight.parafoil import DrogueConfig, ParafoilConfig, ParafoilModel, ParafoilState
from navigation.coordinates import LocalPoint
from guidance.loiter_guidance import (
    LoiterDirection,
    LoiterGuidance,
    LoiterPhase,
)


@dataclass
class LoiterSimulationResult:
    """
    Time-series and summary validation metrics for a loiter dynamic simulation run.
    """

    time: List[float]
    x: List[float]
    y: List[float]
    radial_distance: List[float]
    radial_error: List[float]
    commanded_heading: List[float]
    actual_heading: List[float]
    steering_command: List[float]
    phase: List[LoiterPhase]

    # Validation criteria summary
    time_to_capture: Optional[float]
    max_radial_error_after_capture: float
    steady_state_max_radial_error: float
    steady_state_mean_radial_error: float
    steady_state_mean_steering: float
    steady_state_steering_std: float
    sustained_tracking_achieved: bool


def run_loiter_simulation(
    initial_position: LocalPoint,
    initial_heading: float,
    direction: LoiterDirection,
    center: LocalPoint = LocalPoint(0.0, 0.0),
    loiter_radius: float = 30.0,
    capture_tolerance: float = 5.0,
    forward_speed: float = 8.0,
    descent_rate: float = 2.5,
    max_turn_rate: float = 30.0,
    dt: float = 0.1,
    sim_duration: float = 120.0,
) -> LoiterSimulationResult:
    """
    Run closed-loop dynamic simulation of LoiterGuidance controlling ParafoilModel.

    Assumptions & Limitations:
        - Uses the current simplified kinematic parafoil model (ParafoilModel).
        - Constant forward airspeed (8.0 m/s) and constant descent rate (2.5 m/s).
        - Zero wind.
        - Actuator dynamics are simplified to direct steering-to-turn-rate mapping.
        - Does NOT model spiral dive (turn-induced increase in descent rate).
    """
    state = ParafoilState(
        x=initial_position.x,
        y=initial_position.y,
        altitude=500.0,
        vx=0.0,
        vy=0.0,
        vz=-descent_rate,
        heading=initial_heading,
        heading_rate=0.0,
    )
    parafoil_config = ParafoilConfig(
        forward_speed=forward_speed,
        descent_rate=descent_rate,
        max_turn_rate=max_turn_rate,
        wind_x=0.0,
        wind_y=0.0,
    )
    model = ParafoilModel(
        state=state,
        drogue_config=DrogueConfig(),
        parafoil_config=parafoil_config,
    )
    guidance = LoiterGuidance(
        center=center,
        loiter_radius=loiter_radius,
        direction=direction,
        capture_tolerance=capture_tolerance,
        forward_speed=forward_speed,
        max_turn_rate=max_turn_rate,
    )

    times: List[float] = []
    xs: List[float] = []
    ys: List[float] = []
    r_dists: List[float] = []
    r_errs: List[float] = []
    cmd_headings: List[float] = []
    act_headings: List[float] = []
    steerings: List[float] = []
    phases: List[LoiterPhase] = []

    time_to_capture: Optional[float] = None

    t = 0.0
    while t <= sim_duration:
        pos = LocalPoint(x=model.state.x, y=model.state.y)
        out = guidance.update(position=pos, current_heading=model.state.heading)

        times.append(t)
        xs.append(model.state.x)
        ys.append(model.state.y)
        r_dists.append(out.radial_distance)
        r_errs.append(out.radial_error)
        cmd_headings.append(out.desired_heading)
        act_headings.append(model.state.heading)
        steerings.append(out.steering_command)
        phases.append(out.phase)

        if time_to_capture is None and abs(out.radial_error) <= capture_tolerance:
            time_to_capture = t

        model.update_parafoil(steering=out.steering_command, dt=dt)
        t += dt

    # Post-capture analysis
    if time_to_capture is not None:
        idx_cap = int(round(time_to_capture / dt))
        post_cap_errs = [abs(e) for e in r_errs[idx_cap:]]
        max_after_capture = max(post_cap_errs)
    else:
        max_after_capture = float("nan")

    # Steady-state analysis (final 40 s: ~1.7 complete orbits with T = 2*pi*30/8 = 23.56 s)
    steady_window_s = min(40.0, sim_duration * 0.5)
    idx_steady = int(round((sim_duration - steady_window_s) / dt))
    steady_errs = [abs(e) for e in r_errs[idx_steady:]]
    steady_steerings = steerings[idx_steady:]

    steady_max = max(steady_errs) if steady_errs else float("nan")
    steady_mean = sum(steady_errs) / len(steady_errs) if steady_errs else float("nan")
    steady_steer_mean = sum(steady_steerings) / len(steady_steerings) if steady_steerings else 0.0

    steer_var = sum((s - steady_steer_mean) ** 2 for s in steady_steerings) / len(steady_steerings)
    steer_std = math.sqrt(steer_var)

    # Sustained tracking criteria: steady-state maximum radial error stays within tolerance
    sustained = steady_max <= capture_tolerance if not math.isnan(steady_max) else False

    return LoiterSimulationResult(
        time=times,
        x=xs,
        y=ys,
        radial_distance=r_dists,
        radial_error=r_errs,
        commanded_heading=cmd_headings,
        actual_heading=act_headings,
        steering_command=steerings,
        phase=phases,
        time_to_capture=time_to_capture,
        max_radial_error_after_capture=max_after_capture,
        steady_state_max_radial_error=steady_max,
        steady_state_mean_radial_error=steady_mean,
        steady_state_mean_steering=steady_steer_mean,
        steady_state_steering_std=steer_std,
        sustained_tracking_achieved=sustained,
    )


# =====================================================================
# Standalone Dynamic Validation Test Cases
# =====================================================================


def test_loiter_validation_100m_cw():
    """Scenario 1: Start 100 m away from center, Clockwise."""
    res = run_loiter_simulation(
        initial_position=LocalPoint(0.0, -100.0),
        initial_heading=0.0,  # Heading North toward center
        direction=LoiterDirection.CLOCKWISE,
    )
    assert res.time_to_capture is not None
    assert res.time_to_capture < 15.0  # Captures within 15 seconds
    assert res.sustained_tracking_achieved
    assert res.steady_state_max_radial_error < 0.5  # Sub-meter steady-state precision
    assert math.isclose(res.steady_state_mean_steering, 0.51, abs_tol=0.02)
    assert res.steady_state_steering_std < 0.01  # No oscillation/chatter


def test_loiter_validation_100m_ccw():
    """Scenario 1 (cont): Start 100 m away from center, Counter-Clockwise."""
    res = run_loiter_simulation(
        initial_position=LocalPoint(0.0, -100.0),
        initial_heading=0.0,  # Heading North toward center
        direction=LoiterDirection.COUNTER_CLOCKWISE,
    )
    assert res.time_to_capture is not None
    assert res.time_to_capture < 15.0
    assert res.sustained_tracking_achieved
    assert res.steady_state_max_radial_error < 0.5
    assert math.isclose(res.steady_state_mean_steering, -0.51, abs_tol=0.02)
    assert res.steady_state_steering_std < 0.01


def test_loiter_validation_50m_cw():
    """Scenario 2: Start 50 m away from center, Clockwise."""
    res = run_loiter_simulation(
        initial_position=LocalPoint(0.0, -50.0),
        initial_heading=0.0,
        direction=LoiterDirection.CLOCKWISE,
    )
    assert res.time_to_capture is not None
    assert res.time_to_capture < 5.0
    assert res.sustained_tracking_achieved
    assert res.max_radial_error_after_capture <= 5.0  # Stays within tolerance throughout
    assert res.steady_state_max_radial_error < 0.5
    assert res.steady_state_steering_std < 0.01


def test_loiter_validation_50m_ccw():
    """Scenario 2 (cont): Start 50 m away from center, Counter-Clockwise."""
    res = run_loiter_simulation(
        initial_position=LocalPoint(0.0, -50.0),
        initial_heading=0.0,
        direction=LoiterDirection.COUNTER_CLOCKWISE,
    )
    assert res.time_to_capture is not None
    assert res.time_to_capture < 5.0
    assert res.sustained_tracking_achieved
    assert res.max_radial_error_after_capture <= 5.0
    assert res.steady_state_max_radial_error < 0.5
    assert res.steady_state_steering_std < 0.01


def test_loiter_validation_20m_cw():
    """Scenario 3: Start 20 m away from center (inside orbit), Clockwise."""
    res = run_loiter_simulation(
        initial_position=LocalPoint(0.0, -20.0),
        initial_heading=0.0,  # Moving North across center
        direction=LoiterDirection.CLOCKWISE,
    )
    assert res.time_to_capture is not None
    assert res.time_to_capture < 10.0
    assert res.sustained_tracking_achieved
    assert res.steady_state_max_radial_error < 0.5
    assert res.steady_state_steering_std < 0.01


def test_loiter_validation_20m_ccw():
    """Scenario 3 (cont): Start 20 m away from center (inside orbit), Counter-Clockwise."""
    res = run_loiter_simulation(
        initial_position=LocalPoint(0.0, -20.0),
        initial_heading=0.0,
        direction=LoiterDirection.COUNTER_CLOCKWISE,
    )
    assert res.time_to_capture is not None
    assert res.time_to_capture < 10.0
    assert res.sustained_tracking_achieved
    assert res.steady_state_max_radial_error < 0.5
    assert res.steady_state_steering_std < 0.01


def test_loiter_validation_30m_boundary_cw():
    """Scenario 4: Start exactly on 30 m orbit boundary, Clockwise."""
    # Tangent at (0, -30) for CW is West (270.0°)
    res = run_loiter_simulation(
        initial_position=LocalPoint(0.0, -30.0),
        initial_heading=270.0,
        direction=LoiterDirection.CLOCKWISE,
    )
    assert res.time_to_capture == 0.0
    assert res.sustained_tracking_achieved
    assert res.max_radial_error_after_capture < 0.5  # Sub-meter deviation from t=0
    assert res.steady_state_max_radial_error < 0.5
    assert res.steady_state_steering_std < 0.01


def test_loiter_validation_30m_boundary_ccw():
    """Scenario 4 (cont): Start exactly on 30 m orbit boundary, Counter-Clockwise."""
    # Tangent at (0, -30) for CCW is East (90.0°)
    res = run_loiter_simulation(
        initial_position=LocalPoint(0.0, -30.0),
        initial_heading=90.0,
        direction=LoiterDirection.COUNTER_CLOCKWISE,
    )
    assert res.time_to_capture == 0.0
    assert res.sustained_tracking_achieved
    assert res.max_radial_error_after_capture < 0.5
    assert res.steady_state_max_radial_error < 0.5
    assert res.steady_state_steering_std < 0.01


def test_loiter_validation_sustained_tracking_criteria():
    """Explicit validation that sustained tracking remains inside band for >= 60 seconds."""
    # Test a longer simulation of 180 seconds (~7.6 complete orbital revolutions)
    res = run_loiter_simulation(
        initial_position=LocalPoint(0.0, -100.0),
        initial_heading=0.0,
        direction=LoiterDirection.CLOCKWISE,
        sim_duration=180.0,
    )
    assert res.sustained_tracking_achieved

    # Inspect all samples in the final 60 seconds
    idx_final_60s = int(round((180.0 - 60.0) / 0.1))
    recent_errors = [abs(e) for e in res.radial_error[idx_final_60s:]]

    # Must stay within 5.0 m capture tolerance 100% of the time during sustained tracking
    assert all(e <= 5.0 for e in recent_errors)
    # Average error during sustained tracking should be well below 1.0 m
    assert (sum(recent_errors) / len(recent_errors)) < 0.5


if __name__ == "__main__":
    test_cases = [
        ("100m Start CW", LocalPoint(0.0, -100.0), 0.0, LoiterDirection.CLOCKWISE),
        ("100m Start CCW", LocalPoint(0.0, -100.0), 0.0, LoiterDirection.COUNTER_CLOCKWISE),
        ("50m Start CW", LocalPoint(0.0, -50.0), 0.0, LoiterDirection.CLOCKWISE),
        ("50m Start CCW", LocalPoint(0.0, -50.0), 0.0, LoiterDirection.COUNTER_CLOCKWISE),
        ("20m Start CW", LocalPoint(0.0, -20.0), 0.0, LoiterDirection.CLOCKWISE),
        ("20m Start CCW", LocalPoint(0.0, -20.0), 0.0, LoiterDirection.COUNTER_CLOCKWISE),
        ("30m Exact CW", LocalPoint(0.0, -30.0), 270.0, LoiterDirection.CLOCKWISE),
        ("30m Exact CCW", LocalPoint(0.0, -30.0), 90.0, LoiterDirection.COUNTER_CLOCKWISE),
    ]

    print("==========================================================================================")
    print("LOITER GUIDANCE DYNAMIC VALIDATION REPORT")
    print("==========================================================================================")
    print(
        f"{'Scenario':15s} | {'Capture [s]':11s} | {'Max Post [m]':12s} | "
        f"{'Steady Max [m]':14s} | {'Steady Mean [m]':15s} | {'Steering Cmd':12s} | {'Sustained'}"
    )
    print("------------------------------------------------------------------------------------------")
    for name, pos, heading, direction in test_cases:
        r = run_loiter_simulation(
            initial_position=pos,
            initial_heading=heading,
            direction=direction,
        )
        t_cap_str = f"{r.time_to_capture:5.1f}" if r.time_to_capture is not None else "N/A"
        steer_str = f"{r.steady_state_mean_steering:+5.2f}±{r.steady_state_steering_std:.4f}"
        print(
            f"{name:15s} | {t_cap_str:>11s} | {r.max_radial_error_after_capture:12.2f} | "
            f"{r.steady_state_max_radial_error:14.2f} | {r.steady_state_mean_radial_error:15.2f} | "
            f"{steer_str:>12s} | {str(r.sustained_tracking_achieved)}"
        )
    print("==========================================================================================")
