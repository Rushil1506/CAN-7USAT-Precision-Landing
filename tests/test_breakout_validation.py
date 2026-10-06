import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from control.pid import PIDController
from flight.parafoil import DrogueConfig, ParafoilConfig, ParafoilModel, ParafoilState
from guidance.breakout_guidance import (
    BreakoutGeometry,
    calculate_breakout_geometry,
)
from navigation.bearing import bearing
from navigation.coordinates import LocalPoint
from navigation.distance import distance
from navigation.heading import heading_error


@dataclass
class BreakoutDynamicResult:
    """
    Detailed telemetry and validation metrics for a simulated breakout maneuver.
    """

    geom: BreakoutGeometry
    time: List[float]
    x: List[float]
    y: List[float]
    altitude: List[float]
    heading: List[float]
    steering: List[float]
    heading_error: List[float]
    distance_to_target: List[float]

    # Required validation measurements
    touchdown_position: LocalPoint
    touchdown_error: float
    closest_approach_distance: float
    altitude_at_closest_approach: float
    time_to_closest_approach: float
    touchdown_time: float
    heading_at_start_of_straight: float
    heading_error_at_start_of_straight: float
    max_steering_command: float
    heading_reversal_occurred: bool
    target_crossed_before_touchdown: bool

    # Separated error components
    geometry_error_deg: float        # Discontinuity in heading alignment at turn endpoint
    dynamic_turn_error_m: float      # Distance between actual turn endpoint and predicted P_1
    final_straight_error_m: float    # Closest approach distance to target during straight glide
    touchdown_error_m: float         # Distance from target at ground impact (altitude = 0)


def run_breakout_simulation(
    initial_position: LocalPoint,
    initial_heading: float,
    target: LocalPoint = LocalPoint(0.0, 0.0),
    breakout_altitude_margin: float = 2.0,
    forward_speed: float = 8.0,
    descent_rate: float = 2.5,
    max_turn_rate: float = 30.0,
    dt: float = 0.1,
    kp: float = 0.022,
    ki: float = 0.00015,
    kd: float = 0.004,
) -> BreakoutDynamicResult:
    """
    Simulate the physical vehicle beginning breakout at h = h_required_min + margin.

    Phases:
        1. Turn arc: t < turn_duration -> command constant-rate turn (turn_direction * 1.0)
        2. Final straight flight: t >= turn_duration -> steer toward target using PID
    """
    geom = calculate_breakout_geometry(
        position=initial_position,
        heading=initial_heading,
        target=target,
        forward_speed=forward_speed,
        descent_rate=descent_rate,
        max_turn_rate=max_turn_rate,
        breakout_altitude_margin=breakout_altitude_margin,
    )

    state = ParafoilState(
        x=initial_position.x,
        y=initial_position.y,
        altitude=geom.breakout_altitude,
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

    pid = PIDController(
        kp=kp,
        ki=ki,
        kd=kd,
        output_min=-1.0,
        output_max=1.0,
    )

    times: List[float] = []
    xs: List[float] = []
    ys: List[float] = []
    alts: List[float] = []
    hdgs: List[float] = []
    steers: List[float] = []
    h_errs: List[float] = []
    dists: List[float] = []

    t = 0.0
    min_dist = float("inf")
    min_dist_t = 0.0
    min_dist_alt = 0.0

    prev_bearing: Optional[float] = None
    heading_reversal = False
    max_steer = 0.0

    start_straight_hdg: Optional[float] = None
    start_straight_herr: Optional[float] = None
    actual_turn_endpoint: Optional[LocalPoint] = None

    while model.state.altitude > 0.0:
        pos = LocalPoint(model.state.x, model.state.y)
        d = distance(pos, target)

        if d < min_dist:
            min_dist = d
            min_dist_t = t
            min_dist_alt = model.state.altitude

        b_target = bearing(pos, target)
        h_err = heading_error(model.state.heading, b_target)

        # Detect sharp heading reversals (bearing jump > 90°)
        if prev_bearing is not None:
            delta_b = (b_target - prev_bearing + 180.0) % 360.0 - 180.0
            if abs(delta_b) > 90.0:
                heading_reversal = True
        prev_bearing = b_target

        # Command sequence:
        # Phase 1: Constant-rate turn arc
        # Phase 2: Straight flight toward target
        if t < geom.turn_duration:
            steer = geom.turn_direction
        else:
            if start_straight_hdg is None:
                start_straight_hdg = model.state.heading
                start_straight_herr = h_err
                actual_turn_endpoint = pos
            steer = pid.update(error=h_err, dt=dt)

        times.append(t)
        xs.append(model.state.x)
        ys.append(model.state.y)
        alts.append(model.state.altitude)
        hdgs.append(model.state.heading)
        steers.append(steer)
        h_errs.append(h_err)
        dists.append(d)

        max_steer = max(max_steer, abs(steer))
        model.update_parafoil(steering=steer, dt=dt)
        t += dt

    # Final state at touchdown
    td_pos = LocalPoint(model.state.x, model.state.y)
    td_err = distance(td_pos, target)

    # Defaults for edge cases (e.g. turn_duration == 0)
    if start_straight_hdg is None:
        start_straight_hdg = initial_heading
        start_straight_herr = heading_error(initial_heading, bearing(initial_position, target))
        actual_turn_endpoint = initial_position

    # Separated error components
    # A. Geometry error: angular difference between heading at P_1 and bearing from P_1 to target
    pred_bearing_from_p1 = bearing(geom.turn_endpoint, target)
    expected_hdg_at_p1 = (initial_heading + geom.turn_direction * geom.turn_angle) % 360.0
    geometry_error_deg = abs(heading_error(expected_hdg_at_p1, pred_bearing_from_p1))

    # B. Dynamic turn execution error: displacement between simulated and predicted P_1
    dynamic_turn_error_m = distance(actual_turn_endpoint, geom.turn_endpoint)

    # C. Final straight glide error: closest approach to target
    final_straight_error_m = min_dist

    # D. Touchdown error
    touchdown_error_m = td_err

    # Target crossed before touchdown if closest approach happened before the final step
    target_crossed = (t - min_dist_t) > dt and min_dist_alt > 0.0

    return BreakoutDynamicResult(
        geom=geom,
        time=times,
        x=xs,
        y=ys,
        altitude=alts,
        heading=hdgs,
        steering=steers,
        heading_error=h_errs,
        distance_to_target=dists,
        touchdown_position=td_pos,
        touchdown_error=td_err,
        closest_approach_distance=min_dist,
        altitude_at_closest_approach=min_dist_alt,
        time_to_closest_approach=min_dist_t,
        touchdown_time=t,
        heading_at_start_of_straight=start_straight_hdg,
        heading_error_at_start_of_straight=start_straight_herr,
        max_steering_command=max_steer,
        heading_reversal_occurred=heading_reversal,
        target_crossed_before_touchdown=target_crossed,
        geometry_error_deg=geometry_error_deg,
        dynamic_turn_error_m=dynamic_turn_error_m,
        final_straight_error_m=final_straight_error_m,
        touchdown_error_m=touchdown_error_m,
    )


# =====================================================================
# Standalone Dynamic Validation Tests
# =====================================================================


def test_primary_nominal_breakout_validation():
    """
    Primary test:
        target = (0, 0)
        vehicle position = (0, -30)
        heading = 270° (West)
        CW breakout, margin = 2 m
    """
    target = LocalPoint(0.0, 0.0)
    pos0 = LocalPoint(0.0, -30.0)
    hdg0 = 270.0

    res = run_breakout_simulation(
        initial_position=pos0,
        initial_heading=hdg0,
        target=target,
        breakout_altitude_margin=2.0,
    )

    # 1. Turn duration and initial conditions
    assert res.geom.turn_duration == pytest.approx(3.0)
    assert res.geom.turn_angle == pytest.approx(90.0)
    assert res.geom.breakout_altitude == pytest.approx(16.1303, rel=1e-3)

    # 2. Heading at start of straight segment
    assert res.heading_at_start_of_straight == pytest.approx(0.0)

    # 3. Heading error at start of straight segment
    # Bearing from (-15.28, -14.72) to (0, 0) is atan2(15.28, 14.72) ≈ 46.1°
    assert res.heading_error_at_start_of_straight == pytest.approx(46.08, abs=0.5)

    # 4. Maximum steering command
    assert res.max_steering_command == 1.0

    # 5. No heading reversal occurs
    assert not res.heading_reversal_occurred

    # 6. Target reached before ground impact
    assert res.closest_approach_distance < 3.5
    assert res.altitude_at_closest_approach > 0.5
    assert res.target_crossed_before_touchdown

    # 7. Touchdown error
    assert res.touchdown_error < 5.0

    # 8. Time from breakout to touchdown
    assert 6.0 <= res.touchdown_time <= 7.0

    # 9. Verify separated error components
    assert res.geometry_error_deg == pytest.approx(46.08, abs=0.5)
    assert res.dynamic_turn_error_m < 1.0   # Sub-meter discretization error
    assert res.final_straight_error_m < 3.5 # Closest approach distance
    assert res.touchdown_error_m < 5.0     # Landing error at impact


def test_margin_sweep():
    """
    Margin sweep: margin = 0, 2, 5, 10, 20 m.
    Examines sensitivity of touchdown error and target crossing conditions.
    """
    target = LocalPoint(0.0, 0.0)
    pos0 = LocalPoint(0.0, -30.0)
    hdg0 = 270.0
    margins = [0.0, 2.0, 5.0, 10.0, 20.0]

    results = [
        run_breakout_simulation(pos0, hdg0, target, breakout_altitude_margin=m)
        for m in margins
    ]

    # Touchdown time increases monotonically with margin
    for i in range(len(results) - 1):
        assert results[i].touchdown_time < results[i + 1].touchdown_time

    # Closest approach distance remains approximately equal (~3 m) across all runs
    for r in results:
        assert r.closest_approach_distance < 4.5

    # Target crossing altitude increases directly with margin
    assert results[0].altitude_at_closest_approach < 1.0   # Margin 0 crosses near ground
    assert results[1].altitude_at_closest_approach > 1.0   # Margin 2 crosses at ~1.1 m
    assert results[2].altitude_at_closest_approach > 3.5   # Margin 5 crosses at ~4.1 m
    assert results[3].altitude_at_closest_approach > 8.0   # Margin 10 crosses at ~9.1 m
    assert results[4].altitude_at_closest_approach > 18.0  # Margin 20 crosses at ~19.1 m

    # For excessive margins (10 m, 20 m), vehicle overshoots before touchdown
    assert results[3].touchdown_error > 20.0
    assert results[4].touchdown_error > 20.0


def test_orbit_phase_cardinal_symmetry():
    """
    Validate rotational and reflectional symmetry at all 4 cardinal locations (CW and CCW).
    """
    target = LocalPoint(0.0, 0.0)
    cardinal_cases = [
        ("South CW", LocalPoint(0.0, -30.0), 270.0),
        ("South CCW", LocalPoint(0.0, -30.0), 90.0),
        ("East CW", LocalPoint(30.0, 0.0), 180.0),
        ("East CCW", LocalPoint(30.0, 0.0), 0.0),
        ("North CW", LocalPoint(0.0, 30.0), 90.0),
        ("North CCW", LocalPoint(0.0, 30.0), 270.0),
        ("West CW", LocalPoint(-30.0, 0.0), 0.0),
        ("West CCW", LocalPoint(-30.0, 0.0), 180.0),
    ]

    results = [run_breakout_simulation(p, h, target, 2.0) for _, p, h in cardinal_cases]

    baseline = results[0]
    for r in results[1:]:
        # Exact symmetry in touchdown error, closest approach, and altitude
        assert r.touchdown_error == pytest.approx(baseline.touchdown_error, abs=0.01)
        assert r.closest_approach_distance == pytest.approx(baseline.closest_approach_distance, abs=0.01)
        assert r.altitude_at_closest_approach == pytest.approx(baseline.altitude_at_closest_approach, abs=0.01)
        assert abs(r.heading_error_at_start_of_straight) == pytest.approx(
            abs(baseline.heading_error_at_start_of_straight), abs=0.01
        )
        assert not r.heading_reversal_occurred


def test_radial_variation():
    """
    Validate initial radial variation: r = 25 m, 30 m, 35 m.
    """
    target = LocalPoint(0.0, 0.0)
    radii = [25.0, 30.0, 35.0]

    results = [
        run_breakout_simulation(LocalPoint(0.0, -r), 270.0, target, 2.0)
        for r in radii
    ]

    for r in results:
        assert r.touchdown_error < 6.0
        assert r.closest_approach_distance < 6.0
        assert not r.heading_reversal_occurred


def test_heading_variation():
    """
    Validate representative heading errors: 0°, 30°, 60°, 90°, 120°, 180°.
    """
    target = LocalPoint(0.0, 0.0)
    pos = LocalPoint(0.0, -30.0)  # Bearing to target is 0°
    heading_errors_to_test = [0.0, 30.0, 60.0, 90.0, 120.0, 180.0]

    for h_err in heading_errors_to_test:
        hdg = (360.0 - h_err) % 360.0
        res = run_breakout_simulation(pos, hdg, target, 2.0)

        # For 180°, deterministic turn direction is -1.0 (CCW/left)
        if h_err == 180.0:
            assert res.geom.turn_direction == -1.0

        # Closest approach distance remains within 3.0 m for all angles
        assert res.closest_approach_distance < 3.5
        # Touchdown error remains within 7.5 m
        assert res.touchdown_error < 7.5


def generate_trajectory_plot(
    output_path: str = "breakout_validation_trajectory.png",
) -> None:
    """
    Generate trajectory plot showing breakout point, turn arc, turn endpoint,
    final straight segment, target, and touchdown.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    target = LocalPoint(0.0, 0.0)
    pos0 = LocalPoint(0.0, -30.0)
    hdg0 = 270.0

    res = run_breakout_simulation(pos0, hdg0, target, breakout_altitude_margin=2.0)

    # Segment indices
    dt = 0.1
    idx_turn = int(round(res.geom.turn_duration / dt))

    turn_x = res.x[: idx_turn + 1]
    turn_y = res.y[: idx_turn + 1]
    straight_x = res.x[idx_turn:]
    straight_y = res.y[idx_turn:]

    fig, ax = plt.subplots(figsize=(8, 8))

    # Loiter orbit
    loiter_circle = Circle(
        (target.x, target.y),
        30.0,
        fill=False,
        linestyle=":",
        color="orange",
        label="Holding orbit (30m)",
    )
    ax.add_patch(loiter_circle)

    # Trajectory segments
    ax.plot(turn_x, turn_y, color="tab:blue", linewidth=2.5, label=f"Turn arc ({res.geom.turn_duration:.1f}s)")
    ax.plot(straight_x, straight_y, color="tab:green", linewidth=2.5, label="Final straight segment")

    # Key points
    ax.scatter([pos0.x], [pos0.y], color="red", marker="o", s=80, zorder=5, label=f"Breakout start (alt={res.geom.breakout_altitude:.1f}m)")
    ax.scatter([res.geom.turn_endpoint.x], [res.geom.turn_endpoint.y], color="purple", marker="x", s=80, zorder=5, label="Predicted turn endpoint P1")
    ax.scatter([res.x[idx_turn]], [res.y[idx_turn]], color="darkviolet", marker="s", s=60, zorder=5, label="Actual turn endpoint")
    ax.scatter([target.x], [target.y], color="gold", edgecolors="black", marker="*", s=160, zorder=6, label="Landing target (0,0)")
    ax.scatter([res.touchdown_position.x], [res.touchdown_position.y], color="crimson", marker="v", s=100, zorder=6, label=f"Touchdown (error={res.touchdown_error:.2f}m)")

    ax.set_xlabel("East [m]")
    ax.set_ylabel("North [m]")
    ax.set_title("Autonomous Parafoil Breakout Dynamic Validation (Zero Wind)")
    ax.axis("equal")
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


if __name__ == "__main__":
    target = LocalPoint(0.0, 0.0)
    pos0 = LocalPoint(0.0, -30.0)
    hdg0 = 270.0

    print("==========================================================================================")
    print("AUTONOMOUS BREAKOUT GUIDANCE DYNAMIC VALIDATION REPORT")
    print("==========================================================================================")

    # 1. Primary test
    nom = run_breakout_simulation(pos0, hdg0, target, breakout_altitude_margin=2.0)
    print("\n--- 1. PRIMARY NOMINAL TEST (Margin = 2.0 m) ---")
    print(f"Breakout start position        : ({pos0.x:.1f}, {pos0.y:.1f}) m")
    print(f"Breakout altitude              : {nom.geom.breakout_altitude:.2f} m (min: {nom.geom.minimum_required_altitude:.2f} m + 2.0 m)")
    print(f"Turn duration / arc length     : {nom.geom.turn_duration:.2f} s / {nom.geom.turn_arc_length:.2f} m")
    print(f"Predicted turn endpoint P1     : ({nom.geom.turn_endpoint.x:.2f}, {nom.geom.turn_endpoint.y:.2f}) m")
    print(f"Heading at start of straight   : {nom.heading_at_start_of_straight:.2f}°")
    print(f"Heading error at start of straight: {nom.heading_error_at_start_of_straight:.2f}°")
    print(f"Maximum steering command       : {nom.max_steering_command:.2f}")
    print(f"Heading reversal occurred      : {nom.heading_reversal_occurred}")
    print(f"Target crossed before ground   : {nom.target_crossed_before_touchdown}")
    print(f"Closest approach distance      : {nom.closest_approach_distance:.2f} m at alt = {nom.altitude_at_closest_approach:.2f} m (t = {nom.time_to_closest_approach:.1f} s)")
    print(f"Touchdown position / time      : ({nom.touchdown_position.x:.2f}, {nom.touchdown_position.y:.2f}) m at t = {nom.touchdown_time:.1f} s")
    print(f"Touchdown landing error        : {nom.touchdown_error:.2f} m")
    print("\nSeparated error components:")
    print(f"  A. Geometry error (kink)     : {nom.geometry_error_deg:.2f}°")
    print(f"  B. Dynamic turn error        : {nom.dynamic_turn_error_m:.2f} m")
    print(f"  C. Final straight-glide error: {nom.final_straight_error_m:.2f} m")
    print(f"  D. Touchdown error           : {nom.touchdown_error_m:.2f} m")

    # 2. Margin sweep
    print("\n--- 2. MARGIN SWEEP ---")
    print(f"{'Margin [m]':10s} | {'Alt Init [m]':12s} | {'Touchdown [s]':13s} | {'Min Dist [m]':12s} | {'Alt Crossing [m]':16s} | {'Landing Err [m]'}")
    print("-" * 85)
    for m in [0.0, 2.0, 5.0, 10.0, 20.0]:
        r = run_breakout_simulation(pos0, hdg0, target, breakout_altitude_margin=m)
        print(f"{m:10.1f} | {r.geom.breakout_altitude:12.2f} | {r.touchdown_time:13.1f} | {r.closest_approach_distance:12.2f} | {r.altitude_at_closest_approach:16.2f} | {r.touchdown_error:15.2f}")

    # 3. Cardinal symmetry
    print("\n--- 3. CARDINAL & ROTATIONAL SYMMETRY ---")
    print(f"{'Scenario':12s} | {'Landing Err [m]':15s} | {'Min Dist [m]':12s} | {'Crossing Alt [m]':16s} | {'Heading Err [°]'}")
    print("-" * 80)
    cardinal_cases = [
        ("South CW", LocalPoint(0.0, -30.0), 270.0),
        ("South CCW", LocalPoint(0.0, -30.0), 90.0),
        ("East CW", LocalPoint(30.0, 0.0), 180.0),
        ("East CCW", LocalPoint(30.0, 0.0), 0.0),
        ("North CW", LocalPoint(0.0, 30.0), 90.0),
        ("North CCW", LocalPoint(0.0, 30.0), 270.0),
        ("West CW", LocalPoint(-30.0, 0.0), 0.0),
        ("West CCW", LocalPoint(-30.0, 0.0), 180.0),
    ]
    for name, p, h in cardinal_cases:
        r = run_breakout_simulation(p, h, target, 2.0)
        print(f"{name:12s} | {r.touchdown_error:15.2f} | {r.closest_approach_distance:12.2f} | {r.altitude_at_closest_approach:16.2f} | {r.heading_error_at_start_of_straight:15.2f}")

    # 4. Heading variation
    print("\n--- 4. HEADING VARIATION ---")
    print(f"{'Heading Error':14s} | {'Turn Dir':8s} | {'Alt Init [m]':12s} | {'Landing Err [m]':15s} | {'Min Dist [m]':12s} | {'Reversal'}")
    print("-" * 85)
    for h_err in [0.0, 30.0, 60.0, 90.0, 120.0, 180.0]:
        hdg = (360.0 - h_err) % 360.0
        r = run_breakout_simulation(pos0, hdg, target, 2.0)
        t_dir_str = "+1 (CW)" if r.geom.turn_direction > 0 else "-1 (CCW)"
        print(f"{h_err:11.1f}° | {t_dir_str:8s} | {r.geom.breakout_altitude:12.2f} | {r.touchdown_error:15.2f} | {r.closest_approach_distance:12.2f} | {str(r.heading_reversal_occurred)}")

    # Generate plot
    generate_trajectory_plot("breakout_validation_trajectory.png")
    print("\nTrajectory plot generated: breakout_validation_trajectory.png")
    print("==========================================================================================")
