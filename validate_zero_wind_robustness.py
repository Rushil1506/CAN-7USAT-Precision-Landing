"""
Zero-Wind Robustness Validation and Sensitivity Analysis Script for CAN-7USAT.

This script executes a comprehensive sensitivity evaluation across 5 scenario groups:
  - Group A: Breakout Altitude Margin (0, 1, 2, 3, 5, 10 m)
  - Group B: Loiter Orbit Radius (20, 25, 30, 35, 40 m)
  - Group C: Loiter Direction (Clockwise vs Counter-Clockwise)
  - Group D: Orbit Phase & Initial Condition Perturbations (0°, +45°, +90°, +180° for CW & CCW,
             plus multi-cardinal full-mission approaches)
  - Group E: Small Radial Tracking Offsets (r = 25, 30, 35 m for CW & CCW)

Records all 13 required metrics for each scenario, classifies results into:
  - nominal / acceptable
  - degraded
  - failed
Generates a structured CSV table and an analytical plot.
"""

import csv
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from control.pid import PIDController
from flight.parafoil import DrogueConfig, ParafoilConfig, ParafoilState
from flight.state_machine import FlightState
from guidance.guidance_controller import GuidanceController
from guidance.loiter_guidance import LoiterDirection, LoiterPhase
from navigation.bearing import bearing
from navigation.coordinates import LocalPoint
from navigation.distance import distance
from navigation.heading import heading_error
from navigation.waypoints import generate_waypoints
from navigation.waypoint_manager import WaypointManager
from simulation.simulator import ParafoilSimulator, SimulationSample


@dataclass
class ScenarioResult:
    group: str
    scenario_id: str
    description: str
    # 13 required metrics:
    touchdown_error: float
    closest_approach: float
    target_crossing_altitude: float
    breakout_altitude: Optional[float]
    breakout_distance: Optional[float]
    breakout_heading: Optional[float]
    terminal_duration: float
    orbit_count: float
    capture_time: Optional[float]
    mean_steady_radial_error: float
    max_steady_radial_error: float
    max_steering_command: float
    bearing_reversals: int
    failsafe: bool
    # Classification:
    classification: str
    classification_reason: str
    # Trajectory for plotting:
    final_x: float
    final_y: float
    trajectory_x: List[float]
    trajectory_y: List[float]


def run_full_mission(
    start: LocalPoint = LocalPoint(0.0, 0.0),
    target: LocalPoint = LocalPoint(1200.0, 0.0),
    num_waypoints: int = 10,
    waypoint_acceptance_radius: float = 25.0,
    final_approach_altitude: float = 120.0,
    final_approach_radius: float = 100.0,
    terminal_margin: float = 100.0,
    terminal_capture_radius: float = 200.0,
    loiter_radius: float = 30.0,
    loiter_direction: LoiterDirection = LoiterDirection.CLOCKWISE,
    breakout_altitude_margin: float = 2.0,
    max_terminal_guidance_time: float = 120.0,
    max_time: float = 500.0,
) -> Tuple[ParafoilSimulator, List[SimulationSample]]:
    """Execute end-to-end mission from deployment at 1000m to touchdown."""
    waypoints = generate_waypoints(start=start, target=target, num_waypoints=num_waypoints)
    waypoint_manager = WaypointManager(waypoints=waypoints, acceptance_radius=waypoint_acceptance_radius)

    pid = PIDController(
        kp=0.022,
        ki=0.00015,
        kd=0.004,
        output_min=-1.0,
        output_max=1.0,
        integral_min=-200.0,
        integral_max=200.0,
    )

    guidance = GuidanceController(
        waypoint_manager=waypoint_manager,
        pid=pid,
        final_approach_altitude=final_approach_altitude,
        final_approach_radius=final_approach_radius,
    )

    initial_heading = bearing(start, target)
    state = ParafoilState(
        x=start.x,
        y=start.y,
        altitude=1000.0,
        vx=0.0,
        vy=0.0,
        vz=0.0,
        heading=initial_heading,
        heading_rate=0.0,
    )

    simulator = ParafoilSimulator(
        state=state,
        drogue_config=DrogueConfig(descent_rate=17.0, horizontal_speed=0.0),
        parafoil_config=ParafoilConfig(forward_speed=8.0, descent_rate=2.5, max_turn_rate=30.0),
        dt=0.1,
        parafoil_deployment_altitude=600.0,
        deployment_duration=2.0,
        target_position=target,
        final_approach_altitude=final_approach_altitude,
        final_approach_radius=final_approach_radius,
        terminal_margin=terminal_margin,
        terminal_capture_radius=terminal_capture_radius,
        loiter_radius=loiter_radius,
        loiter_capture_tolerance=5.0,
        loiter_direction=loiter_direction,
        max_terminal_guidance_time=max_terminal_guidance_time,
        breakout_altitude_margin=breakout_altitude_margin,
    )

    def steering_cb(t, st):
        if simulator.state_machine.get_state() == FlightState.FINAL_APPROACH:
            if not waypoint_manager.is_complete:
                waypoint_manager.skip_to_final_target()
                guidance.pid.reset()
        pos = LocalPoint(st.x, st.y)
        out = guidance.update(position=pos, heading=st.heading, altitude=st.altitude, dt=simulator.dt)
        return out.steering

    history = simulator.run(steering_function=steering_cb, max_time=max_time)
    return simulator, history


def run_terminal_orbit_scenario(
    target: LocalPoint = LocalPoint(1200.0, 0.0),
    radius: float = 30.0,
    orbit_bearing_from_target: float = 320.0,
    loiter_direction: LoiterDirection = LoiterDirection.CLOCKWISE,
    initial_altitude: float = 220.0,
    breakout_altitude_margin: float = 2.0,
    max_terminal_guidance_time: float = 120.0,
    max_time: float = 200.0,
) -> Tuple[ParafoilSimulator, List[SimulationSample]]:
    """
    Execute simulation initialized directly in terminal guidance around the target orbit.

    Position is placed at distance `radius` and bearing `orbit_bearing_from_target` from target.
    Initial heading is aligned tangent to the orbit for the given direction.
    """
    # In navigation convention: +X = East, +Y = North
    # bearing beta from target: dx = r * sin(beta), dy = r * cos(beta)
    rad_beta = math.radians(orbit_bearing_from_target)
    px = target.x + radius * math.sin(rad_beta)
    py = target.y + radius * math.cos(rad_beta)

    # Tangent heading
    if loiter_direction == LoiterDirection.CLOCKWISE:
        tangent_heading = (orbit_bearing_from_target + 90.0) % 360.0
    else:
        tangent_heading = (orbit_bearing_from_target - 90.0) % 360.0

    state = ParafoilState(
        x=px,
        y=py,
        altitude=initial_altitude,
        vx=0.0,
        vy=0.0,
        vz=0.0,
        heading=tangent_heading,
        heading_rate=0.0,
    )

    simulator = ParafoilSimulator(
        state=state,
        drogue_config=DrogueConfig(descent_rate=17.0, horizontal_speed=0.0),
        parafoil_config=ParafoilConfig(forward_speed=8.0, descent_rate=2.5, max_turn_rate=30.0),
        dt=0.1,
        parafoil_deployment_altitude=600.0,
        target_position=target,
        final_approach_altitude=120.0,
        final_approach_radius=100.0,
        terminal_margin=100.0,
        terminal_capture_radius=200.0,
        loiter_radius=30.0,  # nominal commanded radius
        loiter_capture_tolerance=5.0,
        loiter_direction=loiter_direction,
        max_terminal_guidance_time=max_terminal_guidance_time,
        breakout_altitude_margin=breakout_altitude_margin,
    )

    # Set state machine directly into TERMINAL_GUIDANCE
    simulator.state_machine.state = FlightState.TERMINAL_GUIDANCE

    waypoint_manager = WaypointManager(waypoints=[target], acceptance_radius=25.0)
    pid = PIDController(
        kp=0.022,
        ki=0.00015,
        kd=0.004,
        output_min=-1.0,
        output_max=1.0,
        integral_min=-200.0,
        integral_max=200.0,
    )
    guidance = GuidanceController(
        waypoint_manager=waypoint_manager,
        pid=pid,
        final_approach_altitude=120.0,
        final_approach_radius=100.0,
    )

    def steering_cb(t, st):
        if simulator.state_machine.get_state() == FlightState.FINAL_APPROACH:
            if not waypoint_manager.is_complete:
                waypoint_manager.skip_to_final_target()
                guidance.pid.reset()
        pos = LocalPoint(st.x, st.y)
        out = guidance.update(position=pos, heading=st.heading, altitude=st.altitude, dt=simulator.dt)
        return out.steering

    history = simulator.run(steering_function=steering_cb, max_time=max_time)
    return simulator, history


def compute_metrics(
    group: str,
    scenario_id: str,
    description: str,
    target: LocalPoint,
    sim: ParafoilSimulator,
    history: List[SimulationSample],
) -> ScenarioResult:
    """Calculate the 13 required metrics from simulation execution and classify performance."""
    final_s = history[-1]
    landing_pos = LocalPoint(final_s.x, final_s.y)
    touchdown_err = distance(landing_pos, target)

    # Closest horizontal approach & altitude at that point
    dists = [distance(LocalPoint(s.x, s.y), target) for s in history]
    min_dist_idx = min(range(len(dists)), key=lambda i: dists[i])
    closest_approach = dists[min_dist_idx]
    target_crossing_alt = history[min_dist_idx].altitude

    # Breakout metrics
    brk_alt = sim.breakout_altitude
    brk_dist = sim.breakout_distance
    brk_hdg = sim.breakout_heading

    # Terminal duration
    if sim.terminal_entry_time is not None:
        if sim.breakout_time is not None:
            term_dur = sim.breakout_time - sim.terminal_entry_time
        else:
            term_dur = final_s.time - sim.terminal_entry_time
    else:
        term_dur = 0.0

    orbit_cnt = sim.loiter_orbits_completed

    if sim.loiter_capture_time is not None and sim.terminal_entry_time is not None:
        cap_time = sim.loiter_capture_time - sim.terminal_entry_time
    else:
        cap_time = None

    # Steady-state orbit tracking metrics (strictly ORBIT_TRACKING phase samples)
    tracking_samples = [
        s for s in history if s.state == FlightState.TERMINAL_GUIDANCE and s.loiter_phase == LoiterPhase.ORBIT_TRACKING
    ]
    if tracking_samples:
        rad_errs = [abs(s.loiter_radial_error) for s in tracking_samples if s.loiter_radial_error is not None]
        mean_steady_err = (sum(rad_errs) / len(rad_errs)) if rad_errs else 0.0
        max_steady_err = max(rad_errs) if rad_errs else 0.0
    else:
        mean_steady_err = 0.0
        max_steady_err = 0.0

    # Max steering during terminal & final approach
    term_app_steer = [
        abs(s.steering)
        for s in history
        if s.state in (FlightState.TERMINAL_GUIDANCE, FlightState.FINAL_APPROACH) and s.steering is not None
    ]
    max_steer = max(term_app_steer) if term_app_steer else 0.0

    # Bearing reversals during FINAL_APPROACH
    final_app_samples = [s for s in history if s.state == FlightState.FINAL_APPROACH]
    reversals = 0
    if len(final_app_samples) >= 2:
        prev_b = bearing(LocalPoint(final_app_samples[0].x, final_app_samples[0].y), target)
        for s in final_app_samples[1:]:
            curr_b = bearing(LocalPoint(s.x, s.y), target)
            delta_b = abs(heading_error(prev_b, curr_b))
            if delta_b > 90.0:
                reversals += 1
            prev_b = curr_b

    failsafe = (sim.state_machine.get_state() == FlightState.FAILSAFE) or any(
        s.state == FlightState.FAILSAFE for s in history
    )

    # Performance classification
    classification = "nominal / acceptable"
    reasons = []

    if failsafe:
        classification = "failed"
        reasons.append("failsafe activated")
    elif touchdown_err > 35.0:
        classification = "failed"
        reasons.append(f"excessive touchdown error ({touchdown_err:.2f} m > 35 m)")
    elif touchdown_err > 15.0 or reversals > 0 or max_steady_err > 8.0:
        classification = "degraded"
        if touchdown_err > 15.0:
            reasons.append(f"touchdown error ({touchdown_err:.2f} m) in 15-35m degraded band")
        if reversals > 0:
            reasons.append(f"{reversals} target bearing reversal(s) during final approach")
        if max_steady_err > 8.0:
            reasons.append(f"elevated steady-state radial tracking error ({max_steady_err:.2f} m)")
    else:
        reasons.append(f"clean convergence ({touchdown_err:.2f} m touchdown error, zero reversals, stable tracking)")

    reason_str = "; ".join(reasons)

    return ScenarioResult(
        group=group,
        scenario_id=scenario_id,
        description=description,
        touchdown_error=touchdown_err,
        closest_approach=closest_approach,
        target_crossing_altitude=target_crossing_alt,
        breakout_altitude=brk_alt,
        breakout_distance=brk_dist,
        breakout_heading=brk_hdg,
        terminal_duration=term_dur,
        orbit_count=orbit_cnt,
        capture_time=cap_time,
        mean_steady_radial_error=mean_steady_err,
        max_steady_radial_error=max_steady_err,
        max_steering_command=max_steer,
        bearing_reversals=reversals,
        failsafe=failsafe,
        classification=classification,
        classification_reason=reason_str,
        final_x=final_s.x,
        final_y=final_s.y,
        trajectory_x=[s.x for s in history],
        trajectory_y=[s.y for s in history],
    )


def run_all_scenarios() -> List[ScenarioResult]:
    """Execute all scenario matrices across Groups A-E."""
    results: List[ScenarioResult] = []
    target = LocalPoint(1200.0, 0.0)

    print("Executing Scenario Group A: Breakout Margin Sensitivity...")
    margins = [0.0, 1.0, 2.0, 3.0, 5.0, 10.0]
    for m in margins:
        sim, hist = run_full_mission(breakout_altitude_margin=m)
        res = compute_metrics("Group A", f"A_margin_{m:.0f}", f"Breakout margin = {m:.1f} m", target, sim, hist)
        results.append(res)
        print(f"  Margin {m:4.1f}m -> Touchdown Error: {res.touchdown_error:5.2f}m [{res.classification}]")

    print("\nExecuting Scenario Group B: Loiter Radius Sensitivity...")
    radii = [20.0, 25.0, 30.0, 35.0, 40.0]
    for r in radii:
        sim, hist = run_full_mission(loiter_radius=r)
        res = compute_metrics("Group B", f"B_radius_{r:.0f}", f"Loiter radius = {r:.1f} m", target, sim, hist)
        results.append(res)
        print(f"  Radius {r:4.1f}m -> Touchdown Error: {res.touchdown_error:5.2f}m [{res.classification}]")

    print("\nExecuting Scenario Group C: Loiter Direction Symmetry...")
    directions = [
        (LoiterDirection.CLOCKWISE, "CW"),
        (LoiterDirection.COUNTER_CLOCKWISE, "CCW"),
    ]
    for d, label in directions:
        sim, hist = run_full_mission(loiter_direction=d)
        res = compute_metrics("Group C", f"C_dir_{label}", f"Direction = {label}", target, sim, hist)
        results.append(res)
        print(f"  Direction {label:3s} -> Touchdown Error: {res.touchdown_error:5.2f}m [{res.classification}]")

    print("\nExecuting Scenario Group D: Orbit Phase & Initial Condition Perturbations...")
    nominal_bearing = 320.0
    phase_offsets = [0.0, 45.0, 90.0, 180.0]
    for d, d_label in directions:
        for p in phase_offsets:
            bearing_angle = (nominal_bearing + p) % 360.0
            sim, hist = run_terminal_orbit_scenario(
                target=target,
                radius=30.0,
                orbit_bearing_from_target=bearing_angle,
                loiter_direction=d,
                initial_altitude=220.0,
                breakout_altitude_margin=2.0,
            )
            scen_id = f"D_phase_{d_label}_{p:+.0f}"
            desc = f"{d_label} orbit phase {p:+.0f}° (bearing {bearing_angle:.0f}°)"
            res = compute_metrics("Group D", scen_id, desc, target, sim, hist)
            results.append(res)
            print(f"  {d_label} Phase {p:+4.0f}° -> Touchdown Error: {res.touchdown_error:5.2f}m [{res.classification}]")

    # Multi-cardinal full-mission approaches
    print("\nExecuting Scenario Group D (Cardinal Approaches)...")
    approaches = [
        (LocalPoint(0.0, 0.0), "West (nominal)"),
        (LocalPoint(1200.0, -1200.0), "South"),
        (LocalPoint(2400.0, 0.0), "East"),
        (LocalPoint(1200.0, 1200.0), "North"),
    ]
    for start_pt, app_label in approaches:
        sim, hist = run_full_mission(start=start_pt, target=target)
        scen_id = f"D_approach_{app_label.split()[0]}"
        desc = f"Full mission approach from {app_label}"
        res = compute_metrics("Group D (Approach)", scen_id, desc, target, sim, hist)
        results.append(res)
        print(f"  Approach from {app_label:14s} -> Touchdown Error: {res.touchdown_error:5.2f}m [{res.classification}]")

    print("\nExecuting Scenario Group E: Small Radial Tracking Offsets...")
    radial_offsets = [25.0, 30.0, 35.0]
    for d, d_label in directions:
        for r_off in radial_offsets:
            sim, hist = run_terminal_orbit_scenario(
                target=target,
                radius=r_off,
                orbit_bearing_from_target=nominal_bearing,
                loiter_direction=d,
                initial_altitude=220.0,
                breakout_altitude_margin=2.0,
            )
            scen_id = f"E_radial_{d_label}_r{r_off:.0f}"
            desc = f"{d_label} initial radius r = {r_off:.1f} m"
            res = compute_metrics("Group E", scen_id, desc, target, sim, hist)
            results.append(res)
            print(f"  {d_label} Radius {r_off:4.1f}m -> Touchdown Error: {res.touchdown_error:5.2f}m [{res.classification}]")

    return results


def export_csv(results: List[ScenarioResult], filename: str = "zero_wind_robustness_results.csv") -> None:
    """Write structured scenario results to CSV."""
    fieldnames = [
        "group",
        "scenario_id",
        "description",
        "touchdown_error_m",
        "closest_approach_m",
        "target_crossing_altitude_m",
        "breakout_altitude_m",
        "breakout_distance_m",
        "breakout_heading_deg",
        "terminal_duration_s",
        "orbit_count",
        "capture_time_s",
        "mean_steady_radial_error_m",
        "max_steady_radial_error_m",
        "max_steering_command",
        "bearing_reversals",
        "failsafe",
        "classification",
        "classification_reason",
    ]
    with open(filename, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)
        for r in results:
            writer.writerow([
                r.group,
                r.scenario_id,
                r.description,
                f"{r.touchdown_error:.2f}",
                f"{r.closest_approach:.2f}",
                f"{r.target_crossing_altitude:.2f}",
                f"{r.breakout_altitude:.2f}" if r.breakout_altitude is not None else "N/A",
                f"{r.breakout_distance:.2f}" if r.breakout_distance is not None else "N/A",
                f"{r.breakout_heading:.2f}" if r.breakout_heading is not None else "N/A",
                f"{r.terminal_duration:.2f}",
                f"{r.orbit_count:.2f}",
                f"{r.capture_time:.2f}" if r.capture_time is not None else "N/A",
                f"{r.mean_steady_radial_error:.2f}",
                f"{r.max_steady_radial_error:.2f}",
                f"{r.max_steering_command:.2f}",
                r.bearing_reversals,
                r.failsafe,
                r.classification,
                r.classification_reason,
            ])
    print(f"\nSaved structured scenario results to {filename}")


def plot_results(results: List[ScenarioResult], filename: str = "zero_wind_robustness.png") -> None:
    """Generate multi-panel analytical visualization of validation results."""
    target = LocalPoint(1200.0, 0.0)
    fig = plt.figure(figsize=(16, 12))

    # Panel 1: Landing positions relative to target
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.scatter([target.x], [target.y], marker="*", s=200, color="gold", edgecolors="black", zorder=5, label="Target (0,0)")
    c25 = Circle((target.x, target.y), 25.0, fill=False, linestyle="--", color="gray", label="Acceptance (25m)")
    c10 = Circle((target.x, target.y), 10.0, fill=False, linestyle=":", color="green", label="Inner Target (10m)")
    ax1.add_patch(c25)
    ax1.add_patch(c10)

    # Color by group
    group_colors = {
        "Group A": "tab:blue",
        "Group B": "tab:orange",
        "Group C": "tab:purple",
        "Group D": "tab:green",
        "Group D (Approach)": "tab:cyan",
        "Group E": "tab:red",
    }
    for grp, color in group_colors.items():
        grp_res = [r for r in results if r.group == grp]
        if grp_res:
            xs = [r.final_x for r in grp_res]
            ys = [r.final_y for r in grp_res]
            ax1.scatter(xs, ys, color=color, s=40, label=grp, alpha=0.8)

    ax1.set_xlabel("East [m]")
    ax1.set_ylabel("North [m]")
    ax1.set_title("Touchdown Dispersion Relative to Landing Target")
    ax1.set_xlim(target.x - 40, target.x + 40)
    ax1.set_ylim(target.y - 40, target.y + 40)
    ax1.set_aspect("equal")
    ax1.grid(True)
    ax1.legend(loc="upper right", fontsize=8)

    # Panel 2: Group A Breakout Margin Sensitivity
    ax2 = fig.add_subplot(2, 2, 2)
    res_a = [r for r in results if r.group == "Group A"]
    margins = [float(r.scenario_id.split("_")[-1]) for r in res_a]
    errors_a = [r.touchdown_error for r in res_a]
    brk_alts_a = [r.breakout_altitude for r in res_a]

    ax2.plot(margins, errors_a, marker="o", color="tab:blue", lw=2, label="Touchdown Error [m]")
    ax2.set_xlabel("Breakout Altitude Margin [m]")
    ax2.set_ylabel("Touchdown Error [m]", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")
    ax2.set_title("Group A: Breakout Margin Sensitivity")
    ax2.grid(True)

    ax2_twin = ax2.twinx()
    ax2_twin.plot(margins, brk_alts_a, marker="s", color="tab:red", linestyle="--", lw=1.5, label="Breakout Altitude [m]")
    ax2_twin.set_ylabel("Breakout Altitude [m]", color="tab:red")
    ax2_twin.tick_params(axis="y", labelcolor="tab:red")

    # Panel 3: Group B Loiter Radius Sensitivity
    ax3 = fig.add_subplot(2, 2, 3)
    res_b = [r for r in results if r.group == "Group B"]
    radii = [float(r.scenario_id.split("_")[-1]) for r in res_b]
    errors_b = [r.touchdown_error for r in res_b]
    orbits_b = [r.orbit_count for r in res_b]

    ax3.plot(radii, errors_b, marker="o", color="tab:orange", lw=2, label="Touchdown Error [m]")
    ax3.set_xlabel("Loiter Radius [m]")
    ax3.set_ylabel("Touchdown Error [m]", color="tab:orange")
    ax3.tick_params(axis="y", labelcolor="tab:orange")
    ax3.set_title("Group B: Loiter Radius Sensitivity")
    ax3.grid(True)

    ax3_twin = ax3.twinx()
    ax3_twin.plot(radii, orbits_b, marker="^", color="tab:green", linestyle="--", lw=1.5, label="Orbits Completed")
    ax3_twin.set_ylabel("Orbits Completed", color="tab:green")
    ax3_twin.tick_params(axis="y", labelcolor="tab:green")

    # Panel 4: Error comparison across all scenario groups
    ax4 = fig.add_subplot(2, 2, 4)
    labels = [r.scenario_id for r in results]
    errors = [r.touchdown_error for r in results]
    colors = [
        "tab:green" if r.classification == "nominal / acceptable" else ("tab:orange" if r.classification == "degraded" else "tab:red")
        for r in results
    ]

    bars = ax4.barh(range(len(results)), errors, color=colors, height=0.7)
    ax4.axvline(25.0, color="gray", linestyle="--", label="Target Acceptance (25m)")
    ax4.axvline(15.0, color="green", linestyle=":", label="Nominal Threshold (15m)")
    ax4.set_yticks(range(len(results)))
    ax4.set_yticklabels(labels, fontsize=7)
    ax4.set_xlabel("Touchdown Error [m]")
    ax4.set_title("Touchdown Error Across All Evaluated Scenarios")
    ax4.grid(True, axis="x")
    ax4.invert_yaxis()
    ax4.legend(loc="lower right", fontsize=8)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"Saved analytical validation plot to {filename}")


if __name__ == "__main__":
    results = run_all_scenarios()
    export_csv(results)
    plot_results(results)
