"""
Comprehensive Phase 4.4c Comparative Benchmark across all required evaluations:
1. Core scenarios & Multiple speeds (0, 2, 4, 6 m/s)
2. Direction reversal & 8 compass sweep (0, 45, 90, 135, 180, 225, 270, 315 deg)
3. Terminal-overlapping deterministic gust (active t in [130, 180] s)
4. Multi-seed stochastic evaluation (N=20 seeds, tau=1.0s and tau=20.0s)
5. Actuator dynamics comparison (with realistic lag, slew rate, deadband)
"""
import sys
sys.path.insert(0, ".")

import math
import numpy as np
import random
from typing import Dict, Tuple, Optional, List

from navigation.coordinates import LocalPoint
from navigation.distance import distance
from navigation.bearing import bearing
from guidance.loiter_guidance import LoiterDirection
from environment.wind import ConstantWindField, CosineGust, ComposedWindField, StochasticWind
from flight.parafoil import ParafoilState, DrogueConfig, ParafoilConfig
from flight.state_machine import FlightState
from control.pid import PIDController
from control.actuator import ActuatorModel, ActuatorConfig
from guidance.guidance_controller import GuidanceController
from navigation.waypoints import generate_waypoints
from navigation.waypoint_manager import WaypointManager
from simulation.simulator import ParafoilSimulator
from guidance.loiter_center_planner import LoiterCenterPlanner, LoiterCenterPlannerConfig


def run_sim(
    wind_field,
    planner: Optional[LoiterCenterPlanner] = None,
    actuator_model: Optional[ActuatorModel] = None,
) -> dict:
    target = LocalPoint(1200.0, 0.0)
    wps = generate_waypoints(LocalPoint(0.0, 0.0), target, num_waypoints=10)
    wpm = WaypointManager(wps, acceptance_radius=25.0)

    pid = PIDController(
        kp=10.0,
        ki=0.1,
        kd=0.5,
        output_min=-1.0,
        output_max=1.0,
        integral_min=-10.0,
        integral_max=10.0,
    )
    guidance = GuidanceController(wpm, pid)

    sim = ParafoilSimulator(
        state=ParafoilState(x=0.0, y=0.0, altitude=1000.0, vx=0.0, vy=0.0, vz=0.0, heading=0.0, heading_rate=0.0),
        drogue_config=DrogueConfig(descent_rate=17.0),
        parafoil_config=ParafoilConfig(forward_speed=8.0, descent_rate=2.5, max_turn_rate=30.0),
        dt=0.1,
        parafoil_deployment_altitude=600.0,
        target_position=target,
        wind_field=wind_field,
        actuator_model=actuator_model,
        loiter_center_planner=planner,
    )

    def steering_cb(t, s):
        if sim.state_machine.get_state() == FlightState.FINAL_APPROACH:
            if not wpm.is_complete:
                wpm.skip_to_final_target()
                guidance.pid.reset()
        pos = LocalPoint(s.x, s.y)
        out = guidance.update(position=pos, heading=s.heading, altitude=s.altitude, dt=sim.dt)
        return out.steering

    history = sim.run(steering_cb, max_time=300.0)
    last = history[-1]
    td_err = distance(LocalPoint(last.x, last.y), target)

    fa_samples = [s for s in history if s.state == FlightState.FINAL_APPROACH]
    bo_pos = (fa_samples[0].x, fa_samples[0].y) if fa_samples else (float('nan'), float('nan'))
    bo_alt = fa_samples[0].altitude if fa_samples else float('nan')
    bo_hdg = fa_samples[0].heading if fa_samples else float('nan')
    bo_dist = distance(LocalPoint(bo_pos[0], bo_pos[1]), target) if fa_samples else float('nan')

    effective_dir = sim.loiter_guidance.direction if sim.loiter_guidance else None

    return {
        "td_err": td_err,
        "final_pos": (last.x, last.y),
        "final_state": last.state.name,
        "orbits": sim.loiter_orbits_completed,
        "effective_dir": effective_dir.name if effective_dir else "NONE",
        "bo_alt": bo_alt,
        "bo_dist": bo_dist,
        "bo_hdg": bo_hdg,
        "wind_est": (sim.wind_estimator.wind_x, sim.wind_estimator.wind_y),
    }


def main():
    planner_a = None # Baseline A
    planner_b = LoiterCenterPlanner(LoiterCenterPlannerConfig(
        enabled=True,
        bias_gain=2.5,
        max_offset=30.0,
        direction="upwind",
        min_wind_speed=0.5,
        latch_on_entry=True,
    ))

    actuator = ActuatorModel(ActuatorConfig(
        delay_seconds=0.15,
        slew_rate_limit=2.5,
        deadband_positive=0.03,
        deadband_negative=0.03,
        min_command=-1.0,
        max_command=1.0,
    ))

    print("==================================================================", flush=True)
    print("1. 8-POINT COMPASS SWEEP (4.0 m/s)", flush=True)
    print("==================================================================", flush=True)
    compass = [
        ("North (+Y, 0 deg)", 0.0, 4.0),
        ("NE (45 deg)", 4.0 * math.sin(math.radians(45)), 4.0 * math.cos(math.radians(45))),
        ("East (+X, 90 deg)", 4.0, 0.0),
        ("SE (135 deg)", 4.0 * math.sin(math.radians(135)), 4.0 * math.cos(math.radians(135))),
        ("South (-Y, 180 deg)", 0.0, -4.0),
        ("SW (225 deg)", 4.0 * math.sin(math.radians(225)), 4.0 * math.cos(math.radians(225))),
        ("West (-X, 270 deg)", -4.0, 0.0),
        ("NW (315 deg)", 4.0 * math.sin(math.radians(315)), 4.0 * math.cos(math.radians(315))),
    ]

    print(f"{'Direction':22s} | {'Cand A (Base)':13s} | {'Cand B (Heur)':13s} | {'Delta B-A':10s}", flush=True)
    print("-" * 65, flush=True)
    for name, wx, wy in compass:
        planner_b.reset()
        res_a = run_sim(ConstantWindField(wx, wy), planner=None)
        res_b = run_sim(ConstantWindField(wx, wy), planner=planner_b)
        print(f"{name:22s} | {res_a['td_err']:10.2f}m   | {res_b['td_err']:10.2f}m   | {res_b['td_err'] - res_a['td_err']:+8.2f}m", flush=True)

    print("\n==================================================================", flush=True)
    print("2. WIND SPEED SWEEP (0, 2, 4, 6 m/s in +Y and -Y)", flush=True)
    print("==================================================================", flush=True)
    speed_cases = [
        ("0 m/s Zero Wind", 0.0, 0.0),
        ("+Y 2 m/s", 0.0, 2.0),
        ("+Y 4 m/s", 0.0, 4.0),
        ("+Y 6 m/s", 0.0, 6.0),
        ("-Y 2 m/s", 0.0, -2.0),
        ("-Y 4 m/s", 0.0, -4.0),
        ("-Y 6 m/s", 0.0, -6.0),
        ("+X 2 m/s (Tailwind)", 2.0, 0.0),
        ("+X 4 m/s (Tailwind)", 4.0, 0.0),
        ("+X 6 m/s (Tailwind)", 6.0, 0.0),
    ]
    print(f"{'Case':22s} | {'Cand A (Base)':13s} | {'Cand B (Heur)':13s} | {'Delta B-A':10s}", flush=True)
    print("-" * 65, flush=True)
    for name, wx, wy in speed_cases:
        planner_b.reset()
        res_a = run_sim(ConstantWindField(wx, wy), planner=None)
        res_b = run_sim(ConstantWindField(wx, wy), planner=planner_b)
        print(f"{name:22s} | {res_a['td_err']:10.2f}m   | {res_b['td_err']:10.2f}m   | {res_b['td_err'] - res_a['td_err']:+8.2f}m", flush=True)

    print("\n==================================================================", flush=True)
    print("3. TERMINAL-OVERLAPPING DETERMINISTIC GUST", flush=True)
    print("==================================================================", flush=True)
    # Gust active during t in [130, 180] s (terminal loiter and breakout phase)
    # CosineGust(gust_x, gust_y, t_start, duration)
    # Base wind = +Y 2 m/s, Gust = +Y 3 m/s during [135, 165] s
    gust_terminal = CosineGust(gust_x=0.0, gust_y=3.0, t_start=135.0, duration=30.0)
    wf_gust_term = ComposedWindField(ConstantWindField(0.0, 2.0), gust_terminal)
    planner_b.reset()
    res_a_gust = run_sim(wf_gust_term, planner=None)
    res_b_gust = run_sim(wf_gust_term, planner=planner_b)
    print(f"Base +Y 2m/s + Terminal Gust (+Y 3m/s, t=135-165s):", flush=True)
    print(f"  Cand A (Base): {res_a_gust['td_err']:.2f}m | End: ({res_a_gust['final_pos'][0]:.1f}, {res_a_gust['final_pos'][1]:.1f}) | bo_alt: {res_a_gust['bo_alt']:.1f}m", flush=True)
    print(f"  Cand B (Heur): {res_b_gust['td_err']:.2f}m | End: ({res_b_gust['final_pos'][0]:.1f}, {res_b_gust['final_pos'][1]:.1f}) | bo_alt: {res_b_gust['bo_alt']:.1f}m", flush=True)
    print(f"  Delta B-A:     {res_b_gust['td_err'] - res_a_gust['td_err']:+.2f}m", flush=True)

    # Southward Terminal Gust (-Y)
    gust_term_south = CosineGust(gust_x=0.0, gust_y=-3.0, t_start=135.0, duration=30.0)
    wf_gust_south = ComposedWindField(ConstantWindField(0.0, -2.0), gust_term_south)
    planner_b.reset()
    res_a_gsouth = run_sim(wf_gust_south, planner=None)
    res_b_gsouth = run_sim(wf_gust_south, planner=planner_b)
    print(f"Base -Y 2m/s + Terminal Gust (-Y 3m/s, t=135-165s):", flush=True)
    print(f"  Cand A (Base): {res_a_gsouth['td_err']:.2f}m | End: ({res_a_gsouth['final_pos'][0]:.1f}, {res_a_gsouth['final_pos'][1]:.1f}) | bo_alt: {res_a_gsouth['bo_alt']:.1f}m", flush=True)
    print(f"  Cand B (Heur): {res_b_gsouth['td_err']:.2f}m | End: ({res_b_gsouth['final_pos'][0]:.1f}, {res_b_gsouth['final_pos'][1]:.1f}) | bo_alt: {res_b_gsouth['bo_alt']:.1f}m", flush=True)
    print(f"  Delta B-A:     {res_b_gsouth['td_err'] - res_a_gsouth['td_err']:+.2f}m", flush=True)

    print("\n==================================================================", flush=True)
    print("4. ACTUATOR DYNAMICS IMPACT (Realistic delay, slew rate, deadband)", flush=True)
    print("==================================================================", flush=True)
    act_cases = [
        ("Zero Wind", ConstantWindField(0.0, 0.0)),
        ("+Y 4m/s", ConstantWindField(0.0, 4.0)),
        ("-Y 4m/s", ConstantWindField(0.0, -4.0)),
        ("Tailwind +X", ConstantWindField(4.0, 0.0)),
    ]
    print(f"{'Case':18s} | {'A (Ideal)':10s} | {'A (Actuator)':12s} | {'B (Ideal)':10s} | {'B (Actuator)':12s}", flush=True)
    print("-" * 72, flush=True)
    for name, wf in act_cases:
        planner_b.reset()
        res_a_id = run_sim(wf, planner=None, actuator_model=None)
        res_a_act = run_sim(wf, planner=None, actuator_model=actuator)
        planner_b.reset()
        res_b_id = run_sim(wf, planner=planner_b, actuator_model=None)
        planner_b.reset()
        res_b_act = run_sim(wf, planner=planner_b, actuator_model=actuator)
        print(f"{name:18s} | {res_a_id['td_err']:8.2f}m  | {res_a_act['td_err']:10.2f}m  | {res_b_id['td_err']:8.2f}m  | {res_b_act['td_err']:10.2f}m", flush=True)

    print("\n==================================================================", flush=True)
    print("5. MULTI-SEED STOCHASTIC EVALUATION (N=20 Seeds)", flush=True)
    print("==================================================================", flush=True)
    seeds = [100 + i for i in range(20)]
    for tau in [1.0, 20.0]:
        print(f"\n--- Correlation Time tau = {tau}s (N=20 seeds) ---", flush=True)
        errs_a = []
        errs_b = []
        a_wins = 0
        b_wins = 0
        for s in seeds:
            wf = ComposedWindField(ConstantWindField(0.0, 0.0), StochasticWind(std_dev=1.5, correlation_time=tau, seed=s))
            wf.reset()
            r_a = run_sim(wf, planner=None)
            wf.reset()
            planner_b.reset()
            r_b = run_sim(wf, planner=planner_b)
            errs_a.append(r_a['td_err'])
            errs_b.append(r_b['td_err'])
            if r_a['td_err'] < r_b['td_err']:
                a_wins += 1
            else:
                b_wins += 1

        print(f"  Candidate A (Base): Mean={np.mean(errs_a):.2f}m | Median={np.median(errs_a):.2f}m | Std={np.std(errs_a):.2f}m | <=10m: {sum(1 for e in errs_a if e <= 10.0)/20*100:.1f}%", flush=True)
        print(f"  Candidate B (Heur): Mean={np.mean(errs_b):.2f}m | Median={np.median(errs_b):.2f}m | Std={np.std(errs_b):.2f}m | <=10m: {sum(1 for e in errs_b if e <= 10.0)/20*100:.1f}%", flush=True)
        print(f"  Pairwise Superiority: Cand A won {a_wins}/20 ({a_wins*5}%), Cand B won {b_wins}/20 ({b_wins*5}%)", flush=True)


if __name__ == "__main__":
    main()
