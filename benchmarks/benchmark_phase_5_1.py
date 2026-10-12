"""
Phase 5.1 Comparative Benchmark Harness:
Evaluates Ideal Baseline, Noisy-Direct, and Noisy-EKF across:
1. Zero Wind
2. Feasible Crosswind (+Y 2.0 m/s and +Y 4.0 m/s)
3. Tailwind (+X 2.0 m/s)
4. Terminal-Overlapping Deterministic Gust
5. Headwind Reachability Shortfall (-X 4.0 m/s)
6. Multi-Seed Stochastic Wind (N=20 seeds, tau=1.0s and tau=20.0s)
7. Noise Sensitivity Sweep (Low, Nominal, High)

Reports estimation RMSE, heading error, touchdown error, loiter radial error,
breakout altitude, and failsafe/landing events.
"""
import sys
sys.path.insert(0, ".")

import math
import numpy as np
from typing import Dict, List, Optional, Tuple

from navigation.coordinates import LocalPoint
from navigation.distance import distance
from navigation.bearing import bearing
from navigation.heading import heading_error
from flight.parafoil import ParafoilState, DrogueConfig, ParafoilConfig
from flight.state_machine import FlightState
from control.pid import PIDController
from control.actuator import ActuatorModel, ActuatorConfig
from guidance.guidance_controller import GuidanceController
from navigation.waypoints import generate_waypoints
from navigation.waypoint_manager import WaypointManager
from simulation.simulator import ParafoilSimulator
from environment.wind import (
    ConstantWindField,
    CosineGust,
    ComposedWindField,
    StochasticWind,
)
from sensors.sensor_model import SensorSuite, SensorNoiseConfig
from estimation.ekf import ExtendedKalmanFilter, EKFConfig


def run_single_simulation(
    mode: str,  # "ideal", "noisy_direct", "noisy_ekf"
    wind_field,
    seed: int = 42,
    sensor_config: Optional[SensorNoiseConfig] = None,
    ekf_config: Optional[EKFConfig] = None,
    actuator_model: Optional[ActuatorModel] = None,
) -> Dict:
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

    sensors = None
    ekf = None

    if mode != "ideal":
        if sensor_config is None:
            sensor_config = SensorNoiseConfig(
                enabled=True,
                seed=seed,
                gnss_pos_std=2.5,
                gnss_vel_std=0.2,
                baro_alt_std=1.5,
                heading_std_deg=3.0,
            )
        else:
            sensor_config.seed = seed
            sensor_config.enabled = True
        sensors = SensorSuite(sensor_config)

    if mode == "noisy_ekf":
        if ekf_config is None:
            ekf_config = EKFConfig(enabled=True)
        else:
            ekf_config.enabled = True
        ekf = ExtendedKalmanFilter(ekf_config)

    sim = ParafoilSimulator(
        state=ParafoilState(x=0.0, y=0.0, altitude=1000.0, vx=0.0, vy=0.0, vz=0.0, heading=0.0, heading_rate=0.0),
        drogue_config=DrogueConfig(descent_rate=17.0),
        parafoil_config=ParafoilConfig(forward_speed=8.0, descent_rate=2.5, max_turn_rate=30.0),
        dt=0.1,
        parafoil_deployment_altitude=600.0,
        target_position=target,
        wind_field=wind_field,
        actuator_model=actuator_model,
        sensor_suite=sensors,
        ekf=ekf,
        state_estimation_mode=mode,
    )

    def steering_cb(t, s):
        if sim.state_machine.get_state() == FlightState.FINAL_APPROACH:
            if not wpm.is_complete:
                wpm.skip_to_final_target()
                guidance.pid.reset()

        nav_pos, nav_hdg, nav_alt = sim.get_navigation_state()
        out = guidance.update(position=nav_pos, heading=nav_hdg, altitude=nav_alt, dt=sim.dt)
        return out.steering

    history = sim.run(steering_cb, max_time=300.0)
    last = history[-1]
    td_err = distance(LocalPoint(last.x, last.y), target)

    # Estimation errors across history (during guided flight phases)
    guided_samples = [s for s in history if s.state in {
        FlightState.PRECISION_GUIDANCE,
        FlightState.TERMINAL_GUIDANCE,
        FlightState.FINAL_APPROACH,
    }]

    pos_rmse = 0.0
    alt_rmse = 0.0
    hdg_rmse = 0.0

    if mode == "noisy_ekf" and guided_samples:
        pos_sq = []
        alt_sq = []
        hdg_sq = []
        for s in guided_samples:
            if s.estimated_x is not None:
                dx = s.estimated_x - s.x
                dy = s.estimated_y - s.y
                dalt = s.estimated_altitude - s.altitude
                dhdg = heading_error(s.heading, s.estimated_heading)
                pos_sq.append(dx * dx + dy * dy)
                alt_sq.append(dalt * dalt)
                hdg_sq.append(dhdg * dhdg)
        if pos_sq:
            pos_rmse = float(np.sqrt(np.mean(pos_sq)))
            alt_rmse = float(np.sqrt(np.mean(alt_sq)))
            hdg_rmse = float(np.sqrt(np.mean(hdg_sq)))

    fa_samples = [s for s in history if s.state == FlightState.FINAL_APPROACH]
    bo_alt = fa_samples[0].altitude if fa_samples else float("nan")

    return {
        "mode": mode,
        "td_err": td_err,
        "final_pos": (last.x, last.y),
        "final_state": last.state.name,
        "pos_rmse": pos_rmse,
        "alt_rmse": alt_rmse,
        "hdg_rmse": hdg_rmse,
        "mean_radial_err": sim.mean_radial_error_during_loiter,
        "max_radial_err": sim.max_radial_error_during_loiter,
        "bo_alt": bo_alt,
        "orbits": sim.loiter_orbits_completed,
    }


def main():
    print("=" * 80)
    print("PHASE 5.1 COMPREHENSIVE BENCHMARK: IDEAL vs NOISY-DIRECT vs NOISY-EKF")
    print("=" * 80)

    modes = ["ideal", "noisy_direct", "noisy_ekf"]

    # 1. Deterministic Core Scenarios
    scenarios = [
        ("Zero Wind", ConstantWindField(0.0, 0.0)),
        ("Crosswind (+Y 2 m/s)", ConstantWindField(0.0, 2.0)),
        ("Crosswind (+Y 4 m/s)", ConstantWindField(0.0, 4.0)),
        ("Tailwind (+X 2 m/s)", ConstantWindField(2.0, 0.0)),
        ("Headwind Shortfall (-X 4 m/s)", ConstantWindField(-4.0, 0.0)),
    ]

    print("\n--- 1. Deterministic Core Scenarios ---")
    print(f"{'Scenario':<30} | {'Mode':<14} | {'TD Err [m]':<10} | {'Pos RMSE':<9} | {'Hdg RMSE':<9} | {'Loiter R Err':<12} | {'State':<10}")
    print("-" * 105)

    for name, wf in scenarios:
        for m in modes:
            res = run_single_simulation(m, wf, seed=42)
            print(f"{name:<30} | {m:<14} | {res['td_err']:<10.2f} | {res['pos_rmse']:<9.2f} | {res['hdg_rmse']:<9.2f} | {res['mean_radial_err']:<12.2f} | {res['final_state']:<10}")

    # 2. Terminal-Overlapping Gust Scenario
    print("\n--- 2. Terminal-Overlapping Gust Scenario ---")
    base_wf = ConstantWindField(0.0, 2.0)
    gust = CosineGust(gust_x=0.0, gust_y=3.0, t_start=140.0, duration=40.0)
    gust_wf = ComposedWindField(base_wf, gust)

    for m in modes:
        res = run_single_simulation(m, gust_wf, seed=42)
        print(f"{'Terminal Gust (+Y 3m/s)':<30} | {m:<14} | {res['td_err']:<10.2f} | {res['pos_rmse']:<9.2f} | {res['hdg_rmse']:<9.2f} | {res['mean_radial_err']:<12.2f} | {res['final_state']:<10}")

    # 3. Multi-Seed Stochastic Monte Carlo (N=20 seeds)
    print("\n--- 3. Multi-Seed Stochastic Monte Carlo (N=20 Seeds) ---")
    seeds = list(range(100, 120))

    for tau_val, label in [(1.0, "High-Freq (tau=1.0s)"), (20.0, "Low-Freq (tau=20.0s)")]:
        print(f"\nTurbulence: {label}")
        results_by_mode = {m: [] for m in modes}
        pos_rmses = []
        hdg_rmses = []

        for s in seeds:
            for m in modes:
                wf = ComposedWindField(
                    ConstantWindField(0.0, 0.0),
                    StochasticWind(std_dev=1.5, correlation_time=tau_val, seed=s)
                )
                wf.reset()
                r = run_single_simulation(m, wf, seed=s)
                results_by_mode[m].append(r["td_err"])
                if m == "noisy_ekf":
                    pos_rmses.append(r["pos_rmse"])
                    hdg_rmses.append(r["hdg_rmse"])

        for m in modes:
            errs = results_by_mode[m]
            mean_e = np.mean(errs)
            med_e = np.median(errs)
            p90_e = np.percentile(errs, 90)
            under_10 = np.mean([1 if e <= 10.0 else 0 for e in errs]) * 100.0
            print(f"  {m:<14} -> Mean: {mean_e:6.2f} m | Median: {med_e:6.2f} m | 90th%: {p90_e:6.2f} m | <=10m: {under_10:5.1f}%")
        print(f"  EKF Avg Filter Performance -> Pos RMSE: {np.mean(pos_rmses):.2f} m | Hdg RMSE: {np.mean(hdg_rmses):.2f} deg")

    # 4. Sensor Noise Sensitivity Sweep (Low, Nominal, High)
    print("\n--- 4. Sensor Noise Sensitivity Sweep (+Y 2.0 m/s wind) ---")
    sens_levels = [
        ("Low Noise (1.0m pos, 1.0 deg)", SensorNoiseConfig(gnss_pos_std=1.0, gnss_vel_std=0.1, baro_alt_std=0.5, heading_std_deg=1.0)),
        ("Nominal Noise (2.5m pos, 3.0 deg)", SensorNoiseConfig(gnss_pos_std=2.5, gnss_vel_std=0.2, baro_alt_std=1.5, heading_std_deg=3.0)),
        ("High Noise (5.0m pos, 6.0 deg)", SensorNoiseConfig(gnss_pos_std=5.0, gnss_vel_std=0.4, baro_alt_std=3.0, heading_std_deg=6.0)),
    ]

    wf_cross = ConstantWindField(0.0, 2.0)
    for lvl_name, s_cfg in sens_levels:
        print(f"\nSensitivity Level: {lvl_name}")
        for m in ["noisy_direct", "noisy_ekf"]:
            r = run_single_simulation(m, wf_cross, seed=42, sensor_config=s_cfg)
            print(f"  {m:<14} -> TD Err: {r['td_err']:6.2f} m | Pos RMSE: {r['pos_rmse']:5.2f} m | Hdg RMSE: {r['hdg_rmse']:5.2f} deg | Loiter Err: {r['mean_radial_err']:5.2f} m")

    print("\n" + "=" * 80)
    print("BENCHMARK EXECUTION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
