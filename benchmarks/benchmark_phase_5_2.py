"""
Phase 5.2 Sensor Faults, Dropouts, and Robust State Estimation Benchmark Harness.

Compares:
1. Ideal Baseline (Ground Truth navigation)
2. Noisy Direct (Raw noisy / degraded sensor measurements fed to guidance)
3. Reference EKF (Phase 5.1 EKF: NIS gating disabled, no stale rejection, no DR Q-inflation)
4. Robust EKF (Phase 5.2 EKF: NIS gating enabled, stale rejection enabled, DR Q-inflation=2.5)

Evaluates 10 scenarios:
- S1: Nominal Baseline (zero wind, seed 42)
- S2: Short GNSS Outage (5s outage during Precision Guidance)
- S3: Extended GNSS Outage (15s outage during Precision Guidance)
- S4: GNSS Outage during Terminal Loiter (10s outage triggered upon TERMINAL_GUIDANCE)
- S5: GNSS Position Outlier Burst (3 outlier spikes: +40m, -50m, +60m)
- S6: Stale GNSS Freeze (5s frozen coordinates)
- S7: Barometer Altitude Bias & Linear Drift (+10m bias, +0.2 m/s drift)
- S8: Heading Sensor Spikes & Bias (+35 deg spike, +5 deg bias)
- S9: Crosswind + Combined Faults (4 m/s crosswind, 5s GNSS outage, heading spike)
- S10: Severe Compound Stress (4 m/s wind, 10s GNSS outage, baro bias, 2x noise burst)
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
from environment.wind import ConstantWindField
from sensors.fault_model import (
    SensorFaultConfig,
    GNSSOutage,
    GNSSOutlier,
    GNSSStaleInterval,
    BaroFault,
    HeadingFault,
    NoiseMultiplier,
    StateTriggeredFault,
)
from sensors.sensor_model import SensorSuite, SensorNoiseConfig
from estimation.ekf import ExtendedKalmanFilter, EKFConfig


def run_scenario(
    mode: str,  # "ideal", "noisy_direct", "reference_ekf", "robust_ekf"
    wind_field,
    fault_config: Optional[SensorFaultConfig] = None,
    seed: int = 42,
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
    sim_mode = "ideal" if mode == "ideal" else ("noisy_direct" if mode == "noisy_direct" else "noisy_ekf")

    if mode != "ideal":
        noise_cfg = SensorNoiseConfig(
            enabled=True,
            seed=seed,
            gnss_pos_std=2.5,
            gnss_vel_std=0.2,
            baro_alt_std=1.5,
            heading_std_deg=3.0,
        )
        f_cfg = fault_config if fault_config is not None else SensorFaultConfig(enabled=False)
        sensors = SensorSuite(config=noise_cfg, fault_config=f_cfg)

    if mode == "reference_ekf":
        ekf_cfg = EKFConfig(
            nis_gating_enabled=False,
            stale_rejection_enabled=False,
            dead_reckoning_q_inflation=1.0,
        )
        ekf = ExtendedKalmanFilter(ekf_cfg)
    elif mode == "robust_ekf":
        ekf_cfg = EKFConfig(
            nis_gating_enabled=True,
            nis_gate_gps_pos=16.0,
            nis_gate_gps_pv=25.0,
            nis_gate_baro=9.0,
            nis_gate_heading=9.0,
            stale_rejection_enabled=True,
            dead_reckoning_q_inflation=2.5,
            dead_reckoning_outage_delay=1.0,
        )
        ekf = ExtendedKalmanFilter(ekf_cfg)

    actuator = ActuatorModel(ActuatorConfig())

    sim = ParafoilSimulator(
        state=ParafoilState(x=0.0, y=0.0, altitude=1000.0, vx=0.0, vy=0.0, vz=0.0, heading=0.0, heading_rate=0.0),
        drogue_config=DrogueConfig(descent_rate=17.0),
        parafoil_config=ParafoilConfig(forward_speed=8.0, descent_rate=2.5, max_turn_rate=30.0),
        dt=0.1,
        parafoil_deployment_altitude=600.0,
        target_position=target,
        wind_field=wind_field,
        actuator_model=actuator,
        sensor_suite=sensors,
        ekf=ekf,
        state_estimation_mode=sim_mode,
    )

    def steering_cb(t, s):
        if sim.state_machine.get_state() == FlightState.FINAL_APPROACH:
            if not wpm.is_complete:
                wpm.skip_to_final_target()
                guidance.pid.reset()

        nav_pos, nav_hdg, nav_alt = sim.get_navigation_state()
        out = guidance.update(position=nav_pos, heading=nav_hdg, altitude=nav_alt, dt=sim.dt)
        return out.steering

    history = sim.run(steering_cb, max_time=320.0)
    last = history[-1]
    td_err = distance(LocalPoint(last.x, last.y), target)

    guided_samples = [s for s in history if s.state in {
        FlightState.PRECISION_GUIDANCE,
        FlightState.TERMINAL_GUIDANCE,
        FlightState.FINAL_APPROACH,
    }]

    pos_rmse = 0.0
    peak_pos_err = 0.0
    alt_rmse = 0.0
    hdg_rmse = 0.0

    if ekf is not None and guided_samples:
        pos_sq = []
        pos_errs = []
        alt_sq = []
        hdg_sq = []
        for s in guided_samples:
            if s.estimated_x is not None:
                dx = s.estimated_x - s.x
                dy = s.estimated_y - s.y
                dalt = s.estimated_altitude - s.altitude
                dhdg = heading_error(s.heading, s.estimated_heading)
                err_2d = math.hypot(dx, dy)
                pos_errs.append(err_2d)
                pos_sq.append(dx * dx + dy * dy)
                alt_sq.append(dalt * dalt)
                hdg_sq.append(dhdg * dhdg)
        if pos_sq:
            pos_rmse = float(np.sqrt(np.mean(pos_sq)))
            peak_pos_err = float(max(pos_errs))
            alt_rmse = float(np.sqrt(np.mean(alt_sq)))
            hdg_rmse = float(np.sqrt(np.mean(hdg_sq)))

    rej_gps = ekf.rejected_gps_count if ekf is not None else 0
    rej_baro = ekf.rejected_baro_count if ekf is not None else 0
    rej_hdg = ekf.rejected_heading_count if ekf is not None else 0

    return {
        "mode": mode,
        "td_err": td_err,
        "final_state": last.state.name,
        "pos_rmse": pos_rmse,
        "peak_pos_err": peak_pos_err,
        "alt_rmse": alt_rmse,
        "hdg_rmse": hdg_rmse,
        "rej_gps": rej_gps,
        "rej_baro": rej_baro,
        "rej_hdg": rej_hdg,
        "mean_radial_err": sim.mean_radial_error_during_loiter,
        "orbits": sim.loiter_orbits_completed,
    }


def main():
    print("=" * 105)
    print("CAN-7USAT PHASE 5.2 SENSOR FAULTS, DROPOUTS & ROBUST STATE ESTIMATION BENCHMARK")
    print("=" * 105)

    scenarios = [
        (
            "S1: Nominal Baseline",
            ConstantWindField(0.0, 0.0),
            SensorFaultConfig(enabled=False),
        ),
        (
            "S2: Short GNSS Outage (5s)",
            ConstantWindField(0.0, 0.0),
            SensorFaultConfig(
                enabled=True,
                gnss_outages=[GNSSOutage(start_time=35.0, end_time=40.0, drop_position=True, drop_velocity=True)],
            ),
        ),
        (
            "S3: Extended GNSS Outage (15s)",
            ConstantWindField(0.0, 0.0),
            SensorFaultConfig(
                enabled=True,
                gnss_outages=[GNSSOutage(start_time=35.0, end_time=50.0, drop_position=True, drop_velocity=True)],
            ),
        ),
        (
            "S4: Loiter GNSS Outage (10s)",
            ConstantWindField(0.0, 0.0),
            SensorFaultConfig(
                enabled=True,
                state_triggered_faults=[
                    StateTriggeredFault(
                        trigger_state=FlightState.TERMINAL_GUIDANCE,
                        duration_seconds=10.0,
                        delay_seconds=2.0,
                        fault_type="gnss_outage",
                    )
                ],
            ),
        ),
        (
            "S5: GNSS Outlier Burst (3x)",
            ConstantWindField(0.0, 0.0),
            SensorFaultConfig(
                enabled=True,
                gnss_outliers=[
                    GNSSOutlier(time=35.0, duration=0.2, offset_x=40.0, offset_y=30.0),
                    GNSSOutlier(time=38.0, duration=0.2, offset_x=-50.0, offset_y=20.0),
                    GNSSOutlier(time=41.0, duration=0.2, offset_x=60.0, offset_y=-40.0),
                ],
            ),
        ),
        (
            "S6: Stale GNSS Freeze (5s)",
            ConstantWindField(0.0, 0.0),
            SensorFaultConfig(
                enabled=True,
                gnss_stale_intervals=[GNSSStaleInterval(start_time=35.0, end_time=40.0)],
            ),
        ),
        (
            "S7: Baro Bias + Drift",
            ConstantWindField(0.0, 0.0),
            SensorFaultConfig(
                enabled=True,
                baro_faults=[BaroFault(start_time=0.0, bias=10.0, drift_rate=0.2)],
            ),
        ),
        (
            "S8: Heading Spikes + Bias",
            ConstantWindField(0.0, 0.0),
            SensorFaultConfig(
                enabled=True,
                heading_faults=[
                    HeadingFault(
                        start_time=0.0,
                        bias_deg=5.0,
                        spike_time=40.0,
                        spike_mag_deg=35.0,
                        spike_duration=0.5,
                    )
                ],
            ),
        ),
        (
            "S9: Crosswind (4m/s) + Combined",
            ConstantWindField(0.0, 4.0),
            SensorFaultConfig(
                enabled=True,
                gnss_outages=[GNSSOutage(start_time=35.0, end_time=40.0, drop_position=True, drop_velocity=True)],
                heading_faults=[
                    HeadingFault(
                        start_time=0.0,
                        bias_deg=2.0,
                        spike_time=42.0,
                        spike_mag_deg=30.0,
                        spike_duration=0.4,
                    )
                ],
            ),
        ),
        (
            "S10: Severe Compound Stress",
            ConstantWindField(0.0, 4.0),
            SensorFaultConfig(
                enabled=True,
                gnss_outages=[GNSSOutage(start_time=35.0, end_time=45.0, drop_position=True, drop_velocity=True)],
                baro_faults=[BaroFault(start_time=0.0, bias=8.0, drift_rate=0.0)],
                noise_multipliers=[NoiseMultiplier(start_time=30.0, end_time=50.0, multiplier=2.0, sensor="all")],
            ),
        ),
    ]

    modes = ["ideal", "noisy_direct", "reference_ekf", "robust_ekf"]

    # Header
    print(f"{'Scenario':<32} | {'Mode':<14} | {'TD Err [m]':<10} | {'Pos RMSE':<9} | {'Peak Err':<9} | {'Hdg RMSE':<9} | {'Alt RMSE':<9} | {'Rej GPS/Baro/Hdg':<16} | {'State':<8}")
    print("-" * 125)

    all_results = {}

    for name, wind, fault in scenarios:
        all_results[name] = {}
        for m in modes:
            res = run_scenario(m, wind, fault, seed=42)
            all_results[name][m] = res
            rej_str = f"{res['rej_gps']}/{res['rej_baro']}/{res['rej_hdg']}"
            pos_str = f"{res['pos_rmse']:.2f} m" if m.endswith("ekf") else "—"
            peak_str = f"{res['peak_pos_err']:.2f} m" if m.endswith("ekf") else "—"
            hdg_str = f"{res['hdg_rmse']:.2f}°" if m.endswith("ekf") else "—"
            alt_str = f"{res['alt_rmse']:.2f} m" if m.endswith("ekf") else "—"
            print(f"{name:<32} | {m:<14} | {res['td_err']:<10.2f} | {pos_str:<9} | {peak_str:<9} | {hdg_str:<9} | {alt_str:<9} | {rej_str:<16} | {res['final_state']:<8}")
        print("-" * 125)

    print("\nBenchmark completed successfully.")


if __name__ == "__main__":
    main()
