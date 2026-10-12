"""
Unit and regression tests for Phase 5.2 Sensor Faults, Dropouts, and Robust State Estimation.

Tests verify:
1. Sensor fault models: deterministic time intervals, outages, outliers, bias, drift, and noise multipliers.
2. State-triggered fault transitions anchored to simulator flight phases.
3. EKF NIS (Normalized Innovation Squared) statistical gating and outlier rejection.
4. Reference EKF degradation under outliers when gating is disabled.
5. Measurement staleness and timestamp handling.
6. Dead-reckoning process noise covariance inflation during sustained GNSS outages.
7. Closed-loop simulator robustness with sensor degradation.
"""

import math
import pytest
import numpy as np

from flight.parafoil import ParafoilConfig, ParafoilState, DrogueConfig
from flight.state_machine import FlightState, FlightStateMachine
from navigation.coordinates import LocalPoint
from navigation.distance import distance
from navigation.waypoints import generate_waypoints
from navigation.waypoint_manager import WaypointManager
from control.pid import PIDController
from control.actuator import ActuatorModel, ActuatorConfig
from guidance.guidance_controller import GuidanceController
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
from sensors.sensor_model import SensorNoiseConfig, SensorSuite, SensorMeasurement
from estimation.ekf import ExtendedKalmanFilter, EKFConfig, EKFStateEstimate


class TestSensorFaultModels:
    """Tests for fault configuration and synthetic degradation logic."""

    def test_default_config_disabled(self):
        cfg = SensorFaultConfig()
        assert not cfg.enabled
        assert not cfg.is_gnss_outage(10.0)[0]
        assert not cfg.is_gnss_stale(10.0)
        assert cfg.get_gnss_outlier(10.0) == (0.0, 0.0, 0.0, 0.0)
        assert not cfg.is_baro_outage(10.0)
        assert cfg.get_baro_error(10.0) == 0.0
        assert not cfg.is_heading_outage(10.0)
        assert cfg.get_heading_error_deg(10.0) == 0.0
        assert cfg.get_noise_multiplier(10.0, "gnss") == 1.0

    def test_gnss_outage_boundaries(self):
        cfg = SensorFaultConfig(
            enabled=True,
            gnss_outages=[GNSSOutage(start_time=10.0, end_time=20.0, drop_position=True, drop_velocity=True)],
        )
        assert cfg.is_gnss_outage(9.9) == (False, False)
        assert cfg.is_gnss_outage(10.0) == (True, True)
        assert cfg.is_gnss_outage(15.0) == (True, True)
        assert cfg.is_gnss_outage(20.0) == (True, True)
        assert cfg.is_gnss_outage(20.1) == (False, False)

    def test_gnss_outlier_injection(self):
        cfg = SensorFaultConfig(
            enabled=True,
            gnss_outliers=[
                GNSSOutlier(time=5.0, duration=0.5, offset_x=30.0, offset_y=-40.0, offset_vx=2.0, offset_vy=0.0)
            ],
        )
        assert cfg.get_gnss_outlier(4.9) == (0.0, 0.0, 0.0, 0.0)
        ox, oy, ovx, ovy = cfg.get_gnss_outlier(5.2)
        assert ox == 30.0
        assert oy == -40.0
        assert ovx == 2.0
        assert ovy == 0.0
        assert cfg.get_gnss_outlier(5.6) == (0.0, 0.0, 0.0, 0.0)

    def test_baro_bias_and_drift(self):
        cfg = SensorFaultConfig(
            enabled=True,
            baro_faults=[BaroFault(start_time=10.0, end_time=30.0, bias=5.0, drift_rate=0.5)],
        )
        assert cfg.get_baro_error(5.0) == 0.0
        assert pytest.approx(cfg.get_baro_error(10.0), abs=1e-6) == 5.0
        assert pytest.approx(cfg.get_baro_error(20.0), abs=1e-6) == 10.0  # 5.0 + 0.5 * 10
        assert cfg.get_baro_error(30.1) == 0.0

    def test_heading_spike(self):
        cfg = SensorFaultConfig(
            enabled=True,
            heading_faults=[
                HeadingFault(
                    start_time=10.0,
                    end_time=20.0,
                    bias_deg=2.0,
                    spike_time=15.0,
                    spike_mag_deg=45.0,
                    spike_duration=0.3,
                )
            ],
        )
        assert pytest.approx(cfg.get_heading_error_deg(12.0), abs=1e-6) == 2.0
        # Spike active at t=15.1
        assert pytest.approx(cfg.get_heading_error_deg(15.1), abs=1e-6) == 47.0
        # Spike ended at t=15.5
        assert pytest.approx(cfg.get_heading_error_deg(15.5), abs=1e-6) == 2.0

    def test_state_triggered_fault_activation(self):
        cfg = SensorFaultConfig(
            enabled=True,
            state_triggered_faults=[
                StateTriggeredFault(
                    trigger_state=FlightState.PRECISION_GUIDANCE,
                    duration_seconds=5.0,
                    delay_seconds=2.0,
                    fault_type="gnss_outage",
                )
            ],
        )
        assert len(cfg.gnss_outages) == 0

        # Unrelated transition
        cfg.notify_state_transition(FlightState.PARAFOIL_DEPLOYMENT, transition_time=10.0)
        assert len(cfg.gnss_outages) == 0

        # Target transition
        cfg.notify_state_transition(FlightState.PRECISION_GUIDANCE, transition_time=12.0)
        assert len(cfg.gnss_outages) == 1
        out = cfg.gnss_outages[0]
        assert out.start_time == 14.0  # 12.0 + 2.0
        assert out.end_time == 19.0    # 14.0 + 5.0


class TestEKFRobustnessAndGating:
    """Tests for EKF NIS statistical gating, staleness detection, and covariance inflation."""

    def test_nis_gating_rejects_large_gps_outlier(self):
        config = EKFConfig(
            nis_gating_enabled=True,
            nis_gate_gps_pos=16.0,  # ~4-sigma
        )
        ekf = ExtendedKalmanFilter(config)
        ekf.reset(init_x=0.0, init_y=0.0, init_alt=600.0, init_heading_deg=0.0)

        # 1. Normal measurement near initial state
        accepted = ekf.correct_gps(meas_x=0.5, meas_y=0.2, meas_time=0.1)
        assert accepted
        assert ekf.rejected_gps_count == 0
        assert ekf.last_nis_gps < 16.0

        # 2. Large 80m outlier measurement
        accepted_outlier = ekf.correct_gps(meas_x=80.0, meas_y=-50.0, meas_time=0.3)
        assert not accepted_outlier
        assert ekf.rejected_gps_count == 1
        assert ekf.last_nis_gps > 16.0
        # Verify state x, y was NOT pulled to outlier
        assert abs(ekf.x[0]) < 2.0
        assert abs(ekf.x[1]) < 2.0

        # 3. Followed by normal measurement
        accepted_normal = ekf.correct_gps(meas_x=1.0, meas_y=0.5, meas_time=0.5)
        assert accepted_normal
        assert ekf.rejected_gps_count == 1  # count did not increase
        assert abs(ekf.x[0] - 1.0) < 1.0

    def test_reference_ekf_contaminates_on_outlier_when_gating_disabled(self):
        # Reference EKF: nis_gating_enabled=False
        config = EKFConfig(nis_gating_enabled=False)
        ekf = ExtendedKalmanFilter(config)
        ekf.reset(init_x=0.0, init_y=0.0, init_alt=600.0, init_heading_deg=0.0)

        # Large 80m outlier measurement
        accepted = ekf.correct_gps(meas_x=80.0, meas_y=0.0, meas_time=0.2)
        assert accepted  # Gating disabled: unconditionally accepted
        assert ekf.rejected_gps_count == 0
        # Verify state was heavily pulled by outlier
        assert ekf.x[0] > 10.0

    def test_stale_measurement_rejected(self):
        config = EKFConfig(stale_rejection_enabled=True)
        ekf = ExtendedKalmanFilter(config)
        ekf.reset(init_x=0.0, init_y=0.0, init_alt=600.0)

        # First measurement at t=1.0
        acc1 = ekf.correct_gps(meas_x=5.0, meas_y=5.0, meas_time=1.0)
        assert acc1
        assert ekf.rejected_gps_count == 0

        # Stale measurement with delta_t == 0 (identical timestamp)
        acc2 = ekf.correct_gps(meas_x=5.0, meas_y=5.0, meas_time=1.0)
        assert not acc2
        assert ekf.rejected_gps_count == 1

        # Fresh measurement at t=1.2
        acc3 = ekf.correct_gps(meas_x=6.0, meas_y=6.0, meas_time=1.2)
        assert acc3
        assert ekf.rejected_gps_count == 1

    def test_baro_outlier_rejection(self):
        config = EKFConfig(
            nis_gating_enabled=True,
            nis_gate_baro=9.0,  # ~3-sigma
            r_baro_alt=1.5,
        )
        ekf = ExtendedKalmanFilter(config)
        ekf.reset(init_alt=500.0)

        # Normal measurement
        assert ekf.correct_baro(meas_alt=501.0, meas_time=0.1)
        assert ekf.rejected_baro_count == 0

        # Massive 50m baro altitude spike
        assert not ekf.correct_baro(meas_alt=550.0, meas_time=0.2)
        assert ekf.rejected_baro_count == 1
        assert abs(ekf.x[2] - 500.0) < 3.0  # Altitude protected from spike

    def test_heading_spike_rejection(self):
        config = EKFConfig(
            nis_gating_enabled=True,
            nis_gate_heading=9.0,  # ~3-sigma
            r_heading_deg=3.0,
        )
        ekf = ExtendedKalmanFilter(config)
        ekf.reset(init_heading_deg=90.0)

        # Normal heading measurement (91 deg)
        assert ekf.correct_heading(meas_heading_deg=91.0, meas_time=0.1)
        assert ekf.rejected_heading_count == 0

        # 90-degree heading spike (180 deg)
        assert not ekf.correct_heading(meas_heading_deg=180.0, meas_time=0.2)
        assert ekf.rejected_heading_count == 1
        # Heading state should not have jumped to 180 deg
        est_hdg = math.degrees(ekf.x[3])
        assert abs(est_hdg - 90.0) < 5.0

    def test_covariance_inflation_during_extended_outage(self):
        config = EKFConfig(
            dead_reckoning_q_inflation=3.0,
            dead_reckoning_outage_delay=1.0,
            q_pos=0.2,
        )
        ekf = ExtendedKalmanFilter(config)
        ekf.reset(init_x=0.0, init_y=0.0, init_alt=600.0, init_time=0.0)

        # Establish last GNSS fix at t=1.0
        ekf.correct_gps(meas_x=0.0, meas_y=0.0, meas_time=1.0)
        p_initial = ekf.P[0, 0]

        # Predict within delay (< 1.0s outage): normal Q
        for _ in range(5):
            ekf.predict(dt=0.1, steering_cmd=0.0)
        p_mid = ekf.P[0, 0]
        rate_normal = (p_mid - p_initial) / 5

        # Predict past outage delay (t > 2.0s): inflated Q
        for _ in range(10):
            ekf.predict(dt=0.1, steering_cmd=0.0)
        p_inflated_start = ekf.P[0, 0]

        for _ in range(5):
            ekf.predict(dt=0.1, steering_cmd=0.0)
        p_inflated_end = ekf.P[0, 0]
        rate_inflated = (p_inflated_end - p_inflated_start) / 5

        # Covariance growth rate during extended outage must be strictly greater than normal
        assert rate_inflated > rate_normal

        # Re-introduce GNSS fix: verify covariance collapses safely and filter remains PSD
        ekf.correct_gps(meas_x=5.0, meas_y=10.0, meas_time=ekf._time)
        assert ekf.P[0, 0] < p_inflated_end
        assert np.all(np.linalg.eigvals(ekf.P) > 0.0)


class TestSensorSuiteWithFaults:
    """Tests for SensorSuite integration with synthetic faults."""

    def test_sensor_suite_stale_gnss_retains_coordinates(self):
        fault_cfg = SensorFaultConfig(
            enabled=True,
            gnss_stale_intervals=[GNSSStaleInterval(start_time=2.0, end_time=4.0)],
        )
        noise_cfg = SensorNoiseConfig(enabled=True, seed=42, gnss_dt=0.2)
        suite = SensorSuite(config=noise_cfg, fault_config=fault_cfg)

        # Before stale interval: fresh measurements moving with vehicle
        m1 = suite.update(time=1.0, true_x=10.0, true_y=20.0, true_altitude=500.0, true_vx=8.0, true_vy=0.0, true_heading_deg=90.0)
        assert m1.gnss_valid
        pos1_x = m1.gnss_x

        # During stale interval: coordinates frozen at previous value
        m2 = suite.update(time=2.2, true_x=20.0, true_y=20.0, true_altitude=490.0, true_vx=8.0, true_vy=0.0, true_heading_deg=90.0)
        assert m2.gnss_valid
        assert m2.gnss_x == pos1_x
        # Timestamp must be held frozen
        assert m2.gnss_timestamp == 1.0


class TestStatisticalGatingAndRecoveryAudit:
    """
    Phase 5.2b Statistical Gating Calibration and Fault-Recovery Audit regression tests.
    Verifies:
    1. Zero nominal GPS false rejections across full mission phases.
    2. Statistical chi-square nominal rejection rates for 1-DOF sensor updates.
    3. Natural outage recovery through covariance inflation without blind force-acceptance.
    4. 100% rejection of persistent outliers when force-acceptance is disabled.
    5. Single-sensor vertical drift observability limitations.
    """

    def test_nominal_gps_zero_false_rejections_under_full_mission(self):
        target = LocalPoint(1200.0, 0.0)
        wps = generate_waypoints(LocalPoint(0.0, 0.0), target, num_waypoints=10)
        wpm = WaypointManager(wps, acceptance_radius=25.0)
        pid = PIDController(kp=10.0, ki=0.1, kd=0.5, output_min=-1.0, output_max=1.0, integral_min=-10.0, integral_max=10.0)
        guidance = GuidanceController(wpm, pid)

        noise_cfg = SensorNoiseConfig(enabled=True, seed=42, gnss_pos_std=2.5, gnss_vel_std=0.2, baro_alt_std=1.5, heading_std_deg=3.0)
        fault_cfg = SensorFaultConfig(enabled=False)
        sensors = SensorSuite(config=noise_cfg, fault_config=fault_cfg)

        ekf_cfg = EKFConfig(
            nis_gating_enabled=True,
            nis_gate_gps_pos=16.0,
            nis_gate_gps_pv=25.0,
            nis_gate_baro=9.0,
            nis_gate_heading=9.0,
            stale_rejection_enabled=True,
            dead_reckoning_q_inflation=10.0,
            dead_reckoning_outage_delay=1.0,
            max_consecutive_rejections=0,
        )
        ekf = ExtendedKalmanFilter(ekf_cfg)
        sim = ParafoilSimulator(
            state=ParafoilState(x=0.0, y=0.0, altitude=1000.0, vx=0.0, vy=0.0, vz=0.0, heading=0.0, heading_rate=0.0),
            drogue_config=DrogueConfig(descent_rate=17.0),
            parafoil_config=ParafoilConfig(forward_speed=8.0, descent_rate=2.5, max_turn_rate=30.0),
            dt=0.1,
            parafoil_deployment_altitude=600.0,
            target_position=target,
            wind_field=ConstantWindField(0.0, 0.0),
            actuator_model=ActuatorModel(ActuatorConfig()),
            sensor_suite=sensors,
            ekf=ekf,
            state_estimation_mode="noisy_ekf",
        )

        def steering_cb(t, s):
            if sim.state_machine.get_state() == FlightState.FINAL_APPROACH:
                if not wpm.is_complete:
                    wpm.skip_to_final_target()
                    guidance.pid.reset()
            nav_pos, nav_hdg, nav_alt = sim.get_navigation_state()
            return guidance.update(position=nav_pos, heading=nav_hdg, altitude=nav_alt, dt=sim.dt).steering

        sim.run(steering_cb, max_time=320.0)

        # Under proper drogue and deployment kinematic alignment, nominal GPS false rejections must be exactly 0
        assert ekf.rejected_gps_count == 0

    def test_statistical_chisq_nominal_rejection_rates(self):
        # Under Gaussian noise, a 1-DOF chi-square gate of 9.0 corresponds to 3-sigma bilateral tail: P(chi^2 >= 9) = 0.00270 (0.27%)
        rng = np.random.RandomState(12345)
        N = 5000
        std_baro = 1.5
        noise = rng.normal(0.0, std_baro, size=N)

        ekf = ExtendedKalmanFilter(EKFConfig(nis_gating_enabled=True, nis_gate_baro=9.0, r_baro_alt=std_baro))
        ekf.reset(init_alt=500.0)

        rejections = 0
        for i in range(N):
            # True altitude is steady at 500.0, meas has unbiased Gaussian noise
            meas_alt = 500.0 + noise[i]
            accepted = ekf.correct_baro(meas_alt=meas_alt, meas_time=0.1 * (i + 1))
            if not accepted:
                rejections += 1
            # Maintain steady state altitude in EKF for scalar test
            ekf.x[2] = 500.0

        rejection_rate = rejections / N
        # Empirical rate should be close to 0.27% (between 0.08% and 0.60%)
        assert 4 <= rejections <= 30
        assert 0.0008 <= rejection_rate <= 0.0060

    def test_outage_recovery_pure_covariance_inflation(self):
        target = LocalPoint(1200.0, 0.0)
        wps = generate_waypoints(LocalPoint(0.0, 0.0), target, num_waypoints=10)
        wpm = WaypointManager(wps, acceptance_radius=25.0)
        pid = PIDController(kp=10.0, ki=0.1, kd=0.5, output_min=-1.0, output_max=1.0, integral_min=-10.0, integral_max=10.0)
        guidance = GuidanceController(wpm, pid)

        noise_cfg = SensorNoiseConfig(enabled=True, seed=42, gnss_pos_std=2.5, gnss_vel_std=0.2, baro_alt_std=1.5, heading_std_deg=3.0)
        # 15s GNSS outage between t=35.0 and t=50.0
        fault_cfg = SensorFaultConfig(enabled=True, gnss_outages=[GNSSOutage(start_time=35.0, end_time=50.0, drop_position=True, drop_velocity=True)])
        sensors = SensorSuite(config=noise_cfg, fault_config=fault_cfg)

        ekf_cfg = EKFConfig(
            nis_gating_enabled=True,
            nis_gate_gps_pos=16.0,
            nis_gate_gps_pv=25.0,
            stale_rejection_enabled=True,
            dead_reckoning_q_inflation=15.0,
            dead_reckoning_outage_delay=1.0,
            max_consecutive_rejections=0,  # Strict zero force-acceptance
        )
        ekf = ExtendedKalmanFilter(ekf_cfg)
        sim = ParafoilSimulator(
            state=ParafoilState(x=0.0, y=0.0, altitude=1000.0, vx=0.0, vy=0.0, vz=0.0, heading=0.0, heading_rate=0.0),
            drogue_config=DrogueConfig(descent_rate=17.0),
            parafoil_config=ParafoilConfig(forward_speed=8.0, descent_rate=2.5, max_turn_rate=30.0),
            dt=0.1,
            parafoil_deployment_altitude=600.0,
            target_position=target,
            wind_field=ConstantWindField(0.0, 0.0),
            actuator_model=ActuatorModel(ActuatorConfig()),
            sensor_suite=sensors,
            ekf=ekf,
            state_estimation_mode="noisy_ekf",
        )

        def steering_cb(t, s):
            if sim.state_machine.get_state() == FlightState.FINAL_APPROACH:
                if not wpm.is_complete:
                    wpm.skip_to_final_target()
                    guidance.pid.reset()
            nav_pos, nav_hdg, nav_alt = sim.get_navigation_state()
            return guidance.update(position=nav_pos, heading=nav_hdg, altitude=nav_alt, dt=sim.dt).steering

        history = sim.run(steering_cb, max_time=320.0)
        last = history[-1]
        td_err = distance(LocalPoint(last.x, last.y), target)

        # Re-acquisition succeeded purely via statistical covariance inflation
        assert td_err < 5.0
        # Post-outage fixes re-acquired without infinite lockout (rejections bounded to transition window)
        assert ekf.rejected_gps_count < 200

    def test_persistent_outlier_never_accepted_with_force_acceptance_disabled(self):
        ekf_cfg = EKFConfig(
            nis_gating_enabled=True,
            nis_gate_gps_pos=16.0,
            nis_gate_gps_pv=25.0,
            max_consecutive_rejections=0,  # Force acceptance disabled
        )
        ekf = ExtendedKalmanFilter(ekf_cfg)
        ekf.reset(init_x=10.0, init_y=20.0, init_alt=500.0)

        # Deliver 50 consecutive large GPS outliers (offset by 100m)
        for i in range(50):
            t = 1.0 + 0.1 * i
            ekf.predict(0.1, 0.0)
            accepted = ekf.correct_gps(meas_x=110.0, meas_y=120.0, meas_time=t)
            assert not accepted

        # Exactly 50 rejections, 0 accepted
        assert ekf.rejected_gps_count == 50
        assert ekf.consecutive_rejected_gps == 50
        # State position should remain unaffected by the 100m outlier stream
        assert abs(ekf.x[0] - 10.0) < 15.0

    def test_barometer_drift_observability_limitation(self):
        # Demonstrates single-sensor unobservability of altitude drift
        ekf = ExtendedKalmanFilter(EKFConfig(nis_gating_enabled=True, nis_gate_baro=9.0, r_baro_alt=1.5))
        ekf.reset(init_alt=500.0)

        # Slow 0.2 m/s unmodeled drift
        for i in range(200):
            t = 0.1 * (i + 1)
            ekf.predict(0.1, 0.0)
            # True altitude is 500.0 - 2.5*t, but baro has +0.2*t drift
            true_alt = 500.0 - 2.5 * t
            meas_alt = true_alt + 0.2 * t
            ekf.correct_baro(meas_alt, t)

        # EKF tracks the drifted baro because no second vertical reference exists
        assert abs(ekf.x[2] - (500.0 - 2.5 * 20.0 + 0.2 * 20.0)) < 2.0

