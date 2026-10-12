"""
Comprehensive unit and regression tests for Phase 5.1:
- Sensor noise models (GNSS, Barometer, Heading sensor)
- 5-state Extended Kalman Filter (EKF)
- Numerical stability, covariance properties, Jacobians
- Heading angle wrapping across the 0/360 boundary
- Mode isolation: Ideal vs Noisy-Direct vs Noisy-EKF
- Closed-loop terminal loiter, breakout, and touchdown under estimation
"""

import math
import numpy as np
import pytest

from navigation.coordinates import LocalPoint
from sensors.sensor_model import (
    SensorNoiseConfig,
    SensorMeasurement,
    SensorSuite,
)
from estimation.ekf import (
    EKFConfig,
    EKFStateEstimate,
    ExtendedKalmanFilter,
    wrap_angle_rad,
    wrap_heading_deg,
)
from flight.parafoil import ParafoilState, ParafoilConfig, DrogueConfig
from flight.state_machine import FlightState
from simulation.simulator import ParafoilSimulator
from environment.wind import ConstantWindField


class TestSensorSuite:
    """Test suite for SensorSuite and SensorNoiseConfig."""

    def test_config_validation(self):
        """Verify invalid or non-finite noise configurations raise ValueError."""
        with pytest.raises(ValueError, match="gnss_pos_std"):
            SensorNoiseConfig(gnss_pos_std=-1.0)
        with pytest.raises(ValueError, match="gnss_vel_std"):
            SensorNoiseConfig(gnss_vel_std=float("nan"))
        with pytest.raises(ValueError, match="gnss_dt"):
            SensorNoiseConfig(gnss_dt=0.0)
        with pytest.raises(ValueError, match="baro_alt_std"):
            SensorNoiseConfig(baro_alt_std=-0.5)
        with pytest.raises(ValueError, match="heading_std_deg"):
            SensorNoiseConfig(heading_std_deg=-2.0)

    def test_ideal_passthrough_when_disabled(self):
        """When enabled=False, sensor suite returns exact ground truth without noise."""
        sensors = SensorSuite(SensorNoiseConfig(enabled=False))
        meas = sensors.update(
            time=1.0,
            true_x=123.45,
            true_y=-67.89,
            true_altitude=543.21,
            true_vx=6.5,
            true_vy=-3.2,
            true_heading_deg=285.0,
        )
        assert meas.gnss_valid is True
        assert meas.gnss_x == pytest.approx(123.45)
        assert meas.gnss_y == pytest.approx(-67.89)
        assert meas.gnss_vx == pytest.approx(6.5)
        assert meas.gnss_vy == pytest.approx(-3.2)
        assert meas.baro_valid is True
        assert meas.baro_altitude == pytest.approx(543.21)
        assert meas.heading_valid is True
        assert meas.heading_deg == pytest.approx(285.0)

    def test_deterministic_seeding(self):
        """Two SensorSuites with identical seeds produce bitwise-identical noise sequences."""
        cfg1 = SensorNoiseConfig(enabled=True, seed=42, gnss_pos_std=3.0, heading_std_deg=2.0)
        cfg2 = SensorNoiseConfig(enabled=True, seed=42, gnss_pos_std=3.0, heading_std_deg=2.0)

        s1 = SensorSuite(cfg1)
        s2 = SensorSuite(cfg2)

        for t in [0.0, 0.1, 0.2, 0.3, 0.4]:
            m1 = s1.update(t, 100.0, 200.0, 500.0, 5.0, 5.0, 45.0)
            m2 = s2.update(t, 100.0, 200.0, 500.0, 5.0, 5.0, 45.0)
            assert m1.gnss_valid == m2.gnss_valid
            assert m1.gnss_x == m2.gnss_x
            assert m1.gnss_y == m2.gnss_y
            assert m1.baro_altitude == m2.baro_altitude
            assert m1.heading_deg == m2.heading_deg

    def test_heading_wrapping_in_sensors(self):
        """Heading output must always wrap cleanly in [0, 360)."""
        cfg = SensorNoiseConfig(enabled=True, seed=123, heading_std_deg=10.0)
        sensors = SensorSuite(cfg)

        for true_hdg in [0.0, 1.0, 359.0, 360.0, 720.0, -10.0]:
            meas = sensors.update(0.1, 0.0, 0.0, 100.0, 0.0, 0.0, true_hdg)
            if meas.heading_valid:
                assert 0.0 <= meas.heading_deg < 360.0

    def test_sampling_intervals(self):
        """GNSS at 5 Hz (dt=0.2s) should only be valid every 0.2s."""
        cfg = SensorNoiseConfig(enabled=True, seed=99, gnss_dt=0.2, baro_dt=0.1, heading_dt=0.1)
        sensors = SensorSuite(cfg)

        m0 = sensors.update(0.0, 0.0, 0.0, 500.0, 0.0, 0.0, 0.0)
        assert m0.gnss_valid is True
        assert m0.baro_valid is True

        m1 = sensors.update(0.1, 0.0, 0.0, 499.7, 0.0, 0.0, 0.0)
        assert m1.gnss_valid is False
        assert m1.baro_valid is True

        m2 = sensors.update(0.2, 0.0, 0.0, 499.5, 0.0, 0.0, 0.0)
        assert m2.gnss_valid is True


class TestExtendedKalmanFilter:
    """Test suite for 5-state Extended Kalman Filter."""

    def test_ekf_config_validation(self):
        """Verify invalid parameters in EKFConfig raise ValueError."""
        with pytest.raises(ValueError, match="forward_speed"):
            EKFConfig(forward_speed=-1.0)
        with pytest.raises(ValueError, match="descent_rate"):
            EKFConfig(descent_rate=0.0)
        with pytest.raises(ValueError, match="q_pos"):
            EKFConfig(q_pos=-0.1)
        with pytest.raises(ValueError, match="r_gps_pos"):
            EKFConfig(r_gps_pos=-1.0)

    def test_heading_wrapping_helpers(self):
        """Test angle wrapping helper functions."""
        assert wrap_angle_rad(0.0) == 0.0
        assert wrap_angle_rad(math.pi) == pytest.approx(-math.pi)
        assert wrap_angle_rad(3.0 * math.pi) == pytest.approx(-math.pi)
        assert wrap_angle_rad(-math.pi) == pytest.approx(-math.pi)

        assert wrap_heading_deg(0.0) == 0.0
        assert wrap_heading_deg(360.0) == 0.0
        assert wrap_heading_deg(370.0) == pytest.approx(10.0)
        assert wrap_heading_deg(-30.0) == pytest.approx(330.0)

    def test_ekf_reset(self):
        """EKF resets correctly to prescribed state and initial diagonal covariance."""
        ekf = ExtendedKalmanFilter()
        ekf.reset(init_x=100.0, init_y=200.0, init_alt=600.0, init_heading_deg=90.0)
        est = ekf.get_estimate()

        assert est.x == pytest.approx(100.0)
        assert est.y == pytest.approx(200.0)
        assert est.altitude == pytest.approx(600.0)
        assert est.heading_deg == pytest.approx(90.0)
        assert est.sigma_x == pytest.approx(ekf.config.init_pos_std)
        assert est.sigma_alt == pytest.approx(ekf.config.init_alt_std)

    def test_ekf_prediction_pure_north(self):
        """With heading=0 (North) and zero steering, x should stay constant and y increase by Va*dt."""
        ekf = ExtendedKalmanFilter()
        ekf.reset(init_x=0.0, init_y=0.0, init_alt=1000.0, init_heading_deg=0.0)

        dt = 0.1
        ekf.predict(dt=dt, steering_cmd=0.0)
        est = ekf.get_estimate()

        assert est.x == pytest.approx(0.0, abs=1e-5)
        assert est.y == pytest.approx(8.0 * dt, abs=1e-5)
        assert est.altitude == pytest.approx(1000.0 - 2.5 * dt, abs=1e-5)
        assert est.heading_deg == pytest.approx(0.0, abs=1e-5)

    def test_ekf_prediction_pure_east(self):
        """With heading=90 (East), x should increase by Va*dt and y stay constant."""
        ekf = ExtendedKalmanFilter()
        ekf.reset(init_x=0.0, init_y=0.0, init_alt=1000.0, init_heading_deg=90.0)

        dt = 0.1
        ekf.predict(dt=dt, steering_cmd=0.0)
        est = ekf.get_estimate()

        assert est.x == pytest.approx(8.0 * dt, abs=1e-5)
        assert est.y == pytest.approx(0.0, abs=1e-5)
        assert est.heading_deg == pytest.approx(90.0, abs=1e-5)

    def test_covariance_symmetry_and_positive_definiteness(self):
        """Verify Joseph form maintains exact symmetry and positive diagonal elements."""
        ekf = ExtendedKalmanFilter()
        ekf.reset()

        for _ in range(50):
            ekf.predict(dt=0.1, steering_cmd=0.5)
            ekf.correct_gps(meas_x=10.0, meas_y=20.0, meas_vx=1.0, meas_vy=7.0)
            ekf.correct_baro(meas_alt=950.0)
            ekf.correct_heading(meas_heading_deg=15.0)

            # Symmetry check
            diff = ekf.P - ekf.P.T
            assert np.max(np.abs(diff)) < 1e-12

            # Diagonal positive definiteness
            for i in range(5):
                assert ekf.P[i, i] > 0.0

            # Eigenvalues all positive (positive definite)
            eigvals = np.linalg.eigvalsh(ekf.P)
            assert np.all(eigvals > 0.0)

    def test_heading_innovation_wrapping_at_zero_boundary(self):
        """When true heading transitions across 0/360, innovation must not jump 360 degrees."""
        ekf = ExtendedKalmanFilter()
        # Initialize at 359 deg
        ekf.reset(init_heading_deg=359.0)

        # Measurement arrives at 1.0 deg (a +2.0 deg true difference)
        ekf.correct_heading(meas_heading_deg=1.0)
        est = ekf.get_estimate()

        # Heading should rotate smoothly toward 0 / 1 deg, NOT down toward 180
        # Specifically, heading should be near 359.5 - 0.5 deg
        assert est.heading_deg > 355.0 or est.heading_deg < 5.0
        assert est.last_heading_residual_deg < 5.0

    def test_zero_measurement_noise_convergence(self):
        """With very low measurement noise, state estimate rapidly tracks measurement."""
        cfg = EKFConfig(
            r_gps_pos=0.01,
            r_baro_alt=0.01,
            r_heading_deg=0.01,
        )
        ekf = ExtendedKalmanFilter(cfg)
        ekf.reset(init_x=100.0, init_y=100.0, init_alt=500.0, init_heading_deg=0.0)

        # Measurement says target is at (50, 50, 400, 45 deg)
        for _ in range(5):
            ekf.predict(0.1, 0.0)
            ekf.correct_gps(meas_x=50.0, meas_y=50.0)
            ekf.correct_baro(meas_alt=400.0)
            ekf.correct_heading(meas_heading_deg=45.0)

        est = ekf.get_estimate()
        assert est.x == pytest.approx(50.0, abs=0.5)
        assert est.y == pytest.approx(50.0, abs=0.5)
        assert est.altitude == pytest.approx(400.0, abs=0.5)
        assert est.heading_deg == pytest.approx(45.0, abs=1.0)

    def test_jacobian_finite_difference_verification(self):
        """Verify analytical F Jacobian matches numerical finite difference."""
        ekf = ExtendedKalmanFilter()
        ekf.reset(init_x=10.0, init_y=20.0, init_alt=800.0, init_heading_deg=37.0)

        dt = 0.1
        u = 0.4
        max_rate = math.radians(ekf.config.max_turn_rate_deg_s)
        cmd_rate = u * max_rate
        tau = ekf.config.yaw_rate_tau
        Va = ekf.config.forward_speed
        Vz = ekf.config.descent_rate

        def f_step(x_vec):
            x, y, z, psi, w = x_vec
            nx = x + (Va * math.sin(psi)) * dt
            ny = y + (Va * math.cos(psi)) * dt
            nz = max(0.0, z - Vz * dt)
            npsi = wrap_angle_rad(psi + w * dt)
            nw = w + ((cmd_rate - w) / tau) * dt
            return np.array([nx, ny, nz, npsi, nw])

        x0 = ekf.x.copy()
        eps = 1e-6
        F_num = np.zeros((5, 5))

        for j in range(5):
            x_plus = x0.copy()
            x_plus[j] += eps
            x_minus = x0.copy()
            x_minus[j] -= eps
            F_num[:, j] = (f_step(x_plus) - f_step(x_minus)) / (2.0 * eps)

        # Analytical Jacobian
        psi0 = x0[3]
        F_ana = np.eye(5)
        F_ana[0, 3] = Va * math.cos(psi0) * dt
        F_ana[1, 3] = -Va * math.sin(psi0) * dt
        F_ana[3, 4] = dt
        F_ana[4, 4] = 1.0 - (dt / tau)

        np.testing.assert_allclose(F_ana, F_num, atol=1e-5)

    def test_non_finite_measurement_rejection(self):
        """EKF must gracefully ignore NaN or Infinite measurements without corrupting state or covariance."""
        ekf = ExtendedKalmanFilter()
        ekf.reset(init_x=0.0, init_y=0.0, init_alt=1000.0, init_heading_deg=0.0)

        # Inject invalid measurements
        ekf.correct_gps(float("nan"), 0.0)
        ekf.correct_gps(0.0, float("inf"))
        ekf.correct_baro(float("nan"))
        ekf.correct_heading(float("nan"))

        est = ekf.get_estimate()
        assert math.isfinite(est.x)
        assert math.isfinite(est.y)
        assert math.isfinite(est.altitude)
        assert math.isfinite(est.heading_deg)
        assert np.all(np.isfinite(ekf.P))


class TestSimulatorEstimationIntegration:
    """Test suite verifying ParafoilSimulator integration with SensorSuite and EKF."""

    def test_ideal_baseline_equivalence(self):
        """Default simulation mode='ideal' exactly matches baseline behavior."""
        state = ParafoilState(
            x=0.0, y=0.0, altitude=1000.0,
            vx=0.0, vy=0.0, vz=0.0,
            heading=0.0, heading_rate=0.0
        )
        sim = ParafoilSimulator(
            state=state,
            drogue_config=DrogueConfig(),
            parafoil_config=ParafoilConfig(),
            target_position=LocalPoint(1200.0, 0.0),
            state_estimation_mode="ideal",
        )
        pos, hdg, alt = sim.get_navigation_state()
        assert pos.x == 0.0
        assert pos.y == 0.0
        assert hdg == 0.0
        assert alt == 1000.0

    def test_invalid_mode_raises(self):
        """Attempting to instantiate simulator with invalid mode raises ValueError."""
        state = ParafoilState(0.0, 0.0, 1000.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        with pytest.raises(ValueError, match="state_estimation_mode"):
            ParafoilSimulator(
                state=state,
                drogue_config=DrogueConfig(),
                parafoil_config=ParafoilConfig(),
                state_estimation_mode="invalid_unknown_mode",
            )

    def test_noisy_ekf_flight_completion(self):
        """Run full simulation in 'noisy_ekf' mode to confirm successful landing without crash/NaN."""
        state = ParafoilState(
            x=0.0, y=0.0, altitude=1000.0,
            vx=0.0, vy=0.0, vz=0.0,
            heading=0.0, heading_rate=0.0
        )
        target = LocalPoint(1200.0, 0.0)
        sensor_cfg = SensorNoiseConfig(
            enabled=True,
            seed=42,
            gnss_pos_std=2.5,
            gnss_vel_std=0.2,
            baro_alt_std=1.5,
            heading_std_deg=3.0,
        )
        sensors = SensorSuite(sensor_cfg)
        ekf_cfg = EKFConfig(enabled=True)
        ekf = ExtendedKalmanFilter(ekf_cfg)

        sim = ParafoilSimulator(
            state=state,
            drogue_config=DrogueConfig(),
            parafoil_config=ParafoilConfig(),
            target_position=target,
            sensor_suite=sensors,
            ekf=ekf,
            state_estimation_mode="noisy_ekf",
        )

        def steering_cb(t, st):
            nav_pos, nav_hdg, _ = sim.get_navigation_state()
            from navigation.bearing import bearing
            from navigation.heading import heading_error
            des_bearing = bearing(nav_pos, target)
            err = heading_error(nav_hdg, des_bearing)
            return max(-1.0, min(1.0, err / 30.0))

        history = sim.run(steering_function=steering_cb, max_time=500.0)

        assert len(history) > 100
        final_sample = history[-1]
        assert final_sample.altitude <= 0.0
        assert final_sample.state in {FlightState.LANDING, FlightState.FINAL_APPROACH}
        assert math.isfinite(final_sample.x)
        assert math.isfinite(final_sample.y)
        assert final_sample.estimated_x is not None
        assert math.isfinite(final_sample.estimated_x)
        assert math.isfinite(final_sample.sigma_x)

    def test_ekf_phase_aware_kinematics(self):
        """Verify drogue kinematics (Va=0, Vz=17) correctly propagate without artificial forward motion."""
        ekf = ExtendedKalmanFilter()
        ekf.reset(init_x=0.0, init_y=0.0, init_alt=1000.0, init_heading_deg=0.0, forward_speed=0.0, descent_rate=17.0)
        ekf.set_wind_estimate(2.0, 3.0)

        dt = 0.1
        ekf.predict(dt=dt, steering_cmd=0.0)
        est = ekf.get_estimate()

        # East position advances purely by wind_x * dt = 2.0 * 0.1 = 0.2m (no airspeed forward flight)
        assert est.x == pytest.approx(0.2, abs=1e-5)
        # North position advances purely by wind_y * dt = 3.0 * 0.1 = 0.3m
        assert est.y == pytest.approx(0.3, abs=1e-5)
        # Altitude drops by 17.0 * 0.1 = 1.7m
        assert est.altitude == pytest.approx(998.3, abs=1e-5)
        # Heading unchanged
        assert est.heading_deg == pytest.approx(0.0, abs=1e-5)

        # GPS velocity update with Va=0 should not alter heading
        ekf.correct_gps(meas_x=0.2, meas_y=0.3, meas_vx=2.0, meas_vy=3.0)
        est_after_gps = ekf.get_estimate()
        assert est_after_gps.heading_deg == pytest.approx(0.0, abs=1e-4)

    def test_initial_seeding_at_t0(self):
        """At simulation creation (t=0), get_navigation_state must return valid initial measurements, not zeros."""
        state = ParafoilState(x=150.0, y=250.0, altitude=1000.0, vx=0.0, vy=0.0, vz=0.0, heading=45.0, heading_rate=0.0)
        sensor_cfg = SensorNoiseConfig(enabled=True, seed=123, gnss_pos_std=0.0, baro_alt_std=0.0, heading_std_deg=0.0)
        sensors = SensorSuite(sensor_cfg)
        ekf = ExtendedKalmanFilter(EKFConfig(enabled=True))

        sim = ParafoilSimulator(
            state=state,
            drogue_config=DrogueConfig(),
            parafoil_config=ParafoilConfig(),
            target_position=LocalPoint(1200.0, 0.0),
            sensor_suite=sensors,
            ekf=ekf,
            state_estimation_mode="noisy_direct",
        )
        pos, hdg, alt = sim.get_navigation_state()
        assert pos.x == pytest.approx(150.0)
        assert pos.y == pytest.approx(250.0)
        assert hdg == pytest.approx(45.0)
        assert alt == pytest.approx(1000.0)

    def test_transition_uses_nav_state(self):
        """Verify update_mission_state triggers using navigation state rather than ground truth."""
        # Vehicle ground truth is at altitude 605m (above 600m deployment), but noisy altimeter reports 598m
        state = ParafoilState(x=0.0, y=0.0, altitude=605.0, vx=0.0, vy=0.0, vz=0.0, heading=0.0, heading_rate=0.0)
        sensor_cfg = SensorNoiseConfig(enabled=True, seed=1, baro_alt_std=0.0)
        sensors = SensorSuite(sensor_cfg)

        sim = ParafoilSimulator(
            state=state,
            drogue_config=DrogueConfig(),
            parafoil_config=ParafoilConfig(),
            target_position=LocalPoint(1200.0, 0.0),
            sensor_suite=sensors,
            parafoil_deployment_altitude=600.0,
            state_estimation_mode="noisy_direct",
        )
        assert sim.state_machine.get_state() == FlightState.DROGUE_DESCENT

        # Artificially set sensor altimeter below threshold
        sensors._current_baro_alt = 598.0
        sim.update_mission_state()
        # Should transition to PARAFOIL_DEPLOYMENT based on navigation state
        assert sim.state_machine.get_state() == FlightState.PARAFOIL_DEPLOYMENT
