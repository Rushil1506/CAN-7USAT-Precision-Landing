"""
Extended Kalman Filter (EKF) for CAN-7USAT Precision Landing State Estimation.

This module provides a 5-state discrete-continuous Extended Kalman Filter:
State vector:
    x = [x, y, z, psi, w_rate]^T
where:
    x      : East position [m] (+X = East)
    y      : North position [m] (+Y = North)
    z      : Altitude above ground [m]
    psi    : Heading angle [rad] (0 = North, pi/2 = East, pi = South, 3*pi/2 = West)
    w_rate : Heading turn rate [rad/s] (positive clockwise / eastward turn)

Observability & Process Model:
- Process Model:
    dot_x = V_a * sin(psi) + W_x
    dot_y = V_a * cos(psi) + W_y
    dot_z = -V_z
    dot_psi = w_rate
    dot_w_rate = (omega_cmd - w_rate) / tau_yaw

  where omega_cmd = u * max_turn_rate_rad_s, with u in [-1, +1] (actual steering input).
  V_a is the nominal forward airspeed (8.0 m/s), V_z is nominal descent rate (2.5 m/s).
  W_x, W_y are estimated wind components or zero.
  All process dynamics use only legitimate flight software inputs and parameters.
  No ground truth is accessed.

- Measurement Model:
    z_gps  = [x_gps, y_gps, vx_gps, vy_gps]^T
    z_baro = [z_baro]^T
    z_mag  = [psi_mag]^T (with rigorous angle wrapping of innovation)

- Numerical Stability:
    Joseph form covariance update:
    P = (I - K H) P (I - K H)^T + K R K^T
    Symmetric projection P = 0.5 * (P + P^T)
    Positive semidefiniteness enforcement on diagonal elements.
"""

from dataclasses import dataclass, field
import math
from typing import Dict, List, Optional, Tuple
import numpy as np

from navigation.coordinates import LocalPoint


@dataclass
class EKFConfig:
    """
    Configuration parameters for Extended Kalman Filter.
    """
    enabled: bool = False

    # Nominal aerodynamic parameters
    forward_speed: float = 8.0  # [m/s]
    descent_rate: float = 2.5   # [m/s]
    max_turn_rate_deg_s: float = 30.0  # [deg/s]
    yaw_rate_tau: float = 0.1   # First-order yaw rate tracking time constant [s]

    # Process noise standard deviations (continuous PSD or discrete equivalent)
    q_pos: float = 0.2          # Position disturbance [m/s]
    q_alt: float = 0.1          # Altitude disturbance [m/s]
    q_psi: float = 0.05         # Heading disturbance [rad/s]
    q_yaw_rate: float = 0.5     # Yaw rate acceleration disturbance [rad/s^2]

    # Measurement noise 1-sigma standard deviations (used to form R)
    r_gps_pos: float = 2.5      # GPS position noise [m]
    r_gps_vel: float = 0.2      # GPS velocity noise [m/s]
    r_baro_alt: float = 1.5     # Barometer altitude noise [m]
    r_heading_deg: float = 3.0  # Heading sensor noise [deg]

    # Initial covariance 1-sigma standard deviations
    init_pos_std: float = 5.0
    init_alt_std: float = 3.0
    init_psi_std_deg: float = 10.0
    init_rate_std_deg: float = 5.0

    # Robust state estimation & fault tolerance (Phase 5.2)
    nis_gating_enabled: bool = False
    nis_gate_gps_pos: float = 16.0       # chi^2 threshold for 2-DOF pos (~4-sigma)
    nis_gate_gps_pv: float = 25.0        # chi^2 threshold for 4-DOF pos+vel (~4-sigma)
    nis_gate_baro: float = 9.0           # chi^2 threshold for 1-DOF baro (~3-sigma)
    nis_gate_heading: float = 9.0        # chi^2 threshold for 1-DOF heading (~3-sigma)
    stale_rejection_enabled: bool = False
    dead_reckoning_q_inflation: float = 10.0
    dead_reckoning_outage_delay: float = 1.0
    max_consecutive_rejections: int = 0  # 0 disables blind force-acceptance; re-acquisition governed by covariance inflation

    def __post_init__(self) -> None:
        if self.forward_speed <= 0.0:
            raise ValueError("forward_speed must be positive")
        if self.descent_rate <= 0.0:
            raise ValueError("descent_rate must be positive")
        if self.max_turn_rate_deg_s <= 0.0:
            raise ValueError("max_turn_rate_deg_s must be positive")
        if self.yaw_rate_tau <= 0.0:
            raise ValueError("yaw_rate_tau must be positive")

        for val, name in [
            (self.q_pos, "q_pos"),
            (self.q_alt, "q_alt"),
            (self.q_psi, "q_psi"),
            (self.q_yaw_rate, "q_yaw_rate"),
            (self.r_gps_pos, "r_gps_pos"),
            (self.r_gps_vel, "r_gps_vel"),
            (self.r_baro_alt, "r_baro_alt"),
            (self.r_heading_deg, "r_heading_deg"),
            (self.init_pos_std, "init_pos_std"),
            (self.init_alt_std, "init_alt_std"),
            (self.init_psi_std_deg, "init_psi_std_deg"),
            (self.init_rate_std_deg, "init_rate_std_deg"),
            (self.nis_gate_gps_pos, "nis_gate_gps_pos"),
            (self.nis_gate_gps_pv, "nis_gate_gps_pv"),
            (self.nis_gate_baro, "nis_gate_baro"),
            (self.nis_gate_heading, "nis_gate_heading"),
            (self.dead_reckoning_q_inflation, "dead_reckoning_q_inflation"),
            (self.dead_reckoning_outage_delay, "dead_reckoning_outage_delay"),
        ]:
            if val < 0.0 or not math.isfinite(val):
                raise ValueError(f"{name} must be non-negative and finite")


@dataclass(frozen=True)
class EKFStateEstimate:
    """
    Immutable state estimate snapshot produced by the EKF.
    """
    time: float
    x: float
    y: float
    altitude: float
    vx: float
    vy: float
    vz: float
    heading_deg: float
    heading_rate_deg_s: float
    # 1-sigma uncertainties
    sigma_x: float
    sigma_y: float
    sigma_alt: float
    sigma_heading_deg: float
    # Innovation diagnostics
    last_gps_pos_residual: float = 0.0
    last_heading_residual_deg: float = 0.0
    last_baro_residual: float = 0.0
    # Robust estimation diagnostics (Phase 5.2)
    last_nis_gps: float = 0.0
    last_nis_baro: float = 0.0
    last_nis_heading: float = 0.0
    rejected_gps_count: int = 0
    rejected_baro_count: int = 0
    rejected_heading_count: int = 0


def wrap_angle_rad(angle_rad: float) -> float:
    """Wrap angle in radians to [-pi, pi)."""
    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi


def wrap_heading_deg(heading_deg: float) -> float:
    """Wrap navigation heading in degrees to [0, 360)."""
    return heading_deg % 360.0


class ExtendedKalmanFilter:
    """
    Extended Kalman Filter for CAN-7USAT precision landing navigation.

    States (5x1):
        x[0]: x (East position [m])
        x[1]: y (North position [m])
        x[2]: z (Altitude [m])
        x[3]: psi (Heading [rad], 0=North, pi/2=East)
        x[4]: w_rate (Turn rate [rad/s])
    """

    def __init__(self, config: Optional[EKFConfig] = None) -> None:
        self.config = config if config is not None else EKFConfig()

        self.x = np.zeros(5, dtype=np.float64)
        self.P = np.eye(5, dtype=np.float64)

        self._time = 0.0
        self._wind_x = 0.0
        self._wind_y = 0.0
        self._initialized = False

        self._last_gps_res = 0.0
        self._last_hdg_res_deg = 0.0
        self._last_baro_res = 0.0
        self._current_forward_speed = self.config.forward_speed
        self._current_descent_rate = self.config.descent_rate

        self.rejected_gps_count = 0
        self.consecutive_rejected_gps = 0
        self.rejected_baro_count = 0
        self.rejected_heading_count = 0
        self.last_nis_gps = 0.0
        self.last_nis_baro = 0.0
        self.last_nis_heading = 0.0
        self.last_gps_time: Optional[float] = None
        self.last_baro_time: Optional[float] = None
        self.last_heading_time: Optional[float] = None

        self.reset()

    def reset(
        self,
        init_x: float = 0.0,
        init_y: float = 0.0,
        init_alt: float = 1000.0,
        init_heading_deg: float = 0.0,
        init_time: float = 0.0,
        forward_speed: Optional[float] = None,
        descent_rate: Optional[float] = None,
    ) -> None:
        """Reset EKF state and covariance to initial values."""
        psi_rad = math.radians(wrap_heading_deg(init_heading_deg))

        self.x = np.array([init_x, init_y, init_alt, psi_rad, 0.0], dtype=np.float64)

        init_psi_std_rad = math.radians(self.config.init_psi_std_deg)
        init_rate_std_rad = math.radians(self.config.init_rate_std_deg)

        self.P = np.diag([
            self.config.init_pos_std ** 2,
            self.config.init_pos_std ** 2,
            self.config.init_alt_std ** 2,
            init_psi_std_rad ** 2,
            init_rate_std_rad ** 2,
        ]).astype(np.float64)

        self._time = init_time
        self._wind_x = 0.0
        self._wind_y = 0.0
        self._initialized = True
        self._last_gps_res = 0.0
        self._last_hdg_res_deg = 0.0
        self._last_baro_res = 0.0
        self._current_forward_speed = forward_speed if forward_speed is not None else self.config.forward_speed
        self._current_descent_rate = descent_rate if descent_rate is not None else self.config.descent_rate

        self.rejected_gps_count = 0
        self.consecutive_rejected_gps = 0
        self.rejected_baro_count = 0
        self.rejected_heading_count = 0
        self.last_nis_gps = 0.0
        self.last_nis_baro = 0.0
        self.last_nis_heading = 0.0
        self.last_gps_time = None
        self.last_baro_time = None
        self.last_heading_time = None

    def set_kinematics(self, forward_speed: float, descent_rate: float) -> None:
        """Set current kinematic forward airspeed and descent rate."""
        if math.isfinite(forward_speed) and forward_speed >= 0.0:
            self._current_forward_speed = forward_speed
        if math.isfinite(descent_rate) and descent_rate >= 0.0:
            self._current_descent_rate = descent_rate

    def set_wind_estimate(self, wind_x: float, wind_y: float) -> None:
        """Provide current in-flight wind estimate for kinematic propagation."""
        if math.isfinite(wind_x) and math.isfinite(wind_y):
            self._wind_x = wind_x
            self._wind_y = wind_y

    def predict(
        self,
        dt: float,
        steering_cmd: float,
        forward_speed: Optional[float] = None,
        descent_rate: Optional[float] = None,
    ) -> None:
        """
        EKF Time-Update / Prediction step.

        Args:
            dt: Timestep duration [s]
            steering_cmd: Commanded/actual steering input u in [-1.0, 1.0]
            forward_speed: Optional airspeed override [m/s]
            descent_rate: Optional descent rate override [m/s]
        """
        if dt <= 0.0 or not math.isfinite(dt):
            raise ValueError(f"Invalid prediction dt: {dt}")
        if not math.isfinite(steering_cmd):
            raise ValueError(f"Invalid steering_cmd: {steering_cmd}")

        u = max(-1.0, min(1.0, steering_cmd))
        max_turn_rate_rad_s = math.radians(self.config.max_turn_rate_deg_s)
        cmd_rate = u * max_turn_rate_rad_s

        x, y, z, psi, w = self.x
        Va = self._current_forward_speed if forward_speed is None else forward_speed
        Vz = self._current_descent_rate if descent_rate is None else descent_rate
        tau = self.config.yaw_rate_tau

        # State transition:
        # dot_x = Va * sin(psi) + W_x
        # dot_y = Va * cos(psi) + W_y
        # dot_z = -Vz
        # dot_psi = w
        # dot_w = (cmd_rate - w) / tau
        new_x = x + (Va * math.sin(psi) + self._wind_x) * dt
        new_y = y + (Va * math.cos(psi) + self._wind_y) * dt
        new_z = max(0.0, z - Vz * dt)
        new_psi = wrap_angle_rad(psi + w * dt)
        new_w = w + ((cmd_rate - w) / tau) * dt

        self.x = np.array([new_x, new_y, new_z, new_psi, new_w], dtype=np.float64)

        # State transition Jacobian F = d(f)/d(x)
        F = np.eye(5, dtype=np.float64)
        F[0, 3] = Va * math.cos(psi) * dt
        F[1, 3] = -Va * math.sin(psi) * dt
        F[3, 4] = dt
        F[4, 4] = 1.0 - (dt / tau)
        if F[4, 4] < 0.0:
            F[4, 4] = 0.0  # Guard against numerical oscillation if dt > tau

        # Process noise covariance Q_d
        # Scaled under sustained GNSS outage to avoid overconfidence during dead reckoning
        time_since_gps = (self._time - self.last_gps_time) if self.last_gps_time is not None else 0.0
        q_pos_scale = (
            self.config.dead_reckoning_q_inflation
            if (self.last_gps_time is not None and time_since_gps > self.config.dead_reckoning_outage_delay)
            else 1.0
        )

        Q = np.diag([
            (self.config.q_pos * q_pos_scale * dt) ** 2,
            (self.config.q_pos * q_pos_scale * dt) ** 2,
            (self.config.q_alt * dt) ** 2,
            (self.config.q_psi * dt) ** 2,
            (self.config.q_yaw_rate * dt) ** 2,
        ]).astype(np.float64)

        # Covariance propagation
        self.P = F @ self.P @ F.T + Q
        self._enforce_covariance_symmetry_and_psd()
        self._time += dt

    def correct_gps(
        self,
        meas_x: float,
        meas_y: float,
        meas_vx: Optional[float] = None,
        meas_vy: Optional[float] = None,
        meas_time: Optional[float] = None,
    ) -> bool:
        """
        EKF Measurement-Update with GNSS position (and optionally velocity).
        Returns True if measurement was accepted, False if rejected (gated, stale, or invalid).
        """
        for val, name in [(meas_x, "meas_x"), (meas_y, "meas_y")]:
            if not math.isfinite(val):
                return False  # Reject invalid/NaN measurement safely

        if self.config.stale_rejection_enabled and meas_time is not None and self.last_gps_time is not None:
            if (meas_time - self.last_gps_time) <= 1e-9:
                self.rejected_gps_count += 1
                return False

        use_vel = (
            meas_vx is not None
            and meas_vy is not None
            and math.isfinite(meas_vx)
            and math.isfinite(meas_vy)
            and self.config.r_gps_vel > 0.0
        )

        psi = self.x[3]
        Va = getattr(self, "_current_forward_speed", self.config.forward_speed)

        if use_vel:
            # Measurement vector z = [x, y, vx, vy]^T
            z = np.array([meas_x, meas_y, meas_vx, meas_vy], dtype=np.float64)
            # Predicted measurement h(x)
            h = np.array([
                self.x[0],
                self.x[1],
                Va * math.sin(psi) + self._wind_x,
                Va * math.cos(psi) + self._wind_y,
            ], dtype=np.float64)

            H = np.zeros((4, 5), dtype=np.float64)
            H[0, 0] = 1.0
            H[1, 1] = 1.0
            H[2, 3] = Va * math.cos(psi)
            H[3, 3] = -Va * math.sin(psi)

            R = np.diag([
                self.config.r_gps_pos ** 2,
                self.config.r_gps_pos ** 2,
                self.config.r_gps_vel ** 2,
                self.config.r_gps_vel ** 2,
            ]).astype(np.float64)
            gate = self.config.nis_gate_gps_pv
        else:
            # Position-only update
            z = np.array([meas_x, meas_y], dtype=np.float64)
            h = np.array([self.x[0], self.x[1]], dtype=np.float64)

            H = np.zeros((2, 5), dtype=np.float64)
            H[0, 0] = 1.0
            H[1, 1] = 1.0

            R = np.diag([
                self.config.r_gps_pos ** 2,
                self.config.r_gps_pos ** 2,
            ]).astype(np.float64)
            gate = self.config.nis_gate_gps_pos

        # Innovation residual
        y = z - h
        self._last_gps_res = float(math.hypot(float(y[0]), float(y[1])))

        # Break lockout cascade if filter has drifted during extended outage
        if self.config.max_consecutive_rejections > 0 and self.consecutive_rejected_gps >= self.config.max_consecutive_rejections:
            effective_gate = None
        else:
            effective_gate = gate

        accepted, nis = self._apply_measurement_update(y, H, R, gate=effective_gate)
        self.last_nis_gps = nis
        if not accepted:
            self.rejected_gps_count += 1
            self.consecutive_rejected_gps += 1
            return False

        self.consecutive_rejected_gps = 0
        self.last_gps_time = meas_time if meas_time is not None else self._time
        return True

    def correct_baro(self, meas_alt: float, meas_time: Optional[float] = None) -> bool:
        """
        EKF Measurement-Update with Barometric altitude.
        Returns True if measurement was accepted, False if rejected (gated, stale, or invalid).
        """
        if not math.isfinite(meas_alt) or meas_alt < 0.0:
            return False  # Reject invalid measurement safely

        if self.config.stale_rejection_enabled and meas_time is not None and self.last_baro_time is not None:
            if (meas_time - self.last_baro_time) <= 1e-9:
                self.rejected_baro_count += 1
                return False

        z = np.array([meas_alt], dtype=np.float64)
        h = np.array([self.x[2]], dtype=np.float64)
        y = z - h
        self._last_baro_res = float(abs(float(y[0])))

        H = np.zeros((1, 5), dtype=np.float64)
        H[0, 2] = 1.0

        R = np.array([[self.config.r_baro_alt ** 2]], dtype=np.float64)
        accepted, nis = self._apply_measurement_update(y, H, R, gate=self.config.nis_gate_baro)
        self.last_nis_baro = nis
        if not accepted:
            self.rejected_baro_count += 1
            return False

        self.last_baro_time = meas_time if meas_time is not None else self._time
        return True

    def correct_heading(self, meas_heading_deg: float, meas_time: Optional[float] = None) -> bool:
        """
        EKF Measurement-Update with Heading sensor measurement.
        Wraps heading innovation strictly to [-pi, pi).
        Returns True if measurement was accepted, False if rejected (gated, stale, or invalid).
        """
        if not math.isfinite(meas_heading_deg):
            return False

        if self.config.stale_rejection_enabled and meas_time is not None and self.last_heading_time is not None:
            if (meas_time - self.last_heading_time) <= 1e-9:
                self.rejected_heading_count += 1
                return False

        meas_psi_rad = math.radians(wrap_heading_deg(meas_heading_deg))
        pred_psi_rad = self.x[3]

        # Innovation with angular wrapping
        y_rad = wrap_angle_rad(meas_psi_rad - pred_psi_rad)
        self._last_hdg_res_deg = float(abs(math.degrees(y_rad)))

        y = np.array([y_rad], dtype=np.float64)
        H = np.zeros((1, 5), dtype=np.float64)
        H[0, 3] = 1.0

        r_rad = math.radians(self.config.r_heading_deg)
        R = np.array([[r_rad ** 2]], dtype=np.float64)

        accepted, nis = self._apply_measurement_update(y, H, R, gate=self.config.nis_gate_heading)
        self.last_nis_heading = nis
        if not accepted:
            self.rejected_heading_count += 1
            return False

        # Re-wrap heading state
        self.x[3] = wrap_angle_rad(self.x[3])
        self.last_heading_time = meas_time if meas_time is not None else self._time
        return True

    def _apply_measurement_update(
        self,
        y: np.ndarray,
        H: np.ndarray,
        R: np.ndarray,
        gate: Optional[float] = None,
    ) -> Tuple[bool, float]:
        """
        Numerically stable Kalman update using Joseph form for covariance.
        Optionally evaluates NIS gating before state/covariance update.

        Returns:
            Tuple of (accepted: bool, nis: float)
        """
        # Innovation covariance S = H P H^T + R
        S = H @ self.P @ H.T + R
        # Inversion with pseudo-inverse fallback for singular edge cases
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)

        # Normalized Innovation Squared (NIS)
        nis = float(np.dot(y, S_inv @ y))

        if gate is not None and self.config.nis_gating_enabled:
            if nis > gate:
                # Reject outlier measurement: do not modify state or covariance
                return False, nis

        # Kalman gain K = P H^T S^-1
        K = self.P @ H.T @ S_inv

        # State update
        dx = K @ y
        self.x += dx
        self.x[3] = wrap_angle_rad(self.x[3])

        # Joseph form covariance update:
        # P = (I - K H) P (I - K H)^T + K R K^T
        I = np.eye(5, dtype=np.float64)
        IKH = I - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R @ K.T

        self._enforce_covariance_symmetry_and_psd()
        return True, nis

    def _enforce_covariance_symmetry_and_psd(self) -> None:
        """Symmetric projection and minimum diagonal clamping."""
        self.P = 0.5 * (self.P + self.P.T)
        for i in range(5):
            if self.P[i, i] < 1e-8:
                self.P[i, i] = 1e-8

    def get_estimate(self) -> EKFStateEstimate:
        """Retrieve current estimated state and uncertainties."""
        x, y, z, psi, w = self.x
        hdg_deg = wrap_heading_deg(math.degrees(psi))
        w_deg_s = math.degrees(w)

        Va = getattr(self, "_current_forward_speed", self.config.forward_speed)
        Vz = getattr(self, "_current_descent_rate", self.config.descent_rate)
        vx = Va * math.sin(psi) + self._wind_x
        vy = Va * math.cos(psi) + self._wind_y
        vz = -Vz

        sigmas = np.sqrt(np.diag(self.P))

        return EKFStateEstimate(
            time=self._time,
            x=float(x),
            y=float(y),
            altitude=float(z),
            vx=float(vx),
            vy=float(vy),
            vz=float(vz),
            heading_deg=float(hdg_deg),
            heading_rate_deg_s=float(w_deg_s),
            sigma_x=float(sigmas[0]),
            sigma_y=float(sigmas[1]),
            sigma_alt=float(sigmas[2]),
            sigma_heading_deg=float(math.degrees(sigmas[3])),
            last_gps_pos_residual=self._last_gps_res,
            last_heading_residual_deg=self._last_hdg_res_deg,
            last_baro_residual=self._last_baro_res,
            last_nis_gps=self.last_nis_gps,
            last_nis_baro=self.last_nis_baro,
            last_nis_heading=self.last_nis_heading,
            rejected_gps_count=self.rejected_gps_count,
            rejected_baro_count=self.rejected_baro_count,
            rejected_heading_count=self.rejected_heading_count,
        )
