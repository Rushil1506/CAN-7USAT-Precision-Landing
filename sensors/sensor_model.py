"""
Sensor noise and synthetic measurement models for CAN-7USAT Precision Landing Guidance.

This module provides configurable, modular, testable synthetic sensor models
representing the onboard sensors of the CanSat:
1. GNSS receiver (horizontal position x, y and horizontal velocity vx, vy)
2. Barometric altimeter (altitude above ground)
3. Heading sensor / Magnetometer (heading angle in degrees [0, 360), wrapped)

CRITICAL ARCHITECTURAL CONVENTIONS:
- Local coordinates: +X = East, +Y = North
- Heading: 0 deg = North, 90 deg = East, 180 deg = South, 270 deg = West
- Sensors are synthetic measurement generators from true physical state at the sensor boundary.
- Downstream guidance and state estimation must NOT access simulator ground truth when operating
  in noisy/estimated mode.
- All noise parameters are explicitly configurable and labeled as provisional engineering assumptions.
"""

from dataclasses import dataclass, field
import math
import random
from typing import Optional, Tuple

from sensors.fault_model import SensorFaultConfig


@dataclass
class SensorNoiseConfig:
    """
    Configuration for synthetic sensor measurement noise.

    Note: These parameters are provisional simulation assumptions and do NOT
    represent validated hardware datasheets unless explicitly documented.
    """
    enabled: bool = False
    seed: Optional[int] = None

    # GNSS horizontal position 1-sigma standard deviation [m]
    gnss_pos_std: float = 2.5
    # GNSS horizontal velocity 1-sigma standard deviation [m/s]
    gnss_vel_std: float = 0.2
    # GNSS measurement sampling interval [s] (e.g. 5 Hz -> 0.2s, 10 Hz -> 0.1s)
    gnss_dt: float = 0.2

    # Barometer altitude 1-sigma standard deviation [m]
    baro_alt_std: float = 1.5
    # Barometer sampling interval [s] (e.g. 10 Hz -> 0.1s)
    baro_dt: float = 0.1

    # Heading sensor / Magnetometer 1-sigma standard deviation [deg]
    heading_std_deg: float = 3.0
    # Heading sampling interval [s] (e.g. 10 Hz -> 0.1s)
    heading_dt: float = 0.1

    def __post_init__(self) -> None:
        if self.gnss_pos_std < 0.0 or not math.isfinite(self.gnss_pos_std):
            raise ValueError("gnss_pos_std must be non-negative and finite")
        if self.gnss_vel_std < 0.0 or not math.isfinite(self.gnss_vel_std):
            raise ValueError("gnss_vel_std must be non-negative and finite")
        if self.gnss_dt <= 0.0 or not math.isfinite(self.gnss_dt):
            raise ValueError("gnss_dt must be strictly positive and finite")

        if self.baro_alt_std < 0.0 or not math.isfinite(self.baro_alt_std):
            raise ValueError("baro_alt_std must be non-negative and finite")
        if self.baro_dt <= 0.0 or not math.isfinite(self.baro_dt):
            raise ValueError("baro_dt must be strictly positive and finite")

        if self.heading_std_deg < 0.0 or not math.isfinite(self.heading_std_deg):
            raise ValueError("heading_std_deg must be non-negative and finite")
        if self.heading_dt <= 0.0 or not math.isfinite(self.heading_dt):
            raise ValueError("heading_dt must be strictly positive and finite")


@dataclass(frozen=True)
class SensorMeasurement:
    """
    Discrete measurement packet produced by the sensor suite.

    Attributes:
        time: Timestamp of measurement [s]
        # GNSS measurements
        gnss_valid: Whether a new GNSS sample arrived at this step
        gnss_x: Measured East position [m] (or None if not updated)
        gnss_y: Measured North position [m] (or None if not updated)
        gnss_vx: Measured East velocity [m/s] (or None if not updated)
        gnss_vy: Measured North velocity [m/s] (or None if not updated)

        # Barometer measurements
        baro_valid: Whether a new baro sample arrived at this step
        baro_altitude: Measured altitude [m] (or None if not updated)

        # Heading sensor measurements
        heading_valid: Whether a new heading sample arrived at this step
        heading_deg: Measured heading [deg, 0-360) (or None if not updated)
    """
    time: float
    gnss_valid: bool = False
    gnss_x: Optional[float] = None
    gnss_y: Optional[float] = None
    gnss_vx: Optional[float] = None
    gnss_vy: Optional[float] = None

    baro_valid: bool = False
    baro_altitude: Optional[float] = None

    heading_valid: bool = False
    heading_deg: Optional[float] = None

    # Measurement sample timestamps [s] (used for staleness and latency tracking)
    gnss_timestamp: Optional[float] = None
    baro_timestamp: Optional[float] = None
    heading_timestamp: Optional[float] = None


class SensorSuite:
    """
    Simulated onboard sensor suite for CAN-7USAT.

    Generates synthetic GNSS, Barometer, and Heading measurements from true physical state
    at configured sampling intervals and with configurable Gaussian noise.
    Supports deterministic seeding, independent sample timers, and zero-noise pass-through.
    """

    def __init__(
        self,
        config: Optional[SensorNoiseConfig] = None,
        fault_config: Optional[SensorFaultConfig] = None,
    ) -> None:
        self.config = config if config is not None else SensorNoiseConfig()
        self.fault_config = fault_config if fault_config is not None else SensorFaultConfig()
        self._rng: Optional[random.Random] = None
        self._last_gnss_time: float = -1e9
        self._last_fresh_gnss_time: float = -1e9
        self._last_baro_time: float = -1e9
        self._last_heading_time: float = -1e9

        self._current_gnss_x: float = 0.0
        self._current_gnss_y: float = 0.0
        self._current_gnss_vx: float = 0.0
        self._current_gnss_vy: float = 0.0

        self._current_baro_alt: float = 0.0
        self._current_heading_deg: float = 0.0

        self.reset(self.config.seed)

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset sensor suite timers and internal RNG."""
        if seed is not None:
            self._rng = random.Random(seed)
        elif self.config.seed is not None:
            self._rng = random.Random(self.config.seed)
        else:
            self._rng = random.Random()

        self._last_gnss_time = -1e9
        self._last_baro_time = -1e9
        self._last_heading_time = -1e9

        self._current_gnss_x = 0.0
        self._current_gnss_y = 0.0
        self._current_gnss_vx = 0.0
        self._current_gnss_vy = 0.0
        self._current_baro_alt = 0.0
        self._current_heading_deg = 0.0

    def update(
        self,
        time: float,
        true_x: float,
        true_y: float,
        true_altitude: float,
        true_vx: float,
        true_vy: float,
        true_heading_deg: float,
    ) -> SensorMeasurement:
        """
        Sample sensors at given simulation time against true physical state.

        Returns a SensorMeasurement object indicating which measurements were updated.
        """
        if not math.isfinite(time) or time < 0.0:
            raise ValueError(f"Invalid simulation time: {time}")
        for val, name in [
            (true_x, "true_x"),
            (true_y, "true_y"),
            (true_altitude, "true_altitude"),
            (true_vx, "true_vx"),
            (true_vy, "true_vy"),
            (true_heading_deg, "true_heading_deg"),
        ]:
            if not math.isfinite(val):
                raise ValueError(f"Non-finite value provided for {name}: {val}")

        if not self.config.enabled:
            # Ideal pass-through: sample on every update with zero noise
            wrapped_heading = true_heading_deg % 360.0
            self._current_gnss_x = true_x
            self._current_gnss_y = true_y
            self._current_gnss_vx = true_vx
            self._current_gnss_vy = true_vy
            self._current_baro_alt = true_altitude
            self._current_heading_deg = wrapped_heading

            return SensorMeasurement(
                time=time,
                gnss_valid=True,
                gnss_x=true_x,
                gnss_y=true_y,
                gnss_vx=true_vx,
                gnss_vy=true_vy,
                baro_valid=True,
                baro_altitude=true_altitude,
                heading_valid=True,
                heading_deg=wrapped_heading,
            )

        # Check sample rate intervals
        # Small epsilon (1e-6) handles floating-point stepping precision
        eps = 1e-6
        gnss_valid = (time - self._last_gnss_time) >= (self.config.gnss_dt - eps)
        baro_valid = (time - self._last_baro_time) >= (self.config.baro_dt - eps)
        heading_valid = (time - self._last_heading_time) >= (self.config.heading_dt - eps)

        meas_gnss_x: Optional[float] = None
        meas_gnss_y: Optional[float] = None
        meas_gnss_vx: Optional[float] = None
        meas_gnss_vy: Optional[float] = None
        meas_baro_alt: Optional[float] = None
        meas_heading_deg: Optional[float] = None

        meas_gnss_time: Optional[float] = None
        meas_baro_time: Optional[float] = None
        meas_hdg_time: Optional[float] = None

        # Determine active fault conditions
        fault_active = self.fault_config.enabled
        drop_pos, drop_vel = self.fault_config.is_gnss_outage(time) if fault_active else (False, False)
        is_stale_gnss = self.fault_config.is_gnss_stale(time) if fault_active else False

        baro_outage = self.fault_config.is_baro_outage(time) if fault_active else False
        heading_outage = self.fault_config.is_heading_outage(time) if fault_active else False

        gnss_mult = self.fault_config.get_noise_multiplier(time, "gnss") if fault_active else 1.0
        baro_mult = self.fault_config.get_noise_multiplier(time, "baro") if fault_active else 1.0
        hdg_mult = self.fault_config.get_noise_multiplier(time, "heading") if fault_active else 1.0

        if gnss_valid:
            self._last_gnss_time = time
            if drop_pos and drop_vel:
                # Total outage: drop packet
                gnss_valid = False
            elif is_stale_gnss:
                # Stale freeze: receiver outputs old frozen coordinates, timestamp does NOT advance
                meas_gnss_x = self._current_gnss_x
                meas_gnss_y = self._current_gnss_y
                meas_gnss_vx = self._current_gnss_vx
                meas_gnss_vy = self._current_gnss_vy
                meas_gnss_time = self._last_fresh_gnss_time
            else:
                self._last_fresh_gnss_time = time
                meas_gnss_time = time
                pos_std = self.config.gnss_pos_std * gnss_mult
                vel_std = self.config.gnss_vel_std * gnss_mult

                noise_x = self._rng.gauss(0.0, pos_std) if pos_std > 0 else 0.0
                noise_y = self._rng.gauss(0.0, pos_std) if pos_std > 0 else 0.0
                noise_vx = self._rng.gauss(0.0, vel_std) if vel_std > 0 else 0.0
                noise_vy = self._rng.gauss(0.0, vel_std) if vel_std > 0 else 0.0

                ox, oy, ovx, ovy = self.fault_config.get_gnss_outlier(time) if fault_active else (0.0, 0.0, 0.0, 0.0)

                self._current_gnss_x = true_x + noise_x + ox
                self._current_gnss_y = true_y + noise_y + oy
                self._current_gnss_vx = true_vx + noise_vx + ovx
                self._current_gnss_vy = true_vy + noise_vy + ovy

                meas_gnss_x = self._current_gnss_x
                meas_gnss_y = self._current_gnss_y
                meas_gnss_vx = self._current_gnss_vx
                meas_gnss_vy = self._current_gnss_vy

        if baro_valid:
            self._last_baro_time = time
            if baro_outage:
                baro_valid = False
            else:
                meas_baro_time = time
                baro_std = self.config.baro_alt_std * baro_mult
                noise_alt = self._rng.gauss(0.0, baro_std) if baro_std > 0 else 0.0
                baro_err = self.fault_config.get_baro_error(time) if fault_active else 0.0

                self._current_baro_alt = max(0.0, true_altitude + noise_alt + baro_err)
                meas_baro_alt = self._current_baro_alt

        if heading_valid:
            self._last_heading_time = time
            if heading_outage:
                heading_valid = False
            else:
                meas_hdg_time = time
                hdg_std = self.config.heading_std_deg * hdg_mult
                noise_hdg = self._rng.gauss(0.0, hdg_std) if hdg_std > 0 else 0.0
                hdg_err = self.fault_config.get_heading_error_deg(time) if fault_active else 0.0

                raw_hdg = (true_heading_deg + noise_hdg + hdg_err) % 360.0
                self._current_heading_deg = raw_hdg
                meas_heading_deg = self._current_heading_deg

        return SensorMeasurement(
            time=time,
            gnss_valid=gnss_valid,
            gnss_x=meas_gnss_x,
            gnss_y=meas_gnss_y,
            gnss_vx=meas_gnss_vx,
            gnss_vy=meas_gnss_vy,
            baro_valid=baro_valid,
            baro_altitude=meas_baro_alt,
            heading_valid=heading_valid,
            heading_deg=meas_heading_deg,
            gnss_timestamp=meas_gnss_time,
            baro_timestamp=meas_baro_time,
            heading_timestamp=meas_hdg_time,
        )

    # Convenience accessors to latest sampled values (for raw noisy guidance mode)
    @property
    def latest_x(self) -> float:
        return self._current_gnss_x

    @property
    def latest_y(self) -> float:
        return self._current_gnss_y

    @property
    def latest_vx(self) -> float:
        return self._current_gnss_vx

    @property
    def latest_vy(self) -> float:
        return self._current_gnss_vy

    @property
    def latest_altitude(self) -> float:
        return self._current_baro_alt

    @property
    def latest_heading_deg(self) -> float:
        return self._current_heading_deg
