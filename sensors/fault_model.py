"""
Sensor fault and degradation models for CAN-7USAT Precision Landing Guidance.

This module provides modular, configurable, opt-in sensor fault models:
1. GNSS Position/Velocity Outages (complete loss of GNSS packets).
2. GNSS Position Outliers (transient position jumps / multipath spikes).
3. GNSS Measurement Staleness (frozen measurement packets).
4. Barometric Altimeter Bias, Drift, and Outages.
5. Heading Sensor Bias, Drift, Outages, and Transient Spikes.
6. Temporary Measurement Noise Degradation (dilution of precision).
7. Mission State-Triggered Fault Windows (relative to flight phase entry).

CRITICAL DESIGN PRINCIPLES:
- All fault models are strictly opt-in (enabled=False by default).
- Zero simulator ground truth is accessed by flight software or estimators.
- Parameters represent explicitly documented sensitivity assumptions.
"""

from dataclasses import dataclass, field
import math
from typing import Dict, List, Optional, Tuple

from flight.state_machine import FlightState


@dataclass
class GNSSOutage:
    """Interval [start_time, end_time] during which GNSS packets are dropped."""
    start_time: float
    end_time: float
    drop_position: bool = True
    drop_velocity: bool = True

    def __post_init__(self) -> None:
        if self.start_time < 0.0 or not math.isfinite(self.start_time):
            raise ValueError(f"Invalid start_time: {self.start_time}")
        if self.end_time < self.start_time or not math.isfinite(self.end_time):
            raise ValueError(f"Invalid end_time: {self.end_time}")

    def is_active(self, t: float) -> bool:
        return self.start_time <= t <= self.end_time


@dataclass
class GNSSOutlier:
    """Transient position/velocity jump occurring at or during [time, time + duration]."""
    time: float
    offset_x: float
    offset_y: float
    offset_vx: float = 0.0
    offset_vy: float = 0.0
    duration: float = 0.2  # Single packet or brief interval

    def __post_init__(self) -> None:
        if self.time < 0.0 or not math.isfinite(self.time):
            raise ValueError(f"Invalid outlier time: {self.time}")
        if self.duration <= 0.0 or not math.isfinite(self.duration):
            raise ValueError(f"Invalid outlier duration: {self.duration}")

    def is_active(self, t: float) -> bool:
        return self.time <= t <= (self.time + self.duration)


@dataclass
class GNSSStaleInterval:
    """Interval [start_time, end_time] where GNSS data is frozen (repeating old sample)."""
    start_time: float
    end_time: float

    def __post_init__(self) -> None:
        if self.start_time < 0.0 or not math.isfinite(self.start_time):
            raise ValueError(f"Invalid start_time: {self.start_time}")
        if self.end_time < self.start_time or not math.isfinite(self.end_time):
            raise ValueError(f"Invalid end_time: {self.end_time}")

    def is_active(self, t: float) -> bool:
        return self.start_time <= t <= self.end_time


@dataclass
class BaroFault:
    """
    Barometric altimeter degradation:
    - Constant bias [m] (e.g., QNH reference error)
    - Linear drift rate [m/s]
    - Complete measurement outage
    """
    start_time: float = 0.0
    end_time: float = float("inf")
    bias: float = 0.0
    drift_rate: float = 0.0
    outage: bool = False

    def is_active(self, t: float) -> bool:
        return self.start_time <= t <= self.end_time

    def get_altitude_error(self, t: float) -> float:
        if not self.is_active(t):
            return 0.0
        dt = t - self.start_time
        return self.bias + self.drift_rate * dt


@dataclass
class HeadingFault:
    """
    Heading sensor degradation:
    - Constant bias [deg] (e.g., magnetic declination error or hard-iron offset)
    - Linear drift rate [deg/s]
    - Transient spike at spike_time
    - Complete measurement outage
    """
    start_time: float = 0.0
    end_time: float = float("inf")
    bias_deg: float = 0.0
    drift_rate_deg_s: float = 0.0
    outage: bool = False
    spike_time: Optional[float] = None
    spike_mag_deg: float = 0.0
    spike_duration: float = 0.2

    def is_active(self, t: float) -> bool:
        return self.start_time <= t <= self.end_time

    def get_heading_error_deg(self, t: float) -> float:
        if not self.is_active(t):
            return 0.0
        dt = t - self.start_time
        err = self.bias_deg + self.drift_rate_deg_s * dt
        if self.spike_time is not None and self.spike_time <= t <= (self.spike_time + self.spike_duration):
            err += self.spike_mag_deg
        return err


@dataclass
class NoiseMultiplier:
    """Temporary scaling of sensor 1-sigma noise during [start_time, end_time]."""
    start_time: float
    end_time: float
    multiplier: float = 1.0
    sensor: str = "all"  # "gnss", "baro", "heading", "all"

    def is_active(self, t: float) -> bool:
        return self.start_time <= t <= self.end_time


@dataclass
class StateTriggeredFault:
    """
    Fault whose activation time is dynamically anchored to a mission state transition.
    
    When simulator enters trigger_state at t_trans:
    fault activates at (t_trans + delay_seconds) for duration_seconds.
    """
    trigger_state: FlightState
    duration_seconds: float
    delay_seconds: float = 0.0
    fault_type: str = "gnss_outage"  # "gnss_outage", "baro_outage", "heading_outage", "heading_spike"
    magnitude: float = 0.0           # Spike magnitude or bias if applicable


@dataclass
class SensorFaultConfig:
    """
    Master configuration for synthetic sensor faults and degradations.
    """
    enabled: bool = False

    gnss_outages: List[GNSSOutage] = field(default_factory=list)
    gnss_outliers: List[GNSSOutlier] = field(default_factory=list)
    gnss_stale_intervals: List[GNSSStaleInterval] = field(default_factory=list)

    baro_faults: List[BaroFault] = field(default_factory=list)
    heading_faults: List[HeadingFault] = field(default_factory=list)
    noise_multipliers: List[NoiseMultiplier] = field(default_factory=list)

    state_triggered_faults: List[StateTriggeredFault] = field(default_factory=list)

    def is_gnss_outage(self, t: float) -> Tuple[bool, bool]:
        """Return (drop_position, drop_velocity) if an outage is active at time t."""
        if not self.enabled:
            return False, False
        drop_pos = False
        drop_vel = False
        for out in self.gnss_outages:
            if out.is_active(t):
                if out.drop_position:
                    drop_pos = True
                if out.drop_velocity:
                    drop_vel = True
        return drop_pos, drop_vel

    def is_gnss_stale(self, t: float) -> bool:
        """Return True if GNSS is frozen/stale at time t."""
        if not self.enabled:
            return False
        return any(stale.is_active(t) for stale in self.gnss_stale_intervals)

    def get_gnss_outlier(self, t: float) -> Tuple[float, float, float, float]:
        """Return (offset_x, offset_y, offset_vx, offset_vy) active at time t."""
        if not self.enabled:
            return 0.0, 0.0, 0.0, 0.0
        ox, oy, ovx, ovy = 0.0, 0.0, 0.0, 0.0
        for out in self.gnss_outliers:
            if out.is_active(t):
                ox += out.offset_x
                oy += out.offset_y
                ovx += out.offset_vx
                ovy += out.offset_vy
        return ox, oy, ovx, ovy

    def is_baro_outage(self, t: float) -> bool:
        """Return True if baro outage is active at time t."""
        if not self.enabled:
            return False
        return any(bf.outage and bf.is_active(t) for bf in self.baro_faults)

    def get_baro_error(self, t: float) -> float:
        """Return cumulative baro bias and drift error [m] at time t."""
        if not self.enabled:
            return 0.0
        return sum(bf.get_altitude_error(t) for bf in self.baro_faults)

    def is_heading_outage(self, t: float) -> bool:
        """Return True if heading outage is active at time t."""
        if not self.enabled:
            return False
        return any(hf.outage and hf.is_active(t) for hf in self.heading_faults)

    def get_heading_error_deg(self, t: float) -> float:
        """Return cumulative heading bias, drift, and spike error [deg] at time t."""
        if not self.enabled:
            return 0.0
        return sum(hf.get_heading_error_deg(t) for hf in self.heading_faults)

    def get_noise_multiplier(self, t: float, sensor: str) -> float:
        """Return noise standard deviation scaling multiplier at time t."""
        if not self.enabled:
            return 1.0
        mult = 1.0
        for nm in self.noise_multipliers:
            if nm.is_active(t) and (nm.sensor == "all" or nm.sensor == sensor):
                mult *= nm.multiplier
        return mult

    def notify_state_transition(self, new_state: FlightState, transition_time: float) -> None:
        """
        Dynamically instantiate concrete faults from state-triggered templates upon flight phase entry.
        """
        if not self.enabled:
            return
        for stf in self.state_triggered_faults:
            if stf.trigger_state == new_state:
                start = transition_time + stf.delay_seconds
                end = start + stf.duration_seconds
                if stf.fault_type == "gnss_outage":
                    self.gnss_outages.append(GNSSOutage(start_time=start, end_time=end))
                elif stf.fault_type == "baro_outage":
                    self.baro_faults.append(BaroFault(start_time=start, end_time=end, outage=True))
                elif stf.fault_type == "heading_outage":
                    self.heading_faults.append(HeadingFault(start_time=start, end_time=end, outage=True))
                elif stf.fault_type == "heading_spike":
                    self.heading_faults.append(
                        HeadingFault(
                            start_time=start,
                            end_time=end,
                            spike_time=start,
                            spike_mag_deg=stf.magnitude,
                        )
                    )
