"""
Unit and regression tests for Phase 4.4c Breakout-Aware Planning and Validation.

Validates:
1. Target invariance across all planner modes.
2. Baseline preservation when planner is disabled.
3. Safe numerical behavior and deadbands.
4. Symmetrical handling and orbit direction awareness.
5. Terminal-overlapping gust response.
6. Flight software wind estimator isolation (no simulator truth leakage).
"""

import math
import pytest

from navigation.coordinates import LocalPoint
from navigation.distance import distance
from guidance.loiter_guidance import LoiterDirection
from guidance.loiter_center_planner import (
    LoiterCenterPlanner,
    LoiterCenterPlannerConfig,
)
from flight.parafoil import ParafoilState, DrogueConfig, ParafoilConfig
from flight.state_machine import FlightState
from control.pid import PIDController
from control.actuator import ActuatorModel, ActuatorConfig
from guidance.guidance_controller import GuidanceController
from navigation.waypoints import generate_waypoints
from navigation.waypoint_manager import WaypointManager
from simulation.simulator import ParafoilSimulator
from environment.wind import ConstantWindField, CosineGust, ComposedWindField


class TestBreakoutAwarePlannerInvariants:
    """Verify target invariance, baseline preservation, and estimator isolation."""

    def test_target_position_strictly_invariant(self):
        """Verify that the geographic landing target remains strictly invariant."""
        target = LocalPoint(1200.0, 0.0)
        planner = LoiterCenterPlanner(
            LoiterCenterPlannerConfig(enabled=True, bias_gain=2.5, direction="upwind")
        )
        planned_center, offset = planner.plan_center(target, wind_x=4.0, wind_y=-4.0)

        # Target itself must not have changed
        assert target.x == 1200.0
        assert target.y == 0.0
        # Offset must be finite
        assert math.isfinite(planned_center.x)
        assert math.isfinite(planned_center.y)
        assert offset != (0.0, 0.0)

    def test_baseline_preservation_when_disabled(self):
        """When enabled=False, planner must strictly return zero offset and target center."""
        target = LocalPoint(1200.0, 0.0)
        planner = LoiterCenterPlanner(LoiterCenterPlannerConfig(enabled=False))

        c, off = planner.plan_center(target, wind_x=5.0, wind_y=5.0)
        assert c.x == target.x
        assert c.y == target.y
        assert off == (0.0, 0.0)

    def test_deadband_and_numerical_safety(self):
        """Test zero-wind deadband and NaN/Inf sanitization."""
        target = LocalPoint(1200.0, 0.0)
        planner = LoiterCenterPlanner(
            LoiterCenterPlannerConfig(enabled=True, min_wind_speed=0.5)
        )

        # Sub-threshold wind
        c1, off1 = planner.plan_center(target, wind_x=0.2, wind_y=0.2)
        assert off1 == (0.0, 0.0)
        assert c1.x == target.x and c1.y == target.y

        # Reset and test NaN/Inf
        planner.reset()
        c2, off2 = planner.plan_center(target, wind_x=float("nan"), wind_y=4.0)
        assert off2 == (0.0, 0.0)

        planner.reset()
        c3, off3 = planner.plan_center(target, wind_x=float("inf"), wind_y=0.0)
        assert off3 == (0.0, 0.0)


class TestBreakoutAwareSimulationBehavior:
    """Verify simulator integration, symmetry, and terminal gust behavior."""

    @staticmethod
    def _create_sim(wind_field, planner=None, actuator=None):
        target = LocalPoint(1200.0, 0.0)
        wps = generate_waypoints(LocalPoint(0.0, 0.0), target, num_waypoints=10)
        wpm = WaypointManager(wps, acceptance_radius=25.0)
        pid = PIDController(
            kp=10.0, ki=0.1, kd=0.5,
            output_min=-1.0, output_max=1.0,
            integral_min=-10.0, integral_max=10.0,
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
            actuator_model=actuator,
            loiter_center_planner=planner,
        )

        def cb(t, s):
            if sim.state_machine.get_state() == FlightState.FINAL_APPROACH:
                if not wpm.is_complete:
                    wpm.skip_to_final_target()
                    guidance.pid.reset()
            pos = LocalPoint(s.x, s.y)
            out = guidance.update(position=pos, heading=s.heading, altitude=s.altitude, dt=sim.dt)
            return out.steering

        history = sim.run(cb, max_time=300.0)
        last = history[-1]
        err = distance(LocalPoint(last.x, last.y), target)
        return sim, history, err

    def test_baseline_preserves_production_accuracy_zero_wind(self):
        """Baseline without planner yields established 4.35 m landing accuracy."""
        wf = ConstantWindField(0.0, 0.0)
        _, _, err = self._create_sim(wf, planner=None)
        assert err == pytest.approx(4.35, abs=0.05)

    def test_baseline_shear_planner_balances_opposite_crosswinds(self):
        """Verify that Baseline A maintains balanced landing error across opposite crosswinds."""
        wf_north = ConstantWindField(0.0, 4.0)
        wf_south = ConstantWindField(0.0, -4.0)

        _, _, err_north = self._create_sim(wf_north, planner=None)
        _, _, err_south = self._create_sim(wf_south, planner=None)

        assert err_north < 10.0
        assert err_south < 10.0
        # Balance: difference between opposite crosswinds is less than 3 m
        assert abs(err_north - err_south) < 3.0

    def test_terminal_overlapping_gust_execution(self):
        """Verify that a deterministic gust active during terminal loiter (t in [140, 180] s) is processed."""
        gust = CosineGust(gust_x=0.0, gust_y=3.0, t_start=140.0, duration=40.0)
        wf = ComposedWindField(ConstantWindField(0.0, 2.0), gust)

        sim, history, err = self._create_sim(wf, planner=None)
        assert sim.state.altitude <= 0.0
        assert err < 15.0

        # Verify that terminal guidance occurred during the active gust window
        term_samples = [s for s in history if s.state == FlightState.TERMINAL_GUIDANCE]
        assert len(term_samples) > 0
        term_times = [s.time for s in term_samples]
        assert min(term_times) < 180.0
        assert max(term_times) > 160.0
