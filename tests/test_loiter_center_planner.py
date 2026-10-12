"""
Unit and integration tests for LoiterCenterPlanner and its simulator integration.
"""

import math
import pytest

from navigation.coordinates import LocalPoint
from navigation.distance import distance
from guidance.loiter_center_planner import (
    LoiterCenterPlanner,
    LoiterCenterPlannerConfig,
)
from flight.parafoil import ParafoilState, DrogueConfig, ParafoilConfig
from flight.state_machine import FlightState
from control.pid import PIDController
from guidance.guidance_controller import GuidanceController
from navigation.waypoints import generate_waypoints
from navigation.waypoint_manager import WaypointManager
from simulation.simulator import ParafoilSimulator
from environment.wind import ConstantWindField


class TestLoiterCenterPlannerConfig:
    """Test configuration validation and default properties."""

    def test_default_config(self):
        cfg = LoiterCenterPlannerConfig()
        assert not cfg.enabled
        assert cfg.bias_gain == 2.5
        assert cfg.max_offset == 30.0
        assert cfg.direction == "upwind"
        assert cfg.min_wind_speed == 0.5
        assert cfg.latch_on_entry is True

    def test_invalid_bias_gain(self):
        with pytest.raises(ValueError, match="bias_gain"):
            LoiterCenterPlannerConfig(bias_gain=-1.0)

    def test_invalid_max_offset(self):
        with pytest.raises(ValueError, match="max_offset"):
            LoiterCenterPlannerConfig(max_offset=-5.0)

    def test_invalid_min_wind_speed(self):
        with pytest.raises(ValueError, match="min_wind_speed"):
            LoiterCenterPlannerConfig(min_wind_speed=-0.1)

    def test_invalid_direction(self):
        with pytest.raises(ValueError, match="direction"):
            LoiterCenterPlannerConfig(direction="crosswind")


class TestLoiterCenterPlannerLogic:
    """Test offset computation, conventions, limits, and safety."""

    def test_disabled_planner_returns_zero(self):
        planner = LoiterCenterPlanner(LoiterCenterPlannerConfig(enabled=False))
        dx, dy = planner.calculate_offset(10.0, 10.0)
        assert dx == 0.0
        assert dy == 0.0

        target = LocalPoint(1200.0, 0.0)
        center, off = planner.plan_center(target, 10.0, 10.0)
        assert center.x == 1200.0
        assert center.y == 0.0
        assert off == (0.0, 0.0)

    def test_direction_none_returns_zero(self):
        planner = LoiterCenterPlanner(
            LoiterCenterPlannerConfig(enabled=True, direction="none")
        )
        dx, dy = planner.calculate_offset(5.0, 5.0)
        assert dx == 0.0
        assert dy == 0.0

    def test_deadband_zero_wind(self):
        planner = LoiterCenterPlanner(
            LoiterCenterPlannerConfig(enabled=True, min_wind_speed=0.5)
        )
        # Below threshold
        assert planner.calculate_offset(0.2, 0.1) == (0.0, 0.0)
        # Exactly zero
        assert planner.calculate_offset(0.0, 0.0) == (0.0, 0.0)

    def test_upwind_sign_convention_east_wind(self):
        # East wind blowing toward +X (+East):
        # Upwind bias must be in -X (-West) direction.
        planner = LoiterCenterPlanner(
            LoiterCenterPlannerConfig(enabled=True, bias_gain=2.5, direction="upwind")
        )
        dx, dy = planner.calculate_offset(4.0, 0.0)
        assert dx == pytest.approx(-10.0, abs=1e-6)
        assert dy == pytest.approx(0.0, abs=1e-6)

        target = LocalPoint(1200.0, 0.0)
        center, off = planner.plan_center(target, 4.0, 0.0)
        assert center.x == pytest.approx(1190.0, abs=1e-6)
        assert center.y == pytest.approx(0.0, abs=1e-6)
        # Target position itself is unchanged
        assert target.x == 1200.0
        assert target.y == 0.0

    def test_upwind_sign_convention_north_wind(self):
        # North wind blowing toward +Y:
        # Upwind bias must be in -Y direction.
        planner = LoiterCenterPlanner(
            LoiterCenterPlannerConfig(enabled=True, bias_gain=2.5, direction="upwind")
        )
        dx, dy = planner.calculate_offset(0.0, 4.0)
        assert dx == pytest.approx(0.0, abs=1e-6)
        assert dy == pytest.approx(-10.0, abs=1e-6)

    def test_downwind_sign_convention(self):
        # East wind blowing toward +X:
        # Downwind bias must be in +X direction.
        planner = LoiterCenterPlanner(
            LoiterCenterPlannerConfig(enabled=True, bias_gain=2.5, direction="downwind")
        )
        dx, dy = planner.calculate_offset(4.0, 0.0)
        assert dx == pytest.approx(10.0, abs=1e-6)
        assert dy == pytest.approx(0.0, abs=1e-6)

    def test_diagonal_wind(self):
        # Wind blowing toward (-3, -4) with speed 5 m/s:
        # Upwind bias opposes wind, so offset should be in (+X, +Y) direction.
        planner = LoiterCenterPlanner(
            LoiterCenterPlannerConfig(enabled=True, bias_gain=2.5, direction="upwind")
        )
        dx, dy = planner.calculate_offset(-3.0, -4.0)
        # offset_dist = 2.5 * 5.0 = 12.5 m
        # unit vector = (-0.6, -0.8)
        # sign = -1.0 -> dx = -1.0 * 12.5 * -0.6 = +7.5, dy = +10.0
        assert dx == pytest.approx(7.5, abs=1e-6)
        assert dy == pytest.approx(10.0, abs=1e-6)

    def test_saturation_clamping(self):
        # Strong wind 20 m/s: nominal offset = 2.5 * 20 = 50 m.
        # Should be clamped to max_offset = 30 m.
        planner = LoiterCenterPlanner(
            LoiterCenterPlannerConfig(enabled=True, bias_gain=2.5, max_offset=30.0)
        )
        dx, dy = planner.calculate_offset(20.0, 0.0)
        assert dx == pytest.approx(-30.0, abs=1e-6)
        assert dy == pytest.approx(0.0, abs=1e-6)

    def test_numerical_safety_nan_inf(self):
        planner = LoiterCenterPlanner(LoiterCenterPlannerConfig(enabled=True))
        assert planner.calculate_offset(float("nan"), 4.0) == (0.0, 0.0)
        assert planner.calculate_offset(4.0, float("nan")) == (0.0, 0.0)
        assert planner.calculate_offset(float("inf"), 4.0) == (0.0, 0.0)
        assert planner.calculate_offset(4.0, float("-inf")) == (0.0, 0.0)

    def test_latching_and_reset(self):
        planner = LoiterCenterPlanner(
            LoiterCenterPlannerConfig(enabled=True, latch_on_entry=True)
        )
        target = LocalPoint(100.0, 200.0)

        # First plan
        c1, off1 = planner.plan_center(target, 0.0, 4.0)
        assert c1.y == pytest.approx(190.0, abs=1e-6)
        assert planner.latched_center == c1
        assert planner.latched_offset == (pytest.approx(0.0), pytest.approx(-10.0))

        # Subsequent call with changed wind should return latched center
        c2, off2 = planner.plan_center(target, 4.0, 0.0)
        assert c2 == c1
        assert off2 == off1

        # Reset clears latched center
        planner.reset()
        assert planner.latched_center is None
        assert planner.latched_offset == (0.0, 0.0)

        # After reset, replanning uses new wind
        c3, off3 = planner.plan_center(target, 4.0, 0.0)
        assert c3.x == pytest.approx(90.0, abs=1e-6)
        assert c3.y == pytest.approx(200.0, abs=1e-6)


class TestLoiterCenterPlannerIntegration:
    """Integration with ParafoilSimulator and telemetry recording."""

    def test_simulator_pass_through_when_disabled(self):
        target = LocalPoint(1200.0, 0.0)
        sim = ParafoilSimulator(
            state=ParafoilState(x=0.0, y=0.0, altitude=1000.0, vx=0.0, vy=0.0, vz=0.0, heading=0.0, heading_rate=0.0),
            drogue_config=DrogueConfig(descent_rate=17.0),
            parafoil_config=ParafoilConfig(forward_speed=8.0, descent_rate=2.5, max_turn_rate=30.0),
            dt=0.1,
            parafoil_deployment_altitude=600.0,
            target_position=target,
            wind_field=ConstantWindField(0.0, 4.0),
            loiter_center_planner=None,
        )
        assert sim.loiter_center_planner is None

    def test_simulator_applies_planned_center_in_terminal(self):
        target = LocalPoint(1200.0, 0.0)
        planner = LoiterCenterPlanner(
            LoiterCenterPlannerConfig(enabled=True, bias_gain=2.5, direction="upwind")
        )
        sim = ParafoilSimulator(
            state=ParafoilState(x=0.0, y=0.0, altitude=1000.0, vx=0.0, vy=0.0, vz=0.0, heading=0.0, heading_rate=0.0),
            drogue_config=DrogueConfig(descent_rate=17.0),
            parafoil_config=ParafoilConfig(forward_speed=8.0, descent_rate=2.5, max_turn_rate=30.0),
            dt=0.1,
            parafoil_deployment_altitude=600.0,
            target_position=target,
            wind_field=ConstantWindField(0.0, 4.0),
            loiter_center_planner=planner,
        )

        waypoints = generate_waypoints(LocalPoint(0.0, 0.0), target, 10)
        manager = WaypointManager(waypoints, acceptance_radius=25.0)
        pid = PIDController(kp=10.0, ki=0.1, kd=0.5, output_min=-1.0, output_max=1.0, integral_min=-10.0, integral_max=10.0)
        guidance = GuidanceController(manager, pid)

        def steering_cb(t, s):
            if sim.state_machine.get_state() == FlightState.FINAL_APPROACH:
                if not manager.is_complete:
                    manager.skip_to_final_target()
                    guidance.pid.reset()
            pos = LocalPoint(s.x, s.y)
            out = guidance.update(position=pos, heading=s.heading, altitude=s.altitude, dt=sim.dt)
            return out.steering

        history = sim.run(steering_cb, max_time=300.0)

        # Geographic target must remain invariant
        assert sim.target_position.x == 1200.0
        assert sim.target_position.y == 0.0

        # Planner latched center must have been set
        assert planner.latched_center is not None
        assert planner.latched_center.y < 0.0  # Upwind for +Y wind is -Y

        # Telemetry verification
        loiter_samples = [s for s in history if s.state == FlightState.TERMINAL_GUIDANCE]
        assert len(loiter_samples) > 0
        for sample in loiter_samples:
            assert sample.loiter_planned_center_x is not None
            assert sample.loiter_planned_center_y is not None
            assert sample.loiter_center_offset_x is not None
            assert sample.loiter_center_offset_y is not None
            assert sample.loiter_center_offset_y == pytest.approx(planner.latched_offset[1], abs=1e-3)
