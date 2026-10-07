import math
import pytest

from flight.parafoil import ParafoilState, DrogueConfig, ParafoilConfig
from flight.state_machine import FlightState
from navigation.coordinates import LocalPoint
from guidance.loiter_guidance import LoiterDirection, LoiterGuidance, LoiterPhase
from simulation.simulator import ParafoilSimulator


def create_simulator(altitude=1000.0):

    state = ParafoilState(
        x=0.0,
        y=0.0,
        altitude=altitude,
        vx=0.0,
        vy=0.0,
        vz=0.0,
        heading=0.0,
        heading_rate=0.0,
    )

    drogue = DrogueConfig(descent_rate=17.0)

    parafoil = ParafoilConfig(forward_speed=8.0, descent_rate=2.5, max_turn_rate=30.0)

    simulator = ParafoilSimulator(
        state=state,
        drogue_config=drogue,
        parafoil_config=parafoil,
        dt=0.1,
        parafoil_deployment_altitude=600.0,
        deployment_duration=2.0,
    )

    return simulator


def test_initial_phase():

    simulator = create_simulator()

    assert simulator.state_machine.get_state() == FlightState.DROGUE_DESCENT


def test_drogue_reaches_600m():

    simulator = create_simulator()

    def no_steering(time, state):
        return 0.0

    history = simulator.run(steering_function=no_steering, max_time=30.0)

    # At 17 m/s, reaching 600 m from 1000 m
    # requires approximately 23.5 seconds.
    assert any(sample.state == FlightState.PARAFOIL_DEPLOYMENT for sample in history)


def test_parafoil_deployment_occurs():

    simulator = create_simulator()

    def no_steering(time, state):
        return 0.0

    history = simulator.run(steering_function=no_steering, max_time=30.0)

    assert any(sample.state == FlightState.PARAFOIL_DEPLOYMENT for sample in history)


def test_precision_guidance_starts_after_deployment():

    simulator = create_simulator()

    def no_steering(time, state):
        return 0.0

    history = simulator.run(steering_function=no_steering, max_time=40.0)

    assert any(sample.state == FlightState.PRECISION_GUIDANCE for sample in history)


def test_drogue_does_not_receive_steering():

    simulator = create_simulator()

    def aggressive_steering(time, state):
        return 1.0

    history = simulator.run(steering_function=aggressive_steering, max_time=20.0)

    drogue_samples = [
        sample for sample in history if sample.state == FlightState.DROGUE_DESCENT
    ]

    assert len(drogue_samples) > 0

    assert all(sample.steering == 0.0 for sample in drogue_samples)


def test_parafoil_receives_steering():

    simulator = create_simulator()

    def right_steering(time, state):
        return 0.5

    history = simulator.run(steering_function=right_steering, max_time=40.0)

    guidance_samples = [
        sample for sample in history if sample.state == FlightState.PRECISION_GUIDANCE
    ]

    assert len(guidance_samples) > 0

    assert any(sample.steering > 0.0 for sample in guidance_samples)


def test_full_descent():

    simulator = create_simulator()

    def no_steering(time, state):
        return 0.0

    simulator.run(steering_function=no_steering, max_time=500.0)

    assert math.isclose(simulator.state.altitude, 0.0)


def test_terminal_guidance_transition_occurs_when_range_exceeds_threshold():
    state = ParafoilState(
        x=0.0,
        y=0.0,
        altitude=300.0,
        vx=0.0,
        vy=0.0,
        vz=0.0,
        heading=0.0,
        heading_rate=0.0,
    )
    # Rg = 8.0 * (600.0 / 2.5) = 1920.0 m.
    # Dt = 150.0 m <= R_capture (200.0 m) AND Rg >= Dt + M (250.0 m).
    # Both spatial and energy conditions are satisfied -> enters TERMINAL_GUIDANCE.
    target = LocalPoint(150.0, 0.0)
    simulator = ParafoilSimulator(
        state=state,
        drogue_config=DrogueConfig(),
        parafoil_config=ParafoilConfig(forward_speed=8.0, descent_rate=2.5),
        dt=0.1,
        target_position=target,
        terminal_margin=100.0,
        terminal_capture_radius=200.0,
    )
    simulator.state_machine.state = FlightState.PRECISION_GUIDANCE
    simulator.update_mission_state()

    assert simulator.state_machine.get_state() == FlightState.TERMINAL_GUIDANCE


def test_terminal_guidance_transition_does_not_occur_when_range_insufficient():
    state = ParafoilState(
        x=0.0,
        y=0.0,
        altitude=200.0,
        vx=0.0,
        vy=0.0,
        vz=0.0,
        heading=0.0,
        heading_rate=0.0,
    )
    # Rg = 8.0 * (200.0 / 2.5) = 640.0 m.
    # Dt = 2000.0 m, M = 100.0 m -> threshold = 2100.0 m.
    # Rg < Dt + M -> should remain in PRECISION_GUIDANCE.
    target = LocalPoint(2000.0, 0.0)
    simulator = ParafoilSimulator(
        state=state,
        drogue_config=DrogueConfig(),
        parafoil_config=ParafoilConfig(forward_speed=8.0, descent_rate=2.5),
        dt=0.1,
        target_position=target,
        terminal_margin=100.0,
    )
    simulator.state_machine.state = FlightState.PRECISION_GUIDANCE
    simulator.update_mission_state()

    assert simulator.state_machine.get_state() == FlightState.PRECISION_GUIDANCE


def test_configured_terminal_margin_is_used():
    state = ParafoilState(
        x=0.0,
        y=0.0,
        altitude=600.0,
        vx=0.0,
        vy=0.0,
        vz=0.0,
        heading=0.0,
        heading_rate=0.0,
    )
    # Rg = 1920.0 m, Dt = 1850.0 m. Spatial capture radius set wide (2000 m) to isolate energy gate.
    target = LocalPoint(1850.0, 0.0)

    # Margin = 50.0 m -> threshold = 1900.0 m -> Rg >= threshold -> enters TERMINAL_GUIDANCE
    sim_low_margin = ParafoilSimulator(
        state=state,
        drogue_config=DrogueConfig(),
        parafoil_config=ParafoilConfig(forward_speed=8.0, descent_rate=2.5),
        dt=0.1,
        target_position=target,
        terminal_margin=50.0,
        terminal_capture_radius=2000.0,
    )
    sim_low_margin.state_machine.state = FlightState.PRECISION_GUIDANCE
    sim_low_margin.update_mission_state()
    assert sim_low_margin.state_machine.get_state() == FlightState.TERMINAL_GUIDANCE

    # Margin = 100.0 m -> threshold = 1950.0 m -> Rg < threshold -> stays in PRECISION_GUIDANCE
    sim_high_margin = ParafoilSimulator(
        state=state,
        drogue_config=DrogueConfig(),
        parafoil_config=ParafoilConfig(forward_speed=8.0, descent_rate=2.5),
        dt=0.1,
        target_position=target,
        terminal_margin=100.0,
        terminal_capture_radius=2000.0,
    )
    sim_high_margin.state_machine.state = FlightState.PRECISION_GUIDANCE
    sim_high_margin.update_mission_state()
    assert sim_high_margin.state_machine.get_state() == FlightState.PRECISION_GUIDANCE


def test_spatial_gate_prevents_premature_entry_in_simulator():
    """At 600m altitude and 1200m target distance, spatial gate keeps vehicle in PRECISION_GUIDANCE."""
    state = ParafoilState(
        x=0.0,
        y=0.0,
        altitude=300.0,
        vx=0.0,
        vy=0.0,
        vz=0.0,
        heading=0.0,
        heading_rate=0.0,
    )
    target = LocalPoint(1200.0, 0.0)
    sim = ParafoilSimulator(
        state=state,
        drogue_config=DrogueConfig(),
        parafoil_config=ParafoilConfig(forward_speed=8.0, descent_rate=2.5),
        target_position=target,
        terminal_margin=100.0,
        terminal_capture_radius=200.0,
    )
    sim.state_machine.state = FlightState.PRECISION_GUIDANCE
    sim.update_mission_state()

    assert sim.state_machine.get_state() == FlightState.PRECISION_GUIDANCE


def test_exact_capture_radius_boundary_in_simulator():
    """When Dt == R_capture exactly, the spatial condition is satisfied."""
    state = ParafoilState(
        x=0.0,
        y=0.0,
        altitude=300.0,
        vx=0.0,
        vy=0.0,
        vz=0.0,
        heading=0.0,
        heading_rate=0.0,
    )
    target = LocalPoint(200.0, 0.0)
    sim = ParafoilSimulator(
        state=state,
        drogue_config=DrogueConfig(),
        parafoil_config=ParafoilConfig(forward_speed=8.0, descent_rate=2.5),
        target_position=target,
        terminal_margin=100.0,
        terminal_capture_radius=200.0,
    )
    sim.state_machine.state = FlightState.PRECISION_GUIDANCE
    sim.update_mission_state()

    assert sim.state_machine.get_state() == FlightState.TERMINAL_GUIDANCE


def test_configured_capture_radius_in_simulator():
    """Varying terminal_capture_radius alters the transition boundary in the simulator."""
    state = ParafoilState(
        x=0.0,
        y=0.0,
        altitude=300.0,
        vx=0.0,
        vy=0.0,
        vz=0.0,
        heading=0.0,
        heading_rate=0.0,
    )
    target = LocalPoint(250.0, 0.0)

    sim_narrow = ParafoilSimulator(
        state=state,
        drogue_config=DrogueConfig(),
        parafoil_config=ParafoilConfig(forward_speed=8.0, descent_rate=2.5),
        target_position=target,
        terminal_margin=100.0,
        terminal_capture_radius=200.0,
    )
    sim_narrow.state_machine.state = FlightState.PRECISION_GUIDANCE
    sim_narrow.update_mission_state()
    assert sim_narrow.state_machine.get_state() == FlightState.PRECISION_GUIDANCE

    sim_broad = ParafoilSimulator(
        state=state,
        drogue_config=DrogueConfig(),
        parafoil_config=ParafoilConfig(forward_speed=8.0, descent_rate=2.5),
        target_position=target,
        terminal_margin=100.0,
        terminal_capture_radius=300.0,
    )
    sim_broad.state_machine.state = FlightState.PRECISION_GUIDANCE
    sim_broad.update_mission_state()
    assert sim_broad.state_machine.get_state() == FlightState.TERMINAL_GUIDANCE


def test_invalid_simulator_capture_radius_raises():
    """ParafoilSimulator must reject zero or negative terminal_capture_radius."""
    state = ParafoilState(
        x=0.0, y=0.0, altitude=600.0, vx=0.0, vy=0.0, vz=0.0, heading=0.0, heading_rate=0.0
    )
    with pytest.raises(ValueError, match="Terminal capture radius must be greater than zero"):
        ParafoilSimulator(
            state=state,
            drogue_config=DrogueConfig(),
            parafoil_config=ParafoilConfig(),
            terminal_capture_radius=0.0,
        )

    with pytest.raises(ValueError, match="Terminal capture radius must be greater than zero"):
        ParafoilSimulator(
            state=state,
            drogue_config=DrogueConfig(),
            parafoil_config=ParafoilConfig(),
            terminal_capture_radius=-50.0,
        )


def test_final_approach_not_replaced_by_terminal_guidance():
    # Inside final approach bounds (altitude <= 120, distance <= 100)
    state_near = ParafoilState(
        x=50.0,
        y=0.0,
        altitude=100.0,
        vx=0.0,
        vy=0.0,
        vz=0.0,
        heading=0.0,
        heading_rate=0.0,
    )
    target = LocalPoint(0.0, 0.0)
    sim = ParafoilSimulator(
        state=state_near,
        drogue_config=DrogueConfig(),
        parafoil_config=ParafoilConfig(),
        target_position=target,
        final_approach_altitude=120.0,
        final_approach_radius=100.0,
        terminal_margin=100.0,
        terminal_capture_radius=200.0,
    )
    # 1. From PRECISION_GUIDANCE, final approach conditions transition to FINAL_APPROACH
    sim.state_machine.state = FlightState.PRECISION_GUIDANCE
    sim.update_mission_state()
    assert sim.state_machine.get_state() == FlightState.FINAL_APPROACH

    # 2. From TERMINAL_GUIDANCE, high altitude (100m) does not force FINAL_APPROACH,
    # but once altitude reaches breakout altitude, it transitions to FINAL_APPROACH.
    sim.state_machine.state = FlightState.TERMINAL_GUIDANCE
    sim.update_mission_state()
    assert sim.state_machine.get_state() == FlightState.TERMINAL_GUIDANCE
    sim.state.altitude = sim.last_breakout_geometry.breakout_altitude
    sim.update_mission_state()
    assert sim.state_machine.get_state() == FlightState.FINAL_APPROACH


def test_landing_from_terminal_guidance():
    state_ground = ParafoilState(
        x=0.0,
        y=0.0,
        altitude=0.0,
        vx=0.0,
        vy=0.0,
        vz=0.0,
        heading=0.0,
        heading_rate=0.0,
    )
    sim = ParafoilSimulator(
        state=state_ground,
        drogue_config=DrogueConfig(),
        parafoil_config=ParafoilConfig(),
    )
    sim.state_machine.state = FlightState.TERMINAL_GUIDANCE
    sim.update_mission_state()
    assert sim.state_machine.get_state() == FlightState.LANDING


# =====================================================================
# Phase 2.4b.2 Loiter Guidance Integration Tests
# =====================================================================


def test_terminal_guidance_invokes_loiter_guidance():
    """TERMINAL_GUIDANCE should invoke LoiterGuidance rather than direct-to-target PID."""
    state = ParafoilState(
        x=100.0,
        y=0.0,
        altitude=250.0,
        vx=0.0,
        vy=0.0,
        vz=0.0,
        heading=0.0,
        heading_rate=0.0,
    )
    target = LocalPoint(0.0, 0.0)
    sim = ParafoilSimulator(
        state=state,
        drogue_config=DrogueConfig(),
        parafoil_config=ParafoilConfig(forward_speed=8.0, descent_rate=2.5),
        target_position=target,
        loiter_radius=30.0,
        loiter_capture_tolerance=5.0,
        terminal_margin=100.0,
        terminal_capture_radius=200.0,
    )
    sim.state_machine.state = FlightState.TERMINAL_GUIDANCE

    called_states = []

    def dummy_steering(time, st):
        called_states.append(sim.state_machine.get_state())
        return 0.0

    history = sim.run(steering_function=dummy_steering, max_time=5.0)

    # Loiter guidance was invoked
    assert len(sim.loiter_history) > 0
    assert sim.last_loiter_output is not None
    # dummy_steering (the external callback) was never invoked while in TERMINAL_GUIDANCE
    assert FlightState.TERMINAL_GUIDANCE not in called_states


def test_direct_pid_not_used_in_terminal_guidance():
    """Verify external steering_function (direct PID) is strictly bypassed during TERMINAL_GUIDANCE."""
    state = ParafoilState(
        x=50.0,
        y=0.0,
        altitude=200.0,
        vx=0.0,
        vy=0.0,
        vz=0.0,
        heading=0.0,
        heading_rate=0.0,
    )
    target = LocalPoint(0.0, 0.0)
    sim = ParafoilSimulator(
        state=state,
        drogue_config=DrogueConfig(),
        parafoil_config=ParafoilConfig(forward_speed=8.0, descent_rate=2.5),
        target_position=target,
    )
    sim.state_machine.state = FlightState.TERMINAL_GUIDANCE

    pid_call_count = 0

    def mock_pid(time, st):
        nonlocal pid_call_count
        pid_call_count += 1
        return 0.99  # distinctive command

    history = sim.run(steering_function=mock_pid, max_time=3.0)

    assert pid_call_count == 0
    # The recorded steerings should be computed by loiter guidance, not mock_pid (0.99)
    terminal_steerings = [s.steering for s in history if s.state == FlightState.TERMINAL_GUIDANCE]
    assert all(s != 0.99 for s in terminal_steerings)


def test_terminal_guidance_orbit_capture_and_tracking_phases():
    """Vehicle starting outside orbit should first enter ORBIT_CAPTURE, then reach ORBIT_TRACKING."""
    state = ParafoilState(
        x=0.0,
        y=-60.0,  # 60m from center with radius 30m
        altitude=300.0,
        vx=0.0,
        vy=0.0,
        vz=0.0,
        heading=0.0,
        heading_rate=0.0,
    )
    target = LocalPoint(0.0, 0.0)
    sim = ParafoilSimulator(
        state=state,
        drogue_config=DrogueConfig(),
        parafoil_config=ParafoilConfig(forward_speed=8.0, descent_rate=2.5, max_turn_rate=30.0),
        target_position=target,
        loiter_radius=30.0,
        loiter_capture_tolerance=5.0,
        loiter_direction=LoiterDirection.CLOCKWISE,
        final_approach_altitude=50.0,  # low so it stays in terminal guidance
    )
    sim.state_machine.state = FlightState.TERMINAL_GUIDANCE

    history = sim.run(steering_function=lambda t, s: 0.0, max_time=25.0)

    terminal_samples = [s for s in history if s.state == FlightState.TERMINAL_GUIDANCE]
    assert len(terminal_samples) > 0

    phases = [s.loiter_phase for s in terminal_samples if s.loiter_phase is not None]
    assert LoiterPhase.ORBIT_CAPTURE in phases
    assert LoiterPhase.ORBIT_TRACKING in phases


def test_terminal_guidance_remains_active_above_breakout_altitude():
    """Terminal guidance remains active when current altitude exceeds dynamic breakout altitude."""
    state = ParafoilState(
        x=28.0,
        y=0.0,
        altitude=60.0,  # Well above ~15-20 m breakout altitude
        vx=0.0,
        vy=0.0,
        vz=0.0,
        heading=90.0,
        heading_rate=0.0,
    )
    target = LocalPoint(0.0, 0.0)
    sim = ParafoilSimulator(
        state=state,
        drogue_config=DrogueConfig(),
        parafoil_config=ParafoilConfig(forward_speed=8.0, descent_rate=2.5),
        target_position=target,
        breakout_altitude_margin=2.0,
    )
    sim.state_machine.state = FlightState.TERMINAL_GUIDANCE
    sim.update_mission_state()

    assert sim.state_machine.get_state() == FlightState.TERMINAL_GUIDANCE
    assert sim.last_breakout_geometry is not None
    assert sim.state.altitude > sim.last_breakout_geometry.breakout_altitude


def test_terminal_guidance_transitions_to_final_approach_at_breakout_altitude():
    """When altitude drops to <= breakout_altitude, TERMINAL_GUIDANCE transitions to FINAL_APPROACH."""
    state = ParafoilState(
        x=28.0,
        y=0.0,
        altitude=22.0,  # Near dynamic breakout altitude (~15-18 m)
        vx=0.0,
        vy=0.0,
        vz=0.0,
        heading=90.0,
        heading_rate=0.0,
    )
    target = LocalPoint(0.0, 0.0)
    sim = ParafoilSimulator(
        state=state,
        drogue_config=DrogueConfig(),
        parafoil_config=ParafoilConfig(forward_speed=8.0, descent_rate=2.5),
        target_position=target,
        breakout_altitude_margin=2.0,
    )
    sim.state_machine.state = FlightState.TERMINAL_GUIDANCE

    history = sim.run(steering_function=lambda t, st: 0.0, max_time=10.0)

    states = [s.state for s in history]
    assert FlightState.FINAL_APPROACH in states
    assert sim.final_approach_entry_time is not None
    assert sim.final_approach_entry_altitude is not None
    assert sim.breakout_time is not None
    assert sim.breakout_altitude is not None
    assert sim.breakout_altitude <= 22.0


def test_fixed_120m_altitude_does_not_force_transition():
    """Old 120m final_approach_altitude does not independently force transition from TERMINAL_GUIDANCE."""
    state = ParafoilState(
        x=28.0,
        y=0.0,
        altitude=125.0,
        vx=0.0,
        vy=0.0,
        vz=0.0,
        heading=90.0,
        heading_rate=0.0,
    )
    target = LocalPoint(0.0, 0.0)
    sim = ParafoilSimulator(
        state=state,
        drogue_config=DrogueConfig(),
        parafoil_config=ParafoilConfig(forward_speed=8.0, descent_rate=2.5),
        target_position=target,
        final_approach_altitude=120.0,
    )
    sim.state_machine.state = FlightState.TERMINAL_GUIDANCE

    # Run for 5 seconds (altitude drops from 125 to 112.5 m, crossing 120 m)
    history = sim.run(steering_function=lambda t, s: 0.0, max_time=5.0)

    states = [s.state for s in history]
    assert FlightState.FINAL_APPROACH not in states
    assert sim.state_machine.get_state() == FlightState.TERMINAL_GUIDANCE


def test_breakout_altitude_margin_affects_transition():
    """Higher breakout altitude margin triggers transition to FINAL_APPROACH earlier (at higher altitude)."""
    state0 = ParafoilState(x=28.0, y=0.0, altitude=30.0, vx=0.0, vy=0.0, vz=0.0, heading=90.0, heading_rate=0.0)
    state5 = ParafoilState(x=28.0, y=0.0, altitude=30.0, vx=0.0, vy=0.0, vz=0.0, heading=90.0, heading_rate=0.0)
    target = LocalPoint(0.0, 0.0)

    sim0 = ParafoilSimulator(
        state=state0,
        drogue_config=DrogueConfig(),
        parafoil_config=ParafoilConfig(forward_speed=8.0, descent_rate=2.5),
        target_position=target,
        breakout_altitude_margin=0.0,
    )
    sim0.state_machine.state = FlightState.TERMINAL_GUIDANCE
    sim0.run(steering_function=lambda t, s: 0.0, max_time=10.0)

    sim5 = ParafoilSimulator(
        state=state5,
        drogue_config=DrogueConfig(),
        parafoil_config=ParafoilConfig(forward_speed=8.0, descent_rate=2.5),
        target_position=target,
        breakout_altitude_margin=5.0,
    )
    sim5.state_machine.state = FlightState.TERMINAL_GUIDANCE
    sim5.run(steering_function=lambda t, s: 0.0, max_time=10.0)

    assert sim0.breakout_altitude is not None
    assert sim5.breakout_altitude is not None
    assert sim5.breakout_altitude > sim0.breakout_altitude
    assert sim5.breakout_time < sim0.breakout_time


def test_breakout_works_with_clockwise_loiter():
    """Energy-aligned breakout operates correctly with CLOCKWISE loiter direction."""
    state = ParafoilState(x=30.0, y=0.0, altitude=25.0, vx=0.0, vy=0.0, vz=0.0, heading=180.0, heading_rate=0.0)
    target = LocalPoint(0.0, 0.0)
    sim = ParafoilSimulator(
        state=state,
        drogue_config=DrogueConfig(),
        parafoil_config=ParafoilConfig(forward_speed=8.0, descent_rate=2.5),
        target_position=target,
        loiter_direction=LoiterDirection.CLOCKWISE,
    )
    sim.state_machine.state = FlightState.TERMINAL_GUIDANCE
    sim.run(steering_function=lambda t, s: 0.0, max_time=10.0)

    assert sim.state_machine.get_state() in (FlightState.FINAL_APPROACH, FlightState.LANDING)
    assert sim.breakout_time is not None


def test_breakout_works_with_counter_clockwise_loiter():
    """Energy-aligned breakout operates correctly with COUNTER_CLOCKWISE loiter direction."""
    state = ParafoilState(x=30.0, y=0.0, altitude=25.0, vx=0.0, vy=0.0, vz=0.0, heading=0.0, heading_rate=0.0)
    target = LocalPoint(0.0, 0.0)
    sim = ParafoilSimulator(
        state=state,
        drogue_config=DrogueConfig(),
        parafoil_config=ParafoilConfig(forward_speed=8.0, descent_rate=2.5),
        target_position=target,
        loiter_direction=LoiterDirection.COUNTER_CLOCKWISE,
    )
    sim.state_machine.state = FlightState.TERMINAL_GUIDANCE
    sim.run(steering_function=lambda t, s: 0.0, max_time=10.0)

    assert sim.state_machine.get_state() in (FlightState.FINAL_APPROACH, FlightState.LANDING)
    assert sim.breakout_time is not None


def test_final_approach_resumes_external_steering():
    """External steering callback is resumed and called during FINAL_APPROACH."""
    state = ParafoilState(x=28.0, y=0.0, altitude=22.0, vx=0.0, vy=0.0, vz=0.0, heading=90.0, heading_rate=0.0)
    target = LocalPoint(0.0, 0.0)
    sim = ParafoilSimulator(
        state=state,
        drogue_config=DrogueConfig(),
        parafoil_config=ParafoilConfig(forward_speed=8.0, descent_rate=2.5),
        target_position=target,
    )
    sim.state_machine.state = FlightState.TERMINAL_GUIDANCE

    external_calls = 0

    def steering_cb(t, st):
        nonlocal external_calls
        if sim.state_machine.get_state() == FlightState.FINAL_APPROACH:
            external_calls += 1
        return 0.0

    sim.run(steering_function=steering_cb, max_time=10.0)
    assert external_calls > 0


def test_breakout_transition_continuity():
    """Ensure kinematic state continuity (no position or altitude jumping) across breakout transition."""
    state = ParafoilState(x=28.0, y=0.0, altitude=25.0, vx=0.0, vy=0.0, vz=0.0, heading=90.0, heading_rate=0.0)
    target = LocalPoint(0.0, 0.0)
    sim = ParafoilSimulator(
        state=state,
        drogue_config=DrogueConfig(),
        parafoil_config=ParafoilConfig(forward_speed=8.0, descent_rate=2.5),
        target_position=target,
        dt=0.1,
    )
    sim.state_machine.state = FlightState.TERMINAL_GUIDANCE
    history = sim.run(steering_function=lambda t, s: 0.0, max_time=10.0)

    trans_idx = None
    for i in range(len(history) - 1):
        if history[i].state == FlightState.TERMINAL_GUIDANCE and history[i + 1].state == FlightState.FINAL_APPROACH:
            trans_idx = i
            break

    assert trans_idx is not None
    s_before = history[trans_idx]
    s_after = history[trans_idx + 1]

    dx = s_after.x - s_before.x
    dy = s_after.y - s_before.y
    horiz_jump = (dx * dx + dy * dy) ** 0.5
    vert_jump = abs(s_after.altitude - s_before.altitude)

    # In 0.1s at 8 m/s, max horizontal travel is 0.8m
    assert horiz_jump < 1.0
    # In 0.1s at 2.5 m/s descent, altitude change is 0.25m
    assert vert_jump < 0.35


def test_low_altitude_infeasibility_triggers_failsafe():
    """Low-altitude guard triggers FAILSAFE if altitude is insufficient to complete breakout turn."""
    # Heading 90 deg, target at (0,0), position at (20, 0). Heading points away from target.
    # Required turn is ~180 deg, min turn altitude L_turn / G ≈ 20 m.
    # At altitude 2.0 m, vehicle cannot turn in time, so failsafe is triggered.
    state = ParafoilState(x=20.0, y=0.0, altitude=2.0, vx=0.0, vy=0.0, vz=0.0, heading=90.0, heading_rate=0.0)
    target = LocalPoint(0.0, 0.0)
    sim = ParafoilSimulator(
        state=state,
        drogue_config=DrogueConfig(),
        parafoil_config=ParafoilConfig(forward_speed=8.0, descent_rate=2.5, max_turn_rate=30.0),
        target_position=target,
    )
    sim.state_machine.state = FlightState.TERMINAL_GUIDANCE
    sim.update_mission_state()

    assert sim.state_machine.get_state() == FlightState.FAILSAFE


def test_landing_from_final_approach():
    """Verify clean landing transition when altitude reaches 0 in FINAL_APPROACH."""
    state = ParafoilState(x=5.0, y=0.0, altitude=1.0, vx=0.0, vy=0.0, vz=0.0, heading=0.0, heading_rate=0.0)
    target = LocalPoint(0.0, 0.0)
    sim = ParafoilSimulator(
        state=state,
        drogue_config=DrogueConfig(),
        parafoil_config=ParafoilConfig(forward_speed=8.0, descent_rate=2.5),
        target_position=target,
    )
    sim.state_machine.state = FlightState.FINAL_APPROACH
    history = sim.run(steering_function=lambda t, s: 0.0, max_time=2.0)

    assert sim.state_machine.get_state() == FlightState.LANDING
    assert history[-1].state == FlightState.LANDING


def test_terminal_guidance_timeout_triggers_failsafe():
    """When max_terminal_guidance_time is exceeded, transition to FAILSAFE occurs."""
    state = ParafoilState(
        x=20.0,
        y=0.0,
        altitude=400.0,  # High altitude
        vx=0.0,
        vy=0.0,
        vz=0.0,
        heading=0.0,
        heading_rate=0.0,
    )
    target = LocalPoint(0.0, 0.0)
    sim = ParafoilSimulator(
        state=state,
        drogue_config=DrogueConfig(),
        parafoil_config=ParafoilConfig(forward_speed=8.0, descent_rate=2.5),
        target_position=target,
        final_approach_altitude=120.0,
        max_terminal_guidance_time=5.0,  # 5-second timeout guard
    )
    sim.state_machine.state = FlightState.TERMINAL_GUIDANCE
    # Run a few steps to let it initialize terminal entry logic
    sim.step()
    # Now force the time past the calculated timeout
    sim.time += sim.max_terminal_guidance_time + 10.0
    sim.step()

    assert sim.state_machine.get_state() == FlightState.FAILSAFE

def test_invalid_loiter_parameters_raise():
    """Verify rejection of invalid loiter parameters in simulator constructor."""
    state = ParafoilState(x=0.0, y=0.0, altitude=500.0, vx=0.0, vy=0.0, vz=0.0, heading=0.0, heading_rate=0.0)
    drogue = DrogueConfig()
    parafoil = ParafoilConfig()

    with pytest.raises(ValueError, match="Loiter radius must be greater than zero"):
        ParafoilSimulator(state=state, drogue_config=drogue, parafoil_config=parafoil, loiter_radius=0.0)

    with pytest.raises(ValueError, match="Loiter radius must be greater than zero"):
        ParafoilSimulator(state=state, drogue_config=drogue, parafoil_config=parafoil, loiter_radius=-10.0)

    with pytest.raises(ValueError, match="Loiter capture tolerance must be greater than zero"):
        ParafoilSimulator(state=state, drogue_config=drogue, parafoil_config=parafoil, loiter_capture_tolerance=0.0)

    with pytest.raises(ValueError, match="Max terminal guidance time must be greater than zero"):
        ParafoilSimulator(state=state, drogue_config=drogue, parafoil_config=parafoil, max_terminal_guidance_time=0.0)


def test_invalid_breakout_margin_raises():
    """Verify rejection of negative breakout altitude margin in simulator constructor."""
    state = ParafoilState(x=0.0, y=0.0, altitude=500.0, vx=0.0, vy=0.0, vz=0.0, heading=0.0, heading_rate=0.0)
    drogue = DrogueConfig()
    parafoil = ParafoilConfig()

    with pytest.raises(ValueError, match="Breakout altitude margin cannot be negative"):
        ParafoilSimulator(
            state=state,
            drogue_config=drogue,
            parafoil_config=parafoil,
            breakout_altitude_margin=-1.0,
        )


# =====================================================================
# Phase 3.1 Wind Telemetry Tests
# =====================================================================

def test_simulator_telemetry_zero_wind():
    sim = create_simulator()
    sim.state.heading = 90.0
    sim.state_machine.state = FlightState.PRECISION_GUIDANCE
    sim.step()
    s = sim.history[-1]
    assert math.isclose(s.ground_speed, 8.0)
    assert math.isclose(s.course_over_ground, 90.0)
    assert math.isclose(s.crab_angle, 0.0)


def test_simulator_telemetry_headwind():
    sim = create_simulator()
    sim.state.heading = 0.0
    sim.parafoil_config.wind_y = -3.0
    sim.state_machine.state = FlightState.PRECISION_GUIDANCE
    sim.step()
    s = sim.history[-1]
    assert math.isclose(s.ground_speed, 5.0)
    assert math.isclose(s.course_over_ground, 0.0)
    assert math.isclose(s.crab_angle, 0.0)


def test_simulator_telemetry_tailwind():
    sim = create_simulator()
    sim.state.heading = 0.0
    sim.parafoil_config.wind_y = 3.0
    sim.state_machine.state = FlightState.PRECISION_GUIDANCE
    sim.step()
    s = sim.history[-1]
    assert math.isclose(s.ground_speed, 11.0)
    assert math.isclose(s.course_over_ground, 0.0)
    assert math.isclose(s.crab_angle, 0.0)


def test_simulator_telemetry_crosswind():
    sim = create_simulator()
    sim.state.heading = 0.0
    sim.parafoil_config.wind_x = 6.0
    sim.state_machine.state = FlightState.PRECISION_GUIDANCE
    sim.step()
    s = sim.history[-1]
    assert math.isclose(s.ground_speed, 10.0)
    expected_cog = math.degrees(math.atan2(6.0, 8.0))
    assert math.isclose(s.course_over_ground, expected_cog)
    assert math.isclose(s.crab_angle, expected_cog)


def test_simulator_telemetry_diagonal_wind():
    sim = create_simulator()
    sim.state.heading = 45.0
    sim.parafoil_config.wind_x = 2.0
    sim.parafoil_config.wind_y = -2.0
    sim.state_machine.state = FlightState.PRECISION_GUIDANCE
    sim.step()
    s = sim.history[-1]
    expected_vx = 8.0 * math.sin(math.radians(45.0)) + 2.0
    expected_vy = 8.0 * math.cos(math.radians(45.0)) - 2.0
    assert math.isclose(s.ground_speed, math.hypot(expected_vx, expected_vy))


def test_simulator_drogue_wind_drift():
    sim = create_simulator()
    sim.parafoil_config.wind_x = 5.0
    sim.parafoil_config.wind_y = -5.0
    sim.state_machine.state = FlightState.DROGUE_DESCENT
    sim.step()
    s = sim.history[-1]
    assert math.isclose(s.vx, 5.0)
    assert math.isclose(s.vy, -5.0)
    assert math.isclose(s.ground_speed, math.hypot(5.0, -5.0))


def test_simulator_crab_angle_wrap():
    sim = create_simulator()
    sim.state.heading = 350.0
    sim.state_machine.state = FlightState.DROGUE_DESCENT
    sim.parafoil_config.wind_x = math.sin(math.radians(10.0))
    sim.parafoil_config.wind_y = math.cos(math.radians(10.0))
    sim.step()
    s = sim.history[-1]
    assert math.isclose(s.course_over_ground, 10.0)
    assert math.isclose(s.crab_angle, 20.0)
    
    sim2 = create_simulator()
    sim2.state.heading = 10.0
    sim2.state_machine.state = FlightState.DROGUE_DESCENT
    sim2.parafoil_config.wind_x = math.sin(math.radians(350.0))
    sim2.parafoil_config.wind_y = math.cos(math.radians(350.0))
    sim2.step()
    s2 = sim2.history[-1]
    assert math.isclose(s2.course_over_ground, 350.0)
    assert math.isclose(s2.crab_angle, -20.0)
