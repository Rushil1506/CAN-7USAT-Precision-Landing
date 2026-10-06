import math

from flight.parafoil import ParafoilState, DrogueConfig, ParafoilConfig

from flight.state_machine import FlightState

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
