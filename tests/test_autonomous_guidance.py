import math

from control.pid import PIDController
from flight.parafoil import DrogueConfig, ParafoilConfig, ParafoilState
from flight.state_machine import FlightState
from guidance.guidance_controller import GuidanceController
from navigation.coordinates import LocalPoint
from navigation.distance import distance
from navigation.waypoints import generate_waypoints
from navigation.waypoint_manager import WaypointManager
from simulation.simulator import ParafoilSimulator


def test_autonomous_guidance_reaches_target_without_manual_steering():
    start = LocalPoint(0.0, 0.0)
    target = LocalPoint(1200.0, 0.0)

    waypoints = generate_waypoints(start, target, 10)
    manager = WaypointManager(waypoints, acceptance_radius=25.0)

    guidance = GuidanceController(
        manager,
        PIDController(
            kp=0.022,
            ki=0.00015,
            kd=0.004,
            output_min=-1.0,
            output_max=1.0,
            integral_min=-200.0,
            integral_max=200.0,
        ),
        final_approach_altitude=120.0,
        final_approach_radius=100.0,
    )

    state = ParafoilState(
        x=0.0,
        y=0.0,
        altitude=1000.0,
        vx=0.0,
        vy=0.0,
        vz=0.0,
        heading=0.0,
        heading_rate=0.0,
    )

    simulator = ParafoilSimulator(
        state=state,
        drogue_config=DrogueConfig(descent_rate=17.0),
        parafoil_config=ParafoilConfig(
            forward_speed=8.0,
            descent_rate=2.5,
            max_turn_rate=30.0,
        ),
        dt=0.1,
        parafoil_deployment_altitude=600.0,
        deployment_duration=2.0,
        target_position=target,
        final_approach_altitude=120.0,
        final_approach_radius=100.0,
    )

    def autonomous_steering(_time, current_state):
        if simulator.state_machine.get_state() == FlightState.FINAL_APPROACH:
            if not manager.is_complete:
                manager.skip_to_final_target()
                guidance.pid.reset()
        output = guidance.update(
            position=LocalPoint(current_state.x, current_state.y),
            heading=current_state.heading,
            altitude=current_state.altitude,
            dt=simulator.dt,
        )
        return output.steering

    history = simulator.run(autonomous_steering, max_time=500.0)

    landing = LocalPoint(simulator.state.x, simulator.state.y)
    landing_error = distance(landing, target)

    assert history[-1].state == FlightState.LANDING
    assert manager.is_complete
    assert math.isclose(simulator.state.altitude, 0.0)
    assert landing_error < 50.0
