import math

from control.pid import PIDController
from guidance.guidance_controller import GuidanceController
from navigation.coordinates import LocalPoint
from navigation.waypoint_manager import WaypointManager


def test_guidance_points_east_from_north_heading():
    manager = WaypointManager(
        [LocalPoint(100.0, 0.0)],
        acceptance_radius=5.0,
    )

    pid = PIDController(
        kp=0.02,
        ki=0.0,
        kd=0.0,
    )

    controller = GuidanceController(manager, pid)

    output = controller.update(
        position=LocalPoint(0.0, 0.0),
        heading=0.0,
        altitude=500.0,
        dt=0.1,
    )

    assert math.isclose(output.desired_bearing, 90.0)
    assert math.isclose(output.heading_error, 90.0)
    assert output.steering > 0.0
    assert output.waypoint_number == 1
    assert not output.target_reached


def test_waypoint_reached_resets_to_next_target():
    manager = WaypointManager(
        [
            LocalPoint(10.0, 0.0),
            LocalPoint(20.0, 0.0),
        ],
        acceptance_radius=2.0,
    )

    pid = PIDController(kp=0.02, ki=0.0, kd=0.0)
    controller = GuidanceController(manager, pid)

    output = controller.update(
        position=LocalPoint(9.0, 0.0),
        heading=90.0,
        altitude=500.0,
        dt=0.1,
    )

    assert output.waypoint_reached
    assert output.waypoint_number == 2
    assert math.isclose(output.desired_bearing, 90.0)


def test_final_approach_flag():
    manager = WaypointManager(
        [LocalPoint(100.0, 0.0)],
        acceptance_radius=5.0,
    )

    controller = GuidanceController(
        manager,
        PIDController(kp=0.02, ki=0.0, kd=0.0),
        final_approach_altitude=120.0,
        final_approach_radius=100.0,
    )

    output = controller.update(
        position=LocalPoint(40.0, 0.0),
        heading=90.0,
        altitude=100.0,
        dt=0.1,
    )

    assert output.final_approach
