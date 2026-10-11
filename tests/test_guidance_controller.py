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


def test_guidance_controller_final_approach_direct_bearing():
    target = LocalPoint(100.0, 0.0)
    manager = WaypointManager([target], acceptance_radius=5.0)
    manager.skip_to_final_target()
    assert manager.is_complete

    pid = PIDController(kp=10.0, ki=0.1, kd=0.5)
    controller = GuidanceController(
        manager,
        pid,
        final_approach_altitude=120.0,
        final_approach_radius=100.0,
    )

    # In final approach: altitude <= 120.0, target at (100.0, 0.0)
    # Vehicle at (50.0, 0.0), heading North (0.0 deg)
    output = controller.update(
        position=LocalPoint(50.0, 0.0),
        heading=0.0,
        altitude=100.0,
        dt=0.1,
    )

    assert output.final_approach
    assert output.target_reached
    assert math.isclose(output.desired_bearing, 90.0)
    assert math.isclose(output.heading_error, 90.0)
    assert output.steering > 0.0  # Turn right toward East


def test_guidance_controller_radian_error_scaling():
    target = LocalPoint(100.0, 0.0)
    manager = WaypointManager([target], acceptance_radius=5.0)
    pid = PIDController(kp=10.0, ki=0.0, kd=0.0)
    controller = GuidanceController(manager, pid)

    # Heading error is 10 degrees = 10 * pi / 180 = ~0.1745 rad
    # Steering should be kp * error_rad = 10.0 * 0.17453 = ~1.745 -> clamped to 1.0
    out = controller.update(
        position=LocalPoint(0.0, 0.0),
        heading=80.0,
        altitude=500.0,
        dt=0.1,
    )
    assert math.isclose(out.heading_error, 10.0)
    assert math.isclose(out.steering, 1.0)  # Saturated at 1.0


def test_guidance_controller_reset_clears_pid_and_manager():
    target = LocalPoint(100.0, 0.0)
    manager = WaypointManager([target], acceptance_radius=5.0)
    pid = PIDController(kp=10.0, ki=0.1, kd=0.5)
    controller = GuidanceController(manager, pid)

    controller.update(
        position=LocalPoint(0.0, 0.0),
        heading=0.0,
        altitude=500.0,
        dt=0.1,
    )
    assert pid.integral != 0.0 or pid.previous_error is not None

    controller.reset()
    assert pid.integral == 0.0
    assert not pid.initialized
    assert manager.current_index == 0
