import math

from control.pid import PIDController
from guidance.guidance_controller import GuidanceController
from navigation.coordinates import LocalPoint
from navigation.waypoint_manager import WaypointManager

def test_radian_conversion_and_pid_response():
    manager = WaypointManager(
        [LocalPoint(100.0, 0.0)],
        acceptance_radius=5.0,
    )
    
    pid = PIDController(
        kp=2.5,
        ki=0.1,
        kd=0.5,
        output_min=-1.0,
        output_max=1.0,
        integral_min=-10.0,
        integral_max=10.0,
    )
    
    controller = GuidanceController(manager, pid)
    
    # 1. Zero error behavior
    output_zero = controller.update(
        position=LocalPoint(0.0, 0.0),
        heading=90.0,
        altitude=500.0,
        dt=0.1,
    )
    assert math.isclose(output_zero.heading_error, 0.0)
    assert math.isclose(output_zero.steering, 0.0)
    
    # 2. Small error (avoiding saturation) to check P, I, D logic
    # Previous error was 0.0. Current heading 89.0 -> Error +1.0 deg.
    output_small = controller.update(
        position=LocalPoint(0.0, 0.0),
        heading=89.0,
        altitude=500.0,
        dt=0.1,
    )
    
    error_deg_small = 1.0
    error_rad_small = math.radians(error_deg_small)
    
    expected_p = 2.5 * error_rad_small
    expected_d = 0.5 * (error_rad_small - 0.0) / 0.1
    expected_i = 0.1 * error_rad_small * 0.1
    expected_steering = expected_p + expected_d + expected_i
    
    assert math.isclose(output_small.steering, expected_steering, rel_tol=1e-5)
    assert output_small.steering < 1.0
    
    # 3. Output Saturation [-1, +1]
    pid.reset()
    output_sat_pos = controller.update(
        position=LocalPoint(0.0, 0.0),
        heading=0.0, # Error = +90 deg
        altitude=500.0,
        dt=0.1,
    )
    assert math.isclose(output_sat_pos.steering, 1.0)

    pid.reset()
    output_sat_neg = controller.update(
        position=LocalPoint(0.0, 0.0),
        heading=180.0, # Error = -90 deg
        altitude=500.0,
        dt=0.1,
    )
    assert math.isclose(output_sat_neg.steering, -1.0)
