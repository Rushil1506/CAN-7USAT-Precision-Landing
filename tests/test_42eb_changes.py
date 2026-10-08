import pytest
from control.pid import PIDController
from guidance.guidance_controller import GuidanceController
from navigation.waypoint_manager import WaypointManager
from navigation.coordinates import LocalPoint

def test_pid_integral_bounds():
    pid = PIDController(kp=1.0, ki=1.0, kd=0.0)
    
    # Wind up the integral
    for _ in range(50):
        pid.update(error=10.0, dt=1.0)
    
    assert pid.integral <= 10.0
    
    # Reverse error to wind down
    for _ in range(100):
        pid.update(error=-10.0, dt=1.0)
        
    assert pid.integral >= -10.0


