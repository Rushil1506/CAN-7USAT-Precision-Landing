import numpy as np
from navigation.coordinates import LocalPoint
from navigation.distance import distance
from validate_wind_baseline import run_mission_with_wind
from flight.state_machine import FlightState

# Let's run a baseline scenario with exactly the identical setup.
sim, history = run_mission_with_wind(3.0, 0.0)

# Check radial error calculation
term_history = [s for s in history if s.state == FlightState.TERMINAL_GUIDANCE]
orbit_tracking = [s for s in term_history if s.loiter_phase is not None and (s.loiter_phase == "ORBIT_TRACKING" or getattr(s.loiter_phase, 'name', None) == "ORBIT_TRACKING")]

if orbit_tracking and hasattr(sim, 'terminal_orbit_plan'):
    center = sim.terminal_orbit_plan.orbit_center
    r_target = sim.loiter_radius
    
    actual_errors = []
    for s in orbit_tracking:
        p = LocalPoint(s.x, s.y)
        dist = distance(p, center)
        err = abs(dist - r_target)
        actual_errors.append(err)
        
    print(f"Computed Actual Mean Radial Error: {np.mean(actual_errors):.4f} m")
    print(f"Computed Actual Max Radial Error: {np.max(actual_errors):.4f} m")
    print(f"Number of tracking samples: {len(orbit_tracking)}")

    # Check reported loiter_radial_error from telemetry
    reported = [abs(s.loiter_radial_error) for s in orbit_tracking if s.loiter_radial_error is not None]
    if reported:
        print(f"Reported Mean Radial Error: {np.mean(reported):.4f} m")
        print(f"Reported Max Radial Error: {np.max(reported):.4f} m")

print(f"Touchdown Error: {distance(LocalPoint(history[-1].x, history[-1].y), LocalPoint(1200.0, 0.0)):.2f} m")
