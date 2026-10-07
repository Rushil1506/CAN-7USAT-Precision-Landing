import math
from flight.state_machine import FlightState
from validate_wind_baseline import run_mission_with_wind

def get_max_crosstrack(wind_x, wind_y):
    sim, history = run_mission_with_wind(wind_x, wind_y)
    # Transit phase = PRECISION_GUIDANCE
    transit_samples = [s for s in history if s.state == FlightState.PRECISION_GUIDANCE]
    
    # Target is (1200, 0)
    # Cross-track error is simply abs(Y) since start is (0,0) and target is along X axis.
    max_cte = max([abs(s.y) for s in transit_samples]) if transit_samples else 0.0
    return max_cte

print(f"CTE for +Y 3 m/s: {get_max_crosstrack(0, 3):.2f}m")
print(f"CTE for -Y 3 m/s: {get_max_crosstrack(0, -3):.2f}m")
print(f"CTE for Diagonal NE (2.12, 2.12): {get_max_crosstrack(2.12, 2.12):.2f}m")
