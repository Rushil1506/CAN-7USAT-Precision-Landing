import math
from flight.state_machine import FlightState
from validate_wind_baseline import run_mission_with_wind

def get_max_crosstrack_uncomp(wind_x, wind_y):
    import guidance.guidance_controller as gc
    old_update = gc.GuidanceController.update
    
    def uncomp_update(self, *args, **kwargs):
        wx, wy = self.estimated_wind_x, self.estimated_wind_y
        self.estimated_wind_x = 0.0
        self.estimated_wind_y = 0.0
        res = old_update(self, *args, **kwargs)
        self.estimated_wind_x = wx
        self.estimated_wind_y = wy
        return res
        
    gc.GuidanceController.update = uncomp_update
    
    sim, history = run_mission_with_wind(wind_x, wind_y)
    transit_samples = [s for s in history if s.state == FlightState.PRECISION_GUIDANCE]
    
    max_cte = max([abs(s.y) for s in transit_samples]) if transit_samples else 0.0
    
    gc.GuidanceController.update = old_update
    return max_cte

print(f"UNCOMP CTE for +Y 3 m/s: {get_max_crosstrack_uncomp(0, 3):.2f}m")
print(f"UNCOMP CTE for -Y 3 m/s: {get_max_crosstrack_uncomp(0, -3):.2f}m")
print(f"UNCOMP CTE for Diagonal NE: {get_max_crosstrack_uncomp(2.12, 2.12):.2f}m")
