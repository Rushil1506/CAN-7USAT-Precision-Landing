from flight.state_machine import FlightState
from validate_wind_baseline import run_mission_with_wind
import guidance.guidance_controller as gc

def get_end_cte(wind_x, wind_y, comp=True):
    if not comp:
        old_update = gc.GuidanceController.update
        def uncomp_update(self, *args, **kwargs):
            self.estimated_wind_x = 0.0
            self.estimated_wind_y = 0.0
            return old_update(self, *args, **kwargs)
        gc.GuidanceController.update = uncomp_update
        
    sim, history = run_mission_with_wind(wind_x, wind_y)
    
    if not comp:
        gc.GuidanceController.update = old_update
        
    transit_samples = [s for s in history if s.state == FlightState.PRECISION_GUIDANCE]
    return abs(transit_samples[-1].y) if transit_samples else 0.0

print(f"COMP END CTE +Y: {get_end_cte(0, 3, True):.2f}m")
print(f"UNCOMP END CTE +Y: {get_end_cte(0, 3, False):.2f}m")
