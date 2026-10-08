import math
import numpy as np

from flight.state_machine import FlightState
from flight.parafoil import ParafoilState, ParafoilConfig, DrogueConfig
from environment.wind import ConstantWindField, StochasticWind, ComposedWindField
from simulation.simulator import ParafoilSimulator
from navigation.coordinates import LocalPoint
from navigation.waypoint_manager import WaypointManager
from guidance.guidance_controller import GuidanceController
from control.pid import PIDController

def run_mission(tau: float):
    start_pos = (0.0, -800.0, 1000.0)
    target = LocalPoint(0.0, 0.0)
    
    state = ParafoilState(x=start_pos[0], y=start_pos[1], altitude=start_pos[2], heading=0.0, vx=0.0, vy=0.0, vz=0.0, heading_rate=0.0)
    drogue_config = DrogueConfig()
    parafoil_config = ParafoilConfig(forward_speed=8.0, descent_rate=2.5, max_turn_rate=30.0)
    
    # 3 m/s disturbance
    wind_field = ComposedWindField(ConstantWindField(0.0, 0.0), StochasticWind(3.0, tau, 101))
    
    sim = ParafoilSimulator(
        state=state, drogue_config=drogue_config, parafoil_config=parafoil_config,
        target_position=target, dt=0.1, wind_field=wind_field
    )
    
    # Defaults should now be integral_min=-10.0, integral_max=10.0
    pid = PIDController(kp=2.5, ki=0.1, kd=0.5)
    guidance = GuidanceController(waypoint_manager=WaypointManager(waypoints=[target]), pid=pid)
    
    steerings = []
    sign_changes = 0
    prev_steering = 0.0
    
    def steering_func(t, s):
        nonlocal sign_changes, prev_steering
        
        wx, wy = wind_field.wind_at(s.altitude, getattr(sim, "time", 0.0))
        
        # Production controller now has EMA filtering built-in
        guidance.set_wind(wx, wy)
        
        out = guidance.update(position=LocalPoint(s.x, s.y), heading=s.heading, altitude=s.altitude, dt=sim.dt)
        
        st = out.steering
            
        steerings.append(st)
        if st * prev_steering < 0:
            sign_changes += 1
        prev_steering = st
        
        return st

    sim.run(steering_function=steering_func, max_time=3000.0)
    
    sat_frac = sum(1 for st in steerings if abs(st) >= 0.99) / len(steerings) if steerings else 0.0
    chatter_rate = sign_changes / sim.time if sim.time > 0 else 0.0
    td_err = math.hypot(sim.state.x, sim.state.y)
    
    print(f"tau={tau}s -> SatFrac: {sat_frac:.2f} | ChatterRate: {chatter_rate:.1f}/s | TD: {td_err:.1f}m")

if __name__ == "__main__":
    print("=== Validation 4.2e-b (Production Code) ===")
    run_mission(1.0)
    run_mission(50.0)
