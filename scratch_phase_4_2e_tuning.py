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

def run_tuning(tau: float, kp: float, ki: float, kd: float, integral_max: float, filter_alpha: float = 1.0):
    start_pos = (0.0, -800.0, 1000.0)
    target = LocalPoint(0.0, 0.0)
    
    state = ParafoilState(x=start_pos[0], y=start_pos[1], altitude=start_pos[2], heading=0.0, vx=0.0, vy=0.0, vz=0.0, heading_rate=0.0)
    drogue_config = DrogueConfig()
    parafoil_config = ParafoilConfig(forward_speed=8.0, descent_rate=2.5, max_turn_rate=30.0)
    
    wind_field = ComposedWindField(ConstantWindField(0.0, 0.0), StochasticWind(3.0, tau, 101))
    
    sim = ParafoilSimulator(
        state=state, drogue_config=drogue_config, parafoil_config=parafoil_config,
        target_position=target, dt=0.1, wind_field=wind_field
    )
    
    pid = PIDController(kp=kp, ki=ki, kd=kd, integral_min=-integral_max, integral_max=integral_max)
    guidance = GuidanceController(waypoint_manager=WaypointManager(waypoints=[target]), pid=pid)
    
    cross_track_errors = []
    steerings = []
    sign_changes = 0
    prev_steering = 0.0
    
    filtered_heading_error = 0.0
    
    def steering_func(t, s):
        nonlocal sign_changes, prev_steering, filtered_heading_error
        
        wx, wy = wind_field.wind_at(s.altitude, getattr(sim, "time", 0.0))
        
        # We can implement EMA on wind if filter_alpha < 1.0
        if not hasattr(steering_func, 'filtered_wx'):
            steering_func.filtered_wx = wx
            steering_func.filtered_wy = wy
            
        if filter_alpha < 1.0:
            steering_func.filtered_wx = filter_alpha * wx + (1.0 - filter_alpha) * steering_func.filtered_wx
            steering_func.filtered_wy = filter_alpha * wy + (1.0 - filter_alpha) * steering_func.filtered_wy
        else:
            steering_func.filtered_wx = wx
            steering_func.filtered_wy = wy
            
        guidance.set_wind(steering_func.filtered_wx, steering_func.filtered_wy)
        
        out = guidance.update(position=LocalPoint(s.x, s.y), heading=s.heading, altitude=s.altitude, dt=sim.dt)
        
        st = out.steering
        if filter_alpha < 1.0:
            # We can't easily hook into GuidanceController without copy-pasting its internals, 
            # let's just do it directly.
            pass
            
        steerings.append(st)
        if st * prev_steering < 0:
            sign_changes += 1
        prev_steering = st
        
        return st

    sim.run(steering_function=steering_func, max_time=3000.0)
    
    sat_frac = sum(1 for st in steerings if abs(st) >= 0.99) / len(steerings) if steerings else 0.0
    chatter_rate = sign_changes / sim.time if sim.time > 0 else 0.0
    td_err = math.hypot(sim.state.x, sim.state.y)
    
    print(f"kp={kp:.1f} ki={ki:.2f} kd={kd:.2f} i_max={integral_max:.1f} -> SatFrac: {sat_frac:.2f} | ChatterRate: {chatter_rate:.1f}/s | TD: {td_err:.1f}m")

if __name__ == "__main__":
    for tau in [1.0, 50.0]:
        print(f"\n=== Testing tau = {tau}s ===")
        print("Baseline:")
        run_tuning(tau, 2.5, 0.1, 0.5, 1.0)
        
        print("Zero KD (Removes derivative chatter):")
        run_tuning(tau, 2.5, 0.1, 0.0, 1.0, 1.0)
        
        print("Larger integral max (Fix sluggishness):")
        run_tuning(tau, 2.5, 0.1, 0.0, 10.0, 1.0)
        
        print("Filtered wind (alpha=0.05) [Fix chatter]:")
        run_tuning(tau, 2.5, 0.1, 0.5, 1.0, 0.05)
        
        print("Filtered wind (alpha=0.05) + Larger I_max [Fix both]:")
        run_tuning(tau, 2.5, 0.1, 0.5, 10.0, 0.05)
