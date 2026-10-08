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
from navigation.heading import heading_error
from navigation.bearing import bearing

def trace_stochastic(tau: float):
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
    
    pid = PIDController(kp=2.5, ki=0.1, kd=0.5, integral_min=-10.0, integral_max=10.0)
    guidance = GuidanceController(waypoint_manager=WaypointManager(waypoints=[target]), pid=pid)
    
    trace_data = []
    
    def steering_func(t, s):
        # Read the shared FSW estimator from the simulator wrapper
        est_wx = sim.wind_estimator.wind_x
        est_wy = sim.wind_estimator.wind_y
        
        guidance.set_wind(est_wx, est_wy)
        
        prev_error = pid.previous_error
        
        out = guidance.update(position=LocalPoint(s.x, s.y), heading=s.heading, altitude=s.altitude, dt=sim.dt)
        
        err = pid.previous_error # This is the current error just set by update
        p_term = pid.kp * err
        i_term = pid.ki * pid.integral
        d_term = pid.kd * (err - prev_error) / sim.dt if sim.dt > 0 else 0
        
        trace_data.append({
            'time': t,
            'phase': sim.state_machine.get_state().name,
            'p_term': p_term,
            'i_term': i_term,
            'd_term': d_term,
            'steering': out.steering
        })
        return out.steering

    sim.run(steering_func, max_time=400.0)
    
    # Analyze
    sat_count = sum(1 for d in trace_data if abs(d['steering']) >= 0.99)
    sat_ratio = sat_count / len(trace_data) if trace_data else 0.0
    
    # Calculate chatter (steering sign flips or large derivatives)
    flips = sum(1 for i in range(1, len(trace_data)) if trace_data[i]['steering'] * trace_data[i-1]['steering'] < -0.5)
    
    print(f"=== TAU={tau}s Trace ===")
    print(f"Saturation: {sat_ratio*100:.1f}%")
    print(f"Chatter (flips): {flips}")
    
    # Look at a slice of P/I/D contributions when saturated
    sats = [d for d in trace_data if abs(d['steering']) >= 0.99]
    if sats:
        s = sats[len(sats)//2]
        print(f"Example Saturated Step: P={s['p_term']:.2f}, I={s['i_term']:.2f}, D={s['d_term']:.2f}, Total={s['p_term']+s['i_term']+s['d_term']:.2f} -> saturated to {s['steering']:.2f}")

def trace_crosswind_cte():
    start_pos = (0.0, -800.0, 1000.0)
    target = LocalPoint(0.0, 0.0)
    
    state = ParafoilState(x=start_pos[0], y=start_pos[1], altitude=start_pos[2], heading=0.0, vx=0.0, vy=0.0, vz=0.0, heading_rate=0.0)
    drogue_config = DrogueConfig()
    parafoil_config = ParafoilConfig(forward_speed=8.0, descent_rate=2.5, max_turn_rate=30.0)
    
    wind_field = ConstantWindField(4.0, 0.0) # 4 m/s pure crosswind (blowing East)
    
    sim = ParafoilSimulator(
        state=state, drogue_config=drogue_config, parafoil_config=parafoil_config,
        target_position=target, dt=0.1, wind_field=wind_field
    )
    
    pid = PIDController(kp=2.5, ki=0.1, kd=0.5, integral_min=-10.0, integral_max=10.0)
    guidance = GuidanceController(waypoint_manager=WaypointManager(waypoints=[target]), pid=pid)
    
    # We want to trace position and heading to show lack of lateral CTE feedback
    trace_data = []
    def steering_func(t, s):
        est_wx = sim.wind_estimator.wind_x
        est_wy = sim.wind_estimator.wind_y
        guidance.set_wind(est_wx, est_wy)
        out = guidance.update(position=LocalPoint(s.x, s.y), heading=s.heading, altitude=s.altitude, dt=sim.dt)
        
        # Calculate ground velocity and ground track
        ground_vx = s.vx
        ground_vy = s.vy
        track = math.degrees(math.atan2(ground_vx, ground_vy))
        
        trace_data.append({
            'time': t,
            'x': s.x,
            'y': s.y,
            'heading': s.heading,
            'track': track,
            'bearing_to_target': bearing(LocalPoint(s.x, s.y), target),
            'desired_heading': out.desired_heading
        })
        return out.steering

    sim.run(steering_func, max_time=400.0)
    
    print("\n=== CTE Crosswind Analysis ===")
    mid_idx = len(trace_data) // 2
    mid = trace_data[mid_idx]
    end = trace_data[-1]
    
    print(f"Mid-flight Pos: X={mid['x']:.1f}, Y={mid['y']:.1f}")
    print(f"Mid-flight Bearing to Target: {mid['bearing_to_target']:.1f} deg")
    print(f"Mid-flight Desired Heading: {mid['desired_heading']:.1f} deg")
    print(f"Mid-flight Actual Heading: {mid['heading']:.1f} deg")
    print(f"Mid-flight Ground Track: {mid['track']:.1f} deg")
    print(f"Mid-flight Track Error vs Bearing: {heading_error(mid['track'], mid['bearing_to_target']):.1f} deg")
    
    print(f"\nTerminal Pos: X={end['x']:.1f}, Y={end['y']:.1f}")
    print(f"Terminal Miss Distance: {math.hypot(end['x'], end['y']):.1f} m")
    print("Conclusion: The vehicle correctly crabs (actual heading != ground track) such that ground track roughly equals bearing to target. However, bearing alone does not correct lateral displacement (CTE), so it steadily drifts parallel or converges too slowly.")

if __name__ == '__main__':
    trace_stochastic(1.0)
    trace_stochastic(50.0)
    trace_crosswind_cte()
