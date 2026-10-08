import math
import numpy as np

from flight.state_machine import FlightState
from flight.parafoil import ParafoilState, ParafoilConfig, DrogueConfig
from environment.wind import ConstantWindField, CosineGust, ComposedWindField
from simulation.simulator import ParafoilSimulator
from navigation.coordinates import LocalPoint
from navigation.waypoint_manager import WaypointManager
from guidance.guidance_controller import GuidanceController
from control.pid import PIDController
from navigation.heading import heading_error

def trace_deterministic_gust():
    start_pos = (0.0, -800.0, 1000.0)
    target = LocalPoint(0.0, 0.0)
    
    state = ParafoilState(x=start_pos[0], y=start_pos[1], altitude=start_pos[2], heading=0.0, vx=0.0, vy=0.0, vz=0.0, heading_rate=0.0)
    drogue_config = DrogueConfig()
    parafoil_config = ParafoilConfig(forward_speed=8.0, descent_rate=2.5, max_turn_rate=30.0)
    
    # 3 m/s high-freq deterministic gusts
    gust1 = CosineGust(0.0, 3.0, 30.0, 2.0)
    gust2 = CosineGust(0.0, -3.0, 32.0, 2.0)
    gust3 = CosineGust(3.0, 0.0, 34.0, 2.0)
    wind_field = ComposedWindField(ConstantWindField(0.0, 0.0), gust1, gust2, gust3)
    
    sim = ParafoilSimulator(
        state=state, drogue_config=drogue_config, parafoil_config=parafoil_config,
        target_position=target, dt=0.1, wind_field=wind_field
    )
    
    pid = PIDController(kp=2.5, ki=0.1, kd=0.5, integral_min=-10.0, integral_max=10.0)
    guidance = GuidanceController(waypoint_manager=WaypointManager(waypoints=[target]), pid=pid)
    
    trace_data = []
    
    def steering_func(t, s):
        sim_time = getattr(sim, "time", 0.0)
        wx, wy = wind_field.wind_at(s.altitude, sim_time)
        
        guidance.set_wind(wx, wy)
        
        filtered_wx = guidance.estimated_wind_x
        filtered_wy = guidance.estimated_wind_y
        
        from guidance.wind_compensation import calculate_wind_corrected_heading
        from navigation.bearing import bearing
        
        raw_comp = calculate_wind_corrected_heading(
            desired_course_deg=bearing(LocalPoint(s.x, s.y), guidance.waypoint_manager.target),
            airspeed=guidance.airspeed,
            wind_x=wx,
            wind_y=wy
        )
        
        out = guidance.update(position=LocalPoint(s.x, s.y), heading=s.heading, altitude=s.altitude, dt=sim.dt)
        
        st = out.steering
            
        if 29.0 <= sim_time <= 37.0:
            trace_data.append({
                'time': sim_time,
                'raw_wx': wx, 'raw_wy': wy,
                'filt_wx': filtered_wx, 'filt_wy': filtered_wy,
                'raw_hdg': raw_comp.heading_deg,
                'filt_hdg': out.desired_heading,
                'err': out.heading_error,
                'p': out.heading_error * pid.kp,
                'i': pid.integral * pid.ki,
                'st': st,
                'sat': abs(st) >= 0.99
            })
            
        return st

    sim.run(steering_function=steering_func, max_time=100.0)
    return trace_data

def trace_integral_authority(i_max):
    start_pos = (0.0, -800.0, 1000.0)
    target = LocalPoint(0.0, 0.0)
    
    state = ParafoilState(x=start_pos[0], y=start_pos[1], altitude=start_pos[2], heading=0.0, vx=0.0, vy=0.0, vz=0.0, heading_rate=0.0)
    drogue_config = DrogueConfig()
    parafoil_config = ParafoilConfig(forward_speed=8.0, descent_rate=2.5, max_turn_rate=30.0)
    
    # 4 m/s steady crosswind (+X direction)
    wind_field = ConstantWindField(4.0, 0.0)
    
    sim = ParafoilSimulator(
        state=state, drogue_config=drogue_config, parafoil_config=parafoil_config,
        target_position=target, dt=0.1, wind_field=wind_field
    )
    
    pid = PIDController(kp=2.5, ki=0.1, kd=0.5, integral_min=-i_max, integral_max=i_max)
    guidance = GuidanceController(waypoint_manager=WaypointManager(waypoints=[target]), pid=pid)
    
    trace_data = []
    
    def steering_func(t, s):
        sim_time = getattr(sim, "time", 0.0)
        wx, wy = wind_field.wind_at(s.altitude, sim_time)
        
        guidance.set_wind(wx, wy)
        out = guidance.update(position=LocalPoint(s.x, s.y), heading=s.heading, altitude=s.altitude, dt=sim.dt)
        st = out.steering
        
        trace_data.append({
            'time': sim_time,
            'i_state': pid.integral,
            'i_term': pid.integral * pid.ki,
            'st': st,
            'cte': abs(s.x) # target is at 0,0, and we start at y=-800, so x is cross-track
        })
            
        return st

    sim.run(steering_function=steering_func, max_time=3000.0)
    
    # Steady state is roughly from t=50 to t=100
    steady = [d for d in trace_data if 50.0 <= d['time'] <= 100.0]
    
    if not steady:
        return 0, 0, 0, 0, 0
        
    max_i = max(abs(d['i_state']) for d in steady)
    mean_i = sum(d['i_state'] for d in steady) / len(steady)
    mean_i_term = sum(d['i_term'] for d in steady) / len(steady)
    sat_frac = sum(1 for d in steady if abs(d['st']) >= 0.99) / len(steady)
    mean_cte = sum(d['cte'] for d in steady) / len(steady)
    
    return max_i, mean_i, mean_i_term, sat_frac, mean_cte

if __name__ == "__main__":
    print("=== EMA VERIFICATION (Deterministic Gusts) ===")
    data = trace_deterministic_gust()
    print("time | raw_wx | raw_wy | filt_wx | filt_wy | raw_hdg | filt_hdg | st")
    for d in data[0:20:2]:
        print(f"{d['time']:4.1f} | {d['raw_wx']:6.2f} | {d['raw_wy']:6.2f} | {d['filt_wx']:7.2f} | {d['filt_wy']:7.2f} | {d['raw_hdg']:7.2f} | {d['filt_hdg']:8.2f} | {d['st']:5.2f}")
        
    rms_wind = np.sqrt(np.mean([(d['raw_wx']-d['filt_wx'])**2 + (d['raw_wy']-d['filt_wy'])**2 for d in data]))
    rms_hdg = np.sqrt(np.mean([heading_error(d['raw_hdg'], d['filt_hdg'])**2 for d in data]))
    
    raw_hdg_steps = [abs(heading_error(data[i]['raw_hdg'], data[i-1]['raw_hdg'])) for i in range(1, len(data))]
    filt_hdg_steps = [abs(heading_error(data[i]['filt_hdg'], data[i-1]['filt_hdg'])) for i in range(1, len(data))]
    
    print(f"\nRMS Wind Diff: {rms_wind:.3f} m/s")
    print(f"RMS Hdg Diff:  {rms_hdg:.3f} deg")
    print(f"Raw Hdg Step Max: {np.max(raw_hdg_steps):.2f} deg")
    print(f"Filt Hdg Step Max: {np.max(filt_hdg_steps):.2f} deg")

    print("\n=== INTEGRAL AUTHORITY VERIFICATION (4m/s Crosswind) ===")
    max_i_1, mean_i_1, mean_iterm_1, sat_1, cte_1 = trace_integral_authority(1.0)
    print(f"Limit =  1.0 -> Max I: {max_i_1:.2f}, Mean I: {mean_i_1:.2f}, I Term: {mean_iterm_1:.2f}, SatFrac: {sat_1:.2f}, CTE: {cte_1:.2f}m")
    
    max_i_10, mean_i_10, mean_iterm_10, sat_10, cte_10 = trace_integral_authority(10.0)
    print(f"Limit = 10.0 -> Max I: {max_i_10:.2f}, Mean I: {mean_i_10:.2f}, I Term: {mean_iterm_10:.2f}, SatFrac: {sat_10:.2f}, CTE: {cte_10:.2f}m")
