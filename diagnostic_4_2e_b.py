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

def trace_mission(tau: float):
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
        sim_time = getattr(sim, "time", 0.0)
        wx, wy = wind_field.wind_at(s.altitude, sim_time)
        
        guidance.set_wind(wx, wy)
        
        filtered_wx = guidance.estimated_wind_x
        filtered_wy = guidance.estimated_wind_y
        
        # Calculate raw desired heading for comparison
        from guidance.wind_compensation import calculate_wind_corrected_heading
        from navigation.bearing import bearing
        raw_comp = calculate_wind_corrected_heading(
            desired_course_deg=bearing(LocalPoint(s.x, s.y), guidance.waypoint_manager.target),
            airspeed=guidance.airspeed,
            wind_x=wx,
            wind_y=wy
        )
        raw_heading = raw_comp.heading_deg
        
        out = guidance.update(position=LocalPoint(s.x, s.y), heading=s.heading, altitude=s.altitude, dt=sim.dt)
        filtered_heading = out.desired_heading
        
        st = out.steering
            
        trace_data.append({
            'time': sim_time,
            'raw_wx': wx,
            'raw_wy': wy,
            'filt_wx': filtered_wx,
            'filt_wy': filtered_wy,
            'raw_hdg': raw_heading,
            'filt_hdg': filtered_heading,
            'st': st,
            'pid_p': pid.previous_error * pid.kp, # approx
            'pid_i': pid.integral,
            'pid_d': pid.kd * (out.heading_error - getattr(steering_func, 'prev_err', out.heading_error)) / sim.dt if sim.dt > 0 else 0.0,
            'error': out.heading_error
        })
        steering_func.prev_err = out.heading_error
        
        return st

    sim.run(steering_function=steering_func, max_time=3000.0)
    return trace_data

if __name__ == "__main__":
    print("=== TRACE tau=1.0s ===")
    data = trace_mission(1.0)
    
    print(f"Total calls: {len(data)}")
    
    # Print sample
    print("time | raw_wx | raw_wy | filt_wx | filt_wy | raw_mag | filt_mag")
    for d in data[100:110]:
        r_mag = math.hypot(d['raw_wx'], d['raw_wy'])
        f_mag = math.hypot(d['filt_wx'], d['filt_wy'])
        print(f"{d['time']:.1f} | {d['raw_wx']:.3f} | {d['raw_wy']:.3f} | {d['filt_wx']:.3f} | {d['filt_wy']:.3f} | {r_mag:.3f} | {f_mag:.3f}")
        
    rms_diff = np.sqrt(np.mean([(d['raw_wx']-d['filt_wx'])**2 + (d['raw_wy']-d['filt_wy'])**2 for d in data]))
    print(f"RMS(raw_wind - filtered_wind) = {rms_diff:.3f} m/s")
    
    rms_hdg = np.sqrt(np.mean([heading_error(d['raw_hdg'], d['filt_hdg'])**2 for d in data]))
    print(f"RMS heading difference = {rms_hdg:.3f} deg")
    
    max_hdg = np.max([abs(heading_error(d['raw_hdg'], d['filt_hdg'])) for d in data])
    print(f"Peak heading difference = {max_hdg:.3f} deg")
    
    st_sat = sum(1 for d in data if abs(d['st']) >= 0.99) / len(data)
    print(f"Saturation fraction = {st_sat:.3f}")
    
    max_i = np.max([abs(d['pid_i']) for d in data])
    print(f"Max integral state = {max_i:.3f}")
    
    # Heading jitter (diff between steps)
    raw_hdg_steps = [abs(heading_error(data[i]['raw_hdg'], data[i-1]['raw_hdg'])) for i in range(1, len(data))]
    filt_hdg_steps = [abs(heading_error(data[i]['filt_hdg'], data[i-1]['filt_hdg'])) for i in range(1, len(data))]
    
    print(f"Raw Hdg Step Max: {np.max(raw_hdg_steps):.2f} deg, Std: {np.std(raw_hdg_steps):.2f} deg")
    print(f"Filt Hdg Step Max: {np.max(filt_hdg_steps):.2f} deg, Std: {np.std(filt_hdg_steps):.2f} deg")
