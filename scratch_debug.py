import math
import numpy as np

from control.pid import PIDController
from flight.parafoil import ParafoilConfig, ParafoilState, DrogueConfig
from environment.wind import ConstantWindField, StochasticWind, CosineGust, ComposedWindField
from simulation.simulator import ParafoilSimulator
from navigation.coordinates import LocalPoint
from navigation.waypoint_manager import WaypointManager
from guidance.guidance_controller import GuidanceController
from navigation.bearing import bearing

def run_scenario(name, wind_field, trace_ab=False):
    start_pos = (0.0, -800.0, 1000.0)
    target = LocalPoint(0.0, 0.0)
    state = ParafoilState(x=start_pos[0], y=start_pos[1], altitude=start_pos[2], heading=0.0, vx=0.0, vy=0.0, vz=0.0, heading_rate=0.0)
    sim = ParafoilSimulator(
        state=state, drogue_config=DrogueConfig(), parafoil_config=ParafoilConfig(forward_speed=8.0, descent_rate=2.5, max_turn_rate=30.0),
        target_position=target, dt=0.1, wind_field=wind_field
    )
    
    pid = PIDController(kp=2.5, ki=0.1, kd=0.5, integral_min=-10.0, integral_max=10.0)
    guidance = GuidanceController(waypoint_manager=WaypointManager(waypoints=[target]), pid=pid)
    
    trace_data = []
    
    def steering_func(t, s):
        est_wx = sim.wind_estimator.wind_x
        est_wy = sim.wind_estimator.wind_y
        guidance.set_wind(est_wx, est_wy)
        
        prev_error = pid.previous_error
        
        # update guidance -> converts to rad internally!
        out = guidance.update(position=LocalPoint(s.x, s.y), heading=s.heading, altitude=s.altitude, dt=sim.dt)
        
        err_rad = pid.previous_error  # This is now in radians because GuidanceController passed radians
        p = pid.kp * err_rad
        i = pid.ki * pid.integral
        d = pid.kd * (err_rad - prev_error) / sim.dt if sim.dt > 0 else 0
        
        trace_data.append({
            'time': t, 'err_deg': math.degrees(err_rad), 'err_rad': err_rad,
            'p': p, 'i': i, 'd': d, 'total_unsat': p+i+d,
            'steer': out.steering, 'heading': s.heading, 'turn_rate': s.heading_rate
        })
        return out.steering

    sim.run(steering_func, max_time=400.0)
    
    steerings = [d['steer'] for d in trace_data]
    sats = sum(1 for st in steerings if abs(st) >= 0.99)
    sat_ratio = sats / max(1, len(steerings))
    
    chatter = 0
    for i in range(1, len(steerings)):
        if steerings[i] * steerings[i-1] < -0.5:
            chatter += 1
            
    max_p = max(abs(d['p']) for d in trace_data)
    max_i = max(abs(d['i']) for d in trace_data)
    max_d = max(abs(d['d']) for d in trace_data)
    td_err = math.hypot(sim.state.x, sim.state.y)
    
    print(f"[{name}] Saturation: {sat_ratio*100:5.1f}% | Chatter: {chatter:4} | Max P/I/D: {max_p:4.2f} / {max_i:4.2f} / {max_d:4.2f} | Landing Err: {td_err:5.1f}m")
    
    if trace_ab:
        print("\n=== A/B REGRESSION TRACE ===")
        print(f"{'Time':<6} | {'Err Deg':<8} | {'Err Rad':<8} | {'P':<6} | {'I':<6} | {'D':<6} | {'Unsat':<6} | {'Steer':<6} | {'Heading':<8} | {'TurnRate':<8}")
        print("-" * 90)
        # Print early sequence where turnaround happens (e.g. t=10 to t=15)
        for d in trace_data:
            if 10.0 <= d['time'] <= 12.0:
                print(f"{d['time']:<6.1f} | {d['err_deg']:<8.2f} | {d['err_rad']:<8.4f} | {d['p']:<6.2f} | {d['i']:<6.2f} | {d['d']:<6.2f} | {d['total_unsat']:<6.2f} | {d['steer']:<6.2f} | {d['heading']:<8.2f} | {d['turn_rate']:<8.2f}")
        print()

def validate_all():
    scenarios = {
        "Zero Wind": ConstantWindField(0.0, 0.0),
        "2m/s Crosswind": ConstantWindField(2.0, 0.0),
        "4m/s Crosswind": ConstantWindField(4.0, 0.0),
        "Deterministic Gust": ComposedWindField(ConstantWindField(0.0, 0.0), CosineGust(0.0, 3.0, 20.0, 2.0)),
        "Stochastic Tau=1s": ComposedWindField(ConstantWindField(0.0, 0.0), StochasticWind(3.0, 1.0, 101)),
        "Stochastic Tau=5s": ComposedWindField(ConstantWindField(0.0, 0.0), StochasticWind(3.0, 5.0, 101)),
        "Stochastic Tau=20s": ComposedWindField(ConstantWindField(0.0, 0.0), StochasticWind(3.0, 20.0, 101)),
        "Stochastic Tau=50s": ComposedWindField(ConstantWindField(0.0, 0.0), StochasticWind(3.0, 50.0, 101))
    }

    print("=== CLOSED-LOOP VALIDATION METRICS ===")
    for name, wf in scenarios.items():
        # Do AB trace for Deterministic Gust
        trace_ab = (name == "Stochastic Tau=1s")
        run_scenario(name, wf, trace_ab)

if __name__ == '__main__':
    validate_all()
