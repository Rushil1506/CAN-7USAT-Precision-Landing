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

def analyze_sensitivity():
    # 3. Heading Unit Scale
    print("=== 3. HEADING-UNIT SCALE ===")
    kp = 2.5
    errors = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    for e in errors:
        p = kp * e
        p_rad = kp * math.radians(e)
        print(f"Error: {e:4.1f} deg -> P_deg = {p:6.2f} | P_rad = {p_rad:6.2f}")
    sat_err = 1.0 / kp
    print(f"P term saturates (|u|=1) at error = {sat_err:.2f} deg\n")

    # 4. Derivative Term Analysis
    print("=== 4. DERIVATIVE TERM ANALYSIS ===")
    kd = 0.5
    dt = 0.1
    for step in errors:
        d = kd * step / dt
        print(f"Step: {step:4.1f} deg in {dt}s -> D = {d:6.2f}")
    sat_step = (1.0 / kd) * dt
    print(f"D term saturates (|u|=1) at step = {sat_step:.2f} deg\n")

    # Run Decompositions
    print("=== 2. DECOMPOSITION ===")
    scenarios = {
        "Zero Wind": ConstantWindField(0.0, 0.0),
        "4m/s Crosswind": ConstantWindField(4.0, 0.0),
        "Deterministic Gust": ComposedWindField(ConstantWindField(0.0, 0.0), CosineGust(0.0, 3.0, 20.0, 2.0)),
        "Stochastic Tau=1s": ComposedWindField(ConstantWindField(0.0, 0.0), StochasticWind(3.0, 1.0, 101)),
        "Stochastic Tau=50s": ComposedWindField(ConstantWindField(0.0, 0.0), StochasticWind(3.0, 50.0, 101))
    }

    for name, wf in scenarios.items():
        run_scenario(name, wf)

def run_scenario(name, wind_field):
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
        out = guidance.update(position=LocalPoint(s.x, s.y), heading=s.heading, altitude=s.altitude, dt=sim.dt)
        err = pid.previous_error
        p = pid.kp * err
        i = pid.ki * pid.integral
        d = pid.kd * (err - prev_error) / sim.dt if sim.dt > 0 else 0
        
        trace_data.append({
            'time': t, 'p': p, 'i': i, 'd': d,
            'total_unsat': p+i+d, 'steer': out.steering,
            'err': err
        })
        return out.steering

    sim.run(steering_func, max_time=400.0)
    
    sats = [d for d in trace_data if abs(d['steer']) >= 0.99]
    unsats = [d for d in trace_data if abs(d['steer']) < 0.99 and abs(d['steer']) > 0.01]
    
    print(f"Scenario: {name}")
    if sats:
        s = sats[len(sats)//2]
        print(f"  Saturated Case -> P: {s['p']:.2f}, I: {s['i']:.2f}, D: {s['d']:.2f} | Unsat Total: {s['total_unsat']:.2f} | Final: {s['steer']:.2f}")
    if unsats:
        u = unsats[len(unsats)//2]
        print(f"  Unsaturated Case -> P: {u['p']:.2f}, I: {u['i']:.2f}, D: {u['d']:.2f} | Unsat Total: {u['total_unsat']:.2f} | Final: {u['steer']:.2f}")

def sweep_gains():
    print("\n=== 6. OFFLINE PARAMETER SENSITIVITY STUDY ===")
    gains = [
        (2.5, 0.5),    # baseline
        (1.0, 0.2),    # reduced both
        (0.5, 0.1),    # further reduced
        (0.1, 0.05),   # moderate
        (0.022, 0.004) # original production?
    ]
    
    wind = ComposedWindField(ConstantWindField(0.0, 0.0), StochasticWind(3.0, 5.0, 101))
    
    for kp, kd in gains:
        run_sweep_sim(kp, kd, wind)

def run_sweep_sim(kp, kd, wind_field):
    start_pos = (0.0, -800.0, 1000.0)
    target = LocalPoint(0.0, 0.0)
    state = ParafoilState(x=start_pos[0], y=start_pos[1], altitude=start_pos[2], heading=0.0, vx=0.0, vy=0.0, vz=0.0, heading_rate=0.0)
    sim = ParafoilSimulator(
        state=state, drogue_config=DrogueConfig(), parafoil_config=ParafoilConfig(forward_speed=8.0, descent_rate=2.5, max_turn_rate=30.0),
        target_position=target, dt=0.1, wind_field=wind_field
    )
    
    pid = PIDController(kp=kp, ki=0.1, kd=kd, integral_min=-10.0, integral_max=10.0)
    guidance = GuidanceController(waypoint_manager=WaypointManager(waypoints=[target]), pid=pid)
    
    steerings = []
    flips = 0
    max_err = 0.0
    last_steer = 0.0
    
    def steering_func(t, s):
        nonlocal flips, max_err, last_steer
        est_wx = sim.wind_estimator.wind_x
        est_wy = sim.wind_estimator.wind_y
        guidance.set_wind(est_wx, est_wy)
        out = guidance.update(position=LocalPoint(s.x, s.y), heading=s.heading, altitude=s.altitude, dt=sim.dt)
        steerings.append(out.steering)
        if last_steer * out.steering < -0.5:
            flips += 1
        last_steer = out.steering
        max_err = max(max_err, abs(pid.previous_error))
        return out.steering

    sim.run(steering_func, max_time=400.0)
    
    sat_ratio = sum(1 for st in steerings if abs(st) >= 0.99) / max(1, len(steerings))
    td_err = math.hypot(sim.state.x, sim.state.y)
    
    print(f"Kp={kp:.3f}, Kd={kd:.3f} | Sat: {sat_ratio*100:5.1f}% | Flips: {flips:4} | MaxErr: {max_err:5.1f} | TD Err: {td_err:5.1f}m")

if __name__ == '__main__':
    analyze_sensitivity()
    sweep_gains()
