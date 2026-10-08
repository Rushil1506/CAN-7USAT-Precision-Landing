import math
from control.pid import PIDController
from flight.parafoil import ParafoilConfig, ParafoilState, DrogueConfig
from environment.wind import ConstantWindField, CosineGust, ComposedWindField, WindField
from simulation.simulator import ParafoilSimulator
from navigation.coordinates import LocalPoint
from navigation.waypoint_manager import WaypointManager
from guidance.guidance_controller import GuidanceController
from navigation.bearing import bearing
from flight.state_machine import FlightState
import copy
import random
from typing import Tuple

class StochasticWind:
    def __init__(self, std_dev: float, correlation_time: float, seed: int = 42):
        self.std_dev = std_dev
        self.correlation_time = correlation_time
        self.seed = seed
        self.rng = random.Random(seed)
        self.last_time = 0.0
        self.current_wx = 0.0
        self.current_wy = 0.0
        
    def evaluate(self, time: float) -> Tuple[float, float]:
        if time == 0.0 and self.last_time > 0.0:
            # Ignore bogus time=0 calls from planners
            return (self.current_wx, self.current_wy)
            
        if time < self.last_time:
            self.rng = random.Random(self.seed)
            self.last_time = 0.0
            self.current_wx = 0.0
            self.current_wy = 0.0
            
        if time <= self.last_time:
            return (self.current_wx, self.current_wy)
            
        dt = time - self.last_time
        self.last_time = time
        
        theta = 1.0 / self.correlation_time
        sigma_noise = self.std_dev * math.sqrt(2.0 * theta)
        
        self.current_wx += -theta * self.current_wx * dt + sigma_noise * math.sqrt(dt) * self.rng.gauss(0.0, 1.0)
        self.current_wy += -theta * self.current_wy * dt + sigma_noise * math.sqrt(dt) * self.rng.gauss(0.0, 1.0)
        
        limit = 3.0 * self.std_dev
        self.current_wx = max(-limit, min(limit, self.current_wx))
        self.current_wy = max(-limit, min(limit, self.current_wy))
        
        return (self.current_wx, self.current_wy)


def get_sim_config(wind_field):
    start_pos = (0.0, -800.0, 1000.0)
    target = LocalPoint(0.0, 0.0)
    state = ParafoilState(x=start_pos[0], y=start_pos[1], altitude=start_pos[2], heading=0.0, vx=0.0, vy=0.0, vz=0.0, heading_rate=0.0)
    sim = ParafoilSimulator(
        state=state, drogue_config=DrogueConfig(descent_rate=17.0), parafoil_config=ParafoilConfig(forward_speed=8.0, descent_rate=2.5, max_turn_rate=30.0),
        target_position=target, dt=0.1, wind_field=wind_field,
        parafoil_deployment_altitude=600.0, deployment_duration=2.0,
        final_approach_altitude=120.0, final_approach_radius=100.0,
        terminal_margin=100.0, terminal_capture_radius=200.0,
        loiter_radius=30.0, loiter_capture_tolerance=5.0
    )
    manager = WaypointManager([target], acceptance_radius=25.0)
    return sim, manager, target

def run_evaluation(kp, ki, kd, wind_field, label=""):
    sim, manager, target = get_sim_config(wind_field)
    pid = PIDController(kp=kp, ki=ki, kd=kd, integral_min=-10.0, integral_max=10.0)
    guidance = GuidanceController(waypoint_manager=manager, pid=pid, final_approach_altitude=120.0, final_approach_radius=100.0)
    
    trace = []
    
    def steering_func(t, s):
        if sim.state_machine.get_state() == FlightState.FINAL_APPROACH:
            if not manager.is_complete:
                manager.skip_to_final_target()
                guidance.pid.reset()
                
        est_wx = sim.wind_estimator.wind_x
        est_wy = sim.wind_estimator.wind_y
        guidance.set_wind(est_wx, est_wy)
        
        prev_err = pid.previous_error
        out = guidance.update(position=LocalPoint(s.x, s.y), heading=s.heading, altitude=s.altitude, dt=sim.dt)
        
        err_rad = pid.previous_error
        p = pid.kp * err_rad
        i = pid.ki * pid.integral
        d = pid.kd * (err_rad - prev_err) / sim.dt if sim.dt > 0 else 0
        
        trace.append({
            'time': t, 'state': sim.state_machine.get_state(),
            'steer': out.steering, 'err_rad': err_rad,
            'p': p, 'i': i, 'd': d, 'turn_rate': s.heading_rate
        })
        return out.steering

    history = sim.run(steering_func, max_time=500.0)
    
    # Calculate metrics
    steerings = [d['steer'] for d in trace if d['state'] in (FlightState.PRECISION_GUIDANCE, FlightState.FINAL_APPROACH)]
    sats = sum(1 for st in steerings if abs(st) >= 0.99)
    sat_ratio = (sats / len(steerings)) if steerings else 0.0
    
    chatter = 0
    for j in range(1, len(steerings)):
        if steerings[j] * steerings[j-1] < -0.5:
            chatter += 1
            
    td_err = math.hypot(sim.state.x, sim.state.y)
    
    p_vals = [abs(d['p']) for d in trace]
    i_vals = [abs(d['i']) for d in trace]
    d_vals = [abs(d['d']) for d in trace]
    
    max_p = max(p_vals) if p_vals else 0
    max_i = max(i_vals) if i_vals else 0
    max_d = max(d_vals) if d_vals else 0
    max_tr = max([abs(d['turn_rate']) for d in trace]) if trace else 0
    
    # Check terminal stability
    terminal_stable = sim.terminal_entry_time is not None and sim.loiter_capture_time is not None
    
    return {
        'sat': sat_ratio,
        'chatter': chatter,
        'td_err': td_err,
        'max_p': max_p, 'max_i': max_i, 'max_d': max_d, 'max_tr': max_tr,
        'terminal_stable': terminal_stable
    }

def print_result(res, label):
    sat_pct = res['sat']*100
    print(f"{label:<25} | Err: {res['td_err']:6.1f}m | Sat: {sat_pct:4.1f}% | Chtr: {res['chatter']:3} | Max P/I/D: {res['max_p']:4.1f}/{res['max_i']:4.1f}/{res['max_d']:4.1f} | Max TR: {res['max_tr']:4.1f} | Term Stable: {res['terminal_stable']}")

def analytical_proportional_band():
    print("=== 2. ANALYTICAL PROPORTIONAL BAND ===")
    errors_deg = [5, 10, 15, 20, 30, 45, 90, 180]
    errors_rad = [math.radians(e) for e in errors_deg]
    
    kps = [2.5, 5.0, 7.5, 10.0, 15.0]
    
    print(f"{'kp':<5} | " + " | ".join([f"{e:^5}°" for e in errors_deg]))
    print("-" * 75)
    for kp in kps:
        p_vals = [kp * e for e in errors_rad]
        print(f"{kp:<5.1f} | " + " | ".join([f"{p:5.2f}" for p in p_vals]))
    
    print("\nError (deg) to reach saturation (P=1.0):")
    for kp in kps:
        e_sat = math.degrees(1.0 / kp)
        print(f"kp = {kp:<4.1f} -> {e_sat:.1f}°")
    print()

def sweep_kp():
    print("=== 3. KP SENSITIVITY STUDY ===")
    # 4m/s crosswind + Tau=1s stochastic
    scenarios = {
        "Zero Wind": ConstantWindField(0.0, 0.0),
        "Crosswind 4m/s": ConstantWindField(4.0, 0.0),
        "Tau=1s Stochastic": ComposedWindField(ConstantWindField(0.0, 0.0), StochasticWind(3.0, 1.0, 101)),
    }
    kps = [2.5, 5.0, 7.5, 10.0, 15.0, 20.0]
    
    for name, wf in scenarios.items():
        print(f"\nScenario: {name}")
        for kp in kps:
            res = run_evaluation(kp, 0.1, 0.5, wf, f"kp={kp}")
            print_result(res, f"kp={kp}")

def sweep_kd():
    print("\n=== 4. KD SENSITIVITY STUDY ===")
    # Fix kp = 10.0
    kp = 10.0
    kds = [0.0, 0.5, 1.0, 2.0, 5.0]
    wf = ComposedWindField(ConstantWindField(0.0, 0.0), StochasticWind(3.0, 1.0, 101))
    
    for kd in kds:
        res = run_evaluation(kp, 0.1, kd, wf, f"kd={kd}")
        print_result(res, f"kd={kd}")

def sweep_ki():
    print("\n=== 5. KI SENSITIVITY STUDY ===")
    kp = 10.0
    kd = 1.0
    kis = [0.0, 0.1, 0.5, 1.0, 2.0]
    wf = ConstantWindField(4.0, 0.0)  # Crosswind to test steady state error
    
    for ki in kis:
        res = run_evaluation(kp, ki, kd, wf, f"ki={ki}")
        print_result(res, f"ki={ki}")

if __name__ == "__main__":
    analytical_proportional_band()
    sweep_kp()
    sweep_kd()
    sweep_ki()
