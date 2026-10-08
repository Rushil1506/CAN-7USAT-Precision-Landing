import math
import numpy as np
from typing import Tuple, List, Dict

from flight.state_machine import FlightState
from flight.parafoil import ParafoilState, ParafoilConfig, DrogueConfig
from environment.wind import ConstantWindField, StochasticWind, ComposedWindField
from simulation.simulator import ParafoilSimulator
from navigation.coordinates import LocalPoint
from navigation.waypoint_manager import WaypointManager
from guidance.guidance_controller import GuidanceController
from control.pid import PIDController

def compute_cross_track_error(x, y, start_wp, target_wp):
    dx = target_wp.x - start_wp.x
    dy = target_wp.y - start_wp.y
    length = math.hypot(dx, dy)
    if length == 0:
        return math.hypot(x - target_wp.x, y - target_wp.y)
    
    # Distance from point to line
    # line eq: (target_y - start_y)*x - (target_x - start_x)*y + target_x*start_y - target_y*start_x = 0
    num = abs(dy * x - dx * y + target_wp.x * start_wp.y - target_wp.y * start_wp.x)
    return num / length

def run_sim(amp: float, tau: float, seed: int, title: str, start_pos: Tuple[float, float, float], phase_trigger: str):
    target = LocalPoint(0.0, 0.0)
    start_wp = LocalPoint(start_pos[0], start_pos[1])
    
    state = ParafoilState(x=start_pos[0], y=start_pos[1], altitude=start_pos[2], heading=0.0, vx=0.0, vy=0.0, vz=0.0, heading_rate=0.0)
    drogue_config = DrogueConfig()
    parafoil_config = ParafoilConfig(forward_speed=8.0, descent_rate=2.5, max_turn_rate=30.0)
    
    base = ConstantWindField(0.0, 0.0)
    # Give a base wind so we have a nominal crab if we want, but let's use 0.0 for pure stochastic
    stoch = StochasticWind(std_dev=amp, correlation_time=tau, seed=seed)
    
    # To test directional cases, maybe base wind is not 0
    # But for amplitude grid, base wind = 0 is fine
    wind_field = ComposedWindField(base, stoch)
    
    sim = ParafoilSimulator(
        state=state,
        drogue_config=drogue_config,
        parafoil_config=parafoil_config,
        target_position=target,
        dt=0.1,
        wind_field=wind_field
    )
    
    pid = PIDController(kp=2.5, ki=0.1, kd=0.5)
    guidance = GuidanceController(waypoint_manager=WaypointManager(waypoints=[target]), pid=pid)
    
    cross_track_errors = []
    steerings = []
    
    def steering_func(t, s):
        wx, wy = wind_field.wind_at(s.altitude, getattr(sim, "time", 0.0))
        guidance.set_wind(wx, wy)
        out = guidance.update(
            position=LocalPoint(s.x, s.y),
            heading=s.heading,
            altitude=s.altitude,
            dt=sim.dt
        )
        
        # Collect metrics
        cte = compute_cross_track_error(s.x, s.y, start_wp, target)
        cross_track_errors.append(cte)
        steerings.append(out.steering)
        
        return out.steering

    sim.run(steering_function=steering_func, max_time=3000.0)
    
    td_err = math.hypot(sim.state.x - target.x, sim.state.y - target.y)
    max_cte = max(cross_track_errors) if cross_track_errors else 0.0
    rms_cte = math.sqrt(sum(e**2 for e in cross_track_errors)/len(cross_track_errors)) if cross_track_errors else 0.0
    
    sat_count = sum(1 for st in steerings if abs(st) >= 0.99)
    sat_frac = sat_count / len(steerings) if steerings else 0.0
    
    # Classification
    classification = "STABLE"
    if amp >= 8.0:
        classification = "PHYSICALLY-INFEASIBLE"
    elif td_err > 50.0:
        classification = "MISSION-FAIL"
    elif sat_frac > 0.3:
        classification = "SATURATION-LIMITED"
    elif max_cte > 25.0:
        classification = "DEGRADED"
        
    return {
        "title": title,
        "seed": seed,
        "amp": amp,
        "tau": tau,
        "td_err": td_err,
        "max_cte": max_cte,
        "rms_cte": rms_cte,
        "sat_frac": sat_frac,
        "classification": classification
    }

if __name__ == "__main__":
    seeds = [101] # just use one seed for fast scanning, or [101, 202]
    amplitudes = [0.0, 1.0, 3.0, 6.0, 10.0]  # low, med, high, infeasible
    taus = [1.0, 5.0, 20.0, 50.0] # very short, short, med, long
    
    results = []
    
    # 2D Grid Sweep
    print("Running 2D Grid Sweep (Amp vs Tau)...")
    for amp in amplitudes:
        for tau in taus:
            res = run_sim(amp, tau, 101, f"Amp={amp}_Tau={tau}", (0.0, -800.0, 1000.0), "PRECISION")
            results.append(res)
            print(f"  Amp: {amp:4.1f}, Tau: {tau:4.1f} -> Class: {res['classification']:15s} | TD: {res['td_err']:6.1f} | MaxCTE: {res['max_cte']:6.1f} | SatFrac: {res['sat_frac']:.2f}")

    print("\n--- Summary ---")
    print("Zero Disturbance Regression check:")
    z = [r for r in results if r['amp'] == 0.0]
    for r in z:
        print(f"  Tau {r['tau']}: TD={r['td_err']:.1f}")

