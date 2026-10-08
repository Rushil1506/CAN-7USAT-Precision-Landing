import math
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List

from flight.state_machine import FlightState
from flight.parafoil import ParafoilState, ParafoilConfig, DrogueConfig
from environment.wind import ConstantWindField, CosineGust, ComposedWindField, PiecewiseWindField, StochasticWind
from simulation.simulator import ParafoilSimulator
from navigation.coordinates import LocalPoint
from navigation.waypoint_manager import WaypointManager
from guidance.guidance_controller import GuidanceController
from control.pid import PIDController

def run_sim(wind_field_factory, title: str, start_pos: Tuple[float, float, float], waypoints: List[LocalPoint], seeds: List[int]):
    print(f"\n--- Running: {title} ---")
    
    td_errors = []
    
    for seed in seeds:
        state = ParafoilState(x=start_pos[0], y=start_pos[1], altitude=start_pos[2], heading=0.0, vx=0.0, vy=0.0, vz=0.0, heading_rate=0.0)
        drogue_config = DrogueConfig()
        parafoil_config = ParafoilConfig(forward_speed=8.0, descent_rate=2.5, max_turn_rate=30.0)
        
        wind_field = wind_field_factory(seed)
        
        sim = ParafoilSimulator(
            state=state,
            drogue_config=drogue_config,
            parafoil_config=parafoil_config,
            target_position=waypoints[-1],
            dt=0.1,
            wind_field=wind_field
        )
        
        pid = PIDController(kp=2.5, ki=0.1, kd=0.5)
        
        guidance = GuidanceController(
            waypoint_manager=WaypointManager(waypoints=waypoints),
            pid=pid
        )
        
        def steering_func(t, s):
            wx, wy = wind_field.wind_at(s.altitude, getattr(sim, "time", 0.0))
            guidance.set_wind(wx, wy)
            out = guidance.update(
                position=LocalPoint(s.x, s.y),
                heading=s.heading,
                altitude=s.altitude,
                dt=sim.dt
            )
            return out.steering

        sim.run(steering_function=steering_func, max_time=3000.0)
        
        td_err = math.hypot(sim.state.x - waypoints[-1].x, sim.state.y - waypoints[-1].y)
        td_errors.append(td_err)
        
        print(f"  Seed {seed:3d}: TD={td_err:5.1f}m | T={sim.time:5.1f}s")
        
        xs = [s.x for s in sim.history]
        ys = [s.y for s in sim.history]
        
        if seed == seeds[0]:  # Only plot first seed to save time
            plt.figure()
            plt.plot(xs, ys, label="Trajectory")
            plt.plot(waypoints[-1].x, waypoints[-1].y, 'rx', label="Target")
            plt.title(f"{title} (Seed {seed})")
            plt.xlabel("X [m]")
            plt.ylabel("Y [m]")
            plt.axis("equal")
            plt.legend()
            plt.grid(True)
            plt.savefig(f"plot_42c_{title.replace(' ', '_').lower()}.png")
            plt.close()
            
    avg_td = sum(td_errors)/len(td_errors)
    max_td = max(td_errors)
    print(f"  --> Avg TD: {avg_td:.1f}m | Max TD: {max_td:.1f}m")


if __name__ == "__main__":
    start_pos = (0.0, -800.0, 1000.0)
    target = LocalPoint(0.0, 0.0)
    seeds = [101, 202, 303, 404, 505]
    
    # A. Zero stochastic amplitude
    run_sim(lambda s: ComposedWindField(ConstantWindField(0.0, 0.0), StochasticWind(0.0, 10.0, s)), "A Zero Turb", start_pos, [target], seeds)
    
    # B. Low turbulence
    run_sim(lambda s: ComposedWindField(ConstantWindField(0.0, 0.0), StochasticWind(0.5, 10.0, s)), "B Low Turb", start_pos, [target], seeds)
    
    # C. Medium turbulence
    run_sim(lambda s: ComposedWindField(ConstantWindField(0.0, 0.0), StochasticWind(1.5, 10.0, s)), "C Med Turb", start_pos, [target], seeds)
    
    # D. Short correlation time
    run_sim(lambda s: ComposedWindField(ConstantWindField(0.0, 0.0), StochasticWind(1.5, 1.0, s)), "D Short Corr", start_pos, [target], seeds)
    
    # E. Long correlation time
    run_sim(lambda s: ComposedWindField(ConstantWindField(0.0, 0.0), StochasticWind(1.5, 50.0, s)), "E Long Corr", start_pos, [target], seeds)
    
    # F. Combined shear + stochastic
    pwf = PiecewiseWindField([(0.0, (2.0, 0.0)), (1000.0, (6.0, 0.0))])
    run_sim(lambda s: ComposedWindField(pwf, StochasticWind(1.5, 10.0, s)), "F Shear Turb", start_pos, [target], seeds)
    
    # G. Combined deterministic gust + stochastic
    run_sim(lambda s: ComposedWindField(ConstantWindField(0.0, 0.0), CosineGust(0.0, 4.0, 50.0, 20.0), StochasticWind(1.0, 10.0, s)), "G Gust Turb", start_pos, [target], seeds)
    
    # H. Terminal Guidance Test
    run_sim(lambda s: ComposedWindField(ConstantWindField(2.0, 0.0), StochasticWind(1.5, 10.0, s)), "H Terminal Turb", (0.0, -100.0, 1000.0), [target], seeds)
    
    # I. Final Approach Test
    run_sim(lambda s: ComposedWindField(ConstantWindField(0.0, 0.0), StochasticWind(2.0, 5.0, s)), "I Final Approach Turb", start_pos, [target], seeds)
