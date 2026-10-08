import math
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List

from flight.state_machine import FlightState
from flight.parafoil import ParafoilState, ParafoilConfig, DrogueConfig
from environment.wind import ConstantWindField, CosineGust, ComposedWindField, PiecewiseWindField
from simulation.simulator import ParafoilSimulator
from navigation.coordinates import LocalPoint
from navigation.waypoint_manager import WaypointManager
from guidance.guidance_controller import GuidanceController
from control.pid import PIDController

def run_sim(wind_field, title: str, start_pos: Tuple[float, float, float], waypoints: List[LocalPoint]):
    print(f"\n--- Running: {title} ---")
    
    state = ParafoilState(x=start_pos[0], y=start_pos[1], altitude=start_pos[2], heading=0.0, vx=0.0, vy=0.0, vz=0.0, heading_rate=0.0)
    drogue_config = DrogueConfig()
    parafoil_config = ParafoilConfig(forward_speed=8.0, descent_rate=2.5, max_turn_rate=30.0)
    
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
        # We can also store metrics here if we want, or just rely on sim history
        return out.steering

    sim.run(steering_function=steering_func, max_time=3000.0)
    
    td_error = math.hypot(sim.state.x - waypoints[-1].x, sim.state.y - waypoints[-1].y)
    
    print(f"Landed at t={sim.time:.1f} s")
    print(f"Touchdown error: {td_error:.1f} m")
    
    xs = [s.x for s in sim.history]
    ys = [s.y for s in sim.history]
    
    plt.figure()
    plt.plot(xs, ys, label="Trajectory")
    plt.plot(waypoints[-1].x, waypoints[-1].y, 'rx', label="Target")
    plt.title(title)
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.axis("equal")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"plot_42b_{title.replace(' ', '_').lower()}.png")
    plt.close()

if __name__ == "__main__":
    start_pos = (0.0, -800.0, 1000.0)
    target = LocalPoint(0.0, 0.0)
    
    # 1. Zero gust
    run_sim(ComposedWindField(ConstantWindField(0.0, 0.0)), "Zero Gust", start_pos, [target])
    
    # 2. Single +Y gust
    run_sim(ComposedWindField(ConstantWindField(0.0, 0.0), CosineGust(0.0, 2.0, 50.0, 20.0)), "Single +Y Gust", start_pos, [target])
    
    # 3. Single -Y gust
    run_sim(ComposedWindField(ConstantWindField(0.0, 0.0), CosineGust(0.0, -2.0, 50.0, 20.0)), "Single -Y Gust", start_pos, [target])
    
    # 4. +Y followed by -Y
    run_sim(ComposedWindField(ConstantWindField(0.0, 0.0), CosineGust(0.0, 3.0, 50.0, 20.0), CosineGust(0.0, -3.0, 80.0, 20.0)), "PlusY then MinusY", start_pos, [target])
    
    # 5. +Y followed by +X
    run_sim(ComposedWindField(ConstantWindField(0.0, 0.0), CosineGust(0.0, 3.0, 50.0, 20.0), CosineGust(3.0, 0.0, 80.0, 20.0)), "PlusY then PlusX", start_pos, [target])
    
    # 6. Two overlapping gusts
    run_sim(ComposedWindField(ConstantWindField(0.0, 0.0), CosineGust(0.0, 3.0, 50.0, 20.0), CosineGust(3.0, 0.0, 60.0, 20.0)), "Overlapping Gusts", start_pos, [target])
    
    # 7. Multiple gusts on PiecewiseWindField
    pwf = PiecewiseWindField([(0.0, (2.0, 0.0)), (1000.0, (5.0, 0.0))])
    run_sim(ComposedWindField(pwf, CosineGust(0.0, 3.0, 50.0, 20.0), CosineGust(0.0, -2.0, 150.0, 20.0)), "Shear + Gusts", start_pos, [target])
    
    # 8. Short-duration gust (2s)
    run_sim(ComposedWindField(ConstantWindField(0.0, 0.0), CosineGust(0.0, 5.0, 50.0, 2.0)), "Short Gust 2s", start_pos, [target])
    
    # 9. High-amplitude gust (6m/s)
    run_sim(ComposedWindField(ConstantWindField(0.0, 0.0), CosineGust(0.0, 6.0, 50.0, 20.0)), "High Amplitude 6mps", start_pos, [target])
    
    # 10. Near-critical gust (base 6, gust 2 vs 8)
    run_sim(ComposedWindField(ConstantWindField(6.0, 0.0), CosineGust(2.0, 0.0, 50.0, 20.0)), "Near Critical Gust", start_pos, [target])
    
    # 11. Infeasible gust (base 6, gust 4 vs 8 -> 10m/s > 8m/s)
    run_sim(ComposedWindField(ConstantWindField(6.0, 0.0), CosineGust(4.0, 0.0, 50.0, 20.0)), "Infeasible Gust", start_pos, [target])
    
    # 12. Gust during TERMINAL_GUIDANCE (need a setup where terminal guidance happens)
    # Give it a higher initial altitude and closer so it orbits
    run_sim(ComposedWindField(ConstantWindField(2.0, 0.0), CosineGust(0.0, 5.0, 250.0, 20.0)), "Terminal Guidance Gust", (0.0, -100.0, 1000.0), [target])
    
    # 13. Gust during FINAL_APPROACH
    # Final approach is usually below 50m. Let's trigger gust at 380s.
    run_sim(ComposedWindField(ConstantWindField(0.0, 0.0), CosineGust(0.0, 3.0, 380.0, 10.0)), "Final Approach Gust", start_pos, [target])
