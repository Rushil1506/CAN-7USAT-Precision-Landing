import math
import matplotlib.pyplot as plt
from typing import Tuple

from flight.state_machine import FlightState
from flight.parafoil import ParafoilState, ParafoilConfig, DrogueConfig
from environment.wind import ConstantWindField, CosineGust, ComposedWindField
from simulation.simulator import ParafoilSimulator
from navigation.coordinates import LocalPoint
from navigation.waypoint_manager import WaypointManager
from guidance.guidance_controller import GuidanceController
from control.pid import PIDController

def run_sim(base_wind: Tuple[float, float], gust_x: float, gust_y: float, gust_start: float, gust_duration: float, title: str):
    print(f"\n--- Running: {title} ---")
    
    target = LocalPoint(0.0, 0.0)
    
    state = ParafoilState(x=0.0, y=-800.0, altitude=1000.0, heading=0.0, vx=0.0, vy=0.0, vz=0.0, heading_rate=0.0)
    drogue_config = DrogueConfig()
    parafoil_config = ParafoilConfig(forward_speed=8.0, descent_rate=2.5, max_turn_rate=30.0)
    
    base = ConstantWindField(base_wind[0], base_wind[1])
    gust = CosineGust(gust_x, gust_y, gust_start, gust_duration)
    wind_field = ComposedWindField(base, gust)
    
    sim = ParafoilSimulator(
        state=state,
        drogue_config=drogue_config,
        parafoil_config=parafoil_config,
        target_position=target,
        dt=0.1,
        wind_field=wind_field
    )
    
    pid = PIDController(kp=2.5, ki=0.1, kd=0.5)
    
    guidance = GuidanceController(
        waypoint_manager=WaypointManager(waypoints=[target]),
        pid=pid
    )
    
    # We'll use sim.time to query wind
    def steering_func(t, s):
        # Local wind gets passed the time now
        wx, wy = wind_field.wind_at(s.altitude, getattr(sim, "time", 0.0))
        guidance.set_wind(wx, wy)
        out = guidance.update(
            position=LocalPoint(s.x, s.y),
            heading=s.heading,
            altitude=s.altitude,
            dt=sim.dt
        )
        return out.steering

    # Let it run
    sim.run(steering_function=steering_func, max_time=3000.0)
    
    print(f"Landed at t={sim.time:.1f} s")
    print(f"Touchdown error: math.hypot({sim.state.x:.1f}, {sim.state.y:.1f}) = {math.hypot(sim.state.x, sim.state.y):.1f} m")
    
    xs = [s.x for s in sim.history]
    ys = [s.y for s in sim.history]
    
    plt.figure()
    plt.plot(xs, ys, label="Trajectory")
    plt.plot(0, 0, 'rx', label="Target")
    plt.title(title)
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.axis("equal")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"plot_{title.replace(' ', '_').lower()}.png")
    plt.close()

if __name__ == "__main__":
    # Gust A (+Y, 2m/s) during precision transit
    run_sim((0.0, 0.0), 0.0, 2.0, 50.0, 20.0, "Gust A +Y 2mps")
    
    # Gust B (-Y, 4m/s)
    run_sim((0.0, 0.0), 0.0, -4.0, 50.0, 20.0, "Gust B -Y 4mps")
    
    # Gust C (+X, 6m/s)
    run_sim((0.0, 0.0), 6.0, 0.0, 50.0, 20.0, "Gust C +X 6mps")
    
    # Near-feasibility gust (base 6m/s crosswind + 2m/s gust vs 8m/s airspeed)
    run_sim((6.0, 0.0), 2.0, 0.0, 100.0, 20.0, "Near Feasibility Gust")
    
    # Terminal-guidance observation (run gust during orbit and verify stability without modifying the planner)
    # Give it a higher initial position so it does terminal orbit
    run_sim((2.0, 0.0), 4.0, 0.0, 300.0, 20.0, "Terminal Gust Observation")
