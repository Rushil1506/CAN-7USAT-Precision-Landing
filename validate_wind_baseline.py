def plot_trajectories(scenarios, results_file='wind_baseline_results.csv'):
    plt.figure(figsize=(12, 10))
    target = LocalPoint(1200.0, 0.0)
    plt.plot(0, 0, 'ko', label='Start', markersize=8)
    plt.plot(target.x, target.y, 'r*', label='Target', markersize=12)
    
    # Draw terminal capture radius and final approach radius
    circle = plt.Circle((target.x, target.y), 200.0, color='gray', fill=False, linestyle='--', label='Terminal Capture (200m)')
    plt.gca().add_patch(circle)
    circle2 = plt.Circle((target.x, target.y), 100.0, color='red', fill=False, linestyle=':', label='Final Approach (100m)')
    plt.gca().add_patch(circle2)
    
    for name, wx, wy in scenarios:
        sim, history = run_mission_with_wind(wx, wy)
        x_vals = [s.x for s in history]
        y_vals = [s.y for s in history]
        plt.plot(x_vals, y_vals, label=f"{name}")
        
    plt.title('CAN-7USAT Wind Baseline Validation Trajectories')
    plt.xlabel('East (m)')
    plt.ylabel('North (m)')
    plt.grid(True)
    plt.legend(loc='lower left', fontsize='small')
    plt.axis('equal')
    plt.savefig('wind_validation_trajectories.png')

import csv
import math
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt

from control.pid import PIDController
from flight.parafoil import DrogueConfig, ParafoilConfig, ParafoilState
from flight.state_machine import FlightState
from guidance.guidance_controller import GuidanceController
from guidance.loiter_guidance import LoiterDirection, LoiterPhase
from navigation.coordinates import LocalPoint
from navigation.distance import distance
from navigation.waypoints import generate_waypoints
from navigation.waypoint_manager import WaypointManager
from simulation.simulator import ParafoilSimulator, SimulationSample

def run_mission_with_wind(wind_x: float, wind_y: float):
    start = LocalPoint(0.0, 0.0)
    target = LocalPoint(1200.0, 0.0)
    waypoints = generate_waypoints(start=start, target=target, num_waypoints=10)
    waypoint_manager = WaypointManager(waypoints=waypoints, acceptance_radius=25.0)

    pid = PIDController(kp=0.022, ki=0.00015, kd=0.004, output_min=-1.0, output_max=1.0)
    guidance = GuidanceController(waypoint_manager=waypoint_manager, pid=pid)

    state = ParafoilState(x=0.0, y=0.0, altitude=1000.0, vx=0.0, vy=0.0, vz=0.0, heading=0.0, heading_rate=0.0)

    sim = ParafoilSimulator(
        state=state,
        drogue_config=DrogueConfig(),
        parafoil_config=ParafoilConfig(forward_speed=8.0, descent_rate=2.5),
        target_position=target,
        wind_x=wind_x,
        wind_y=wind_y,
        loiter_radius=30.0,
        breakout_altitude_margin=2.0
    )

    def steering_cb(time, st):
        # Inject known wind values into the controller for this milestone
        guidance.set_wind(wind_x, wind_y)
        
        if sim.state_machine.get_state() == FlightState.FINAL_APPROACH:
            if not waypoint_manager.is_complete:
                waypoint_manager.skip_to_final_target()
                guidance.pid.reset()
        output = guidance.update(
            position=LocalPoint(st.x, st.y),
            heading=st.heading,
            altitude=st.altitude,
            dt=sim.dt
        )
        return output.steering

    history = sim.run(steering_function=steering_cb, max_time=600.0)
    return sim, history

def main():
    scenarios = [
        ("Baseline", 0.0, 0.0),
        ("North (Tailwind)", 0.0, 3.0),
        ("South (Headwind)", 0.0, -3.0),
        ("East (Crosswind)", 3.0, 0.0),
        ("West (Crosswind)", -3.0, 0.0),
        ("Diagonal (NE)", 2.12, 2.12),
        ("Strong (90% Airspeed N)", 0.0, 7.2),
        ("Strong (90% Airspeed S)", 0.0, -7.2),
        ("Strong (90% Airspeed E)", 7.2, 0.0),
        ("Strong (90% Airspeed W)", -7.2, 0.0),
    ]

    results = []
    
    for name, wx, wy in scenarios:
        sim, history = run_mission_with_wind(wx, wy)
        
        target = LocalPoint(1200.0, 0.0)
        final = history[-1]
        touchdown_err = distance(LocalPoint(final.x, final.y), target)
        
        terminal_samples = [s for s in history if s.state == FlightState.TERMINAL_GUIDANCE]
        fa_samples = [s for s in history if s.state == FlightState.FINAL_APPROACH]
        
        orbit_count = sim.loiter_orbits_completed
        mean_radial_err = sim.mean_radial_error_during_loiter
        max_radial_err = sim.max_radial_error_during_loiter
        breakout_alt = sim.breakout_altitude if sim.breakout_altitude else 0.0
        
        failsafe = sim.state_machine.get_state() == FlightState.FAILSAFE
        
        # We must output 19 specific metrics
        metrics = {
            "Scenario": name,
            "Wind X": wx,
            "Wind Y": wy,
            "Failsafe": failsafe,
            "Landing Error": touchdown_err,
            "Drogue Drift X": sim.state.x if len(history)>0 else 0, # Note: this is end of flight x. We need drogue drift specifically.
            "Drogue Drift Y": sim.state.y if len(history)>0 else 0,
            "Terminal Entry Alt": sim.terminal_entry_altitude,
            "Terminal Entry Time": sim.terminal_entry_time,
            "Terminal Entry Dist": sim.terminal_entry_distance,
            "Terminal Entry Glide": sim.terminal_entry_glide_range,
            "Terminal Duration": len(terminal_samples)*0.1,
            "Breakout Alt": sim.breakout_altitude,
            "Breakout Dist": sim.breakout_distance,
            "Breakout Heading": sim.breakout_heading,
            "Orbit Count": orbit_count,
            "Mean Radial Err": mean_radial_err,
            "Max Radial Err": max_radial_err,
            "FA Duration": len(fa_samples)*0.1
        }
        
        # fix drogue drift
        drogue_end = [s for s in history if s.state == FlightState.DROGUE_DESCENT][-1]
        metrics["Drogue Drift X"] = drogue_end.x
        metrics["Drogue Drift Y"] = drogue_end.y
        
        results.append(metrics)
        print(f"[{name}] Wind:({wx},{wy}) Error:{touchdown_err:.2f}m Orbits:{orbit_count:.2f} Failsafe:{failsafe}")

    with open("wind_baseline_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    plot_trajectories(scenarios)

if __name__ == "__main__":
    main()

