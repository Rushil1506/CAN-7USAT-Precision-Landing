import sys
import math
import numpy as np

from flight.parafoil import DrogueConfig, ParafoilConfig, ParafoilState
from flight.state_machine import FlightState
from guidance.guidance_controller import GuidanceController
from navigation.coordinates import LocalPoint
from navigation.waypoints import generate_waypoints
from navigation.waypoint_manager import WaypointManager
from control.pid import PIDController
from simulation.simulator import ParafoilSimulator, SimulationSample
from navigation.distance import distance

def main():
    target = LocalPoint(1200.0, 0.0)
    wind_x = 3.0
    wind_y = 0.0
    
    start = LocalPoint(0.0, 0.0)
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

    outputs = []

    def steering_cb(time, st):
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
        outputs.append((time, output))
        return output.steering

    history = sim.run(steering_function=steering_cb, max_time=600.0)
    
    # Process history
    phases = {}
    
    for i, h in enumerate(history):
        if "B" not in phases and h.state == FlightState.TERMINAL_GUIDANCE:
            phases["B"] = h
        if "C" not in phases and h.state == FlightState.TERMINAL_GUIDANCE and h.loiter_phase is not None:
            phases["C"] = h
        if "F" not in phases and h.state in [FlightState.FINAL_APPROACH, FlightState.FAILSAFE]:
            phases["F"] = h
    
    # A. Original terminal capture opportunity
    for h in history:
        target_dist = distance(LocalPoint(h.x, h.y), target)
        from guidance.glide_range import calculate_glide_range
        from navigation.bearing import bearing
        rg = calculate_glide_range(h.altitude, 2.5, 8.0, wind_x, wind_y, bearing(LocalPoint(h.x, h.y), target))
        if rg >= target_dist + 100.0 and target_dist <= 200.0:
            phases["A"] = h
            break

    # Find breakout-turn endpoint (G)
    # The turn takes `geom.turn_duration` seconds from the start of FINAL_APPROACH.
    # Actually wait, in full mission, breakout starts at FINAL_APPROACH in the simulator because we transition immediately when altitude <= breakout_altitude!
    # Wait, the loiter guidance just provides commands. In FINAL_APPROACH, it switches to `steering_cb` which uses `GuidanceController`!
    
    def print_phase(name, h):
        if h:
            print(f"{name}: Time={h.time:.1f}s Alt={h.altitude:.2f}m Pos=({h.x:.1f}, {h.y:.1f}) Dist={distance(LocalPoint(h.x, h.y), target):.2f}m Hdg={h.heading:.1f}°")
            # find the guidance output for this time
            for t, out in outputs:
                if abs(t - h.time) < 0.05:
                    print(f"   -> Desired Hdg={out.desired_heading:.1f}° Crab={out.crab_angle:.1f}° GS={out.ground_speed_along_track:.1f}")
                    break
    
    print_phase("A. Original terminal capture opp", phases.get("A"))
    print_phase("B. Actual temporal-gate terminal entry", phases.get("B"))
    print_phase("F. Breakout trigger / FINAL APPROACH", phases.get("F"))
    td = history[-1]
    print_phase("H. Touchdown", td)
    
    fa_samples = [s for s in history if s.state == FlightState.FINAL_APPROACH]
    if len(fa_samples) > 0:
        start_fa = fa_samples[0]
        p0 = np.array([start_fa.x, start_fa.y])
        p1 = np.array([target.x, target.y])
        v = p1 - p0
        v_norm = np.linalg.norm(v)
        u = v / v_norm if v_norm > 0 else np.array([1, 0])
        n = np.array([-u[1], u[0]])
        
        c_errs = []
        for s in fa_samples:
            p = np.array([s.x, s.y])
            dp = p - p0
            cte = np.dot(dp, n)
            c_errs.append(abs(cte))
            
        print(f"Final FA max CTE: {max(c_errs):.2f} m")
        print(f"Final FA RMS CTE: {np.sqrt(np.mean(np.array(c_errs)**2)):.2f} m")
        print(f"Final CTE at TD: {c_errs[-1]:.2f} m")
        
        # Calculate along-track error
        p_td = np.array([td.x, td.y])
        dp_td = p_td - p0
        ate_td = np.dot(dp_td, u)
        # distance remaining along track
        dist_remaining = v_norm - ate_td
        print(f"Final Along-Track Error (distance short of target): {dist_remaining:.2f} m")

    geom = sim.last_breakout_geometry
    if geom:
        dx_wind = wind_x * geom.turn_duration
        dy_wind = wind_y * geom.turn_duration
        endpoint_ground_x = geom.turn_endpoint.x
        endpoint_ground_y = geom.turn_endpoint.y
        endpoint_air_x = endpoint_ground_x - dx_wind
        endpoint_air_y = endpoint_ground_y - dy_wind
        
        print("\n=== Breakout Geometry ===")
        print(f"Turn duration: {geom.turn_duration:.2f} s")
        print(f"Aero endpoint: ({endpoint_air_x:.2f}, {endpoint_air_y:.2f})")
        print(f"Wind drift during turn: dx={dx_wind:.2f}, dy={dy_wind:.2f}")
        print(f"Corrected ground endpoint: ({endpoint_ground_x:.2f}, {endpoint_ground_y:.2f})")
        print(f"Straight distance: {geom.straight_distance:.2f} m")
        print(f"Breakout Altitude: {geom.breakout_altitude:.2f} m")
        
        # heading at breakout trigger vs desired aero heading
        trigger_state = phases.get("F")
        if trigger_state:
            print(f"Heading at breakout trigger: {trigger_state.heading:.1f}°")
            # get desired heading
            for t, out in outputs:
                if abs(t - trigger_state.time) < 0.05:
                    print(f"Desired aero heading: {out.desired_heading:.1f}°")
                    break


    zero_sim = ParafoilSimulator(
        state=ParafoilState(x=0.0, y=0.0, altitude=1000.0, vx=0.0, vy=0.0, vz=0.0, heading=0.0, heading_rate=0.0),
        drogue_config=DrogueConfig(),
        parafoil_config=ParafoilConfig(forward_speed=8.0, descent_rate=2.5),
        target_position=target,
        wind_x=0.0,
        wind_y=0.0,
        loiter_radius=30.0,
        breakout_altitude_margin=2.0
    )
    def zero_steering_cb(time, st):
        guidance.set_wind(0.0, 0.0)
        if zero_sim.state_machine.get_state() == FlightState.FINAL_APPROACH:
            if not waypoint_manager.is_complete:
                waypoint_manager.skip_to_final_target()
                guidance.pid.reset()
        out = guidance.update(LocalPoint(st.x, st.y), st.heading, st.altitude, zero_sim.dt)
        return out.steering
    
    waypoint_manager.reset()
    guidance.pid.reset()
    zero_history = zero_sim.run(steering_function=zero_steering_cb, max_time=600.0)
    z_td = zero_history[-1]
    print("\n=== Zero-Wind Quick Check ===")
    print(f"Touchdown Dist: {distance(LocalPoint(z_td.x, z_td.y), target):.2f} m")
    
if __name__ == '__main__':
    main()
