import math
from flight.parafoil import DrogueConfig, ParafoilConfig, ParafoilState
from flight.state_machine import FlightState
from guidance.guidance_controller import GuidanceController
from navigation.coordinates import LocalPoint
from navigation.waypoints import generate_waypoints
from navigation.waypoint_manager import WaypointManager
from control.pid import PIDController
from simulation.simulator import ParafoilSimulator
from navigation.distance import distance
from navigation.bearing import bearing

cases = [
    ("Zero wind", 0.0, 0.0),
    ("+3 m/s East tailwind", 3.0, 0.0),
    ("+3 m/s North crosswind", 0.0, 3.0),
    ("-3 m/s North crosswind", 0.0, -3.0),
    ("diagonal 3 m/s wind", 2.12, 2.12)
]

for name, wx, wy in cases:
    target = LocalPoint(1200.0, 0.0)
    start = LocalPoint(0.0, 0.0)
    waypoints = generate_waypoints(start=start, target=target, num_waypoints=10)
    wm = WaypointManager(waypoints=waypoints, acceptance_radius=25.0)
    pid = PIDController(kp=0.022, ki=0.00015, kd=0.004, output_min=-1.0, output_max=1.0)
    guidance = GuidanceController(waypoint_manager=wm, pid=pid)
    state = ParafoilState(x=0.0, y=0.0, altitude=1000.0, vx=0.0, vy=0.0, vz=0.0, heading=0.0, heading_rate=0.0)
    
    sim = ParafoilSimulator(
        state=state, drogue_config=DrogueConfig(),
        parafoil_config=ParafoilConfig(forward_speed=8.0, descent_rate=2.5, wind_x=wx, wind_y=wy),
        target_position=target, wind_x=wx, wind_y=wy, loiter_radius=30.0, breakout_altitude_margin=2.0
    )
    
    def steering_cb(t, st):
        guidance.set_wind(wx, wy)
        c_state = sim.state_machine.get_state()
            
        if c_state == FlightState.FINAL_APPROACH:
            if not wm.is_complete:
                wm.skip_to_final_target()
                guidance.pid.reset()
        out = guidance.update(LocalPoint(st.x, st.y), st.heading, st.altitude, sim.dt)
        return out.steering

    history = sim.run(steering_function=steering_cb, max_time=600.0)
    td = history[-1]
    
    terminal_duration = 0.0
    if sim.terminal_entry_time is not None:
        if sim.breakout_time is not None:
            terminal_duration = sim.breakout_time - sim.terminal_entry_time
        else:
            terminal_duration = td.time - sim.terminal_entry_time
            
    geom = sim.last_breakout_geometry
    
    print(f"--- {name} ---")
    if geom:
        dx = wx * geom.turn_duration
        dy = wy * geom.turn_duration
        ex = geom.turn_endpoint.x
        ey = geom.turn_endpoint.y
        b_tgt = bearing(geom.turn_endpoint, target)
        
        final_heading = (history[0].heading + geom.turn_direction * geom.turn_angle) % 360 # approx
        
        # we can just recalculate the ground course from the end of the turn using airspeed and wind
        # wait, we can just grab geom.ground_course if we added it, but we didn't. 
        # let's recalculate it exactly as the geometry does:
        # turn_angle, turn_dir, heading at breakout = ? 
        # The history has the actual heading at breakout. But geom was last updated at breakout!
        # Actually, let's just find the last terminal guidance state
        trigger_states = [h for h in history if h.state == FlightState.TERMINAL_GUIDANCE]
        if not trigger_states:
             trigger_states = [h for h in history if h.state == FlightState.FINAL_APPROACH]
        
        if trigger_states:
            last_tg = trigger_states[-1]
            end_hdg = (last_tg.heading + geom.turn_direction * geom.turn_angle) % 360.0
            ax = 8.0 * math.sin(math.radians(end_hdg))
            ay = 8.0 * math.cos(math.radians(end_hdg))
            gx = ax + wx
            gy = ay + wy
            gc = math.degrees(math.atan2(gx, gy)) % 360.0
            
            mismatch = (gc - b_tgt + 180) % 360 - 180
            
            print(f"turn duration: {geom.turn_duration:.2f} s")
            print(f"turn endpoint: ({ex:.2f}, {ey:.2f})")
            print(f"wind drift: ({dx:.2f}, {dy:.2f})")
            print(f"ground course at endpoint: {gc:.2f} deg")
            print(f"bearing endpoint -> target: {b_tgt:.2f} deg")
            print(f"course/bearing mismatch: {mismatch:.2f} deg")
            print(f"breakout altitude: {geom.breakout_altitude:.2f} m")
    
    print(f"touchdown error: {distance(LocalPoint(td.x, td.y), target):.2f} m")
    print(f"terminal duration: {terminal_duration:.2f} s")
    print(f"failsafe status: {td.state == FlightState.FAILSAFE}\n")
