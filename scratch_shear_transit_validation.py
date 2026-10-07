import math
from environment.wind import PiecewiseWindField
from flight.parafoil import DrogueConfig, ParafoilConfig, ParafoilState
from flight.state_machine import FlightState
from guidance.guidance_controller import GuidanceController
from navigation.coordinates import LocalPoint
from navigation.waypoint_manager import WaypointManager
from control.pid import PIDController
from simulation.simulator import ParafoilSimulator
from navigation.distance import distance

def run_shear_transit(name, wind_field, start_pos, target_pos):
    wm = WaypointManager(waypoints=[LocalPoint(*start_pos[:2]), LocalPoint(*target_pos[:2])], acceptance_radius=25.0)
    pid = PIDController(kp=0.022, ki=0.00015, kd=0.004, output_min=-1.0, output_max=1.0)
    guidance = GuidanceController(waypoint_manager=wm, pid=pid)
    
    state = ParafoilState(x=start_pos[0], y=start_pos[1], altitude=1000.0, vx=0.0, vy=0.0, vz=0.0, heading=0.0, heading_rate=0.0)
    
    sim = ParafoilSimulator(
        state=state, 
        drogue_config=DrogueConfig(),
        parafoil_config=ParafoilConfig(forward_speed=8.0, descent_rate=2.5),
        wind_field=wind_field
    )
    
    def steering_cb(t, st):
        # The correct architecture for Phase 4.1c:
        # Guidance controller MUST receive local wind at every update!
        wx, wy = sim.wind_field.wind_at_altitude(st.altitude)
        guidance.set_wind(wx, wy)
        
        c_state = sim.state_machine.get_state()
        if c_state == FlightState.FINAL_APPROACH:
            if not wm.is_complete:
                wm.skip_to_final_target()
                guidance.pid.reset()
                
        out = guidance.update(LocalPoint(st.x, st.y), st.heading, st.altitude, sim.dt)
        return out.steering

    # Override state to bypass drogue and deployment for pure transit validation
    # Actually, we should just let it run or start in precision guidance.
    # The prompt says: "Separate any initial drogue/deployment displacement from the actual parafoil precision-guidance tracking."
    
    history = sim.run(steering_function=steering_cb, max_time=600.0)
    
    # Analyze PRECISION_GUIDANCE portion only
    precision_samples = [s for s in history if s.state == FlightState.PRECISION_GUIDANCE]
    
    print(f"--- {name} ---")
    if not precision_samples:
        print("No precision guidance samples!")
        return
        
    start_x, start_y = precision_samples[0].x, precision_samples[0].y
    tgt = LocalPoint(*target_pos[:2])
    
    # Course commanded
    chi = math.degrees(math.atan2(tgt.x - start_x, tgt.y - start_y)) % 360.0
    print(f"Commanded Ground Course: {chi:.2f} deg")
    
    max_cte = 0.0
    sq_cte = 0.0
    max_ce = 0.0
    sq_ce = 0.0
    max_crab = 0.0
    max_steer = 0.0
    
    import navigation.distance as nav_dist
    
    for s in precision_samples:
        # Cross track error
        # Line from start point to target
        dx = tgt.x - start_x
        dy = tgt.y - start_y
        L = math.hypot(dx, dy)
        if L > 1e-6:
            # (y - y1)(x2 - x1) - (x - x1)(y2 - y1) / L
            cte = abs((s.y - start_y)*dx - (s.x - start_x)*dy) / L
        else:
            cte = 0.0
            
        max_cte = max(max_cte, cte)
        sq_cte += cte*cte
        
        ce = abs((s.course_over_ground - chi + 180) % 360 - 180)
        max_ce = max(max_ce, ce)
        sq_ce += ce*ce
        
        max_crab = max(max_crab, abs(s.crab_angle))
        max_steer = max(max_steer, abs(s.steering))
        
    n = len(precision_samples)
    rms_cte = math.sqrt(sq_cte / n)
    rms_ce = math.sqrt(sq_ce / n)
    final_cte = max_cte # Actually need the last one
    
    last_s = precision_samples[-1]
    dx = tgt.x - start_x; dy = tgt.y - start_y; L = math.hypot(dx, dy)
    final_cte = abs((last_s.y - start_y)*dx - (last_s.x - start_x)*dy) / L if L > 1e-6 else 0.0
    
    print(f"Max CTE: {max_cte:.2f} m, RMS CTE: {rms_cte:.2f} m, Final CTE: {final_cte:.2f} m")
    print(f"Max CE: {max_ce:.2f} deg, RMS CE: {rms_ce:.2f} deg")
    print(f"Max Crab Angle: {max_crab:.2f} deg")
    print(f"Max Steering: {max_steer:.2f}")
    print()

if __name__ == '__main__':
    # Test A: Zero-shear regression
    run_shear_transit("Test A - Zero Shear Regression", PiecewiseWindField([(0.0, (3.0, 0.0)), (1000.0, (3.0, 0.0))]), (0, 0, 1000), (1200, 0, 0))
    
    # Test B: Pure Magnitude Shear (along track)
    run_shear_transit("Test B - Pure Magnitude Shear", PiecewiseWindField([(0.0, (1.0, 0.0)), (1000.0, (4.0, 0.0))]), (0, 0, 1000), (1200, 0, 0))
    
    # Test C: Pure Crosswind Shear
    run_shear_transit("Test C - Pure Crosswind Shear", PiecewiseWindField([(0.0, (0.0, 1.0)), (1000.0, (0.0, 4.0))]), (0, 0, 1000), (1200, 0, 0))
    
    # Test D: Directional Wind Shear
    points_d = [
        (0.0,   (-1.0,  0.0)),
        (200.0, ( 0.0,  2.0)),
        (400.0, ( 2.0,  2.0)),
        (600.0, ( 4.0,  0.0)),
        (1000.0,( 4.0,  0.0)) # Flat top
    ]
    run_shear_transit("Test D - Directional Wind Shear", PiecewiseWindField(points_d), (0, 0, 1000), (1200, 0, 0))
    
    # Test E: Near-feasibility Test
    points_e = [
        (0.0,   (0.0, 2.0)),
        (200.0, (0.0, 4.0)),
        (400.0, (0.0, 6.0)),
        (600.0, (0.0, 7.5)),
        (1000.0,(0.0, 7.5))
    ]
    run_shear_transit("Test E - Near-feasibility", PiecewiseWindField(points_e), (0, 0, 1000), (1200, 0, 0))
    
    # Test F: Infeasible Test
    points_f = [
        (0.0,   (0.0, 2.0)),
        (200.0, (0.0, 9.0)), # V_air is 8.0, so this is > V_air!
        (1000.0,(0.0, 9.0))
    ]
    run_shear_transit("Test F - Infeasible", PiecewiseWindField(points_f), (0, 0, 1000), (1200, 0, 0))
