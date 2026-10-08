import math
import numpy as np
from typing import List, Tuple
from navigation.coordinates import LocalPoint
from navigation.distance import distance
from environment.wind import ConstantWindField, PiecewiseWindField
from flight.parafoil import ParafoilState, ParafoilConfig, DrogueConfig
from simulation.simulator import ParafoilSimulator
from flight.state_machine import FlightState
from control.pid import PIDController
from navigation.waypoints import generate_waypoints
from navigation.waypoint_manager import WaypointManager
from guidance.guidance_controller import GuidanceController

def run_mission(wind_field, target=LocalPoint(500.0, 500.0), dt=0.1, initial_alt=1000.0):
    state = ParafoilState(x=0.0, y=0.0, altitude=initial_alt, heading=0.0, vx=0.0, vy=0.0, vz=0.0, heading_rate=0.0)
    drogue_config = DrogueConfig()
    parafoil_config = ParafoilConfig(forward_speed=8.0, descent_rate=2.5, max_turn_rate=30.0)
    
    sim = ParafoilSimulator(
        state=state,
        drogue_config=drogue_config,
        parafoil_config=parafoil_config,
        target_position=target,
        dt=dt,
        wind_field=wind_field
    )
    
    pid = PIDController(kp=0.5, ki=0.0, kd=0.1)
    
    guidance = GuidanceController(
        waypoint_manager=WaypointManager(waypoints=generate_waypoints(LocalPoint(0,0), target, 1)),
        pid=pid
    )
    
    def steering_func(t, s):
        wx, wy = wind_field.wind_at_altitude(s.altitude)
        guidance.set_wind(wx, wy)
        out = guidance.update(
            position=LocalPoint(s.x, s.y),
            heading=s.heading,
            altitude=s.altitude,
            dt=sim.dt
        )
        return out.steering
        
    sim.run(steering_function=steering_func, max_time=3000.0)
    return sim

def analyze_telemetry(sim):
    history = sim.history
    term_history = [s for s in history if s.state == FlightState.TERMINAL_GUIDANCE]
    
    # Check plan freeze
    plan = getattr(sim, 'terminal_orbit_plan', None)
    plan_frozen = plan is not None
    
    # Check local control
    winds = set()
    crabs = set()
    for s in term_history:
        winds.add((round(s.wind_x, 3), round(s.wind_y, 3)))
        crabs.add(round(s.crab_angle, 1))
        
    local_control_active = len(winds) > 1 and len(crabs) > 1
    
    # Steady state radial error
    orbit_tracking = [s for s in term_history if getattr(s, 'loiter_phase', None) == 1] # ORBIT_TRACKING
    radial_errors = [abs(s.loiter_radial_error) for s in orbit_tracking if s.loiter_radial_error is not None]
    
    mean_err = np.mean(radial_errors) if radial_errors else 0.0
    max_err = np.max(radial_errors) if radial_errors else 0.0
    
    # Touchdown error
    final_s = history[-1]
    td_error = distance(LocalPoint(final_s.x, final_s.y), sim.target_position)
    
    return {
        'term_entry_alt': sim.terminal_entry_altitude,
        'plan_feasible': plan.feasible if plan else None,
        'plan_frozen': plan_frozen,
        'local_control_active': local_control_active,
        'mean_radial_err': mean_err,
        'max_radial_err': max_err,
        'td_error': td_error,
        'failsafe': final_s.state == FlightState.FAILSAFE,
        'final_state': final_s.state.name
    }

def print_metrics(name, metrics):
    print(f"--- {name} ---")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.2f}")
        else:
            print(f"  {k}: {v}")

def run_validations():
    # 1. Feasible Shear Cases
    print("=========================================")
    print("1. FEASIBLE SHEAR END-TO-END VALIDATION")
    
    # A. Weak magnitude shear
    w1 = PiecewiseWindField([
        (0.0, (1.0, 0.0)),
        (1000.0, (3.0, 0.0))
    ])
    sim1 = run_mission(w1)
    print_metrics("Weak Magnitude Shear", analyze_telemetry(sim1))
    
    # B. Strong magnitude shear
    w2 = PiecewiseWindField([
        (0.0, (4.0, 0.0)),
        (1000.0, (7.0, 0.0))
    ])
    sim2 = run_mission(w2)
    print_metrics("Strong Magnitude Shear", analyze_telemetry(sim2))
    
    # C. Directional shear
    w3 = PiecewiseWindField([
        (0.0, (3.0, 0.0)),
        (500.0, (3.0, 3.0)),
        (1000.0, (0.0, 3.0))
    ])
    sim3 = run_mission(w3)
    print_metrics("Directional Shear", analyze_telemetry(sim3))
    
    # D. Crosswind reversal
    w4 = PiecewiseWindField([
        (0.0, (2.0, 0.0)),
        (500.0, (0.0, 0.0)),
        (1000.0, (-2.0, 0.0))
    ])
    sim4 = run_mission(w4)
    print_metrics("Crosswind Reversal", analyze_telemetry(sim4))
    
    # E. Diagonal shear
    w5 = PiecewiseWindField([
        (0.0, (2.0, 2.0)),
        (1000.0, (5.0, 5.0))
    ])
    sim5 = run_mission(w5)
    print_metrics("Diagonal Shear", analyze_telemetry(sim5))
    
    # 2 & 3. Plan-Freeze and Local-Control
    print("=========================================")
    print("2 & 3. PLAN-FREEZE & LOCAL-CONTROL VALIDATION")
    print(f"Plan frozen: {analyze_telemetry(sim3)['plan_frozen']}")
    print(f"Local control active: {analyze_telemetry(sim3)['local_control_active']}")
    
    # 4. Infeasible-Plan Fallback
    print("=========================================")
    print("4. INFEASIBLE-PLAN FALLBACK")
    # Wind > forward_speed (8.0)
    w_inf = PiecewiseWindField([
        (0.0, (10.0, 0.0)),
        (1000.0, (10.0, 0.0))
    ])
    sim_inf = run_mission(w_inf)
    
    term_history = [s for s in sim_inf.history if s.state == FlightState.TERMINAL_GUIDANCE]
    print(f"Entered TERMINAL_GUIDANCE: {len(term_history) > 0}")
    print(f"Final state: {sim_inf.history[-1].state.name}")
    
    # Trace what happens if terminal entry is rejected
    # Fly over target while waiting?
    prec_history = [s for s in sim_inf.history if s.state == FlightState.PRECISION_GUIDANCE]
    distances = [distance(LocalPoint(s.x, s.y), sim_inf.target_position) for s in prec_history]
    min_dist = min(distances) if distances else float('inf')
    print(f"Minimum distance to target during PRECISION_GUIDANCE: {min_dist:.2f} m")
    
    w_inf_shear = PiecewiseWindField([
        (0.0, (3.0, 0.0)), # Feasible at bottom
        (200.0, (3.0, 0.0)),
        (400.0, (10.0, 0.0)), # Infeasible at terminal entry (around 300-400m)
        (1000.0, (10.0, 0.0))
    ])
    sim_inf2 = run_mission(w_inf_shear)
    print(f"Infeasible Shear - Final State: {sim_inf2.history[-1].state.name}")
    prec_history2 = [s for s in sim_inf2.history if s.state == FlightState.PRECISION_GUIDANCE]
    dists2 = [distance(LocalPoint(s.x, s.y), sim_inf2.target_position) for s in prec_history2]
    min_dist2 = min(dists2) if dists2 else float('inf')
    print(f"Infeasible Shear - Min distance to target during PRECISION_GUIDANCE: {min_dist2:.2f} m")
    
    # 5. Constant-Wind Regression
    print("=========================================")
    print("5. CONSTANT-WIND REGRESSION")
    cw_scenarios = {
        'zero': (0.0, 0.0),
        '+3X': (3.0, 0.0),
        '-3X': (-3.0, 0.0),
        '+3Y': (0.0, 3.0),
        '-3Y': (0.0, -3.0),
        'diagonal': (3.0, 3.0)
    }
    for name, (wx, wy) in cw_scenarios.items():
        sim_cw = run_mission(ConstantWindField(wx, wy))
        td_err = analyze_telemetry(sim_cw)['td_error']
        print(f"CW {name}: TD Error = {td_err:.2f} m, Failsafe: {sim_cw.history[-1].state == FlightState.FAILSAFE}")
        
    print("=========================================")
    print("FINISHED VALIDATION SCRIPT")

if __name__ == '__main__':
    run_validations()
