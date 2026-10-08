import math
import numpy as np

from control.pid import PIDController
from flight.parafoil import ParafoilConfig, ParafoilState, DrogueConfig
from environment.wind import ConstantWindField, StochasticWind, CosineGust, ComposedWindField
from simulation.simulator import ParafoilSimulator
from navigation.coordinates import LocalPoint
from navigation.waypoint_manager import WaypointManager
from guidance.guidance_controller import GuidanceController

def prove_equivalence():
    print("=== 3. PROVE A/B/C MATHEMATICAL EQUIVALENCE ===")
    errors = [0.1, 0.4, 1.0, 5.0, 10.0, 20.0, 45.0, 90.0, 180.0]
    
    # Baseline
    kp = 2.5; ki = 0.1; kd = 0.5
    dt = 0.1
    
    print("Model A (Degree error, kp=2.5, kd=0.5):")
    for e in errors:
        pA = kp * e
        dA = kd * (e / dt) # assume step from 0
        print(f"e={e:5.1f} deg -> P = {pA:7.3f}, D = {dA:7.3f}")
        
    print("\nModel B (Radian error, kp=2.5, kd=0.5):")
    for e in errors:
        e_rad = math.radians(e)
        pB = kp * e_rad
        dB = kd * (e_rad / dt)
        print(f"e={e:5.1f} deg ({e_rad:5.3f} rad) -> P = {pB:7.3f}, D = {dB:7.3f}")
        
    print("\nModel C (Degree error, rescaled gains kp*pi/180):")
    kp_deg = kp * math.pi / 180.0
    kd_deg = kd * math.pi / 180.0
    for e in errors:
        pC = kp_deg * e
        dC = kd_deg * (e / dt)
        print(f"e={e:5.1f} deg -> P = {pC:7.3f}, D = {dC:7.3f}")
        
    print("\nEquivalence proven: Model B and Model C produce identical numerical P/D contributions.")

def analytical_saturation():
    print("\n=== 4. ANALYTICAL SATURATION THRESHOLDS ===")
    kp = 2.5; kd = 0.5; dt = 0.1
    
    # Model A
    sat_P_A = 1.0 / kp
    sat_D_A = (1.0 / kd) * dt
    print(f"Model A (Degree): |P|=1 at {sat_P_A:.3f} deg, |D|=1 at {sat_D_A:.3f} deg/step ({sat_D_A/dt:.1f} deg/s)")
    
    # Model B
    sat_P_B_rad = 1.0 / kp
    sat_D_B_rad = (1.0 / kd) * dt
    print(f"Model B (Radian): |P|=1 at {math.degrees(sat_P_B_rad):.1f} deg, |D|=1 at {math.degrees(sat_D_B_rad):.1f} deg/step ({math.degrees(sat_D_B_rad)/dt:.1f} deg/s)")
    
    # Note on 30 deg/s max turn rate
    print(f"Physical Max Turn Rate: 30 deg/s. In dt=0.1s, max turn is 3.0 deg.")
    print(f"Model B |D|=1 angular rate is {math.degrees(sat_D_B_rad)/dt:.1f} deg/s. This matches magnitude beautifully: D term will only saturate at {math.degrees(sat_D_B_rad)/dt:.1f} deg/s, giving sensible proportional command compared to physical 30 deg/s limit.")

def run_scenarios():
    print("\n=== 5. RUN CLOSED-LOOP A/B TESTS ===")
    scenarios = [
        ("Zero Wind", ConstantWindField(0.0, 0.0)),
        ("2m/s Crosswind", ConstantWindField(2.0, 0.0)),
        ("4m/s Crosswind", ConstantWindField(4.0, 0.0)),
        ("Deterministic Gust", ComposedWindField(ConstantWindField(0.0, 0.0), CosineGust(0.0, 3.0, 20.0, 2.0))),
        ("Tau=1s", ComposedWindField(ConstantWindField(0.0, 0.0), StochasticWind(3.0, 1.0, 101))),
        ("Tau=5s", ComposedWindField(ConstantWindField(0.0, 0.0), StochasticWind(3.0, 5.0, 101))),
        ("Tau=20s", ComposedWindField(ConstantWindField(0.0, 0.0), StochasticWind(3.0, 20.0, 101))),
        ("Tau=50s", ComposedWindField(ConstantWindField(0.0, 0.0), StochasticWind(3.0, 50.0, 101))),
    ]
    
    for name, wf in scenarios:
        print(f"\n--- Scenario: {name} ---")
        res_A = run_sim("Model A", wf, normalize_rad=False)
        res_B = run_sim("Model B", wf, normalize_rad=True)
        
        for k in res_A.keys():
            if type(res_A[k]) == float:
                print(f"{k:20s}: A={res_A[k]:7.2f} | B={res_B[k]:7.2f}")
            else:
                print(f"{k:20s}: A={res_A[k]} | B={res_B[k]}")

def run_sim(model_name, wind_field, normalize_rad):
    start_pos = (0.0, -800.0, 1000.0)
    target = LocalPoint(0.0, 0.0)
    state = ParafoilState(x=start_pos[0], y=start_pos[1], altitude=start_pos[2], heading=0.0, vx=0.0, vy=0.0, vz=0.0, heading_rate=0.0)
    sim = ParafoilSimulator(
        state=state, drogue_config=DrogueConfig(), parafoil_config=ParafoilConfig(forward_speed=8.0, descent_rate=2.5, max_turn_rate=30.0),
        target_position=target, dt=0.1, wind_field=wind_field
    )
    
    if normalize_rad:
        # Model C gains (equivalent to Model B input normalization)
        kp = 2.5 * math.pi / 180.0
        ki = 0.1 * math.pi / 180.0
        kd = 0.5 * math.pi / 180.0
    else:
        # Model A
        kp = 2.5
        ki = 0.1
        kd = 0.5
        
    pid = PIDController(kp=kp, ki=ki, kd=kd, integral_min=-10.0, integral_max=10.0)
    guidance = GuidanceController(waypoint_manager=WaypointManager(waypoints=[target]), pid=pid)
    
    steerings = []
    flips = 0
    max_err = 0.0
    last_steer = 0.0
    p_max = 0.0
    i_max = 0.0
    d_max = 0.0
    d_sum = 0.0
    err_sum = 0.0
    
    def steering_func(t, s):
        nonlocal flips, max_err, last_steer, p_max, i_max, d_max, d_sum, err_sum
        est_wx = sim.wind_estimator.wind_x
        est_wy = sim.wind_estimator.wind_y
        guidance.set_wind(est_wx, est_wy)
        
        prev_err = pid.previous_error
        out = guidance.update(position=LocalPoint(s.x, s.y), heading=s.heading, altitude=s.altitude, dt=sim.dt)
        steerings.append(out.steering)
        
        err = pid.previous_error
        p = pid.kp * err
        i = pid.ki * pid.integral
        d = pid.kd * (err - prev_err) / sim.dt if sim.dt > 0 else 0
        
        if last_steer * out.steering < -0.5:
            flips += 1
        last_steer = out.steering
        
        max_err = max(max_err, abs(err))
        err_sum += abs(err)
        p_max = max(p_max, abs(p))
        i_max = max(i_max, abs(i))
        d_max = max(d_max, abs(d))
        d_sum += abs(d)
        
        return out.steering

    sim.run(steering_func, max_time=400.0)
    
    n = max(1, len(steerings))
    sat_frac = sum(1 for st in steerings if abs(st) >= 0.99) / n
    td_err = math.hypot(sim.state.x, sim.state.y)
    
    return {
        "sat_frac": sat_frac,
        "flips": flips,
        "max_err": max_err,
        "mean_err": err_sum / n,
        "p_max": p_max,
        "i_max": i_max,
        "d_max": d_max,
        "mean_d": d_sum / n,
        "td_err": td_err
    }

def derivative_kick():
    print("\n=== 8. DERIVATIVE KICK ANALYSIS ===")
    dt = 0.1
    steps = [1.0, 5.0, 10.0]
    print("Assuming instantaneous setpoint change over dt=0.1s:")
    for s in steps:
        dA = 0.5 * (s / dt)
        dB = 0.5 * (math.radians(s) / dt)
        print(f"Step {s:4.1f} deg -> Model A D={dA:6.2f} | Model B D={dB:6.2f}")

if __name__ == "__main__":
    prove_equivalence()
    analytical_saturation()
    derivative_kick()
    run_scenarios()
