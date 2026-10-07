import math
from environment.wind import PiecewiseWindField
from flight.parafoil import DrogueConfig, ParafoilConfig, ParafoilState
from flight.state_machine import FlightState
from simulation.simulator import ParafoilSimulator

def run_shear_validation():
    # Deterministic baseline shear scenario
    # Do NOT claim this represents real atmospheric data.
    points = [
        (0.0,   (1.0, 0.0)),
        (200.0, (1.5, 0.0)),
        (400.0, (2.0, 0.5)),
        (600.0, (3.0, 1.0)),
        (800.0, (3.5, 1.5)),
        (1000.0,(4.0, 2.0))
    ]
    
    wind_field = PiecewiseWindField(points)
    
    state = ParafoilState(x=0.0, y=0.0, altitude=1000.0, vx=0.0, vy=0.0, vz=0.0, heading=0.0, heading_rate=0.0)
    
    sim = ParafoilSimulator(
        state=state, 
        drogue_config=DrogueConfig(),
        parafoil_config=ParafoilConfig(forward_speed=8.0, descent_rate=2.5),
        wind_field=wind_field
    )
    
    def steering_cb(t, st):
        return 0.0

    history = sim.run(steering_function=steering_cb, max_time=600.0)
    
    print("Shear Validation Telemetry:")
    print(f"{'Altitude (m)':<15} | {'Wind X (m/s)':<15} | {'Wind Y (m/s)':<15}")
    print("-" * 50)
    
    target_alts = [1000.0, 800.0, 600.0, 400.0, 200.0, 100.0, 0.0]
    
    for target in target_alts:
        # Find the sample closest to the target altitude
        sample = min(history, key=lambda s: abs(s.altitude - target))
        print(f"{sample.altitude:<15.2f} | {sample.wind_x:<15.2f} | {sample.wind_y:<15.2f}")

if __name__ == '__main__':
    run_shear_validation()
