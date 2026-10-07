from environment.wind import PiecewiseWindField
from simulation.simulator import ParafoilSimulator
from flight.parafoil import ParafoilConfig
from navigation.waypoints import LocalPoint
import math

def test_dynamic():
    # Strong headwind up high, tailwind down low
    wind = PiecewiseWindField([
        (0.0, (5.0, 0.0)),
        (500.0, (0.0, 0.0)),
        (1000.0, (-10.0, 0.0))
    ])
    
    config = ParafoilConfig(
        forward_speed=8.0,
        descent_rate=2.5,
        max_turn_rate=math.radians(30),
    )
    
    sim = ParafoilSimulator(
        config=config,
        initial_x=0.0,
        initial_y=0.0,
        initial_altitude=1000.0,
        initial_heading=90.0,
        target_position=LocalPoint(2000.0, 0.0),
        wind_field=wind,
        terminal_margin=100.0
    )
    
    while sim.state.altitude > 0 and sim.time < 1000:
        sim.update(0.1)
        if sim.state_machine.current_state.name == "TERMINAL_GUIDANCE":
            dist = math.hypot(sim.state.x - 2000.0, sim.state.y - 0.0)
            print(f"Entered TERMINAL at alt={sim.state.altitude:.2f}, dist={dist:.2f}, time={sim.time:.1f}")
            break
            
if __name__ == "__main__":
    test_dynamic()
