import pytest
from navigation.coordinates import LocalPoint
from flight.parafoil import ParafoilState, ParafoilConfig, DrogueConfig
from environment.wind import ConstantWindField, PiecewiseWindField
from simulation.simulator import ParafoilSimulator
from flight.state_machine import FlightState

def test_simulator_shear_terminal_entry_feasible():
    # Setup simulator with wind that is feasible
    state = ParafoilState(x=0.0, y=0.0, altitude=200.0, heading=0.0, vx=0.0, vy=0.0, vz=0.0, heading_rate=0.0)
    dconfig = DrogueConfig()
    pconfig = ParafoilConfig(forward_speed=10.0, descent_rate=2.0)
    wind = ConstantWindField(5.0, 0.0) # Feasible wind
    
    sim = ParafoilSimulator(
        state=state,
        drogue_config=dconfig,
        parafoil_config=pconfig,
        target_position=LocalPoint(100.0, 0.0),
        wind_field=wind,
        terminal_capture_radius=1000.0 # Large capture radius
    )
    
    sim.state_machine.state = FlightState.PRECISION_GUIDANCE
    sim.step()
    
    # Should enter TERMINAL_GUIDANCE
    assert sim.state_machine.get_state() == FlightState.TERMINAL_GUIDANCE
    assert hasattr(sim, 'terminal_orbit_plan')
    assert sim.terminal_orbit_plan.feasible == True

def test_simulator_shear_terminal_entry_infeasible():
    # Setup simulator with wind that is infeasible (wind > forward_speed)
    state = ParafoilState(x=0.0, y=0.0, altitude=200.0, heading=0.0, vx=0.0, vy=0.0, vz=0.0, heading_rate=0.0)
    dconfig = DrogueConfig()
    pconfig = ParafoilConfig(forward_speed=10.0, descent_rate=2.0)
    wind = ConstantWindField(15.0, 0.0) # Infeasible wind
    
    sim = ParafoilSimulator(
        state=state,
        drogue_config=dconfig,
        parafoil_config=pconfig,
        target_position=LocalPoint(100.0, 0.0),
        wind_field=wind,
        terminal_capture_radius=1000.0 # Large capture radius
    )
    
    sim.state_machine.state = FlightState.PRECISION_GUIDANCE
    sim.step()
    
    # Should NOT enter TERMINAL_GUIDANCE
    assert sim.state_machine.get_state() == FlightState.PRECISION_GUIDANCE
