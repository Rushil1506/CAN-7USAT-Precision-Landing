import math
import pytest
from navigation.coordinates import LocalPoint
from environment.wind import ConstantWindField, PiecewiseWindField
from guidance.shear_orbit_planner import solve_shear_terminal_plan

def test_shear_planner_zero_wind():
    wind = ConstantWindField(0.0, 0.0)
    plan = solve_shear_terminal_plan(
        position=LocalPoint(0.0, 0.0),
        target=LocalPoint(100.0, 100.0),
        current_altitude=1000.0,
        wind_field=wind,
        forward_speed=8.0,
        descent_rate=2.5,
        orbit_radius=40.0
    )
    assert plan.feasible
    assert plan.effective_wind_x == 0.0
    assert plan.breakout_distance == 0.0
    assert plan.planning_iterations == 1
    assert plan.convergence_error == 0.0

def test_shear_planner_constant_wind():
    wind = ConstantWindField(3.0, 0.0)
    plan = solve_shear_terminal_plan(
        position=LocalPoint(0.0, 0.0),
        target=LocalPoint(100.0, 100.0),
        current_altitude=1000.0,
        wind_field=wind,
        forward_speed=8.0,
        descent_rate=2.5,
        orbit_radius=40.0
    )
    assert plan.feasible
    assert math.isclose(plan.effective_wind_x, 3.0, abs_tol=1e-5)
    # Final course should be directly into wind (West, which is 270 deg)
    assert math.isclose(plan.final_ground_course_deg, 270.0, abs_tol=1e-3)
    # Should converge rapidly
    assert plan.planning_iterations <= 2
    
def test_shear_planner_direction_change():
    wind = PiecewiseWindField([
        (0.0, (-1.0, 0.0)),
        (200.0, (0.0, 2.0)),
        (400.0, (2.0, 2.0)),
        (600.0, (4.0, 0.0))
    ])
    plan = solve_shear_terminal_plan(
        position=LocalPoint(0.0, 0.0),
        target=LocalPoint(100.0, 100.0),
        current_altitude=1000.0,
        wind_field=wind,
        forward_speed=8.0,
        descent_rate=2.5,
        orbit_radius=40.0
    )
    assert plan.feasible
    # Ensures no infinite loop
    assert plan.planning_iterations <= 10
    assert plan.breakout_altitude > 0.0
    assert plan.convergence_error <= 0.5

def test_shear_planner_infeasible_layer():
    # Wind > 8.0 m/s airspeed
    wind = PiecewiseWindField([
        (0.0, (0.0, 0.0)),
        (10.0, (10.0, 0.0)),
        (100.0, (10.0, 0.0))
    ])
    plan = solve_shear_terminal_plan(
        position=LocalPoint(0.0, 0.0),
        target=LocalPoint(100.0, 100.0),
        current_altitude=1000.0,
        wind_field=wind,
        forward_speed=8.0,
        descent_rate=2.5,
        orbit_radius=40.0,
        min_final_distance=100.0
    )
    assert not plan.feasible
