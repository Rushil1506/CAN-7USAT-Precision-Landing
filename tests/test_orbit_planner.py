import math
import pytest
from guidance.orbit_planner import plan_terminal_orbit
from navigation.coordinates import LocalPoint
from guidance.loiter_guidance import LoiterDirection

def test_zero_wind():
    plan = plan_terminal_orbit(
        position=LocalPoint(100, 100),
        target=LocalPoint(0, 0),
        current_altitude=500.0,
        wind_x=0.0,
        wind_y=0.0,
        forward_speed=8.0,
        descent_rate=2.5,
        orbit_radius=30.0
    )
    assert plan.feasible
    assert plan.orbit_center.x == 0.0
    assert plan.orbit_center.y == 0.0
    assert plan.breakout_altitude == 2.0
    assert plan.d_selected == 0.0

def test_plus_x_wind():
    # Wind moves East (+X). Final course should be West.
    # Breakout point should be East (downwind) of target.
    plan = plan_terminal_orbit(
        position=LocalPoint(100, 100),
        target=LocalPoint(0, 0),
        current_altitude=500.0,
        wind_x=3.0,
        wind_y=0.0,
        forward_speed=8.0,
        descent_rate=2.5,
        orbit_radius=30.0
    )
    assert plan.feasible
    assert plan.breakout_point.x > 0.0
    assert plan.breakout_point.y == 0.0
    assert plan.final_ground_speed == 5.0
    
def test_minus_x_wind():
    plan = plan_terminal_orbit(
        position=LocalPoint(-100, -100),
        target=LocalPoint(0, 0),
        current_altitude=500.0,
        wind_x=-3.0,
        wind_y=0.0,
        forward_speed=8.0,
        descent_rate=2.5,
        orbit_radius=30.0
    )
    assert plan.feasible
    assert plan.breakout_point.x < 0.0
    assert plan.breakout_point.y == 0.0
    
def test_plus_y_wind():
    plan = plan_terminal_orbit(
        position=LocalPoint(100, 100),
        target=LocalPoint(0, 0),
        current_altitude=500.0,
        wind_x=0.0,
        wind_y=3.0,
        forward_speed=8.0,
        descent_rate=2.5,
        orbit_radius=30.0
    )
    assert plan.feasible
    assert plan.breakout_point.x == 0.0
    assert plan.breakout_point.y > 0.0
    
def test_minus_y_wind():
    plan = plan_terminal_orbit(
        position=LocalPoint(-100, -100),
        target=LocalPoint(0, 0),
        current_altitude=500.0,
        wind_x=0.0,
        wind_y=-3.0,
        forward_speed=8.0,
        descent_rate=2.5,
        orbit_radius=30.0
    )
    assert plan.feasible
    assert plan.breakout_point.x == 0.0
    assert plan.breakout_point.y < 0.0
    
def test_diagonal_wind():
    plan = plan_terminal_orbit(
        position=LocalPoint(100, 100),
        target=LocalPoint(0, 0),
        current_altitude=500.0,
        wind_x=3.0,
        wind_y=3.0,
        forward_speed=8.0,
        descent_rate=2.5,
        orbit_radius=30.0
    )
    assert plan.feasible
    assert plan.breakout_point.x > 0.0
    assert plan.breakout_point.y > 0.0
    assert math.isclose(plan.breakout_point.x, plan.breakout_point.y)

def test_near_critical_wind():
    plan = plan_terminal_orbit(
        position=LocalPoint(100, 100),
        target=LocalPoint(0, 0),
        current_altitude=1500.0,
        wind_x=7.9,
        wind_y=0.0,
        forward_speed=8.0,
        descent_rate=2.5,
        orbit_radius=30.0
    )
    assert plan.feasible
    assert math.isclose(plan.final_ground_speed, 0.1)
    
def test_critical_infeasible_wind():
    plan = plan_terminal_orbit(
        position=LocalPoint(100, 100),
        target=LocalPoint(0, 0),
        current_altitude=500.0,
        wind_x=8.5,
        wind_y=0.0,
        forward_speed=8.0,
        descent_rate=2.5,
        orbit_radius=30.0
    )
    assert not plan.feasible
    
def test_cw_ccw_symmetry():
    # Pos at (100, 100) -> picks CW if it's closer
    plan_pos1 = plan_terminal_orbit(
        position=LocalPoint(0, 100),
        target=LocalPoint(0, 0),
        current_altitude=500.0,
        wind_x=3.0,
        wind_y=0.0,
        forward_speed=8.0,
        descent_rate=2.5,
        orbit_radius=30.0
    )
    
    # Pos at (0, -100)
    plan_pos2 = plan_terminal_orbit(
        position=LocalPoint(0, -100),
        target=LocalPoint(0, 0),
        current_altitude=500.0,
        wind_x=3.0,
        wind_y=0.0,
        forward_speed=8.0,
        descent_rate=2.5,
        orbit_radius=30.0
    )
    
    # Because breakout point is East of target, one side is North, one is South
    # (0, 100) is North -> should pick center that is North
    # (0, -100) is South -> should pick center that is South
    assert plan_pos1.orbit_direction != plan_pos2.orbit_direction
    assert plan_pos1.orbit_center.y > 0
    assert plan_pos2.orbit_center.y < 0

def test_heading_cross_product_geometry():
    # wind is East (+3, 0)
    plan = plan_terminal_orbit(
        position=LocalPoint(0, 100),
        target=LocalPoint(0, 0),
        current_altitude=500.0,
        wind_x=3.0,
        wind_y=0.0,
        forward_speed=8.0,
        descent_rate=2.5,
        orbit_radius=30.0
    )
    # the tangent from the orbit at P_b must point West (-1, 0)
    dx = plan.breakout_point.x - plan.orbit_center.x
    dy = plan.breakout_point.y - plan.orbit_center.y
    # For CW (direction=1), velocity is rotated -90 deg from position vector
    if plan.orbit_direction == LoiterDirection.CLOCKWISE:
        tangent_x = dy
        tangent_y = -dx
    else:
        tangent_x = -dy
        tangent_y = dx
        
    assert tangent_x < 0 # pointing West
    assert tangent_y == 0
