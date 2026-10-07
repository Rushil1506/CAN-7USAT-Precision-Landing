import math
import pytest
from guidance.wind_compensation import calculate_wind_corrected_heading

def test_zero_wind():
    # Test A: Zero wind
    res = calculate_wind_corrected_heading(desired_course_deg=90.0, airspeed=8.0, wind_x=0.0, wind_y=0.0)
    assert res.feasible is True
    assert math.isclose(res.crab_angle_deg, 0.0, abs_tol=1e-9)
    assert math.isclose(res.heading_deg, 90.0)
    assert math.isclose(res.ground_speed_along_track, 8.0)

def test_northward_crosswind_flying_east():
    # Test B
    res = calculate_wind_corrected_heading(desired_course_deg=90.0, airspeed=8.0, wind_x=0.0, wind_y=4.0)
    assert res.feasible is True
    assert math.isclose(res.crosswind_component, -4.0)
    assert math.isclose(res.crab_angle_deg, 30.0)
    assert math.isclose(res.heading_deg, 120.0)

def test_southward_crosswind_flying_east():
    # Test C
    res = calculate_wind_corrected_heading(desired_course_deg=90.0, airspeed=8.0, wind_x=0.0, wind_y=-4.0)
    assert res.feasible is True
    assert math.isclose(res.crab_angle_deg, -30.0)
    assert math.isclose(res.heading_deg, 60.0)

def test_pure_headwind():
    # Test D: Flying East, Wind West
    res = calculate_wind_corrected_heading(desired_course_deg=90.0, airspeed=8.0, wind_x=-3.0, wind_y=0.0)
    assert res.feasible is True
    assert math.isclose(res.crab_angle_deg, 0.0, abs_tol=1e-9)
    assert math.isclose(res.heading_deg, 90.0)
    assert math.isclose(res.ground_speed_along_track, 5.0)

def test_pure_tailwind():
    # Test E: Flying East, Wind East
    res = calculate_wind_corrected_heading(desired_course_deg=90.0, airspeed=8.0, wind_x=3.0, wind_y=0.0)
    assert res.feasible is True
    assert math.isclose(res.crab_angle_deg, 0.0, abs_tol=1e-9)
    assert math.isclose(res.heading_deg, 90.0)
    assert math.isclose(res.ground_speed_along_track, 11.0)

def test_exactly_critical_crosswind():
    # Test F
    res = calculate_wind_corrected_heading(desired_course_deg=90.0, airspeed=8.0, wind_x=0.0, wind_y=8.0)
    assert res.feasible is True
    assert math.isclose(abs(res.crab_angle_deg), 90.0)
    assert math.isclose(res.ground_speed_along_track, 0.0, abs_tol=1e-6)

def test_infeasible_crosswind():
    # Test G
    res = calculate_wind_corrected_heading(desired_course_deg=90.0, airspeed=8.0, wind_x=0.0, wind_y=10.0)
    assert res.feasible is False

def test_heading_wraparound():
    # Test H: Course 350. If crab = +20, heading = 10.
    # W_perp must be -V_air * sin(+20). Let's construct this W_perp directly.
    # Course 350 => cos(350)=0.9848, sin(350)=-0.1736
    # Let W_parallel = 0.
    # We want W_perp = 8.0 * math.sin(math.radians(-20))
    # wind_x = W_parallel*sin(chi) + W_perp*cos(chi)
    # wind_y = W_parallel*cos(chi) - W_perp*sin(chi)
    chi = math.radians(350.0)
    w_perp = 8.0 * math.sin(math.radians(-20.0))
    wind_x = w_perp * math.cos(chi)
    wind_y = -w_perp * math.sin(chi)
    
    res = calculate_wind_corrected_heading(desired_course_deg=350.0, airspeed=8.0, wind_x=wind_x, wind_y=wind_y)
    assert res.feasible is True
    assert math.isclose(res.crab_angle_deg, 20.0, abs_tol=1e-5)
    assert math.isclose(res.heading_deg, 10.0, abs_tol=1e-5)
    
    # Course 10. If crab = -20, heading = 350.
    chi2 = math.radians(10.0)
    w_perp2 = 8.0 * math.sin(math.radians(20.0))
    wind_x2 = w_perp2 * math.cos(chi2)
    wind_y2 = -w_perp2 * math.sin(chi2)
    
    res2 = calculate_wind_corrected_heading(desired_course_deg=10.0, airspeed=8.0, wind_x=wind_x2, wind_y=wind_y2)
    assert res2.feasible is True
    assert math.isclose(res2.crab_angle_deg, -20.0, abs_tol=1e-5)
    assert math.isclose(res2.heading_deg, 350.0, abs_tol=1e-5)

def test_other_course_angles():
    # Test I: Non-cardinal
    res = calculate_wind_corrected_heading(desired_course_deg=45.0, airspeed=8.0, wind_x=2.0, wind_y=2.0)
    assert res.feasible is True
    assert math.isclose(res.crab_angle_deg, 0.0, abs_tol=1e-9)
    assert math.isclose(res.heading_deg, 45.0)
    assert math.isclose(res.ground_speed_along_track, 8.0 + math.hypot(2.0, 2.0))
