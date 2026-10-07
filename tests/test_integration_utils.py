import math
import pytest
from environment.wind import ConstantWindField, PiecewiseWindField
from guidance.integration_utils import (
    integrate_altitude_scalar,
    effective_wind,
    evaluate_interval_feasibility
)
from guidance.glide_range import estimate_glide_range

def test_constant_zero_wind():
    wind = ConstantWindField(0.0, 0.0)
    wx, wy = effective_wind(wind, 0.0, 1000.0)
    assert wx == 0.0
    assert wy == 0.0

def test_constant_nonzero_wind():
    wind = ConstantWindField(3.0, -2.0)
    wx, wy = effective_wind(wind, 0.0, 1000.0)
    assert wx == 3.0
    assert wy == -2.0

def test_linear_scalar_wind_component():
    wind = PiecewiseWindField([(0.0, (0.0, 0.0)), (1000.0, (10.0, 0.0))])
    wx, wy = effective_wind(wind, 0.0, 1000.0)
    # Average of 0 and 10 is 5
    assert math.isclose(wx, 5.0, abs_tol=1e-5)
    assert math.isclose(wy, 0.0, abs_tol=1e-5)

def test_direction_changing_wind():
    wind = PiecewiseWindField([(0.0, (10.0, 0.0)), (1000.0, (0.0, 10.0))])
    wx, wy = effective_wind(wind, 0.0, 1000.0)
    assert math.isclose(wx, 5.0, abs_tol=1e-5)
    assert math.isclose(wy, 5.0, abs_tol=1e-5)

def test_narrow_interval():
    wind = PiecewiseWindField([(0.0, (0.0, 0.0)), (1000.0, (10.0, 0.0))])
    wx, wy = effective_wind(wind, 400.0, 600.0)
    # At 400: 4.0. At 600: 6.0. Average: 5.0.
    assert math.isclose(wx, 5.0, abs_tol=1e-5)
    
def test_constant_profile_exactness():
    wind = PiecewiseWindField([(0.0, (3.0, -2.0)), (1000.0, (3.0, -2.0))])
    wx, wy = effective_wind(wind, 0.0, 1000.0)
    assert wx == 3.0
    assert wy == -2.0

def test_integration_convergence():
    wind = PiecewiseWindField([(0.0, (0.0, 0.0)), (1000.0, (10.0, 0.0))])
    wx_coarse, _ = effective_wind(wind, 0.0, 1000.0, dh=100.0)
    wx_medium, _ = effective_wind(wind, 0.0, 1000.0, dh=10.0)
    wx_fine, _ = effective_wind(wind, 0.0, 1000.0, dh=1.0)
    assert math.isclose(wx_coarse, wx_medium)
    assert math.isclose(wx_medium, wx_fine)
    
def test_feasibility_over_altitude_interval():
    wind = PiecewiseWindField([(0.0, (2.0, 0.0)), (1000.0, (5.0, 0.0))])
    res = evaluate_interval_feasibility(wind, 0.0, 1000.0, desired_course_deg=90.0, airspeed=8.0)
    assert res.feasible
    assert math.isclose(res.maximum_crosswind_component, 0.0, abs_tol=1e-5)
    assert res.minimum_projected_ground_speed > 0.0

def test_feasibility_failure():
    wind = PiecewiseWindField([(0.0, (0.0, 0.0)), (1000.0, (0.0, 10.0))])
    res = evaluate_interval_feasibility(wind, 0.0, 1000.0, desired_course_deg=90.0, airspeed=8.0)
    assert not res.feasible
    assert res.maximum_crosswind_component > 8.0
    
def test_negative_projected_ground_speed():
    wind = PiecewiseWindField([(0.0, (-10.0, 0.0)), (1000.0, (-10.0, 0.0))])
    res = evaluate_interval_feasibility(wind, 0.0, 1000.0, desired_course_deg=90.0, airspeed=8.0)
    assert not res.feasible
    assert res.minimum_projected_ground_speed <= 0.0

def test_exact_vs_effective_regression():
    # Prove that exact integrated range != V_ground(W_eff) * time
    wind = PiecewiseWindField([(0.0, (0.0, 7.9)), (1000.0, (0.0, -7.9))])
    # Course 90 (East). Wind is pure crosswind.
    # W_eff over [0, 1000] is (0, 0).
    # V_ground(W_eff) = 8.0 m/s. Range = 8.0 * 400 = 3200m.
    
    wx_eff, wy_eff = effective_wind(wind, 0.0, 1000.0)
    assert math.isclose(wx_eff, 0.0, abs_tol=1e-5)
    assert math.isclose(wy_eff, 0.0, abs_tol=1e-5)
    
    res = estimate_glide_range(altitude=1000.0, descent_rate=2.5, horizontal_speed=8.0, desired_course_deg=90.0, wind_field=wind)
    assert res.feasible
    assert res.glide_range < 3190.0 # strictly less than 3200
