import math
import pytest
from environment.wind import PiecewiseWindField, ConstantWindField

def test_piecewise_wind_exact_points():
    points = [
        (0.0, (1.0, 2.0)),
        (100.0, (3.0, 4.0)),
        (600.0, (5.0, 6.0)),
        (1000.0, (7.0, 8.0))
    ]
    wf = PiecewiseWindField(points)
    assert wf.wind_at_altitude(0.0) == (1.0, 2.0)
    assert wf.wind_at_altitude(100.0) == (3.0, 4.0)
    assert wf.wind_at_altitude(600.0) == (5.0, 6.0)
    assert wf.wind_at_altitude(1000.0) == (7.0, 8.0)

def test_piecewise_wind_midpoint_interpolation():
    points = [
        (0.0, (0.0, 0.0)),
        (100.0, (10.0, 20.0))
    ]
    wf = PiecewiseWindField(points)
    wx, wy = wf.wind_at_altitude(50.0)
    assert math.isclose(wx, 5.0)
    assert math.isclose(wy, 10.0)

def test_piecewise_wind_independent_xy():
    points = [
        (0.0, (0.0, 10.0)),
        (100.0, (10.0, 0.0))
    ]
    wf = PiecewiseWindField(points)
    wx, wy = wf.wind_at_altitude(50.0)
    assert math.isclose(wx, 5.0)
    assert math.isclose(wy, 5.0)

def test_piecewise_wind_direction_changing():
    points = [
        (0.0, (0.0, -4.0)),
        (1000.0, (4.0, 0.0))
    ]
    wf = PiecewiseWindField(points)
    wx, wy = wf.wind_at_altitude(500.0)
    assert math.isclose(wx, 2.0)
    assert math.isclose(wy, -2.0)

def test_piecewise_wind_below_min_altitude():
    points = [
        (100.0, (1.0, 1.0)),
        (1000.0, (2.0, 2.0))
    ]
    wf = PiecewiseWindField(points)
    assert wf.wind_at_altitude(50.0) == (1.0, 1.0)
    assert wf.wind_at_altitude(-50.0) == (1.0, 1.0)

def test_piecewise_wind_above_max_altitude():
    points = [
        (0.0, (1.0, 1.0)),
        (1000.0, (2.0, 2.0))
    ]
    wf = PiecewiseWindField(points)
    assert wf.wind_at_altitude(1500.0) == (2.0, 2.0)
    assert wf.wind_at_altitude(2000.0) == (2.0, 2.0)

def test_piecewise_wind_invalid_profile():
    with pytest.raises(ValueError):
        PiecewiseWindField([])
    
    with pytest.raises(ValueError):
        PiecewiseWindField([(0.0, (1.0, 1.0))])
        
    with pytest.raises(ValueError):
        PiecewiseWindField([(100.0, (1.0, 1.0)), (100.0, (2.0, 2.0))])
        
    with pytest.raises(ValueError):
        PiecewiseWindField([(100.0, (1.0, 1.0)), (50.0, (2.0, 2.0))])
        
    with pytest.raises(ValueError):
        PiecewiseWindField([(float('nan'), (1.0, 1.0)), (100.0, (2.0, 2.0))])
        
    wf = PiecewiseWindField([(0.0, (1.0, 1.0)), (100.0, (2.0, 2.0))])
    with pytest.raises(ValueError):
        wf.wind_at_altitude(float('nan'))

def test_piecewise_wind_constant_profile_equivalence():
    points = [
        (0.0, (3.0, -2.0)),
        (100.0, (3.0, -2.0)),
        (600.0, (3.0, -2.0)),
        (1000.0, (3.0, -2.0))
    ]
    pwf = PiecewiseWindField(points)
    cwf = ConstantWindField(3.0, -2.0)
    
    for alt in [-100.0, 0.0, 50.0, 500.0, 1000.0, 1500.0]:
        assert pwf.wind_at_altitude(alt) == cwf.wind_at_altitude(alt)
