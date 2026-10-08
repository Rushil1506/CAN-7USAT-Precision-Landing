import pytest
import math
from environment.wind import ConstantWindField, CosineGust, ComposedWindField, PiecewiseWindField

def test_cosine_gust_evaluation():
    gust = CosineGust(gust_x=5.0, gust_y=-3.0, t_start=10.0, duration=20.0)
    
    # Before start
    assert gust.evaluate(5.0) == (0.0, 0.0)
    
    # At start
    assert gust.evaluate(10.0) == (0.0, 0.0)
    
    # At peak (t=20.0, halfway)
    peak_gx, peak_gy = gust.evaluate(20.0)
    assert math.isclose(peak_gx, 5.0, abs_tol=1e-5)
    assert math.isclose(peak_gy, -3.0, abs_tol=1e-5)
    
    # At end
    end_gx, end_gy = gust.evaluate(30.0)
    assert math.isclose(end_gx, 0.0, abs_tol=1e-5)
    assert math.isclose(end_gy, 0.0, abs_tol=1e-5)
    
    # After end
    assert gust.evaluate(35.0) == (0.0, 0.0)

def test_composed_wind_field():
    base = ConstantWindField(2.0, 1.0)
    gust = CosineGust(3.0, 4.0, 5.0, 10.0)
    composed = ComposedWindField(base, gust)
    
    # wind_at_altitude (time=0)
    assert composed.wind_at_altitude(100.0) == (2.0, 1.0)
    
    # Before gust
    assert composed.wind_at(100.0, 0.0) == (2.0, 1.0)
    
    # Peak gust (t=10.0)
    peak_wx, peak_wy = composed.wind_at(100.0, 10.0)
    assert math.isclose(peak_wx, 5.0, abs_tol=1e-5)
    assert math.isclose(peak_wy, 5.0, abs_tol=1e-5)
    
    # After gust
    assert composed.wind_at(100.0, 20.0) == (2.0, 1.0)

def test_cosine_gust_validation():
    with pytest.raises(ValueError):
        CosineGust(math.inf, 1.0, 0.0, 10.0)
    with pytest.raises(ValueError):
        CosineGust(1.0, 1.0, math.nan, 10.0)
    with pytest.raises(ValueError):
        CosineGust(1.0, 1.0, 0.0, -5.0)
