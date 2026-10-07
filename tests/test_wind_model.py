import pytest
from environment.wind import ConstantWindField

def test_constant_wind_field_zero():
    wind = ConstantWindField(0.0, 0.0)
    assert wind.wind_at_altitude(0) == (0.0, 0.0)
    assert wind.wind_at_altitude(100) == (0.0, 0.0)
    assert wind.wind_at_altitude(600) == (0.0, 0.0)
    assert wind.wind_at_altitude(1000) == (0.0, 0.0)
    assert wind.wind_at_altitude(-50) == (0.0, 0.0)

def test_constant_wind_field_nonzero():
    wind = ConstantWindField(3.0, -2.0)
    assert wind.wind_at_altitude(0) == (3.0, -2.0)
    assert wind.wind_at_altitude(100) == (3.0, -2.0)
    assert wind.wind_at_altitude(600) == (3.0, -2.0)
    assert wind.wind_at_altitude(1000) == (3.0, -2.0)
    assert wind.wind_at_altitude(-50) == (3.0, -2.0)
