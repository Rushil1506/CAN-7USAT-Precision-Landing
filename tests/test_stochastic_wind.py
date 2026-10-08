import pytest
import math
from environment.wind import StochasticWind, ConstantWindField, ComposedWindField

def test_stochastic_wind_sanity():
    # Zero amplitude
    wind = StochasticWind(std_dev=0.0, correlation_time=1.0, seed=42)
    wx, wy = wind.evaluate(1.0)
    assert wx == 0.0
    assert wy == 0.0

def test_stochastic_wind_reproducibility():
    wind1 = StochasticWind(std_dev=2.0, correlation_time=10.0, seed=123)
    wind2 = StochasticWind(std_dev=2.0, correlation_time=10.0, seed=123)
    
    for t in [0.1, 0.5, 1.2, 5.0]:
        w1x, w1y = wind1.evaluate(t)
        w2x, w2y = wind2.evaluate(t)
        assert w1x == w2x
        assert w1y == w2y

def test_stochastic_wind_different_seeds():
    wind1 = StochasticWind(std_dev=2.0, correlation_time=10.0, seed=123)
    wind2 = StochasticWind(std_dev=2.0, correlation_time=10.0, seed=456)
    
    # Evolve a bit
    for t in [0.1, 0.2, 0.3, 0.4]:
        w1 = wind1.evaluate(t)
        w2 = wind2.evaluate(t)
        
    w1 = wind1.evaluate(0.5)
    w2 = wind2.evaluate(0.5)
    
    assert w1 != w2

def test_stochastic_wind_bounds():
    wind = StochasticWind(std_dev=1.0, correlation_time=0.1, seed=999)
    # Run for a long time to see if it bounds to 3*std_dev = 3.0
    
    max_wx = 0.0
    max_wy = 0.0
    
    t = 0.0
    for _ in range(1000):
        t += 0.1
        wx, wy = wind.evaluate(t)
        max_wx = max(max_wx, abs(wx))
        max_wy = max(max_wy, abs(wy))
        
    assert max_wx <= 3.0
    assert max_wy <= 3.0

def test_stochastic_wind_composition():
    base = ConstantWindField(5.0, -3.0)
    stoch = StochasticWind(std_dev=1.0, correlation_time=1.0, seed=42)
    composed = ComposedWindField(base, stoch)
    
    wx, wy = composed.wind_at(100.0, 0.1)
    # The stochastic part won't be zero, so it should differ from base
    assert wx != 5.0
    assert wy != -3.0
