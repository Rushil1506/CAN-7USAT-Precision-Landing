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

# --- Regression tests for Phase 4.2f-3a ---

def test_stochastic_wind_explicit_reset():
    # A. Reset behavior
    wind = StochasticWind(std_dev=2.0, correlation_time=1.0, seed=123)
    seq1 = [wind.evaluate(t) for t in [1.0, 2.0, 3.0]]
    
    wind.reset()
    seq2 = [wind.evaluate(t) for t in [1.0, 2.0, 3.0]]
    
    assert seq1 == seq2

def test_stochastic_wind_t0_safety():
    # C. t=0 safety
    wind = StochasticWind(std_dev=2.0, correlation_time=1.0, seed=123)
    wind.evaluate(1.0)
    wind.evaluate(2.0)
    
    val_at_2 = wind.evaluate(2.0)
    
    # Query t=0
    val_at_0 = wind.evaluate(0.0)
    
    # Ensure it just returns current snapshot
    assert val_at_0 == val_at_2
    
    # Next advance should be normal, not huge jump
    val_at_3 = wind.evaluate(3.0)
    
    # Compare with a pure sequence
    wind_pure = StochasticWind(std_dev=2.0, correlation_time=1.0, seed=123)
    wind_pure.evaluate(1.0)
    wind_pure.evaluate(2.0)
    pure_val_at_3 = wind_pure.evaluate(3.0)
    
    assert val_at_3 == pure_val_at_3

def test_stochastic_wind_query_non_interference():
    # B, D. Query non-interference and repeated query safety
    wind_query = StochasticWind(std_dev=2.0, correlation_time=1.0, seed=123)
    wind_pure = StochasticWind(std_dev=2.0, correlation_time=1.0, seed=123)
    
    for t in [1.0, 2.0, 3.0]:
        wind_query.evaluate(t)
        wind_pure.evaluate(t)
        
        # Planner queries at t=0
        wind_query.evaluate(0.0)
        wind_query.evaluate(0.0)
        
    val_query = wind_query.evaluate(4.0)
    val_pure = wind_pure.evaluate(4.0)
    
    assert val_query == val_pure

def test_composed_wind_field_safety():
    # F. ComposedWindField safety
    stoch_query = StochasticWind(std_dev=2.0, correlation_time=1.0, seed=123)
    comp_query = ComposedWindField(ConstantWindField(0.0, 0.0), stoch_query)
    
    stoch_pure = StochasticWind(std_dev=2.0, correlation_time=1.0, seed=123)
    comp_pure = ComposedWindField(ConstantWindField(0.0, 0.0), stoch_pure)
    
    for t in [1.0, 2.0, 3.0]:
        comp_query.wind_at(100.0, t)
        comp_pure.wind_at(100.0, t)
        
        # Planner query via wind_at_altitude (which passes time=0.0)
        comp_query.wind_at_altitude(100.0)
        
    val_query = comp_query.wind_at(100.0, 4.0)
    val_pure = comp_pure.wind_at(100.0, 4.0)
    
    assert val_query == val_pure

def test_stochastic_wind_reproduce_failure_and_fix():
    # 6. Recreate the failure before fixing (which now passes because it's fixed)
    # The failure was: querying wind_at_altitude (time=0) caused reset,
    # leading to the stochastic wind freezing at max bound.
    wind = ComposedWindField(ConstantWindField(0.0, 0.0), StochasticWind(std_dev=3.0, correlation_time=1.0, seed=101))
    
    # Run a simulated sequence with planner queries
    dt = 0.1
    time = 0.0
    for _ in range(50):
        time += dt
        
        # Planner queries
        wind.wind_at_altitude(1000.0)
        
        # Actual environment evolution
        wx, wy = wind.wind_at(1000.0, time)
        
    # If the bug were present, wx and wy would be stuck at exactly 9.0 and -9.0 (from seed 101 bounding)
    # With the fix, they will be normally evolving values, not stuck at boundaries.
    assert wx != 9.0
    assert wy != -9.0
    assert wx != -9.0
    assert wy != 9.0
