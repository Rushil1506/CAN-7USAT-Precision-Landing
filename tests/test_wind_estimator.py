from guidance.wind_estimator import WindEstimator

def test_wind_estimator_initialization():
    est = WindEstimator(alpha=0.1)
    assert not est.initialized
    assert est.wind_x == 0.0
    assert est.wind_y == 0.0

def test_wind_estimator_first_update():
    est = WindEstimator(alpha=0.1)
    est.update(5.0, -3.0)
    assert est.initialized
    assert est.wind_x == 5.0
    assert est.wind_y == -3.0

def test_wind_estimator_ema():
    est = WindEstimator(alpha=0.5)
    est.update(10.0, 10.0) # initializes to 10, 10
    
    # next update should be 0.5 * 20.0 + 0.5 * 10.0 = 15.0
    est.update(20.0, 20.0)
    assert est.wind_x == 15.0
    assert est.wind_y == 15.0

def test_wind_estimator_reset():
    est = WindEstimator(alpha=0.1)
    est.update(5.0, -3.0)
    est.reset()
    assert not est.initialized
    assert est.wind_x == 0.0
    assert est.wind_y == 0.0
