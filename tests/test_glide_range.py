import math
import pytest

from flight.parafoil import ParafoilConfig
from guidance.glide_range import (
    GlideRangeEstimate,
    calculate_time_to_ground,
    calculate_glide_range,
    estimate_glide_range,
    estimate_glide_range_from_config,
)


def test_zero_altitude():
    """At ground level (0 m), time to ground and glide range should be 0."""
    result = estimate_glide_range(
        altitude=0.0,
        descent_rate=2.5,
        horizontal_speed=8.0,
    )
    assert math.isclose(result.time_to_ground, 0.0)
    assert math.isclose(result.glide_range, 0.0)
    assert math.isclose(calculate_glide_range(0.0, 2.5, 8.0), 0.0)
    assert math.isclose(calculate_time_to_ground(0.0, 2.5), 0.0)


def test_normal_altitude():
    """Standard altitude check with nominal parameters."""
    # 600 m altitude / 2.5 m/s descent = 240 s
    # 240 s * 8.0 m/s = 1920 m
    result = estimate_glide_range(
        altitude=600.0,
        descent_rate=2.5,
        horizontal_speed=8.0,
    )
    assert math.isclose(result.time_to_ground, 240.0)
    assert math.isclose(result.glide_range, 1920.0)
    assert math.isclose(calculate_glide_range(600.0, 2.5, 8.0), 1920.0)
    assert math.isclose(calculate_time_to_ground(600.0, 2.5), 240.0)


def test_faster_descent_rate():
    """Higher descent rate reduces time to ground and glide range."""
    # 600 m / 5.0 m/s = 120 s
    # 120 s * 8.0 m/s = 960 m
    nominal = estimate_glide_range(altitude=600.0, descent_rate=2.5, horizontal_speed=8.0)
    faster = estimate_glide_range(altitude=600.0, descent_rate=5.0, horizontal_speed=8.0)

    assert math.isclose(faster.time_to_ground, 120.0)
    assert math.isclose(faster.glide_range, 960.0)
    assert faster.glide_range < nominal.glide_range
    assert faster.time_to_ground < nominal.time_to_ground


def test_slower_descent_rate():
    """Lower descent rate increases time to ground and glide range."""
    # 600 m / 1.25 m/s = 480 s
    # 480 s * 8.0 m/s = 3840 m
    nominal = estimate_glide_range(altitude=600.0, descent_rate=2.5, horizontal_speed=8.0)
    slower = estimate_glide_range(altitude=600.0, descent_rate=1.25, horizontal_speed=8.0)

    assert math.isclose(slower.time_to_ground, 480.0)
    assert math.isclose(slower.glide_range, 3840.0)
    assert slower.glide_range > nominal.glide_range
    assert slower.time_to_ground > nominal.time_to_ground


def test_zero_horizontal_speed():
    """With zero forward speed, vehicle descends vertically with zero glide range."""
    result = estimate_glide_range(
        altitude=600.0,
        descent_rate=2.5,
        horizontal_speed=0.0,
    )
    assert math.isclose(result.time_to_ground, 240.0)
    assert math.isclose(result.glide_range, 0.0)
    assert math.isclose(calculate_glide_range(600.0, 2.5, 0.0), 0.0)


def test_zero_descent_rate_raises():
    """A zero descent rate must raise ValueError (cannot divide by zero)."""
    with pytest.raises(ValueError, match="descent_rate must be greater than zero"):
        estimate_glide_range(altitude=600.0, descent_rate=0.0, horizontal_speed=8.0)

    with pytest.raises(ValueError, match="descent_rate must be greater than zero"):
        calculate_glide_range(altitude=600.0, descent_rate=0.0, horizontal_speed=8.0)

    with pytest.raises(ValueError, match="descent_rate must be greater than zero"):
        calculate_time_to_ground(altitude=600.0, descent_rate=0.0)


def test_negative_descent_rate_raises():
    """A negative descent rate must raise ValueError."""
    with pytest.raises(ValueError, match="descent_rate must be greater than zero"):
        estimate_glide_range(altitude=600.0, descent_rate=-2.5, horizontal_speed=8.0)

    with pytest.raises(ValueError, match="descent_rate must be greater than zero"):
        calculate_glide_range(altitude=600.0, descent_rate=-2.5, horizontal_speed=8.0)

    with pytest.raises(ValueError, match="descent_rate must be greater than zero"):
        calculate_time_to_ground(altitude=600.0, descent_rate=-2.5)


def test_negative_altitude_raises():
    """A negative altitude must raise ValueError."""
    with pytest.raises(ValueError, match="altitude cannot be negative"):
        estimate_glide_range(altitude=-10.0, descent_rate=2.5, horizontal_speed=8.0)

    with pytest.raises(ValueError, match="altitude cannot be negative"):
        calculate_glide_range(altitude=-10.0, descent_rate=2.5, horizontal_speed=8.0)

    with pytest.raises(ValueError, match="altitude cannot be negative"):
        calculate_time_to_ground(altitude=-10.0, descent_rate=2.5)


def test_negative_horizontal_speed_raises():
    """A negative horizontal speed must raise ValueError."""
    with pytest.raises(ValueError, match="horizontal_speed cannot be negative"):
        estimate_glide_range(altitude=600.0, descent_rate=2.5, horizontal_speed=-8.0)

    with pytest.raises(ValueError, match="horizontal_speed cannot be negative"):
        calculate_glide_range(altitude=600.0, descent_rate=2.5, horizontal_speed=-8.0)


def test_estimate_from_config():
    """Verify convenience wrapper using ParafoilConfig."""
    config = ParafoilConfig(forward_speed=8.0, descent_rate=2.5)
    result = estimate_glide_range_from_config(altitude=600.0, config=config)

    assert isinstance(result, GlideRangeEstimate)
    assert math.isclose(result.time_to_ground, 240.0)
    assert math.isclose(result.glide_range, 1920.0)

def test_pure_tailwind():
    # Test B: Pure 3 m/s tailwind along East course
    result = estimate_glide_range(altitude=600.0, descent_rate=2.5, horizontal_speed=8.0, wind_x=3.0, wind_y=0.0, desired_course_deg=90.0)
    assert result.feasible is True
    assert math.isclose(result.projected_ground_speed, 11.0)
    assert math.isclose(result.glide_range, 2640.0)

def test_pure_headwind():
    # Test C: Pure 3 m/s headwind along East course
    result = estimate_glide_range(altitude=600.0, descent_rate=2.5, horizontal_speed=8.0, wind_x=-3.0, wind_y=0.0, desired_course_deg=90.0)
    assert result.feasible is True
    assert math.isclose(result.projected_ground_speed, 5.0)
    assert math.isclose(result.glide_range, 1200.0)

def test_crosswind():
    # Test D: 4 m/s crosswind with 8 m/s airspeed
    result = estimate_glide_range(altitude=600.0, descent_rate=2.5, horizontal_speed=8.0, wind_x=0.0, wind_y=4.0, desired_course_deg=90.0)
    assert result.feasible is True
    # crab = 30 deg -> cos(30) = sqrt(3)/2 ~= 0.866
    expected_speed = 8.0 * math.cos(math.radians(30.0))
    assert math.isclose(result.projected_ground_speed, expected_speed, abs_tol=1e-5)
    assert math.isclose(result.glide_range, expected_speed * (600.0/2.5), abs_tol=1e-5)

def test_critical_crosswind():
    # Test E: Critical crosswind
    result = estimate_glide_range(altitude=600.0, descent_rate=2.5, horizontal_speed=8.0, wind_x=0.0, wind_y=8.0, desired_course_deg=90.0)
    assert result.feasible is True
    assert math.isclose(result.projected_ground_speed, 0.0, abs_tol=1e-5)
    assert math.isclose(result.glide_range, 0.0, abs_tol=1e-5)

def test_infeasible_crosswind():
    # Test F: Infeasible crosswind
    result = estimate_glide_range(altitude=600.0, descent_rate=2.5, horizontal_speed=8.0, wind_x=0.0, wind_y=10.0, desired_course_deg=90.0)
    assert result.feasible is False
    assert math.isclose(result.glide_range, 0.0)

def test_strong_headwind():
    # Test G: Strong headwind
    result = estimate_glide_range(altitude=600.0, descent_rate=2.5, horizontal_speed=8.0, wind_x=-9.0, wind_y=0.0, desired_course_deg=90.0)
    assert result.feasible is False
    assert result.projected_ground_speed < 0.0
    assert math.isclose(result.glide_range, 0.0)

def test_diagonal_wind():
    # Test H: Diagonal wind
    # non-cardinal course 45 deg, tailwind (2, 2)
    result = estimate_glide_range(altitude=600.0, descent_rate=2.5, horizontal_speed=8.0, wind_x=2.0, wind_y=2.0, desired_course_deg=45.0)
    assert result.feasible is True
    assert math.isclose(result.projected_ground_speed, 8.0 + math.hypot(2.0, 2.0))

