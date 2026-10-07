import math
import pytest

from navigation.coordinates import LocalPoint
from guidance.loiter_guidance import (
    LoiterDirection,
    LoiterGuidance,
    LoiterGuidanceOutput,
    LoiterPhase,
    calculate_nominal_turn_rate,
    calculate_radial_distance,
    calculate_radial_error,
    calculate_tangent_heading,
)


def test_radial_distance_calculation():
    """Verify exact Euclidean distance calculation from orbit center."""
    center = LocalPoint(x=0.0, y=0.0)
    pos = LocalPoint(x=30.0, y=40.0)
    assert math.isclose(calculate_radial_distance(center, pos), 50.0)

    # Offset center
    offset_center = LocalPoint(x=100.0, y=200.0)
    offset_pos = LocalPoint(x=103.0, y=204.0)
    assert math.isclose(calculate_radial_distance(offset_center, offset_pos), 5.0)


def test_vehicle_inside_orbit():
    """When r < R, radial error should be negative and vehicle detected inside."""
    center = LocalPoint(x=0.0, y=0.0)
    radius = 30.0
    pos_inside = LocalPoint(x=20.0, y=0.0)

    r_dist = calculate_radial_distance(center, pos_inside)
    r_err = calculate_radial_error(center, pos_inside, radius)

    assert r_dist < radius
    assert math.isclose(r_err, -10.0)

    guidance = LoiterGuidance(center=center, loiter_radius=radius, capture_tolerance=5.0)
    out = guidance.update(position=pos_inside, current_heading=90.0)

    assert out.radial_distance < radius
    assert out.radial_error < 0.0
    assert out.phase == LoiterPhase.ORBIT_CAPTURE


def test_vehicle_outside_orbit():
    """When r > R, radial error should be positive and vehicle detected outside."""
    center = LocalPoint(x=0.0, y=0.0)
    radius = 30.0
    pos_outside = LocalPoint(x=50.0, y=0.0)

    r_dist = calculate_radial_distance(center, pos_outside)
    r_err = calculate_radial_error(center, pos_outside, radius)

    assert r_dist > radius
    assert math.isclose(r_err, 20.0)

    guidance = LoiterGuidance(center=center, loiter_radius=radius, capture_tolerance=5.0)
    out = guidance.update(position=pos_outside, current_heading=90.0)

    assert out.radial_distance > radius
    assert out.radial_error > 0.0
    assert out.phase == LoiterPhase.ORBIT_CAPTURE


def test_exact_orbit_radius_boundary():
    """At r == R, radial error is 0, phase is ORBIT_TRACKING, and heading is pure tangent."""
    center = LocalPoint(x=0.0, y=0.0)
    radius = 30.0
    pos_boundary = LocalPoint(x=0.0, y=30.0)  # North position

    r_err = calculate_radial_error(center, pos_boundary, radius)
    assert math.isclose(r_err, 0.0, abs_tol=1e-9)

    guidance_cw = LoiterGuidance(
        center=center, loiter_radius=radius, direction=LoiterDirection.CLOCKWISE
    )
    out_cw = guidance_cw.update(position=pos_boundary, current_heading=90.0)

    assert math.isclose(out_cw.radial_distance, radius)
    assert math.isclose(out_cw.radial_error, 0.0, abs_tol=1e-9)
    assert out_cw.phase == LoiterPhase.ORBIT_TRACKING
    expected_tangent_cw = calculate_tangent_heading(
        center, pos_boundary, LoiterDirection.CLOCKWISE
    )
    assert math.isclose(out_cw.desired_heading, expected_tangent_cw)
    assert math.isclose(out_cw.desired_heading, 90.0)  # North moving CW -> East (90 deg)


def test_clockwise_orbit_direction():
    """Clockwise orbit should produce positive desired turn direction (+1.0)."""
    center = LocalPoint(x=0.0, y=0.0)
    guidance = LoiterGuidance(
        center=center,
        loiter_radius=30.0,
        direction=LoiterDirection.CLOCKWISE,
    )
    out = guidance.update(position=LocalPoint(x=30.0, y=0.0), current_heading=180.0)

    assert out.direction == LoiterDirection.CLOCKWISE
    assert out.desired_turn_direction == 1.0


def test_counter_clockwise_orbit_direction():
    """Counter-clockwise orbit should produce negative desired turn direction (-1.0)."""
    center = LocalPoint(x=0.0, y=0.0)
    guidance = LoiterGuidance(
        center=center,
        loiter_radius=30.0,
        direction=LoiterDirection.COUNTER_CLOCKWISE,
    )
    out = guidance.update(position=LocalPoint(x=30.0, y=0.0), current_heading=0.0)

    assert out.direction == LoiterDirection.COUNTER_CLOCKWISE
    assert out.desired_turn_direction == -1.0


def test_desired_tangent_heading():
    """Verify tangent heading calculations at cardinal positions for CW and CCW."""
    center = LocalPoint(x=0.0, y=0.0)
    r = 30.0

    # North (0, 30): CW -> East (90°), CCW -> West (270°)
    pos_north = LocalPoint(x=0.0, y=r)
    assert math.isclose(
        calculate_tangent_heading(center, pos_north, LoiterDirection.CLOCKWISE), 90.0
    )
    assert math.isclose(
        calculate_tangent_heading(center, pos_north, LoiterDirection.COUNTER_CLOCKWISE), 270.0
    )

    # East (30, 0): CW -> South (180°), CCW -> North (0°)
    pos_east = LocalPoint(x=r, y=0.0)
    assert math.isclose(
        calculate_tangent_heading(center, pos_east, LoiterDirection.CLOCKWISE), 180.0
    )
    assert math.isclose(
        calculate_tangent_heading(center, pos_east, LoiterDirection.COUNTER_CLOCKWISE), 0.0
    )

    # South (0, -30): CW -> West (270°), CCW -> East (90°)
    pos_south = LocalPoint(x=0.0, y=-r)
    assert math.isclose(
        calculate_tangent_heading(center, pos_south, LoiterDirection.CLOCKWISE), 270.0
    )
    assert math.isclose(
        calculate_tangent_heading(center, pos_south, LoiterDirection.COUNTER_CLOCKWISE), 90.0
    )

    # West (-30, 0): CW -> North (0°), CCW -> South (180°)
    pos_west = LocalPoint(x=-r, y=0.0)
    assert math.isclose(
        calculate_tangent_heading(center, pos_west, LoiterDirection.CLOCKWISE), 0.0
    )
    assert math.isclose(
        calculate_tangent_heading(center, pos_west, LoiterDirection.COUNTER_CLOCKWISE), 180.0
    )


def test_orbit_capture_behavior():
    """Verify tangent-to-circle intercept heading when vehicle is outside the orbit."""
    center = LocalPoint(x=0.0, y=0.0)
    radius = 30.0
    # Vehicle at (0, -60), r = 60 m, bearing to center = 0° (North)
    # alpha = asin(R / r) = asin(30 / 60) = 30°
    pos_outside = LocalPoint(x=0.0, y=-60.0)

    guidance_cw = LoiterGuidance(
        center=center,
        loiter_radius=radius,
        direction=LoiterDirection.CLOCKWISE,
        capture_tolerance=5.0,
    )
    out_cw = guidance_cw.update(position=pos_outside, current_heading=0.0)
    assert out_cw.phase == LoiterPhase.ORBIT_CAPTURE
    # Bearing to center is 0°. CW intercept: (0° - 30°) mod 360 = 330°
    assert math.isclose(out_cw.desired_heading, 330.0)

    guidance_ccw = LoiterGuidance(
        center=center,
        loiter_radius=radius,
        direction=LoiterDirection.COUNTER_CLOCKWISE,
        capture_tolerance=5.0,
    )
    out_ccw = guidance_ccw.update(position=pos_outside, current_heading=0.0)
    assert out_ccw.phase == LoiterPhase.ORBIT_CAPTURE
    # Bearing to center is 0°. CCW intercept: (0° + 30°) mod 360 = 30°
    assert math.isclose(out_ccw.desired_heading, 30.0)


def test_capture_tolerance():
    """Verify state switching between ORBIT_TRACKING and ORBIT_CAPTURE based on tolerance."""
    center = LocalPoint(x=0.0, y=0.0)
    radius = 30.0
    tolerance = 5.0
    guidance = LoiterGuidance(
        center=center,
        loiter_radius=radius,
        capture_tolerance=tolerance,
    )

    # Distance 34 m (|34 - 30| = 4 <= 5) -> ORBIT_TRACKING
    out = guidance.update(LocalPoint(x=34.0, y=0.0), current_heading=180.0)
    assert out.phase == LoiterPhase.ORBIT_TRACKING

    # Distance 26 m (|26 - 30| = 4 <= 5) -> ORBIT_TRACKING
    out = guidance.update(LocalPoint(x=26.0, y=0.0), current_heading=180.0)
    assert out.phase == LoiterPhase.ORBIT_TRACKING

    # Exact boundary 35 m (|35 - 30| = 5 <= 5) -> ORBIT_TRACKING
    out = guidance.update(LocalPoint(x=35.0, y=0.0), current_heading=180.0)
    assert out.phase == LoiterPhase.ORBIT_TRACKING

    # Distance 36 m (|36 - 30| = 6 > 5) -> ORBIT_CAPTURE
    out = guidance.update(LocalPoint(x=36.0, y=0.0), current_heading=180.0)
    assert out.phase == LoiterPhase.ORBIT_CAPTURE

    # Distance 24 m (|24 - 30| = 6 > 5) -> ORBIT_CAPTURE
    out = guidance.update(LocalPoint(x=24.0, y=0.0), current_heading=180.0)
    assert out.phase == LoiterPhase.ORBIT_CAPTURE


def test_invalid_radius_raises():
    """loiter_radius <= 0 must raise ValueError."""
    center = LocalPoint(x=0.0, y=0.0)
    pos = LocalPoint(x=10.0, y=10.0)

    with pytest.raises(ValueError):
        calculate_radial_error(center, pos, loiter_radius=0.0)

    with pytest.raises(ValueError):
        calculate_radial_error(center, pos, loiter_radius=-10.0)

    with pytest.raises(ValueError):
        calculate_nominal_turn_rate(forward_speed=8.0, loiter_radius=0.0)

    with pytest.raises(ValueError):
        calculate_nominal_turn_rate(forward_speed=8.0, loiter_radius=-5.0)

    with pytest.raises(ValueError):
        LoiterGuidance(center=center, loiter_radius=0.0)

    with pytest.raises(ValueError):
        LoiterGuidance(center=center, loiter_radius=-30.0)


def test_invalid_capture_tolerance_raises():
    """capture_tolerance <= 0 must raise ValueError."""
    center = LocalPoint(x=0.0, y=0.0)

    with pytest.raises(ValueError):
        LoiterGuidance(center=center, capture_tolerance=0.0)

    with pytest.raises(ValueError):
        LoiterGuidance(center=center, capture_tolerance=-1.0)


def test_nominal_turn_rate():
    """Verify nominal turn rate calculation omega = V / R."""
    # V = 8.0 m/s, R = 30.0 m
    # omega = 8.0 / 30.0 rad/s = 0.266667 rad/s ≈ 15.27887 deg/s
    rate = calculate_nominal_turn_rate(forward_speed=8.0, loiter_radius=30.0)
    expected_deg_s = math.degrees(8.0 / 30.0)
    assert math.isclose(rate, expected_deg_s)
    assert 15.2 < rate < 15.3

    # Invalid forward speed
    with pytest.raises(ValueError):
        calculate_nominal_turn_rate(forward_speed=0.0, loiter_radius=30.0)
    with pytest.raises(ValueError):
        calculate_nominal_turn_rate(forward_speed=-8.0, loiter_radius=30.0)


def test_invalid_speeds_and_limits_raise():
    """forward_speed <= 0 and max_turn_rate <= 0 must raise ValueError."""
    center = LocalPoint(x=0.0, y=0.0)

    with pytest.raises(ValueError):
        LoiterGuidance(center=center, forward_speed=0.0)

    with pytest.raises(ValueError):
        LoiterGuidance(center=center, max_turn_rate=0.0)


def test_wind_aware_loiter_tangent_heading():
    center = LocalPoint(0.0, 0.0)
    radius = 30.0
    guidance = LoiterGuidance(center=center, loiter_radius=radius, forward_speed=8.0)
    
    guidance.set_wind(0.0, 0.0)
    pos = LocalPoint(30.0, 0.0)
    out = guidance.update(position=pos, current_heading=180.0)
    assert pytest.approx(out.desired_heading, abs=0.1) == 180.0
    
    guidance.set_wind(3.0, 0.0)
    out_wind = guidance.update(position=pos, current_heading=180.0)
    assert pytest.approx(out_wind.desired_heading, abs=0.5) == 202.0
