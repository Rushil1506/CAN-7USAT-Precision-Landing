import math
import pytest

from flight.parafoil import ParafoilConfig
from guidance.breakout_guidance import (
    BreakoutGeometry,
    calculate_breakout_geometry,
    calculate_breakout_geometry_from_config,
    calculate_turn_endpoint,
)
from navigation.coordinates import LocalPoint


def test_nominal_breakout_geometry():
    """
    Test nominal scenario:
        CanSat at (0, -30) heading 270° (West) in a 30 m CW loiter around (0, 0).
        Target is at (0, 0) (North, bearing 0°).
        Heading change is a 90° right turn (CW).
    """
    position = LocalPoint(x=0.0, y=-30.0)
    heading = 270.0
    target = LocalPoint(x=0.0, y=0.0)

    geom = calculate_breakout_geometry(
        position=position,
        heading=heading,
        target=target,
        forward_speed=8.0,
        descent_rate=2.5,
        max_turn_rate=30.0,
        breakout_altitude_margin=2.0,
    )

    # Basic verification
    assert isinstance(geom, BreakoutGeometry)
    assert geom.radial_distance == pytest.approx(30.0)
    assert geom.target_bearing == pytest.approx(0.0)
    assert geom.heading_error == pytest.approx(90.0)
    assert geom.turn_angle == pytest.approx(90.0)
    assert geom.turn_direction == 1.0

    # Turn geometry: omega = 30 deg/s = pi/6 rad/s
    # R_turn = 8.0 / (pi/6) = 48/pi ≈ 15.27887 m
    expected_r_turn = 8.0 / math.radians(30.0)
    assert geom.turn_radius == pytest.approx(expected_r_turn, rel=1e-5)
    assert geom.turn_duration == pytest.approx(3.0, rel=1e-5)
    assert geom.turn_arc_length == pytest.approx(24.0, rel=1e-5)

    # Endpoint: P_1 = (-R_turn, -30 + R_turn) = (-15.27887, -14.72113)
    assert geom.turn_endpoint.x == pytest.approx(-expected_r_turn, rel=1e-5)
    assert geom.turn_endpoint.y == pytest.approx(-30.0 + expected_r_turn, rel=1e-5)

    # Straight distance: sqrt((-15.27887)^2 + (-14.72113)^2) ≈ 21.21727 m
    expected_l_straight = math.hypot(-expected_r_turn, -30.0 + expected_r_turn)
    assert geom.straight_distance == pytest.approx(expected_l_straight, rel=1e-5)

    # Required path length: 24.0 + 21.21727 ≈ 45.21727 m
    expected_l_required = 24.0 + expected_l_straight
    assert geom.required_path_length == pytest.approx(expected_l_required, rel=1e-5)

    # Glide ratio and altitudes: G = 8.0 / 2.5 = 3.2
    assert geom.glide_ratio == pytest.approx(3.2, rel=1e-5)
    expected_h_min = expected_l_required / 3.2
    assert geom.minimum_required_altitude == pytest.approx(expected_h_min, rel=1e-5)
    assert geom.breakout_altitude == pytest.approx(expected_h_min + 2.0, rel=1e-5)


def test_cardinal_rotational_symmetry():
    """
    Test that the path length and required altitude are invariant to orientation
    at the 4 cardinal positions on a 30 m circle with tangent headings.
    """
    target = LocalPoint(x=0.0, y=0.0)
    cardinal_cases = [
        # (position, heading for CW loiter)
        (LocalPoint(x=0.0, y=-30.0), 270.0),  # South, heading West
        (LocalPoint(x=30.0, y=0.0), 180.0),   # East, heading South
        (LocalPoint(x=0.0, y=30.0), 90.0),    # North, heading East
        (LocalPoint(x=-30.0, y=0.0), 0.0),    # West, heading North
    ]

    results = [
        calculate_breakout_geometry(pos, hdg, target, 8.0, 2.5, 30.0, 2.0)
        for pos, hdg in cardinal_cases
    ]

    for res in results:
        assert res.turn_angle == pytest.approx(90.0)
        assert res.turn_arc_length == pytest.approx(24.0, rel=1e-4)
        assert res.straight_distance == pytest.approx(21.21727, rel=1e-4)
        assert res.required_path_length == pytest.approx(45.21727, rel=1e-4)
        assert res.minimum_required_altitude == pytest.approx(14.13040, rel=1e-4)
        assert res.breakout_altitude == pytest.approx(16.13040, rel=1e-4)


def test_turn_direction_symmetry_ccw():
    """
    Test CCW loiter point:
        Position at (0, -30), heading 90° (East).
        Target at (0, 0) (North, bearing 0°).
        Heading change is a 90° left turn (CCW).
    """
    position = LocalPoint(x=0.0, y=-30.0)
    heading = 90.0
    target = LocalPoint(x=0.0, y=0.0)

    geom = calculate_breakout_geometry(
        position=position,
        heading=heading,
        target=target,
        forward_speed=8.0,
        descent_rate=2.5,
        max_turn_rate=30.0,
        breakout_altitude_margin=2.0,
    )

    assert geom.heading_error == pytest.approx(-90.0)
    assert geom.turn_angle == pytest.approx(90.0)
    assert geom.turn_direction == -1.0
    # Endpoint should mirror X coordinate to +15.27887 m
    expected_r_turn = 8.0 / math.radians(30.0)
    assert geom.turn_endpoint.x == pytest.approx(expected_r_turn, rel=1e-5)
    assert geom.turn_endpoint.y == pytest.approx(-30.0 + expected_r_turn, rel=1e-5)
    assert geom.straight_distance == pytest.approx(21.21727, rel=1e-4)
    assert geom.required_path_length == pytest.approx(45.21727, rel=1e-4)
    assert geom.minimum_required_altitude == pytest.approx(14.13040, rel=1e-4)


def test_already_aligned_heading():
    """
    When the vehicle is already pointing directly at the target,
    turn angle and arc length should be 0, and straight distance equals radial distance.
    """
    position = LocalPoint(x=0.0, y=-30.0)
    heading = 0.0  # North, directly toward (0, 0)
    target = LocalPoint(x=0.0, y=0.0)

    geom = calculate_breakout_geometry(
        position=position,
        heading=heading,
        target=target,
        forward_speed=8.0,
        descent_rate=2.5,
        max_turn_rate=30.0,
        breakout_altitude_margin=2.0,
    )

    assert geom.heading_error == pytest.approx(0.0)
    assert geom.turn_angle == pytest.approx(0.0)
    assert geom.turn_duration == pytest.approx(0.0)
    assert geom.turn_arc_length == pytest.approx(0.0)
    assert geom.turn_endpoint.x == pytest.approx(0.0)
    assert geom.turn_endpoint.y == pytest.approx(-30.0)
    assert geom.straight_distance == pytest.approx(30.0)
    assert geom.required_path_length == pytest.approx(30.0)
    assert geom.minimum_required_altitude == pytest.approx(30.0 / 3.2)
    assert geom.breakout_altitude == pytest.approx(30.0 / 3.2 + 2.0)


def test_opposite_heading_180_turn():
    """
    When pointing directly away from the target, a 180° turn is required.
    """
    position = LocalPoint(x=0.0, y=-30.0)
    heading = 180.0  # South, directly away from (0, 0)
    target = LocalPoint(x=0.0, y=0.0)

    geom = calculate_breakout_geometry(
        position=position,
        heading=heading,
        target=target,
        forward_speed=8.0,
        descent_rate=2.5,
        max_turn_rate=30.0,
        breakout_altitude_margin=2.0,
    )

    assert geom.turn_angle == pytest.approx(180.0)
    expected_r_turn = 8.0 / math.radians(30.0)
    expected_l_turn = expected_r_turn * math.pi
    assert geom.turn_arc_length == pytest.approx(expected_l_turn, rel=1e-5)
    # A 180° turn shifts position by 2 * R_turn in X (left turn, so +2 * R_turn)
    assert geom.turn_endpoint.x == pytest.approx(2.0 * expected_r_turn, rel=1e-5)
    assert geom.turn_endpoint.y == pytest.approx(-30.0, rel=1e-5)
    expected_straight = math.hypot(2.0 * expected_r_turn, 30.0)
    assert geom.straight_distance == pytest.approx(expected_straight, rel=1e-5)
    assert geom.minimum_required_altitude == pytest.approx(
        (expected_l_turn + expected_straight) / 3.2, rel=1e-5
    )


def test_monotonicity_with_radial_distance():
    """
    As radial distance increases at tangent headings, the required path length
    and minimum altitude must increase strictly monotonically.
    """
    target = LocalPoint(x=0.0, y=0.0)
    radii = [20.0, 25.0, 30.0, 35.0, 40.0, 50.0]
    geoms = [
        calculate_breakout_geometry(
            position=LocalPoint(x=0.0, y=-r),
            heading=270.0,
            target=target,
        )
        for r in radii
    ]

    for i in range(len(geoms) - 1):
        assert geoms[i].straight_distance < geoms[i + 1].straight_distance
        assert geoms[i].required_path_length < geoms[i + 1].required_path_length
        assert geoms[i].minimum_required_altitude < geoms[i + 1].minimum_required_altitude


def test_non_zero_target_origin():
    """
    Calculations should be translation-invariant.
    """
    target = LocalPoint(x=250.0, y=-150.0)
    position = LocalPoint(x=250.0, y=-180.0)  # 30 m South of target
    heading = 270.0

    geom = calculate_breakout_geometry(
        position=position,
        heading=heading,
        target=target,
    )

    expected_r_turn = 8.0 / math.radians(30.0)
    assert geom.radial_distance == pytest.approx(30.0)
    assert geom.target_bearing == pytest.approx(0.0)
    assert geom.turn_endpoint.x == pytest.approx(250.0 - expected_r_turn, rel=1e-5)
    assert geom.turn_endpoint.y == pytest.approx(-180.0 + expected_r_turn, rel=1e-5)
    assert geom.straight_distance == pytest.approx(21.21727, rel=1e-4)
    assert geom.minimum_required_altitude == pytest.approx(14.13040, rel=1e-4)


def test_already_at_target():
    """
    When position coincides with target, required distance and minimum altitude are 0.
    """
    target = LocalPoint(x=100.0, y=200.0)
    position = LocalPoint(x=100.0, y=200.0)

    geom = calculate_breakout_geometry(
        position=position,
        heading=45.0,
        target=target,
        breakout_altitude_margin=3.5,
    )

    assert geom.radial_distance == 0.0
    assert geom.turn_angle == 0.0
    assert geom.turn_arc_length == 0.0
    assert geom.straight_distance == 0.0
    assert geom.required_path_length == 0.0
    assert geom.minimum_required_altitude == 0.0
    assert geom.breakout_altitude == 3.5


def test_margin_sensitivity():
    """
    The breakout_altitude should strictly equal minimum_required_altitude + margin.
    """
    target = LocalPoint(x=0.0, y=0.0)
    position = LocalPoint(x=0.0, y=-30.0)

    geom_m0 = calculate_breakout_geometry(position, 270.0, target, breakout_altitude_margin=0.0)
    geom_m5 = calculate_breakout_geometry(position, 270.0, target, breakout_altitude_margin=5.0)

    assert geom_m0.breakout_altitude == pytest.approx(geom_m0.minimum_required_altitude)
    assert geom_m5.breakout_altitude == pytest.approx(geom_m5.minimum_required_altitude + 5.0)


def test_config_convenience_wrapper():
    """
    calculate_breakout_geometry_from_config should extract forward_speed, descent_rate,
    and max_turn_rate correctly from ParafoilConfig.
    """
    config = ParafoilConfig(
        forward_speed=8.0,
        descent_rate=2.5,
        max_turn_rate=30.0,
    )
    target = LocalPoint(x=0.0, y=0.0)
    position = LocalPoint(x=0.0, y=-30.0)

    geom_config = calculate_breakout_geometry_from_config(
        position=position,
        heading=270.0,
        target=target,
        config=config,
        breakout_altitude_margin=2.0,
    )
    geom_direct = calculate_breakout_geometry(
        position=position,
        heading=270.0,
        target=target,
        forward_speed=8.0,
        descent_rate=2.5,
        max_turn_rate=30.0,
        breakout_altitude_margin=2.0,
    )

    assert geom_config == geom_direct


def test_calculate_turn_endpoint():
    """
    Test turn endpoint calculation directly.
    """
    # 0 degree turn produces no displacement
    p0 = LocalPoint(x=10.0, y=20.0)
    p1 = calculate_turn_endpoint(p0, 45.0, 0.0, 1.0, 15.0)
    assert p1.x == pytest.approx(10.0)
    assert p1.y == pytest.approx(20.0)

    # 0 turn radius produces no displacement
    p2 = calculate_turn_endpoint(p0, 45.0, 90.0, 1.0, 0.0)
    assert p2.x == pytest.approx(10.0)
    assert p2.y == pytest.approx(20.0)


def test_invalid_parameter_exceptions():
    """
    Verify ValueError is raised on non-positive speeds/rates or negative margins.
    """
    pos = LocalPoint(x=0.0, y=-30.0)
    tgt = LocalPoint(x=0.0, y=0.0)

    with pytest.raises(ValueError, match="forward_speed"):
        calculate_breakout_geometry(pos, 0.0, tgt, forward_speed=0.0)

    with pytest.raises(ValueError, match="forward_speed"):
        calculate_breakout_geometry(pos, 0.0, tgt, forward_speed=-5.0)

    with pytest.raises(ValueError, match="descent_rate"):
        calculate_breakout_geometry(pos, 0.0, tgt, descent_rate=0.0)

    with pytest.raises(ValueError, match="descent_rate"):
        calculate_breakout_geometry(pos, 0.0, tgt, descent_rate=-1.0)

    with pytest.raises(ValueError, match="max_turn_rate"):
        calculate_breakout_geometry(pos, 0.0, tgt, max_turn_rate=0.0)

    with pytest.raises(ValueError, match="breakout_altitude_margin"):
        calculate_breakout_geometry(pos, 0.0, tgt, breakout_altitude_margin=-0.5)

    with pytest.raises(ValueError, match="turn_radius"):
        calculate_turn_endpoint(pos, 0.0, 90.0, 1.0, turn_radius=-1.0)

    with pytest.raises(ValueError, match="turn_angle"):
        calculate_turn_endpoint(pos, 0.0, turn_angle=-10.0, turn_direction=1.0, turn_radius=15.0)
