import math

from navigation.coordinates import (
    GPSPoint,
    LocalPoint,
    LocalCoordinateSystem
)

from navigation.distance import distance
from navigation.bearing import bearing
from navigation.waypoints import generate_waypoints


def test_distance():

    a = LocalPoint(0, 0)
    b = LocalPoint(3, 4)

    assert math.isclose(
        distance(a, b),
        5.0
    )


def test_bearing_north():

    a = LocalPoint(0, 0)
    b = LocalPoint(0, 100)

    assert math.isclose(
        bearing(a, b),
        0.0
    )


def test_bearing_east():

    a = LocalPoint(0, 0)
    b = LocalPoint(100, 0)

    assert math.isclose(
        bearing(a, b),
        90.0
    )


def test_bearing_south():

    a = LocalPoint(0, 0)
    b = LocalPoint(0, -100)

    assert math.isclose(
        bearing(a, b),
        180.0
    )


def test_bearing_west():

    a = LocalPoint(0, 0)
    b = LocalPoint(-100, 0)

    assert math.isclose(
        bearing(a, b),
        270.0
    )


def test_waypoints():

    start = LocalPoint(0, 0)
    target = LocalPoint(100, 0)

    waypoints = generate_waypoints(
        start,
        target,
        10
    )

    assert len(waypoints) == 10

    assert math.isclose(
        waypoints[0].x,
        10.0
    )

    assert math.isclose(
        waypoints[-1].x,
        100.0
    )


def test_gps_conversion():

    origin = GPSPoint(
        latitude=13.3521,
        longitude=74.7920
    )

    coordinate_system = LocalCoordinateSystem(origin)

    result = coordinate_system.gps_to_local(origin)

    assert math.isclose(result.x, 0.0, abs_tol=1e-6)
    assert math.isclose(result.y, 0.0, abs_tol=1e-6)
from navigation.heading import heading_error


def test_heading_error_normal():

    assert heading_error(0, 30) == 30


def test_heading_error_left():

    assert heading_error(30, 0) == -30


def test_heading_error_wrap_right():

    assert heading_error(350, 10) == 20


def test_heading_error_wrap_left():

    assert heading_error(10, 350) == -20


def test_heading_error_east():

    assert heading_error(180, 270) == 90


def test_heading_error_west():

    assert heading_error(270, 180) == -90