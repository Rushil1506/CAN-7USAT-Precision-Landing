from navigation.coordinates import LocalPoint
from navigation.waypoint_manager import WaypointManager


def test_waypoint_progression():
    manager = WaypointManager(
        [
            LocalPoint(10.0, 0.0),
            LocalPoint(20.0, 0.0),
            LocalPoint(30.0, 0.0),
        ],
        acceptance_radius=2.0,
    )

    assert manager.waypoint_number == 1
    assert not manager.is_complete

    assert not manager.update(LocalPoint(5.0, 0.0))
    assert manager.waypoint_number == 1

    assert manager.update(LocalPoint(9.5, 0.0))
    assert manager.waypoint_number == 2

    assert manager.update(LocalPoint(19.5, 0.0))
    assert manager.waypoint_number == 3

    assert manager.update(LocalPoint(29.5, 0.0))
    assert manager.is_complete


def test_large_step_can_skip_reached_waypoints():
    manager = WaypointManager(
        [
            LocalPoint(10.0, 0.0),
            LocalPoint(20.0, 0.0),
            LocalPoint(30.0, 0.0),
        ],
        acceptance_radius=100.0,
    )

    assert manager.update(LocalPoint(0.0, 0.0))
    assert manager.is_complete
