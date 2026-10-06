import math
from dataclasses import dataclass


# Mean Earth radius in metres
EARTH_RADIUS = 6_371_000.0


@dataclass
class GPSPoint:
    latitude: float
    longitude: float


@dataclass
class LocalPoint:
    x: float  # East, metres
    y: float  # North, metres


class LocalCoordinateSystem:
    """
    Converts GPS latitude/longitude to a local Cartesian coordinate system.

    Origin:
        (0, 0) = deployment location

    Axes:
        +X = East
        +Y = North
    """

    def __init__(self, origin: GPSPoint):
        self.origin = origin

        self.lat0_rad = math.radians(origin.latitude)
        self.lon0_rad = math.radians(origin.longitude)

    def gps_to_local(self, point: GPSPoint) -> LocalPoint:
        """
        Convert GPS coordinates to local X/Y coordinates.

        Suitable for the relatively small distances involved
        in a CanSat precision-landing scenario.
        """

        lat_rad = math.radians(point.latitude)
        lon_rad = math.radians(point.longitude)

        delta_lat = lat_rad - self.lat0_rad
        delta_lon = lon_rad - self.lon0_rad

        # North/South displacement
        y = EARTH_RADIUS * delta_lat

        # East/West displacement
        x = EARTH_RADIUS * math.cos(self.lat0_rad) * delta_lon

        return LocalPoint(x=x, y=y)

    def local_to_gps(self, point: LocalPoint) -> GPSPoint:
        """
        Convert local X/Y coordinates back to GPS latitude/longitude.
        """

        delta_lat = point.y / EARTH_RADIUS

        delta_lon = point.x / (
            EARTH_RADIUS * math.cos(self.lat0_rad)
        )

        latitude = math.degrees(self.lat0_rad + delta_lat)
        longitude = math.degrees(self.lon0_rad + delta_lon)

        return GPSPoint(
            latitude=latitude,
            longitude=longitude
        )