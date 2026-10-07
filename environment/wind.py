from typing import Protocol, Tuple

class WindField(Protocol):
    """Protocol for an altitude-dependent wind field.
    
    Coordinate convention:
    +X = East
    +Y = North
    
    Wind vector represents the direction the air is moving TOWARD.
    """
    def wind_at_altitude(self, altitude: float) -> Tuple[float, float]:
        """Get the (wind_x, wind_y) vector at the given altitude [m/s]."""
        ...

class ConstantWindField:
    """A WindField that returns the same wind vector at all altitudes."""
    def __init__(self, wind_x: float, wind_y: float):
        self.wind_x = wind_x
        self.wind_y = wind_y
        
    def wind_at_altitude(self, altitude: float) -> Tuple[float, float]:
        return (self.wind_x, self.wind_y)

import math
from typing import List

class PiecewiseWindField:
    """A WindField that linearly interpolates between defined altitude points.
    
    Boundary behavior:
    - h < min(altitudes) -> clamps to minimum altitude wind
    - h > max(altitudes) -> clamps to maximum altitude wind
    """
    def __init__(self, altitude_wind_points: List[Tuple[float, Tuple[float, float]]]):
        if not altitude_wind_points:
            raise ValueError("altitude_wind_points must not be empty")
        if len(altitude_wind_points) < 2:
            raise ValueError("PiecewiseWindField requires at least 2 points")
            
        # Validate ordering and duplicates
        for i in range(1, len(altitude_wind_points)):
            if altitude_wind_points[i][0] <= altitude_wind_points[i-1][0]:
                raise ValueError("Altitudes must be strictly increasing without duplicates")
                
        # Validate finite values
        for alt, (wx, wy) in altitude_wind_points:
            if not math.isfinite(alt) or not math.isfinite(wx) or not math.isfinite(wy):
                raise ValueError("Altitudes and wind components must be finite")
                
        self.points = list(altitude_wind_points)
        
    def wind_at_altitude(self, altitude: float) -> Tuple[float, float]:
        if not math.isfinite(altitude):
            raise ValueError("Altitude must be finite")
            
        # Clamping
        if altitude <= self.points[0][0]:
            return self.points[0][1]
        if altitude >= self.points[-1][0]:
            return self.points[-1][1]
            
        # Find the segment
        for i in range(len(self.points) - 1):
            h0, (wx0, wy0) = self.points[i]
            h1, (wx1, wy1) = self.points[i+1]
            
            if h0 <= altitude <= h1:
                t = (altitude - h0) / (h1 - h0)
                wx = wx0 + (wx1 - wx0) * t
                wy = wy0 + (wy1 - wy0) * t
                return (wx, wy)
                
        # Should never reach here due to clamping
        return self.points[-1][1]
