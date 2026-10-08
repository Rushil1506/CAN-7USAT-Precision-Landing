from typing import Protocol, Tuple
import math

class WindField(Protocol):
    """Protocol for a wind field.
    
    Coordinate convention:
    +X = East
    +Y = North
    
    Wind vector represents the direction the air is moving TOWARD.
    """
    def wind_at_altitude(self, altitude: float) -> Tuple[float, float]:
        """Get the (wind_x, wind_y) vector at the given altitude [m/s] (assumes time=0 or steady)."""
        ...
        
    def wind_at(self, altitude: float, time: float) -> Tuple[float, float]:
        """Get the (wind_x, wind_y) vector at given altitude and time."""
        ...

class ConstantWindField:
    """A WindField that returns the same wind vector at all altitudes and times."""
    def __init__(self, wind_x: float, wind_y: float):
        self.wind_x = wind_x
        self.wind_y = wind_y
        
    def wind_at_altitude(self, altitude: float) -> Tuple[float, float]:
        return (self.wind_x, self.wind_y)
        
    def wind_at(self, altitude: float, time: float) -> Tuple[float, float]:
        return (self.wind_x, self.wind_y)

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
                
        return self.points[-1][1]
        
    def wind_at(self, altitude: float, time: float) -> Tuple[float, float]:
        return self.wind_at_altitude(altitude)


class CosineGust:
    """Deterministic 1-Cosine Gust Model."""
    def __init__(self, gust_x: float, gust_y: float, t_start: float, duration: float):
        if not math.isfinite(gust_x) or not math.isfinite(gust_y):
            raise ValueError("Gust components must be finite")
        if not math.isfinite(t_start):
            raise ValueError("t_start must be finite")
        if not math.isfinite(duration) or duration <= 0.0:
            raise ValueError("duration must be finite and positive")
            
        self.gust_x = gust_x
        self.gust_y = gust_y
        self.t_start = t_start
        self.duration = duration
        
    def evaluate(self, time: float) -> Tuple[float, float]:
        if not math.isfinite(time):
            raise ValueError("Time must be finite")
            
        if time < self.t_start or time > self.t_start + self.duration:
            return (0.0, 0.0)
            
        scale = 0.5 * (1.0 - math.cos(2.0 * math.pi * (time - self.t_start) / self.duration))
        return (self.gust_x * scale, self.gust_y * scale)


class ComposedWindField:
    """WindField that composes a base altitude wind with one or more time-varying gusts."""
    def __init__(self, base_wind: WindField, *gusts: CosineGust):
        self.base_wind = base_wind
        self.gusts = gusts
        
    def wind_at_altitude(self, altitude: float) -> Tuple[float, float]:
        """Provides steady state wind (time=0)."""
        return self.wind_at(altitude, 0.0)
        
    def wind_at(self, altitude: float, time: float) -> Tuple[float, float]:
        wx, wy = self.base_wind.wind_at_altitude(altitude)
        for gust in self.gusts:
            gx, gy = gust.evaluate(time)
            wx += gx
            wy += gy
        return (wx, wy)

