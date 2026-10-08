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

    def reset(self) -> None:
        """Explicitly reset any stochastic or time-varying state to reproducible initial conditions."""
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

    def reset(self) -> None:
        pass

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

    def reset(self) -> None:
        pass

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

    def reset(self) -> None:
        pass

class ComposedWindField:
    """WindField that composes a base altitude wind with one or more time-varying modifiers (gusts, turbulence)."""
    def __init__(self, base_wind: WindField, *modifiers):
        self.base_wind = base_wind
        self.modifiers = modifiers
        
    def wind_at_altitude(self, altitude: float) -> Tuple[float, float]:
        """Provides steady state wind (time=0)."""
        return self.wind_at(altitude, 0.0)
        
    def wind_at(self, altitude: float, time: float) -> Tuple[float, float]:
        wx, wy = self.base_wind.wind_at_altitude(altitude)
        for modifier in self.modifiers:
            gx, gy = modifier.evaluate(time)
            wx += gx
            wy += gy
        return (wx, wy)

    def reset(self) -> None:
        self.base_wind.reset()
        for modifier in self.modifiers:
            if hasattr(modifier, 'reset'):
                modifier.reset()

import random

class StochasticWind:
    """Synthetic stochastic wind disturbance for controller robustness characterization.
    
    Implements a bounded Ornstein-Uhlenbeck process.
    """
    def __init__(self, std_dev: float, correlation_time: float, seed: int = 42):
        if std_dev < 0.0 or not math.isfinite(std_dev):
            raise ValueError("std_dev must be non-negative and finite")
        if correlation_time <= 0.0 or not math.isfinite(correlation_time):
            raise ValueError("correlation_time must be positive and finite")
            
        self.std_dev = std_dev
        self.correlation_time = correlation_time
        self.seed = seed
        self.rng = random.Random(seed)
        
        self.last_time = 0.0
        self.current_wx = 0.0
        self.current_wy = 0.0
        
    def reset(self) -> None:
        """Explicitly reset the stochastic process to its initial seeded state."""
        self.rng = random.Random(self.seed)
        self.last_time = 0.0
        self.current_wx = 0.0
        self.current_wy = 0.0
        
    def evaluate(self, time: float) -> Tuple[float, float]:
        if not math.isfinite(time):
            raise ValueError("Time must be finite")
            
        if time <= self.last_time:
            # Snapshot query: return the current state without advancing or resetting
            return (self.current_wx, self.current_wy)
            
        dt = time - self.last_time
        self.last_time = time
        
        theta = 1.0 / self.correlation_time
        sigma_noise = self.std_dev * math.sqrt(2.0 * theta)
        
        noise_x = self.rng.gauss(0.0, 1.0)
        self.current_wx += -theta * self.current_wx * dt + sigma_noise * math.sqrt(dt) * noise_x
        
        noise_y = self.rng.gauss(0.0, 1.0)
        self.current_wy += -theta * self.current_wy * dt + sigma_noise * math.sqrt(dt) * noise_y
        
        # Bounded
        limit = 3.0 * self.std_dev
        self.current_wx = max(-limit, min(limit, self.current_wx))
        self.current_wy = max(-limit, min(limit, self.current_wy))
        
        return (self.current_wx, self.current_wy)


