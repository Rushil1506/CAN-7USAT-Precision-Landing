import math
from typing import Callable, Tuple
from environment.wind import WindField
from guidance.wind_compensation import calculate_wind_corrected_heading

def integrate_altitude_scalar(
    func: Callable[[float], float],
    h_low: float,
    h_high: float,
    dh: float = 1.0
) -> float:
    """
    Deterministically integrate a scalar function over an altitude interval [h_low, h_high]
    using the trapezoidal rule.
    
    Args:
        func: A callable that takes altitude as input and returns a scalar.
        h_low: Lower altitude bound (must be <= h_high).
        h_high: Upper altitude bound.
        dh: Resolution of the integration step.
        
    Returns:
        The integrated value: \int_{h_low}^{h_high} func(h) dh
    """
    if h_low > h_high:
        raise ValueError("h_low must be less than or equal to h_high")
        
    delta_h = h_high - h_low
    if delta_h < 1e-6:
        return 0.0
        
    num_steps = int(math.ceil(delta_h / dh))
    actual_dh = delta_h / num_steps
    
    integral = 0.0
    v_prev = func(h_high)
    
    for i in range(1, num_steps + 1):
        h_curr = h_high - i * actual_dh
        v_curr = func(h_curr)
        integral += 0.5 * (v_prev + v_curr) * actual_dh
        v_prev = v_curr
        
    return integral

def effective_wind(wind_field: WindField, h_low: float, h_high: float, dh: float = 1.0) -> Tuple[float, float]:
    """
    Calculate the effective planning wind over an altitude interval.
    This represents the average wind vector (time-equivalent weighting for constant descent).
    
    W_eff = (1 / (h_high - h_low)) * \int_{h_low}^{h_high} W(h) dh
    """
    if h_low > h_high:
        raise ValueError("h_low must be less than or equal to h_high")
        
    if (h_high - h_low) < 1e-6:
        return wind_field.wind_at_altitude(h_high)
        
    def wx_func(h: float) -> float:
        return wind_field.wind_at_altitude(h)[0]
        
    def wy_func(h: float) -> float:
        return wind_field.wind_at_altitude(h)[1]
        
    wx_eff = integrate_altitude_scalar(wx_func, h_low, h_high, dh) / (h_high - h_low)
    wy_eff = integrate_altitude_scalar(wy_func, h_low, h_high, dh) / (h_high - h_low)
    
    return wx_eff, wy_eff

from dataclasses import dataclass

@dataclass(frozen=True)
class IntervalFeasibilityResult:
    feasible: bool
    limiting_altitude: float
    minimum_projected_ground_speed: float
    maximum_crosswind_component: float

def evaluate_interval_feasibility(
    wind_field: WindField,
    h_low: float,
    h_high: float,
    desired_course_deg: float,
    airspeed: float,
    dh: float = 1.0
) -> IntervalFeasibilityResult:
    """
    Determine whether a wind profile is kinematically feasible over an altitude interval
    for a selected course.
    """
    if h_low > h_high:
        raise ValueError("h_low must be less than or equal to h_high")
        
    delta_h = h_high - h_low
    num_steps = int(math.ceil(delta_h / dh)) if delta_h > 1e-6 else 1
    actual_dh = delta_h / num_steps if num_steps > 0 else 0.0
    
    feasible = True
    min_vg = float('inf')
    max_cross = 0.0
    limit_h = h_high
    
    for i in range(num_steps + 1):
        h_curr = h_high - i * actual_dh
        wx, wy = wind_field.wind_at_altitude(h_curr)
        comp = calculate_wind_corrected_heading(desired_course_deg, airspeed, wx, wy)
        
        vg = comp.ground_speed_along_track
        cross = abs(comp.crosswind_component)
        
        if cross > max_cross:
            max_cross = cross
            
        if vg < min_vg:
            min_vg = vg
            limit_h = h_curr
            
        if not comp.feasible or vg <= 0:
            feasible = False
            
    return IntervalFeasibilityResult(
        feasible=feasible,
        limiting_altitude=limit_h,
        minimum_projected_ground_speed=min_vg if min_vg != float('inf') else 0.0,
        maximum_crosswind_component=max_cross
    )
