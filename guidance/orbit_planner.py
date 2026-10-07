import math
from dataclasses import dataclass
from typing import Optional, Tuple

from navigation.coordinates import LocalPoint
from navigation.distance import distance
from guidance.loiter_guidance import LoiterDirection

@dataclass(frozen=True)
class OrbitPlan:
    feasible: bool
    orbit_center: LocalPoint
    orbit_direction: LoiterDirection
    breakout_point: LocalPoint
    breakout_altitude: float
    final_ground_speed: float
    final_approach_time: float
    d_selected: float
    
def plan_terminal_orbit(
    position: LocalPoint,
    target: LocalPoint,
    current_altitude: float,
    wind_x: float,
    wind_y: float,
    forward_speed: float,
    descent_rate: float,
    orbit_radius: float,
    min_final_time: float = 5.0,
    min_final_distance: float = 30.0,
    breakout_altitude_margin: float = 2.0,
) -> OrbitPlan:
    """
    Plan a wind-aware terminal orbit and final approach path.
    """
    w_mag = math.hypot(wind_x, wind_y)
    
    # Zero-wind fallback
    if w_mag < 1e-3:
        return OrbitPlan(
            feasible=True,
            orbit_center=LocalPoint(target.x, target.y),
            orbit_direction=LoiterDirection.CLOCKWISE,
            breakout_point=LocalPoint(target.x, target.y),
            breakout_altitude=breakout_altitude_margin,
            final_ground_speed=forward_speed,
            final_approach_time=0.0,
            d_selected=0.0
        )
        
    if w_mag >= forward_speed:
        # Infeasible wind
        return OrbitPlan(
            feasible=False,
            orbit_center=LocalPoint(target.x, target.y),
            orbit_direction=LoiterDirection.CLOCKWISE,
            breakout_point=LocalPoint(target.x, target.y),
            breakout_altitude=0.0,
            final_ground_speed=0.0,
            final_approach_time=0.0,
            d_selected=0.0
        )
        
    w_hat_x = wind_x / w_mag
    w_hat_y = wind_y / w_mag
    
    # Perpendicular vector (90 deg right of wind vector)
    # Wind vector is (w_hat_x, w_hat_y)
    # Right of wind is (w_hat_y, -w_hat_x)
    w_perp_x = w_hat_y
    w_perp_y = -w_hat_x
    
    v_final_ground = forward_speed - w_mag
    
    # D must be large enough to allow min_final_time, and at least min_final_distance
    d_min = max(min_final_distance, min_final_time * v_final_ground)
    
    best_cost = float('inf')
    best_plan = None
    
    # Sweep D candidates
    for d_cand in range(int(math.ceil(d_min)), int(math.ceil(d_min)) + 150, 10):
        d = float(d_cand)
        t_final = d / v_final_ground
        h_breakout_req = t_final * descent_rate + breakout_altitude_margin
        
        # If required altitude exceeds what we currently have (with a small buffer), skip
        if h_breakout_req > current_altitude - 5.0:
            continue
            
        p_breakout = LocalPoint(target.x + d * w_hat_x, target.y + d * w_hat_y)
        
        # Candidate 1: CW
        c_cw = LocalPoint(p_breakout.x - orbit_radius * w_perp_x, p_breakout.y - orbit_radius * w_perp_y)
        dist_cw = distance(position, c_cw)
        cost_cw = d + dist_cw
        
        if cost_cw < best_cost:
            best_cost = cost_cw
            best_plan = OrbitPlan(
                feasible=True,
                orbit_center=c_cw,
                orbit_direction=LoiterDirection.CLOCKWISE,
                breakout_point=p_breakout,
                breakout_altitude=h_breakout_req,
                final_ground_speed=v_final_ground,
                final_approach_time=t_final,
                d_selected=d
            )
            
        # Candidate 2: CCW
        c_ccw = LocalPoint(p_breakout.x + orbit_radius * w_perp_x, p_breakout.y + orbit_radius * w_perp_y)
        dist_ccw = distance(position, c_ccw)
        cost_ccw = d + dist_ccw
        
        if cost_ccw < best_cost:
            best_cost = cost_ccw
            best_plan = OrbitPlan(
                feasible=True,
                orbit_center=c_ccw,
                orbit_direction=LoiterDirection.COUNTER_CLOCKWISE,
                breakout_point=p_breakout,
                breakout_altitude=h_breakout_req,
                final_ground_speed=v_final_ground,
                final_approach_time=t_final,
                d_selected=d
            )
            
    if best_plan is None:
        return OrbitPlan(
            feasible=False,
            orbit_center=LocalPoint(target.x, target.y),
            orbit_direction=LoiterDirection.CLOCKWISE,
            breakout_point=LocalPoint(target.x, target.y),
            breakout_altitude=0.0,
            final_ground_speed=0.0,
            final_approach_time=0.0,
            d_selected=0.0
        )
        
    return best_plan
