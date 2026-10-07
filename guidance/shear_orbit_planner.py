import math
from dataclasses import dataclass
from typing import Optional, Tuple

from navigation.coordinates import LocalPoint
from navigation.distance import distance
from navigation.bearing import bearing
from guidance.loiter_guidance import LoiterDirection
from environment.wind import WindField, ConstantWindField
from guidance.integration_utils import (
    effective_wind,
    evaluate_interval_feasibility,
    integrate_altitude_scalar
)
from guidance.wind_compensation import calculate_wind_corrected_heading

@dataclass(frozen=True)
class ShearTerminalPlan:
    feasible: bool
    effective_wind_x: float
    effective_wind_y: float
    final_ground_course_deg: float
    orbit_center: LocalPoint
    breakout_point: LocalPoint
    orbit_direction: LoiterDirection
    breakout_distance: float
    breakout_altitude: float
    final_ground_speed: float
    final_approach_time: float
    planning_iterations: int
    convergence_error: float

def _solve_required_breakout_altitude(
    d_target: float,
    course_deg: float,
    wind_field: WindField,
    forward_speed: float,
    descent_rate: float,
    h_max: float = 2000.0,
    tol: float = 0.1
) -> Optional[Tuple[float, float, float]]:
    """
    Solve for h_breakout using bisection on the exact integral equation.
    Returns (h_breakout, final_approach_time, average_ground_speed) if feasible.
    """
    h_low = 0.0
    h_high = h_max
    
    def dist_at(h: float) -> float:
        if h < 1e-3:
            return 0.0
        feas = evaluate_interval_feasibility(wind_field, 0.0, h, course_deg, forward_speed)
        if not feas.feasible:
            return -1.0 # Infeasible signals it can't reach this distance
            
        def vg_func(alt: float) -> float:
            wx, wy = wind_field.wind_at_altitude(alt)
            return calculate_wind_corrected_heading(course_deg, forward_speed, wx, wy).ground_speed_along_track
            
        return integrate_altitude_scalar(vg_func, 0.0, h) / descent_rate

    # First check if max altitude is feasible and enough distance
    dist_max = dist_at(h_high)
    if dist_max >= 0 and dist_max < d_target:
        return None # Even at max alt, we can't glide far enough
        
    for _ in range(50):
        h_mid = (h_low + h_high) / 2.0
        d_mid = dist_at(h_mid)
        
        if d_mid < 0:
            # Infeasible at this altitude, must lower altitude
            h_high = h_mid
        elif d_mid < d_target:
            h_low = h_mid
        else:
            h_high = h_mid
            
        if (h_high - h_low) < tol:
            break
            
    h_sol = (h_low + h_high) / 2.0
    # Evaluate final feasibility
    feas = evaluate_interval_feasibility(wind_field, 0.0, h_sol, course_deg, forward_speed)
    if not feas.feasible:
        return None
        
    t_final = h_sol / descent_rate
    vg_avg = d_target / t_final if t_final > 1e-3 else forward_speed
    return h_sol, t_final, vg_avg

def solve_shear_terminal_plan(
    position: LocalPoint,
    target: LocalPoint,
    current_altitude: float,
    wind_field: WindField,
    forward_speed: float,
    descent_rate: float,
    orbit_radius: float,
    min_final_time: float = 5.0,
    min_final_distance: float = 30.0,
    breakout_altitude_margin: float = 2.0,
    max_iterations: int = 10,
    altitude_tolerance: float = 0.5
) -> ShearTerminalPlan:
    
    # 1. Initial guess for h_b
    h_b = 30.0
    
    best_plan = None
    convergence_err = float('inf')
    
    for iteration in range(1, max_iterations + 1):
        wx_eff, wy_eff = effective_wind(wind_field, 0.0, h_b)
        w_mag = math.hypot(wx_eff, wy_eff)
        
        if w_mag < 1e-3:
            # Zero-wind fallback
            best_plan = ShearTerminalPlan(
                feasible=True,
                effective_wind_x=0.0,
                effective_wind_y=0.0,
                final_ground_course_deg=90.0, # Arbitrary safe default
                orbit_center=LocalPoint(target.x, target.y),
                breakout_point=LocalPoint(target.x, target.y),
                orbit_direction=LoiterDirection.CLOCKWISE,
                breakout_distance=0.0,
                breakout_altitude=breakout_altitude_margin,
                final_ground_speed=forward_speed,
                final_approach_time=0.0,
                planning_iterations=iteration,
                convergence_error=0.0
            )
            break
            
        if w_mag >= forward_speed:
            return ShearTerminalPlan(
                feasible=False,
                effective_wind_x=wx_eff,
                effective_wind_y=wy_eff,
                final_ground_course_deg=0.0,
                orbit_center=LocalPoint(target.x, target.y),
                breakout_point=LocalPoint(target.x, target.y),
                orbit_direction=LoiterDirection.CLOCKWISE,
                breakout_distance=0.0,
                breakout_altitude=0.0,
                final_ground_speed=0.0,
                final_approach_time=0.0,
                planning_iterations=iteration,
                convergence_error=0.0
            )
            
        w_hat_x = wx_eff / w_mag
        w_hat_y = wy_eff / w_mag
        
        w_perp_x = w_hat_y
        w_perp_y = -w_hat_x
        
        # u_final points INTO the effective wind
        u_final_x = -w_hat_x
        u_final_y = -w_hat_y
        course_deg = math.degrees(math.atan2(u_final_x, u_final_y)) % 360.0
        
        v_final_est = forward_speed - w_mag
        d_min = max(min_final_distance, min_final_time * v_final_est)
        
        cand_best_cost = float('inf')
        cand_best_plan = None
        cand_best_h = h_b
        
        for d_cand in range(int(math.ceil(d_min)), int(math.ceil(d_min)) + 150, 10):
            d = float(d_cand)
            p_breakout = LocalPoint(target.x + d * w_hat_x, target.y + d * w_hat_y)
            
            # Solve exact h_breakout for this distance
            res = _solve_required_breakout_altitude(d, course_deg, wind_field, forward_speed, descent_rate)
            if res is None:
                continue
                
            h_req, t_final, vg_avg = res
            h_req += breakout_altitude_margin
            
            if h_req > current_altitude - 5.0:
                continue
                
            # Score CW
            c_cw = LocalPoint(p_breakout.x - orbit_radius * w_perp_x, p_breakout.y - orbit_radius * w_perp_y)
            cost_cw = d + distance(position, c_cw)
            if cost_cw < cand_best_cost:
                cand_best_cost = cost_cw
                cand_best_h = h_req
                cand_best_plan = ShearTerminalPlan(
                    feasible=True,
                    effective_wind_x=wx_eff,
                    effective_wind_y=wy_eff,
                    final_ground_course_deg=course_deg,
                    orbit_center=c_cw,
                    breakout_point=p_breakout,
                    orbit_direction=LoiterDirection.CLOCKWISE,
                    breakout_distance=d,
                    breakout_altitude=h_req,
                    final_ground_speed=vg_avg,
                    final_approach_time=t_final,
                    planning_iterations=iteration,
                    convergence_error=0.0
                )
                
            # Score CCW
            c_ccw = LocalPoint(p_breakout.x + orbit_radius * w_perp_x, p_breakout.y + orbit_radius * w_perp_y)
            cost_ccw = d + distance(position, c_ccw)
            if cost_ccw < cand_best_cost:
                cand_best_cost = cost_ccw
                cand_best_h = h_req
                cand_best_plan = ShearTerminalPlan(
                    feasible=True,
                    effective_wind_x=wx_eff,
                    effective_wind_y=wy_eff,
                    final_ground_course_deg=course_deg,
                    orbit_center=c_ccw,
                    breakout_point=p_breakout,
                    orbit_direction=LoiterDirection.COUNTER_CLOCKWISE,
                    breakout_distance=d,
                    breakout_altitude=h_req,
                    final_ground_speed=vg_avg,
                    final_approach_time=t_final,
                    planning_iterations=iteration,
                    convergence_error=0.0
                )
                
        if cand_best_plan is None:
            # Failed to find any feasible candidate
            return ShearTerminalPlan(
                feasible=False,
                effective_wind_x=wx_eff,
                effective_wind_y=wy_eff,
                final_ground_course_deg=course_deg,
                orbit_center=LocalPoint(target.x, target.y),
                breakout_point=LocalPoint(target.x, target.y),
                orbit_direction=LoiterDirection.CLOCKWISE,
                breakout_distance=0.0,
                breakout_altitude=0.0,
                final_ground_speed=0.0,
                final_approach_time=0.0,
                planning_iterations=iteration,
                convergence_error=convergence_err
            )
            
        h_new = cand_best_h
        convergence_err = abs(h_new - h_b)
        
        # Need to update planning_iterations and convergence_error in the returned plan
        best_plan = ShearTerminalPlan(
            feasible=True,
            effective_wind_x=cand_best_plan.effective_wind_x,
            effective_wind_y=cand_best_plan.effective_wind_y,
            final_ground_course_deg=cand_best_plan.final_ground_course_deg,
            orbit_center=cand_best_plan.orbit_center,
            breakout_point=cand_best_plan.breakout_point,
            orbit_direction=cand_best_plan.orbit_direction,
            breakout_distance=cand_best_plan.breakout_distance,
            breakout_altitude=cand_best_plan.breakout_altitude,
            final_ground_speed=cand_best_plan.final_ground_speed,
            final_approach_time=cand_best_plan.final_approach_time,
            planning_iterations=iteration,
            convergence_error=convergence_err
        )
        
        if convergence_err <= altitude_tolerance:
            break
            
        h_b = h_new
        
    return best_plan
