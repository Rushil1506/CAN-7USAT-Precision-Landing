from dataclasses import dataclass
import math
from typing import Optional

from flight.parafoil import ParafoilConfig
from navigation.bearing import bearing
from navigation.coordinates import LocalPoint
from navigation.distance import distance
from navigation.heading import heading_error
from guidance.wind_compensation import calculate_wind_corrected_heading


@dataclass(frozen=True)
class BreakoutGeometry:
    """
    Geometric and energetic metrics for loiter breakout to landing target.

    Based on tangent-to-circle intercept geometry:
        A turn arc at maximum turn rate to align heading toward the target,
        followed by a straight-line glide segment to the landing target.

    Kinematic Model:
        R_turn = V / omega_rad
        L_turn = R_turn * Delta_psi_rad
        L_straight = ||P_1 - Target||
        L_required = L_turn + L_straight
        G = V / V_z
        h_required_min = L_required / G
        h_required = h_required_min + margin

    Assumptions and Limitations:
        - Simplified kinematic model (constant forward airspeed, constant descent rate).
        - Assumes zero wind and symmetric aerodynamic turn response.
        - Ignores roll transients, bank angle buildup, and spiral dive sink rate increases.
        - This is a kinematic guidance estimate, NOT a full 6-DOF aerodynamic simulation.

    Attributes:
        radial_distance: Euclidean distance from current position to target [m].
        target_bearing: Navigation bearing to landing target [deg, 0-360).
        heading_error: Shortest signed heading error to target bearing [deg, -180 to 180).
        turn_angle: Required heading turn angle magnitude [deg, 0 to 180].
        turn_direction: Turn direction (+1.0 for CW/right, -1.0 for CCW/left).
        turn_radius: Kinematic turn radius R_turn [m].
        turn_duration: Estimated duration of turn arc t_turn [s].
        turn_arc_length: Distance traveled along turn arc L_turn [m].
        turn_endpoint: Estimated position P_1 at completion of turn arc [LocalPoint].
        straight_distance: Straight-line distance from P_1 to target L_straight [m].
        required_path_length: Total required path length L_turn + L_straight [m].
        glide_ratio: Effective kinematic glide ratio V / V_z.
        minimum_required_altitude: Minimum theoretical altitude required to reach target [m].
        breakout_altitude: Recommended operational breakout altitude including margin [m].
    """

    radial_distance: float
    target_bearing: float
    heading_error: float
    turn_angle: float
    turn_direction: float
    turn_radius: float
    turn_duration: float
    turn_arc_length: float
    turn_endpoint: LocalPoint
    straight_distance: float
    required_path_length: float
    glide_ratio: float
    minimum_required_altitude: float
    breakout_altitude: float
    feasible: bool = True


def calculate_turn_endpoint(
    start_point: LocalPoint,
    start_heading: float,
    turn_angle: float,
    turn_direction: float,
    turn_radius: float,
) -> LocalPoint:
    """
    Calculate the kinematic endpoint of a constant-radius turn arc.

    Navigation Coordinate Convention:
        +X = East
        +Y = North
        Heading: 0° = North, 90° = East, 180° = South, 270° = West

    Kinematic Displacement Formula:
        For heading psi measured clockwise from North:
            v_x = V * sin(psi)
            v_y = V * cos(psi)

        Integrating over a constant turn rate omega = s * |omega|:
            x_1 = x_0 + s * R * (cos(psi_0) - cos(psi_1))
            y_1 = y_0 - s * R * (sin(psi_0) - sin(psi_1))

        where:
            s = +1.0 (CW / right) or -1.0 (CCW / left)
            psi_1 = (psi_0 + s * turn_angle) mod 360°

    Args:
        start_point: Starting position P_0.
        start_heading: Starting heading in degrees [0, 360).
        turn_angle: Magnitude of heading change in degrees (must be >= 0).
        turn_direction: Turn direction (+1.0 for CW, -1.0 for CCW).
        turn_radius: Kinematic turn radius in metres (must be >= 0).

    Returns:
        LocalPoint at completion of the turn arc.

    Raises:
        ValueError: If turn_radius < 0 or turn_angle < 0.
    """
    if turn_radius < 0.0:
        raise ValueError("turn_radius cannot be negative")
    if turn_angle < 0.0:
        raise ValueError("turn_angle cannot be negative")

    if turn_radius == 0.0 or turn_angle == 0.0:
        return LocalPoint(x=start_point.x, y=start_point.y)

    s = 1.0 if turn_direction >= 0.0 else -1.0
    psi_0_rad = math.radians(start_heading)
    psi_1_rad = math.radians(start_heading + s * turn_angle)

    dx = s * turn_radius * (math.cos(psi_0_rad) - math.cos(psi_1_rad))
    dy = -s * turn_radius * (math.sin(psi_0_rad) - math.sin(psi_1_rad))

    return LocalPoint(x=start_point.x + dx, y=start_point.y + dy)



def _wrap_180(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0

def _evaluate_turn_angle(ang, position, heading, turn_dir, turn_radius, forward_speed, wind_x, wind_y, target):
    t_turn = 0.0
    if ang > 0:
        num_steps = max(2, int(ang / 5.0))
        dchi_deg = ang / num_steps
        dchi_rad = math.radians(dchi_deg)
        ds = turn_radius * dchi_rad
        for i in range(num_steps):
            course_deg = (heading + turn_dir * (i + 0.5) * dchi_deg) % 360.0
            comp = calculate_wind_corrected_heading(course_deg, forward_speed, wind_x, wind_y)
            if not comp.feasible or comp.ground_speed_along_track <= 0:
                return None
            t_turn += ds / comp.ground_speed_along_track
            
    endpoint_air = calculate_turn_endpoint(position, heading, ang, turn_dir, turn_radius)
    endpoint = LocalPoint(endpoint_air.x + wind_x * t_turn, endpoint_air.y + wind_y * t_turn)
    
    final_heading = (heading + turn_dir * ang) % 360.0
    air_x = forward_speed * math.sin(math.radians(final_heading))
    air_y = forward_speed * math.cos(math.radians(final_heading))
    vg_x = air_x + wind_x
    vg_y = air_y + wind_y
    ground_course = math.degrees(math.atan2(vg_x, vg_y)) % 360.0
    
    b_target = bearing(endpoint, target)
    
    mismatch = _wrap_180(ground_course - b_target)
    return mismatch, t_turn, endpoint, ground_course

def _find_best_turn_angle(position, heading, target, forward_speed, wind_x, wind_y, turn_radius, initial_h_err):
    if abs(wind_x) < 1e-6 and abs(wind_y) < 1e-6:
        return abs(initial_h_err), 1.0 if initial_h_err >= 0.0 else -1.0
        
    best_ang = None
    best_dir = None
    min_mismatch = 999
    
    primary_dir = 1.0 if initial_h_err >= 0.0 else -1.0
    
    for turn_dir in [primary_dir, -primary_dir]:
        prev_err = None
        prev_ang = None
        
        # Scan 0 to 180 in 5 deg steps
        for step in range(37):
            ang = step * 5.0
            res = _evaluate_turn_angle(ang, position, heading, turn_dir, turn_radius, forward_speed, wind_x, wind_y, target)
            if res is None: continue
            err = res[0]
            
            if abs(err) < abs(min_mismatch):
                min_mismatch = err
                best_ang = ang
                best_dir = turn_dir
                
            if prev_err is not None:
                if prev_err * err <= 0 and abs(prev_err - err) < 180:
                    a, b = prev_ang, ang
                    for _ in range(15):
                        mid = (a + b) / 2
                        m_res = _evaluate_turn_angle(mid, position, heading, turn_dir, turn_radius, forward_speed, wind_x, wind_y, target)
                        if m_res is None: break
                        m_err = m_res[0]
                        if m_err * prev_err <= 0:
                            b = mid
                            err = m_err
                        else:
                            a = mid
                            prev_err = m_err
                    if abs(m_err) < abs(min_mismatch):
                        min_mismatch = m_err
                        best_ang = mid
                        best_dir = turn_dir
            prev_err = err
            prev_ang = ang
            
        if abs(min_mismatch) < 0.5:
            break
            
    return best_ang, best_dir


def calculate_breakout_geometry(
    position: LocalPoint,
    heading: float,
    target: LocalPoint,
    forward_speed: float = 8.0,
    descent_rate: float = 2.5,
    max_turn_rate: float = 30.0,
    breakout_altitude_margin: float = 2.0,
    wind_x: float = 0.0,
    wind_y: float = 0.0,
) -> BreakoutGeometry:
    """
    Calculate the path and altitude required to exit loiter and glide to target.

    Uses tangent-to-circle intercept geometry to estimate:
        1. Turn arc to steer onto the target bearing.
        2. Endpoint of turn arc accounting for lateral displacement.
        3. Remaining straight-line glide from turn endpoint to target.
        4. Minimum required altitude and operational breakout altitude.

    Args:
        position: Current CanSat position.
        heading: Current CanSat heading in degrees [0, 360).
        target: Landing target position.
        forward_speed: Forward air-relative speed in m/s (must be > 0, default 8.0).
        descent_rate: Vertical descent rate in m/s (must be > 0, default 2.5).
        max_turn_rate: Maximum kinematic yaw turn rate in deg/s (must be > 0, default 30.0).
        breakout_altitude_margin: Safety margin added to minimum altitude in metres
            (must be >= 0, default 2.0).

    Returns:
        BreakoutGeometry dataclass with detailed path and altitude metrics.

    Raises:
        ValueError: If forward_speed <= 0, descent_rate <= 0, max_turn_rate <= 0,
            or breakout_altitude_margin < 0.
    """
    if forward_speed <= 0.0:
        raise ValueError("forward_speed must be greater than zero")
    if descent_rate <= 0.0:
        raise ValueError("descent_rate must be greater than zero")
    if max_turn_rate <= 0.0:
        raise ValueError("max_turn_rate must be greater than zero")
    if breakout_altitude_margin < 0.0:
        raise ValueError("breakout_altitude_margin cannot be negative")

    rad_dist = distance(position, target)
    glide_ratio = forward_speed / descent_rate
    omega_rad = math.radians(max_turn_rate)
    turn_radius = forward_speed / omega_rad

    if rad_dist < 1e-9:
        target_b = heading % 360.0
        h_err = 0.0
        turn_ang = 0.0
        turn_dir = 1.0
        t_turn = 0.0
        l_turn = 0.0
        endpoint = LocalPoint(x=position.x, y=position.y)
        l_straight = 0.0
        l_required = 0.0
        h_min = 0.0
        h_breakout = breakout_altitude_margin
        feasible = True
    else:
        target_b = bearing(position, target)
        h_err = heading_error(heading, target_b)
        
        # Determine geometric compatibility root finding
        best_ang, best_dir = _find_best_turn_angle(
            position, heading, target, forward_speed, wind_x, wind_y, turn_radius, h_err
        )
        
        if best_ang is not None:
            turn_ang = best_ang
            turn_dir = best_dir
        else:
            # Fallback to simple zero-wind heading error
            turn_ang = abs(h_err)
            turn_dir = 1.0 if h_err >= 0.0 else -1.0
            
        feasible = True
        t_turn = 0.0
        if turn_ang > 0:
            num_steps = max(2, int(turn_ang / 5.0))
            dchi_deg = turn_ang / num_steps
            dchi_rad = math.radians(dchi_deg)
            ds = turn_radius * dchi_rad
            for i in range(num_steps):
                course_deg = (heading + turn_dir * (i + 0.5) * dchi_deg) % 360.0
                comp = calculate_wind_corrected_heading(
                    desired_course_deg=course_deg,
                    airspeed=forward_speed,
                    wind_x=wind_x,
                    wind_y=wind_y
                )
                if not comp.feasible or comp.ground_speed_along_track <= 0.0:
                    feasible = False
                    break
                t_turn += ds / comp.ground_speed_along_track

        # Calculate kinematic turn endpoint (air mass frame)
        endpoint_air = calculate_turn_endpoint(
            start_point=position,
            start_heading=heading,
            turn_angle=turn_ang,
            turn_direction=turn_dir,
            turn_radius=turn_radius,
        )

        # Apply wind drift over the turn duration
        dx_wind = wind_x * t_turn
        dy_wind = wind_y * t_turn
        endpoint = LocalPoint(x=endpoint_air.x + dx_wind, y=endpoint_air.y + dy_wind)

        l_straight = distance(endpoint, target)
        
        t_straight = 0.0
        if feasible and l_straight > 0.0:
            straight_course = bearing(endpoint, target)
            comp_straight = calculate_wind_corrected_heading(
                desired_course_deg=straight_course,
                airspeed=forward_speed,
                wind_x=wind_x,
                wind_y=wind_y
            )
            if not comp_straight.feasible or comp_straight.ground_speed_along_track <= 0.0:
                feasible = False
            else:
                t_straight = l_straight / comp_straight.ground_speed_along_track
                
        l_turn = forward_speed * (turn_ang / max_turn_rate)
        l_required = l_turn + l_straight
        
        if feasible:
            t_total = t_turn + t_straight
            h_min = descent_rate * t_total
            h_breakout = h_min + breakout_altitude_margin
        else:
            h_min = 0.0
            h_breakout = breakout_altitude_margin
            t_turn = 0.0

    return BreakoutGeometry(
        radial_distance=rad_dist,
        target_bearing=target_b,
        heading_error=h_err,
        turn_angle=turn_ang,
        turn_direction=turn_dir,
        turn_radius=turn_radius,
        turn_duration=t_turn,
        turn_arc_length=l_turn,
        turn_endpoint=endpoint,
        straight_distance=l_straight,
        required_path_length=l_required,
        glide_ratio=glide_ratio,
        minimum_required_altitude=h_min,
        breakout_altitude=h_breakout,
        feasible=feasible,
    )


def calculate_breakout_geometry_from_config(
    position: LocalPoint,
    heading: float,
    target: LocalPoint,
    config: ParafoilConfig,
    breakout_altitude_margin: float = 2.0,
) -> BreakoutGeometry:
    return calculate_breakout_geometry(
        position=position,
        heading=heading,
        target=target,
        forward_speed=config.forward_speed,
        descent_rate=config.descent_rate,
        max_turn_rate=config.max_turn_rate,
        breakout_altitude_margin=breakout_altitude_margin,
        wind_x=config.wind_x,
        wind_y=config.wind_y,
    )
