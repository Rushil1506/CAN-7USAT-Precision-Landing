import math
from guidance.breakout_guidance import calculate_turn_endpoint
from guidance.wind_compensation import calculate_wind_corrected_heading
from navigation.coordinates import LocalPoint
from navigation.bearing import bearing
from navigation.distance import distance

def wrap_180(angle):
    return (angle + 180.0) % 360.0 - 180.0

def evaluate_turn_angle(ang, position, heading, turn_dir, turn_radius, forward_speed, wind_x, wind_y, target, max_turn_rate):
    t_turn = 0.0
    if ang > 0:
        num_steps = 20
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
    
    mismatch = wrap_180(ground_course - b_target)
    return mismatch, t_turn, endpoint, ground_course

def find_best_turn_angle(position, heading, target, forward_speed, wind_x, wind_y, max_turn_rate):
    turn_radius = forward_speed / math.radians(max_turn_rate)
    
    best_ang = None
    best_dir = None
    min_mismatch = 999
    
    for turn_dir in [1.0, -1.0]:
        prev_err = None
        prev_ang = None
        
        # Scan 0 to 360 in 5 deg steps
        for step in range(73):
            ang = step * 5.0
            res = evaluate_turn_angle(ang, position, heading, turn_dir, turn_radius, forward_speed, wind_x, wind_y, target, max_turn_rate)
            if res is None: continue
            err = res[0]
            
            if prev_err is not None:
                # Check for zero crossing
                if prev_err * err <= 0 and abs(prev_err - err) < 180:
                    # Bisection
                    a, b = prev_ang, ang
                    for _ in range(15):
                        mid = (a + b) / 2
                        m_res = evaluate_turn_angle(mid, position, heading, turn_dir, turn_radius, forward_speed, wind_x, wind_y, target, max_turn_rate)
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
            
    return best_ang, best_dir, min_mismatch

# 3m/s East Tailwind (Target East at 1000, 0?)
pos = LocalPoint(0.0, -30.0)
heading = 270.0
target = LocalPoint(0.0, 0.0)

print(find_best_turn_angle(pos, heading, target, 8.0, 3.0, 0.0, 30.0))
