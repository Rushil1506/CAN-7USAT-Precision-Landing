import re

with open("guidance/breakout_guidance.py", "r") as f:
    content = f.read()

helper_code = """
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
    best_ang = None
    best_dir = None
    min_mismatch = 999
    
    # Prioritize the direction suggested by initial heading error
    primary_dir = 1.0 if initial_h_err >= 0.0 else -1.0
    
    for turn_dir in [primary_dir, -primary_dir]:
        prev_err = None
        prev_ang = None
        
        # Scan 0 to 360 in 5 deg steps
        for step in range(73):
            ang = step * 5.0
            res = _evaluate_turn_angle(ang, position, heading, turn_dir, turn_radius, forward_speed, wind_x, wind_y, target)
            if res is None: continue
            err = res[0]
            
            # Record absolute minimum just in case bisection doesn't find a zero crossing
            if abs(err) < abs(min_mismatch):
                min_mismatch = err
                best_ang = ang
                best_dir = turn_dir
                
            if prev_err is not None:
                # Check for zero crossing
                if prev_err * err <= 0 and abs(prev_err - err) < 180:
                    # Bisection
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
            
        # If we found a very good match in the primary direction, no need to check the other
        if abs(min_mismatch) < 1.0:
            break
            
    return best_ang, best_dir

"""

replacement = """
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
"""

if "_evaluate_turn_angle" not in content:
    # insert helper above calculate_breakout_geometry
    parts = content.split("def calculate_breakout_geometry(")
    content = parts[0] + helper_code + "\ndef calculate_breakout_geometry(" + parts[1]

# replace the else block inside calculate_breakout_geometry
import re
pattern = r"    else:\n        target_b = bearing\(position, target\)\n        h_err = heading_error\(heading, target_b\)\n        turn_ang = abs\(h_err\)\n        turn_dir = 1\.0 if h_err >= 0\.0 else -1\.0\n        \n        feasible = True\n        t_turn = 0\.0\n        if turn_ang > 0:\n            num_steps = 20"
content = re.sub(pattern, replacement.strip("\n"), content)

with open("guidance/breakout_guidance.py", "w") as f:
    f.write(content)
print("done")
